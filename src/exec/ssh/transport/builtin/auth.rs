//! Public key authentication for the built-in client: the agent's keys, then
//! the key files, skipping what the client cannot use (an encrypted file the
//! agent does not hold, an RSA key, a file others may read) with a note that
//! ends up in the error when nothing authenticates.

use std::fs;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use russh::client::{AuthResult, Handle, Handler};
use russh::keys::agent::AgentIdentity;
use russh::keys::agent::client::AgentClient;
use russh::keys::{PrivateKey, PrivateKeyWithHashAlg, PublicKey};
use secrecy::zeroize::Zeroizing;

use super::config::{AgentChoice, HostConfig};

/// What [`authenticate`] offers the server, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthPlan {
    /// The agent socket whose keys come first, if any.
    pub agent: Option<PathBuf>,
    /// The key files tried after the agent's keys.
    pub files: Vec<PathBuf>,
    /// `IdentitiesOnly`: the agent offers only the keys of [`Self::files`].
    pub identities_only: bool,
}

/// The plan for `host`: the agent (`IdentityAgent`, or `auth_sock` from
/// `SSH_AUTH_SOCK`, unless `none`) then the key files. With `IdentitiesOnly`
/// the agent offers only the keys of those files, as OpenSSH does.
pub fn plan(host: &HostConfig, auth_sock: Option<&Path>) -> AuthPlan {
    let agent = match &host.identity_agent {
        AgentChoice::Env => auth_sock.map(Path::to_path_buf),
        AgentChoice::None => None,
        AgentChoice::Path(path) => Some(path.clone()),
    };
    AuthPlan {
        agent,
        files: host.identity_files.clone(),
        identities_only: host.identities_only,
    }
}

/// A key file as the built-in client sees it.
#[derive(Debug)]
pub enum KeyFile {
    /// No file at that path: skipped without a note, as OpenSSH does.
    Missing,
    /// A key the client can sign with.
    Usable(Box<PrivateKey>),
    /// A key behind a passphrase: usable only through the agent.
    Encrypted,
    /// A key the client cannot use, and why, in words for the user.
    Skipped(String),
}

/// What `path` holds: never prompts for a passphrase. A file its group or
/// others may read is skipped, as OpenSSH does. The note of a skipped file
/// names the path only, never the key.
pub fn load_key(path: &Path) -> KeyFile {
    match fs::metadata(path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => return KeyFile::Missing,
        Ok(metadata) if metadata.permissions().mode() & 0o077 != 0 => {
            return KeyFile::Skipped(format!(
                "Permissions {:04o} for '{}' are too open",
                metadata.permissions().mode() & 0o7777,
                path.display()
            ));
        },
        _ => {},
    }
    let text = match fs::read_to_string(path) {
        Ok(text) => Zeroizing::new(text),
        Err(error) if error.kind() == io::ErrorKind::NotFound => return KeyFile::Missing,
        Err(error) => {
            return KeyFile::Skipped(format!(
                "key {} cannot be read: {}",
                path.display(),
                error.kind()
            ));
        },
    };
    if text.contains("-----BEGIN RSA PRIVATE KEY-----") {
        return KeyFile::Skipped(rsa_note(path));
    }
    if text.contains("-----BEGIN OPENSSH PRIVATE KEY-----") {
        return match PrivateKey::from_openssh(text.as_str()) {
            Ok(key) if key.algorithm().is_rsa() => KeyFile::Skipped(rsa_note(path)),
            Ok(key) if key.is_encrypted() => KeyFile::Encrypted,
            Ok(key) => KeyFile::Usable(Box::new(key)),
            Err(_) => KeyFile::Skipped(unreadable_note(path)),
        };
    }
    if text.contains("ENCRYPTED") {
        return KeyFile::Encrypted;
    }
    match russh::keys::decode_secret_key(&text, None) {
        Ok(key) if key.algorithm().is_rsa() => KeyFile::Skipped(rsa_note(path)),
        Ok(key) => KeyFile::Usable(Box::new(key)),
        Err(russh::keys::Error::KeyIsEncrypted) => KeyFile::Encrypted,
        Err(_) => KeyFile::Skipped(unreadable_note(path)),
    }
}

/// The public key of the key file at `path`: from `<path>.pub` when it holds
/// one, else from the private file. An OpenSSH private key carries its public
/// key in clear, so an encrypted one gives it too; an encrypted PEM file does
/// not.
fn file_public_key(path: &Path) -> Option<PublicKey> {
    let mut public = path.as_os_str().to_owned();
    public.push(".pub");
    if let Some(key) = fs::read_to_string(&public)
        .ok()
        .and_then(|text| PublicKey::from_openssh(text.trim()).ok())
    {
        return Some(key);
    }
    let text = Zeroizing::new(fs::read_to_string(path).ok()?);
    if text.contains("-----BEGIN OPENSSH PRIVATE KEY-----") {
        return PrivateKey::from_openssh(text.as_str())
            .ok()
            .map(|key| key.public_key().clone());
    }
    russh::keys::decode_secret_key(&text, None)
        .ok()
        .map(|key| key.public_key().clone())
}

/// Whether `key` is in `keys`, comments aside.
fn holds(keys: &[PublicKey], key: &PublicKey) -> bool {
    keys.iter().any(|held| held.key_data() == key.key_data())
}

/// Whether the agent key `key` is offered: not RSA (S3), and with
/// `IdentitiesOnly` (`only` holding the keys of the key files) one of those.
fn offers_agent_key(key: &PublicKey, only: Option<&[PublicKey]>) -> bool {
    !key.algorithm().is_rsa() && only.is_none_or(|files| holds(files, key))
}

/// The note for the encrypted key file at `path`, or `None` when, with
/// `identities_only`, the agent holds its key among `held`: the agent has
/// offered it already.
fn encrypted_skip_note(path: &Path, identities_only: bool, held: &[PublicKey]) -> Option<String> {
    let through_agent =
        identities_only && file_public_key(path).is_some_and(|key| holds(held, &key));
    (!through_agent).then(|| encrypted_note(path))
}

/// The note for an encrypted key file the agent does not hold (D11).
fn encrypted_note(path: &Path) -> String {
    format!("key {} is encrypted: add it to ssh-agent", path.display())
}

/// The note for an RSA key file (S3, D11).
fn rsa_note(path: &Path) -> String {
    format!(
        "key {} is RSA, which the built-in SSH client does not support: use an ed25519 key, or ssh_client = \"openssh\"",
        path.display()
    )
}

/// The note for a file that holds no private key the client reads.
fn unreadable_note(path: &Path) -> String {
    format!(
        "key {} is not a private key the built-in SSH client can read",
        path.display()
    )
}

/// What the offers so far gave: the keys offered, the notes of what was
/// skipped, and the keys the agent holds.
#[derive(Debug, Default)]
struct Tally {
    /// How many keys were offered.
    offered: usize,
    /// Why keys were skipped, and agent failures.
    notes: Vec<String>,
    /// The keys the agent listed.
    held: Vec<PublicKey>,
}

/// Offers the keys of `plan` for `user` until the server accepts one.
///
/// # Errors
///
/// Returns why nothing authenticated: how many keys the server refused, and
/// every note of a skipped key or an unreachable agent.
pub async fn authenticate<H: Handler>(
    handle: &mut Handle<H>,
    user: &str,
    plan: &AuthPlan,
) -> Result<(), String> {
    let mut tally = Tally::default();
    if let Some(socket) = &plan.agent {
        let file_keys: Option<Vec<PublicKey>> = plan.identities_only.then(|| {
            plan.files
                .iter()
                .filter_map(|path| file_public_key(path))
                .collect()
        });
        match agent_keys(handle, user, socket, file_keys.as_deref(), &mut tally).await {
            Ok(true) => return Ok(()),
            Ok(false) => {},
            Err(note) => tally.notes.push(note),
        }
    }
    for path in &plan.files {
        let key = match load_key(path) {
            KeyFile::Missing => continue,
            KeyFile::Encrypted => {
                tally
                    .notes
                    .extend(encrypted_skip_note(path, plan.identities_only, &tally.held));
                continue;
            },
            KeyFile::Skipped(note) => {
                tally.notes.push(note);
                continue;
            },
            KeyFile::Usable(key) => key,
        };
        tally.offered += 1;
        let key = PrivateKeyWithHashAlg::new(Arc::new(*key), None);
        match handle.authenticate_publickey(user, key).await {
            Ok(AuthResult::Success) => return Ok(()),
            Ok(AuthResult::Failure { .. }) => {},
            Err(error) => {
                return Err(format!(
                    "the connection failed during authentication: {error}"
                ));
            },
        }
    }
    Err(failure(tally.offered, &tally.notes))
}

/// Offers the agent's keys at `socket` (with `IdentitiesOnly`, only those in
/// `only`), recording in `tally` the keys it holds, the keys offered and a key
/// the agent failed to sign with: `true` once one is accepted.
///
/// # Errors
///
/// Returns a note when the agent cannot be reached or listed.
async fn agent_keys<H: Handler>(
    handle: &mut Handle<H>,
    user: &str,
    socket: &Path,
    only: Option<&[PublicKey]>,
    tally: &mut Tally,
) -> Result<bool, String> {
    let unreachable = |_| format!("the agent at {} cannot be reached", socket.display());
    let mut agent = AgentClient::connect_uds(socket)
        .await
        .map_err(unreachable)?;
    let identities = agent.request_identities().await.map_err(unreachable)?;
    for identity in identities {
        // Certificates and RSA keys are left to OpenSSH (S3).
        let AgentIdentity::PublicKey { key, .. } = identity else {
            continue;
        };
        tally.held.push(key.clone());
        if !offers_agent_key(&key, only) {
            continue;
        }
        tally.offered += 1;
        let result = handle
            .authenticate_publickey_with(user, key, None, &mut agent)
            .await;
        match agent_attempt(&result, socket) {
            Attempt::Accepted => return Ok(true),
            Attempt::Refused => {},
            Attempt::Failed(note) => {
                if !tally.notes.contains(&note) {
                    tally.notes.push(note);
                }
            },
        }
    }
    Ok(false)
}

/// What one offer of an agent key gave.
#[derive(Debug, PartialEq, Eq)]
enum Attempt {
    /// The server accepted the key.
    Accepted,
    /// The server refused it: the next key is tried.
    Refused,
    /// The agent could not sign, with the note saying so: the next key is
    /// tried too.
    Failed(String),
}

/// The [`Attempt`] of an agent key offer that gave `result`, the agent
/// listening at `socket`.
fn agent_attempt<E>(result: &Result<AuthResult, E>, socket: &Path) -> Attempt {
    match result {
        Ok(AuthResult::Success) => Attempt::Accepted,
        Ok(AuthResult::Failure { .. }) => Attempt::Refused,
        Err(_) => Attempt::Failed(format!(
            "the agent at {} failed to sign with one of its keys",
            socket.display()
        )),
    }
}

/// The reason of an authentication failure after `offered` refused keys and
/// the `notes` of what was skipped.
fn failure(offered: usize, notes: &[String]) -> String {
    let head = if offered == 0 {
        "no key to offer".to_string()
    } else {
        format!("the server refused the {offered} key(s) offered")
    };
    if notes.is_empty() {
        head
    } else {
        format!("{head}; {}", notes.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use std::process::{Command, Stdio};
    use std::time::Duration;

    use russh::keys::ssh_key::LineEnding;
    use russh::keys::{Algorithm, PrivateKey};

    use super::*;

    /// The result of a test that may fail on I/O or key handling.
    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// Writes `text` to `path` readable by its owner only, as a private key
    /// file must be.
    fn write_key(path: &Path, text: &str) -> io::Result<()> {
        use std::os::unix::fs::PermissionsExt;
        fs::write(path, text.as_bytes())?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
    }

    /// A resolved host with `identities_only` and `agent`, two key files.
    fn host(identities_only: bool, agent: AgentChoice) -> HostConfig {
        HostConfig {
            alias: "h".into(),
            host_name: "h".into(),
            user: "u".into(),
            port: 22,
            identity_files: vec!["/k/one".into(), "/k/two".into()],
            identities_only,
            identity_agent: agent,
            known_hosts: Vec::new(),
            host_key_alias: None,
            proxy_jump: Vec::new(),
            connect_timeout: Duration::from_secs(30),
            alive_interval: Duration::from_secs(15),
            alive_count: 3,
        }
    }

    /// The agent comes before the files, unless `IdentityAgent none` keeps it
    /// out; `IdentityAgent <path>` wins over `SSH_AUTH_SOCK`. `IdentitiesOnly`
    /// keeps the agent and is carried in the plan.
    #[test]
    fn the_plan_follows_identities_only_and_identity_agent() {
        let sock = Path::new("/run/agent.sock");
        let files = vec![PathBuf::from("/k/one"), PathBuf::from("/k/two")];
        let with_env = plan(&host(false, AgentChoice::Env), Some(sock));
        assert_eq!(with_env.agent.as_deref(), Some(sock));
        assert_eq!(with_env.files, files);
        assert_eq!(plan(&host(false, AgentChoice::Env), None).agent, None);
        assert!(!with_env.identities_only);
        let only = plan(&host(true, AgentChoice::Env), Some(sock));
        assert_eq!(only.agent.as_deref(), Some(sock));
        assert!(only.identities_only);
        assert_eq!(only.files, files);
        assert_eq!(
            plan(&host(false, AgentChoice::None), Some(sock)).agent,
            None
        );
        let own = plan(
            &host(false, AgentChoice::Path("/own.sock".into())),
            Some(sock),
        );
        assert_eq!(own.agent.as_deref(), Some(Path::new("/own.sock")));
        let only_own = plan(&host(true, AgentChoice::Path("/own.sock".into())), None);
        assert_eq!(only_own.agent.as_deref(), Some(Path::new("/own.sock")));
    }

    /// A fresh ed25519 key in OpenSSH format.
    fn ed25519() -> Result<PrivateKey, Box<dyn std::error::Error>> {
        Ok(PrivateKey::random(&mut rand::rng(), Algorithm::Ed25519)?)
    }

    /// A plain ed25519 file is usable, an absent file is skipped silently.
    #[test]
    fn a_plain_key_is_usable_and_a_missing_one_is_skipped() -> TestResult {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("id_ed25519");
        write_key(&path, &ed25519()?.to_openssh(LineEnding::LF)?)?;
        assert!(matches!(load_key(&path), KeyFile::Usable(_)));
        assert!(matches!(
            load_key(&dir.path().join("absent")),
            KeyFile::Missing
        ));
        Ok(())
    }

    /// An encrypted key is skipped with the note D11 gives, naming the path.
    #[test]
    fn an_encrypted_key_is_skipped_with_its_note() -> TestResult {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("id_ed25519");
        let key = ed25519()?.encrypt(&mut rand::rng(), "passphrase")?;
        write_key(&path, &key.to_openssh(LineEnding::LF)?)?;
        assert!(matches!(load_key(&path), KeyFile::Encrypted));
        assert_eq!(
            encrypted_note(&path),
            format!("key {} is encrypted: add it to ssh-agent", path.display())
        );
        Ok(())
    }

    /// A private key file that its group or others may read is skipped with
    /// OpenSSH's words, as OpenSSH ignores it.
    #[test]
    fn a_key_others_may_read_is_skipped() -> TestResult {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("id_ed25519");
        write_key(&path, &ed25519()?.to_openssh(LineEnding::LF)?)?;
        for mode in [0o644, 0o640, 0o604] {
            fs::set_permissions(&path, fs::Permissions::from_mode(mode))?;
            let KeyFile::Skipped(note) = load_key(&path) else {
                return Err("the open key was not skipped".into());
            };
            assert_eq!(
                note,
                format!(
                    "Permissions {mode:04o} for '{}' are too open",
                    path.display()
                )
            );
        }
        fs::set_permissions(&path, fs::Permissions::from_mode(0o400))?;
        assert!(matches!(load_key(&path), KeyFile::Usable(_)));
        Ok(())
    }

    /// The public key of a key file comes from `<file>.pub` when present, else
    /// from the private file, encrypted or not; nothing for a missing file or
    /// one that holds no key.
    #[test]
    fn a_key_file_gives_its_public_key() -> TestResult {
        let dir = tempfile::tempdir()?;
        let plain = ed25519()?;
        let plain_path = dir.path().join("plain");
        write_key(&plain_path, &plain.to_openssh(LineEnding::LF)?)?;
        assert_eq!(
            file_public_key(&plain_path),
            Some(plain.public_key().clone())
        );

        let locked = ed25519()?;
        let locked_path = dir.path().join("locked");
        let encrypted = locked.encrypt(&mut rand::rng(), "passphrase")?;
        write_key(&locked_path, &encrypted.to_openssh(LineEnding::LF)?)?;
        assert_eq!(
            file_public_key(&locked_path),
            Some(locked.public_key().clone())
        );

        let other = ed25519()?;
        fs::write(
            dir.path().join("plain.pub"),
            other.public_key().to_openssh()?,
        )?;
        assert_eq!(
            file_public_key(&plain_path),
            Some(other.public_key().clone())
        );

        fs::write(dir.path().join("notes"), "hello\n")?;
        assert_eq!(file_public_key(&dir.path().join("notes")), None);
        assert_eq!(file_public_key(&dir.path().join("absent")), None);
        Ok(())
    }

    /// Without `IdentitiesOnly` every agent key is offered; with it, only the
    /// keys of the key files, whatever their comments.
    #[test]
    fn identities_only_offers_the_agent_keys_of_the_key_files() -> TestResult {
        let file = ed25519()?.public_key().clone();
        let mut same = file.clone();
        same.set_comment("agent comment");
        let stranger = ed25519()?.public_key().clone();
        assert!(offers_agent_key(&stranger, None));
        assert!(offers_agent_key(&same, Some(std::slice::from_ref(&file))));
        assert!(!offers_agent_key(
            &stranger,
            Some(std::slice::from_ref(&file))
        ));
        assert!(!offers_agent_key(&file, Some(&[])));
        Ok(())
    }

    /// With `IdentitiesOnly`, an encrypted key file the agent holds goes
    /// through the agent without a note; one it does not hold keeps the "add
    /// it to ssh-agent" note. Without `IdentitiesOnly` the note stays.
    #[test]
    fn an_encrypted_key_the_agent_holds_has_no_note() -> TestResult {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("id_ed25519");
        let key = ed25519()?;
        let encrypted = key.encrypt(&mut rand::rng(), "passphrase")?;
        write_key(&path, &encrypted.to_openssh(LineEnding::LF)?)?;
        let held = [key.public_key().clone()];
        let stranger = [ed25519()?.public_key().clone()];
        let note = Some(encrypted_note(&path));
        assert_eq!(encrypted_skip_note(&path, true, &held), None);
        assert_eq!(encrypted_skip_note(&path, true, &stranger), note);
        assert_eq!(encrypted_skip_note(&path, true, &[]), note);
        assert_eq!(encrypted_skip_note(&path, false, &held), note);
        Ok(())
    }

    /// An RSA key, in the OpenSSH or the PEM format, is skipped with the note
    /// D11 gives. Generated with `ssh-keygen`, since this build has no RSA.
    #[test]
    fn an_rsa_key_is_skipped_with_its_note() -> TestResult {
        let dir = tempfile::tempdir()?;
        for (name, format) in [("id_rsa", "RFC4716"), ("id_rsa_pem", "PEM")] {
            let path = dir.path().join(name);
            let made = Command::new("ssh-keygen")
                .args([
                    "-q", "-t", "rsa", "-b", "2048", "-N", "", "-m", format, "-f",
                ])
                .arg(&path)
                .stdin(Stdio::null())
                .status();
            if !made.is_ok_and(|status| status.success()) {
                eprintln!("skipped: ssh-keygen cannot make an RSA key here");
                return Ok(());
            }
            let KeyFile::Skipped(note) = load_key(&path) else {
                return Err("the RSA key was not skipped".into());
            };
            assert_eq!(
                note,
                format!(
                    "key {} is RSA, which the built-in SSH client does not support: use an ed25519 key, or ssh_client = \"openssh\"",
                    path.display()
                )
            );
        }
        Ok(())
    }

    /// A file that is no key is skipped with a note naming only the path.
    #[test]
    fn a_file_that_is_no_key_is_skipped() -> TestResult {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("notes.txt");
        write_key(&path, "hello\n")?;
        let KeyFile::Skipped(note) = load_key(&path) else {
            return Err("the file was not skipped".into());
        };
        assert!(note.starts_with(&format!("key {} is not", path.display())));
        Ok(())
    }

    /// An agent signing failure is a note and the next key is tried; a
    /// refusal moves on without one.
    #[test]
    fn an_agent_signing_failure_is_noted_and_skipped() {
        let socket = Path::new("/run/agent.sock");
        assert_eq!(
            agent_attempt::<()>(&Err(()), socket),
            Attempt::Failed(
                "the agent at /run/agent.sock failed to sign with one of its keys".into()
            )
        );
        assert_eq!(
            agent_attempt::<()>(&Ok(AuthResult::Success), socket),
            Attempt::Accepted
        );
        let refused = AuthResult::Failure {
            remaining_methods: russh::MethodSet::empty(),
            partial_success: false,
        };
        assert_eq!(agent_attempt::<()>(&Ok(refused), socket), Attempt::Refused);
    }

    /// The failure reason counts the refused keys and lists every note.
    #[test]
    fn the_failure_lists_what_was_skipped() {
        assert_eq!(failure(0, &[]), "no key to offer");
        assert_eq!(
            failure(2, &["key a is encrypted: add it to ssh-agent".into()]),
            "the server refused the 2 key(s) offered; key a is encrypted: add it to ssh-agent"
        );
        assert_eq!(
            failure(0, &["n1".into(), "n2".into()]),
            "no key to offer; n1; n2"
        );
    }
}
