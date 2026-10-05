//! Public key authentication for the built-in client: the agent's keys, then
//! the key files, skipping what the client cannot use (an encrypted file, an
//! RSA key) with a note that ends up in the error when nothing authenticates.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use russh::client::{AuthResult, Handle, Handler};
use russh::keys::agent::AgentIdentity;
use russh::keys::agent::client::AgentClient;
use russh::keys::{PrivateKey, PrivateKeyWithHashAlg};

use super::config::{AgentChoice, HostConfig};

/// What [`authenticate`] offers the server, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthPlan {
    /// The agent socket whose keys come first, if any.
    pub agent: Option<PathBuf>,
    /// The key files tried after the agent's keys.
    pub files: Vec<PathBuf>,
}

/// The plan for `host`: with `IdentitiesOnly` only its key files, otherwise the
/// agent (`IdentityAgent`, or `auth_sock` from `SSH_AUTH_SOCK`, unless `none`)
/// then the key files.
pub fn plan(host: &HostConfig, auth_sock: Option<&Path>) -> AuthPlan {
    let agent = if host.identities_only {
        None
    } else {
        match &host.identity_agent {
            AgentChoice::Env => auth_sock.map(Path::to_path_buf),
            AgentChoice::None => None,
            AgentChoice::Path(path) => Some(path.clone()),
        }
    };
    AuthPlan {
        agent,
        files: host.identity_files.clone(),
    }
}

/// A key file as the built-in client sees it.
#[derive(Debug)]
pub enum KeyFile {
    /// No file at that path: skipped without a note, as OpenSSH does.
    Missing,
    /// A key the client can sign with.
    Usable(Box<PrivateKey>),
    /// A key the client cannot use, and why, in words for the user.
    Skipped(String),
}

/// What `path` holds: never prompts for a passphrase. The note of a skipped
/// file names the path only, never the key.
pub fn load_key(path: &Path) -> KeyFile {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
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
        return match PrivateKey::from_openssh(&text) {
            Ok(key) if key.algorithm().is_rsa() => KeyFile::Skipped(rsa_note(path)),
            Ok(key) if key.is_encrypted() => KeyFile::Skipped(encrypted_note(path)),
            Ok(key) => KeyFile::Usable(Box::new(key)),
            Err(_) => KeyFile::Skipped(unreadable_note(path)),
        };
    }
    if text.contains("ENCRYPTED") {
        return KeyFile::Skipped(encrypted_note(path));
    }
    match russh::keys::decode_secret_key(&text, None) {
        Ok(key) if key.algorithm().is_rsa() => KeyFile::Skipped(rsa_note(path)),
        Ok(key) => KeyFile::Usable(Box::new(key)),
        Err(russh::keys::Error::KeyIsEncrypted) => KeyFile::Skipped(encrypted_note(path)),
        Err(_) => KeyFile::Skipped(unreadable_note(path)),
    }
}

/// The note for an encrypted key file (D11).
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
    let mut notes = Vec::new();
    let mut offered = 0_usize;
    if let Some(socket) = &plan.agent {
        match agent_keys(handle, user, socket, (&mut offered, &mut notes)).await {
            Ok(true) => return Ok(()),
            Ok(false) => {},
            Err(note) => notes.push(note),
        }
    }
    for path in &plan.files {
        let key = match load_key(path) {
            KeyFile::Missing => continue,
            KeyFile::Skipped(note) => {
                notes.push(note);
                continue;
            },
            KeyFile::Usable(key) => key,
        };
        offered += 1;
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
    Err(failure(offered, &notes))
}

/// Offers the agent's keys at `socket`, counting them in `offered` and noting
/// in `notes` a key the agent failed to sign with: `true` once one is
/// accepted.
///
/// # Errors
///
/// Returns a note when the agent cannot be reached or listed.
async fn agent_keys<H: Handler>(
    handle: &mut Handle<H>,
    user: &str,
    socket: &Path,
    (offered, notes): (&mut usize, &mut Vec<String>),
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
        if key.algorithm().is_rsa() {
            continue;
        }
        *offered += 1;
        let result = handle
            .authenticate_publickey_with(user, key, None, &mut agent)
            .await;
        match agent_attempt(&result, socket) {
            Attempt::Accepted => return Ok(true),
            Attempt::Refused => {},
            Attempt::Failed(note) => {
                if !notes.contains(&note) {
                    notes.push(note);
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

    type TestResult = Result<(), Box<dyn std::error::Error>>;

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

    /// The agent comes before the files, unless `IdentitiesOnly` or
    /// `IdentityAgent none` keeps it out; `IdentityAgent <path>` wins over
    /// `SSH_AUTH_SOCK`.
    #[test]
    fn the_plan_follows_identities_only_and_identity_agent() {
        let sock = Path::new("/run/agent.sock");
        let files = vec![PathBuf::from("/k/one"), PathBuf::from("/k/two")];
        let with_env = plan(&host(false, AgentChoice::Env), Some(sock));
        assert_eq!(with_env.agent.as_deref(), Some(sock));
        assert_eq!(with_env.files, files);
        assert_eq!(plan(&host(false, AgentChoice::Env), None).agent, None);
        let only = plan(&host(true, AgentChoice::Env), Some(sock));
        assert_eq!(only.agent, None);
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
        assert_eq!(only_own.agent, None);
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
        fs::write(&path, ed25519()?.to_openssh(LineEnding::LF)?.as_bytes())?;
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
        fs::write(&path, key.to_openssh(LineEnding::LF)?.as_bytes())?;
        let KeyFile::Skipped(note) = load_key(&path) else {
            return Err("the encrypted key was not skipped".into());
        };
        assert_eq!(
            note,
            format!("key {} is encrypted: add it to ssh-agent", path.display())
        );
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
        fs::write(&path, "hello\n")?;
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
