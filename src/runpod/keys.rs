//! Per-run SSH keys and the per-run ssh config: `runs/<run-id>/ssh/`.
//!
//! The pod's host key is generated here and sent in the create call's `env`; its
//! public half is pinned in a `known_hosts` keyed by a `HostKeyAlias`, so a new
//! public port after a pod reset only changes the config, never the pinned key.
//! The config is passed to `ssh -F`, which then ignores `~/.ssh/config` and
//! `/etc/ssh/ssh_config`: no user setting can weaken or break the connection.

use std::fs;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use secrecy::SecretString;

use super::{PodError, SshEndpoint};

/// Directory of a run's SSH files, in its local run directory.
pub const SSH_DIR: &str = "ssh";
/// The run's client private key, in [`SSH_DIR`].
pub const CLIENT_KEY: &str = "id_ed25519";
/// The run's ssh config, in [`SSH_DIR`].
pub const SSH_CONFIG: &str = "config";
/// The pod's pinned host key, in [`SSH_DIR`].
pub const KNOWN_HOSTS: &str = "known_hosts";
/// Name of the host key while it is generated, in [`SSH_DIR`]; removed at once.
const HOST_KEY: &str = "host_ed25519";

/// The SSH host alias of a run's pod, also its `HostKeyAlias`.
#[must_use]
pub fn alias(run_id: &str) -> String {
    format!("overbrainer-{run_id}")
}

/// The keys of a run: the client key overbrainer logs in with, and the pod's host
/// key. The private host key only lives in memory, as the base64 of its OpenSSH
/// file, until it is sent in the create call.
#[derive(Debug, Clone)]
pub struct PodKeys {
    /// The client private key file, absolute.
    pub client_key: PathBuf,
    /// The client public key, one `ssh-ed25519 AAAA... comment` line.
    pub client_public: String,
    /// The pod's public host key, `ssh-ed25519 AAAA...`.
    pub host_public: String,
    host_private: SecretString,
}

impl PodKeys {
    /// Keys from their parts. `host_private` is the base64 of the OpenSSH private
    /// host key file.
    #[must_use]
    pub fn new(
        client_key: PathBuf,
        client_public: String,
        host_public: String,
        host_private: SecretString,
    ) -> Self {
        Self {
            client_key,
            client_public,
            host_public,
            host_private,
        }
    }

    /// Generates both keys with `ssh-keygen` (ed25519, no passphrase), the client
    /// key into `dir` (created with mode 700). The host key file is read and
    /// removed at once.
    ///
    /// # Errors
    ///
    /// Returns [`PodError::Keygen`] when `ssh-keygen` is missing or fails, and
    /// [`PodError::Io`] when a file cannot be written, read or removed.
    pub fn generate(dir: &Path, comment: &str) -> Result<Self, PodError> {
        fs::create_dir_all(dir).map_err(io_error(dir))?;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700)).map_err(io_error(dir))?;
        let dir = std::path::absolute(dir).map_err(io_error(dir))?;
        let client_key = dir.join(CLIENT_KEY);
        keygen(&client_key, comment)?;
        let client_public = read_public(&client_key)?;
        let host_key = dir.join(HOST_KEY);
        keygen(&host_key, comment)?;
        let host_public = read_public(&host_key)?;
        let host_private = fs::read(&host_key).map_err(io_error(&host_key))?;
        remove_pair(&host_key)?;
        Ok(Self {
            client_key,
            client_public,
            host_public: key_fields(&host_public),
            host_private: SecretString::from(base64(&host_private)),
        })
    }

    /// The base64 of the OpenSSH private host key file, for the create call.
    #[must_use]
    pub fn host_private(&self) -> &SecretString {
        &self.host_private
    }
}

/// Runs `ssh-keygen` to write a fresh ed25519 key pair at `path`.
fn keygen(path: &Path, comment: &str) -> Result<(), PodError> {
    remove_pair(path)?;
    let output = Command::new("ssh-keygen")
        .args(["-q", "-t", "ed25519", "-N", "", "-C", comment, "-f"])
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .map_err(|error| {
            PodError::Keygen(format!(
                "cannot run ssh-keygen ({error}); it ships with the OpenSSH client"
            ))
        })?;
    if output.status.success() {
        Ok(())
    } else {
        Err(PodError::Keygen(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ))
    }
}

/// The public key next to the private key `path`, on one line.
fn read_public(path: &Path) -> Result<String, PodError> {
    let public = public_path(path);
    let text = fs::read_to_string(&public).map_err(io_error(&public))?;
    Ok(text.trim().to_string())
}

fn public_path(path: &Path) -> PathBuf {
    let mut public = path.as_os_str().to_owned();
    public.push(".pub");
    PathBuf::from(public)
}

/// Removes the key pair at `path`, if any.
fn remove_pair(path: &Path) -> Result<(), PodError> {
    for file in [path.to_path_buf(), public_path(path)] {
        match fs::remove_file(&file) {
            Ok(()) => {},
            Err(e) if e.kind() == io::ErrorKind::NotFound => {},
            Err(source) => return Err(PodError::Io { path: file, source }),
        }
    }
    Ok(())
}

/// The key type and the key of a public key line, without its comment.
fn key_fields(line: &str) -> String {
    line.split_whitespace()
        .take(2)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Writes `known_hosts` into `dir`: one line pinning `host_public` for `alias`.
/// Returns its path.
///
/// # Errors
///
/// Returns [`PodError::Io`] when the file cannot be written.
pub fn write_known_hosts(dir: &Path, alias: &str, host_public: &str) -> Result<PathBuf, PodError> {
    let path = dir.join(KNOWN_HOSTS);
    let line = format!("{alias} {}\n", key_fields(host_public));
    fs::write(&path, line).map_err(io_error(&path))?;
    Ok(path)
}

/// Writes the ssh config of a run into `dir` (created if needed), with
/// `endpoint` as the pod's address, and the `known_hosts` it points to. Rewritten before every connection,
/// since a pod reset changes its public port. Returns the config's path.
///
/// # Errors
///
/// Returns [`PodError::InvalidPath`] when a path cannot be written in an ssh
/// config, [`PodError::InvalidEndpoint`] when Runpod's endpoint holds characters
/// no host or user name has, and [`PodError::Io`] when a file cannot be written.
pub fn write_config(
    dir: &Path,
    alias: &str,
    endpoint: &SshEndpoint,
    keys: &PodKeys,
) -> Result<PathBuf, PodError> {
    fs::create_dir_all(dir).map_err(io_error(dir))?;
    let dir = std::path::absolute(dir).map_err(io_error(dir))?;
    let known_hosts = write_known_hosts(&dir, alias, &keys.host_public)?;
    let text = ssh_config(alias, endpoint, &keys.client_key, &known_hosts)?;
    let path = dir.join(SSH_CONFIG);
    fs::write(&path, text).map_err(io_error(&path))?;
    Ok(path)
}

/// The text of a run's ssh config.
///
/// # Errors
///
/// See [`write_config`].
pub fn ssh_config(
    alias: &str,
    endpoint: &SshEndpoint,
    identity: &Path,
    known_hosts: &Path,
) -> Result<String, PodError> {
    let host_ok = !endpoint.host.is_empty()
        && endpoint
            .host
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | ':' | '-'));
    let user_ok = !endpoint.user.is_empty()
        && endpoint
            .user
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    if !host_ok || !user_ok || endpoint.port == 0 {
        return Err(PodError::InvalidEndpoint(format!(
            "{}@{}:{}",
            endpoint.user.escape_debug(),
            endpoint.host.escape_debug(),
            endpoint.port
        )));
    }
    let lines = [
        format!("Host {alias}"),
        format!("  HostName {}", endpoint.host),
        format!("  Port {}", endpoint.port),
        format!("  User {}", endpoint.user),
        format!("  IdentityFile {}", config_path(identity)?),
        "  IdentitiesOnly yes".to_string(),
        "  IdentityAgent none".to_string(),
        format!("  UserKnownHostsFile {}", config_path(known_hosts)?),
        "  GlobalKnownHostsFile /dev/null".to_string(),
        format!("  HostKeyAlias {alias}"),
        "  CheckHostIP no".to_string(),
        "  StrictHostKeyChecking yes".to_string(),
        "  UpdateHostKeys no".to_string(),
        "  PasswordAuthentication no".to_string(),
        "  KbdInteractiveAuthentication no".to_string(),
    ];
    Ok(format!("{}\n", lines.join("\n")))
}

/// `path` as a double-quoted ssh config value, `%` doubled (ssh expands `%`
/// tokens in these options).
fn config_path(path: &Path) -> Result<String, PodError> {
    let text = path
        .to_str()
        .ok_or_else(|| PodError::InvalidPath(format!("{} is not valid UTF-8", path.display())))?;
    if text.contains(['"', '\n', '\r']) {
        return Err(PodError::InvalidPath(format!(
            "{} holds a double quote or a line break, which an ssh config cannot hold: move the project",
            text.escape_debug()
        )));
    }
    Ok(format!("\"{}\"", text.replace('%', "%%")))
}

/// `bytes` in standard base64, with padding and no line break.
#[must_use]
pub fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for (index, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            if index <= chunk.len() {
                let sextet = usize::try_from((n >> shift) & 63).unwrap_or(0);
                out.push(char::from(ALPHABET[sextet]));
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn io_error(path: &Path) -> impl FnOnce(io::Error) -> PodError + '_ {
    move |source| PodError::Io {
        path: path.to_path_buf(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use secrecy::ExposeSecret;

    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn base64_matches_the_rfc_vectors() {
        let vectors = [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ];
        for (plain, encoded) in vectors {
            assert_eq!(base64(plain.as_bytes()), encoded, "{plain}");
        }
        assert_eq!(base64(&[0xff, 0xfe, 0xfd]), "//79");
    }

    fn endpoint() -> SshEndpoint {
        SshEndpoint {
            host: "203.0.113.7".into(),
            port: 40122,
            user: "root".into(),
        }
    }

    #[test]
    fn the_config_pins_the_alias_and_ignores_the_agent() -> TestResult {
        let text = ssh_config(
            "overbrainer-r1",
            &endpoint(),
            Path::new("/p/a b%c/runs/r1/ssh/id_ed25519"),
            Path::new("/p/a b%c/runs/r1/ssh/known_hosts"),
        )?;
        assert_eq!(
            text,
            "Host overbrainer-r1\n  HostName 203.0.113.7\n  Port 40122\n  User root\n  IdentityFile \"/p/a b%%c/runs/r1/ssh/id_ed25519\"\n  IdentitiesOnly yes\n  IdentityAgent none\n  UserKnownHostsFile \"/p/a b%%c/runs/r1/ssh/known_hosts\"\n  GlobalKnownHostsFile /dev/null\n  HostKeyAlias overbrainer-r1\n  CheckHostIP no\n  StrictHostKeyChecking yes\n  UpdateHostKeys no\n  PasswordAuthentication no\n  KbdInteractiveAuthentication no\n"
        );
        Ok(())
    }

    #[test]
    fn unquotable_paths_and_odd_endpoints_are_refused() {
        let refused = ssh_config(
            "overbrainer-r1",
            &endpoint(),
            Path::new("/p/a\"b/id"),
            Path::new("/p/known_hosts"),
        );
        assert!(matches!(refused, Err(PodError::InvalidPath(_))));
        for bad in [
            SshEndpoint {
                host: "1.2.3.4\n  ProxyCommand x".into(),
                ..endpoint()
            },
            SshEndpoint {
                user: "root x".into(),
                ..endpoint()
            },
            SshEndpoint {
                port: 0,
                ..endpoint()
            },
        ] {
            let result = ssh_config("a", &bad, Path::new("/id"), Path::new("/kh"));
            assert!(
                matches!(result, Err(PodError::InvalidEndpoint(_))),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn known_hosts_pins_the_key_under_the_alias() -> TestResult {
        let dir = tempfile::tempdir()?;
        let path = write_known_hosts(dir.path(), "overbrainer-r1", "ssh-ed25519 AAAAkey comment")?;
        assert_eq!(
            fs::read_to_string(path)?,
            "overbrainer-r1 ssh-ed25519 AAAAkey\n"
        );
        Ok(())
    }

    fn keygen_available() -> bool {
        Command::new("ssh-keygen")
            .arg("-?")
            .stderr(Stdio::null())
            .stdout(Stdio::null())
            .status()
            .is_ok()
    }

    #[test]
    fn generated_keys_hold_together() -> TestResult {
        if !keygen_available() {
            eprintln!("skipped: ssh-keygen is not installed");
            return Ok(());
        }
        let root = tempfile::tempdir()?;
        let dir = root.path().join("ssh");
        let keys = PodKeys::generate(&dir, "overbrainer-r1")?;
        assert_eq!(fs::metadata(&dir)?.permissions().mode() & 0o777, 0o700);
        assert_eq!(
            fs::metadata(&keys.client_key)?.permissions().mode() & 0o777,
            0o600
        );
        assert!(keys.client_public.starts_with("ssh-ed25519 "));
        assert!(keys.client_public.ends_with(" overbrainer-r1"));
        assert!(keys.host_public.starts_with("ssh-ed25519 "));
        assert_eq!(keys.host_public.split_whitespace().count(), 2);
        assert!(!dir.join(HOST_KEY).exists());
        assert!(!format!("{keys:?}").contains(keys.host_private().expose_secret()));
        // Decoded the way the pod's bootstrap does, the private host key gives back
        // the pinned public key.
        let decoded = root.path().join("decoded");
        let output = Command::new("sh")
            .arg("-c")
            .arg("printf '%s' \"$KEY\" | base64 -d > \"$OUT\" && chmod 600 \"$OUT\" && ssh-keygen -y -f \"$OUT\"")
            .env("KEY", keys.host_private().expose_secret())
            .env("OUT", &decoded)
            .output()?;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            key_fields(&String::from_utf8(output.stdout)?),
            keys.host_public
        );
        Ok(())
    }
}
