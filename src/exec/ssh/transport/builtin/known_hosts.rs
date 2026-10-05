//! Host key checking for the built-in client against OpenSSH `known_hosts`
//! files: plain and hashed host patterns, the `[host]:port` form, a
//! `HostKeyAlias`, and the `@revoked` and `@cert-authority` markers.

use std::fs;
use std::io;
use std::path::PathBuf;

use hmac::{Hmac, KeyInit, Mac};
use russh::keys::PublicKey;
use russh::keys::ssh_key::known_hosts::{Entry, HostPatterns, Marker};
use sha1::Sha1;

use super::config::{DEFAULT_PORT, pattern_list_matches};
use crate::exec::ssh::SshError;

/// The reason for a host no file names.
const UNKNOWN: &str = "unknown host (add it with ssh-keyscan or a first connection with ssh)";
/// The reason for a host named with another key.
const CHANGED: &str = "the host key changed";
/// The reason for a key listed under `@revoked`.
const REVOKED: &str = "the host key is revoked";
/// The reason for a host named only by `@cert-authority` lines.
const CERT_AUTHORITY: &str = "certificate authorities are not supported by the built-in SSH client";

/// What a line naming the host says about the server key, in increasing
/// precedence: the strongest finding over all lines decides.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Finding {
    /// No line names the host.
    Unknown,
    /// An `@cert-authority` line names the host.
    CertAuthority,
    /// A plain line names the host with another key.
    Changed,
    /// A plain line carries the key.
    Accepted,
    /// An `@revoked` line carries the key: refused whatever else accepts it.
    Revoked,
}

/// Checks the server key `key` of `host` on `port` against the `known_hosts`
/// `files`, looking up `alias` instead of the host when set.
///
/// The lookup name is the alias as is, or the host for port 22, or
/// `[host]:port` for another port, compared without case. Missing files and
/// malformed lines are skipped, as OpenSSH does.
///
/// # Errors
///
/// Returns [`SshError::HostKey`] when the host is unknown, its key changed, the
/// key is revoked or only a certificate authority names the host, and
/// [`SshError::Other`] for a file that exists but cannot be read.
pub fn check(
    files: &[PathBuf],
    host: &str,
    port: u16,
    alias: Option<&str>,
    key: &PublicKey,
) -> Result<(), SshError> {
    let name = match alias {
        Some(alias) => alias.to_ascii_lowercase(),
        None if port == DEFAULT_PORT => host.to_ascii_lowercase(),
        None => format!("[{}]:{port}", host.to_ascii_lowercase()),
    };
    let mut finding = Finding::Unknown;
    for file in files {
        let text = match fs::read_to_string(file) {
            Ok(text) => text,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(SshError::Other(format!(
                    "cannot read {}: {error}",
                    file.display()
                )));
            },
        };
        for entry in text.lines().filter_map(parse_line) {
            if !names_host(entry.host_patterns(), &name) {
                continue;
            }
            let same_key = entry.public_key().key_data() == key.key_data();
            let line = match entry.marker() {
                Some(Marker::Revoked) if same_key => Finding::Revoked,
                Some(Marker::Revoked) => Finding::Unknown,
                Some(Marker::CertAuthority) => Finding::CertAuthority,
                None if same_key => Finding::Accepted,
                None => Finding::Changed,
            };
            finding = finding.max(line);
        }
    }
    let reason = match finding {
        Finding::Accepted => return Ok(()),
        Finding::Unknown => UNKNOWN,
        Finding::CertAuthority => CERT_AUTHORITY,
        Finding::Changed => CHANGED,
        Finding::Revoked => REVOKED,
    };
    Err(SshError::HostKey {
        host: host.to_string(),
        reason: reason.to_string(),
    })
}

/// The entry on a `known_hosts` line, or `None` for a blank, comment or
/// malformed line. Fields may be separated by any run of blanks.
fn parse_line(line: &str) -> Option<Entry> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    line.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .parse()
        .ok()
}

/// Whether `patterns` name the lowercase lookup `name`: a hashed name by its
/// HMAC-SHA1, a pattern list when a pattern matches and no negated one does.
fn names_host(patterns: &HostPatterns, name: &str) -> bool {
    match patterns {
        HostPatterns::HashedName { salt, hash } => Hmac::<Sha1>::new_from_slice(salt)
            .is_ok_and(|mac| mac.chain_update(name).verify_slice(hash).is_ok()),
        HostPatterns::Patterns(list) => pattern_list_matches(
            list.iter().map(|pattern| pattern.to_ascii_lowercase()),
            name,
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use hmac::{Hmac, KeyInit, Mac};
    use russh::keys::ssh_key::known_hosts::HostPatterns;
    use russh::keys::{Algorithm, PrivateKey, PublicKey};
    use sha1::Sha1;
    use tempfile::TempDir;

    use super::check;
    use crate::exec::ssh::SshError;

    /// What a test returns: `?` fails it with the error.
    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// The reason for a host no file names.
    const UNKNOWN: &str = "unknown host (add it with ssh-keyscan or a first connection with ssh)";
    /// The reason for a host named with another key.
    const CHANGED: &str = "the host key changed";
    /// The reason for a key listed under `@revoked`.
    const REVOKED: &str = "the host key is revoked";
    /// The reason for a host named only by `@cert-authority` lines.
    const CA: &str = "certificate authorities are not supported by the built-in SSH client";

    /// The public half of a test key generated by `/usr/bin/ssh-keygen -t ed25519`
    /// (test data, not a secret).
    const KEYGEN_KEY: &str =
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIE0ciO3An6y64oH3KTEOVx961eqCr6Q2Bos9QTWoQfru";
    /// `pod.example.test` with [`KEYGEN_KEY`], as hashed by `ssh-keygen -H`.
    const KEYGEN_HASHED: &str = "|1|lHWs7Bj8SyK4hlfsuAzMI8ObENc=|StehVpsvp2z8XTA70Ey6DWtGtFE= \
         ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIE0ciO3An6y64oH3KTEOVx961eqCr6Q2Bos9QTWoQfru";
    /// `[pod.example.test]:2222` with [`KEYGEN_KEY`], as hashed by `ssh-keygen -H`.
    const KEYGEN_HASHED_PORT: &str = "|1|vg2aifQz8iZcmygtMuJ5B/UoCyM=|elzjWepPLfo1C5Nnzon1SePoWvc= \
         ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIE0ciO3An6y64oH3KTEOVx961eqCr6Q2Bos9QTWoQfru";

    /// A fresh ed25519 public key.
    fn new_key() -> Result<PublicKey, Box<dyn std::error::Error>> {
        Ok(PrivateKey::random(&mut rand::rng(), Algorithm::Ed25519)?
            .public_key()
            .clone())
    }

    /// The `type base64` form of `key`, as a `known_hosts` line carries it.
    fn line_key(key: &PublicKey) -> Result<String, Box<dyn std::error::Error>> {
        Ok(key.to_openssh()?)
    }

    /// A temporary directory holding `known_hosts` files.
    struct Files {
        /// The temporary directory, removed on drop.
        dir: TempDir,
    }

    impl Files {
        /// Creates an empty directory.
        fn new() -> std::io::Result<Self> {
            Ok(Self {
                dir: TempDir::new()?,
            })
        }

        /// Writes `text` to the file `name` and returns its path.
        fn write(&self, name: &str, text: &str) -> std::io::Result<PathBuf> {
            let path = self.dir.path().join(name);
            fs::write(&path, text)?;
            Ok(path)
        }
    }

    /// The reason of a [`SshError::HostKey`], or `None` for any other result.
    fn reason(result: Result<(), SshError>) -> Option<String> {
        match result {
            Err(SshError::HostKey { reason, .. }) => Some(reason),
            _ => None,
        }
    }

    /// A plain entry with the server key accepts it.
    #[test]
    fn plain_entry_accepts() -> TestResult {
        let key = new_key()?;
        let files = Files::new()?;
        let file = files.write(
            "known_hosts",
            &format!("pod.example.test {}\n", line_key(&key)?),
        )?;
        assert!(
            check(&[file], "pod.example.test", 22, None, &key).is_ok(),
            "plain entry refused"
        );
        Ok(())
    }

    /// Host names compare without case and patterns support `*`, `?`, lists
    /// and negation.
    #[test]
    fn patterns_match_as_openssh() -> TestResult {
        let key = new_key()?;
        let files = Files::new()?;
        let text = format!(
            "other.test,*.example.test,!bad.example.test {k}\nnode?.lab {k}\n",
            k = line_key(&key)?
        );
        let file = files.write("known_hosts", &text)?;
        let paths = [file];
        assert!(
            check(&paths, "Pod.Example.Test", 22, None, &key).is_ok(),
            "wildcard refused"
        );
        assert!(
            check(&paths, "node7.lab", 22, None, &key).is_ok(),
            "question mark refused"
        );
        assert_eq!(
            reason(check(&paths, "bad.example.test", 22, None, &key)).as_deref(),
            Some(UNKNOWN),
            "negation ignored"
        );
        assert_eq!(
            reason(check(&paths, "node17.lab", 22, None, &key)).as_deref(),
            Some(UNKNOWN),
            "question mark matched two characters"
        );
        Ok(())
    }

    /// A hashed entry computed with HMAC-SHA1 over the host name matches.
    #[test]
    fn hashed_entry_accepts() -> TestResult {
        let key = new_key()?;
        let salt: [u8; 20] = rand::random();
        let mut mac = Hmac::<Sha1>::new_from_slice(&salt)?;
        mac.update(b"pod.example.test");
        let hash: [u8; 20] = mac.finalize().into_bytes().into();
        let patterns = HostPatterns::HashedName {
            salt: salt.to_vec(),
            hash,
        };
        let files = Files::new()?;
        let text = format!("{} {}\n", patterns.to_string(), line_key(&key)?);
        let file = files.write("known_hosts", &text)?;
        let paths = [file];
        assert!(
            check(&paths, "pod.example.test", 22, None, &key).is_ok(),
            "hashed entry refused"
        );
        assert_eq!(
            reason(check(&paths, "other.example.test", 22, None, &key)).as_deref(),
            Some(UNKNOWN),
            "hashed entry matched another host"
        );
        Ok(())
    }

    /// Lines written by `ssh-keygen -H` match, with and without a port.
    #[test]
    fn ssh_keygen_hashed_lines_accept() -> TestResult {
        let key: PublicKey = KEYGEN_KEY.parse()?;
        let files = Files::new()?;
        let file = files.write(
            "known_hosts",
            &format!("{KEYGEN_HASHED}\n{KEYGEN_HASHED_PORT}\n"),
        )?;
        let paths = [file];
        assert!(
            check(&paths, "pod.example.test", 22, None, &key).is_ok(),
            "ssh-keygen line refused"
        );
        assert!(
            check(&paths, "pod.example.test", 2222, None, &key).is_ok(),
            "ssh-keygen port line refused"
        );
        assert_eq!(
            reason(check(&paths, "pod.example.test", 2200, None, &key)).as_deref(),
            Some(UNKNOWN),
            "hashed line matched another port"
        );
        Ok(())
    }

    /// A port other than 22 looks up `[host]:port`; port 22 the bare host.
    #[test]
    fn port_form_is_used_off_port_22() -> TestResult {
        let key = new_key()?;
        let files = Files::new()?;
        let text = format!("[pod.example.test]:2222 {}\n", line_key(&key)?);
        let file = files.write("known_hosts", &text)?;
        let paths = [file];
        assert!(
            check(&paths, "pod.example.test", 2222, None, &key).is_ok(),
            "port entry refused"
        );
        assert_eq!(
            reason(check(&paths, "pod.example.test", 22, None, &key)).as_deref(),
            Some(UNKNOWN),
            "port entry matched port 22"
        );
        let bare = files.write("bare", &format!("pod.example.test {}\n", line_key(&key)?))?;
        assert_eq!(
            reason(check(&[bare], "pod.example.test", 2222, None, &key)).as_deref(),
            Some(UNKNOWN),
            "bare entry matched port 2222"
        );
        Ok(())
    }

    /// `HostKeyAlias` replaces the host name and port for the lookup.
    #[test]
    fn alias_replaces_the_host() -> TestResult {
        let key = new_key()?;
        let files = Files::new()?;
        let file = files.write("known_hosts", &format!("my-pod {}\n", line_key(&key)?))?;
        let paths = [file];
        assert!(
            check(&paths, "203.0.113.9", 40022, Some("my-pod"), &key).is_ok(),
            "alias entry refused"
        );
        assert_eq!(
            reason(check(&paths, "my-pod", 22, Some("other"), &key)).as_deref(),
            Some(UNKNOWN),
            "host name used despite the alias"
        );
        Ok(())
    }

    /// A host named with another key is a changed key, not an unknown host;
    /// the error names the host.
    #[test]
    fn mismatch_is_a_changed_key() -> TestResult {
        let known = new_key()?;
        let served = new_key()?;
        let files = Files::new()?;
        let file = files.write(
            "known_hosts",
            &format!("pod.example.test {}\n", line_key(&known)?),
        )?;
        let result = check(&[file], "pod.example.test", 22, None, &served);
        assert!(
            matches!(&result, Err(SshError::HostKey { host, .. }) if host == "pod.example.test"),
            "error does not name the host"
        );
        assert_eq!(
            reason(result).as_deref(),
            Some(CHANGED),
            "mismatch not reported as changed"
        );
        Ok(())
    }

    /// Another entry with the right key wins over a mismatching one.
    #[test]
    fn any_matching_entry_accepts() -> TestResult {
        let old = new_key()?;
        let key = new_key()?;
        let files = Files::new()?;
        let text = format!(
            "pod.example.test {}\npod.example.test {}\n",
            line_key(&old)?,
            line_key(&key)?
        );
        let file = files.write("known_hosts", &text)?;
        assert!(
            check(&[file], "pod.example.test", 22, None, &key).is_ok(),
            "second entry ignored"
        );
        Ok(())
    }

    /// `@revoked` with the server key refuses even when another line accepts it.
    #[test]
    fn revoked_refuses_over_an_accepting_line() -> TestResult {
        let key = new_key()?;
        let files = Files::new()?;
        let accept = files.write(
            "known_hosts",
            &format!("pod.example.test {}\n", line_key(&key)?),
        )?;
        let revoke = files.write("global", &format!("@revoked * {}\n", line_key(&key)?))?;
        assert_eq!(
            reason(check(&[accept, revoke], "pod.example.test", 22, None, &key)).as_deref(),
            Some(REVOKED),
            "revoked key accepted"
        );
        Ok(())
    }

    /// `@revoked` with another key leaves an accepting line alone.
    #[test]
    fn revoked_other_key_is_ignored() -> TestResult {
        let key = new_key()?;
        let other = new_key()?;
        let files = Files::new()?;
        let text = format!(
            "@revoked * {}\npod.example.test {}\n",
            line_key(&other)?,
            line_key(&key)?
        );
        let file = files.write("known_hosts", &text)?;
        assert!(
            check(&[file], "pod.example.test", 22, None, &key).is_ok(),
            "unrelated revocation refused"
        );
        Ok(())
    }

    /// `@cert-authority` alone names the host: refused with the CA reason;
    /// beside a plain entry it is ignored.
    #[test]
    fn cert_authority_only_refuses() -> TestResult {
        let ca = new_key()?;
        let key = new_key()?;
        let files = Files::new()?;
        let ca_line = format!("@cert-authority *.example.test {}\n", line_key(&ca)?);
        let only = files.write("only", &ca_line)?;
        assert_eq!(
            reason(check(
                std::slice::from_ref(&only),
                "pod.example.test",
                22,
                None,
                &key
            ))
            .as_deref(),
            Some(CA),
            "CA-only host not refused with the CA reason"
        );
        let plain = files.write("plain", &format!("pod.example.test {}\n", line_key(&key)?))?;
        assert!(
            check(&[only.clone(), plain], "pod.example.test", 22, None, &key).is_ok(),
            "CA line blocked a plain entry"
        );
        let changed = files.write("changed", &format!("pod.example.test {}\n", line_key(&ca)?))?;
        assert_eq!(
            reason(check(&[only, changed], "pod.example.test", 22, None, &key)).as_deref(),
            Some(CHANGED),
            "CA line hid a changed key"
        );
        Ok(())
    }

    /// Comments, blank lines, odd spacing and malformed lines are skipped.
    #[test]
    fn comments_blank_and_malformed_lines_are_skipped() -> TestResult {
        let key = new_key()?;
        let files = Files::new()?;
        let text = format!(
            "# a comment\n\n   \n@bogus pod.example.test {k}\npod.example.test\n\
             pod.example.test ssh-ed25519 !!!notbase64\n|1|bad|hash {k}\n\
             \t pod.example.test\t {k} laptop key\n",
            k = line_key(&key)?
        );
        let file = files.write("known_hosts", &text)?;
        assert!(
            check(&[file], "pod.example.test", 22, None, &key).is_ok(),
            "valid line after junk refused"
        );
        Ok(())
    }

    /// Missing files are skipped; with no file at all the host is unknown.
    #[test]
    fn missing_files_are_skipped() -> TestResult {
        let key = new_key()?;
        let files = Files::new()?;
        let missing = files.dir.path().join("absent");
        let file = files.write(
            "known_hosts",
            &format!("pod.example.test {}\n", line_key(&key)?),
        )?;
        assert!(
            check(&[missing.clone(), file], "pod.example.test", 22, None, &key).is_ok(),
            "missing file not skipped"
        );
        assert_eq!(
            reason(check(&[missing], "pod.example.test", 22, None, &key)).as_deref(),
            Some(UNKNOWN),
            "no file did not give an unknown host"
        );
        Ok(())
    }

    /// A file that exists but cannot be read is an error, not a skip.
    #[test]
    fn unreadable_file_is_an_error() -> TestResult {
        let key = new_key()?;
        let files = Files::new()?;
        let result = check(
            &[files.dir.path().to_path_buf()],
            "pod.example.test",
            22,
            None,
            &key,
        );
        assert!(
            matches!(result, Err(SshError::Other(_))),
            "unreadable file not reported"
        );
        Ok(())
    }

    /// The message carries the documented prefix and no key material.
    #[test]
    fn message_has_no_key_material() -> TestResult {
        let known = new_key()?;
        let served = new_key()?;
        let files = Files::new()?;
        let file = files.write(
            "known_hosts",
            &format!("pod.example.test {}\n", line_key(&known)?),
        )?;
        let message = check(&[file], "pod.example.test", 22, None, &served)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert_eq!(
            message, "host key verification failed for pod.example.test: the host key changed",
            "unexpected message"
        );
        Ok(())
    }
}
