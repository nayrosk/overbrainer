//! The OpenSSH transport: the user's `ssh` with `~/.ssh/config`, the agent and
//! `known_hosts`, one master connection carrying every command.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use ::openssh::{Child, Error, KnownHosts, Session, SessionBuilder, Stdio};

use super::{Pipes, RemoteProcess, Waiter};
use crate::exec::ssh::SshError;

/// A master connection of the user's `ssh` to one destination.
#[derive(Debug)]
pub struct OpenSshTransport {
    /// The session over the master's control socket.
    session: Arc<Session>,
    /// The destination, as given, for error messages.
    host: String,
    /// The master's `-E` log.
    log: PathBuf,
}

impl OpenSshTransport {
    /// Starts a master connection to `destination` (`user@host` or a
    /// `~/.ssh/config` alias), refusing unknown host keys. Tests pass
    /// `config_file` in place of `~/.ssh/config`.
    ///
    /// # Errors
    ///
    /// Returns the [`SshError`] the master failed with, [`SshError::HostKey`]
    /// when `ssh` refused the server's host key.
    pub async fn connect(destination: &str, config_file: Option<&Path>) -> Result<Self, SshError> {
        let mut builder = SessionBuilder::default();
        builder
            .known_hosts_check(KnownHosts::Strict)
            .connect_timeout(Duration::from_secs(30))
            .server_alive_interval(Duration::from_secs(15));
        if let Some(config_file) = config_file {
            builder.config_file(config_file);
        }
        // What `connect_mux` does, keeping the master's `-E` log path: `log` in
        // the control directory, the path `Session::detach` documents as the
        // "ssh multiplex output log".
        let (builder, resolved) = builder.resolve(destination);
        let control = builder
            .launch_master(resolved)
            .await
            .map_err(|error| map_error(&error, destination))?;
        let log = control.path().join("log");
        Ok(Self {
            session: Arc::new(Session::new_native_mux(control)),
            host: destination.to_string(),
            log,
        })
    }

    /// The master's own log, which says why it ended.
    pub fn master_log(&self) -> &Path {
        &self.log
    }

    /// Starts `command` with `sh -c` over the master (see [`super::Transport::exec`]).
    ///
    /// # Errors
    ///
    /// Returns the [`SshError`] that kept the command from starting.
    pub async fn exec(&self, command: &str, pipes: Pipes) -> Result<RemoteProcess, SshError> {
        let mut builder = Arc::clone(&self.session).arc_command("sh");
        builder
            .arg("-c")
            .arg(command)
            .stdin(stdio(pipes.stdin))
            .stdout(stdio(pipes.stdout))
            .stderr(stdio(pipes.stderr));
        let mut child = builder
            .spawn()
            .await
            .map_err(|error| map_error(&error, &self.host))?;
        let stdin = child
            .stdin()
            .take()
            .map(|pipe| Box::new(pipe) as Box<dyn tokio::io::AsyncWrite + Send + Unpin>);
        let stdout = child
            .stdout()
            .take()
            .map(|pipe| Box::new(pipe) as Box<dyn tokio::io::AsyncRead + Send + Unpin>);
        let stderr = child
            .stderr()
            .take()
            .map(|pipe| Box::new(pipe) as Box<dyn tokio::io::AsyncRead + Send + Unpin>);
        Ok(RemoteProcess {
            stdin,
            stdout,
            stderr,
            waiter: Waiter::OpenSsh(Box::new(OpenSshWaiter {
                child,
                host: self.host.clone(),
            })),
        })
    }
}

/// A command started over the master, waited on for its exit status.
pub(crate) struct OpenSshWaiter {
    /// The `ssh` multiplex child.
    child: Child<Arc<Session>>,
    /// The destination, for error messages.
    host: String,
}

impl OpenSshWaiter {
    /// The command's exit code.
    ///
    /// # Errors
    ///
    /// Returns [`SshError::Terminated`] when the command ended without an exit
    /// status, [`SshError::Disconnected`] when the master is gone.
    pub(crate) async fn wait(self) -> Result<i32, SshError> {
        let status = self
            .child
            .wait()
            .await
            .map_err(|error| map_error(&error, &self.host))?;
        status.code().ok_or(SshError::Terminated)
    }
}

/// A piped stream when `piped`, nothing otherwise.
fn stdio(piped: bool) -> Stdio {
    if piped { Stdio::piped() } else { Stdio::null() }
}

/// `error` from the `ssh` talking to `host` as an [`SshError`]. Only this module
/// knows `openssh`'s errors. `ssh` reports a refused host key only in its error
/// output, which `openssh` passes on as the error's source.
fn map_error(error: &Error, host: &str) -> SshError {
    match error {
        Error::Disconnected => SshError::Disconnected,
        Error::RemoteProcessTerminated => SshError::Terminated,
        _ => {
            let text = causes(error);
            if text.contains(HOST_KEY_REFUSED) {
                let reason = std::error::Error::source(error).map_or_else(|| text.clone(), causes);
                SshError::HostKey {
                    host: host.to_string(),
                    reason,
                }
            } else {
                SshError::Other(text)
            }
        },
    }
}

/// What OpenSSH's `ssh` prints when it refuses the server's host key, unknown
/// or changed.
const HOST_KEY_REFUSED: &str = "Host key verification failed";

/// `error` and its sources on one line, joined with `: `.
fn causes(error: &(dyn std::error::Error + 'static)) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(cause.to_string().trim());
        source = cause.source();
    }
    text
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;

    /// A failed master connection as `openssh` reports it: `ssh`'s error output
    /// in an I/O error.
    fn connect_error(stderr: &str) -> Error {
        Error::Connect(io::Error::new(io::ErrorKind::ConnectionAborted, stderr))
    }

    /// Each `openssh` error lands on its own `SshError`, keeping `ssh`'s words.
    #[test]
    fn openssh_errors_map_to_ssh_errors() {
        assert!(matches!(
            map_error(&Error::Disconnected, "h"),
            SshError::Disconnected
        ));
        assert!(matches!(
            map_error(&Error::RemoteProcessTerminated, "h"),
            SshError::Terminated
        ));
        let refused = map_error(&connect_error("Host key verification failed."), "pod");
        assert!(
            matches!(&refused, SshError::HostKey { host, reason }
                if host == "pod" && reason == "Host key verification failed."),
            "{refused:?}"
        );
        assert_eq!(
            refused.to_string(),
            "host key verification failed for pod: Host key verification failed."
        );
        let other = map_error(&connect_error("Connection refused"), "pod");
        assert_eq!(
            other.to_string(),
            "failed to connect to the remote host: Connection refused"
        );
        assert!(matches!(other, SshError::Other(_)), "{other:?}");
    }

    /// A changed host key, whose warning comes before the refusal, is a host key
    /// failure too.
    #[test]
    fn a_changed_host_key_is_a_host_key_failure() {
        let stderr = "@@@@@@@@\n@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @\n@@@@@@@@\nHost key verification failed.";
        let error = map_error(&connect_error(stderr), "pod");
        assert!(matches!(error, SshError::HostKey { .. }), "{error:?}");
    }
}
