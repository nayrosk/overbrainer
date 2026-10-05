/// Why an SSH transport failed, whichever client carries the connection. Messages
/// never hold key material.
#[derive(Debug, thiserror::Error)]
pub enum SshError {
    /// The connection could not be established.
    #[error("cannot connect to {host}: {reason}")]
    Connect {
        /// The destination, as given.
        host: String,
        /// Why the connection failed.
        reason: String,
    },
    /// The server's host key is unknown, changed or revoked.
    #[error("host key verification failed for {host}: {reason}")]
    HostKey {
        /// The destination, as given.
        host: String,
        /// What the client reported about the key.
        reason: String,
    },
    /// No key was accepted.
    #[error("authentication failed for {user}@{host}: {reason}")]
    Auth {
        /// The remote user.
        user: String,
        /// The destination, as given.
        host: String,
        /// Why every key was refused.
        reason: String,
    },
    /// A `ssh_config` directive the built-in client does not support.
    #[error(
        "{file}: {directive} for host {host} is not supported by the built-in SSH client: use ssh_client = \"openssh\", or a host entry without it"
    )]
    Unsupported {
        /// The directive, as written in the file.
        directive: String,
        /// The host entry it applies to.
        host: String,
        /// The `ssh_config` file holding it.
        file: String,
    },
    /// The connection was lost.
    #[error("the connection was terminated")]
    Disconnected,
    /// The remote command ended without an exit status.
    #[error("the remote process was terminated")]
    Terminated,
    /// Any other transport failure, with its causes on one line. [`ExecError::Ssh`]
    /// already says "ssh failed" in front of it.
    ///
    /// [`ExecError::Ssh`]: crate::exec::ExecError::Ssh
    #[error("{0}")]
    Other(String),
}
