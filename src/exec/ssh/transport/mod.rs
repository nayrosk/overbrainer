//! The client carrying an [`SshExecutor`](super::SshExecutor)'s commands: the
//! user's OpenSSH through a master connection, or the built-in client.

#[cfg(feature = "builtin-ssh")]
pub mod builtin;
mod openssh;

use tokio::io::{AsyncRead, AsyncWrite};

#[cfg(feature = "builtin-ssh")]
pub use self::builtin::BuiltinTransport;
pub use self::openssh::OpenSshTransport;
use super::SshError;

/// The SSH client a connection goes through.
#[derive(Debug)]
pub enum Transport {
    /// The user's `ssh`, through one master connection.
    OpenSsh(OpenSshTransport),
    /// The built-in client, one russh session.
    #[cfg(feature = "builtin-ssh")]
    Builtin(BuiltinTransport),
}

impl Transport {
    /// Starts `command` with the remote user's `sh -c`, opening the standard
    /// streams `pipes` asks for; the others read from or write to nothing.
    ///
    /// # Errors
    ///
    /// Returns the [`SshError`] that kept the command from starting.
    pub async fn exec(&self, command: &str, pipes: Pipes) -> Result<RemoteProcess, SshError> {
        match self {
            // Boxed: the `ssh` multiplex future is large, and every executor
            // future holding it would grow with it.
            Self::OpenSsh(transport) => Box::pin(transport.exec(command, pipes)).await,
            #[cfg(feature = "builtin-ssh")]
            Self::Builtin(transport) => transport.exec(command, pipes).await,
        }
    }
}

/// Which pipes [`Transport::exec`] opens.
#[derive(Debug, Clone, Copy)]
pub struct Pipes {
    /// The command's standard input.
    pub stdin: bool,
    /// The command's standard output.
    pub stdout: bool,
    /// The command's standard error.
    pub stderr: bool,
}

/// A started remote command.
pub struct RemoteProcess {
    /// Its standard input, when piped and not taken yet.
    pub stdin: Option<Box<dyn AsyncWrite + Send + Unpin>>,
    /// Its standard output, when piped and not taken yet.
    pub stdout: Option<Box<dyn AsyncRead + Send + Unpin>>,
    /// Its standard error, when piped and not taken yet.
    pub stderr: Option<Box<dyn AsyncRead + Send + Unpin>>,
    /// What waits for its end.
    pub(crate) waiter: Waiter,
}

/// The client-specific handle [`RemoteProcess::wait`] waits on.
pub(crate) enum Waiter {
    /// A command started through the OpenSSH master.
    OpenSsh(Box<self::openssh::OpenSshWaiter>),
    /// A command started on a channel of the built-in client.
    #[cfg(feature = "builtin-ssh")]
    Builtin(self::builtin::BuiltinWaiter),
}

impl RemoteProcess {
    /// Waits for the exit status, after closing the standard input when it is
    /// still held.
    ///
    /// # Errors
    ///
    /// Returns [`SshError::Terminated`] when the command ended without an exit
    /// status, and the [`SshError`] that broke the connection otherwise.
    pub async fn wait(mut self) -> Result<i32, SshError> {
        drop(self.stdin.take());
        match self.waiter {
            Waiter::OpenSsh(waiter) => Box::pin((*waiter).wait()).await,
            #[cfg(feature = "builtin-ssh")]
            Waiter::Builtin(waiter) => waiter.wait().await,
        }
    }
}
