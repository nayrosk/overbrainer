//! The pure-Rust SSH client behind the `builtin-ssh` feature: one russh session
//! per transport, carrying a channel per command.

pub mod auth;
pub mod config;
pub mod known_hosts;

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use russh::client::{self, DisconnectReason, Handle, Msg};
use russh::keys::{PublicKey, PublicKeyOrCertificate};
use russh::{ChannelMsg, ChannelReadHalf, ChannelWriteHalf};
use tokio::io::{AsyncWrite, AsyncWriteExt, DuplexStream};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use self::auth::AuthPlan;
use self::config::{ConfigSources, HostConfig};
use super::{Pipes, RemoteProcess, Waiter};
use crate::exec::quote;
use crate::exec::ssh::SshError;

/// How long the connection to a pod may take (D14).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// The keepalive interval of a pod connection (D14).
const ALIVE_INTERVAL: Duration = Duration::from_secs(15);
/// How many unanswered keepalives close a pod connection (D14).
const ALIVE_COUNT: u32 = 3;
/// Bytes a command's output may hold before its reader takes them.
const PIPE_BUFFER: usize = 64 * 1024;
/// The extended data type of standard error (RFC 4254, section 5.2).
const STDERR: u32 = 1;

/// How to reach a host without a config file (Runpod, D8).
#[derive(Debug, Clone)]
pub struct DirectTarget {
    /// The name the errors give the host, such as the run's alias.
    pub name: String,
    /// The address to connect to.
    pub host: String,
    /// The port to connect to.
    pub port: u16,
    /// The remote user.
    pub user: String,
    /// The only key file offered.
    pub key: PathBuf,
    /// The only host key accepted.
    pub host_key: PublicKey,
}

/// A russh session to one host.
pub struct BuiltinTransport {
    /// The session.
    handle: Handle<Client>,
    /// The destination, as given, for error messages.
    host: String,
    /// Set once the session ended.
    closed: Arc<AtomicBool>,
}

impl fmt::Debug for BuiltinTransport {
    /// The destination only: the session holds nothing worth printing.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BuiltinTransport")
            .field("host", &self.host)
            .finish_non_exhaustive()
    }
}

/// How the server's host key is checked.
#[derive(Debug, Clone)]
enum HostCheck {
    /// Only this key is accepted (a pod's pinned key).
    Pinned(PublicKey),
    /// The key must be in `known_hosts` (D12).
    KnownHosts {
        /// The `known_hosts` files, user files first.
        files: Vec<PathBuf>,
        /// The host name connected to.
        host: String,
        /// The port connected to.
        port: u16,
        /// The `HostKeyAlias`, looked up in place of the host.
        alias: Option<String>,
    },
}

/// Everything a connection needs, whichever way it was described.
struct Plan {
    /// The destination, as given, for error messages.
    name: String,
    /// The address to connect to.
    host: String,
    /// The port to connect to.
    port: u16,
    /// The remote user.
    user: String,
    /// How the host key is checked.
    check: HostCheck,
    /// The keys offered.
    auth: AuthPlan,
    /// How long the connection may take.
    connect_timeout: Duration,
    /// The interval between keepalive messages.
    alive_interval: Duration,
    /// How many unanswered keepalives close the connection.
    alive_count: u32,
}

impl BuiltinTransport {
    /// Connects to a pod (D8): its endpoint, its client key and its pinned
    /// host key, no config file and no agent.
    ///
    /// # Errors
    ///
    /// Returns [`SshError::HostKey`] when the server's key is not the pinned
    /// one, [`SshError::Auth`] when the key is refused, and
    /// [`SshError::Connect`] when the host cannot be reached in time.
    pub async fn connect_direct(target: &DirectTarget) -> Result<Self, SshError> {
        Self::connect(Plan {
            name: target.name.clone(),
            host: target.host.clone(),
            port: target.port,
            user: target.user.clone(),
            check: HostCheck::Pinned(target.host_key.clone()),
            auth: AuthPlan {
                agent: None,
                files: vec![target.key.clone()],
            },
            connect_timeout: CONNECT_TIMEOUT,
            alive_interval: ALIVE_INTERVAL,
            alive_count: ALIVE_COUNT,
        })
        .await
    }

    /// Connects to `destination` (`[user@]host` or an alias) as the files of
    /// `sources` describe it, the agent at `SSH_AUTH_SOCK` included unless the
    /// files say otherwise.
    ///
    /// # Errors
    ///
    /// Returns the [`SshError`] of [`config::resolve`], [`SshError::HostKey`]
    /// when `known_hosts` does not hold the server's key, [`SshError::Auth`]
    /// when no key is accepted, and [`SshError::Connect`] when the host cannot
    /// be reached in time.
    pub async fn connect_config(
        destination: &str,
        sources: &ConfigSources,
    ) -> Result<Self, SshError> {
        let host = config::resolve(destination, sources)?;
        if !direct(&host.proxy_jump) {
            return Err(SshError::Other("ProxyJump is not supported yet".into()));
        }
        let auth_sock = std::env::var_os("SSH_AUTH_SOCK")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from);
        Self::connect(config_plan(destination, &host, auth_sock.as_deref())).await
    }

    /// Opens the session of `plan`, checks the host key and authenticates.
    async fn connect(plan: Plan) -> Result<Self, SshError> {
        let verdict = Arc::new(Mutex::new(None));
        let closed = Arc::new(AtomicBool::new(false));
        let handler = Client {
            name: plan.name.clone(),
            check: plan.check.clone(),
            verdict: Arc::clone(&verdict),
            closed: Arc::clone(&closed),
        };
        let config = client::Config {
            keepalive_interval: Some(plan.alive_interval),
            keepalive_max: usize::try_from(plan.alive_count).unwrap_or(usize::MAX),
            ..client::Config::default()
        };
        let connecting =
            client::connect(Arc::new(config), (plan.host.as_str(), plan.port), handler);
        let connected = tokio::time::timeout(plan.connect_timeout, connecting).await;
        let refused = verdict.lock().ok().and_then(|mut slot| slot.take());
        let mut handle = match (connected, refused) {
            (_, Some(refused)) => return Err(refused),
            (Ok(Ok(handle)), None) => handle,
            (Ok(Err(error)), None) => {
                return Err(SshError::Connect {
                    host: plan.name,
                    reason: error.to_string(),
                });
            },
            (Err(_), None) => {
                return Err(SshError::Connect {
                    host: plan.name,
                    reason: format!("timed out after {}s", plan.connect_timeout.as_secs()),
                });
            },
        };
        auth::authenticate(&mut handle, &plan.user, &plan.auth)
            .await
            .map_err(|reason| SshError::Auth {
                user: plan.user.clone(),
                host: plan.name.clone(),
                reason,
            })?;
        Ok(Self {
            handle,
            host: plan.name,
            closed,
        })
    }

    /// Starts `command` with `sh -c` on a channel of its own (see
    /// [`super::Transport::exec`]).
    ///
    /// # Errors
    ///
    /// Returns [`SshError::Disconnected`] when the session is gone, and
    /// [`SshError::Other`] when the server refuses the channel or the command.
    pub async fn exec(&self, command: &str, pipes: Pipes) -> Result<RemoteProcess, SshError> {
        let channel = self
            .handle
            .channel_open_session()
            .await
            .map_err(|error| self.lost(&error))?;
        channel
            .exec(true, format!("sh -c {}", quote(command)))
            .await
            .map_err(|error| self.lost(&error))?;
        let (mut read, write) = channel.split();
        let mut early = Vec::new();
        loop {
            match read.wait().await {
                Some(ChannelMsg::Success) => break,
                Some(ChannelMsg::Failure) => {
                    return Err(SshError::Other(format!(
                        "{} refused to run the command",
                        self.host
                    )));
                },
                Some(other) => early.push(other),
                None => return Err(self.ended()),
            }
        }
        let (stdin, eof) = if pipes.stdin {
            let (signal, eof) = oneshot::channel();
            let stdin = ChannelStdin {
                writer: Box::pin(write.make_writer()),
                eof: Some(signal),
            };
            (
                Some(Box::new(stdin) as Box<dyn AsyncWrite + Send + Unpin>),
                Some(eof),
            )
        } else {
            write.eof().await.map_err(|error| self.lost(&error))?;
            (None, None)
        };
        let (stdout, out) = pipe(pipes.stdout);
        let (stderr, err) = pipe(pipes.stderr);
        let pump = Pump {
            read,
            write,
            out,
            err,
            eof,
            closed: Arc::clone(&self.closed),
        };
        let task = tokio::spawn(pump.run(early));
        Ok(RemoteProcess {
            stdin,
            stdout: stdout
                .map(|reader| Box::new(reader) as Box<dyn tokio::io::AsyncRead + Send + Unpin>),
            stderr: stderr
                .map(|reader| Box::new(reader) as Box<dyn tokio::io::AsyncRead + Send + Unpin>),
            waiter: Waiter::Builtin(BuiltinWaiter { task }),
        })
    }

    /// `error` from the session as an [`SshError`]: [`SshError::Disconnected`]
    /// once the session ended.
    fn lost(&self, error: &russh::Error) -> SshError {
        if self.handle.is_closed() || self.closed.load(Ordering::SeqCst) {
            SshError::Disconnected
        } else {
            SshError::Other(format!("{}: {error}", self.host))
        }
    }

    /// Why a channel ended before it was set up.
    fn ended(&self) -> SshError {
        if self.handle.is_closed() || self.closed.load(Ordering::SeqCst) {
            SshError::Disconnected
        } else {
            SshError::Other(format!("{} closed the channel of the command", self.host))
        }
    }
}

/// Whether `proxy_jump` asks for a direct connection: empty or `none`.
fn direct(proxy_jump: &[String]) -> bool {
    match proxy_jump {
        [] => true,
        [only] => only.eq_ignore_ascii_case("none"),
        _ => false,
    }
}

/// The connection plan of `destination` resolved to `host`, the agent at
/// `auth_sock` being the one of `SSH_AUTH_SOCK`.
fn config_plan(destination: &str, host: &HostConfig, auth_sock: Option<&Path>) -> Plan {
    Plan {
        name: destination.to_string(),
        host: host.host_name.clone(),
        port: host.port,
        user: host.user.clone(),
        check: HostCheck::KnownHosts {
            files: host.known_hosts.clone(),
            host: host.host_name.clone(),
            port: host.port,
            alias: host.host_key_alias.clone(),
        },
        auth: auth::plan(host, auth_sock),
        connect_timeout: host.connect_timeout,
        alive_interval: host.alive_interval,
        alive_count: host.alive_count,
    }
}

/// Whether `check` accepts the server key `key` of the host `name`.
///
/// # Errors
///
/// Returns [`SshError::HostKey`] for a key other than the pinned one, a key
/// `known_hosts` refuses, or a host certificate, and the [`SshError::Other`]
/// of an unreadable `known_hosts` file.
fn server_key_verdict(
    check: &HostCheck,
    name: &str,
    key: &PublicKeyOrCertificate,
) -> Result<(), SshError> {
    let PublicKeyOrCertificate::PublicKey { key, .. } = key else {
        return Err(SshError::HostKey {
            host: name.to_string(),
            reason: "host certificates are not supported by the built-in SSH client".into(),
        });
    };
    match check {
        HostCheck::Pinned(pinned) if pinned.key_data() == key.key_data() => Ok(()),
        HostCheck::Pinned(_) => Err(SshError::HostKey {
            host: name.to_string(),
            reason: "the host key is not the one pinned for this run".into(),
        }),
        HostCheck::KnownHosts {
            files,
            host,
            port,
            alias,
        } => known_hosts::check(files, host, *port, alias.as_deref(), key),
    }
}

/// The russh handler of a session: checks the host key and notes the end of
/// the session.
struct Client {
    /// The destination, as given, for error messages.
    name: String,
    /// How the host key is checked.
    check: HostCheck,
    /// Why the host key was refused, for the caller of `connect`.
    verdict: Arc<Mutex<Option<SshError>>>,
    /// Set once the session ended.
    closed: Arc<AtomicBool>,
}

impl client::Handler for Client {
    type Error = russh::Error;

    /// Accepts the key [`server_key_verdict`] accepts; otherwise keeps the
    /// reason for `connect` and refuses it.
    fn check_server_key(
        &mut self,
        server_public_key: &PublicKeyOrCertificate,
    ) -> impl Future<Output = Result<bool, Self::Error>> + Send {
        let accepted = match server_key_verdict(&self.check, &self.name, server_public_key) {
            Ok(()) => true,
            Err(refused) => {
                if let Ok(mut slot) = self.verdict.lock() {
                    *slot = Some(refused);
                }
                false
            },
        };
        std::future::ready(Ok(accepted))
    }

    /// Notes that the session ended, before its channels close.
    fn disconnected(
        &mut self,
        reason: DisconnectReason<Self::Error>,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send {
        self.closed.store(true, Ordering::SeqCst);
        std::future::ready(match reason {
            DisconnectReason::ReceivedDisconnect(_) => Ok(()),
            DisconnectReason::Error(error) => Err(error),
        })
    }
}

/// A reader for one output stream of a command and the pump's end of it, or
/// nothing when the stream is not piped.
fn pipe(piped: bool) -> (Option<DuplexStream>, Option<DuplexStream>) {
    if piped {
        let (reader, writer) = tokio::io::duplex(PIPE_BUFFER);
        (Some(reader), Some(writer))
    } else {
        (None, None)
    }
}

/// The standard input of a command: the channel's writer, which sends the
/// end of file on shutdown or, when dropped without one, through the pump.
struct ChannelStdin {
    /// The channel's data writer.
    writer: Pin<Box<dyn AsyncWrite + Send>>,
    /// Asks the pump for the end of file when dropped; gone once sent.
    eof: Option<oneshot::Sender<()>>,
}

impl AsyncWrite for ChannelStdin {
    /// Writes to the channel.
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.writer.as_mut().poll_write(cx, buf)
    }

    /// Flushes the channel's writer.
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.writer.as_mut().poll_flush(cx)
    }

    /// Sends the end of file, once.
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let done = self.writer.as_mut().poll_shutdown(cx);
        if matches!(done, Poll::Ready(Ok(()))) {
            self.eof = None;
        }
        done
    }
}

impl Drop for ChannelStdin {
    /// Asks the pump for the end of file unless shutdown already sent it.
    fn drop(&mut self) {
        if let Some(eof) = self.eof.take() {
            // The pump may be gone with its command: nothing left to close.
            let _ = eof.send(());
        }
    }
}

/// Moves a command's channel messages to its output pipes and keeps its exit
/// status.
struct Pump {
    /// The channel's incoming messages.
    read: ChannelReadHalf,
    /// The channel's sending side, for the end of file.
    write: ChannelWriteHalf<Msg>,
    /// The standard output pipe, until it ends or its reader is gone.
    out: Option<DuplexStream>,
    /// The standard error pipe, until it ends or its reader is gone.
    err: Option<DuplexStream>,
    /// The end of file request of the standard input, while expected.
    eof: Option<oneshot::Receiver<()>>,
    /// Set once the session ended.
    closed: Arc<AtomicBool>,
}

impl Pump {
    /// Runs until the channel closes, after handling the `early` messages
    /// received while the command was being started.
    async fn run(mut self, early: Vec<ChannelMsg>) -> Result<i32, SshError> {
        let mut outcome = Outcome::default();
        for message in early {
            if self.handle(message, &mut outcome).await {
                return outcome.result(&self.closed);
            }
        }
        loop {
            let message = match self.eof.as_mut() {
                Some(eof) => tokio::select! {
                    message = self.read.wait() => message,
                    sent = eof => {
                        self.eof = None;
                        if sent.is_ok() {
                            // A failure means the channel is gone: its end
                            // shows on the read side.
                            drop(self.write.eof().await);
                        }
                        continue;
                    },
                },
                None => self.read.wait().await,
            };
            let Some(message) = message else {
                break;
            };
            if self.handle(message, &mut outcome).await {
                break;
            }
        }
        self.out = None;
        self.err = None;
        outcome.result(&self.closed)
    }

    /// Handles one message: `true` once the channel is closed.
    async fn handle(&mut self, message: ChannelMsg, outcome: &mut Outcome) -> bool {
        match message {
            ChannelMsg::Data { data } => forward(&mut self.out, &data).await,
            ChannelMsg::ExtendedData { data, ext } if ext == STDERR => {
                forward(&mut self.err, &data).await;
            },
            ChannelMsg::ExitStatus { exit_status } => {
                outcome.status = Some(i32::try_from(exit_status).unwrap_or(i32::MAX));
            },
            ChannelMsg::ExitSignal { .. } => outcome.signalled = true,
            ChannelMsg::Eof => {
                self.out = None;
                self.err = None;
            },
            ChannelMsg::Close => return true,
            _ => {},
        }
        false
    }
}

/// Writes `data` to `pipe`, dropping the pipe once its reader is gone.
async fn forward(pipe: &mut Option<DuplexStream>, data: &[u8]) {
    if let Some(writer) = pipe
        && writer.write_all(data).await.is_err()
    {
        *pipe = None;
    }
}

/// How a command ended, as far as its channel said.
#[derive(Default)]
struct Outcome {
    /// Its exit status, when sent.
    status: Option<i32>,
    /// Whether it was killed by a signal.
    signalled: bool,
}

impl Outcome {
    /// The exit code, [`SshError::Terminated`] without one, or
    /// [`SshError::Disconnected`] when the session ended first.
    fn result(&self, closed: &AtomicBool) -> Result<i32, SshError> {
        match self.status {
            Some(code) if !self.signalled => Ok(code),
            _ if closed.load(Ordering::SeqCst) => Err(SshError::Disconnected),
            _ => Err(SshError::Terminated),
        }
    }
}

/// A command started on a channel, waited on for its exit status.
pub(crate) struct BuiltinWaiter {
    /// The pump of its channel.
    task: JoinHandle<Result<i32, SshError>>,
}

impl BuiltinWaiter {
    /// The command's exit code.
    ///
    /// # Errors
    ///
    /// Returns [`SshError::Terminated`] when the command ended without an exit
    /// status, [`SshError::Disconnected`] when the session is gone.
    pub(crate) async fn wait(self) -> Result<i32, SshError> {
        self.task
            .await
            .map_err(|_| SshError::Other("the reader of a remote command stopped".into()))?
    }
}

#[cfg(test)]
mod tests {
    use russh::keys::{Algorithm, PrivateKey};

    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// A fresh ed25519 public key.
    fn public() -> Result<PublicKey, Box<dyn std::error::Error>> {
        Ok(PrivateKey::random(&mut rand::rng(), Algorithm::Ed25519)?
            .public_key()
            .clone())
    }

    /// The server key as russh hands it to the handler.
    fn offered(key: &PublicKey) -> PublicKeyOrCertificate {
        PublicKeyOrCertificate::PublicKey {
            key: key.clone(),
            hash_alg: None,
        }
    }

    /// A pod accepts its pinned key only, and a refusal reads as a host key
    /// verification failure.
    #[test]
    fn a_pinned_key_accepts_only_itself() -> TestResult {
        let pinned = public()?;
        let check = HostCheck::Pinned(pinned.clone());
        assert!(server_key_verdict(&check, "pod", &offered(&pinned)).is_ok());
        let refused = server_key_verdict(&check, "pod", &offered(&public()?));
        let Err(error) = refused else {
            return Err("another key was accepted".into());
        };
        assert!(matches!(error, SshError::HostKey { .. }), "{error:?}");
        assert!(
            error
                .to_string()
                .starts_with("host key verification failed for pod: ")
        );
        Ok(())
    }

    /// A configured host defers to `known_hosts`: a listed key passes, an
    /// unknown host is refused.
    #[test]
    fn a_configured_host_checks_known_hosts() -> TestResult {
        let dir = tempfile::tempdir()?;
        let key = public()?;
        let file = dir.path().join("known_hosts");
        std::fs::write(&file, format!("[h.example]:2222 {}\n", key.to_openssh()?))?;
        let check = |host: &str| HostCheck::KnownHosts {
            files: vec![file.clone()],
            host: host.to_string(),
            port: 2222,
            alias: None,
        };
        assert!(server_key_verdict(&check("h.example"), "h", &offered(&key)).is_ok());
        let unknown = server_key_verdict(&check("other.example"), "o", &offered(&key));
        assert!(
            matches!(unknown, Err(SshError::HostKey { .. })),
            "{unknown:?}"
        );
        Ok(())
    }

    /// Only an empty `ProxyJump` or `none` connects directly.
    #[test]
    fn proxy_jump_none_is_direct() {
        assert!(direct(&[]));
        assert!(direct(&["none".to_string()]));
        assert!(!direct(&["bastion".to_string()]));
        assert!(!direct(&["a".to_string(), "b".to_string()]));
    }

    /// A command's end: its status, or terminated, or disconnected when the
    /// session ended without one.
    #[test]
    fn the_outcome_maps_to_status_terminated_or_disconnected() {
        let open = AtomicBool::new(false);
        let gone = AtomicBool::new(true);
        let exited = Outcome {
            status: Some(3),
            signalled: false,
        };
        assert!(matches!(exited.result(&open), Ok(3)));
        assert!(matches!(exited.result(&gone), Ok(3)));
        let silent = Outcome::default();
        assert!(matches!(silent.result(&open), Err(SshError::Terminated)));
        assert!(matches!(silent.result(&gone), Err(SshError::Disconnected)));
        let killed = Outcome {
            status: None,
            signalled: true,
        };
        assert!(matches!(killed.result(&open), Err(SshError::Terminated)));
    }

    /// A resolved host becomes a plan checking `known_hosts` under its alias,
    /// with its timeouts.
    #[test]
    fn a_resolved_host_becomes_its_plan() {
        let host = HostConfig {
            alias: "gpu".into(),
            host_name: "10.0.0.5".into(),
            user: "me".into(),
            port: 2200,
            identity_files: vec!["/k".into()],
            identities_only: true,
            identity_agent: config::AgentChoice::Env,
            known_hosts: vec!["/kh".into()],
            host_key_alias: Some("gpu-alias".into()),
            proxy_jump: Vec::new(),
            connect_timeout: Duration::from_secs(7),
            alive_interval: Duration::from_secs(9),
            alive_count: 2,
        };
        let plan = config_plan("me@gpu", &host, Some(Path::new("/sock")));
        assert_eq!(plan.name, "me@gpu");
        assert_eq!((plan.host.as_str(), plan.port), ("10.0.0.5", 2200));
        assert_eq!(plan.user, "me");
        assert!(matches!(
            &plan.check,
            HostCheck::KnownHosts { alias: Some(alias), port: 2200, .. } if alias == "gpu-alias"
        ));
        assert_eq!(plan.auth.agent, None);
        assert_eq!(plan.connect_timeout, Duration::from_secs(7));
        assert_eq!(plan.alive_interval, Duration::from_secs(9));
        assert_eq!(plan.alive_count, 2);
    }
}
