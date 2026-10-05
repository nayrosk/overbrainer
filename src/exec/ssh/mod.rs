mod error;
mod transport;

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;

use secrecy::{ExposeSecret, SecretString};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::process::Child;

pub use self::error::SshError;
#[cfg(feature = "builtin-ssh")]
use self::transport::builtin::config::ConfigSources;
use self::transport::{OpenSshTransport, Pipes, RemoteProcess, Transport};
use super::tar;
use super::{
    CANCEL_FILE, CANCELLING_FILE, EXIT_FILE, ExecError, Executor, FileDigest, JOB_LOG, JobCommand,
    JobId, JobStatus, PID_FILE, Pid, cancel_script, check_job_env, claim_script, job_script,
    manifest_script, parse_manifest, parse_status, quote, shell_path, status_script,
};
use crate::config::SshClient;

/// Runs jobs on a remote Linux machine through the user's `ssh` or the built-in
/// client: `~/.ssh/config`, the agent and `known_hosts` apply, and unknown host
/// keys are refused. One connection (an OpenSSH master, or one built-in
/// session) carries every command.
///
/// A job's secrets travel on the standard input of the command starting it and are
/// read into the job's environment there: they never appear on a command line,
/// remote or local, in a file or in an error.
///
/// A job starts with whatever environment the remote non-interactive `sh -c` gives
/// it: the SSH server's environment for that user, without a login shell and
/// without any interactive startup file, plus the job's secrets. Nothing of
/// overbrainer's own environment crosses the connection. A `native` target whose
/// `axolotl` relies on a shell setup, such as a conda activation or a `PATH` entry
/// written by a profile, therefore needs that setup in the job's own commands or in
/// the server's environment; the same target run locally would inherit it from the
/// caller.
///
/// The user `destination` logs in as must own the jobs it checks and cancels: status
/// and cancel signal the job's process group with `kill`. When that group belongs to
/// another user (for example a job started as root, or inside a rootful container
/// whose processes run as another user on the host), `kill -s 0` fails with a
/// permission error, which the status script cannot tell apart from a group that is
/// gone. Such a job reads [`JobStatus::Lost`] while it is really running, and a
/// cancel cannot stop it.
#[derive(Debug)]
pub struct SshExecutor {
    transport: Transport,
    workdir: String,
}

impl SshExecutor {
    /// Connects to `destination` with `client` and creates `workdir` there,
    /// relative to the remote home unless absolute.
    ///
    /// # Errors
    ///
    /// Returns [`ExecError::Ssh`] when the connection fails, for example on an
    /// unknown host key, [`ExecError::MasterDied`], with the tail of the master's
    /// log, when an OpenSSH master connection ends right after it started, and
    /// [`ExecError::Command`] when `workdir` cannot be created.
    pub async fn connect(
        destination: &SshDestination<'_>,
        workdir: &str,
        client: SshClient,
    ) -> Result<Self, ExecError> {
        let transport = open(destination, client).await.map_err(ExecError::Ssh)?;
        let dir = shell_path(workdir);
        let resolved = run(
            &transport,
            &format!("mkdir -p -- {dir} && cd -- {dir} && pwd -P"),
            "create the work directory",
        )
        .await
        .map_err(|error| first_failure(&transport, error))?;
        let workdir = String::from_utf8_lossy(&resolved).trim().to_string();
        if !workdir.starts_with('/') {
            return Err(ExecError::Protocol(format!(
                "the work directory resolved to `{workdir}`"
            )));
        }
        Ok(Self { transport, workdir })
    }

    /// Starts `job` detached on the target, its secrets fed on the launcher's
    /// standard input, and returns its process ID.
    async fn start(&self, job: &JobCommand) -> Result<JobId, ExecError> {
        let launcher = launcher(job)?;
        let all = Pipes {
            stdin: true,
            stdout: true,
            stderr: true,
        };
        let mut process = self
            .transport
            .exec(&launcher, all)
            .await
            .map_err(ExecError::Ssh)?;
        let stdin = process.stdin.take();
        // Fed while the output is read, so a launcher failing before it has read
        // every secret reports its own error rather than a broken pipe.
        let (fed, output) = tokio::join!(feed_secrets(stdin, &job.secrets), output(process));
        let output = output.map_err(ExecError::Ssh)?;
        if !output.status.success() {
            return Err(ExecError::Command {
                action: "start the job",
                message: tar::failure(&output.stderr, output.status),
            });
        }
        fed.map_err(|error| remote_io(&error))?;
        let text = String::from_utf8_lossy(&output.stdout);
        let raw: u32 = text
            .trim()
            .parse()
            .map_err(|_| ExecError::Protocol(format!("`{}` is not a process ID", text.trim())))?;
        Ok(JobId {
            dir: job.dir.clone(),
            pid: Pid::new(raw)?,
            container: job.container.clone(),
        })
    }

    /// Copies `local`, without the `skip` entries, into `remote` through a `tar`
    /// stream.
    async fn send(&self, local: &Path, remote: &str, skip: &[String]) -> Result<(), ExecError> {
        // Listed up front: a local `tar` that cannot even open the directory would
        // send an empty stream, and the remote `tar` complaining about that would hide
        // the real cause.
        let entries = tar::upload_entries(local, skip)?;
        let dir = quote(remote);
        if entries.is_empty() {
            run(&self.transport, &format!("mkdir -p -- {dir}"), "upload").await?;
            return Ok(());
        }
        let pipes = Pipes {
            stdin: true,
            stdout: false,
            stderr: true,
        };
        let mut process = self
            .transport
            .exec(&format!("mkdir -p -- {dir} && tar -C {dir} -xf -"), pipes)
            .await
            .map_err(ExecError::Ssh)?;
        let create = tar::spawn_create(local, &entries, &[])?;
        let (copied, errors, created) =
            transfer(create, process.stdin.take(), process.stderr.take()).await;
        let received = process
            .wait()
            .await
            .map(exit_status)
            .map_err(ExecError::Ssh);
        upload_outcome(received, &errors, created, copied)
    }

    /// Copies the `entries` of `remote`, without `exclude`, into `local` through a
    /// `tar` stream.
    async fn fetch(
        &self,
        remote: &str,
        local: &Path,
        entries: &[String],
        exclude: &[String],
    ) -> Result<(), ExecError> {
        let pipes = Pipes {
            stdin: false,
            stdout: true,
            stderr: true,
        };
        let mut process = self
            .transport
            .exec(&archive_script(remote, entries, exclude), pipes)
            .await
            .map_err(ExecError::Ssh)?;
        let (extracted, errors) =
            receive(process.stdout.take(), process.stderr.take(), local).await;
        let sent = process
            .wait()
            .await
            .map(exit_status)
            .map_err(ExecError::Ssh);
        download_outcome(extracted, sent, &errors)
    }
}

impl Executor for SshExecutor {
    fn workdir(&self) -> &str {
        &self.workdir
    }

    async fn claim(&self, dir: &str, owner: &str) -> Result<bool, ExecError> {
        let script = claim_script(dir, owner);
        let output = run(&self.transport, &script, "claim the run directory").await?;
        match String::from_utf8_lossy(&output).trim() {
            "claimed" => Ok(true),
            "taken" => Ok(false),
            other => Err(ExecError::Protocol(format!(
                "`{other}` is not an answer to a claim"
            ))),
        }
    }

    async fn upload(&self, local: &Path, remote: &str, skip: &[String]) -> Result<(), ExecError> {
        self.send(local, remote, skip).await
    }

    async fn spawn(&self, job: &JobCommand) -> Result<JobId, ExecError> {
        self.start(job).await
    }

    async fn read_from(&self, path: &str, offset: u64, limit: u64) -> Result<Vec<u8>, ExecError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        run(&self.transport, &read_script(path, offset, limit), "read").await
    }

    async fn status(&self, job: &JobId) -> Result<JobStatus, ExecError> {
        let script = status_script(&job.dir, job.pid.get());
        let output = run(&self.transport, &script, "status").await?;
        let text = String::from_utf8_lossy(&output);
        parse_status(&text)
            .ok_or_else(|| ExecError::Protocol(format!("unknown job status {:?}", text.trim())))
    }

    async fn cancel(&self, job: &JobId) -> Result<(), ExecError> {
        let script = cancel_script(&job.dir, job.pid.get(), job.container.as_ref());
        run(&self.transport, &script, "cancel").await.map(drop)
    }

    async fn download(
        &self,
        remote: &str,
        local: &Path,
        entries: &[String],
        exclude: &[String],
    ) -> Result<(), ExecError> {
        self.fetch(remote, local, entries, exclude).await
    }

    async fn manifest(
        &self,
        remote: &str,
        entries: &[String],
        exclude: &[String],
    ) -> Result<Vec<FileDigest>, ExecError> {
        let script = manifest_script(remote, entries, exclude);
        let output = run(&self.transport, &script, "manifest").await?;
        parse_manifest(&String::from_utf8_lossy(&output))
    }

    async fn probe(&self, script: &str) -> Result<Vec<u8>, ExecError> {
        run(&self.transport, script, "probe").await
    }

    async fn put_file(&self, path: &str, content: &str) -> Result<(), ExecError> {
        run(&self.transport, &put_script(path, content), "write a file")
            .await
            .map(drop)
    }
}

/// Where [`SshExecutor::connect`] goes.
#[derive(Debug, Clone, Copy)]
pub enum SshDestination<'a> {
    /// `user@host` or an alias of the user's ssh configuration.
    Config {
        /// The destination, as given.
        destination: &'a str,
        /// A file read in place of `~/.ssh/config` and the system file, as
        /// `ssh -F` does (tests pass one).
        config_file: Option<&'a Path>,
    },
    /// A Runpod pod, reached without the user's configuration.
    Pod(&'a PodEndpoint),
}

/// What reaches a pod (D6, D8): its endpoint, the run's client key, its pinned
/// host key, and the per-run config the OpenSSH client reads.
#[derive(Debug, Clone)]
pub struct PodEndpoint {
    /// The run's host alias, also its `HostKeyAlias`.
    pub alias: String,
    /// The per-run ssh config defining `alias`, read by OpenSSH.
    pub config: PathBuf,
    /// The pod's public address.
    pub host: String,
    /// The pod's public SSH port.
    pub port: u16,
    /// The remote user.
    pub user: String,
    /// The run's client private key file.
    pub key: PathBuf,
    /// The pod's pinned public host key, `ssh-ed25519 AAAA...`.
    pub host_key: String,
}

/// Opens the transport of `client` to `destination`.
///
/// # Errors
///
/// Returns the [`SshError`] of the connection, and [`SshError::Other`] for
/// `builtin` in a build without the `builtin-ssh` feature.
async fn open(destination: &SshDestination<'_>, client: SshClient) -> Result<Transport, SshError> {
    match (client, destination) {
        (
            SshClient::Openssh,
            SshDestination::Config {
                destination,
                config_file,
            },
        ) => Ok(Transport::OpenSsh(
            OpenSshTransport::connect(destination, *config_file).await?,
        )),
        (SshClient::Openssh, SshDestination::Pod(pod)) => Ok(Transport::OpenSsh(
            OpenSshTransport::connect(&pod.alias, Some(&pod.config)).await?,
        )),
        #[cfg(feature = "builtin-ssh")]
        (
            SshClient::Builtin,
            SshDestination::Config {
                destination,
                config_file,
            },
        ) => {
            let sources = config_sources(*config_file, &LocalEnv::read())?;
            // Boxed, as the russh handshake future is large.
            Ok(Transport::Builtin(
                Box::pin(transport::BuiltinTransport::connect_config(
                    destination,
                    &sources,
                ))
                .await?,
            ))
        },
        #[cfg(feature = "builtin-ssh")]
        (SshClient::Builtin, SshDestination::Pod(pod)) => Ok(Transport::Builtin(
            Box::pin(transport::BuiltinTransport::connect_direct(&direct_target(
                pod,
            )?))
            .await?,
        )),
        #[cfg(not(feature = "builtin-ssh"))]
        (SshClient::Builtin, _) => Err(SshError::Other(
            crate::config::BUILTIN_SSH_REFUSED.to_string(),
        )),
    }
}

/// The built-in client's view of `pod`.
///
/// # Errors
///
/// Returns [`SshError::HostKey`] when the pinned host key cannot be read.
#[cfg(feature = "builtin-ssh")]
fn direct_target(pod: &PodEndpoint) -> Result<transport::builtin::DirectTarget, SshError> {
    let host_key = russh::keys::PublicKey::from_openssh(pod.host_key.trim()).map_err(|_| {
        SshError::HostKey {
            host: pod.alias.clone(),
            reason: "the host key pinned for this run cannot be read".into(),
        }
    })?;
    Ok(transport::builtin::DirectTarget {
        name: pod.alias.clone(),
        host: pod.host.clone(),
        port: pod.port,
        user: pod.user.clone(),
        key: pod.key.clone(),
        host_key,
    })
}

/// What the built-in client reads of the local environment.
#[cfg(feature = "builtin-ssh")]
#[derive(Debug, Default)]
struct LocalEnv {
    /// `HOME`.
    home: Option<std::ffi::OsString>,
    /// `USER`.
    user: Option<std::ffi::OsString>,
    /// `LOGNAME`.
    logname: Option<std::ffi::OsString>,
}

#[cfg(feature = "builtin-ssh")]
impl LocalEnv {
    /// The variables of this process.
    fn read() -> Self {
        Self {
            home: std::env::var_os("HOME"),
            user: std::env::var_os("USER"),
            logname: std::env::var_os("LOGNAME"),
        }
    }
}

/// The ssh configuration files the built-in client reads: `config_file` alone
/// when given (as `ssh -F`), otherwise `~/.ssh/config` then
/// `/etc/ssh/ssh_config`; the home and the local user from `env`.
///
/// # Errors
///
/// Returns [`SshError::Other`] when `HOME`, or both `USER` and `LOGNAME`, are
/// unset or empty: the client does not guess them.
#[cfg(feature = "builtin-ssh")]
fn config_sources(config_file: Option<&Path>, env: &LocalEnv) -> Result<ConfigSources, SshError> {
    let home = env
        .home
        .as_ref()
        .filter(|home| !home.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| {
            SshError::Other(
                "HOME is not set: the built-in SSH client needs it to find ~/.ssh".into(),
            )
        })?;
    let local_user = [&env.user, &env.logname]
        .into_iter()
        .flatten()
        .filter_map(|name| name.to_str())
        .find(|name| !name.is_empty())
        .map(str::to_string)
        .ok_or_else(|| {
            SshError::Other(
                "neither USER nor LOGNAME is set: the built-in SSH client needs the local user name"
                    .into(),
            )
        })?;
    let files = match config_file {
        Some(file) => vec![file.to_path_buf()],
        None => vec![
            home.join(".ssh").join("config"),
            PathBuf::from("/etc/ssh/ssh_config"),
        ],
    };
    Ok(ConfigSources {
        files,
        home,
        local_user,
    })
}

/// Replaces `path` with `content` through a sibling temporary file of the
/// remote shell's own (`<path>.<pid>.tmp`, never shared with a concurrent
/// writer such as the pod's watchdog), creating its directory first. The
/// content travels quoted on the command line: it is never a secret.
fn put_script(path: &str, content: &str) -> String {
    let dir = path.rsplit_once('/').map_or(".", |(dir, _)| dir);
    let dir = if dir.is_empty() { "/" } else { dir };
    let tmp = format!("{}.$$.tmp", quote(path));
    format!(
        "mkdir -p -- {dir} && printf '%s' {content} > {tmp} && mv -f {tmp} {path}",
        dir = quote(dir),
        content = quote(content),
        path = quote(path)
    )
}

/// The command starting `job` detached from the SSH session, printing its process ID.
/// It first reads one line per secret from its standard input into the environment
/// variable of that name: secret values are never written here.
///
/// # Errors
///
/// Returns [`ExecError::InvalidSecret`] when a secret cannot be passed to a job
/// (see [`check_job_env`]).
fn launcher(job: &JobCommand) -> Result<String, ExecError> {
    check_job_env(&job.secrets)?;
    let mut parts = Vec::new();
    for (name, _) in &job.secrets {
        parts.push(format!("IFS= read -r {name} && export {name}"));
    }
    parts.push(format!(
        "mkdir -p -- {dir} && cd -- {dir} && rm -f {EXIT_FILE} {CANCELLING_FILE} {CANCEL_FILE} {PID_FILE} && \
         {{ setsid nohup sh -c {inner} > {JOB_LOG} 2>&1 < /dev/null & }} && echo \"$!\"",
        dir = quote(&job.dir),
        inner = quote(&job_script(&job.script)),
    ));
    Ok(parts.join(" && "))
}

/// Writes each secret on a line of its own to `stdin`, then closes it.
async fn feed_secrets<W: AsyncWrite + Unpin>(
    stdin: Option<W>,
    secrets: &[(String, SecretString)],
) -> io::Result<()> {
    let Some(mut stdin) = stdin else {
        return Err(io::Error::other("the launcher has no standard input"));
    };
    for (_, value) in secrets {
        stdin.write_all(value.expose_secret().as_bytes()).await?;
        stdin.write_all(b"\n").await?;
    }
    stdin.shutdown().await
}

/// Prints at most `limit` bytes of `path` from byte `offset`; a missing file prints
/// nothing.
fn read_script(path: &str, offset: u64, limit: u64) -> String {
    format!(
        "[ -f {path} ] || exit 0\ntail -c +{start} -- {path} | head -c {limit}\n",
        path = quote(path),
        start = offset.saturating_add(1)
    )
}

/// Writes to stdout a `tar` archive of the `entries` of `remote` that exist, without
/// the names matching an `exclude` pattern, or nothing when none exists. Fails, saying
/// so, when `remote` itself is not a directory.
fn archive_script(remote: &str, entries: &[String], exclude: &[String]) -> String {
    let candidates: Vec<String> = entries.iter().map(|entry| quote(entry)).collect();
    let args: Vec<String> = tar::create_args(".", &[], exclude)
        .iter()
        .map(|arg| quote(arg))
        .collect();
    format!(
        "[ -d {dir} ] || {{ printf '%s does not exist\\n' {dir} >&2; exit 1; }}\ncd -- {dir} || exit 1\nset --\nfor entry in {candidates}; do [ -e \"$entry\" ] && set -- \"$@\" \"$entry\"; done\n[ \"$#\" -gt 0 ] || exit 0\nexec tar {args} \"$@\"\n",
        dir = quote(remote),
        candidates = candidates.join(" "),
        args = args.join(" "),
    )
}

/// Streams the archive of the local `tar` `create` into `sink`, closing it at the
/// end, while reading the receiver's error output `errors` and `create`'s own, all at
/// once so no error pipe can fill up and stall its writer. Returns the result of the
/// copy, the receiver's error output and the result of `create`.
async fn transfer<W, E>(
    mut create: Child,
    sink: Option<W>,
    errors: Option<E>,
) -> (io::Result<()>, Vec<u8>, Result<(), ExecError>)
where
    W: AsyncWrite + Unpin,
    E: AsyncRead + Unpin,
{
    let source = create.stdout.take();
    // Both ends are dropped as soon as the copy stops, so a receiver that dies early
    // makes the local `tar` fail on a broken pipe instead of blocking forever.
    let copy = async move {
        let (Some(mut source), Some(mut sink)) = (source, sink) else {
            return Err(io::Error::other("a transfer pipe is missing"));
        };
        tokio::io::copy(&mut source, &mut sink).await?;
        sink.shutdown().await
    };
    tokio::join!(copy, drain(errors), tar::finish(create, "archive"))
}

/// Extracts the archive read from `archive` into `local` while reading the sender's
/// error output `errors`, both at once. Returns the extraction's result and that
/// output.
async fn receive<R, E>(
    archive: Option<R>,
    errors: Option<E>,
    local: &Path,
) -> (Result<(), ExecError>, Vec<u8>)
where
    R: AsyncRead + Unpin,
    E: AsyncRead + Unpin,
{
    // `archive` is dropped once the extraction ends, early or not, so a sender still
    // writing sees a broken pipe instead of blocking forever.
    let extract = async move {
        match archive {
            Some(mut archive) => tar::extract(&mut archive, local).await,
            None => Err(ExecError::Protocol("the remote tar has no output".into())),
        }
    };
    tokio::join!(extract, drain(errors))
}

/// Everything `errors` holds until its end. A failed read only loses diagnostics.
async fn drain<E: AsyncRead + Unpin>(errors: Option<E>) -> Vec<u8> {
    let mut bytes = Vec::new();
    if let Some(mut errors) = errors
        && errors.read_to_end(&mut bytes).await.is_err()
    {
        bytes.extend_from_slice(b"\n(the error output could not be read to its end)");
    }
    bytes
}

/// The result of an upload. The receiving `tar`'s own failure comes first: when it
/// dies early, the local `tar` and the copy only fail on a broken pipe. A receiver
/// killed by a signal has no exit status: its error output, when it printed any,
/// says more than a bare [`SshError::Terminated`].
fn upload_outcome(
    received: Result<ExitStatus, ExecError>,
    errors: &[u8],
    created: Result<(), ExecError>,
    copied: io::Result<()>,
) -> Result<(), ExecError> {
    let status = match received {
        Err(ExecError::Ssh(SshError::Terminated)) => {
            return Err(tar::error_text(errors).map_or(
                ExecError::Ssh(SshError::Terminated),
                |message| ExecError::Command {
                    action: "upload",
                    message,
                },
            ));
        },
        other => other?,
    };
    if !status.success() {
        return Err(ExecError::Command {
            action: "upload",
            message: tar::failure(errors, status),
        });
    }
    created?;
    copied.map_err(|error| remote_io(&error))
}

/// The result of a download. The local extraction's failure comes first: when it
/// dies early, the remote `tar` only fails on a broken pipe.
fn download_outcome(
    extracted: Result<(), ExecError>,
    sent: Result<ExitStatus, ExecError>,
    errors: &[u8],
) -> Result<(), ExecError> {
    extracted?;
    let status = sent?;
    if status.success() {
        Ok(())
    } else {
        Err(ExecError::Command {
            action: "download",
            message: tar::failure(errors, status),
        })
    }
}

/// `error` of the first command on the new connection `transport`: with the
/// OpenSSH client, [`master_died`] reads the master's log; the built-in client
/// has no master, and its error stays as is.
fn first_failure(transport: &Transport, error: ExecError) -> ExecError {
    match transport {
        Transport::OpenSsh(openssh) => master_died(error, openssh.master_log()),
        #[cfg(feature = "builtin-ssh")]
        Transport::Builtin(_) => error,
    }
}

/// `error` of the first command on a new session, as [`ExecError::MasterDied`]
/// with the tail of the master's log at `log` when the master was gone.
fn master_died(error: ExecError, log: &Path) -> ExecError {
    match error {
        ExecError::Ssh(SshError::Disconnected) => ExecError::MasterDied {
            log: master_log(log),
        },
        other => other,
    }
}

/// Bytes read from the end of the master's log.
const LOG_READ: u64 = 4096;
/// Lines kept of the master's log.
const LOG_LINES: usize = 5;
/// Characters kept of the master's log, `...` included.
const LOG_CHARS: usize = 300;

/// The tail of the master's log at `path` (see [`log_tail`]), empty when it cannot
/// be read. Only its last [`LOG_READ`] bytes are read.
fn master_log(path: &Path) -> String {
    let read = || -> io::Result<Vec<u8>> {
        let mut file = File::open(path)?;
        let len = file.metadata()?.len();
        file.seek(SeekFrom::Start(len.saturating_sub(LOG_READ)))?;
        let mut bytes = Vec::new();
        file.take(LOG_READ).read_to_end(&mut bytes)?;
        Ok(bytes)
    };
    read().map_or_else(
        |_| String::new(),
        |bytes| log_tail(&String::from_utf8_lossy(&bytes)),
    )
}

/// The last [`LOG_LINES`] non-blank lines of `text`, control characters dropped,
/// joined with ` | ` and cut from the start to [`LOG_CHARS`] characters.
fn log_tail(text: &str) -> String {
    let lines: Vec<String> = text
        .lines()
        .map(|line| line.chars().filter(|c| !c.is_control()).collect::<String>())
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();
    let joined = lines[lines.len().saturating_sub(LOG_LINES)..].join(" | ");
    let count = joined.chars().count();
    if count <= LOG_CHARS {
        return joined;
    }
    let kept: String = joined.chars().skip(count - (LOG_CHARS - 3)).collect();
    format!("...{kept}")
}

/// Runs `script` with `sh` on the target and returns its stdout.
async fn run(
    transport: &Transport,
    script: &str,
    action: &'static str,
) -> Result<Vec<u8>, ExecError> {
    let pipes = Pipes {
        stdin: false,
        stdout: true,
        stderr: true,
    };
    let process = transport
        .exec(script, pipes)
        .await
        .map_err(ExecError::Ssh)?;
    let output = output(process).await.map_err(ExecError::Ssh)?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(ExecError::Command {
            action,
            message: tar::failure(&output.stderr, output.status),
        })
    }
}

/// What a finished remote command printed, and how it ended.
struct Output {
    /// Its exit status.
    status: ExitStatus,
    /// Everything it wrote to its standard output.
    stdout: Vec<u8>,
    /// Everything it wrote to its standard error.
    stderr: Vec<u8>,
}

/// Reads `process`'s standard output and error to their ends, both at once so
/// neither pipe can fill up and stall it, then waits for its exit status.
async fn output(mut process: RemoteProcess) -> Result<Output, SshError> {
    let (stdout, stderr) = tokio::try_join!(
        read_all(process.stdout.take()),
        read_all(process.stderr.take())
    )?;
    let status = exit_status(process.wait().await?);
    Ok(Output {
        status,
        stdout,
        stderr,
    })
}

/// Everything `pipe` holds until its end, nothing when there is no pipe.
async fn read_all<R: AsyncRead + Unpin>(pipe: Option<R>) -> Result<Vec<u8>, SshError> {
    let mut bytes = Vec::new();
    if let Some(mut pipe) = pipe {
        pipe.read_to_end(&mut bytes).await.map_err(|error| {
            SshError::Other(format!(
                "failure while accessing standard i/o of remote process: {error}"
            ))
        })?;
    }
    Ok(bytes)
}

/// A remote exit `code` as the local [`ExitStatus`] of a process that exited
/// with it.
fn exit_status(code: i32) -> ExitStatus {
    ExitStatus::from_raw((code & 0xff) << 8)
}

/// A failure feeding or copying to a remote command's pipes.
fn remote_io(source: &io::Error) -> ExecError {
    ExecError::Ssh(SshError::Other(format!(
        "the remote command could not be executed: {source}"
    )))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command as StdCommand;
    use std::time::Duration;

    use tempfile::tempdir;
    use tokio::process::Command;

    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn job(secret: &str) -> JobCommand {
        JobCommand {
            dir: "/w/r1".into(),
            script: "true".into(),
            secrets: vec![("HF_TOKEN".into(), SecretString::from(secret.to_string()))],
            container: None,
        }
    }

    fn with_name(name: &str) -> JobCommand {
        JobCommand {
            secrets: vec![(name.into(), SecretString::from("v".to_string()))],
            ..job("v")
        }
    }

    #[test]
    fn the_put_script_creates_the_directory_and_replaces_the_file() -> TestResult {
        let dir = tempdir()?;
        let path = dir.path().join("r1/.pod/it's here");
        let path = path.to_string_lossy().into_owned();
        for content in ["deadline", "a 'quoted' $value"] {
            let status = StdCommand::new("sh")
                .args(["-c", &put_script(&path, content)])
                .status()?;
            assert!(status.success());
            assert_eq!(fs::read_to_string(&path)?, content);
        }
        let left: Vec<_> = fs::read_dir(dir.path().join("r1/.pod"))?
            .filter_map(Result::ok)
            .map(|entry| entry.file_name())
            .collect();
        assert_eq!(left.len(), 1, "a temporary file is left: {left:?}");
        assert!(put_script(&path, "x").contains(".$$.tmp"));
        Ok(())
    }

    #[test]
    fn the_master_log_tail_is_its_last_lines_on_one_bounded_line() {
        assert_eq!(log_tail(""), "");
        assert_eq!(log_tail("\n \r\n"), "");
        let log = "one\ntwo\nthree\nfour\nfive\nsix\u{1b}[31m\r\n\n";
        assert_eq!(log_tail(log), "two | three | four | five | six[31m");
        let lengthy = format!("start\n{}end", "x".repeat(5000));
        let tail = log_tail(&lengthy);
        assert_eq!(tail.chars().count(), LOG_CHARS);
        assert!(tail.starts_with("...") && tail.ends_with("xend"), "{tail}");
        assert_eq!(log_tail("caf\u{e9}\u{7}\n"), "caf\u{e9}");
    }

    #[test]
    fn the_master_log_is_read_from_its_end() -> TestResult {
        let dir = tempdir()?;
        assert_eq!(master_log(&dir.path().join("log")), "");
        let path = dir.path().join("log");
        fs::write(&path, format!("{}\nlast words\n", "y".repeat(12_000)))?;
        assert!(master_log(&path).ends_with("last words"));
        Ok(())
    }

    /// A connection lost at the first command reads as a dead master, with its log.
    #[test]
    fn a_session_gone_at_its_first_command_is_a_dead_master() -> TestResult {
        let dir = tempdir()?;
        let log = dir.path().join("log");
        fs::write(&log, "debug1: forking\r\nkilled\u{1b}[0m\n")?;
        let error = master_died(ExecError::Ssh(SshError::Disconnected), &log);
        assert!(
            matches!(&error, ExecError::MasterDied { log } if log == "debug1: forking | killed[0m"),
            "{error:?}"
        );
        let error = master_died(ExecError::Protocol("x".into()), &log);
        assert!(matches!(error, ExecError::Protocol(_)), "{error:?}");
        Ok(())
    }

    #[test]
    fn a_dead_master_names_its_log() {
        let error = ExecError::MasterDied {
            log: "Connection closed".into(),
        };
        assert_eq!(
            error.to_string(),
            "the ssh master connection ended right after it started (ssh log: Connection closed)"
        );
        let error = ExecError::MasterDied { log: String::new() };
        assert_eq!(
            error.to_string(),
            "the ssh master connection ended right after it started (ssh log: empty)"
        );
    }

    #[test]
    fn the_launcher_reads_secrets_from_stdin() -> Result<(), ExecError> {
        let launcher = launcher(&job("hf_value"))?;
        assert!(
            launcher.starts_with("IFS= read -r HF_TOKEN && export HF_TOKEN && mkdir -p -- '/w/r1'")
        );
        assert!(!launcher.contains("hf_value"));
        assert!(launcher.contains("rm -f exit_code cancelling cancelled job.pid"));
        Ok(())
    }

    #[test]
    fn multi_line_secrets_are_refused_without_their_value() {
        for value in ["line\nbreak", "line\rbreak", "nul\0byte"] {
            let error = launcher(&job(value)).err().map(|error| error.to_string());
            assert_eq!(
                error.as_deref(),
                Some("HF_TOKEN cannot be passed to the job")
            );
        }
    }

    #[test]
    fn secret_names_must_be_variable_names() {
        for name in ["", "1ABC", "A-B", "A B", "A;rm -rf /", "$(x)"] {
            assert!(
                matches!(launcher(&with_name(name)), Err(ExecError::InvalidSecret(_))),
                "{name:?} was accepted"
            );
        }
        assert!(launcher(&with_name("_A1")).is_ok());
    }

    /// Whether `setsid` (util-linux or busybox), which the launcher runs, is on `PATH`.
    fn setsid_available() -> bool {
        StdCommand::new("sh")
            .arg("-c")
            .arg("command -v setsid")
            .stdout(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    /// Runs `script` with the local `sh`, feeding it `stdin`, and returns its stdout.
    fn sh(script: &str, stdin: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        use std::io::Write;
        let mut child = StdCommand::new("sh")
            .arg("-c")
            .arg(script)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()?;
        if let Some(mut input) = child.stdin.take() {
            input.write_all(stdin)?;
        }
        let output = child.wait_with_output()?;
        if !output.status.success() {
            return Err(format!("sh failed: {}", output.status).into());
        }
        Ok(output.stdout)
    }

    #[tokio::test]
    async fn the_launcher_hands_secrets_to_the_job_verbatim() -> TestResult {
        if !setsid_available() {
            eprintln!("skipped: setsid is not installed");
            return Ok(());
        }
        let root = tempdir()?;
        let dir = root.path().join("r1");
        let job = JobCommand {
            dir: dir.to_string_lossy().into_owned(),
            script: "printf '%s|%s' \"$A\" \"$B\" > seen".into(),
            secrets: vec![
                ("A".into(), SecretString::from(" s3cr3t 'value' \\n ")),
                ("B".into(), SecretString::from(String::new())),
            ],
            container: None,
        };
        let mut stdin = Vec::new();
        feed_secrets(Some(&mut stdin), &job.secrets).await?;
        let printed = sh(&launcher(&job)?, &stdin)?;
        let pid: u32 = String::from_utf8(printed)?.trim().parse()?;
        assert!(pid > 1);
        for _ in 0..100 {
            if dir.join(EXIT_FILE).exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!(fs::read_to_string(dir.join(EXIT_FILE))?.trim(), "0");
        assert_eq!(
            fs::read_to_string(dir.join(PID_FILE))?.trim(),
            pid.to_string()
        );
        assert_eq!(
            fs::read_to_string(dir.join("seen"))?,
            " s3cr3t 'value' \\n |"
        );
        Ok(())
    }

    #[test]
    fn read_script_honours_offset_and_limit() -> TestResult {
        let root = tempdir()?;
        let file = root.path().join("data.txt");
        fs::write(&file, "0123456789")?;
        let path = file.to_string_lossy().into_owned();
        assert_eq!(sh(&read_script(&path, 2, 3), b"")?, b"234");
        assert_eq!(sh(&read_script(&path, 8, 100), b"")?, b"89");
        assert_eq!(sh(&read_script(&path, 20, 5), b"")?, b"");
        assert_eq!(
            sh(&read_script(&format!("{path}.missing"), 0, 5), b"")?,
            b""
        );
        Ok(())
    }

    #[test]
    fn archive_script_skips_missing_entries_and_excludes() -> TestResult {
        let root = tempdir()?;
        fs::create_dir_all(root.path().join("output/checkpoint-5"))?;
        fs::write(root.path().join("output/adapter.bin"), "w")?;
        fs::write(root.path().join("output/checkpoint-5/state"), "s")?;
        let remote = root.path().to_string_lossy().into_owned();
        let script = archive_script(
            &remote,
            &["output".into(), "missing".into()],
            &["checkpoint-*".into()],
        );
        let archive = sh(&script, b"")?;
        let listing = sh("tar -tf -", &archive)?;
        let listing = String::from_utf8(listing)?;
        assert!(listing.contains("output/adapter.bin"), "{listing}");
        assert!(!listing.contains("checkpoint"), "{listing}");
        assert_eq!(
            sh(&archive_script(&remote, &["missing".into()], &[]), b"")?,
            [] as [u8; 0]
        );

        let gone = format!("{remote}/never-created");
        let output = StdCommand::new("sh")
            .arg("-c")
            .arg(archive_script(&gone, &["output".into()], &[]))
            .output()?;
        assert_eq!(output.status.code(), Some(1));
        assert_eq!(output.stdout, [] as [u8; 0]);
        assert_eq!(
            String::from_utf8(output.stderr)?,
            format!("{gone} does not exist\n")
        );
        Ok(())
    }

    /// Whether permissions are enforced for this user (they are not for root).
    fn permissions_enforced(locked: &Path) -> bool {
        fs::read_dir(locked).is_err()
    }

    /// A local directory holding one file far larger than a pipe holds.
    fn big_source(root: &Path) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
        let source = root.join("source");
        fs::create_dir(&source)?;
        fs::write(source.join("big"), vec![b'x'; 4 * 1024 * 1024])?;
        Ok(source)
    }

    /// A local `tar -x` into `dir`, standing in for the remote receiver.
    fn receiver(dir: &Path) -> Result<Child, Box<dyn std::error::Error>> {
        Ok(Command::new("tar")
            .arg("-C")
            .arg(dir)
            .args(["-xf", "-"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()?)
    }

    async fn upload_to(
        source: &Path,
        mut receiver: Child,
    ) -> Result<Result<(), ExecError>, Box<dyn std::error::Error>> {
        let create = tar::spawn_create(source, &[".".to_string()], &[])?;
        let (copied, errors, created) =
            transfer(create, receiver.stdin.take(), receiver.stderr.take()).await;
        let status = receiver.wait().await?;
        Ok(upload_outcome(Ok(status), &errors, created, copied))
    }

    #[tokio::test]
    async fn an_upload_reports_the_receiving_tar_error_not_the_broken_pipe() -> TestResult {
        let root = tempdir()?;
        let source = big_source(root.path())?;
        let locked = root.path().join("locked");
        fs::create_dir(&locked)?;
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o000))?;
        if !permissions_enforced(&locked) {
            fs::set_permissions(&locked, fs::Permissions::from_mode(0o755))?;
            eprintln!("skipped: permissions are not enforced for this user");
            return Ok(());
        }
        let result = tokio::time::timeout(
            Duration::from_secs(30),
            upload_to(&source, receiver(&locked)?),
        )
        .await;
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o755))?;
        match result {
            Err(_) => return Err("the upload hung once the receiver died".into()),
            Ok(outcome) => match outcome? {
                Err(ExecError::Command { action, message }) => {
                    assert_eq!(action, "upload");
                    assert!(message.contains("Permission denied"), "{message}");
                    assert!(!message.contains("Broken pipe"), "{message}");
                },
                other => return Err(format!("expected tar's own error, got {other:?}").into()),
            },
        }
        Ok(())
    }

    #[tokio::test]
    async fn an_upload_survives_a_flood_of_receiver_warnings() -> TestResult {
        let root = tempdir()?;
        let source = root.path().join("source");
        fs::create_dir(&source)?;
        for index in 0..3000 {
            fs::write(source.join(format!("{index:05}-{}", "n".repeat(150))), "")?;
        }
        // Enterable but not writable: tar warns once per entry and keeps reading.
        let target = root.path().join("target");
        fs::create_dir(&target)?;
        fs::set_permissions(&target, fs::Permissions::from_mode(0o555))?;
        if fs::write(target.join("probe"), "").is_ok() {
            fs::set_permissions(&target, fs::Permissions::from_mode(0o755))?;
            eprintln!("skipped: permissions are not enforced for this user");
            return Ok(());
        }
        let result = tokio::time::timeout(
            Duration::from_secs(30),
            upload_to(&source, receiver(&target)?),
        )
        .await;
        fs::set_permissions(&target, fs::Permissions::from_mode(0o755))?;
        match result {
            Err(_) => return Err("the upload deadlocked on the receiver's warnings".into()),
            Ok(outcome) => match outcome? {
                Err(ExecError::Command { message, .. }) => {
                    assert!(message.contains("more lines)"), "{message}");
                    assert!(message.len() < 8 * 1024, "{} bytes", message.len());
                },
                other => return Err(format!("expected tar's warnings, got {other:?}").into()),
            },
        }
        Ok(())
    }

    /// A local `sh` standing in for the remote sender: runs `script` with its stdout
    /// and stderr piped.
    fn sender(script: &str) -> Result<Child, Box<dyn std::error::Error>> {
        Ok(Command::new("sh")
            .arg("-c")
            .arg(script)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?)
    }

    async fn download_from(
        mut sender: Child,
        local: &Path,
    ) -> Result<Result<(), ExecError>, Box<dyn std::error::Error>> {
        let (extracted, errors) = receive(sender.stdout.take(), sender.stderr.take(), local).await;
        let sent = sender.wait().await?;
        Ok(download_outcome(extracted, Ok(sent), &errors))
    }

    #[tokio::test]
    async fn a_download_reports_the_local_tar_error_not_the_broken_pipe() -> TestResult {
        let root = tempdir()?;
        let source = big_source(root.path())?;
        let locked = root.path().join("locked");
        fs::create_dir(&locked)?;
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o000))?;
        if !permissions_enforced(&locked) {
            fs::set_permissions(&locked, fs::Permissions::from_mode(0o755))?;
            eprintln!("skipped: permissions are not enforced for this user");
            return Ok(());
        }
        let script = archive_script(&source.to_string_lossy(), &["big".into()], &[]);
        let result = tokio::time::timeout(
            Duration::from_secs(30),
            download_from(sender(&script)?, &locked),
        )
        .await;
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o755))?;
        match result {
            Err(_) => return Err("the download hung once the extraction died".into()),
            Ok(outcome) => match outcome? {
                Err(ExecError::Command { action, message }) => {
                    assert_eq!(action, "extract");
                    assert!(message.contains("Permission denied"), "{message}");
                },
                other => return Err(format!("expected tar's own error, got {other:?}").into()),
            },
        }
        Ok(())
    }

    #[tokio::test]
    async fn a_download_survives_a_flood_of_sender_warnings() -> TestResult {
        let root = tempdir()?;
        let source = root.path().join("source");
        fs::create_dir(&source)?;
        fs::write(source.join("file"), "content\n")?;
        // 256 KiB of warnings before the archive: without draining them at the same
        // time, the sender blocks on its error pipe and the extraction on its output.
        let script = format!(
            "i=0; while [ \"$i\" -lt 4096 ]; do echo 'warning: {}' >&2; i=$((i + 1)); done\n{}",
            "w".repeat(54),
            archive_script(&source.to_string_lossy(), &["file".into()], &[])
        );
        let local = root.path().join("back");
        let result = tokio::time::timeout(
            Duration::from_secs(30),
            download_from(sender(&script)?, &local),
        )
        .await;
        match result {
            Err(_) => return Err("the download deadlocked on the sender's warnings".into()),
            Ok(outcome) => outcome??,
        }
        assert_eq!(fs::read_to_string(local.join("file"))?, "content\n");
        Ok(())
    }

    /// A receiver ended without an exit status reports its error output when it
    /// printed any.
    #[test]
    fn a_receiver_killed_by_a_signal_is_reported_with_its_error_output() {
        let killed = || Err(ExecError::Ssh(SshError::Terminated));
        let result = upload_outcome(killed(), b"tar: write error\n", Ok(()), Ok(()));
        assert!(
            matches!(
                &result,
                Err(ExecError::Command { action: "upload", message }) if message == "tar: write error"
            ),
            "{result:?}"
        );
        let silent = upload_outcome(killed(), b" \n", Ok(()), Ok(()));
        assert!(
            matches!(silent, Err(ExecError::Ssh(SshError::Terminated))),
            "{silent:?}"
        );
    }

    /// A remote exit code reads as the status of a local process that exited
    /// with it, so failure messages stay as they were.
    #[test]
    fn a_remote_exit_code_is_an_exit_status() {
        assert_eq!(exit_status(0).code(), Some(0));
        assert!(exit_status(0).success());
        assert_eq!(exit_status(2).code(), Some(2));
        assert_eq!(exit_status(2).to_string(), "exit status: 2");
        assert_eq!(exit_status(255).code(), Some(255));
    }

    #[test]
    fn a_failed_sender_is_reported_with_its_error_output() {
        let status = std::os::unix::process::ExitStatusExt::from_raw(1 << 8);
        let result = download_outcome(Ok(()), Ok(status), b"sh: cd: can't cd to /nowhere\n");
        assert!(
            matches!(
                &result,
                Err(ExecError::Command { action: "download", message })
                    if message == "sh: cd: can't cd to /nowhere"
            ),
            "{result:?}"
        );
    }

    /// The built-in client reads `~/.ssh/config` then the system file, or the
    /// given file alone, with the home and user of the environment.
    #[cfg(feature = "builtin-ssh")]
    #[test]
    fn the_builtin_client_reads_the_user_and_system_files() -> TestResult {
        let env = LocalEnv {
            home: Some("/home/me".into()),
            user: None,
            logname: Some("me".into()),
        };
        let sources = config_sources(None, &env)?;
        assert_eq!(
            sources.files,
            [
                PathBuf::from("/home/me/.ssh/config"),
                PathBuf::from("/etc/ssh/ssh_config")
            ]
        );
        assert_eq!(sources.home, PathBuf::from("/home/me"));
        assert_eq!(sources.local_user, "me");
        let given = config_sources(Some(Path::new("/tmp/cfg")), &env)?;
        assert_eq!(given.files, [PathBuf::from("/tmp/cfg")]);
        Ok(())
    }

    /// Without `HOME`, or without both `USER` and `LOGNAME`, the built-in
    /// client refuses rather than guesses.
    #[cfg(feature = "builtin-ssh")]
    #[test]
    fn the_builtin_client_does_not_guess_home_or_user() {
        let no_home = LocalEnv {
            home: None,
            user: Some("me".into()),
            logname: None,
        };
        let error = config_sources(None, &no_home).err().map(|e| e.to_string());
        assert!(error.is_some_and(|e| e.starts_with("HOME is not set")));
        let no_user = LocalEnv {
            home: Some("/h".into()),
            user: Some(String::new().into()),
            logname: None,
        };
        let error = config_sources(None, &no_user).err().map(|e| e.to_string());
        assert!(error.is_some_and(|e| e.starts_with("neither USER nor LOGNAME")));
    }

    /// A pod's pinned key line becomes the built-in client's target; a line
    /// that is no key is a host key failure.
    #[cfg(feature = "builtin-ssh")]
    #[test]
    fn a_pod_endpoint_becomes_a_direct_target() -> TestResult {
        use russh::keys::{Algorithm, PrivateKey};

        let key = PrivateKey::random(&mut rand::rng(), Algorithm::Ed25519)?;
        let mut pod = PodEndpoint {
            alias: "overbrainer-r1".into(),
            config: "/runs/r1/ssh/config".into(),
            host: "1.2.3.4".into(),
            port: 2201,
            user: "root".into(),
            key: "/runs/r1/ssh/id_ed25519".into(),
            host_key: key.public_key().to_openssh()?,
        };
        let target = direct_target(&pod)?;
        assert_eq!(target.name, "overbrainer-r1");
        assert_eq!((target.host.as_str(), target.port), ("1.2.3.4", 2201));
        assert_eq!(target.host_key.key_data(), key.public_key().key_data());
        pod.host_key = "not a key".into();
        assert!(matches!(direct_target(&pod), Err(SshError::HostKey { .. })));
        Ok(())
    }
}
