//! Running a training job on a target: the `Executor` trait and its `Local` and `Ssh`
//! implementations.
//!
//! A job is detached from overbrainer: it runs in its own session (local process
//! group, or `setsid nohup` over SSH), writes its output to `job.log`, its process ID
//! to `job.pid` and, when it ends, its exit code to `exit_code`, all in its run
//! directory. Any later overbrainer process can read its status, tail its files or
//! cancel it from that directory alone.

mod digest;
mod lines;
mod local;
mod runtime;
mod script;
mod ssh;
mod tar;

use std::future::Future;
use std::path::{Path, PathBuf};

use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

pub use digest::{
    FileDigest, glob_match, local_manifest, manifest_script, parse_manifest, sha256_file,
};
pub use lines::{LineStream, MAX_TAIL_READ, complete_lines};
pub use local::LocalExecutor;
pub use runtime::{JobRuntime, JobSpec, shell_path};
pub use script::{GROUP_SIGNAL, cancel_script, job_script, parse_status, quote, status_script};
pub use ssh::SshExecutor;

use crate::config::Engine;

/// Output of the job (stdout and stderr), in its run directory.
pub const JOB_LOG: &str = "job.log";
/// Process ID of the job's session leader, in its run directory.
pub const PID_FILE: &str = "job.pid";
/// Exit code of the job once it has ended, in its run directory.
pub const EXIT_FILE: &str = "exit_code";
/// Marker written by [`Executor::cancel`] before it signals anything, in the run
/// directory. While this exists and [`CANCEL_FILE`] does not, [`Executor::status`]
/// reports [`JobStatus::Running`] for as long as the process group is alive, ahead
/// of [`EXIT_FILE`] (stopping a job's container makes its wrapper write an exit
/// code while the cancel is still stopping the group), and
/// [`JobStatus::Cancelled`] once the group is gone and no [`EXIT_FILE`] was
/// written, even before [`CANCEL_FILE`] itself lands.
///
/// [`EXIT_FILE`] deliberately wins over this marker once the group is gone: a
/// cancel killed right after writing it (a Ctrl-C in its window) would otherwise
/// leave it behind for ever and label a job that then finished normally as
/// cancelled.
pub const CANCELLING_FILE: &str = "cancelling";
/// Marker written by [`Executor::cancel`] once the process group is confirmed gone,
/// in the run directory. Once this exists, [`Executor::status`] reports
/// [`JobStatus::Cancelled`] unconditionally: not even a live process at a recycled
/// `pid`, or an [`EXIT_FILE`] written before the cancel finished, can change it.
pub const CANCEL_FILE: &str = "cancelled";

/// Errors from running or reaching a job.
#[derive(Debug, thiserror::Error)]
pub enum ExecError {
    /// A local file or directory could not be read or written.
    #[error("cannot access {}", path.display())]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// A local program could not be started or waited for.
    #[error("cannot run `{program}`")]
    Spawn {
        /// The program.
        program: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// A command on the target failed. The message is the command's error output,
    /// which never holds a secret: secrets only travel on standard input.
    #[error("{action} failed: {message}")]
    Command {
        /// What was attempted, for example `upload`.
        action: &'static str,
        /// Error output of the command, or its exit status.
        message: String,
    },
    /// The SSH connection failed or broke.
    #[error("ssh failed")]
    Ssh(#[source] openssh::Error),
    /// A secret cannot be passed to the job: its name is not an env variable name, or
    /// its value holds a line break or a NUL byte. The message names the variable
    /// only.
    #[error("{0} cannot be passed to the job")]
    InvalidSecret(String),
    /// The target answered something unexpected.
    #[error("unexpected answer from the target: {0}")]
    Protocol(String),
}

/// A container running the job, stopped by [`Executor::cancel`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Container {
    /// Engine running it.
    pub engine: Engine,
    /// Container name.
    pub name: String,
}

/// Checks that every secret can be handed to a job, whatever the target: its name
/// must be an environment variable name (letters, digits and `_`, not starting with
/// a digit, not empty), and its value must hold no line break and no NUL byte. A
/// line break would end the `read` that feeds a remote job early, and a NUL byte
/// cannot live in an environment: either would hand the job a different value.
///
/// # Errors
///
/// Returns [`ExecError::InvalidSecret`], naming the variable and never its value,
/// for the first secret that cannot be passed.
fn check_secrets(secrets: &[(String, SecretString)]) -> Result<(), ExecError> {
    for (name, value) in secrets {
        let valid_name = name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            && name.chars().next().is_some_and(|c| !c.is_ascii_digit());
        if !valid_name || value.expose_secret().contains(['\n', '\r', '\0']) {
            return Err(ExecError::InvalidSecret(name.clone()));
        }
    }
    Ok(())
}

/// A job's process ID, validated to be neither `0` nor `1`. Process group `0` is a
/// wildcard for `kill`, and `1` is `init` (or, over SSH, sometimes a container's own
/// PID 1): signalling either would never correctly target a single job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct Pid(u32);

impl Pid {
    /// Validates `pid`.
    ///
    /// # Errors
    ///
    /// Returns [`ExecError::Protocol`] when `pid` is `0` or `1`.
    pub fn new(pid: u32) -> Result<Self, ExecError> {
        if pid <= 1 {
            return Err(ExecError::Protocol(format!("invalid process id {pid}")));
        }
        Ok(Self(pid))
    }

    /// The validated process ID.
    #[must_use]
    pub fn get(self) -> u32 {
        self.0
    }
}

impl<'de> Deserialize<'de> for Pid {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = u32::deserialize(deserializer)?;
        Self::new(raw).map_err(serde::de::Error::custom)
    }
}

/// A job to start in the background.
#[derive(Debug, Clone)]
pub struct JobCommand {
    /// Run directory on the target, absolute. The job runs there.
    pub dir: String,
    /// POSIX shell commands of the job.
    pub script: String,
    /// Secret env variables of the job. They reach the job's environment only: never
    /// a command line, a file or a log.
    pub secrets: Vec<(String, SecretString)>,
    /// The container the script starts, if any.
    pub container: Option<Container>,
}

/// Handle of a started job, enough to find it again from another process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobId {
    /// Run directory on the target.
    pub dir: String,
    /// Process ID of the job's session leader, also its process group ID. Never `0`
    /// or `1`.
    pub pid: Pid,
    /// The container the job runs in, if any.
    pub container: Option<Container>,
}

/// What a job is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    /// Still running.
    Running,
    /// Ended with this exit code.
    Exited(i32),
    /// Stopped by [`Executor::cancel`].
    Cancelled,
    /// Neither running nor ended normally: killed without writing its exit code, or
    /// its run directory is gone.
    Lost,
}

impl JobStatus {
    /// Whether the job has stopped for good.
    #[must_use]
    pub fn is_finished(self) -> bool {
        self != Self::Running
    }
}

/// Runs jobs on a target and moves files to and from it. Paths on the target are
/// absolute strings; the target may be another machine.
///
/// Futures are `Send` so the CLI and the TUI can drive them from any task.
pub trait Executor: Send + Sync {
    /// Directory holding the run directories on the target, absolute.
    fn workdir(&self) -> &str;

    /// Copies the content of the local directory `local` into `remote`, creating it.
    ///
    /// # Errors
    ///
    /// Returns an [`ExecError`] when the directory cannot be read or copied.
    fn upload(
        &self,
        local: &Path,
        remote: &str,
    ) -> impl Future<Output = Result<(), ExecError>> + Send;

    /// Starts `job` in the background and returns its handle. The job keeps running
    /// when this process exits or loses its connection to the target.
    ///
    /// The environment the job starts with depends on the target. A local job
    /// inherits overbrainer's own environment, minus every variable whose name
    /// starts with `OVERBRAINER_` or `VAULT_` (see [`LocalExecutor`]). A job over
    /// SSH gets whatever the remote non-interactive `sh -c` gives it, which is the
    /// SSH server's environment for that user: no login shell and no interactive
    /// startup file are read (see [`SshExecutor`]). The job's secrets are then set
    /// on top of that, on its environment only. A `native` target whose `axolotl`
    /// relies on the caller's shell setup, such as a conda activation or a `PATH`
    /// entry from a profile, can therefore behave differently on each.
    ///
    /// # Errors
    ///
    /// Returns [`ExecError::InvalidSecret`], naming the variable only, when a
    /// secret's name is not an environment variable name or its value holds a line
    /// break or a NUL byte (see [`check_secrets`]); nothing is started then. Returns
    /// another [`ExecError`] when the job cannot be started.
    fn spawn(&self, job: &JobCommand) -> impl Future<Output = Result<JobId, ExecError>> + Send;

    /// Bytes of the file `path` from byte `offset`, up to `limit` bytes: fewer when
    /// less than `limit` remains past `offset`. A missing file reads as empty.
    ///
    /// # Errors
    ///
    /// Returns an [`ExecError`] when the file cannot be read.
    fn read_from(
        &self,
        path: &str,
        offset: u64,
        limit: u64,
    ) -> impl Future<Output = Result<Vec<u8>, ExecError>> + Send;

    /// What `job` is doing now.
    ///
    /// # Errors
    ///
    /// Returns an [`ExecError`] when the target cannot be asked.
    fn status(&self, job: &JobId) -> impl Future<Output = Result<JobStatus, ExecError>> + Send;

    /// Stops `job`: its container first, then its whole process group (`SIGTERM`,
    /// then `SIGKILL` after 10 seconds). A job that has already exited is left as
    /// is: its status stays [`JobStatus::Exited`], it is never marked cancelled.
    /// Otherwise, [`CANCELLING_FILE`] is written before anything is signalled, so
    /// [`Executor::status`] reports [`JobStatus::Running`] while the group is being
    /// stopped and [`JobStatus::Cancelled`] as soon as it is gone, never
    /// [`JobStatus::Lost`]; [`CANCEL_FILE`] is written last, once the group is
    /// confirmed gone, and then wins unconditionally over everything else. Calling
    /// `cancel` again on an already cancelled job is a no-op.
    ///
    /// Cancelling a container job has a short window where the status is wrong:
    /// once the engine's `stop` makes the job's wrapper write [`EXIT_FILE`] and
    /// until the group is gone and [`CANCEL_FILE`] lands (roughly a second, up to
    /// the `<engine> stop -t 30` timeout), [`Executor::status`] reports
    /// [`JobStatus::Exited`] rather than [`JobStatus::Running`]. This is the price
    /// of letting a real exit code win over a [`CANCELLING_FILE`] left behind by a
    /// cancel that was killed; the final [`CANCEL_FILE`] still wins afterwards.
    ///
    /// The caller runs this to its end: an interrupted cancel can leave
    /// [`CANCELLING_FILE`] behind, and over a local target the scripts run in their
    /// own process group so a Ctrl-C at the terminal cannot reach them.
    ///
    /// # Errors
    ///
    /// Returns an [`ExecError`] when the target cannot be reached.
    fn cancel(&self, job: &JobId) -> impl Future<Output = Result<(), ExecError>> + Send;

    /// Copies `entries` (files or directories, relative to the target directory
    /// `remote`) into the local directory `local`, leaving out names matching an
    /// `exclude` pattern at any depth. Missing entries are skipped, and `local` is
    /// not created when none exists.
    ///
    /// # Errors
    ///
    /// Returns [`ExecError::Command`] (action `download`, message
    /// `<remote> does not exist`) when `remote` itself is not a directory, and an
    /// [`ExecError`] when the copy fails.
    fn download(
        &self,
        remote: &str,
        local: &Path,
        entries: &[String],
        exclude: &[String],
    ) -> impl Future<Output = Result<(), ExecError>> + Send;

    /// Every regular file under `entries` (relative to the target directory
    /// `remote`) with the SHA-256 of its content, leaving out names matching an
    /// `exclude` pattern at any depth, as [`Executor::download`] does. Missing
    /// entries are skipped; symbolic links are not followed. Paths are relative to
    /// `remote`, sorted.
    ///
    /// # Errors
    ///
    /// Returns [`ExecError::Command`] (message `<remote> does not exist`) when
    /// `remote` itself is not a directory, and an [`ExecError`] when a file cannot
    /// be read or the target cannot be reached.
    fn manifest(
        &self,
        remote: &str,
        entries: &[String],
        exclude: &[String],
    ) -> impl Future<Output = Result<Vec<FileDigest>, ExecError>> + Send;

    /// Follows the file `path` from byte `offset`, one complete line at a time.
    fn tail(&self, path: &str, offset: u64) -> LineStream<'_, Self>
    where
        Self: Sized,
    {
        LineStream::new(self, path, offset)
    }
}

/// The executor of a local or SSH target.
#[derive(Debug)]
pub enum AnyExecutor {
    /// Runs on this machine.
    Local(LocalExecutor),
    /// Runs over SSH.
    Ssh(SshExecutor),
}

impl Executor for AnyExecutor {
    fn workdir(&self) -> &str {
        match self {
            Self::Local(executor) => executor.workdir(),
            Self::Ssh(executor) => executor.workdir(),
        }
    }

    async fn upload(&self, local: &Path, remote: &str) -> Result<(), ExecError> {
        match self {
            Self::Local(executor) => executor.upload(local, remote).await,
            Self::Ssh(executor) => executor.upload(local, remote).await,
        }
    }

    async fn spawn(&self, job: &JobCommand) -> Result<JobId, ExecError> {
        match self {
            Self::Local(executor) => executor.spawn(job).await,
            Self::Ssh(executor) => executor.spawn(job).await,
        }
    }

    async fn read_from(&self, path: &str, offset: u64, limit: u64) -> Result<Vec<u8>, ExecError> {
        match self {
            Self::Local(executor) => executor.read_from(path, offset, limit).await,
            Self::Ssh(executor) => executor.read_from(path, offset, limit).await,
        }
    }

    async fn status(&self, job: &JobId) -> Result<JobStatus, ExecError> {
        match self {
            Self::Local(executor) => executor.status(job).await,
            Self::Ssh(executor) => executor.status(job).await,
        }
    }

    async fn cancel(&self, job: &JobId) -> Result<(), ExecError> {
        match self {
            Self::Local(executor) => executor.cancel(job).await,
            Self::Ssh(executor) => executor.cancel(job).await,
        }
    }

    async fn download(
        &self,
        remote: &str,
        local: &Path,
        entries: &[String],
        exclude: &[String],
    ) -> Result<(), ExecError> {
        match self {
            Self::Local(executor) => executor.download(remote, local, entries, exclude).await,
            Self::Ssh(executor) => executor.download(remote, local, entries, exclude).await,
        }
    }

    async fn manifest(
        &self,
        remote: &str,
        entries: &[String],
        exclude: &[String],
    ) -> Result<Vec<FileDigest>, ExecError> {
        match self {
            Self::Local(executor) => executor.manifest(remote, entries, exclude).await,
            Self::Ssh(executor) => executor.manifest(remote, entries, exclude).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn any_executor_delegates_with_the_limit() -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let executor = AnyExecutor::Local(LocalExecutor::new(root.path())?);
        let file = Path::new(executor.workdir()).join("data.txt");
        std::fs::write(&file, "0123456789")?;
        let path = file.to_string_lossy().into_owned();
        assert_eq!(executor.read_from(&path, 2, 3).await?, b"234");
        Ok(())
    }

    #[test]
    fn pid_rejects_zero_and_one() {
        assert!(Pid::new(0).is_err());
        assert!(Pid::new(1).is_err());
        assert!(Pid::new(2).is_ok_and(|pid| pid.get() == 2));
    }

    #[test]
    fn deserializing_a_job_id_rejects_an_invalid_pid() -> Result<(), Box<dyn std::error::Error>> {
        let valid: JobId = serde_json::from_str(r#"{"dir":"/w/r1","pid":42,"container":null}"#)?;
        assert_eq!(valid.pid.get(), 42);
        let invalid = serde_json::from_str::<JobId>(r#"{"dir":"/w/r1","pid":1,"container":null}"#);
        assert!(invalid.is_err());
        Ok(())
    }
}
