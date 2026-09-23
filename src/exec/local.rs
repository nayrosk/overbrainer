use std::ffi::OsString;
use std::fs::{self, File};
use std::future::{Future, ready};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Output, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use secrecy::ExposeSecret;
use tokio::process::{Child, Command};

use super::tar;
use super::{
    CANCEL_FILE, CANCELLING_FILE, EXIT_FILE, ExecError, Executor, FileDigest, JOB_LOG, JobCommand,
    JobId, JobStatus, PID_FILE, Pid, cancel_script, check_secrets, job_script, local_manifest,
    parse_status, status_script,
};

/// How often this process reaps its exited jobs while a status or cancel script runs.
/// A job leader that has exited but is not reaped stays a zombie, and a zombie still
/// counts as a live member of its process group, so a cancel would otherwise wait
/// for its full grace period.
const REAP_EVERY: Duration = Duration::from_millis(150);

/// Prefixes of the variables of this process that never reach a job: overbrainer's
/// own settings and the Vault client's, which can hold credentials.
const PRIVATE_PREFIXES: [&str; 2] = ["OVERBRAINER_", "VAULT_"];

/// Runs jobs on this machine, each in its own process group so a Ctrl-C in the
/// terminal does not reach it. The status and cancel scripts get their own process
/// group too, so an interrupted command cannot kill a cancel halfway through.
///
/// A job inherits the environment of this process (`PATH`, CUDA, conda and so on)
/// except every variable whose name starts with `OVERBRAINER_` or `VAULT_`; its
/// secrets are then set explicitly, on its environment only.
#[derive(Debug)]
pub struct LocalExecutor {
    workdir: String,
    /// Jobs started by this process, reaped before each status check and during a
    /// cancel so they do not stay zombies (which would still look alive).
    children: Mutex<Vec<Child>>,
}

impl LocalExecutor {
    /// Runs jobs under `workdir`, created if needed.
    ///
    /// # Errors
    ///
    /// Returns [`ExecError::Io`] when `workdir` cannot be created or its absolute
    /// path is not valid UTF-8.
    pub fn new(workdir: &Path) -> Result<Self, ExecError> {
        fs::create_dir_all(workdir).map_err(io_error(workdir))?;
        let absolute = fs::canonicalize(workdir).map_err(io_error(workdir))?;
        let workdir = absolute
            .to_str()
            .ok_or_else(|| ExecError::Io {
                path: absolute.clone(),
                source: io::Error::other("the path is not valid UTF-8"),
            })?
            .to_string();
        Ok(Self {
            workdir,
            children: Mutex::new(Vec::new()),
        })
    }

    fn start(&self, job: &JobCommand) -> Result<JobId, ExecError> {
        check_secrets(&job.secrets)?;
        let dir = Path::new(&job.dir);
        fs::create_dir_all(dir).map_err(io_error(dir))?;
        for stale in [EXIT_FILE, CANCELLING_FILE, CANCEL_FILE, PID_FILE] {
            remove(&dir.join(stale))?;
        }
        let log_path = dir.join(JOB_LOG);
        let log = File::create(&log_path).map_err(io_error(&log_path))?;
        let err = log.try_clone().map_err(io_error(&log_path))?;
        let mut command = Command::new("sh");
        command
            .arg("-c")
            .arg(job_script(&job.script))
            .current_dir(dir)
            .stdin(Stdio::null())
            .stdout(log)
            .stderr(err)
            .process_group(0)
            .kill_on_drop(false);
        for name in private_names(std::env::vars_os().map(|(name, _)| name)) {
            command.env_remove(name);
        }
        for (name, value) in &job.secrets {
            command.env(name, value.expose_secret());
        }
        let child = command.spawn().map_err(spawn_error)?;
        let raw = child
            .id()
            .ok_or_else(|| ExecError::Protocol("the job exited before it started".into()))?;
        if let Ok(mut children) = self.children.lock() {
            children.push(child);
        }
        // Cannot fail for a freshly spawned child, whose pid is never 0 or 1, so the
        // running job is not left without a handle in practice.
        Ok(JobId {
            dir: job.dir.clone(),
            pid: Pid::new(raw)?,
            container: job.container.clone(),
        })
    }

    /// Reaps the jobs of this process that have exited.
    fn reap(&self) {
        if let Ok(mut children) = self.children.lock() {
            children.retain_mut(|child| matches!(child.try_wait(), Ok(None)));
        }
    }

    /// Runs `script` with `sh`, reaping this process's exited jobs before it starts
    /// and every [`REAP_EVERY`] until it ends. Fails with `action` when it exits
    /// non-zero.
    ///
    /// The script runs in its own process group, so a Ctrl-C in the terminal cannot
    /// kill a cancel in flight and leave its `cancelling` marker behind.
    async fn run_script(&self, script: String, action: &'static str) -> Result<Output, ExecError> {
        self.reap();
        let output = Command::new("sh")
            .arg("-c")
            .arg(script)
            .stdin(Stdio::null())
            .process_group(0)
            .output();
        tokio::pin!(output);
        let mut tick = tokio::time::interval(REAP_EVERY);
        let output = loop {
            tokio::select! {
                result = &mut output => break result.map_err(spawn_error)?,
                _ = tick.tick() => self.reap(),
            }
        };
        if output.status.success() {
            Ok(output)
        } else {
            Err(ExecError::Command {
                action,
                message: tar::failure(&output.stderr, output.status),
            })
        }
    }

    async fn check(&self, job: &JobId) -> Result<JobStatus, ExecError> {
        let output = self
            .run_script(status_script(&job.dir, job.pid.get()), "status")
            .await?;
        let text = String::from_utf8_lossy(&output.stdout);
        parse_status(&text)
            .ok_or_else(|| ExecError::Protocol(format!("unknown job status {:?}", text.trim())))
    }

    async fn stop(&self, job: &JobId) -> Result<(), ExecError> {
        let script = cancel_script(&job.dir, job.pid.get(), job.container.as_ref());
        self.run_script(script, "cancel").await.map(drop)
    }

    async fn copy(
        &self,
        from: &Path,
        to: &Path,
        entries: &[String],
        exclude: &[String],
    ) -> Result<(), ExecError> {
        if same_dir(from, to) {
            return Ok(());
        }
        if nested(from, to) {
            return Err(ExecError::Command {
                action: "copy",
                message: format!(
                    "{} is inside {}: copying there would copy the destination into itself",
                    to.display(),
                    from.display()
                ),
            });
        }
        let entries: Vec<String> = entries
            .iter()
            .filter(|entry| from.join(entry.as_str()).exists())
            .cloned()
            .collect();
        if entries.is_empty() {
            return Ok(());
        }
        let mut create = tar::spawn_create(from, &entries, exclude)?;
        let mut stdout = create
            .stdout
            .take()
            .ok_or_else(|| ExecError::Protocol("tar has no output".into()))?;
        // Both run at once, so the archiving tar's error output is drained while its
        // archive is extracted. `stdout` is dropped once the extraction ends.
        let extract = async move { tar::extract(&mut stdout, to).await };
        let (extracted, created) = tokio::join!(extract, tar::finish(create, "archive"));
        extracted.and(created)
    }
}

impl Executor for LocalExecutor {
    fn workdir(&self) -> &str {
        &self.workdir
    }

    async fn upload(&self, local: &Path, remote: &str) -> Result<(), ExecError> {
        self.copy(local, Path::new(remote), &[".".to_string()], &[])
            .await
    }

    fn spawn(&self, job: &JobCommand) -> impl Future<Output = Result<JobId, ExecError>> + Send {
        ready(self.start(job))
    }

    fn read_from(
        &self,
        path: &str,
        offset: u64,
        limit: u64,
    ) -> impl Future<Output = Result<Vec<u8>, ExecError>> + Send {
        ready(read_file(Path::new(path), offset, limit))
    }

    async fn status(&self, job: &JobId) -> Result<JobStatus, ExecError> {
        self.check(job).await
    }

    async fn cancel(&self, job: &JobId) -> Result<(), ExecError> {
        self.stop(job).await
    }

    async fn download(
        &self,
        remote: &str,
        local: &Path,
        entries: &[String],
        exclude: &[String],
    ) -> Result<(), ExecError> {
        let from = Path::new(remote);
        if !from.is_dir() {
            return Err(ExecError::Command {
                action: "download",
                message: format!("{remote} does not exist"),
            });
        }
        self.copy(from, local, entries, exclude).await
    }

    /// Computed here with the `sha2` crate, off the async runtime's threads, so a
    /// local target needs no `sha256sum` (macOS has none).
    async fn manifest(
        &self,
        remote: &str,
        entries: &[String],
        exclude: &[String],
    ) -> Result<Vec<FileDigest>, ExecError> {
        let dir = PathBuf::from(remote);
        let entries = entries.to_vec();
        let exclude = exclude.to_vec();
        tokio::task::spawn_blocking(move || local_manifest(&dir, &entries, &exclude))
            .await
            .map_err(|error| ExecError::Protocol(format!("the manifest task failed: {error}")))?
    }
}

/// At most `limit` bytes of `path` from `offset`; a missing file reads as empty.
fn read_file(path: &Path, offset: u64, limit: u64) -> Result<Vec<u8>, ExecError> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(io_error(path)(e)),
    };
    let mut bytes = Vec::new();
    file.seek(SeekFrom::Start(offset))
        .and_then(|_| file.take(limit).read_to_end(&mut bytes))
        .map_err(io_error(path))?;
    Ok(bytes)
}

/// Whether `from` and `to` are the same existing directory.
fn same_dir(from: &Path, to: &Path) -> bool {
    match (fs::canonicalize(from), fs::canonicalize(to)) {
        (Ok(from), Ok(to)) => from == to,
        _ => false,
    }
}

/// Whether `to`, once resolved (it may not exist yet), is strictly inside the
/// existing directory `from`.
fn nested(from: &Path, to: &Path) -> bool {
    match (fs::canonicalize(from), resolve(to)) {
        (Ok(from), Some(to)) => to != from && to.starts_with(&from),
        _ => false,
    }
}

/// The absolute form of `path` with its longest existing prefix canonicalized.
fn resolve(path: &Path) -> Option<PathBuf> {
    let absolute = std::path::absolute(path).ok()?;
    let mut missing = Vec::new();
    let mut current = absolute.as_path();
    loop {
        if let Ok(real) = fs::canonicalize(current) {
            return Some(missing.iter().rev().fold(real, |acc, part| acc.join(part)));
        }
        missing.push(current.file_name()?);
        current = current.parent()?;
    }
}

/// The names among `names` that must not reach a job (see [`PRIVATE_PREFIXES`]).
fn private_names(names: impl IntoIterator<Item = OsString>) -> Vec<OsString> {
    names
        .into_iter()
        .filter(|name| {
            PRIVATE_PREFIXES
                .iter()
                .any(|prefix| name.as_encoded_bytes().starts_with(prefix.as_bytes()))
        })
        .collect()
}

fn remove(path: &Path) -> Result<(), ExecError> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(io_error(path)(e)),
    }
}

fn spawn_error(source: io::Error) -> ExecError {
    ExecError::Spawn {
        program: "sh".to_string(),
        source,
    }
}

fn io_error(path: &Path) -> impl FnOnce(io::Error) -> ExecError + '_ {
    move |source| ExecError::Io {
        path: PathBuf::from(path),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_overbrainer_and_vault_variables_are_private() {
        let names = [
            "OVERBRAINER_CONFIG",
            "VAULT_TOKEN",
            "VAULT_ADDR",
            "PATH",
            "CUDA_VISIBLE_DEVICES",
            "HF_TOKEN",
            "MY_OVERBRAINER_X",
            "vault_token",
        ]
        .map(OsString::from);
        assert_eq!(
            private_names(names),
            ["OVERBRAINER_CONFIG", "VAULT_TOKEN", "VAULT_ADDR"].map(OsString::from)
        );
    }

    #[test]
    fn a_destination_inside_the_source_is_nested() -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let source = root.path().join("source");
        fs::create_dir_all(source.join("existing"))?;
        assert!(nested(&source, &source.join("existing")));
        assert!(nested(&source, &source.join("missing/deeper")));
        assert!(!nested(&source, &source));
        assert!(!nested(&source, &root.path().join("sibling")));
        assert!(!nested(&source, &root.path().join("source-2")));
        Ok(())
    }
}
