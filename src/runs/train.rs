use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use secrecy::SecretString;

use super::{MetricsSummary, RunRecord, RunState, Runs, RunsError, new_run_id, rfc3339};
use crate::events::{Event, EventBus};
use crate::exec::{
    ExecError, Executor, FileDigest, JOB_LOG, JobId, JobRuntime, JobSpec, JobStatus, LineStream,
    local_manifest, sha256_file,
};
use crate::train::{TrainError, Trainer};

/// Hugging Face cache on the target, under the executor's work directory. Shared
/// by the runs of that target so a base model is downloaded once.
pub const HF_CACHE_DIR: &str = ".hf-cache";

/// Failures in a row to reach the target that a watch retries: the sixth one in a
/// row gives up (the job keeps running and can be attached again). A success in the
/// poll loop starts the count again; the drain that follows the job's end spends
/// what is left of it.
const MAX_FAILURES: u32 = 5;

/// Start of the note added to a run's message when its artifacts could not be
/// retrieved.
const NOT_RETRIEVED: &str = "artifacts not retrieved: ";

/// Errors of the train, attach and cancel flows.
#[derive(Debug, thiserror::Error)]
pub enum RunError {
    /// A run record cannot be read or written.
    #[error(transparent)]
    Runs(#[from] RunsError),
    /// The run's files cannot be prepared.
    #[error(transparent)]
    Train(#[from] TrainError),
    /// The target failed or cannot be reached.
    #[error(transparent)]
    Exec(#[from] ExecError),
    /// The run never started a job.
    #[error("run {0} has no job: it stopped before the job started")]
    NotStarted(String),
    /// [`collect`] was asked for a run that has not ended yet.
    #[error(
        "run {0} has not ended yet: watch or attach it, not collect, while it is preparing or running"
    )]
    NotEnded(String),
}

/// What the flows share: where runs live, the target, the event bus, and how often
/// the job is polled.
#[derive(Debug, Clone, Copy)]
pub struct RunCtx<'a, E> {
    /// The project's `runs/`.
    pub runs: &'a Runs,
    /// The target.
    pub executor: &'a E,
    /// Where metric and status events go.
    pub bus: &'a EventBus,
    /// Time between two looks at the job.
    pub poll: Duration,
}

/// How to start the job of a new run.
#[derive(Debug)]
pub struct Launch<'a> {
    /// How the job runs on the target.
    pub runtime: &'a JobRuntime,
    /// Secret env variables of the job (the Hugging Face token).
    pub secrets: Vec<(String, SecretString)>,
}

/// How a run ended.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// The final record: `Succeeded`, `Failed` (with its message) or `Cancelled`.
    pub record: RunRecord,
    /// What the metric lines add up to.
    pub summary: MetricsSummary,
    /// Whether the artifacts and the job log were downloaded and each file checked
    /// against the SHA-256 the target computed for it.
    pub retrieved: bool,
}

/// Creates a run for `target`, whose run directories live in `workdir` on the
/// target: a new ID, and its record saved as `Preparing`. `workdir` is the
/// executor's [`Executor::workdir`], or, for a target whose executor only exists
/// later (a Runpod pod), the directory it will have.
///
/// # Errors
///
/// Returns [`RunsError`] when the record cannot be saved.
pub fn create(runs: &Runs, workdir: &str, target: &str) -> Result<RunRecord, RunsError> {
    let now = SystemTime::now();
    let id = new_run_id(now);
    let record = RunRecord {
        remote_dir: format!("{workdir}/{id}"),
        id,
        target: target.to_string(),
        created: rfc3339(now),
        job: None,
        state: RunState::Preparing,
        message: None,
    };
    runs.save(&record)?;
    Ok(record)
}

/// Prepares the files of the run `record` (from [`create`]), copies them to the
/// target and starts the job. The record is saved with the job, as `Running`.
/// The `Running` status event is left to [`watch`], which reports it on its first
/// look at the job.
///
/// # Errors
///
/// Returns a [`RunError`] when a file cannot be prepared, copied, or the job cannot
/// start; the record is then saved as `Failed` with the reason (a failure to save
/// it is only logged, and the original error returned). When the job started but
/// its record cannot be saved, nothing could find the job again: it is cancelled,
/// on a best effort basis, and the save error is returned.
pub async fn start<E: Executor, T: Trainer>(
    ctx: &RunCtx<'_, E>,
    trainer: &T,
    launch: Launch<'_>,
    mut record: RunRecord,
) -> Result<RunRecord, RunError> {
    match launch_job(ctx, trainer, launch, &record).await {
        Ok(job) => record_started(ctx, record, job).await,
        Err(error) => {
            record.state = RunState::Failed;
            record.message = Some(error.to_string());
            if let Err(save_error) = ctx.runs.save(&record) {
                tracing::warn!("cannot record run {} as failed: {save_error}", record.id);
            }
            Err(error)
        },
    }
}

/// Saves `record` as `Running` with its started `job`. When that fails, the job
/// is cancelled on a best effort basis, since nothing could find it again.
async fn record_started<E: Executor>(
    ctx: &RunCtx<'_, E>,
    mut record: RunRecord,
    job: JobId,
) -> Result<RunRecord, RunError> {
    record.job = Some(job.clone());
    record.state = RunState::Running;
    let Err(error) = ctx.runs.save(&record) else {
        return Ok(record);
    };
    if let Err(cancel_error) = ctx.executor.cancel(&job).await {
        tracing::warn!(
            "cannot cancel the job of run {}, which is not recorded: {cancel_error}",
            record.id
        );
    }
    Err(error.into())
}

async fn launch_job<E: Executor, T: Trainer>(
    ctx: &RunCtx<'_, E>,
    trainer: &T,
    launch: Launch<'_>,
    record: &RunRecord,
) -> Result<JobId, RunError> {
    let local = ctx.runs.run_dir(&record.id)?;
    let root = launch.runtime.root(&record.remote_dir);
    trainer.prepare(&local, &root)?;
    ctx.executor.upload(&local, &record.remote_dir).await?;
    let cache_dir = format!("{}/{HF_CACHE_DIR}", ctx.executor.workdir());
    let job = launch.runtime.job(JobSpec {
        run_id: &record.id,
        run_dir: &record.remote_dir,
        cache_dir: &cache_dir,
        commands: &trainer.commands(),
        env: &trainer.env(&root),
        secrets: launch.secrets,
    });
    Ok(ctx.executor.spawn(&job).await?)
}

/// Follows a started run until its job ends, publishing metrics and status changes,
/// then retrieves the artifacts and saves the outcome. A run already ended is not
/// followed again: its outcome comes from the local metrics file, and
/// [`Outcome::retrieved`] holds only when its message carries no
/// `artifacts not retrieved` note and its job log was downloaded.
///
/// Retrieving means taking the SHA-256 manifest of the files on the target,
/// downloading them, and checking that the local files are exactly the manifest's:
/// none missing, none with a different hash, and none the target did not have.
///
/// A transient failure to retrieve (the target unreachable, a download error, a
/// file missing or mismatched locally) leaves a run that would have succeeded
/// `Running` and returns the error, so a later attach tries again. A run that
/// failed or was cancelled is saved in that state anyway, with
/// `artifacts not retrieved: <error>` added to its message and
/// [`Outcome::retrieved`] false: there is nothing worth retrying for, though
/// [`collect`] can still fetch its files.
///
/// A succeeded job whose target has no file in the trainer's required entry
/// ([`Artifacts::required`](crate::train::Artifacts::required)) is different: that
/// is permanent, not something a retry could fix, so once what does exist (the job
/// log, the metrics) has been downloaded and verified, the run is recorded
/// `Failed` instead, with a message saying so.
///
/// # Errors
///
/// Returns [`RunError::NotStarted`] for a run without a job, and an error when the
/// target stays unreachable or the artifacts of a successful run cannot be
/// retrieved; the job keeps running (or its files stay on the target) and the run
/// can be attached again.
pub async fn watch<E: Executor, T: Trainer>(
    ctx: &RunCtx<'_, E>,
    trainer: &T,
    mut record: RunRecord,
) -> Result<Outcome, RunError> {
    let job = record
        .job
        .clone()
        .ok_or_else(|| RunError::NotStarted(record.id.clone()))?;
    let local = ctx.runs.run_dir(&record.id)?;
    if record.state != RunState::Running {
        let summary = local_summary(&local.join(trainer.metrics_file()))?;
        let retrieved = !artifacts_missing(&record) && local.join(JOB_LOG).is_file();
        return Ok(Outcome {
            record,
            summary,
            retrieved,
        });
    }
    let metrics = format!("{}/{}", record.remote_dir, trainer.metrics_file());
    let mut stream = ctx.executor.tail(&metrics, 0);
    let mut summary = MetricsSummary::default();
    let status = follow(ctx, &job, &mut stream, &mut summary).await?;
    let (state, message) = outcome(status, &summary, &record.id);
    let succeeded = state == RunState::Succeeded;
    let (state, message, retrieved) =
        match retrieve(ctx.executor, trainer, &record.remote_dir, &local, succeeded).await {
            Ok(Retrieved::Ok) => (state, message, true),
            Ok(Retrieved::NoOutput) => {
                (RunState::Failed, Some(no_output_message(&record.id)), false)
            },
            Err(error) if succeeded => return Err(error.into()),
            Err(error) => (state, Some(not_retrieved(message, &error)), false),
        };
    record.message = message;
    record.state = state;
    ctx.runs.save(&record)?;
    Ok(Outcome {
        record,
        summary,
        retrieved,
    })
}

/// What [`retrieve`] found while checking a succeeded job's required output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Retrieved {
    /// Every local file matched the target's manifest exactly (`succeeded` was
    /// false, or the trainer names no required entry, or that entry held a file).
    Ok,
    /// The job succeeded, but the target's required entry held no file: a
    /// permanent condition, since nothing more will appear there for a retry to
    /// find.
    NoOutput,
}

/// Copies the trainer's artifacts and the job log from `remote` into `local`, then
/// checks that the local files under the trainer's entries are exactly the ones
/// the target's SHA-256 manifest lists, with matching hashes: no file missing, none
/// with a different hash, and none extra that the target did not have. When
/// `succeeded`, a target whose required entry holds no file is reported as
/// [`Retrieved::NoOutput`] rather than as an error, once what does exist has been
/// downloaded and verified.
async fn retrieve<E: Executor, T: Trainer>(
    executor: &E,
    trainer: &T,
    remote: &str,
    local: &Path,
    succeeded: bool,
) -> Result<Retrieved, ExecError> {
    let artifacts = trainer.artifacts();
    let mut entries = artifacts.entries;
    entries.push(JOB_LOG.to_string());
    let manifest = executor
        .manifest(remote, &entries, &artifacts.exclude)
        .await?;
    let no_output = succeeded
        && artifacts
            .required
            .as_deref()
            .is_some_and(|required| !has_required(&manifest, required));
    executor
        .download(remote, local, &entries, &artifacts.exclude)
        .await?;
    let local = local.to_path_buf();
    let exclude = artifacts.exclude.clone();
    tokio::task::spawn_blocking(move || verify(&local, &entries, &exclude, &manifest))
        .await
        .map_err(|error| ExecError::Protocol(format!("the verification task failed: {error}")))??;
    Ok(if no_output {
        Retrieved::NoOutput
    } else {
        Retrieved::Ok
    })
}

/// Whether `manifest` holds a file at or under `required`.
fn has_required(manifest: &[FileDigest], required: &str) -> bool {
    let inside = format!("{required}/");
    manifest
        .iter()
        .any(|file| file.path == required || file.path.starts_with(&inside))
}

/// Checks that the local files under `entries` of `local` are exactly the files
/// `manifest` lists, each with the SHA-256 the target computed for it: none
/// missing, none with a different hash, and none the target did not have.
fn verify(
    local: &Path,
    entries: &[String],
    exclude: &[String],
    manifest: &[FileDigest],
) -> Result<(), ExecError> {
    for file in manifest {
        let path: PathBuf = local.join(&file.path);
        let sha256 = match sha256_file(&path) {
            Ok(sha256) => sha256,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(verify_error(format!("{:?} is missing locally", file.path)));
            },
            Err(source) => return Err(ExecError::Io { path, source }),
        };
        if sha256 != file.sha256 {
            return Err(verify_error(format!(
                "{:?} differs from the target (SHA-256 mismatch)",
                file.path
            )));
        }
    }
    let local_files = local_manifest(local, entries, exclude)?;
    if let Some(extra) = local_files
        .iter()
        .find(|file| !manifest.iter().any(|listed| listed.path == file.path))
    {
        return Err(verify_error(format!(
            "{:?} is not on the target",
            extra.path
        )));
    }
    Ok(())
}

fn verify_error(message: String) -> ExecError {
    ExecError::Command {
        action: "verify",
        message,
    }
}

/// The message recorded when a succeeded job leaves no file in its required
/// entry: a permanent condition, since a retry cannot make the target produce
/// what was never written.
fn no_output_message(id: &str) -> String {
    format!("the job succeeded but left no output (see runs/{id}/{JOB_LOG})")
}

/// `message` with the reason the artifacts could not be retrieved added to it.
fn not_retrieved(message: Option<String>, error: &ExecError) -> String {
    match message {
        Some(message) => format!("{message} ({NOT_RETRIEVED}{error})"),
        None => format!("{NOT_RETRIEVED}{error}"),
    }
}

/// Whether the record of an ended run says its artifacts were not retrieved.
#[must_use]
pub fn artifacts_missing(record: &RunRecord) -> bool {
    record
        .message
        .as_deref()
        .is_some_and(|message| message.contains(NOT_RETRIEVED))
}

/// `message` without the note [`not_retrieved`] added to it.
fn without_note(message: Option<String>) -> Option<String> {
    let message = message?;
    if message.starts_with(NOT_RETRIEVED) {
        return None;
    }
    match message.find(&format!(" ({NOT_RETRIEVED}")) {
        Some(index) => Some(message[..index].to_string()),
        None => Some(message),
    }
}

/// Retrieves again the artifacts of a run already in a final state, whose target
/// still holds its files: downloads them, checks them against the target's
/// SHA-256 manifest, and on success removes the `artifacts not retrieved` note
/// from the record's message and saves it. A failure to retrieve is logged and
/// leaves the record as it was. Returns the record and whether the files were
/// retrieved.
///
/// # Errors
///
/// Returns [`RunError::NotEnded`] for a run still `Preparing` or `Running`: there
/// is nothing to collect yet, `watch` or `attach` follows it instead. Also returns
/// [`RunError::Runs`] when the run directory is invalid or the updated record
/// cannot be saved.
pub async fn collect<E: Executor, T: Trainer>(
    ctx: &RunCtx<'_, E>,
    trainer: &T,
    mut record: RunRecord,
) -> Result<(RunRecord, bool), RunError> {
    if matches!(record.state, RunState::Preparing | RunState::Running) {
        return Err(RunError::NotEnded(record.id));
    }
    let local = ctx.runs.run_dir(&record.id)?;
    let succeeded = record.state == RunState::Succeeded;
    if let Err(error) = retrieve(ctx.executor, trainer, &record.remote_dir, &local, succeeded).await
    {
        tracing::warn!(
            "cannot retrieve the artifacts of run {}: {error}",
            record.id
        );
        return Ok((record, false));
    }
    if artifacts_missing(&record) {
        record.message = without_note(record.message.take());
        ctx.runs.save(&record)?;
    }
    Ok((record, true))
}

/// Reads new metric lines and the job status until the job ends, then drains the
/// rest of the metrics file: one read fetches at most [`MAX_TAIL_READ`] bytes, so a
/// file larger than that needs several. Up to [`MAX_FAILURES`] failures in a row are
/// retried, for the drain as for the rest.
///
/// [`MAX_TAIL_READ`]: crate::exec::MAX_TAIL_READ
async fn follow<E: Executor>(
    ctx: &RunCtx<'_, E>,
    job: &JobId,
    stream: &mut LineStream<'_, E>,
    summary: &mut MetricsSummary,
) -> Result<JobStatus, RunError> {
    let mut failures = 0;
    let mut last = None;
    let status = loop {
        match poll(ctx, job, stream, summary).await {
            Ok(status) => {
                failures = 0;
                if last != Some(status) {
                    ctx.bus.publish(Event::JobStatus(status));
                    last = Some(status);
                }
                if status.is_finished() {
                    break status;
                }
            },
            Err(error) => retry(&mut failures, error)?,
        }
        tokio::time::sleep(ctx.poll).await;
    };
    loop {
        let offset = stream.offset();
        match stream.read().await {
            Ok(lines) => {
                // The budget is not given back here: it is the same one the poll
                // loop was left with, spent over the whole drain.
                let drained = lines.is_empty() && stream.offset() == offset;
                publish(ctx.bus, lines, summary);
                if drained {
                    return Ok(status);
                }
            },
            Err(error) => {
                retry(&mut failures, error)?;
                tokio::time::sleep(ctx.poll).await;
            },
        }
    }
}

/// Counts one more failure in a row to reach the job, and returns it as an error
/// once [`MAX_FAILURES`] have been retried.
fn retry(failures: &mut u32, error: ExecError) -> Result<(), RunError> {
    if *failures < MAX_FAILURES {
        *failures += 1;
        unreachable_target(&error, *failures);
        Ok(())
    } else {
        Err(error.into())
    }
}

async fn poll<E: Executor>(
    ctx: &RunCtx<'_, E>,
    job: &JobId,
    stream: &mut LineStream<'_, E>,
    summary: &mut MetricsSummary,
) -> Result<JobStatus, ExecError> {
    publish(ctx.bus, stream.read().await?, summary);
    ctx.executor.status(job).await
}

fn publish(bus: &EventBus, lines: Vec<String>, summary: &mut MetricsSummary) {
    for line in lines {
        if let Some(metric) = summary.add(&line) {
            bus.publish(Event::Metric(metric));
        }
    }
}

fn unreachable_target(error: &ExecError, failures: u32) {
    tracing::warn!("cannot reach the job ({failures}/{MAX_FAILURES}): {error}");
}

/// The final state of a run whose job ended with `status`.
fn outcome(status: JobStatus, summary: &MetricsSummary, id: &str) -> (RunState, Option<String>) {
    let log = format!("see runs/{id}/{JOB_LOG}");
    match status {
        JobStatus::Exited(0) if summary.lines > 0 => (RunState::Succeeded, None),
        JobStatus::Exited(0) => (
            RunState::Failed,
            Some(format!(
                "the job wrote no metrics: the overbrainer metrics plugin was not loaded ({log})"
            )),
        ),
        JobStatus::Exited(code) => (
            RunState::Failed,
            Some(format!("the job exited with code {code} ({log})")),
        ),
        JobStatus::Cancelled => (RunState::Cancelled, None),
        JobStatus::Lost | JobStatus::Running => (
            RunState::Failed,
            Some(format!("the job stopped without an exit code ({log})")),
        ),
    }
}

/// Summary of a local `metrics.jsonl`; a missing file sums to nothing.
fn local_summary(path: &Path) -> Result<MetricsSummary, RunError> {
    let content = match std::fs::read(path) {
        Ok(content) => content,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(source) => {
            return Err(RunError::Runs(RunsError::Io {
                path: path.to_path_buf(),
                source,
            }));
        },
    };
    let mut summary = MetricsSummary::default();
    for line in String::from_utf8_lossy(&content).lines() {
        if !line.trim().is_empty() {
            summary.add(line);
        }
    }
    Ok(summary)
}

/// Cancels the job of a started run, then reads its status back.
///
/// Only when the job reads [`JobStatus::Cancelled`] is the record changed: it is
/// saved as `Cancelled`, without a message, and the trainer's artifacts and the
/// job log are then copied into the local run directory and checked against the
/// target's SHA-256 manifest, since nobody may be watching the run to do it. That
/// copy is best effort: when it fails, the failure is logged and the record stays
/// `Cancelled` with `artifacts not retrieved: <error>` as its message. Saving that
/// message is best effort too: `Cancelled` is already on disk, so a failure to
/// save it is logged and the cancel still succeeds. The returned flag says whether
/// the files were retrieved and verified.
///
/// A job that had already ended before the cancel is left alone by the target, so
/// its status stays [`JobStatus::Exited`] or [`JobStatus::Lost`]; the record is
/// then returned unchanged, keeping whatever state it held (`Running`, or a final
/// state for a run cancelled after it was recorded as ended), so a later watch or
/// attach records the real outcome and retrieves its artifacts. The status is
/// returned with the record so the caller can tell which case happened.
///
/// The record's state is not looked at: a run recorded as ended may still have a
/// job, or a container, left on the target, and cancelling it stops that.
///
/// # Errors
///
/// Returns [`RunError::NotStarted`] for a run without a job, and an error when the
/// target cannot be reached or the `Cancelled` record cannot be saved.
pub async fn cancel<E: Executor, T: Trainer>(
    runs: &Runs,
    executor: &E,
    trainer: &T,
    mut record: RunRecord,
) -> Result<(RunRecord, JobStatus, bool), RunError> {
    let job = record
        .job
        .clone()
        .ok_or_else(|| RunError::NotStarted(record.id.clone()))?;
    executor.cancel(&job).await?;
    let status = executor.status(&job).await?;
    let mut retrieved = false;
    if status == JobStatus::Cancelled {
        record.state = RunState::Cancelled;
        record.message = None;
        runs.save(&record)?;
        retrieved = retrieve_cancelled(runs, executor, trainer, &mut record).await?;
    }
    Ok((record, status, retrieved))
}

/// Copies the artifacts of the cancelled run `record`, best effort: a failure is
/// logged and noted in its message, which is saved when possible. Returns whether
/// they were retrieved.
async fn retrieve_cancelled<E: Executor, T: Trainer>(
    runs: &Runs,
    executor: &E,
    trainer: &T,
    record: &mut RunRecord,
) -> Result<bool, RunError> {
    let local = runs.run_dir(&record.id)?;
    let Err(error) = retrieve(executor, trainer, &record.remote_dir, &local, false).await else {
        return Ok(true);
    };
    cancelled_warning("retrieve the artifacts of", &record.id, &error);
    record.message = Some(not_retrieved(None, &error));
    if let Err(save_error) = runs.save(record) {
        cancelled_warning("note the missing artifacts of", &record.id, &save_error);
    }
    Ok(false)
}

fn cancelled_warning(what: &str, id: &str, error: &dyn std::fmt::Display) {
    tracing::warn!("cannot {what} cancelled run {id}: {error}");
}

#[cfg(test)]
mod tests {
    use std::future::{Future, ready};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;
    use crate::exec::{FileDigest, JobCommand, MAX_TAIL_READ, Pid};
    use crate::runs::RECORD_FILE;
    use crate::train::Artifacts;

    #[test]
    fn outcomes_follow_the_exit_code_and_the_metrics() {
        let mut summary = MetricsSummary::default();
        let (state, message) = outcome(JobStatus::Exited(0), &summary, "r1");
        assert_eq!(state, RunState::Failed);
        assert!(message.is_some_and(|message| message.contains("plugin was not loaded")));
        summary.add(r#"{"event": "begin", "time": 1}"#);
        assert_eq!(
            outcome(JobStatus::Exited(0), &summary, "r1"),
            (RunState::Succeeded, None)
        );
        assert_eq!(
            outcome(JobStatus::Exited(2), &summary, "r1"),
            (
                RunState::Failed,
                Some("the job exited with code 2 (see runs/r1/job.log)".to_string())
            )
        );
        assert_eq!(
            outcome(JobStatus::Cancelled, &summary, "r1"),
            (RunState::Cancelled, None)
        );
        assert_eq!(outcome(JobStatus::Lost, &summary, "r1").0, RunState::Failed);
    }

    #[test]
    fn the_download_error_is_added_to_the_message() {
        let error = ExecError::Command {
            action: "download",
            message: "/w/r1 does not exist".to_string(),
        };
        assert_eq!(
            not_retrieved(Some("the job exited with code 2".to_string()), &error),
            "the job exited with code 2 (artifacts not retrieved: download failed: /w/r1 does not exist)"
        );
        assert_eq!(
            not_retrieved(None, &error),
            "artifacts not retrieved: download failed: /w/r1 does not exist"
        );
    }

    const METRICS: &str = "{\"event\": \"begin\", \"time\": 1}\n";
    const RUN_ID: &str = "20260922-143005-abcd";

    /// A scripted target. Its job reads `status`, its metrics file holds
    /// `metrics`, and the read numbered `failing_read` (from 0) fails.
    struct Fake {
        status: JobStatus,
        metrics: String,
        spawn_fails: bool,
        download_fails: bool,
        failing_read: Option<u32>,
        /// A directory created when a download is asked for, before it fails.
        dir_on_download: Option<PathBuf>,
        /// What the target's manifest lists.
        manifest: Vec<FileDigest>,
        reads: AtomicU32,
        cancels: AtomicU32,
    }

    impl Fake {
        fn new(status: JobStatus) -> Self {
            Self {
                status,
                metrics: METRICS.to_string(),
                spawn_fails: false,
                download_fails: false,
                failing_read: None,
                dir_on_download: None,
                manifest: Vec::new(),
                reads: AtomicU32::new(0),
                cancels: AtomicU32::new(0),
            }
        }
    }

    fn broken(action: &'static str) -> ExecError {
        ExecError::Command {
            action,
            message: "connection reset".to_string(),
        }
    }

    fn job() -> Result<JobId, ExecError> {
        Ok(JobId {
            dir: format!("/w/{RUN_ID}"),
            pid: Pid::new(42)?,
            container: None,
        })
    }

    impl Executor for Fake {
        fn workdir(&self) -> &'static str {
            "/w"
        }

        fn upload(
            &self,
            _local: &Path,
            _remote: &str,
        ) -> impl Future<Output = Result<(), ExecError>> + Send {
            ready(Ok(()))
        }

        fn spawn(
            &self,
            _job: &JobCommand,
        ) -> impl Future<Output = Result<JobId, ExecError>> + Send {
            ready(if self.spawn_fails {
                Err(broken("spawn"))
            } else {
                job()
            })
        }

        fn read_from(
            &self,
            _path: &str,
            offset: u64,
            limit: u64,
        ) -> impl Future<Output = Result<Vec<u8>, ExecError>> + Send {
            let read = self.reads.fetch_add(1, Ordering::SeqCst);
            let start = usize::try_from(offset).unwrap_or(self.metrics.len());
            let mut bytes = self
                .metrics
                .as_bytes()
                .get(start..)
                .unwrap_or_default()
                .to_vec();
            bytes.truncate(usize::try_from(limit).unwrap_or(usize::MAX));
            ready(if self.failing_read == Some(read) {
                Err(broken("read"))
            } else {
                Ok(bytes)
            })
        }

        fn status(
            &self,
            _job: &JobId,
        ) -> impl Future<Output = Result<JobStatus, ExecError>> + Send {
            ready(Ok(self.status))
        }

        fn cancel(&self, _job: &JobId) -> impl Future<Output = Result<(), ExecError>> + Send {
            self.cancels.fetch_add(1, Ordering::SeqCst);
            ready(Ok(()))
        }

        fn download(
            &self,
            _remote: &str,
            _local: &Path,
            _entries: &[String],
            _exclude: &[String],
        ) -> impl Future<Output = Result<(), ExecError>> + Send {
            if let Some(dir) = &self.dir_on_download {
                std::fs::create_dir(dir).ok();
            }
            ready(if self.download_fails {
                Err(broken("download"))
            } else {
                Ok(())
            })
        }

        fn manifest(
            &self,
            _remote: &str,
            _entries: &[String],
            _exclude: &[String],
        ) -> impl Future<Output = Result<Vec<FileDigest>, ExecError>> + Send {
            ready(Ok(self.manifest.clone()))
        }
    }

    struct NoFiles;

    impl Trainer for NoFiles {
        fn prepare(&self, _run_dir: &Path, _root: &str) -> Result<(), TrainError> {
            Ok(())
        }

        fn commands(&self) -> Vec<Vec<String>> {
            Vec::new()
        }

        fn env(&self, _root: &str) -> Vec<(String, String)> {
            Vec::new()
        }

        fn metrics_file(&self) -> &'static str {
            "metrics.jsonl"
        }

        fn artifacts(&self) -> Artifacts {
            Artifacts {
                entries: Vec::new(),
                exclude: Vec::new(),
                required: None,
            }
        }
    }

    /// A trainer whose successful runs must leave a file in `output/`.
    struct WithOutput;

    impl Trainer for WithOutput {
        fn prepare(&self, _run_dir: &Path, _root: &str) -> Result<(), TrainError> {
            Ok(())
        }

        fn commands(&self) -> Vec<Vec<String>> {
            Vec::new()
        }

        fn env(&self, _root: &str) -> Vec<(String, String)> {
            Vec::new()
        }

        fn metrics_file(&self) -> &'static str {
            "metrics.jsonl"
        }

        fn artifacts(&self) -> Artifacts {
            Artifacts {
                entries: vec!["output".to_string()],
                exclude: Vec::new(),
                required: Some("output".to_string()),
            }
        }
    }

    fn running() -> Result<RunRecord, ExecError> {
        Ok(RunRecord {
            id: RUN_ID.to_string(),
            target: "box".to_string(),
            created: "2026-09-22T14:30:05Z".to_string(),
            remote_dir: format!("/w/{RUN_ID}"),
            job: Some(job()?),
            state: RunState::Running,
            message: None,
        })
    }

    fn ctx<'a>(runs: &'a Runs, executor: &'a Fake, bus: &'a EventBus) -> RunCtx<'a, Fake> {
        RunCtx {
            runs,
            executor,
            bus,
            poll: Duration::from_millis(1),
        }
    }

    /// Makes every later save of the run `id` fail: its temporary file is taken by
    /// a directory.
    fn break_saves(runs: &Runs, id: &str) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir(runs.run_dir(id)?.join(format!(".{RECORD_FILE}.tmp")))?;
        Ok(())
    }

    #[tokio::test]
    async fn a_successful_run_whose_artifacts_cannot_be_retrieved_stays_running()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let fake = Fake {
            download_fails: true,
            ..Fake::new(JobStatus::Exited(0))
        };
        let record = running()?;
        runs.save(&record)?;
        let error = watch(&ctx(&runs, &fake, &bus), &NoFiles, record.clone())
            .await
            .err()
            .ok_or("the watch succeeded")?;
        assert!(error.to_string().contains("connection reset"), "{error}");
        assert_eq!(runs.load(&record.id)?, record);
        Ok(())
    }

    #[tokio::test]
    async fn the_last_read_after_the_end_is_retried() -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let fake = Fake {
            failing_read: Some(1),
            ..Fake::new(JobStatus::Exited(0))
        };
        let record = running()?;
        runs.save(&record)?;
        let outcome = watch(&ctx(&runs, &fake, &bus), &NoFiles, record).await?;
        assert_eq!(outcome.record.state, RunState::Succeeded);
        assert_eq!(outcome.summary.lines, 1);
        assert_eq!(fake.reads.load(Ordering::SeqCst), 3);
        Ok(())
    }

    #[tokio::test]
    async fn the_final_drain_reads_the_whole_metrics_file() -> Result<(), Box<dyn std::error::Error>>
    {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        // Over twice what one read fetches, so the drain after the job ends has to
        // read again, and again, instead of summarizing a prefix.
        let line = r#"{"event": "log", "time": 1, "step": 7, "loss": 1.0}"#;
        let count = 2 * usize::try_from(MAX_TAIL_READ)? / (line.len() + 1) + 1;
        let mut metrics = String::with_capacity(count * (line.len() + 1));
        for _ in 0..count {
            metrics.push_str(line);
            metrics.push('\n');
        }
        let fake = Fake {
            metrics,
            ..Fake::new(JobStatus::Exited(0))
        };
        let record = running()?;
        runs.save(&record)?;
        let outcome = watch(&ctx(&runs, &fake, &bus), &NoFiles, record).await?;
        assert_eq!(outcome.record.state, RunState::Succeeded);
        assert_eq!(outcome.summary.lines, count);
        Ok(())
    }

    #[tokio::test]
    async fn a_job_whose_run_cannot_be_recorded_is_cancelled()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let fake = Fake::new(JobStatus::Running);
        let created = create(&runs, fake.workdir(), "box")?;
        break_saves(&runs, &created.id)?;
        let launch = Launch {
            runtime: &JobRuntime::Native {
                venv: None,
                env_file: None,
            },
            secrets: Vec::new(),
        };
        let result = start(&ctx(&runs, &fake, &bus), &NoFiles, launch, created).await;
        assert!(matches!(result, Err(RunError::Runs(_))), "{result:?}");
        assert_eq!(fake.cancels.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn a_failed_start_returns_its_own_error_when_it_cannot_be_recorded()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let fake = Fake {
            spawn_fails: true,
            ..Fake::new(JobStatus::Running)
        };
        let created = create(&runs, fake.workdir(), "box")?;
        break_saves(&runs, &created.id)?;
        let launch = Launch {
            runtime: &JobRuntime::Native {
                venv: None,
                env_file: None,
            },
            secrets: Vec::new(),
        };
        let result = start(&ctx(&runs, &fake, &bus), &NoFiles, launch, created).await;
        assert!(
            matches!(
                &result,
                Err(RunError::Exec(ExecError::Command {
                    action: "spawn",
                    ..
                }))
            ),
            "{result:?}"
        );
        assert_eq!(fake.cancels.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[tokio::test]
    async fn a_cancelled_run_keeps_cancelled_when_its_artifacts_cannot_be_retrieved()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let fake = Fake {
            download_fails: true,
            ..Fake::new(JobStatus::Cancelled)
        };
        let record = running()?;
        runs.save(&record)?;
        let (cancelled, status, retrieved) = cancel(&runs, &fake, &NoFiles, record).await?;
        assert!(!retrieved);
        assert_eq!(status, JobStatus::Cancelled);
        assert_eq!(cancelled.state, RunState::Cancelled);
        assert_eq!(
            cancelled.message.as_deref(),
            Some("artifacts not retrieved: download failed: connection reset")
        );
        assert_eq!(runs.load(&cancelled.id)?, cancelled);
        Ok(())
    }

    #[tokio::test]
    async fn a_cancel_is_reported_even_when_its_note_cannot_be_saved()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let record = running()?;
        runs.save(&record)?;
        // The download breaks every later save, after `Cancelled` is on disk.
        let fake = Fake {
            download_fails: true,
            dir_on_download: Some(runs.run_dir(RUN_ID)?.join(format!(".{RECORD_FILE}.tmp"))),
            ..Fake::new(JobStatus::Cancelled)
        };
        let (cancelled, status, retrieved) = cancel(&runs, &fake, &NoFiles, record).await?;
        assert!(!retrieved);
        assert_eq!(status, JobStatus::Cancelled);
        assert_eq!(cancelled.state, RunState::Cancelled);
        assert!(
            cancelled
                .message
                .as_deref()
                .is_some_and(|message| message.starts_with("artifacts not retrieved: ")),
            "{:?}",
            cancelled.message
        );
        let saved = runs.load(RUN_ID)?;
        assert_eq!(saved.state, RunState::Cancelled);
        assert_eq!(saved.message, None);
        Ok(())
    }

    #[tokio::test]
    async fn a_job_that_ended_before_the_first_poll_reports_only_its_end()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let mut receiver = bus.subscribe();
        let fake = Fake::new(JobStatus::Exited(0));
        let record = running()?;
        runs.save(&record)?;
        watch(&ctx(&runs, &fake, &bus), &NoFiles, record).await?;
        let mut statuses = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            if let Event::JobStatus(status) = event {
                statuses.push(status);
            }
        }
        assert_eq!(statuses, vec![JobStatus::Exited(0)]);
        Ok(())
    }

    /// SHA-256 of `b"w"`.
    const W: &str = "50e721e49c013f00c62cf59f2163542a9d8df02464efeb615d31051b0fddc326";

    fn listed(path: &str) -> FileDigest {
        FileDigest {
            path: path.to_string(),
            sha256: W.to_string(),
        }
    }

    /// Writes `content` at `path` inside the local run directory, as a download
    /// would have.
    fn downloaded(
        runs: &Runs,
        path: &str,
        content: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = runs.run_dir(RUN_ID)?.join(path);
        if let Some(parent) = file.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(file, content)?;
        Ok(())
    }

    async fn watch_with(
        manifest: Vec<FileDigest>,
        files: &[(&str, &str)],
    ) -> Result<(Runs, tempfile::TempDir, Result<Outcome, RunError>), Box<dyn std::error::Error>>
    {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let fake = Fake {
            manifest,
            ..Fake::new(JobStatus::Exited(0))
        };
        runs.save(&running()?)?;
        for (path, content) in files {
            downloaded(&runs, path, content)?;
        }
        let result = watch(&ctx(&runs, &fake, &bus), &WithOutput, running()?).await;
        Ok((runs, project, result))
    }

    #[tokio::test]
    async fn verified_files_make_a_retrieved_success() -> Result<(), Box<dyn std::error::Error>> {
        let (_, _project, result) = watch_with(
            vec![listed("output/adapter.bin"), listed("job.log")],
            &[("output/adapter.bin", "w"), ("job.log", "w")],
        )
        .await?;
        let outcome = result?;
        assert_eq!(outcome.record.state, RunState::Succeeded);
        assert!(outcome.retrieved);
        Ok(())
    }

    #[tokio::test]
    async fn a_missing_different_or_extra_file_leaves_a_success_running()
    -> Result<(), Box<dyn std::error::Error>> {
        let cases = [
            (
                &[("job.log", "w")][..],
                "verify failed: \"output/adapter.bin\" is missing locally",
            ),
            (
                &[("output/adapter.bin", "x"), ("job.log", "w")][..],
                "verify failed: \"output/adapter.bin\" differs from the target (SHA-256 mismatch)",
            ),
            (
                &[
                    ("output/adapter.bin", "w"),
                    ("job.log", "w"),
                    ("output/extra.bin", "w"),
                ][..],
                "verify failed: \"output/extra.bin\" is not on the target",
            ),
        ];
        for (files, expected) in cases {
            let (runs, _project, result) =
                watch_with(vec![listed("output/adapter.bin"), listed("job.log")], files).await?;
            let error = result.err().ok_or("the watch succeeded")?;
            assert_eq!(error.to_string(), expected, "{files:?}");
            assert_eq!(runs.load(RUN_ID)?.state, RunState::Running);
        }
        Ok(())
    }

    #[tokio::test]
    async fn a_success_without_output_is_a_permanent_failure()
    -> Result<(), Box<dyn std::error::Error>> {
        let (runs, _project, result) =
            watch_with(vec![listed("job.log")], &[("job.log", "w")]).await?;
        let outcome = result?;
        assert_eq!(outcome.record.state, RunState::Failed);
        assert!(!outcome.retrieved);
        assert_eq!(
            outcome.record.message.as_deref(),
            Some(
                format!("the job succeeded but left no output (see runs/{RUN_ID}/job.log)")
                    .as_str()
            )
        );
        assert_eq!(runs.load(RUN_ID)?, outcome.record);
        Ok(())
    }

    #[tokio::test]
    async fn a_failed_run_whose_verification_fails_gets_the_note()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let fake = Fake {
            manifest: vec![listed("job.log")],
            ..Fake::new(JobStatus::Exited(2))
        };
        runs.save(&running()?)?;
        // job.log is never written locally: verify fails, but the run already
        // failed, so it is recorded anyway, with the note added.
        let outcome = watch(&ctx(&runs, &fake, &bus), &NoFiles, running()?).await?;
        assert_eq!(outcome.record.state, RunState::Failed);
        assert!(!outcome.retrieved);
        let message = outcome.record.message.clone().unwrap_or_default();
        assert!(message.contains("the job exited with code 2"), "{message}");
        assert!(
            message.contains(
                "(artifacts not retrieved: verify failed: \"job.log\" is missing locally)"
            ),
            "{message}"
        );
        assert!(artifacts_missing(&outcome.record));
        assert_eq!(runs.load(RUN_ID)?, outcome.record);
        Ok(())
    }

    #[tokio::test]
    async fn a_final_record_is_retrieved_only_without_a_note_and_with_a_local_job_log()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let fake = Fake::new(JobStatus::Cancelled);
        let mut record = running()?;
        record.state = RunState::Cancelled;
        runs.save(&record)?;
        // No note on the message, but the job log was never downloaded (a cancel
        // whose note itself failed to save leaves exactly this).
        let outcome = watch(&ctx(&runs, &fake, &bus), &NoFiles, record.clone()).await?;
        assert!(!outcome.retrieved);
        // Once the job log is there, the record is reported retrieved.
        downloaded(&runs, JOB_LOG, "log")?;
        let outcome = watch(&ctx(&runs, &fake, &bus), &NoFiles, record).await?;
        assert!(outcome.retrieved);
        Ok(())
    }

    #[tokio::test]
    async fn a_cancelled_run_is_retrieved_when_nothing_prevents_it()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let fake = Fake::new(JobStatus::Cancelled);
        let record = running()?;
        runs.save(&record)?;
        let (cancelled, status, retrieved) = cancel(&runs, &fake, &NoFiles, record).await?;
        assert_eq!(status, JobStatus::Cancelled);
        assert_eq!(cancelled.state, RunState::Cancelled);
        assert!(retrieved);
        Ok(())
    }

    #[tokio::test]
    async fn collect_refuses_a_run_that_has_not_ended() -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let fake = Fake::new(JobStatus::Running);
        let record = running()?;
        runs.save(&record)?;
        let error = collect(&ctx(&runs, &fake, &bus), &NoFiles, record)
            .await
            .err()
            .ok_or("collect succeeded")?;
        assert!(matches!(error, RunError::NotEnded(_)), "{error:?}");
        Ok(())
    }

    #[tokio::test]
    async fn collect_retrieves_again_and_clears_the_note() -> Result<(), Box<dyn std::error::Error>>
    {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let mut record = running()?;
        record.state = RunState::Failed;
        record.message = Some(
            "the job exited with code 2 (artifacts not retrieved: download failed: reset)"
                .to_string(),
        );
        runs.save(&record)?;
        let broken = Fake {
            download_fails: true,
            ..Fake::new(JobStatus::Exited(2))
        };
        let (kept, retrieved) = collect(&ctx(&runs, &broken, &bus), &NoFiles, record).await?;
        assert!(!retrieved);
        assert!(artifacts_missing(&kept));
        let fake = Fake {
            manifest: vec![listed("job.log")],
            ..Fake::new(JobStatus::Exited(2))
        };
        downloaded(&runs, "job.log", "w")?;
        let (collected, retrieved) = collect(&ctx(&runs, &fake, &bus), &NoFiles, kept).await?;
        assert!(retrieved);
        assert_eq!(
            collected.message.as_deref(),
            Some("the job exited with code 2")
        );
        assert_eq!(runs.load(RUN_ID)?, collected);
        Ok(())
    }

    #[test]
    fn the_note_is_removed_from_either_form_of_message() {
        assert_eq!(
            without_note(Some("artifacts not retrieved: x".to_string())),
            None
        );
        assert_eq!(
            without_note(Some("boom (artifacts not retrieved: x)".to_string())),
            Some("boom".to_string())
        );
        assert_eq!(
            without_note(Some("boom".to_string())),
            Some("boom".to_string())
        );
        assert_eq!(without_note(None), None);
    }
}
