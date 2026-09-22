use std::path::Path;
use std::time::{Duration, SystemTime};

use secrecy::SecretString;

use super::{MetricsSummary, RunRecord, RunState, Runs, RunsError, new_run_id, rfc3339};
use crate::events::{Event, EventBus};
use crate::exec::{
    ExecError, Executor, JOB_LOG, JobId, JobRuntime, JobSpec, JobStatus, LineStream,
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
}

/// Creates a run for `target` on `executor`: a new ID, and its record saved as
/// `Preparing`.
///
/// # Errors
///
/// Returns [`RunsError`] when the record cannot be saved.
pub fn create<E: Executor>(
    runs: &Runs,
    executor: &E,
    target: &str,
) -> Result<RunRecord, RunsError> {
    let now = SystemTime::now();
    let id = new_run_id(now);
    let record = RunRecord {
        remote_dir: format!("{}/{id}", executor.workdir()),
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
/// followed again: its outcome comes from the local metrics file.
///
/// When the artifacts cannot be retrieved, a run that would have succeeded is left
/// `Running` and the error is returned, so a later attach tries again. A run that
/// failed or was cancelled is saved in that state anyway, with
/// `artifacts not retrieved: <error>` added to its message: there is nothing worth
/// retrying for.
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
        return Ok(Outcome { record, summary });
    }
    let metrics = format!("{}/{}", record.remote_dir, trainer.metrics_file());
    let mut stream = ctx.executor.tail(&metrics, 0);
    let mut summary = MetricsSummary::default();
    let status = follow(ctx, &job, &mut stream, &mut summary).await?;
    let (state, message) = outcome(status, &summary, &record.id);
    let downloaded = retrieve(ctx.executor, trainer, &record.remote_dir, &local).await;
    record.message = match downloaded {
        Ok(()) => message,
        Err(error) if state == RunState::Succeeded => return Err(error.into()),
        Err(error) => Some(not_retrieved(message, &error)),
    };
    record.state = state;
    ctx.runs.save(&record)?;
    Ok(Outcome { record, summary })
}

/// Copies the trainer's artifacts and the job log from `remote` into `local`.
async fn retrieve<E: Executor, T: Trainer>(
    executor: &E,
    trainer: &T,
    remote: &str,
    local: &Path,
) -> Result<(), ExecError> {
    let artifacts = trainer.artifacts();
    let mut entries = artifacts.entries;
    entries.push(JOB_LOG.to_string());
    executor
        .download(remote, local, &entries, &artifacts.exclude)
        .await
}

/// `message` with the reason the artifacts could not be retrieved added to it.
fn not_retrieved(message: Option<String>, error: &ExecError) -> String {
    match message {
        Some(message) => format!("{message} (artifacts not retrieved: {error})"),
        None => format!("artifacts not retrieved: {error}"),
    }
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
/// job log are then copied into the local run directory, since nobody may be
/// watching the run to do it. That copy is best effort: when it fails, the failure
/// is logged and the record stays `Cancelled` with
/// `artifacts not retrieved: <error>` as its message. Saving that message is best
/// effort too: `Cancelled` is already on disk, so a failure to save it is logged
/// and the cancel still succeeds.
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
) -> Result<(RunRecord, JobStatus), RunError> {
    let job = record
        .job
        .clone()
        .ok_or_else(|| RunError::NotStarted(record.id.clone()))?;
    executor.cancel(&job).await?;
    let status = executor.status(&job).await?;
    if status == JobStatus::Cancelled {
        record.state = RunState::Cancelled;
        record.message = None;
        runs.save(&record)?;
        retrieve_cancelled(runs, executor, trainer, &mut record).await?;
    }
    Ok((record, status))
}

/// Copies the artifacts of the cancelled run `record`, best effort: a failure is
/// logged and noted in its message, which is saved when possible.
async fn retrieve_cancelled<E: Executor, T: Trainer>(
    runs: &Runs,
    executor: &E,
    trainer: &T,
    record: &mut RunRecord,
) -> Result<(), RunError> {
    let local = runs.run_dir(&record.id)?;
    let Err(error) = retrieve(executor, trainer, &record.remote_dir, &local).await else {
        return Ok(());
    };
    cancelled_warning("retrieve the artifacts of", &record.id, &error);
    record.message = Some(not_retrieved(None, &error));
    if let Err(save_error) = runs.save(record) {
        cancelled_warning("note the missing artifacts of", &record.id, &save_error);
    }
    Ok(())
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
    use crate::exec::{JobCommand, MAX_TAIL_READ, Pid};
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
        let created = create(&runs, &fake, "box")?;
        break_saves(&runs, &created.id)?;
        let launch = Launch {
            runtime: &JobRuntime::Native { venv: None },
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
        let created = create(&runs, &fake, "box")?;
        break_saves(&runs, &created.id)?;
        let launch = Launch {
            runtime: &JobRuntime::Native { venv: None },
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
        let (cancelled, status) = cancel(&runs, &fake, &NoFiles, record).await?;
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
        let (cancelled, status) = cancel(&runs, &fake, &NoFiles, record).await?;
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
}
