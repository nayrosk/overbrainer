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

/// Consecutive failures to reach the target before a watch gives up (the job keeps
/// running and can be attached again).
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
///
/// # Errors
///
/// Returns a [`RunError`] when a file cannot be prepared, copied, or the job cannot
/// start; the record is then saved as `Failed` with the reason.
pub async fn start<E: Executor, T: Trainer>(
    ctx: &RunCtx<'_, E>,
    trainer: &T,
    launch: Launch<'_>,
    mut record: RunRecord,
) -> Result<RunRecord, RunError> {
    match launch_job(ctx, trainer, launch, &record).await {
        Ok(job) => {
            record.job = Some(job);
            record.state = RunState::Running;
            ctx.runs.save(&record)?;
            ctx.bus.publish(Event::JobStatus(JobStatus::Running));
            Ok(record)
        },
        Err(error) => {
            record.state = RunState::Failed;
            record.message = Some(error.to_string());
            ctx.runs.save(&record)?;
            Err(error)
        },
    }
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
    let artifacts = trainer.artifacts();
    let mut entries = artifacts.entries;
    entries.push(JOB_LOG.to_string());
    let downloaded = ctx
        .executor
        .download(&record.remote_dir, &local, &entries, &artifacts.exclude)
        .await;
    record.message = match downloaded {
        Ok(()) => message,
        Err(error) if state == RunState::Succeeded => return Err(error.into()),
        Err(error) => Some(not_retrieved(message, &error)),
    };
    record.state = state;
    ctx.runs.save(&record)?;
    Ok(Outcome { record, summary })
}

/// `message` with the reason the artifacts could not be retrieved added to it.
fn not_retrieved(message: Option<String>, error: &ExecError) -> String {
    match message {
        Some(message) => format!("{message} (artifacts not retrieved: {error})"),
        None => format!("artifacts not retrieved: {error}"),
    }
}

/// Reads new metric lines and the job status until the job ends, then reads the
/// last lines. Up to [`MAX_FAILURES`] failures in a row are retried.
async fn follow<E: Executor>(
    ctx: &RunCtx<'_, E>,
    job: &JobId,
    stream: &mut LineStream<'_, E>,
    summary: &mut MetricsSummary,
) -> Result<JobStatus, RunError> {
    let mut failures = 0;
    let mut last = None;
    loop {
        match poll(ctx, job, stream, summary).await {
            Ok(status) => {
                failures = 0;
                if last != Some(status) {
                    ctx.bus.publish(Event::JobStatus(status));
                    last = Some(status);
                }
                if status.is_finished() {
                    publish(ctx.bus, stream.read().await?, summary);
                    return Ok(status);
                }
            },
            Err(error) if failures < MAX_FAILURES => {
                failures += 1;
                unreachable_target(&error, failures);
            },
            Err(error) => return Err(error.into()),
        }
        tokio::time::sleep(ctx.poll).await;
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
/// saved as `Cancelled`, without a message. A job that had already ended before
/// the cancel is left alone by the target, so its status stays
/// [`JobStatus::Exited`] or [`JobStatus::Lost`]; the record is then returned
/// unchanged (still `Running`), so a later watch or attach records the real
/// outcome and retrieves its artifacts. The status is returned with the record so
/// the caller can tell which case happened.
///
/// # Errors
///
/// Returns [`RunError::NotStarted`] for a run without a job, and an error when the
/// target cannot be reached or the record cannot be saved.
pub async fn cancel<E: Executor>(
    runs: &Runs,
    executor: &E,
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
    }
    Ok((record, status))
}

#[cfg(test)]
mod tests {
    use std::future::{Future, ready};

    use super::*;
    use crate::exec::{JobCommand, Pid};
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

    /// A target whose job has exited with code 0 after writing [`METRICS`], and
    /// from which nothing can be downloaded.
    struct BrokenDownload;

    impl Executor for BrokenDownload {
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
            ready(Err(ExecError::Protocol("unused".to_string())))
        }

        fn read_from(
            &self,
            _path: &str,
            offset: u64,
            _limit: u64,
        ) -> impl Future<Output = Result<Vec<u8>, ExecError>> + Send {
            let start = usize::try_from(offset).unwrap_or(METRICS.len());
            ready(Ok(METRICS
                .as_bytes()
                .get(start..)
                .unwrap_or_default()
                .to_vec()))
        }

        fn status(
            &self,
            _job: &JobId,
        ) -> impl Future<Output = Result<JobStatus, ExecError>> + Send {
            ready(Ok(JobStatus::Exited(0)))
        }

        fn cancel(&self, _job: &JobId) -> impl Future<Output = Result<(), ExecError>> + Send {
            ready(Ok(()))
        }

        fn download(
            &self,
            _remote: &str,
            _local: &Path,
            _entries: &[String],
            _exclude: &[String],
        ) -> impl Future<Output = Result<(), ExecError>> + Send {
            ready(Err(ExecError::Command {
                action: "download",
                message: "connection reset".to_string(),
            }))
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

    #[tokio::test]
    async fn a_successful_run_whose_artifacts_cannot_be_retrieved_stays_running()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let ctx = RunCtx {
            runs: &runs,
            executor: &BrokenDownload,
            bus: &bus,
            poll: Duration::from_millis(1),
        };
        let record = RunRecord {
            id: "20260922-143005-abcd".to_string(),
            target: "box".to_string(),
            created: "2026-09-22T14:30:05Z".to_string(),
            remote_dir: "/w/20260922-143005-abcd".to_string(),
            job: Some(JobId {
                dir: "/w/20260922-143005-abcd".to_string(),
                pid: Pid::new(42)?,
                container: None,
            }),
            state: RunState::Running,
            message: None,
        };
        runs.save(&record)?;
        let error = watch(&ctx, &NoFiles, record.clone())
            .await
            .err()
            .ok_or("the watch succeeded")?;
        assert!(error.to_string().contains("connection reset"), "{error}");
        assert_eq!(runs.load(&record.id)?, record);
        Ok(())
    }
}
