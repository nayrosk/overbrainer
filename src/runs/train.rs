use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::path::Path;
use std::time::{Duration, SystemTime};

use secrecy::SecretString;

use super::snapshot::{Proof, Snapshot, read_proof};
use super::{MetricsSummary, RunRecord, RunState, Runs, RunsError, new_run_id, rfc3339};
use crate::events::{Event, EventBus};
use crate::exec::{
    ExecError, Executor, FileDigest, JOB_LOG, JobId, JobRuntime, JobSpec, JobStatus, LineStream,
    local_manifest,
};
use crate::runpod::{POD_FILE, SSH_DIR, chain};
use crate::system::{self, SystemSample, probe_script};
use crate::train::{SNAPSHOT_FILE, TrainError, Trainer};

/// Hugging Face cache on the target, under the executor's work directory. Shared
/// by the runs of that target so a base model is downloaded once.
pub const HF_CACHE_DIR: &str = ".hf-cache";

/// Failures in a row to reach the target that a watch retries: the sixth one in a
/// row gives up (the job keeps running and can be attached again). A success in the
/// poll loop starts the count again; the drain that follows the job's end spends
/// what is left of it.
const MAX_FAILURES: u32 = 5;

/// Time between two samples of the target's machine while a job is followed.
pub const PROBE_EVERY: Duration = Duration::from_secs(10);

/// How long one sample may take before it is given up: longer than the probe
/// script's own bound (see [`probe_script`]), so a sample is only abandoned
/// when the connection itself hangs, and the next one waits for it: probes
/// never run side by side.
const PROBE_TIMEOUT: Duration = Duration::from_secs(20);

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
    /// The run directory on the target belongs to another run.
    #[error(
        "{0} on the target belongs to another run with the same ID, from another checkout of the project; start the run again"
    )]
    Taken(String),
    /// A snapshot was asked for a run whose job already ended.
    #[error("run {0} is not running ({1}): there is nothing to snapshot")]
    NotRunning(String, String),
    /// A snapshot was asked for a run whose job cannot save one: it was
    /// started by an overbrainer older than 0.5.0, whose metrics plugin does
    /// not read the request.
    #[error(
        "run {0} was started by an overbrainer older than 0.5.0: its job cannot save a snapshot; cancel it with `overbrainer train cancel {0}`, or let it finish"
    )]
    NoSnapshots(String),
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
    /// Whether every file the target has for this run (its artifacts and its job
    /// log, whatever that turns out to be) was downloaded and verified against the
    /// SHA-256 the target computed for it: what decides whether the pod that ran
    /// it may be deleted without losing anything.
    ///
    /// A succeeded job whose required entry holds no file is recorded `Failed`
    /// (see [`watch`]), but is still `retrieved` once what did exist was
    /// downloaded and verified: there was nothing more on the target to lose. Only
    /// a file that could not be retrieved or verified, or, for an already-ended
    /// run, a job log never downloaded locally, makes this false. `watch` and
    /// [`collect`] agree on this for the same run.
    pub retrieved: bool,
}

/// Creates a run of the project named `project` for `target`, whose run
/// directories live in `workdir` on the target: a new ID (see [`new_run_id`]
/// and [`Runs::claim`]), and its record saved as `Preparing`. For a target whose
/// executor only exists later (a Runpod pod): `workdir` is the directory it will
/// have, and [`reserve`] claims the run directory there once it exists. A target
/// with an executor uses [`create_on`].
///
/// # Errors
///
/// Returns [`RunsError`] when the run directory cannot be claimed or the record
/// cannot be saved.
pub fn create(
    runs: &Runs,
    project: &str,
    workdir: &str,
    target: &str,
) -> Result<RunRecord, RunsError> {
    let now = SystemTime::now();
    let (id, _) = runs.claim(&new_run_id(project, now), 0)?;
    let record = preparing(id, workdir, target, now);
    runs.save(&record)?;
    Ok(record)
}

/// [`create`] on the target of `executor`: the run directory is also claimed
/// there ([`Executor::claim`]) before the record is saved. When another run
/// already owns it on the target, such as a run started in the same second from
/// another checkout of the project against the same work directory, the next ID
/// is tried, `_2`, `_3` and so on, as [`Runs::claim`] does locally.
///
/// # Errors
///
/// Returns [`RunError::Runs`] when no local ID is left or the record cannot be
/// saved, and [`RunError::Exec`] when the target cannot claim the directory.
pub async fn create_on<E: Executor>(
    runs: &Runs,
    executor: &E,
    project: &str,
    target: &str,
) -> Result<RunRecord, RunError> {
    let now = SystemTime::now();
    create_named(runs, executor, &new_run_id(project, now), target, now).await
}

/// [`create_on`] with the base ID `base`, created `now`.
async fn create_named<E: Executor>(
    runs: &Runs,
    executor: &E,
    base: &str,
    target: &str,
    now: SystemTime,
) -> Result<RunRecord, RunError> {
    // Only this call ever claims with it: a directory that holds anything
    // else belongs to another run.
    let owner = format!("{:016x}", fastrand::u64(..));
    let mut after = 0;
    loop {
        let (id, n) = runs.claim(base, after)?;
        let remote_dir = format!("{}/{id}", executor.workdir());
        match executor.claim(&remote_dir, &owner).await {
            Ok(true) => {
                let record = preparing(id, executor.workdir(), target, now);
                runs.save(&record)?;
                return Ok(record);
            },
            Ok(false) => {
                tracing::debug!("{remote_dir} belongs to another run: trying the next ID");
                runs.release(&id);
                after = n;
            },
            Err(error) => {
                runs.release(&id);
                return Err(error.into());
            },
        }
    }
}

/// Claims the run directory of `record` on the target for `owner`
/// ([`Executor::claim`]), for a run made by [`create`] before its target
/// existed. `owner` is the run's own value, the same the target itself may
/// have claimed the directory with already (a Runpod pod's bootstrap does), so
/// that claim is accepted. The run's ID can no longer change: when another run
/// owns the directory, the run fails, saved `Failed` like a run [`start`]
/// cannot start, before anything is copied there.
///
/// # Errors
///
/// Returns [`RunError::Taken`] when another run owns the directory, and
/// [`RunError::Exec`] when the target cannot claim it.
pub async fn reserve<E: Executor>(
    ctx: &RunCtx<'_, E>,
    record: &mut RunRecord,
    owner: &str,
) -> Result<(), RunError> {
    let error = match ctx.executor.claim(&record.remote_dir, owner).await {
        Ok(true) => return Ok(()),
        Ok(false) => RunError::Taken(record.remote_dir.clone()),
        Err(error) => error.into(),
    };
    record.state = RunState::Failed;
    record.message = Some(error.to_string());
    if let Err(save_error) = ctx.runs.save(record) {
        tracing::warn!("cannot record run {} as failed: {save_error}", record.id);
    }
    Err(error)
}

/// The record of the new run `id`, `Preparing`, created `now`.
fn preparing(id: String, workdir: &str, target: &str, now: SystemTime) -> RunRecord {
    RunRecord {
        remote_dir: format!("{workdir}/{id}"),
        id,
        target: target.to_string(),
        created: rfc3339(now),
        job: None,
        state: RunState::Preparing,
        message: None,
        snapshot: None,
        resumed_from: None,
        snapshots: false,
    }
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
    // The job runs this version's metrics plugin, which reads snapshot requests.
    record.snapshots = true;
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

/// Prepares and uploads the run, excluding its local SSH keys and `pod.json`,
/// then starts the trainer through the selected runtime. Secrets are passed to
/// the job separately from the uploaded run directory.
async fn launch_job<E: Executor, T: Trainer>(
    ctx: &RunCtx<'_, E>,
    trainer: &T,
    launch: Launch<'_>,
    record: &RunRecord,
) -> Result<JobId, RunError> {
    let local = ctx.runs.run_dir(&record.id)?;
    let root = launch.runtime.root(&record.remote_dir);
    trainer.prepare(&local, &root)?;
    // A Runpod run's private SSH keys and pod record never leave this machine:
    // on a network volume, what the pod receives outlives the pod.
    let local_only = [SSH_DIR.to_string(), POD_FILE.to_string()];
    ctx.executor
        .upload(&local, &record.remote_dir, &local_only)
        .await?;
    let cache_dir = format!("{}/{HF_CACHE_DIR}", ctx.executor.workdir());
    let job = launch.runtime.job(JobSpec {
        run_id: &record.id,
        run_dir: &record.remote_dir,
        cache_dir: &cache_dir,
        commands: &trainer.commands(),
        env: &trainer.env(&root),
        stop_marker: trainer.stop_marker(),
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
/// `Failed` instead, with a message saying so, and [`Outcome::retrieved`] true:
/// nothing the target had was left behind.
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
        let retrieved = !artifacts_missing(&record)
            && local.join(JOB_LOG).is_file()
            && record
                .snapshot
                .as_ref()
                .is_none_or(|snapshot| local.join(&snapshot.checkpoint).is_dir());
        return Ok(Outcome {
            record,
            summary,
            retrieved,
        });
    }
    ctx.bus.publish(Event::RunWatched {
        run_id: record.id.clone(),
    });
    let metrics = format!("{}/{}", record.remote_dir, trainer.metrics_file());
    let mut stream = ctx.executor.tail(&metrics, 0);
    let mut summary = MetricsSummary::default();
    // The samples stop with the follow: they never end it, and their failures
    // never count against its retries. Boxed, so the watch's own future stays
    // small.
    let sampler = Box::pin(sample(ctx, &record.remote_dir));
    let status = tokio::select! {
        status = follow(ctx, &job, &mut stream, &mut summary) => status?,
        never = sampler => match never {},
    };
    let (state, message) = outcome(status, &summary, &record.id);
    // A job cancelled once its proof was written (a stop whose job did not
    // end in time) saved its checkpoint all the same: it is stopped too.
    let (state, message, snapshot) = if matches!(state, RunState::Succeeded | RunState::Cancelled) {
        match read_proof(ctx.executor, &record.remote_dir).await? {
            Proof::None => (state, message, None),
            Proof::Saved(snapshot) => (RunState::Stopped, None, Some(snapshot)),
            Proof::Invalid(_) if state == RunState::Cancelled => (state, message, None),
            Proof::Invalid(why) => (
                RunState::Failed,
                Some(format!("the job stopped with a snapshot, but {why}")),
                None,
            ),
        }
    } else {
        (state, message, None)
    };
    let expect = Expect::of(state, snapshot.as_ref());
    let kept = matches!(state, RunState::Succeeded | RunState::Stopped);
    let (state, message, retrieved) =
        match retrieve(ctx.executor, trainer, &record.remote_dir, &local, expect).await {
            Ok(Retrieved::Ok) => (state, message, true),
            // Recorded Failed, not Succeeded, but everything the target had was
            // still downloaded and verified: retrieved is still true.
            Ok(Retrieved::NoOutput) => (
                RunState::Failed,
                Some(no_output_message(&record.id, snapshot.as_ref())),
                true,
            ),
            Err(error) if kept => return Err(error.into()),
            Err(error) => (state, Some(not_retrieved(message, &error)), false),
        };
    record.message = message;
    record.snapshot = snapshot.filter(|_| state == RunState::Stopped);
    record.state = state;
    ctx.runs.save(&record)?;
    Ok(Outcome {
        record,
        summary,
        retrieved,
    })
}

/// What [`retrieve`] checks beyond the files themselves.
#[derive(Debug, Clone, Copy)]
enum Expect<'a> {
    /// Nothing: the run failed or was cancelled.
    Nothing,
    /// The trainer's required entry holds a file: the run succeeded.
    Output,
    /// The snapshot's checkpoint, retrieved in a second pass: the run stopped.
    Snapshot(&'a Snapshot),
}

impl<'a> Expect<'a> {
    /// What a run in `state`, with `snapshot` when stopped, must have left.
    fn of(state: RunState, snapshot: Option<&'a Snapshot>) -> Self {
        match (state, snapshot) {
            (RunState::Succeeded, _) => Self::Output,
            (RunState::Stopped, Some(snapshot)) => Self::Snapshot(snapshot),
            _ => Self::Nothing,
        }
    }
}

/// What [`retrieve`] found while checking what the run must have left.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Retrieved {
    /// Every local file matched the target's manifest exactly, and what the run
    /// must have left is there (see [`Expect`]).
    Ok,
    /// The job succeeded or stopped, but the target's required entry, or the
    /// snapshot's checkpoint, held no file: a permanent condition, since nothing
    /// more will appear there for a retry to find.
    NoOutput,
}

/// Copies the trainer's artifacts and the job log from `remote` into `local`, then
/// checks that the local files under the trainer's entries are exactly the ones
/// the target's SHA-256 manifest lists, with matching hashes: no file missing, none
/// with a different hash, and none extra that the target did not have. A stopped
/// run then gets its checkpoint (which the artifacts leave out) and its proof the
/// same way, in a second pass. A succeeded run whose required entry holds no file,
/// or a stopped one whose checkpoint holds none, is reported as
/// [`Retrieved::NoOutput`] rather than as an error, once what does exist has been
/// downloaded and verified.
async fn retrieve<E: Executor, T: Trainer>(
    executor: &E,
    trainer: &T,
    remote: &str,
    local: &Path,
    expect: Expect<'_>,
) -> Result<Retrieved, ExecError> {
    let artifacts = trainer.artifacts();
    let mut entries = artifacts.entries;
    entries.push(JOB_LOG.to_string());
    // Boxed, as below: their state would otherwise weigh on every caller's future.
    let manifest = Box::pin(retrieve_pass(
        executor,
        remote,
        local,
        entries,
        artifacts.exclude,
    ))
    .await?;
    let found = match expect {
        Expect::Nothing => true,
        Expect::Output => artifacts
            .required
            .as_deref()
            .is_none_or(|required| has_required(&manifest, required)),
        Expect::Snapshot(snapshot) => {
            let entries = vec![snapshot.checkpoint.clone(), SNAPSHOT_FILE.to_string()];
            let manifest =
                Box::pin(retrieve_pass(executor, remote, local, entries, Vec::new())).await?;
            has_required(&manifest, &snapshot.checkpoint)
        },
    };
    Ok(if found {
        Retrieved::Ok
    } else {
        Retrieved::NoOutput
    })
}

/// One retrieval: the target's manifest of `entries` (less `exclude`), the
/// download, then the check that the local files are exactly the manifest's.
/// Returns the manifest.
async fn retrieve_pass<E: Executor>(
    executor: &E,
    remote: &str,
    local: &Path,
    entries: Vec<String>,
    exclude: Vec<String>,
) -> Result<Vec<FileDigest>, ExecError> {
    let manifest = executor.manifest(remote, &entries, &exclude).await?;
    executor.download(remote, local, &entries, &exclude).await?;
    let local = local.to_path_buf();
    let listed = manifest.clone();
    tokio::task::spawn_blocking(move || verify(&local, &entries, &exclude, &listed))
        .await
        .map_err(|error| ExecError::Protocol(format!("the verification task failed: {error}")))??;
    Ok(manifest)
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
/// missing, none with a different hash, and none the target did not have. Every
/// local file is hashed exactly once, by [`local_manifest`], and compared as a
/// map: a multi-gigabyte file is never hashed twice to answer this.
fn verify(
    local: &Path,
    entries: &[String],
    exclude: &[String],
    manifest: &[FileDigest],
) -> Result<(), ExecError> {
    let local_files = local_manifest(local, entries, exclude)?;
    let local_by_path: HashMap<&str, &str> = local_files
        .iter()
        .map(|file| (file.path.as_str(), file.sha256.as_str()))
        .collect();
    for file in manifest {
        let Some(sha256) = local_by_path.get(file.path.as_str()).copied() else {
            return Err(verify_error(format!("{:?} is missing locally", file.path)));
        };
        if sha256 != file.sha256.as_str() {
            return Err(verify_error(format!(
                "{:?} differs from the target (SHA-256 mismatch)",
                file.path
            )));
        }
    }
    let manifest_paths: HashSet<&str> = manifest.iter().map(|file| file.path.as_str()).collect();
    if let Some(extra) = local_files
        .iter()
        .find(|file| !manifest_paths.contains(file.path.as_str()))
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
/// entry, or a stopped one no file in its checkpoint: a permanent condition,
/// since a retry cannot make the target produce what was never written.
fn no_output_message(id: &str, snapshot: Option<&Snapshot>) -> String {
    match snapshot {
        Some(snapshot) => format!(
            "the job stopped with a snapshot but left no file in {} (see runs/{id}/{JOB_LOG})",
            snapshot.checkpoint
        ),
        None => format!("the job succeeded but left no output (see runs/{id}/{JOB_LOG})"),
    }
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
    let expect = Expect::of(record.state, record.snapshot.as_ref());
    if let Err(error) = retrieve(ctx.executor, trainer, &record.remote_dir, &local, expect).await {
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

/// Samples the target's machine at once, then every [`PROBE_EVERY`], and
/// publishes each sample as an [`Event::System`]; `run_dir` is the run
/// directory on the target, whose file system is sampled first. A sample that
/// fails or takes longer than [`PROBE_TIMEOUT`] is logged at debug level and
/// skipped. Never ends: the caller drops it.
async fn sample<E: Executor>(ctx: &RunCtx<'_, E>, run_dir: &str) -> Infallible {
    let script = probe_script(run_dir);
    let mut previous: Option<SystemSample> = None;
    let mut every = tokio::time::interval(PROBE_EVERY);
    every.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        every.tick().await;
        match sample_once(ctx.executor, &script, previous.as_ref()).await {
            Ok(sample) => {
                ctx.bus.publish(Event::System(sample.clone()));
                previous = Some(sample);
            },
            Err(error) => tracing::debug!("cannot sample the target: {error}"),
        }
    }
}

/// One sample of the target: `script` run there, its answer parsed with the
/// counters of `previous`. Fails, saying why, when the target does not answer
/// within [`PROBE_TIMEOUT`].
async fn sample_once<E: Executor>(
    executor: &E,
    script: &str,
    previous: Option<&SystemSample>,
) -> Result<SystemSample, String> {
    let output = tokio::time::timeout(PROBE_TIMEOUT, executor.probe(script))
        .await
        .map_err(|_| format!("no answer within {}s", PROBE_TIMEOUT.as_secs()))?
        .map_err(|error| chain(&error))?;
    let text = String::from_utf8_lossy(&output);
    Ok(system::parse(&text, SystemTime::now(), previous))
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
    tracing::warn!("{}", unreachable_line(error, failures));
}

/// The warning for the `failures`th failed poll: `error` with its sources, so an
/// SSH failure says why (`ssh failed: the connection was terminated`).
fn unreachable_line(error: &ExecError, failures: u32) -> String {
    format!(
        "cannot reach the job ({failures}/{MAX_FAILURES}): {}",
        chain(error)
    )
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
    let Err(error) = retrieve(
        executor,
        trainer,
        &record.remote_dir,
        &local,
        Expect::Nothing,
    )
    .await
    else {
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

    #[test]
    fn an_unreachable_job_says_why_ssh_failed() {
        let error = ExecError::Ssh(openssh::Error::Disconnected);
        assert_eq!(
            unreachable_line(&error, 2),
            format!(
                "cannot reach the job (2/{MAX_FAILURES}): ssh failed: the connection was terminated"
            )
        );
    }
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

    /// How [`Fake`]'s probe answers.
    #[derive(Debug, Clone)]
    enum Probe {
        /// It prints this.
        Answers(Vec<u8>),
        /// It fails at once.
        Fails,
        /// It never answers.
        Hangs,
    }

    /// A scripted target. Its job reads `status`, its metrics file holds
    /// `metrics`, and the read numbered `failing_read` (from 0) fails.
    struct Fake {
        status: JobStatus,
        metrics: String,
        spawn_fails: bool,
        /// Whether another run owns every run directory on the target.
        claim_taken: bool,
        download_fails: bool,
        failing_read: Option<u32>,
        /// A run directory whose later saves [`break_saves_in`] breaks when a
        /// download is asked for, before it fails.
        break_on_download: Option<PathBuf>,
        /// What the target's manifest lists, under the entries asked for.
        manifest: Vec<FileDigest>,
        /// The content of the target's `snapshot.json`.
        proof: String,
        reads: AtomicU32,
        cancels: AtomicU32,
        /// How the probe answers.
        probe: Probe,
        probes: AtomicU32,
        /// Status polls answered `Running` before [`Fake::status`].
        running_polls: u32,
        polls: AtomicU32,
        /// Every file written with `put_file`, with its content.
        puts: std::sync::Mutex<Vec<(String, String)>>,
        /// How reads of the snapshot request answer.
        request_reads: RequestReads,
    }

    impl Fake {
        fn new(status: JobStatus) -> Self {
            Self {
                status,
                metrics: METRICS.to_string(),
                spawn_fails: false,
                claim_taken: false,
                download_fails: false,
                failing_read: None,
                break_on_download: None,
                manifest: Vec::new(),
                proof: String::new(),
                reads: AtomicU32::new(0),
                cancels: AtomicU32::new(0),
                probe: Probe::Fails,
                probes: AtomicU32::new(0),
                running_polls: 0,
                polls: AtomicU32::new(0),
                puts: std::sync::Mutex::new(Vec::new()),
                request_reads: RequestReads::Answered,
            }
        }
    }

    /// How the fake answers reads of the snapshot request.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum RequestReads {
        /// With what `put_file` wrote there, or nothing.
        Answered,
        /// Never: the read hangs.
        Hung,
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

        fn claim(
            &self,
            _dir: &str,
            _owner: &str,
        ) -> impl Future<Output = Result<bool, ExecError>> + Send {
            ready(Ok(!self.claim_taken))
        }

        fn upload(
            &self,
            _local: &Path,
            _remote: &str,
            _skip: &[String],
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
            path: &str,
            offset: u64,
            limit: u64,
        ) -> impl Future<Output = Result<Vec<u8>, ExecError>> + Send {
            use futures::future::{Either, pending};
            if path.ends_with(SNAPSHOT_FILE) {
                return Either::Right(ready(Ok(self.proof.clone().into_bytes())));
            }
            if path.ends_with(crate::train::SNAPSHOT_REQUEST) {
                if self.request_reads == RequestReads::Hung {
                    return Either::Left(pending());
                }
                let request = self.puts.lock().ok().and_then(|puts| {
                    puts.iter()
                        .rfind(|(put, _)| put == path)
                        .map(|(_, content)| content.clone().into_bytes())
                });
                return Either::Right(ready(Ok(request.unwrap_or_default())));
            }
            let read = self.reads.fetch_add(1, Ordering::SeqCst);
            let start = usize::try_from(offset).unwrap_or(self.metrics.len());
            let mut bytes = self
                .metrics
                .as_bytes()
                .get(start..)
                .unwrap_or_default()
                .to_vec();
            bytes.truncate(usize::try_from(limit).unwrap_or(usize::MAX));
            Either::Right(ready(if self.failing_read == Some(read) {
                Err(broken("read"))
            } else {
                Ok(bytes)
            }))
        }

        fn status(
            &self,
            _job: &JobId,
        ) -> impl Future<Output = Result<JobStatus, ExecError>> + Send {
            let poll = self.polls.fetch_add(1, Ordering::SeqCst);
            ready(Ok(if poll < self.running_polls {
                JobStatus::Running
            } else {
                self.status
            }))
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
            if let Some(dir) = &self.break_on_download {
                break_saves_in(dir).ok();
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
            entries: &[String],
            exclude: &[String],
        ) -> impl Future<Output = Result<Vec<FileDigest>, ExecError>> + Send {
            let listed = self
                .manifest
                .iter()
                .filter(|file| {
                    entries.iter().any(|entry| {
                        file.path == *entry || file.path.starts_with(&format!("{entry}/"))
                    }) && !file.path.split('/').any(|part| {
                        exclude
                            .iter()
                            .any(|pattern| crate::exec::glob_match(pattern, part))
                    })
                })
                .cloned()
                .collect();
            ready(Ok(listed))
        }

        fn probe(&self, _script: &str) -> impl Future<Output = Result<Vec<u8>, ExecError>> + Send {
            self.probes.fetch_add(1, Ordering::SeqCst);
            let probe = self.probe.clone();
            async move {
                match probe {
                    Probe::Answers(output) => Ok(output),
                    Probe::Fails => Err(broken("probe")),
                    Probe::Hangs => std::future::pending().await,
                }
            }
        }

        fn put_file(
            &self,
            path: &str,
            content: &str,
        ) -> impl Future<Output = Result<(), ExecError>> + Send {
            if let Ok(mut puts) = self.puts.lock() {
                puts.push((path.to_string(), content.to_string()));
            }
            ready(Ok(()))
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

    /// A trainer whose successful runs must leave a file in `output/`, which
    /// leaves the checkpoints out, as Axolotl's does.
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
                exclude: vec!["checkpoint-*".to_string()],
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
            snapshot: None,
            resumed_from: None,
            snapshots: true,
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

    /// Makes every later save of the run `id` fail: see [`break_saves_in`].
    fn break_saves(runs: &Runs, id: &str) -> Result<(), Box<dyn std::error::Error>> {
        break_saves_in(&runs.run_dir(id)?)?;
        Ok(())
    }

    /// Makes every later save in the run directory `dir` fail, root or not: its
    /// `run.json` moves to [`SAVED_BEFORE`] and a directory takes its place, which
    /// no save can be renamed over.
    fn break_saves_in(dir: &Path) -> std::io::Result<()> {
        let record = dir.join(RECORD_FILE);
        if record.is_file() {
            std::fs::rename(&record, dir.join(SAVED_BEFORE))?;
        }
        std::fs::create_dir(record)
    }

    /// Where [`break_saves_in`] moves the last saved `run.json`.
    const SAVED_BEFORE: &str = "run.json.before";

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
    async fn the_target_is_sampled_while_the_job_is_followed()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let mut events = bus.subscribe();
        let df = "@df.run\nFilesystem 1024-blocks Used Available Capacity Mounted on\n\
                  /dev/sda1 100 60 40 60% /workspace\n";
        let fake = Fake {
            probe: Probe::Answers(df.as_bytes().to_vec()),
            running_polls: 3,
            ..Fake::new(JobStatus::Exited(0))
        };
        let record = running()?;
        runs.save(&record)?;
        let outcome = watch(&ctx(&runs, &fake, &bus), &NoFiles, record).await?;
        assert_eq!(outcome.record.state, RunState::Succeeded);
        // At once, then every ten seconds: once in a watch this short.
        assert_eq!(fake.probes.load(Ordering::SeqCst), 1);
        let mut samples = Vec::new();
        while let Ok(event) = events.try_recv() {
            if let Event::System(sample) = event {
                samples.push(sample);
            }
        }
        let [sample] = samples.as_slice() else {
            return Err(format!("{samples:?}").into());
        };
        let disk = sample.run_disk().ok_or("no disk")?;
        assert_eq!(
            (disk.mount.as_str(), disk.used_bytes),
            ("/workspace", 60 * 1024)
        );
        Ok(())
    }

    /// Watches a run whose job runs for 80 polls of a second each on `fake`,
    /// on the paused clock: long enough for several samples.
    async fn watch_long(fake: Fake) -> Result<(Outcome, Fake), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let fake = Fake {
            running_polls: 80,
            ..fake
        };
        let record = running()?;
        runs.save(&record)?;
        let ctx = RunCtx {
            poll: Duration::from_secs(1),
            ..ctx(&runs, &fake, &bus)
        };
        let outcome = watch(&ctx, &NoFiles, record).await?;
        Ok((outcome, fake))
    }

    #[tokio::test(start_paused = true)]
    async fn failing_samples_never_count_against_the_follow()
    -> Result<(), Box<dyn std::error::Error>> {
        let (outcome, fake) = watch_long(Fake::new(JobStatus::Exited(0))).await?;
        assert_eq!(outcome.record.state, RunState::Succeeded);
        // Every sample failed, far more than the follow's retries allow.
        let probes = fake.probes.load(Ordering::SeqCst);
        assert!(probes > MAX_FAILURES + 2, "{probes}");
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn a_sample_that_never_answers_never_holds_the_follow()
    -> Result<(), Box<dyn std::error::Error>> {
        let fake = Fake {
            probe: Probe::Hangs,
            ..Fake::new(JobStatus::Exited(0))
        };
        let (outcome, fake) = watch_long(fake).await?;
        assert_eq!(outcome.record.state, RunState::Succeeded);
        // Each one given up after its timeout, the next started only then: at
        // most one every 20 seconds over the job's 80, never side by side.
        let probes = fake.probes.load(Ordering::SeqCst);
        assert!((2..=5).contains(&probes), "{probes}");
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
    async fn a_run_directory_another_checkout_owns_on_the_target_is_skipped()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let target = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let executor = crate::exec::LocalExecutor::new(target.path())?;
        let base = "demo_20260930-120000";
        let now = SystemTime::UNIX_EPOCH;
        // Another checkout of the project, against the same work directory, got
        // there first with the same ID; its own `runs/` is elsewhere.
        assert!(
            executor
                .claim(&format!("{}/{base}", executor.workdir()), "elsewhere")
                .await?
        );
        let record = create_named(&runs, &executor, base, "box", now).await?;
        assert_eq!(record.id, format!("{base}_2"));
        assert_eq!(
            record.remote_dir,
            format!("{}/{base}_2", executor.workdir())
        );
        assert!(
            Path::new(&record.remote_dir)
                .join(crate::exec::CLAIM_FILE)
                .is_file()
        );
        // The local directory of the ID the target refused is not left behind.
        assert!(!runs.dir().join(base).exists());
        assert_eq!(runs.list()?, vec![record]);
        Ok(())
    }

    #[tokio::test]
    async fn a_local_target_claims_the_run_directory_it_shares_with_runs()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let executor = crate::exec::LocalExecutor::new(runs.dir())?;
        let first = create_on(&runs, &executor, "demo", "here").await?;
        let second = create_on(&runs, &executor, "demo", "here").await?;
        assert_ne!(first.id, second.id);
        for record in [&first, &second] {
            assert_eq!(runs.load(&record.id)?, *record);
            assert!(
                runs.run_dir(&record.id)?
                    .join(crate::exec::CLAIM_FILE)
                    .is_file()
            );
        }
        Ok(())
    }

    #[tokio::test]
    async fn a_run_whose_directory_is_taken_on_its_pod_fails_before_any_copy()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let fake = Fake {
            claim_taken: true,
            ..Fake::new(JobStatus::Running)
        };
        let mut record = create(&runs, "demo", fake.workdir(), "cloud")?;
        let result = reserve(&ctx(&runs, &fake, &bus), &mut record, "mine").await;
        assert!(matches!(&result, Err(RunError::Taken(dir)) if *dir == record.remote_dir));
        let saved = runs.load(&record.id)?;
        assert_eq!(saved.state, RunState::Failed);
        assert!(
            saved
                .message
                .as_deref()
                .is_some_and(|message| message.contains("belongs to another run")),
            "{saved:?}"
        );
        let mut free = create(&runs, "demo", "/w", "cloud")?;
        reserve(
            &ctx(&runs, &Fake::new(JobStatus::Running), &bus),
            &mut free,
            "mine",
        )
        .await?;
        assert_eq!(runs.load(&free.id)?.state, RunState::Preparing);
        Ok(())
    }

    #[tokio::test]
    async fn reserve_accepts_the_runs_own_claim_and_refuses_another()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let target = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let executor = crate::exec::LocalExecutor::new(target.path())?;
        let ctx = RunCtx {
            runs: &runs,
            executor: &executor,
            bus: &bus,
            poll: Duration::from_millis(1),
        };
        // As a pod's bootstrap claims it, with the run's own value.
        let mut ours = create(&runs, "demo", executor.workdir(), "cloud")?;
        assert!(
            executor
                .claim(&ours.remote_dir, "ssh-ed25519 AAAAours")
                .await?
        );
        reserve(&ctx, &mut ours, "ssh-ed25519 AAAAours").await?;
        assert_eq!(runs.load(&ours.id)?.state, RunState::Preparing);
        // Claimed by a run of another checkout.
        let mut theirs = create(&runs, "demo", executor.workdir(), "cloud")?;
        assert!(
            executor
                .claim(&theirs.remote_dir, "ssh-ed25519 AAAAtheirs")
                .await?
        );
        let refused = reserve(&ctx, &mut theirs, "ssh-ed25519 AAAAours").await;
        assert!(matches!(refused, Err(RunError::Taken(_))), "{refused:?}");
        assert_eq!(runs.load(&theirs.id)?.state, RunState::Failed);
        assert_eq!(
            std::fs::read_to_string(Path::new(&theirs.remote_dir).join(crate::exec::CLAIM_FILE))?,
            "ssh-ed25519 AAAAtheirs\n"
        );
        Ok(())
    }

    #[test]
    fn runs_created_in_the_same_second_keep_their_own_records()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let first = create(&runs, "Demo Project", "/w", "box")?;
        let second = create(&runs, "Demo Project", "/w", "other")?;
        assert!(first.id.starts_with("demo_project_"), "{}", first.id);
        assert_ne!(first.id, second.id);
        assert_eq!(second.remote_dir, format!("/w/{}", second.id));
        assert_eq!(runs.load(&first.id)?.target, "box");
        assert_eq!(runs.load(&second.id)?.target, "other");
        Ok(())
    }

    #[tokio::test]
    async fn a_job_whose_run_cannot_be_recorded_is_cancelled()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let fake = Fake::new(JobStatus::Running);
        let created = create(&runs, "demo", fake.workdir(), "box")?;
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
    async fn a_started_run_is_recorded_as_able_to_snapshot()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let fake = Fake::new(JobStatus::Running);
        let created = create(&runs, "demo", fake.workdir(), "box")?;
        assert!(!created.snapshots, "no job yet");
        let launch = Launch {
            runtime: &JobRuntime::Native {
                venv: None,
                env_file: None,
            },
            secrets: Vec::new(),
        };
        let started = start(&ctx(&runs, &fake, &bus), &NoFiles, launch, created).await?;
        assert!(started.snapshots);
        assert!(runs.load(&started.id)?.snapshots);
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
        let created = create(&runs, "demo", fake.workdir(), "box")?;
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
            break_on_download: Some(runs.run_dir(RUN_ID)?),
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
        let saved: RunRecord =
            serde_json::from_slice(&std::fs::read(runs.run_dir(RUN_ID)?.join(SAVED_BEFORE))?)?;
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
        // Everything the target had (just the job log here) was downloaded and
        // verified, so this still counts as retrieved: the pod may be deleted.
        assert!(outcome.retrieved);
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
    async fn a_permanent_failure_without_output_is_retrieved_on_every_path()
    -> Result<(), Box<dyn std::error::Error>> {
        let (runs, _project, result) =
            watch_with(vec![listed("job.log")], &[("job.log", "w")]).await?;
        let outcome = result?;
        assert_eq!(outcome.record.state, RunState::Failed);
        assert!(outcome.retrieved);

        let bus = EventBus::new();
        let fake = Fake {
            manifest: vec![listed("job.log")],
            ..Fake::new(JobStatus::Exited(0))
        };
        // A later watch, on the now-ended record, agrees.
        let again = watch(
            &ctx(&runs, &fake, &bus),
            &WithOutput,
            outcome.record.clone(),
        )
        .await?;
        assert!(again.retrieved);
        // So does collect.
        let (_, collected) = collect(&ctx(&runs, &fake, &bus), &WithOutput, outcome.record).await?;
        assert!(collected);
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

    const PROOF: &str =
        r#"{"checkpoint": "output/checkpoint-3", "step": 3, "time": 1.0, "reason": "deadline"}"#;

    /// What a stopped job leaves on the target: its output, its log, the
    /// checkpoint and the proof.
    fn stopped_manifest() -> Vec<FileDigest> {
        vec![
            listed("job.log"),
            listed("output/adapter.bin"),
            listed("output/checkpoint-3/optimizer.pt"),
            listed("snapshot.json"),
        ]
    }

    /// Watches a run whose job exits 0 with `proof` on a target listing
    /// `manifest`, after writing `files` locally as the downloads would.
    async fn watch_stopped(
        proof: &str,
        manifest: Vec<FileDigest>,
        files: &[&str],
    ) -> Result<(Runs, tempfile::TempDir, Result<Outcome, RunError>), Box<dyn std::error::Error>>
    {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let fake = Fake {
            manifest,
            proof: proof.to_string(),
            ..Fake::new(JobStatus::Exited(0))
        };
        runs.save(&running()?)?;
        for path in files {
            downloaded(&runs, path, "w")?;
        }
        let result = watch(&ctx(&runs, &fake, &bus), &WithOutput, running()?).await;
        Ok((runs, project, result))
    }

    #[tokio::test]
    async fn a_job_that_left_a_proof_is_stopped_with_its_checkpoint()
    -> Result<(), Box<dyn std::error::Error>> {
        let files = [
            "job.log",
            "output/adapter.bin",
            "output/checkpoint-3/optimizer.pt",
            "snapshot.json",
        ];
        let (runs, _project, result) = watch_stopped(PROOF, stopped_manifest(), &files).await?;
        let outcome = result?;
        assert_eq!(outcome.record.state, RunState::Stopped);
        assert!(outcome.retrieved);
        assert_eq!(outcome.record.message, None);
        assert_eq!(
            outcome.record.snapshot,
            Some(Snapshot {
                checkpoint: "output/checkpoint-3".into(),
                step: 3,
                reason: crate::runs::SnapshotReason::Deadline,
            })
        );
        assert_eq!(runs.load(RUN_ID)?, outcome.record);
        // Ended, it reads as retrieved while its checkpoint is there.
        let bus = EventBus::new();
        let fake = Fake::new(JobStatus::Exited(0));
        let again = watch(
            &ctx(&runs, &fake, &bus),
            &WithOutput,
            outcome.record.clone(),
        )
        .await?;
        assert!(again.retrieved);
        std::fs::remove_dir_all(runs.run_dir(RUN_ID)?.join("output/checkpoint-3"))?;
        let again = watch(&ctx(&runs, &fake, &bus), &WithOutput, outcome.record).await?;
        assert!(!again.retrieved);
        Ok(())
    }

    /// A proof at the final step (the plugin ignores a request there, but a job
    /// that wrote one anyway) still records the run stopped: the proof, not
    /// the step, decides, and the checkpoint is resumable.
    #[tokio::test]
    async fn a_proof_at_the_final_step_still_records_the_run_stopped()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let fake = Fake {
            manifest: stopped_manifest(),
            proof: PROOF.to_string(),
            metrics: "{\"event\": \"begin\", \"time\": 1, \"max_steps\": 3}\n\
                      {\"event\": \"log\", \"time\": 2, \"step\": 3, \"max_steps\": 3, \"loss\": 1.0}\n"
                .to_string(),
            ..Fake::new(JobStatus::Exited(0))
        };
        runs.save(&running()?)?;
        for path in [
            "job.log",
            "output/adapter.bin",
            "output/checkpoint-3/optimizer.pt",
            "snapshot.json",
        ] {
            downloaded(&runs, path, "w")?;
        }
        let outcome = watch(&ctx(&runs, &fake, &bus), &WithOutput, running()?).await?;
        assert_eq!(outcome.summary.lines, 2);
        assert_eq!(outcome.record.state, RunState::Stopped);
        assert_eq!(
            outcome.record.snapshot.map(|snapshot| snapshot.step),
            Some(3)
        );
        assert!(outcome.retrieved);
        Ok(())
    }

    #[tokio::test]
    async fn a_job_cancelled_after_its_proof_is_stopped() -> Result<(), Box<dyn std::error::Error>>
    {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let fake = Fake {
            manifest: stopped_manifest(),
            proof: PROOF.to_string(),
            ..Fake::new(JobStatus::Cancelled)
        };
        runs.save(&running()?)?;
        for path in [
            "job.log",
            "output/adapter.bin",
            "output/checkpoint-3/optimizer.pt",
            "snapshot.json",
        ] {
            downloaded(&runs, path, "w")?;
        }
        let outcome = watch(&ctx(&runs, &fake, &bus), &WithOutput, running()?).await?;
        assert_eq!(outcome.record.state, RunState::Stopped);
        assert!(outcome.retrieved);
        Ok(())
    }

    #[tokio::test]
    async fn a_checkpoint_not_retrieved_leaves_a_stopped_run_running()
    -> Result<(), Box<dyn std::error::Error>> {
        let files = ["job.log", "output/adapter.bin", "snapshot.json"];
        let (runs, _project, result) = watch_stopped(PROOF, stopped_manifest(), &files).await?;
        let error = result.err().ok_or("the watch succeeded")?;
        assert_eq!(
            error.to_string(),
            "verify failed: \"output/checkpoint-3/optimizer.pt\" is missing locally"
        );
        assert_eq!(runs.load(RUN_ID)?.state, RunState::Running);
        Ok(())
    }

    #[tokio::test]
    async fn a_snapshot_without_a_checkpoint_or_a_valid_proof_is_a_failure()
    -> Result<(), Box<dyn std::error::Error>> {
        let manifest = vec![
            listed("job.log"),
            listed("output/adapter.bin"),
            listed("snapshot.json"),
        ];
        let files = ["job.log", "output/adapter.bin", "snapshot.json"];
        let (_, _project, result) = watch_stopped(PROOF, manifest.clone(), &files).await?;
        let outcome = result?;
        assert_eq!(outcome.record.state, RunState::Failed);
        assert_eq!(outcome.record.snapshot, None);
        assert_eq!(
            outcome.record.message.as_deref(),
            Some(
                format!(
                    "the job stopped with a snapshot but left no file in output/checkpoint-3 \
                     (see runs/{RUN_ID}/job.log)"
                )
                .as_str()
            )
        );
        let (_, _project, result) =
            watch_stopped(r#"{"checkpoint": "../x", "step": 3}"#, manifest, &files).await?;
        let outcome = result?;
        assert_eq!(outcome.record.state, RunState::Failed);
        assert_eq!(
            outcome.record.message.as_deref(),
            Some(
                "the job stopped with a snapshot, but snapshot.json names an invalid checkpoint path"
            )
        );
        Ok(())
    }

    #[tokio::test]
    async fn collect_retrieves_the_checkpoint_of_a_stopped_run_again()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let bus = EventBus::new();
        let mut record = running()?;
        record.state = RunState::Stopped;
        record.snapshot = Some(Snapshot {
            checkpoint: "output/checkpoint-3".into(),
            step: 3,
            reason: crate::runs::SnapshotReason::Requested,
        });
        runs.save(&record)?;
        let fake = Fake {
            manifest: stopped_manifest(),
            ..Fake::new(JobStatus::Exited(0))
        };
        for path in ["job.log", "output/adapter.bin", "snapshot.json"] {
            downloaded(&runs, path, "w")?;
        }
        let (_, retrieved) = collect(&ctx(&runs, &fake, &bus), &WithOutput, record.clone()).await?;
        assert!(!retrieved, "the checkpoint is missing locally");
        downloaded(&runs, "output/checkpoint-3/optimizer.pt", "w")?;
        let (collected, retrieved) = collect(&ctx(&runs, &fake, &bus), &WithOutput, record).await?;
        assert!(retrieved);
        assert_eq!(collected.state, RunState::Stopped);
        Ok(())
    }

    #[tokio::test]
    async fn a_snapshot_request_carries_its_reason_to_a_running_run_only()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::runs::{SnapshotReason, request_snapshot};
        let fake = Fake::new(JobStatus::Running);
        let record = running()?;
        request_snapshot(&fake, &record, SnapshotReason::Cost).await?;
        let puts = fake.puts.lock().map_err(|_| "poisoned")?.clone();
        assert_eq!(
            puts,
            vec![(format!("/w/{RUN_ID}/snapshot.request"), "cost".to_string())]
        );
        let mut ended = record.clone();
        ended.state = RunState::Succeeded;
        let refused = request_snapshot(&fake, &ended, SnapshotReason::Requested).await;
        assert!(
            matches!(refused, Err(RunError::NotRunning(..))),
            "{refused:?}"
        );
        let mut preparing = record;
        preparing.job = None;
        let refused = request_snapshot(&fake, &preparing, SnapshotReason::Requested).await;
        assert!(
            matches!(refused, Err(RunError::NotStarted(_))),
            "{refused:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_run_started_before_snapshots_is_never_asked_for_one()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::runs::{SnapshotReason, request_snapshot};
        let fake = Fake::new(JobStatus::Running);
        let mut old = running()?;
        old.snapshots = false;
        let refused = request_snapshot(&fake, &old, SnapshotReason::Disk).await;
        let Err(error @ RunError::NoSnapshots(_)) = refused else {
            return Err(format!("{refused:?}").into());
        };
        assert_eq!(
            error.to_string(),
            format!(
                "run {RUN_ID} was started by an overbrainer older than 0.5.0: its job cannot \
                 save a snapshot; cancel it with `overbrainer train cancel {RUN_ID}`, or let it \
                 finish"
            )
        );
        assert!(fake.puts.lock().map_err(|_| "poisoned")?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn a_stop_with_no_snapshot_in_time_cancels_the_job()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::runs::{StopLimits, with_stop_fallback};
        let job = job()?;
        let limits = StopLimits {
            proof: Duration::from_millis(1),
            end: Duration::from_millis(50),
        };
        // (proof, status, how long the flow takes, cancels): no proof is
        // cancelled at once; a proof gets `end` more, then is cancelled if the
        // job still runs; a flow ending first cancels nothing.
        for (proof, status, lasts, cancels) in [
            ("", JobStatus::Running, 200, 1),
            (PROOF, JobStatus::Running, 200, 1),
            (PROOF, JobStatus::Running, 20, 0),
            ("", JobStatus::Exited(0), 200, 0),
        ] {
            let fake = Fake {
                proof: proof.to_string(),
                ..Fake::new(status)
            };
            let flow = tokio::time::sleep(Duration::from_millis(lasts));
            with_stop_fallback(&fake, &job, limits, flow).await;
            assert_eq!(
                fake.cancels.load(Ordering::SeqCst),
                cancels,
                "{proof:?} {status:?} {lasts}"
            );
        }
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn a_follow_cancels_a_job_only_from_a_request_it_sees()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::runs::{SnapshotReason, StopLimits, with_request_watch};
        let run = running()?;
        let request = format!("{}/{}", run.remote_dir, crate::train::SNAPSHOT_REQUEST);
        let limits = StopLimits {
            proof: Duration::from_secs(100),
            end: Duration::from_secs(50),
        };
        let every = Duration::from_secs(10);
        let mut old = run.clone();
        old.snapshots = false;
        // (run, when the request is written, how long the flow takes, cancels):
        // no request cancels nothing; the limits count from the request seen,
        // not from the start of the follow; the job of a run started before
        // snapshots ignores a request, so it is never cancelled for one.
        for (run, written, lasts, cancels) in [
            (&run, None, 500, 0),
            (&run, Some(80), 150, 0),
            (&run, Some(80), 250, 1),
            (&old, Some(80), 250, 0),
        ] {
            let fake = Fake::new(JobStatus::Running);
            let flow = async {
                if let Some(at) = written {
                    tokio::time::sleep(Duration::from_secs(at)).await;
                    fake.put_file(&request, SnapshotReason::Requested.name())
                        .await?;
                    tokio::time::sleep(Duration::from_secs(lasts - at)).await;
                } else {
                    tokio::time::sleep(Duration::from_secs(lasts)).await;
                }
                Ok::<_, ExecError>(())
            };
            with_request_watch(&fake, run, limits, every, flow).await?;
            assert_eq!(
                fake.cancels.load(Ordering::SeqCst),
                cancels,
                "{} {written:?} {lasts}",
                run.snapshots
            );
        }
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn a_follow_ends_with_its_job_while_a_request_read_hangs()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::runs::{StopLimits, with_request_watch};
        let run = running()?;
        let mut fake = Fake::new(JobStatus::Running);
        fake.request_reads = RequestReads::Hung;
        let limits = StopLimits {
            proof: Duration::from_secs(100),
            end: Duration::from_secs(50),
        };
        let flow = async {
            tokio::time::sleep(Duration::from_secs(25)).await;
            Ok::<_, ExecError>(())
        };
        let watched = with_request_watch(&fake, &run, limits, Duration::from_secs(10), flow);
        tokio::time::timeout(Duration::from_secs(60), watched).await??;
        assert_eq!(fake.cancels.load(Ordering::SeqCst), 0);
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
