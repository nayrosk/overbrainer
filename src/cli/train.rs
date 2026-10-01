//! The training commands: `train`, `train attach`, `train stop`, `train cancel`
//! and `runs ls`.

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, anyhow, bail};

use super::front::{Frontend, Interrupt};
use super::progress::status_name;
use super::reload::Reloader;
use super::runpod_train::RunpodStart;
use super::{TrainArgs, TrainCommand};
use crate::config::{DEFAULT_WORKDIR, Settings, Source, Target, Training};
use crate::dataset::DataFiles;
use crate::exec::{AnyExecutor, Executor, JobRuntime, JobStatus, LocalExecutor, SshExecutor};
use crate::runpod::{PodRecord, RunpodTarget, connect_followed};
use crate::runs::{
    Launch, Outcome, REQUEST_POLL, RUNS_DIR, RunCtx, RunError, RunRecord, RunState, Runs,
    STOP_LIMITS, SnapshotReason, cancel, create_on, request_snapshot, start, watch,
    with_request_watch, with_stop_fallback,
};
use crate::train::{Axolotl, OUTPUT_DIR, Outputs, Resume, reasoning_template_warning};

/// Time between two looks at a running job.
pub(super) const POLL: Duration = Duration::from_secs(2);

/// Runs `overbrainer train` or one of its subcommands, for `front`, with the
/// settings of `source`.
///
/// # Errors
///
/// Returns an error when the configuration or the target cannot be used, when the
/// run fails, or when it is interrupted (the job keeps running).
pub async fn run(
    project_dir: &Path,
    args: &TrainArgs,
    front: &Frontend,
    source: &Source,
) -> anyhow::Result<()> {
    match &args.command {
        None => {
            let settings = source.load(project_dir)?;
            train(project_dir, &settings, args, front).await
        },
        Some(TrainCommand::Attach { run_id }) => attach(project_dir, run_id, front, source).await,
        Some(TrainCommand::Stop { run_id }) => stop(project_dir, run_id, front, source).await,
        Some(TrainCommand::Cancel { run_id }) => {
            cancel_run(project_dir, run_id, front, source).await
        },
    }
}

/// Trains after `overbrainer run` when `[training]` is set, with the settings
/// `reloader` reads again when their files changed since the last stage.
///
/// # Errors
///
/// Returns an error when the changed configuration cannot be used or training
/// fails.
pub(crate) async fn after_run(
    project_dir: &Path,
    front: &Frontend,
    reloader: &mut Reloader,
) -> anyhow::Result<()> {
    let settings = match reloader.changed()? {
        Some(settings) => settings,
        None => crate::config::load(project_dir, reloader.env().clone())?,
    };
    if settings.training.is_none() {
        no_training();
        return Ok(());
    }
    train(project_dir, &settings, &TrainArgs::default(), front).await
}

fn no_training() {
    tracing::info!("no [training] section in overbrainer.toml: run stops after split");
}

async fn train(
    project_dir: &Path,
    settings: &Settings,
    args: &TrainArgs,
    front: &Frontend,
) -> anyhow::Result<()> {
    let keep_pod = args.keep_pod;
    let training = training(settings)?;
    let name = args.target.as_deref().unwrap_or(&training.target);
    let target = settings
        .targets
        .get(name)
        .with_context(|| format!("unknown target `{name}`"))?;
    let runs = Runs::new(project_dir);
    let trainer = trainer(project_dir, training, &runs, args.resume_from.as_deref())?;
    if let Some(spec) = RunpodTarget::from_target(target) {
        let start = RunpodStart {
            name,
            spec: &spec,
            keep: keep_pod,
            vram_floor: args.vram_floor,
            trainer: &trainer,
        };
        // Boxed: the Runpod flow would otherwise weigh on every caller's future.
        return Box::pin(super::runpod_train::train(
            project_dir,
            settings,
            start,
            front,
        ))
        .await;
    }
    if keep_pod {
        bail!("--keep-pod only applies to a runpod target");
    }
    let runtime = JobRuntime::from_target(target).with_context(|| runpod_only(name))?;
    if let Some(warning) = reasoning_template_warning(training) {
        warn(&warning);
    }
    // Caught from before the preparation, so Ctrl-C stops it without a run (the
    // command line's handler may already be installed by `overbrainer run`), and
    // never kills the process while the job is being started and not yet recorded.
    let mut interrupt = front.interrupt();
    let (secrets, executor) = prepare(&mut interrupt, async {
        let secrets = secrets(settings).await?;
        let executor = executor(project_dir, name, target).await?;
        Ok((secrets, executor))
    })
    .await?;
    let record = create_on(&runs, &executor, &settings.project.name, name).await?;
    let record = resumed(&runs, record, &trainer)?;
    let trainer = exporting(&trainer, &record.id, &settings.export);
    started(&record);
    front.run_created(&record.id);
    let id = record.id.clone();
    let guard = front.open_bus();
    let ctx = RunCtx {
        runs: &runs,
        executor: &executor,
        bus: &guard.bus,
        poll: POLL,
    };
    let launch = Launch {
        runtime: &runtime,
        secrets,
    };
    // Starting is never interrupted: dropping it after the job is spawned and
    // before its record is saved would leave a job nothing can find again.
    let result = match interrupt
        .shield(start(&ctx, &trainer, launch, record))
        .await
    {
        Err(error) => Err(error.into()),
        Ok(_) if interrupt.caught() => Ok(None),
        Ok(record) => {
            let flow = async {
                watch_requests(&ctx, &trainer, record)
                    .await
                    .with_context(|| reattach(&id))
            };
            interrupt.race(flow).await.transpose()
        },
    };
    guard.close().await;
    finish(
        &runs,
        &id,
        result,
        front,
        settings.export.ollama_name.as_deref(),
    )
}

/// `trainer` for the new run `run_id`, exporting its model at the end of its
/// job when `[export] after_training` is set.
pub(super) fn exporting<'a>(
    trainer: &Axolotl<'a>,
    run_id: &str,
    export: &crate::config::Export,
) -> Axolotl<'a> {
    if export.after_training {
        trainer.clone().exporting(run_id, &export.quantize)
    } else {
        trainer.clone()
    }
}

async fn attach(
    project_dir: &Path,
    run_id: &str,
    front: &Frontend,
    source: &Source,
) -> anyhow::Result<()> {
    let settings = source.load(project_dir)?;
    let training = training(&settings)?;
    let runs = Runs::new(project_dir);
    let record = runs.load(run_id)?;
    if let Some(pod) = PodRecord::load(&runs, run_id)? {
        return super::runpod_train::attach(project_dir, &settings, record, pod, front).await;
    }
    let executor = run_executor(project_dir, &settings, &record).await?;
    let trainer = Axolotl::new(training, &DataFiles::new(project_dir));
    let guard = front.open_bus();
    let ctx = RunCtx {
        runs: &runs,
        executor: &executor,
        bus: &guard.bus,
        poll: POLL,
    };
    let flow = async {
        watch_requests(&ctx, &trainer, record)
            .await
            .with_context(|| reattach(run_id))
    };
    let result = front.interrupt().race(flow).await.transpose();
    guard.close().await;
    finish(
        &runs,
        run_id,
        result,
        front,
        settings.export.ollama_name.as_deref(),
    )
}

/// Follows `record` as [`watch`] does. A snapshot request written on the
/// target meanwhile, by `train stop` from another process for instance, is
/// then held to [`STOP_LIMITS`] as `train stop` holds its own (never for a run
/// whose job cannot save a snapshot).
async fn watch_requests<E: Executor>(
    ctx: &RunCtx<'_, E>,
    trainer: &Axolotl<'_>,
    record: RunRecord,
) -> Result<Outcome, RunError> {
    let run = record.clone();
    // Boxed: its state would otherwise weigh on every caller's future.
    let watched = Box::pin(watch(ctx, trainer, record));
    with_request_watch(ctx.executor, &run, STOP_LIMITS, REQUEST_POLL, watched).await
}

/// The trainer of a new run of `training`: resuming from the stopped run
/// `resume_from` when given, which must hold its snapshot locally and have
/// been trained with the same settings.
///
/// # Errors
///
/// Returns an error when `resume_from` is not a stopped run with its
/// checkpoint in `runs/`, or its training settings differ from `training`.
pub(crate) fn trainer<'a>(
    project_dir: &Path,
    training: &'a Training,
    runs: &Runs,
    resume_from: Option<&str>,
) -> anyhow::Result<Axolotl<'a>> {
    let trainer = Axolotl::new(training, &DataFiles::new(project_dir));
    let Some(id) = resume_from else {
        return Ok(trainer);
    };
    let source = runs.load(id)?;
    let snapshot = match (source.state, &source.snapshot) {
        (RunState::Stopped, Some(snapshot)) => snapshot,
        (state, _) => bail!(
            "cannot resume from run {id}: it is {}, not stopped with a snapshot (see `overbrainer train stop`)",
            state.name()
        ),
    };
    let dir = runs.run_dir(id)?;
    if !dir.join(&snapshot.checkpoint).is_dir() {
        bail!(
            "cannot resume from run {id}: its snapshot is not in {RUNS_DIR}/{id}/{}; retrieve it \
             with `overbrainer train attach {id}`",
            snapshot.checkpoint
        );
    }
    let trainer = trainer.resuming(Resume {
        run_id: id.to_string(),
        dir,
        checkpoint: snapshot.checkpoint.clone(),
    });
    let differ = trainer.resume_mismatch()?;
    if !differ.is_empty() {
        bail!(
            "cannot resume from run {id}: the training settings differ from the ones it ran \
             with ({}); set them back to resume, or start a new run",
            differ.join(", ")
        );
    }
    Ok(trainer)
}

/// The new run `record` of `trainer`, recording the run it resumes from.
///
/// # Errors
///
/// Returns an error when the record cannot be saved.
pub(super) fn resumed(
    runs: &Runs,
    mut record: RunRecord,
    trainer: &Axolotl<'_>,
) -> anyhow::Result<RunRecord> {
    if let Some(resume) = trainer.resume() {
        record.resumed_from = Some(resume.run_id.clone());
        runs.save(&record)?;
        tracing::info!(
            "train: run {} resumes from the snapshot of run {} ({})",
            record.id,
            resume.run_id,
            resume.checkpoint
        );
    }
    Ok(record)
}

/// `train stop`: asks the job for a snapshot, then follows it as `train attach`
/// does until it ends, stopped; a job that gives no snapshot in time, or does
/// not end once it gave one, is cancelled instead (see [`STOP_LIMITS`]).
async fn stop(
    project_dir: &Path,
    run_id: &str,
    front: &Frontend,
    source: &Source,
) -> anyhow::Result<()> {
    let settings = source.load(project_dir)?;
    let training = training(&settings)?;
    let runs = Runs::new(project_dir);
    let record = runs.load(run_id)?;
    let job = stoppable(&record)?;
    if let Some(pod) = PodRecord::load(&runs, run_id)? {
        return super::runpod_train::stop(project_dir, &settings, record, pod, front).await;
    }
    let executor = run_executor(project_dir, &settings, &record).await?;
    request_snapshot(&executor, &record, SnapshotReason::Requested).await?;
    front.line(&stop_requested(run_id));
    let trainer = Axolotl::new(training, &DataFiles::new(project_dir));
    let guard = front.open_bus();
    let ctx = RunCtx {
        runs: &runs,
        executor: &executor,
        bus: &guard.bus,
        poll: POLL,
    };
    let flow = async {
        // Boxed: its state would otherwise weigh on every caller's future.
        let watched = Box::pin(watch(&ctx, &trainer, record));
        with_stop_fallback(&executor, &job, STOP_LIMITS, watched)
            .await
            .with_context(|| reattach(run_id))
    };
    let result = front.interrupt().race(flow).await.transpose();
    guard.close().await;
    finish(
        &runs,
        run_id,
        result,
        front,
        settings.export.ollama_name.as_deref(),
    )
}

/// `train stop` while another overbrainer process, `holder`, holds the project,
/// most likely following the run: only writes the snapshot request on the
/// target and leaves the snapshot to the process that follows the run. Reads
/// `run.json` and `pod.json`, and writes nothing in the run directory but, on a
/// local target, the request itself.
///
/// # Errors
///
/// Returns an error when the configuration cannot be used, the run is not
/// running, its target cannot be reached, or the request cannot be written.
pub(super) async fn request_stop(
    project_dir: &Path,
    run_id: &str,
    holder: Option<u32>,
    front: &Frontend,
    source: &Source,
) -> anyhow::Result<()> {
    let settings = source.load(project_dir)?;
    let runs = Runs::new(project_dir);
    let record = runs.load(run_id)?;
    stoppable(&record)?;
    if let Some(pod) = PodRecord::load(&runs, run_id)? {
        let Some(executor) = connect_followed(&runs, &pod, &record).await? else {
            bail!("run {run_id} has no pod to reach: nothing to stop");
        };
        request_snapshot(&executor, &record, SnapshotReason::Requested).await?;
    } else {
        let executor = run_executor(project_dir, &settings, &record).await?;
        request_snapshot(&executor, &record, SnapshotReason::Requested).await?;
    }
    front.line(&stop_requested(run_id));
    front.line(&left_to_holder(run_id, holder));
    Ok(())
}

/// What `train stop` says when another overbrainer process, `holder`, holds
/// the project and so collects the snapshot.
fn left_to_holder(run_id: &str, holder: Option<u32>) -> String {
    let holder = holder.map_or_else(
        || "another overbrainer".to_string(),
        |pid| format!("overbrainer (pid {pid})"),
    );
    format!(
        "train: {holder} is using this project: if it follows run {run_id}, it collects the \
         snapshot; if not, collect it with `overbrainer train attach {run_id}`"
    )
}

/// The job of `record` when a snapshot can be asked of it: the run is running,
/// and its job can save one (an older overbrainer's cannot).
pub(super) fn stoppable(record: &RunRecord) -> anyhow::Result<crate::exec::JobId> {
    let id = &record.id;
    match (&record.job, record.state) {
        (Some(_), RunState::Running) if !record.snapshots => {
            Err(RunError::NoSnapshots(id.clone()).into())
        },
        (Some(job), RunState::Running) => Ok(job.clone()),
        (None, _) | (_, RunState::Preparing) => {
            bail!("run {id} has not started: there is no job to snapshot yet")
        },
        (Some(_), state) => bail!("run {id} already ended: {}", state.name()),
    }
}

/// What `train stop` says once the request is written.
pub(super) fn stop_requested(run_id: &str) -> String {
    format!(
        "train: snapshot of run {run_id} requested: its job saves a checkpoint at the end of \
         its current step, then stops"
    )
}

async fn cancel_run(
    project_dir: &Path,
    run_id: &str,
    front: &Frontend,
    source: &Source,
) -> anyhow::Result<()> {
    let settings = source.load(project_dir)?;
    let runs = Runs::new(project_dir);
    let record = runs.load(run_id)?;
    match (record.state, record.job.is_some()) {
        (RunState::Cancelled, _) => bail!("run {run_id} already ended: cancelled"),
        (_, false) => bail!("run {run_id} has not started"),
        (RunState::Preparing | RunState::Running, true) => {},
        // A run recorded as ended can still have a job left on the target: a
        // container that outlived its wrapper keeps the GPU until it is stopped.
        (state, true) => tracing::info!(
            "train: run {run_id} already ended: {}; stopping any job left on the target",
            state.name()
        ),
    }
    let training = settings.training.as_ref().context(
        "cancel needs [training] to retrieve the run's artifacts: \
         no [training] section in overbrainer.toml",
    )?;
    if let Some(pod) = PodRecord::load(&runs, run_id)? {
        return super::runpod_train::cancel(project_dir, &settings, record, pod, front).await;
    }
    let executor = run_executor(project_dir, &settings, &record).await?;
    let trainer = Axolotl::new(training, &DataFiles::new(project_dir));
    // Cancelling is never interrupted: dropping it between the `cancelling` marker
    // and the signal would leave that marker on the target for ever.
    let (record, status, _) = front
        .interrupt()
        .shield(cancel(&runs, &executor, &trainer, record))
        .await?;
    if status != JobStatus::Cancelled {
        front.line(&format!(
            "train: the job of run {run_id} had already ended ({}): collect it with `overbrainer train attach {run_id}`",
            status_name(status)
        ));
        return Ok(());
    }
    match &record.message {
        Some(message) => front.line(&format!("train: run {run_id} cancelled ({message})")),
        None => front.line(&format!("train: run {run_id} cancelled")),
    }
    Ok(())
}

/// Prints the runs in `runs/`, oldest first: ID, state, target, creation time,
/// the step and reason of a stopped run's snapshot (the model its `output/`
/// holds is partial, from that step), and, for a Runpod run, what
/// `pod.json` says of its pod (no API call).
///
/// # Errors
///
/// Returns an error when a run record cannot be read.
pub fn list(project_dir: &Path) -> anyhow::Result<()> {
    let runs = Runs::new(project_dir);
    let records = runs.list()?;
    let width = records
        .iter()
        .map(|record| record.id.len())
        .max()
        .unwrap_or(0);
    for record in records {
        let pod = match PodRecord::load(&runs, &record.id) {
            Ok(Some(pod)) => format!("  {}", pod.summary()),
            Ok(None) => String::new(),
            Err(error) => {
                warn(&format!(
                    "cannot read the pod record of run {}: {error}",
                    record.id
                ));
                String::new()
            },
        };
        let snapshot = record
            .snapshot
            .as_ref()
            .map_or_else(String::new, |snapshot| {
                format!(
                    "  step {} ({}): partial model in {OUTPUT_DIR}/",
                    snapshot.step,
                    snapshot.reason.name()
                )
            });
        println!(
            "{:<width$}  {:<9}  {}  {}{snapshot}{pod}",
            record.id,
            record.state.name(),
            record.target,
            record.created
        );
    }
    Ok(())
}

/// Runs `steps`, the preparation of a run that does not exist yet, unless the
/// interruption comes first.
///
/// # Errors
///
/// Returns the error of `steps`, or an error when the interruption came first.
pub(super) async fn prepare<T>(
    interrupt: &mut Interrupt,
    steps: impl Future<Output = anyhow::Result<T>>,
) -> anyhow::Result<T> {
    interrupt
        .race(steps)
        .await
        .unwrap_or_else(|| Err(anyhow!("interrupted before the run started")))
}

pub(super) fn training(settings: &Settings) -> anyhow::Result<&Training> {
    settings
        .training
        .as_ref()
        .context("no [training] section in overbrainer.toml")
}

/// The job's environment variable holding the Hugging Face token.
pub(super) const HF_TOKEN: &str = "HF_TOKEN";

/// The Hugging Face token, resolved only now, for the job's environment.
pub(super) async fn secrets(
    settings: &Settings,
) -> anyhow::Result<Vec<(String, secrecy::SecretString)>> {
    let Some(token) = &settings.hf_token else {
        if settings
            .training
            .as_ref()
            .is_some_and(|training| training.hub_model_id.is_some())
        {
            warn(
                "training.hub_model_id is set but OVERBRAINER_HF_TOKEN is not: the push will fail",
            );
        }
        return Ok(Vec::new());
    };
    let token = hf_token(token).await?;
    Ok(vec![(HF_TOKEN.to_string(), token)])
}

/// The Hugging Face token `token` of the settings, resolved only now.
///
/// # Errors
///
/// Returns an error when it cannot be resolved.
pub(crate) async fn hf_token(
    token: &secrecy::SecretString,
) -> anyhow::Result<secrecy::SecretString> {
    super::resolver()
        .resolve(token)
        .await
        .context("cannot resolve hf_token")
}

/// The executor of the local or SSH target `name`.
///
/// # Errors
///
/// Returns an error for a Runpod target, an SSH target without a host, or
/// one that cannot be reached.
pub(super) async fn executor(
    project_dir: &Path,
    name: &str,
    target: &Target,
) -> anyhow::Result<AnyExecutor> {
    match target {
        Target::Local { .. } => Ok(AnyExecutor::Local(LocalExecutor::new(
            &project_dir.join(RUNS_DIR),
        )?)),
        Target::Ssh { host, workdir, .. } => {
            let host = host.as_deref().with_context(|| {
                format!(
                    "targets.{name}.host is not set: set OVERBRAINER_TARGETS__{}__HOST",
                    name.to_ascii_uppercase()
                )
            })?;
            let workdir = workdir.as_deref().unwrap_or(DEFAULT_WORKDIR);
            let executor = SshExecutor::connect(host, workdir, None)
                .await
                .with_context(|| format!("cannot reach target `{name}`"))?;
            Ok(AnyExecutor::Ssh(executor))
        },
        Target::Runpod { .. } => bail!(runpod_only(name)),
    }
}

/// A Runpod target has no executor until its run's pod exists: its runs go
/// through `runpod_train`.
fn runpod_only(name: &str) -> String {
    format!("target `{name}` is a runpod target: its runs are reached through their pod")
}

/// The executor of the target a run was started on.
async fn run_executor(
    project_dir: &Path,
    settings: &Settings,
    record: &RunRecord,
) -> anyhow::Result<AnyExecutor> {
    let target = settings.targets.get(&record.target).with_context(|| {
        format!(
            "target `{}` of run {} is no longer in overbrainer.toml",
            record.target, record.id
        )
    })?;
    executor(project_dir, &record.target, target).await
}

/// Emits the outcome through `front` (stdout on the command line), then, for a
/// run that succeeded, where its model is, read from the run's own files (the
/// settings may have changed since it started; nothing is said when they
/// cannot be read), and the GGUF of an export in its job with its Modelfile,
/// creating the Ollama model `ollama`; for a stopped one where its snapshot is
/// and how to resume from it; or explains how to follow an interrupted run.
pub(super) fn finish(
    runs: &Runs,
    id: &str,
    result: anyhow::Result<Option<Outcome>>,
    front: &Frontend,
    ollama: Option<&str>,
) -> anyhow::Result<()> {
    let Some(outcome) = result? else {
        return interrupted(runs, id);
    };
    let record = &outcome.record;
    let output = if record.state == RunState::Succeeded {
        format!("; output in runs/{}/{OUTPUT_DIR}", record.id)
    } else {
        String::new()
    };
    front.line(&format!(
        "train: run {} {}; {}{output}",
        record.id,
        record.state.name(),
        outcome.summary.describe()
    ));
    match record.state {
        RunState::Succeeded => {
            let outputs = runs
                .run_dir(&record.id)
                .ok()
                .and_then(|dir| Outputs::recorded(&dir));
            for (what, path) in outputs.map(|o| o.paths(&record.id)).unwrap_or_default() {
                front.line(&format!("train: {what} in {path}"));
            }
            super::export::deliver_in_job(runs, record, ollama, front);
            Ok(())
        },
        RunState::Stopped => {
            for line in stopped_lines(record) {
                front.line(&line);
            }
            if let Some(snapshot) = &record.snapshot {
                front.run_stopped(snapshot.step);
            }
            Ok(())
        },
        RunState::Cancelled => bail!("run {} was cancelled", record.id),
        _ => bail!(
            "{}",
            record
                .message
                .clone()
                .unwrap_or_else(|| format!("run {} failed", record.id))
        ),
    }
}

/// What a stopped run says: where its snapshot is, how to resume from it, and
/// that the model Axolotl saved in `output/` when training ended early is
/// partial, from the snapshot's step. Nothing without a snapshot.
pub(crate) fn stopped_lines(record: &RunRecord) -> Vec<String> {
    let Some(snapshot) = record.snapshot.as_ref() else {
        return Vec::new();
    };
    let id = &record.id;
    vec![
        format!(
            "train: run {id} stopped at step {} ({}): snapshot in {RUNS_DIR}/{id}/{}; resume with \
             `overbrainer train --resume-from {id}`",
            snapshot.step,
            snapshot.reason.name(),
            snapshot.checkpoint
        ),
        format!(
            "train: {RUNS_DIR}/{id}/{OUTPUT_DIR} also holds the partial model at step {}, not a \
             finished one",
            snapshot.step
        ),
    ]
}

/// After Ctrl-C, which only stops following a started job: it keeps running. A
/// start is never interrupted, so a run without a job here never spawned one; its
/// record already says why.
fn interrupted(runs: &Runs, id: &str) -> anyhow::Result<()> {
    let record = runs.load(id)?;
    if record.job.is_none() {
        bail!("interrupted: run {id} has no job");
    }
    bail!(
        "interrupted: run {id} keeps running on target `{}`; {}",
        record.target,
        reattach(id)
    )
}

pub(super) fn reattach(id: &str) -> String {
    format!("follow it again with `overbrainer train attach {id}`")
}

pub(super) fn started(record: &RunRecord) {
    tracing::info!(
        "train: run {} on target `{}`, in {}",
        record.id,
        record.target,
        record.remote_dir
    );
}

pub(super) fn warn(message: &str) {
    tracing::warn!("{message}");
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::events::EventBus;

    #[tokio::test]
    async fn an_interruption_stops_the_preparation() -> anyhow::Result<()> {
        let detach = CancellationToken::new();
        let front = Frontend::Tui {
            bus: EventBus::with_capacity(8),
            detach: detach.clone(),
            abandon: Arc::new(AtomicBool::new(false)),
            report: Arc::new(|_| {}),
        };
        let mut interrupt = front.interrupt();
        let steps = async {
            detach.cancel();
            std::future::pending::<anyhow::Result<()>>().await
        };
        let prepared =
            tokio::time::timeout(Duration::from_secs(10), prepare(&mut interrupt, steps)).await?;
        let Err(error) = prepared else {
            anyhow::bail!("the preparation was not interrupted");
        };
        assert_eq!(error.to_string(), "interrupted before the run started");
        assert!(interrupt.caught());
        Ok(())
    }
}
