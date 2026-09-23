//! The training commands: `train`, `train attach`, `train cancel` and `runs ls`.

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, bail};

use super::front::Frontend;
use super::progress::status_name;
use super::runpod_train::RunpodStart;
use super::{TrainArgs, TrainCommand};
use crate::config::{DEFAULT_WORKDIR, EnvSource, Settings, Target, Training};
use crate::dataset::DataFiles;
use crate::exec::{AnyExecutor, Executor, JobRuntime, JobStatus, LocalExecutor, SshExecutor};
use crate::runpod::{PodRecord, RunpodTarget};
use crate::runs::{
    Launch, Outcome, RUNS_DIR, RunCtx, RunRecord, RunState, Runs, cancel, create, start, watch,
};
use crate::train::{Axolotl, OUTPUT_DIR, reasoning_template_warning};

/// Time between two looks at a running job.
pub(super) const POLL: Duration = Duration::from_secs(2);

/// Runs `overbrainer train` or one of its subcommands, for `front`.
///
/// # Errors
///
/// Returns an error when the configuration or the target cannot be used, when the
/// run fails, or when it is interrupted (the job keeps running).
pub async fn run(project_dir: &Path, args: &TrainArgs, front: &Frontend) -> anyhow::Result<()> {
    match &args.command {
        None => {
            let settings = crate::config::load(project_dir, EnvSource::Process)?;
            train(
                project_dir,
                &settings,
                args.target.as_deref(),
                args.keep_pod,
                front,
            )
            .await
        },
        Some(TrainCommand::Attach { run_id }) => attach(project_dir, run_id, front).await,
        Some(TrainCommand::Cancel { run_id }) => cancel_run(project_dir, run_id, front).await,
    }
}

/// Trains after `overbrainer run` when `[training]` is set.
///
/// # Errors
///
/// Returns an error when training fails.
pub async fn after_run(project_dir: &Path, front: &Frontend) -> anyhow::Result<()> {
    let settings = crate::config::load(project_dir, EnvSource::Process)?;
    if settings.training.is_none() {
        no_training();
        return Ok(());
    }
    train(project_dir, &settings, None, false, front).await
}

fn no_training() {
    tracing::info!("no [training] section in overbrainer.toml: run stops after split");
}

async fn train(
    project_dir: &Path,
    settings: &Settings,
    target: Option<&str>,
    keep_pod: bool,
    front: &Frontend,
) -> anyhow::Result<()> {
    let training = training(settings)?;
    let name = target.unwrap_or(&training.target);
    let target = settings
        .targets
        .get(name)
        .with_context(|| format!("unknown target `{name}`"))?;
    if let Some(spec) = RunpodTarget::from_target(target) {
        let start = RunpodStart {
            name,
            spec: &spec,
            keep: keep_pod,
        };
        return super::runpod_train::train(project_dir, settings, start, front).await;
    }
    if keep_pod {
        bail!("--keep-pod only applies to a runpod target");
    }
    let runtime = JobRuntime::from_target(target).with_context(|| runpod_only(name))?;
    if let Some(warning) = reasoning_template_warning(training) {
        warn(&warning);
    }
    let secrets = secrets(settings).await?;
    let executor = executor(project_dir, name, target).await?;
    let runs = Runs::new(project_dir);
    // Caught from before the run exists, so Ctrl-C never kills the process while
    // its job is being started and not yet recorded.
    let mut interrupt = front.interrupt();
    let record = create(&runs, executor.workdir(), name)?;
    started(&record);
    front.run_created(&record.id);
    let trainer = Axolotl::new(training, &DataFiles::new(project_dir));
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
                watch(&ctx, &trainer, record)
                    .await
                    .with_context(|| reattach(&id))
            };
            interrupt.race(flow).await.transpose()
        },
    };
    guard.close().await;
    finish(&runs, &id, result, front)
}

async fn attach(project_dir: &Path, run_id: &str, front: &Frontend) -> anyhow::Result<()> {
    let settings = crate::config::load(project_dir, EnvSource::Process)?;
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
        watch(&ctx, &trainer, record)
            .await
            .with_context(|| reattach(run_id))
    };
    let result = front.interrupt().race(flow).await.transpose();
    guard.close().await;
    finish(&runs, run_id, result, front)
}

async fn cancel_run(project_dir: &Path, run_id: &str, front: &Frontend) -> anyhow::Result<()> {
    let settings = crate::config::load(project_dir, EnvSource::Process)?;
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
/// and, for a Runpod run, what `pod.json` says of its pod (no API call).
///
/// # Errors
///
/// Returns an error when a run record cannot be read.
pub fn list(project_dir: &Path) -> anyhow::Result<()> {
    let runs = Runs::new(project_dir);
    for record in runs.list()? {
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
        println!(
            "{}  {:<9}  {}  {}{pod}",
            record.id,
            record.state.name(),
            record.target,
            record.created
        );
    }
    Ok(())
}

pub(super) fn training(settings: &Settings) -> anyhow::Result<&Training> {
    settings
        .training
        .as_ref()
        .context("no [training] section in overbrainer.toml")
}

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
    let token = super::resolver()
        .resolve(token)
        .await
        .context("cannot resolve hf_token")?;
    Ok(vec![("HF_TOKEN".to_string(), token)])
}

async fn executor(project_dir: &Path, name: &str, target: &Target) -> anyhow::Result<AnyExecutor> {
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

/// Emits the outcome through `front` (stdout on the command line), or explains how
/// to follow an interrupted run.
pub(super) fn finish(
    runs: &Runs,
    id: &str,
    result: anyhow::Result<Option<Outcome>>,
    front: &Frontend,
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
        RunState::Succeeded => Ok(()),
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
