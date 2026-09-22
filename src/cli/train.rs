//! The training commands: `train`, `train attach`, `train cancel` and `runs ls`.

use std::future::Future;
use std::io;
use std::path::Path;
use std::pin::{Pin, pin};
use std::task::{Context as TaskContext, Poll, Waker};
use std::time::Duration;

use anyhow::{Context, bail};

use super::progress::status_name;
use super::{TrainArgs, TrainCommand};
use crate::config::{DEFAULT_WORKDIR, EnvSource, Settings, Target, Training};
use crate::dataset::DataFiles;
use crate::events::EventBus;
use crate::exec::{AnyExecutor, JobRuntime, JobStatus, LocalExecutor, SshExecutor};
use crate::runs::{
    Launch, Outcome, RUNS_DIR, RunCtx, RunRecord, RunState, Runs, cancel, create, start, watch,
};
use crate::train::{Axolotl, OUTPUT_DIR, reasoning_template_warning};

/// Time between two looks at a running job.
const POLL: Duration = Duration::from_secs(2);

/// Runs `overbrainer train` or one of its subcommands.
///
/// # Errors
///
/// Returns an error when the configuration or the target cannot be used, when the
/// run fails, or when it is interrupted (the job keeps running).
pub async fn run(project_dir: &Path, args: &TrainArgs) -> anyhow::Result<()> {
    match &args.command {
        None => {
            let settings = crate::config::load(project_dir, EnvSource::Process)?;
            train(project_dir, &settings, args.target.as_deref()).await
        },
        Some(TrainCommand::Attach { run_id }) => attach(project_dir, run_id).await,
        Some(TrainCommand::Cancel { run_id }) => cancel_run(project_dir, run_id).await,
    }
}

/// Trains after `overbrainer run` when `[training]` is set.
///
/// # Errors
///
/// Returns an error when training fails.
pub async fn after_run(project_dir: &Path) -> anyhow::Result<()> {
    let settings = crate::config::load(project_dir, EnvSource::Process)?;
    if settings.training.is_none() {
        no_training();
        return Ok(());
    }
    train(project_dir, &settings, None).await
}

fn no_training() {
    tracing::info!("no [training] section in overbrainer.toml: run stops after split");
}

async fn train(
    project_dir: &Path,
    settings: &Settings,
    target: Option<&str>,
) -> anyhow::Result<()> {
    let training = training(settings)?;
    let name = target.unwrap_or(&training.target);
    let target = settings
        .targets
        .get(name)
        .with_context(|| format!("unknown target `{name}`"))?;
    let runtime = JobRuntime::from_target(target).with_context(|| runpod_refused(name))?;
    if let Some(warning) = reasoning_template_warning(training) {
        warn(&warning);
    }
    let secrets = secrets(settings).await?;
    let executor = executor(project_dir, name, target).await?;
    let runs = Runs::new(project_dir);
    // Caught from before the run exists, so Ctrl-C never kills the process while
    // its job is being started and not yet recorded.
    let mut interrupt = Interrupt::catch();
    let record = create(&runs, &executor, name)?;
    started(&record);
    let trainer = Axolotl::new(training, &DataFiles::new(project_dir));
    let id = record.id.clone();
    let bus = EventBus::new();
    let renderer = tokio::spawn(super::progress::render(bus.subscribe()));
    let ctx = RunCtx {
        runs: &runs,
        executor: &executor,
        bus: &bus,
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
    drop(bus);
    renderer.await.ok();
    finish(&runs, &id, result)
}

async fn attach(project_dir: &Path, run_id: &str) -> anyhow::Result<()> {
    let settings = crate::config::load(project_dir, EnvSource::Process)?;
    let training = training(&settings)?;
    let runs = Runs::new(project_dir);
    let record = runs.load(run_id)?;
    let executor = run_executor(project_dir, &settings, &record).await?;
    let trainer = Axolotl::new(training, &DataFiles::new(project_dir));
    let bus = EventBus::new();
    let renderer = tokio::spawn(super::progress::render(bus.subscribe()));
    let ctx = RunCtx {
        runs: &runs,
        executor: &executor,
        bus: &bus,
        poll: POLL,
    };
    let flow = async {
        watch(&ctx, &trainer, record)
            .await
            .with_context(|| reattach(run_id))
    };
    let result = Interrupt::catch().race(flow).await.transpose();
    drop(bus);
    renderer.await.ok();
    finish(&runs, run_id, result)
}

async fn cancel_run(project_dir: &Path, run_id: &str) -> anyhow::Result<()> {
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
    let executor = run_executor(project_dir, &settings, &record).await?;
    let trainer = Axolotl::new(training, &DataFiles::new(project_dir));
    // Cancelling is never interrupted: dropping it between the `cancelling` marker
    // and the signal would leave that marker on the target for ever.
    let (record, status) = Interrupt::catch()
        .shield(cancel(&runs, &executor, &trainer, record))
        .await?;
    if status != JobStatus::Cancelled {
        println!(
            "train: the job of run {run_id} had already ended ({}): collect it with `overbrainer train attach {run_id}`",
            status_name(status)
        );
        return Ok(());
    }
    match &record.message {
        Some(message) => println!("train: run {run_id} cancelled ({message})"),
        None => println!("train: run {run_id} cancelled"),
    }
    Ok(())
}

/// Prints the runs in `runs/`, oldest first: ID, state, target, creation time.
///
/// # Errors
///
/// Returns an error when a run record cannot be read.
pub fn list(project_dir: &Path) -> anyhow::Result<()> {
    for record in Runs::new(project_dir).list()? {
        println!(
            "{}  {:<9}  {}  {}",
            record.id,
            record.state.name(),
            record.target,
            record.created
        );
    }
    Ok(())
}

fn training(settings: &Settings) -> anyhow::Result<&Training> {
    settings
        .training
        .as_ref()
        .context("no [training] section in overbrainer.toml")
}

/// The Hugging Face token, resolved only now, for the job's environment.
async fn secrets(settings: &Settings) -> anyhow::Result<Vec<(String, secrecy::SecretString)>> {
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
        Target::Runpod { .. } => bail!(runpod_refused(name)),
    }
}

fn runpod_refused(name: &str) -> String {
    format!(
        "target `{name}` is a runpod target, which overbrainer cannot train on yet \
         (planned for M4): use a local or ssh target"
    )
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

type CtrlC = Pin<Box<dyn Future<Output = io::Result<()>> + Send>>;

/// Ctrl-C, caught from [`Interrupt::catch`] on so that it no longer stops the
/// process: a flow either runs to its end regardless ([`Interrupt::shield`]) or
/// stops at the first Ctrl-C ([`Interrupt::race`]).
enum Interrupt {
    /// Waiting for Ctrl-C.
    Listening(CtrlC),
    /// Ctrl-C was pressed.
    Caught,
    /// Ctrl-C cannot be caught: it stops the process as usual.
    Off,
}

impl Interrupt {
    /// Starts catching Ctrl-C now.
    fn catch() -> Self {
        let mut signal: CtrlC = Box::pin(tokio::signal::ctrl_c());
        // The handler is installed on the first poll; a later poll registers the
        // real waker.
        let mut cx = TaskContext::from_waker(Waker::noop());
        match signal.as_mut().poll(&mut cx) {
            Poll::Pending => Self::Listening(signal),
            Poll::Ready(result) => Self::after(result),
        }
    }

    fn after(result: io::Result<()>) -> Self {
        match result {
            Ok(()) => Self::Caught,
            Err(error) => {
                warn(&format!("cannot catch Ctrl-C: {error}"));
                Self::Off
            },
        }
    }

    /// Whether Ctrl-C was pressed.
    fn caught(&self) -> bool {
        matches!(self, Self::Caught)
    }

    /// Runs `flow` to its end, noting a Ctrl-C pressed meanwhile.
    async fn shield<T>(&mut self, flow: impl Future<Output = T>) -> T {
        let mut flow = pin!(flow);
        let result = match self {
            Self::Listening(signal) => tokio::select! {
                output = &mut flow => return output,
                result = signal.as_mut() => result,
            },
            Self::Caught | Self::Off => return flow.await,
        };
        *self = Self::after(result);
        flow.await
    }

    /// Runs `flow`, or `None` when Ctrl-C is pressed first (or was already).
    async fn race<T>(&mut self, flow: impl Future<Output = T>) -> Option<T> {
        let mut flow = pin!(flow);
        let result = match self {
            Self::Listening(signal) => tokio::select! {
                output = &mut flow => return Some(output),
                result = signal.as_mut() => result,
            },
            Self::Caught => return None,
            Self::Off => return Some(flow.await),
        };
        *self = Self::after(result);
        if self.caught() {
            None
        } else {
            Some(flow.await)
        }
    }
}

/// Prints the outcome on stdout, or explains how to follow an interrupted run.
fn finish(runs: &Runs, id: &str, result: anyhow::Result<Option<Outcome>>) -> anyhow::Result<()> {
    let Some(outcome) = result? else {
        return interrupted(runs, id);
    };
    let record = &outcome.record;
    let output = if record.state == RunState::Succeeded {
        format!("; output in runs/{}/{OUTPUT_DIR}", record.id)
    } else {
        String::new()
    };
    println!(
        "train: run {} {}; {}{output}",
        record.id,
        record.state.name(),
        outcome.summary.describe()
    );
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

fn reattach(id: &str) -> String {
    format!("follow it again with `overbrainer train attach {id}`")
}

fn started(record: &RunRecord) {
    tracing::info!(
        "train: run {} on target `{}`, in {}",
        record.id,
        record.target,
        record.remote_dir
    );
}

fn warn(message: &str) {
    tracing::warn!("{message}");
}

#[cfg(test)]
mod tests {
    use tokio::signal::unix::{SignalKind, signal};

    use super::*;

    /// Sends SIGINT to this process, as Ctrl-C does.
    async fn ctrl_c() -> io::Result<()> {
        let status = tokio::process::Command::new("kill")
            .args(["-INT", &std::process::id().to_string()])
            .status()
            .await?;
        if status.success() {
            Ok(())
        } else {
            Err(io::Error::other("kill failed"))
        }
    }

    #[tokio::test]
    async fn ctrl_c_never_stops_a_shielded_flow_and_stops_a_raced_one()
    -> Result<(), Box<dyn std::error::Error>> {
        let limit = Duration::from_secs(10);
        let mut interrupt = Interrupt::catch();
        assert!(matches!(interrupt, Interrupt::Listening(_)));
        // Ctrl-C arrives while the flow waits: it still ends, and Ctrl-C is noted.
        let mut received = signal(SignalKind::interrupt())?;
        let shielded = async {
            ctrl_c().await?;
            received.recv().await;
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok::<_, io::Error>("ended")
        };
        let ended = tokio::time::timeout(limit, interrupt.shield(shielded)).await??;
        assert_eq!(ended, "ended");
        assert!(interrupt.caught());
        // Once Ctrl-C was pressed, a race does not even start its flow.
        assert_eq!(interrupt.race(std::future::ready(())).await, None);

        let mut interrupt = Interrupt::catch();
        let raced = async {
            ctrl_c().await?;
            std::future::pending::<()>().await;
            Ok::<_, io::Error>(())
        };
        let raced = tokio::time::timeout(limit, interrupt.race(raced)).await?;
        assert!(raced.is_none());
        assert!(interrupt.caught());
        Ok(())
    }
}
