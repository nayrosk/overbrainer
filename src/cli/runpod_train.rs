//! `train`, `train attach`, `train stop` and `train cancel` on a Runpod target: the pod is
//! created before the run's job and deleted once its results are retrieved.
//!
//! Ctrl-C before the job exists deletes the pod and fails the run. Once the job is
//! started, Ctrl-C only stops following it: the job and the pod keep running, and
//! the pod's watchdog bounds the cost.

use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, SystemTime};

use anyhow::{Context, bail};
use secrecy::SecretString;

use super::front::{BusGuard, Flag, Frontend, Interrupt};
use super::train::{
    HF_TOKEN, POLL, finish, prepare, reattach, resumed, secrets, started, stop_requested,
    stoppable, training, warn,
};
use crate::config::{Settings, Training};
use crate::dataset::DataFiles;
use crate::exec::{JobStatus, LocalExecutor, SshExecutor};
use crate::runpod::{
    DeleteReason, DeletedBy, Ending, LEASE_TTL, PodCtx, PodError, PodRecord, PodState,
    RunpodClient, RunpodTarget, Timing, arm_cost_cap, chain, end_pod, forget_client_key,
    job_started, limit_reached, listed_rows, orphan_warnings, reconnect, remove, settle_watch,
    ssh_command, start_pod, watch_leased, with_pod_logs,
};
use crate::runs::{
    Launch, Outcome, RunCtx, RunRecord, RunState, Runs, STOP_LIMITS, SnapshotReason,
    artifacts_missing, cancel as cancel_job, collect, create, request_snapshot, reserve, start,
    watch, with_stop_fallback,
};
use crate::train::sizing::{HF_URL, VramFloor, estimate_model};
use crate::train::{Axolotl, reasoning_template_warning};

/// What every Runpod command sets up: the API client, the front end's bus and
/// the interrupted flag provisioning checks between its steps.
struct Session<'f> {
    client: RunpodClient,
    runs: Runs,
    guard: BusGuard,
    timing: Timing,
    flag: Flag,
    front: &'f Frontend,
}

impl<'f> Session<'f> {
    /// Opens the API client and run store, then the front end's interrupted flag
    /// and bus.
    ///
    /// # Errors
    ///
    /// Returns an error when the Runpod API key cannot be resolved, the client
    /// cannot be built, or the interrupted flag cannot be set up.
    async fn open(
        project_dir: &Path,
        settings: &Settings,
        front: &'f Frontend,
    ) -> anyhow::Result<Self> {
        let client = super::pod::client(settings).await?;
        // Set up now, so an interruption from here on is seen by provisioning.
        let flag = front.provisioning_flag()?;
        let guard = front.open_bus();
        Ok(Self {
            client,
            runs: Runs::new(project_dir),
            guard,
            timing: Timing::standard(),
            flag,
            front,
        })
    }

    fn ctx(&self) -> PodCtx<'_> {
        PodCtx {
            client: &self.client,
            runs: &self.runs,
            bus: &self.guard.bus,
            timing: &self.timing,
            interrupted: &self.flag.interrupted,
        }
    }

    /// Stops watching for the interruption and lets the bus show what is left.
    async fn close(self) {
        self.flag.close();
        self.guard.close().await;
    }
}

/// Which Runpod target `overbrainer train` starts a run on, and how.
pub(super) struct RunpodStart<'a> {
    /// Name of the target in `overbrainer.toml`.
    pub(super) name: &'a str,
    /// The target, defaults applied.
    pub(super) spec: &'a RunpodTarget,
    /// `--keep-pod`.
    pub(super) keep: bool,
    /// The VRAM floor of `auto` GPU types, when the caller already estimated
    /// it; else it is estimated here (see [`vram_floor`]).
    pub(super) vram_floor: VramFloor,
    /// The trainer, resuming a stopped run or not.
    pub(super) trainer: &'a Axolotl<'a>,
}

/// `overbrainer train` on the Runpod target `start.name`.
///
/// # Errors
///
/// Returns an error when the pod cannot be provisioned, the run fails, or it is
/// interrupted.
pub(super) async fn train(
    project_dir: &Path,
    settings: &Settings,
    start: RunpodStart<'_>,
    front: &Frontend,
) -> anyhow::Result<()> {
    let RunpodStart {
        name,
        spec,
        keep,
        vram_floor: known,
        trainer,
    } = start;
    let training = training(settings)?;
    if let Some(warning) = reasoning_template_warning(training) {
        warn(&warning);
    }
    // Caught from before the preparation: Ctrl-C stops it without a run.
    let mut interrupt = front.interrupt();
    let (secrets, vram_floor_gb, session) = prepare(&mut interrupt, async {
        let secrets = secrets(settings).await?;
        let token = secrets
            .iter()
            .find(|(name, _)| name == HF_TOKEN)
            .map(|(_, token)| token);
        let floor = async {
            match known {
                VramFloor::Known(floor) => floor,
                VramFloor::ToEstimate => vram_floor(training, spec, token, HF_URL).await,
            }
        };
        let (vram_floor_gb, session) =
            tokio::join!(floor, Session::open(project_dir, settings, front));
        Ok((secrets, vram_floor_gb, session?))
    })
    .await?;
    let job = Job {
        session: &session,
        spec,
        trainer,
        vram_floor_gb,
        stopping: false,
    };
    let result = async {
        warn_orphans(&session.ctx()).await;
        let record = create(&session.runs, &settings.project.name, spec.workdir(), name)?;
        let record = resumed(&session.runs, record, trainer)?;
        started(&record);
        front.run_created(&record.id);
        job.run(&mut interrupt, record, keep, secrets).await
    }
    .await;
    session.close().await;
    result
}

/// How long the model's shape may take to read from Hugging Face.
const ESTIMATE_LIMIT: Duration = Duration::from_secs(10);

/// The least VRAM the `auto` GPU types of `spec` need: the estimate of what a
/// run of `training` needs per GPU, its model's shape read from the Hugging
/// Face Hub at `hub`, only when `spec` has `auto` GPU types and no
/// `min_vram_gb`. Without an estimate (the model's shape cannot be read, or
/// the run shards the model), a warning says so and `auto` picks as it would
/// without one.
async fn vram_floor(
    training: &Training,
    spec: &RunpodTarget,
    token: Option<&SecretString>,
    hub: &str,
) -> Option<u32> {
    if !spec.gpu_types.is_auto() || spec.min_vram_gb.is_some() {
        return None;
    }
    match estimate_model(training, token, hub, ESTIMATE_LIMIT).await {
        Ok(need) => {
            tracing::info!(
                "the run needs {need} of VRAM per GPU (estimate): gpu_types = \"auto\" keeps \
                 GPU types with at least {} GB",
                need.floor_gb()
            );
            Some(need.floor_gb())
        },
        Err(error) => {
            warn(&format!(
                "cannot estimate the VRAM the run needs ({error}): gpu_types = \"auto\" picks \
                 without a VRAM floor"
            ));
            None
        },
    }
}

/// A run's job on its pod.
struct Job<'a> {
    session: &'a Session<'a>,
    spec: &'a RunpodTarget,
    trainer: &'a Axolotl<'a>,
    /// See [`vram_floor`]; used only to start a pod.
    vram_floor_gb: Option<u32>,
    /// Whether a snapshot was asked of it: following it then cancels a job
    /// that gives none within [`STOP_LIMITS`].
    stopping: bool,
}

impl Job<'_> {
    fn run_ctx<'e>(&'e self, executor: &'e SshExecutor) -> RunCtx<'e, SshExecutor> {
        RunCtx {
            runs: &self.session.runs,
            executor,
            bus: &self.session.guard.bus,
            poll: POLL,
        }
    }

    /// Provisions the pod of the new run `record`, starts its job there and
    /// follows it. `pod.json` holds the pod's ID from its creation on, so before
    /// the run is saved `Running`.
    async fn run(
        &self,
        interrupt: &mut Interrupt,
        record: RunRecord,
        keep: bool,
        secrets: Vec<(String, secrecy::SecretString)>,
    ) -> anyhow::Result<()> {
        let ctx = self.session.ctx();
        let id = record.id.clone();
        // Provisioning checks the Ctrl-C flag between its steps and deletes its pod.
        let provisioned = interrupt
            .shield(start_pod(
                &ctx,
                self.spec,
                record.clone(),
                keep,
                self.vram_floor_gb,
            ))
            .await;
        let (mut pod, provisioned) = match provisioned {
            Ok(ready) => ready,
            Err(PodError::Interrupted) => bail!(interrupted_before_job(&self.session.runs, &id)),
            Err(error) => return Err(error.into()),
        };
        if interrupt.caught() || self.session.flag.interrupted.load(Ordering::SeqCst) {
            return abandon(&ctx, interrupt, &mut pod, &id).await;
        }
        let executor = provisioned.executor;
        if let Err(error) = arm_cost_cap(&executor, &pod, &record).await {
            return refuse_uncapped(&ctx, interrupt, &mut pod, record, error).await;
        }
        let runtime = self.spec.runtime();
        let launch = Launch {
            runtime: &runtime,
            secrets,
        };
        // Starting is never interrupted: dropped after the job is spawned and
        // before its record is saved, it would leave a job nothing finds again.
        // The run directory is claimed before anything is copied: on a network
        // volume, a run from another checkout could own it.
        let run_ctx = self.run_ctx(&executor);
        // The run's own value, the one its pod's bootstrap claimed the directory
        // with: the pod's public host key, unique to the run.
        let owner = pod.host_key.clone();
        let started_run = interrupt
            .shield(async {
                let mut record = record;
                reserve(&run_ctx, &mut record, &owner).await?;
                start(&run_ctx, self.trainer, launch, record).await
            })
            .await;
        let started_run = match started_run {
            Ok(run) => run,
            Err(error) => {
                // No job runs, so the pod has nothing left to do.
                release(&ctx, interrupt, &mut pod, DeleteReason::Requested).await;
                return Err(error.into());
            },
        };
        if keep {
            // Only now: until the job started, Ctrl-C or a failure deleted the pod.
            warn(&keep_warning(&pod, &id));
        }
        // The job runs on a billed pod: the error says how to reach or remove it.
        job_started(&self.session.runs, &mut pod).with_context(|| {
            format!(
                "the job of run {id} runs on pod {}{}; {}, or delete the pod with \
                 `overbrainer pod rm {id}`",
                pod_name(&pod),
                rate(&pod),
                reattach(&id)
            )
        })?;
        if interrupt.caught() {
            bail!(detached(&pod, &id, self.spec));
        }
        self.follow(interrupt, &executor, started_run, &mut pod)
            .await
    }

    /// Follows a started run, then ends its pod and reports the outcome. Ctrl-C
    /// detaches: the job and the pod keep running.
    async fn follow(
        &self,
        interrupt: &mut Interrupt,
        executor: &SshExecutor,
        record: RunRecord,
        pod: &mut PodRecord,
    ) -> anyhow::Result<()> {
        let ctx = self.session.ctx();
        let id = record.id.clone();
        let job = record.job.clone().filter(|_| self.stopping);
        // Only the watch is raced: it never changes the pod, so Ctrl-C drops
        // nothing half done (the capture of the pod's logs resumes from its
        // cursor). Acting on it (deleting the pod past the deadline)
        // is shielded.
        let run_ctx = self.run_ctx(executor);
        // Boxed: its state would otherwise weigh on every caller's future.
        let leased = Box::pin(watch_leased(&run_ctx, self.trainer, record, pod));
        let watch = async move {
            match job {
                Some(job) => with_stop_fallback(executor, &job, STOP_LIMITS, leased).await,
                None => leased.await,
            }
        };
        let watched = interrupt
            .race(with_pod_logs(&ctx, pod, Box::pin(watch)))
            .await;
        let Some(watched) = watched else {
            bail!(detached(pod, &id, self.spec));
        };
        let settled = interrupt
            .shield(settle_watch(&ctx, pod, &id, watched))
            .await;
        let outcome = match settled {
            Err(error @ PodError::Run(_)) => {
                return Err(anyhow::Error::new(error).context(reattach(&id)));
            },
            Err(error) => return Err(error.into()),
            Ok(outcome) => outcome,
        };
        self.end(interrupt, executor, pod, outcome).await
    }

    /// Ends the pod of a run whose job ended, then prints its outcome. The
    /// outcome is printed even when the pod could not be ended.
    async fn end(
        &self,
        interrupt: &mut Interrupt,
        executor: &SshExecutor,
        pod: &mut PodRecord,
        outcome: Outcome,
    ) -> anyhow::Result<()> {
        let ctx = self.session.ctx();
        let id = outcome.record.id.clone();
        let ending = interrupt
            .shield(end_pod(
                &ctx,
                pod,
                executor,
                &outcome.record,
                outcome.retrieved,
            ))
            .await;
        let finished = finish(
            &self.session.runs,
            &id,
            Ok(Some(outcome)),
            self.session.front,
        );
        ended(pod, ending, &id, finished)
    }

    /// `train attach` of the started run `record` on its pod.
    async fn attach(
        &self,
        interrupt: &mut Interrupt,
        record: RunRecord,
        pod: &mut PodRecord,
    ) -> anyhow::Result<()> {
        let ctx = self.session.ctx();
        let Some(executor) = reconnect(&ctx, pod, &record).await? else {
            return from_local_files(self.session, self.trainer, record, pod).await;
        };
        if record.state == RunState::Running {
            return self.follow(interrupt, &executor, record, pod).await;
        }
        let run_ctx = self.run_ctx(&executor);
        let (record, retrieved) = if pod.state == PodState::Kept && !artifacts_missing(&record) {
            (record, true)
        } else {
            interrupt
                .shield(collect(&run_ctx, self.trainer, record))
                .await?
        };
        let mut outcome = watch(&run_ctx, self.trainer, record).await?;
        outcome.retrieved = retrieved;
        self.end(interrupt, &executor, pod, outcome).await
    }

    /// `train stop` of the running run `record` on its pod: asks its job for a
    /// snapshot, then follows it until it ends, and ends the pod.
    async fn stop(
        &self,
        interrupt: &mut Interrupt,
        record: RunRecord,
        pod: &mut PodRecord,
    ) -> anyhow::Result<()> {
        let ctx = self.session.ctx();
        let id = record.id.clone();
        let Some(executor) = reconnect(&ctx, pod, &record).await? else {
            bail!("run {id} has no pod left: nothing to stop");
        };
        request_snapshot(&executor, &record, SnapshotReason::Requested).await?;
        self.session.front.line(&stop_requested(&id));
        self.follow(interrupt, &executor, record, pod).await
    }

    /// `train cancel` of the started run `record` on its pod: cancels the job,
    /// retrieves its results best effort, then ends the pod.
    async fn cancel(
        &self,
        interrupt: &mut Interrupt,
        record: RunRecord,
        pod: &mut PodRecord,
    ) -> anyhow::Result<()> {
        let ctx = self.session.ctx();
        let id = record.id.clone();
        let Some(executor) = reconnect(&ctx, pod, &record).await? else {
            bail!("run {id} has no pod left: nothing to cancel");
        };
        // Cancelling is never interrupted, like on any other target.
        let (record, status, retrieved) = interrupt
            .shield(cancel_job(
                &self.session.runs,
                &executor,
                self.trainer,
                record,
            ))
            .await?;
        let front = self.session.front;
        if status != JobStatus::Cancelled {
            front.line(&format!(
                "train: the job of run {id} had already ended ({}): collect it with `overbrainer train attach {id}`",
                super::progress::status_name(status)
            ));
            return Ok(());
        }
        let ending = interrupt
            .shield(end_pod(&ctx, pod, &executor, &record, retrieved))
            .await;
        match &record.message {
            Some(message) => front.line(&format!("train: run {id} cancelled ({message})")),
            None => front.line(&format!("train: run {id} cancelled")),
        }
        ended(pod, ending, &id, Ok(()))
    }
}

/// Reports what became of the pod of the run `id` once its job ended, then
/// returns `finished`, or the error of ending the pod, which matters more: the
/// pod may still bill.
fn ended(
    pod: &PodRecord,
    ending: Result<Ending, PodError>,
    id: &str,
    finished: anyhow::Result<()>,
) -> anyhow::Result<()> {
    match ending {
        Ok(ending) => {
            report(pod, ending, id);
            finished
        },
        Err(error) => {
            if let Err(run_error) = finished {
                warn(&format!("{run_error:#}"));
            }
            Err(anyhow::Error::new(error).context(format!("run {id} ended, but not its pod")))
        },
    }
}

/// Deletes the pod of a run interrupted before its job started, and fails the run.
async fn abandon(
    ctx: &PodCtx<'_>,
    interrupt: &mut Interrupt,
    pod: &mut PodRecord,
    id: &str,
) -> anyhow::Result<()> {
    release(ctx, interrupt, pod, DeleteReason::Interrupted).await;
    let mut run = ctx.runs.load(id)?;
    run.state = RunState::Failed;
    run.message = Some(PodError::Interrupted.to_string());
    ctx.runs.save(&run)?;
    bail!(interrupted_before_job(ctx.runs, id))
}

/// A cost cap could not be applied to the pod of `run`: deletes the pod, fails
/// the run with `error` and returns it, since nothing would stop the run at
/// its cap.
async fn refuse_uncapped(
    ctx: &PodCtx<'_>,
    interrupt: &mut Interrupt,
    pod: &mut PodRecord,
    mut run: RunRecord,
    error: PodError,
) -> anyhow::Result<()> {
    release(ctx, interrupt, pod, DeleteReason::CostCapUnset).await;
    run.state = RunState::Failed;
    run.message = Some(error.to_string());
    ctx.runs.save(&run)?;
    Err(error.into())
}

/// Deletes the pod of a run that will not run its job, shielded from Ctrl-C; a
/// failure is only warned about, with where to look.
async fn release(
    ctx: &PodCtx<'_>,
    interrupt: &mut Interrupt,
    pod: &mut PodRecord,
    reason: DeleteReason,
) {
    let removed = interrupt
        .shield(remove(ctx, pod, reason, DeletedBy::Client))
        .await;
    match removed {
        Ok(()) => forget_client_key(ctx.runs, &pod.run_id),
        Err(error) => warn(&format!(
            "{}; remove the pod with `overbrainer pod rm {}`",
            chain(&error),
            pod.run_id
        )),
    }
}

/// The error of a run interrupted before its job started, saying whether a pod
/// of it may be left, from its `pod.json`.
fn interrupted_before_job(runs: &Runs, id: &str) -> String {
    let left = match PodRecord::load(runs, id) {
        Ok(Some(pod)) => {
            (pod.pod_id.is_some() && pod.state != PodState::Deleted) || !pod.stray_pods.is_empty()
        },
        Ok(None) => false,
        Err(_) => true,
    };
    if left {
        format!(
            "interrupted: run {id} stopped before its job started, but a pod of it may be left: \
             check `overbrainer pod ls`"
        )
    } else {
        format!("interrupted: run {id} stopped before its job started; it has no pod left")
    }
}

/// Warns about the pods nothing will delete, from one list of the account's
/// pods, best effort. Never deletes anything.
async fn warn_orphans(ctx: &PodCtx<'_>) {
    match listed_rows(ctx).await {
        Ok(rows) => {
            for warning in orphan_warnings(&rows) {
                warn(&warning);
            }
        },
        Err(error) => warn(&format!("cannot look for leftover pods: {}", chain(&error))),
    }
}

fn rate(pod: &PodRecord) -> String {
    pod.cost_per_hour
        .map_or_else(String::new, |rate| format!(" (${rate:.2}/h)"))
}

fn pod_name(pod: &PodRecord) -> String {
    pod.pod_id
        .as_ref()
        .map_or_else(|| "(none)".to_string(), ToString::to_string)
}

fn keep_warning(pod: &PodRecord, id: &str) -> String {
    let rate = pod.cost_per_hour.map_or_else(
        || "at an hourly rate Runpod did not give".to_string(),
        |rate| format!("at ${rate:.2}/h"),
    );
    format!(
        "--keep-pod: pod {} is kept with no time limit, {rate}, even once the run ends; \
         nothing deletes it but `overbrainer pod rm {id}`",
        pod_name(pod)
    )
}

/// The message of a run left running on its pod after Ctrl-C.
fn detached(pod: &PodRecord, id: &str, spec: &RunpodTarget) -> String {
    let guard = if pod.keep {
        format!(
            "The pod is kept with no time limit (--keep-pod): `{}`.",
            ssh_command(id)
        )
    } else {
        format!(
            "The watchdog deletes the pod {} min after the job ends if its results are not retrieved, and at {} or {} min from now (once the lease this command held runs out), whichever is later.",
            spec.retrieve_grace.as_secs() / 60,
            pod.deadline.as_deref().unwrap_or("its deadline"),
            LEASE_TTL.as_secs() / 60
        )
    };
    format!(
        "interrupted: run {id} keeps running on pod {}{}; {}, or delete the pod with \
         `overbrainer pod rm {id}`. {guard}",
        pod_name(pod),
        rate(pod),
        reattach(id)
    )
}

/// Says what became of the pod once the job ended: see [`report_line`].
fn report(pod: &PodRecord, ending: Ending, id: &str) {
    if let Some(line) = report_line(pod, ending, id) {
        warn(&line);
    }
}

/// What to say of the pod once the job ended. Nothing for a deleted pod, which
/// its events already reported, nor for one awaiting retrieval, which `end_pod`
/// reported: how to reach a kept pod, its results retrieved or not, is said here.
fn report_line(pod: &PodRecord, ending: Ending, id: &str) -> Option<String> {
    match ending {
        Ending::Deleted | Ending::AwaitingRetrieval => None,
        Ending::Kept => Some(format!(
            "pod {} kept (--keep-pod): `{}`; remove it with `overbrainer pod rm {id}`; \
             nothing deletes it automatically",
            pod_name(pod),
            ssh_command(id)
        )),
    }
}

/// `overbrainer train attach` on a Runpod run.
///
/// # Errors
///
/// Returns an error when the run has no job, the pod or the run cannot be
/// followed, or the run failed.
pub(super) async fn attach(
    project_dir: &Path,
    settings: &Settings,
    record: RunRecord,
    mut pod: PodRecord,
    front: &Frontend,
) -> anyhow::Result<()> {
    let training = training(settings)?;
    let spec = target_of(settings, &record)?;
    if record.job.is_none() {
        bail!(
            "run {} has no job yet: its pod is still starting, or it stopped before its job started",
            record.id
        );
    }
    let session = Session::open(project_dir, settings, front).await?;
    let mut interrupt = front.interrupt();
    let trainer = Axolotl::new(training, &DataFiles::new(project_dir));
    let job = Job {
        session: &session,
        spec: &spec,
        trainer: &trainer,
        vram_floor_gb: None,
        stopping: false,
    };
    let result = job.attach(&mut interrupt, record, &mut pod).await;
    session.close().await;
    result
}

/// Reports a run whose pod is gone from its local files; a run still `Running`
/// is failed first, since its job went with the pod.
async fn from_local_files(
    session: &Session<'_>,
    trainer: &Axolotl<'_>,
    mut record: RunRecord,
    pod: &PodRecord,
) -> anyhow::Result<()> {
    tracing::info!("pod: {} already deleted", pod_name(pod));
    if record.state == RunState::Running {
        record.state = RunState::Failed;
        // Its watchdog deleted it on `max_hours` or `max_cost_usd` once one
        // passed.
        record.message = Some(limit_reached(pod, SystemTime::now()).map_or_else(
            || format!("pod {} no longer exists", pod_name(pod)),
            str::to_string,
        ));
        session.runs.save(&record)?;
    }
    let id = record.id.clone();
    // Never contacted: a run not `Running` is summarized from its local files.
    let nowhere = LocalExecutor::new(&session.runs.run_dir(&id)?)?;
    let run_ctx = RunCtx {
        runs: &session.runs,
        executor: &nowhere,
        bus: &session.guard.bus,
        poll: POLL,
    };
    let outcome = watch(&run_ctx, trainer, record).await?;
    finish(&session.runs, &id, Ok(Some(outcome)), session.front)
}

/// `overbrainer train cancel` on a Runpod run.
///
/// # Errors
///
/// Returns an error when the pod is gone or cannot be reached, or the cancel
/// fails.
pub(super) async fn cancel(
    project_dir: &Path,
    settings: &Settings,
    record: RunRecord,
    mut pod: PodRecord,
    front: &Frontend,
) -> anyhow::Result<()> {
    let training = training(settings)?;
    let spec = target_of(settings, &record)?;
    let session = Session::open(project_dir, settings, front).await?;
    let mut interrupt = front.interrupt();
    let trainer = Axolotl::new(training, &DataFiles::new(project_dir));
    let job = Job {
        session: &session,
        spec: &spec,
        trainer: &trainer,
        vram_floor_gb: None,
        stopping: false,
    };
    let result = job.cancel(&mut interrupt, record, &mut pod).await;
    session.close().await;
    result
}

/// `overbrainer train stop` on a Runpod run.
///
/// # Errors
///
/// Returns an error when the run is not running, its pod is gone or cannot be
/// reached, the request cannot be written, or the run cannot be followed.
pub(super) async fn stop(
    project_dir: &Path,
    settings: &Settings,
    record: RunRecord,
    mut pod: PodRecord,
    front: &Frontend,
) -> anyhow::Result<()> {
    let training = training(settings)?;
    let spec = target_of(settings, &record)?;
    stoppable(&record)?;
    let session = Session::open(project_dir, settings, front).await?;
    let mut interrupt = front.interrupt();
    let trainer = Axolotl::new(training, &DataFiles::new(project_dir));
    let job = Job {
        session: &session,
        spec: &spec,
        trainer: &trainer,
        vram_floor_gb: None,
        stopping: true,
    };
    let result = job.stop(&mut interrupt, record, &mut pod).await;
    session.close().await;
    result
}

/// The Runpod target a run was started on.
fn target_of(settings: &Settings, record: &RunRecord) -> anyhow::Result<RunpodTarget> {
    settings
        .targets
        .get(&record.target)
        .and_then(RunpodTarget::from_target)
        .with_context(|| {
            format!(
                "target `{}` of run {} is no longer a runpod target in overbrainer.toml",
                record.target, record.id
            )
        })
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime};

    use super::*;
    use crate::config::ListOrAuto;
    use crate::runpod::{AttemptResult, Pod};

    /// The floor asks Hugging Face only for `auto` GPU types without
    /// `min_vram_gb`.
    #[tokio::test]
    async fn the_vram_floor_is_estimated_only_when_auto_uses_it()
    -> Result<(), Box<dyn std::error::Error>> {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let hub = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/Qwen/Qwen3-4B"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(
                    serde_json::json!({"safetensors": {"total": 4_022_468_096_u64}}),
                ),
            )
            .mount(&hub)
            .await;
        Mock::given(method("GET"))
            .and(path("/Qwen/Qwen3-4B/resolve/main/config.json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "hidden_size": 2560, "num_hidden_layers": 36, "vocab_size": 151_936
            })))
            .mount(&hub)
            .await;
        let training: Training = serde_json::from_value(serde_json::json!({
            "target": "cloud", "base_model": "Qwen/Qwen3-4B", "adapter": "lora"
        }))?;
        let listed = target();
        let mut with_min = target();
        with_min.gpu_types = ListOrAuto::Auto;
        with_min.min_vram_gb = Some(24);
        for spec in [&listed, &with_min] {
            assert_eq!(vram_floor(&training, spec, None, &hub.uri()).await, None);
        }
        let asked = hub.received_requests().await.unwrap_or_default();
        assert!(asked.is_empty(), "{asked:?}");
        let mut auto = target();
        auto.gpu_types = ListOrAuto::Auto;
        assert_eq!(
            vram_floor(&training, &auto, None, &hub.uri()).await,
            Some(21)
        );
        Ok(())
    }

    fn target() -> RunpodTarget {
        RunpodTarget {
            gpu_types: ListOrAuto::List(vec!["NVIDIA A40".into()]),
            min_vram_gb: None,
            max_price_per_hour: None,
            gpu_count: 1,
            image: "img".into(),
            venv: "/venv".into(),
            container_disk_gb: 50,
            max_hours: 6.0,
            max_cost_usd: None,
            boot_grace: Duration::from_secs(1800),
            retrieve_grace: Duration::from_secs(3600),
            data_center_ids: ListOrAuto::default(),
            network_volume_id: None,
        }
    }

    fn pod(keep: bool) -> Result<PodRecord, serde_json::Error> {
        let mut pod = PodRecord::new("r1", keep, 1, "ssh-ed25519 AAAAhost");
        let remote: Pod = serde_json::from_value(
            serde_json::json!({"id": "k3x9abc", "status": "RUNNING", "cost": 0.53}),
        )?;
        pod.begin_attempt("NVIDIA A40", SystemTime::UNIX_EPOCH, 6.0);
        pod.created(&remote, AttemptResult::Created, SystemTime::UNIX_EPOCH);
        Ok(pod)
    }

    #[test]
    fn keep_pod_warns_of_no_time_limit_with_the_rate_and_pod_rm() -> Result<(), serde_json::Error> {
        assert_eq!(
            keep_warning(&pod(true)?, "r1"),
            "--keep-pod: pod k3x9abc is kept with no time limit, at $0.53/h, even once the run \
             ends; nothing deletes it but `overbrainer pod rm r1`"
        );
        Ok(())
    }

    #[test]
    fn a_detached_run_names_attach_pod_rm_and_what_bounds_its_pod() -> Result<(), serde_json::Error>
    {
        let guarded = pod(false)?;
        let deadline = guarded.deadline.clone().unwrap_or_default();
        assert_eq!(
            detached(&guarded, "r1", &target()),
            format!(
                "interrupted: run r1 keeps running on pod k3x9abc ($0.53/h); follow it again \
                 with `overbrainer train attach r1`, or delete the pod with `overbrainer pod rm \
                 r1`. The watchdog deletes the pod 60 min after the job ends if its results are \
                 not retrieved, and at {deadline} or 15 min from now (once the lease this command held \
                 runs out), whichever is later."
            )
        );
        assert!(detached(&pod(true)?, "r1", &target()).ends_with(
            "The pod is kept with no time limit (--keep-pod): \
             `ssh -F runs/r1/ssh/config overbrainer-r1`."
        ));
        Ok(())
    }

    #[test]
    fn a_kept_pod_is_reported_with_its_ssh_command() -> Result<(), serde_json::Error> {
        let kept = "pod k3x9abc kept (--keep-pod): `ssh -F runs/r1/ssh/config overbrainer-r1`; \
                    remove it with `overbrainer pod rm r1`; nothing deletes it automatically";
        let (keep, guarded) = (pod(true)?, pod(false)?);
        assert_eq!(
            report_line(&keep, Ending::Kept, "r1").as_deref(),
            Some(kept)
        );
        assert_eq!(report_line(&guarded, Ending::AwaitingRetrieval, "r1"), None);
        assert_eq!(report_line(&guarded, Ending::Deleted, "r1"), None);
        Ok(())
    }
}
