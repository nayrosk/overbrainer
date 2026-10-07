//! `train`, `train attach`, `train stop` and `train cancel` on a Runpod target: the pod is
//! created before the run's job and deleted once its results are retrieved.
//!
//! Ctrl-C before the job exists deletes the pod and fails the run. Once the job is
//! started, Ctrl-C only stops following it: the job and the pod keep running, and
//! the pod's watchdog bounds the cost.

use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, SystemTime};

use anyhow::{Context, anyhow, bail};
use secrecy::SecretString;

use super::export::{Delivery, EXPORT_PREFIX, Plan, fail, finish_export, started_line};
use super::front::{BusGuard, Flag, Frontend, Interrupt};
use super::train::{
    HF_TOKEN, POLL, exporting, finish, prepare, push_after_training, resumed, secrets, started,
    stop_requested, stoppable, training, warn,
};
use crate::compare::{COMPARE_PREFIX, ModelSource, discard_model};
use crate::config::{Settings, SshClient, Training, process_client};
use crate::dataset::DataFiles;
use crate::exec::{JobStatus, LocalExecutor, SshExecutor};
use crate::export::ExportJob;
use crate::runpod::export_room_warning;
use crate::runpod::{
    DeleteReason, DeletedBy, Ending, LEASE_TTL, PodCtx, PodError, PodRecord, PodState,
    RunpodClient, RunpodTarget, Timing, arm_cost_cap, chain, end_pod, forget_keys, job_started,
    limit_reached, listed_rows, orphan_warnings, reconnect, remove, settle_watch, ssh_command,
    start_pod, sweep_host_keys, watch_leased, with_pod_logs,
};
use crate::runs::{
    Launch, Outcome, REQUEST_POLL, RunCtx, RunRecord, RunState, Runs, STOP_LIMITS, SnapshotReason,
    artifacts_missing, cancel as cancel_job, collect, create, request_snapshot, reserve, start,
    watch, with_request_watch, with_stop_fallback,
};
use crate::train::sizing::{
    HF_URL, VramFloor, estimate_model, export_disk_bytes, export_estimate, fetch_shape,
};
use crate::train::{Axolotl, CONFIG_FILE, Trainer, reasoning_template_warning};

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
        Self::open_in(Runs::new(project_dir), settings, front).await
    }

    /// [`Session::open`] with the runs of `runs`: an export's, whose pod and
    /// record live in its own directory.
    ///
    /// # Errors
    ///
    /// As [`Session::open`].
    async fn open_in(runs: Runs, settings: &Settings, front: &'f Frontend) -> anyhow::Result<Self> {
        let client = super::pod::client(settings).await?;
        // Set up now, so an interruption from here on is seen by provisioning.
        let flag = front.provisioning_flag()?;
        let guard = front.open_bus();
        Ok(Self {
            client,
            runs,
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
        let disk = warn_export_disk(training, &settings.export, spec, token, HF_URL);
        let (vram_floor_gb, (), session) =
            tokio::join!(floor, disk, Session::open(project_dir, settings, front));
        Ok((secrets, vram_floor_gb, session?))
    })
    .await?;
    let ollama = settings.export.ollama_name.as_deref();
    let report =
        |runs: &Runs, id: &str, result, front: &Frontend| finish(runs, id, result, front, ollama);
    // The run's ID, once it was created, beside the job's result.
    let result = async {
        warn_orphans(&session.ctx()).await;
        let record = create(&session.runs, &settings.project.name, spec.workdir(), name)?;
        let record = resumed(&session.runs, record, trainer)?;
        let trainer = exporting(trainer, &record.id, &settings.export);
        started(&record);
        front.run_created(&record.id);
        let id = record.id.clone();
        let job = Job {
            session: &session,
            spec,
            trainer: &trainer,
            vram_floor_gb,
            stopping: false,
            report: &report,
            cancel_on_interrupt: false,
        };
        anyhow::Ok((id, job.run(&mut interrupt, record, keep, secrets).await))
    }
    .await;
    session.close().await;
    let (id, result) = result?;
    // After the pod ended and the report: the GGUF of an export is in place.
    if result.is_ok() {
        push_after_training(&Runs::new(project_dir), settings, &id, front).await;
    }
    result
}

/// `overbrainer export` of `plan` on the Runpod target `spec`: a new pod with
/// the target's GPU settings, the run's model and `axolotl.yaml` uploaded with
/// the export job, its GGUF retrieved, then the pod deleted (kept with
/// `keep`), under the same lease, deadline and cost cap as a training job.
/// The export's record and pod live in `runs/<run-id>/exports/<export-id>/`.
/// Ctrl-C cancels the export and ends its pod.
///
/// # Errors
///
/// Returns an error when the pod cannot be provisioned, the export fails or
/// cannot be delivered, or it is interrupted.
pub(super) async fn export(
    project_dir: &Path,
    settings: &Settings,
    (spec, keep): (&RunpodTarget, bool),
    plan: &Plan<'_>,
    front: &Frontend,
) -> anyhow::Result<()> {
    let run = plan.run;
    let runs = Runs::new(project_dir);
    let exports = runs.exports(&run.id)?;
    let local_run = runs.run_dir(&run.id)?;
    // Caught from before the preparation: Ctrl-C stops it without an export.
    let mut interrupt = front.interrupt();
    let (secrets, vram_floor_gb, session) = prepare(&mut interrupt, async {
        let secrets = secrets(settings).await?;
        let token = secrets
            .iter()
            .find(|(name, _)| name == HF_TOKEN)
            .map(|(_, token)| token);
        let sizing = export_sizing(&local_run, plan, spec, token, HF_URL);
        let (floor, session) =
            tokio::join!(sizing, Session::open_in(exports.clone(), settings, front));
        Ok((secrets, floor, session?))
    })
    .await?;
    let trainer = ExportJob::staged(&run.id, plan.quantize, &local_run, plan.model);
    let file = trainer.file_name();
    let report =
        |exports: &Runs, id: &str, result: anyhow::Result<Option<Outcome>>, front: &Frontend| {
            let outcome = result?.with_context(|| format!("interrupted: export {id} stopped"))?;
            let delivery = Delivery {
                runs: &runs,
                exports,
                plan,
                file: &file,
            };
            finish_export(&delivery, &outcome, front)
        };
    let result = async {
        let record = create(&session.runs, EXPORT_PREFIX, spec.workdir(), &run.target)?;
        front.line(&started_line(&record.id, plan));
        let job = Job {
            session: &session,
            spec,
            trainer: &trainer,
            vram_floor_gb,
            stopping: false,
            report: &report,
            cancel_on_interrupt: true,
        };
        job.run(&mut interrupt, record, keep, secrets).await
    }
    .await;
    session.close().await;
    result
}

/// `overbrainer compare` of `plan` on the Runpod target `spec`: a new pod
/// with the target's GPU settings and the compare image (`[compare] image`,
/// else [`DEFAULT_COMPARE_IMAGE`](crate::config::DEFAULT_COMPARE_IMAGE), its
/// job run by the image's `python3`), the compare job uploaded with the run's
/// GGUF and the questions, its answers retrieved, then the pod deleted (kept
/// with `keep`), under the same lease, deadline and cost cap as a training
/// job. The pod's price and the questions are said before it starts. The
/// compare's record and pod live in `runs/<run-id>/compares/<compare-id>/`.
/// Ctrl-C cancels the job and ends its pod. Returns the compares of the run
/// and the compare's ID once its answers are back.
///
/// # Errors
///
/// Returns an error when the pod cannot be provisioned, the job fails, or it
/// is interrupted.
pub(super) async fn compare(
    project_dir: &Path,
    settings: &Settings,
    (spec, keep): (&RunpodTarget, bool),
    plan: &super::compare::Plan,
    front: &Frontend,
) -> anyhow::Result<(Runs, String)> {
    let run = &plan.run;
    let spec = &spec.for_compare(settings.compare.image.as_deref());
    let compares = Runs::new(project_dir).compares(&run.id)?;
    let trainer = super::compare::job_of(plan, settings, ModelSource::Upload(plan.gguf.clone()))?;
    let vram_floor_gb = super::compare::compare_vram_floor(spec, plan.export.size);
    // Caught from before the preparation: Ctrl-C stops it without a compare.
    let mut interrupt = front.interrupt();
    let session = prepare(&mut interrupt, async {
        let session = Session::open_in(compares.clone(), settings, front).await?;
        let price = super::compare::pod_price(&session.client, spec, vram_floor_gb).await;
        tracing::info!(
            "compare: {} questions on a new Runpod pod {}",
            plan.questions.len(),
            price.words()
        );
        Ok(session)
    })
    .await?;
    let report =
        |compares: &Runs, id: &str, result: anyhow::Result<Option<Outcome>>, _front: &Frontend| {
            let outcome = result?.with_context(|| format!("interrupted: compare {id} stopped"))?;
            super::compare::generated(compares, &outcome)
        };
    let result = async {
        let record = create(&session.runs, COMPARE_PREFIX, spec.workdir(), &run.target)?;
        let id = record.id.clone();
        let dir = session.runs.run_dir(&id)?;
        if let Err(error) = super::compare::setup_of(plan, &id, settings.pipeline.seed).save(&dir) {
            let error = anyhow::Error::from(error);
            fail(&session.runs, &id, &error);
            return Err(error);
        }
        tracing::info!("compare: {id}: {}", session.runs.relative_dir(&id));
        let job = Job {
            session: &session,
            spec,
            trainer: &trainer,
            vram_floor_gb,
            stopping: false,
            report: &report,
            cancel_on_interrupt: true,
        };
        let ran = job.run(&mut interrupt, record, keep, Vec::new()).await;
        // The GGUF linked into the job directory for the upload is of no use now.
        if let Err(error) = discard_model(&dir) {
            warn(&format!("cannot clean the compare {id}: {error}"));
        }
        ran.map(|()| id)
    }
    .await;
    session.close().await;
    Ok((compares, result?))
}

/// The VRAM floor of `auto` GPU types for a compare of a GGUF of `bytes`:
/// the file and its cache, a fifth more, plus 2 GB, in whole GB.
pub(super) fn gguf_vram_floor(bytes: u64) -> u32 {
    let gb = bytes.saturating_mul(6).div_ceil(5).div_ceil(1_000_000_000);
    u32::try_from(gb).unwrap_or(u32::MAX).saturating_add(2)
}

/// What an export of `plan` needs of a pod of `spec`, from its base model's
/// shape on the Hub at `hub`: the VRAM floor of `auto` GPU types without
/// `min_vram_gb` (the merge holds the base model), and, without a network
/// volume, a warning when the container disk may be too small. Without the
/// shape, a warning says so and there is no floor.
async fn export_sizing(
    run_dir: &Path,
    plan: &Plan<'_>,
    spec: &RunpodTarget,
    token: Option<&SecretString>,
    hub: &str,
) -> Option<u32> {
    let wants_floor = spec.gpu_types.is_auto() && spec.min_vram_gb.is_none();
    let wants_disk = spec.network_volume_id.is_none();
    if !wants_floor && !wants_disk {
        return None;
    }
    let base_model = recorded_base_model(run_dir)?;
    match fetch_shape(hub, &base_model, token, ESTIMATE_LIMIT).await {
        Ok(shape) => {
            let model = run_dir.join(plan.model);
            let merge = model.join("adapter_config.json").is_file()
                && !model.join("merged/config.json").is_file();
            let need = export_disk_bytes(shape.params, merge, plan.quantize);
            if let Some(warning) =
                export_room_warning(spec.container_disk_gb, need).filter(|_| wants_disk)
            {
                warn(&warning);
            }
            wants_floor.then(|| export_estimate(&shape).floor_gb())
        },
        Err(error) => {
            warn(&format!(
                "cannot estimate what the export needs ({error}): gpu_types = \"auto\" picks \
                 without a VRAM floor"
            ));
            None
        },
    }
}

/// The `base_model` the run in `run_dir` trained, from its `axolotl.yaml`.
fn recorded_base_model(run_dir: &Path) -> Option<String> {
    let config = std::fs::read_to_string(run_dir.join(CONFIG_FILE)).ok()?;
    crate::train::top_level_scalar(&config, "base_model")
}

/// Warns, without a network volume, when the container disk of `spec` may be
/// too small for the export at the end of a training job of `training`, its
/// model's shape read from the Hub at `hub`. Silent when the shape cannot be
/// read: the VRAM floor says so already.
async fn warn_export_disk(
    training: &Training,
    export: &crate::config::Export,
    spec: &RunpodTarget,
    token: Option<&SecretString>,
    hub: &str,
) {
    if !export.after_training || spec.network_volume_id.is_some() {
        return;
    }
    let Ok(shape) = fetch_shape(hub, &training.base_model, token, ESTIMATE_LIMIT).await else {
        return;
    };
    let merge = training.adapter != crate::config::Adapter::Full;
    let need = export_disk_bytes(shape.params, merge, &export.quantize);
    if let Some(warning) = export_room_warning(spec.container_disk_gb, need) {
        warn(&warning);
    }
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

/// How a Runpod command reports a job that ended, from the runs it was
/// saved in and its ID: `finish` for a training run.
type Report<'a> = &'a (
        dyn Fn(&Runs, &str, anyhow::Result<Option<Outcome>>, &Frontend) -> anyhow::Result<()> + Sync
    );

/// A run's job on its pod: a training job, or an export.
struct Job<'a, T> {
    session: &'a Session<'a>,
    spec: &'a RunpodTarget,
    trainer: &'a T,
    /// See [`vram_floor`]; used only to start a pod.
    vram_floor_gb: Option<u32>,
    /// Whether a snapshot was asked of it: following it then cancels a job
    /// that gives none within [`STOP_LIMITS`].
    stopping: bool,
    /// Reports the job once it ended.
    report: Report<'a>,
    /// Whether Ctrl-C cancels the job and ends its pod (an export), rather
    /// than leaving both running (a training job).
    cancel_on_interrupt: bool,
}

impl<T: Trainer + Sync> Job<'_, T> {
    /// The SSH client of the target, `OVERBRAINER_SSH_CLIENT` applied.
    fn client(&self) -> anyhow::Result<SshClient> {
        process_client(self.spec.ssh_client).map_err(|message| anyhow!(message))
    }

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
            warn(&keep_warning(&pod, &self.session.runs, &id));
        }
        // The job runs on a billed pod: the error says how to reach or remove it.
        job_started(&self.session.runs, &mut pod).with_context(|| {
            format!(
                "the job of {} runs on pod {}{}; {}, or delete the pod with \
                 `overbrainer pod rm {id}`",
                self.session.runs.subject(&id),
                pod_name(&pod),
                rate(&pod),
                self.session.runs.follow_hint(&id)
            )
        })?;
        if interrupt.caught() {
            if self.cancel_on_interrupt {
                return self
                    .abort(interrupt, &executor, started_run, &mut pod)
                    .await;
            }
            bail!(detached(&pod, &id, self.spec, &self.session.runs));
        }
        self.follow(interrupt, &executor, started_run, &mut pod)
            .await
    }

    /// Cancels the job of `record` after Ctrl-C, retrieves what it left
    /// best effort, then ends its pod.
    async fn abort(
        &self,
        interrupt: &mut Interrupt,
        executor: &SshExecutor,
        record: RunRecord,
        pod: &mut PodRecord,
    ) -> anyhow::Result<()> {
        let ctx = self.session.ctx();
        let id = record.id.clone();
        let (record, _, retrieved) = interrupt
            .shield(cancel_job(
                &self.session.runs,
                executor,
                self.trainer,
                record,
            ))
            .await?;
        let ending = interrupt
            .shield(end_pod(&ctx, pod, executor, &record, retrieved))
            .await;
        ended(
            pod,
            ending,
            (&self.session.runs, &id),
            Err(anyhow!(
                "interrupted: {} cancelled",
                self.session.runs.subject(&id)
            )),
        )
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
        let run = record.clone();
        let interrupted = record.clone();
        let stopping = self.stopping;
        // Only the watch is raced: it never changes the pod, so Ctrl-C drops
        // nothing half done (the capture of the pod's logs resumes from its
        // cursor). Acting on it (deleting the pod past the deadline)
        // is shielded.
        let run_ctx = self.run_ctx(executor);
        // Boxed: its state would otherwise weigh on every caller's future.
        let leased = Box::pin(watch_leased(&ctx, &run_ctx, self.trainer, record, pod));
        let watch = async move {
            match &run.job {
                Some(job) if stopping => {
                    with_stop_fallback(executor, job, STOP_LIMITS, leased).await
                },
                // A request written by another process, `train stop` or the
                // pod's watchdog, is held to the same limits from when it is
                // seen; never for a run whose job cannot save a snapshot.
                _ => with_request_watch(executor, &run, STOP_LIMITS, REQUEST_POLL, leased).await,
            }
        };
        let watched = interrupt
            .race(with_pod_logs(&ctx, pod, Box::pin(watch)))
            .await;
        let Some(watched) = watched else {
            if self.cancel_on_interrupt {
                return self.abort(interrupt, executor, interrupted, pod).await;
            }
            bail!(detached(pod, &id, self.spec, &self.session.runs));
        };
        let settled = interrupt
            .shield(settle_watch(&ctx, pod, &id, watched))
            .await;
        let outcome = match settled {
            Err(error @ PodError::Run(_)) => {
                return Err(anyhow::Error::new(error).context(self.session.runs.follow_hint(&id)));
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
        let finished = (self.report)(
            &self.session.runs,
            &id,
            Ok(Some(outcome)),
            self.session.front,
        );
        ended(pod, ending, (&self.session.runs, &id), finished)
    }

    /// `train attach` of the started run `record` on its pod.
    async fn attach(
        &self,
        interrupt: &mut Interrupt,
        record: RunRecord,
        pod: &mut PodRecord,
    ) -> anyhow::Result<()> {
        let ctx = self.session.ctx();
        let Some(executor) = reconnect(&ctx, pod, &record, self.client()?).await? else {
            return from_local_files(self.session, self.trainer, record, pod, self.report).await;
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
        let Some(executor) = reconnect(&ctx, pod, &record, self.client()?).await? else {
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
        let Some(executor) = reconnect(&ctx, pod, &record, self.client()?).await? else {
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
        ended(pod, ending, (&self.session.runs, &id), Ok(()))
    }
}

/// Reports what became of the pod of the run `id` once its job ended, then
/// returns `finished`, or the error of ending the pod, which matters more: the
/// pod may still bill.
fn ended(
    pod: &PodRecord,
    ending: Result<Ending, PodError>,
    (runs, id): (&Runs, &str),
    finished: anyhow::Result<()>,
) -> anyhow::Result<()> {
    match ending {
        Ok(ending) => {
            report(pod, ending, runs, id);
            finished
        },
        Err(error) => {
            if let Err(run_error) = finished {
                warn(&format!("{run_error:#}"));
            }
            Err(anyhow::Error::new(error)
                .context(format!("{} ended, but not its pod", runs.subject(id))))
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
        Ok(()) => forget_keys(ctx, &pod.run_id).await,
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
    let subject = runs.subject(id);
    if left {
        format!(
            "interrupted: {subject} stopped before its job started, but a pod of it may be left: \
             check `overbrainer pod ls`"
        )
    } else {
        format!("interrupted: {subject} stopped before its job started; it has no pod left")
    }
}

/// Warns about the pods nothing will delete, from one list of the account's
/// pods, best effort, and sweeps the run host key secrets no pod needs any
/// more (see [`sweep_host_keys`]). Never deletes a pod.
async fn warn_orphans(ctx: &PodCtx<'_>) {
    let rows = match listed_rows(ctx).await {
        Ok(rows) => rows,
        Err(error) => {
            warn(&format!("cannot look for leftover pods: {}", chain(&error)));
            return;
        },
    };
    for warning in orphan_warnings(&rows) {
        warn(&warning);
    }
    match sweep_host_keys(ctx, &rows).await {
        Ok(warnings) => {
            for warning in warnings {
                warn(&warning);
            }
        },
        Err(error) => warn(&format!(
            "cannot look for leftover host key secrets: {}",
            chain(&error)
        )),
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

/// The warning of `--keep-pod` for the pod of `id`, a run or a job (an export
/// or a compare) of `runs`: nothing deletes the pod once it ends.
fn keep_warning(pod: &PodRecord, runs: &Runs, id: &str) -> String {
    let rate = pod.cost_per_hour.map_or_else(
        || "at an hourly rate Runpod did not give".to_string(),
        |rate| format!("at ${rate:.2}/h"),
    );
    let ends = runs.job_of().map_or_else(
        || "the run".to_string(),
        |(_, kind)| format!("the {}", kind.noun()),
    );
    format!(
        "--keep-pod: pod {} is kept with no time limit, {rate}, even once {ends} ends; \
         nothing deletes it but `overbrainer pod rm {id}`",
        pod_name(pod)
    )
}

/// The message of a run left running on its pod after Ctrl-C.
fn detached(pod: &PodRecord, id: &str, spec: &RunpodTarget, runs: &Runs) -> String {
    let guard = if pod.keep {
        format!(
            "The pod is kept with no time limit (--keep-pod): `{}`.",
            ssh_command(runs, id)
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
        "interrupted: {} keeps running on pod {}{}; {}, or delete the pod with \
         `overbrainer pod rm {id}`. {guard}",
        runs.subject(id),
        pod_name(pod),
        rate(pod),
        runs.follow_hint(id)
    )
}

/// Says what became of the pod once the job ended: see [`report_line`].
fn report(pod: &PodRecord, ending: Ending, runs: &Runs, id: &str) {
    if let Some(line) = report_line(pod, ending, runs, id) {
        warn(&line);
    }
}

/// What to say of the pod once the job ended. Nothing for a deleted pod, which
/// its events already reported, nor for one awaiting retrieval, which `end_pod`
/// reported: how to reach a kept pod, its results retrieved or not, is said here.
fn report_line(pod: &PodRecord, ending: Ending, runs: &Runs, id: &str) -> Option<String> {
    match ending {
        Ending::Deleted | Ending::AwaitingRetrieval => None,
        Ending::Kept => Some(format!(
            "pod {} kept (--keep-pod): `{}`; remove it with `overbrainer pod rm {id}`; \
             nothing deletes it automatically",
            pod_name(pod),
            ssh_command(runs, id)
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
    let ollama = settings.export.ollama_name.as_deref();
    let report =
        |runs: &Runs, id: &str, result, front: &Frontend| finish(runs, id, result, front, ollama);
    let job = Job {
        session: &session,
        spec: &spec,
        trainer: &trainer,
        vram_floor_gb: None,
        stopping: false,
        report: &report,
        cancel_on_interrupt: false,
    };
    let id = record.id.clone();
    let result = job.attach(&mut interrupt, record, &mut pod).await;
    session.close().await;
    if result.is_ok() {
        push_after_training(&Runs::new(project_dir), settings, &id, front).await;
    }
    result
}

/// Reports a run whose pod is gone from its local files; a run still `Running`
/// is failed first, since its job went with the pod.
async fn from_local_files<T: Trainer>(
    session: &Session<'_>,
    trainer: &T,
    mut record: RunRecord,
    pod: &PodRecord,
    report: Report<'_>,
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
    report(&session.runs, &id, Ok(Some(outcome)), session.front)
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
    let ollama = settings.export.ollama_name.as_deref();
    let report =
        |runs: &Runs, id: &str, result, front: &Frontend| finish(runs, id, result, front, ollama);
    let job = Job {
        session: &session,
        spec: &spec,
        trainer: &trainer,
        vram_floor_gb: None,
        stopping: false,
        report: &report,
        cancel_on_interrupt: false,
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
    let ollama = settings.export.ollama_name.as_deref();
    let report =
        |runs: &Runs, id: &str, result, front: &Frontend| finish(runs, id, result, front, ollama);
    let job = Job {
        session: &session,
        spec: &spec,
        trainer: &trainer,
        vram_floor_gb: None,
        stopping: true,
        report: &report,
        cancel_on_interrupt: false,
    };
    let id = record.id.clone();
    let result = job.stop(&mut interrupt, record, &mut pod).await;
    session.close().await;
    if result.is_ok() {
        push_after_training(&Runs::new(project_dir), settings, &id, front).await;
    }
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
    #[test]
    fn the_base_model_is_read_from_the_run_s_own_config() -> Result<(), Box<dyn std::error::Error>>
    {
        let run = tempfile::tempdir()?;
        assert_eq!(recorded_base_model(run.path()), None, "no axolotl.yaml");
        std::fs::write(
            run.path().join(CONFIG_FILE),
            "datasets:\n  - base_model: nested\nbase_model: 'Qwen/Qwen3-0.6B'  # edited\n",
        )?;
        assert_eq!(
            recorded_base_model(run.path()).as_deref(),
            Some("Qwen/Qwen3-0.6B")
        );
        Ok(())
    }

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

    /// A Runpod target with an A40 and no limits.
    fn target() -> RunpodTarget {
        RunpodTarget {
            gpu_types: ListOrAuto::List(vec!["NVIDIA A40".into()]),
            min_vram_gb: None,
            max_price_per_hour: None,
            gpu_count: 1,
            image: "img".into(),
            venv: Some("/venv".into()),
            container_disk_gb: 50,
            max_hours: 6.0,
            max_cost_usd: None,
            boot_grace: Duration::from_mins(30),
            retrieve_grace: Duration::from_secs(3600),
            data_center_ids: ListOrAuto::default(),
            network_volume_id: None,
            max_volume_gb: None,
            ssh_client: crate::config::SshClient::Openssh,
        }
    }

    /// The runs of a project in `project`, relative paths only.
    fn runs() -> Runs {
        Runs::new(Path::new("project"))
    }

    #[test]
    fn an_export_s_messages_name_the_export_and_never_train_attach()
    -> Result<(), Box<dyn std::error::Error>> {
        let exports = runs().exports("r1")?;
        let kept = pod(true)?;
        let warning = keep_warning(&kept, &exports, "e1");
        assert!(warning.contains("even once the export ends"), "{warning}");
        assert!(warning.ends_with("`overbrainer pod rm e1`"), "{warning}");
        assert_eq!(
            report_line(&kept, Ending::Kept, &exports, "e1").as_deref(),
            Some(
                "pod k3x9abc kept (--keep-pod): `ssh -F runs/r1/exports/e1/ssh/config \
                 overbrainer-e1`; remove it with `overbrainer pod rm e1`; nothing deletes it \
                 automatically"
            )
        );
        let detached = detached(&kept, "e1", &target(), &exports);
        assert!(
            detached.starts_with("interrupted: export e1 of run r1 keeps running on pod k3x9abc"),
            "{detached}"
        );
        assert!(detached.contains("`overbrainer export r1`"), "{detached}");
        let interrupted = interrupted_before_job(&exports, "e1");
        assert_eq!(
            interrupted,
            "interrupted: export e1 of run r1 stopped before its job started; it has no pod left"
        );
        for message in [warning, detached, interrupted] {
            assert!(!message.contains("train attach"), "{message}");
        }
        Ok(())
    }

    /// A compare's messages name the compare and how to run it again.
    #[test]
    fn a_compare_s_messages_name_the_compare_and_never_train_attach()
    -> Result<(), Box<dyn std::error::Error>> {
        let compares = runs().compares("r1")?;
        let kept = pod(true)?;
        let warning = keep_warning(&kept, &compares, "c1");
        assert!(warning.contains("even once the compare ends"), "{warning}");
        let detached = detached(&kept, "c1", &target(), &compares);
        assert!(
            detached.starts_with("interrupted: compare c1 of run r1 keeps running on pod k3x9abc"),
            "{detached}"
        );
        assert!(
            detached.contains("`overbrainer compare --run r1`"),
            "{detached}"
        );
        assert!(!detached.contains("train attach"), "{detached}");
        Ok(())
    }

    /// The GGUF and its cache must fit: a fifth more than the file, plus 2 GB.
    #[test]
    fn the_vram_floor_of_a_compare_follows_the_gguf() {
        assert_eq!(gguf_vram_floor(0), 2);
        assert_eq!(gguf_vram_floor(2_500_000_000), 5);
        assert_eq!(gguf_vram_floor(4_000_000_001), 7);
    }

    /// The record of pod k3x9abc, an A40 at $0.53/h, kept when `keep`.
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
            keep_warning(&pod(true)?, &runs(), "r1"),
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
            detached(&guarded, "r1", &target(), &runs()),
            format!(
                "interrupted: run r1 keeps running on pod k3x9abc ($0.53/h); follow it again \
                 with `overbrainer train attach r1`, or delete the pod with `overbrainer pod rm \
                 r1`. The watchdog deletes the pod 60 min after the job ends if its results are \
                 not retrieved, and at {deadline} or 15 min from now (once the lease this command held \
                 runs out), whichever is later."
            )
        );
        assert!(detached(&pod(true)?, "r1", &target(), &runs()).ends_with(
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
            report_line(&keep, Ending::Kept, &runs(), "r1").as_deref(),
            Some(kept)
        );
        assert_eq!(
            report_line(&guarded, Ending::AwaitingRetrieval, &runs(), "r1"),
            None
        );
        assert_eq!(report_line(&guarded, Ending::Deleted, &runs(), "r1"), None);
        Ok(())
    }
}
