//! `train`, `train attach` and `train cancel` on a Runpod target: the pod is
//! created before the run's job and deleted once its results are retrieved.
//!
//! Ctrl-C before the job exists deletes the pod and fails the run. Once the job is
//! started, Ctrl-C only stops following it: the job and the pod keep running, and
//! the pod's watchdog bounds the cost.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, bail};
use tokio::signal::unix::{SignalKind, signal};
use tokio::task::JoinHandle;

use super::train::{Interrupt, POLL, finish, reattach, secrets, started, training, warn};
use crate::config::Settings;
use crate::dataset::DataFiles;
use crate::events::EventBus;
use crate::exec::{JobStatus, LocalExecutor, SshExecutor};
use crate::runpod::{
    DeleteReason, DeletedBy, Ending, PodCtx, PodError, PodRecord, PodState, RunpodClient,
    RunpodTarget, Timing, chain, end_pod, follow, forget_client_key, job_started, listed_rows,
    orphan_warnings, reconnect, remove, ssh_command, start_pod,
};
use crate::runs::{
    Launch, Outcome, RunCtx, RunRecord, RunState, Runs, artifacts_missing, cancel as cancel_job,
    collect, create, start, watch,
};
use crate::train::{Axolotl, reasoning_template_warning};

/// What every Runpod command sets up: the API client, the event rendering and
/// the Ctrl-C flag provisioning checks between its steps.
struct Session {
    client: RunpodClient,
    runs: Runs,
    bus: EventBus,
    timing: Timing,
    interrupted: Arc<AtomicBool>,
    watcher: JoinHandle<()>,
    renderer: JoinHandle<()>,
}

impl Session {
    async fn open(project_dir: &Path, settings: &Settings) -> anyhow::Result<Self> {
        let client = super::pod::client(settings).await?;
        // Registered now, so a Ctrl-C from here on is seen by provisioning.
        let mut sigint = signal(SignalKind::interrupt()).context("cannot catch Ctrl-C")?;
        let interrupted = Arc::new(AtomicBool::new(false));
        let seen = Arc::clone(&interrupted);
        let watcher = tokio::spawn(async move {
            if sigint.recv().await.is_some() {
                seen.store(true, Ordering::SeqCst);
            }
        });
        let bus = EventBus::new();
        let renderer = tokio::spawn(super::progress::render(bus.subscribe()));
        Ok(Self {
            client,
            runs: Runs::new(project_dir),
            bus,
            timing: Timing::standard(),
            interrupted,
            watcher,
            renderer,
        })
    }

    fn ctx(&self) -> PodCtx<'_> {
        PodCtx {
            client: &self.client,
            runs: &self.runs,
            bus: &self.bus,
            timing: &self.timing,
            interrupted: &self.interrupted,
        }
    }

    /// Stops watching Ctrl-C and lets the renderer print what is left.
    async fn close(self) {
        self.watcher.abort();
        drop(self.bus);
        self.renderer.await.ok();
    }
}

/// `overbrainer train` on the Runpod target `name`.
///
/// # Errors
///
/// Returns an error when the pod cannot be provisioned, the run fails, or it is
/// interrupted.
pub(super) async fn train(
    project_dir: &Path,
    settings: &Settings,
    name: &str,
    spec: &RunpodTarget,
    keep: bool,
) -> anyhow::Result<()> {
    let training = training(settings)?;
    if let Some(warning) = reasoning_template_warning(training) {
        warn(&warning);
    }
    let secrets = secrets(settings).await?;
    let session = Session::open(project_dir, settings).await?;
    let mut interrupt = Interrupt::catch();
    let trainer = Axolotl::new(training, &DataFiles::new(project_dir));
    let job = Job {
        session: &session,
        spec,
        trainer: &trainer,
    };
    let result = async {
        warn_orphans(&session.ctx()).await;
        let record = create(&session.runs, spec.workdir(), name)?;
        started(&record);
        job.run(&mut interrupt, record, keep, secrets).await
    }
    .await;
    session.close().await;
    result
}

/// A run's job on its pod.
struct Job<'a> {
    session: &'a Session,
    spec: &'a RunpodTarget,
    trainer: &'a Axolotl<'a>,
}

impl Job<'_> {
    fn run_ctx<'e>(&'e self, executor: &'e SshExecutor) -> RunCtx<'e, SshExecutor> {
        RunCtx {
            runs: &self.session.runs,
            executor,
            bus: &self.session.bus,
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
            .shield(start_pod(&ctx, self.spec, record.clone(), keep))
            .await;
        let (mut pod, provisioned) = match provisioned {
            Ok(ready) => ready,
            Err(PodError::Interrupted) => bail!(interrupted_before_job(&self.session.runs, &id)),
            Err(error) => return Err(error.into()),
        };
        if keep {
            warn(&keep_warning(&pod, &id));
        }
        if interrupt.caught() || self.session.interrupted.load(Ordering::SeqCst) {
            return abandon(&ctx, interrupt, &mut pod, &id).await;
        }
        let executor = provisioned.executor;
        let runtime = self.spec.runtime();
        let launch = Launch {
            runtime: &runtime,
            secrets,
        };
        // Starting is never interrupted: dropped after the job is spawned and
        // before its record is saved, it would leave a job nothing finds again.
        let started_run = interrupt
            .shield(start(
                &self.run_ctx(&executor),
                self.trainer,
                launch,
                record,
            ))
            .await;
        let started_run = match started_run {
            Ok(run) => run,
            Err(error) => {
                // No job runs, so the pod has nothing left to do.
                release(&ctx, interrupt, &mut pod, DeleteReason::Requested).await;
                return Err(error.into());
            },
        };
        job_started(&self.session.runs, &mut pod)?;
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
        let followed = interrupt
            .race(follow(
                &ctx,
                &self.run_ctx(executor),
                self.trainer,
                record,
                pod,
            ))
            .await;
        let outcome = match followed {
            None => bail!(detached(pod, &id, self.spec)),
            Some(Err(error @ PodError::Run(_))) => {
                return Err(anyhow::Error::new(error).context(reattach(&id)));
            },
            Some(Err(error)) => return Err(error.into()),
            Some(Ok(outcome)) => outcome,
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
        let finished = finish(&self.session.runs, &id, Ok(Some(outcome)));
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
        if status != JobStatus::Cancelled {
            println!(
                "train: the job of run {id} had already ended ({}): collect it with `overbrainer train attach {id}`",
                super::progress::status_name(status)
            );
            return Ok(());
        }
        let ending = interrupt
            .shield(end_pod(&ctx, pod, &executor, &record, retrieved))
            .await;
        match &record.message {
            Some(message) => println!("train: run {id} cancelled ({message})"),
            None => println!("train: run {id} cancelled"),
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
        "The pod is kept with no time limit (--keep-pod).".to_string()
    } else {
        format!(
            "The watchdog deletes the pod {} min after the job ends if its results are not retrieved, and by {} in any case.",
            spec.retrieve_grace.as_secs() / 60,
            pod.deadline.as_deref().unwrap_or("its deadline")
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

/// Says what became of the pod once the job ended. A deleted pod was already
/// reported by its events, and one awaiting retrieval by `end_pod`.
fn report(pod: &PodRecord, ending: Ending, id: &str) {
    match ending {
        Ending::Deleted | Ending::AwaitingRetrieval => {},
        Ending::Kept => warn(&format!(
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
) -> anyhow::Result<()> {
    let training = training(settings)?;
    let spec = target_of(settings, &record)?;
    if record.job.is_none() {
        bail!(
            "run {} has no job yet: its pod is still starting, or it stopped before its job started",
            record.id
        );
    }
    let session = Session::open(project_dir, settings).await?;
    let mut interrupt = Interrupt::catch();
    let trainer = Axolotl::new(training, &DataFiles::new(project_dir));
    let job = Job {
        session: &session,
        spec: &spec,
        trainer: &trainer,
    };
    let result = job.attach(&mut interrupt, record, &mut pod).await;
    session.close().await;
    result
}

/// Reports a run whose pod is gone from its local files; a run still `Running`
/// is failed first, since its job went with the pod.
async fn from_local_files(
    session: &Session,
    trainer: &Axolotl<'_>,
    mut record: RunRecord,
    pod: &PodRecord,
) -> anyhow::Result<()> {
    tracing::info!("pod: {} already deleted", pod_name(pod));
    if record.state == RunState::Running {
        record.state = RunState::Failed;
        record.message = Some(format!("pod {} no longer exists", pod_name(pod)));
        session.runs.save(&record)?;
    }
    let id = record.id.clone();
    // Never contacted: a run not `Running` is summarized from its local files.
    let nowhere = LocalExecutor::new(&session.runs.run_dir(&id)?)?;
    let run_ctx = RunCtx {
        runs: &session.runs,
        executor: &nowhere,
        bus: &session.bus,
        poll: POLL,
    };
    let outcome = watch(&run_ctx, trainer, record).await?;
    finish(&session.runs, &id, Ok(Some(outcome)))
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
) -> anyhow::Result<()> {
    let training = training(settings)?;
    let spec = target_of(settings, &record)?;
    let session = Session::open(project_dir, settings).await?;
    let mut interrupt = Interrupt::catch();
    let trainer = Axolotl::new(training, &DataFiles::new(project_dir));
    let job = Job {
        session: &session,
        spec: &spec,
        trainer: &trainer,
    };
    let result = job.cancel(&mut interrupt, record, &mut pod).await;
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
    use crate::runpod::{AttemptResult, Pod};

    fn target() -> RunpodTarget {
        RunpodTarget {
            gpu_types: vec!["NVIDIA A40".into()],
            gpu_count: 1,
            image: "img".into(),
            venv: "/venv".into(),
            container_disk_gb: 50,
            max_hours: 6.0,
            boot_grace: Duration::from_secs(1800),
            retrieve_grace: Duration::from_secs(3600),
            data_center_ids: Vec::new(),
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
                 not retrieved, and by {deadline} in any case."
            )
        );
        assert!(
            detached(&pod(true)?, "r1", &target())
                .ends_with("The pod is kept with no time limit (--keep-pod).")
        );
        Ok(())
    }
}
