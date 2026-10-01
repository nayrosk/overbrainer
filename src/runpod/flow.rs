//! The steps of a Runpod run around the generic run flows: provisioning its pod,
//! following its job under a client-side deadline, and ending the pod once the
//! results are retrieved. The CLI drives them and decides what Ctrl-C does.

use std::fs;
use std::io;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use secrecy::SecretString;
use tokio::sync::broadcast::Receiver;
use tokio::sync::broadcast::error::RecvError;

use crate::events::Event;
use crate::exec::{ExecError, Executor, SshExecutor};
use crate::runs::{Outcome, RunCtx, RunError, RunRecord, RunState, Runs, SnapshotReason, watch};
use crate::train::{Pace, SNAPSHOT_REQUEST, Trainer};

use super::disk::{DiskWatch, VolumeDisk};
use super::provision::note_strays;
use super::{
    BOOTSTRAP_LOG, CLIENT_KEY, CostCap, DeleteReason, DeletedBy, Pod, PodCtx, PodError, PodId,
    PodKeys, PodPlan, PodRecord, PodState, PodStatus, Provisioned, RemoteStatus, RunpodTarget,
    SSH_DIR, SshEndpoint, VOLUME_MOUNT, VOLUME_WORKDIR, alias, keep_file, provision, remove, sweep,
    with_pod_logs, write_config,
};
use crate::secrets::Redactor;

/// How long after the watchdog's deadline the client deletes the pod itself, when
/// it is still there: the watchdog should have done it first.
pub const DEADLINE_MARGIN: Duration = Duration::from_secs(5 * 60);

/// Where the watchdog looks for the client's "retrieved" marker, in a run
/// directory on the pod.
pub const RETRIEVED_MARKER: &str = ".pod/retrieved";

/// Why a run fails when its pod was deleted on `max_hours`, by the client's own
/// guard or by the pod's watchdog.
pub const MAX_HOURS_REACHED: &str = "max_hours reached: the pod was deleted before the job ended";

/// Why a run fails when its pod was deleted on `max_cost_usd` by its watchdog.
pub const MAX_COST_REACHED: &str = "max_cost_usd reached: the pod was deleted before the job ended";

/// Where the client writes, on the pod, when the job is stopped with a snapshot
/// for the cost cap (Unix seconds), in the run directory: read by the watchdog.
pub const SNAPSHOT_AT_FILE: &str = ".pod/snapshot_at";

/// Where the client writes, on the pod, when the watchdog deletes the pod for
/// the cost cap (Unix seconds), in the run directory.
pub const COST_CAP_FILE: &str = ".pod/cost_cap_at";

/// The lease a client following a job keeps on its pod, in the run directory:
/// the watchdog skips its `max_hours` deadline while the file was touched less
/// than [`LEASE_TTL`] ago.
pub const LEASE_FILE: &str = ".pod/lease";

/// How long a renewed lease holds the deadline off; the watchdog's
/// `OVERBRAINER_LEASE_TTL` default.
pub const LEASE_TTL: Duration = Duration::from_secs(15 * 60);

/// How often the following client renews the lease.
const LEASE_RENEW: Duration = Duration::from_secs(5 * 60);

/// A job with no new metric for this long counts as stalled: its lease is no
/// longer renewed, so `max_hours` applies again.
const STALL: Duration = Duration::from_secs(30 * 60);

/// How often the lease holder looks at the time.
const LEASE_TICK: Duration = Duration::from_secs(60);

/// Looks in a row that must answer Runpod's 404 before a pod is declared gone.
const GONE_LOOKS: u32 = 3;

/// Creates the pod of the new run `run` (from `runs::create`) and waits until it is
/// ready: keys in `runs/<id>/ssh/`, then `pod.json`, then provisioning. When that
/// fails, the run is saved `Failed` with the reason (`interrupted before the job
/// started` after Ctrl-C), whatever pod was created is deleted, and once every
/// pod of the run is confirmed deleted, the run's private client key is removed.
/// `vram_floor_gb` is the least VRAM `auto` GPU types need when the target sets
/// no `min_vram_gb` (see [`PodPlan::vram_floor_gb`]).
///
/// # Errors
///
/// Returns the [`PodError`] of the step that failed.
pub async fn start_pod(
    ctx: &PodCtx<'_>,
    target: &RunpodTarget,
    mut run: RunRecord,
    keep: bool,
    vram_floor_gb: Option<u32>,
) -> Result<(PodRecord, Provisioned), PodError> {
    let result = provision_run(ctx, target, &run, keep, vram_floor_gb).await;
    if let Err(error) = &result {
        run.state = RunState::Failed;
        run.message = Some(run_message(error));
        if let Err(save_error) = ctx.runs.save(&run) {
            tracing::warn!("cannot record run {} as failed: {save_error}", run.id);
        }
        if no_pod_left(ctx.runs, &run.id) {
            forget_client_key(ctx.runs, &run.id);
        }
    }
    result
}

/// Whether `pod.json` shows no pod of the run that may still exist: no pod, or
/// one confirmed deleted, and no stray. A `pod.json` that cannot be read shows
/// nothing for sure; none at all means no pod was ever asked for.
fn no_pod_left(runs: &Runs, run_id: &str) -> bool {
    match PodRecord::load(runs, run_id) {
        Ok(Some(record)) => {
            record.stray_pods.is_empty()
                && (record.pod_id.is_none() || record.state == PodState::Deleted)
        },
        Ok(None) => true,
        Err(error) => {
            tracing::warn!("cannot read the pod record of run {run_id}: {error}");
            false
        },
    }
}

/// The reason a failed provisioning gives in `run.json`: the error's message,
/// except that an answer of the Runpod API is reduced to its status. A
/// non-create call's error carries Runpod's own text (cleaned of the account
/// key), which the log shows but `run.json` never keeps.
fn run_message(error: &PodError) -> String {
    match error {
        PodError::Api(api) => match api.status() {
            Some(status) => format!("Runpod answered {status}"),
            None => error.to_string(),
        },
        _ => error.to_string(),
    }
}

/// Generates the run's SSH keys, saves its initial `pod.json`, then provisions
/// a reachable pod. The record exists before the first create request, so every
/// attempt can be tracked even when Runpod's answer is ambiguous.
async fn provision_run(
    ctx: &PodCtx<'_>,
    target: &RunpodTarget,
    run: &RunRecord,
    keep: bool,
    vram_floor_gb: Option<u32>,
) -> Result<(PodRecord, Provisioned), PodError> {
    let ssh_dir = ctx.runs.run_dir(&run.id)?.join(SSH_DIR);
    let keys = PodKeys::generate(&ssh_dir, &alias(&run.id))?;
    let mut record = PodRecord::new(&run.id, keep, target.gpu_count, &keys.host_public);
    record.max_cost_usd = target.max_cost_usd.filter(|_| !keep);
    record
        .network_volume_id
        .clone_from(&target.network_volume_id);
    record.max_volume_gb = target.max_volume_gb;
    record.save(ctx.runs)?;
    let plan = PodPlan {
        run_id: &run.id,
        target,
        keys: &keys,
        ssh_dir: &ssh_dir,
        workdir: target.workdir(),
        api_url: ctx.client.base_url(),
        vram_floor_gb,
        volume_gb: None,
    };
    let provisioned = provision(ctx, &plan, &mut record).await?;
    Ok((record, provisioned))
}

/// Hands the cost cap of `pod` to its watchdog before the job of `run` starts:
/// when to stop the job with a snapshot and when to delete the pod, in files
/// the watchdog reads every minute. Nothing to do without a cap, or for a kept
/// pod.
///
/// # Errors
///
/// Returns [`PodError::CostCapUnset`] when a cap is set but cannot be applied:
/// Runpod gave no hourly rate for the pod, or the files cannot be written. No
/// job may start then, since nothing would stop it at its cap.
pub async fn arm_cost_cap<E: Executor>(
    executor: &E,
    pod: &PodRecord,
    run: &RunRecord,
) -> Result<(), PodError> {
    match pod.cost_cap() {
        Some(cap) => write_cost_cap(executor, cap, &run.remote_dir)
            .await
            .map_err(|error| {
                PodError::CostCapUnset(format!("cannot write it on the pod: {error}"))
            }),
        None if pod.max_cost_usd.is_some() && !pod.keep => Err(PodError::CostCapUnset(
            "Runpod gave no hourly rate for the pod".to_string(),
        )),
        None => Ok(()),
    }
}

/// Writes the times of `cap` where the watchdog of the run in `remote_dir`
/// reads them.
async fn write_cost_cap<E: Executor>(
    executor: &E,
    cap: CostCap,
    remote_dir: &str,
) -> Result<(), ExecError> {
    for (file, at) in [
        (SNAPSHOT_AT_FILE, cap.snapshot_at),
        (COST_CAP_FILE, cap.delete_at),
    ] {
        executor
            .put_file(&format!("{remote_dir}/{file}"), &at.to_string())
            .await?;
    }
    Ok(())
}

/// Records that the job of the run was started on the pod.
///
/// # Errors
///
/// Returns [`PodError::Runs`] when `pod.json` cannot be saved.
pub fn job_started(runs: &Runs, pod: &mut PodRecord) -> Result<(), PodError> {
    pod.state = PodState::Running;
    pod.save(runs)?;
    Ok(())
}

/// Follows the started run `record` with `runs::watch`, until its job ends, with
/// the client's own guard: at the watchdog's deadline plus [`DEADLINE_MARGIN`], a
/// pod still there is deleted and the run saved `Failed` (never for a kept pod).
/// That is [`watch_on_pod`], the pod's logs kept meanwhile ([`with_pod_logs`]),
/// then [`settle_watch`].
///
/// # Errors
///
/// Returns [`PodError::DeadlineReached`] after that guard fired, or when the
/// target stays unreachable because the watchdog deleted the pod at its
/// deadline, [`PodError::PodGone`] when it stays unreachable because the pod no
/// longer exists for another reason (the run is saved `Failed` in both cases),
/// and [`PodError::Run`] for any other failure of the watch (the job may keep
/// running: attach again).
pub async fn follow<E: Executor, T: Trainer>(
    ctx: &PodCtx<'_>,
    run_ctx: &RunCtx<'_, E>,
    trainer: &T,
    record: RunRecord,
    pod: &mut PodRecord,
) -> Result<Outcome, PodError> {
    let id = record.id.clone();
    let watched = with_pod_logs(ctx, pod, watch_on_pod(ctx, run_ctx, trainer, record, pod)).await;
    settle_watch(ctx, pod, &id, watched).await
}

/// How [`watch_on_pod`] ended.
#[derive(Debug)]
pub enum Watched {
    /// The watch ended first, with its outcome or its error.
    Ended(Box<Result<Outcome, RunError>>),
    /// The client's deadline passed first; nothing was deleted yet.
    DeadlinePassed,
    /// The pod spent its cost cap first; nothing was deleted yet.
    CostCapPassed,
}

/// Watches the started run `record` until its job ends or the client's deadline
/// (the watchdog's plus [`DEADLINE_MARGIN`]; none for a kept pod) passes, and
/// warns once when the job's pace cannot end it before the watchdog's deadline.
/// The [disk policy](DiskWatch) runs beside it. It never deletes anything, so
/// dropping it (on Ctrl-C) loses nothing: what it found is acted on by
/// [`settle_watch`].
pub async fn watch_on_pod<E: Executor, T: Trainer>(
    ctx: &PodCtx<'_>,
    run_ctx: &RunCtx<'_, E>,
    trainer: &T,
    record: RunRecord,
    pod: &PodRecord,
) -> Watched {
    let run = record.clone();
    let disk = Box::pin(disk_watch(ctx, run_ctx.executor, &run, pod).run(run_ctx.bus.subscribe()));
    let Some(wait) = until_deadline(pod) else {
        return tokio::select! {
            outcome = watch(run_ctx, trainer, record) => Watched::Ended(Box::new(outcome)),
            never = disk => match never {},
        };
    };
    let events = run_ctx.bus.subscribe();
    tokio::select! {
        outcome = watch(run_ctx, trainer, record) => Watched::Ended(Box::new(outcome)),
        () = tokio::time::sleep(wait) => Watched::DeadlinePassed,
        // Never ends: it only warns.
        () = warn_overrun(events, pod) => Watched::DeadlinePassed,
        never = disk => match never {},
    }
}

/// The disk policy of the run `run` on `pod`.
fn disk_watch<'a, E: Executor>(
    ctx: &'a PodCtx<'_>,
    executor: &'a E,
    run: &'a RunRecord,
    pod: &'a PodRecord,
) -> DiskWatch<'a, E> {
    let volume = pod.network_volume_id.as_deref().map(|id| VolumeDisk {
        id,
        mount: VOLUME_MOUNT,
        max_gb: pod.max_volume_gb,
    });
    let watch = DiskWatch::new(executor, ctx.client, run, volume);
    // A run on a network volume whose ID `pod.json` does not hold (written
    // before overbrainer kept it) cannot be measured: `df` there is the whole
    // shared cluster.
    if volume.is_none() && run.remote_dir.starts_with(VOLUME_WORKDIR) {
        return watch.idle();
    }
    watch
}

/// [`watch_on_pod`] for a client that stays with the job: while the job keeps
/// making progress (a metric in the last 30 minutes), it renews the pod's lease
/// ([`LEASE_FILE`]) every few minutes, so `max_hours` deletes nothing, neither
/// the watchdog nor the client's own guard. Once the job stalls or the lease
/// cannot be renewed, the lease runs out after [`LEASE_TTL`] and the deadline
/// applies again. It warns once, as information, when the job ends after the
/// deadline. Once the pod has spent [`SNAPSHOT_SHARE`](super::SNAPSHOT_SHARE) of
/// its cost cap, it asks the job for a snapshot, as the watchdog does. A kept
/// pod has no deadline and needs no lease. The [disk policy](DiskWatch) runs
/// beside it, kept pod or not.
pub async fn watch_leased<E: Executor, T: Trainer>(
    ctx: &PodCtx<'_>,
    run_ctx: &RunCtx<'_, E>,
    trainer: &T,
    record: RunRecord,
    pod: &PodRecord,
) -> Watched {
    let run = record.clone();
    // Boxed: its state would otherwise weigh on every caller's future.
    let disk = Box::pin(disk_watch(ctx, run_ctx.executor, &run, pod).run(run_ctx.bus.subscribe()));
    if until_deadline(pod).is_none() {
        return tokio::select! {
            outcome = watch(run_ctx, trainer, record) => Watched::Ended(Box::new(outcome)),
            never = disk => match never {},
        };
    }
    let events = run_ctx.bus.subscribe();
    let remote_dir = record.remote_dir.clone();
    tokio::select! {
        outcome = watch(run_ctx, trainer, record) => Watched::Ended(Box::new(outcome)),
        // Boxed: its state would otherwise weigh on every caller's future.
        watched = Box::pin(hold_lease(run_ctx.executor, &remote_dir, events, pod)) => watched,
        never = disk => match never {},
    }
}

/// Renews the lease of the run in `remote_dir` while the job progresses, asks
/// for a snapshot at the cost cap, warns once when the job ends after the
/// deadline, and returns once the client's own deadline passed with no lease
/// held.
async fn hold_lease<E: Executor>(
    executor: &E,
    remote_dir: &str,
    mut events: Receiver<Event>,
    pod: &PodRecord,
) -> Watched {
    let mut lease = Lease::new(pod, remote_dir);
    let mut open = true;
    let mut tick = tokio::time::interval(LEASE_TICK);
    loop {
        tokio::select! {
            event = events.recv(), if open => match event {
                Ok(event) => lease.saw(&event),
                Err(RecvError::Lagged(_)) => {},
                Err(RecvError::Closed) => open = false,
            },
            _ = tick.tick() => {
                if let Some(guard) = lease.tick(executor).await {
                    return guard;
                }
            },
        }
    }
}

/// What [`hold_lease`] knows of the followed job, of its lease and of its cost
/// cap.
struct Lease<'a> {
    pod: &'a PodRecord,
    remote_dir: &'a str,
    pace: Pace,
    last_metric: SystemTime,
    renewed: Option<SystemTime>,
    warned: bool,
    /// Whether the snapshot for the cost cap was asked for.
    capped: bool,
}

impl<'a> Lease<'a> {
    fn new(pod: &'a PodRecord, remote_dir: &'a str) -> Self {
        Self {
            pod,
            remote_dir,
            pace: Pace::default(),
            last_metric: SystemTime::now(),
            renewed: None,
            warned: false,
            capped: false,
        }
    }

    /// Takes a training metric into account, and warns once when the job
    /// ends after the deadline.
    fn saw(&mut self, event: &Event) {
        let Event::Metric(metric) = event else {
            return;
        };
        self.pace.add(metric);
        self.last_metric = SystemTime::now();
        if self.warned {
            return;
        }
        if let Some(note) = overrun(&self.pace, self.pod, self.last_metric, true) {
            tracing::warn!("{note}");
            self.warned = true;
        }
    }

    /// Renews the lease when due and asks for the cost cap's snapshot once it
    /// is due; which of the client's guards fires, if one does: the cost cap
    /// (whatever the lease) or the deadline.
    async fn tick<E: Executor>(&mut self, executor: &E) -> Option<Watched> {
        let now = SystemTime::now();
        if renewal_due(self.renewed, self.last_metric, now) {
            self.renew(executor, now).await;
        }
        if !self.capped && cap_snapshot_due(self.pod, now) {
            self.cap(executor).await;
        }
        if past_cost_cap(self.pod, now) {
            return Some(Watched::CostCapPassed);
        }
        guard_fires(self.pod, self.renewed, now).then_some(Watched::DeadlinePassed)
    }

    /// Renews the lease at `now`.
    async fn renew<E: Executor>(&mut self, executor: &E, now: SystemTime) {
        let path = format!("{}/{LEASE_FILE}", self.remote_dir);
        match executor.put_file(&path, "").await {
            Ok(()) => self.renewed = Some(now),
            Err(error) => tracing::debug!("cannot renew the pod's lease: {error}"),
        }
    }

    /// Asks the job for the cost cap's snapshot; tried again next tick when
    /// the request cannot be written.
    async fn cap<E: Executor>(&mut self, executor: &E) {
        let path = format!("{}/{SNAPSHOT_REQUEST}", self.remote_dir);
        let written = executor.put_file(&path, SnapshotReason::Cost.name()).await;
        self.capped = written.is_ok();
        let note = match written {
            Ok(()) => format!(
                "the pod spent {:.0}% of max_cost_usd: its job is stopped with a snapshot",
                super::SNAPSHOT_SHARE * 100.0
            ),
            Err(error) => format!("cannot ask for the cost cap's snapshot: {error}"),
        };
        tracing::warn!("{note}");
    }
}

/// Whether the job on `pod` is due for its cost cap's snapshot at `now`.
fn cap_snapshot_due(pod: &PodRecord, now: SystemTime) -> bool {
    pod.cost_cap()
        .is_some_and(|cap| unix_secs(now) >= cap.snapshot_at)
}

/// Seconds since the Unix epoch at `now`, 0 before it.
fn unix_secs(now: SystemTime) -> u64 {
    now.duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

/// Whether the lease, last renewed at `renewed`, should be renewed at `now`:
/// once [`LEASE_RENEW`] passed, and only while the job made progress (its last
/// metric at `last_metric`) within [`STALL`].
fn renewal_due(renewed: Option<SystemTime>, last_metric: SystemTime, now: SystemTime) -> bool {
    let since = |at: SystemTime| now.duration_since(at).unwrap_or_default();
    since(last_metric) < STALL && renewed.is_none_or(|at| since(at) >= LEASE_RENEW)
}

/// Whether the client's own guard deletes `pod` at `now`: its deadline plus
/// [`DEADLINE_MARGIN`] passed, and no lease renewed within [`LEASE_TTL`].
fn guard_fires(pod: &PodRecord, renewed: Option<SystemTime>, now: SystemTime) -> bool {
    let held = renewed.is_some_and(|at| now.duration_since(at).unwrap_or_default() < LEASE_TTL);
    !held && until_deadline_at(pod, now).is_some_and(|left| left.is_zero())
}

/// Warns once, from the training metrics on the bus, when the job of `pod`
/// would need longer than its watchdog's deadline; then waits forever, so it
/// never ends the watch.
async fn warn_overrun(mut events: Receiver<Event>, pod: &PodRecord) {
    let mut pace = Pace::default();
    loop {
        match events.recv().await {
            Ok(Event::Metric(metric)) => {
                pace.add(&metric);
                if let Some(warning) = overrun(&pace, pod, SystemTime::now(), false) {
                    tracing::warn!("{warning}");
                    break;
                }
            },
            Ok(_) | Err(RecvError::Lagged(_)) => {},
            Err(RecvError::Closed) => break,
        }
    }
    std::future::pending::<()>().await;
}

/// The warning when, at `pace`, the job of `pod` still running at `now` would
/// end after its watchdog's deadline; `None` when it ends in time, for a kept
/// pod, or while the pace is unknown. With a `leased` pod the deadline only
/// applies once the client stops following the job.
fn overrun(pace: &Pace, pod: &PodRecord, now: SystemTime, leased: bool) -> Option<String> {
    if pod.keep {
        return None;
    }
    let deadline = pod.deadline_unix?;
    let left = pace.eta()?;
    let end = now.checked_add(left)?.duration_since(UNIX_EPOCH).ok()?;
    if end.as_secs() <= deadline {
        return None;
    }
    let hours = left.as_secs_f64() / 3600.0;
    let at = pod.deadline.as_deref().unwrap_or("its deadline");
    if leased {
        return Some(format!(
            "at this pace the job needs about {hours:.1}h more, past max_hours ({at}): \
             the pod stays while overbrainer follows the job; if it stops following, \
             the pod's watchdog deletes the pod"
        ));
    }
    Some(format!(
        "at this pace the job needs about {hours:.1}h more, but the pod's watchdog deletes it by {at} (max_hours): \
         raise the target's max_hours, or train with --keep-pod, to let it finish"
    ))
}

/// Acts on what [`watch_on_pod`] found for the run `run_id`: past the deadline
/// or the cost cap, deletes the pod and fails the run; after a failed watch, fails the run when
/// its pod is gone. Callers must not drop it midway (the CLI shields it from
/// Ctrl-C), since it may be deleting the pod.
///
/// # Errors
///
/// As [`follow`].
pub async fn settle_watch(
    ctx: &PodCtx<'_>,
    pod: &mut PodRecord,
    run_id: &str,
    watched: Watched,
) -> Result<Outcome, PodError> {
    match watched {
        Watched::Ended(ended) => match *ended {
            Ok(outcome) => Ok(outcome),
            Err(error) => Err(unreachable(ctx, pod, run_id, error.into()).await),
        },
        Watched::DeadlinePassed => Err(deadline_reached(ctx, pod, run_id).await),
        Watched::CostCapPassed => Err(cost_cap_reached(ctx, pod, run_id).await),
    }
}

/// Time left until the client's own deadline, `None` for a kept pod.
fn until_deadline(pod: &PodRecord) -> Option<Duration> {
    until_deadline_at(pod, SystemTime::now())
}

/// Time left at `now` until the client's own deadline (the watchdog's plus
/// [`DEADLINE_MARGIN`]), zero once it passed; `None` for a kept pod, or one
/// without a deadline.
pub(super) fn until_deadline_at(pod: &PodRecord, now: SystemTime) -> Option<Duration> {
    if pod.keep {
        return None;
    }
    let deadline = pod.deadline_unix?.saturating_add(DEADLINE_MARGIN.as_secs());
    let now = now
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    Some(Duration::from_secs(deadline.saturating_sub(now)))
}

/// The pod outlived its deadline: deletes it (confirmed; recorded deleted by the
/// watchdog when it was already gone), and fails the run.
async fn deadline_reached(ctx: &PodCtx<'_>, pod: &mut PodRecord, run_id: &str) -> PodError {
    if let Err(error) = remove(ctx, pod, DeleteReason::Deadline, DeletedBy::Client).await {
        return error;
    }
    fail_run(ctx.runs, run_id, MAX_HOURS_REACHED);
    PodError::DeadlineReached
}

/// The pod spent its cost cap: deletes it (confirmed), as its watchdog does
/// too, and fails the run.
async fn cost_cap_reached(ctx: &PodCtx<'_>, pod: &mut PodRecord, run_id: &str) -> PodError {
    if let Err(error) = remove(ctx, pod, DeleteReason::CostCap, DeletedBy::Client).await {
        return error;
    }
    fail_run(ctx.runs, run_id, MAX_COST_REACHED);
    PodError::CostCapReached
}

/// The watch failed with `error`: when the pod is gone, the run is failed with
/// that reason instead. A pod gone once its watchdog's deadline passed was
/// deleted by that watchdog: the run failed on `max_hours`, before the client's
/// own guard (the deadline plus [`DEADLINE_MARGIN`]) could fire.
async fn unreachable(
    ctx: &PodCtx<'_>,
    pod: &mut PodRecord,
    run_id: &str,
    error: PodError,
) -> PodError {
    match gone(ctx, pod).await {
        Ok(true) if past_cost_cap(pod, SystemTime::now()) => {
            fail_run(ctx.runs, run_id, MAX_COST_REACHED);
            PodError::CostCapReached
        },
        Ok(true) if past_deadline(pod, SystemTime::now()) => {
            fail_run(ctx.runs, run_id, MAX_HOURS_REACHED);
            PodError::DeadlineReached
        },
        Ok(true) => {
            let id = pod.pod_id.clone();
            if let Some(id) = &id {
                fail_run(ctx.runs, run_id, &format!("pod {id} no longer exists"));
            }
            id.map_or(error, PodError::PodGone)
        },
        Ok(false) => error,
        Err(check_error) => {
            tracing::warn!("cannot look the pod up: {check_error}");
            error
        },
    }
}

/// Whether the run's pod no longer exists; if so, `pod.json` records it deleted
/// by [`gone_by`]. Nothing is ever deleted here: see [`look_again`].
async fn gone(ctx: &PodCtx<'_>, pod: &mut PodRecord) -> Result<bool, PodError> {
    let Some(id) = pod.pod_id.clone() else {
        return Ok(true);
    };
    if pod.state == PodState::Deleted {
        return Ok(true);
    }
    if ctx.client.get_pod(&id).await?.is_some() || look_again(ctx, &id).await?.is_some() {
        return Ok(false);
    }
    mark_gone(ctx, pod, id, gone_by(pod, SystemTime::now()))?;
    Ok(true)
}

/// Whether the watchdog's own deadline for `pod` has passed at `now`; never for
/// a kept pod, or one without a deadline. A pod found gone then was deleted by
/// its watchdog on `max_hours`.
#[must_use]
pub fn past_deadline(pod: &PodRecord, now: SystemTime) -> bool {
    let now = now
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    !pod.keep && pod.deadline_unix.is_some_and(|deadline| now >= deadline)
}

/// Whether the cost cap of `pod` was spent at `now`: a pod found gone then was
/// deleted by its watchdog on `max_cost_usd`.
fn past_cost_cap(pod: &PodRecord, now: SystemTime) -> bool {
    pod.cost_cap()
        .is_some_and(|cap| unix_secs(now) >= cap.delete_at)
}

/// Why the run of `pod`, found gone at `now`, failed when a limit of its
/// watchdog had passed: [`MAX_COST_REACHED`] or [`MAX_HOURS_REACHED`].
#[must_use]
pub fn limit_reached(pod: &PodRecord, now: SystemTime) -> Option<&'static str> {
    if past_cost_cap(pod, now) {
        Some(MAX_COST_REACHED)
    } else if past_deadline(pod, now) {
        Some(MAX_HOURS_REACHED)
    } else {
        None
    }
}

/// Who deleted `pod`, found gone at `now` without overbrainer deleting it: its
/// watchdog once its deadline or its cost cap passed, otherwise nobody known.
fn gone_by(pod: &PodRecord, now: SystemTime) -> DeletedBy {
    if limit_reached(pod, now).is_some() {
        DeletedBy::Watchdog
    } else {
        DeletedBy::Unknown
    }
}

/// The pod `id` as the API shows it, or `None` when it is really gone: a 404
/// that [`look_again`] confirms. Nothing is ever deleted here.
///
/// # Errors
///
/// Returns [`PodError::Api`] when a look fails: the pod is then undetermined.
pub(super) async fn look_up(ctx: &PodCtx<'_>, id: &PodId) -> Result<Option<Pod>, PodError> {
    match ctx.client.get_pod(id).await? {
        Some(pod) => Ok(Some(pod)),
        None => look_again(ctx, id).await,
    }
}

/// Looks again at the pod `id`, whose first look just answered Runpod's 404:
/// `None` when it is really gone, that is [`GONE_LOOKS`] looks in a row answer
/// that 404, [`Timing::gone_interval`] apart, and the pod list no longer shows
/// it; otherwise the pod as last seen. A pod that only looks gone may be kept,
/// or hold results not yet retrieved, so it is never deleted: any answer showing
/// the pod means it is not gone, and any error leaves it undetermined (the error
/// is returned and nothing is recorded).
///
/// [`Timing::gone_interval`]: super::Timing::gone_interval
async fn look_again(ctx: &PodCtx<'_>, id: &PodId) -> Result<Option<Pod>, PodError> {
    for _ in 1..GONE_LOOKS {
        tokio::time::sleep(ctx.timing.gone_interval).await;
        if let Some(pod) = ctx.client.get_pod(id).await? {
            return Ok(Some(pod));
        }
    }
    Ok(ctx
        .client
        .list_pods()
        .await?
        .into_iter()
        .find(|listed| listed.id == *id))
}

/// What a batched look found of one pod (see [`look_up_all`]).
#[derive(Debug)]
pub(super) enum Look {
    /// The API shows the pod.
    Found(Box<Pod>),
    /// Confirmed gone: [`GONE_LOOKS`] 404s in a row and absent from the list.
    Gone,
    /// A look failed: the pod is undetermined. Holds the error's message.
    Failed(String),
}

/// [`look_up`] for several pods at once, with the same strength per pod but
/// one wait for all: every pod is looked at once; those answering Runpod's 404
/// are looked at again [`GONE_LOOKS`] - 1 times, [`Timing::gone_interval`]
/// apart, then checked against one pod list. No wait at all when every pod
/// answers. Nothing is ever deleted here. The looks are in the order of `ids`.
///
/// [`Timing::gone_interval`]: super::Timing::gone_interval
pub(super) async fn look_up_all(ctx: &PodCtx<'_>, ids: &[PodId]) -> Vec<Look> {
    let mut looks: Vec<Option<Look>> = ids.iter().map(|_| None).collect();
    let mut pending: Vec<usize> = (0..ids.len()).collect();
    for round in 0..GONE_LOOKS {
        if pending.is_empty() {
            break;
        }
        if round > 0 {
            tokio::time::sleep(ctx.timing.gone_interval).await;
        }
        pending = look_round(ctx, ids, &pending, &mut looks).await;
    }
    look_again_all(ctx, ids, &pending, &mut looks).await;
    looks
        .into_iter()
        .map(|look| {
            // Every pod gets a look above; one without is undetermined, never gone.
            look.unwrap_or_else(|| Look::Failed("the pod was not looked at".to_string()))
        })
        .collect()
}

/// Looks at the pods at `pending` once; returns those answering Runpod's 404.
async fn look_round(
    ctx: &PodCtx<'_>,
    ids: &[PodId],
    pending: &[usize],
    looks: &mut [Option<Look>],
) -> Vec<usize> {
    let mut missing = Vec::new();
    for &index in pending {
        match ctx.client.get_pod(&ids[index]).await {
            Ok(Some(pod)) => looks[index] = Some(Look::Found(Box::new(pod))),
            Ok(None) => missing.push(index),
            Err(error) => looks[index] = Some(Look::Failed(error.to_string())),
        }
    }
    missing
}

/// The last step of [`look_up_all`]: one pod list, which must not show the pods
/// at `pending` (each answered [`GONE_LOOKS`] 404s in a row) for them to be
/// gone.
async fn look_again_all(
    ctx: &PodCtx<'_>,
    ids: &[PodId],
    pending: &[usize],
    looks: &mut [Option<Look>],
) {
    if pending.is_empty() {
        return;
    }
    match ctx.client.list_pods().await {
        Ok(listed) => {
            for &index in pending {
                looks[index] = Some(match listed.iter().find(|pod| pod.id == ids[index]) {
                    Some(pod) => Look::Found(Box::new(pod.clone())),
                    None => Look::Gone,
                });
            }
        },
        Err(error) => {
            for &index in pending {
                looks[index] = Some(Look::Failed(error.to_string()));
            }
        },
    }
}

/// Records the pod `id` as deleted by `by`.
pub(super) fn mark_gone(
    ctx: &PodCtx<'_>,
    pod: &mut PodRecord,
    id: PodId,
    by: DeletedBy,
) -> Result<(), PodError> {
    let now = SystemTime::now();
    let uptime = pod.uptime(now);
    pod.deleted(by, now);
    pod.save(ctx.runs)?;
    forget_client_key(ctx.runs, &pod.run_id);
    ctx.bus.publish(Event::PodStatus(PodStatus::Deleted {
        pod_id: id,
        uptime,
        estimated_spend: pod.estimated_spend,
    }));
    Ok(())
}

/// Saves the run `run_id` as `Failed` with `message`, best effort.
fn fail_run(runs: &Runs, run_id: &str, message: &str) {
    let result = runs.load(run_id).and_then(|mut run| {
        run.state = RunState::Failed;
        run.message = Some(message.to_string());
        runs.save(&run)
    });
    if let Err(error) = result {
        tracing::warn!("cannot record run {run_id} as failed: {error}");
    }
}

/// What became of a pod once its job ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ending {
    /// Deleted after its results were retrieved.
    Deleted,
    /// Kept (`--keep-pod`), its results retrieved or not: only `overbrainer pod
    /// rm` deletes it.
    Kept,
    /// Its results were not retrieved and it is not kept: it stays for the
    /// watchdog's retrieve grace, or until `train attach` retrieves them.
    AwaitingRetrieval,
}

/// Ends the pod of the run `run` once its job ended, after copying the watchdog's
/// and the bootstrap's logs into `runs/<run-id>/.pod/`. Only when the results were `retrieved`
/// ([`Outcome::retrieved`]: every file the pod has was downloaded and verified)
/// is the "retrieved" marker written on the pod, first (so its watchdog deletes
/// it even if this client's delete fails), then the pod is deleted unless kept
/// and its deletion confirmed, other pods of the run are swept (any whose
/// deletion cannot be confirmed is recorded in `pod.json`), and the run's
/// private client key is removed. Otherwise the pod is left to its watchdog's
/// retrieve grace (a kept pod stays kept), with a warning telling how to
/// retrieve the results again or remove the pod.
///
/// # Errors
///
/// Returns a [`PodError`] when the delete fails or cannot be confirmed (the pod
/// then stays in `pod.json`, as `Deleting`) or `pod.json` cannot be saved.
pub async fn end_pod(
    ctx: &PodCtx<'_>,
    pod: &mut PodRecord,
    executor: &SshExecutor,
    run: &RunRecord,
    retrieved: bool,
) -> Result<Ending, PodError> {
    fetch_watchdog_log(ctx, executor, run).await;
    if !retrieved {
        return unretrieved(ctx, pod, &run.id);
    }
    let marker = format!("{}/{RETRIEVED_MARKER}", run.remote_dir);
    if let Err(error) = executor.put_file(&marker, "").await {
        tracing::warn!(
            "cannot mark the results of run {} retrieved on the pod: {error}",
            run.id
        );
    }
    if pod.keep {
        pod.state = PodState::Kept;
        pod.save(ctx.runs)?;
        if let Some(id) = &pod.pod_id {
            ctx.bus
                .publish(Event::PodStatus(PodStatus::Kept { pod_id: id.clone() }));
        }
        return Ok(Ending::Kept);
    }
    remove(ctx, pod, DeleteReason::Retrieved, DeletedBy::Client).await?;
    let stray = sweep(ctx, &run.id, None).await;
    note_strays(ctx, pod, stray);
    forget_client_key(ctx.runs, &run.id);
    Ok(Ending::Deleted)
}

/// The results of the run `run_id` were not retrieved: the pod stays, for its
/// watchdog's retrieve grace or, kept, until removed.
fn unretrieved(ctx: &PodCtx<'_>, pod: &mut PodRecord, run_id: &str) -> Result<Ending, PodError> {
    let name = pod
        .pod_id
        .as_ref()
        .map_or_else(|| "(none)".to_string(), ToString::to_string);
    let (stays, ending) = if pod.keep {
        pod.state = PodState::Kept;
        (
            "stays (--keep-pod), nothing deletes it automatically",
            Ending::Kept,
        )
    } else {
        pod.state = PodState::AwaitingRetrieval;
        (
            "stays until its watchdog's retrieve grace ends",
            Ending::AwaitingRetrieval,
        )
    };
    pod.save(ctx.runs)?;
    tracing::warn!(
        "the results of run {run_id} were not retrieved: pod {name} {stays}; retrieve them with `overbrainer train attach {run_id}`, or remove the pod with `overbrainer pod rm {run_id}`"
    );
    Ok(ending)
}

/// The watchdog's log on the pod, in the run directory on the pod.
pub const WATCHDOG_LOG: &str = ".pod/watchdog.log";

/// Largest copy of the watchdog's or the bootstrap's log, in bytes.
const MAX_POD_FILE_LOG: u64 = 4 * 1024 * 1024;

/// Copies the watchdog's and the bootstrap's logs of the run into
/// `runs/<run-id>/.pod/`, best effort: the pod's own logs keep them too.
/// Each is read as bytes (never extracted from an archive the pod builds, so
/// the pod cannot make the copy land anywhere else), redacted line by line,
/// then written like the pod's log (see [`keep_file`]). A pod of an older
/// overbrainer has no bootstrap log: nothing is written for it.
async fn fetch_watchdog_log(ctx: &PodCtx<'_>, executor: &SshExecutor, run: &RunRecord) {
    let Ok(local) = ctx.runs.run_dir(&run.id) else {
        return;
    };
    for (entry, what) in [(WATCHDOG_LOG, "watchdog"), (BOOTSTRAP_LOG, "bootstrap")] {
        if let Err(error) = copy_pod_file(ctx, executor, &run.remote_dir, &local, entry).await {
            tracing::warn!("cannot copy the {what}'s log of run {}: {error}", run.id);
        }
    }
}

/// Copies `entry` of the run directory `remote_dir` on the pod to the run
/// directory `local`, redacted; nothing when it is empty or absent.
async fn copy_pod_file(
    ctx: &PodCtx<'_>,
    executor: &SshExecutor,
    remote_dir: &str,
    local: &Path,
    entry: &str,
) -> Result<(), PodError> {
    let remote = format!("{remote_dir}/{entry}");
    let bytes = executor.read_from(&remote, 0, MAX_POD_FILE_LOG).await?;
    if bytes.is_empty() {
        return Ok(());
    }
    let mut redactor = Redactor::new(ctx.client.log_secrets());
    let mut text = String::new();
    for line in String::from_utf8_lossy(&bytes).lines() {
        text.push_str(&redactor.line(line));
        text.push('\n');
    }
    keep_file(local, entry, text.as_bytes()).map_err(|source| PodError::Io {
        path: local.join(entry),
        source,
    })
}

/// Removes the private client key of a run whose pod is gone, best effort.
pub fn forget_client_key(runs: &Runs, run_id: &str) {
    let Ok(dir) = runs.run_dir(run_id) else {
        return;
    };
    let key = dir.join(SSH_DIR).join(CLIENT_KEY);
    match fs::remove_file(&key) {
        Ok(()) => {},
        Err(e) if e.kind() == io::ErrorKind::NotFound => {},
        Err(error) => tracing::warn!("cannot remove {}: {error}", key.display()),
    }
}

/// Connects again to the pod of the run `run`, for `train attach` and `train
/// cancel`: looks it up (its public port may have changed), rewrites the run's ssh
/// config and connects. `None` when the pod no longer exists (confirmed by
/// repeated looks, never by a delete), which `pod.json` then records.
///
/// # Errors
///
/// Returns [`PodError::PodStopped`] when the pod is stopped (`EXITED`),
/// [`PodError::NoEndpoint`] when it has no SSH endpoint yet (restarting), and
/// another [`PodError`] when the API, the files or the connection fail.
pub async fn reconnect(
    ctx: &PodCtx<'_>,
    pod: &mut PodRecord,
    run: &RunRecord,
) -> Result<Option<SshExecutor>, PodError> {
    let Some(id) = pod.pod_id.clone() else {
        return Ok(None);
    };
    if pod.state == PodState::Deleted {
        return Ok(None);
    }
    let Some(remote) = look_up(ctx, &id).await? else {
        mark_gone(ctx, pod, id, gone_by(pod, SystemTime::now()))?;
        return Ok(None);
    };
    if remote.status == RemoteStatus::Exited {
        return Err(PodError::PodStopped {
            pod_id: id,
            run_id: run.id.clone(),
        });
    }
    let direct = remote
        .direct()
        .ok_or_else(|| PodError::NoEndpoint(id.clone(), remote.status.name().to_string()))?;
    let endpoint = SshEndpoint {
        host: direct.host.clone(),
        port: direct.port,
        user: direct.username.clone(),
    };
    let ssh_dir = ctx.runs.run_dir(&run.id)?.join(SSH_DIR);
    let keys = PodKeys::new(
        ssh_dir.join(CLIENT_KEY),
        String::new(),
        pod.host_key.clone(),
        SecretString::from(String::new()),
    );
    let alias = alias(&run.id);
    let config = write_config(&ssh_dir, &alias, &endpoint, &keys)?;
    let executor = SshExecutor::connect(&alias, pod_workdir(run), Some(&config)).await?;
    pod.ssh = Some(endpoint);
    pod.save(ctx.runs)?;
    Ok(Some(executor))
}

/// Connects to the pod of the run `run` through the ssh config the process
/// following it wrote, for `train stop` while that process holds the project:
/// no Runpod API call, and nothing written locally, so the run directory stays
/// that process's alone. `None` when the run has no pod left or no ssh config
/// yet.
///
/// # Errors
///
/// Returns a [`PodError`] when the run directory is invalid or the connection
/// fails.
pub async fn connect_followed(
    runs: &Runs,
    pod: &PodRecord,
    run: &RunRecord,
) -> Result<Option<SshExecutor>, PodError> {
    if pod.pod_id.is_none() || pod.state == PodState::Deleted {
        return Ok(None);
    }
    let config = runs.run_dir(&run.id)?.join(SSH_DIR).join(super::SSH_CONFIG);
    if !config.is_file() {
        return Ok(None);
    }
    let executor = SshExecutor::connect(&alias(&run.id), pod_workdir(run), Some(&config)).await?;
    Ok(Some(executor))
}

/// The work directory on the pod of the run `run`: the parent of its run
/// directory there.
fn pod_workdir(run: &RunRecord) -> &str {
    run.remote_dir
        .rsplit_once('/')
        .map_or(run.remote_dir.as_str(), |(parent, _)| parent)
}

/// The command reaching a kept pod: `ssh -F runs/<id>/ssh/config overbrainer-<id>`.
#[must_use]
pub fn ssh_command(run_id: &str) -> String {
    format!(
        "ssh -F runs/{run_id}/{SSH_DIR}/{} {}",
        super::SSH_CONFIG,
        alias(run_id)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runpod::ApiError;
    use crate::train::TrainMetric;

    /// A pace of 1 step per 10 s at step 100 of 7300: 20 hours left.
    fn slow_pace() -> Pace {
        let mut pace = Pace::default();
        for (time, step) in [(0.0, 90), (100.0, 100)] {
            pace.add(&TrainMetric {
                time,
                step,
                epoch: None,
                max_steps: Some(7300),
                loss: Some(1.0),
                eval_loss: None,
                learning_rate: None,
                grad_norm: None,
            });
        }
        pace
    }

    /// A pod whose watchdog deletes it `hours` after `now`.
    fn pod_due_in(hours: u64, keep: bool, now: SystemTime) -> PodRecord {
        let mut pod = PodRecord::new("r1", keep, 1, "ssh-ed25519 AAAAhost");
        let deadline = now + Duration::from_secs(hours * 3600);
        pod.deadline_unix = deadline
            .duration_since(UNIX_EPOCH)
            .ok()
            .map(|since| since.as_secs());
        pod.deadline = Some("2026-09-29T19:54:50Z".into());
        pod
    }

    #[test]
    fn a_job_that_cannot_end_before_the_deadline_is_warned_about() {
        let now = SystemTime::now();
        assert_eq!(
            overrun(&slow_pace(), &pod_due_in(6, false, now), now, false).as_deref(),
            Some(
                "at this pace the job needs about 20.0h more, but the pod's watchdog deletes it by \
                 2026-09-29T19:54:50Z (max_hours): raise the target's max_hours, or train with \
                 --keep-pod, to let it finish"
            )
        );
    }

    #[test]
    fn a_followed_job_past_the_deadline_is_only_noted() {
        let now = SystemTime::now();
        let note = overrun(&slow_pace(), &pod_due_in(6, false, now), now, true);
        assert_eq!(
            note.as_deref(),
            Some(
                "at this pace the job needs about 20.0h more, past max_hours \
                 (2026-09-29T19:54:50Z): the pod stays while overbrainer follows the job; \
                 if it stops following, the pod's watchdog deletes the pod"
            )
        );
    }

    #[test]
    fn the_watchdog_holds_the_lease_as_long_as_the_client_counts_on() {
        let default = format!("OVERBRAINER_LEASE_TTL:-{}}}", LEASE_TTL.as_secs());
        assert!(super::super::watchdog_script().contains(&default));
    }

    #[test]
    fn the_lease_is_renewed_every_few_minutes_while_the_job_progresses() {
        let now = SystemTime::now();
        let ago = |minutes: u64| now - Duration::from_secs(minutes * 60);
        assert!(renewal_due(None, now, now));
        assert!(!renewal_due(Some(ago(1)), ago(1), now));
        assert!(renewal_due(Some(ago(5)), ago(1), now));
        // A job with no new metric for 30 minutes stalled: no renewal.
        assert!(!renewal_due(Some(ago(5)), ago(30), now));
        assert!(!renewal_due(None, ago(31), now));
    }

    #[test]
    fn the_client_guard_waits_for_the_lease_to_run_out() {
        let now = SystemTime::now();
        let ago = |minutes: u64| Some(now - Duration::from_secs(minutes * 60));
        // Deadline passed an hour ago, well beyond the margin.
        let late = pod_due_in(0, false, now - Duration::from_secs(3600));
        assert!(!guard_fires(&late, ago(1), now));
        assert!(!guard_fires(&late, ago(14), now));
        assert!(guard_fires(&late, ago(15), now));
        assert!(guard_fires(&late, None, now));
        // Before the deadline, or kept: never.
        assert!(!guard_fires(&pod_due_in(1, false, now), None, now));
        assert!(!guard_fires(
            &pod_due_in(0, true, now - Duration::from_secs(3600)),
            None,
            now
        ));
    }

    #[test]
    fn no_overrun_in_time_for_a_kept_pod_or_without_a_pace() {
        let now = SystemTime::now();
        assert_eq!(
            overrun(&slow_pace(), &pod_due_in(21, false, now), now, false),
            None
        );
        assert_eq!(
            overrun(&slow_pace(), &pod_due_in(6, true, now), now, false),
            None
        );
        assert_eq!(
            overrun(&Pace::default(), &pod_due_in(6, false, now), now, false),
            None
        );
    }

    #[test]
    fn the_cost_cap_asks_for_a_snapshot_then_names_why_the_pod_went() {
        let now = SystemTime::now();
        // Created an hour ago at $1/h, with a cap of $2: deleted at 2h, the
        // snapshot 15 min before (95% would leave only 6 min).
        let mut pod = pod_due_in(6, false, now);
        pod.created_unix = Some(unix_secs(now) - 3600);
        pod.cost_per_hour = Some(1.0);
        assert!(!cap_snapshot_due(&pod, now), "no cap");
        pod.max_cost_usd = Some(2.0);
        let hours = |h: f64| now + Duration::from_secs_f64(h * 3600.0);
        assert!(!cap_snapshot_due(&pod, hours(0.7)));
        assert!(cap_snapshot_due(&pod, hours(0.75)));
        assert_eq!(limit_reached(&pod, hours(0.95)), None);
        assert_eq!(limit_reached(&pod, hours(1.0)), Some(MAX_COST_REACHED));
        assert_eq!(gone_by(&pod, hours(1.0)), DeletedBy::Watchdog);
        assert_eq!(limit_reached(&pod, hours(6.0)), Some(MAX_COST_REACHED));
        pod.max_cost_usd = None;
        assert_eq!(limit_reached(&pod, hours(6.0)), Some(MAX_HOURS_REACHED));
        assert_eq!(gone_by(&pod, hours(1.0)), DeletedBy::Unknown);
    }

    #[test]
    fn the_watchdog_asks_for_a_snapshot_as_early_as_the_client_counts_on() {
        let default = format!(
            "OVERBRAINER_SNAPSHOT_LEAD:-{}}}",
            super::super::SNAPSHOT_LEAD.as_secs()
        );
        assert!(super::super::watchdog_script().contains(&default));
    }

    /// A pod created an hour before `now` at $1/h.
    fn billed_pod(now: SystemTime, cap: Option<f64>) -> PodRecord {
        let mut pod = pod_due_in(6, false, now);
        pod.created_unix = Some(unix_secs(now) - 3600);
        pod.cost_per_hour = Some(1.0);
        pod.max_cost_usd = cap;
        pod
    }

    fn run_in(remote_dir: &str) -> RunRecord {
        RunRecord {
            id: "r1".into(),
            target: "gpu".into(),
            created: "2026-09-22T14:30:05Z".into(),
            remote_dir: remote_dir.into(),
            job: None,
            state: RunState::Preparing,
            message: None,
            snapshot: None,
            resumed_from: None,
        }
    }

    #[tokio::test]
    async fn a_cost_cap_that_cannot_be_armed_refuses_the_run()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let executor = crate::exec::LocalExecutor::new(root.path())?;
        let now = SystemTime::now();
        let dir = root.path().join("r1");
        let run = run_in(&dir.to_string_lossy());
        // Armed: both times written where the watchdog reads them.
        arm_cost_cap(&executor, &billed_pod(now, Some(2.0)), &run).await?;
        let written = std::fs::read_to_string(dir.join(COST_CAP_FILE))?;
        assert_eq!(written, (unix_secs(now) + 3600).to_string());
        assert!(dir.join(SNAPSHOT_AT_FILE).is_file());
        // Nothing to arm without a cap.
        arm_cost_cap(&executor, &billed_pod(now, None), &run).await?;
        // No rate: refused.
        let mut unrated = billed_pod(now, Some(2.0));
        unrated.cost_per_hour = None;
        let refused = arm_cost_cap(&executor, &unrated, &run).await;
        assert!(
            matches!(refused, Err(PodError::CostCapUnset(_))),
            "{refused:?}"
        );
        // Files that cannot be written: refused.
        let blocked = root.path().join("blocked");
        std::fs::write(&blocked, "")?;
        let refused = arm_cost_cap(
            &executor,
            &billed_pod(now, Some(2.0)),
            &run_in(&blocked.to_string_lossy()),
        )
        .await;
        assert!(
            matches!(refused, Err(PodError::CostCapUnset(_))),
            "{refused:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn the_client_deletes_the_pod_at_its_cost_cap_whatever_the_lease()
    -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let executor = crate::exec::LocalExecutor::new(root.path())?;
        let now = SystemTime::now();
        let remote = root.path().join("r1").to_string_lossy().into_owned();
        let spent = billed_pod(now, Some(0.5));
        let mut lease = Lease::new(&spent, &remote);
        let fired = lease.tick(&executor).await;
        assert!(matches!(fired, Some(Watched::CostCapPassed)), "{fired:?}");
        // Its snapshot was asked for on the way.
        assert_eq!(
            std::fs::read_to_string(root.path().join("r1").join(SNAPSHOT_REQUEST))?,
            "cost"
        );
        let within = billed_pod(now, Some(20.0));
        let mut lease = Lease::new(&within, &remote);
        assert!(lease.tick(&executor).await.is_none());
        Ok(())
    }

    #[test]
    fn a_failed_run_keeps_only_the_status_of_an_api_answer() {
        let answered = PodError::Api(ApiError::Status {
            status: 500,
            message: "server text".into(),
            retry_after: None,
            capacity: false,
        });
        assert_eq!(run_message(&answered), "Runpod answered 500");
        assert_eq!(
            run_message(&PodError::Interrupted),
            "interrupted before the job started"
        );
    }
}
