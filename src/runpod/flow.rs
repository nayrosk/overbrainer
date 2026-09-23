//! The steps of a Runpod run around the generic run flows: provisioning its pod,
//! following its job under a client-side deadline, and ending the pod once the
//! results are retrieved. The CLI drives them and decides what Ctrl-C does.

use std::fs;
use std::io;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use secrecy::SecretString;

use crate::events::Event;
use crate::exec::{Executor, SshExecutor};
use crate::runs::{Outcome, RunCtx, RunError, RunRecord, RunState, Runs, watch};
use crate::train::Trainer;

use super::provision::note_strays;
use super::{
    CLIENT_KEY, DeleteReason, DeletedBy, Pod, PodCtx, PodError, PodId, PodKeys, PodPlan, PodRecord,
    PodState, PodStatus, Provisioned, RemoteStatus, RunpodTarget, SSH_DIR, SshEndpoint, alias,
    provision, remove, sweep, write_config,
};

/// How long after the watchdog's deadline the client deletes the pod itself, when
/// it is still there: the watchdog should have done it first.
pub const DEADLINE_MARGIN: Duration = Duration::from_secs(5 * 60);

/// Where the watchdog looks for the client's "retrieved" marker, in a run
/// directory on the pod.
pub const RETRIEVED_MARKER: &str = ".pod/retrieved";

/// Looks in a row that must answer Runpod's 404 before a pod is declared gone.
const GONE_LOOKS: u32 = 3;

/// Creates the pod of the new run `run` (from `runs::create`) and waits until it is
/// ready: keys in `runs/<id>/ssh/`, then `pod.json`, then provisioning. When that
/// fails, the run is saved `Failed` with the reason (`interrupted before the job
/// started` after Ctrl-C), whatever pod was created is deleted, and once every
/// pod of the run is confirmed deleted, the run's private client key is removed.
///
/// # Errors
///
/// Returns the [`PodError`] of the step that failed.
pub async fn start_pod(
    ctx: &PodCtx<'_>,
    target: &RunpodTarget,
    mut run: RunRecord,
    keep: bool,
) -> Result<(PodRecord, Provisioned), PodError> {
    let result = provision_run(ctx, target, &run, keep).await;
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

async fn provision_run(
    ctx: &PodCtx<'_>,
    target: &RunpodTarget,
    run: &RunRecord,
    keep: bool,
) -> Result<(PodRecord, Provisioned), PodError> {
    let ssh_dir = ctx.runs.run_dir(&run.id)?.join(SSH_DIR);
    let keys = PodKeys::generate(&ssh_dir, &alias(&run.id))?;
    let mut record = PodRecord::new(&run.id, keep, target.gpu_count, &keys.host_public);
    record.save(ctx.runs)?;
    let plan = PodPlan {
        run_id: &run.id,
        target,
        keys: &keys,
        ssh_dir: &ssh_dir,
        workdir: target.workdir(),
        api_url: ctx.client.base_url(),
    };
    let provisioned = provision(ctx, &plan, &mut record).await?;
    Ok((record, provisioned))
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
/// That is [`watch_on_pod`], then [`settle_watch`].
///
/// # Errors
///
/// Returns [`PodError::DeadlineReached`] after that guard fired,
/// [`PodError::PodGone`] when the target stays unreachable because the pod no
/// longer exists (the run is then saved `Failed`), and [`PodError::Run`] for any
/// other failure of the watch (the job may keep running: attach again).
pub async fn follow<E: Executor, T: Trainer>(
    ctx: &PodCtx<'_>,
    run_ctx: &RunCtx<'_, E>,
    trainer: &T,
    record: RunRecord,
    pod: &mut PodRecord,
) -> Result<Outcome, PodError> {
    let id = record.id.clone();
    let watched = watch_on_pod(run_ctx, trainer, record, pod).await;
    settle_watch(ctx, pod, &id, watched).await
}

/// How [`watch_on_pod`] ended.
#[derive(Debug)]
pub enum Watched {
    /// The watch ended first, with its outcome or its error.
    Ended(Box<Result<Outcome, RunError>>),
    /// The client's deadline passed first; nothing was deleted yet.
    DeadlinePassed,
}

/// Watches the started run `record` until its job ends or the client's deadline
/// (the watchdog's plus [`DEADLINE_MARGIN`]; none for a kept pod) passes. It
/// never calls the Runpod API, so dropping it (on Ctrl-C) loses nothing: what
/// it found is acted on by [`settle_watch`].
pub async fn watch_on_pod<E: Executor, T: Trainer>(
    run_ctx: &RunCtx<'_, E>,
    trainer: &T,
    record: RunRecord,
    pod: &PodRecord,
) -> Watched {
    match until_deadline(pod) {
        Some(wait) => tokio::select! {
            outcome = watch(run_ctx, trainer, record) => Watched::Ended(Box::new(outcome)),
            () = tokio::time::sleep(wait) => Watched::DeadlinePassed,
        },
        None => Watched::Ended(Box::new(watch(run_ctx, trainer, record).await)),
    }
}

/// Acts on what [`watch_on_pod`] found for the run `run_id`: past the deadline,
/// deletes the pod and fails the run; after a failed watch, fails the run when
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
    fail_run(
        ctx.runs,
        run_id,
        "max_hours reached: the pod was deleted before the job ended",
    );
    PodError::DeadlineReached
}

/// The watch failed with `error`: when the pod is gone, the run is failed with
/// that reason instead.
async fn unreachable(
    ctx: &PodCtx<'_>,
    pod: &mut PodRecord,
    run_id: &str,
    error: PodError,
) -> PodError {
    match gone(ctx, pod).await {
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
/// by [`DeletedBy::Unknown`]. Nothing is ever deleted here: see [`look_again`].
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
    mark_gone(ctx, pod, id, DeletedBy::Unknown)?;
    Ok(true)
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
    /// Kept (`--keep-pod`): only `overbrainer pod rm` deletes it.
    Kept,
    /// Its results were not retrieved: it stays for the watchdog's retrieve
    /// grace, or until `train attach` retrieves them.
    AwaitingRetrieval,
}

/// Ends the pod of the run `run` once its job ended, after copying the watchdog's
/// log into `runs/<run-id>/.pod/`. Only when the results were `retrieved`
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
    fetch_watchdog_log(ctx.runs, executor, run).await;
    if !retrieved {
        return unretrieved(ctx, pod, &run.id);
    }
    let marker = format!("{}/{RETRIEVED_MARKER}", run.remote_dir);
    if let Err(error) = executor.write_marker(&marker).await {
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
    let stays = if pod.keep {
        pod.state = PodState::Kept;
        "stays (--keep-pod), nothing deletes it automatically"
    } else {
        pod.state = PodState::AwaitingRetrieval;
        "stays until its watchdog's retrieve grace ends"
    };
    pod.save(ctx.runs)?;
    tracing::warn!(
        "the results of run {run_id} were not retrieved: pod {name} {stays}; retrieve them with `overbrainer train attach {run_id}`, or remove the pod with `overbrainer pod rm {run_id}`"
    );
    Ok(Ending::AwaitingRetrieval)
}

/// The watchdog's log on the pod, in the run directory on the pod.
pub const WATCHDOG_LOG: &str = ".pod/watchdog.log";

/// Copies the watchdog's log of the run into `runs/<run-id>/.pod/`, best effort:
/// the pod's own logs keep it too.
async fn fetch_watchdog_log(runs: &Runs, executor: &SshExecutor, run: &RunRecord) {
    let Ok(local) = runs.run_dir(&run.id) else {
        return;
    };
    let entries = [WATCHDOG_LOG.to_string()];
    if let Err(error) = executor
        .download(&run.remote_dir, &local, &entries, &[])
        .await
    {
        tracing::warn!("cannot copy the watchdog's log of run {}: {error}", run.id);
    }
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
        mark_gone(ctx, pod, id, DeletedBy::Unknown)?;
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
    let workdir = run
        .remote_dir
        .rsplit_once('/')
        .map_or(run.remote_dir.as_str(), |(parent, _)| parent);
    let executor = SshExecutor::connect(&alias, workdir, Some(&config)).await?;
    pod.ssh = Some(endpoint);
    pod.save(ctx.runs)?;
    Ok(Some(executor))
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
