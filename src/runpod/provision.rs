//! A run's pod from create to ready, and its deletion: the ordered GPU list, the
//! reconciliation of an ambiguous create, SSH readiness, the watchdog's verdict,
//! and deletes confirmed by the API.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime};

use crate::events::{Event, EventBus};
use crate::exec::{Executor, SshExecutor};
use crate::runs::Runs;

use super::{
    ApiError, Attempt, AttemptResult, CreateEnv, CreatePod, DeleteReason, DeletedBy, GpuRequest,
    HOST_KEY_ENV, MIN_CUDA_VERSION, Mounts, NetworkMount, Pod, PodError, PodId, PodKeys, PodRecord,
    PodSettings, PodStatus, RunpodClient, RunpodTarget, SshEndpoint, VOLUME_MOUNT, alias,
    pod_command, pod_env, write_config,
};

/// Ambiguous creates tried per GPU type before moving to the next one.
const MAX_AMBIGUOUS: u32 = 2;

/// Bytes read of the watchdog's verdict file.
const VERDICT_BYTES: u64 = 4096;

/// Characters kept of the watchdog's verdict, once reduced to one line of
/// printable ASCII: it reaches `pod.json`, errors and messages.
const VERDICT_CHARS: usize = 200;

/// Prefix of the watchdog's verdict reason when the bootstrap itself failed
/// (`failed bootstrap: <reason>`).
const BOOTSTRAP_FAILED: &str = "bootstrap: ";

/// Least time between two warnings that the verdict cannot be read.
const VERDICT_WARN_EVERY: Duration = Duration::from_secs(60);

/// How long the provisioning steps wait. [`Timing::standard`] in production;
/// tests use milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timing {
    /// Time between two looks at the pod.
    pub poll: Duration,
    /// Longest wait for an SSH endpoint and a working handshake, together.
    pub ready_timeout: Duration,
    /// Longest wait for the watchdog's verdict once SSH works.
    pub preflight_timeout: Duration,
    /// Waits before each look for the pod of an ambiguous create.
    pub reconcile_waits: [Duration; 2],
    /// Longest wait for a deleted pod to disappear from the API.
    pub delete_timeout: Duration,
    /// Time between two looks at a pod that looks gone, before it is declared
    /// gone (see `runpod::follow` and `runpod::reconnect`).
    pub gone_interval: Duration,
}

impl Timing {
    /// Production timing: a look every 5 s, 15 min to become reachable (an 8.5 GB
    /// image took about 3.5 min to pull), 3 min for the verdict, reconciliation
    /// after 5 s and 15 s, 60 s for a delete to show, 10 s between two looks at a
    /// pod that looks gone.
    #[must_use]
    pub fn standard() -> Self {
        Self {
            poll: Duration::from_secs(5),
            ready_timeout: Duration::from_secs(15 * 60),
            preflight_timeout: Duration::from_secs(3 * 60),
            reconcile_waits: [Duration::from_secs(5), Duration::from_secs(15)],
            delete_timeout: Duration::from_secs(60),
            gone_interval: Duration::from_secs(10),
        }
    }
}

/// What the pod flows share.
#[derive(Debug, Clone, Copy)]
pub struct PodCtx<'a> {
    /// The Runpod API, with the account key.
    pub client: &'a RunpodClient,
    /// The project's `runs/`.
    pub runs: &'a Runs,
    /// Where pod events go.
    pub bus: &'a EventBus,
    /// How long the steps wait.
    pub timing: &'a Timing,
    /// Set by Ctrl-C: provisioning stops at its next step and deletes its pod.
    pub interrupted: &'a AtomicBool,
}

impl PodCtx<'_> {
    fn publish(&self, status: PodStatus) {
        self.bus.publish(Event::PodStatus(status));
    }

    fn check(&self) -> Result<(), PodError> {
        if self.interrupted.load(Ordering::SeqCst) {
            Err(PodError::Interrupted)
        } else {
            Ok(())
        }
    }
}

/// What to create for a run.
#[derive(Debug, Clone, Copy)]
pub struct PodPlan<'a> {
    /// The run.
    pub run_id: &'a str,
    /// The target.
    pub target: &'a RunpodTarget,
    /// The run's keys.
    pub keys: &'a PodKeys,
    /// The run's local `ssh/` directory, where the ssh config is written.
    pub ssh_dir: &'a Path,
    /// Directory of the run directories on the pod: the target's
    /// [`RunpodTarget::workdir`] (tests use a directory of the test sshd).
    pub workdir: &'a str,
    /// Base URL of the Runpod API, which the pod's watchdog calls too.
    pub api_url: &'a str,
}

/// A pod ready for the run: reachable over SSH with its pinned host key, with a
/// watchdog that proved it can delete it.
#[derive(Debug)]
pub struct Provisioned {
    /// The pod.
    pub pod_id: PodId,
    /// The connection to it.
    pub executor: SshExecutor,
}

/// Why a pod did not become ready.
enum Wait {
    /// Not reachable in time, or dead: try the next GPU type.
    NotReady(String),
    /// Its watchdog cannot delete it: refuse the run.
    Refused(String),
    /// Interrupted, or a failure that no other pod would fix.
    Failed(PodError),
}

/// What one create call gave.
enum Created {
    Pod(Box<Pod>, AttemptResult),
    Unavailable(String),
    Ambiguous,
}

/// Creates the run's pod, trying `plan.target.gpu_types` in order, and waits
/// until it is ready. `record` (`pod.json`) is saved before every create call and
/// after every answer. A pod that dies or stays unreachable is deleted and the
/// next GPU type tried. When provisioning fails after a create call got no clear
/// answer, every pod of the run still listed is deleted (see [`sweep`]).
///
/// # Errors
///
/// Returns [`PodError::NoCapacity`] when no GPU type could be placed or gave a
/// ready pod,
/// [`PodError::Unanswered`] when no create call got a clear answer,
/// [`PodError::NoCredits`] on a 402, [`PodError::Rejected`] on a 422 or a 400
/// that is not a capacity failure, [`PodError::WatchdogRefused`] when the
/// watchdog cannot delete its pod and [`PodError::BootstrapFailed`] when the
/// pod's bootstrap failed (the pod is deleted in both cases),
/// [`PodError::Interrupted`] after Ctrl-C (the pod, if any, is deleted),
/// [`PodError::NotDeleted`] when a pod that did not become ready cannot be
/// confirmed deleted (it stays in `record`), and another [`PodError`] when the
/// API or a local file fails.
pub async fn provision(
    ctx: &PodCtx<'_>,
    plan: &PodPlan<'_>,
    record: &mut PodRecord,
) -> Result<Provisioned, PodError> {
    let result = walk(ctx, plan, record).await;
    if result.is_err() {
        after_failure(ctx, record).await;
    }
    result
}

/// After a failed walk: a create call with no clear answer may still produce a
/// pod, so every pod of the run is deleted, and any that cannot be confirmed
/// deleted is recorded in `record`.
async fn after_failure(ctx: &PodCtx<'_>, record: &mut PodRecord) {
    let unclear: Vec<String> = record
        .attempts
        .iter()
        .filter(|attempt| attempt.result == AttemptResult::Ambiguous)
        .map(|attempt| attempt.name.clone())
        .collect();
    if unclear.is_empty() {
        return;
    }
    let stray = sweep(ctx, &record.run_id.clone(), None).await;
    note_strays(ctx, record, stray);
    warn(&format!(
        "the create calls {} got no clear answer from Runpod: a pod may still appear; check `overbrainer pod ls`",
        unclear.join(", ")
    ));
}

/// Records `stray` pods in `pod.json`, best effort.
pub(super) fn note_strays(ctx: &PodCtx<'_>, record: &mut PodRecord, stray: Vec<PodId>) {
    if stray.is_empty() {
        return;
    }
    for id in stray {
        record.note_stray(id);
    }
    if let Err(error) = record.save(ctx.runs) {
        warn(&format!("cannot record the stray pods: {}", chain(&error)));
    }
}

/// Tries the GPU types in order: see [`provision`].
async fn walk(
    ctx: &PodCtx<'_>,
    plan: &PodPlan<'_>,
    record: &mut PodRecord,
) -> Result<Provisioned, PodError> {
    let first = record.attempts.len();
    let mut last_detail = None;
    for gpu in &plan.target.gpu_types {
        let mut ambiguous = 0;
        loop {
            ctx.check()?;
            match create(ctx, plan, record, gpu).await? {
                Created::Pod(pod, result) => match settle(ctx, plan, record, &pod, result).await? {
                    Some(provisioned) => return Ok(provisioned),
                    None => break,
                },
                Created::Unavailable(detail) => {
                    last_detail = Some(detail);
                    break;
                },
                Created::Ambiguous => {
                    ambiguous += 1;
                    if ambiguous >= MAX_AMBIGUOUS {
                        break;
                    }
                },
            }
        }
    }
    let tried = &record.attempts[first..];
    if !tried.is_empty()
        && tried
            .iter()
            .all(|attempt| attempt.result == AttemptResult::Ambiguous)
    {
        return Err(PodError::Unanswered);
    }
    Err(no_capacity(
        tried,
        last_detail,
        plan.target.network_volume_id.is_some(),
    ))
}

/// The error once every GPU type was tried. `tried` are the attempts of this
/// walk: when a pod was created but never became ready, the message says so,
/// with the last one's reason. `detail` is the last capacity or 403 answer, as
/// the client's fixed message: a request Runpod rejected for any other reason
/// stopped the walk at once instead.
fn no_capacity(tried: &[Attempt], detail: Option<String>, volume: bool) -> PodError {
    let not_ready: Vec<&Attempt> = tried
        .iter()
        .filter(|attempt| attempt.result == AttemptResult::NotReady)
        .collect();
    let mut message = match not_ready.last() {
        None => match detail {
            Some(detail) => format!("no gpu_types entry could be placed: {detail}"),
            None => "no gpu_types entry could be placed".to_string(),
        },
        Some(last) => {
            let pods = match not_ready.len() {
                1 => "1 pod was".to_string(),
                count => format!("{count} pods were"),
            };
            let because = last.detail.as_deref().map_or_else(String::new, |reason| {
                format!(", the last one because {reason}")
            });
            let mut message = format!(
                "no gpu_types entry gave a ready pod: {pods} created but never became ready{because}"
            );
            if let Some(detail) = detail {
                message = format!("{message}; the other types could not be placed: {detail}");
            }
            message
        },
    };
    if volume {
        message = format!("{message} (check network_volume_id and data_center_ids)");
    }
    PodError::NoCapacity(message)
}

/// Sends one create call for `gpu`, recording it first.
async fn create(
    ctx: &PodCtx<'_>,
    plan: &PodPlan<'_>,
    record: &mut PodRecord,
    gpu: &str,
) -> Result<Created, PodError> {
    let attempt = record
        .begin_attempt(gpu, SystemTime::now(), plan.target.max_hours)
        .clone();
    record.save(ctx.runs)?;
    ctx.publish(PodStatus::Creating {
        name: attempt.name.clone(),
        gpu_type: gpu.to_string(),
    });
    let request = request(plan, record.keep, &attempt);
    let error = match ctx.client.create_pod(&request).await {
        Ok(pod) => return Ok(Created::Pod(Box::new(pod), AttemptResult::Created)),
        Err(error) => error,
    };
    // The client's fixed message for the failure: never text from Runpod.
    let detail = error.to_string();
    match skipped(&error) {
        Some(result) => {
            record.end_attempt(result, Some(detail.clone()));
            record.save(ctx.runs)?;
            ctx.publish(PodStatus::Unavailable {
                gpu_type: gpu.to_string(),
                reason: detail.clone(),
            });
            Ok(Created::Unavailable(detail))
        },
        None if error.is_ambiguous() => {
            record.end_attempt(AttemptResult::Ambiguous, Some(detail));
            record.save(ctx.runs)?;
            tracing::warn!("no clear answer to the create call ({error}): looking for the pod");
            Ok(match reconcile(ctx, record, &attempt.name).await? {
                Some(pod) => Created::Pod(Box::new(pod), AttemptResult::Adopted),
                None => Created::Ambiguous,
            })
        },
        None => {
            record.end_attempt(AttemptResult::Rejected, Some(detail));
            record.save(ctx.runs)?;
            Err(stop(error))
        },
    }
}

/// How a failed create ends its attempt when the next GPU type may still be
/// placed: no capacity left for this type (a 400 the client recognized as such),
/// or a 403. `None` for any other failure.
fn skipped(error: &ApiError) -> Option<AttemptResult> {
    if error.is_capacity() {
        Some(AttemptResult::Unavailable)
    } else if error.status() == Some(403) {
        Some(AttemptResult::Forbidden)
    } else {
        None
    }
}

/// The error of a create failure that no other GPU type would fix: a 402, a
/// 422 or a 400 that is not about capacity (overbrainer's request is wrong), or
/// anything else the client reported.
fn stop(error: ApiError) -> PodError {
    match error.status() {
        Some(402) => PodError::NoCredits,
        Some(400 | 422) => PodError::Rejected(error.to_string()),
        _ => error.into(),
    }
}

/// The create call of `attempt`.
fn request(plan: &PodPlan<'_>, keep: bool, attempt: &Attempt) -> CreatePod {
    let target = plan.target;
    let env = pod_env(&PodSettings {
        run_id: plan.run_id,
        workdir: plan.workdir,
        deadline_unix: attempt.deadline_unix,
        boot_grace: target.boot_grace,
        retrieve_grace: target.retrieve_grace,
        keep,
        api_url: plan.api_url,
        authorized_key: &plan.keys.client_public,
    });
    CreatePod {
        name: attempt.name.clone(),
        image: target.image.clone(),
        cloud: "SECURE",
        gpu: GpuRequest {
            id: attempt.gpu_type.clone(),
            count: target.gpu_count,
            min_cuda_version: MIN_CUDA_VERSION,
        },
        disk: target.container_disk_gb,
        ports: vec!["22/tcp".to_string()],
        start_ssh: false,
        data_center_ids: (!target.data_center_ids.is_empty())
            .then(|| target.data_center_ids.clone()),
        mounts: target.network_volume_id.as_ref().map(|volume| Mounts {
            network: vec![NetworkMount {
                volume_id: volume.clone(),
                path: VOLUME_MOUNT.to_string(),
            }],
        }),
        env: CreateEnv {
            plain: env,
            host_key_name: HOST_KEY_ENV,
            host_key: plan.keys.host_private().clone(),
        },
        cmd: pod_command(),
    }
}

/// Records `pod` as the run's pod and waits for it. `Some` once ready; `None`
/// when it was deleted as not ready, so the next GPU type is tried.
async fn settle(
    ctx: &PodCtx<'_>,
    plan: &PodPlan<'_>,
    record: &mut PodRecord,
    pod: &Pod,
    result: AttemptResult,
) -> Result<Option<Provisioned>, PodError> {
    record.created(pod, result, SystemTime::now());
    let saved = record.save(ctx.runs);
    if saved.is_err() {
        // Nothing else knows this pod: delete it before giving up.
        let removed = remove(ctx, record, DeleteReason::Requested, DeletedBy::Client).await;
        saved_then_removed(saved, removed)?;
    }
    ctx.publish(PodStatus::Created {
        pod_id: pod.id.clone(),
        gpu_type: record.gpu_type.clone().unwrap_or_default(),
        data_center: record.data_center_id.clone(),
        cost_per_hour: record.cost_per_hour,
    });
    match ready(ctx, plan, record, &pod.id).await {
        Ok(provisioned) => Ok(Some(provisioned)),
        Err(Wait::NotReady(reason)) => {
            not_ready(ctx, record, &pod.id, reason).await?;
            Ok(None)
        },
        Err(Wait::Refused(reason)) => Err(refuse(ctx, record, &pod.id, reason).await),
        Err(Wait::Failed(error)) => Err(abandon(ctx, record, error).await),
    }
}

/// Deletes a pod that did not become ready and forgets it, so the next GPU type
/// can be tried. A pod whose delete cannot be confirmed stays in `record`
/// (state `Deleting`) and its error stops the walk: another pod must not be
/// created while this one may still bill.
async fn not_ready(
    ctx: &PodCtx<'_>,
    record: &mut PodRecord,
    id: &PodId,
    reason: String,
) -> Result<(), PodError> {
    warn(&format!("pod {id} is not ready: {reason}"));
    record.end_attempt(AttemptResult::NotReady, Some(reason));
    let saved = record.save(ctx.runs);
    let removed = remove(ctx, record, DeleteReason::NotReady, DeletedBy::Client).await;
    saved_then_removed(saved, removed)?;
    record.forget_pod();
    record.save(ctx.runs)?;
    Ok(())
}

/// The outcome of a `pod.json` save made before a delete that was sent anyway:
/// the delete's error first, since the pod may then still run, else the save's.
fn saved_then_removed(
    saved: Result<(), crate::runs::RunsError>,
    removed: Result<(), PodError>,
) -> Result<(), PodError> {
    match (saved, removed) {
        (Err(save), Err(delete)) => {
            warn(&format!("cannot save pod.json: {}", chain(&save)));
            Err(delete)
        },
        (_, Err(delete)) => Err(delete),
        (Err(save), Ok(())) => Err(save.into()),
        (Ok(()), Ok(())) => Ok(()),
    }
}

/// Deletes a pod whose watchdog refused it (it cannot delete it, or the pod's
/// bootstrap failed), and returns the refusal.
async fn refuse(ctx: &PodCtx<'_>, record: &mut PodRecord, id: &PodId, reason: String) -> PodError {
    record.end_attempt(AttemptResult::Refused, Some(reason.clone()));
    let saved = record.save(ctx.runs);
    let (delete_reason, refusal) = refusal(id, reason);
    let removed = remove(ctx, record, delete_reason, DeletedBy::Client).await;
    if let Err(error) = saved_then_removed(saved, removed) {
        return error;
    }
    refusal
}

/// Why the pod `id`, refused by its watchdog's verdict `failed <reason>`, is
/// deleted, and the error that refuses the run.
fn refusal(id: &PodId, reason: String) -> (DeleteReason, PodError) {
    match reason.strip_prefix(BOOTSTRAP_FAILED) {
        Some(bootstrap) => (
            DeleteReason::BootstrapFailed,
            PodError::BootstrapFailed {
                pod_id: id.clone(),
                reason: bootstrap.to_string(),
            },
        ),
        None => (
            DeleteReason::Refused,
            PodError::WatchdogRefused {
                pod_id: id.clone(),
                reason,
            },
        ),
    }
}

/// Deletes the pod after an interruption or a failure, best effort, and returns
/// `error`.
async fn abandon(ctx: &PodCtx<'_>, record: &mut PodRecord, error: PodError) -> PodError {
    let reason = if matches!(error, PodError::Interrupted) {
        DeleteReason::Interrupted
    } else {
        DeleteReason::Requested
    };
    if let Err(delete_error) = remove(ctx, record, reason, DeletedBy::Client).await {
        warn(&delete_error.to_string());
    }
    error
}

fn warn(message: &str) {
    tracing::warn!("{message}");
}

/// Waits until the pod `id` answers SSH with the pinned host key and its watchdog
/// wrote `ready`, then records it as ready.
async fn ready(
    ctx: &PodCtx<'_>,
    plan: &PodPlan<'_>,
    record: &mut PodRecord,
    id: &PodId,
) -> Result<Provisioned, Wait> {
    let (executor, endpoint) = reach(ctx, plan, record, id).await?;
    let verdict = format!("{}/{}/.pod/watchdog", executor.workdir(), plan.run_id);
    preflight(ctx, &executor, &verdict, id).await?;
    let now = SystemTime::now();
    record.ready(endpoint, now);
    record
        .save(ctx.runs)
        .map_err(|error| Wait::Failed(error.into()))?;
    ctx.publish(PodStatus::Ready {
        pod_id: id.clone(),
        after: record.uptime(now).unwrap_or_default(),
        deadline: record.deadline.clone(),
    });
    let stray = sweep(ctx, plan.run_id, Some(id)).await;
    note_strays(ctx, record, stray);
    Ok(Provisioned {
        pod_id: id.clone(),
        executor,
    })
}

/// Polls the pod until it has an SSH endpoint and a strict handshake with it
/// succeeds, within [`Timing::ready_timeout`].
async fn reach(
    ctx: &PodCtx<'_>,
    plan: &PodPlan<'_>,
    record: &mut PodRecord,
    id: &PodId,
) -> Result<(SshExecutor, SshEndpoint), Wait> {
    let started = Instant::now();
    let alias = alias(plan.run_id);
    let mut last = "no SSH endpoint yet".to_string();
    loop {
        ctx.check().map_err(Wait::Failed)?;
        if started.elapsed() >= ctx.timing.ready_timeout {
            return Err(Wait::NotReady(format!(
                "not reachable over SSH after {}s: {last}",
                ctx.timing.ready_timeout.as_secs()
            )));
        }
        let pod = alive(ctx, id).await?;
        record.note_rate(&pod);
        if let Some(direct) = pod.direct() {
            let endpoint = SshEndpoint {
                host: direct.host.clone(),
                port: direct.port,
                user: direct.username.clone(),
            };
            let config =
                write_config(plan.ssh_dir, &alias, &endpoint, plan.keys).map_err(Wait::Failed)?;
            match SshExecutor::connect(&alias, plan.workdir, Some(&config)).await {
                Ok(executor) => return Ok((executor, endpoint)),
                Err(error) => last = chain(&error),
            }
        }
        tokio::time::sleep(ctx.timing.poll).await;
    }
}

/// The pod `id` as the API shows it, or [`Wait::NotReady`] at once when it is
/// gone or dead: its watchdog deletes it right away after a failed bootstrap, so
/// no later look would find it reachable.
async fn alive(ctx: &PodCtx<'_>, id: &PodId) -> Result<Pod, Wait> {
    let pod = ctx
        .client
        .get_pod(id)
        .await
        .map_err(|error| Wait::Failed(error.into()))?
        .ok_or_else(|| Wait::NotReady("the pod disappeared".to_string()))?;
    if pod.status.is_dead() {
        return Err(Wait::NotReady(format!("the pod is {}", pod.status.name())));
    }
    Ok(pod)
}

/// Reads the watchdog's verdict at `path` until it says `ready` or
/// `failed <reason>` (`failed bootstrap: <reason>` when the bootstrap failed),
/// within [`Timing::preflight_timeout`]. While there is no verdict, the pod `id`
/// is checked on every poll, so a pod that died meanwhile ends the wait at once.
/// A read failure is logged when it first happens, then at most once a minute.
async fn preflight(
    ctx: &PodCtx<'_>,
    executor: &SshExecutor,
    path: &str,
    id: &PodId,
) -> Result<(), Wait> {
    let started = Instant::now();
    let mut warned: Option<Instant> = None;
    loop {
        ctx.check().map_err(Wait::Failed)?;
        match executor.read_from(path, 0, VERDICT_BYTES).await {
            Ok(bytes) => {
                let text = one_line(&String::from_utf8_lossy(&bytes));
                if text == "ready" {
                    return Ok(());
                }
                if let Some(reason) = text.strip_prefix("failed ") {
                    return Err(Wait::Refused(reason.to_string()));
                }
            },
            Err(error) => {
                if warned.is_none_or(|at| at.elapsed() >= VERDICT_WARN_EVERY) {
                    tracing::warn!("cannot read the watchdog's verdict: {}", chain(&error));
                    warned = Some(Instant::now());
                }
            },
        }
        // No verdict yet (a missing file reads as empty): a pod that died
        // meanwhile will never write one.
        alive(ctx, id).await?;
        if started.elapsed() >= ctx.timing.preflight_timeout {
            return Err(Wait::Refused(format!(
                "no verdict within {}s",
                ctx.timing.preflight_timeout.as_secs()
            )));
        }
        tokio::time::sleep(ctx.timing.poll).await;
    }
}

/// `text` as one short line of printable ASCII: control and non-ASCII
/// characters dropped, at most [`VERDICT_CHARS`] kept.
fn one_line(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_ascii_graphic() || *c == ' ')
        .take(VERDICT_CHARS)
        .collect::<String>()
        .trim()
        .to_string()
}

/// Deletes the run's current pod and waits until the API no longer knows it,
/// then records it deleted by `by`, or by [`DeletedBy::Watchdog`] when it was
/// already gone. Does nothing when there is no pod. The delete is sent even
/// when `pod.json` cannot be saved first.
///
/// # Errors
///
/// Returns [`PodError::NotDeleted`] when the pod still shows after
/// [`Timing::delete_timeout`] (the record stays `Deleting`, with the pod), and
/// another [`PodError`] when the API or `pod.json` fails.
pub async fn remove(
    ctx: &PodCtx<'_>,
    record: &mut PodRecord,
    reason: DeleteReason,
    by: DeletedBy,
) -> Result<(), PodError> {
    let Some(id) = record.pod_id.clone() else {
        return Ok(());
    };
    ctx.publish(PodStatus::Deleting {
        pod_id: id.clone(),
        reason,
    });
    record.state = super::PodState::Deleting;
    let saved = record.save(ctx.runs);
    // The first look only names who deleted the pod: the delete is always sent,
    // since a one-off 404 must never pass a live pod for deleted. When both the
    // look and the delete say Runpod no longer knows it, its watchdog deleted it
    // (after a failed bootstrap, for example).
    let gone = matches!(ctx.client.get_pod(&id).await, Ok(None));
    let found = match delete_confirmed(ctx, &id).await {
        Ok(found) => found,
        Err(error) => return saved_then_removed(saved, Err(error)),
    };
    let now = SystemTime::now();
    let uptime = record.uptime(now);
    record.deleted(
        if gone && !found {
            DeletedBy::Watchdog
        } else {
            by
        },
        now,
    );
    let recorded = record.save(ctx.runs);
    ctx.publish(PodStatus::Deleted {
        pod_id: id,
        uptime,
        estimated_spend: record.estimated_spend,
    });
    recorded?;
    saved?;
    Ok(())
}

/// Sends the delete of `id` and waits until the API no longer knows it. `true`
/// when the delete found the pod, `false` when Runpod already did not know it.
pub(super) async fn delete_confirmed(ctx: &PodCtx<'_>, id: &PodId) -> Result<bool, PodError> {
    let found = ctx.client.delete_pod(id).await?;
    wait_gone(ctx, id).await?;
    Ok(found)
}

/// Polls until the API no longer knows the pod `id`, for at most
/// [`Timing::delete_timeout`].
///
/// # Errors
///
/// Returns [`PodError::NotDeleted`] when the pod still shows after that, and
/// [`PodError::Api`] when the API fails.
pub async fn wait_gone(ctx: &PodCtx<'_>, id: &PodId) -> Result<(), PodError> {
    let started = Instant::now();
    loop {
        if ctx.client.get_pod(id).await?.is_none() {
            return Ok(());
        }
        if started.elapsed() >= ctx.timing.delete_timeout {
            return Err(PodError::NotDeleted(id.clone()));
        }
        tokio::time::sleep(ctx.timing.poll).await;
    }
}

/// Looks for the pod of an ambiguous create by the run marker, twice. The pod
/// named `name` is adopted; any other pod of the run is a duplicate of an earlier
/// ambiguous create and is deleted (recorded in `record` when that cannot be
/// confirmed). After Ctrl-C the remaining wait is skipped but the pods are still
/// listed once, so a pod the call created is adopted and then deleted.
///
/// # Errors
///
/// Returns [`PodError::Interrupted`] after Ctrl-C when no pod was found.
async fn reconcile(
    ctx: &PodCtx<'_>,
    record: &mut PodRecord,
    name: &str,
) -> Result<Option<Pod>, PodError> {
    for wait in ctx.timing.reconcile_waits {
        nap(ctx, wait).await;
        let interrupted = ctx.check().is_err();
        let mine: Vec<Pod> = ctx
            .client
            .list_pods()
            .await?
            .into_iter()
            .filter(|pod| pod.run_id() == Some(record.run_id.as_str()))
            .collect();
        if let Some(found) = mine.iter().find(|pod| pod.name == name) {
            let mut stray = Vec::new();
            for duplicate in mine.iter().filter(|pod| pod.id != found.id) {
                if !delete_duplicate(ctx, &duplicate.id).await {
                    stray.push(duplicate.id.clone());
                }
            }
            note_strays(ctx, record, stray);
            return Ok(Some(found.clone()));
        }
        if interrupted {
            return Err(PodError::Interrupted);
        }
    }
    Ok(None)
}

/// Sleeps `wait`, or less once Ctrl-C was pressed.
async fn nap(ctx: &PodCtx<'_>, wait: Duration) {
    let started = Instant::now();
    while ctx.check().is_ok() {
        let left = wait.saturating_sub(started.elapsed());
        if left.is_zero() {
            return;
        }
        tokio::time::sleep(left.min(ctx.timing.poll)).await;
    }
}

/// Deletes every pod carrying the marker of `run_id` but `keep`: a pod left over
/// by an ambiguous create that showed up late. Returns the pods whose deletion
/// could not be confirmed, for `pod.json`.
pub async fn sweep(ctx: &PodCtx<'_>, run_id: &str, keep: Option<&PodId>) -> Vec<PodId> {
    let pods = match ctx.client.list_pods().await {
        Ok(pods) => pods,
        Err(error) => {
            tracing::warn!("cannot look for duplicate pods of run {run_id}: {error}");
            return Vec::new();
        },
    };
    let mut stray = Vec::new();
    for pod in pods {
        if pod.run_id() == Some(run_id)
            && Some(&pod.id) != keep
            && !delete_duplicate(ctx, &pod.id).await
        {
            stray.push(pod.id);
        }
    }
    stray
}

/// Deletes the duplicate pod `id` and confirms it is gone; false when that
/// failed, so the caller records it.
async fn delete_duplicate(ctx: &PodCtx<'_>, id: &PodId) -> bool {
    ctx.publish(PodStatus::Deleting {
        pod_id: id.clone(),
        reason: DeleteReason::Duplicate,
    });
    match delete_confirmed(ctx, id).await {
        Ok(_) => true,
        Err(error) => {
            tracing::warn!(
                "cannot confirm the deletion of duplicate pod {id}: {error}; its watchdog deletes it after its boot grace"
            );
            false
        },
    }
}

/// `error` and its sources, joined with `: `.
#[must_use]
pub fn chain(error: &dyn std::error::Error) -> String {
    let mut parts = vec![error.to_string()];
    let mut source = error.source();
    while let Some(cause) = source {
        parts.push(cause.to_string());
        source = cause.source();
    }
    parts.join(": ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_verdict_is_one_short_line_of_printable_ascii() {
        assert_eq!(one_line("ready\n"), "ready");
        assert_eq!(
            one_line("failed http_403\u{1b}[31m\r\nsecond line\u{7}"),
            "failed http_403[31msecond line"
        );
        assert_eq!(one_line("failed caf\u{e9}"), "failed caf");
        let long = format!("failed {}", "x".repeat(5000));
        assert_eq!(one_line(&long).len(), VERDICT_CHARS);
    }

    #[test]
    fn a_failed_bootstrap_has_its_own_delete_reason() -> Result<(), String> {
        let id = PodId::new("p1")?;
        let (reason, error) = refusal(&id, "bootstrap: cannot start sshd".to_string());
        assert_eq!(reason, DeleteReason::BootstrapFailed);
        assert_eq!(reason.describe(), "its bootstrap failed");
        assert!(matches!(error, PodError::BootstrapFailed { .. }), "{error}");
        let (reason, error) = refusal(&id, "http_403".to_string());
        assert_eq!(reason, DeleteReason::Refused);
        assert!(matches!(error, PodError::WatchdogRefused { .. }), "{error}");
        Ok(())
    }
}
