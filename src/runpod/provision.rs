//! A run's pod from create to ready, and its deletion: the ordered GPU list, the
//! reconciliation of an ambiguous create, SSH readiness, the watchdog's verdict,
//! and deletes confirmed by the API.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime};

use crate::events::{Event, EventBus};
use crate::exec::{ExecError, Executor, SshExecutor};
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

/// Handshakes in a row where the pod's sshd sent its banner but the local
/// master connection died, after which the cause is taken to be local.
const LOCAL_STRIKES: u32 = 2;

/// Longest wait for the pod's sshd to send its banner.
const BANNER_TIMEOUT: Duration = Duration::from_secs(5);

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
    /// Its sshd answers but the local `ssh` cannot keep its connection: stop,
    /// since no other pod would do better.
    Local(String),
    /// Interrupted, or a failure that no other pod would fix.
    Failed(PodError),
}

/// What one create call gave.
enum Created {
    Pod(Box<Pod>, AttemptResult),
    Unavailable(String),
    Ambiguous,
}

/// Creates the run's pod, trying `plan.target.gpu_types` in order (its `auto`
/// choices resolved first, see [`resolve_target`]), and waits until it is
/// ready. `record` (`pod.json`) is saved before every create call and after
/// every answer. A pod that dies or stays unreachable is deleted and the
/// next GPU type tried. When provisioning fails after a create call got no clear
/// answer, every pod of the run still listed is deleted (see [`sweep`]).
///
/// # Errors
///
/// Returns [`PodError::NotInStock`] when an `auto` choice finds nothing in
/// stock (no create call is made), [`PodError::NoCapacity`] when no GPU type
/// could be placed or gave a ready pod,
/// [`PodError::Unanswered`] when no create call got a clear answer,
/// [`PodError::NoCredits`] on a 402, [`PodError::Rejected`] on a 422 or a 400
/// that is not a capacity failure, [`PodError::WatchdogRefused`] when the
/// watchdog cannot delete its pod and [`PodError::BootstrapFailed`] when the
/// pod's bootstrap failed (the pod is deleted in both cases),
/// [`PodError::Interrupted`] after Ctrl-C (the pod, if any, is deleted),
/// [`PodError::LocalSsh`] when the pod answers SSH but the local `ssh` cannot
/// keep its master connection (the pod is deleted, no other GPU type tried),
/// [`PodError::NotDeleted`] when a pod that did not become ready cannot be
/// confirmed deleted (it stays in `record`), and another [`PodError`] when the
/// API or a local file fails.
pub async fn provision(
    ctx: &PodCtx<'_>,
    plan: &PodPlan<'_>,
    record: &mut PodRecord,
) -> Result<Provisioned, PodError> {
    ctx.check()?;
    let target = resolve_target(ctx.client, plan.target).await?;
    let plan = &PodPlan {
        target: &target,
        ..*plan
    };
    let result = walk(ctx, plan, record).await;
    if result.is_err() {
        after_failure(ctx, record).await;
    }
    result
}

/// `target` with its `auto` choices resolved from the Runpod catalog (see
/// [`resolve`](super::resolve)), logged at info level; a target without any
/// is returned as it is, without an API call. Both choices are resolved from
/// the same GPU listing, scoped to `target.gpu_count`: `catalog/datacenters`
/// is never read for this.
///
/// # Errors
///
/// Returns [`PodError::NotInStock`] when nothing in stock matches, and
/// [`PodError::Api`] when the catalog cannot be read.
pub async fn resolve_target(
    client: &RunpodClient,
    target: &RunpodTarget,
) -> Result<RunpodTarget, PodError> {
    if !target.gpu_types.is_auto() && !target.data_center_ids.is_auto() {
        return Ok(target.clone());
    }
    let gpus = client.list_gpu_types(target.gpu_count).await?;
    let resolved = super::resolve(target, &gpus).map_err(PodError::NotInStock)?;
    log_picks(target, &resolved);
    Ok(resolved)
}

/// Logs what the `auto` choices of `target` became in `resolved`.
fn log_picks(target: &RunpodTarget, resolved: &RunpodTarget) {
    let picks = [
        ("gpu_types", &target.gpu_types, &resolved.gpu_types),
        (
            "data_center_ids",
            &target.data_center_ids,
            &resolved.data_center_ids,
        ),
    ];
    for (field, asked, picked) in picks {
        if asked.is_auto() {
            tracing::info!("{field} = \"auto\" picked {picked}");
        }
    }
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
    for gpu in plan.target.gpu_types.list() {
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
        data_center_ids: (!target.data_center_ids.list().is_empty())
            .then(|| target.data_center_ids.list().to_vec()),
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
        Err(Wait::Local(reason)) => Err(local_ssh(ctx, record, &pod.id, reason).await),
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

/// Deletes the pod `id` that answers SSH while the local `ssh` cannot keep its
/// connection, and returns [`PodError::LocalSsh`], or the delete's error when
/// the pod may still run.
async fn local_ssh(
    ctx: &PodCtx<'_>,
    record: &mut PodRecord,
    id: &PodId,
    reason: String,
) -> PodError {
    record.end_attempt(AttemptResult::NotReady, Some(reason.clone()));
    let saved = record.save(ctx.runs);
    let removed = remove(ctx, record, DeleteReason::LocalSsh, DeletedBy::Client).await;
    if let Err(error) = saved_then_removed(saved, removed) {
        return error;
    }
    PodError::LocalSsh {
        pod_id: id.clone(),
        reason,
    }
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
/// succeeds, within [`Timing::ready_timeout`]. When the pod's sshd sends its
/// banner but the local master connection dies, [`LOCAL_STRIKES`] times in a
/// row, the wait stops with [`PodError::LocalSsh`]: no pod would do better.
async fn reach(
    ctx: &PodCtx<'_>,
    plan: &PodPlan<'_>,
    record: &mut PodRecord,
    id: &PodId,
) -> Result<(SshExecutor, SshEndpoint), Wait> {
    let workdir = plan.workdir;
    let connect = move |alias: String, config: PathBuf| async move {
        SshExecutor::connect(&alias, workdir, Some(&config)).await
    };
    reach_with(ctx, plan, record, id, connect).await
}

/// [`reach`], connecting with `connect(alias, config)`: tests stand in for ssh.
async fn reach_with<C, F>(
    ctx: &PodCtx<'_>,
    plan: &PodPlan<'_>,
    record: &mut PodRecord,
    id: &PodId,
    connect: C,
) -> Result<(SshExecutor, SshEndpoint), Wait>
where
    C: Fn(String, PathBuf) -> F,
    F: Future<Output = Result<SshExecutor, ExecError>>,
{
    let started = Instant::now();
    let alias = alias(plan.run_id);
    let mut last = "no SSH endpoint yet".to_string();
    let mut strikes = 0;
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
            match connect(alias.clone(), config).await {
                Ok(executor) => return Ok((executor, endpoint)),
                Err(error) => {
                    let died =
                        matches!(&error, ExecError::MasterDied { log } if !closed_remotely(log));
                    let banner =
                        died && ssh_banner(&endpoint.host, endpoint.port, BANNER_TIMEOUT).await;
                    strikes = strikes_after(strikes, banner, died);
                    last = chain(&error);
                    if strikes >= LOCAL_STRIKES {
                        return Err(Wait::Local(last));
                    }
                },
            }
        } else {
            strikes = 0;
        }
        tokio::time::sleep(ctx.timing.poll).await;
    }
}

/// Whether the master's `log` tail says the server ended the connection: a pod
/// dropping it is not a local failure.
fn closed_remotely(log: &str) -> bool {
    let log = log.to_ascii_lowercase();
    ["closed by remote host", "connection reset", "broken pipe"]
        .iter()
        .any(|sign| log.contains(sign))
}

/// The count of handshakes in a row blamed on this machine, `strikes` before
/// this one: it grows when the pod's sshd sent its `banner` but the local master
/// `died`, and restarts on any other outcome (no banner yet, a timeout, a
/// refused key), which keeps waiting as before.
fn strikes_after(strikes: u32, banner: bool, died: bool) -> u32 {
    if banner && died { strikes + 1 } else { 0 }
}

/// Whether `host:port` answers with an SSH banner (a first line starting with
/// `SSH-`) within `within`. Any failure reads as no banner.
async fn ssh_banner(host: &str, port: u16, within: Duration) -> bool {
    let host = host.to_string();
    let probe = tokio::task::spawn_blocking(move || -> std::io::Result<bool> {
        use std::io::{BufRead, BufReader, Read};
        use std::net::{TcpStream, ToSocketAddrs};
        let mut last = std::io::Error::other("no address");
        for addr in (host.as_str(), port).to_socket_addrs()? {
            match TcpStream::connect_timeout(&addr, within) {
                Ok(stream) => {
                    stream.set_read_timeout(Some(within))?;
                    let mut line = Vec::new();
                    BufReader::new(stream.take(256)).read_until(b'\n', &mut line)?;
                    return Ok(line.starts_with(b"SSH-"));
                },
                Err(error) => last = error,
            }
        }
        Err(last)
    });
    // The blocking probe is bounded by its own timeouts; this one also covers
    // a slow name lookup.
    matches!(
        tokio::time::timeout(within.saturating_mul(3), probe).await,
        Ok(Ok(Ok(true)))
    )
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
                let text = one_line(&String::from_utf8_lossy(&bytes), VERDICT_CHARS);
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

/// `text`, from a pod or the Runpod account, as one short line of printable
/// ASCII: control and non-ASCII characters dropped, at most `max` kept, and
/// the spaces around it trimmed.
pub(super) fn one_line(text: &str, max: usize) -> String {
    text.chars()
        .filter(|c| c.is_ascii_graphic() || *c == ' ')
        .take(max)
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
pub(crate) async fn wait_gone(ctx: &PodCtx<'_>, id: &PodId) -> Result<(), PodError> {
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
        assert_eq!(one_line("ready\n", VERDICT_CHARS), "ready");
        assert_eq!(
            one_line(
                "failed http_403\u{1b}[31m\r\nsecond line\u{7}",
                VERDICT_CHARS
            ),
            "failed http_403[31msecond line"
        );
        assert_eq!(one_line("failed caf\u{e9}", VERDICT_CHARS), "failed caf");
        assert_eq!(one_line("run\u{1b}[31m\nx\u{e9}", 64), "run[31mx");
        let long = format!("failed {}", "x".repeat(5000));
        assert_eq!(one_line(&long, VERDICT_CHARS).len(), VERDICT_CHARS);
        assert_eq!(one_line(&long, 64).len(), 64);
    }

    #[test]
    fn only_a_banner_with_a_dead_master_counts_as_a_local_failure() {
        assert_eq!(strikes_after(0, true, true), 1);
        assert_eq!(strikes_after(1, true, true), LOCAL_STRIKES);
        // No banner yet, or another failure: keep waiting from scratch.
        assert_eq!(strikes_after(1, false, true), 0);
        assert_eq!(strikes_after(1, true, false), 0);
        assert_eq!(strikes_after(0, false, false), 0);
    }

    #[test]
    fn a_local_ssh_failure_names_the_cause_and_the_fix() -> Result<(), crate::runpod::InvalidPodId>
    {
        let error = PodError::LocalSsh {
            pod_id: PodId::new("p1")?,
            reason: "the ssh master connection ended right after it started (ssh log: empty)"
                .into(),
        };
        let text = error.to_string();
        assert!(text.contains("pod p1"), "{text}");
        assert!(text.contains("firejail"), "{text}");
        assert!(text.contains("first on PATH"), "{text}");
        assert!(text.contains("(ssh log: empty)"), "{text}");
        Ok(())
    }

    /// A local server answering each connection with `answer`, then closing it.
    fn server(answer: &'static [u8]) -> std::io::Result<u16> {
        use std::io::Write;
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                std::thread::spawn(move || {
                    let _ = stream.write_all(answer);
                    std::thread::sleep(Duration::from_millis(500));
                });
            }
        });
        Ok(port)
    }

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// A stub Runpod API, a project and the pod `p1` whose SSH endpoint is a
    /// local server answering `SSH-2.0-...`.
    struct Stub {
        server: wiremock::MockServer,
        project: tempfile::TempDir,
        client: RunpodClient,
        runs: Runs,
        bus: EventBus,
        timing: Timing,
        interrupted: AtomicBool,
        target: RunpodTarget,
        keys: PodKeys,
        pod: Pod,
    }

    impl Stub {
        /// `deletable`: the pod disappears once deleted, else it stays listed.
        async fn new(deletable: bool) -> Result<Self, Box<dyn std::error::Error>> {
            use std::sync::Arc;
            use wiremock::matchers::{method, path};
            use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

            struct Get(Arc<AtomicBool>, serde_json::Value);
            impl Respond for Get {
                fn respond(&self, _: &Request) -> ResponseTemplate {
                    if self.0.load(Ordering::SeqCst) {
                        ResponseTemplate::new(404).set_body_json(serde_json::json!({
                            "detail": "pod not found", "status": 404, "title": "Not Found"
                        }))
                    } else {
                        ResponseTemplate::new(200).set_body_json(&self.1)
                    }
                }
            }
            struct Delete(Arc<AtomicBool>, bool);
            impl Respond for Delete {
                fn respond(&self, _: &Request) -> ResponseTemplate {
                    self.0.store(self.1, Ordering::SeqCst);
                    ResponseTemplate::new(204)
                }
            }

            let port = server(b"SSH-2.0-OpenSSH_9.6\r\n")?;
            let body = serde_json::json!({
                "id": "p1",
                "name": "overbrainer-r1-1",
                "status": "RUNNING",
                "cost": 0.25,
                "env": {"OVERBRAINER_RUN_ID": "r1"},
                "ssh": {"direct": {"host": "127.0.0.1", "port": port, "username": "root"}}
            });
            let server = MockServer::start().await;
            let deleted = Arc::new(AtomicBool::new(false));
            Mock::given(method("GET"))
                .and(path("/v2/pods/p1"))
                .respond_with(Get(Arc::clone(&deleted), body))
                .mount(&server)
                .await;
            Mock::given(method("DELETE"))
                .and(path("/v2/pods/p1"))
                .respond_with(Delete(deleted, deletable))
                .mount(&server)
                .await;
            let client = RunpodClient::new(
                &format!("{}/v2", server.uri()),
                &secrecy::SecretString::from("rp_key"),
            )?
            .with_policy(crate::retry::RetryPolicy {
                max_retries: 1,
                base: Duration::from_millis(1),
                cap: Duration::from_millis(2),
            });
            let pod = client
                .get_pod(&PodId::new("p1")?)
                .await?
                .ok_or("no pod p1")?;
            let project = tempfile::tempdir()?;
            let runs = Runs::new(project.path());
            Ok(Self {
                server,
                project,
                client,
                runs,
                bus: EventBus::new(),
                timing: Timing {
                    poll: Duration::from_millis(5),
                    ready_timeout: Duration::from_millis(400),
                    preflight_timeout: Duration::from_millis(300),
                    reconcile_waits: [Duration::from_millis(5); 2],
                    delete_timeout: Duration::from_millis(200),
                    gone_interval: Duration::from_millis(5),
                },
                interrupted: AtomicBool::new(false),
                target: RunpodTarget {
                    gpu_types: crate::config::ListOrAuto::List(vec!["A".into()]),
                    min_vram_gb: None,
                    max_price_per_hour: None,
                    gpu_count: 1,
                    image: "img@sha256:abc".into(),
                    venv: "/workspace/axolotl-venv".into(),
                    container_disk_gb: 50,
                    max_hours: 6.0,
                    boot_grace: Duration::from_secs(1800),
                    retrieve_grace: Duration::from_secs(3600),
                    data_center_ids: crate::config::ListOrAuto::default(),
                    network_volume_id: None,
                },
                keys: PodKeys::new(
                    std::path::PathBuf::from("/nonexistent/id_ed25519"),
                    "ssh-ed25519 AAAAclient overbrainer".into(),
                    "ssh-ed25519 AAAAhost".into(),
                    secrecy::SecretString::from("aG9zdC1rZXk"),
                ),
                pod,
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

        fn plan(&self) -> PodPlan<'_> {
            PodPlan {
                run_id: "r1",
                target: &self.target,
                keys: &self.keys,
                ssh_dir: self.project.path(),
                workdir: "/workspace/overbrainer",
                api_url: self.client.base_url(),
            }
        }

        fn record(&self) -> PodRecord {
            let mut record = PodRecord::new("r1", false, 1, "ssh-ed25519 AAAAhost");
            record.created(&self.pod, AttemptResult::Created, SystemTime::now());
            record
        }

        /// Waits for SSH on `p1` with every connection failing on a dead master
        /// that logged `log`. Returns the outcome and the connection count.
        async fn reach(&self, log: &str) -> (Result<(), Wait>, usize) {
            let calls = std::sync::atomic::AtomicUsize::new(0);
            let connect = |_: String, _: PathBuf| {
                calls.fetch_add(1, Ordering::SeqCst);
                std::future::ready(Err(ExecError::MasterDied { log: log.into() }))
            };
            let mut record = self.record();
            let outcome = reach_with(
                &self.ctx(),
                &self.plan(),
                &mut record,
                &self.pod.id,
                connect,
            )
            .await
            .map(drop);
            (outcome, calls.load(Ordering::SeqCst))
        }

        async fn deletes(&self) -> usize {
            self.server
                .received_requests()
                .await
                .unwrap_or_default()
                .iter()
                .filter(|request| request.method.as_str() == "DELETE")
                .count()
        }
    }

    #[tokio::test]
    async fn a_banner_with_a_master_dying_twice_stops_the_wait() -> TestResult {
        let stub = Stub::new(true).await?;
        let (outcome, calls) = stub.reach("").await;
        assert!(
            matches!(&outcome, Err(Wait::Local(reason)) if reason.contains("ssh log: empty")),
            "{:?}",
            outcome.err().map(|wait| matches!(wait, Wait::NotReady(_)))
        );
        assert_eq!(calls, LOCAL_STRIKES as usize);
        Ok(())
    }

    #[tokio::test]
    async fn a_master_dropped_by_the_server_keeps_the_wait() -> TestResult {
        let stub = Stub::new(true).await?;
        let (outcome, calls) = stub
            .reach("Connection to 127.0.0.1 closed by remote host.")
            .await;
        assert!(matches!(outcome, Err(Wait::NotReady(_))));
        assert!(calls > LOCAL_STRIKES as usize, "{calls}");
        Ok(())
    }

    #[tokio::test]
    async fn a_local_ssh_failure_deletes_the_pod() -> TestResult {
        let stub = Stub::new(true).await?;
        let mut record = stub.record();
        let error = local_ssh(&stub.ctx(), &mut record, &stub.pod.id, "dead".into()).await;
        assert!(matches!(error, PodError::LocalSsh { .. }), "{error}");
        assert_eq!(stub.deletes().await, 1);
        assert_eq!(record.state, super::super::PodState::Deleted);
        Ok(())
    }

    #[tokio::test]
    async fn a_local_ssh_failure_never_claims_an_undeleted_pod_gone() -> TestResult {
        let stub = Stub::new(false).await?;
        let mut record = stub.record();
        let error = local_ssh(&stub.ctx(), &mut record, &stub.pod.id, "dead".into()).await;
        assert!(matches!(error, PodError::NotDeleted(_)), "{error}");
        assert!(!error.to_string().contains("was deleted"), "{error}");
        assert_eq!(record.state, super::super::PodState::Deleting);
        Ok(())
    }

    #[tokio::test]
    async fn an_sshd_is_told_by_its_banner() -> std::io::Result<()> {
        let within = Duration::from_millis(300);
        let port = server(b"SSH-2.0-OpenSSH_9.6\r\n")?;
        assert!(ssh_banner("127.0.0.1", port, within).await);
        let port = server(b"HTTP/1.1 400 Bad Request\r\n")?;
        assert!(!ssh_banner("127.0.0.1", port, within).await);
        // Silent until the timeout.
        let port = server(b"")?;
        assert!(!ssh_banner("127.0.0.1", port, within).await);
        // Nothing listening.
        let closed = std::net::TcpListener::bind("127.0.0.1:0")?
            .local_addr()?
            .port();
        assert!(!ssh_banner("127.0.0.1", closed, within).await);
        Ok(())
    }

    #[test]
    fn a_failed_bootstrap_has_its_own_delete_reason() -> Result<(), crate::runpod::InvalidPodId> {
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
