//! Runpod as a training target: a pod created for a run over the REST API (v2),
//! reached over SSH with per-run keys, guarded by a watchdog running on the pod,
//! and deleted once the run's results are retrieved.
//!
//! A Runpod target is not an executor of its own: once its pod is ready, the run
//! goes through the [`SshExecutor`](crate::exec::SshExecutor) and the `native`
//! runtime, like an SSH target.

use std::path::PathBuf;

mod bootstrap;
mod catalog;
mod client;
mod disk;
mod flow;
mod keys;
mod logs;
mod orphans;
mod provision;
mod record;
mod secret;
mod status;
mod target;
mod types;

pub use bootstrap::{
    HOST_KEY_ENV, JOB_ENV, PodSettings, bootstrap_functions, pod_command, pod_env, watchdog_script,
};
pub use catalog::{
    DataCenterStock, GpuFilter, ResolveError, by_price, data_center_stock, data_center_table,
    gpu_table, printable, resolve, resolve_with_floor, select_gpus, stocked_data_centers,
    template_table, volume_table,
};
pub use client::{ApiError, RunpodClient, SECRETS_FORBIDDEN_MESSAGE, USER_AGENT};
pub use disk::{
    ACT_PERCENT, Assessment, Critical, DISK_PROBE_EVERY, DiskProbe, DiskWatch, GROW_WINDOW, Usage,
    VOLUME_SIZE_FILE, VOLUME_USABLE_PERCENT, VolumeDisk, WARN_PERCENT, WARN_STEP,
    WATCHDOG_ACT_PERCENT, assess, disk_probe_script, export_room_warning, grown_size,
    parse_disk_probe,
};
pub use flow::{
    COST_CAP_FILE, DEADLINE_MARGIN, Ending, LEASE_FILE, LEASE_TTL, MAX_COST_REACHED,
    MAX_HOURS_REACHED, RETRIEVED_MARKER, SNAPSHOT_AT_FILE, WATCHDOG_LOG, Watched, arm_cost_cap,
    connect_followed, end_pod, follow, forget_client_key, job_started, limit_reached,
    past_deadline, reconnect, settle_watch, ssh_command, start_pod, watch_leased, watch_on_pod,
};
pub use keys::{
    CLIENT_KEY, KNOWN_HOSTS, PodKeys, SSH_CONFIG, SSH_DIR, alias, base64, ssh_config, write_config,
    write_known_hosts,
};
pub use logs::{
    BOOTSTRAP_LOG, Capture, DRAIN_WAIT, LogError, LogQuery, LogSource, POD_LOG, POD_LOG_CAP,
    POD_LOG_CURSOR, PodLogLine, TAIL_MAX, capture, drain, follow as follow_logs, keep_file,
    kept_cursor, kept_lines, one_line, open_kept, parse_kept_line, read_kept, read_kept_from,
    snapshot, with_pod_logs,
};
pub use orphans::{
    PodRow, Removal, Removed, RowKind, listed_rows, orphan_warnings, pod_rows, remove_run_pods,
    sweep_host_keys, table,
};
pub use provision::{
    PodCtx, PodPlan, Provisioned, Timing, chain, provision, remove, resolve_target, sweep,
};
pub use record::{
    Attempt, AttemptResult, CostCap, DeletedBy, MIN_CAP_TIME, POD_FILE, POD_RECORD_VERSION,
    PodRecord, PodState, SNAPSHOT_LEAD, SNAPSHOT_SHARE, SshEndpoint, hours, short_cap_warning,
};
pub use secret::{
    HOST_KEY_SECRET_PREFIX, drop_host_key, drop_host_key_or_warn, forget_keys,
    host_key_placeholder, host_key_run, host_key_secret, store_host_key,
};
pub use status::{DeleteReason, PodStatus};
pub use target::{MIN_CUDA_VERSION, RunpodTarget, VOLUME_MOUNT, VOLUME_WORKDIR, WORKDIR};
pub use types::{
    Availability, CreatePod, CudaVersion, DataCenter, DataCenterList, GpuMaxCount, GpuPrice,
    GpuRequest, GpuType, GpuTypeList, InvalidPodId, Mounts, NetworkMount, NetworkVolume,
    NetworkVolumeList, NewSecret, Pagination, Pod, PodEnv, PodGpu, PodId, PodPage, PodSsh,
    RemoteStatus, Secret, SecretList, SshDirect, Stock, Template, TemplatePage,
};

/// Errors of a Runpod run's pod: provisioning, keys, readiness, deletion.
#[derive(Debug, thiserror::Error)]
pub enum PodError {
    /// The Runpod API failed.
    #[error(transparent)]
    Api(#[from] ApiError),
    /// A local file could not be read or written.
    #[error("cannot access {}", path.display())]
    Io {
        /// The file.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The run's SSH keys could not be generated.
    #[error("cannot generate the run's SSH keys: {0}")]
    Keygen(String),
    /// A local path cannot be written into an ssh config.
    #[error("{0}")]
    InvalidPath(String),
    /// Runpod reported an SSH endpoint that is not a plain host, port and user.
    #[error("Runpod gave an unusable SSH endpoint: {0}")]
    InvalidEndpoint(String),
    /// The pod cannot be reached or used.
    #[error(transparent)]
    Exec(#[from] crate::exec::ExecError),
    /// A run record cannot be read or written.
    #[error(transparent)]
    Runs(#[from] crate::runs::RunsError),
    /// No GPU type of the target gave a ready pod: none could be placed, or the
    /// pods created never became ready.
    #[error("{0}")]
    NoCapacity(String),
    /// An `auto` choice of the target found nothing in stock in the Runpod
    /// catalog; no pod was asked for. Says what was asked.
    #[error(transparent)]
    NotInStock(#[from] ResolveError),
    /// The size of the target's network volume cannot be read, even after
    /// retries: the pod's watchdog could not tell when it fills, so no pod is
    /// asked for.
    #[error(
        "cannot read the size of network volume {id} ({reason}): no pod was created, since its watchdog could not watch the volume's disk space"
    )]
    VolumeSize {
        /// The volume's ID.
        id: String,
        /// Why: Runpod's status or a fixed message, never Runpod's own text.
        reason: String,
    },
    /// Runpod asks for credits (402).
    #[error(
        "Runpod refused for lack of credits (402): deploying needs at least one hour of credits"
    )]
    NoCredits,
    /// Runpod rejected the create request itself (a 422, or a 400 that is not a
    /// capacity failure), which is a bug. Holds the client's fixed message,
    /// which already says so.
    #[error("{0}")]
    Rejected(String),
    /// No create call got a clear answer (transport errors, timeouts, 5xx).
    #[error("Runpod did not answer the create calls clearly; check `overbrainer pod ls`")]
    Unanswered,
    /// The pod's watchdog could not prove it can delete its pod; the pod was
    /// deleted.
    #[error(
        "the pod's watchdog cannot remove its own pod ({reason}): pod {pod_id} was deleted and training refused"
    )]
    WatchdogRefused {
        /// The pod.
        pod_id: PodId,
        /// The watchdog's verdict.
        reason: String,
    },
    /// The pod's bootstrap failed (its watchdog's verdict was `failed bootstrap:
    /// <reason>`); the pod was deleted.
    #[error("the pod's bootstrap failed ({reason}): pod {pod_id} was deleted and training refused")]
    BootstrapFailed {
        /// The pod.
        pod_id: PodId,
        /// What the bootstrap reported.
        reason: String,
    },
    /// The pod's sshd answered, but the local `ssh` master connection kept ending
    /// right after it started: the cause is on this machine, so no other GPU type
    /// is tried. The pod was deleted.
    #[error(
        "pod {pod_id} answers SSH, but the local ssh cannot keep its connection ({reason}); a wrapper around `ssh` on PATH, such as firejail, may kill its background process: put the real ssh first on PATH. The pod was deleted and no other GPU type tried"
    )]
    LocalSsh {
        /// The pod.
        pod_id: PodId,
        /// Why the connection failed, with the tail of ssh's log.
        reason: String,
    },
    /// The pod's host key cannot be stored as a Runpod secret: no pod was
    /// created. Holds the client's fixed message, never Runpod's text.
    #[error("cannot store the pod's host key as a Runpod secret: {0}; no pod was created")]
    HostKeySecret(String),
    /// Ctrl-C before the job started.
    #[error("interrupted before the job started")]
    Interrupted,
    /// A deleted pod still shows in the API.
    #[error("pod {0} could not be confirmed deleted: check `overbrainer pod ls`")]
    NotDeleted(PodId),
    /// Following the run failed.
    #[error(transparent)]
    Run(#[from] crate::runs::RunError),
    /// `max_hours` ran out: the client's own deadline fired, or the pod's watchdog
    /// deleted the pod at its deadline, before the job ended.
    #[error("max_hours reached: the pod was deleted before the job ended")]
    DeadlineReached,
    /// `max_cost_usd` ran out: the pod's watchdog deleted the pod once it had
    /// spent it, before the job ended.
    #[error("max_cost_usd reached: the pod was deleted before the job ended")]
    CostCapReached,
    /// `max_cost_usd` is set but could not be handed to the pod's watchdog:
    /// nothing would stop the run at its cap, so no job was started.
    #[error("max_cost_usd cannot be applied ({0}): the pod was deleted and training refused")]
    CostCapUnset(String),
    /// The run's pod no longer exists.
    #[error("pod {0} no longer exists")]
    PodGone(PodId),
    /// `pod rm` of a run in progress (preparing or running), without `--force`:
    /// its training pod was kept, and only its other pods were deleted.
    #[error(
        "run {run_id} is still running{kept}; stop it with `overbrainer train cancel {run_id}` first, or pass --force"
    )]
    RunStillRunning {
        /// The run.
        run_id: String,
        /// What was kept and deleted, starting with `: `, or empty.
        kept: String,
    },
    /// `pod rm` of a run still starting its pod, without `--force`: nothing was
    /// deleted, since a pod being provisioned must never be at risk.
    #[error(
        "run {0} is still starting its pod; wait for it, or use `overbrainer train cancel {0}` once it runs, or `pod rm {0} --force`"
    )]
    StillStarting(String),
    /// `pod rm` of a run in progress whose training pod is not recorded, without
    /// `--force`: nothing was deleted, since any pod of the run may be training.
    #[error(
        "run {0} is in progress but its pod is not recorded, so its pods are left alone; if the run is really dead, use `overbrainer pod rm {0} --force`"
    )]
    PodNotRecorded(String),
    /// `pod rm` of a run absent from this project's `runs/`, without `--force`:
    /// its pods may belong to another checkout.
    #[error(
        "run {0} is not in this project's runs/: its pods may belong to another checkout; if none owns it, pass --force"
    )]
    NotInRuns(String),
    /// The pod exists but offers no SSH endpoint yet.
    #[error("pod {0} has no SSH endpoint (status {1}); try again once it runs")]
    NoEndpoint(PodId, String),
    /// The pod is stopped (`EXITED`), and will not run again by itself.
    #[error(
        "pod {pod_id} is stopped and will not run again by itself; remove it with `overbrainer pod rm {run_id}`"
    )]
    PodStopped {
        /// The pod.
        pod_id: PodId,
        /// Its run.
        run_id: String,
    },
}
