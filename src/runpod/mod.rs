//! Runpod as a training target: a pod created for a run over the REST API (v2),
//! reached over SSH with per-run keys, guarded by a watchdog running on the pod,
//! and deleted once the run's results are retrieved.
//!
//! A Runpod target is not an executor of its own: once its pod is ready, the run
//! goes through the [`SshExecutor`](crate::exec::SshExecutor) and the `native`
//! runtime, like an SSH target.

use std::path::PathBuf;

mod bootstrap;
mod client;
mod flow;
mod keys;
mod orphans;
mod provision;
mod record;
mod status;
mod target;
mod types;

pub use bootstrap::{
    HOST_KEY_ENV, JOB_ENV, PodSettings, bootstrap_functions, pod_command, pod_env, watchdog_script,
};
pub use client::{ApiError, RunpodClient, USER_AGENT};
pub use flow::{
    DEADLINE_MARGIN, Ending, RETRIEVED_MARKER, WATCHDOG_LOG, end_pod, follow, forget_client_key,
    job_started, reconnect, ssh_command, start_pod,
};
pub use keys::{
    CLIENT_KEY, KNOWN_HOSTS, PodKeys, SSH_CONFIG, SSH_DIR, alias, base64, ssh_config, write_config,
    write_known_hosts,
};
pub use orphans::{PodRow, Removed, orphan_warnings, pod_rows, remove_run_pods, table};
pub use provision::{
    PodCtx, PodPlan, Provisioned, Timing, chain, provision, remove, sweep, wait_gone,
};
pub use record::{
    Attempt, AttemptResult, DeletedBy, POD_FILE, POD_RECORD_VERSION, PodRecord, PodState,
    SshEndpoint, hours,
};
pub use status::{DeleteReason, PodStatus};
pub use target::{MIN_CUDA_VERSION, RunpodTarget, VOLUME_MOUNT, VOLUME_WORKDIR, WORKDIR};
pub use types::{
    CreateEnv, CreatePod, GpuRequest, Mounts, NetworkMount, Pagination, Pod, PodEnv, PodGpu, PodId,
    PodPage, PodSsh, RemoteStatus, SshDirect,
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
    /// Every GPU type of the target was unavailable.
    #[error("{0}")]
    NoCapacity(String),
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
    /// Ctrl-C before the job started.
    #[error("interrupted before the job started")]
    Interrupted,
    /// A deleted pod still shows in the API.
    #[error("pod {0} could not be confirmed deleted: check `overbrainer pod ls`")]
    NotDeleted(PodId),
    /// Following the run failed.
    #[error(transparent)]
    Run(#[from] crate::runs::RunError),
    /// The client's own deadline fired: the pod was deleted before the job ended.
    #[error("max_hours reached: the pod was deleted before the job ended")]
    DeadlineReached,
    /// The run's pod no longer exists.
    #[error("pod {0} no longer exists")]
    PodGone(PodId),
    /// `pod rm` of a run whose job still runs, without `--force`.
    #[error(
        "run {0} is still running; stop it with `overbrainer train cancel {0}` first, or pass --force"
    )]
    RunStillRunning(String),
    /// The pod exists but offers no SSH endpoint.
    #[error("pod {0} has no SSH endpoint (status {1}); try again once it runs")]
    NoEndpoint(PodId, String),
}
