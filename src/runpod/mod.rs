//! Runpod as a training target: a pod created for a run over the REST API (v2),
//! reached over SSH with per-run keys, guarded by a watchdog running on the pod,
//! and deleted once the run's results are retrieved.
//!
//! A Runpod target is not an executor of its own: once its pod is ready, the run
//! goes through the [`SshExecutor`](crate::exec::SshExecutor) and the `native`
//! runtime, like an SSH target.

mod client;
mod record;
mod status;
mod types;

pub use client::{ApiError, RunpodClient, USER_AGENT};
pub use record::{
    Attempt, AttemptResult, DeletedBy, POD_FILE, POD_RECORD_VERSION, PodRecord, PodState,
    SshEndpoint, hours,
};
pub use status::{DeleteReason, PodStatus};
pub use types::{
    CreateEnv, CreatePod, GpuRequest, Mounts, NetworkMount, Pagination, Pod, PodEnv, PodGpu, PodId,
    PodPage, PodSsh, RemoteStatus, SshDirect,
};
