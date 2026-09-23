//! Runpod as a training target: a pod created for a run over the REST API (v2),
//! reached over SSH with per-run keys, guarded by a watchdog running on the pod,
//! and deleted once the run's results are retrieved.
//!
//! A Runpod target is not an executor of its own: once its pod is ready, the run
//! goes through the [`SshExecutor`](crate::exec::SshExecutor) and the `native`
//! runtime, like an SSH target.

use std::path::PathBuf;

mod client;
mod keys;
mod record;
mod status;
mod types;

pub use client::{ApiError, RunpodClient, USER_AGENT};
pub use keys::{
    CLIENT_KEY, KNOWN_HOSTS, PodKeys, SSH_CONFIG, SSH_DIR, alias, base64, ssh_config, write_config,
    write_known_hosts,
};
pub use record::{
    Attempt, AttemptResult, DeletedBy, POD_FILE, POD_RECORD_VERSION, PodRecord, PodState,
    SshEndpoint, hours,
};
pub use status::{DeleteReason, PodStatus};
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
}
