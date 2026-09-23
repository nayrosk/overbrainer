//! Pod lifecycle events, published on the event bus as `Event::PodStatus`.

use std::time::Duration;

use super::PodId;

/// Why overbrainer deletes a pod.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeleteReason {
    /// The run's results were downloaded and verified.
    Retrieved,
    /// The pod never became reachable over SSH in time, or died while starting.
    NotReady,
    /// The pod's watchdog could not prove it can delete its own pod.
    Refused,
    /// The pod's bootstrap failed: no job can run on it.
    BootstrapFailed,
    /// Ctrl-C before the job started.
    Interrupted,
    /// `max_hours` has passed.
    Deadline,
    /// `overbrainer pod rm`, or a failure after the pod was created.
    Requested,
    /// A second pod of the same run, left by an ambiguous create.
    Duplicate,
}

impl DeleteReason {
    /// The reason in words.
    #[must_use]
    pub fn describe(self) -> &'static str {
        match self {
            Self::Retrieved => "results retrieved",
            Self::NotReady => "not ready in time",
            Self::Refused => "its watchdog cannot delete it",
            Self::BootstrapFailed => "its bootstrap failed",
            Self::Interrupted => "interrupted before the job started",
            Self::Deadline => "max_hours reached",
            Self::Requested => "requested",
            Self::Duplicate => "duplicate of this run's pod",
        }
    }
}

/// A step of a pod's life.
#[derive(Debug, Clone, PartialEq)]
pub enum PodStatus {
    /// A pod is being asked for.
    Creating {
        /// Name of the pod, `overbrainer-<run-id>-<attempt>`.
        name: String,
        /// GPU type asked for.
        gpu_type: String,
    },
    /// A GPU type could not be placed; the next one is tried.
    Unavailable {
        /// The GPU type.
        gpu_type: String,
        /// What Runpod said.
        reason: String,
    },
    /// Runpod created the pod; it is starting.
    Created {
        /// The pod.
        pod_id: PodId,
        /// Its GPU type.
        gpu_type: String,
        /// Its data center, when known.
        data_center: Option<String>,
        /// USD per hour, when known.
        cost_per_hour: Option<f64>,
    },
    /// SSH answers with the pod's host key and its watchdog proved it can delete
    /// the pod.
    Ready {
        /// The pod.
        pod_id: PodId,
        /// Time from creation to ready.
        after: Duration,
        /// When the watchdog deletes the pod at the latest, RFC 3339 UTC; `None`
        /// for a pod kept with `--keep-pod`, which is never deleted automatically
        /// once its job starts.
        deadline: Option<String>,
    },
    /// The pod is being deleted.
    Deleting {
        /// The pod.
        pod_id: PodId,
        /// Why.
        reason: DeleteReason,
    },
    /// Runpod no longer knows the pod.
    Deleted {
        /// The pod.
        pod_id: PodId,
        /// How long it existed, when known.
        uptime: Option<Duration>,
        /// Its rate times its uptime, in USD, when both are known.
        estimated_spend: Option<f64>,
    },
    /// The pod stays, as `--keep-pod` asked.
    Kept {
        /// The pod.
        pod_id: PodId,
    },
}
