//! `runs/<run-id>/pod.json`: what overbrainer knows of a run's pod, written before
//! every call that can create or delete one, so a crash never loses a pod.

use std::fs;
use std::io;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::{Pod, PodId};
use crate::runs::{Runs, RunsError, rfc3339};

/// The pod record of a run, in its local run directory.
pub const POD_FILE: &str = "pod.json";

/// Format version of [`POD_FILE`].
pub const POD_RECORD_VERSION: u32 = 1;

/// Where a run's pod stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PodState {
    /// A create call is about to be sent, or was sent.
    Creating,
    /// Runpod created the pod; overbrainer waits for SSH and the watchdog.
    Provisioning,
    /// Reachable, with a watchdog that can delete it.
    Ready,
    /// The job was started on it.
    Running,
    /// The job ended but its results were not retrieved; the watchdog deletes the
    /// pod once its retrieve grace has passed.
    AwaitingRetrieval,
    /// Kept with `--keep-pod`: only `overbrainer pod rm` deletes it.
    Kept,
    /// A delete call is about to be sent, or was sent.
    Deleting,
    /// Runpod no longer knows the pod.
    Deleted,
}

impl PodState {
    /// The state in words.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Creating => "creating",
            Self::Provisioning => "provisioning",
            Self::Ready => "ready",
            Self::Running => "running",
            Self::AwaitingRetrieval => "awaiting retrieval",
            Self::Kept => "kept",
            Self::Deleting => "deleting",
            Self::Deleted => "deleted",
        }
    }
}

/// How one create call ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptResult {
    /// Sent; no answer recorded yet.
    Sent,
    /// Runpod created the pod.
    Created,
    /// Runpod could not place the GPU type (400).
    Unavailable,
    /// Runpod refused the GPU type (403).
    Forbidden,
    /// No usable answer: the pod may or may not exist.
    Ambiguous,
    /// No usable answer, then found by its run marker.
    Adopted,
    /// Created, but not reachable in time; deleted.
    NotReady,
    /// Created, but its watchdog could not prove it can delete it; deleted.
    Refused,
    /// Refused by Runpod for a reason no other GPU type would fix (402, 422).
    Rejected,
}

/// One create call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attempt {
    /// The pod name sent, `overbrainer-<run-id>-<n>`.
    pub name: String,
    /// The GPU type asked for.
    pub gpu_type: String,
    /// When the call was sent, RFC 3339 UTC.
    pub sent_at: String,
    /// When this pod's watchdog deletes it, in Unix seconds; `None` when kept.
    pub deadline_unix: Option<u64>,
    /// How it ended.
    pub result: AttemptResult,
    /// What Runpod said, for a failed call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// The pod it created, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pod_id: Option<PodId>,
}

/// Who removed a pod.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeletedBy {
    /// overbrainer, during `train`, `train attach` or `train cancel`.
    Client,
    /// The pod's watchdog, found out later.
    Watchdog,
    /// `overbrainer pod rm`.
    PodRm,
    /// Found gone without knowing why.
    Unknown,
}

/// The SSH endpoint of a pod as last seen.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SshEndpoint {
    /// Public IP or host name.
    pub host: String,
    /// Public port.
    pub port: u16,
    /// User.
    pub user: String,
}

/// `runs/<run-id>/pod.json`. It never holds a private key, the API key or the
/// pod's environment: only the public host key and what Runpod reported.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PodRecord {
    /// [`POD_RECORD_VERSION`].
    pub version: u32,
    /// The run.
    pub run_id: String,
    /// Where the pod stands.
    pub state: PodState,
    /// Whether `--keep-pod` was given.
    pub keep: bool,
    /// When the watchdog deletes the current pod at the latest, RFC 3339 UTC;
    /// `None` when kept.
    pub deadline: Option<String>,
    /// [`PodRecord::deadline`] in Unix seconds.
    pub deadline_unix: Option<u64>,
    /// Every create call, in order.
    pub attempts: Vec<Attempt>,
    /// The current pod, once created.
    pub pod_id: Option<PodId>,
    /// Its GPU type.
    pub gpu_type: Option<String>,
    /// GPUs asked for.
    pub gpu_count: u32,
    /// Its data center.
    pub data_center_id: Option<String>,
    /// USD per hour, the first non-zero rate Runpod reported.
    pub cost_per_hour: Option<f64>,
    /// When it was created, RFC 3339 UTC, from the local clock.
    pub created_at: Option<String>,
    /// [`PodRecord::created_at`] in Unix seconds.
    pub created_unix: Option<u64>,
    /// When it became ready, RFC 3339 UTC.
    pub ready_at: Option<String>,
    /// Its SSH endpoint as last seen.
    pub ssh: Option<SshEndpoint>,
    /// The pod's public host key, `ssh-ed25519 AAAA...`.
    pub host_key: String,
    /// When it was found deleted, RFC 3339 UTC.
    pub deleted_at: Option<String>,
    /// Who deleted it.
    pub deleted_by: Option<DeletedBy>,
    /// Rate times uptime, in USD.
    pub estimated_spend: Option<f64>,
}

/// Seconds since the Unix epoch at `time`.
fn unix(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

/// `max_hours` as a duration, zero when it cannot be one (the configuration
/// rejects such values).
#[must_use]
pub fn hours(max_hours: f64) -> Duration {
    Duration::try_from_secs_f64(max_hours * 3600.0).unwrap_or_default()
}

impl PodRecord {
    /// The record of a run whose pod does not exist yet.
    #[must_use]
    pub fn new(run_id: &str, keep: bool, gpu_count: u32, host_key: &str) -> Self {
        Self {
            version: POD_RECORD_VERSION,
            run_id: run_id.to_string(),
            state: PodState::Creating,
            keep,
            deadline: None,
            deadline_unix: None,
            attempts: Vec::new(),
            pod_id: None,
            gpu_type: None,
            gpu_count,
            data_center_id: None,
            cost_per_hour: None,
            created_at: None,
            created_unix: None,
            ready_at: None,
            ssh: None,
            host_key: host_key.to_string(),
            deleted_at: None,
            deleted_by: None,
            estimated_spend: None,
        }
    }

    /// Reads the pod record of run `id`, `None` when it has none.
    ///
    /// # Errors
    ///
    /// Returns [`RunsError::InvalidId`] for an invalid run ID, and
    /// [`RunsError::Io`] or [`RunsError::Invalid`] when the file cannot be read.
    pub fn load(runs: &Runs, id: &str) -> Result<Option<Self>, RunsError> {
        let path = runs.run_dir(id)?.join(POD_FILE);
        let content = match fs::read(&path) {
            Ok(content) => content,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(source) => return Err(RunsError::Io { path, source }),
        };
        serde_json::from_slice(&content)
            .map(Some)
            .map_err(|source| RunsError::Invalid { path, source })
    }

    /// Writes the record, replacing the file atomically.
    ///
    /// # Errors
    ///
    /// Returns [`RunsError::InvalidId`] for an invalid run ID, and
    /// [`RunsError::Io`] or [`RunsError::Invalid`] when it cannot be written.
    pub fn save(&self, runs: &Runs) -> Result<(), RunsError> {
        let dir = runs.run_dir(&self.run_id)?;
        fs::create_dir_all(&dir).map_err(|source| RunsError::Io {
            path: dir.clone(),
            source,
        })?;
        let path = dir.join(POD_FILE);
        let tmp = dir.join(format!(".{POD_FILE}.tmp"));
        let mut content = serde_json::to_vec_pretty(self).map_err(|source| RunsError::Invalid {
            path: path.clone(),
            source,
        })?;
        content.push(b'\n');
        write(&tmp, &content)?;
        fs::rename(&tmp, &path).map_err(|source| RunsError::Io { path, source })
    }

    /// Appends a create call for `gpu_type`, sent `now`, and returns it. Its pod
    /// would be deleted by its watchdog `max_hours` from now, unless kept.
    pub fn begin_attempt(&mut self, gpu_type: &str, now: SystemTime, max_hours: f64) -> &Attempt {
        let name = format!("overbrainer-{}-{}", self.run_id, self.attempts.len() + 1);
        let deadline_unix = (!self.keep).then(|| unix(now) + hours(max_hours).as_secs());
        self.state = PodState::Creating;
        self.attempts.push(Attempt {
            name,
            gpu_type: gpu_type.to_string(),
            sent_at: rfc3339(now),
            deadline_unix,
            result: AttemptResult::Sent,
            detail: None,
            pod_id: None,
        });
        let last = self.attempts.len() - 1;
        &self.attempts[last]
    }

    /// Records how the last create call ended.
    pub fn end_attempt(&mut self, result: AttemptResult, detail: Option<String>) {
        if let Some(attempt) = self.attempts.last_mut() {
            attempt.result = result;
            attempt.detail = detail;
        }
    }

    /// Makes `pod`, created (or adopted) by the last call at `now`, the current
    /// pod: its deadline is that call's.
    pub fn created(&mut self, pod: &Pod, result: AttemptResult, now: SystemTime) {
        let deadline_unix = self
            .attempts
            .last()
            .and_then(|attempt| attempt.deadline_unix);
        if let Some(attempt) = self.attempts.last_mut() {
            attempt.result = result;
            attempt.pod_id = Some(pod.id.clone());
            self.gpu_type = Some(
                pod.gpu_type()
                    .map_or_else(|| attempt.gpu_type.clone(), str::to_string),
            );
        }
        self.state = PodState::Provisioning;
        self.pod_id = Some(pod.id.clone());
        self.data_center_id.clone_from(&pod.data_center_id);
        self.cost_per_hour = pod.rate();
        self.created_at = Some(rfc3339(now));
        self.created_unix = Some(unix(now));
        self.deadline_unix = deadline_unix;
        self.deadline = deadline_unix.map(|at| rfc3339(UNIX_EPOCH + Duration::from_secs(at)));
        self.ready_at = None;
        self.ssh = None;
    }

    /// Keeps the first non-zero rate Runpod reports.
    pub fn note_rate(&mut self, pod: &Pod) {
        if self.cost_per_hour.is_none() {
            self.cost_per_hour = pod.rate();
        }
    }

    /// Forgets the current pod once it is deleted, before another attempt.
    pub fn forget_pod(&mut self) {
        self.pod_id = None;
        self.state = PodState::Creating;
        self.created_at = None;
        self.created_unix = None;
        self.cost_per_hour = None;
        self.deadline = None;
        self.deadline_unix = None;
        self.ssh = None;
    }

    /// The pod is reachable at `endpoint` with a working watchdog.
    pub fn ready(&mut self, endpoint: SshEndpoint, now: SystemTime) {
        self.state = PodState::Ready;
        self.ssh = Some(endpoint);
        self.ready_at = Some(rfc3339(now));
    }

    /// How long the current pod has existed at `now`.
    #[must_use]
    pub fn uptime(&self, now: SystemTime) -> Option<Duration> {
        let created = self.created_unix?;
        Some(Duration::from_secs(unix(now).saturating_sub(created)))
    }

    /// The pod was found gone at `now`, removed by `by`: the spend is estimated
    /// from its rate and uptime, to a hundredth of a cent.
    pub fn deleted(&mut self, by: DeletedBy, now: SystemTime) {
        self.estimated_spend = match (self.cost_per_hour, self.uptime(now)) {
            // Rounded to a hundredth of a cent, which also keeps the number short
            // enough to read back from `pod.json` exactly.
            (Some(rate), Some(uptime)) => {
                Some((rate * uptime.as_secs_f64() / 3600.0 * 10_000.0).round() / 10_000.0)
            },
            _ => None,
        };
        self.state = PodState::Deleted;
        self.deleted_at = Some(rfc3339(now));
        self.deleted_by = Some(by);
    }

    /// Whether the watchdog may still hold the pod after `now` + `margin`.
    #[must_use]
    pub fn past_deadline(&self, now: SystemTime, margin: Duration) -> bool {
        self.deadline_unix
            .is_some_and(|deadline| unix(now) >= deadline.saturating_add(margin.as_secs()))
    }

    /// One line for `runs ls`: `pod running k3x9abc $0.53/h`, `pod deleted about
    /// $0.64`, `pod kept k3x9abc`, and so on.
    #[must_use]
    pub fn summary(&self) -> String {
        let id = self
            .pod_id
            .as_ref()
            .map_or_else(|| "(no pod)".to_string(), ToString::to_string);
        match self.state {
            PodState::Deleted => match self.estimated_spend {
                Some(spend) => format!("pod deleted about ${spend:.2}"),
                None => "pod deleted".to_string(),
            },
            PodState::Kept | PodState::AwaitingRetrieval => {
                format!("pod {} {id}", self.state.name())
            },
            _ => match self.cost_per_hour {
                Some(rate) => format!("pod {} {id} ${rate:.2}/h", self.state.name()),
                None => format!("pod {} {id}", self.state.name()),
            },
        }
    }
}

fn write(path: &Path, content: &[u8]) -> Result<(), RunsError> {
    fs::write(path, content).map_err(|source| RunsError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    const RUN: &str = "20260922-143005-a1b2";

    fn at(seconds: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(seconds)
    }

    fn pod() -> Result<Pod, serde_json::Error> {
        serde_json::from_str(
            r#"{"id": "k3x9abc", "name": "overbrainer-20260922-143005-a1b2-2", "status": "RUNNING",
                "cost": 0.53, "dataCenterId": "EU-RO-1", "gpu": {"id": "NVIDIA RTX A6000", "count": 1}}"#,
        )
    }

    #[test]
    fn a_pod_goes_from_attempts_to_deleted_with_its_spend() -> TestResult {
        let mut record = PodRecord::new(RUN, false, 1, "ssh-ed25519 AAAAhost");
        let first = record.begin_attempt("NVIDIA GeForce RTX 4090", at(1_790_000_000), 6.0);
        assert_eq!(first.name, "overbrainer-20260922-143005-a1b2-1");
        assert_eq!(first.deadline_unix, Some(1_790_021_600));
        record.end_attempt(AttemptResult::Unavailable, Some("no capacity".into()));
        let second = record.begin_attempt("NVIDIA RTX A6000", at(1_790_000_002), 6.0);
        assert_eq!(second.name, "overbrainer-20260922-143005-a1b2-2");
        record.created(&pod()?, AttemptResult::Created, at(1_790_000_003));
        assert_eq!(record.state, PodState::Provisioning);
        assert_eq!(record.pod_id.as_ref().map(PodId::as_str), Some("k3x9abc"));
        assert_eq!(record.deadline.as_deref(), Some("2026-09-21T20:13:22Z"));
        assert_eq!(record.gpu_type.as_deref(), Some("NVIDIA RTX A6000"));
        assert_eq!(record.summary(), "pod provisioning k3x9abc $0.53/h");
        record.ready(
            SshEndpoint {
                host: "203.0.113.7".into(),
                port: 40122,
                user: "root".into(),
            },
            at(1_790_000_224),
        );
        record.state = PodState::Running;
        assert_eq!(record.summary(), "pod running k3x9abc $0.53/h");
        assert!(!record.past_deadline(at(1_790_021_901), Duration::from_secs(300)));
        assert!(record.past_deadline(at(1_790_021_902), Duration::from_secs(300)));
        record.deleted(DeletedBy::Client, at(1_790_004_323));
        assert_eq!(record.summary(), "pod deleted about $0.64");
        assert_eq!(record.deleted_by, Some(DeletedBy::Client));
        Ok(())
    }

    #[test]
    fn a_kept_pod_has_no_deadline() -> TestResult {
        let mut record = PodRecord::new(RUN, true, 1, "ssh-ed25519 AAAAhost");
        let attempt = record.begin_attempt("NVIDIA A40", at(1_790_000_000), 6.0);
        assert_eq!(attempt.deadline_unix, None);
        record.created(&pod()?, AttemptResult::Created, at(1_790_000_001));
        assert_eq!(record.deadline, None);
        assert!(!record.past_deadline(at(u64::from(u32::MAX)), Duration::ZERO));
        record.state = PodState::Kept;
        assert_eq!(record.summary(), "pod kept k3x9abc");
        Ok(())
    }

    #[test]
    fn the_record_round_trips_atomically_and_holds_no_secret() -> TestResult {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        assert_eq!(PodRecord::load(&runs, RUN)?, None);
        let mut record = PodRecord::new(RUN, false, 1, "ssh-ed25519 AAAAhost");
        record.begin_attempt("NVIDIA A40", at(1_790_000_000), 1.5);
        record.save(&runs)?;
        assert_eq!(PodRecord::load(&runs, RUN)?, Some(record.clone()));
        let text = fs::read_to_string(runs.run_dir(RUN)?.join(POD_FILE))?;
        assert!(text.contains("\"state\": \"creating\""), "{text}");
        assert!(text.contains("\"result\": \"sent\""), "{text}");
        assert!(!runs.run_dir(RUN)?.join(".pod.json.tmp").exists());
        assert!(matches!(
            PodRecord::load(&runs, "../x"),
            Err(RunsError::InvalidId(_))
        ));
        Ok(())
    }

    #[test]
    fn forgetting_a_pod_clears_what_belonged_to_it() -> TestResult {
        let mut record = PodRecord::new(RUN, false, 1, "ssh-ed25519 AAAAhost");
        record.begin_attempt("NVIDIA A40", at(1_790_000_000), 1.0);
        record.created(&pod()?, AttemptResult::Created, at(1_790_000_001));
        record.end_attempt(AttemptResult::NotReady, Some("no SSH".into()));
        record.forget_pod();
        assert_eq!(record.pod_id, None);
        assert_eq!(record.state, PodState::Creating);
        assert_eq!(record.attempts[0].result, AttemptResult::NotReady);
        assert_eq!(
            record.attempts[0].pod_id.as_ref().map(PodId::as_str),
            Some("k3x9abc")
        );
        Ok(())
    }
}
