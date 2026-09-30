//! Snapshots: a run stopped with a saved checkpoint, the request that asks for
//! one, and the proof its job leaves.
//!
//! Anyone who can write the run directory on the target asks for a snapshot by
//! writing [`SNAPSHOT_REQUEST`] there, holding the reason: overbrainer (`train
//! stop`, the TUI, the Runpod cost cap or the disk policy) or the pod's watchdog.
//! The trainer's plugin then saves a checkpoint at the end of the current step,
//! stops, and writes [`SNAPSHOT_FILE`]; the job exits 0 and the run is recorded
//! [`RunState::Stopped`](super::RunState::Stopped).

use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::{RunError, RunRecord, RunState};
use crate::exec::{ExecError, Executor, JobId, JobStatus};
use crate::train::{SNAPSHOT_FILE, SNAPSHOT_REQUEST};

/// How long a stop waits for the snapshot before it cancels the job instead:
/// a job that neither wrote its proof nor ended by then will not.
pub const STOP_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// Largest `snapshot.json` read from the target.
const PROOF_LIMIT: u64 = 64 * 1024;

/// Why a snapshot was taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SnapshotReason {
    /// Asked for: `overbrainer train stop`, or `s` in the TUI.
    Requested,
    /// The Runpod watchdog's deadline (`max_hours`) was near.
    Deadline,
    /// The Runpod spending cap (`max_cost_usd`) was near.
    Cost,
    /// The target's disk was nearly full.
    Disk,
}

impl SnapshotReason {
    /// Lowercase name, as the request and `run.json` hold it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Requested => "requested",
            Self::Deadline => "deadline",
            Self::Cost => "cost",
            Self::Disk => "disk",
        }
    }

    /// The reason `text` names; any other text, or none, is
    /// [`SnapshotReason::Requested`].
    #[must_use]
    pub fn parse(text: &str) -> Self {
        match text.trim() {
            "deadline" => Self::Deadline,
            "cost" => Self::Cost,
            "disk" => Self::Disk,
            _ => Self::Requested,
        }
    }
}

/// The snapshot of a stopped run, in its `run.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    /// The checkpoint, relative to the run directory, for example
    /// `output/checkpoint-120`.
    pub checkpoint: String,
    /// The optimizer step it was saved at.
    pub step: u64,
    /// Why it was taken.
    pub reason: SnapshotReason,
}

/// `snapshot.json` as the plugin writes it; its `time` is not kept.
#[derive(Debug, Deserialize)]
struct Written {
    checkpoint: String,
    step: u64,
    #[serde(default)]
    reason: Option<String>,
}

/// What the target's `snapshot.json` says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Proof {
    /// No proof: the job was not stopped with a snapshot.
    None,
    /// A snapshot was saved.
    Saved(Snapshot),
    /// The file exists but cannot be used; why.
    Invalid(String),
}

/// Reads the proof `content` of a `snapshot.json`; empty content is no proof.
/// The checkpoint must be a relative path of plain names (letters, digits, `.`,
/// `_`, `-`), never `.` or `..`: it names local directories later.
#[must_use]
pub fn parse_proof(content: &[u8]) -> Proof {
    if content.iter().all(u8::is_ascii_whitespace) {
        return Proof::None;
    }
    let written: Written = match serde_json::from_slice(content) {
        Ok(written) => written,
        Err(error) => {
            return Proof::Invalid(format!(
                "{SNAPSHOT_FILE} is not valid (line {}, column {})",
                error.line(),
                error.column()
            ));
        },
    };
    if !valid_checkpoint(&written.checkpoint) {
        return Proof::Invalid(format!("{SNAPSHOT_FILE} names an invalid checkpoint path"));
    }
    Proof::Saved(Snapshot {
        checkpoint: written.checkpoint,
        step: written.step,
        reason: SnapshotReason::parse(written.reason.as_deref().unwrap_or_default()),
    })
}

/// Whether `path` is relative and made of plain names only.
fn valid_checkpoint(path: &str) -> bool {
    !path.is_empty()
        && path.split('/').all(|part| {
            !part.is_empty()
                && part != "."
                && part != ".."
                && part
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        })
}

/// Reads the proof the job of the run in `remote_dir` left on the target.
///
/// # Errors
///
/// Returns an [`ExecError`] when the target cannot be read.
pub async fn read_proof<E: Executor>(executor: &E, remote_dir: &str) -> Result<Proof, ExecError> {
    let path = format!("{remote_dir}/{SNAPSHOT_FILE}");
    let content = executor.read_from(&path, 0, PROOF_LIMIT).await?;
    Ok(parse_proof(&content))
}

/// Asks the job of the running run `record` for a snapshot, for `reason`: writes
/// the request in its run directory on the target. The job saves a checkpoint at
/// the end of its current step and stops; whoever follows the run records it
/// stopped once the job ends. Asking again changes nothing but the reason.
///
/// # Errors
///
/// Returns [`RunError::NotStarted`] for a run without a job,
/// [`RunError::NotRunning`] for a run that already ended, and
/// [`RunError::Exec`] when the request cannot be written.
pub async fn request_snapshot<E: Executor>(
    executor: &E,
    record: &RunRecord,
    reason: SnapshotReason,
) -> Result<(), RunError> {
    if record.job.is_none() {
        return Err(RunError::NotStarted(record.id.clone()));
    }
    if record.state != RunState::Running {
        return Err(RunError::NotRunning(
            record.id.clone(),
            record.state.name().to_string(),
        ));
    }
    let path = format!("{}/{SNAPSHOT_REQUEST}", record.remote_dir);
    executor.put_file(&path, reason.name()).await?;
    Ok(())
}

/// What a stop does when its snapshot does not come: after `timeout`, a job
/// still running without having written its proof is cancelled, and whoever
/// follows it records it cancelled. It never ends: race it with the watch.
pub async fn stop_fallback<E: Executor>(executor: &E, job: &JobId, timeout: Duration) {
    tokio::time::sleep(timeout).await;
    late_stop(executor, job, timeout).await;
    std::future::pending::<()>().await;
}

/// Cancels `job` when it is still running without a proof `waited` after the
/// request, and says so; a failure is only logged.
async fn late_stop<E: Executor>(executor: &E, job: &JobId, waited: Duration) {
    let note = match cancel_unproven(executor, job).await {
        Ok(true) => format!(
            "no snapshot {} min after it was asked for: the job was cancelled",
            waited.as_secs() / 60
        ),
        Ok(false) => return,
        Err(error) => format!("cannot cancel a job that gave no snapshot: {error}"),
    };
    tracing::warn!("{note}");
}

/// Cancels `job` when it is still running without a proof; whether it did.
async fn cancel_unproven<E: Executor>(executor: &E, job: &JobId) -> Result<bool, ExecError> {
    if read_proof(executor, &job.dir).await? != Proof::None
        || executor.status(job).await? != JobStatus::Running
    {
        return Ok(false);
    }
    executor.cancel(job).await?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_proof_names_its_checkpoint_step_and_reason() {
        let proof = parse_proof(
            br#"{"checkpoint": "output/checkpoint-120", "step": 120, "time": 1.5, "reason": "cost"}"#,
        );
        assert_eq!(
            proof,
            Proof::Saved(Snapshot {
                checkpoint: "output/checkpoint-120".into(),
                step: 120,
                reason: SnapshotReason::Cost,
            })
        );
        let without = parse_proof(br#"{"checkpoint": "output/checkpoint-1", "step": 1}"#);
        assert!(
            matches!(
                without,
                Proof::Saved(Snapshot {
                    reason: SnapshotReason::Requested,
                    ..
                })
            ),
            "{without:?}"
        );
        assert_eq!(parse_proof(b""), Proof::None);
        assert_eq!(parse_proof(b" \n"), Proof::None);
    }

    #[test]
    fn a_proof_never_names_a_path_outside_the_run() {
        for checkpoint in [
            "/etc/passwd",
            "../other",
            "output/../..",
            "output//checkpoint-1",
            "output/./checkpoint-1",
            "output/check point",
            "",
        ] {
            let text = format!(r#"{{"checkpoint": {checkpoint:?}, "step": 1}}"#);
            assert!(
                matches!(parse_proof(text.as_bytes()), Proof::Invalid(_)),
                "{checkpoint:?}"
            );
        }
        assert!(matches!(parse_proof(b"{not json"), Proof::Invalid(_)));
    }

    #[test]
    fn reasons_read_back_and_default_to_requested() {
        for reason in [
            SnapshotReason::Requested,
            SnapshotReason::Deadline,
            SnapshotReason::Cost,
            SnapshotReason::Disk,
        ] {
            assert_eq!(SnapshotReason::parse(reason.name()), reason);
        }
        assert_eq!(
            SnapshotReason::parse(" deadline\n"),
            SnapshotReason::Deadline
        );
        assert_eq!(SnapshotReason::parse("anything"), SnapshotReason::Requested);
    }
}
