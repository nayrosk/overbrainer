//! Snapshots: a run stopped with a saved checkpoint, the request that asks for
//! one, and the proof its job leaves.
//!
//! Anyone who can write the run directory on the target asks for a snapshot by
//! writing [`SNAPSHOT_REQUEST`] there, holding the reason: overbrainer (`train
//! stop`, the TUI, the Runpod cost cap or the disk policy) or the pod's watchdog.
//! The trainer's plugin then saves a checkpoint at the end of the current step,
//! stops, and writes [`SNAPSHOT_FILE`]; the job exits 0 and the run is recorded
//! [`RunState::Stopped`](super::RunState::Stopped).

use std::future::Future;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::{RunError, RunRecord, RunState};
use crate::exec::{ExecError, Executor, JobId, JobStatus};
use crate::train::{OUTPUT_DIR, SNAPSHOT_FILE, SNAPSHOT_REQUEST};

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
/// The checkpoint must be exactly `output/checkpoint-<step>` for its step: it
/// names local directories later.
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
    if written.checkpoint != expected_checkpoint(written.step) {
        return Proof::Invalid(format!("{SNAPSHOT_FILE} names an invalid checkpoint path"));
    }
    Proof::Saved(Snapshot {
        checkpoint: written.checkpoint,
        step: written.step,
        reason: SnapshotReason::parse(written.reason.as_deref().unwrap_or_default()),
    })
}

/// The only checkpoint a proof may name for a snapshot at `step`: where the
/// trainer saves it, `output/checkpoint-<step>`.
fn expected_checkpoint(step: u64) -> String {
    format!("{OUTPUT_DIR}/checkpoint-{step}")
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

/// How long a stop waits before it cancels the job instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StopLimits {
    /// For the proof, from the request: a job that neither wrote it nor ended
    /// by then will not.
    pub proof: Duration,
    /// For the job to end once its proof is there: its checkpoint is saved,
    /// what is left is the final save of the model.
    pub end: Duration,
}

/// The limits of `train stop`: [`STOP_TIMEOUT`] for the proof, then 10 more
/// minutes for the job to end.
pub const STOP_LIMITS: StopLimits = StopLimits {
    proof: STOP_TIMEOUT,
    end: Duration::from_secs(10 * 60),
};

/// Runs `flow`, which follows the job `job` after its snapshot was asked for,
/// and returns what it returns. Beside it runs what a stop does when its
/// snapshot does not come: after `limits.proof`, a job still running without
/// having written its proof is cancelled; one that wrote it but still runs
/// `limits.end` later is cancelled too (its checkpoint is saved). `flow` then
/// sees it end cancelled.
pub async fn with_stop_fallback<E: Executor, F: Future>(
    executor: &E,
    job: &JobId,
    limits: StopLimits,
    flow: F,
) -> F::Output {
    let mut flow = std::pin::pin!(flow);
    tokio::select! {
        output = &mut flow => return output,
        () = tokio::time::sleep(limits.proof) => {},
    }
    if late_check(executor, job, limits.proof).await {
        tokio::select! {
            output = &mut flow => return output,
            () = tokio::time::sleep(limits.end) => {},
        }
        late_stop(executor, job).await;
    }
    flow.await
}

/// How often a follow that did not ask for a snapshot looks for a request
/// someone else wrote: `train stop` from another process, the cost cap or the
/// pod's watchdog.
pub const REQUEST_POLL: Duration = Duration::from_secs(30);

/// Runs `flow`, which follows the job `job`, and returns what it returns.
/// Every `every`, it looks for a snapshot request in the job's run directory;
/// once one is there, the job is handled as [`with_stop_fallback`] does, with
/// `limits` counted from then. A job asked for no snapshot is never cancelled.
pub async fn with_request_watch<E: Executor, F: Future>(
    executor: &E,
    job: &JobId,
    limits: StopLimits,
    every: Duration,
    flow: F,
) -> F::Output {
    let mut flow = std::pin::pin!(flow);
    let mut ticks = tokio::time::interval(every);
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            biased;
            output = &mut flow => return output,
            _ = ticks.tick() => {},
        }
        if requested(executor, &job.dir).await {
            break;
        }
    }
    with_stop_fallback(executor, job, limits, flow).await
}

/// Whether a snapshot was asked of the job of `run_dir`: its request holds a
/// reason. A failure to read it counts as no request and is only logged.
async fn requested<E: Executor>(executor: &E, run_dir: &str) -> bool {
    let path = format!("{run_dir}/{SNAPSHOT_REQUEST}");
    match executor.read_from(&path, 0, 1).await {
        Ok(content) => !content.is_empty(),
        Err(error) => {
            tracing::debug!("cannot look for a snapshot request: {error}");
            false
        },
    }
}

/// Where a stop stands once `waited` passed: a job still running without a
/// proof is cancelled, and said so; returns whether it wrote its proof and
/// still runs, so it gets more time to end. A failure is only logged.
async fn late_check<E: Executor>(executor: &E, job: &JobId, waited: Duration) -> bool {
    let checked = async {
        let proven = read_proof(executor, &job.dir).await? != Proof::None;
        let running = executor.status(job).await? == JobStatus::Running;
        if running && !proven {
            executor.cancel(job).await?;
        }
        Ok::<_, ExecError>((proven, running))
    };
    let note = match checked.await {
        Ok((true, running)) => return running,
        Ok((false, false)) => return false,
        Ok((false, true)) => format!(
            "no snapshot {} min after it was asked for: the job was cancelled",
            waited.as_secs() / 60
        ),
        Err(error) => format!("cannot cancel a job that gave no snapshot: {error}"),
    };
    tracing::warn!("{note}");
    false
}

/// Cancels `job`, which saved its snapshot but did not end in time, when it
/// still runs; a failure is only logged.
async fn late_stop<E: Executor>(executor: &E, job: &JobId) {
    let stopped = async {
        if executor.status(job).await? != JobStatus::Running {
            return Ok(false);
        }
        executor.cancel(job).await?;
        Ok::<_, ExecError>(true)
    };
    let note = match stopped.await {
        Ok(true) => "the job saved its snapshot but did not end: it was cancelled".to_string(),
        Ok(false) => return,
        Err(error) => format!("cannot cancel a job that did not end: {error}"),
    };
    tracing::warn!("{note}");
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
            "output/other-1",
            "output/checkpoint-2",
            "output/checkpoint-1/x",
            "checkpoint-1",
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
