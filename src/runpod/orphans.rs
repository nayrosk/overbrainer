//! `overbrainer pod ls` and `pod rm`, and the orphan warning of `train`: the pods
//! of the account that overbrainer created, matched with `runs/`.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use crate::runs::{RECORD_FILE, RunRecord, RunState, Runs, RunsError};

use super::flow::{Look, look_up_all, mark_gone};
use super::provision::{delete_confirmed, one_line};
use super::{
    DeleteReason, DeletedBy, Pod, PodCtx, PodError, PodId, PodRecord, PodState, forget_client_key,
    remove,
};

/// Characters kept of a table cell taken from the Runpod account.
const CELL_CHARS: usize = 64;

/// `text`, from the Runpod account or a local record, as a table cell: one
/// short line of printable ASCII (see [`one_line`]).
fn cell_text(text: &str) -> String {
    one_line(text, CELL_CHARS)
}

/// The note of a stray pod: another pod of a run, left by an ambiguous create,
/// whose delete could not be confirmed (`stray_pods` in `pod.json`), or any pod
/// carrying the run's marker that `pod.json` does not record.
const STRAY: &str = "stray, not deleted";

/// The note of a pod named like overbrainer's without a usable run marker.
const NO_MARKER: &str = "no run marker";

/// The message of a forced `pod rm` that deleted the pod of a run in progress.
const FAILED_BY_POD_RM: &str = "pod deleted by `pod rm` before its results were retrieved";

/// What overbrainer knows of a pod of `pod ls`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    /// Its run is preparing or running.
    InProgress,
    /// Its run has ended, and the pod is still there.
    Ended,
    /// Kept with `--keep-pod`: only `pod rm` deletes it.
    Kept,
    /// Its results were not retrieved; its watchdog deletes it later.
    AwaitingRetrieval,
    /// One of its run's `stray_pods`, or any pod carrying its run's marker other
    /// than the one in its `pod.json`.
    Stray,
    /// A stray, beside the run's pod kept or awaiting retrieval: `pod rm` of the
    /// run deletes that pod too.
    StrayBesideKept,
    /// Its run is not in this project's `runs/`: it may belong to another
    /// checkout.
    NotInRuns,
    /// Its run directory is here, but its run record cannot be read.
    Unreadable,
    /// Named like overbrainer's pods, but without a usable run marker.
    NoMarker,
    /// Recorded in `runs/`, and confirmed gone.
    Gone,
}

/// A row of `overbrainer pod ls`.
#[derive(Debug, Clone, PartialEq)]
pub struct PodRow {
    /// The run the pod belongs to, from its marker; `-` without one.
    pub run: String,
    /// The pod.
    pub pod_id: String,
    /// What Runpod says it is doing, `GONE` for a recorded pod it no longer knows,
    /// or `UNCHECKED` when the look at a recorded pod failed.
    pub status: String,
    /// Its GPU type.
    pub gpu: String,
    /// USD per hour.
    pub rate: Option<f64>,
    /// When it was created.
    pub created: String,
    /// What overbrainer knows of it, in words (see [`pod_rows`]).
    pub note: String,
    /// The same, for code.
    pub kind: RowKind,
}

impl PodRow {
    /// Whether nothing will delete this pod on its own: its run has ended, it is
    /// a stray, its run is not in this project, or it has no run marker. A kept
    /// pod, or one waiting for its results to be retrieved, is not.
    #[must_use]
    pub fn is_orphan(&self) -> bool {
        matches!(
            self.kind,
            RowKind::Ended
                | RowKind::Stray
                | RowKind::StrayBesideKept
                | RowKind::NotInRuns
                | RowKind::NoMarker
        )
    }
}

/// Every pod of the account whose run marker is set or whose name starts with
/// `overbrainer-`, plus every pod recorded in `runs/` (the run's pod and its
/// `stray_pods`) that the list does not show. Those are looked up together
/// (see `look_up_all`) and declared gone only after three 404s in a row and a
/// list without them (never by a delete): the run's pod is then recorded
/// `deleted` in `pod.json`, and a stray is dropped from `stray_pods`. A
/// `pod.json` that cannot be read is skipped with a warning.
///
/// The note of a row says `run <state>` for a run in progress, `run <state>, not
/// deleted` for an ended run, `kept`, `awaiting retrieval`, `stray, not deleted`
/// (a pod of `stray_pods`, or any pod carrying the run's marker other than the
/// one `pod.json` records), `no run marker`, `run record unreadable`, `gone`,
/// or, for a run not in this project's `runs/`, how to remove it if no other
/// checkout owns it.
///
/// # Errors
///
/// Returns a [`PodError`] when the API or `runs/` cannot be read.
pub async fn pod_rows(ctx: &PodCtx<'_>) -> Result<Vec<PodRow>, PodError> {
    let pods = ctx.client.list_pods().await?;
    let known = Known::load(ctx.runs)?;
    let mut rows = known.listed(&pods);
    let unlisted = known.unlisted(&pods);
    let ids: Vec<PodId> = unlisted.iter().map(|entry| entry.id.clone()).collect();
    let looks = look_up_all(ctx, &ids).await;
    for (entry, look) in unlisted.iter().zip(looks) {
        if let Some(row) = settle(ctx, &known, entry, look)? {
            rows.push(row);
        }
    }
    rows.sort_by(|a, b| (&a.run, &a.pod_id).cmp(&(&b.run, &b.pod_id)));
    Ok(rows)
}

/// The rows of the pods that one list of the account shows, as [`pod_rows`]
/// makes them, with no look at any single pod and nothing recorded: what the
/// orphan warning of `train` is built from. A recorded pod the list does not
/// show has no row: it is not billing, or `pod ls` finds it.
///
/// # Errors
///
/// Returns a [`PodError`] when the API or `runs/` cannot be read.
pub async fn listed_rows(ctx: &PodCtx<'_>) -> Result<Vec<PodRow>, PodError> {
    let pods = ctx.client.list_pods().await?;
    let mut rows = Known::load(ctx.runs)?.listed(&pods);
    rows.sort_by(|a, b| (&a.run, &a.pod_id).cmp(&(&b.run, &b.pod_id)));
    Ok(rows)
}

/// A recorded pod the list does not show.
struct Unlisted {
    run_id: String,
    id: PodId,
    stray: bool,
}

/// What `runs/` holds, read once.
struct Known<'a> {
    runs: &'a Runs,
    run_records: HashMap<String, RunRecord>,
    pod_records: HashMap<String, PodRecord>,
}

impl<'a> Known<'a> {
    /// Loads the project's run records and their readable pod records once.
    /// An unreadable `pod.json` is warned about and omitted so account pods can
    /// still be listed and classified from the remaining evidence.
    fn load(runs: &'a Runs) -> Result<Self, PodError> {
        let run_records: HashMap<String, RunRecord> = runs
            .list()?
            .into_iter()
            .map(|run| (run.id.clone(), run))
            .collect();
        let mut pod_records = HashMap::new();
        for id in run_records.keys() {
            match PodRecord::load(runs, id) {
                Ok(Some(record)) => {
                    pod_records.insert(id.clone(), record);
                },
                Ok(None) => {},
                Err(error) => {
                    tracing::warn!("skipping the unreadable pod record of run {id}: {error}");
                },
            }
        }
        Ok(Self {
            runs,
            run_records,
            pod_records,
        })
    }

    /// The rows of the pods of `pods` that overbrainer created: those with a run
    /// marker or named `overbrainer-*`.
    fn listed(&self, pods: &[Pod]) -> Vec<PodRow> {
        pods.iter()
            .filter(|pod| pod.run_id().is_some() || pod.name.starts_with("overbrainer-"))
            .map(|pod| self.row(pod.run_id(), pod))
            .collect()
    }

    /// The recorded pods, and strays, that `pods` does not show.
    fn unlisted(&self, pods: &[Pod]) -> Vec<Unlisted> {
        let listed = |id: &PodId| pods.iter().any(|pod| pod.id == *id);
        let mut unlisted = Vec::new();
        for (run_id, record) in &self.pod_records {
            if let Some(id) = &record.pod_id
                && record.state != PodState::Deleted
                && !listed(id)
            {
                unlisted.push(Unlisted {
                    run_id: run_id.clone(),
                    id: id.clone(),
                    stray: false,
                });
            }
            for id in &record.stray_pods {
                if !listed(id) && record.pod_id.as_ref() != Some(id) {
                    unlisted.push(Unlisted {
                        run_id: run_id.clone(),
                        id: id.clone(),
                        stray: true,
                    });
                }
            }
        }
        unlisted
    }

    fn is_stray(&self, id: &PodId) -> bool {
        self.pod_records
            .values()
            .any(|record| record.stray_pods.contains(id))
    }

    /// What `runs/` says about the pod `id` of the run `run`.
    fn classify(&self, run: Option<&str>, id: &PodId) -> (RowKind, String) {
        let Some((run, dir)) = run.and_then(|run| Some((run, self.runs.run_dir(run).ok()?))) else {
            return (RowKind::NoMarker, NO_MARKER.to_string());
        };
        let Some(record) = self.run_records.get(run) else {
            if dir.join(RECORD_FILE).exists() {
                return (RowKind::Unreadable, "run record unreadable".to_string());
            }
            return (
                RowKind::NotInRuns,
                format!(
                    "not in this project's runs/; if no other checkout owns it, `overbrainer pod rm {run} --force`"
                ),
            );
        };
        let pod_record = self.pod_records.get(run);
        let pod_state = pod_record.map(|pod| pod.state);
        // A pod carrying the marker that is not the run's recorded pod (an extra
        // pod of an unclear create) is a stray too: the run's state says nothing
        // about it, and beside a kept pod no watchdog would ever delete it.
        let extra = pod_record
            .and_then(|pod| pod.pod_id.as_ref())
            .is_some_and(|recorded| recorded != id);
        if extra || self.is_stray(id) {
            let kept = matches!(
                pod_state,
                Some(PodState::Kept | PodState::AwaitingRetrieval)
            );
            let kind = if kept {
                RowKind::StrayBesideKept
            } else {
                RowKind::Stray
            };
            return (kind, STRAY.to_string());
        }
        match (pod_state, record.state) {
            (Some(PodState::Kept), _) => (RowKind::Kept, "kept".to_string()),
            (Some(PodState::AwaitingRetrieval), _) => {
                (RowKind::AwaitingRetrieval, "awaiting retrieval".to_string())
            },
            (_, state @ (RunState::Preparing | RunState::Running)) => {
                (RowKind::InProgress, format!("run {}", state.name()))
            },
            (_, ended) => (RowKind::Ended, format!("run {}, not deleted", ended.name())),
        }
    }

    /// The row of `pod`, a pod of the run `run`.
    fn row(&self, run: Option<&str>, pod: &Pod) -> PodRow {
        let (kind, note) = self.classify(run, &pod.id);
        PodRow {
            run: cell_text(run.unwrap_or("-")),
            pod_id: pod.id.to_string(),
            status: pod.status.name().to_string(),
            gpu: cell_text(pod.gpu_type().unwrap_or("-")),
            rate: pod.cost,
            created: cell_text(pod.created_at.as_deref().unwrap_or_default()),
            note,
            kind,
        }
    }

    /// The row of the recorded pod `entry`, from what `pod.json` says.
    fn recorded_row(&self, entry: &Unlisted, status: &str, kind: RowKind, note: String) -> PodRow {
        let record = self.pod_records.get(&entry.run_id).filter(|_| !entry.stray);
        PodRow {
            run: cell_text(&entry.run_id),
            pod_id: entry.id.to_string(),
            status: status.to_string(),
            gpu: cell_text(
                record
                    .and_then(|record| record.gpu_type.as_deref())
                    .unwrap_or("-"),
            ),
            rate: None,
            created: cell_text(
                record
                    .and_then(|record| record.created_at.as_deref())
                    .unwrap_or_default(),
            ),
            note,
            kind,
        }
    }
}

/// The row of an unlisted recorded pod once looked up, recording it gone (or
/// dropping it from `stray_pods`) when it is.
fn settle(
    ctx: &PodCtx<'_>,
    known: &Known<'_>,
    entry: &Unlisted,
    look: Look,
) -> Result<Option<PodRow>, PodError> {
    match look {
        Look::Found(pod) => Ok(Some(known.row(Some(&entry.run_id), &pod))),
        Look::Gone if entry.stray => {
            forget_stray(ctx.runs, &entry.run_id, &entry.id)?;
            Ok(None)
        },
        Look::Gone => {
            record_gone(ctx, &entry.run_id, &entry.id)?;
            let row = known.recorded_row(entry, "GONE", RowKind::Gone, "gone".to_string());
            Ok(Some(row))
        },
        Look::Failed(error) => {
            tracing::warn!(
                "cannot look up pod {} of run {}: {error}",
                entry.id,
                entry.run_id
            );
            let (kind, note) = known.classify(Some(&entry.run_id), &entry.id);
            Ok(Some(known.recorded_row(entry, "UNCHECKED", kind, note)))
        },
    }
}

/// Records the pod `id` of the run `run_id` gone, on a fresh `pod.json`, when it
/// is still the run's pod.
fn record_gone(ctx: &PodCtx<'_>, run_id: &str, id: &PodId) -> Result<(), PodError> {
    if let Some(mut record) = PodRecord::load(ctx.runs, run_id)?
        && record.pod_id.as_ref() == Some(id)
        && record.state != PodState::Deleted
    {
        mark_gone(ctx, &mut record, id.clone(), DeletedBy::Unknown)?;
    }
    Ok(())
}

/// Drops the stray pod `id`, confirmed gone, from the run's `pod.json`.
fn forget_stray(runs: &Runs, run_id: &str, id: &PodId) -> Result<(), PodError> {
    edit_record(runs, run_id, |record| {
        record.stray_pods.retain(|stray| stray != id);
    })?;
    tracing::info!("stray pod {id} of run {run_id} is gone");
    Ok(())
}

/// Applies `edit` to the run's `pod.json`, read again just before, and saves
/// it, so that a concurrent writer is not overwritten with a stale copy.
/// Nothing when the run has no `pod.json`.
fn edit_record(
    runs: &Runs,
    run_id: &str,
    edit: impl FnOnce(&mut PodRecord),
) -> Result<(), PodError> {
    if let Some(mut record) = PodRecord::load(runs, run_id)? {
        edit(&mut record);
        record.save(runs)?;
    }
    Ok(())
}

/// The rows as the aligned table `pod ls` prints, header first; nothing when
/// there is no row.
#[must_use]
pub fn table(rows: &[PodRow]) -> Vec<String> {
    if rows.is_empty() {
        return Vec::new();
    }
    let mut lines = vec![format!(
        "{:<20}  {:<14}  {:<12}  {:<24}  {:>5}  {:<20}  NOTE",
        "RUN", "POD", "STATUS", "GPU", "$/H", "CREATED"
    )];
    for row in rows {
        let rate = row
            .rate
            .map_or_else(|| "-".to_string(), |rate| format!("{rate:.2}"));
        lines.push(format!(
            "{:<20}  {:<14}  {:<12}  {:<24}  {rate:>5}  {:<20}  {}",
            row.run, row.pod_id, row.status, row.gpu, row.created, row.note
        ));
    }
    lines
}

/// One warning per orphan pod among `rows`, saying how to remove it. Never
/// deletes anything.
#[must_use]
pub fn orphan_warnings(rows: &[PodRow]) -> Vec<String> {
    rows.iter()
        .filter(|row| row.is_orphan())
        .map(|row| {
            let rate = row
                .rate
                .map_or_else(String::new, |rate| format!(" at ${rate:.2}/h"));
            match row.kind {
                RowKind::NoMarker => format!(
                    "pod {} is named like an overbrainer pod but has no run marker; if it is yours, delete it from the Runpod console",
                    row.pod_id
                ),
                RowKind::NotInRuns => {
                    format!("pod {} is still on Runpod{rate}: {}", row.pod_id, row.note)
                },
                RowKind::StrayBesideKept => format!(
                    "pod {} ({}) is still on Runpod{rate}: remove it with `overbrainer pod rm {}` (this also deletes the run's kept pod)",
                    row.pod_id, row.note, row.run
                ),
                _ => format!(
                    "pod {} ({}) is still on Runpod{rate}: remove it with `overbrainer pod rm {}`",
                    row.pod_id, row.note, row.run
                ),
            }
        })
        .collect()
}

/// A pod `pod rm` deleted.
#[derive(Debug, Clone, PartialEq)]
pub struct Removed {
    /// The pod.
    pub pod_id: PodId,
    /// How long it existed, when recorded.
    pub uptime: Option<Duration>,
    /// Its estimated spend, when recorded.
    pub estimated_spend: Option<f64>,
}

impl Removed {
    fn unrecorded(pod_id: PodId) -> Self {
        Self {
            pod_id,
            uptime: None,
            estimated_spend: None,
        }
    }
}

/// What `pod rm` did: the pods it deleted, even when it failed.
#[derive(Debug)]
pub struct Removal {
    /// Every pod deleted and confirmed gone.
    pub removed: Vec<Removed>,
    /// How it ended.
    pub result: Result<(), PodError>,
}

/// Deletes the pods of the run `run_id`: the one in its `pod.json` (even kept),
/// its `stray_pods`, and any other carrying its marker. Every delete is sent and
/// confirmed; a stray is dropped from `stray_pods` once gone, and a pod found by
/// its marker whose delete fails is added to it. Every pod is tried; the first
/// failure is returned at the end.
///
/// A run absent from this project's `runs/` is refused unless `force`: its pods
/// may belong to another checkout. So is a run still starting its pod (preparing,
/// or running with a last create call that has no pod yet), with
/// [`PodError::StillStarting`], and a running run whose pod is not recorded,
/// with [`PodError::PodNotRecorded`]: nothing is deleted. For a running run
/// without `force`, its training pod (the one in `pod.json`) is kept, every other
/// pod is deleted, and [`PodError::RunStillRunning`] says so. Forced, the run is
/// saved `Failed` once a pod was deleted. The run's private client key is removed
/// once every pod is.
#[must_use]
pub async fn remove_run_pods(ctx: &PodCtx<'_>, run_id: &str, force: bool) -> Removal {
    let mut removed = Vec::new();
    let result = remove_checked(ctx, run_id, force, &mut removed).await;
    Removal { removed, result }
}

async fn remove_checked(
    ctx: &PodCtx<'_>,
    run_id: &str,
    force: bool,
    removed: &mut Vec<Removed>,
) -> Result<(), PodError> {
    ctx.runs.run_dir(run_id)?;
    let state = match ctx.runs.load(run_id) {
        Ok(run) => Some(run.state),
        Err(RunsError::NotFound(_)) if force => None,
        Err(RunsError::NotFound(_)) => return Err(PodError::NotInRuns(run_id.to_string())),
        Err(error) => return Err(error.into()),
    };
    let in_progress = matches!(state, Some(RunState::Preparing | RunState::Running));
    let record = PodRecord::load(ctx.runs, run_id)?;
    if !force && starting(state, record.as_ref()) {
        return Err(PodError::StillStarting(run_id.to_string()));
    }
    let keep_training = in_progress && !force;
    let training = record
        .as_ref()
        .filter(|record| record.state != PodState::Deleted)
        .and_then(|record| record.pod_id.clone());
    if keep_training && training.is_none() {
        // Any pod carrying the marker may be the one training.
        return Err(PodError::PodNotRecorded(run_id.to_string()));
    }
    let mut failed = None;
    if !keep_training
        && let Some(mut record) = record
        && let Err(error) = remove_recorded(ctx, &mut record, removed).await
    {
        failed.get_or_insert(error);
    }
    let mut tried: Vec<PodId> = training.iter().cloned().collect();
    remove_strays(ctx, run_id, removed, &mut tried, &mut failed).await;
    remove_marked(ctx, run_id, removed, &tried, &mut failed).await;
    if in_progress
        && !keep_training
        && !removed.is_empty()
        && let Err(error) = fail_run_by_pod_rm(ctx.runs, run_id)
    {
        failed.get_or_insert(error);
    }
    if let Some(error) = failed {
        return Err(error);
    }
    if keep_training {
        return Err(PodError::RunStillRunning {
            run_id: run_id.to_string(),
            kept: kept_words(training.as_ref(), removed),
        });
    }
    forget_client_key(ctx.runs, run_id);
    Ok(())
}

/// Whether the run is still starting its pod: preparing, or in progress with a
/// last create call that has no pod yet. Such a run's pods are all left alone
/// without `--force`: provisioning sweeps its own duplicates.
fn starting(state: Option<RunState>, record: Option<&PodRecord>) -> bool {
    let unanswered = record
        .and_then(|record| record.attempts.last())
        .is_some_and(|attempt| attempt.pod_id.is_none());
    match state {
        Some(RunState::Preparing) => true,
        Some(RunState::Running) => unanswered,
        _ => false,
    }
}

/// Deletes the pod of `record`, unless it is already deleted.
async fn remove_recorded(
    ctx: &PodCtx<'_>,
    record: &mut PodRecord,
    removed: &mut Vec<Removed>,
) -> Result<(), PodError> {
    let Some(id) = record
        .pod_id
        .clone()
        .filter(|_| record.state != PodState::Deleted)
    else {
        return Ok(());
    };
    let uptime = record.uptime(SystemTime::now());
    remove(ctx, record, DeleteReason::Requested, DeletedBy::PodRm).await?;
    removed.push(Removed {
        pod_id: id,
        uptime,
        estimated_spend: record.estimated_spend,
    });
    Ok(())
}

/// Deletes every stray pod of the run, dropping each from `stray_pods` once it
/// is confirmed gone. Each one tried is added to `tried`.
async fn remove_strays(
    ctx: &PodCtx<'_>,
    run_id: &str,
    removed: &mut Vec<Removed>,
    tried: &mut Vec<PodId>,
    failed: &mut Option<PodError>,
) {
    let strays = match PodRecord::load(ctx.runs, run_id) {
        Ok(record) => record.map(|record| record.stray_pods).unwrap_or_default(),
        Err(error) => {
            failed.get_or_insert(error.into());
            return;
        },
    };
    for id in strays {
        tried.push(id.clone());
        match delete_confirmed(ctx, &id).await {
            Ok(_) => {
                if let Err(error) = forget_stray(ctx.runs, run_id, &id) {
                    failed.get_or_insert(error);
                }
                removed.push(Removed::unrecorded(id));
            },
            Err(error) => {
                tracing::warn!("cannot confirm the deletion of stray pod {id}: {error}");
                failed.get_or_insert(error);
            },
        }
    }
}

/// Deletes every other listed pod carrying the run's marker, but those in
/// `tried`. One whose delete fails is added to `stray_pods`.
async fn remove_marked(
    ctx: &PodCtx<'_>,
    run_id: &str,
    removed: &mut Vec<Removed>,
    tried: &[PodId],
    failed: &mut Option<PodError>,
) {
    let pods = match ctx.client.list_pods().await {
        Ok(pods) => pods,
        Err(error) => {
            failed.get_or_insert(error.into());
            return;
        },
    };
    for pod in pods {
        let done = tried.contains(&pod.id) || removed.iter().any(|done| done.pod_id == pod.id);
        if pod.run_id() != Some(run_id) || done {
            continue;
        }
        match delete_confirmed(ctx, &pod.id).await {
            Ok(_) => removed.push(Removed::unrecorded(pod.id)),
            Err(error) => {
                record_stray(ctx.runs, run_id, &pod.id, &error);
                failed.get_or_insert(error);
            },
        }
    }
}

/// Adds the pod `id`, whose delete failed with `error`, to the run's
/// `stray_pods`, best effort.
fn record_stray(runs: &Runs, run_id: &str, id: &PodId, error: &PodError) {
    let stray = id.clone();
    let recorded = match edit_record(runs, run_id, |record| record.note_stray(stray)) {
        Ok(()) => String::new(),
        Err(record_error) => format!(" (and cannot record it as stray: {record_error})"),
    };
    tracing::warn!("cannot confirm the deletion of pod {id}: {error}{recorded}");
}

/// Saves the run `run_id`, read again just before, `Failed` by `pod rm`.
fn fail_run_by_pod_rm(runs: &Runs, run_id: &str) -> Result<(), PodError> {
    let mut run = runs.load(run_id)?;
    run.state = RunState::Failed;
    run.message = Some(FAILED_BY_POD_RM.to_string());
    runs.save(&run)?;
    Ok(())
}

/// What a `pod rm` of a run in progress kept and deleted, for its error.
fn kept_words(training: Option<&PodId>, removed: &[Removed]) -> String {
    let others: Vec<String> = removed.iter().map(|pod| pod.pod_id.to_string()).collect();
    let deleted = if others.is_empty() {
        String::new()
    } else {
        format!("its other pods {} were deleted", others.join(", "))
    };
    match (training, deleted.is_empty()) {
        (Some(id), true) => format!(": its training pod {id} was kept"),
        (Some(id), false) => format!(": its training pod {id} was kept, and {deleted}"),
        (None, false) => format!(": {deleted}"),
        (None, true) => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(run: &str, kind: RowKind, note: &str) -> PodRow {
        PodRow {
            run: run.to_string(),
            pod_id: "k3x9abc".to_string(),
            status: "RUNNING".to_string(),
            gpu: "NVIDIA A40".to_string(),
            rate: Some(0.49),
            created: "2026-09-21T09:00:03Z".to_string(),
            note: note.to_string(),
            kind,
        }
    }

    #[test]
    fn only_pods_nothing_will_delete_are_orphans_and_each_gets_a_safe_hint() {
        let rows = [
            row("r1", RowKind::InProgress, "run running"),
            row("r2", RowKind::Ended, "run succeeded, not deleted"),
            row(
                "r3",
                RowKind::NotInRuns,
                "not in this project's runs/; if no other checkout owns it, `overbrainer pod rm r3 --force`",
            ),
            row("r4", RowKind::Kept, "kept"),
            row("r5", RowKind::AwaitingRetrieval, "awaiting retrieval"),
            row("r6", RowKind::Stray, STRAY),
            row("-", RowKind::NoMarker, NO_MARKER),
            row("r8", RowKind::Unreadable, "run record unreadable"),
            row("r9", RowKind::Gone, "gone"),
        ];
        assert_eq!(
            orphan_warnings(&rows),
            vec![
                "pod k3x9abc (run succeeded, not deleted) is still on Runpod at $0.49/h: remove it with `overbrainer pod rm r2`".to_string(),
                "pod k3x9abc is still on Runpod at $0.49/h: not in this project's runs/; if no other checkout owns it, `overbrainer pod rm r3 --force`".to_string(),
                "pod k3x9abc (stray, not deleted) is still on Runpod at $0.49/h: remove it with `overbrainer pod rm r6`".to_string(),
                "pod k3x9abc is named like an overbrainer pod but has no run marker; if it is yours, delete it from the Runpod console".to_string(),
            ]
        );
    }

    #[test]
    fn the_table_aligns_its_columns() {
        assert!(table(&[]).is_empty());
        let lines = table(&[row(
            "20260921-090000-ffff",
            RowKind::Ended,
            "run succeeded, not deleted",
        )]);
        assert_eq!(
            lines,
            vec![
                "RUN                   POD             STATUS        GPU                         $/H  CREATED               NOTE".to_string(),
                "20260921-090000-ffff  k3x9abc         RUNNING       NVIDIA A40                 0.49  2026-09-21T09:00:03Z  run succeeded, not deleted".to_string(),
            ]
        );
    }

    #[test]
    fn a_refused_pod_rm_says_what_it_kept_and_deleted() -> Result<(), crate::runpod::InvalidPodId> {
        let training = PodId::new("p1")?;
        let others = [
            Removed::unrecorded(PodId::new("s1")?),
            Removed::unrecorded(PodId::new("x2")?),
        ];
        assert_eq!(
            kept_words(Some(&training), &others),
            ": its training pod p1 was kept, and its other pods s1, x2 were deleted"
        );
        assert_eq!(
            kept_words(Some(&training), &[]),
            ": its training pod p1 was kept"
        );
        assert_eq!(kept_words(None, &[]), "");
        Ok(())
    }
}
