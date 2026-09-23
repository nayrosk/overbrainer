//! `overbrainer pod ls` and `pod rm`, and the orphan warning of `train`: the pods
//! of the account that overbrainer created, matched with `runs/`.

use std::time::{Duration, SystemTime};

use crate::runs::{RunState, Runs, RunsError};

use super::flow::{look_up, mark_gone};
use super::provision::delete_confirmed;
use super::{
    DeleteReason, DeletedBy, Pod, PodCtx, PodError, PodId, PodRecord, PodState, forget_client_key,
    remove,
};

/// Characters kept of a table cell taken from the Runpod account.
const CELL_CHARS: usize = 64;

/// The note of a stray pod: another pod of a run, left by an ambiguous create,
/// whose delete could not be confirmed (`stray_pods` in `pod.json`).
const STRAY: &str = "stray, not deleted";

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
    /// What overbrainer knows of it (see [`pod_rows`]).
    pub note: String,
}

impl PodRow {
    /// Whether nothing will delete this pod on its own: its run is unknown or has
    /// ended, or it is a stray, and it is neither kept nor waiting for its results
    /// to be retrieved.
    #[must_use]
    pub fn is_orphan(&self) -> bool {
        self.note == "not in runs/" || self.note.ends_with(", not deleted")
    }
}

/// Every pod of the account whose run marker is set or whose name starts with
/// `overbrainer-`, plus every pod recorded in `runs/` (the run's pod and its
/// `stray_pods`) that the list does not show. Such a pod is looked up again and
/// declared gone only after three 404s in a row and a list without it (never by
/// a delete): the run's pod is then recorded `deleted` in `pod.json`, and a stray
/// is dropped from `stray_pods`. The note of a row says `run <state>` for a run
/// in progress, `run <state>, not deleted` for an ended run, `kept`, `awaiting
/// retrieval`, `stray, not deleted`, `not in runs/` for a marker without a local
/// run, or `gone`.
///
/// # Errors
///
/// Returns a [`PodError`] when the API or `runs/` cannot be read.
pub async fn pod_rows(ctx: &PodCtx<'_>) -> Result<Vec<PodRow>, PodError> {
    let pods = ctx.client.list_pods().await?;
    let mut records = Vec::new();
    for run in ctx.runs.list()? {
        if let Some(record) = PodRecord::load(ctx.runs, &run.id)? {
            records.push(record);
        }
    }
    let mut rows = Vec::new();
    for pod in &pods {
        if pod.run_id().is_some() || pod.name.starts_with("overbrainer-") {
            let stray = records
                .iter()
                .any(|record| record.stray_pods.contains(&pod.id));
            let run = pod.run_id().unwrap_or("-");
            rows.push(row(ctx.runs, run, pod, stray)?);
        }
    }
    for record in &mut records {
        unlisted_pod(ctx, record, &pods, &mut rows).await?;
        unlisted_strays(ctx, record, &pods, &mut rows).await?;
    }
    rows.sort_by(|a, b| (&a.run, &a.pod_id).cmp(&(&b.run, &b.pod_id)));
    Ok(rows)
}

/// The row of the run's recorded pod when the list does not show it: the pod as
/// a look finds it, or `GONE` once confirmed gone (then recorded so).
async fn unlisted_pod(
    ctx: &PodCtx<'_>,
    record: &mut PodRecord,
    pods: &[Pod],
    rows: &mut Vec<PodRow>,
) -> Result<(), PodError> {
    let Some(id) = record.pod_id.clone() else {
        return Ok(());
    };
    if record.state == PodState::Deleted || pods.iter().any(|pod| pod.id == id) {
        return Ok(());
    }
    let run_id = record.run_id.clone();
    match look_up(ctx, &id).await {
        Ok(Some(pod)) => rows.push(row(ctx.runs, &run_id, &pod, false)?),
        Ok(None) => {
            mark_gone(ctx, record, id.clone(), DeletedBy::Unknown)?;
            rows.push(recorded_row(record, &id, "GONE", "gone".to_string()));
        },
        Err(error) => {
            tracing::warn!("cannot look up pod {id} of run {run_id}: {error}");
            let note = note(ctx.runs, &run_id, false)?;
            rows.push(recorded_row(record, &id, "UNCHECKED", note));
        },
    }
    Ok(())
}

/// The rows of the run's stray pods the list does not show. A stray confirmed
/// gone is dropped from `stray_pods` instead.
async fn unlisted_strays(
    ctx: &PodCtx<'_>,
    record: &mut PodRecord,
    pods: &[Pod],
    rows: &mut Vec<PodRow>,
) -> Result<(), PodError> {
    let unlisted: Vec<PodId> = record
        .stray_pods
        .iter()
        .filter(|id| pods.iter().all(|pod| pod.id != **id))
        .cloned()
        .collect();
    for id in unlisted {
        unlisted_stray(ctx, record, id, rows).await?;
    }
    Ok(())
}

/// The row of the stray pod `id` of `record`, or nothing once it is confirmed
/// gone and dropped from `stray_pods`.
async fn unlisted_stray(
    ctx: &PodCtx<'_>,
    record: &mut PodRecord,
    id: PodId,
    rows: &mut Vec<PodRow>,
) -> Result<(), PodError> {
    let run_id = record.run_id.clone();
    match look_up(ctx, &id).await {
        Ok(Some(pod)) => rows.push(row(ctx.runs, &run_id, &pod, true)?),
        Ok(None) => forget_stray(ctx.runs, record, &id)?,
        Err(error) => {
            tracing::warn!("cannot look up stray pod {id} of run {run_id}: {error}");
            let mut unchecked = recorded_row(record, &id, "UNCHECKED", STRAY.to_string());
            unchecked.gpu = "-".to_string();
            unchecked.created = String::new();
            rows.push(unchecked);
        },
    }
    Ok(())
}

/// Drops the stray pod `id`, confirmed gone, from `record`.
fn forget_stray(runs: &Runs, record: &mut PodRecord, id: &PodId) -> Result<(), PodError> {
    record.stray_pods.retain(|stray| stray != id);
    record.save(runs)?;
    tracing::info!("stray pod {id} of run {} is gone", record.run_id);
    Ok(())
}

/// The row of `pod`, a pod of the run `run`.
fn row(runs: &Runs, run: &str, pod: &Pod, stray: bool) -> Result<PodRow, PodError> {
    Ok(PodRow {
        run: cell(run),
        pod_id: pod.id.to_string(),
        status: pod.status.name().to_string(),
        gpu: cell(pod.gpu_type().unwrap_or("-")),
        rate: pod.cost,
        created: cell(pod.created_at.as_deref().unwrap_or_default()),
        note: note(runs, run, stray)?,
    })
}

/// The row of the pod `id` of `record`, from what `pod.json` says.
fn recorded_row(record: &PodRecord, id: &PodId, status: &str, note: String) -> PodRow {
    PodRow {
        run: record.run_id.clone(),
        pod_id: id.to_string(),
        status: status.to_string(),
        gpu: cell(record.gpu_type.as_deref().unwrap_or("-")),
        rate: None,
        created: record.created_at.clone().unwrap_or_default(),
        note,
    }
}

/// `text`, from the Runpod account, as one short line of printable ASCII.
fn cell(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_ascii_graphic() || *c == ' ')
        .take(CELL_CHARS)
        .collect()
}

/// What `runs/` says about a pod of `run_id`; `stray` for one of its
/// `stray_pods`.
fn note(runs: &Runs, run_id: &str, stray: bool) -> Result<String, PodError> {
    let run = match runs.load(run_id) {
        Ok(run) => run,
        Err(RunsError::NotFound(_) | RunsError::InvalidId(_)) => return Ok("not in runs/".into()),
        Err(error) => return Err(error.into()),
    };
    if stray {
        return Ok(STRAY.to_string());
    }
    let state = PodRecord::load(runs, run_id)?.map(|record| record.state);
    Ok(match (state, run.state) {
        (Some(PodState::Kept), _) => "kept".to_string(),
        (Some(PodState::AwaitingRetrieval), _) => "awaiting retrieval".to_string(),
        (_, RunState::Preparing | RunState::Running) => format!("run {}", run.state.name()),
        (_, ended) => format!("run {}, not deleted", ended.name()),
    })
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

/// One warning per orphan pod among `rows`, naming the command that removes it.
/// Never deletes anything: only `overbrainer pod rm` does.
#[must_use]
pub fn orphan_warnings(rows: &[PodRow]) -> Vec<String> {
    rows.iter()
        .filter(|row| row.is_orphan())
        .map(|row| {
            let rate = row
                .rate
                .map_or_else(String::new, |rate| format!(" at ${rate:.2}/h"));
            format!(
                "pod {} ({}) is still on Runpod{rate}: remove it with `overbrainer pod rm {}`",
                row.pod_id, row.note, row.run
            )
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

/// Deletes every pod of the run `run_id`: the one in its `pod.json` (even kept),
/// its `stray_pods`, and any other carrying its marker (so a pod whose run
/// directory is gone, or a duplicate, goes too). The delete is always sent and
/// each pod confirmed gone; a stray is dropped from `stray_pods` once it is. A
/// run still `Running` is refused unless `force`; forced, it is saved `Failed`
/// once a pod was deleted. The run's private client key is removed once every
/// pod is.
///
/// # Errors
///
/// Returns [`PodError::RunStillRunning`] for a running run without `force`,
/// and another [`PodError`] when the API or the files fail, or a delete cannot
/// be confirmed (every stray is still tried first).
pub async fn remove_run_pods(
    ctx: &PodCtx<'_>,
    run_id: &str,
    force: bool,
) -> Result<Vec<Removed>, PodError> {
    ctx.runs.run_dir(run_id)?;
    let run = match ctx.runs.load(run_id) {
        Ok(run) => Some(run),
        Err(RunsError::NotFound(_)) => None,
        Err(error) => return Err(error.into()),
    };
    let running = run
        .as_ref()
        .is_some_and(|run| run.state == RunState::Running);
    if running && !force {
        return Err(PodError::RunStillRunning(run_id.to_string()));
    }
    let mut removed = Vec::new();
    let result = remove_all(ctx, run_id, &mut removed).await;
    if let Some(mut run) = run.filter(|_| running && !removed.is_empty()) {
        run.state = RunState::Failed;
        run.message = Some("pod deleted by `pod rm` before its results were retrieved".to_string());
        ctx.runs.save(&run)?;
    }
    result?;
    forget_client_key(ctx.runs, run_id);
    Ok(removed)
}

/// See [`remove_run_pods`]: every deleted pod is pushed to `removed`.
async fn remove_all(
    ctx: &PodCtx<'_>,
    run_id: &str,
    removed: &mut Vec<Removed>,
) -> Result<(), PodError> {
    if let Some(mut record) = PodRecord::load(ctx.runs, run_id)? {
        if record.state != PodState::Deleted
            && let Some(id) = record.pod_id.clone()
        {
            let uptime = record.uptime(SystemTime::now());
            remove(ctx, &mut record, DeleteReason::Requested, DeletedBy::PodRm).await?;
            removed.push(Removed {
                pod_id: id,
                uptime,
                estimated_spend: record.estimated_spend,
            });
        }
        remove_strays(ctx, &mut record, removed).await?;
    }
    for pod in ctx.client.list_pods().await? {
        if pod.run_id() == Some(run_id) && removed.iter().all(|done| done.pod_id != pod.id) {
            delete_confirmed(ctx, &pod.id).await?;
            removed.push(Removed::unrecorded(pod.id));
        }
    }
    Ok(())
}

/// Deletes every stray pod of `record`, dropping each from `stray_pods` once it
/// is confirmed gone. Every stray is tried; the first failure is returned.
async fn remove_strays(
    ctx: &PodCtx<'_>,
    record: &mut PodRecord,
    removed: &mut Vec<Removed>,
) -> Result<(), PodError> {
    let mut failed = None;
    for id in record.stray_pods.clone() {
        match delete_confirmed(ctx, &id).await {
            Ok(_) => {
                record.stray_pods.retain(|stray| *stray != id);
                record.save(ctx.runs)?;
                removed.push(Removed::unrecorded(id));
            },
            Err(error) => {
                tracing::warn!("cannot confirm the deletion of stray pod {id}: {error}");
                failed.get_or_insert(error);
            },
        }
    }
    failed.map_or(Ok(()), Err)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(run: &str, note: &str) -> PodRow {
        PodRow {
            run: run.to_string(),
            pod_id: "k3x9abc".to_string(),
            status: "RUNNING".to_string(),
            gpu: "NVIDIA A40".to_string(),
            rate: Some(0.49),
            created: "2026-09-21T09:00:03Z".to_string(),
            note: note.to_string(),
        }
    }

    #[test]
    fn only_pods_nothing_will_delete_are_orphans() {
        let rows = [
            row("r1", "run running"),
            row("r2", "run succeeded, not deleted"),
            row("r3", "not in runs/"),
            row("r4", "kept"),
            row("r5", "awaiting retrieval"),
            row("r6", "stray, not deleted"),
        ];
        assert_eq!(
            orphan_warnings(&rows),
            vec![
                "pod k3x9abc (run succeeded, not deleted) is still on Runpod at $0.49/h: remove it with `overbrainer pod rm r2`".to_string(),
                "pod k3x9abc (not in runs/) is still on Runpod at $0.49/h: remove it with `overbrainer pod rm r3`".to_string(),
                "pod k3x9abc (stray, not deleted) is still on Runpod at $0.49/h: remove it with `overbrainer pod rm r6`".to_string(),
            ]
        );
    }

    #[test]
    fn the_table_aligns_its_columns() {
        assert!(table(&[]).is_empty());
        let lines = table(&[row("20260921-090000-ffff", "run succeeded, not deleted")]);
        assert_eq!(
            lines,
            vec![
                "RUN                   POD             STATUS        GPU                         $/H  CREATED               NOTE".to_string(),
                "20260921-090000-ffff  k3x9abc         RUNNING       NVIDIA A40                 0.49  2026-09-21T09:00:03Z  run succeeded, not deleted".to_string(),
            ]
        );
    }

    #[test]
    fn text_from_the_account_is_one_short_printable_line() {
        assert_eq!(cell("run\u{1b}[31m\nx\u{e9}"), "run[31mx");
        assert_eq!(cell(&"y".repeat(500)).len(), CELL_CHARS);
    }
}
