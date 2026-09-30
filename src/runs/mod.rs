//! Training runs: `runs/<run-id>/` directories, their `run.json` record, and the
//! train, attach and cancel flows.

mod id;
mod snapshot;
mod summary;
mod train;

use std::fs;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub use id::{
    RUN_ID_MAX, compact_utc, is_safe_name, is_valid_run_id, new_run_id, parse_rfc3339,
    project_slug, rfc3339,
};
pub use snapshot::{
    Proof, STOP_TIMEOUT, Snapshot, SnapshotReason, parse_proof, read_proof, request_snapshot,
    stop_fallback,
};
pub use summary::MetricsSummary;
pub use train::{
    HF_CACHE_DIR, Launch, Outcome, PROBE_EVERY, RunCtx, RunError, artifacts_missing, cancel,
    collect, create, create_on, reserve, start, watch,
};

use crate::exec::JobId;

/// Directory holding the runs, in the project directory.
pub const RUNS_DIR: &str = "runs";
/// Record of a run, in its directory.
pub const RECORD_FILE: &str = "run.json";

/// Errors reading or writing run records.
#[derive(Debug, thiserror::Error)]
pub enum RunsError {
    /// A run file or directory could not be read or written.
    #[error("cannot access {}", path.display())]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// A `run.json` or `pod.json` is not a valid record. Only where the JSON
    /// error is is kept: its message can quote a value straight from the file.
    #[error("{} is not a valid run record (line {line}, column {column})", path.display())]
    Invalid {
        /// The file.
        path: PathBuf,
        /// Line of the JSON error, from 1 (0 when not reading a file).
        line: usize,
        /// Column of the JSON error, from 1 (0 when not reading a file).
        column: usize,
    },
    /// An ID cannot name a run directory.
    #[error("{0} is not a valid run ID")]
    InvalidId(String),
    /// No run has this ID.
    #[error("no run `{0}` in runs/")]
    NotFound(String),
}

impl RunsError {
    /// [`RunsError::Invalid`] for `path`, keeping only the position of `error`.
    pub(crate) fn invalid(path: PathBuf, error: &serde_json::Error) -> Self {
        Self::Invalid {
            path,
            line: error.line(),
            column: error.column(),
        }
    }
}

/// Where a run stands, as last seen by overbrainer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RunState {
    /// Files are being prepared or uploaded; the job has not started.
    Preparing,
    /// The job was started and has not been seen ending.
    Running,
    /// The job ended with exit code 0 and wrote metrics.
    Succeeded,
    /// The job failed, stopped without an exit code, or wrote no metrics.
    Failed,
    /// The job was cancelled.
    Cancelled,
    /// The job was stopped with a snapshot: it saved a checkpoint, ended with
    /// exit code 0, and a new run can resume from it.
    Stopped,
}

impl RunState {
    /// Lowercase name, as stored in `run.json`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Preparing => "preparing",
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Stopped => "stopped",
        }
    }
}

/// `runs/<run-id>/run.json`: what is needed to find the job again.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunRecord {
    /// The run ID, also the directory name.
    pub id: String,
    /// Name of the target in `overbrainer.toml`.
    pub target: String,
    /// Creation time, RFC 3339 UTC.
    pub created: String,
    /// The run directory on the target.
    pub remote_dir: String,
    /// The job, once started.
    pub job: Option<JobId>,
    /// Where the run stands.
    pub state: RunState,
    /// Why the run failed, when it did.
    pub message: Option<String>,
    /// The snapshot of a stopped run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<Snapshot>,
    /// The run this one resumed from, when it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resumed_from: Option<String>,
}

/// Most runs [`Runs::claim`] makes with one base ID: `base`, then `base_2` to
/// `base_99`.
const MAX_CLAIMS: u32 = 99;

/// The `runs/` directory of a project.
#[derive(Debug, Clone)]
pub struct Runs {
    dir: PathBuf,
}

impl Runs {
    /// The runs of the project in `project_dir`.
    #[must_use]
    pub fn new(project_dir: &Path) -> Self {
        Self {
            dir: project_dir.join(RUNS_DIR),
        }
    }

    /// The `runs/` directory.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Local directory of the run `id`.
    ///
    /// # Errors
    ///
    /// Returns [`RunsError::InvalidId`] when `id` is not a valid run ID, so a
    /// directory outside `runs/` (an absolute path, or one with `..` or `/`) is
    /// never returned.
    pub fn run_dir(&self, id: &str) -> Result<PathBuf, RunsError> {
        if is_valid_run_id(id) {
            Ok(self.dir.join(id))
        } else {
            Err(RunsError::InvalidId(id.to_string()))
        }
    }

    /// Creates the directory of a new run and returns its ID with its number:
    /// `base` is 1, `base_2` is 2, and so on up to `base_99`. Only numbers past
    /// `after` are tried, and a number whose directory exists is skipped. Each
    /// directory is created exclusively, so no existing run is ever overwritten,
    /// even by another process creating a run at the same second.
    ///
    /// # Errors
    ///
    /// Returns [`RunsError::InvalidId`] when `base` is not a valid run ID, and
    /// [`RunsError::Io`] when a directory cannot be created, or no number up to
    /// 99 is left (an `AlreadyExists` error).
    pub fn claim(&self, base: &str, after: u32) -> Result<(String, u32), RunsError> {
        let first = self.run_dir(base)?;
        fs::create_dir_all(&self.dir).map_err(io_error(&self.dir))?;
        for n in after + 1..=MAX_CLAIMS {
            let id = if n == 1 {
                base.to_string()
            } else {
                format!("{base}_{n}")
            };
            let dir = self.run_dir(&id)?;
            match fs::create_dir(&dir) {
                Ok(()) => return Ok((id, n)),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {},
                Err(e) => return Err(io_error(&dir)(e)),
            }
        }
        Err(RunsError::Io {
            path: first,
            source: io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("runs {base} to {base}_{MAX_CLAIMS} all exist"),
            ),
        })
    }

    /// Removes the directory of run `id` made by [`Runs::claim`] when the run
    /// cannot use it after all. Only an empty directory is removed; a failure is
    /// logged, since an empty directory without a record is never listed.
    pub(crate) fn release(&self, id: &str) {
        let Ok(dir) = self.run_dir(id) else {
            return;
        };
        if let Err(error) = fs::remove_dir(&dir) {
            tracing::warn!(
                "cannot remove the unused run directory {}: {error}",
                dir.display()
            );
        }
    }

    /// Writes `record` to its `run.json`, replacing it atomically, and creates the
    /// run directory if needed.
    ///
    /// # Errors
    ///
    /// Returns [`RunsError::InvalidId`] when `record.id` is not a valid run ID,
    /// [`RunsError::Io`] when the file cannot be written, and [`RunsError::Invalid`]
    /// when the record cannot be serialized to JSON.
    pub fn save(&self, record: &RunRecord) -> Result<(), RunsError> {
        let dir = self.run_dir(&record.id)?;
        fs::create_dir_all(&dir).map_err(io_error(&dir))?;
        let path = dir.join(RECORD_FILE);
        let mut content =
            serde_json::to_vec_pretty(record).map_err(|e| RunsError::invalid(path, &e))?;
        content.push(b'\n');
        write_atomic(&dir, RECORD_FILE, &content)
    }

    /// Reads the record of run `id`.
    ///
    /// # Errors
    ///
    /// Returns [`RunsError::InvalidId`] when `id` is not a valid run ID,
    /// [`RunsError::NotFound`] when it is valid but has no record, and
    /// [`RunsError::Io`] or [`RunsError::Invalid`] when the record cannot be read.
    pub fn load(&self, id: &str) -> Result<RunRecord, RunsError> {
        let dir = self.run_dir(id)?;
        let path = dir.join(RECORD_FILE);
        let content = match fs::read(&path) {
            Ok(content) => content,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Err(RunsError::NotFound(id.to_string()));
            },
            Err(e) => return Err(io_error(&path)(e)),
        };
        serde_json::from_slice(&content).map_err(|e| RunsError::invalid(path, &e))
    }

    /// Every run with a readable record, oldest first. Directories without
    /// `run.json`, and records that fail to load ([`RunsError::Invalid`] or
    /// [`RunsError::Io`]), are skipped and logged with `tracing::warn!`.
    ///
    /// # Errors
    ///
    /// Returns an error when `runs/` itself cannot be read.
    pub fn list(&self) -> Result<Vec<RunRecord>, RunsError> {
        self.list_with(|id, error| {
            tracing::warn!("skipping unreadable run record for {id}: {error}");
        })
    }

    /// [`Runs::list`], calling `skipped` with the ID and error of each record
    /// that fails to load instead of logging it.
    ///
    /// # Errors
    ///
    /// Returns an error when `runs/` itself cannot be read.
    pub fn list_with(
        &self,
        mut skipped: impl FnMut(&str, &RunsError),
    ) -> Result<Vec<RunRecord>, RunsError> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(io_error(&self.dir)(e)),
        };
        let mut ids = Vec::new();
        for entry in entries {
            let entry = entry.map_err(io_error(&self.dir))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if is_valid_run_id(&name) && entry.path().join(RECORD_FILE).is_file() {
                ids.push(name);
            }
        }
        ids.sort();
        let mut records = Vec::with_capacity(ids.len());
        for id in &ids {
            match self.load(id) {
                Ok(record) => records.push(record),
                Err(error @ (RunsError::Invalid { .. } | RunsError::Io { .. })) => {
                    skipped(id, &error);
                },
                Err(error) => return Err(error),
            }
        }
        // `created` is RFC 3339 UTC to the second, as `rfc3339` writes it: its
        // text order is its time order.
        records.sort_by(|a, b| (&a.created, &a.id).cmp(&(&b.created, &b.id)));
        Ok(records)
    }
}

/// Replaces `dir/name` with `content` atomically: writes a temporary file of its
/// own in `dir` (`.<name>.<random>.tmp`, never shared with a concurrent save),
/// then renames it over `dir/name`, so a reader sees either the old or the new
/// content. The temporary file is removed when the write or the rename fails.
///
/// # Errors
///
/// Returns [`RunsError::Io`] when the temporary file cannot be created or
/// written, or cannot be renamed over `dir/name`.
pub(crate) fn write_atomic(dir: &Path, name: &str, content: &[u8]) -> Result<(), RunsError> {
    let tmp = dir.join(format!(".{name}.{:016x}.tmp", fastrand::u64(..)));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .map_err(io_error(&tmp))?;
    let written = file.write_all(content).map_err(io_error(&tmp));
    drop(file);
    let path = dir.join(name);
    let renamed = written.and_then(|()| fs::rename(&tmp, &path).map_err(io_error(&path)));
    if renamed.is_err()
        && let Err(error) = fs::remove_file(&tmp)
    {
        tracing::warn!("cannot remove {}: {error}", tmp.display());
    }
    renamed
}

fn io_error(path: &Path) -> impl FnOnce(io::Error) -> RunsError + '_ {
    move |source| RunsError::Io {
        path: path.to_path_buf(),
        source,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn record(id: &str) -> RunRecord {
        RunRecord {
            id: id.to_string(),
            target: "box".to_string(),
            created: "2026-09-22T14:30:05Z".to_string(),
            remote_dir: format!("/w/{id}"),
            job: None,
            state: RunState::Preparing,
            message: None,
            snapshot: None,
            resumed_from: None,
        }
    }

    #[test]
    fn a_stopped_record_keeps_its_snapshot_and_an_old_one_reads_without()
    -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let mut stopped = record("20260922-143005-bbbb");
        stopped.state = RunState::Stopped;
        stopped.snapshot = Some(Snapshot {
            checkpoint: "output/checkpoint-120".into(),
            step: 120,
            reason: SnapshotReason::Deadline,
        });
        stopped.resumed_from = Some("20260921-000000-aaaa".into());
        runs.save(&stopped)?;
        assert_eq!(runs.load(&stopped.id)?, stopped);
        let json = fs::read_to_string(runs.run_dir(&stopped.id)?.join(RECORD_FILE))?;
        assert!(json.contains("\"state\": \"stopped\""), "{json}");
        assert!(json.contains("\"reason\": \"deadline\""), "{json}");
        // A record without them, as older versions wrote it, has neither.
        let plain = record("20260922-143005-cccc");
        runs.save(&plain)?;
        let json = fs::read_to_string(runs.run_dir(&plain.id)?.join(RECORD_FILE))?;
        assert!(
            !json.contains("snapshot") && !json.contains("resumed_from"),
            "{json}"
        );
        assert_eq!(runs.load(&plain.id)?, plain);
        Ok(())
    }

    #[test]
    fn records_round_trip_and_list_in_order() -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        assert!(runs.list()?.is_empty());
        runs.save(&record("20260922-143005-bbbb"))?;
        let mut first = record("20260921-000000-aaaa");
        first.state = RunState::Succeeded;
        runs.save(&first)?;
        fs::create_dir_all(runs.dir().join(".hf-cache"))?;
        fs::create_dir_all(runs.dir().join("20260923-000000-cccc"))?;
        let ids: Vec<String> = runs.list()?.into_iter().map(|run| run.id).collect();
        assert_eq!(ids, vec!["20260921-000000-aaaa", "20260922-143005-bbbb"]);
        assert_eq!(runs.load("20260921-000000-aaaa")?, first);
        let json = fs::read_to_string(runs.run_dir(&first.id)?.join(RECORD_FILE))?;
        assert!(json.contains("\"state\": \"succeeded\""), "{json}");
        Ok(())
    }

    #[test]
    fn runs_list_in_creation_order_whatever_their_ids() -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let mut old_form = record("20260930-120000-ffff");
        old_form.created = "2026-09-30T12:00:00Z".to_string();
        let mut named = record("alpha_20260929-080000");
        named.created = "2026-09-29T08:00:00Z".to_string();
        let mut later = record("zeta_20260929-080000");
        later.created = "2026-09-29T08:00:00Z".to_string();
        let mut first = record("zeta_20260101-000000");
        first.created = "2026-01-01T00:00:00Z".to_string();
        for run in [&old_form, &later, &named, &first] {
            runs.save(run)?;
        }
        let ids: Vec<String> = runs.list()?.into_iter().map(|run| run.id).collect();
        assert_eq!(
            ids,
            vec![
                "zeta_20260101-000000",
                "alpha_20260929-080000",
                "zeta_20260929-080000",
                "20260930-120000-ffff",
            ]
        );
        Ok(())
    }

    #[test]
    fn a_claimed_run_directory_is_never_claimed_again() -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let base = "p_20260930-120000";
        assert_eq!(runs.claim(base, 0)?, (base.to_string(), 1));
        assert_eq!(runs.claim(base, 0)?, (format!("{base}_2"), 2));
        fs::create_dir(runs.dir().join(format!("{base}_3")))?;
        assert_eq!(runs.claim(base, 0)?, (format!("{base}_4"), 4));
        // Past `after` only: 5 is skipped although it is free.
        assert_eq!(runs.claim(base, 5)?, (format!("{base}_6"), 6));
        assert_eq!(runs.claim(base, 0)?, (format!("{base}_5"), 5));
        for n in 7..=99 {
            assert_eq!(runs.claim(base, 0)?, (format!("{base}_{n}"), n));
        }
        let full = runs.claim(base, 0);
        assert!(
            matches!(&full, Err(RunsError::Io { source, .. }) if source.kind() == io::ErrorKind::AlreadyExists),
            "{full:?}"
        );
        assert!(runs.claim(base, 99).is_err());
        assert!(matches!(
            runs.claim("../p", 0),
            Err(RunsError::InvalidId(_))
        ));
        runs.release(&format!("{base}_99"));
        assert_eq!(runs.claim(base, 0)?, (format!("{base}_99"), 99));
        Ok(())
    }

    /// The temporary files a save left in `dir`: `.run.json.<random>.tmp` or
    /// `.pod.json.<random>.tmp`.
    pub(crate) fn leftover_temp_files(dir: &Path) -> io::Result<Vec<String>> {
        let mut left = Vec::new();
        for entry in fs::read_dir(dir)? {
            let name = entry?.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') && Path::new(&name).extension().is_some_and(|e| e == "tmp") {
                left.push(name);
            }
        }
        Ok(left)
    }

    /// `error` and its sources, joined as the CLI prints an error with `{:#}`.
    pub(crate) fn chain(error: &dyn std::error::Error) -> String {
        let mut text = error.to_string();
        let mut source = error.source();
        while let Some(cause) = source {
            text.push_str(": ");
            text.push_str(&cause.to_string());
            source = cause.source();
        }
        text
    }

    #[test]
    fn a_malformed_run_json_is_reported_by_position_only() -> Result<(), Box<dyn std::error::Error>>
    {
        const MARKER: &str = "MARKER-0d4a8e52-never-printed";
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let id = "20260922-143005-bbbb";
        let dir = runs.run_dir(id)?;
        fs::create_dir_all(&dir)?;
        let content = format!("{{\n  \"id\": \"{id}\",\n  \"state\": \"{MARKER}\"\n}}\n");
        fs::write(dir.join(RECORD_FILE), content)?;
        let error = runs.load(id).err().ok_or("expected an error")?;
        let chain = chain(&error);
        assert!(!chain.contains(MARKER), "{chain}");
        assert!(chain.contains("is not a valid run record"), "{chain}");
        assert!(chain.contains("line 3"), "{chain}");
        let debug = format!("{error:?}");
        assert!(!debug.contains(MARKER), "{debug}");
        Ok(())
    }

    #[test]
    fn concurrent_saves_always_leave_a_valid_record() -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let id = "20260922-143005-bbbb";
        let results: Vec<Result<(), String>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|writer| {
                    let runs = &runs;
                    scope.spawn(move || -> Result<(), RunsError> {
                        for save in 0..40 {
                            let mut run = record(id);
                            run.message = Some(format!("writer {writer} save {save}"));
                            runs.save(&run)?;
                            runs.load(id)?;
                        }
                        Ok(())
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| match handle.join() {
                    Ok(result) => result.map_err(|error| format!("{error:#?}")),
                    Err(_) => Err("a writer panicked".to_string()),
                })
                .collect()
        });
        for result in results {
            result?;
        }
        let saved = runs.load(id)?;
        assert!(
            saved
                .message
                .as_deref()
                .is_some_and(|message| message.ends_with("save 39")),
            "{saved:?}"
        );
        assert_eq!(
            leftover_temp_files(&runs.run_dir(id)?)?,
            Vec::<String>::new()
        );
        Ok(())
    }

    #[test]
    fn a_failed_save_leaves_no_temporary_file() -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let run = record("20260922-143005-bbbb");
        let dir = runs.run_dir(&run.id)?;
        fs::create_dir_all(dir.join(RECORD_FILE))?;
        assert!(matches!(runs.save(&run), Err(RunsError::Io { .. })));
        assert_eq!(leftover_temp_files(&dir)?, Vec::<String>::new());
        Ok(())
    }

    #[test]
    fn unknown_ids_are_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        assert!(matches!(runs.load("nope"), Err(RunsError::NotFound(_))));
        Ok(())
    }

    #[test]
    fn invalid_ids_are_rejected_everywhere() -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        for id in ["../etc", "/tmp/x", "a/b"] {
            assert!(
                matches!(runs.run_dir(id), Err(RunsError::InvalidId(_))),
                "{id}"
            );
            assert!(
                matches!(runs.load(id), Err(RunsError::InvalidId(_))),
                "{id}"
            );
            let result = runs.save(&record(id));
            assert!(
                matches!(result, Err(RunsError::InvalidId(_))),
                "{id}: {result:?}"
            );
        }
        assert!(!runs.dir().exists());
        Ok(())
    }

    fn record_with_pid(id: &str, pid: u32) -> String {
        format!(
            r#"{{
  "id": "{id}",
  "target": "box",
  "created": "2026-09-22T14:30:05Z",
  "remote_dir": "/w/{id}",
  "job": {{"dir": "/w/run", "pid": {pid}, "container": null}},
  "state": "running",
  "message": null
}}
"#
        )
    }

    #[test]
    fn a_run_json_with_pid_0_or_1_fails_to_load() -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        for (id, pid) in [("20260922-143005-cccc", 1), ("20260922-143005-dddd", 0)] {
            let dir = runs.run_dir(id)?;
            fs::create_dir_all(&dir)?;
            fs::write(dir.join(RECORD_FILE), record_with_pid(id, pid))?;
            assert!(
                matches!(runs.load(id), Err(RunsError::Invalid { .. })),
                "{id}"
            );
        }
        Ok(())
    }

    #[test]
    fn list_skips_unreadable_records_and_keeps_the_rest() -> Result<(), Box<dyn std::error::Error>>
    {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        runs.save(&record("20260921-000000-good"))?;
        let garbage_dir = runs.run_dir("20260922-000000-bad")?;
        fs::create_dir_all(&garbage_dir)?;
        fs::write(garbage_dir.join(RECORD_FILE), "not json")?;
        let pid_dir = runs.run_dir("20260923-000000-pid1")?;
        fs::create_dir_all(&pid_dir)?;
        fs::write(
            pid_dir.join(RECORD_FILE),
            record_with_pid("20260923-000000-pid1", 1),
        )?;
        let ids: Vec<String> = runs.list()?.into_iter().map(|run| run.id).collect();
        assert_eq!(ids, vec!["20260921-000000-good"]);
        Ok(())
    }
}
