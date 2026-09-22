//! Training runs: `runs/<run-id>/` directories, their `run.json` record, and the
//! train, attach and cancel flows.

mod id;
mod summary;
mod train;

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub use id::{is_valid_run_id, new_run_id, rfc3339};
pub use summary::MetricsSummary;
pub use train::{HF_CACHE_DIR, Launch, Outcome, RunCtx, RunError, cancel, create, start, watch};

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
    /// A `run.json` is not a valid record.
    #[error("{} is not a valid run record", path.display())]
    Invalid {
        /// The file.
        path: PathBuf,
        /// Underlying JSON error.
        #[source]
        source: serde_json::Error,
    },
    /// An ID cannot name a run directory.
    #[error("{0} is not a valid run ID")]
    InvalidId(String),
    /// No run has this ID.
    #[error("no run `{0}` in runs/")]
    NotFound(String),
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
}

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
        let tmp = dir.join(format!(".{RECORD_FILE}.tmp"));
        let mut content =
            serde_json::to_vec_pretty(record).map_err(|source| RunsError::Invalid {
                path: path.clone(),
                source,
            })?;
        content.push(b'\n');
        fs::write(&tmp, content).map_err(io_error(&tmp))?;
        fs::rename(&tmp, &path).map_err(io_error(&path))
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
        serde_json::from_slice(&content).map_err(|source| RunsError::Invalid { path, source })
    }

    /// Every run with a readable record, oldest first. Directories without
    /// `run.json`, and records that fail to load ([`RunsError::Invalid`] or
    /// [`RunsError::Io`]), are skipped and logged with `tracing::warn!`.
    ///
    /// # Errors
    ///
    /// Returns an error when `runs/` itself cannot be read.
    pub fn list(&self) -> Result<Vec<RunRecord>, RunsError> {
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
                    tracing::warn!("skipping unreadable run record for {id}: {error}");
                },
                Err(error) => return Err(error),
            }
        }
        Ok(records)
    }
}

fn io_error(path: &Path) -> impl FnOnce(io::Error) -> RunsError + '_ {
    move |source| RunsError::Io {
        path: path.to_path_buf(),
        source,
    }
}

#[cfg(test)]
mod tests {
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
        }
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
