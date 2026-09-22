//! Training runs: `runs/<run-id>/` directories, their `run.json` record, and the
//! train, attach and cancel flows.

mod id;
mod summary;

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub use id::{is_valid_run_id, new_run_id, rfc3339};
pub use summary::MetricsSummary;

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
    #[must_use]
    pub fn run_dir(&self, id: &str) -> PathBuf {
        self.dir.join(id)
    }

    /// Writes `record` to its `run.json`, replacing it atomically, and creates the
    /// run directory if needed.
    ///
    /// # Errors
    ///
    /// Returns [`RunsError::Io`] when the file cannot be written.
    pub fn save(&self, record: &RunRecord) -> Result<(), RunsError> {
        let dir = self.run_dir(&record.id);
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
    /// Returns [`RunsError::NotFound`] when `id` is not a valid run ID or has no
    /// record, and [`RunsError::Io`] or [`RunsError::Invalid`] when the record cannot
    /// be read.
    pub fn load(&self, id: &str) -> Result<RunRecord, RunsError> {
        if !is_valid_run_id(id) {
            return Err(RunsError::NotFound(id.to_string()));
        }
        let path = self.run_dir(id).join(RECORD_FILE);
        let content = match fs::read(&path) {
            Ok(content) => content,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return Err(RunsError::NotFound(id.to_string()));
            },
            Err(e) => return Err(io_error(&path)(e)),
        };
        serde_json::from_slice(&content).map_err(|source| RunsError::Invalid { path, source })
    }

    /// Every run with a record, oldest first. Directories without `run.json` are
    /// skipped.
    ///
    /// # Errors
    ///
    /// Returns an error when `runs/` or a record cannot be read.
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
        ids.iter().map(|id| self.load(id)).collect()
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
        let json = fs::read_to_string(runs.run_dir(&first.id).join(RECORD_FILE))?;
        assert!(json.contains("\"state\": \"succeeded\""), "{json}");
        Ok(())
    }

    #[test]
    fn unknown_or_invalid_ids_are_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        assert!(matches!(runs.load("nope"), Err(RunsError::NotFound(_))));
        assert!(matches!(runs.load("../etc"), Err(RunsError::NotFound(_))));
        Ok(())
    }

    #[test]
    fn a_run_json_with_an_invalid_pid_fails_to_load() -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let id = "20260922-143005-cccc";
        let dir = runs.run_dir(id);
        fs::create_dir_all(&dir)?;
        fs::write(
            dir.join(RECORD_FILE),
            r#"{
  "id": "20260922-143005-cccc",
  "target": "box",
  "created": "2026-09-22T14:30:05Z",
  "remote_dir": "/w/20260922-143005-cccc",
  "job": {"dir": "/w/run", "pid": 1, "container": null},
  "state": "running",
  "message": null
}
"#,
        )?;
        assert!(matches!(runs.load(id), Err(RunsError::Invalid { .. })));
        Ok(())
    }
}
