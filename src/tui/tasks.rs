//! The TUI's background work: each task runs on its own, owns its inputs and
//! returns what the app needs when it ends. The app only ever sees results.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tokio::task::JoinSet;

use crate::dataset::{DataFiles, Dataset};

/// Identifies a task for the app.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct TaskId(pub(super) u64);

/// Work the app asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Task {
    /// Reads the dataset files.
    Load,
}

/// What a task gives back.
#[derive(Debug)]
pub(super) enum Done {
    /// The dataset files, or why they cannot be read.
    Loaded(Result<Dataset, String>),
}

/// The running tasks.
pub(super) struct Tasks {
    project_dir: PathBuf,
    set: JoinSet<Done>,
    ids: HashMap<tokio::task::Id, TaskId>,
}

impl Tasks {
    /// No task yet, for the project in `project_dir`.
    pub(super) fn new(project_dir: &Path) -> Self {
        Self {
            project_dir: project_dir.to_path_buf(),
            set: JoinSet::new(),
            ids: HashMap::new(),
        }
    }

    /// Whether no task runs.
    pub(super) fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    /// Starts `task` as `id`.
    pub(super) fn spawn(&mut self, id: TaskId, task: Task) {
        let files = DataFiles::new(&self.project_dir);
        let handle = match task {
            Task::Load => self.set.spawn(async move {
                let read = tokio::task::spawn_blocking(move || Dataset::read(&files)).await;
                Done::Loaded(match read {
                    Ok(Ok(data)) => Ok(data),
                    Ok(Err(error)) => Err(format!("{:#}", anyhow::Error::from(error))),
                    Err(error) => Err(format!("cannot read the data files: {error}")),
                })
            }),
        };
        self.ids.insert(handle.id(), id);
    }

    /// The next task to end, with what it gave back or how it failed (a panic).
    /// `None` when no task runs.
    pub(super) async fn next(&mut self) -> Option<(TaskId, Result<Done, String>)> {
        loop {
            let joined = self.set.join_next_with_id().await?;
            let (task, result) = match joined {
                Ok((task, done)) => (task, Ok(done)),
                Err(error) => (
                    error.id(),
                    Err(format!("a background task failed: {error}")),
                ),
            };
            if let Some(id) = self.ids.remove(&task) {
                return Some((id, result));
            }
        }
    }
}
