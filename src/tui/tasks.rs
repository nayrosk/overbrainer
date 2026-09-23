//! The TUI's background work: each task runs on its own, owns its inputs and
//! returns what the app needs when it ends. The app only ever sees results.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;

use tokio::task::JoinSet;

use super::editor::Edited;
use crate::config::EnvSource;
use crate::dataset::{Counts, DataFiles, Dataset, Deletion};
use crate::events::EventBus;
use crate::pipeline::{Ctx, SplitReport};
use crate::prompts::Prompts;

/// Identifies a task for the app.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct TaskId(pub(super) u64);

/// Work the app asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Task {
    /// Reads the dataset files.
    Load,
    /// Changes the dataset files, then runs `split`.
    Edit(Edit),
}

/// A change to the dataset files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Edit {
    /// An edit made in the editor.
    Change(Edited),
    /// A confirmed deletion, which must still remove `counts`.
    Delete {
        /// What is deleted.
        deletion: Deletion,
        /// What the confirmation said it removes.
        counts: Counts,
    },
}

/// What a task gives back.
#[derive(Debug)]
pub(super) enum Done {
    /// The dataset files, or why they cannot be read.
    Loaded(Result<Dataset, String>),
    /// What an edit did, or why nothing changed.
    Saved(Result<Saved, String>),
}

/// A saved edit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Saved {
    /// What was changed, for the status line.
    pub(super) message: String,
    /// The `split` run after it, or why it failed (the edit stays saved).
    pub(super) split: Result<SplitReport, String>,
}

/// What arrives from tasks while they run.
#[derive(Debug)]
pub(super) enum Msg {
    /// The editor ended, or could not be started.
    EditorExited(io::Result<ExitStatus>),
}

/// Applies `edit` to the files of the project in `project_dir`, freshly read,
/// then rewrites train and eval with `split`, with the settings of `env` (a
/// configuration error refuses the edit before anything is written).
///
/// # Errors
///
/// Returns why nothing changed: the settings cannot be loaded, the files cannot
/// be read or written, or the edit is refused.
pub(super) fn save(project_dir: &Path, edit: &Edit, env: EnvSource) -> Result<Saved, String> {
    let error = |error: anyhow::Error| format!("{error:#}");
    let settings = crate::config::load(project_dir, env).map_err(|e| error(e.into()))?;
    let files = DataFiles::new(project_dir);
    let mut data = Dataset::read(&files).map_err(|e| error(e.into()))?;
    let change = match edit {
        Edit::Change(Edited::Question { id, before, text }) => data.edit_question(id, before, text),
        Edit::Change(Edited::Answer { id, before, after }) => {
            data.edit_answer(id, before, after.clone())
        },
        Edit::Change(Edited::Subtopic { id, before, name }) => {
            data.rename_subtopic(id, before, name)
        },
        Edit::Delete { deletion, counts } => data.delete(deletion, *counts),
    }
    .map_err(|e| error(e.into()))?;
    data.save(&files, &change).map_err(|e| error(e.into()))?;
    let ctx = Ctx {
        settings: &settings,
        files: &files,
        prompts: &Prompts::empty(),
        bus: &EventBus::new(),
        topic: None,
        force: false,
    };
    let split = crate::pipeline::split(&ctx).map_err(|e| error(e.into()));
    Ok(Saved {
        message: change.message,
        split,
    })
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
            Task::Edit(edit) => {
                let dir = self.project_dir.clone();
                self.set.spawn(async move {
                    let saved =
                        tokio::task::spawn_blocking(move || save(&dir, &edit, EnvSource::Process))
                            .await;
                    Done::Saved(match saved {
                        Ok(saved) => saved,
                        Err(error) => Err(format!("the edit failed: {error}")),
                    })
                })
            },
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::dataset::{Example, Id, Question, rewrite};
    use crate::tui::snapshots::{MOVED, dataset, project};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    const LIMIT: Duration = Duration::from_secs(10);

    #[tokio::test]
    async fn a_load_reads_the_data_files_and_ends_with_its_id() -> TestResult {
        let dir = tempfile::tempdir()?;
        let files = DataFiles::new(dir.path());
        rewrite(&files.subtopics, &dataset().subtopics)?;
        let mut tasks = Tasks::new(dir.path());
        assert!(tasks.is_empty());
        tasks.spawn(TaskId(7), Task::Load);
        assert!(!tasks.is_empty());
        let next = tokio::time::timeout(LIMIT, tasks.next()).await?;
        let Some((TaskId(7), Ok(Done::Loaded(Ok(data))))) = next else {
            return Err(format!("unexpected end: {next:?}").into());
        };
        assert_eq!(data.subtopics.len(), 3);
        assert!(tasks.is_empty());
        assert!(tasks.next().await.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn a_load_of_a_broken_file_ends_with_its_error() -> TestResult {
        let dir = tempfile::tempdir()?;
        let files = DataFiles::new(dir.path());
        if let Some(parent) = files.answers.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&files.answers, "not json\n")?;
        let mut tasks = Tasks::new(dir.path());
        tasks.spawn(TaskId(1), Task::Load);
        let next = tokio::time::timeout(LIMIT, tasks.next()).await?;
        let Some((TaskId(1), Ok(Done::Loaded(Err(error))))) = next else {
            return Err(format!("unexpected end: {next:?}").into());
        };
        assert!(error.contains("answers.jsonl"), "{error}");
        Ok(())
    }

    #[test]
    fn a_saved_edit_rewrites_the_files_then_train_and_eval() -> TestResult {
        let dir = project()?;
        let files = DataFiles::new(dir.path());
        let subtopic = Id::subtopic("ownership", "Borrowing");
        let edit = Edit::Change(Edited::Question {
            id: Id::question(&subtopic, MOVED),
            before: MOVED.into(),
            text: "What happens to a borrow after a move?".into(),
        });
        let saved = save(dir.path(), &edit, EnvSource::Vars(Vec::new()))?;
        assert_eq!(
            saved.message,
            "question saved; its old answer was deleted: 1 question unanswered, run answers to answer it"
        );
        let report = saved.split?;
        assert_eq!((report.train, report.eval, report.orphaned), (1, 1, 1));
        let questions: Vec<Question> = crate::dataset::read(&files.questions)?;
        assert!(
            questions
                .iter()
                .any(|q| q.text == "What happens to a borrow after a move?")
        );
        let train: Vec<Example> = crate::dataset::read(&files.train)?;
        assert_eq!(train.len(), 1);
        Ok(())
    }

    #[test]
    fn a_broken_config_refuses_the_edit_before_writing() -> TestResult {
        let dir = project()?;
        let files = DataFiles::new(dir.path());
        let before = std::fs::read(&files.answers)?;
        std::fs::write(dir.path().join("overbrainer.toml"), "[project\n")?;
        let edit = Edit::Delete {
            deletion: Deletion::Answer(Id::question(
                &Id::subtopic("ownership", "Borrowing"),
                MOVED,
            )),
            counts: Counts {
                questions: 0,
                answers: 1,
            },
        };
        assert!(save(dir.path(), &edit, EnvSource::Vars(Vec::new())).is_err());
        assert_eq!(std::fs::read(&files.answers)?, before);
        assert!(!files.train.exists());
        Ok(())
    }
}
