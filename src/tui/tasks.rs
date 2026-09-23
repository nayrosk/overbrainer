//! The TUI's background work: each task runs on its own, owns its inputs and
//! returns what the app needs when it ends. The app only ever sees results.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use tokio::sync::broadcast::Receiver;
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::{AbortHandle, JoinHandle, JoinSet};
use tokio_util::sync::CancellationToken;

use super::editor::Edited;
use super::training::{Listing, list_runs, read_series};
use crate::cli::data::Command;
use crate::cli::front::{Frontend, Report};
use crate::cli::{StageArgs, TrainArgs, TrainCommand};
use crate::config::EnvSource;
use crate::dataset::{Counts, DataFiles, Dataset, Deletion};
use crate::events::{Event, EventBus};
use crate::pipeline::{Ctx, SplitReport};
use crate::prompts::Prompts;
use crate::train::TrainMetric;

/// Events kept for a TUI task's forwarder when it falls behind. `watch` publishes
/// every line of one tail read at once, at most 1 MiB, and a metrics line is at
/// least about 40 bytes: about 26 000 events, then an await lets the forwarder
/// drain.
pub(super) const BUS_CAPACITY: usize = 32_768;

/// How long a pipeline or training task waits for its forwarder once its flow is dropped:
/// only a sender kept by a task the flow spawned and has not yet dropped can
/// hold it that long.
const FORWARD_GRACE: Duration = Duration::from_secs(5);

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
    /// A pipeline command, on every topic, without `--force`.
    Pipeline(Command),
    /// A training flow of the command line.
    Train(TrainJob),
    /// Reads `runs/`, with the pod records.
    Runs,
    /// Reads the local metrics file of a run.
    Series(String),
}

/// A training flow, run exactly as `overbrainer train` runs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum TrainJob {
    /// `train attach <run-id>`.
    Attach(String),
    /// `train cancel <run-id>`.
    Cancel(String),
}

impl TrainJob {
    /// The arguments `overbrainer train` would get.
    fn args(&self) -> TrainArgs {
        let command = match self {
            Self::Attach(run_id) => TrainCommand::Attach {
                run_id: run_id.clone(),
            },
            Self::Cancel(run_id) => TrainCommand::Cancel {
                run_id: run_id.clone(),
            },
        };
        TrainArgs {
            command: Some(command),
            ..TrainArgs::default()
        }
    }
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
    /// How a pipeline command ended.
    Pipeline(Result<(), String>),
    /// How a training flow ended.
    Trained(Result<(), String>),
    /// What `runs/` holds.
    Runs(Listing),
    /// The local metrics of run `run`, `None` when its file cannot be read.
    Series {
        /// The run.
        run: String,
        /// Its metrics.
        series: Option<Vec<TrainMetric>>,
    },
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
    /// An event of a task's bus.
    Event(TaskId, Event),
    /// A task's forwarder fell behind and skipped this many events.
    Lagged(TaskId, u64),
    /// A line a task's flow reported.
    Report(TaskId, Report),
}

/// Forwards the events of task `id` from `events` as messages, until every
/// sender of its bus is dropped (the task ended). Lag is reported, never fatal.
/// The handle ends once the last event is sent.
pub(super) fn forward(
    id: TaskId,
    mut events: Receiver<Event>,
    messages: UnboundedSender<Msg>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let message = match events.recv().await {
                Ok(event) => Msg::Event(id, event),
                Err(RecvError::Lagged(skipped)) => Msg::Lagged(id, skipped),
                Err(RecvError::Closed) => break,
            };
            if messages.send(message).is_err() {
                break;
            }
        }
    })
}

/// Drops `front`, which closes its bus, then waits for `forwarder` to send what
/// is left and end, so every message of the task is in the app's inbox before
/// its end is. `what` names the task in the warning of a forwarder that does not
/// end within [`FORWARD_GRACE`].
async fn forwarded(front: Frontend, forwarder: JoinHandle<()>, what: &str) {
    drop(front);
    if tokio::time::timeout(FORWARD_GRACE, forwarder)
        .await
        .is_err()
    {
        tracing::warn!("the events of the {what} were not all forwarded");
    }
}

/// A TUI front end for task `id`: its own bus, forwarded to `messages` by the
/// task returned with it, its detach token and abandon flag, and its reports
/// sent as messages.
fn front_end(
    id: TaskId,
    messages: &UnboundedSender<Msg>,
    detach: &CancellationToken,
    abandon: &Arc<AtomicBool>,
) -> (Frontend, JoinHandle<()>) {
    let bus = EventBus::with_capacity(BUS_CAPACITY);
    let forwarder = forward(id, bus.subscribe(), messages.clone());
    let sink = messages.clone();
    let front = Frontend::Tui {
        bus,
        detach: detach.clone(),
        abandon: Arc::clone(abandon),
        report: Arc::new(move |report| {
            sink.send(Msg::Report(id, report)).ok();
        }),
    };
    (front, forwarder)
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
    messages: UnboundedSender<Msg>,
    set: JoinSet<Done>,
    ids: HashMap<tokio::task::Id, TaskId>,
    tokens: HashMap<TaskId, CancellationToken>,
    abandons: HashMap<TaskId, Arc<AtomicBool>>,
}

impl Tasks {
    /// No task yet, for the project in `project_dir`; running tasks send their
    /// messages to `messages`.
    pub(super) fn new(project_dir: &Path, messages: UnboundedSender<Msg>) -> Self {
        Self {
            project_dir: project_dir.to_path_buf(),
            messages,
            set: JoinSet::new(),
            ids: HashMap::new(),
            tokens: HashMap::new(),
            abandons: HashMap::new(),
        }
    }

    /// Sets the abandon flag of task `id` and cancels its token: a Runpod run
    /// still provisioning deletes its pod and fails, as Ctrl-C does on the
    /// command line; any other flow detaches.
    pub(super) fn abandon(&self, id: TaskId) {
        if let Some(abandon) = self.abandons.get(&id) {
            abandon.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        self.cancel(id);
    }

    /// Cancels the token of task `id`: a pipeline stops, a training detaches.
    pub(super) fn cancel(&self, id: TaskId) {
        if let Some(token) = self.tokens.get(&id) {
            token.cancel();
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
            Task::Pipeline(command) => self.spawn_pipeline(id, command),
            Task::Train(job) => self.spawn_train(id, job),
            Task::Runs => {
                let dir = self.project_dir.clone();
                self.set.spawn(async move {
                    let listing = tokio::task::spawn_blocking(move || list_runs(&dir)).await;
                    Done::Runs(listing.unwrap_or_else(|error| Listing {
                        runs: Err(format!("cannot list the runs: {error}")),
                        skipped: Vec::new(),
                    }))
                })
            },
            Task::Series(run) => {
                let dir = self.project_dir.clone();
                self.set.spawn(async move {
                    let id = run.clone();
                    let series = tokio::task::spawn_blocking(move || read_series(&dir, &id)).await;
                    Done::Series {
                        run,
                        series: series.ok().flatten(),
                    }
                })
            },
        };
        self.ids.insert(handle.id(), id);
    }

    /// Starts the pipeline `command` as `id`: its token drops the stage.
    fn spawn_pipeline(&mut self, id: TaskId, command: Command) -> AbortHandle {
        let dir = self.project_dir.clone();
        let token = CancellationToken::new();
        let abandon = Arc::new(AtomicBool::new(false));
        let (front, forwarder) = front_end(id, &self.messages, &token, &abandon);
        self.tokens.insert(id, token.clone());
        self.set.spawn(async move {
            let args = StageArgs::default();
            let outcome = {
                let run = crate::cli::data::run(&dir, command, &args, &front);
                tokio::select! {
                    result = run => result.map_err(|error| format!("{error:#}")),
                    () = token.cancelled() => {
                        Err("interrupted: the stage resumes on its next run".to_string())
                    },
                }
            };
            forwarded(front, forwarder, "stage").await;
            Done::Pipeline(outcome)
        })
    }

    /// Starts the training flow `job` as `id`: its token detaches the flow, and
    /// its abandon flag makes a Runpod provisioning delete its pod.
    fn spawn_train(&mut self, id: TaskId, job: TrainJob) -> AbortHandle {
        let dir = self.project_dir.clone();
        let token = CancellationToken::new();
        let abandon = Arc::new(AtomicBool::new(false));
        let (front, forwarder) = front_end(id, &self.messages, &token, &abandon);
        self.tokens.insert(id, token);
        self.abandons.insert(id, abandon);
        self.set.spawn(async move {
            // Never aborted: the flow shields its starts and cancels, and only
            // its token detaches it.
            let result = crate::cli::train::run(&dir, &job.args(), &front).await;
            forwarded(front, forwarder, "training").await;
            Done::Trained(result.map_err(|error| format!("{error:#}")))
        })
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
                self.tokens.remove(&id);
                self.abandons.remove(&id);
                return Some((id, result));
            }
        }
    }
}

#[cfg(test)]
mod tests {

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
        let mut tasks = Tasks::new(dir.path(), tokio::sync::mpsc::unbounded_channel().0);
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
        let mut tasks = Tasks::new(dir.path(), tokio::sync::mpsc::unbounded_channel().0);
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

    /// A real `train cancel` task on a run that has no job: the command line's
    /// flow refuses it, and its error comes back as the task's end.
    #[tokio::test]
    async fn a_cancel_task_runs_the_command_line_flow() -> TestResult {
        let dir = project()?;
        let run = "20260921-133200-a1b2";
        let mut record = crate::tui::snapshots::run(run, "homelab", crate::runs::RunState::Failed);
        record.job = None;
        crate::runs::Runs::new(dir.path()).save(&record)?;
        let mut tasks = Tasks::new(dir.path(), tokio::sync::mpsc::unbounded_channel().0);
        tasks.spawn(TaskId(3), Task::Train(TrainJob::Cancel(run.into())));
        let next = tokio::time::timeout(LIMIT, tasks.next()).await?;
        let Some((TaskId(3), Ok(Done::Trained(Err(error))))) = next else {
            return Err(format!("unexpected end: {next:?}").into());
        };
        assert_eq!(error, format!("run {run} has not started"));
        assert!(tasks.is_empty());
        Ok(())
    }

    /// The runs and a run's metrics are read in tasks, off the loop's thread.
    #[tokio::test]
    async fn runs_and_metrics_are_read_in_tasks() -> TestResult {
        let dir = tempfile::tempdir()?;
        let run = "20260921-133200-a1b2";
        let runs = crate::runs::Runs::new(dir.path());
        runs.save(&crate::tui::snapshots::run(
            run,
            "homelab",
            crate::runs::RunState::Running,
        ))?;
        std::fs::write(
            runs.run_dir(run)?.join(crate::train::METRICS_FILE),
            "{\"event\": \"log\", \"time\": 2, \"step\": 1, \"loss\": 1.5}\n",
        )?;
        let mut tasks = Tasks::new(dir.path(), tokio::sync::mpsc::unbounded_channel().0);
        tasks.spawn(TaskId(1), Task::Runs);
        let next = tokio::time::timeout(LIMIT, tasks.next()).await?;
        let Some((TaskId(1), Ok(Done::Runs(Listing { runs: Ok(rows), .. })))) = next else {
            return Err(format!("unexpected end: {next:?}").into());
        };
        assert_eq!(rows.len(), 1);
        tasks.spawn(TaskId(2), Task::Series(run.into()));
        let next = tokio::time::timeout(LIMIT, tasks.next()).await?;
        let Some((TaskId(2), Ok(Done::Series { series, .. }))) = next else {
            return Err(format!("unexpected end: {next:?}").into());
        };
        assert_eq!(series.map(|s| s.len()), Some(1));
        Ok(())
    }

    /// Abandoning a task sets its flag and cancels its token; a task already
    /// ended is ignored.
    #[test]
    fn abandon_sets_the_flag_then_cancels_the_token() {
        let mut tasks = Tasks::new(
            Path::new("/nonexistent"),
            tokio::sync::mpsc::unbounded_channel().0,
        );
        let token = CancellationToken::new();
        let flag = Arc::new(AtomicBool::new(false));
        tasks.tokens.insert(TaskId(1), token.clone());
        tasks.abandons.insert(TaskId(1), Arc::clone(&flag));
        tasks.abandon(TaskId(2));
        assert!(!flag.load(std::sync::atomic::Ordering::SeqCst));
        tasks.abandon(TaskId(1));
        assert!(flag.load(std::sync::atomic::Ordering::SeqCst));
        assert!(token.is_cancelled());
    }

    /// A copy of a message of the pipeline task, which [`Msg`] cannot clone.
    fn copy(message: &Msg) -> Option<Msg> {
        match message {
            Msg::Event(id, event) => Some(Msg::Event(*id, event.clone())),
            Msg::Lagged(id, skipped) => Some(Msg::Lagged(*id, *skipped)),
            Msg::Report(id, report) => Some(Msg::Report(*id, report.clone())),
            Msg::EditorExited(_) => None,
        }
    }

    /// A real `split` task: every message is sent before it ends, and the app
    /// keeps them whether they are handled before or after its end.
    #[tokio::test]
    async fn a_split_task_sends_every_message_before_it_ends() -> TestResult {
        use crossterm::event::KeyCode;

        use crate::events::Stage;
        use crate::tui::app::Effect;
        use crate::tui::pipeline::StageState;
        use crate::tui::snapshots::{app, key};

        let dir = project()?;
        let (messages, mut inbox) = tokio::sync::mpsc::unbounded_channel();
        let mut tasks = Tasks::new(dir.path(), messages);
        tasks.spawn(TaskId(1), Task::Pipeline(Command::Split));
        let next = tokio::time::timeout(LIMIT, tasks.next()).await?;
        let Some((TaskId(1), Ok(Done::Pipeline(Ok(()))))) = next else {
            return Err(format!("unexpected end: {next:?}").into());
        };
        let mut received = Vec::new();
        while let Ok(message) = inbox.try_recv() {
            received.push(message);
        }
        let started = received.iter().any(|message| {
            matches!(
                message,
                Msg::Event(
                    TaskId(1),
                    Event::StageStarted {
                        stage: Stage::Split,
                        ..
                    }
                )
            )
        });
        let finished = received.iter().any(|message| {
            matches!(
                message,
                Msg::Event(
                    TaskId(1),
                    Event::StageFinished {
                        stage: Stage::Split,
                        ..
                    }
                )
            )
        });
        assert!(started && finished, "{received:?}");
        let line = received.iter().find_map(|message| match message {
            Msg::Report(TaskId(1), Report::Line(line)) => Some(line.clone()),
            _ => None,
        });
        let Some(line) = line else {
            return Err(format!("no split line: {received:?}").into());
        };
        assert!(line.starts_with("split: "), "{line}");
        for done_first in [false, true] {
            let mut app = app();
            let mut effects = Vec::new();
            for code in [
                KeyCode::Char('r'),
                KeyCode::Down,
                KeyCode::Down,
                KeyCode::Down,
                KeyCode::Enter,
            ] {
                effects.extend(app.on_input(&key(code)));
            }
            assert_eq!(
                effects,
                [Effect::Spawn(TaskId(1), Task::Pipeline(Command::Split))]
            );
            if done_first {
                app.on_done(TaskId(1), Ok(Done::Pipeline(Ok(()))));
            }
            for message in received.iter().filter_map(copy) {
                app.on_message(message);
            }
            if !done_first {
                app.on_done(TaskId(1), Ok(Done::Pipeline(Ok(()))));
            }
            assert_eq!(
                app.pipeline.results,
                std::slice::from_ref(&line),
                "done first: {done_first}"
            );
            let row = app.pipeline.row(Stage::Split);
            assert_eq!(row.state, StageState::Done, "done first: {done_first}");
            assert_eq!(row.finished, row.total, "done first: {done_first}");
            assert!(row.total > 0);
        }
        Ok(())
    }
}
