//! The state of the TUI and what keys and events do to it. Nothing here draws,
//! reads the clock or touches the terminal: the loop feeds it input, ticks and
//! signals, and renders it.

use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::time::{Duration, SystemTime};

use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use tracing::Level;

use super::dataset::{DatasetView, Model, Node, TopicInfo};
use super::editor::{self, Session, Target};
use super::motion::{Motion, MotionLevel};
use super::pipeline::{PipelineView, STAGES, command_name};
use super::start::StartPlan;
use super::tasks::{Done, Edit, Msg, Saved, Task, TaskId};
use super::theme::Theme;
use super::training::TrainingView;
use crate::cli::data::Command;
use crate::cli::front::Report;
use crate::config::Settings;
use crate::dataset::{AnswerText, Counts, Dataset, Deletion, Id};
use crate::logging::LogBuffer;

/// How long a status message stays on the status line.
const STATUS_FOR: Duration = Duration::from_secs(10);
/// Lines moved by `PgUp` and `PgDn`.
pub(super) const PAGE: u16 = 10;

/// What the TUI knows of the project, read from `overbrainer.toml` when it starts.
#[derive(Debug, Clone)]
pub(super) struct Project {
    /// `project.name`.
    pub(super) name: String,
    /// The project directory.
    pub(super) dir: PathBuf,
    /// The configured topics, in order.
    pub(super) topics: Vec<TopicInfo>,
    /// `pipeline.eval_ratio`.
    pub(super) eval_ratio: f64,
    /// `pipeline.concurrency`.
    pub(super) concurrency: usize,
    /// `training.target`, when training is configured.
    pub(super) target: Option<String>,
}

impl Project {
    /// The project in `dir`, configured by `settings`.
    pub(super) fn new(dir: &Path, settings: &Settings) -> Self {
        Self {
            name: settings.project.name.clone(),
            dir: dir.to_path_buf(),
            topics: settings
                .topics
                .iter()
                .map(|topic| TopicInfo {
                    name: topic.name.clone(),
                    description: topic.description.clone(),
                    subtopics: topic.subtopics,
                    questions_per_subtopic: topic.questions_per_subtopic,
                })
                .collect(),
            eval_ratio: settings.pipeline.eval_ratio,
            concurrency: settings.pipeline.concurrency,
            target: settings
                .training
                .as_ref()
                .map(|training| training.target.clone()),
        }
    }
}

/// What the app asks the loop to do.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Effect {
    /// Start a background task.
    Spawn(TaskId, Task),
    /// Cancel the token of a task: a pipeline stops, a training detaches.
    Cancel(TaskId),
    /// Set the abandon flag of a training task and cancel its token.
    Abandon(TaskId),
    /// Hand the terminal to the editor `command`, on the file `path`.
    OpenEditor {
        /// The program and its arguments.
        command: Vec<String>,
        /// The file to edit.
        path: PathBuf,
    },
}

/// The four views, switched with `1` to `4`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum View {
    /// The dataset tree, its details and stats.
    Dataset,
    /// The pipeline stages.
    Pipeline,
    /// Training runs.
    Training,
    /// Log lines.
    Logs,
}

impl View {
    /// Every view, in tab order.
    pub(super) const ALL: [Self; 4] = [Self::Dataset, Self::Pipeline, Self::Training, Self::Logs];

    /// The tab title.
    pub(super) fn title(self) -> &'static str {
        match self {
            Self::Dataset => "Dataset",
            Self::Pipeline => "Pipeline",
            Self::Training => "Training",
            Self::Logs => "Logs",
        }
    }

    /// Position in [`View::ALL`].
    pub(super) fn index(self) -> usize {
        match self {
            Self::Dataset => 0,
            Self::Pipeline => 1,
            Self::Training => 2,
            Self::Logs => 3,
        }
    }

    fn shifted(self, by: usize) -> Self {
        Self::ALL[(self.index() + by) % Self::ALL.len()]
    }
}

/// How a status message is shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Severity {
    /// A result or a note.
    Info,
    /// A refusal or a warning.
    Warn,
    /// A failure.
    Error,
}

/// The message on the left of the status line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Status {
    /// The message.
    pub(super) text: String,
    /// How it is shown.
    pub(super) severity: Severity,
    /// When it was set, by the app's clock.
    pub(super) at: SystemTime,
}

/// What is drawn over the view.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Overlay {
    /// The key table.
    Help,
    /// A question answered with `y` or `n`.
    Confirm(Confirm),
    /// The `r` menu, with its selected entry.
    Menu(usize),
}

/// The entries of the `r` menu: every pipeline command, all topics, no `--force`.
pub(super) const MENU: [(Command, &str); 5] = [
    (Command::Subtopics, "generate the missing subtopics"),
    (Command::Questions, "fill the subtopics with questions"),
    (Command::Answers, "ask the parent to answer (paid requests)"),
    (Command::Split, "rebuild train and eval"),
    (Command::Run, "all four stages; training starts only with t"),
];

/// A confirmation dialog: `y` runs its action, anything else closes it.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Confirm {
    /// The title.
    pub(super) title: String,
    /// The question, one paragraph per entry.
    pub(super) text: Vec<String>,
    /// What `y` does, in one word.
    pub(super) yes: &'static str,
    /// What `n` does, in one word.
    pub(super) no: &'static str,
    /// What `y` runs.
    pub(super) action: Action,
}

/// What a confirmed dialog runs.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Action {
    /// A deletion, which must still remove `counts`.
    Delete {
        /// What is deleted.
        deletion: Deletion,
        /// What the dialog said it removes.
        counts: Counts,
    },
    /// Quitting while work runs.
    Quit,
    /// Cancelling the job of run `0`.
    Cancel(String),
    /// Starting a training run as planned.
    Start(Box<StartPlan>),
    /// Abandoning Runpod runs still provisioning, when quitting.
    Abandon(Vec<TaskId>),
    /// Abandoning the Runpod start `0` still provisioning, asked for with `c`.
    AbandonStart(TaskId),
}

/// What an exit note added while leaving is about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum NoteOf {
    /// The pipeline stage stopped.
    Stage,
    /// Run `0`, detached or not cancelled.
    Run(String),
}

/// Why the loop ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Exit {
    /// The user quit.
    Quit,
    /// A process signal (SIGINT, SIGTERM or SIGHUP) came from outside.
    Signal,
}

/// The Logs view's state: the level shown and where it is scrolled.
///
/// Scrolling back pins the view on a line, so the lines on screen stay put while
/// new ones arrive; `G` or End follows the tail again. `f` changes which lines
/// match, so it follows the tail again too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LogView {
    /// The least severe level shown.
    pub(super) min: Level,
    /// While scrolled back, the `seq` of the line at the bottom of the view;
    /// `None` follows the tail.
    pub(super) anchor: Option<u64>,
    /// Rows of lines the view showed at the last draw.
    pub(super) height: usize,
}

impl LogView {
    /// Matching lines newer than the bottom of the view. Exact: it never goes
    /// past what a full view can show, and an anchor dropped from the buffer
    /// counts as its oldest line.
    pub(super) fn offset(&self, logs: &LogBuffer) -> usize {
        let Some(anchor) = self.anchor else {
            return 0;
        };
        let matching = logs.window(self.min, 0, 0).matching;
        logs.newer(self.min, anchor)
            .min(matching.saturating_sub(self.height.max(1)))
    }

    /// The next level `f` selects: error, warn, info, debug, trace, then error.
    fn next_level(level: Level) -> Level {
        match level {
            Level::ERROR => Level::WARN,
            Level::WARN => Level::INFO,
            Level::INFO => Level::DEBUG,
            Level::DEBUG => Level::TRACE,
            _ => Level::ERROR,
        }
    }
}

/// The whole state of the TUI.
pub(super) struct App {
    /// The project.
    pub(super) project: Project,
    /// Styles.
    pub(super) theme: Theme,
    /// The log lines captured in TUI mode.
    pub(super) logs: LogBuffer,
    /// The view shown.
    pub(super) view: View,
    /// What is drawn over the view, if anything.
    pub(super) overlay: Option<Overlay>,
    /// The status message, until it expires.
    pub(super) status: Option<Status>,
    /// The app's clock, set by each tick: the only time the views use.
    pub(super) now: SystemTime,
    /// The Dataset view's state.
    pub(super) dataset: DatasetView,
    /// The Logs view's state.
    pub(super) log_view: LogView,
    /// The load of the data files running, if any.
    pub(super) load: Option<TaskId>,
    /// When the last load started, by the motion clock.
    pub(super) load_at: Option<Duration>,
    /// What moves on screen, and its clock.
    pub(super) motion: Motion,
    /// Whether a reload was asked for while a load ran: it starts once that load
    /// ends, so it reads what changed meanwhile.
    reload_pending: bool,
    /// The edit being saved, if any.
    pub(super) edit: Option<TaskId>,
    /// The edit open in the editor or being saved.
    pub(super) editing: Option<Session>,
    /// The editor command.
    pub(super) editor: Vec<String>,
    /// The Pipeline view's state.
    pub(super) pipeline: PipelineView,
    /// The pipeline task running, if any.
    pub(super) pipeline_task: Option<TaskId>,
    /// The pipeline task that ended last, until the next one starts: its late
    /// messages still update the view.
    pipeline_last: Option<TaskId>,
    /// The Training view's state, with the training tasks.
    pub(super) training: TrainingView,
    /// When `runs/` was last read.
    pub(super) refreshed: SystemTime,
    /// The task preparing a training start, if any.
    pub(super) prepare: Option<TaskId>,
    /// The task looking up list prices, if any; an earlier one's result is
    /// ignored.
    pub(super) prices: Option<TaskId>,
    /// Lines printed on stderr once the terminal is restored.
    pub(super) exit_notes: Vec<String>,
    /// Which of `exit_notes` were added because the TUI was leaving (a stage
    /// stopped, a run detached), with what they are about: dropped once that
    /// work is taken up again, never merely because the user stays.
    leaving_notes: Vec<(usize, NoteOf)>,
    /// How the TUI ends, set once it is to end while it waits for work to end:
    /// the pipeline task it stopped, and an edit being saved (never cut).
    pub(super) leaving: Option<Exit>,
    next_task: u64,
    /// Whether something changed since the last draw.
    pub(super) dirty: bool,
    /// Set once the loop must end, with why.
    pub(super) exit: Option<Exit>,
    /// The log sequence number last looked at.
    seen_log: u64,
}

impl App {
    /// A new app for `project`, logging into `logs`, at `now`.
    pub(super) fn new(project: Project, logs: LogBuffer, theme: &Theme, now: SystemTime) -> Self {
        Self {
            project,
            theme: *theme,
            seen_log: logs.seq(),
            logs,
            view: View::Dataset,
            overlay: None,
            status: None,
            now,
            dataset: DatasetView {
                warn: theme.warn,
                ..DatasetView::default()
            },
            load: None,
            load_at: None,
            motion: Motion::new(MotionLevel::Off),
            reload_pending: false,
            edit: None,
            editing: None,
            editor: vec!["vi".to_string()],
            pipeline: PipelineView::default(),
            pipeline_task: None,
            pipeline_last: None,
            training: TrainingView::default(),
            refreshed: Self::never(),
            prepare: None,
            prices: None,
            exit_notes: Vec::new(),
            leaving_notes: Vec::new(),
            leaving: None,
            next_task: 0,
            log_view: LogView {
                min: Level::INFO,
                anchor: None,
                height: 0,
            },
            dirty: true,
            exit: None,
        }
    }

    /// Sets the status message.
    pub(super) fn say(&mut self, severity: Severity, text: impl Into<String>) {
        self.status = Some(Status {
            text: text.into(),
            severity,
            at: self.now,
        });
        self.dirty = true;
    }

    /// What to do when the loop starts: load the data.
    pub(super) fn start(&mut self) -> Vec<Effect> {
        self.reload()
    }

    /// A new task ID.
    pub(super) fn task_id(&mut self) -> TaskId {
        self.next_task += 1;
        TaskId(self.next_task)
    }

    /// Reloads the data files; while a load runs, one more starts when it ends.
    fn reload(&mut self) -> Vec<Effect> {
        if self.load.is_some() {
            self.reload_pending = true;
            return Vec::new();
        }
        let id = self.task_id();
        self.load = Some(id);
        self.load_at = Some(self.motion.clock());
        vec![Effect::Spawn(id, Task::Load)]
    }

    /// The work running, as the footer shows it, each with a spinner.
    pub(super) fn work(&self) -> Vec<String> {
        let mut work = Vec::new();
        if self.leaving.is_some() {
            work.push("quitting: waiting".to_string());
        }
        if self.load.is_some() {
            work.push("loading".to_string());
        }
        if let Some(progress) = self.pipeline.progress() {
            work.push(progress);
        }
        if self.edit.is_some() {
            work.push("saving".to_string());
        }
        if self.prepare.is_some() || self.prices.is_some() {
            work.push("preparing a run".to_string());
        }
        let followed = self.training.tasks.len();
        if followed > 0 {
            work.push(count(followed, "training task"));
        }
        work
    }

    /// Why the dataset cannot be changed now, if it cannot: a stage, an edit or
    /// a training start (until its job started) runs in this TUI.
    pub(super) fn lock(&self) -> Option<String> {
        if self.pipeline_task.is_some() {
            let running = self.pipeline.progress().unwrap_or_default();
            let stage = running.split(' ').next().unwrap_or("a stage");
            return Some(format!("{stage} is running"));
        }
        if self.edit.is_some() || self.editing.is_some() {
            return Some("an edit is being saved".to_string());
        }
        if let Some(follow) = self.training.tasks.values().find(|f| f.starting()) {
            return Some(format!("{} is starting", follow.run()));
        }
        None
    }

    /// Refuses a change to the dataset while it is locked or the TUI is leaving.
    fn locked(&mut self) -> bool {
        self.refuse_new("edits resume when it ends", "edit")
    }

    /// Refuses new work of kind `new` (a stage, an edit, a run) while the data
    /// is locked, saying `wait` after why, or while the TUI is leaving (a quit
    /// waits or a signal came): the status line says why. Returns whether it
    /// refused.
    pub(super) fn refuse_new(&mut self, wait: &str, new: &str) -> bool {
        let reason = match self.lock() {
            Some(reason) => format!("{reason}; {wait}"),
            None if self.leaving.is_some() => format!("quitting; no {new} starts"),
            None => return false,
        };
        self.say(Severity::Warn, format!("refused: {reason}"));
        true
    }

    /// Handles the end of task `id`: what it gave back, or how it failed.
    /// A load's result is used only when `id` is the load running; a reload asked
    /// for meanwhile starts then. An edit's end reloads the data.
    pub(super) fn on_done(&mut self, id: TaskId, result: Result<Done, String>) -> Vec<Effect> {
        self.dirty = true;
        match result {
            Ok(Done::Loaded(loaded)) => self.on_loaded(id, loaded),
            Ok(Done::Saved(saved)) => self.saved(saved),
            Ok(Done::Pipeline(outcome)) => self.pipeline_ended(id, outcome),
            Ok(Done::Trained(result)) => self.trained(id, result),
            Ok(Done::Runs(listing)) => self.listed(id, listing),
            Ok(Done::Series { run, series }) => self.series_read(id, run, series),
            Ok(Done::Prepared(plan)) if self.prepare == Some(id) => self.prepared(plan),
            Ok(Done::Prices(prices)) if self.prices == Some(id) => {
                self.priced(&prices);
                Vec::new()
            },
            Ok(Done::Prepared(_) | Done::Prices(_)) => Vec::new(),
            Err(error) => self.failed(id, error),
        }
    }

    /// Load `id` read `loaded`: shown when it is the load running, else ignored.
    fn on_loaded(&mut self, id: TaskId, loaded: Result<Dataset, String>) -> Vec<Effect> {
        if self.load != Some(id) {
            return Vec::new();
        }
        self.load = None;
        match loaded {
            Ok(data) => self.dataset.loaded(data, &self.project.topics),
            Err(error) => self.load_failed(error),
        }
        self.pending_reload()
    }

    /// Task `id` failed (a panic): an edit keeps its typed text, a load and a
    /// stage show why.
    fn failed(&mut self, id: TaskId, error: String) -> Vec<Effect> {
        tracing::error!("{error}");
        if self.edit == Some(id) {
            return self.saved(Err(error));
        }
        if self.pipeline_task == Some(id) {
            return self.pipeline_ended(id, Err(error));
        }
        if self.training.tasks.contains_key(&id) {
            return self.trained(id, Err(error));
        }
        if self.prepare == Some(id) {
            return self.prepared(Err(error));
        }
        if self.prices == Some(id) {
            // Every price unknown: the dialog never keeps waiting.
            self.priced(&Vec::new());
            return Vec::new();
        }
        if self.read_failed(id, &error) {
            return Vec::new();
        }
        if self.load == Some(id) {
            self.load = None;
            self.load_failed(error);
            return self.pending_reload();
        }
        self.say(Severity::Error, error);
        Vec::new()
    }

    /// Starts the reload asked for while a load ran, once no load runs.
    fn pending_reload(&mut self) -> Vec<Effect> {
        if self.load.is_none() && std::mem::take(&mut self.reload_pending) {
            return self.reload();
        }
        Vec::new()
    }

    /// An edit task ended: says what it did, or keeps the typed text of a refused
    /// edit, then reloads the data.
    fn saved(&mut self, saved: Result<Saved, String>) -> Vec<Effect> {
        self.edit = None;
        let session = self.editing.take();
        match saved {
            Ok(saved) => self.applied(session.as_ref(), saved),
            Err(error) => self.refused(session.as_ref(), error),
        }
        self.leave_when_idle();
        if self.exit.is_some() {
            return Vec::new();
        }
        self.reload()
    }

    /// Pipeline task `id` ended with `outcome`: says so, then reloads the data.
    fn pipeline_ended(&mut self, id: TaskId, outcome: Result<(), String>) -> Vec<Effect> {
        if self.pipeline_task != Some(id) {
            return Vec::new();
        }
        self.pipeline_task = None;
        self.pipeline_last = Some(id);
        self.pipeline.stopped();
        let name = self.pipeline.command.map_or("stage", command_name);
        let (severity, said) = match &outcome {
            Ok(()) if self.pipeline.command == Some(Command::Run) => (
                Severity::Info,
                "run finished after split; training starts only with t".to_string(),
            ),
            Ok(()) => (Severity::Info, format!("{name} finished")),
            Err(error) => (Severity::Error, format!("{name}: {error}")),
        };
        if self.leaving.is_some() {
            self.note_leaving(said.clone(), NoteOf::Stage);
        }
        self.say(severity, said);
        self.pipeline.outcome = Some(outcome);
        self.leave_when_idle();
        if self.exit.is_some() {
            return Vec::new();
        }
        self.reload()
    }

    /// A message of a task: the events, lag and lines of a training task, or of
    /// the pipeline task running or the last one (a message handled after its
    /// end); those of any other task are dropped.
    pub(super) fn on_message(&mut self, message: Msg) -> Vec<Effect> {
        self.dirty = true;
        let id = match &message {
            Msg::Event(id, _) | Msg::Lagged(id, _) | Msg::Report(id, _) => *id,
            Msg::EditorExited(_) => return Vec::new(),
        };
        if self.training.is_training(id) {
            return self.on_training_message(id, message);
        }
        if !self.is_pipeline(id) {
            return Vec::new();
        }
        match message {
            Msg::Event(_, event) => self.pipeline.event(&event),
            Msg::Lagged(_, skipped) => self.pipeline.skipped += skipped,
            Msg::Report(_, Report::Line(line)) => self.pipeline.results.push(line),
            Msg::Report(_, Report::RunCreated(_)) | Msg::EditorExited(_) => {},
        }
        Vec::new()
    }

    /// Whether task `id` is the pipeline task running or the last one.
    fn is_pipeline(&self, id: TaskId) -> bool {
        self.pipeline_task == Some(id) || self.pipeline_last == Some(id)
    }

    /// `r`: the menu of pipeline commands, unless the data is locked or the TUI
    /// is leaving.
    fn run_menu(&mut self) {
        if self.refuse_new("one task at a time", "stage") {
            return;
        }
        self.overlay = Some(Overlay::Menu(0));
    }

    /// A key in the `r` menu: moves, runs the selected command, or closes it.
    fn on_menu_key(&mut self, selected: usize, code: KeyCode) -> Vec<Effect> {
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.overlay = Some(Overlay::Menu(selected.saturating_sub(1)));
            },
            KeyCode::Down | KeyCode::Char('j') => {
                self.overlay = Some(Overlay::Menu((selected + 1).min(MENU.len() - 1)));
            },
            KeyCode::Enter => {
                self.overlay = None;
                let (command, _) = MENU[selected.min(MENU.len() - 1)];
                return self.run_pipeline(command);
            },
            _ => self.overlay = None,
        }
        Vec::new()
    }

    /// Starts the pipeline `command` and shows the Pipeline view, unless the
    /// data is locked or the TUI is leaving.
    fn run_pipeline(&mut self, command: Command) -> Vec<Effect> {
        if self.refuse_new("one task at a time", "stage") {
            return Vec::new();
        }
        self.forget_notes(&NoteOf::Stage);
        let id = self.task_id();
        self.pipeline_task = Some(id);
        self.pipeline_last = None;
        self.pipeline.started(command, self.project.concurrency);
        self.view = View::Pipeline;
        vec![Effect::Spawn(id, Task::Pipeline(command))]
    }

    /// Adds `note`, about `of`, to the exit notes because the TUI is leaving:
    /// it stays if the user stays, until `of` is taken up again.
    pub(super) fn note_leaving(&mut self, note: String, of: NoteOf) {
        self.leaving_notes.push((self.exit_notes.len(), of));
        self.exit_notes.push(note);
    }

    /// Drops the exit notes added while leaving about `of`, which is taken up
    /// again (a run followed again, the stage run again): they no longer apply.
    pub(super) fn forget_notes(&mut self, of: &NoteOf) {
        let dropped: Vec<usize> = self
            .leaving_notes
            .iter()
            .filter(|(_, about)| about == of)
            .map(|(index, _)| *index)
            .collect();
        if dropped.is_empty() {
            return;
        }
        let mut index = 0;
        self.exit_notes.retain(|_| {
            let keep = !dropped.contains(&index);
            index += 1;
            keep
        });
        self.leaving_notes.retain(|(_, about)| about != of);
        for (index, _) in &mut self.leaving_notes {
            *index -= dropped.iter().filter(|gone| **gone < *index).count();
        }
    }

    /// Ends the TUI once nothing it waits for runs.
    pub(super) fn leave_when_idle(&mut self) {
        let idle =
            self.edit.is_none() && self.pipeline_task.is_none() && self.training.tasks.is_empty();
        if self.leaving.is_some() && idle {
            self.exit = self.leaving;
        }
    }

    /// A change was saved: drops the temp file of `session`, and says what the
    /// change did, and whether `split` failed after it.
    fn applied(&mut self, session: Option<&Session>, saved: Saved) {
        if let Some(session) = session {
            std::fs::remove_file(&session.path).ok();
        }
        let (message, severity) = match &saved.split {
            Ok(_) => (saved.message, Severity::Info),
            Err(error) => (
                format!("{}, but split failed: {error}; run split", saved.message),
                Severity::Warn,
            ),
        };
        self.dataset.split = saved.split.ok();
        if self.leaving.is_some() {
            self.exit_notes
                .push(format!("a change was saved: {message}"));
        }
        self.say(severity, message);
    }

    /// A change was refused: keeps the typed text of `session`, if any.
    fn refused(&mut self, session: Option<&Session>, error: String) {
        if let Some(session) = session {
            self.kept(session, &error);
            return;
        }
        if self.leaving.is_some() {
            self.exit_notes
                .push(format!("a change was refused ({error}); nothing changed"));
        }
        self.say(Severity::Error, error);
    }

    /// Keeps the temp file of `session` after `error`, saying where it is, on the
    /// status line and on exit.
    fn kept(&mut self, session: &Session, error: &str) {
        let path = session.path.display();
        self.exit_notes.push(format!(
            "an edit was refused ({error}); its text is kept in {path}"
        ));
        self.say(
            Severity::Error,
            format!("refused: {error}; your text is kept in {path}"),
        );
    }

    /// The editor ended with `status`: checks the edited file and saves it.
    pub(super) fn on_editor_exit(&mut self, status: io::Result<ExitStatus>) -> Vec<Effect> {
        let Some(session) = self.editing.clone() else {
            return Vec::new();
        };
        let failed = match status {
            Ok(status) if status.success() => None,
            Ok(status) => Some(status.code().map_or_else(
                || "the editor was killed; nothing changed".to_string(),
                |code| format!("the editor exited with status {code}; nothing changed"),
            )),
            Err(error) => Some(format!("cannot run the editor: {error}; nothing changed")),
        };
        if let Some(message) = failed {
            return self.drop_edit(&session, Severity::Warn, message);
        }
        let bytes = match std::fs::read(&session.path) {
            Ok(bytes) => bytes,
            Err(error) => {
                self.editing = None;
                self.kept(&session, &format!("cannot read the edited file: {error}"));
                return Vec::new();
            },
        };
        match editor::check(&session, bytes) {
            Ok(edited) => {
                let id = self.task_id();
                self.edit = Some(id);
                vec![Effect::Spawn(id, Task::Edit(Edit::Change(edited)))]
            },
            Err(refusal) if refusal.keep => {
                self.editing = None;
                self.kept(&session, &refusal.message);
                Vec::new()
            },
            Err(refusal) => self.drop_edit(&session, Severity::Info, refusal.message),
        }
    }

    /// Ends an edit that changed nothing, removing its temp file.
    fn drop_edit(&mut self, session: &Session, severity: Severity, message: String) -> Vec<Effect> {
        std::fs::remove_file(&session.path).ok();
        self.editing = None;
        self.say(severity, message);
        Vec::new()
    }

    /// Notes, for the exit, the temp file of an edit the TUI ended during (a
    /// signal, a terminal error): it may hold typed text.
    pub(super) fn abandon_edit(&mut self) {
        if let Some(session) = self.editing.take() {
            self.exit_notes.push(format!(
                "an edit was not saved; its text is kept in {}",
                session.path.display()
            ));
        }
    }

    /// Shows why the data could not be loaded, in the view and on the status line.
    fn load_failed(&mut self, error: String) {
        self.dataset.model = None;
        self.dataset.error = Some(error.clone());
        self.say(Severity::Error, error);
    }

    /// Handles one terminal event.
    pub(super) fn on_input(&mut self, event: &Event) -> Vec<Effect> {
        match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => self.on_key(*key),
            Event::Resize(..) => {
                self.dirty = true;
                Vec::new()
            },
            _ => Vec::new(),
        }
    }

    fn on_key(&mut self, key: KeyEvent) -> Vec<Effect> {
        self.dirty = true;
        let ctrl_c =
            key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c');
        if ctrl_c {
            self.close_overlay();
            self.dataset.input = None;
            return self.quit();
        }
        if self.view == View::Dataset && self.dataset.input.is_some() {
            if key.modifiers.difference(KeyModifiers::SHIFT).is_empty() {
                self.on_filter_key(key.code);
            }
            return Vec::new();
        }
        match &self.overlay {
            Some(Overlay::Menu(selected)) => return self.on_menu_key(*selected, key.code),
            Some(Overlay::Help) => {
                if matches!(key.code, KeyCode::Esc | KeyCode::Char('?' | 'q')) {
                    self.overlay = None;
                }
                return Vec::new();
            },
            Some(Overlay::Confirm(_)) => return self.on_confirm_key(key.code),
            None => {},
        }
        if key.code == KeyCode::Char('q') {
            return self.quit();
        }
        if self.leaving == Some(Exit::Quit) && matches!(key.code, KeyCode::Esc | KeyCode::Char('n'))
        {
            self.stay();
            return Vec::new();
        }
        match key.code {
            KeyCode::Char('R') => {
                let mut effects = self.reload();
                effects.extend(self.refresh_runs());
                return effects;
            },
            KeyCode::Char('r') => self.run_menu(),
            KeyCode::Char('?') => self.overlay = Some(Overlay::Help),
            KeyCode::Char('1') => return self.show(View::Dataset),
            KeyCode::Char('2') => return self.show(View::Pipeline),
            KeyCode::Char('3') => return self.show(View::Training),
            KeyCode::Char('4') => return self.show(View::Logs),
            KeyCode::Tab => return self.show(self.view.shifted(1)),
            KeyCode::BackTab => return self.show(self.view.shifted(View::ALL.len() - 1)),
            code => return self.on_view_key(code),
        }
        Vec::new()
    }

    /// Shows `view`; the Training view reads `runs/` again. Leaving a view never
    /// touches a task.
    fn show(&mut self, view: View) -> Vec<Effect> {
        self.view = view;
        if view == View::Training {
            return self.refresh_runs();
        }
        Vec::new()
    }

    /// Closes the overlay and returns it. Closing the start dialog forgets its
    /// list price lookup: nothing shows its prices any more.
    fn close_overlay(&mut self) -> Option<Overlay> {
        let overlay = self.overlay.take();
        if let Some(Overlay::Confirm(Confirm {
            action: Action::Start(_),
            ..
        })) = &overlay
        {
            self.prices = None;
        }
        overlay
    }

    /// `y` runs the dialog's action; any other key closes it.
    fn on_confirm_key(&mut self, code: KeyCode) -> Vec<Effect> {
        let Some(Overlay::Confirm(confirm)) = self.close_overlay() else {
            return Vec::new();
        };
        if code != KeyCode::Char('y') {
            return Vec::new();
        }
        match confirm.action {
            Action::Quit => {
                let effects = self.leave(Exit::Quit);
                self.offer_abandon();
                effects
            },
            Action::Cancel(run_id) => self.cancel_run(&run_id),
            Action::Start(plan) => self.start_run(&plan),
            Action::Abandon(tasks) => self.abandon(&tasks),
            Action::AbandonStart(task) => self.abandon_start(task),
            Action::Delete { deletion, counts } => {
                if self.locked() {
                    return Vec::new();
                }
                let id = self.task_id();
                self.edit = Some(id);
                vec![Effect::Spawn(
                    id,
                    Task::Edit(Edit::Delete { deletion, counts }),
                )]
            },
        }
    }

    fn on_view_key(&mut self, code: KeyCode) -> Vec<Effect> {
        match self.view {
            View::Dataset => return self.on_dataset_key(code),
            View::Training => return self.on_training_key(code),
            View::Logs => self.on_logs_key(code),
            View::Pipeline => {},
        }
        Vec::new()
    }

    fn on_dataset_key(&mut self, code: KeyCode) -> Vec<Effect> {
        match code {
            KeyCode::Char('e') => return self.edit_selected(),
            KeyCode::Char('d') => {
                self.delete_selected();
                return Vec::new();
            },
            _ => {},
        }
        let view = &mut self.dataset;
        match code {
            KeyCode::Up | KeyCode::Char('k') => view.step(false),
            KeyCode::Down | KeyCode::Char('j') => view.step(true),
            KeyCode::Right | KeyCode::Char('l') | KeyCode::Enter => {
                view.tree.key_right();
            },
            KeyCode::Left | KeyCode::Char('h') => {
                view.tree.key_left();
                view.scroll = 0;
            },
            KeyCode::PageDown => view.scroll = view.scroll.saturating_add(PAGE),
            KeyCode::PageUp => view.scroll = view.scroll.saturating_sub(PAGE),
            KeyCode::Char('s') => view.stats = !view.stats,
            KeyCode::Char('/') => view.input = Some(view.filter.clone()),
            KeyCode::Esc if !view.filter.is_empty() => view.apply_filter(String::new()),
            _ => {},
        }
        Vec::new()
    }

    /// `e`: opens the selected question, answer or subtopic name in the editor.
    fn edit_selected(&mut self) -> Vec<Effect> {
        if self.locked() {
            return Vec::new();
        }
        let target = match self.selected_target() {
            Ok(target) => target,
            Err(refusal) => {
                self.say(Severity::Warn, refusal);
                return Vec::new();
            },
        };
        let text = match editor::text_of(&target) {
            Ok(text) => text,
            Err(refusal) => {
                self.say(Severity::Warn, refusal);
                return Vec::new();
            },
        };
        match editor::open(&self.project.dir.join("data"), target, &text) {
            Ok(session) => {
                let path = session.path.clone();
                self.editing = Some(session);
                vec![Effect::OpenEditor {
                    command: self.editor.clone(),
                    path,
                }]
            },
            Err(error) => {
                self.say(
                    Severity::Error,
                    format!("cannot write the edit file: {error}"),
                );
                Vec::new()
            },
        }
    }

    /// What `e` edits for the selected node.
    fn selected_target(&self) -> Result<Target, &'static str> {
        let model = self.dataset.model.as_ref().ok_or("nothing to edit")?;
        match self.dataset.tree.selected().last() {
            Some(Node::Question(id)) => model
                .question(id)
                .map(|q| Target::Question {
                    id: id.clone(),
                    text: q.text.clone(),
                })
                .ok_or("changed on disk; press R"),
            Some(Node::Answer(id)) => model
                .answer(id)
                .and_then(AnswerText::of)
                .map(|before| Target::Answer {
                    id: id.clone(),
                    before,
                })
                .ok_or("changed on disk; press R"),
            Some(Node::Subtopic(id)) => model
                .data
                .subtopics
                .iter()
                .find(|s| &s.id == id)
                .map(|s| Target::Subtopic {
                    id: id.clone(),
                    name: s.name.clone(),
                })
                .ok_or("changed on disk; press R"),
            Some(Node::Topic(_)) => Err("refused: topics live in overbrainer.toml"),
            Some(Node::MissingSubtopic(_)) | None => Err("nothing to edit here"),
        }
    }

    /// `d`: asks to delete the selected node, with what goes with it.
    fn delete_selected(&mut self) {
        if self.locked() {
            return;
        }
        let Some(model) = &self.dataset.model else {
            return;
        };
        match deletion(model, self.dataset.tree.selected(), &self.project.topics) {
            Ok(confirm) => self.overlay = Some(Overlay::Confirm(confirm)),
            Err(refusal) => self.say(Severity::Warn, refusal),
        }
    }

    /// A key while the filter is typed: Enter applies it, Esc clears it.
    fn on_filter_key(&mut self, code: KeyCode) {
        let view = &mut self.dataset;
        match code {
            KeyCode::Char(c) => {
                if let Some(input) = &mut view.input {
                    input.push(c);
                }
            },
            KeyCode::Backspace => {
                if let Some(input) = &mut view.input {
                    input.pop();
                }
            },
            KeyCode::Enter => {
                let filter = view.input.take().unwrap_or_default();
                view.apply_filter(filter);
            },
            KeyCode::Esc => {
                view.input = None;
                view.apply_filter(String::new());
            },
            _ => {},
        }
    }

    fn on_logs_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Up | KeyCode::Char('k') => self.scroll_logs(true, 1),
            KeyCode::Down | KeyCode::Char('j') => self.scroll_logs(false, 1),
            KeyCode::PageUp => self.scroll_logs(true, usize::from(PAGE)),
            KeyCode::PageDown => self.scroll_logs(false, usize::from(PAGE)),
            KeyCode::End | KeyCode::Char('G') => self.log_view.anchor = None,
            KeyCode::Char('f') => {
                self.log_view.min = LogView::next_level(self.log_view.min);
                self.log_view.anchor = None;
            },
            _ => {},
        }
    }

    /// Moves the Logs view `by` lines back (older) or forward, pinning it on the
    /// line then at its bottom. Scrolling forward while following does nothing.
    fn scroll_logs(&mut self, back: bool, by: usize) {
        let view = self.log_view;
        let offset = view.offset(&self.logs);
        let matching = self.logs.window(view.min, 0, 0).matching;
        let target = if back {
            offset.saturating_add(by)
        } else {
            offset.saturating_sub(by)
        }
        .min(matching.saturating_sub(view.height.max(1)));
        if view.anchor.is_none() && target == 0 {
            return;
        }
        self.log_view.anchor = self
            .logs
            .window(view.min, 1, target)
            .lines
            .first()
            .map(|line| line.seq);
    }

    /// `q` or Ctrl-C: quits at once when nothing runs, else asks, saying what
    /// becomes of each piece of work.
    fn quit(&mut self) -> Vec<Effect> {
        if self.leaving.is_some() {
            // A run still provisioning can be abandoned rather than waited for.
            self.offer_abandon();
            if self.overlay.is_none() {
                self.say(Severity::Info, "quitting once the work running ends");
            }
            return Vec::new();
        }
        let mut text = Vec::new();
        if self.pipeline_task.is_some() {
            let command = self.pipeline.command.map_or("stage", command_name);
            let progress = self.pipeline.progress().unwrap_or_default();
            let flight: usize = STAGES
                .iter()
                .map(|stage| self.pipeline.in_flight(*stage))
                .sum();
            text.push(format!(
                "{progress}, {} in flight. It stops now; the next `{command}` resumes it. \
                 The requests in flight are lost (already paid).",
                count(flight, "request"),
            ));
        }
        if self.edit.is_some() {
            text.push("An edit is being saved: quitting waits for it.".to_string());
        }
        text.extend(self.training_quit_text());
        if text.is_empty() {
            self.exit = Some(Exit::Quit);
            return Vec::new();
        }
        self.overlay = Some(Overlay::Confirm(Confirm {
            title: " Quit overbrainer? ".to_string(),
            text,
            yes: "quit",
            no: "stay",
            action: Action::Quit,
        }));
        Vec::new()
    }

    /// Ends the TUI for `why`: stops the pipeline task, detaches the followed
    /// runs (abandons them on a signal), and waits for them, for cancels and for
    /// an edit being saved: a run still starting is detached once its job
    /// started, never before. A signal is never turned back into a plain quit.
    fn leave(&mut self, why: Exit) -> Vec<Effect> {
        if self.leaving != Some(Exit::Signal) {
            self.leaving = Some(why);
        }
        let mut effects: Vec<Effect> = self.pipeline_task.map(Effect::Cancel).into_iter().collect();
        effects.extend(match self.leaving {
            Some(Exit::Signal) => self.abandon_all(),
            _ => self.detach_all(),
        });
        self.leave_when_idle();
        effects
    }

    /// What the TUI waits for once its loop ended, one line each: an edit being
    /// saved, the stage stopping, and each training task.
    pub(super) fn waiting_for(&self) -> Vec<String> {
        let mut lines = Vec::new();
        if self.edit.is_some() {
            lines.push("waiting for an edit to be saved...".to_string());
        }
        if self.pipeline_task.is_some() {
            let name = self.pipeline.command.map_or("stage", command_name);
            lines.push(format!("waiting for {name} to stop..."));
        }
        lines.extend(self.training.tasks.values().map(|follow| {
            let run = &follow.run_id;
            match follow.job {
                super::training::Job::Cancel => format!("waiting for the cancel of run {run}..."),
                super::training::Job::Start { .. } if follow.starting() => {
                    if follow.detach == super::training::Detach::Done {
                        format!("waiting for {} to be abandoned...", follow.run())
                    } else {
                        format!(
                            "waiting for {} to start; Ctrl-C abandons it...",
                            follow.run()
                        )
                    }
                },
                super::training::Job::Start { .. } | super::training::Job::Attach => {
                    format!("waiting for run {run} to detach...")
                },
            }
        }));
        lines
    }

    /// `n` or Esc while quitting: stays. A stage stopped and a run detached
    /// still end; a run waiting for its job to detach keeps being followed.
    /// The exit notes stay: a run detached is no longer followed, a stage
    /// stopped stays stopped, until taken up again.
    fn stay(&mut self) {
        self.leaving = None;
        let mut still = Vec::new();
        if self.pipeline_task.is_some() {
            let name = self.pipeline.command.map_or("stage", command_name);
            still.push(format!("{name} still stops"));
        }
        let (detached, abandoned) = self.keep_following();
        if detached > 0 {
            still.push(format!("{} still detached", count(detached, "run")));
        }
        if abandoned > 0 {
            still.push(format!("{} still abandoned", count(abandoned, "run")));
        }
        let said = if still.is_empty() {
            "not quitting".to_string()
        } else {
            format!("not quitting; {}", still.join(", "))
        };
        self.say(Severity::Info, said);
    }

    /// The loop ended with work left (a terminal error, a panic while drawing,
    /// its input gone): quits as a confirmed quit does, without asking. The
    /// stage stops, followed runs detach, a run still starting detaches once
    /// its job started and is never abandoned, and cancels and an edit being
    /// saved are waited for. Only a signal abandons (design 3.6, 6.7).
    pub(super) fn on_loop_end(&mut self) -> Vec<Effect> {
        self.close_overlay();
        self.dataset.input = None;
        self.leave(Exit::Quit)
    }

    /// SIGINT, SIGTERM or SIGHUP from outside: quits without asking, as Ctrl-C
    /// does on the command line: the stage stops, the training tasks are
    /// abandoned, an edit being saved and a cancel are waited for (they are never
    /// cut, design 3.6).
    pub(super) fn on_signal(&mut self) -> Vec<Effect> {
        self.close_overlay();
        let effects = self.leave(Exit::Signal);
        if self.exit.is_none() {
            let mut waited = Vec::new();
            if self.edit.is_some() {
                waited.push("the edit is saved".to_string());
            }
            if self.pipeline_task.is_some() {
                let name = self.pipeline.command.map_or("stage", command_name);
                waited.push(format!("{name} stops"));
            }
            if !self.training.tasks.is_empty() {
                waited.push("the training tasks end".to_string());
            }
            self.say(
                Severity::Warn,
                format!("interrupted: exiting once {}", waited.join(" and ")),
            );
        }
        effects
    }

    /// Moves the clock to `now`: expires the status message and shows new log
    /// lines, and the newest warning or error on the status line; reads `runs/`
    /// again when the Training view is shown and it is time.
    pub(super) fn on_tick(&mut self, now: SystemTime) -> Vec<Effect> {
        self.now = now;
        let effects = self.refresh_when_due();
        if self.status.as_ref().is_some_and(|status| {
            now.duration_since(status.at)
                .is_ok_and(|shown| shown >= STATUS_FOR)
        }) {
            self.status = None;
            self.dirty = true;
        }
        let seq = self.logs.seq();
        if seq == self.seen_log {
            return effects;
        }
        if self.view == View::Logs {
            self.dirty = true;
        }
        if let Some(line) = self.logs.latest(Level::WARN)
            && line.seq > self.seen_log
        {
            let severity = if line.level == Level::ERROR {
                Severity::Error
            } else {
                Severity::Warn
            };
            self.say(severity, line.message);
        }
        self.seen_log = seq;
        effects
    }
}

/// The confirmation of `d` on the node at `path`, with what it removes.
fn deletion(model: &Model, path: &[Node], topics: &[TopicInfo]) -> Result<Confirm, String> {
    let data = &model.data;
    let deletion = match path.last() {
        Some(Node::Subtopic(id)) => Deletion::Subtopic(id.clone()),
        Some(Node::Question(id)) => Deletion::Question(id.clone()),
        Some(Node::Answer(id)) => Deletion::Answer(id.clone()),
        Some(Node::MissingSubtopic(topic)) => Deletion::MissingSubtopic(topic.clone()),
        Some(Node::Topic(_)) => {
            return Err("refused: topics live in overbrainer.toml".to_string());
        },
        None => return Err("nothing to delete".to_string()),
    };
    let counts = data
        .counts(&deletion)
        .ok_or_else(|| "changed on disk; press R".to_string())?;
    let (title, question) = match &deletion {
        Deletion::Subtopic(id) => (
            " Delete a subtopic? ",
            subtopic_text(data, id, counts, topics),
        ),
        Deletion::Question(id) => (
            " Delete a question? ",
            question_text(data, id, counts, topics),
        ),
        Deletion::Answer(id) => (" Delete an answer? ", answer_text(data, id, topics)),
        Deletion::MissingSubtopic(_) => (
            " Delete questions? ",
            format!(
                "Delete {} whose subtopic no longer exists, and their {}? {REBUILT}",
                count(counts.questions, "question"),
                count(counts.answers, "answer"),
            ),
        ),
    };
    Ok(Confirm {
        title: title.to_string(),
        text: vec![question],
        yes: "delete",
        no: "cancel",
        action: Action::Delete { deletion, counts },
    })
}

/// The end of every deletion's text.
const REBUILT: &str = "train and eval are rebuilt.";

/// What the dialog says of a topic that is not in `overbrainer.toml`: no stage
/// runs on it, so nothing comes back.
fn not_configured(topic: &str) -> String {
    format!("its topic \"{topic}\" is not in overbrainer.toml, so no run replaces it")
}

/// The dialog text for deleting subtopic `id`, which removes `counts`.
fn subtopic_text(data: &Dataset, id: &Id, counts: Counts, topics: &[TopicInfo]) -> String {
    let subtopic = data.subtopics.iter().find(|s| &s.id == id);
    let name = subtopic.map_or("", |s| s.name.as_str());
    let topic = subtopic.map_or("", |s| s.topic.as_str());
    let then = topics.iter().find(|t| t.name == topic).map_or_else(
        || format!("; {}", not_configured(topic)),
        |topic| {
            format!(
                " so the subtopics stage does not generate it again; the next subtopics run \
                 generates a replacement to reach {}",
                topic.subtopics
            )
        },
    );
    format!(
        "Delete subtopic \"{name}\" with its {} and {}? It is recorded in \
         data/rejected.jsonl{then}. {REBUILT}",
        count(counts.questions, "question"),
        count(counts.answers, "answer"),
    )
}

/// The dialog text for deleting question `id`, which removes `counts`.
fn question_text(data: &Dataset, id: &Id, counts: Counts, topics: &[TopicInfo]) -> String {
    let question = data.questions.iter().find(|q| &q.id == id);
    let subtopic = question.map_or("", |q| q.subtopic.as_str());
    let topic = question.map_or("", |q| q.topic.as_str());
    let then = topics.iter().find(|t| t.name == topic).map_or_else(
        || not_configured(topic),
        |topic| {
            format!(
                "the next questions run tops \"{subtopic}\" up to {} without it",
                topic.questions_per_subtopic
            )
        },
    );
    let answer = if counts.answers > 0 {
        " and its answer"
    } else {
        ""
    };
    format!(
        "Delete this question{answer}? It is recorded in data/rejected.jsonl; {then}. {REBUILT}"
    )
}

/// The dialog text for deleting the answer of question `id`.
fn answer_text(data: &Dataset, id: &Id, topics: &[TopicInfo]) -> String {
    let topic = data
        .answers
        .iter()
        .find(|a| &a.id == id)
        .map_or("", |a| a.topic.as_str());
    let then = if topics.iter().any(|t| t.name == topic) {
        "the next answers run asks the parent again (a paid request)".to_string()
    } else {
        not_configured(topic)
    };
    format!("Delete this answer? The question stays; {then}. {REBUILT}")
}

/// `count` `noun`s, plural unless 1.
fn count(count: usize, noun: &str) -> String {
    if count == 1 {
        format!("1 {noun}")
    } else {
        format!("{count} {noun}s")
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use super::*;
    use crate::dataset::Id;
    use crate::logging::LogLine;
    use crate::tui::dataset::Node;
    use crate::tui::editor::Edited;
    use crate::tui::snapshots::{
        MOVED, NOW, app, at, ctrl_c, dataset, dataset_app, draw, key, open_to, path_to, project,
        text,
    };
    use crate::tui::tasks::TrainJob;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn press(app: &mut App, codes: &[KeyCode]) -> Vec<Effect> {
        codes
            .iter()
            .flat_map(|code| app.on_input(&key(*code)))
            .collect()
    }

    #[test]
    fn digits_and_tabs_switch_views() {
        let mut app = app();
        assert_eq!(app.view, View::Dataset);
        press(&mut app, &[KeyCode::Char('3')]);
        assert_eq!(app.view, View::Training);
        press(&mut app, &[KeyCode::Tab, KeyCode::Tab]);
        assert_eq!(app.view, View::Dataset);
        press(&mut app, &[KeyCode::BackTab]);
        assert_eq!(app.view, View::Logs);
        press(&mut app, &[KeyCode::Char('2')]);
        assert_eq!(app.view, View::Pipeline);
    }

    #[test]
    fn the_help_overlay_opens_and_closes_and_ignores_other_keys() {
        let mut app = app();
        press(&mut app, &[KeyCode::Char('?')]);
        assert_eq!(app.overlay, Some(Overlay::Help));
        press(&mut app, &[KeyCode::Char('3')]);
        assert_eq!(app.view, View::Dataset);
        press(&mut app, &[KeyCode::Esc]);
        assert_eq!(app.overlay, None);
        press(&mut app, &[KeyCode::Char('?'), KeyCode::Char('?')]);
        assert_eq!(app.overlay, None);
        press(&mut app, &[KeyCode::Char('?'), KeyCode::Char('q')]);
        assert_eq!(app.overlay, None);
        assert_eq!(app.exit, None, "q closes the help, it does not quit");
    }

    #[test]
    fn q_and_ctrl_c_quit_when_nothing_runs() {
        let mut first = app();
        press(&mut first, &[KeyCode::Char('q')]);
        assert_eq!(first.exit, Some(Exit::Quit));
        let mut second = app();
        second.on_input(&ctrl_c());
        assert_eq!(second.exit, Some(Exit::Quit));
    }

    #[test]
    fn a_signal_quits_and_is_remembered() {
        let mut app = app();
        app.on_signal();
        assert_eq!(app.exit, Some(Exit::Signal));
    }

    fn log(app: &App, level: Level, message: &str) {
        app.logs.push(LogLine {
            seq: 0,
            level,
            target: "overbrainer".into(),
            time: at(NOW),
            message: message.into(),
        });
    }

    /// The Logs view at 80x24 shows 20 lines.
    const ROWS: usize = 20;

    /// The title row and the rows of lines of the Logs view drawn at 80x24:
    /// the title, a blank row, then the lines.
    fn logs_view(app: &mut App) -> Result<(String, Vec<String>), Infallible> {
        let rows = text(&draw(app, 80, 24)?);
        let title = rows.get(1).cloned().unwrap_or_default();
        Ok((title, rows.into_iter().skip(3).take(ROWS).collect()))
    }

    #[test]
    fn scrolling_back_stops_at_the_oldest_line_with_an_exact_count() -> TestResult {
        let mut app = app();
        for n in 0..30 {
            log(&app, Level::INFO, &format!("line {n}"));
        }
        press(&mut app, &[KeyCode::Char('4')]);
        logs_view(&mut app)?;
        press(&mut app, &[KeyCode::Char('k'), KeyCode::Up]);
        assert_eq!(app.log_view.offset(&app.logs), 2);
        press(
            &mut app,
            &[KeyCode::PageUp, KeyCode::PageUp, KeyCode::PageUp],
        );
        let bound = 30 - ROWS;
        assert_eq!(
            app.log_view.offset(&app.logs),
            bound,
            "the oldest line on top"
        );
        let (title, lines) = logs_view(&mut app)?;
        assert!(
            title.contains("(10 newer lines below: G follows)"),
            "{title}"
        );
        assert!(lines.first().is_some_and(|row| row.contains("line 0 ")));
        assert!(lines.last().is_some_and(|row| row.contains("line 19 ")));
        press(&mut app, &[KeyCode::Char('j'), KeyCode::PageDown]);
        assert_eq!(app.log_view.offset(&app.logs), 0);
        assert!(app.log_view.anchor.is_some(), "still pinned at the bottom");
        press(&mut app, &[KeyCode::Char('G')]);
        assert_eq!(app.log_view.anchor, None);
        Ok(())
    }

    #[test]
    fn a_scrolled_back_view_stays_put_while_lines_arrive() -> TestResult {
        let mut app = app();
        for n in 0..30 {
            log(&app, Level::INFO, &format!("line {n}"));
        }
        press(&mut app, &[KeyCode::Char('4')]);
        logs_view(&mut app)?;
        press(&mut app, &[KeyCode::Char('k'), KeyCode::Char('k')]);
        let (title, before) = logs_view(&mut app)?;
        assert!(
            title.contains("(2 newer lines below: G follows)"),
            "{title}"
        );
        for n in 30..35 {
            log(&app, Level::INFO, &format!("line {n}"));
        }
        log(&app, Level::DEBUG, "not shown at info");
        app.on_tick(at(NOW + 1));
        let (title, after) = logs_view(&mut app)?;
        assert_eq!(after, before);
        assert!(
            title.contains("(7 newer lines below: G follows)"),
            "{title}"
        );
        press(&mut app, &[KeyCode::End]);
        let (title, lines) = logs_view(&mut app)?;
        assert!(!title.contains("newer"), "{title}");
        assert!(lines.last().is_some_and(|row| row.contains("line 34 ")));
        Ok(())
    }

    #[test]
    fn a_pinned_line_dropped_from_the_buffer_clamps_to_the_oldest() -> TestResult {
        let mut app = App::new(
            Project {
                name: "rust_expert".into(),
                dir: "/nonexistent/rust_expert".into(),
                topics: Vec::new(),
                eval_ratio: 0.1,
                concurrency: 8,
                target: None,
            },
            LogBuffer::new(25),
            &Theme::new(crate::tui::theme::ColorLevel::TrueColor),
            at(NOW),
        );
        for n in 0..25 {
            log(&app, Level::INFO, &format!("line {n}"));
        }
        press(&mut app, &[KeyCode::Char('4')]);
        logs_view(&mut app)?;
        press(&mut app, &[KeyCode::PageUp]);
        assert_eq!(app.log_view.anchor, Some(20), "line 19 at the bottom");
        for n in 25..50 {
            log(&app, Level::INFO, &format!("line {n}"));
        }
        let (title, lines) = logs_view(&mut app)?;
        assert!(
            title.contains("(5 newer lines below: G follows)"),
            "{title}"
        );
        assert!(lines.first().is_some_and(|row| row.contains("line 25 ")));
        Ok(())
    }

    #[test]
    fn f_cycles_the_level_and_follows_again() {
        let mut app = app();
        for n in 0..30 {
            log(&app, Level::INFO, &format!("line {n}"));
        }
        press(&mut app, &[KeyCode::Char('4'), KeyCode::Char('k')]);
        assert!(app.log_view.anchor.is_some());
        let levels: Vec<Level> = (0..5)
            .map(|_| {
                press(&mut app, &[KeyCode::Char('f')]);
                app.log_view.min
            })
            .collect();
        assert_eq!(
            levels,
            [
                Level::DEBUG,
                Level::TRACE,
                Level::ERROR,
                Level::WARN,
                Level::INFO
            ]
        );
        assert_eq!(app.log_view.anchor, None);
    }

    #[test]
    fn a_tick_shows_the_newest_warning_then_expires_it() {
        let mut app = app();
        log(&app, Level::INFO, "plain");
        log(&app, Level::WARN, "careful");
        app.on_tick(at(NOW + 1));
        let status = app.status.clone();
        assert_eq!(status.as_ref().map(|s| s.text.as_str()), Some("careful"));
        assert_eq!(status.map(|s| s.severity), Some(Severity::Warn));
        log(&app, Level::INFO, "plain again");
        app.say(Severity::Warn, "refused");
        app.on_tick(at(NOW + 2));
        assert_eq!(
            app.status.as_ref().map(|s| s.text.as_str()),
            Some("refused")
        );
        app.on_tick(at(NOW + 10));
        assert_eq!(
            app.status.as_ref().map(|s| s.text.as_str()),
            Some("refused")
        );
        app.on_tick(at(NOW + 11));
        assert_eq!(app.status, None);
    }

    #[test]
    fn the_dataset_view_moves_expands_and_scrolls() {
        let mut app = dataset_app();
        press(
            &mut app,
            &[KeyCode::Enter, KeyCode::Char('j'), KeyCode::Char('l')],
        );
        assert_eq!(
            app.dataset.tree.selected(),
            [
                Node::Topic("ownership".into()),
                Node::Subtopic(Id::subtopic("ownership", "Borrowing"))
            ]
        );
        press(&mut app, &[KeyCode::PageDown, KeyCode::PageDown]);
        assert_eq!(app.dataset.scroll, 20);
        press(&mut app, &[KeyCode::Down]);
        assert_eq!(app.dataset.scroll, 0);
        press(&mut app, &[KeyCode::Char('s')]);
        assert!(app.dataset.stats);
        press(&mut app, &[KeyCode::Char('h')]);
        assert_eq!(app.dataset.tree.selected().len(), 2, "back to the subtopic");
        press(&mut app, &[KeyCode::Char('h'), KeyCode::Char('h')]);
        assert_eq!(
            app.dataset.tree.selected().len(),
            1,
            "closed, then its topic"
        );
    }

    #[test]
    fn the_filter_takes_every_typed_key_until_enter_or_esc() {
        let mut app = dataset_app();
        press(
            &mut app,
            &[KeyCode::Char('/'), KeyCode::Char('q'), KeyCode::Char('x')],
        );
        assert_eq!(app.exit, None, "q is typed into the filter");
        press(
            &mut app,
            &[KeyCode::Backspace, KeyCode::Char('?'), KeyCode::Enter],
        );
        assert_eq!(app.dataset.filter, "q?");
        assert_eq!(app.dataset.input, None);
        press(&mut app, &[KeyCode::Char('/'), KeyCode::Esc]);
        assert_eq!(app.dataset.filter, "");
        app.on_input(&ctrl_c());
        assert_eq!(app.exit, Some(Exit::Quit));
    }

    /// The one load in `effects`.
    fn only_load(effects: &[Effect]) -> Result<TaskId, String> {
        match effects {
            [Effect::Spawn(id, Task::Load)] => Ok(*id),
            other => Err(format!("expected one load, got {other:?}")),
        }
    }

    /// `effects` but the reads of `runs/` that `R` starts too.
    fn but_runs(effects: Vec<Effect>) -> Vec<Effect> {
        effects
            .into_iter()
            .filter(|effect| !matches!(effect, Effect::Spawn(_, Task::Runs)))
            .collect()
    }

    #[test]
    fn a_failed_load_task_is_shown_in_the_view_and_frees_the_load() -> TestResult {
        let mut app = app();
        let id = only_load(&app.start())?;
        assert_eq!(app.work(), ["loading"]);
        let error = "a background task failed: task 1 panicked";
        assert_eq!(app.on_done(id, Err(error.into())), []);
        assert_eq!(app.load, None);
        assert_eq!(app.dataset.error.as_deref(), Some(error));
        assert_eq!(
            app.status.as_ref().map(|s| s.severity),
            Some(Severity::Error)
        );
        only_load(&but_runs(press(&mut app, &[KeyCode::Char('R')])))?;
        Ok(())
    }

    #[test]
    fn a_reload_asked_for_during_a_load_starts_once_it_ends() -> TestResult {
        let mut app = app();
        let first = only_load(&app.start())?;
        assert_eq!(
            but_runs(press(&mut app, &[KeyCode::Char('R'), KeyCode::Char('R')])),
            []
        );
        let second = only_load(&app.on_done(first, Ok(Done::Loaded(Ok(dataset())))))?;
        assert_ne!(second, first);
        assert_eq!(app.load, Some(second));
        assert!(app.dataset.model.is_some(), "the first load is shown");
        assert_eq!(app.on_done(second, Ok(Done::Loaded(Ok(dataset())))), []);
        assert_eq!(app.load, None);
        Ok(())
    }

    #[test]
    fn a_load_that_is_not_the_current_one_is_ignored() -> TestResult {
        let mut app = app();
        let id = only_load(&app.start())?;
        let stale = TaskId(id.0 + 100);
        assert_eq!(app.on_done(stale, Ok(Done::Loaded(Ok(dataset())))), []);
        assert!(app.dataset.model.is_none());
        assert_eq!(app.load, Some(id));
        app.on_done(
            id,
            Ok(Done::Loaded(Err("data/answers.jsonl:1: bad".into()))),
        );
        assert_eq!(
            app.dataset.error.as_deref(),
            Some("data/answers.jsonl:1: bad")
        );
        assert_eq!(app.load, None);
        Ok(())
    }

    #[test]
    fn esc_clears_an_applied_filter() {
        let mut app = dataset_app();
        press(
            &mut app,
            &[
                KeyCode::Char('/'),
                KeyCode::Char('n'),
                KeyCode::Char('l'),
                KeyCode::Char('l'),
                KeyCode::Enter,
            ],
        );
        assert_eq!(app.dataset.filter, "nll");
        assert_eq!(app.dataset.model.as_ref().and_then(|m| m.matches), Some(1));
        press(&mut app, &[KeyCode::Esc]);
        assert_eq!(app.dataset.filter, "");
        assert_eq!(app.dataset.model.as_ref().and_then(|m| m.matches), None);
    }

    #[test]
    fn the_filter_ignores_ctrl_and_alt_keys_but_takes_shifted_ones() {
        let mut app = dataset_app();
        let with = |code, modifiers| Event::Key(KeyEvent::new(KeyCode::Char(code), modifiers));
        press(&mut app, &[KeyCode::Char('/')]);
        app.on_input(&with('x', KeyModifiers::CONTROL));
        app.on_input(&with('y', KeyModifiers::ALT));
        app.on_input(&with('N', KeyModifiers::SHIFT));
        press(&mut app, &[KeyCode::Char('l'), KeyCode::Enter]);
        assert_eq!(app.dataset.filter, "Nl");
        assert_eq!(app.exit, None);
    }

    #[test]
    fn the_detail_scroll_is_clamped_to_the_text() -> TestResult {
        let mut app = dataset_app();
        open_to(&mut app, &path_to(MOVED, true));
        app.dataset.scroll = 1000;
        let rows = text(&draw(&mut app, 120, 40)?);
        assert_eq!(app.dataset.scroll, 0, "14 lines fit in 38 rows");
        assert!(rows.iter().any(|row| row.contains("model deepseek-r1")));
        let rows = text(&draw(&mut app, 80, 24)?);
        assert_eq!(app.dataset.scroll, 0, "18 lines fit in 20 rows");
        assert!(rows.iter().any(|row| row.contains("model deepseek-r1")));
        Ok(())
    }

    #[test]
    fn the_stats_pane_shows_all_topics_when_no_topic_is_selected() -> TestResult {
        let mut app = dataset_app();
        app.dataset.stats = true;
        app.dataset.tree.select(Vec::new());
        let rows = text(&draw(&mut app, 80, 24)?);
        let title = rows.get(1).cloned().unwrap_or_default();
        assert!(title.contains("stats: all topics"), "{title}");
        assert!(
            rows.iter().any(|row| row.contains("answers          5 ")),
            "{rows:#?}"
        );
        Ok(())
    }

    /// [`dataset_app`] on a copy of the fixture's files in a temp directory.
    fn project_app() -> Result<(tempfile::TempDir, App), Box<dyn std::error::Error>> {
        let dir = project()?;
        let mut app = dataset_app();
        app.project.dir = dir.path().to_path_buf();
        Ok((dir, app))
    }

    fn exited(code: i32) -> ExitStatus {
        std::os::unix::process::ExitStatusExt::from_raw(code << 8)
    }

    fn status(app: &App) -> Option<&str> {
        app.status.as_ref().map(|status| status.text.as_str())
    }

    #[test]
    fn e_opens_the_selected_question_then_saves_what_was_typed()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = project_app()?;
        app.editor = vec!["my-editor".into(), "--wait".into()];
        open_to(&mut app, &path_to(MOVED, false));
        let effects = press(&mut app, &[KeyCode::Char('e')]);
        let [Effect::OpenEditor { command, path }] = effects.as_slice() else {
            return Err(format!("{effects:?}").into());
        };
        assert_eq!(command, &["my-editor", "--wait"]);
        assert_eq!(std::fs::read_to_string(path)?, format!("{MOVED}\n"));
        assert_eq!(app.lock().as_deref(), Some("an edit is being saved"));
        std::fs::write(path, "What happens to a borrow after a move?\n")?;
        let effects = app.on_editor_exit(Ok(exited(0)));
        let [Effect::Spawn(id, Task::Edit(Edit::Change(Edited::Question { text, .. })))] =
            effects.as_slice()
        else {
            return Err(format!("{effects:?}").into());
        };
        assert_eq!(text, "What happens to a borrow after a move?");
        let saved = Saved {
            message: "question saved".into(),
            split: Ok(crate::pipeline::SplitReport::default()),
        };
        let reload = app.on_done(*id, Ok(Done::Saved(Ok(saved))));
        assert!(matches!(reload.as_slice(), [Effect::Spawn(_, Task::Load)]));
        assert_eq!(status(&app), Some("question saved"));
        assert!(!path.exists(), "the temp file is removed once saved");
        assert_eq!(app.lock(), None);
        Ok(())
    }

    #[test]
    fn an_editor_that_fails_or_changes_nothing_saves_nothing()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = project_app()?;
        open_to(&mut app, &path_to(MOVED, false)[..2]);
        for (exit, message) in [
            (
                Ok(exited(1)),
                "the editor exited with status 1; nothing changed",
            ),
            (Ok(exited(0)), "unchanged"),
            (
                Err(io::Error::from(io::ErrorKind::NotFound)),
                "cannot run the editor: entity not found; nothing changed",
            ),
        ] {
            let effects = press(&mut app, &[KeyCode::Char('e')]);
            let [Effect::OpenEditor { path, .. }] = effects.as_slice() else {
                return Err(format!("{effects:?}").into());
            };
            assert_eq!(app.on_editor_exit(exit), []);
            assert_eq!(status(&app), Some(message));
            assert!(!path.exists());
            assert_eq!(app.lock(), None);
        }
        Ok(())
    }

    #[test]
    fn a_refused_save_keeps_the_typed_text_and_says_where() -> Result<(), Box<dyn std::error::Error>>
    {
        let (_dir, mut app) = project_app()?;
        open_to(&mut app, &path_to(MOVED, false));
        let effects = press(&mut app, &[KeyCode::Char('e')]);
        let [Effect::OpenEditor { path, .. }] = effects.as_slice() else {
            return Err(format!("{effects:?}").into());
        };
        std::fs::write(path, "When does NLL end a borrow?")?;
        let effects = app.on_editor_exit(Ok(exited(0)));
        let [Effect::Spawn(id, _)] = effects.as_slice() else {
            return Err(format!("{effects:?}").into());
        };
        let refused = "another question of this subtopic already has this text";
        app.on_done(*id, Ok(Done::Saved(Err(refused.into()))));
        assert!(path.exists());
        let kept = format!(
            "refused: {refused}; your text is kept in {}",
            path.display()
        );
        assert_eq!(status(&app), Some(kept.as_str()));
        assert_eq!(app.exit_notes.len(), 1);
        Ok(())
    }

    #[test]
    fn e_and_d_are_refused_on_a_topic_and_while_an_edit_is_saved() {
        let mut app = dataset_app();
        press(&mut app, &[KeyCode::Char('e')]);
        assert_eq!(
            status(&app),
            Some("refused: topics live in overbrainer.toml")
        );
        press(&mut app, &[KeyCode::Char('d')]);
        assert_eq!(app.overlay, None);
        open_to(&mut app, &path_to(MOVED, true));
        app.edit = Some(TaskId(9));
        for code in ['e', 'd'] {
            assert_eq!(press(&mut app, &[KeyCode::Char(code)]), []);
            assert_eq!(
                status(&app),
                Some("refused: an edit is being saved; edits resume when it ends")
            );
        }
        press(&mut app, &[KeyCode::Char('q')]);
        assert!(matches!(app.overlay, Some(Overlay::Confirm(_))));
        press(&mut app, &[KeyCode::Char('y')]);
        assert_eq!(app.exit, None, "quitting waits for the edit");
        app.on_done(TaskId(9), Err("a background task failed: panicked".into()));
        assert_eq!(app.exit, Some(Exit::Quit));
    }

    #[test]
    fn d_asks_then_deletes_the_selected_answer_only_on_y() {
        let mut app = dataset_app();
        open_to(&mut app, &path_to(MOVED, true));
        press(&mut app, &[KeyCode::Char('d')]);
        let Some(Overlay::Confirm(confirm)) = app.overlay.clone() else {
            return assert_eq!(app.overlay, None);
        };
        assert_eq!(
            confirm.text,
            [
                "Delete this answer? The question stays; the next answers run asks the parent \
              again (a paid request). train and eval are rebuilt."
            ]
        );
        assert_eq!(press(&mut app, &[KeyCode::Char('n')]), []);
        assert_eq!(app.overlay, None);
        press(&mut app, &[KeyCode::Char('d')]);
        let effects = press(&mut app, &[KeyCode::Char('y')]);
        let id = Id::question(&Id::subtopic("ownership", "Borrowing"), MOVED);
        assert!(matches!(
            effects.as_slice(),
            [Effect::Spawn(_, Task::Edit(Edit::Delete { deletion: Deletion::Answer(answer), counts }))]
                if *answer == id && *counts == Counts { questions: 0, answers: 1 }
        ));
    }

    /// [`dataset_app`] on an answer, following a run, and leaving: after a
    /// signal, or with a quit waiting for that run.
    fn leaving_app(signal: bool) -> App {
        let mut app = dataset_app();
        open_to(&mut app, &path_to(MOVED, true));
        app.training.tasks.insert(
            TaskId(9),
            crate::tui::training::Follow::new(
                crate::tui::training::Job::Attach,
                "20260921-133200-a1b2",
            ),
        );
        if signal {
            app.on_signal();
        } else {
            press(&mut app, &[KeyCode::Char('q'), KeyCode::Char('y')]);
        }
        app
    }

    /// Once the TUI is leaving, after a signal or while a quit waits for a
    /// followed run, no stage, edit or deletion starts, not even one confirmed
    /// in a dialog.
    #[test]
    fn no_new_work_starts_while_leaving() -> Result<(), String> {
        for signal in [true, false] {
            let mut app = leaving_app(signal);
            assert!(app.leaving.is_some(), "{signal}");
            assert_eq!(app.lock(), None, "nothing locks the data");
            let effects = press(&mut app, &[KeyCode::Char('r'), KeyCode::Enter]);
            assert!(
                !effects.iter().any(|e| matches!(e, Effect::Spawn(..))),
                "{effects:?}"
            );
            assert_eq!(app.overlay, None);
            assert_eq!(status(&app), Some("refused: quitting; no stage starts"));
            app.status = None;
            app.overlay = Some(Overlay::Menu(0));
            assert_eq!(press(&mut app, &[KeyCode::Enter]), []);
            assert_eq!(status(&app), Some("refused: quitting; no stage starts"));
            no_edit_starts(&mut app)?;
        }
        Ok(())
    }

    /// `e`, `d` and a confirmed deletion start nothing in `app`, and say why.
    fn no_edit_starts(app: &mut App) -> Result<(), String> {
        for code in ['e', 'd'] {
            app.status = None;
            assert_eq!(press(app, &[KeyCode::Char(code)]), []);
            assert_eq!(app.overlay, None);
            assert_eq!(status(app), Some("refused: quitting; no edit starts"));
        }
        let model = app.dataset.model.as_ref().ok_or("no model")?;
        let confirm = deletion(model, app.dataset.tree.selected(), &app.project.topics)?;
        app.overlay = Some(Overlay::Confirm(confirm));
        app.status = None;
        assert_eq!(press(app, &[KeyCode::Char('y')]), []);
        assert_eq!(status(app), Some("refused: quitting; no edit starts"));
        Ok(())
    }

    #[test]
    fn an_edit_open_when_the_tui_ends_is_noted_for_the_exit()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = project_app()?;
        open_to(&mut app, &path_to(MOVED, false));
        let effects = press(&mut app, &[KeyCode::Char('e')]);
        let [Effect::OpenEditor { path, .. }] = effects.as_slice() else {
            return Err(format!("{effects:?}").into());
        };
        app.abandon_edit();
        assert!(path.exists(), "it may hold typed text");
        assert_eq!(
            app.exit_notes,
            [format!(
                "an edit was not saved; its text is kept in {}",
                path.display()
            )]
        );
        app.abandon_edit();
        assert_eq!(app.exit_notes.len(), 1);
        Ok(())
    }

    #[test]
    fn a_signal_during_a_save_waits_for_it_then_says_it_was_saved()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = project_app()?;
        open_to(&mut app, &path_to(MOVED, false));
        let effects = press(&mut app, &[KeyCode::Char('e')]);
        let [Effect::OpenEditor { path, .. }] = effects.as_slice() else {
            return Err(format!("{effects:?}").into());
        };
        std::fs::write(path, "What happens to a borrow after a move?\n")?;
        let effects = app.on_editor_exit(Ok(exited(0)));
        let [Effect::Spawn(id, _)] = effects.as_slice() else {
            return Err(format!("{effects:?}").into());
        };
        app.on_signal();
        assert_eq!(app.exit, None, "the save is never cut");
        let saved = Saved {
            message: "question saved".into(),
            split: Err("cannot write data/train.jsonl".into()),
        };
        assert_eq!(app.on_done(*id, Ok(Done::Saved(Ok(saved)))), []);
        assert_eq!(app.exit, Some(Exit::Signal));
        assert!(!path.exists());
        let said = "question saved, but split failed: cannot write data/train.jsonl; run split";
        assert_eq!(app.exit_notes, [format!("a change was saved: {said}")]);
        assert_eq!(status(&app), Some(said));
        assert_eq!(
            app.status.as_ref().map(|s| s.severity),
            Some(Severity::Warn)
        );
        app.abandon_edit();
        assert_eq!(app.exit_notes.len(), 1, "no stale note");
        Ok(())
    }

    #[test]
    fn a_signal_during_a_refused_save_notes_the_kept_file() -> Result<(), Box<dyn std::error::Error>>
    {
        let (_dir, mut app) = project_app()?;
        open_to(&mut app, &path_to(MOVED, false));
        let effects = press(&mut app, &[KeyCode::Char('e')]);
        let [Effect::OpenEditor { path, .. }] = effects.as_slice() else {
            return Err(format!("{effects:?}").into());
        };
        std::fs::write(path, "When does NLL end a borrow?")?;
        let effects = app.on_editor_exit(Ok(exited(0)));
        let [Effect::Spawn(id, _)] = effects.as_slice() else {
            return Err(format!("{effects:?}").into());
        };
        app.on_signal();
        let refused = "another question of this subtopic already has this text";
        app.on_done(*id, Ok(Done::Saved(Err(refused.into()))));
        assert_eq!(app.exit, Some(Exit::Signal));
        assert!(path.exists());
        assert_eq!(
            app.exit_notes,
            [format!(
                "an edit was refused ({refused}); its text is kept in {}",
                path.display()
            )]
        );
        Ok(())
    }

    #[test]
    fn an_edited_file_that_cannot_be_read_is_noted_for_the_exit()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = project_app()?;
        open_to(&mut app, &path_to(MOVED, false));
        let effects = press(&mut app, &[KeyCode::Char('e')]);
        let [Effect::OpenEditor { path, .. }] = effects.as_slice() else {
            return Err(format!("{effects:?}").into());
        };
        std::fs::remove_file(path)?;
        std::fs::create_dir(path)?;
        assert_eq!(app.on_editor_exit(Ok(exited(0))), []);
        assert_eq!(app.lock(), None);
        assert_eq!(app.exit_notes.len(), 1);
        assert!(
            app.exit_notes[0].contains("cannot read the edited file")
                && app.exit_notes[0].contains(&path.display().to_string()),
            "{:?}",
            app.exit_notes
        );
        Ok(())
    }

    /// The text of the dialog `d` opens on the node at `path`.
    fn dialog_text(app: &mut App, path: &[Node]) -> Result<String, String> {
        open_to(app, path);
        press(app, &[KeyCode::Char('d')]);
        match app.overlay.take() {
            Some(Overlay::Confirm(confirm)) => Ok(confirm.text.concat()),
            other => Err(format!("{other:?}")),
        }
    }

    #[test]
    fn a_deletion_in_a_topic_not_configured_promises_no_replacement() -> Result<(), String> {
        let mut app = dataset_app();
        let legacy = Id::subtopic("old_topic", "Legacy");
        let question = Id::question(&legacy, "Is this question still used?");
        let topic = Node::Topic("old_topic".into());
        let not_configured = "its topic \"old_topic\" is not in overbrainer.toml, so no run \
                              replaces it. train and eval are rebuilt.";
        let subtopic = dialog_text(&mut app, &[topic.clone(), Node::Subtopic(legacy.clone())])?;
        assert_eq!(
            subtopic,
            format!(
                "Delete subtopic \"Legacy\" with its 1 question and 1 answer? It is recorded in \
                 data/rejected.jsonl; {not_configured}"
            )
        );
        let path = [
            topic,
            Node::Subtopic(legacy),
            Node::Question(question.clone()),
        ];
        let text = dialog_text(&mut app, &path)?;
        assert_eq!(
            text,
            format!(
                "Delete this question and its answer? It is recorded in data/rejected.jsonl; \
                 {not_configured}"
            )
        );
        let mut answer = path.to_vec();
        answer.push(Node::Answer(question));
        let text = dialog_text(&mut app, &answer)?;
        assert_eq!(
            text,
            format!("Delete this answer? The question stays; {not_configured}")
        );
        for text in [subtopic, text] {
            assert!(!text.contains("next"), "{text}");
        }
        Ok(())
    }

    #[test]
    fn r_runs_the_chosen_command_one_task_at_a_time() {
        let mut app = dataset_app();
        press(
            &mut app,
            &[KeyCode::Char('r'), KeyCode::Down, KeyCode::Down],
        );
        assert_eq!(app.overlay, Some(Overlay::Menu(2)));
        let effects = press(&mut app, &[KeyCode::Enter]);
        let [Effect::Spawn(id, Task::Pipeline(Command::Answers))] = effects.as_slice() else {
            return assert_eq!(effects, []);
        };
        assert_eq!(app.view, View::Pipeline);
        assert_eq!(app.pipeline.concurrency, 8);
        app.on_message(Msg::Event(
            *id,
            crate::events::Event::StageStarted {
                stage: crate::events::Stage::Answers,
                total: 3,
            },
        ));
        app.on_message(Msg::Event(
            TaskId(99),
            crate::events::Event::StageStarted {
                stage: crate::events::Stage::Split,
                total: 3,
            },
        ));
        assert_eq!(
            app.pipeline.row(crate::events::Stage::Split).state,
            super::super::pipeline::StageState::Idle,
            "another task's events are ignored"
        );
        app.on_message(Msg::Lagged(*id, 5));
        app.on_message(Msg::Report(*id, Report::Line("answers: 3 done".into())));
        assert_eq!((app.pipeline.skipped, app.pipeline.results.len()), (5, 1));
        assert_eq!(press(&mut app, &[KeyCode::Char('r')]), []);
        assert_eq!(
            status(&app),
            Some("refused: answers is running; one task at a time")
        );
        app.view = View::Dataset;
        press(&mut app, &[KeyCode::Char('e')]);
        assert_eq!(
            status(&app),
            Some("refused: answers is running; edits resume when it ends")
        );
        let reload = app.on_done(*id, Ok(Done::Pipeline(Ok(()))));
        assert!(matches!(reload.as_slice(), [Effect::Spawn(_, Task::Load)]));
        assert_eq!(status(&app), Some("answers finished"));
        assert_eq!(app.lock(), None);
    }

    #[test]
    fn quitting_asks_then_stops_the_stage_and_waits_for_it() {
        let mut app = app();
        crate::tui::snapshots::pipeline_running(&mut app);
        app.on_input(&ctrl_c());
        let Some(Overlay::Confirm(confirm)) = app.overlay.clone() else {
            return assert_eq!(app.overlay, None);
        };
        assert_eq!(
            confirm.text,
            [
                "answers 120/400, 8 requests in flight. It stops now; the next `run` resumes \
                 it. The requests in flight are lost (already paid)."
            ]
        );
        assert_eq!(press(&mut app, &[KeyCode::Char('n')]), []);
        assert_eq!(app.leaving, None);
        press(&mut app, &[KeyCode::Char('q')]);
        let effects = press(&mut app, &[KeyCode::Char('y')]);
        assert_eq!(effects, [Effect::Cancel(TaskId(7))]);
        assert_eq!(app.exit, None);
        assert_eq!(
            app.work().first().map(String::as_str),
            Some("quitting: waiting")
        );
        app.on_done(TaskId(7), Ok(Done::Pipeline(Err("interrupted".into()))));
        assert_eq!(app.exit, Some(Exit::Quit));
        assert_eq!(app.exit_notes, ["run: interrupted"]);
    }

    #[test]
    fn a_signal_stops_the_stage_without_asking_and_exits_once_it_ended() {
        let mut app = app();
        crate::tui::snapshots::pipeline_running(&mut app);
        assert_eq!(app.on_signal(), [Effect::Cancel(TaskId(7))]);
        assert_eq!(app.overlay, None);
        assert_eq!(app.exit, None);
        app.on_done(TaskId(7), Err("a background task failed".into()));
        assert_eq!(app.exit, Some(Exit::Signal));
    }

    #[test]
    fn a_signal_while_quitting_stays_a_signal_and_n_cannot_undo_it() {
        let mut app = app();
        crate::tui::snapshots::pipeline_running(&mut app);
        press(&mut app, &[KeyCode::Char('q'), KeyCode::Char('y')]);
        assert_eq!(app.leaving, Some(Exit::Quit));
        press(&mut app, &[KeyCode::Char('q')]);
        assert_eq!(app.overlay, None, "no second dialog while quitting");
        assert_eq!(app.on_signal(), [Effect::Cancel(TaskId(7))]);
        assert_eq!(app.leaving, Some(Exit::Signal));
        press(&mut app, &[KeyCode::Char('n'), KeyCode::Esc]);
        assert_eq!(app.leaving, Some(Exit::Signal));
        app.on_done(TaskId(7), Ok(Done::Pipeline(Err("interrupted".into()))));
        assert_eq!(app.exit, Some(Exit::Signal));
    }

    /// The loop ended with a stage running: it stops as on a confirmed quit,
    /// and its end is noted for the exit.
    #[test]
    fn a_stage_running_when_the_loop_ends_stops_and_is_noted() {
        let mut app = app();
        crate::tui::snapshots::pipeline_running(&mut app);
        assert_eq!(app.on_loop_end(), [Effect::Cancel(TaskId(7))]);
        assert_eq!(app.leaving, Some(Exit::Quit));
        assert_eq!(app.waiting_for(), ["waiting for run to stop..."]);
        app.on_done(TaskId(7), Ok(Done::Pipeline(Err("interrupted".into()))));
        assert_eq!(app.exit, Some(Exit::Quit));
        assert_eq!(app.exit_notes, ["run: interrupted"]);
    }

    /// The loop ended with a Runpod run still provisioning: it is never
    /// abandoned, only detached once its job started, unless a signal comes.
    #[test]
    fn a_start_is_never_abandoned_when_the_loop_ends() {
        let mut app = app();
        starting(&mut app, TaskId(6));
        assert_eq!(app.on_loop_end(), [], "never during the start");
        assert_eq!(app.overlay, None, "nothing asks");
        assert_eq!(
            app.waiting_for(),
            ["waiting for run 20260921-133200-a1b2 to start; Ctrl-C abandons it..."]
        );
        assert_eq!(watching(&mut app, TaskId(6)), [Effect::Cancel(TaskId(6))]);

        let mut app = super::super::snapshots::app();
        starting(&mut app, TaskId(6));
        app.on_loop_end();
        assert_eq!(app.on_signal(), [Effect::Abandon(TaskId(6))]);
        assert_eq!(
            app.waiting_for(),
            ["waiting for run 20260921-133200-a1b2 to be abandoned..."]
        );
    }

    #[test]
    fn staying_after_y_says_the_stage_still_stops() {
        let mut app = app();
        crate::tui::snapshots::pipeline_running(&mut app);
        let effects = press(&mut app, &[KeyCode::Char('q'), KeyCode::Char('y')]);
        assert_eq!(effects, [Effect::Cancel(TaskId(7))]);
        press(&mut app, &[KeyCode::Esc]);
        assert_eq!(app.leaving, None);
        assert_eq!(status(&app), Some("not quitting; run still stops"));
        app.on_done(
            TaskId(7),
            Ok(Done::Pipeline(Err(
                "interrupted: the stage resumes on its next run".into(),
            ))),
        );
        assert_eq!(app.exit, None);
        assert!(
            app.pipeline
                .rows
                .iter()
                .all(|row| row.state != super::super::pipeline::StageState::Running)
        );
    }

    /// Runs the reads among `effects` at once, as their tasks would, and hands
    /// their results to `app`; returns the other effects.
    fn read(app: &mut App, effects: Vec<Effect>) -> Vec<Effect> {
        use crate::tui::training::{list_runs, read_series};
        let mut left = Vec::new();
        let mut queue: std::collections::VecDeque<Effect> = effects.into();
        while let Some(effect) = queue.pop_front() {
            let more = match effect {
                Effect::Spawn(id, Task::Runs) => {
                    let listing = list_runs(&app.project.dir);
                    app.on_done(id, Ok(Done::Runs(listing)))
                },
                Effect::Spawn(id, Task::Series(run)) => {
                    let series = read_series(&app.project.dir, &run);
                    app.on_done(id, Ok(Done::Series { run, series }))
                },
                other => {
                    left.push(other);
                    continue;
                },
            };
            queue.extend(more);
        }
        left
    }

    /// [`press`], with the reads it starts run at once.
    fn keys(app: &mut App, codes: &[KeyCode]) -> Vec<Effect> {
        let effects = press(app, codes);
        read(app, effects)
    }

    /// [`App::on_done`], with the reads it starts run at once.
    fn ended(app: &mut App, id: TaskId, result: Result<Done, String>) -> Vec<Effect> {
        let effects = app.on_done(id, result);
        read(app, effects)
    }

    const FIRST: &str = "20260921-133200-a1b2";
    const SECOND: &str = "20260920-101500-9f00";

    fn runs_app() -> Result<(tempfile::TempDir, App), Box<dyn std::error::Error>> {
        let (dir, mut app) = project_app()?;
        let runs = crate::runs::Runs::new(dir.path());
        for (id, state) in [
            (FIRST, crate::runs::RunState::Running),
            (SECOND, crate::runs::RunState::Succeeded),
        ] {
            runs.save(&crate::tui::snapshots::run(id, "homelab", state))?;
        }
        std::fs::write(
            runs.run_dir(SECOND)?.join(crate::train::METRICS_FILE),
            "{\"event\": \"log\", \"time\": 2, \"step\": 1, \"loss\": 1.5}\n",
        )?;
        keys(&mut app, &[KeyCode::Char('3')]);
        Ok((dir, app))
    }

    /// `a` on the selected run: the task it spawns.
    fn attach(app: &mut App) -> Result<TaskId, String> {
        let effects = keys(app, &[KeyCode::Char('a')]);
        match effects.as_slice() {
            [Effect::Spawn(id, Task::Train(TrainJob::Attach(_)))] => Ok(*id),
            _ => Err(format!("{effects:?}")),
        }
    }

    /// The only effect of `effects`, a cancel task started.
    fn cancel_started(effects: &[Effect]) -> Result<TaskId, String> {
        match effects {
            [Effect::Spawn(id, Task::Train(TrainJob::Cancel(run)))] if run == FIRST => Ok(*id),
            _ => Err(format!("{effects:?}")),
        }
    }

    /// The first job status of task `id`: its watch began.
    fn watching(app: &mut App, id: TaskId) -> Vec<Effect> {
        app.on_message(Msg::Event(
            id,
            crate::events::Event::JobStatus(crate::exec::JobStatus::Running),
        ))
    }

    fn metric(step: u64) -> crate::train::TrainMetric {
        crate::train::TrainMetric {
            time: 1.0,
            step,
            epoch: None,
            max_steps: Some(4),
            loss: Some(2.0),
            eval_loss: None,
            learning_rate: None,
            grad_norm: None,
        }
    }

    const DETACHED: &str = "interrupted: run 20260921-133200-a1b2 keeps running on target \
                            `homelab`; follow it again with `overbrainer train attach \
                            20260921-133200-a1b2`";

    #[test]
    fn the_training_view_lists_runs_newest_first_and_reads_local_metrics()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = runs_app()?;
        let ids: Vec<&str> = app
            .training
            .runs
            .iter()
            .map(|r| r.record.id.as_str())
            .collect();
        assert_eq!(ids, [FIRST, SECOND]);
        assert!(!app.training.series.contains_key(FIRST));
        let effects = press(&mut app, &[KeyCode::Char('j')]);
        assert!(
            matches!(effects.as_slice(), [Effect::Spawn(_, Task::Series(run))] if run == SECOND),
            "read in a task: {effects:?}"
        );
        read(&mut app, effects);
        assert_eq!(app.training.series.get(SECOND).map(Vec::len), Some(1));
        let again = keys(&mut app, &[KeyCode::Char('k')]);
        assert_eq!(again, [], "its read ran");
        let again = press(&mut app, &[KeyCode::Char('j')]);
        assert_eq!(again, [], "SECOND is read once");
        Ok(())
    }

    #[test]
    fn the_runs_are_read_again_every_two_seconds_only_while_shown()
    -> Result<(), Box<dyn std::error::Error>> {
        let (dir, mut app) = runs_app()?;
        let runs = crate::runs::Runs::new(dir.path());
        let record =
            |id| crate::tui::snapshots::run(id, "homelab", crate::runs::RunState::Preparing);
        runs.save(&record("20260922-080000-beef"))?;
        let effects = app.on_tick(at(NOW + 1));
        assert_eq!(effects, [], "read at most every 2 s");
        let effects = app.on_tick(at(NOW + 2));
        read(&mut app, effects);
        assert_eq!(app.training.runs.len(), 3);
        assert_eq!(app.training.runs[0].record.id, "20260922-080000-beef");
        runs.save(&record("20260923-080000-cafe"))?;
        keys(&mut app, &[KeyCode::Char('1')]);
        assert_eq!(app.on_tick(at(NOW + 10)), [], "not read while hidden");
        Ok(())
    }

    #[test]
    fn a_listing_asked_for_while_one_runs_starts_after_it_and_stale_reads_are_ignored()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::tui::training::{Listing, list_runs};
        let (_dir, mut app) = runs_app()?;
        let first = app.refresh_runs();
        let [Effect::Spawn(listing, Task::Runs)] = first.as_slice() else {
            return Err(format!("{first:?}").into());
        };
        assert_eq!(app.refresh_runs(), [], "one read at a time");
        let again = app.on_done(*listing, Ok(Done::Runs(list_runs(&app.project.dir))));
        assert!(
            again
                .iter()
                .any(|e| matches!(e, Effect::Spawn(_, Task::Runs))),
            "{again:?}"
        );
        let stale = Listing {
            runs: Ok(Vec::new()),
            skipped: Vec::new(),
        };
        assert_eq!(app.on_done(*listing, Ok(Done::Runs(stale))), []);
        assert_eq!(app.training.runs.len(), 2, "a stale listing is ignored");
        // A read of the metrics started before an attach no longer counts.
        let reading = press(&mut app, &[KeyCode::Char('j')]);
        let [Effect::Spawn(read_id, Task::Series(_))] = reading.as_slice() else {
            return Err(format!("{reading:?}").into());
        };
        attach(&mut app)?;
        let old = Some(vec![metric(1)]);
        app.on_done(
            *read_id,
            Ok(Done::Series {
                run: SECOND.into(),
                series: old,
            }),
        );
        assert_eq!(app.training.series.get(SECOND), None);
        Ok(())
    }

    #[test]
    fn a_failed_listing_keeps_the_runs_shown_and_marks_them_stale()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::tui::training::Listing;
        let (_dir, mut app) = runs_app()?;
        let effects = app.refresh_runs();
        let [Effect::Spawn(listing, Task::Runs)] = effects.as_slice() else {
            return Err(format!("{effects:?}").into());
        };
        let failed = Listing {
            runs: Err("cannot list the runs: permission denied".into()),
            skipped: Vec::new(),
        };
        app.on_done(*listing, Ok(Done::Runs(failed)));
        assert_eq!(app.training.runs.len(), 2);
        let rows = text(&draw(&mut app, 80, 24)?);
        assert!(
            rows[1].contains(" runs (stale: cannot list the runs) "),
            "{}",
            rows[1]
        );
        Ok(())
    }

    #[test]
    fn a_skipped_record_is_warned_about_once() -> Result<(), Box<dyn std::error::Error>> {
        use crate::tui::training::list_runs;
        let (dir, mut app) = runs_app()?;
        let broken = dir.path().join("runs/20260922-080000-beef");
        std::fs::create_dir_all(&broken)?;
        std::fs::write(broken.join("run.json"), "not json")?;
        let listing = list_runs(dir.path());
        assert_eq!(listing.skipped.len(), 1, "{listing:?}");
        assert_eq!(app.training.listed(listing.clone()).len(), 1);
        assert_eq!(app.training.listed(listing), Vec::<String>::new());
        Ok(())
    }

    #[test]
    fn a_attaches_once_and_its_events_feed_the_run() -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = runs_app()?;
        let id = attach(&mut app)?;
        assert_eq!(keys(&mut app, &[KeyCode::Char('a')]), []);
        assert_eq!(
            status(&app),
            Some("run 20260921-133200-a1b2 is already followed")
        );
        app.on_message(Msg::Event(id, crate::events::Event::Metric(metric(1))));
        assert_eq!(watching(&mut app, id), []);
        app.on_message(Msg::Lagged(id, 3));
        app.on_message(Msg::Report(
            id,
            Report::Line("train: run ... succeeded".into()),
        ));
        let follow = app.training.tasks.get(&id).cloned();
        assert_eq!(
            follow
                .as_ref()
                .map(|f| (f.watching, f.skipped, f.lines.len())),
            Some((true, 3, 1))
        );
        assert_eq!(app.training.series.get(FIRST).map(Vec::len), Some(1));
        ended(&mut app, id, Ok(Done::Trained(Ok(()))));
        assert!(app.training.tasks.is_empty());
        let end = app.training.ended.get(FIRST).cloned().unwrap_or_default();
        assert_eq!((end.lines.len(), end.skipped, end.healed), (1, 3, false));
        let rows = text(&draw(&mut app, 120, 40)?).join("\n");
        assert!(
            rows.contains("3 points missing (no local metrics file to read them from)"),
            "{rows}"
        );
        // Messages handled after the end still count: no local file holds them.
        app.on_message(Msg::Report(id, Report::Line("late".into())));
        app.on_message(Msg::Event(id, crate::events::Event::Metric(metric(2))));
        assert_eq!(
            app.training.ended.get(FIRST).map(|e| e.lines.len()),
            Some(2)
        );
        assert_eq!(app.training.series.get(FIRST).map(Vec::len), Some(2));
        Ok(())
    }

    #[test]
    fn late_metrics_of_a_run_read_again_from_its_file_are_dropped()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = runs_app()?;
        keys(&mut app, &[KeyCode::Char('j')]);
        let id = attach(&mut app)?;
        assert_eq!(
            app.training.series.get(SECOND),
            None,
            "attach replays it all"
        );
        app.on_message(Msg::Lagged(id, 2));
        ended(&mut app, id, Ok(Done::Trained(Ok(()))));
        assert_eq!(app.training.series.get(SECOND).map(Vec::len), Some(1));
        assert_eq!(app.training.ended.get(SECOND).map(|e| e.healed), Some(true));
        app.on_message(Msg::Event(id, crate::events::Event::Metric(metric(1))));
        assert_eq!(app.training.series.get(SECOND).map(Vec::len), Some(1));
        let rows = text(&draw(&mut app, 120, 40)?).join("\n");
        assert!(!rows.contains("points missing"), "healed: {rows}");
        Ok(())
    }

    #[test]
    fn cancelling_a_followed_run_detaches_it_first() -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = runs_app()?;
        let follow = attach(&mut app)?;
        watching(&mut app, follow);
        keys(&mut app, &[KeyCode::Char('c')]);
        assert!(matches!(app.overlay, Some(Overlay::Confirm(_))));
        assert_eq!(
            keys(&mut app, &[KeyCode::Char('y')]),
            [Effect::Cancel(follow)]
        );
        for key in ['c', 'a'] {
            assert_eq!(keys(&mut app, &[KeyCode::Char(key)]), []);
            assert_eq!(app.overlay, None);
            assert_eq!(
                status(&app),
                Some("run 20260921-133200-a1b2 is being cancelled")
            );
        }
        let detached = "interrupted: run 20260921-133200-a1b2 keeps running on target `homelab`";
        let effects = ended(&mut app, follow, Ok(Done::Trained(Err(detached.into()))));
        cancel_started(&effects)?;
        keys(&mut app, &[KeyCode::Char('c')]);
        assert_eq!(app.overlay, None, "a cancel already runs");
        Ok(())
    }

    #[test]
    fn a_run_whose_job_has_not_started_is_detached_once_it_did()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = runs_app()?;
        let follow = attach(&mut app)?;
        keys(&mut app, &[KeyCode::Char('c')]);
        assert_eq!(keys(&mut app, &[KeyCode::Char('y')]), [], "no token cut");
        let creating = crate::events::Event::PodStatus(crate::runpod::PodStatus::Creating {
            name: "overbrainer-20260921-133200-a1b2-1".into(),
            gpu_type: "NVIDIA GeForce RTX 4090".into(),
        });
        assert_eq!(app.on_message(Msg::Event(follow, creating)), []);
        assert_eq!(watching(&mut app, follow), [Effect::Cancel(follow)]);
        assert_eq!(watching(&mut app, follow), []);
        Ok(())
    }

    #[test]
    fn quitting_detaches_followed_runs_and_waits_for_cancels()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = runs_app()?;
        let follow = attach(&mut app)?;
        watching(&mut app, follow);
        keys(&mut app, &[KeyCode::Char('j'), KeyCode::Char('c')]);
        let cancel = keys(&mut app, &[KeyCode::Char('y')]);
        let [Effect::Spawn(cancelling, _)] = cancel.as_slice() else {
            return Err(format!("{cancel:?}").into());
        };
        keys(&mut app, &[KeyCode::Char('q')]);
        assert_eq!(
            keys(&mut app, &[KeyCode::Char('y')]),
            [Effect::Cancel(follow)]
        );
        ended(&mut app, follow, Ok(Done::Trained(Err(DETACHED.into()))));
        assert_eq!(app.exit, None, "the cancel is waited for");
        ended(&mut app, *cancelling, Ok(Done::Trained(Ok(()))));
        assert_eq!(app.exit, Some(Exit::Quit));
        assert_eq!(app.exit_notes, [DETACHED]);
        Ok(())
    }

    /// The text of the dialog shown.
    fn dialog(app: &App) -> String {
        match &app.overlay {
            Some(Overlay::Confirm(confirm)) => confirm.text.join("\n"),
            _ => String::new(),
        }
    }

    #[test]
    fn a_cancel_confirmed_before_quitting_still_runs_and_is_waited_for()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = runs_app()?;
        let follow = attach(&mut app)?;
        watching(&mut app, follow);
        keys(&mut app, &[KeyCode::Char('c'), KeyCode::Char('y')]);
        keys(&mut app, &[KeyCode::Char('q')]);
        assert_eq!(
            dialog(&app),
            "Run 20260921-133200-a1b2: cancel pending, it starts once the run is detached; \
             quitting waits for it."
        );
        assert_eq!(
            keys(&mut app, &[KeyCode::Char('y')]),
            [],
            "already detaching"
        );
        let effects = ended(&mut app, follow, Ok(Done::Trained(Err(DETACHED.into()))));
        let cancel = cancel_started(&effects)?;
        assert_eq!(app.exit, None, "the cancel is waited for");
        app.on_message(Msg::Report(
            cancel,
            Report::Line("train: run 20260921-133200-a1b2 cancelled".into()),
        ));
        ended(&mut app, cancel, Ok(Done::Trained(Ok(()))));
        assert_eq!(app.exit, Some(Exit::Quit));
        assert_eq!(
            app.exit_notes,
            ["train: run 20260921-133200-a1b2 cancelled"]
        );
        Ok(())
    }

    #[test]
    fn a_cancel_confirmed_while_quitting_still_runs_and_is_waited_for()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = runs_app()?;
        let follow = attach(&mut app)?;
        watching(&mut app, follow);
        assert_eq!(
            keys(&mut app, &[KeyCode::Char('q'), KeyCode::Char('y')]),
            [Effect::Cancel(follow)]
        );
        keys(&mut app, &[KeyCode::Char('c')]);
        assert!(matches!(app.overlay, Some(Overlay::Confirm(_))));
        assert_eq!(
            keys(&mut app, &[KeyCode::Char('y')]),
            [],
            "already detached"
        );
        let effects = ended(&mut app, follow, Ok(Done::Trained(Err(DETACHED.into()))));
        let cancel = cancel_started(&effects)?;
        assert_eq!(app.exit, None);
        ended(&mut app, cancel, Ok(Done::Trained(Ok(()))));
        assert_eq!(app.exit, Some(Exit::Quit));
        Ok(())
    }

    #[test]
    fn a_signal_before_a_pending_cancel_says_how_to_run_it()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = runs_app()?;
        let follow = attach(&mut app)?;
        watching(&mut app, follow);
        keys(&mut app, &[KeyCode::Char('c'), KeyCode::Char('y')]);
        assert_eq!(app.on_signal(), [Effect::Abandon(follow)]);
        assert_eq!(
            ended(&mut app, follow, Ok(Done::Trained(Err(DETACHED.into())))),
            []
        );
        assert_eq!(app.exit, Some(Exit::Signal));
        assert_eq!(
            app.exit_notes,
            [
                DETACHED.to_string(),
                "run 20260921-133200-a1b2 was not cancelled: interrupted before its cancel \
                 started; cancel it with `overbrainer train cancel 20260921-133200-a1b2`"
                    .to_string(),
            ]
        );
        Ok(())
    }

    /// Start tasks bound to [`FIRST`] and [`SECOND`], on Runpod when `runpod`.
    fn starts(app: &mut App, runpod: bool) -> [TaskId; 2] {
        use crate::tui::training::{Follow, Job};
        for (id, run) in [(TaskId(90), FIRST), (TaskId(91), SECOND)] {
            app.training
                .tasks
                .insert(id, Follow::new(Job::Start { runpod }, run));
        }
        [TaskId(90), TaskId(91)]
    }

    /// The tasks the open dialog abandons, if it is an abandon dialog.
    fn abandoning(app: &App) -> Option<&[TaskId]> {
        match &app.overlay {
            Some(Overlay::Confirm(Confirm {
                action: Action::AbandonStart(task),
                ..
            })) => Some(std::slice::from_ref(task)),
            _ => None,
        }
    }

    #[test]
    fn c_on_a_starting_runpod_run_offers_to_abandon_it_and_n_keeps_it()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::tui::training::Detach;
        let (_dir, mut app) = runs_app()?;
        let [first, _] = starts(&mut app, true);
        for code in [KeyCode::Char('n'), KeyCode::Esc] {
            keys(&mut app, &[KeyCode::Char('c')]);
            assert!(
                dialog(&app).starts_with(
                    "Run 20260921-133200-a1b2 is still starting, so it has no job to cancel \
                     yet. Abandon it instead? If its pod is still being prepared, it is \
                     deleted"
                ),
                "{}",
                dialog(&app)
            );
            assert_eq!(abandoning(&app), Some([first].as_slice()));
            assert_eq!(keys(&mut app, &[code]), [], "{code:?}");
            assert_eq!(app.overlay, None);
            let follow = app
                .training
                .tasks
                .get(&first)
                .map(|f| (f.detach, f.cancel_after));
            assert_eq!(follow, Some((Detach::No, false)), "{code:?}");
        }
        Ok(())
    }

    #[test]
    fn c_then_y_abandons_only_the_selected_start_and_the_tui_stays()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::tui::training::Detach;
        let (_dir, mut app) = runs_app()?;
        let [first, second] = starts(&mut app, true);
        assert_eq!(
            keys(&mut app, &[KeyCode::Char('c'), KeyCode::Char('y')]),
            [Effect::Abandon(first)]
        );
        assert_eq!(status(&app), Some("abandoning run 20260921-133200-a1b2"));
        assert_eq!((app.leaving, app.exit), (None, None));
        let other = app.training.tasks.get(&second).map(|f| f.detach);
        assert_eq!(other, Some(Detach::No), "only the selected run");
        keys(&mut app, &[KeyCode::Char('c')]);
        assert_eq!(app.overlay, None);
        assert_eq!(
            status(&app),
            Some("run 20260921-133200-a1b2 is being abandoned")
        );
        let failed = "interrupted: run 20260921-133200-a1b2 stopped before its job started; it \
                      has no pod left";
        ended(&mut app, first, Ok(Done::Trained(Err(failed.into()))));
        assert_eq!((app.leaving, app.exit), (None, None));
        let error = app.training.ended.get(FIRST).and_then(|e| e.error.clone());
        assert_eq!(error.as_deref(), Some(failed));
        assert_eq!(
            status(&app).map(String::from),
            Some(format!("run {FIRST}: {failed}"))
        );
        assert!(app.training.tasks.contains_key(&second));
        Ok(())
    }

    #[test]
    fn y_after_the_job_started_does_not_abandon_and_says_so()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = runs_app()?;
        let [first, _] = starts(&mut app, true);
        keys(&mut app, &[KeyCode::Char('c')]);
        assert_eq!(abandoning(&app), Some([first].as_slice()));
        assert_eq!(watching(&mut app, first), [], "still followed");
        assert_eq!(keys(&mut app, &[KeyCode::Char('y')]), [], "no abandon");
        assert_eq!(
            status(&app),
            Some(
                "run 20260921-133200-a1b2: its job started, so it was not abandoned and its pod \
                 keeps billing; press c to cancel it"
            )
        );
        assert_eq!(
            app.status.as_ref().map(|status| status.severity),
            Some(Severity::Warn)
        );
        keys(&mut app, &[KeyCode::Char('c')]);
        assert!(
            matches!(
                &app.overlay,
                Some(Overlay::Confirm(Confirm { action: Action::Cancel(run), .. })) if run == FIRST
            ),
            "{:?}",
            app.overlay
        );
        Ok(())
    }

    #[test]
    fn c_then_y_while_a_quit_is_pending_abandons_only_that_start()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::tui::training::Detach;
        let (_dir, mut app) = runs_app()?;
        let [first, second] = starts(&mut app, true);
        keys(&mut app, &[KeyCode::Char('q'), KeyCode::Char('y')]);
        assert!(
            dialog(&app).starts_with("Quitting waits"),
            "{}",
            dialog(&app)
        );
        keys(&mut app, &[KeyCode::Char('n')]);
        assert_eq!(app.leaving, Some(Exit::Quit));
        keys(&mut app, &[KeyCode::Char('c')]);
        assert_eq!(abandoning(&app), Some([first].as_slice()));
        assert_eq!(
            keys(&mut app, &[KeyCode::Char('y')]),
            [Effect::Abandon(first)]
        );
        assert_eq!(status(&app), Some("abandoning run 20260921-133200-a1b2"));
        let other = app.training.tasks.get(&second).map(|f| f.detach);
        assert_eq!(other, Some(Detach::OnStart), "still waited for");
        assert_eq!((app.leaving, app.exit), (Some(Exit::Quit), None));
        Ok(())
    }

    #[test]
    fn a_signal_while_the_c_dialog_is_open_closes_it_and_abandons_every_task()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = runs_app()?;
        let [first, second] = starts(&mut app, true);
        keys(&mut app, &[KeyCode::Char('c')]);
        assert_eq!(abandoning(&app), Some([first].as_slice()));
        let mut effects = app.on_signal();
        effects.sort_by_key(|effect| format!("{effect:?}"));
        assert_eq!(effects, [Effect::Abandon(first), Effect::Abandon(second)]);
        assert_eq!(app.overlay, None);
        assert_eq!(app.leaving, Some(Exit::Signal));
        Ok(())
    }

    #[test]
    fn c_on_a_starting_local_run_still_cancels_it_once_its_job_started()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = runs_app()?;
        let [first, _] = starts(&mut app, false);
        keys(&mut app, &[KeyCode::Char('c')]);
        assert!(
            matches!(
                &app.overlay,
                Some(Overlay::Confirm(Confirm { action: Action::Cancel(run), .. })) if run == FIRST
            ),
            "{:?}",
            app.overlay
        );
        assert_eq!(
            keys(&mut app, &[KeyCode::Char('y')]),
            [],
            "never during the start"
        );
        assert_eq!(
            status(&app),
            Some("run 20260921-133200-a1b2 is starting: it is cancelled once its job started")
        );
        assert_eq!(watching(&mut app, first), [Effect::Cancel(first)]);
        assert_eq!(app.leaving, None);
        Ok(())
    }

    #[test]
    fn c_is_refused_after_a_signal() -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = runs_app()?;
        app.on_signal();
        assert_eq!(keys(&mut app, &[KeyCode::Char('c')]), []);
        assert_eq!(app.overlay, None);
        assert_eq!(status(&app), Some("refused: interrupted, exiting"));
        Ok(())
    }

    #[test]
    fn staying_keeps_following_a_run_whose_job_has_not_started()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = runs_app()?;
        let follow = attach(&mut app)?;
        assert_eq!(
            keys(&mut app, &[KeyCode::Char('q'), KeyCode::Char('y')]),
            []
        );
        assert_eq!(app.leaving, Some(Exit::Quit));
        assert_eq!(keys(&mut app, &[KeyCode::Char('a')]), []);
        assert_eq!(
            status(&app),
            Some("refused: quitting; nothing new is followed")
        );
        keys(&mut app, &[KeyCode::Char('n')]);
        assert_eq!(status(&app), Some("not quitting"));
        assert_eq!(watching(&mut app, follow), [], "still followed");
        Ok(())
    }

    #[test]
    fn quitting_waits_for_a_start_then_detaches_it() -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = runs_app()?;
        let follow = attach(&mut app)?;
        assert_eq!(
            keys(&mut app, &[KeyCode::Char('q'), KeyCode::Char('y')]),
            []
        );
        assert_eq!(watching(&mut app, follow), [Effect::Cancel(follow)]);
        ended(&mut app, follow, Ok(Done::Trained(Err(DETACHED.into()))));
        assert_eq!(app.exit, Some(Exit::Quit));
        assert_eq!(app.exit_notes, [DETACHED]);
        Ok(())
    }

    #[test]
    fn a_signal_abandons_followed_runs() -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = runs_app()?;
        let follow = attach(&mut app)?;
        assert_eq!(app.on_signal(), [Effect::Abandon(follow)]);
        assert_eq!(
            status(&app),
            Some("interrupted: exiting once the training tasks end")
        );
        ended(&mut app, follow, Err("a background task failed".into()));
        assert_eq!(app.exit, Some(Exit::Signal));
        assert_eq!(app.exit_notes, ["a background task failed"]);
        Ok(())
    }

    #[test]
    fn a_signal_while_quitting_abandons_the_runs_already_detached()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = runs_app()?;
        let follow = attach(&mut app)?;
        watching(&mut app, follow);
        assert_eq!(
            keys(&mut app, &[KeyCode::Char('q'), KeyCode::Char('y')]),
            [Effect::Cancel(follow)]
        );
        assert_eq!(app.on_signal(), [Effect::Abandon(follow)]);
        assert_eq!(app.leaving, Some(Exit::Signal));
        assert_eq!(
            app.waiting_for(),
            ["waiting for run 20260921-133200-a1b2 to detach..."]
        );
        Ok(())
    }

    #[test]
    fn t_prepares_asks_with_list_prices_then_starts_one_run() -> Result<(), String> {
        let mut app = app();
        press(&mut app, &[KeyCode::Char('3')]);
        let effects = press(&mut app, &[KeyCode::Char('t')]);
        let [Effect::Spawn(prepare, Task::Prepare)] = effects.as_slice() else {
            return Err(format!("{effects:?}"));
        };
        let effects = app.on_done(
            *prepare,
            Ok(Done::Prepared(Ok(crate::tui::snapshots::runpod_plan()))),
        );
        let [Effect::Spawn(prices, Task::Prices(gpus))] = effects.as_slice() else {
            return Err(format!("{effects:?}"));
        };
        assert_eq!(gpus.len(), 3);
        let listed = vec![("NVIDIA A40".to_string(), Some(0.44))];
        app.on_done(*prices, Ok(Done::Prices(listed)));
        let Some(Overlay::Confirm(confirm)) = &app.overlay else {
            return Err("no dialog".into());
        };
        assert!(confirm.text.iter().any(|line| line.ends_with("$0.44/h")));
        let effects = press(&mut app, &[KeyCode::Char('y')]);
        let [Effect::Spawn(start, Task::Train(TrainJob::Start))] = effects.as_slice() else {
            return Err(format!("{effects:?}"));
        };
        assert_eq!(app.lock().as_deref(), Some("a new run is starting"));
        for code in ['t', 'r'] {
            assert_eq!(press(&mut app, &[KeyCode::Char(code)]), []);
        }
        app.on_message(Msg::Report(
            *start,
            Report::RunCreated("20260921-141320-ab12".into()),
        ));
        assert_eq!(
            app.lock().as_deref(),
            Some("run 20260921-141320-ab12 is starting")
        );
        app.on_message(Msg::Event(
            *start,
            crate::events::Event::JobStatus(crate::exec::JobStatus::Running),
        ));
        assert_eq!(app.lock(), None, "the lock ends once the job started");
        Ok(())
    }

    #[test]
    fn t_is_refused_without_training_or_while_the_data_is_locked() {
        let mut app = app();
        press(&mut app, &[KeyCode::Char('3')]);
        let effects = press(&mut app, &[KeyCode::Char('t')]);
        let [Effect::Spawn(prepare, _)] = effects.as_slice() else {
            return assert_eq!(effects, []);
        };
        let refused = "no [training] section in overbrainer.toml";
        assert_eq!(
            app.on_done(*prepare, Ok(Done::Prepared(Err(refused.into())))),
            []
        );
        assert_eq!(
            status(&app),
            Some("refused: no [training] section in overbrainer.toml")
        );
        app.edit = Some(TaskId(40));
        assert_eq!(press(&mut app, &[KeyCode::Char('t')]), []);
        assert_eq!(
            status(&app),
            Some("refused: an edit is being saved; one task at a time")
        );
    }

    #[test]
    fn quitting_never_cuts_a_start_and_abandons_only_when_confirmed() {
        let mut app = app();
        app.training.tasks.insert(
            TaskId(4),
            crate::tui::training::Follow::new(
                crate::tui::training::Job::Start { runpod: true },
                "",
            ),
        );
        press(&mut app, &[KeyCode::Char('q')]);
        assert_eq!(
            press(&mut app, &[KeyCode::Char('y')]),
            [],
            "never during the start"
        );
        assert!(matches!(
            &app.overlay,
            Some(Overlay::Confirm(Confirm {
                action: Action::Abandon(_),
                ..
            }))
        ));
        assert_eq!(press(&mut app, &[KeyCode::Char('n')]), [], "waiting");
        let effects = app.on_message(Msg::Event(
            TaskId(4),
            crate::events::Event::JobStatus(crate::exec::JobStatus::Running),
        ));
        assert_eq!(
            effects,
            [Effect::Cancel(TaskId(4))],
            "detached once its job started"
        );

        let mut app = super::super::snapshots::app();
        app.training.tasks.insert(
            TaskId(5),
            crate::tui::training::Follow::new(
                crate::tui::training::Job::Start { runpod: true },
                "",
            ),
        );
        press(&mut app, &[KeyCode::Char('q'), KeyCode::Char('y')]);
        assert_eq!(
            press(&mut app, &[KeyCode::Char('y')]),
            [Effect::Abandon(TaskId(5))]
        );
    }

    /// A start task on a Runpod target, bound to run [`FIRST`].
    fn starting(app: &mut App, id: TaskId) {
        app.training.tasks.insert(
            id,
            crate::tui::training::Follow::new(
                crate::tui::training::Job::Start { runpod: true },
                "",
            ),
        );
        app.on_message(Msg::Report(id, Report::RunCreated(FIRST.into())));
    }

    #[test]
    fn q_while_waiting_offers_to_abandon_again_and_a_signal_never_asks() {
        let mut app = app();
        starting(&mut app, TaskId(6));
        press(&mut app, &[KeyCode::Char('q'), KeyCode::Char('y')]);
        assert!(dialog(&app).starts_with("Quitting waits until the job of run 20260921-133200"));
        press(&mut app, &[KeyCode::Char('n')]);
        assert_eq!(app.overlay, None);
        press(&mut app, &[KeyCode::Char('q')]);
        assert!(
            dialog(&app).starts_with("Quitting waits"),
            "asked again: {:?}",
            app.overlay
        );
        assert_eq!(
            app.on_signal(),
            [Effect::Abandon(TaskId(6))],
            "a signal abandons at once"
        );
        assert_eq!(app.overlay, None);
        press(&mut app, &[KeyCode::Char('q')]);
        assert_eq!(app.overlay, None, "nothing left to abandon");
        assert_eq!(
            app.waiting_for(),
            ["waiting for run 20260921-133200-a1b2 to be abandoned..."]
        );
    }

    #[test]
    fn staying_after_the_abandon_dialog_keeps_the_start_followed() {
        let mut app = app();
        starting(&mut app, TaskId(6));
        press(&mut app, &[KeyCode::Char('q'), KeyCode::Char('y')]);
        press(&mut app, &[KeyCode::Char('n'), KeyCode::Char('n')]);
        assert_eq!(app.leaving, None);
        assert_eq!(status(&app), Some("not quitting"));
        assert_eq!(watching(&mut app, TaskId(6)), [], "still followed");
    }

    #[test]
    fn c_on_a_starting_run_cancels_it_once_its_job_started() {
        let mut app = app();
        starting(&mut app, TaskId(6));
        assert_eq!(app.cancel_run(FIRST), [], "never during the start");
        assert_eq!(
            status(&app),
            Some("run 20260921-133200-a1b2 is starting: it is cancelled once its job started")
        );
        assert_eq!(watching(&mut app, TaskId(6)), [Effect::Cancel(TaskId(6))]);
        let effects = app.on_done(TaskId(6), Ok(Done::Trained(Err(DETACHED.into()))));
        assert!(
            matches!(effects.as_slice(), [.., Effect::Spawn(_, Task::Train(TrainJob::Cancel(run)))] if run == FIRST),
            "{effects:?}"
        );
    }

    #[test]
    fn a_start_that_fails_before_its_run_exists_says_so() {
        let mut app = app();
        app.training.tasks.insert(
            TaskId(6),
            crate::tui::training::Follow::new(
                crate::tui::training::Job::Start { runpod: false },
                "",
            ),
        );
        let effects = app.on_done(
            TaskId(6),
            Ok(Done::Trained(Err("training.target: unknown".into()))),
        );
        assert!(
            effects
                .iter()
                .all(|e| !matches!(e, Effect::Spawn(_, Task::Series(_)))),
            "{effects:?}"
        );
        assert_eq!(status(&app), Some("a new run: training.target: unknown"));
        assert!(app.training.ended.is_empty());
    }

    #[test]
    fn stale_or_failed_price_lookups_never_leave_the_dialog_waiting() -> Result<(), String> {
        let mut app = app();
        let plan = crate::tui::snapshots::runpod_plan();
        let first = app.prepared(Ok(plan.clone()));
        app.overlay = None;
        let second = app.prepared(Ok(plan));
        let ([Effect::Spawn(old, _)], [Effect::Spawn(new, _)]) =
            (first.as_slice(), second.as_slice())
        else {
            return Err(format!("{first:?} {second:?}"));
        };
        let listed = vec![("NVIDIA A40".to_string(), Some(0.44))];
        app.on_done(*old, Ok(Done::Prices(listed)));
        assert!(dialog(&app).contains("looking up list prices..."), "stale");
        app.on_done(*new, Err("a background task failed: cancelled".into()));
        assert!(!dialog(&app).contains("looking up"), "{}", dialog(&app));
        assert!(dialog(&app).contains("NVIDIA A40                 list price unknown"));
        assert_eq!(app.prices, None);
        Ok(())
    }

    #[test]
    fn a_plan_never_replaces_an_open_dialog_and_y_still_quits() -> Result<(), String> {
        let mut app = app();
        // A followed run: `q` asks first.
        app.training.tasks.insert(
            TaskId(9),
            crate::tui::training::Follow::new(crate::tui::training::Job::Attach, FIRST),
        );
        let effects = press(&mut app, &[KeyCode::Char('3'), KeyCode::Char('t')]);
        let Some(Effect::Spawn(prepare, Task::Prepare)) = effects.last() else {
            return Err(format!("{effects:?}"));
        };
        press(&mut app, &[KeyCode::Char('q')]);
        let effects = app.on_done(
            *prepare,
            Ok(Done::Prepared(Ok(crate::tui::snapshots::runpod_plan()))),
        );
        assert_eq!(effects, [], "no price lookup either");
        assert!(
            matches!(
                &app.overlay,
                Some(Overlay::Confirm(Confirm {
                    action: Action::Quit,
                    ..
                }))
            ),
            "{:?}",
            app.overlay
        );
        assert_eq!(
            status(&app),
            Some("a run is prepared: press t again to start it")
        );
        let effects = press(&mut app, &[KeyCode::Char('y')]);
        assert!(
            !effects
                .iter()
                .any(|e| matches!(e, Effect::Spawn(_, Task::Train(TrainJob::Start)))),
            "{effects:?}"
        );
        assert_eq!(app.leaving, Some(Exit::Quit), "y quits");
        Ok(())
    }

    #[test]
    fn a_plan_waits_while_the_help_or_the_filter_is_open() {
        let mut app = app();
        for open in [true, false] {
            if open {
                app.overlay = Some(Overlay::Help);
            } else {
                app.overlay = None;
                app.dataset.input = Some("bor".into());
            }
            app.prepare = Some(TaskId(3));
            let effects = app.on_done(
                TaskId(3),
                Ok(Done::Prepared(Ok(crate::tui::snapshots::runpod_plan()))),
            );
            assert_eq!(effects, []);
            assert!(!matches!(app.overlay, Some(Overlay::Confirm(_))));
        }
    }

    #[test]
    fn t_while_a_run_is_prepared_says_so() {
        let mut app = app();
        press(&mut app, &[KeyCode::Char('3'), KeyCode::Char('t')]);
        assert_eq!(press(&mut app, &[KeyCode::Char('t')]), []);
        assert_eq!(status(&app), Some("already preparing a run"));
    }

    #[test]
    fn closing_the_start_dialog_forgets_its_price_lookup() -> Result<(), String> {
        for code in [KeyCode::Char('n'), KeyCode::Char('y'), KeyCode::Esc] {
            let mut app = app();
            let effects = app.prepared(Ok(crate::tui::snapshots::runpod_plan()));
            let [Effect::Spawn(lookup, Task::Prices(_))] = effects.as_slice() else {
                return Err(format!("{effects:?}"));
            };
            assert!(app.work().contains(&"preparing a run".to_string()));
            press(&mut app, &[code]);
            assert_eq!(app.prices, None, "{code:?}");
            assert!(!app.work().contains(&"preparing a run".to_string()));
            app.on_done(*lookup, Ok(Done::Prices(Vec::new())));
        }
        let mut app = app();
        app.prepared(Ok(crate::tui::snapshots::runpod_plan()));
        app.on_input(&ctrl_c());
        assert_eq!(app.prices, None, "Ctrl-C");
        Ok(())
    }

    #[test]
    fn abandoning_drops_a_cancel_asked_for_during_the_start() {
        let mut app = app();
        starting(&mut app, TaskId(6));
        app.cancel_run(FIRST);
        press(&mut app, &[KeyCode::Char('q'), KeyCode::Char('y')]);
        assert!(
            dialog(&app).starts_with("Quitting waits"),
            "{}",
            dialog(&app)
        );
        assert_eq!(
            press(&mut app, &[KeyCode::Char('y')]),
            [Effect::Abandon(TaskId(6))]
        );
        let failed = "interrupted before its job started: run 20260921-133200-a1b2 failed";
        let effects = app.on_done(TaskId(6), Ok(Done::Trained(Err(failed.into()))));
        assert!(
            !effects
                .iter()
                .any(|e| matches!(e, Effect::Spawn(_, Task::Train(TrainJob::Cancel(_))))),
            "{effects:?}"
        );
        assert_eq!(app.exit, Some(Exit::Quit));
        assert_eq!(app.exit_notes, [failed]);
    }

    /// A signal abandons a start whose cancel was asked for: the cancel never
    /// runs, and a note says how to run it, true whether the start failed with
    /// no job or detached with one.
    #[test]
    fn a_signal_during_a_start_notes_its_cancel_never_runs() {
        let failed = "interrupted before its job started: run 20260921-133200-a1b2 failed";
        for error in [failed, DETACHED] {
            let mut app = app();
            starting(&mut app, TaskId(6));
            app.cancel_run(FIRST);
            assert_eq!(app.on_signal(), [Effect::Abandon(TaskId(6))]);
            let effects = app.on_done(TaskId(6), Ok(Done::Trained(Err(error.into()))));
            assert!(
                !effects
                    .iter()
                    .any(|e| matches!(e, Effect::Spawn(_, Task::Train(TrainJob::Cancel(_))))),
                "{effects:?}"
            );
            assert_eq!(app.exit, Some(Exit::Signal));
            assert_eq!(
                app.exit_notes,
                [
                    error.to_string(),
                    "run 20260921-133200-a1b2 was not cancelled (a signal came during its \
                     start); if it is running, cancel it with `overbrainer train cancel \
                     20260921-133200-a1b2`"
                        .to_string(),
                ]
            );
        }
    }

    /// Staying after `y` keeps the note of a run the quit detached, and the
    /// flow's lines: the run is no longer followed, so a later quit with
    /// nothing running still says it was left running.
    #[test]
    fn staying_keeps_the_note_of_a_run_left_detached() -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = runs_app()?;
        crate::tui::snapshots::pipeline_running(&mut app);
        let follow = attach(&mut app)?;
        watching(&mut app, follow);
        let effects = keys(&mut app, &[KeyCode::Char('q'), KeyCode::Char('y')]);
        assert!(effects.contains(&Effect::Cancel(follow)), "{effects:?}");
        let warning = "train: remove the pod with `overbrainer pod rm`";
        app.on_message(Msg::Report(follow, Report::Line(warning.into())));
        ended(&mut app, follow, Ok(Done::Trained(Err(DETACHED.into()))));
        keys(&mut app, &[KeyCode::Char('n')]);
        assert_eq!(app.leaving, None);
        ended(
            &mut app,
            TaskId(7),
            Ok(Done::Pipeline(Err("interrupted".into()))),
        );
        assert_eq!(app.exit_notes, [warning, DETACHED]);
        keys(&mut app, &[KeyCode::Char('q')]);
        assert_eq!(app.exit, Some(Exit::Quit), "nothing left to wait for");
        assert_eq!(app.exit_notes, [warning, DETACHED]);
        Ok(())
    }

    /// A note of leaving goes once its work is taken up again: a run attached
    /// again, the stage run again. The flow's lines stay.
    #[test]
    fn a_note_of_leaving_goes_once_its_work_is_taken_up_again()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = runs_app()?;
        crate::tui::snapshots::pipeline_running(&mut app);
        let follow = attach(&mut app)?;
        watching(&mut app, follow);
        app.edit = Some(TaskId(40));
        keys(&mut app, &[KeyCode::Char('q'), KeyCode::Char('y')]);
        let warning = "train: remove the pod with `overbrainer pod rm`";
        app.on_message(Msg::Report(follow, Report::Line(warning.into())));
        ended(&mut app, follow, Ok(Done::Trained(Err(DETACHED.into()))));
        ended(
            &mut app,
            TaskId(7),
            Ok(Done::Pipeline(Err("interrupted".into()))),
        );
        assert_eq!(app.exit_notes, [warning, DETACHED, "run: interrupted"]);
        keys(&mut app, &[KeyCode::Char('n')]);
        ended(&mut app, TaskId(40), Err("a background task failed".into()));
        attach(&mut app)?;
        assert_eq!(app.exit_notes, [warning, "run: interrupted"]);
        app.overlay = Some(Overlay::Menu(0));
        keys(&mut app, &[KeyCode::Enter]);
        assert!(app.pipeline_task.is_some(), "the stage runs again");
        assert_eq!(app.exit_notes, [warning]);
        Ok(())
    }

    #[test]
    fn capital_r_reloads_the_data_files_and_the_runs() {
        let mut app = app();
        let effects = press(&mut app, &[KeyCode::Char('R')]);
        assert!(
            matches!(
                effects.as_slice(),
                [Effect::Spawn(_, Task::Load), Effect::Spawn(_, Task::Runs)]
            ),
            "{effects:?}"
        );
    }

    #[test]
    fn a_plan_that_arrives_while_quitting_is_dropped() {
        let mut app = app();
        let effects = press(&mut app, &[KeyCode::Char('3'), KeyCode::Char('t')]);
        let Some(Effect::Spawn(prepare, Task::Prepare)) = effects.last() else {
            return assert_eq!(effects, []);
        };
        app.leaving = Some(Exit::Quit);
        let effects = app.on_done(
            *prepare,
            Ok(Done::Prepared(Ok(crate::tui::snapshots::runpod_plan()))),
        );
        assert_eq!(effects, []);
        assert_eq!(app.overlay, None);
        assert_eq!(app.prepare, None);
    }
}
