//! The state of the TUI and what keys and events do to it. Nothing here draws,
//! reads the clock or touches the terminal: the loop feeds it input, ticks and
//! signals, and renders it.

use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::time::{Duration, SystemTime};

use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use tracing::Level;

use super::catalog::{CatalogKind, Listed, Query};
use super::config_watch::ConfigWatch;
use super::dataset::{DatasetView, Model, Node, TopicInfo};
use super::editor::{self, Session, Target};
use super::follow::REFRESH;
use super::motion::{Motion, MotionLevel};
use super::pipeline::{PipelineView, STAGES, command_name};
use super::project::{ProjectConfig, ProjectView};
use super::project_edit::{Removal, SaveRefusal, editor_failure};
use super::start::{Gpus, StartPlan};
use super::tasks::{Done, Edit, History, Msg, Saved, Task, TaskId};
use super::theme::Theme;
use super::training::TrainingView;
use super::views::logs::{export_line, level_name};
use super::widgets::form::{Input, InputOutcome};
use super::widgets::picker::{Choice, Entry, Picker, PickerOutcome};
use crate::cli::data::Command;
use crate::cli::front::Report;
use crate::config::edit::FieldPath;
use crate::config::{EnvSource, Settings, Source};
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

/// The repository `g` opens.
pub(super) const REPOSITORY: &str = env!("CARGO_PKG_REPOSITORY");

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
    /// Open this URL in a browser, detached.
    OpenUrl(String),
    /// Has the tasks started from now read the settings from this source:
    /// the configuration the app keeps, read at start, saved or read again.
    UseConfig(Source),
    /// Writes `lines` to `.overbrainer/<name>`, off the UI thread, never
    /// overwriting an existing file.
    ExportLogs {
        /// File name inside `.overbrainer/`.
        name: String,
        /// The lines to write, oldest first.
        lines: Vec<String>,
    },
}

/// The five views, switched with `1` to `5`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum View {
    /// The configuration and the project's stats.
    Project,
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
    pub(super) const ALL: [Self; 5] = [
        Self::Project,
        Self::Dataset,
        Self::Pipeline,
        Self::Training,
        Self::Logs,
    ];

    /// The tab title.
    pub(super) fn title(self) -> &'static str {
        match self {
            Self::Project => "Project",
            Self::Dataset => "Dataset",
            Self::Pipeline => "Pipeline",
            Self::Training => "Training",
            Self::Logs => "Logs",
        }
    }

    /// Position in [`View::ALL`].
    pub(super) fn index(self) -> usize {
        match self {
            Self::Project => 0,
            Self::Dataset => 1,
            Self::Pipeline => 2,
            Self::Training => 3,
            Self::Logs => 4,
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
    /// A picker of the Runpod catalog.
    Picker(Box<Picking>),
}

/// What opened a picker, which its kept choice goes back to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Origin {
    /// Enter on this field of the Project view: the choice becomes a pending
    /// edit of it.
    Field(FieldPath),
    /// `g` or `c` in the dialog starting a run on this Runpod target: the
    /// choice is used for the run and saved on `y`.
    Start(String),
}

/// A picker open over the view, and the listing that fills it.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Picking {
    /// What it lists.
    pub(super) kind: CatalogKind,
    /// What opened it.
    pub(super) origin: Origin,
    /// The task reading its entries; a result of any other is ignored.
    pub(super) task: TaskId,
    /// Its state.
    pub(super) picker: Picker,
}

/// A choice kept in a picker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Picked {
    /// What the picker listed.
    pub(super) kind: CatalogKind,
    /// What was kept.
    pub(super) choice: Choice,
    /// The entries of the IDs kept, as listed.
    pub(super) entries: Vec<Entry>,
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
    /// Taking a topic, a provider or a target out of the pending changes.
    Remove(Removal),
    /// Dropping the pending changes to `overbrainer.toml`.
    DropChanges,
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
    /// Whether the load running was started in the background, while a stage
    /// runs: the footer does not show it.
    quiet_load: bool,
    /// When the last load started or ended.
    reloaded: SystemTime,
    /// The edit being saved, if any.
    pub(super) edit: Option<TaskId>,
    /// The edit open in the editor or being saved.
    pub(super) editing: Option<Session>,
    /// The editor command.
    pub(super) editor: Vec<String>,
    /// The cost of the stages the history recorded at the last load, `None`
    /// while none spent anything.
    pub(super) history_cost: Option<crate::history::Cost>,
    /// What the history recorded at the last load that read it.
    pub(super) history: History,
    /// The configuration the Project view shows, read when the TUI starts.
    pub(super) config: Option<ProjectConfig>,
    /// The Project view's state.
    pub(super) project_view: ProjectView,
    /// The environment the configuration is read with.
    pub(super) env: EnvSource,
    /// The look at the configuration files every few seconds, `None` when
    /// they are not watched (the tests).
    pub(super) watch: Option<ConfigWatch>,
    /// The last error reading the history, warned once, until a read works.
    history_error: Option<String>,
    /// While a pipeline task runs, and until the reload after its end read the
    /// history: the history cost it started with, which its own rows add to.
    pub(super) cost_base: Option<super::cost::Base>,
    /// The load in flight when that task started: none of its stages is in the
    /// history it reads, so it sets the base.
    base_load: Option<TaskId>,
    /// The reload after that task ended: once it read the history, the base goes.
    settle: Option<TaskId>,
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
    /// The task reading the GPU catalog of the start dialog, if any; an
    /// earlier one's result is ignored.
    pub(super) start_catalog: Option<TaskId>,
    /// The GPU catalog of the start dialog, while it or its picker is open.
    pub(super) start_gpus: Option<Gpus>,
    /// The plan of the start dialog while its picker is open.
    pub(super) start_held: Option<Box<StartPlan>>,
    /// The run started once the save of its choices ends well.
    pub(super) start_after_save: Option<Box<StartPlan>>,
    /// The catalog listings running, a picker's or a hint's: their failure
    /// shows in their picker, if still open, and nowhere else.
    catalog_reads: Vec<(TaskId, CatalogKind)>,
    /// The GPU types a catalog listing read last, for the hints of the Runpod
    /// target fields.
    pub(super) gpu_catalog: Option<Vec<crate::runpod::GpuType>>,
    /// The network volumes a catalog listing read last, for a volume typed
    /// instead of picked.
    pub(super) volume_catalog: Option<Vec<Entry>>,
    /// The volume listing started last: only its result is kept, so an older
    /// one finishing later never replaces it.
    volume_read: Option<TaskId>,
    /// The release on crates.io, when newer than the one running.
    pub(super) newer: Option<String>,
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
            view: View::Project,
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
            quiet_load: false,
            reloaded: Self::never(),
            edit: None,
            editing: None,
            editor: vec!["vi".to_string()],
            history_cost: None,
            history: History::default(),
            config: None,
            project_view: ProjectView::default(),
            env: EnvSource::Process,
            watch: None,
            history_error: None,
            cost_base: None,
            base_load: None,
            settle: None,
            pipeline: PipelineView::default(),
            pipeline_task: None,
            pipeline_last: None,
            training: TrainingView::default(),
            refreshed: Self::never(),
            prepare: None,
            start_catalog: None,
            start_gpus: None,
            start_held: None,
            start_after_save: None,
            catalog_reads: Vec::new(),
            gpu_catalog: None,
            volume_catalog: None,
            volume_read: None,
            newer: None,
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

    /// What to do when the loop starts: have the tasks read the configuration
    /// kept, if any, and load the data.
    pub(super) fn start(&mut self) -> Vec<Effect> {
        let mut effects: Vec<Effect> = self
            .config
            .as_ref()
            .map(|config| self.use_config(config))
            .into_iter()
            .collect();
        effects.extend(self.reload());
        effects
    }

    /// Has the tasks read their settings from `config`, with the environment
    /// the app reads the configuration with.
    pub(super) fn use_config(&self, config: &ProjectConfig) -> Effect {
        Effect::UseConfig(Source {
            text: Some(config.text.clone()),
            env: self.env.clone(),
        })
    }

    /// A new task ID.
    pub(super) fn task_id(&mut self) -> TaskId {
        self.next_task += 1;
        TaskId(self.next_task)
    }

    /// Reloads the data files; while a load runs, one more starts when it ends,
    /// and the footer shows the running load even when it started quietly.
    pub(super) fn reload(&mut self) -> Vec<Effect> {
        if self.load.is_some() {
            self.reload_pending = true;
            self.quiet_load = false;
            return Vec::new();
        }
        let id = self.task_id();
        if self.pipeline_task.is_none() && self.cost_base.is_some() {
            self.settle = Some(id);
        }
        self.load = Some(id);
        self.load_at = Some(self.motion.clock());
        self.quiet_load = false;
        self.reloaded = self.now;
        vec![Effect::Spawn(id, Task::Load)]
    }

    /// Reloads the data files in the background while a stage runs and the
    /// Dataset view is shown, so its counts follow what the stage writes: at
    /// once when `now` is set, else once the last load ended long enough ago (or
    /// the clock went back), so a slow load is not followed by another at once. A load already running is left alone.
    fn reload_while_running(&mut self, now: bool) -> Vec<Effect> {
        if self.view != View::Dataset || self.pipeline_task.is_none() || self.load.is_some() {
            return Vec::new();
        }
        let recent = matches!(
            self.now.duration_since(self.reloaded),
            Ok(since) if since < REFRESH
        );
        if recent && !now {
            return Vec::new();
        }
        let effects = self.reload();
        self.quiet_load = true;
        effects
    }

    /// The work running, as the footer shows it, each with a spinner.
    pub(super) fn work(&self) -> Vec<String> {
        let mut work = Vec::new();
        if self.leaving.is_some() {
            work.push("quitting: waiting".to_string());
        }
        if self.load.is_some() && !self.quiet_load {
            work.push("loading".to_string());
        }
        if let Some(progress) = self.pipeline.progress() {
            work.push(progress);
        }
        if self.edit.is_some() || self.project_view.save.is_some() {
            work.push("saving".to_string());
        }
        if self.prepare.is_some() || self.start_catalog.is_some() {
            work.push("preparing a run".to_string());
        }
        if let Some(Overlay::Picker(picking)) = &self.overlay
            && picking.picker.loading()
        {
            work.push("reading the catalog".to_string());
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
        if self.start_after_save.is_some() {
            return Some("a new run is starting".to_string());
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
            Ok(Done::Loaded(loaded, history)) => self.on_loaded(id, loaded, history),
            Ok(Done::Saved(saved)) => self.saved(saved),
            Ok(Done::Pipeline(outcome)) => self.pipeline_ended(id, outcome),
            Ok(Done::Trained(result)) => self.trained(id, result),
            Ok(Done::Runs(listing)) => self.listed(id, listing),
            Ok(Done::Series { run, series }) => self.series_read(id, run, series),
            Ok(Done::Prepared(plan)) if self.prepare == Some(id) => self.prepared(plan),
            Ok(Done::StartCatalog(gpus)) if self.start_catalog == Some(id) => {
                self.start_catalog_read(gpus);
                Vec::new()
            },
            Ok(Done::ConfigSaved(saved)) if self.project_view.save == Some(id) => {
                self.config_saved(saved)
            },
            Ok(Done::Catalog(listed)) => {
                self.listed_catalog(id, listed);
                Vec::new()
            },
            Ok(Done::ConfigChecked(checked)) => self.config_checked(id, *checked),
            Ok(Done::Prepared(_) | Done::StartCatalog(_) | Done::ConfigSaved(_)) => Vec::new(),
            Err(error) => self.failed(id, error),
        }
    }

    /// Load `id` read `loaded` and the cost in the history: shown when it is the
    /// load running, else ignored. A history that cannot be read keeps the cost
    /// known so far; its error is warned once, not at every quiet reload.
    fn on_loaded(
        &mut self,
        id: TaskId,
        loaded: Result<Dataset, String>,
        history: Result<History, String>,
    ) -> Vec<Effect> {
        if self.load != Some(id) {
            return Vec::new();
        }
        self.load = None;
        self.reloaded = self.now;
        match history {
            Ok(history) => {
                let cost = history.cost;
                self.history_cost = cost;
                self.history = history;
                self.history_error = None;
                if self.base_load == Some(id) {
                    self.base_load = None;
                    if let Some(base) = &mut self.cost_base {
                        base.0 = cost;
                    }
                }
                if self.settle == Some(id) {
                    self.settle = None;
                    self.cost_base = None;
                }
            },
            // Every quiet reload reads it again: warn once per new error.
            Err(error) if self.history_error.as_ref() != Some(&error) => {
                tracing::warn!("{error}");
                self.history_error = Some(error);
            },
            Err(_) => {},
        }
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
        if self.project_view.save == Some(id) {
            return self.config_saved(Err(SaveRefusal::Failed(error)));
        }
        if self.check_failed(id) {
            return Vec::new();
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
        if self.start_catalog == Some(id) {
            // The dialog never keeps waiting.
            self.start_catalog_read(Err(error));
            return Vec::new();
        }
        if self.catalog_reads.iter().any(|(read, _)| *read == id) {
            self.listed_catalog(id, Err(error));
            return Vec::new();
        }
        if self.read_failed(id, &error) {
            return Vec::new();
        }
        if self.load == Some(id) {
            self.load = None;
            self.reloaded = self.now;
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
            Msg::BrowserFailed(url) => {
                self.say(Severity::Warn, format!("cannot open a browser: {url}"));
                return Vec::new();
            },
            Msg::LogsExported(result) => {
                match result {
                    Ok((name, count)) => self.say(
                        Severity::Info,
                        format!("exported {count} lines to .overbrainer/{name}"),
                    ),
                    Err(error) => {
                        self.say(Severity::Error, format!("cannot export the logs: {error}"));
                    },
                }
                return Vec::new();
            },
            Msg::NewerRelease(version) => {
                self.newer = Some(version.clone());
                return Vec::new();
            },
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
            Msg::Report(_, Report::RunCreated(_))
            | Msg::EditorExited(_)
            | Msg::BrowserFailed(_)
            | Msg::LogsExported(_)
            | Msg::NewerRelease(_) => {},
        }
        Vec::new()
    }

    /// Whether task `id` is the pipeline task running or the last one.
    fn is_pipeline(&self, id: TaskId) -> bool {
        self.pipeline_task == Some(id) || self.pipeline_last == Some(id)
    }

    /// Opens the picker `query` asks for over the view, `preselected` chosen,
    /// and reads its entries in the background; what it keeps goes back to
    /// `origin`.
    pub(super) fn open_picker(
        &mut self,
        query: Query,
        preselected: Choice,
        origin: Origin,
    ) -> Vec<Effect> {
        let kind = query.kind;
        let task = self.task_id();
        let picker = Picker::new(kind.spec(), preselected);
        // A choice at start is used as picked, never typed.
        let picker = match origin {
            Origin::Field(_) => picker,
            Origin::Start(_) => picker.untyped(),
        };
        self.overlay = Some(Overlay::Picker(Box::new(Picking {
            kind,
            origin,
            task,
            picker,
        })));
        self.catalog_reads.push((task, kind));
        if kind == CatalogKind::Volumes {
            self.volume_read = Some(task);
        }
        vec![Effect::Spawn(task, Task::Catalog(query))]
    }

    /// Reads the GPU types for the field hints in the background, unless they
    /// were read or a listing reading them runs.
    pub(super) fn read_gpu_catalog(&mut self, gpu_count: u32) -> Vec<Effect> {
        let reading = self
            .catalog_reads
            .iter()
            .any(|(_, kind)| matches!(kind, CatalogKind::Gpus | CatalogKind::DataCenters));
        if self.gpu_catalog.is_some() || reading {
            return Vec::new();
        }
        let task = self.task_id();
        self.catalog_reads.push((task, CatalogKind::Gpus));
        let query = Query {
            kind: CatalogKind::Gpus,
            gpu_count,
            gpu_types: Vec::new(),
        };
        vec![Effect::Spawn(task, Task::Catalog(query))]
    }

    /// Listing `id` read `listed`: its GPU types kept for the hints, its
    /// volumes (from the volume listing started last only) for a volume typed
    /// and the data centers they hold, its entries shown when it fills the
    /// picker open. Its failure shows in that picker only: once the picker is
    /// closed, nothing is said.
    fn listed_catalog(&mut self, id: TaskId, listed: Result<Listed, String>) {
        let Some(at) = self.catalog_reads.iter().position(|(read, _)| *read == id) else {
            return;
        };
        let (_, kind) = self.catalog_reads.remove(at);
        let volumes = kind == CatalogKind::Volumes && self.volume_read == Some(id);
        let entries = listed.map(|listed| {
            if !listed.gpus.is_empty() {
                self.gpu_catalog = Some(listed.gpus);
            }
            if volumes {
                self.volume_catalog = Some(listed.entries.clone());
            }
            listed.entries
        });
        if volumes && entries.is_ok() {
            self.reconcile_volume_centers();
        }
        if let Some(Overlay::Picker(picking)) = &mut self.overlay
            && picking.task == id
        {
            picking.picker.loaded(entries);
        }
    }

    /// A picker opened from `origin` kept `picked`: it goes back there.
    fn picked(&mut self, origin: Origin, picked: Picked) -> Vec<Effect> {
        match origin {
            Origin::Field(path) => self.picked_field(&path, picked),
            Origin::Start(target) => self.picked_start(&target, picked),
        }
        Vec::new()
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
        // What the history holds so far, the last task's stages included. The
        // load in flight may set it only when no task's stages are missing from
        // it: no window was open, or it is the last task's settling reload.
        let fresh = self.cost_base.is_none() || (self.load.is_some() && self.settle == self.load);
        self.cost_base = Some(super::cost::Base(super::cost::history_so_far(self)));
        self.base_load = if fresh { self.load } else { None };
        self.settle = None;
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
        let idle = self.edit.is_none()
            && self.project_view.save.is_none()
            && self.pipeline_task.is_none()
            && self.training.tasks.is_empty();
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
        if std::mem::take(&mut self.project_view.editing) {
            return self.config_edited(status);
        }
        let Some(session) = self.editing.clone() else {
            return Vec::new();
        };
        let failed = editor_failure(status).map(|failed| format!("{failed}; nothing changed"));
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
            Event::Paste(text) => {
                self.on_paste(text);
                Vec::new()
            },
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
            self.project_view.form = None;
            return self.quit();
        }
        if matches!(self.overlay, Some(Overlay::Picker(_))) {
            if !key.modifiers.difference(KeyModifiers::SHIFT).is_empty() {
                return Vec::new();
            }
            return self.on_picker_key(key.code);
        }
        if self.view == View::Project && self.project_view.form.is_some() {
            if key.modifiers.difference(KeyModifiers::SHIFT).is_empty() {
                self.on_form_key(key.code);
            }
            return Vec::new();
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
            Some(Overlay::Picker(_)) | None => {},
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
            KeyCode::Char('g') => return vec![Effect::OpenUrl(REPOSITORY.to_string())],
            KeyCode::Char('1') => return self.show(View::Project),
            KeyCode::Char('2') => return self.show(View::Dataset),
            KeyCode::Char('3') => return self.show(View::Pipeline),
            KeyCode::Char('4') => return self.show(View::Training),
            KeyCode::Char('5') => return self.show(View::Logs),
            KeyCode::Tab => return self.show(self.view.shifted(1)),
            KeyCode::BackTab => return self.show(self.view.shifted(View::ALL.len() - 1)),
            code => return self.on_view_key(code),
        }
        Vec::new()
    }

    /// A key in the picker open: it goes back to what opened it once kept,
    /// cancelled or typed instead.
    fn on_picker_key(&mut self, code: KeyCode) -> Vec<Effect> {
        let Some(Overlay::Picker(picking)) = &mut self.overlay else {
            return Vec::new();
        };
        match picking.picker.on_key(code) {
            PickerOutcome::Open => Vec::new(),
            PickerOutcome::Cancelled => {
                let origin = picking.origin.clone();
                self.overlay = None;
                if let Origin::Start(_) = origin {
                    self.reopen_start();
                }
                Vec::new()
            },
            PickerOutcome::Typed => {
                let origin = picking.origin.clone();
                self.overlay = None;
                match origin {
                    Origin::Field(path) => self.type_field(&path),
                    // Never: a picker at start is untyped.
                    Origin::Start(_) => self.reopen_start(),
                }
                Vec::new()
            },
            PickerOutcome::Kept(choice) => {
                let entries = match &choice {
                    Choice::Auto => Vec::new(),
                    Choice::List(ids) => ids
                        .iter()
                        .filter_map(|id| picking.picker.entry(id).cloned())
                        .collect(),
                };
                let (kind, origin) = (picking.kind, picking.origin.clone());
                self.overlay = None;
                let picked = Picked {
                    kind,
                    choice,
                    entries,
                };
                self.picked(origin, picked)
            },
        }
    }

    /// Shows `view`; the Training view reads `runs/` again, and switching to the
    /// Dataset view reads the data files again while a stage runs. Leaving a
    /// view never touches a task.
    fn show(&mut self, view: View) -> Vec<Effect> {
        let entered = self.view != view;
        self.view = view;
        match view {
            View::Training => self.refresh_runs(),
            View::Dataset if entered => self.reload_while_running(true),
            View::Project | View::Dataset | View::Pipeline | View::Logs => Vec::new(),
        }
    }

    /// Closes the overlay and returns it. Closing the start dialog, or its
    /// picker, forgets its plan and its GPU catalog lookup: nothing shows them
    /// any more.
    fn close_overlay(&mut self) -> Option<Overlay> {
        let overlay = self.overlay.take();
        let start = match &overlay {
            Some(Overlay::Confirm(confirm)) => matches!(confirm.action, Action::Start(_)),
            Some(Overlay::Picker(picking)) => matches!(picking.origin, Origin::Start(_)),
            _ => false,
        };
        if start {
            self.start_catalog = None;
            self.start_gpus = None;
            self.start_held = None;
        }
        overlay
    }

    /// `y` runs the dialog's action; `g` and `c` in the dialog starting a run
    /// on a Runpod target choose its GPU types and data centers; any other key
    /// closes it.
    fn on_confirm_key(&mut self, code: KeyCode) -> Vec<Effect> {
        if let KeyCode::Char(key @ ('g' | 'c')) = code
            && let Some(Overlay::Confirm(Confirm {
                action: Action::Start(plan),
                ..
            })) = &self.overlay
            && plan.runpod.is_some()
        {
            return self.choose_for_start(key == 'g');
        }
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
            Action::Start(plan) => self.confirm_start(plan),
            Action::Abandon(tasks) => self.abandon(&tasks),
            Action::AbandonStart(task) => self.abandon_start(task),
            Action::Remove(removal) => {
                self.remove(&removal);
                Vec::new()
            },
            Action::DropChanges => {
                if !self.refuse_change() {
                    self.drop_changes();
                    self.say(Severity::Info, "pending changes dropped");
                }
                Vec::new()
            },
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
            View::Project => return self.on_project_key(code),
            View::Dataset => return self.on_dataset_key(code),
            View::Training => return self.on_training_key(code),
            View::Logs => return self.on_logs_key(code),
            View::Pipeline => {},
        }
        Vec::new()
    }

    fn on_dataset_key(&mut self, code: KeyCode) -> Vec<Effect> {
        match code {
            KeyCode::Char(key @ ('e' | 'E')) => return self.edit_selected(key == 'E'),
            KeyCode::Char(key @ ('d' | 'D')) => {
                self.delete_selected(key == 'D');
                return Vec::new();
            },
            _ => {},
        }
        let view = &mut self.dataset;
        match code {
            KeyCode::Up | KeyCode::Char('k') => view.step(false),
            KeyCode::Down | KeyCode::Char('j') => view.step(true),
            KeyCode::Right | KeyCode::Char('l') => {
                view.tree.key_right();
            },
            KeyCode::Enter => {
                view.tree.toggle_selected();
            },
            KeyCode::Left | KeyCode::Char('h') => {
                view.tree.key_left();
                view.scroll = 0;
                view.sections.clear();
            },
            KeyCode::PageDown => view.scroll = view.scroll.saturating_add(PAGE),
            KeyCode::PageUp => view.scroll = view.scroll.saturating_sub(PAGE),
            KeyCode::Char(']') => {
                if let Some(&next) = view.sections.iter().find(|&&at| at > view.scroll) {
                    view.scroll = next;
                }
            },
            KeyCode::Char('[') => {
                if let Some(&previous) = view.sections.iter().rfind(|&&at| at < view.scroll) {
                    view.scroll = previous;
                }
            },
            KeyCode::Char('s') => view.stats = !view.stats,
            KeyCode::Char('/') => view.input = Some(Input::new(view.filter.clone())),
            KeyCode::Esc if !view.filter.is_empty() => view.apply_filter(String::new()),
            _ => {},
        }
        Vec::new()
    }

    /// `e`: opens the selected question or subtopic name in the editor; `E`
    /// (`answer`) opens the selected question's answer.
    fn edit_selected(&mut self, answer: bool) -> Vec<Effect> {
        if self.locked() {
            return Vec::new();
        }
        let target = match self.selected_target(answer) {
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

    /// What `e` edits for the selected node, or `E` (`answer`).
    fn selected_target(&self, answer: bool) -> Result<Target, &'static str> {
        let model = self.dataset.model.as_ref().ok_or("nothing to edit")?;
        if answer {
            let Some(Node::Question(id)) = self.dataset.tree.selected().last() else {
                return Err("no answer to edit");
            };
            let example = model.answer(id).ok_or("no answer to edit")?;
            return AnswerText::of(example)
                .map(|before| Target::Answer {
                    id: id.clone(),
                    before,
                })
                .ok_or("changed on disk; press R");
        }
        match self.dataset.tree.selected().last() {
            Some(Node::Question(id)) => model
                .question(id)
                .map(|q| Target::Question {
                    id: id.clone(),
                    text: q.text.clone(),
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

    /// `d`: asks to delete the selected node, with what goes with it; `D`
    /// (`answer`): the selected question's answer only.
    fn delete_selected(&mut self, answer: bool) {
        if self.locked() {
            return;
        }
        let Some(model) = &self.dataset.model else {
            return;
        };
        match deletion(
            model,
            self.dataset.tree.selected(),
            &self.project.topics,
            answer,
        ) {
            Ok(confirm) => self.overlay = Some(Overlay::Confirm(confirm)),
            Err(refusal) => self.say(Severity::Warn, refusal),
        }
    }

    /// A key while the filter is typed: Enter applies it, Esc clears it.
    fn on_filter_key(&mut self, code: KeyCode) {
        let view = &mut self.dataset;
        let Some(input) = &mut view.input else {
            return;
        };
        match input.on_key(code) {
            InputOutcome::Editing => {},
            InputOutcome::Done(filter) => {
                view.input = None;
                view.apply_filter(filter);
            },
            InputOutcome::Cancelled => {
                view.input = None;
                view.apply_filter(String::new());
            },
        }
    }

    /// A bracketed paste goes to the input being typed; with none, it is dropped.
    fn on_paste(&mut self, text: &str) {
        if let Some(Overlay::Picker(picking)) = &mut self.overlay {
            picking.picker.paste(text);
            return;
        }
        if self.view == View::Project {
            self.on_project_paste(text);
        }
        if self.view == View::Dataset
            && let Some(input) = &mut self.dataset.input
        {
            input.paste(text);
            self.dirty = true;
        }
    }

    fn on_logs_key(&mut self, code: KeyCode) -> Vec<Effect> {
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
            KeyCode::Char('x') => return self.export_logs(),
            _ => {},
        }
        Vec::new()
    }

    /// `x`: exports every retained line at the shown level or more severe,
    /// oldest first, to a new file under `.overbrainer/`; no file without one.
    fn export_logs(&mut self) -> Vec<Effect> {
        let window = self.logs.window(self.log_view.min, usize::MAX, 0);
        if window.lines.is_empty() {
            let level = level_name(self.log_view.min);
            self.say(Severity::Warn, format!("nothing to export at {level}"));
            return Vec::new();
        }
        let lines = window.lines.iter().map(export_line).collect();
        let name = format!("logs-{}.log", crate::runs::compact_utc(self.now));
        vec![Effect::ExportLogs { name, lines }]
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
        if self.project_view.save.is_some() {
            text.push("overbrainer.toml is being saved: quitting waits for it.".to_string());
        }
        if self.project_view.pending.is_some() {
            text.push(
                "The pending changes to overbrainer.toml are not saved: quitting drops them."
                    .to_string(),
            );
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

    /// What the TUI waits for once its loop ended, one line each: an edit or
    /// `overbrainer.toml` being saved, the stage stopping, and each training
    /// task.
    pub(super) fn waiting_for(&self) -> Vec<String> {
        let mut lines = Vec::new();
        if self.edit.is_some() {
            lines.push("waiting for an edit to be saved...".to_string());
        }
        if self.project_view.save.is_some() {
            lines.push(format!(
                "waiting for {} to be saved...",
                crate::config::CONFIG_FILE
            ));
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
        self.project_view.form = None;
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
            if self.project_view.save.is_some() {
                waited.push("overbrainer.toml is saved".to_string());
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
    /// again when the Training view is shown and it is time, the data files
    /// when the Dataset view is shown while a stage runs, and the configuration
    /// when its files changed.
    pub(super) fn on_tick(&mut self, now: SystemTime) -> Vec<Effect> {
        self.now = now;
        let mut effects = self.refresh_when_due();
        effects.extend(self.reload_while_running(false));
        effects.extend(self.check_config());
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

/// The confirmation of `d` on the node at `path`, with what it removes, or of
/// `D` (`answer`): the answer of the question at `path`.
fn deletion(
    model: &Model,
    path: &[Node],
    topics: &[TopicInfo],
    answer: bool,
) -> Result<Confirm, String> {
    let data = &model.data;
    let deletion = match path.last() {
        Some(Node::Question(id)) if answer && model.answer(id).is_some() => {
            Deletion::Answer(id.clone())
        },
        _ if answer => return Err("no answer to delete".to_string()),
        Some(Node::Subtopic(id)) => Deletion::Subtopic(id.clone()),
        Some(Node::Question(id)) => Deletion::Question(id.clone()),
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
    fn the_tui_opens_on_the_project_view() {
        let project = app().project;
        let theme = Theme::new(crate::tui::theme::ColorLevel::TrueColor);
        let app = App::new(project, LogBuffer::new(1), &theme, at(NOW));
        assert_eq!(app.view, View::Project);
    }

    #[test]
    fn digits_and_tabs_switch_the_five_views() {
        let mut app = app();
        for (digit, view) in ['1', '2', '3', '4', '5'].into_iter().zip(View::ALL) {
            press(&mut app, &[KeyCode::Char(digit)]);
            assert_eq!(app.view, view, "{digit}");
        }
        press(&mut app, &[KeyCode::Char('1')]);
        for view in View::ALL.into_iter().cycle().skip(1).take(5) {
            press(&mut app, &[KeyCode::Tab]);
            assert_eq!(app.view, view);
        }
        press(&mut app, &[KeyCode::BackTab]);
        assert_eq!(app.view, View::Logs);
        press(&mut app, &[KeyCode::Char('4')]);
        assert_eq!(app.view, View::Training);
    }

    #[test]
    fn the_project_view_moves_by_field_and_by_page() -> TestResult {
        let mut app = app();
        let config = crate::tui::snapshots::project_config()?;
        let count =
            crate::tui::project::rows(&config, None, &crate::tui::project::Locks::default())
                .iter()
                .filter(|row| matches!(row, crate::tui::project::Row::Field(_)))
                .count();
        app.config = Some(config);
        press(
            &mut app,
            &[KeyCode::Char('1'), KeyCode::Char('j'), KeyCode::Down],
        );
        assert_eq!(app.project_view.selected, 2);
        press(&mut app, &[KeyCode::PageDown]);
        assert_eq!(app.project_view.selected, 12);
        press(&mut app, &[KeyCode::End]);
        assert_eq!(app.project_view.selected, count - 1);
        press(&mut app, &[KeyCode::Char('j')]);
        assert_eq!(app.project_view.selected, count - 1, "stays on the last");
        press(&mut app, &[KeyCode::PageUp, KeyCode::Char('k')]);
        assert_eq!(app.project_view.selected, count - 12);
        press(&mut app, &[KeyCode::Home, KeyCode::Up]);
        assert_eq!(app.project_view.selected, 0);
        Ok(())
    }

    #[test]
    fn the_help_overlay_opens_and_closes_and_ignores_other_keys() {
        let mut app = app();
        press(&mut app, &[KeyCode::Char('?')]);
        assert_eq!(app.overlay, Some(Overlay::Help));
        press(&mut app, &[KeyCode::Char('4')]);
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
    fn g_opens_the_repository_in_every_view() {
        let mut app = app();
        for view in View::ALL {
            app.view = view;
            assert_eq!(
                press(&mut app, &[KeyCode::Char('g')]),
                [Effect::OpenUrl(REPOSITORY.to_string())],
                "{view:?}"
            );
        }
    }

    #[test]
    fn g_is_ignored_in_a_dialog_the_help_the_menu_and_the_filter() {
        let mut app = dataset_app();
        open_to(&mut app, &path_to(MOVED));
        press(&mut app, &[KeyCode::Char('d')]);
        assert!(matches!(app.overlay, Some(Overlay::Confirm(_))));
        assert_eq!(press(&mut app, &[KeyCode::Char('g')]), []);
        assert_eq!(app.overlay, None, "as any key but y, g answers no");
        app.overlay = Some(Overlay::Help);
        assert_eq!(press(&mut app, &[KeyCode::Char('g')]), []);
        assert_eq!(app.overlay, Some(Overlay::Help), "the help stays");
        app.overlay = Some(Overlay::Menu(0));
        assert_eq!(press(&mut app, &[KeyCode::Char('g')]), []);
        assert_eq!(app.overlay, None, "as any other key, g closes the menu");
        press(&mut app, &[KeyCode::Char('/'), KeyCode::Char('g')]);
        assert_eq!(
            app.dataset.input.as_ref().map(Input::text),
            Some("g"),
            "g is typed"
        );
    }

    #[test]
    fn a_browser_that_fails_is_said_and_stays_said() {
        let mut app = app();
        let url = REPOSITORY.to_string();
        assert_eq!(app.on_message(Msg::BrowserFailed(url.clone())), []);
        let status = app.status.as_ref().map(|s| (s.severity, s.text.as_str()));
        let said = format!("cannot open a browser: {url}");
        assert_eq!(status, Some((Severity::Warn, said.as_str())));
        // The loop logs why at debug: the next tick keeps the message. A newer
        // warning from elsewhere would replace it, as any status.
        log(&app, Level::DEBUG, "xdg-open exited with 3");
        app.on_tick(at(NOW + 1));
        let status = app.status.as_ref().map(|s| (s.severity, s.text.as_str()));
        assert_eq!(status, Some((Severity::Warn, said.as_str())));
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
        press(&mut app, &[KeyCode::Char('5')]);
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
        press(&mut app, &[KeyCode::Char('5')]);
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
        press(&mut app, &[KeyCode::Char('5')]);
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
        press(&mut app, &[KeyCode::Char('5'), KeyCode::Char('k')]);
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
    fn x_exports_only_the_shown_lines_oldest_first() -> TestResult {
        let mut app = app();
        log(&app, Level::DEBUG, "too quiet");
        log(&app, Level::WARN, "first");
        log(&app, Level::INFO, "second");
        log(&app, Level::ERROR, "third");
        press(&mut app, &[KeyCode::Char('5')]);
        let effects = press(&mut app, &[KeyCode::Char('x')]);
        let [Effect::ExportLogs { name, lines }] = effects.as_slice() else {
            return Err(format!("expected an export effect: {effects:?}").into());
        };
        assert_eq!(
            name,
            &format!("logs-{}.log", crate::runs::compact_utc(at(NOW)))
        );
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert!(
            lines[0].contains("WARN") && lines[0].contains("first"),
            "{lines:?}"
        );
        assert!(
            lines[1].contains("INFO") && lines[1].contains("second"),
            "{lines:?}"
        );
        assert!(
            lines[2].contains("ERROR") && lines[2].contains("third"),
            "{lines:?}"
        );
        assert!(
            !lines.iter().any(|line| line.contains("too quiet")),
            "{lines:?}"
        );
        Ok(())
    }

    #[test]
    fn x_with_no_line_at_the_shown_level_exports_nothing() {
        let mut app = app();
        log(&app, Level::DEBUG, "too quiet");
        press(&mut app, &[KeyCode::Char('5')]);
        assert_eq!(press(&mut app, &[KeyCode::Char('x')]), []);
        let status = app.status.as_ref().map(|s| (s.severity, s.text.as_str()));
        assert_eq!(status, Some((Severity::Warn, "nothing to export at INFO")));
    }

    #[test]
    fn a_newer_release_is_kept_for_the_footer_without_a_status() {
        let mut app = app();
        app.dirty = false;
        assert_eq!(app.on_message(Msg::NewerRelease("0.9.0".into())), []);
        assert_eq!(app.newer.as_deref(), Some("0.9.0"));
        assert!(app.dirty);
        assert_eq!(app.status, None);
    }

    #[test]
    fn logs_exported_says_the_outcome() {
        let mut app = app();
        let effects = app.on_message(Msg::LogsExported(Ok(("logs-x.log".into(), 3))));
        assert_eq!(effects, []);
        let status = app.status.clone();
        assert_eq!(
            status.as_ref().map(|s| s.text.as_str()),
            Some("exported 3 lines to .overbrainer/logs-x.log")
        );
        assert_eq!(status.map(|s| s.severity), Some(Severity::Info));
        let effects = app.on_message(Msg::LogsExported(Err("file exists".into())));
        assert_eq!(effects, []);
        let status = app.status.clone();
        assert_eq!(
            status.as_ref().map(|s| s.text.as_str()),
            Some("cannot export the logs: file exists")
        );
        assert_eq!(status.map(|s| s.severity), Some(Severity::Error));
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
    fn enter_opens_then_closes_the_selected_node() {
        let mut app = dataset_app();
        let topic = vec![Node::Topic("ownership".into())];
        press(&mut app, &[KeyCode::Enter]);
        assert!(app.dataset.tree.opened().contains(&topic));
        press(&mut app, &[KeyCode::Enter]);
        assert!(!app.dataset.tree.opened().contains(&topic));
        assert_eq!(app.dataset.tree.selected(), topic, "the selection stays");
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

    #[test]
    fn the_filter_edits_at_its_cursor_and_takes_a_paste() {
        let mut app = dataset_app();
        press(
            &mut app,
            &[KeyCode::Char('/'), KeyCode::Char('b'), KeyCode::Home],
        );
        assert_eq!(app.on_input(&Event::Paste("bor\nde".into())), []);
        press(
            &mut app,
            &[KeyCode::Char('a'), KeyCode::End, KeyCode::Enter],
        );
        assert_eq!(app.dataset.filter, "bor deab");
    }

    #[test]
    fn a_paste_without_an_input_is_ignored() {
        let mut app = dataset_app();
        assert_eq!(app.on_input(&Event::Paste("q".into())), []);
        assert_eq!(app.exit, None);
        assert_eq!(app.dataset.filter, "");
        assert_eq!(app.dataset.input, None);
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
        let second = only_load(&app.on_done(
            first,
            Ok(Done::Loaded(Ok(dataset()), Ok(History::default()))),
        ))?;
        assert_ne!(second, first);
        assert_eq!(app.load, Some(second));
        assert!(app.dataset.model.is_some(), "the first load is shown");
        assert_eq!(
            app.on_done(
                second,
                Ok(Done::Loaded(Ok(dataset()), Ok(History::default())))
            ),
            []
        );
        assert_eq!(app.load, None);
        Ok(())
    }

    #[test]
    fn a_load_sets_the_history_cost_and_keeps_it_when_the_history_cannot_be_read() -> TestResult {
        let mut app = app();
        let known = Ok(History::costing(crate::history::Cost::Known(1.5)));
        let id = only_load(&app.start())?;
        app.on_done(id, Ok(Done::Loaded(Ok(dataset()), known)));
        assert_eq!(app.history_cost, Some(crate::history::Cost::Known(1.5)));
        let id = only_load(&but_runs(press(&mut app, &[KeyCode::Char('R')])))?;
        app.on_done(
            id,
            Ok(Done::Loaded(Ok(dataset()), Err("permission denied".into()))),
        );
        assert_eq!(app.history_cost, Some(crate::history::Cost::Known(1.5)));
        Ok(())
    }

    #[test]
    fn a_history_read_error_is_warned_once_until_it_changes() -> TestResult {
        let mut app = app();
        let (loaded, buffer) = crate::logging::capture(|| -> TestResult {
            let mut id = only_load(&app.start())?;
            // A read that works again forgets the error.
            for history in ["denied", "denied", "gone", "gone"]
                .map(|error| Err(error.to_string()))
                .into_iter()
                .chain([Ok(History::default()), Err("gone".to_string())])
            {
                app.on_done(id, Ok(Done::Loaded(Ok(dataset()), history)));
                id = only_load(&but_runs(press(&mut app, &[KeyCode::Char('R')])))?;
            }
            Ok(())
        });
        loaded?;
        let warned: Vec<String> = buffer
            .since(tracing::Level::WARN, 0)
            .into_iter()
            .map(|line| line.message)
            .collect();
        assert_eq!(warned, ["denied", "gone", "gone"]);
        Ok(())
    }

    #[test]
    fn a_load_that_is_not_the_current_one_is_ignored() -> TestResult {
        let mut app = app();
        let id = only_load(&app.start())?;
        let stale = TaskId(id.0 + 100);
        assert_eq!(
            app.on_done(
                stale,
                Ok(Done::Loaded(Ok(dataset()), Ok(History::default())))
            ),
            []
        );
        assert!(app.dataset.model.is_none());
        assert_eq!(app.load, Some(id));
        app.on_done(
            id,
            Ok(Done::Loaded(
                Err("data/answers.jsonl:1: bad".into()),
                Ok(History::default()),
            )),
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
        open_to(&mut app, &path_to(MOVED));
        app.dataset.scroll = 1000;
        let rows = text(&draw(&mut app, 120, 40)?);
        assert_eq!(
            app.dataset.scroll, 0,
            "the question and its answer fit in 38 rows"
        );
        assert!(rows.iter().any(|row| row.contains("model deepseek-r1")));
        app.dataset.scroll = 1000;
        let rows = text(&draw(&mut app, 80, 24)?);
        assert_eq!(app.dataset.scroll, 3, "23 lines, 20 rows");
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
        open_to(&mut app, &path_to(MOVED));
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
        open_to(&mut app, &path_to(MOVED)[..2]);
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
        open_to(&mut app, &path_to(MOVED));
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
        open_to(&mut app, &path_to(MOVED));
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
    fn shift_d_asks_then_deletes_the_selected_question_s_answer_only_on_y() {
        let mut app = dataset_app();
        open_to(&mut app, &path_to(MOVED));
        press(&mut app, &[KeyCode::Char('D')]);
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
        press(&mut app, &[KeyCode::Char('D')]);
        let effects = press(&mut app, &[KeyCode::Char('y')]);
        let id = Id::question(&Id::subtopic("ownership", "Borrowing"), MOVED);
        assert!(matches!(
            effects.as_slice(),
            [Effect::Spawn(_, Task::Edit(Edit::Delete { deletion: Deletion::Answer(answer), counts }))]
                if *answer == id && *counts == Counts { questions: 0, answers: 1 }
        ));
    }

    #[test]
    fn shift_d_and_shift_e_warn_on_a_question_without_answer() {
        let mut app = dataset_app();
        open_to(&mut app, &path_to("When does NLL end a borrow?"));
        assert_eq!(press(&mut app, &[KeyCode::Char('D')]), []);
        assert_eq!(app.overlay, None);
        assert_eq!(status(&app), Some("no answer to delete"));
        assert_eq!(press(&mut app, &[KeyCode::Char('E')]), []);
        assert_eq!(status(&app), Some("no answer to edit"));
        assert!(app.editing.is_none());
    }

    #[test]
    fn shift_e_opens_the_selected_question_s_answer() -> Result<(), Box<dyn std::error::Error>> {
        let (_dir, mut app) = project_app()?;
        open_to(&mut app, &path_to(MOVED));
        let effects = press(&mut app, &[KeyCode::Char('E')]);
        let [Effect::OpenEditor { path, .. }] = effects.as_slice() else {
            return Err(format!("{effects:?}").into());
        };
        let text = std::fs::read_to_string(path)?;
        assert!(
            text.contains("The borrow must end before the move."),
            "{text}"
        );
        assert!(
            matches!(
                app.editing.as_ref().map(|session| &session.target),
                Some(Target::Answer { .. })
            ),
            "the answer is edited"
        );
        Ok(())
    }

    #[test]
    fn brackets_jump_between_the_parts_of_a_question() -> TestResult {
        let mut app = dataset_app();
        let mut data = dataset();
        let reply = data.answers[0].messages.last_mut().ok_or("no reply")?;
        reply.content.push_str(&"\nOne more line.".repeat(30));
        app.dataset.loaded(data, &app.project.topics.clone());
        open_to(&mut app, &path_to(MOVED));
        draw(&mut app, 80, 24)?;
        let sections = app.dataset.sections.clone();
        assert_eq!(
            sections.len(),
            3,
            "question, reasoning, answer: {sections:?}"
        );
        assert_eq!(sections.first(), Some(&0));
        press(&mut app, &[KeyCode::Char(']')]);
        assert_eq!(app.dataset.scroll, sections[1]);
        press(&mut app, &[KeyCode::Char(']')]);
        assert_eq!(app.dataset.scroll, sections[2]);
        press(&mut app, &[KeyCode::Char(']')]);
        assert_eq!(app.dataset.scroll, sections[2], "the last part stays");
        press(&mut app, &[KeyCode::Char('[')]);
        assert_eq!(app.dataset.scroll, sections[1]);
        press(&mut app, &[KeyCode::Char('['), KeyCode::Char('[')]);
        assert_eq!(app.dataset.scroll, 0);
        let rows = text(&draw(&mut app, 80, 24)?);
        assert!(
            rows.iter()
                .any(|row| row.contains(MOVED.get(..20).unwrap_or(MOVED)))
        );
        Ok(())
    }

    #[test]
    fn brackets_do_not_jump_by_the_parts_of_the_node_left_before_a_draw() -> TestResult {
        let mut app = dataset_app();
        let mut data = dataset();
        let reply = data.answers[0].messages.last_mut().ok_or("no reply")?;
        reply.content.push_str(&"\nOne more line.".repeat(30));
        app.dataset.loaded(data, &app.project.topics.clone());
        for key in [KeyCode::Char('j'), KeyCode::Char('k'), KeyCode::Char('h')] {
            open_to(&mut app, &path_to(MOVED));
            draw(&mut app, 80, 24)?;
            assert_eq!(app.dataset.sections.len(), 3);
            press(&mut app, &[key, KeyCode::Char(']')]);
            assert_eq!(app.dataset.scroll, 0, "{key:?}");
        }
        Ok(())
    }

    #[test]
    fn brackets_do_nothing_when_the_detail_fits() -> TestResult {
        let mut app = dataset_app();
        open_to(&mut app, &path_to(MOVED));
        draw(&mut app, 120, 40)?;
        press(&mut app, &[KeyCode::Char(']')]);
        assert_eq!(app.dataset.scroll, 0);
        press(&mut app, &[KeyCode::Char('[')]);
        assert_eq!(app.dataset.scroll, 0);
        open_to(&mut app, &path_to(MOVED)[..2]);
        draw(&mut app, 80, 24)?;
        assert_eq!(
            app.dataset.sections,
            Vec::<u16>::new(),
            "only a question has parts"
        );
        Ok(())
    }

    /// [`dataset_app`] on an answered question, following a run, and leaving: after a
    /// signal, or with a quit waiting for that run.
    fn leaving_app(signal: bool) -> App {
        let mut app = dataset_app();
        open_to(&mut app, &path_to(MOVED));
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

    /// `e`, `E`, `d`, `D` and a confirmed deletion start nothing in `app`, and say why.
    fn no_edit_starts(app: &mut App) -> Result<(), String> {
        for code in ['e', 'E', 'd', 'D'] {
            app.status = None;
            assert_eq!(press(app, &[KeyCode::Char(code)]), []);
            assert_eq!(app.overlay, None);
            assert_eq!(status(app), Some("refused: quitting; no edit starts"));
        }
        let model = app.dataset.model.as_ref().ok_or("no model")?;
        let confirm = deletion(
            model,
            app.dataset.tree.selected(),
            &app.project.topics,
            false,
        )?;
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
        open_to(&mut app, &path_to(MOVED));
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
        open_to(&mut app, &path_to(MOVED));
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
        open_to(&mut app, &path_to(MOVED));
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
        open_to(&mut app, &path_to(MOVED));
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
    fn dialog_text(app: &mut App, path: &[Node], key: char) -> Result<String, String> {
        open_to(app, path);
        press(app, &[KeyCode::Char(key)]);
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
        let subtopic = dialog_text(
            &mut app,
            &[topic.clone(), Node::Subtopic(legacy.clone())],
            'd',
        )?;
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
        let text = dialog_text(&mut app, &path, 'd')?;
        assert_eq!(
            text,
            format!(
                "Delete this question and its answer? It is recorded in data/rejected.jsonl; \
                 {not_configured}"
            )
        );
        let text = dialog_text(&mut app, &path, 'D')?;
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

    /// Stage `stage` of task `id` starts, spends `cost` on one item, and finishes
    /// when `finish`.
    fn spend(app: &mut App, id: TaskId, stage: crate::events::Stage, cost: f64, finish: bool) {
        use crate::events::{Event, StageStats};
        app.on_message(Msg::Event(id, Event::StageStarted { stage, total: 2 }));
        app.on_message(Msg::Event(
            id,
            Event::ItemDone {
                stage,
                id: "x".into(),
                usage: Some(crate::llm::Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                }),
                cost: Some(cost),
            },
        ));
        if finish {
            let stats = StageStats {
                cost: Some(cost),
                ..StageStats::default()
            };
            app.on_message(Msg::Event(id, Event::StageFinished { stage, stats }));
        }
    }

    #[test]
    fn the_footer_keeps_the_finished_stages_of_a_run_until_the_history_has_them() -> TestResult {
        use crate::events::Stage;
        use crate::history::Cost;
        use crate::tui::cost::{project_cost, rows_cost};

        let mut app = dataset_app();
        let load = only_load(&app.start())?;
        app.on_done(
            load,
            Ok(Done::Loaded(
                Ok(dataset()),
                Ok(History::costing(Cost::Known(1.0))),
            )),
        );
        let mut keys = vec![KeyCode::Char('r')];
        keys.extend([KeyCode::Down; 4]);
        keys.push(KeyCode::Enter);
        let effects = press(&mut app, &keys);
        let [Effect::Spawn(id, Task::Pipeline(Command::Run))] = effects.as_slice() else {
            return Err(format!("expected a run: {effects:?}").into());
        };
        let id = *id;
        spend(&mut app, id, Stage::Subtopics, 0.25, true);
        assert_eq!(project_cost(&app), Cost::Known(1.25));
        spend(&mut app, id, Stage::Questions, 0.5, false);
        assert_eq!(project_cost(&app), Cost::Known(1.75));
        let finished = crate::events::Event::StageFinished {
            stage: Stage::Questions,
            stats: crate::events::StageStats {
                cost: Some(0.5),
                ..crate::events::StageStats::default()
            },
        };
        app.on_message(Msg::Event(id, finished));
        assert_eq!(
            project_cost(&app),
            Cost::Known(1.75),
            "a finished stage stays"
        );
        spend(&mut app, id, Stage::Answers, 0.125, false);
        let pipeline = Cost::Known(1.0).plus(match rows_cost(&app.pipeline.rows) {
            Cost::Known(cost) => Some(cost),
            other => return Err(format!("the rows cost is {other:?}").into()),
        });
        assert_eq!(pipeline, Cost::Known(1.875));
        assert_eq!(
            project_cost(&app),
            pipeline,
            "the base plus the Pipeline view"
        );
        // A quiet reload lands with the finished stages in the history: the sum
        // still counts each once.
        let quiet = only_load(&press(&mut app, &[KeyCode::Char('2')]))?;
        app.on_done(
            quiet,
            Ok(Done::Loaded(
                Ok(dataset()),
                Ok(History::costing(Cost::Known(1.75))),
            )),
        );
        assert_eq!(
            project_cost(&app),
            pipeline,
            "a quiet reload changes nothing"
        );
        // Stopped: the stage stays counted until the reload after the end lands.
        let reload = only_load(&app.on_done(id, Ok(Done::Pipeline(Err("interrupted".into())))))?;
        assert_eq!(
            app.pipeline.row(Stage::Answers).state,
            super::super::pipeline::StageState::Stopped
        );
        assert_eq!(project_cost(&app), pipeline, "a stopped stage stays");
        app.on_done(
            reload,
            Ok(Done::Loaded(
                Ok(dataset()),
                Ok(History::costing(Cost::Known(1.875))),
            )),
        );
        assert_eq!(
            project_cost(&app),
            Cost::Known(1.875),
            "the history has them all"
        );
        Ok(())
    }

    #[test]
    fn a_load_started_before_a_stage_sets_the_history_it_adds_to() -> TestResult {
        use crate::events::Stage;
        use crate::history::Cost;
        use crate::tui::cost::project_cost;

        let mut app = dataset_app();
        let load = only_load(&app.start())?;
        let effects = press(
            &mut app,
            &[KeyCode::Char('r'), KeyCode::Down, KeyCode::Enter],
        );
        let [Effect::Spawn(id, Task::Pipeline(_))] = effects.as_slice() else {
            return Err(format!("expected a stage: {effects:?}").into());
        };
        let id = *id;
        spend(&mut app, id, Stage::Questions, 0.5, true);
        app.on_done(
            load,
            Ok(Done::Loaded(
                Ok(dataset()),
                Ok(History::costing(Cost::Known(1.0))),
            )),
        );
        assert_eq!(project_cost(&app), Cost::Known(1.5));
        Ok(())
    }

    #[test]
    fn a_load_older_than_the_last_task_leaves_the_next_base_alone() -> TestResult {
        use crate::events::Stage;
        use crate::history::Cost;
        use crate::tui::cost::project_cost;

        let mut app = dataset_app();
        let load = only_load(&app.start())?;
        app.on_done(
            load,
            Ok(Done::Loaded(
                Ok(dataset()),
                Ok(History::costing(Cost::Known(1.0))),
            )),
        );
        let questions = [KeyCode::Char('r'), KeyCode::Down, KeyCode::Enter];
        let [Effect::Spawn(first, Task::Pipeline(_))] = press(&mut app, &questions)[..] else {
            return Err("expected a stage".into());
        };
        spend(&mut app, first, Stage::Questions, 0.5, true);
        // A quiet load starts during the task, which ends before it lands.
        let quiet = only_load(&press(&mut app, &[KeyCode::Char('2')]))?;
        assert_eq!(app.on_done(first, Ok(Done::Pipeline(Ok(())))), []);
        let [Effect::Spawn(_, Task::Pipeline(_))] = press(&mut app, &questions)[..] else {
            return Err("expected a second stage".into());
        };
        assert_eq!(project_cost(&app), Cost::Known(1.5));
        // Its history predates the first task's entry: the base keeps that task.
        app.on_done(
            quiet,
            Ok(Done::Loaded(
                Ok(dataset()),
                Ok(History::costing(Cost::Known(1.0))),
            )),
        );
        assert_eq!(project_cost(&app), Cost::Known(1.5));
        Ok(())
    }

    #[test]
    fn the_dataset_view_reloads_quietly_while_a_stage_runs() {
        let mut app = dataset_app();
        let effects = press(
            &mut app,
            &[KeyCode::Char('r'), KeyCode::Down, KeyCode::Enter],
        );
        let [Effect::Spawn(id, Task::Pipeline(_))] = effects.as_slice() else {
            return assert_eq!(effects, []);
        };
        let pipeline = *id;
        assert_eq!(
            app.on_tick(at(NOW + 10)),
            [],
            "not shown on the Pipeline view"
        );
        let effects = press(&mut app, &[KeyCode::Char('2')]);
        let [Effect::Spawn(load, Task::Load)] = effects.as_slice() else {
            return assert_eq!(effects, []);
        };
        let load = *load;
        assert!(!app.work().contains(&"loading".to_string()), "quiet");
        assert_eq!(app.on_tick(at(NOW + 11)), [], "one load at a time");
        app.on_done(
            load,
            Ok(Done::Loaded(Ok(dataset()), Ok(History::default()))),
        );
        assert_eq!(app.on_tick(at(NOW + 12)), [], "ended less than 2 s ago");
        let effects = app.on_tick(at(NOW + 13));
        assert!(matches!(effects.as_slice(), [Effect::Spawn(_, Task::Load)]));
        let effects = app.on_done(pipeline, Ok(Done::Pipeline(Ok(()))));
        assert_eq!(effects, [], "the end's reload waits for the running load");
        assert!(
            app.work().contains(&"loading".to_string()),
            "a reload asked for shows, even behind a quiet one"
        );
        assert_eq!(app.on_tick(at(NOW + 30)), [], "no stage runs");
    }

    #[test]
    fn the_dataset_view_reloads_at_once_only_when_it_is_entered() {
        let mut app = dataset_app();
        let effects = press(
            &mut app,
            &[KeyCode::Char('r'), KeyCode::Down, KeyCode::Enter],
        );
        assert!(matches!(
            effects.as_slice(),
            [Effect::Spawn(_, Task::Pipeline(_))]
        ));
        let effects = press(&mut app, &[KeyCode::Char('2')]);
        let [Effect::Spawn(load, Task::Load)] = effects.as_slice() else {
            return assert_eq!(effects, []);
        };
        let load = *load;
        app.on_tick(at(NOW + 1));
        app.on_done(
            load,
            Ok(Done::Loaded(Ok(dataset()), Ok(History::default()))),
        );
        assert_eq!(
            press(&mut app, &[KeyCode::Char('2')]),
            [],
            "already shown: the tick paces the reloads"
        );
    }

    #[test]
    fn a_failed_quiet_load_waits_before_the_next_one() {
        let mut app = dataset_app();
        let effects = press(
            &mut app,
            &[KeyCode::Char('r'), KeyCode::Down, KeyCode::Enter],
        );
        assert!(matches!(
            effects.as_slice(),
            [Effect::Spawn(_, Task::Pipeline(_))]
        ));
        let effects = press(&mut app, &[KeyCode::Char('2')]);
        let [Effect::Spawn(load, Task::Load)] = effects.as_slice() else {
            return assert_eq!(effects, []);
        };
        let load = *load;
        assert_eq!(app.on_tick(at(NOW + 10)), [], "the load still runs");
        app.on_done(load, Err("boom".into()));
        assert_eq!(
            app.on_tick(at(NOW + 11)),
            [],
            "failed less than 2 s ago, though it started long before"
        );
        let effects = app.on_tick(at(NOW + 12));
        assert!(matches!(effects.as_slice(), [Effect::Spawn(_, Task::Load)]));
    }

    #[test]
    fn the_dataset_view_does_not_reload_on_its_own_when_no_stage_runs() {
        let mut app = dataset_app();
        assert_eq!(app.on_tick(at(NOW + 10)), []);
        assert_eq!(
            press(&mut app, &[KeyCode::Char('3'), KeyCode::Char('2')]),
            []
        );
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

    #[test]
    fn a_save_in_flight_is_waited_for() {
        let mut app = app();
        app.project_view.save = Some(TaskId(12));
        assert_eq!(
            app.waiting_for(),
            ["waiting for overbrainer.toml to be saved..."]
        );
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
        keys(&mut app, &[KeyCode::Char('4')]);
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
        keys(&mut app, &[KeyCode::Char('2')]);
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
    fn t_prepares_asks_with_the_gpu_catalog_then_starts_one_run() -> Result<(), String> {
        let mut app = app();
        press(&mut app, &[KeyCode::Char('4')]);
        let effects = press(&mut app, &[KeyCode::Char('t')]);
        let [Effect::Spawn(prepare, Task::Prepare)] = effects.as_slice() else {
            return Err(format!("{effects:?}"));
        };
        let effects = app.on_done(
            *prepare,
            Ok(Done::Prepared(Ok(crate::tui::snapshots::runpod_plan()))),
        );
        let [Effect::Spawn(catalog, Task::StartCatalog(1))] = effects.as_slice() else {
            return Err(format!("{effects:?}"));
        };
        let listed = crate::tui::snapshots::gpu_types().map_err(|error| error.to_string())?;
        app.on_done(*catalog, Ok(Done::StartCatalog(Ok(listed))));
        let Some(Overlay::Confirm(confirm)) = &app.overlay else {
            return Err("no dialog".into());
        };
        assert!(
            confirm
                .text
                .iter()
                .any(|line| line.ends_with("$0.40/h        48 GB  HIGH")),
            "{:?}",
            confirm.text
        );
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
        press(&mut app, &[KeyCode::Char('4')]);
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
    fn stale_or_failed_catalog_lookups_never_leave_the_dialog_waiting() -> Result<(), String> {
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
        let listed = crate::tui::snapshots::gpu_types().map_err(|error| error.to_string())?;
        app.on_done(*old, Ok(Done::StartCatalog(Ok(listed))));
        assert!(dialog(&app).contains("looking up the catalog..."), "stale");
        app.on_done(*new, Err("a background task failed: cancelled".into()));
        assert!(!dialog(&app).contains("looking up"), "{}", dialog(&app));
        assert!(dialog(&app).contains("NVIDIA A40               catalog unread"));
        assert!(dialog(&app).contains("catalog     a background task failed: cancelled"));
        assert_eq!(app.start_catalog, None);
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
        let effects = press(&mut app, &[KeyCode::Char('4'), KeyCode::Char('t')]);
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
                app.dataset.input = Some(Input::new("bor"));
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
        press(&mut app, &[KeyCode::Char('4'), KeyCode::Char('t')]);
        assert_eq!(press(&mut app, &[KeyCode::Char('t')]), []);
        assert_eq!(status(&app), Some("already preparing a run"));
    }

    #[test]
    fn closing_the_start_dialog_forgets_its_catalog_lookup() -> Result<(), String> {
        for code in [KeyCode::Char('n'), KeyCode::Char('y'), KeyCode::Esc] {
            let mut app = app();
            let effects = app.prepared(Ok(crate::tui::snapshots::runpod_plan()));
            let [Effect::Spawn(lookup, Task::StartCatalog(_))] = effects.as_slice() else {
                return Err(format!("{effects:?}"));
            };
            assert!(app.work().contains(&"preparing a run".to_string()));
            press(&mut app, &[code]);
            assert_eq!(app.start_catalog, None, "{code:?}");
            assert!(!app.work().contains(&"preparing a run".to_string()));
            app.on_done(*lookup, Ok(Done::StartCatalog(Ok(Vec::new()))));
        }
        let mut app = app();
        app.prepared(Ok(crate::tui::snapshots::runpod_plan()));
        app.on_input(&ctrl_c());
        assert_eq!(app.start_catalog, None, "Ctrl-C");
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
        let effects = press(&mut app, &[KeyCode::Char('4'), KeyCode::Char('t')]);
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

    /// The Runpod target field `field` of the Project view's configuration.
    fn gpu_cloud(field: &'static str) -> FieldPath {
        FieldPath::Target {
            name: "gpu_cloud".into(),
            field,
        }
    }

    /// The GPU picker of the Project view's `gpu_types` open on the fixture
    /// catalog, `A40` then `4090` chosen.
    fn gpu_picker() -> Result<(App, TaskId), Box<dyn std::error::Error>> {
        let mut app = crate::tui::snapshots::project_app()?;
        let chosen = Choice::List(vec!["NVIDIA A40".into(), "NVIDIA GeForce RTX 4090".into()]);
        let query = Query {
            kind: CatalogKind::Gpus,
            gpu_count: 2,
            gpu_types: Vec::new(),
        };
        let effects = app.open_picker(query.clone(), chosen, Origin::Field(gpu_cloud("gpu_types")));
        let [Effect::Spawn(id, Task::Catalog(asked))] = effects.as_slice() else {
            return Err(format!("{effects:?}").into());
        };
        assert_eq!(asked, &query);
        let id = *id;
        Ok((app, id))
    }

    /// The fixture GPU catalog, listed for 2 GPUs.
    fn gpus_listed() -> Result<Listed, Box<dyn std::error::Error>> {
        Ok(Listed {
            entries: crate::tui::snapshots::gpu_catalog(2)?,
            gpus: crate::tui::snapshots::gpu_types()?,
        })
    }

    /// The `gpu_types` the Project view shows.
    fn gpu_types_shown(app: &App) -> Option<String> {
        app.project_view
            .pending
            .as_ref()
            .and_then(|pending| pending.doc.get(&gpu_cloud("gpu_types")))
    }

    fn picker_of(app: &App) -> Option<&Picker> {
        match &app.overlay {
            Some(Overlay::Picker(picking)) => Some(&picking.picker),
            _ => None,
        }
    }

    #[test]
    fn a_picker_reads_its_catalog_then_enter_keeps_the_choice() -> TestResult {
        let (mut app, id) = gpu_picker()?;
        assert!(picker_of(&app).is_some_and(Picker::loading));
        assert_eq!(app.work(), ["reading the catalog"]);
        app.on_done(TaskId(id.0 + 100), Ok(Done::Catalog(Ok(gpus_listed()?))));
        assert!(
            picker_of(&app).is_some_and(Picker::loading),
            "another listing is ignored"
        );
        assert_eq!(app.gpu_catalog, None, "nor kept");
        app.on_done(id, Ok(Done::Catalog(Ok(gpus_listed()?))));
        assert!(app.work().is_empty());
        assert!(app.gpu_catalog.is_some(), "kept for the hints");
        press(&mut app, &[KeyCode::Char('J'), KeyCode::Enter]);
        assert_eq!(app.overlay, None);
        assert_eq!(
            gpu_types_shown(&app).as_deref(),
            Some("NVIDIA GeForce RTX 4090, NVIDIA A40"),
            "back to the field that opened it"
        );
        Ok(())
    }

    #[test]
    fn esc_or_ctrl_c_closes_a_picker_and_keeps_nothing() -> TestResult {
        let (mut app, id) = gpu_picker()?;
        press(&mut app, &[KeyCode::Esc]);
        assert_eq!(app.overlay, None);
        assert!(app.project_view.pending.is_none());
        assert_eq!(app.on_done(id, Ok(Done::Catalog(Ok(gpus_listed()?)))), []);
        assert_eq!(app.overlay, None, "a late listing opens nothing");
        let (mut app, _) = gpu_picker()?;
        app.on_input(&ctrl_c());
        assert_eq!(app.overlay, None);
        assert!(app.project_view.pending.is_none());
        Ok(())
    }

    #[test]
    fn a_listing_failing_after_its_picker_closed_says_nothing() -> TestResult {
        let (mut app, id) = gpu_picker()?;
        press(&mut app, &[KeyCode::Esc]);
        app.status = None;
        assert_eq!(
            app.on_done(id, Err("a background task failed: boom".into())),
            []
        );
        assert_eq!(app.status, None);
        assert_eq!(app.overlay, None);
        let (mut app, id) = gpu_picker()?;
        press(&mut app, &[KeyCode::Esc]);
        app.status = None;
        app.on_done(id, Ok(Done::Catalog(Err("cannot read".into()))));
        assert_eq!(app.status, None);
        Ok(())
    }

    #[test]
    fn a_failed_listing_shows_in_the_picker() -> TestResult {
        let (mut app, id) = gpu_picker()?;
        app.on_done(id, Err("a background task failed: boom".into()));
        let picker = picker_of(&app).ok_or("the picker closed")?;
        assert_eq!(picker.error(), Some("a background task failed: boom"));
        let (mut app, id) = gpu_picker()?;
        let refused = "cannot read the Runpod catalog: no Runpod API key".to_string();
        app.on_done(id, Ok(Done::Catalog(Err(refused.clone()))));
        let picker = picker_of(&app).ok_or("the picker closed")?;
        assert_eq!(picker.error(), Some(refused.as_str()));
        press(&mut app, &[KeyCode::Enter]);
        assert!(picker_of(&app).is_some(), "Enter keeps nothing");
        press(&mut app, &[KeyCode::Esc]);
        assert_eq!(app.overlay, None);
        Ok(())
    }

    #[test]
    fn keys_go_to_the_picker_not_the_view() -> TestResult {
        let (mut app, id) = gpu_picker()?;
        app.on_done(id, Ok(Done::Catalog(Ok(gpus_listed()?))));
        app.on_input(&Event::Paste("A40".into()));
        assert!(
            !picker_of(&app).is_some_and(Picker::typing),
            "no filter typed"
        );
        press(&mut app, &[KeyCode::Char('/')]);
        app.on_input(&Event::Paste("A40\n".into()));
        press(
            &mut app,
            &[KeyCode::Enter, KeyCode::Char(' '), KeyCode::Enter],
        );
        // A40, found through the pasted filter, is taken out.
        assert_eq!(
            gpu_types_shown(&app).as_deref(),
            Some("NVIDIA GeForce RTX 4090")
        );
        let (mut app, id) = gpu_picker()?;
        app.on_done(id, Ok(Done::Catalog(Ok(gpus_listed()?))));
        let view = app.view;
        let effects = press(
            &mut app,
            &[KeyCode::Char('q'), KeyCode::Char('2'), KeyCode::Char('r')],
        );
        assert_eq!(effects, []);
        assert_eq!(app.view, view);
        assert_eq!(app.exit, None);
        assert!(picker_of(&app).is_some());
        Ok(())
    }
    #[test]
    fn an_older_volume_listing_finishing_last_is_ignored() -> Result<(), String> {
        let mut app = app();
        let query = Query {
            kind: CatalogKind::Volumes,
            gpu_count: 1,
            gpu_types: Vec::new(),
        };
        let origin = Origin::Field(crate::config::edit::FieldPath::Target {
            name: "gpu_cloud".into(),
            field: "network_volume_id",
        });
        let spawned = |effects: Vec<Effect>| match effects.as_slice() {
            [Effect::Spawn(id, _)] => Ok(*id),
            _ => Err(format!("{effects:?}")),
        };
        let older =
            spawned(app.open_picker(query.clone(), Choice::List(Vec::new()), origin.clone()))?;
        app.overlay = None;
        let newer = spawned(app.open_picker(query, Choice::List(Vec::new()), origin))?;
        let volumes = |center: &str| Listed {
            entries: crate::tui::catalog::volume_entries(&[crate::runpod::NetworkVolume {
                id: "vol1".into(),
                name: "data".into(),
                size: 10,
                data_center: center.into(),
            }]),
            gpus: Vec::new(),
        };
        app.on_done(newer, Ok(Done::Catalog(Ok(volumes("EU-RO-1")))));
        app.on_done(older, Ok(Done::Catalog(Ok(volumes("US-KS-2")))));
        let kept = app
            .volume_catalog
            .as_ref()
            .and_then(|entries| entries.iter().find(|entry| entry.id == "vol1"))
            .map(|entry| entry.columns[3].clone());
        assert_eq!(kept.as_deref(), Some("EU-RO-1"));
        Ok(())
    }
}
