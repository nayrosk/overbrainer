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
use super::tasks::{Done, Edit, Saved, Task, TaskId};
use super::theme::Theme;
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
        }
    }
}

/// What the app asks the loop to do.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Effect {
    /// Start a background task.
    Spawn(TaskId, Task),
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Overlay {
    /// The key table.
    Help,
    /// A question answered with `y` or `n`.
    Confirm(Confirm),
}

/// A confirmation dialog: `y` runs its action, anything else closes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Confirm {
    /// The title.
    pub(super) title: String,
    /// The question, one paragraph per entry.
    pub(super) text: Vec<String>,
    /// What `y` does, in one word.
    pub(super) yes: &'static str,
    /// What `y` runs.
    pub(super) action: Action,
}

/// What a confirmed dialog runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Action {
    /// A deletion, which must still remove `counts`.
    Delete {
        /// What is deleted.
        deletion: Deletion,
        /// What the dialog said it removes.
        counts: Counts,
    },
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
    /// Whether a reload was asked for while a load ran: it starts once that load
    /// ends, so it reads what changed meanwhile.
    reload_pending: bool,
    /// The edit being saved, if any.
    pub(super) edit: Option<TaskId>,
    /// The edit open in the editor or being saved.
    pub(super) editing: Option<Session>,
    /// The editor command.
    pub(super) editor: Vec<String>,
    /// Lines printed on stderr once the terminal is restored.
    pub(super) exit_notes: Vec<String>,
    /// How the TUI ends once the edit being saved is saved: `q` or a signal
    /// came while it was saved.
    pub(super) quitting: Option<Exit>,
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
    pub(super) fn new(project: Project, logs: LogBuffer, theme: Theme, now: SystemTime) -> Self {
        Self {
            project,
            theme,
            seen_log: logs.seq(),
            logs,
            view: View::Dataset,
            overlay: None,
            status: None,
            now,
            dataset: DatasetView::default(),
            load: None,
            reload_pending: false,
            edit: None,
            editing: None,
            editor: vec!["vi".to_string()],
            exit_notes: Vec::new(),
            quitting: None,
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
    fn task_id(&mut self) -> TaskId {
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
        vec![Effect::Spawn(id, Task::Load)]
    }

    /// The work running, as the status line shows it.
    pub(super) fn work(&self) -> Vec<String> {
        let mut work = Vec::new();
        if self.load.is_some() {
            work.push("loading".to_string());
        }
        if self.edit.is_some() {
            work.push("saving".to_string());
        }
        if self.lock().is_some() {
            work.push("edits locked".to_string());
        }
        work
    }

    /// Why the dataset cannot be changed now, if it cannot.
    pub(super) fn lock(&self) -> Option<&'static str> {
        if self.edit.is_some() || self.editing.is_some() {
            return Some("an edit is being saved");
        }
        None
    }

    /// Refuses a change to the dataset while it is locked.
    fn locked(&mut self) -> bool {
        let Some(reason) = self.lock() else {
            return false;
        };
        self.say(
            Severity::Warn,
            format!("refused: {reason}; edits resume when it ends"),
        );
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

    /// Task `id` failed (a panic): an edit keeps its typed text, a load shows why.
    fn failed(&mut self, id: TaskId, error: String) -> Vec<Effect> {
        tracing::error!("{error}");
        if self.edit == Some(id) {
            return self.saved(Err(error));
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
        if let Some(exit) = self.quitting.take() {
            self.exit = Some(exit);
            return Vec::new();
        }
        self.reload()
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
        if self.quitting.is_some() {
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
        if self.quitting.is_some() {
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
            self.quit();
            return Vec::new();
        }
        if self.view == View::Dataset && self.dataset.input.is_some() {
            if key.modifiers.difference(KeyModifiers::SHIFT).is_empty() {
                self.on_filter_key(key.code);
            }
            return Vec::new();
        }
        match &self.overlay {
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
            self.quit();
            return Vec::new();
        }
        match key.code {
            KeyCode::Char('R') => return self.reload(),
            KeyCode::Char('?') => self.overlay = Some(Overlay::Help),
            KeyCode::Char('1') => self.view = View::Dataset,
            KeyCode::Char('2') => self.view = View::Pipeline,
            KeyCode::Char('3') => self.view = View::Training,
            KeyCode::Char('4') => self.view = View::Logs,
            KeyCode::Tab => self.view = self.view.shifted(1),
            KeyCode::BackTab => self.view = self.view.shifted(View::ALL.len() - 1),
            code => return self.on_view_key(code),
        }
        Vec::new()
    }

    /// `y` runs the dialog's action; any other key closes it.
    fn on_confirm_key(&mut self, code: KeyCode) -> Vec<Effect> {
        let Some(Overlay::Confirm(confirm)) = self.overlay.take() else {
            return Vec::new();
        };
        if code != KeyCode::Char('y') {
            return Vec::new();
        }
        match confirm.action {
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
            View::Logs => self.on_logs_key(code),
            View::Pipeline | View::Training => {},
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

    /// `q` or Ctrl-C: quits, once an edit being saved is saved.
    fn quit(&mut self) {
        if self.edit.is_some() {
            if self.quitting.is_none() {
                self.quitting = Some(Exit::Quit);
            }
            self.say(Severity::Info, "quitting once the edit is saved");
            return;
        }
        self.exit = Some(Exit::Quit);
    }

    /// SIGINT, SIGTERM or SIGHUP from outside: quits without asking, once an
    /// edit being saved is saved (it is never cut, design 3.6).
    pub(super) fn on_signal(&mut self) {
        if self.edit.is_some() {
            self.quitting = Some(Exit::Signal);
            self.say(
                Severity::Warn,
                "interrupted: exiting once the edit is saved",
            );
            return;
        }
        self.exit = Some(Exit::Signal);
    }

    /// Moves the clock to `now`: expires the status message and shows new log
    /// lines, and the newest warning or error on the status line.
    pub(super) fn on_tick(&mut self, now: SystemTime) {
        self.now = now;
        if self.status.as_ref().is_some_and(|status| {
            now.duration_since(status.at)
                .is_ok_and(|shown| shown >= STATUS_FOR)
        }) {
            self.status = None;
            self.dirty = true;
        }
        let seq = self.logs.seq();
        if seq == self.seen_log {
            return;
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

    /// The title row and the rows of lines of the Logs view drawn at 80x24.
    fn logs_view(app: &mut App) -> Result<(String, Vec<String>), Infallible> {
        let rows = text(&draw(app, 80, 24)?);
        let title = rows.get(1).cloned().unwrap_or_default();
        Ok((title, rows.into_iter().skip(2).take(ROWS).collect()))
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
            },
            LogBuffer::new(25),
            Theme::color(),
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
        only_load(&press(&mut app, &[KeyCode::Char('R')]))?;
        Ok(())
    }

    #[test]
    fn a_reload_asked_for_during_a_load_starts_once_it_ends() -> TestResult {
        let mut app = app();
        let first = only_load(&app.start())?;
        assert_eq!(
            press(&mut app, &[KeyCode::Char('R'), KeyCode::Char('R')]),
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
        assert_eq!(app.lock(), Some("an edit is being saved"));
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
}
