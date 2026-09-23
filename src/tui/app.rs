//! The state of the TUI and what keys and events do to it. Nothing here draws,
//! reads the clock or touches the terminal: the loop feeds it input, ticks and
//! signals, and renders it.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use tracing::Level;

use super::dataset::{DatasetView, TopicInfo};
use super::tasks::{Done, Task, TaskId};
use super::theme::Theme;
use crate::config::Settings;
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
        work
    }

    /// Handles the end of task `id`: what it gave back, or how it failed.
    /// A load's result is used only when `id` is the load running; a reload asked
    /// for meanwhile starts then.
    pub(super) fn on_done(&mut self, id: TaskId, result: Result<Done, String>) -> Vec<Effect> {
        self.dirty = true;
        let is_load = self.load == Some(id);
        match result {
            Ok(Done::Loaded(_)) if !is_load => return Vec::new(),
            Ok(Done::Loaded(loaded)) => {
                self.load = None;
                match loaded {
                    Ok(data) => self.dataset.loaded(data, &self.project.topics),
                    Err(error) => self.load_failed(error),
                }
            },
            Err(error) => {
                tracing::error!("{error}");
                if is_load {
                    self.load = None;
                    self.load_failed(error);
                } else {
                    self.say(Severity::Error, error);
                }
            },
        }
        if self.load.is_none() && std::mem::take(&mut self.reload_pending) {
            return self.reload();
        }
        Vec::new()
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
        if key.code == KeyCode::Char('q') {
            self.quit();
            return Vec::new();
        }
        if self.overlay == Some(Overlay::Help) {
            if matches!(key.code, KeyCode::Esc | KeyCode::Char('?')) {
                self.overlay = None;
            }
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
            code => self.on_view_key(code),
        }
        Vec::new()
    }

    fn on_view_key(&mut self, code: KeyCode) {
        match self.view {
            View::Dataset => self.on_dataset_key(code),
            View::Logs => self.on_logs_key(code),
            View::Pipeline | View::Training => {},
        }
    }

    fn on_dataset_key(&mut self, code: KeyCode) {
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

    /// `q` or Ctrl-C: quits.
    fn quit(&mut self) {
        self.exit = Some(Exit::Quit);
    }

    /// SIGINT, SIGTERM or SIGHUP from outside: quits without asking.
    pub(super) fn on_signal(&mut self) {
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

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use super::*;
    use crate::dataset::Id;
    use crate::logging::LogLine;
    use crate::tui::dataset::Node;
    use crate::tui::snapshots::{
        MOVED, NOW, app, at, ctrl_c, dataset, dataset_app, draw, key, open_to, path_to, text,
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
}
