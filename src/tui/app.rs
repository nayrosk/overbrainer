//! The state of the TUI and what keys and events do to it. Nothing here draws,
//! reads the clock or touches the terminal: the loop feeds it input, ticks and
//! signals, and renders it.

use std::time::{Duration, SystemTime};

use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use tracing::Level;

use super::theme::Theme;
use crate::config::Settings;
use crate::logging::LogBuffer;

/// How long a status message stays on the status line.
const STATUS_FOR: Duration = Duration::from_secs(10);
/// Lines moved by `PgUp` and `PgDn`.
pub(super) const PAGE: usize = 10;

/// What the TUI knows of the project, read from `overbrainer.toml` when it starts.
#[derive(Debug, Clone)]
pub(super) struct Project {
    /// `project.name`.
    pub(super) name: String,
}

impl Project {
    /// The project configured by `settings`.
    pub(super) fn new(settings: &Settings) -> Self {
        Self {
            name: settings.project.name.clone(),
        }
    }
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
    /// The Logs view's state.
    pub(super) log_view: LogView,
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

    /// Handles one terminal event.
    pub(super) fn on_input(&mut self, event: &Event) {
        match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => self.on_key(*key),
            Event::Resize(..) => self.dirty = true,
            _ => {},
        }
    }

    fn on_key(&mut self, key: KeyEvent) {
        self.dirty = true;
        let ctrl_c =
            key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c');
        if ctrl_c || key.code == KeyCode::Char('q') {
            self.quit();
            return;
        }
        if self.overlay == Some(Overlay::Help) {
            if matches!(key.code, KeyCode::Esc | KeyCode::Char('?')) {
                self.overlay = None;
            }
            return;
        }
        match key.code {
            KeyCode::Char('?') => self.overlay = Some(Overlay::Help),
            KeyCode::Char('1') => self.view = View::Dataset,
            KeyCode::Char('2') => self.view = View::Pipeline,
            KeyCode::Char('3') => self.view = View::Training,
            KeyCode::Char('4') => self.view = View::Logs,
            KeyCode::Tab => self.view = self.view.shifted(1),
            KeyCode::BackTab => self.view = self.view.shifted(View::ALL.len() - 1),
            code => self.on_view_key(code),
        }
    }

    fn on_view_key(&mut self, code: KeyCode) {
        if self.view == View::Logs {
            self.on_logs_key(code);
        }
    }

    fn on_logs_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Up | KeyCode::Char('k') => self.scroll_logs(true, 1),
            KeyCode::Down | KeyCode::Char('j') => self.scroll_logs(false, 1),
            KeyCode::PageUp => self.scroll_logs(true, PAGE),
            KeyCode::PageDown => self.scroll_logs(false, PAGE),
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
    use crate::logging::LogLine;
    use crate::tui::snapshots::{NOW, app, at, ctrl_c, draw, key, text};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn press(app: &mut App, codes: &[KeyCode]) {
        for code in codes {
            app.on_input(&key(*code));
        }
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
}
