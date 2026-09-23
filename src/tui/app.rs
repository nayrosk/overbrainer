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

/// The Logs view's state: the level shown and how far back it is scrolled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LogView {
    /// The least severe level shown.
    pub(super) min: Level,
    /// Matching lines between the newest one and the bottom of the view; 0
    /// follows the tail.
    pub(super) offset: usize,
}

impl LogView {
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
                offset: 0,
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
        let matching = self.logs.window(self.log_view.min, 0, 0).matching;
        let last = matching.saturating_sub(1);
        let view = &mut self.log_view;
        match code {
            KeyCode::Up | KeyCode::Char('k') => view.offset = (view.offset + 1).min(last),
            KeyCode::Down | KeyCode::Char('j') => view.offset = view.offset.saturating_sub(1),
            KeyCode::PageUp => view.offset = (view.offset + PAGE).min(last),
            KeyCode::PageDown => view.offset = view.offset.saturating_sub(PAGE),
            KeyCode::End | KeyCode::Char('G') => view.offset = 0,
            KeyCode::Char('f') => {
                view.min = LogView::next_level(view.min);
                view.offset = 0;
            },
            _ => {},
        }
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
    use super::*;
    use crate::logging::LogLine;
    use crate::tui::snapshots::{NOW, app, at, ctrl_c, key};

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

    #[test]
    fn the_logs_view_scrolls_follows_and_cycles_its_level() {
        let mut app = app();
        for n in 0..30 {
            log(&app, Level::INFO, &format!("line {n}"));
        }
        press(
            &mut app,
            &[KeyCode::Char('4'), KeyCode::Char('k'), KeyCode::Up],
        );
        assert_eq!(app.log_view.offset, 2);
        press(
            &mut app,
            &[KeyCode::PageUp, KeyCode::PageUp, KeyCode::PageUp],
        );
        assert_eq!(app.log_view.offset, 29, "clamped to the oldest line");
        press(&mut app, &[KeyCode::Char('j'), KeyCode::PageDown]);
        assert_eq!(app.log_view.offset, 18);
        press(&mut app, &[KeyCode::Char('G')]);
        assert_eq!(app.log_view.offset, 0);
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
