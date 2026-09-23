//! Snapshot tests of every view on `TestBackend`, at 80x24 and 120x40, and the
//! monochrome theme. Fixtures build the app from fixed records and a fixed clock,
//! so no runtime, file or network is needed.

use std::convert::Infallible;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::style::{Color, Modifier};
use tracing::Level;

use super::app::{App, Overlay, Project, View};
use super::theme::Theme;
use super::ui;
use crate::logging::{LogBuffer, LogLine};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Where the snapshot files are kept: outside `src/`, so the published crate does
/// not ship them.
const SNAPSHOTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/snapshots/tui");

/// The fixtures' clock, 2026-09-21 14:13:20 UTC.
pub(super) const NOW: u64 = 1_790_000_000;

/// `seconds` after the Unix epoch.
pub(super) fn at(seconds: u64) -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(seconds)
}

/// An app on the project `rust_expert` at [`NOW`], with the color theme.
pub(super) fn app() -> App {
    let project = Project {
        name: "rust_expert".into(),
    };
    App::new(project, LogBuffer::new(100), Theme::color(), at(NOW))
}

/// A key press.
pub(super) fn key(code: KeyCode) -> Event {
    Event::Key(KeyEvent {
        code,
        modifiers: KeyModifiers::NONE,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    })
}

/// Ctrl-C, a key in raw mode.
pub(super) fn ctrl_c() -> Event {
    Event::Key(KeyEvent {
        code: KeyCode::Char('c'),
        modifiers: KeyModifiers::CONTROL,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    })
}

/// `app` drawn on a `width` by `height` test terminal.
pub(super) fn draw(
    app: &mut App,
    width: u16,
    height: u16,
) -> Result<Terminal<TestBackend>, Infallible> {
    let mut terminal = Terminal::new(TestBackend::new(width, height))?;
    terminal.draw(|frame| ui::render(frame, app))?;
    Ok(terminal)
}

/// Checks `app` drawn at `width` by `height` against the snapshot `name`.
fn snapshot_at(name: &str, app: &mut App, width: u16, height: u16) -> Result<(), Infallible> {
    let terminal = draw(app, width, height)?;
    let mut settings = insta::Settings::clone_current();
    settings.set_snapshot_path(SNAPSHOTS);
    settings.set_prepend_module_to_snapshot(false);
    settings.set_omit_expression(true);
    settings.bind(|| insta::assert_snapshot!(name.to_string(), terminal.backend()));
    Ok(())
}

/// Checks `app` against the snapshots `<name>_80x24` and `<name>_120x40`.
pub(super) fn snapshot(name: &str, app: &mut App) -> Result<(), Infallible> {
    snapshot_at(&format!("{name}_80x24"), app, 80, 24)?;
    snapshot_at(&format!("{name}_120x40"), app, 120, 40)
}

fn log(buffer: &LogBuffer, second: u64, level: Level, target: &str, message: &str) {
    buffer.push(LogLine {
        seq: 0,
        level,
        target: target.into(),
        time: at(NOW - 60 + second),
        message: message.into(),
    });
}

/// Log lines of every level.
fn logs(app: &App) {
    let buffer = &app.logs;
    log(
        buffer,
        1,
        Level::INFO,
        "overbrainer::cli::progress",
        "answers: 400 to process",
    );
    log(
        buffer,
        2,
        Level::DEBUG,
        "overbrainer::cli::progress",
        "answers: 3f1c done",
    );
    log(
        buffer,
        3,
        Level::WARN,
        "overbrainer::cli::progress",
        "answers: 77aa: 429 Too Many Requests; retrying in 4.0s",
    );
    log(
        buffer,
        4,
        Level::ERROR,
        "overbrainer::cli::progress",
        "answers: 77aa failed: the answer is not a JSON array of strings",
    );
    log(buffer, 5, Level::TRACE, "overbrainer::llm", "request sent");
    log(
        buffer,
        6,
        Level::INFO,
        "overbrainer::cli::progress",
        "answers: 40/400",
    );
}

#[test]
fn logs_of_mixed_levels() -> TestResult {
    let mut app = app();
    logs(&app);
    app.view = View::Logs;
    snapshot("logs", &mut app)?;
    Ok(())
}

#[test]
fn logs_filtered_at_warn() -> TestResult {
    let mut app = app();
    logs(&app);
    app.view = View::Logs;
    app.log_view.min = Level::WARN;
    snapshot("logs_warn", &mut app)?;
    Ok(())
}

#[test]
fn logs_at_trace_with_nothing_captured() -> TestResult {
    let mut app = app();
    app.view = View::Logs;
    app.log_view.min = Level::TRACE;
    snapshot("logs_empty_trace", &mut app)?;
    Ok(())
}

#[test]
fn the_help_overlay_lists_global_and_view_keys() -> TestResult {
    let mut app = app();
    app.view = View::Logs;
    app.overlay = Some(Overlay::Help);
    snapshot("help_logs", &mut app)?;
    Ok(())
}

#[test]
fn a_terminal_below_80x24_says_so() -> TestResult {
    let mut app = app();
    snapshot_at("too_small_79x24", &mut app, 79, 24)?;
    snapshot_at("too_small_80x23", &mut app, 80, 23)?;
    Ok(())
}

/// Every view drawn with the monochrome theme uses no color, and the selection
/// shows as reversed.
#[test]
fn the_monochrome_theme_uses_no_color() -> TestResult {
    let mut app = app();
    logs(&app);
    app.theme = Theme::mono();
    for view in View::ALL {
        app.view = view;
        for overlay in [None, Some(Overlay::Help)] {
            app.overlay = overlay.clone();
            let terminal = draw(&mut app, 120, 40)?;
            let buffer = terminal.backend().buffer();
            let colored = buffer
                .content()
                .iter()
                .filter(|cell| cell.fg != Color::Reset || cell.bg != Color::Reset)
                .count();
            assert_eq!(colored, 0, "{view:?}, {overlay:?}");
            assert!(
                buffer
                    .content()
                    .iter()
                    .any(|cell| cell.modifier.contains(Modifier::REVERSED)),
                "{view:?}: no reversed selection"
            );
        }
    }
    Ok(())
}
