//! The footer, on the last row: the key hints of what the keys act on, or the
//! latest status message while it lasts, on the left; the work running, each
//! with its spinner, `locked` while the data lock holds, and `? help` on the
//! right.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::tui::app::{App, Overlay, Severity, Status, View};
use crate::tui::keys::{self, Context, HELP_HINT, Hint, SEPARATOR};
use crate::tui::theme::Theme;

/// Draws the footer in `area`.
pub(in crate::tui) fn render(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let locked = app.lock().is_some();
    let spinner = app.motion.spinner();
    let mut right = Vec::new();
    for work in app.work() {
        right.push(Span::styled(format!("{spinner} "), theme.accent));
        right.push(Span::styled(work, theme.dim));
        right.push(Span::styled(SEPARATOR, theme.dim));
    }
    if locked {
        right.push(Span::styled("locked", theme.dim));
        right.push(Span::styled(SEPARATOR, theme.dim));
    }
    let (key, label) = HELP_HINT.split_at(1);
    right.push(Span::styled(key, theme.key));
    right.push(Span::styled(format!("{label} "), theme.dim));
    let right = Line::from(right).right_aligned();
    let width = u16::try_from(right.width()).unwrap_or(area.width);
    let [left_area, right_area] =
        Layout::horizontal([Constraint::Fill(1), Constraint::Length(width)]).areas(area);
    let left = match &app.status {
        Some(status) => status_line(status, theme),
        None => hints_line(&keys::footer(context(app)), locked, left_area.width, theme),
    };
    frame.render_widget(Paragraph::new(left), left_area);
    frame.render_widget(Paragraph::new(right), right_area);
}

/// What the keys act on now.
pub(in crate::tui) fn context(app: &App) -> Context {
    match &app.overlay {
        Some(Overlay::Confirm(confirm)) => Context::Dialog {
            yes: confirm.yes,
            no: confirm.no,
        },
        Some(Overlay::Help) => Context::Help,
        Some(Overlay::Menu(_)) => Context::Menu,
        None if app.view == View::Dataset && app.dataset.input.is_some() => Context::Filter,
        None if app.view == View::Training && app.training.selected_abandons() => Context::Abandon,
        None => Context::View(app.view),
    }
}

/// The status message, marked `✓` or `✗`.
fn status_line(status: &Status, theme: &Theme) -> Line<'static> {
    let (mark, style) = match status.severity {
        Severity::Info => ("✓", theme.ok),
        Severity::Warn => ("✗", theme.warn),
        Severity::Error => ("✗", theme.error),
    };
    Line::from(Span::styled(format!(" {mark} {}", status.text), style))
}

/// As many of `hints` as fit `width`, in order, a column kept free on the
/// right; the keys the data lock refuses crossed out while `locked`.
fn hints_line(hints: &[Hint], locked: bool, width: u16, theme: &Theme) -> Line<'static> {
    let room = usize::from(width).saturating_sub(1);
    let mut spans = vec![Span::raw(" ")];
    let mut used = 1;
    for (index, hint) in hints.iter().enumerate() {
        let separator = if index == 0 { "" } else { SEPARATOR };
        let cost =
            separator.chars().count() + hint.key.chars().count() + 1 + hint.label.chars().count();
        if used + cost > room {
            break;
        }
        used += cost;
        let (key, label) = if locked && hint.locks {
            (theme.locked, theme.locked)
        } else {
            (theme.key, theme.dim)
        };
        spans.push(Span::styled(separator, theme.dim));
        spans.push(Span::styled(hint.key, key));
        spans.push(Span::styled(format!(" {}", hint.label), label));
    }
    Line::from(spans)
}

#[cfg(test)]
mod tests {
    use ratatui::style::Modifier;

    use crate::runs::RunState;
    use crate::tui::app::{Severity, View};
    use crate::tui::snapshots::{NOW, app, at, draw, pipeline_running, run, text};
    use crate::tui::tasks::TaskId;
    use crate::tui::training::{Follow, Job, RunRow};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// The last row of `app` drawn at 80x24.
    fn footer(app: &mut crate::tui::app::App) -> Result<String, Box<dyn std::error::Error>> {
        let rows = text(&draw(app, 80, 24)?);
        Ok(rows.last().cloned().unwrap_or_default())
    }

    #[test]
    fn a_locked_key_is_crossed_out_and_locked_joins_the_work() -> TestResult {
        let mut app = app();
        app.view = View::Pipeline;
        pipeline_running(&mut app);
        let terminal = draw(&mut app, 80, 24)?;
        let rows = text(&terminal);
        assert_eq!(
            rows.last().map(String::as_str),
            Some(
                " r run a stage · 1-4 views · q quit         … answers 120/400 · locked · ? help "
            )
        );
        let buffer = terminal.backend().buffer();
        assert!(
            buffer[(1, 23)].modifier.contains(Modifier::CROSSED_OUT),
            "r is locked"
        );
        assert!(
            !buffer[(17, 23)].modifier.contains(Modifier::CROSSED_OUT),
            "1-4 is not"
        );
        Ok(())
    }

    #[test]
    fn a_status_replaces_the_hints_until_it_expires() -> TestResult {
        let mut app = app();
        app.say(Severity::Error, "cannot write the edit file");
        assert!(footer(&mut app)?.starts_with(" ✗ cannot write the edit file "));
        app.say(Severity::Info, "deletion saved");
        assert!(footer(&mut app)?.starts_with(" ✓ deletion saved "));
        app.on_tick(at(NOW + 11));
        assert!(footer(&mut app)?.starts_with(" j/k move · l open"));
        Ok(())
    }

    #[test]
    fn c_reads_abandon_on_a_runpod_run_still_starting() -> TestResult {
        let mut app = app();
        app.view = View::Training;
        let id = "20260921-133200-a1b2";
        app.training.runs = vec![RunRow {
            record: run(id, "gpu_cloud", RunState::Preparing),
            pod: None,
        }];
        assert!(footer(&mut app)?.contains("c cancel"));
        app.training
            .tasks
            .insert(TaskId(4), Follow::new(Job::Start { runpod: true }, id));
        assert!(footer(&mut app)?.contains("c abandon"));
        Ok(())
    }
}
