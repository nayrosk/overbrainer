//! The footer, on the last row: the key hints of what the keys act on, or the
//! latest status message while it lasts, on the left; the work running, each
//! with its spinner, `locked` while the data lock holds, the project cost and
//! the version, marked `↑ X.Y.Z` when a newer release exists (not under a
//! dialog, the help or the menu) and `? help` on the right.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::history::Cost;
use crate::tui::app::{Action, App, Confirm, Overlay, Severity, Status, View};
use crate::tui::cost::project_cost;
use crate::tui::keys::{self, Context, HELP_HINT, Hint, SEPARATOR};
use crate::tui::project::Form;
use crate::tui::theme::Theme;
use crate::tui::training::RunActivity;

/// The version shown before `? help`.
pub(in crate::tui) const VERSION: &str = concat!("v", env!("CARGO_PKG_VERSION"));

/// Draws the footer in `area`; returns where the status message is, when one
/// shows.
pub(in crate::tui) fn render(frame: &mut Frame, area: Rect, app: &App) -> Option<Rect> {
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
    // An overlay's hints need the room: no cost and no version under it.
    if app.overlay.is_none() {
        let cost = match project_cost(app) {
            Cost::Known(usd) => Some(format!("${usd:.2}")),
            Cost::Partial(usd) => Some(format!("${usd:.2}+")),
            Cost::Unknown => None,
        };
        if let Some(cost) = cost {
            right.push(Span::styled(format!("{cost}  "), theme.dim));
        }
        right.push(Span::styled(VERSION, theme.dim));
        if let Some(newer) = &app.newer {
            right.push(Span::styled(format!(" ↑ {newer}"), theme.accent));
        }
        right.push(Span::styled("  ", theme.dim));
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
    app.status.as_ref().map(|_| left_area)
}

/// What the keys act on now.
pub(in crate::tui) fn context(app: &App) -> Context {
    match &app.overlay {
        Some(Overlay::Confirm(Confirm {
            action: Action::Start(plan),
            ..
        })) if plan.runpod.is_some() => Context::Start,
        Some(Overlay::Confirm(confirm)) => Context::Dialog {
            yes: confirm.yes,
            no: confirm.no,
        },
        Some(Overlay::Help) => Context::Help,
        Some(Overlay::Menu(_)) => Context::Menu,
        Some(Overlay::Picker(picking)) => {
            let picker = &picking.picker;
            let typed = picker.typed();
            if picker.typing() {
                Context::Filter
            } else if picker.loading() || picker.error().is_some() {
                Context::Listing { typed }
            } else {
                Context::Picker {
                    mode: picker.mode(),
                    typed,
                }
            }
        },
        None if app.view == View::Dataset && app.dataset.input.is_some() => Context::Filter,
        None if app.view == View::Project => match &app.project_view.form {
            Some(Form::Value { .. } | Form::Name { .. }) => Context::Form,
            Some(Form::Adding(_) | Form::Kind { .. }) => Context::Pick,
            None => Context::View(View::Project),
        },
        None if app.view == View::Training
            && app.training.selected_activity() == RunActivity::Starting { runpod: true } =>
        {
            Context::Abandon
        },
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

    use super::VERSION;
    use crate::history::Cost;
    use crate::runs::RunState;
    use crate::tui::app::{Action, Confirm, Overlay, Severity, View};
    use crate::tui::snapshots::{NOW, app, at, draw, pipeline_running, run, text};
    use crate::tui::tasks::TaskId;
    use crate::tui::training::{Detach, Follow, Job, RunRow};

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
        let shown = text(&terminal).last().cloned().unwrap_or_default();
        let right = format!("… answers 120/400 · locked · {VERSION}  ? help ");
        assert!(
            shown.starts_with(" r run a stage · 1-5 views · q quit "),
            "{shown}"
        );
        assert!(shown.ends_with(&right), "{shown}");
        let buffer = terminal.backend().buffer();
        assert!(
            buffer[(1, 23)].modifier.contains(Modifier::CROSSED_OUT),
            "r is locked"
        );
        assert!(
            !buffer[(17, 23)].modifier.contains(Modifier::CROSSED_OUT),
            "1-5 is not"
        );
        Ok(())
    }

    #[test]
    fn the_version_comes_before_help() -> TestResult {
        let shown = footer(&mut app())?;
        let version = format!("v{}  ? help", env!("CARGO_PKG_VERSION"));
        assert!(shown.trim_end().ends_with(&version), "{shown}");
        assert!(!shown.contains('$'), "nothing spent yet: {shown}");
        Ok(())
    }

    #[test]
    fn the_project_cost_shows_with_two_decimals() -> TestResult {
        let mut app = app();
        let version = format!("  v{}", env!("CARGO_PKG_VERSION"));
        app.history_cost = Some(Cost::Known(1.5));
        let shown = footer(&mut app)?;
        assert!(shown.contains(&format!(" $1.50{version}")), "{shown}");
        app.history_cost = Some(Cost::Partial(1.5));
        let shown = footer(&mut app)?;
        assert!(shown.contains(&format!(" $1.50+{version}")), "{shown}");
        app.history_cost = Some(Cost::Unknown);
        let shown = footer(&mut app)?;
        assert!(!shown.contains('$'), "{shown}");
        Ok(())
    }

    #[test]
    fn only_the_help_key_stands_out_on_the_right() -> TestResult {
        let mut app = app();
        app.history_cost = Some(Cost::Known(1.5));
        let terminal = draw(&mut app, 80, 24)?;
        let buffer = terminal.backend().buffer();
        let column = |symbol: &str| (0..80).find(|&x| buffer[(x, 23)].symbol() == symbol);
        let style = |x: u16| buffer[(x, 23)].style();
        let dollar = column("$").ok_or("no cost")?;
        let help = column("?").ok_or("no help")?;
        let dim = style(help + 2);
        assert_eq!(style(dollar), dim, "the cost is dim");
        assert_eq!(style(help - 3), dim, "the version is dim");
        assert_ne!(style(help), dim, "? stands out");
        Ok(())
    }

    #[test]
    fn a_dialog_keeps_its_hints_over_the_cost_and_the_version() -> TestResult {
        let mut app = app();
        pipeline_running(&mut app);
        app.history_cost = Some(Cost::Known(1.5));
        app.overlay = Some(Overlay::Confirm(Confirm {
            title: " Quit overbrainer? ".to_string(),
            text: vec!["A pipeline is running.".to_string()],
            yes: "quit",
            no: "stay",
            action: Action::Quit,
        }));
        let shown = footer(&mut app)?;
        assert!(shown.contains("n, Esc or Enter stay"), "{shown}");
        assert!(!shown.contains('$'), "{shown}");
        assert!(!shown.contains(VERSION), "{shown}");
        Ok(())
    }

    #[test]
    fn a_newer_release_follows_the_version_in_the_accent() -> TestResult {
        let mut app = app();
        app.newer = Some("0.9.0".into());
        let terminal = draw(&mut app, 120, 40)?;
        let shown = text(&terminal).last().cloned().unwrap_or_default();
        assert!(
            shown.ends_with(&format!("{VERSION} ↑ 0.9.0  ? help ")),
            "{shown}"
        );
        let buffer = terminal.backend().buffer();
        let arrow = (0..120)
            .find(|&x| buffer[(x, 39)].symbol() == "↑")
            .ok_or("no marker")?;
        assert_eq!(Some(buffer[(arrow, 39)].fg), app.theme.accent.fg);
        app.overlay = Some(Overlay::Confirm(Confirm {
            title: " Quit overbrainer? ".to_string(),
            text: vec![],
            yes: "quit",
            no: "stay",
            action: Action::Quit,
        }));
        let shown = text(&draw(&mut app, 120, 40)?)
            .last()
            .cloned()
            .unwrap_or_default();
        assert!(!shown.contains('↑'), "hidden with the version: {shown}");
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
        if let Some(follow) = app.training.tasks.get_mut(&TaskId(4)) {
            follow.detach = Detach::Done;
        }
        let shown = footer(&mut app)?;
        assert!(!shown.contains("c abandon"), "{shown}");
        Ok(())
    }
}
