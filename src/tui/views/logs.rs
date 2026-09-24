//! The Logs view: the captured log lines, newest at the bottom.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Margin, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{List, ListItem, Paragraph};
use tracing::Level;

use crate::logging::LogLine;
use crate::tui::app::App;
use crate::tui::format::clock;

/// Draws the Logs view in `area`: no frame, a two-column margin, a title row,
/// a blank row, then the lines. Records the rows the lines have, which bound
/// how far back the view scrolls.
pub(in crate::tui) fn render(frame: &mut Frame, area: Rect, app: &mut App) {
    let area = area.inner(Margin::new(2, 0));
    let [title_row, _, lines] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Fill(1),
    ])
    .areas(area);
    app.log_view.height = usize::from(lines.height);
    let theme = &app.theme;
    let view = app.log_view;
    let offset = view.offset(&app.logs);
    let window = app.logs.window(view.min, view.height, offset);
    let mut title = vec![Span::styled(
        format!("Logs · {} and above", level_name(view.min)),
        theme.title,
    )];
    if view.anchor.is_some() {
        title.push(Span::styled(
            format!("  ({offset} newer lines below: G follows)"),
            theme.dim,
        ));
    }
    frame.render_widget(Paragraph::new(Line::from(title)), title_row);
    let mut items: Vec<ListItem> = window
        .lines
        .iter()
        .map(|line| ListItem::new(render_line(line, theme)))
        .collect();
    if items.is_empty() && matches!(view.min, Level::DEBUG | Level::TRACE) {
        items.push(ListItem::new(Span::styled(
            "nothing captured at this level: OVERBRAINER_LOG sets what is captured",
            theme.dim,
        )));
    }
    frame.render_widget(List::new(items), lines);
}

fn render_line<'a>(line: &'a LogLine, theme: &crate::tui::theme::Theme) -> Line<'a> {
    Line::from(vec![
        Span::styled(clock(line.time), theme.dim),
        Span::raw(" "),
        Span::styled(
            format!("{:<5}", level_name(line.level)),
            theme.level(line.level),
        ),
        Span::raw(" "),
        Span::styled(format!("{}: ", short_target(&line.target)), theme.dim),
        Span::raw(line.message.as_str()),
    ])
}

/// A log target without the crate's own `overbrainer::` prefix.
fn short_target(target: &str) -> &str {
    target.strip_prefix("overbrainer::").unwrap_or(target)
}

/// The level in upper case.
fn level_name(level: Level) -> &'static str {
    match level {
        Level::ERROR => "ERROR",
        Level::WARN => "WARN",
        Level::INFO => "INFO",
        Level::DEBUG => "DEBUG",
        _ => "TRACE",
    }
}
