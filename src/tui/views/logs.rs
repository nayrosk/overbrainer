//! The Logs view: the captured log lines, newest at the bottom.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, List, ListItem};
use tracing::Level;

use crate::logging::LogLine;
use crate::tui::app::App;
use crate::tui::format::clock;

/// Draws the Logs view in `area`.
/// Records the rows it has, which bound how far back the view scrolls.
pub(in crate::tui) fn render(frame: &mut Frame, area: Rect, app: &mut App) {
    app.log_view.height = usize::from(area.height.saturating_sub(2));
    let theme = &app.theme;
    let view = app.log_view;
    let offset = view.offset(&app.logs);
    let window = app.logs.window(view.min, view.height, offset);
    let note = if view.anchor.is_some() {
        format!("({offset} newer lines below: G follows) ")
    } else {
        String::new()
    };
    let title = format!(" logs: {} and above {note}", level_name(view.min));
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
    let block = Block::bordered()
        .title(Span::styled(title, theme.title))
        .border_style(theme.dim);
    frame.render_widget(List::new(items).block(block), area);
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
        Span::styled(format!("{}: ", line.target), theme.dim),
        Span::raw(line.message.as_str()),
    ])
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
