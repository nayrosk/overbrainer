//! The status line: the latest message on the left, the work running on the right.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::tui::app::{App, Severity};

/// Draws the status line in `area`.
pub(in crate::tui) fn render(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let mut spans: Vec<Span> = app
        .work()
        .into_iter()
        .map(|work| Span::styled(format!("{work}  "), theme.dim))
        .collect();
    spans.push(Span::styled("? help", theme.key));
    let right = Line::from(spans).right_aligned();
    let width = u16::try_from(right.width()).unwrap_or(area.width);
    let [left_area, right_area] =
        Layout::horizontal([Constraint::Fill(1), Constraint::Length(width + 1)]).areas(area);
    if let Some(status) = &app.status {
        let style = match status.severity {
            Severity::Info => theme.ok,
            Severity::Warn => theme.warn,
            Severity::Error => theme.error,
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(format!(" {}", status.text), style))),
            left_area,
        );
    }
    frame.render_widget(Paragraph::new(right), right_area);
}
