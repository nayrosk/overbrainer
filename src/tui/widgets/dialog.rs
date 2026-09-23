//! A confirmation dialog: its question, then the keys that answer it.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph, Wrap};

use super::centered;
use crate::tui::app::Confirm;
use crate::tui::theme::Theme;

/// Draws `confirm` centered over `area`.
pub(in crate::tui) fn render(frame: &mut Frame, area: Rect, confirm: &Confirm, theme: &Theme) {
    let width = area.width.saturating_sub(4).min(76);
    let mut lines: Vec<Line> = Vec::new();
    for paragraph in &confirm.text {
        lines.push(Line::from(paragraph.as_str()));
    }
    lines.push(Line::from(""));
    lines.push(
        Line::from(vec![
            Span::styled("[y]", theme.key),
            Span::raw(format!(" {}   ", confirm.yes)),
            Span::styled("[n]", theme.key),
            Span::raw(" cancel"),
        ])
        .right_aligned(),
    );
    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: true });
    let height = u16::try_from(paragraph.line_count(width.saturating_sub(2)))
        .unwrap_or(u16::MAX)
        .saturating_add(2);
    let popup = centered(area, width, height);
    let block = Block::bordered()
        .title(Span::styled(confirm.title.as_str(), theme.title))
        .border_style(theme.title);
    frame.render_widget(Clear, popup);
    frame.render_widget(paragraph.block(block), popup);
}
