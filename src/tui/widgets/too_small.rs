//! The screen shown instead of the views when the terminal is below 80x24.

use ratatui::Frame;
use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Wrap};

/// The smallest terminal the views are drawn in.
pub(in crate::tui) const MIN_WIDTH: u16 = 80;
/// See [`MIN_WIDTH`].
pub(in crate::tui) const MIN_HEIGHT: u16 = 24;

/// Whether `area` is too small for the views.
pub(in crate::tui) fn too_small(area: Rect) -> bool {
    area.width < MIN_WIDTH || area.height < MIN_HEIGHT
}

/// Says the terminal is too small, centered in `area`.
pub(in crate::tui) fn render(frame: &mut Frame, area: Rect) {
    let text = format!(
        "terminal too small: {MIN_WIDTH}x{MIN_HEIGHT} needed, this one is {}x{}",
        area.width, area.height
    );
    let [row] = Layout::vertical([Constraint::Length(2)])
        .flex(Flex::Center)
        .areas(area);
    let paragraph = Paragraph::new(vec![
        Line::from(text).centered(),
        Line::from("q quits").centered(),
    ])
    .wrap(Wrap { trim: true });
    frame.render_widget(paragraph, row);
}
