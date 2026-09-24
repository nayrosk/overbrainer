//! Pieces drawn over or around the views.

pub(super) mod dialog;
pub(super) mod help;
pub(super) mod menu;
pub(super) mod status;
pub(super) mod too_small;

use ratatui::Frame;
use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::widgets::{Block, BorderType, Clear};

use crate::tui::theme::Theme;

/// A `width` by `height` rectangle centered in `area`, shrunk to fit it.
pub(super) fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let [row] = Layout::vertical([Constraint::Length(height.min(area.height))])
        .flex(Flex::Center)
        .areas(area);
    let [cell] = Layout::horizontal([Constraint::Length(width.min(area.width))])
        .flex(Flex::Center)
        .areas(row);
    cell
}

/// Clears `popup` and paints it as an overlay: the surface color, and a
/// rounded crimson border. Returns the block, to draw its title and content.
pub(super) fn overlay<'a>(frame: &mut Frame, popup: Rect, theme: &Theme) -> Block<'a> {
    frame.render_widget(Clear, popup);
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme.accent)
        .style(theme.surface)
}
