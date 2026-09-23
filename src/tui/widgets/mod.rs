//! Pieces drawn over or around the views.

pub(super) mod help;
pub(super) mod status;
pub(super) mod too_small;

use ratatui::layout::{Constraint, Flex, Layout, Rect};

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
