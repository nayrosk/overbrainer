//! The `r` menu: the pipeline commands, run on every topic without `--force`.

use ratatui::Frame;
use ratatui::layout::{Constraint, Rect};
use ratatui::text::Span;
use ratatui::widgets::{Padding, Row, Table, TableState};

use super::{centered, overlay};
use crate::tui::app::MENU;
use crate::tui::pipeline::command_name;
use crate::tui::theme::Theme;

/// Width of the menu, borders included.
const WIDTH: u16 = 70;

/// Draws the menu with entry `selected` highlighted, centered over `area`.
pub(in crate::tui) fn render(frame: &mut Frame, area: Rect, selected: usize, theme: &Theme) {
    let rows: Vec<Row> = MENU
        .iter()
        .map(|(command, what)| {
            Row::new(vec![
                Span::styled(command_name(*command), theme.key),
                Span::raw(*what),
            ])
        })
        .collect();
    let height = u16::try_from(MENU.len())
        .unwrap_or(u16::MAX)
        .saturating_add(2);
    let popup = centered(area, WIDTH, height);
    let block = overlay(frame, popup, theme)
        .title(Span::styled(" Run on every topic ", theme.title))
        .title_bottom(Span::styled(" Enter runs, Esc closes ", theme.dim))
        .padding(Padding::horizontal(1));
    let table = Table::new(rows, [Constraint::Length(11), Constraint::Fill(1)])
        .block(block)
        .style(theme.text_hi)
        .row_highlight_style(theme.selected)
        .highlight_symbol("▶ ");
    let mut state = TableState::default().with_selected(Some(selected));
    frame.render_stateful_widget(table, popup, &mut state);
}
