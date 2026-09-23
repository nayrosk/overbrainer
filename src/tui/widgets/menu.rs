//! The `r` menu: the pipeline commands, run on every topic without `--force`.

use ratatui::Frame;
use ratatui::layout::{Constraint, Rect};
use ratatui::text::Span;
use ratatui::widgets::{Block, Clear, Row, Table, TableState};

use super::centered;
use crate::tui::app::MENU;
use crate::tui::pipeline::command_name;
use crate::tui::theme::Theme;

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
    let popup = centered(area, 66, 7);
    let block = Block::bordered()
        .title(Span::styled(" Run on every topic ", theme.title))
        .title_bottom(Span::styled(" Enter runs, Esc closes ", theme.dim))
        .border_style(theme.title);
    let table = Table::new(rows, [Constraint::Length(11), Constraint::Fill(1)])
        .block(block)
        .row_highlight_style(theme.selected)
        .highlight_symbol("▶ ");
    let mut state = TableState::default().with_selected(Some(selected));
    frame.render_widget(Clear, popup);
    frame.render_stateful_widget(table, popup, &mut state);
}
