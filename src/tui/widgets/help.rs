//! The help overlay: the keys of every view, then those of the current one.

use ratatui::Frame;
use ratatui::layout::{Constraint, Rect};
use ratatui::text::Span;
use ratatui::widgets::{Block, Clear, Row, Table};

use super::centered;
use crate::tui::app::App;
use crate::tui::keys::{self, KeyHelp};

/// Draws the help overlay over `area`.
pub(in crate::tui) fn render(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let view = keys::of(app.view);
    let rows: Vec<Row> = keys::GLOBAL
        .iter()
        .chain(view)
        .map(|KeyHelp { keys, action }| {
            Row::new(vec![Span::styled(*keys, theme.key), Span::raw(*action)])
        })
        .collect();
    let height = u16::try_from(rows.len())
        .unwrap_or(u16::MAX)
        .saturating_add(2);
    let popup = centered(area, 76, height);
    let block = Block::bordered()
        .title(Span::styled(
            format!(" keys: {} ", app.view.title()),
            theme.title,
        ))
        .border_style(theme.title);
    let table = Table::new(rows, [Constraint::Length(26), Constraint::Fill(1)]).block(block);
    frame.render_widget(Clear, popup);
    frame.render_widget(table, popup);
}
