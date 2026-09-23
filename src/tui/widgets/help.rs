//! The help overlay: the keys of every view, then those of the current one, and
//! the note on what the data lock refuses.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::Span;
use ratatui::widgets::{Block, Clear, Paragraph, Row, Table, Wrap};

use super::centered;
use crate::tui::app::App;
use crate::tui::keys::{self, ACTION_WIDTH, HELP_WIDTH, KEYS_WIDTH, KeyHelp};

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
    let note = Paragraph::new(Span::styled(keys::NOTE, theme.dim)).wrap(Wrap { trim: true });
    let note_height =
        u16::try_from(note.line_count(HELP_WIDTH.saturating_sub(2))).unwrap_or(u16::MAX);
    let table_height = u16::try_from(rows.len()).unwrap_or(u16::MAX);
    // The table, a blank row, the note, and the two borders.
    let height = table_height.saturating_add(note_height).saturating_add(3);
    let popup = centered(area, HELP_WIDTH, height);
    let block = Block::bordered()
        .title(Span::styled(
            format!(" keys: {} ", app.view.title()),
            theme.title,
        ))
        .border_style(theme.title);
    let inner = block.inner(popup);
    let [keys_area, _, note_area] = Layout::vertical([
        Constraint::Length(table_height),
        Constraint::Length(1),
        Constraint::Fill(1),
    ])
    .areas(inner);
    let table = Table::new(
        rows,
        [
            Constraint::Length(KEYS_WIDTH),
            Constraint::Length(ACTION_WIDTH),
        ],
    );
    frame.render_widget(Clear, popup);
    frame.render_widget(block, popup);
    frame.render_widget(table, keys_area);
    frame.render_widget(note, note_area);
}
