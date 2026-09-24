//! The help overlay: the keys of every view, then those of the current one, and
//! the note on what the data lock refuses.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::Span;
use ratatui::widgets::{Cell, Padding, Paragraph, Row, Table, Wrap};

use super::{centered, overlay};
use crate::tui::app::App;
use crate::tui::keys::{self, ACTION_WIDTH, HELP_PADDING, HELP_WIDTH, KEYS_WIDTH, KeyHelp};

/// Draws the help overlay over `area`; returns where.
pub(in crate::tui) fn render(frame: &mut Frame, area: Rect, app: &App) -> Rect {
    let theme = &app.theme;
    let row = |KeyHelp { keys, action }: &KeyHelp| {
        Row::new(vec![Span::styled(*keys, theme.key), Span::raw(*action)])
    };
    let section = |name: &str| {
        Row::new(vec![Cell::from(Span::styled(
            name.to_string(),
            theme.title,
        ))])
    };
    let mut rows = vec![section("Everywhere")];
    rows.extend(keys::GLOBAL.iter().map(row));
    let view = keys::of(app.view);
    if !view.is_empty() {
        rows.push(Row::new(Vec::<Cell>::new()));
        rows.push(section(app.view.title()));
        rows.extend(view.iter().map(row));
    }
    let text_width = HELP_WIDTH.saturating_sub(2 + 2 * HELP_PADDING);
    let note = Paragraph::new(Span::styled(keys::NOTE, theme.dim)).wrap(Wrap { trim: true });
    let note_height = u16::try_from(note.line_count(text_width)).unwrap_or(u16::MAX);
    let table_height = u16::try_from(rows.len()).unwrap_or(u16::MAX);
    // The table, a blank row, the note, and the two borders.
    let height = table_height.saturating_add(note_height).saturating_add(3);
    let popup = centered(area, HELP_WIDTH, height);
    let block = overlay(frame, popup, theme)
        .title(Span::styled(" Keys ", theme.title))
        .padding(Padding::horizontal(HELP_PADDING));
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
    )
    .style(theme.text_hi);
    frame.render_widget(block, popup);
    frame.render_widget(table, keys_area);
    frame.render_widget(note, note_area);
    popup
}
