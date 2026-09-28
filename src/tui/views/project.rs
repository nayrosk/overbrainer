//! The Project view: the configuration on the left, one row per field, and the
//! project's stats on the right.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Padding, Paragraph};

use crate::tui::app::App;
use crate::tui::format::{cut, hang};
use crate::tui::project::{self, Field, Locks, ProjectView, Row, Shown, Stat};
use crate::tui::theme::Theme;

/// Columns of a stat's label: a longer one, or a value that does not fit
/// beside it, gets its own line.
const LABEL_WIDTH: usize = 11;
/// Rows of the selected field's detail, under the list.
const DETAIL_ROWS: u16 = 2;

/// A rounded pane with a column of padding, its border drawn in `border`.
fn pane<'a>(border: Style) -> Block<'a> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(border)
        .padding(Padding::horizontal(1))
}

/// Draws the Project view in `area`.
pub(in crate::tui) fn render(frame: &mut Frame, area: Rect, app: &mut App) {
    let theme = app.theme;
    let [left, right] =
        Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(area);
    let block = pane(theme.border_focus).title(Span::styled(" configuration ", theme.title));
    let inner = block.inner(left);
    frame.render_widget(block, left);
    match &app.config {
        Some(config) => {
            let rows = project::rows(config, &Locks::of(app));
            render_rows(frame, inner, (&rows, &mut app.project_view), &theme);
        },
        None => frame.render_widget(
            Paragraph::new(Span::styled("No configuration read.", theme.dim)),
            inner,
        ),
    }
    let block = pane(theme.border).title(Span::styled(" stats ", theme.title));
    let inner = block.inner(right);
    frame.render_widget(block, right);
    let lines = stat_lines(&project::stats(app), inner.width, &theme);
    frame.render_widget(Paragraph::new(lines), inner);
}

/// The rows, scrolled so the selected field shows, then its detail.
fn render_rows(
    frame: &mut Frame,
    area: Rect,
    (rows, view): (&[Row], &mut ProjectView),
    theme: &Theme,
) {
    let [list, detail] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(DETAIL_ROWS)]).areas(area);
    let fields: Vec<usize> = rows
        .iter()
        .enumerate()
        .filter_map(|(at, row)| matches!(row, Row::Field(_)).then_some(at))
        .collect();
    view.selected = view.selected.min(fields.len().saturating_sub(1));
    let selected = fields.get(view.selected).copied().unwrap_or(0);
    let height = usize::from(list.height).max(1);
    // Keep the heading of the selected field's table in view when there is room.
    let top = match selected.checked_sub(1).and_then(|above| rows.get(above)) {
        Some(Row::Heading(_)) => selected - 1,
        _ => selected,
    };
    if top < view.offset {
        view.offset = top;
    } else if selected >= view.offset + height {
        view.offset = selected + 1 - height;
    }
    let width = usize::from(list.width);
    let name_width = rows
        .iter()
        .filter_map(|row| match row {
            Row::Field(field) => Some(field.name.chars().count() + 2),
            Row::Heading(_) => None,
        })
        .max()
        .unwrap_or(0)
        .min(width * 3 / 5);
    let lines: Vec<Line> = rows
        .iter()
        .enumerate()
        .skip(view.offset)
        .take(height)
        .map(|(at, row)| match row {
            Row::Heading(text) => Line::from(Span::styled(cut(text, width), theme.title)),
            Row::Field(field) => field_line(field, (name_width, width), at == selected, theme),
        })
        .collect();
    frame.render_widget(Paragraph::new(lines), list);
    if let Some(Row::Field(field)) = rows.get(selected) {
        let lines: Vec<Line> = hang(&field.detail(), detail.width, 0)
            .into_iter()
            .take(usize::from(DETAIL_ROWS))
            .map(|line| Line::from(Span::styled(line, theme.dim)))
            .collect();
        frame.render_widget(Paragraph::new(lines), detail);
    }
}

/// One field: its name, its value, `(env)` after an env value and
/// `(used by …)` after a locked one, which goes dim.
fn field_line(
    field: &Field,
    (name_width, width): (usize, usize),
    selected: bool,
    theme: &Theme,
) -> Line<'static> {
    let name = cut(field.name, name_width.saturating_sub(2));
    let name = format!("  {name:<pad$}", pad = name_width.saturating_sub(2));
    let value_style = match (&field.shown, field.lock.is_some()) {
        (_, true) | (Shown::Default(_) | Shown::Unset, _) => theme.dim,
        (Shown::VaultRef, _) => theme.info,
        (Shown::Value(_) | Shown::Set, _) => Style::new(),
    };
    let room = width.saturating_sub(name_width + 1);
    let value = field.shown.text();
    // `(used by …)` becomes `(used)` when the value and it do not fit: the
    // detail line under the list says by what.
    let mut marks = String::new();
    if field.env {
        marks.push_str(" (env)");
    }
    if let Some(user) = &field.lock {
        let long = format!(" (used by {user})");
        let fits = value.chars().count() + marks.chars().count() + long.chars().count() <= room;
        marks.push_str(if fits { &long } else { " (used)" });
    }
    // The marks come first: a long value is cut to what they leave.
    let value = cut(value, room.saturating_sub(marks.chars().count()));
    let marks = cut(&marks, room.saturating_sub(value.chars().count()));
    let name_style = if field.lock.is_some() {
        theme.dim
    } else {
        Style::new()
    };
    let line = Line::from(vec![
        Span::styled(name, name_style),
        Span::raw(" "),
        Span::styled(value, value_style),
        Span::styled(marks, theme.dim),
    ]);
    if selected {
        line.style(theme.selected)
    } else {
        line
    }
}

/// The stats as lines `width` columns wide, a blank line before each group
/// but the first.
fn stat_lines(stats: &[Stat], width: u16, theme: &Theme) -> Vec<Line<'static>> {
    let width = usize::from(width);
    let mut lines = Vec::new();
    for stat in stats {
        match stat {
            Stat::Heading(text) => {
                if !lines.is_empty() {
                    lines.push(Line::from(""));
                }
                lines.push(Line::from(Span::styled(cut(text, width), theme.title)));
            },
            Stat::Note(text) => {
                lines.push(Line::from(Span::styled(
                    cut(&format!("  {text}"), width),
                    theme.dim,
                )));
            },
            Stat::Pair(label, value)
                if label.chars().count() > LABEL_WIDTH
                    || LABEL_WIDTH + 3 + value.chars().count() > width =>
            {
                lines.push(Line::from(Span::styled(
                    cut(&format!("  {label}"), width),
                    theme.dim,
                )));
                let indent = " ".repeat((LABEL_WIDTH + 3).min(width / 3));
                lines.push(Line::from(cut(&format!("{indent}{value}"), width)));
            },
            Stat::Pair(label, value) => {
                let label = format!("  {label:<LABEL_WIDTH$} ");
                let value = cut(value, width.saturating_sub(label.chars().count()));
                lines.push(Line::from(vec![
                    Span::styled(label, theme.dim),
                    Span::raw(value),
                ]));
            },
        }
    }
    lines
}
