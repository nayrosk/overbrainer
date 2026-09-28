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
    let lines = stat_lines(&project::stats(app), (inner.width, inner.height), &theme);
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

/// The lines of one stat, `width` columns wide: a pair whose label or value
/// does not fit beside the other takes two.
fn stat_line(stat: &Stat, width: usize, theme: &Theme) -> Vec<Line<'static>> {
    match stat {
        Stat::Heading(text) => vec![Line::from(Span::styled(cut(text, width), theme.title))],
        Stat::Note(text) => vec![Line::from(Span::styled(
            cut(&format!("  {text}"), width),
            theme.dim,
        ))],
        Stat::Pair(label, value)
            if label.chars().count() > LABEL_WIDTH
                || LABEL_WIDTH + 3 + value.chars().count() > width =>
        {
            let indent = " ".repeat((LABEL_WIDTH + 3).min(width / 3));
            vec![
                Line::from(Span::styled(cut(&format!("  {label}"), width), theme.dim)),
                Line::from(cut(&format!("{indent}{value}"), width)),
            ]
        },
        Stat::Pair(label, value) => {
            let label = format!("  {label:<LABEL_WIDTH$} ");
            let value = cut(value, width.saturating_sub(label.chars().count()));
            vec![Line::from(vec![
                Span::styled(label, theme.dim),
                Span::raw(value),
            ])]
        },
    }
}

/// The stats in `width` columns and `height` rows, a blank line between two
/// groups when all fit with them. A group shows only with its heading and at
/// least its first stat, and a stat never loses its second line: what does
/// not fit is left out, never a lone heading.
fn stat_lines(stats: &[Stat], (width, height): (u16, u16), theme: &Theme) -> Vec<Line<'static>> {
    let width = usize::from(width);
    let height = usize::from(height);
    let mut groups: Vec<Vec<Vec<Line<'static>>>> = Vec::new();
    for stat in stats {
        let lines = stat_line(stat, width, theme);
        match (stat, groups.last_mut()) {
            (Stat::Heading(_), _) | (_, None) => groups.push(vec![lines]),
            (_, Some(group)) => group.push(lines),
        }
    }
    let count = |group: &Vec<Vec<Line>>| group.iter().map(Vec::len).sum::<usize>();
    let blanks = groups.iter().map(count).sum::<usize>() + groups.len().saturating_sub(1) <= height;
    let mut lines: Vec<Line<'static>> = Vec::new();
    for group in groups {
        let blank = usize::from(blanks && !lines.is_empty());
        let mut room = height.saturating_sub(lines.len() + blank);
        let first = group.iter().take(2).map(Vec::len).sum::<usize>();
        if first > room {
            continue;
        }
        if blank == 1 {
            lines.push(Line::from(""));
        }
        for stat in group {
            if stat.len() > room {
                break;
            }
            room -= stat.len();
            lines.extend(stat);
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::theme::ColorLevel;

    fn texts(lines: &[Line]) -> Vec<String> {
        lines.iter().map(ToString::to_string).collect()
    }

    fn stats() -> Vec<Stat> {
        let pair = |label: &str, value: &str| Stat::Pair(label.into(), value.into());
        vec![
            Stat::Heading("cost"),
            pair("all", "$1"),
            Stat::Heading("dataset"),
            pair("topics", "2"),
            pair("questions", "6"),
            Stat::Heading("models"),
            pair("a-model-with-a-long-name", "$1"),
            pair("gen", "$0"),
        ]
    }

    #[test]
    fn groups_are_parted_by_a_blank_line_when_all_fit() {
        let theme = Theme::new(ColorLevel::TrueColor);
        let lines = texts(&stat_lines(&stats(), (30, 11), &theme));
        assert_eq!(lines.len(), 11, "{lines:?}");
        assert_eq!((lines[2].as_str(), lines[6].as_str()), ("", ""));
    }

    #[test]
    fn short_on_rows_the_blanks_go_then_whole_stats() {
        let theme = Theme::new(ColorLevel::TrueColor);
        let lines = texts(&stat_lines(&stats(), (30, 10), &theme));
        assert_eq!(lines.len(), 9, "{lines:?}");
        assert!(!lines.contains(&String::new()), "{lines:?}");
        let lines = texts(&stat_lines(&stats(), (30, 8), &theme));
        assert_eq!(lines.len(), 8, "{lines:?}");
        assert_eq!(
            lines.last().map(|line| line.trim()),
            Some("$1"),
            "the long model name and its value go together: {lines:?}"
        );
        let lines = texts(&stat_lines(&stats(), (30, 7), &theme));
        assert!(
            !lines.iter().any(|line| line == "models"),
            "no heading without a stat under it: {lines:?}"
        );
        assert_eq!(lines.len(), 5);
    }
}
