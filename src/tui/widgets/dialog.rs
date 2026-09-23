//! A confirmation dialog: its question, then the keys that answer it.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph, Wrap};

use super::centered;
use crate::tui::app::{Action, Confirm};
use crate::tui::start;
use crate::tui::theme::Theme;

/// What marks text cut from a dialog too tall for the terminal.
const CUT: &str = "…";

/// Draws `confirm` centered over `area`. The key line always shows; when the
/// text does not fit, the paragraphs that do come first, then [`CUT`], then
/// the start dialog's most-it-can-cost line, which is never cut.
pub(in crate::tui) fn render(frame: &mut Frame, area: Rect, confirm: &Confirm, theme: &Theme) {
    let width = area.width.saturating_sub(4).min(76);
    let pinned = match &confirm.action {
        Action::Start(plan) => start::cost_line(plan),
        _ => None,
    };
    // Rows left for the text: the borders, a blank row and the key row taken.
    let room = usize::from(area.height.saturating_sub(4));
    let (lines, rows) = fitted(&confirm.text, width.saturating_sub(2), room, pinned, theme);
    let height = u16::try_from(rows).unwrap_or(u16::MAX).saturating_add(4);
    let popup = centered(area, width, height);
    let block = Block::bordered()
        .title(Span::styled(confirm.title.as_str(), theme.title))
        .border_style(theme.title);
    let inner = block.inner(popup);
    let [body, keys] = Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(inner);
    let answers = Line::from(vec![
        Span::styled("[y]", theme.key),
        Span::raw(format!(" {}   ", confirm.yes)),
        Span::styled("[n]", theme.key),
        Span::raw(format!(" {}", confirm.no)),
    ])
    .right_aligned();
    frame.render_widget(Clear, popup);
    frame.render_widget(block, popup);
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), body);
    frame.render_widget(Paragraph::new(answers), keys);
}

/// Rows `paragraph` takes wrapped at `width`.
fn rows_of(paragraph: &str, width: u16) -> usize {
    Paragraph::new(paragraph)
        .wrap(Wrap { trim: true })
        .line_count(width)
}

/// The paragraphs of `text` shown in `room` rows at `width`, with the rows
/// they take. All of them when they fit; otherwise, in order, those that fit
/// before the first that does not, [`CUT`], and paragraph `pinned` wherever it
/// is (it is kept whenever it fits with the cut mark).
fn fitted<'a>(
    text: &'a [String],
    width: u16,
    room: usize,
    pinned: Option<usize>,
    theme: &Theme,
) -> (Vec<Line<'a>>, usize) {
    let rows: Vec<usize> = text.iter().map(|p| rows_of(p, width)).collect();
    let total: usize = rows.iter().sum();
    if total <= room {
        return (text.iter().map(|p| Line::from(p.as_str())).collect(), total);
    }
    let kept = pinned.filter(|index| rows.get(*index).is_some_and(|r| *r < room));
    let mut left = room.saturating_sub(1 + kept.and_then(|i| rows.get(i)).map_or(0, |r| *r));
    let (mut lines, mut used) = (Vec::new(), 0);
    let mut cut = false;
    for (index, (paragraph, height)) in text.iter().zip(&rows).enumerate() {
        if Some(index) == kept {
            lines.push(Line::from(paragraph.as_str()));
            used += height;
        } else if !cut && *height <= left {
            left -= height;
            lines.push(Line::from(paragraph.as_str()));
            used += height;
        } else if !cut {
            cut = true;
            lines.push(Line::styled(CUT, theme.dim));
            used += 1;
        }
    }
    (lines, used)
}
