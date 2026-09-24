//! A confirmation dialog: its question, then the keys that answer it, `n`
//! highlighted as the default.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Padding, Paragraph, Wrap};

use super::{centered, overlay};
use crate::tui::app::{Action, Confirm};
use crate::tui::start;
use crate::tui::theme::Theme;

/// What marks text cut from a dialog too tall for the terminal.
const CUT: &str = "…";
/// The widest a dialog gets, borders included.
const MAX_WIDTH: u16 = 72;
/// Columns kept free on each side of a dialog.
const MARGIN: u16 = 4;
/// Columns of padding inside the borders, on each side.
const PADDING: u16 = 2;
/// Rows a dialog takes besides its text: two borders, a blank row above the
/// text, a blank row and the key row under it, and a blank row at the bottom.
const CHROME_ROWS: u16 = 6;

/// Whether `y` on a dialog running `action` loses something: a deletion, a
/// cancel, an abandon, or quitting while a stage runs (its requests in flight
/// are lost). Its `y` is then drawn as an error.
pub(in crate::tui) fn destructive(action: &Action, stage_running: bool) -> bool {
    match action {
        Action::Delete { .. }
        | Action::Cancel(_)
        | Action::Abandon(_)
        | Action::AbandonStart(_) => true,
        Action::Quit => stage_running,
        Action::Start(_) => false,
    }
}

/// Draws `confirm` centered over `area`. The key line always shows; when the
/// text does not fit, the paragraphs that do come first, then [`CUT`], then
/// the start dialog's most-it-can-cost line, which is never cut. `y` is drawn
/// as an error when `destructive`. Returns where the dialog is.
pub(in crate::tui) fn render(
    frame: &mut Frame,
    area: Rect,
    confirm: &Confirm,
    theme: &Theme,
    destructive: bool,
) -> Rect {
    let width = area.width.saturating_sub(2 * MARGIN).min(MAX_WIDTH);
    let pinned = match &confirm.action {
        Action::Start(plan) => start::cost_line(plan),
        _ => None,
    };
    let room = usize::from(area.height.saturating_sub(CHROME_ROWS));
    let text_width = width.saturating_sub(2 + 2 * PADDING);
    let (lines, rows) = fitted(&confirm.text, text_width, room, pinned, theme);
    let height = u16::try_from(rows)
        .unwrap_or(u16::MAX)
        .saturating_add(CHROME_ROWS);
    let popup = centered(area, width, height);
    let block = overlay(frame, popup, theme)
        .title(Span::styled(confirm.title.as_str(), theme.title))
        .padding(Padding::new(PADDING, PADDING, 1, 1));
    let inner = block.inner(popup);
    let [body, _, keys] = Layout::vertical([
        Constraint::Fill(1),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .areas(inner);
    let yes = if destructive {
        theme.error
    } else {
        theme.accent
    };
    let answers = Line::from(vec![
        Span::styled(format!(" y {} ", confirm.yes), yes),
        Span::raw("  "),
        Span::styled(format!(" n {} ", confirm.no), theme.selected),
    ])
    .right_aligned();
    frame.render_widget(block, popup);
    frame.render_widget(
        Paragraph::new(lines)
            .style(theme.text_hi)
            .wrap(Wrap { trim: true }),
        body,
    );
    frame.render_widget(Paragraph::new(answers), keys);
    popup
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

#[cfg(test)]
mod tests {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use super::*;
    use crate::tui::app::Action;
    use crate::tui::theme::ColorLevel;

    fn confirm(action: Action) -> Confirm {
        Confirm {
            title: " Quit overbrainer? ".into(),
            text: vec!["A stage stops now.".into()],
            yes: "quit",
            no: "stay",
            action,
        }
    }

    /// The cell holding the first character of `word` in `terminal`.
    fn find(terminal: &Terminal<TestBackend>, word: &str) -> Option<ratatui::buffer::Cell> {
        let buffer = terminal.backend().buffer();
        let width = usize::from(buffer.area.width);
        let rows: Vec<String> = buffer
            .content()
            .chunks(width)
            .map(|row| row.iter().map(ratatui::buffer::Cell::symbol).collect())
            .collect();
        rows.iter().enumerate().find_map(|(y, row)| {
            let x = row.find(word)?;
            let x = row[..x].chars().count();
            let (x, y) = (u16::try_from(x).ok()?, u16::try_from(y).ok()?);
            buffer.cell((x, y)).cloned()
        })
    }

    #[test]
    fn a_destructive_yes_is_an_error_and_no_is_the_highlighted_default()
    -> Result<(), Box<dyn std::error::Error>> {
        let theme = Theme::new(ColorLevel::TrueColor);
        let quit = confirm(Action::Quit);
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;
        for (stage_running, style) in [(true, theme.error), (false, theme.accent)] {
            let destructive = destructive(&quit.action, stage_running);
            terminal.draw(|frame| {
                render(frame, frame.area(), &quit, &theme, destructive);
            })?;
            let yes = find(&terminal, "y quit").ok_or("no y key")?;
            assert_eq!(Some(yes.fg), style.fg, "stage running: {stage_running}");
            let no = find(&terminal, "n stay").ok_or("no n key")?;
            assert_eq!(Some(no.bg), theme.selected.bg);
        }
        assert!(destructive(&Action::Cancel("run".into()), false));
        Ok(())
    }
}
