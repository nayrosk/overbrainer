//! A one-line text input with a cursor, for the filter and the forms.

use crossterm::event::KeyCode;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

/// The most chars an [`Input`] holds: typing and pasting stop there.
pub(in crate::tui) const MAX_CHARS: usize = 4096;

/// A one-line text being edited, at most [`MAX_CHARS`] chars. The cursor is a
/// char index, from 0 (before the first char) to the char count (after the
/// last).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::tui) struct Input {
    text: String,
    cursor: usize,
}

/// What a key did to an [`Input`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::tui) enum InputOutcome {
    /// Still editing.
    Editing,
    /// Enter: the text is kept.
    Done(String),
    /// Esc: the edit is dropped.
    Cancelled,
}

impl Input {
    /// An input holding `text`, the cursor after its last char.
    pub(in crate::tui) fn new(text: impl Into<String>) -> Self {
        let text = text.into();
        let cursor = text.chars().count();
        Self { text, cursor }
    }

    /// The text typed so far.
    pub(in crate::tui) fn text(&self) -> &str {
        &self.text
    }

    /// Edits the text or moves the cursor; Enter and Esc end the edit.
    pub(in crate::tui) fn on_key(&mut self, code: KeyCode) -> InputOutcome {
        match code {
            KeyCode::Enter => return InputOutcome::Done(self.text.clone()),
            KeyCode::Esc => return InputOutcome::Cancelled,
            KeyCode::Char(c) if !c.is_control() && self.len() < MAX_CHARS => {
                self.text.insert(self.byte(self.cursor), c);
                self.cursor += 1;
            },
            KeyCode::Backspace if self.cursor > 0 => {
                self.cursor -= 1;
                self.text.remove(self.byte(self.cursor));
            },
            KeyCode::Delete if self.cursor < self.len() => {
                self.text.remove(self.byte(self.cursor));
            },
            KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Right => self.cursor = (self.cursor + 1).min(self.len()),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.len(),
            _ => {},
        }
        InputOutcome::Editing
    }

    /// Inserts pasted text at the cursor, on one line: the line breaks at its
    /// ends are dropped, the others become spaces, other control chars go. What
    /// goes past [`MAX_CHARS`] is dropped.
    pub(in crate::tui) fn paste(&mut self, pasted: &str) {
        let room = MAX_CHARS.saturating_sub(self.len());
        let clean: String = pasted
            .trim_matches(['\r', '\n'])
            .replace("\r\n", "\n")
            .chars()
            .filter_map(|c| match c {
                '\r' | '\n' => Some(' '),
                c if c.is_control() => None,
                c => Some(c),
            })
            .take(room)
            .collect();
        self.text.insert_str(self.byte(self.cursor), &clean);
        self.cursor += clean.chars().count();
    }

    /// The part of the text that fits in `width` cells, the cursor cell in
    /// `style` reversed. The cursor after the last char is a reversed space;
    /// a text too wide scrolls so the cursor stays shown.
    pub(in crate::tui) fn line(&self, width: u16, style: Style) -> Line<'static> {
        let width = usize::from(width);
        let chars: Vec<char> = self.text.chars().collect();
        let under = chars.get(self.cursor).copied().unwrap_or(' ');
        let mut used = cells(under);
        if width == 0 || used > width {
            return Line::default();
        }
        let mut start = self.cursor;
        while let Some(&c) = start.checked_sub(1).and_then(|at| chars.get(at)) {
            if used + cells(c) > width {
                break;
            }
            used += cells(c);
            start -= 1;
        }
        let before: String = chars
            .get(start..self.cursor)
            .unwrap_or_default()
            .iter()
            .collect();
        let mut after = String::new();
        for &c in chars.get(self.cursor + 1..).unwrap_or_default() {
            if used + cells(c) > width {
                break;
            }
            used += cells(c);
            after.push(c);
        }
        Line::from(vec![
            Span::styled(before, style),
            Span::styled(under.to_string(), style.add_modifier(Modifier::REVERSED)),
            Span::styled(after, style),
        ])
    }

    /// Draws [`Self::line`] in `area`.
    pub(in crate::tui) fn render(&self, frame: &mut Frame, area: Rect, style: Style) {
        frame.render_widget(self.line(area.width, style), area);
    }

    /// The char count.
    fn len(&self) -> usize {
        self.text.chars().count()
    }

    /// The byte offset of the char at index `at`, or the text's length past it.
    fn byte(&self, at: usize) -> usize {
        self.text
            .char_indices()
            .nth(at)
            .map_or(self.text.len(), |(offset, _)| offset)
    }
}

/// The cells `c` takes on screen.
fn cells(c: char) -> usize {
    let mut buffer = [0; 4];
    Span::raw(&*c.encode_utf8(&mut buffer)).width()
}

#[cfg(test)]
mod tests {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use super::*;

    fn typed(input: &mut Input, codes: &[KeyCode]) -> Vec<InputOutcome> {
        codes.iter().map(|code| input.on_key(*code)).collect()
    }

    fn chars(text: &str) -> Vec<KeyCode> {
        text.chars().map(KeyCode::Char).collect()
    }

    /// The line's text, and the text of its cursor cell.
    fn shown(input: &Input, width: u16) -> (String, String) {
        let line = input.line(width, Style::default());
        let text = line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        let cursor = line
            .spans
            .iter()
            .filter(|span| span.style.add_modifier.contains(Modifier::REVERSED))
            .map(|span| span.content.as_ref())
            .collect();
        (text, cursor)
    }

    #[test]
    fn a_new_input_has_its_cursor_at_the_end() {
        let mut input = Input::new("ab");
        typed(&mut input, &chars("c"));
        assert_eq!(input.text(), "abc");
    }

    #[test]
    fn keys_insert_and_delete_in_the_middle() {
        let mut input = Input::new("");
        typed(&mut input, &chars("acd"));
        typed(&mut input, &[KeyCode::Left, KeyCode::Left]);
        typed(&mut input, &chars("b"));
        assert_eq!(input.text(), "abcd");
        typed(&mut input, &[KeyCode::Delete]);
        assert_eq!(input.text(), "abd", "Delete takes the char at the cursor");
        typed(&mut input, &[KeyCode::Backspace]);
        assert_eq!(input.text(), "ad", "Backspace takes the char before it");
        typed(
            &mut input,
            &[KeyCode::Right, KeyCode::Right, KeyCode::Delete],
        );
        assert_eq!(input.text(), "ad", "Delete at the end does nothing");
        typed(
            &mut input,
            &[KeyCode::Home, KeyCode::Backspace, KeyCode::Left],
        );
        assert_eq!(input.text(), "ad", "Backspace at the start does nothing");
        typed(&mut input, &chars("_"));
        assert_eq!(input.text(), "_ad");
    }

    #[test]
    fn home_and_end_jump_to_the_ends() {
        let mut input = Input::new("mid");
        typed(&mut input, &[KeyCode::Home]);
        typed(&mut input, &chars("<"));
        typed(&mut input, &[KeyCode::End]);
        typed(&mut input, &chars(">"));
        assert_eq!(input.text(), "<mid>");
    }

    #[test]
    fn multi_byte_chars_are_edited_whole() {
        let mut input = Input::new("é日");
        typed(&mut input, &[KeyCode::Left]);
        typed(&mut input, &chars("ü"));
        assert_eq!(input.text(), "éü日");
        typed(&mut input, &[KeyCode::Backspace, KeyCode::Backspace]);
        assert_eq!(input.text(), "日");
        typed(
            &mut input,
            &[KeyCode::Delete, KeyCode::Right, KeyCode::Right],
        );
        assert_eq!(input.text(), "");
        typed(&mut input, &chars("🦀x"));
        assert_eq!(input.text(), "🦀x");
    }

    #[test]
    fn a_paste_keeps_one_line_without_control_chars() {
        let mut input = Input::new("ab");
        typed(&mut input, &[KeyCode::Left]);
        input.paste("one\r\ntwo\nthree\u{1b}[31m\u{7}\n");
        assert_eq!(input.text(), "aone two three[31mb");
        typed(&mut input, &chars("!"));
        assert_eq!(
            input.text(),
            "aone two three[31m!b",
            "the cursor is after the paste"
        );
    }

    #[test]
    fn typing_and_pasting_stop_at_the_cap() {
        let mut input = Input::new("a".repeat(MAX_CHARS - 2));
        input.paste("bcd");
        assert_eq!(input.text().chars().count(), MAX_CHARS);
        assert!(input.text().ends_with("abc"), "the paste is cut at the cap");
        typed(&mut input, &[KeyCode::Home]);
        typed(&mut input, &chars("x"));
        assert_eq!(
            input.text().chars().count(),
            MAX_CHARS,
            "no key goes past it"
        );
        assert!(input.text().starts_with('a'));
        typed(
            &mut input,
            &[KeyCode::Backspace, KeyCode::End, KeyCode::Backspace],
        );
        typed(&mut input, &chars("日"));
        assert!(input.text().ends_with("ab日"), "room again after a delete");
    }

    #[test]
    fn enter_keeps_the_text_and_esc_drops_it() {
        let mut input = Input::new("");
        assert_eq!(
            typed(&mut input, &[KeyCode::Char('x'), KeyCode::Enter]),
            [InputOutcome::Editing, InputOutcome::Done("x".into())]
        );
        assert_eq!(input.on_key(KeyCode::Esc), InputOutcome::Cancelled);
        assert_eq!(input.on_key(KeyCode::F(2)), InputOutcome::Editing);
    }

    #[test]
    fn the_cursor_cell_is_reversed() {
        let mut input = Input::new("abc");
        assert_eq!(
            shown(&input, 10),
            ("abc ".into(), " ".into()),
            "a cell after the end"
        );
        typed(&mut input, &[KeyCode::Left]);
        assert_eq!(shown(&input, 10), ("abc".into(), "c".into()));
        assert_eq!(shown(&Input::new(""), 0), (String::new(), String::new()));
    }

    #[test]
    fn a_long_text_scrolls_to_keep_the_cursor_shown() {
        let mut input = Input::new("abcdefgh");
        assert_eq!(shown(&input, 4), ("fgh ".into(), " ".into()));
        typed(&mut input, &[KeyCode::Home]);
        assert_eq!(shown(&input, 4), ("abcd".into(), "a".into()));
        typed(&mut input, &[KeyCode::Right, KeyCode::Right]);
        assert_eq!(shown(&input, 4), ("abcd".into(), "c".into()));
    }

    #[test]
    fn wide_chars_count_two_cells() {
        let input = Input::new("日本語");
        assert_eq!(shown(&input, 5), ("本語 ".into(), " ".into()));
        let mut input = Input::new("日本語");
        typed(&mut input, &[KeyCode::Home]);
        assert_eq!(shown(&input, 5), ("日本".into(), "日".into()));
    }

    #[test]
    fn render_draws_the_line_in_the_area() -> Result<(), Box<dyn std::error::Error>> {
        let mut terminal = Terminal::new(TestBackend::new(6, 1))?;
        let input = Input::new("ab");
        terminal.draw(|frame| input.render(frame, frame.area(), Style::default()))?;
        let buffer = terminal.backend().buffer();
        assert_eq!(buffer[(0, 0)].symbol(), "a");
        assert!(!buffer[(0, 0)].modifier.contains(Modifier::REVERSED));
        assert!(buffer[(2, 0)].modifier.contains(Modifier::REVERSED));
        Ok(())
    }
}
