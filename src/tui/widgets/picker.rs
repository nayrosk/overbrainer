//! The picker overlay: entries in columns, several chosen in order (Space
//! toggles, `J`/`K` move the entry under the cursor in that order) or one
//! picked with Enter, filtered with `/`. The entries arrive later: a spinner
//! shows until then, and an error in their place when they cannot be read.

use crossterm::event::KeyCode;
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Cell, Padding, Paragraph, Row as TableRow, Table, TableState, Wrap};

use super::form::{Input, InputOutcome};
use super::{centered, overlay};
use crate::tui::app::PAGE;
use crate::tui::theme::Theme;

/// The widest the picker gets, borders included.
const MAX_WIDTH: u16 = 100;
/// Columns kept free on each side of the picker.
const MARGIN: u16 = 2;
/// Rows kept free above and under the picker.
const MARGIN_ROWS: u16 = 1;
/// What the `auto` entry is called.
pub(in crate::tui) const AUTO: &str = "auto";

/// How entries are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::tui) enum Mode {
    /// Several, in order: Space toggles, `J`/`K` reorder, Enter keeps them.
    Multi,
    /// One: Enter picks the entry under the cursor.
    Single,
}

/// One entry of the picker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::tui) struct Entry {
    /// What choosing it gives, such as a GPU type ID or an image.
    pub(in crate::tui) id: String,
    /// Its cells, one per column of the header.
    pub(in crate::tui) columns: Vec<String>,
    /// Whether it can be chosen; drawn dim when not.
    pub(in crate::tui) selectable: bool,
}

/// What a picker holds chosen, or kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::tui) enum Choice {
    /// The `auto` entry.
    Auto,
    /// These IDs, in order; one at most in [`Mode::Single`].
    List(Vec<String>),
}

/// What a key did to a picker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::tui) enum PickerOutcome {
    /// Still open.
    Open,
    /// Enter: this choice is kept.
    Kept(Choice),
    /// Esc: nothing changes.
    Cancelled,
}

/// What a picker shows, besides its entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::tui) struct Spec {
    /// The title.
    pub(in crate::tui) title: &'static str,
    /// The column names.
    pub(in crate::tui) header: &'static [&'static str],
    /// The column holding an entry's ID: an entry chosen but not listed shows
    /// its ID there.
    pub(in crate::tui) id_column: usize,
    /// How entries are chosen.
    pub(in crate::tui) mode: Mode,
    /// What the `auto` entry at the top says, when there is one.
    pub(in crate::tui) auto: Option<&'static str>,
    /// What shows when there is no entry at all.
    pub(in crate::tui) empty: &'static str,
}

/// The entries: being read, read, or why they cannot be.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Load {
    Loading,
    Ready(Vec<Entry>),
    Failed(String),
}

/// A row shown: the `auto` entry, or an entry by its index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Row {
    Auto,
    Entry(usize),
}

/// A picker's state. Its rows are the `auto` entry, then in
/// [`Mode::Multi`] the chosen entries in their order, then the others in the
/// order they came in, each kept only when it matches the filter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::tui) struct Picker {
    spec: Spec,
    load: Load,
    /// Whether the `auto` entry is chosen.
    auto: bool,
    /// The chosen IDs, in order.
    chosen: Vec<String>,
    /// The row under the cursor.
    cursor: usize,
    /// The filter, while there is one.
    filter: Option<Input>,
    /// Whether the filter is being typed.
    typing: bool,
}

impl Picker {
    /// A picker showing `spec`, with `preselected` chosen, its entries still
    /// being read. `auto` is dropped without an `auto` entry, and all but the
    /// first ID in [`Mode::Single`].
    pub(in crate::tui) fn new(spec: Spec, preselected: Choice) -> Self {
        let (auto, chosen) = match preselected {
            Choice::Auto => (spec.auto.is_some(), Vec::new()),
            Choice::List(ids) => (false, ids),
        };
        let mut unique: Vec<String> = Vec::new();
        for id in chosen {
            if !unique.contains(&id) {
                unique.push(id);
            }
        }
        if spec.mode == Mode::Single {
            unique.truncate(1);
        }
        Self {
            spec,
            load: Load::Loading,
            auto,
            chosen: unique,
            cursor: 0,
            filter: None,
            typing: false,
        }
    }

    /// Shows the entries read, or why they cannot be. A chosen ID no entry
    /// has gets an entry of its own, so it stays in view and can be taken out.
    /// The cursor goes to the first chosen row.
    pub(in crate::tui) fn loaded(&mut self, entries: Result<Vec<Entry>, String>) {
        let mut entries = match entries {
            Ok(entries) => entries,
            Err(error) => {
                self.load = Load::Failed(error);
                return;
            },
        };
        let width = self.spec.header.len().max(self.spec.id_column + 1);
        for id in &self.chosen {
            if !entries.iter().any(|entry| &entry.id == id) {
                let mut columns = vec!["-".to_string(); width];
                if let Some(cell) = columns.get_mut(self.spec.id_column) {
                    cell.clone_from(id);
                }
                entries.push(Entry {
                    id: id.clone(),
                    columns,
                    selectable: true,
                });
            }
        }
        self.load = Load::Ready(entries);
        let rows = self.rows();
        self.cursor = rows
            .iter()
            .position(|row| self.is_chosen(*row))
            .unwrap_or(0);
    }

    /// Whether the entries are still being read.
    pub(in crate::tui) fn loading(&self) -> bool {
        self.load == Load::Loading
    }

    /// Why the entries cannot be read, if they cannot.
    pub(in crate::tui) fn error(&self) -> Option<&str> {
        match &self.load {
            Load::Failed(error) => Some(error),
            Load::Loading | Load::Ready(_) => None,
        }
    }

    /// How entries are chosen.
    pub(in crate::tui) fn mode(&self) -> Mode {
        self.spec.mode
    }

    /// Whether the filter is being typed.
    pub(in crate::tui) fn typing(&self) -> bool {
        self.typing
    }

    /// What is chosen now.
    pub(in crate::tui) fn choice(&self) -> Choice {
        if self.auto {
            Choice::Auto
        } else {
            Choice::List(self.chosen.clone())
        }
    }

    /// The entry of `id`, once the entries are read.
    pub(in crate::tui) fn entry(&self, id: &str) -> Option<&Entry> {
        self.entries().iter().find(|entry| entry.id == id)
    }

    /// Handles `code`: moves, toggles, reorders, filters, keeps or cancels.
    /// Until the entries are read, only Esc does something.
    pub(in crate::tui) fn on_key(&mut self, code: KeyCode) -> PickerOutcome {
        if self.typing {
            self.on_filter_key(code);
            return PickerOutcome::Open;
        }
        if code == KeyCode::Esc {
            return PickerOutcome::Cancelled;
        }
        if !matches!(self.load, Load::Ready(_)) {
            return PickerOutcome::Open;
        }
        let last = self.rows().len().saturating_sub(1);
        match code {
            KeyCode::Up | KeyCode::Char('k') => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => self.cursor = (self.cursor + 1).min(last),
            KeyCode::PageUp => self.cursor = self.cursor.saturating_sub(usize::from(PAGE)),
            KeyCode::PageDown => self.cursor = (self.cursor + usize::from(PAGE)).min(last),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = last,
            KeyCode::Char(' ') if self.spec.mode == Mode::Multi => self.toggle(),
            KeyCode::Char('J') if self.spec.mode == Mode::Multi => self.shift(true),
            KeyCode::Char('K') if self.spec.mode == Mode::Multi => self.shift(false),
            KeyCode::Char('/') => {
                self.typing = true;
                self.filter.get_or_insert_with(|| Input::new(""));
            },
            KeyCode::Enter => return self.enter(),
            _ => {},
        }
        PickerOutcome::Open
    }

    /// Pastes `text` into the filter while it is typed; ignored otherwise.
    pub(in crate::tui) fn paste(&mut self, text: &str) {
        if self.typing
            && let Some(input) = &mut self.filter
        {
            input.paste(text);
            self.cursor = self.cursor.min(self.rows().len().saturating_sub(1));
        }
    }

    /// A key while the filter is typed: Enter keeps it (an empty one goes),
    /// Esc clears it; the cursor stays on a row.
    fn on_filter_key(&mut self, code: KeyCode) {
        let Some(input) = &mut self.filter else {
            self.typing = false;
            return;
        };
        match input.on_key(code) {
            InputOutcome::Editing => {},
            InputOutcome::Done(text) => {
                self.typing = false;
                if text.is_empty() {
                    self.filter = None;
                }
            },
            InputOutcome::Cancelled => {
                self.typing = false;
                self.filter = None;
            },
        }
        self.cursor = self.cursor.min(self.rows().len().saturating_sub(1));
    }

    /// Enter: in [`Mode::Multi`] keeps what is chosen, even nothing; in
    /// [`Mode::Single`] picks the row under the cursor, when it can be chosen.
    fn enter(&self) -> PickerOutcome {
        if self.spec.mode == Mode::Multi {
            return PickerOutcome::Kept(self.choice());
        }
        match self.rows().get(self.cursor) {
            Some(Row::Auto) => PickerOutcome::Kept(Choice::Auto),
            Some(Row::Entry(index)) => match self.entries().get(*index) {
                Some(entry) if entry.selectable => {
                    PickerOutcome::Kept(Choice::List(vec![entry.id.clone()]))
                },
                _ => PickerOutcome::Open,
            },
            None => PickerOutcome::Open,
        }
    }

    /// Space: chooses the row under the cursor, or takes it out. Choosing
    /// `auto` takes every entry out; choosing an entry takes `auto` out. The
    /// cursor follows the row.
    fn toggle(&mut self) {
        let Some(row) = self.rows().get(self.cursor).copied() else {
            return;
        };
        match row {
            Row::Auto => {
                self.auto = !self.auto;
                if self.auto {
                    self.chosen.clear();
                }
            },
            Row::Entry(index) => {
                let Some(entry) = self.entries().get(index) else {
                    return;
                };
                if !entry.selectable {
                    return;
                }
                let id = entry.id.clone();
                if let Some(at) = self.chosen.iter().position(|chosen| *chosen == id) {
                    self.chosen.remove(at);
                } else {
                    self.chosen.push(id);
                    self.auto = false;
                }
            },
        }
        self.follow(row);
    }

    /// `J` (`down`) or `K`: moves the chosen entry under the cursor one place
    /// later or earlier in the chosen order; the cursor follows it.
    fn shift(&mut self, down: bool) {
        let Some(Row::Entry(index)) = self.rows().get(self.cursor).copied() else {
            return;
        };
        let Some(entry) = self.entries().get(index) else {
            return;
        };
        let Some(at) = self.chosen.iter().position(|id| *id == entry.id) else {
            return;
        };
        let to = if down {
            at + 1
        } else {
            match at.checked_sub(1) {
                Some(to) => to,
                None => return,
            }
        };
        if to < self.chosen.len() {
            self.chosen.swap(at, to);
            self.follow(Row::Entry(index));
        }
    }

    /// Puts the cursor on `row`, wherever it is now.
    fn follow(&mut self, row: Row) {
        if let Some(at) = self.rows().iter().position(|shown| *shown == row) {
            self.cursor = at;
        }
    }

    /// The entries read, none until then.
    fn entries(&self) -> &[Entry] {
        match &self.load {
            Load::Ready(entries) => entries,
            Load::Loading | Load::Failed(_) => &[],
        }
    }

    /// The filter's text, empty without one.
    fn filter_text(&self) -> &str {
        self.filter.as_ref().map_or("", Input::text)
    }

    /// Whether `row` is chosen.
    fn is_chosen(&self, row: Row) -> bool {
        match row {
            Row::Auto => self.auto,
            Row::Entry(index) => self
                .entries()
                .get(index)
                .is_some_and(|entry| self.chosen.contains(&entry.id)),
        }
    }

    /// The rows shown, in order (see [`Picker`]).
    fn rows(&self) -> Vec<Row> {
        let needle = self.filter_text().to_lowercase();
        let matches = |text: &str| needle.is_empty() || text.to_lowercase().contains(&needle);
        let entries = self.entries();
        let mut rows = Vec::new();
        if let Some(note) = self.spec.auto
            && (matches(AUTO) || matches(note))
            && matches!(self.load, Load::Ready(_))
        {
            rows.push(Row::Auto);
        }
        let shown = |entry: &Entry| entry.columns.iter().any(|cell| matches(cell));
        let mut first: Vec<usize> = Vec::new();
        if self.spec.mode == Mode::Multi {
            for id in &self.chosen {
                if let Some(index) = entries.iter().position(|entry| &entry.id == id)
                    && shown(&entries[index])
                {
                    first.push(index);
                }
            }
        }
        rows.extend(first.iter().map(|index| Row::Entry(*index)));
        rows.extend(
            entries
                .iter()
                .enumerate()
                .filter(|(index, entry)| !first.contains(index) && shown(entry))
                .map(|(index, _)| Row::Entry(index)),
        );
        rows
    }

    /// The mark of `row`: its place in the chosen order in [`Mode::Multi`],
    /// `●` for a chosen `auto` or the entry chosen in [`Mode::Single`].
    fn mark(&self, row: Row) -> String {
        match row {
            Row::Auto if self.auto => "●".to_string(),
            Row::Entry(index) => {
                let place = self
                    .entries()
                    .get(index)
                    .and_then(|entry| self.chosen.iter().position(|id| *id == entry.id));
                match (place, self.spec.mode) {
                    (Some(place), Mode::Multi) => (place + 1).to_string(),
                    (Some(_), Mode::Single) => "●".to_string(),
                    (None, _) => String::new(),
                }
            },
            Row::Auto => String::new(),
        }
    }
}

/// Draws `picker` centered over `area`, `spinner` turning while its entries
/// are read; returns where.
pub(in crate::tui) fn render(
    frame: &mut Frame,
    area: Rect,
    picker: &Picker,
    theme: &Theme,
    spinner: &str,
) -> Rect {
    let width = area.width.saturating_sub(2 * MARGIN).min(MAX_WIDTH);
    let inner_width = width.saturating_sub(4);
    let room = area.height.saturating_sub(2 * MARGIN_ROWS + 2);
    let rows = picker.rows();
    let filter_rows = u16::from(picker.filter.is_some());
    let body_rows = match &picker.load {
        Load::Loading => 1,
        Load::Failed(error) => {
            let text = Paragraph::new(error.as_str()).wrap(Wrap { trim: true });
            u16::try_from(text.line_count(inner_width)).unwrap_or(u16::MAX)
        },
        Load::Ready(entries) if entries.is_empty() => 1,
        Load::Ready(_) => u16::try_from(rows.len().max(1) + 1).unwrap_or(u16::MAX),
    };
    let height = body_rows
        .saturating_add(filter_rows)
        .min(room)
        .saturating_add(2);
    let popup = centered(area, width, height);
    let chosen = picker.chosen.len();
    let title = match (picker.spec.mode, picker.auto) {
        (Mode::Multi, true) => format!(" {}: {AUTO} ", picker.spec.title),
        (Mode::Multi, false) if chosen > 0 => format!(" {}: {chosen} chosen ", picker.spec.title),
        _ => format!(" {} ", picker.spec.title),
    };
    let keys = match (&picker.load, picker.spec.mode) {
        (Load::Ready(_), Mode::Multi) => " Space toggles, Enter keeps, Esc cancels ",
        (Load::Ready(_), Mode::Single) => " Enter picks, Esc cancels ",
        _ => " Esc closes ",
    };
    let block = overlay(frame, popup, theme)
        .title(Span::styled(title, theme.title))
        .title_bottom(Span::styled(keys, theme.dim))
        .padding(Padding::horizontal(1));
    let inner = block.inner(popup);
    frame.render_widget(block, popup);
    let [body, filter] =
        Layout::vertical([Constraint::Fill(1), Constraint::Length(filter_rows)]).areas(inner);
    match &picker.load {
        Load::Loading => {
            let line = Line::from(vec![
                Span::styled(format!("{spinner} "), theme.accent),
                Span::styled("reading the Runpod catalog", theme.dim),
            ]);
            frame.render_widget(Paragraph::new(line), body);
        },
        Load::Failed(error) => {
            let text = Paragraph::new(error.as_str())
                .style(theme.error)
                .wrap(Wrap { trim: true });
            frame.render_widget(text, body);
        },
        Load::Ready(entries) if entries.is_empty() => {
            frame.render_widget(
                Paragraph::new(Span::styled(picker.spec.empty, theme.dim)),
                body,
            );
        },
        Load::Ready(_) if rows.is_empty() => {
            let none = Span::styled("nothing matches the filter", theme.dim);
            frame.render_widget(Paragraph::new(none), body);
        },
        Load::Ready(_) => render_table(frame, body, picker, &rows, theme),
    }
    if let Some(input) = &picker.filter {
        let [slash, text] =
            Layout::horizontal([Constraint::Length(2), Constraint::Fill(1)]).areas(filter);
        frame.render_widget(Paragraph::new(Span::styled("/ ", theme.key)), slash);
        if picker.typing {
            input.render(frame, text, theme.text_hi);
        } else {
            frame.render_widget(Paragraph::new(Span::styled(input.text(), theme.dim)), text);
        }
    }
    popup
}

/// Draws `rows` of `picker` as a table in `area`, columns as wide as their
/// widest cell, the last one taking what is left.
fn render_table(frame: &mut Frame, area: Rect, picker: &Picker, rows: &[Row], theme: &Theme) {
    let entries = picker.entries();
    let header = picker.spec.header;
    // The `auto` row says what it does in the first column.
    let cells = |row: Row| -> Vec<String> {
        match row {
            Row::Auto => vec![format!("{AUTO}: {}", picker.spec.auto.unwrap_or(""))],
            Row::Entry(index) => entries
                .get(index)
                .map(|entry| entry.columns.clone())
                .unwrap_or_default(),
        }
    };
    let shown: Vec<(Row, Vec<String>)> = rows.iter().map(|row| (*row, cells(*row))).collect();
    let mut widths: Vec<usize> = header.iter().map(|name| name.chars().count()).collect();
    for (_, cells) in &shown {
        for (at, cell) in cells.iter().enumerate() {
            if let Some(width) = widths.get_mut(at) {
                *width = (*width).max(cell.chars().count());
            }
        }
    }
    let mut constraints = vec![Constraint::Length(2)];
    let last = widths.len().saturating_sub(1);
    for (at, width) in widths.iter().enumerate() {
        constraints.push(if at == last {
            Constraint::Fill(1)
        } else {
            Constraint::Length(u16::try_from(*width).unwrap_or(u16::MAX))
        });
    }
    let table_rows: Vec<TableRow> = shown
        .into_iter()
        .map(|(row, cells)| {
            let mark = Cell::from(Span::styled(picker.mark(row), theme.accent));
            let Row::Entry(index) = row else {
                let note = format!(": {}", picker.spec.auto.unwrap_or(""));
                let auto = Line::from(vec![
                    Span::styled(AUTO, theme.key),
                    Span::styled(note, theme.dim),
                ]);
                return TableRow::new(vec![mark, Cell::from(auto)]);
            };
            let selectable = entries.get(index).is_some_and(|entry| entry.selectable);
            let style = if selectable { theme.text_hi } else { theme.dim };
            let line = std::iter::once(mark).chain(cells.into_iter().map(Cell::from));
            TableRow::new(line.collect::<Vec<_>>()).style(style)
        })
        .collect();
    let header_row = TableRow::new(
        std::iter::once(Cell::from(""))
            .chain(header.iter().map(|name| Cell::from(*name)))
            .collect::<Vec<_>>(),
    )
    .style(theme.dim);
    let table = Table::new(table_rows, constraints)
        .header(header_row)
        .column_spacing(2)
        .row_highlight_style(theme.selected)
        .highlight_symbol("▶ ");
    let mut state = TableState::default().with_selected(Some(picker.cursor));
    frame.render_stateful_widget(table, area, &mut state);
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &[&str] = &["ID", "VRAM GB"];

    fn spec(mode: Mode, auto: bool) -> Spec {
        Spec {
            title: "GPU types",
            header: HEADER,
            id_column: 0,
            mode,
            auto: auto.then_some("chosen at start"),
            empty: "nothing listed",
        }
    }

    fn entry(id: &str, selectable: bool) -> Entry {
        Entry {
            id: id.into(),
            columns: vec![id.into(), "48".into()],
            selectable,
        }
    }

    /// A picker on `a`, `b`, `c` (`c` cannot be chosen), `chosen` preselected.
    fn picker(mode: Mode, auto: bool, chosen: Choice) -> Picker {
        let mut picker = Picker::new(spec(mode, auto), chosen);
        picker.loaded(Ok(vec![
            entry("a", true),
            entry("b", true),
            entry("c", false),
        ]));
        picker
    }

    fn list(ids: &[&str]) -> Choice {
        Choice::List(ids.iter().map(|id| (*id).to_string()).collect())
    }

    fn keys(picker: &mut Picker, codes: &[KeyCode]) -> PickerOutcome {
        let mut outcome = PickerOutcome::Open;
        for code in codes {
            outcome = picker.on_key(*code);
        }
        outcome
    }

    fn ids(picker: &Picker) -> Vec<String> {
        picker
            .rows()
            .iter()
            .map(|row| match row {
                Row::Auto => AUTO.to_string(),
                Row::Entry(index) => picker.entries()[*index].id.clone(),
            })
            .collect()
    }

    #[test]
    fn space_toggles_in_order_and_enter_keeps() {
        let mut picker = picker(Mode::Multi, false, list(&[]));
        let outcome = keys(
            &mut picker,
            &[
                KeyCode::Down,
                KeyCode::Char(' '),
                KeyCode::Char('j'),
                KeyCode::Char(' '),
                KeyCode::Enter,
            ],
        );
        // `b` chosen first moves to the top; the cursor follows it, then `a`.
        assert_eq!(outcome, PickerOutcome::Kept(list(&["b", "a"])));
        assert_eq!(ids(&picker), ["b", "a", "c"]);
        keys(&mut picker, &[KeyCode::Home, KeyCode::Char(' ')]);
        assert_eq!(picker.choice(), list(&["a"]), "Space takes it out");
    }

    #[test]
    fn an_entry_that_cannot_be_chosen_is_not() {
        let mut picker = picker(Mode::Multi, false, list(&[]));
        keys(&mut picker, &[KeyCode::End, KeyCode::Char(' ')]);
        assert_eq!(picker.choice(), list(&[]));
        let mut single = picker_single();
        assert_eq!(
            keys(&mut single, &[KeyCode::End, KeyCode::Enter]),
            PickerOutcome::Open
        );
    }

    fn picker_single() -> Picker {
        picker(Mode::Single, false, list(&[]))
    }

    #[test]
    fn capital_j_and_k_move_the_entry_in_the_chosen_order() {
        let mut picker = picker(Mode::Multi, false, list(&["a", "b"]));
        assert_eq!(picker.cursor, 0, "the cursor starts on the first chosen");
        keys(&mut picker, &[KeyCode::Char('J')]);
        assert_eq!(picker.choice(), list(&["b", "a"]));
        assert_eq!(ids(&picker), ["b", "a", "c"]);
        assert_eq!(picker.cursor, 1, "the cursor follows the entry");
        keys(&mut picker, &[KeyCode::Char('J')]);
        assert_eq!(picker.choice(), list(&["b", "a"]), "already last");
        keys(&mut picker, &[KeyCode::Char('K'), KeyCode::Char('K')]);
        assert_eq!(picker.choice(), list(&["a", "b"]));
        assert_eq!(picker.cursor, 0);
        keys(&mut picker, &[KeyCode::End, KeyCode::Char('K')]);
        assert_eq!(picker.choice(), list(&["a", "b"]), "c is not chosen");
    }

    #[test]
    fn slash_filters_enter_keeps_the_filter_and_esc_clears_it() {
        let mut picker = picker(Mode::Multi, true, list(&[]));
        assert_eq!(ids(&picker), [AUTO, "a", "b", "c"]);
        keys(&mut picker, &[KeyCode::Char('/'), KeyCode::Char('B')]);
        assert!(picker.typing());
        assert_eq!(ids(&picker), ["b"], "case does not matter");
        keys(&mut picker, &[KeyCode::Enter]);
        assert!(!picker.typing());
        assert_eq!(ids(&picker), ["b"], "Enter keeps the filter");
        assert_eq!(
            keys(&mut picker, &[KeyCode::Char(' '), KeyCode::Enter]),
            PickerOutcome::Kept(list(&["b"]))
        );
        keys(&mut picker, &[KeyCode::Char('/'), KeyCode::Esc]);
        assert_eq!(ids(&picker), [AUTO, "b", "a", "c"]);
        assert!(picker.filter.is_none());
        keys(&mut picker, &[KeyCode::Char('/'), KeyCode::Char('z')]);
        assert!(ids(&picker).is_empty());
        keys(&mut picker, &[KeyCode::Backspace, KeyCode::Enter]);
        assert!(picker.filter.is_none(), "an empty filter goes");
    }

    #[test]
    fn auto_excludes_the_entries() {
        let mut picker = picker(Mode::Multi, true, list(&["a"]));
        assert_eq!(ids(&picker), [AUTO, "a", "b", "c"]);
        assert_eq!(picker.cursor, 1);
        keys(&mut picker, &[KeyCode::Home, KeyCode::Char(' ')]);
        assert_eq!(picker.choice(), Choice::Auto);
        assert_eq!(
            keys(&mut picker, &[KeyCode::Enter]),
            PickerOutcome::Kept(Choice::Auto)
        );
        keys(&mut picker, &[KeyCode::Down, KeyCode::Char(' ')]);
        assert_eq!(picker.choice(), list(&["a"]));
        let preselected = self::picker(Mode::Multi, true, Choice::Auto);
        assert_eq!(
            (preselected.choice(), preselected.cursor),
            (Choice::Auto, 0)
        );
        let without = self::picker(Mode::Multi, false, Choice::Auto);
        assert_eq!(without.choice(), list(&[]), "no auto entry, no auto");
    }

    #[test]
    fn single_choice_picks_the_row_under_the_cursor() {
        let mut picker = picker(Mode::Single, false, list(&["b"]));
        assert_eq!(ids(&picker), ["a", "b", "c"], "the catalog order stays");
        assert_eq!(picker.cursor, 1);
        assert_eq!(picker.mark(Row::Entry(1)), "●");
        keys(&mut picker, &[KeyCode::Char(' '), KeyCode::Char('J')]);
        assert_eq!(picker.choice(), list(&["b"]), "no toggle, no order");
        assert_eq!(
            keys(&mut picker, &[KeyCode::Up, KeyCode::Enter]),
            PickerOutcome::Kept(list(&["a"]))
        );
    }

    #[test]
    fn esc_cancels_even_while_loading_and_nothing_else_does() {
        let mut picker = Picker::new(spec(Mode::Multi, true), list(&["a"]));
        assert!(picker.loading());
        assert_eq!(
            keys(&mut picker, &[KeyCode::Char(' '), KeyCode::Enter]),
            PickerOutcome::Open
        );
        assert_eq!(keys(&mut picker, &[KeyCode::Esc]), PickerOutcome::Cancelled);
        picker.loaded(Err("no Runpod API key".into()));
        assert_eq!(picker.error(), Some("no Runpod API key"));
        assert_eq!(keys(&mut picker, &[KeyCode::Enter]), PickerOutcome::Open);
        assert_eq!(keys(&mut picker, &[KeyCode::Esc]), PickerOutcome::Cancelled);
    }

    #[test]
    fn the_row_under_the_cursor_stays_in_view_in_a_long_list()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut picker = Picker::new(spec(Mode::Single, false), list(&[]));
        picker.loaded(Ok((0..40)
            .map(|n| entry(&format!("gpu-{n:02}"), true))
            .collect()));
        keys(&mut picker, &[KeyCode::End]);
        let theme = Theme::mono();
        let mut terminal = ratatui::Terminal::new(ratatui::backend::TestBackend::new(80, 24))?;
        terminal.draw(|frame| {
            render(frame, frame.area(), &picker, &theme, "*");
        })?;
        let screen = terminal.backend().to_string();
        let cursor = screen.lines().find(|line| line.contains('▶'));
        assert!(
            cursor.is_some_and(|line| line.contains("gpu-39")),
            "{screen}"
        );
        assert!(!screen.contains("gpu-00"), "{screen}");
        Ok(())
    }

    #[test]
    fn a_chosen_id_the_catalog_lacks_stays_in_view() {
        let mut picker = picker(Mode::Multi, false, list(&["gone", "a", "gone"]));
        assert_eq!(picker.choice(), list(&["gone", "a"]), "no duplicate");
        assert_eq!(ids(&picker), ["gone", "a", "b", "c"]);
        let gone = picker.entries().iter().find(|entry| entry.id == "gone");
        assert_eq!(
            gone.map(|entry| entry.columns.clone()),
            Some(vec!["gone".to_string(), "-".to_string()])
        );
        keys(&mut picker, &[KeyCode::Char(' ')]);
        assert_eq!(picker.choice(), list(&["a"]));
    }
}
