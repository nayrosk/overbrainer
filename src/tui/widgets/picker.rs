//! The picker overlay: entries in columns, several chosen in order (Space
//! toggles, `J`/`K` move the entry under the cursor in that order) or one
//! picked with Enter, filtered with `/`, sorted another way with `o`. The entries arrive later: a spinner
//! shows until then, and an error in their place when they cannot be read.

use crossterm::event::KeyCode;
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Cell, Padding, Paragraph, Row as TableRow, Table, TableState, Wrap};

use super::form::{Input, InputOutcome};
use super::{centered, overlay};
use crate::config::ListOrAuto;
use crate::runpod::printable;
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
    /// Its place in each order of [`Spec::orders`] after the first, which is
    /// the order entries come in; an entry without one comes last.
    pub(in crate::tui) ranks: Vec<usize>,
}

/// What a picker holds chosen, or kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::tui) enum Choice {
    /// The `auto` entry.
    Auto,
    /// These IDs, in order; one at most in [`Mode::Single`].
    List(Vec<String>),
}

impl From<&ListOrAuto> for Choice {
    /// `auto` as the `auto` entry, a list as its IDs.
    fn from(value: &ListOrAuto) -> Self {
        match value {
            ListOrAuto::Auto => Self::Auto,
            ListOrAuto::List(ids) => Self::List(ids.clone()),
        }
    }
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
    /// `t`: the value is to be typed instead, in what opened the picker.
    Typed,
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
    /// The names of the orders `o` cycles through, the first being the order
    /// entries come in; with fewer than two, `o` does nothing.
    pub(in crate::tui) orders: &'static [&'static str],
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
/// order shown (see [`Spec::orders`]), each kept only when it matches the
/// filter.
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
    /// Whether `t` asks to type the value instead.
    typed: bool,
    /// The order shown, an index into [`Spec::orders`].
    order: usize,
    /// What the title adds once the entries are read, such as the VRAM a
    /// run needs.
    note: Option<String>,
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
            typed: true,
            order: 0,
            note: None,
        }
    }

    /// The picker without `t`: what it keeps cannot be typed instead.
    pub(in crate::tui) fn untyped(mut self) -> Self {
        self.typed = false;
        self
    }

    /// Whether `t` asks to type the value instead.
    pub(in crate::tui) fn typed(&self) -> bool {
        self.typed
    }

    /// Shows the entries read, or why they cannot be; their cells, from the
    /// Runpod API, are made [`printable`]. A chosen ID no entry has gets an
    /// entry of its own, so it stays in view and can be taken out. The cursor
    /// goes to the first chosen row.
    pub(in crate::tui) fn loaded(&mut self, entries: Result<Vec<Entry>, String>) {
        let mut entries = match entries {
            Ok(mut entries) => {
                for cell in entries.iter_mut().flat_map(|entry| &mut entry.columns) {
                    *cell = printable(cell);
                }
                entries
            },
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
                    ranks: Vec::new(),
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

    /// Sets what the title adds.
    pub(in crate::tui) fn set_note(&mut self, note: Option<String>) {
        self.note = note;
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

    /// Whether `o` sorts the entries another way.
    pub(in crate::tui) fn sortable(&self) -> bool {
        self.spec.orders.len() > 1
    }

    /// The name of the order shown, when there are several.
    fn order_name(&self) -> Option<&'static str> {
        self.spec
            .orders
            .get(self.order)
            .copied()
            .filter(|_| self.sortable())
    }

    /// Handles `code`: moves, toggles, reorders, sorts, filters, keeps or cancels;
    /// `t` asks to type the value instead, unless [`Self::untyped`]. Until
    /// the entries are read, only Esc and `t` do something.
    pub(in crate::tui) fn on_key(&mut self, code: KeyCode) -> PickerOutcome {
        if self.typing {
            self.on_filter_key(code);
            return PickerOutcome::Open;
        }
        if code == KeyCode::Esc {
            return PickerOutcome::Cancelled;
        }
        if code == KeyCode::Char('t') && self.typed {
            return PickerOutcome::Typed;
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
            KeyCode::Char('o') if self.sortable() => self.sort(),
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
                let id = entry.id.clone();
                // One that cannot be chosen can still be taken out.
                if let Some(at) = self.chosen.iter().position(|chosen| *chosen == id) {
                    self.chosen.remove(at);
                } else if !entry.selectable {
                    return;
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

    /// `o`: shows the next order, after the last the first; the cursor stays
    /// on its row.
    fn sort(&mut self) {
        let row = self.rows().get(self.cursor).copied();
        self.order = (self.order + 1) % self.spec.orders.len();
        if let Some(row) = row {
            self.follow(row);
        }
    }

    /// The indices of the entries in the order shown: by their rank there,
    /// ties (and the first order) in the order they came in.
    fn sorted(&self) -> Vec<usize> {
        let entries = self.entries();
        let mut indices: Vec<usize> = (0..entries.len()).collect();
        if let Some(at) = self.order.checked_sub(1) {
            let rank = |index: usize| {
                entries
                    .get(index)
                    .and_then(|entry| entry.ranks.get(at))
                    .copied()
                    .unwrap_or(usize::MAX)
            };
            indices.sort_by_key(|index| (rank(*index), *index));
        }
        indices
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
            self.sorted()
                .into_iter()
                .filter(|index| !first.contains(index) && entries.get(*index).is_some_and(shown))
                .map(Row::Entry),
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
    let title = title(picker);
    let keys = keys(picker);
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

/// The title of `picker`: its name, the order shown when there are several,
/// its note, and what is chosen in [`Mode::Multi`].
fn title(picker: &Picker) -> String {
    let mut name = match picker.order_name() {
        Some(order) => format!("{} by {order}", picker.spec.title),
        None => picker.spec.title.to_string(),
    };
    if let Some(note) = &picker.note {
        name = format!("{name} ({note})");
    }
    let chosen = picker.chosen.len();
    match (picker.spec.mode, picker.auto) {
        (Mode::Multi, true) => format!(" {name}: {AUTO} "),
        (Mode::Multi, false) if chosen > 0 => format!(" {name}: {chosen} chosen "),
        _ => format!(" {name} "),
    }
}

/// The keys of `picker` in its state, for its bottom border.
fn keys(picker: &Picker) -> String {
    let mut keys: Vec<&str> = Vec::new();
    if matches!(picker.load, Load::Ready(_)) {
        keys.push(match picker.spec.mode {
            Mode::Multi => "Space toggles, Enter keeps",
            Mode::Single => "Enter picks",
        });
        if picker.sortable() {
            keys.push("o sorts");
        }
        if picker.typed {
            keys.push("t types");
        }
        keys.push("Esc cancels");
    } else {
        if picker.typed {
            keys.push("t types the value");
        }
        keys.push("Esc closes");
    }
    format!(" {} ", keys.join(", "))
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
            orders: &[],
        }
    }

    fn entry(id: &str, selectable: bool) -> Entry {
        Entry {
            id: id.into(),
            columns: vec![id.into(), "48".into()],
            selectable,
            ranks: Vec::new(),
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

    #[test]
    fn t_asks_to_type_the_value_in_any_state_but_the_filter() {
        let mut loading = Picker::new(spec(Mode::Multi, true), list(&[]));
        assert_eq!(
            keys(&mut loading, &[KeyCode::Char('t')]),
            PickerOutcome::Typed
        );
        let mut failed = Picker::new(spec(Mode::Single, false), list(&[]));
        failed.loaded(Err("no key".into()));
        assert_eq!(
            keys(&mut failed, &[KeyCode::Char('t')]),
            PickerOutcome::Typed
        );
        let mut single = picker_single();
        assert_eq!(
            keys(&mut single, &[KeyCode::Char('t')]),
            PickerOutcome::Typed
        );
        let mut filtered = picker_single();
        let outcome = keys(&mut filtered, &[KeyCode::Char('/'), KeyCode::Char('t')]);
        assert_eq!(outcome, PickerOutcome::Open, "typed in the filter");
        assert_eq!(filtered.filter_text(), "t");
    }

    #[test]
    fn an_untyped_picker_ignores_t() {
        let mut loading = Picker::new(spec(Mode::Multi, true), list(&[])).untyped();
        assert!(!loading.typed());
        assert_eq!(
            keys(&mut loading, &[KeyCode::Char('t')]),
            PickerOutcome::Open
        );
        let mut single = picker_single().untyped();
        assert_eq!(
            keys(&mut single, &[KeyCode::Char('t'), KeyCode::Enter]),
            PickerOutcome::Kept(list(&["a"]))
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

    #[test]
    fn entries_are_drawn_without_control_characters() -> Result<(), Box<dyn std::error::Error>> {
        let mut picker = Picker::new(spec(Mode::Single, false), list(&[]));
        picker.loaded(Ok(vec![Entry {
            id: "odd".into(),
            columns: vec!["odd\u{1b}[2J\nname".into(), "48".into()],
            selectable: true,
            ranks: Vec::new(),
        }]));
        assert_eq!(
            picker.entry("odd").map(|entry| entry.columns.clone()),
            Some(vec!["odd name".to_string(), "48".to_string()])
        );
        let theme = Theme::mono();
        let mut terminal = ratatui::Terminal::new(ratatui::backend::TestBackend::new(80, 24))?;
        terminal.draw(|frame| {
            render(frame, frame.area(), &picker, &theme, "*");
        })?;
        let screen = terminal.backend().to_string();
        assert!(screen.contains("odd name"), "{screen}");
        Ok(())
    }

    /// A picker on `a`, `b`, `c` ordered by name, `c`, `a`, `b` by size, with
    /// `chosen` preselected.
    fn sortable(mode: Mode, chosen: Choice) -> Picker {
        let mut spec = spec(mode, false);
        spec.orders = &["name", "size"];
        let mut picker = Picker::new(spec, chosen);
        let ranked = |id: &str, rank: usize| Entry {
            ranks: vec![rank],
            ..entry(id, true)
        };
        picker.loaded(Ok(vec![ranked("a", 1), ranked("b", 2), ranked("c", 0)]));
        picker
    }

    #[test]
    fn o_cycles_the_orders_and_the_cursor_stays_on_its_row() {
        let mut picker = sortable(Mode::Single, list(&["b"]));
        assert_eq!(picker.order_name(), Some("name"));
        assert_eq!(title(&picker), " GPU types by name ");
        assert_eq!(ids(&picker), ["a", "b", "c"]);
        keys(&mut picker, &[KeyCode::Char('o')]);
        assert_eq!(ids(&picker), ["c", "a", "b"]);
        assert_eq!(title(&picker), " GPU types by size ");
        assert_eq!(picker.cursor, 2, "still on b");
        keys(&mut picker, &[KeyCode::Char('o')]);
        assert_eq!(ids(&picker), ["a", "b", "c"], "back to the first order");
        assert!(self::keys_hint(&picker).contains("o sorts"));
    }

    fn keys_hint(picker: &Picker) -> String {
        super::keys(picker)
    }

    #[test]
    fn chosen_entries_stay_first_in_any_order() {
        let mut picker = sortable(Mode::Multi, list(&["b"]));
        keys(&mut picker, &[KeyCode::Char('o')]);
        assert_eq!(ids(&picker), ["b", "c", "a"]);
        assert_eq!(title(&picker), " GPU types by size: 1 chosen ");
    }

    #[test]
    fn the_note_shows_in_the_title() {
        let mut picker = sortable(Mode::Multi, list(&["b"]));
        picker.set_note(Some("about 19.1 GB per GPU needed".into()));
        assert_eq!(
            title(&picker),
            " GPU types by name (about 19.1 GB per GPU needed): 1 chosen "
        );
    }

    #[test]
    fn an_entry_that_cannot_be_chosen_can_be_taken_out() {
        let mut picker = picker(Mode::Multi, false, list(&["c"]));
        keys(&mut picker, &[KeyCode::Char(' ')]);
        assert_eq!(picker.choice(), list(&[]), "c taken out");
        keys(&mut picker, &[KeyCode::End, KeyCode::Char(' ')]);
        assert_eq!(picker.choice(), list(&[]), "c cannot be chosen again");
    }

    #[test]
    fn o_does_nothing_with_one_order_and_is_typed_in_the_filter() {
        let mut plain = picker(Mode::Single, false, list(&[]));
        keys(&mut plain, &[KeyCode::Char('o')]);
        assert_eq!(ids(&plain), ["a", "b", "c"]);
        assert_eq!(title(&plain), " GPU types ");
        assert!(!keys_hint(&plain).contains("o sorts"));
        let mut filtered = sortable(Mode::Single, list(&[]));
        keys(&mut filtered, &[KeyCode::Char('/'), KeyCode::Char('o')]);
        assert_eq!(filtered.filter_text(), "o");
        assert_eq!(filtered.order, 0);
    }

    #[test]
    fn an_entry_without_a_rank_comes_last() {
        let mut picker = sortable(Mode::Multi, list(&[]));
        if let Load::Ready(entries) = &mut picker.load {
            entries.push(entry("d", true));
            entries.swap(0, 3);
        }
        keys(&mut picker, &[KeyCode::Char('o')]);
        assert_eq!(ids(&picker), ["c", "a", "b", "d"]);
    }
}
