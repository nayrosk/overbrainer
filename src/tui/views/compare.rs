//! The Compare view: the compares on top, the selected one's summary on
//! their right; its questions below, the selected one's detail on their
//! right, its two answers side by side from 120 columns, stacked below.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Row, Table, TableState, Wrap};

use crate::compare::{CompareEntry, QuestionResult, Report, Verdict};
use crate::tui::app::App;
use crate::tui::compare::{CompareView, Focus};
use crate::tui::format::cut;
use crate::tui::theme::Theme;
use crate::tui::views::dataset::{failed, pane};

/// Terminal width from which the answers sit side by side.
const SIDE_BY_SIDE: u16 = 120;
/// Rows of the top panes, frames included, at least.
const TOP_MIN: u16 = 5;
/// Rows of the top panes, frames included, at most on a small terminal; a
/// taller one gives them a third of the view.
const TOP_MAX: u16 = 8;
/// Columns of a pane's frame and padding, both sides.
const FRAME: u16 = 4;
/// Columns of the quantize type column.
const QUANT: u16 = 6;
/// Columns of the judge column, shown side by side only.
const JUDGE: u16 = 12;
/// Columns of the win or tie column.
const RATE: u16 = 7;
/// Columns of the questions count column.
const COUNT: u16 = 3;
/// The summary pane's title.
const SUMMARY_TITLE: &str = " summary ";
/// Columns of the summary pane's top border beside its title and the
/// progress: two corners and at least one column of line between them.
const SUMMARY_BORDER: usize = 3;
/// The progress's mark, before its text.
const PROGRESS_MARK: &str = " ● ";
/// Characters of a question ID shown whole; longer ones keep both ends.
const ID_SHOWN: usize = 8;

/// A pane titled `title`, its border the focus color when `focused`.
fn framed<'a>(title: String, focused: bool, theme: &Theme) -> Block<'a> {
    let border = if focused {
        theme.border_focus
    } else {
        theme.border
    };
    pane(border).title(Span::styled(title, theme.title))
}

/// Draws the Compare view in `area`.
pub(in crate::tui) fn render(frame: &mut Frame, area: Rect, app: &mut App) {
    let wide = frame.area().width >= SIDE_BY_SIDE;
    let theme = app.theme;
    let view = &mut app.compare;
    let halves = || Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)]);
    let [_, summary_width] = halves().areas(area);
    let summary = summary_lines(view, summary_width.width.saturating_sub(FRAME), &theme);
    let summary_paragraph = Paragraph::new(summary).wrap(Wrap { trim: false });
    let summary_rows = summary_paragraph.line_count(summary_width.width.saturating_sub(FRAME));
    let list_rows = view.rows.len().saturating_add(1);
    let top_rows = u16::try_from(summary_rows.max(list_rows))
        .unwrap_or(u16::MAX)
        .saturating_add(2)
        .clamp(TOP_MIN, TOP_MAX.max(area.height / 3));
    let [top, bottom] =
        Layout::vertical([Constraint::Length(top_rows), Constraint::Fill(1)]).areas(area);
    let [list_area, summary_area] = halves().areas(top);
    render_list(frame, list_area, (view, wide), &theme);
    let mut block = framed(SUMMARY_TITLE.to_string(), false, &theme);
    if let Some(progress) = progress(view, summary_area.width, &theme) {
        block = block.title_top(progress);
    }
    frame.render_widget(summary_paragraph.block(block), summary_area);
    let [questions, detail] =
        Layout::horizontal([Constraint::Percentage(40), Constraint::Percentage(60)]).areas(bottom);
    render_questions(frame, questions, view, &theme);
    render_detail(frame, detail, (view, wide), &theme);
}

/// The compares: run, quantize type, judge (side by side only), win or tie
/// rate and questions; or why there is none.
fn render_list(frame: &mut Frame, area: Rect, (view, wide): (&CompareView, bool), theme: &Theme) {
    let block = framed(
        " compares ".to_string(),
        view.focus == Focus::Compares,
        theme,
    );
    if view.rows.is_empty() {
        let text = match &view.error {
            Some(error) => failed(error, theme),
            None => vec![Line::styled(
                "No compare yet. C compares the newest run with a GGUF, as `overbrainer \
                 compare` does.",
                theme.dim,
            )],
        };
        frame.render_widget(
            Paragraph::new(text).wrap(Wrap { trim: false }).block(block),
            area,
        );
        return;
    }
    let mut widths = vec![
        Constraint::Fill(1),
        Constraint::Length(QUANT),
        Constraint::Length(RATE),
        Constraint::Length(COUNT),
    ];
    let mut header = vec!["run", "quant", "win+tie", "n"];
    if wide {
        widths.insert(2, Constraint::Length(JUDGE));
        header.insert(2, "judge");
    }
    let rows: Vec<Row> = view
        .rows
        .iter()
        .map(|row| {
            let mut cells = list_cells(row);
            if !wide {
                cells.remove(2);
            }
            Row::new(cells)
        })
        .collect();
    let table = Table::new(rows, widths)
        .header(Row::new(header).style(theme.dim))
        .row_highlight_style(theme.selected)
        .block(block);
    let mut state = TableState::default().with_selected(Some(view.selected));
    frame.render_stateful_widget(table, area, &mut state);
}

/// The cells of a compare's row: run, quantize type, judge, win or tie rate
/// and questions; a compare without a report says its state instead.
fn list_cells(row: &CompareEntry) -> Vec<String> {
    match &row.report {
        Some(report) => vec![
            row.run.clone(),
            report.quantize.clone(),
            judge_word(report),
            rate(report),
            report.summary.questions.to_string(),
        ],
        None => vec![
            row.run.clone(),
            String::new(),
            String::new(),
            cut(row.record.state.name(), usize::from(RATE)),
            String::new(),
        ],
    }
}

/// The win or tie rate of `report`, as `71%`, or `-` when no question counts.
fn rate(report: &Report) -> String {
    report
        .summary
        .win_or_tie
        .map_or_else(|| "-".to_string(), |rate| format!("{:.0}%", rate * 100.0))
}

/// The judge in a word: `parent`, else its model's last part.
fn judge_word(report: &Report) -> String {
    if report.judge_is_parent {
        return "parent".to_string();
    }
    let model = report.judge.rsplit('/').next().unwrap_or_default();
    cut(model, usize::from(JUDGE))
}

/// What names the compare running: its run, else its ID once known.
fn running_name(view: &CompareView) -> Option<&str> {
    let running = view.running.as_ref()?;
    running.run.as_deref().or(running.compare.as_deref())
}

/// The progress of the compare running, for the summary pane's top right:
/// `● demo_...: judging 12/100`, or `● judging 12/100` when the run does
/// not fit in `width` columns beside the pane's title.
fn progress(view: &CompareView, width: u16, theme: &Theme) -> Option<Line<'static>> {
    let running = view.running.as_ref()?;
    let label = running.label();
    // The mark, the text and a space before the corner.
    let room = usize::from(width)
        .saturating_sub(SUMMARY_TITLE.chars().count() + SUMMARY_BORDER)
        .saturating_sub(PROGRESS_MARK.chars().count() + 1);
    let text = running_name(view)
        .map(|name| format!("{name}: {label}"))
        .filter(|text| text.chars().count() <= room)
        .unwrap_or(label);
    Some(
        Line::from(vec![
            Span::styled(PROGRESS_MARK, theme.accent),
            Span::raw(format!("{text} ")),
        ])
        .right_aligned(),
    )
}

/// Whether the compare running is the selected one.
fn running_is_selected(view: &CompareView) -> bool {
    let Some(running) = &view.running else {
        return false;
    };
    let selected = view.selected_row().map(|row| row.record.id.as_str());
    running.compare.is_some() && running.compare.as_deref() == selected
}

/// The summary pane's lines, `width` columns wide: which run is being
/// compared when it is not the selected compare, then the selected compare's
/// counts, latency, cost per 1,000 requests and hardware.
fn summary_lines(view: &CompareView, width: u16, theme: &Theme) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if view.running.is_some() && !running_is_selected(view) {
        let name = running_name(view).unwrap_or("the newest run");
        lines.push(Line::styled(
            cut(&format!("comparing {name}"), usize::from(width)),
            theme.accent,
        ));
    }
    let Some(row) = view.selected_row() else {
        return lines;
    };
    let Some(report) = &row.report else {
        lines.push(Line::styled(
            format!("{}: {}, no report", row.record.id, row.record.state.name()),
            theme.dim,
        ));
        return lines;
    };
    let summary = &report.summary;
    lines.push(Line::raw(format!(
        "win {}  tie {}  loss {}  unparsed {}",
        summary.wins,
        summary.ties,
        summary.losses + summary.errors,
        summary.unparsed
    )));
    let seconds =
        |value: Option<f64>| value.map_or_else(|| "-".to_string(), |s| format!("{s:.2}s"));
    lines.push(Line::raw(format!(
        "latency p50 {}  p95 {}",
        seconds(summary.latency_p50),
        seconds(summary.latency_p95)
    )));
    let dollars =
        |value: Option<f64>| value.map_or_else(|| "-".to_string(), |d| format!("${d:.2}"));
    lines.push(Line::raw(format!(
        "cost/1k parent {} child {}",
        dollars(report.costs.parent_per_1k),
        dollars(report.costs.child_per_1k)
    )));
    if let Some(hardware) = &report.hardware {
        lines.push(Line::styled(hardware.describe(), theme.dim));
    }
    lines
}

/// The mark of a verdict and its style.
fn mark(verdict: Verdict, theme: &Theme) -> (&'static str, Style) {
    match verdict {
        Verdict::Win => ("✓", theme.ok),
        Verdict::Tie => ("=", theme.info),
        Verdict::Loss => ("✗", theme.error),
        Verdict::Unparsed => ("?", theme.warn),
        Verdict::Error => ("!", theme.error),
    }
}

/// `id` whole when short, else its first and last four characters.
fn short(id: &str) -> String {
    let count = id.chars().count();
    if count <= ID_SHOWN {
        return id.to_string();
    }
    let head: String = id.chars().take(4).collect();
    let tail: String = id.chars().skip(count - 4).collect();
    format!("{head}…{tail}")
}

/// The selected compare's questions the filter keeps, each with its
/// verdict's mark.
fn render_questions(frame: &mut Frame, area: Rect, view: &CompareView, theme: &Theme) {
    let title = format!(" questions: {} ", view.filter.label());
    let block = framed(title, view.focus == Focus::Questions, theme);
    let questions = view.shown_questions();
    let id_width = questions
        .iter()
        .map(|question| short(&question.id).chars().count())
        .max()
        .unwrap_or(0)
        .saturating_add(2);
    let id_width = u16::try_from(id_width).unwrap_or(u16::MAX);
    let text_width = usize::from(area.width.saturating_sub(FRAME + id_width + 1));
    let rows: Vec<Row> = questions
        .iter()
        .map(|question| {
            let (symbol, style) = mark(question.verdict, theme);
            Row::new(vec![
                Line::from(vec![
                    Span::styled(symbol, style),
                    Span::raw(format!(" {}", short(&question.id))),
                ]),
                Line::raw(cut(&question.question.replace('\n', " "), text_width)),
            ])
        })
        .collect();
    let table = Table::new(rows, [Constraint::Length(id_width), Constraint::Fill(1)])
        .row_highlight_style(theme.selected)
        .block(block);
    let mut state =
        TableState::default().with_selected((!questions.is_empty()).then_some(view.question));
    frame.render_stateful_widget(table, area, &mut state);
}

/// A band of the detail: paragraphs side by side, starting on the same row.
struct Band {
    /// Each column's lines.
    columns: Vec<Vec<Line<'static>>>,
}

impl Band {
    /// A band of one column.
    fn one(lines: Vec<Line<'static>>) -> Self {
        Self {
            columns: vec![lines],
        }
    }

    /// The columns' areas in `width` columns at `x`, one column between two.
    fn areas(&self, x: u16, width: u16) -> Vec<(u16, u16)> {
        let count = u16::try_from(self.columns.len()).unwrap_or(1).max(1);
        let gaps = count - 1;
        let each = width.saturating_sub(gaps) / count;
        (0..count)
            .map(|i| (x.saturating_add(i * (each + 1)), each))
            .collect()
    }

    /// Rows the band takes at `width` columns: its tallest column's.
    fn rows(&self, width: u16) -> usize {
        self.columns
            .iter()
            .zip(self.areas(0, width))
            .map(|(lines, (_, each))| {
                Paragraph::new(lines.clone())
                    .wrap(Wrap { trim: false })
                    .line_count(each)
            })
            .max()
            .unwrap_or(0)
    }
}

/// A part's heading and its text, then a blank line.
fn part(title: &str, text: &str, theme: &Theme) -> Vec<Line<'static>> {
    let mut lines = vec![Line::styled(format!("── {title} ──"), theme.dim)];
    lines.extend(text.lines().map(|line| Line::raw(line.to_string())));
    lines.push(Line::raw(""));
    lines
}

/// The question's lines: its text, then its topic and timings, dim; the
/// topic alone when the timings would not fit in `room` columns (`None`:
/// they always show, as side by side).
fn question_lines(
    question: &QuestionResult,
    room: Option<u16>,
    theme: &Theme,
) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = question
        .question
        .lines()
        .map(|line| Line::raw(line.to_string()))
        .collect();
    let mut facts = vec![format!("topic {}", question.topic)];
    if let Some(seconds) = question.seconds {
        facts.push(format!("{seconds:.2}s"));
    }
    if let Some(first) = question.first_token_seconds {
        facts.push(format!("first token {first:.2}s"));
    }
    if let Some(speed) = question.tokens_per_second {
        facts.push(format!("{speed:.0} tok/s"));
    }
    let mut facts = facts.join("  ");
    if room.is_some_and(|room| facts.chars().count() > usize::from(room)) {
        facts = format!("topic {}", question.topic);
    }
    lines.push(Line::styled(facts, theme.dim));
    lines.push(Line::raw(""));
    lines
}

/// The selected question: its text, both answers and the judge's reason,
/// scrolled; keeps where each part starts, in rows, for `[` and `]`.
fn render_detail(
    frame: &mut Frame,
    area: Rect,
    (view, wide): (&mut CompareView, bool),
    theme: &Theme,
) {
    let Some(question) = view.selected_question() else {
        view.sections.clear();
        view.scroll = 0;
        frame.render_widget(framed(" detail ".to_string(), false, theme), area);
        return;
    };
    let child = part("child", &child_text(question), theme);
    let parent = part("parent", &question.parent, theme);
    let judge = part(
        "judge",
        question.reason.as_deref().unwrap_or("(no reason)"),
        theme,
    );
    let room = (!wide).then(|| area.width.saturating_sub(FRAME));
    let mut bands = vec![Band::one(question_lines(question, room, theme))];
    if wide {
        bands.push(Band {
            columns: vec![child, parent],
        });
    } else {
        bands.push(Band::one(child));
        bands.push(Band::one(parent));
    }
    bands.push(Band::one(judge));
    let block = framed(
        format!(" {}  {} ", short(&question.id), question.verdict.name()),
        false,
        theme,
    );
    let inner = block.inner(area);
    let heights: Vec<u16> = bands
        .iter()
        .map(|band| u16::try_from(band.rows(inner.width)).unwrap_or(u16::MAX))
        .collect();
    let total = heights
        .iter()
        .fold(0u16, |sum, rows| sum.saturating_add(*rows));
    let last = total.saturating_sub(inner.height);
    view.scroll = view.scroll.min(last);
    view.sections.clear();
    let mut start = 0u16;
    let mut starts = Vec::with_capacity(bands.len());
    for rows in &heights {
        starts.push(start);
        let section = start.min(last);
        if view.sections.last() != Some(&section) {
            view.sections.push(section);
        }
        start = start.saturating_add(*rows);
    }
    let position = format!(
        " {}/{} ",
        view.scroll.saturating_add(inner.height).min(total),
        total
    );
    frame.render_widget(
        block.title_bottom(Line::styled(position, theme.dim).right_aligned()),
        area,
    );
    let scroll = view.scroll;
    for ((band, start), rows) in bands.into_iter().zip(starts).zip(heights) {
        draw_band(frame, inner, (band, start, rows), scroll);
    }
}

/// Draws `band`, which starts `start` rows into the detail and takes `rows`,
/// in `inner` scrolled by `scroll` rows: only its visible part.
fn draw_band(frame: &mut Frame, inner: Rect, (band, start, rows): (Band, u16, u16), scroll: u16) {
    let end = start.saturating_add(rows);
    let bottom = scroll.saturating_add(inner.height);
    if end <= scroll || start >= bottom {
        return;
    }
    let top = start.max(scroll);
    let y = inner.y.saturating_add(top - scroll);
    let height = end.min(bottom) - top;
    let skip = top - start;
    let areas = band.areas(inner.x, inner.width);
    for (lines, (x, width)) in band.columns.into_iter().zip(areas) {
        frame.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: false })
                .scroll((skip, 0)),
            Rect {
                x,
                y,
                width,
                height,
            },
        );
    }
}

/// The child's answer, or why there is none.
fn child_text(question: &QuestionResult) -> String {
    match (&question.child, &question.error) {
        (Some(child), _) => child.clone(),
        (None, Some(error)) => format!("(no answer: {error})"),
        (None, None) => "(no answer)".to_string(),
    }
}
