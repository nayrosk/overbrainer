//! The Pipeline view: one row per stage, then tokens, cost, recent errors and the
//! summary lines of the current or last pipeline task.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Margin, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use crate::events::Stage;
use crate::tui::app::App;
use crate::tui::format::{WORKING, hang};
use crate::tui::pipeline::{PipelineView, STAGES, StageState, command_name};
use crate::tui::theme::Theme;
use crate::tui::training::float;
use crate::tui::widgets::bar::bar;

/// Draws the Pipeline view in `area`: no frame, a two-column margin, a title
/// row that also heads the count columns, the stage rows, a blank row, then
/// the details.
pub(in crate::tui) fn render(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let view = &app.pipeline;
    let area = area.inner(Margin::new(2, 0));
    let [title_row, rows, _, rest] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(4),
        Constraint::Length(1),
        Constraint::Fill(1),
    ])
    .areas(area);
    let title = match view.command {
        Some(command) => format!("Pipeline · {}", command_name(command)),
        None => "Pipeline".to_string(),
    };
    frame.render_widget(Paragraph::new(Span::styled(title, theme.title)), title_row);
    let [.., counts] = columns(title_row);
    frame.render_widget(
        Paragraph::new(Span::styled(" in flight  retries  failed", theme.dim)),
        counts,
    );
    let row_areas = Layout::vertical([Constraint::Length(1); 4]).split(rows);
    for (stage, row_area) in STAGES.iter().zip(row_areas.iter()) {
        render_row(frame, *row_area, view, *stage, theme);
    }
    // The oldest item failures give way first, so the summary lines and the
    // final error stay in view on a small terminal.
    let mut errors = view.errors.len();
    let paragraph = loop {
        let lines = details(view, theme, errors, rest.width);
        let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
        if errors == 0 || paragraph.line_count(rest.width) <= usize::from(rest.height) {
            break paragraph;
        }
        errors -= 1;
    };
    frame.render_widget(paragraph, rest);
}

/// The glyph, name, bar, count and counts columns of a row.
fn columns(area: Rect) -> [Rect; 5] {
    Layout::horizontal([
        Constraint::Length(2),
        Constraint::Length(12),
        Constraint::Fill(1),
        Constraint::Length(10),
        Constraint::Length(27),
    ])
    .areas(area)
}

/// The share of a stage's items finished, 1 when it has none.
pub(in crate::tui) fn ratio(finished: usize, total: usize) -> f64 {
    let count = |n: usize| float(u64::try_from(n).unwrap_or(u64::MAX));
    if total == 0 {
        1.0
    } else {
        count(finished) / count(total)
    }
}

/// One stage: `✓` done, [`WORKING`] running, `·` pending, `✗` stopped; its
/// name, its bar (the word `pending` or `stopped` in the empty part), its
/// count and its request counters.
fn render_row(frame: &mut Frame, area: Rect, view: &PipelineView, stage: Stage, theme: &Theme) {
    let [glyph_area, name_area, bar_area, label_area, counts_area] = columns(area);
    let row = view.row(stage);
    let (glyph, glyph_style, name_style) = match row.state {
        StageState::Idle => (" ", theme.dim, Style::new()),
        StageState::Pending => ("·", theme.dim, theme.dim),
        StageState::Running => (WORKING, theme.accent, theme.accent),
        StageState::Done => ("✓", theme.ok, Style::new()),
        StageState::Stopped => ("✗", theme.error, Style::new()),
    };
    frame.render_widget(Paragraph::new(Span::styled(glyph, glyph_style)), glyph_area);
    frame.render_widget(
        Paragraph::new(Span::styled(stage.to_string(), name_style)),
        name_area,
    );
    if row.state == StageState::Pending {
        frame.render_widget(Paragraph::new(Span::styled("pending", theme.dim)), bar_area);
    }
    if !matches!(
        row.state,
        StageState::Running | StageState::Done | StageState::Stopped
    ) {
        return;
    }
    let filled = bar(ratio(row.finished, row.total), bar_area.width);
    let mut spans = vec![Span::styled(filled.trim_end().to_string(), theme.gauge)];
    if row.state == StageState::Stopped {
        let room = filled.chars().count() - filled.trim_end().chars().count();
        if room > " stopped".len() {
            spans.push(Span::styled(" stopped", theme.error));
        }
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), bar_area);
    frame.render_widget(
        Paragraph::new(format!("{}/{}", row.finished, row.total)).right_aligned(),
        label_area,
    );
    frame.render_widget(
        Paragraph::new(format!(
            " {:>9}  {:>7}  {:>6}",
            view.in_flight(stage),
            row.retries,
            row.failed
        )),
        counts_area,
    );
}

/// The lines under the stage rows, with the newest `errors` item failures,
/// wrapped at `width` with a hanging indent.
fn details(view: &PipelineView, theme: &Theme, errors: usize, width: u16) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if view.command.is_none() {
        lines.push(Line::from(Span::styled(
            "nothing ran yet in this TUI: r runs a stage",
            theme.dim,
        )));
        return lines;
    }
    let usage = view.usage();
    let cost = STAGES
        .iter()
        .filter_map(|stage| view.row(*stage).stats.as_ref())
        .filter_map(|stats| stats.cost)
        .reduce(|a, b| a + b)
        .map_or_else(
            || {
                if view.running {
                    "cost: at the end of each stage".to_string()
                } else {
                    "cost unknown".to_string()
                }
            },
            |cost| format!("cost ${cost:.4}"),
        );
    lines.push(Line::from(format!(
        "tokens  in {}  out {}        {cost}",
        usage.input_tokens, usage.output_tokens
    )));
    if view.skipped > 0 {
        lines.push(Line::from(Span::styled(
            format!(
                "{} events skipped: counts catch up when each stage finishes",
                view.skipped
            ),
            theme.warn,
        )));
    }
    if errors > 0 {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Last errors", theme.title)));
        let older = view.errors.len().saturating_sub(errors);
        for failure in view.errors.iter().skip(older) {
            let text = format!(
                "{} {}  {}",
                failure.stage,
                short(&failure.id),
                failure.error
            );
            lines.extend(hang(&text, width, 2).into_iter().map(Line::from));
        }
    }
    if !view.results.is_empty() || view.outcome.is_some() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Results", theme.title)));
    }
    for result in &view.results {
        lines.extend(hang(result, width, 2).into_iter().map(Line::from));
    }
    if let Some(Err(error)) = &view.outcome {
        let wrapped = hang(&format!("✗ {error}"), width, 2);
        lines.extend(
            wrapped
                .into_iter()
                .map(|line| Line::from(Span::styled(line, theme.error))),
        );
    }
    lines
}

/// An item ID shortened like the Dataset view's; topic names stay whole.
fn short(id: &str) -> String {
    if id.len() == 32 && id.chars().all(|c| c.is_ascii_hexdigit()) {
        format!("{}…{}", &id[..4], &id[28..])
    } else {
        id.to_string()
    }
}
