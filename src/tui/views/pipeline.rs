//! The Pipeline view: one row per stage, then tokens, cost, recent errors and the
//! summary lines of the current or last pipeline task.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, LineGauge, Paragraph, Wrap};

use crate::events::Stage;
use crate::tui::app::App;
use crate::tui::pipeline::{PipelineView, STAGES, StageState, command_name};
use crate::tui::theme::Theme;

/// Draws the Pipeline view in `area`.
pub(in crate::tui) fn render(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let view = &app.pipeline;
    let title = match (view.command, view.running) {
        (Some(command), true) => format!(" pipeline: {} running ", command_name(command)),
        (Some(command), false) => format!(" pipeline: last run {} ", command_name(command)),
        (None, _) => " pipeline ".to_string(),
    };
    let block = Block::bordered()
        .title(Span::styled(title, theme.title))
        .border_style(theme.dim);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let [header, rows, rest] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(4),
        Constraint::Fill(1),
    ])
    .areas(inner);
    let [name, state, progress, counts] = columns(header);
    for (text, column) in [
        (" stage", name),
        ("state", state),
        ("progress", progress),
        ("in flight  retries  failed", counts),
    ] {
        frame.render_widget(Paragraph::new(Span::styled(text, theme.dim)), column);
    }
    let row_areas = Layout::vertical([Constraint::Length(1); 4]).split(rows);
    for (stage, row_area) in STAGES.iter().zip(row_areas.iter()) {
        render_row(frame, *row_area, view, *stage, theme);
    }
    // The oldest item failures give way first, so the summary lines and the
    // final error stay in view on a small terminal.
    let mut errors = view.errors.len();
    let paragraph = loop {
        let paragraph = Paragraph::new(details(view, theme, errors)).wrap(Wrap { trim: false });
        if errors == 0 || paragraph.line_count(rest.width) <= usize::from(rest.height) {
            break paragraph;
        }
        errors -= 1;
    };
    frame.render_widget(paragraph, rest);
}

/// The stage, state, progress and counts columns of a row.
fn columns(area: Rect) -> [Rect; 4] {
    Layout::horizontal([
        Constraint::Length(12),
        Constraint::Length(10),
        Constraint::Fill(1),
        Constraint::Length(27),
    ])
    .areas(area)
}

fn render_row(frame: &mut Frame, area: Rect, view: &PipelineView, stage: Stage, theme: &Theme) {
    let [name_area, state_area, gauge_area, counts_area] = columns(area);
    let row = view.row(stage);
    frame.render_widget(Paragraph::new(format!(" {stage}")), name_area);
    let (label, style) = match row.state {
        StageState::Idle => ("", theme.dim),
        StageState::Pending => ("pending", theme.dim),
        StageState::Running => ("running", theme.warn),
        StageState::Done => ("done", theme.ok),
    };
    frame.render_widget(Paragraph::new(Span::styled(label, style)), state_area);
    if matches!(row.state, StageState::Running | StageState::Done) {
        let ratio = if row.total == 0 {
            1.0
        } else {
            f64::from(u32::try_from(row.finished).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(row.total).unwrap_or(u32::MAX))
        };
        let gauge_widget = LineGauge::default()
            .ratio(ratio)
            .label(format!("{}/{}", row.finished, row.total))
            .filled_style(theme.gauge)
            .unfilled_style(theme.dim);
        frame.render_widget(gauge_widget, gauge_area);
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
}

/// The lines under the stage rows, with the newest `errors` item failures.
fn details(view: &PipelineView, theme: &Theme, errors: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if view.command.is_none() {
        lines.push(Line::from(Span::styled(
            " nothing ran yet in this TUI: r runs a stage",
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
        " tokens  in {}  out {}        {cost}",
        usage.input_tokens, usage.output_tokens
    )));
    if view.skipped > 0 {
        lines.push(Line::from(Span::styled(
            format!(
                " {} events skipped: counts catch up when each stage finishes",
                view.skipped
            ),
            theme.warn,
        )));
    }
    if errors > 0 {
        lines.push(Line::from(Span::styled(" last errors", theme.title)));
        let older = view.errors.len().saturating_sub(errors);
        for failure in view.errors.iter().skip(older) {
            lines.push(Line::from(format!(
                "   {} {}  {}",
                failure.stage,
                short(&failure.id),
                failure.error
            )));
        }
    }
    if !view.results.is_empty() || view.outcome.is_some() {
        lines.push(Line::from(Span::styled(" results", theme.title)));
    }
    for result in &view.results {
        lines.push(Line::from(format!("   {result}")));
    }
    if let Some(Err(error)) = &view.outcome {
        lines.push(Line::from(Span::styled(format!("   {error}"), theme.error)));
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
