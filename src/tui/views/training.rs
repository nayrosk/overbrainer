//! The Training view: the runs of `runs/` on top, with the system panel of the
//! selected run on their right on a wide terminal, the selected run below with
//! its progress, pod, loss chart and sparklines.

use std::fmt::Write as _;

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Margin, Rect};
use ratatui::style::Style;
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, BorderType, Chart, Dataset, GraphType, Padding, Paragraph, Row, Sparkline, Table,
    TableState, Wrap,
};

use crate::runpod::{PodRecord, PodState, PodStatus};
use crate::runs::{RunRecord, RunState};
use crate::train::TrainMetric;
use crate::tui::app::App;
use crate::tui::format::{cut, duration};
use crate::tui::motion::Bar;
use crate::tui::theme::Theme;
use crate::tui::training::{
    Ended, Follow, RunActivity, RunRow, TrainingView, float, pod_rate, pod_spend, progress,
};
use crate::tui::views::dataset::failed;
use crate::tui::views::system;
use crate::tui::widgets::bar::bar;

/// Rows the pod line may wrap to.
const POD_ROWS: u16 = 2;
/// Rows the messages under the chart may take.
const MESSAGE_ROWS: u16 = 4;
/// Cells of the selected run's step bar.
const STEP_BAR: u16 = 16;
/// Rows of the view from which a blank row parts the detail's sections (its
/// head, its pod, its chart): a terminal of 30 rows or more. Below, the rows
/// go to the chart.
const GAPS_FROM: u16 = 28;

/// Draws the Training view in `area`; returns the cell of the followed run's
/// `●`, when it shows.
pub(in crate::tui) fn render(frame: &mut Frame, area: Rect, app: &mut App) -> Option<Rect> {
    let view = &app.training;
    let theme = &app.theme;
    if view.runs.is_empty() {
        render_no_runs(frame, area, app);
        return None;
    }
    let detail = render_top(frame, area, app);
    // The selected run's detail has no frame: a two-column margin.
    let inner = detail.inner(Margin::new(2, 0));
    let row = view.selected_run()?;
    let follow = view.task_of(&row.record.id).map(|(_, follow)| follow);
    let series = view
        .series
        .get(&row.record.id)
        .map_or(&[][..], Vec::as_slice);
    let ended = view.ended.get(&row.record.id);
    let messages = Paragraph::new(messages(follow, ended, theme)).wrap(Wrap { trim: true });
    let pod = shown_pod(view, row).map(|record| {
        Paragraph::new(pod_line(record, follow.and_then(|f| f.pod.as_ref()), app))
            .wrap(Wrap { trim: true })
    });
    let rows = |paragraph: &Paragraph, most: u16| {
        u16::try_from(paragraph.line_count(inner.width)).map_or(most, |rows| rows.min(most))
    };
    let pod_rows = pod.as_ref().map_or(0, |pod| rows(pod, POD_ROWS));
    let message_rows = rows(&messages, MESSAGE_ROWS);
    let activity = view.activity(&row.record.id);
    let shown = app
        .motion
        .bar(Bar::Step, view.selected_ratio().unwrap_or(0.0), STEP_BAR);
    let head = head_line(&row.record, series, (activity, shown), theme);
    let facts = facts_line(
        (series, &row.record),
        (follow, ended),
        (activity, app.motion.spinner()),
        theme,
    );
    let facts_rows = u16::from(!facts.spans.is_empty());
    let gap = u16::from(area.height >= GAPS_FROM);
    let [status, facts_area, _, pod_area, _, chart, lr, grad, notes] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(facts_rows),
        Constraint::Length(gap),
        Constraint::Length(pod_rows),
        Constraint::Length(gap.min(pod_rows)),
        Constraint::Fill(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(message_rows),
    ])
    .areas(inner);
    frame.render_widget(Paragraph::new(head), status);
    frame.render_widget(Paragraph::new(facts), facts_area);
    if let Some(pod) = pod {
        frame.render_widget(pod, pod_area);
    }
    if series.is_empty() {
        let note = if follow.is_some() {
            "no metrics yet"
        } else {
            "not followed: press a to attach"
        };
        frame.render_widget(Paragraph::new(Span::styled(note, theme.dim)), chart);
    } else {
        let [legend, plot] =
            Layout::vertical([Constraint::Length(1), Constraint::Fill(1)]).areas(chart);
        render_legend(frame, legend, theme);
        render_chart(frame, plot, series, theme);
        render_sparkline(frame, lr, ("lr", |m| m.learning_rate), series, theme.lr);
        render_sparkline(
            frame,
            grad,
            ("grad_norm", |m| m.grad_norm),
            series,
            theme.grad_norm,
        );
    }
    // The newest lines stay in view when they need more rows than they get.
    let hidden = u16::try_from(messages.line_count(inner.width))
        .map_or(0, |total| total.saturating_sub(message_rows));
    frame.render_widget(messages.scroll((hidden, 0)), notes);
    (activity == RunActivity::Followed).then(|| Rect::new(status.x, status.y, 1, 1))
}

/// Draws the runs, and the system panel of the selected run on their right
/// where it fits; returns the area left below them.
fn render_top(frame: &mut Frame, area: Rect, app: &App) -> Rect {
    let view = &app.training;
    let theme = &app.theme;
    let shown = u16::try_from(view.runs.len().min(5)).unwrap_or(5);
    let samples = view
        .selected_run()
        .and_then(|row| view.system.get(&row.record.id));
    // The system panel of the selected run, right of the runs, where it fits
    // and leaves the detail enough rows.
    let with_panel = (shown + 3).max(system::height(samples).saturating_add(2));
    let panel = area.width >= system::PANEL_FROM
        && area.height >= with_panel.saturating_add(system::MIN_DETAIL);
    let top = if panel { with_panel } else { shown + 3 };
    let [list, detail] =
        Layout::vertical([Constraint::Length(top), Constraint::Fill(1)]).areas(area);
    let list = if panel {
        let [list, side] =
            Layout::horizontal([Constraint::Fill(1), Constraint::Length(system::PANEL_WIDTH)])
                .areas(list);
        let followed = view
            .selected_run()
            .is_some_and(|row| view.task_of(&row.record.id).is_some());
        system::render(frame, side, samples, (followed, app.now), theme);
        list
    } else {
        list
    };
    render_runs(frame, list, app);
    detail
}

/// The runs box with no run in it: what to do to start one, or why the runs
/// cannot be listed, in place of the table.
fn render_no_runs(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let text = match (&app.training.error, &app.project.target) {
        (Some(error), _) => failed(error, theme),
        (None, _) if !app.training.hidden.is_empty() => vec![Line::from(
            "No runs to show: the failed runs are cleared until the TUI restarts. Press t to \
             start one.",
        )],
        (None, Some(target)) => vec![Line::from(format!(
            "No runs yet: press t to start one on {target}."
        ))],
        (None, None) => vec![Line::from(
            "No runs yet: add a [training] section to overbrainer.toml, then press t.",
        )],
    };
    let block = runs_block(" runs ", theme);
    let paragraph = Paragraph::new(text).wrap(Wrap { trim: true });
    let inner_width = block.inner(area).width;
    let rows = u16::try_from(paragraph.line_count(inner_width)).unwrap_or(u16::MAX);
    let [list, _] = Layout::vertical([
        Constraint::Length(rows.saturating_add(2)),
        Constraint::Fill(1),
    ])
    .areas(area);
    frame.render_widget(paragraph.block(block), list);
}

/// The rounded, focused box of the runs, titled `title`.
fn runs_block<'a>(title: &'a str, theme: &Theme) -> Block<'a> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme.border_focus)
        .padding(Padding::horizontal(1))
        .title(Span::styled(title, theme.title))
}

fn render_runs(frame: &mut Frame, area: Rect, app: &App) {
    let view = &app.training;
    let theme = &app.theme;
    // What the pod column gets: the box's borders and padding, the other
    // columns and the four gaps between the five taken off.
    let pod_width = usize::from(area.width.saturating_sub(4 + 21 + 10 + 10 + 10 + 4));
    let rows: Vec<Row> = view
        .runs
        .iter()
        .map(|row: &RunRow| {
            Row::new(vec![
                row.record.id.clone(),
                row.record.state.name().to_string(),
                row.record.target.clone(),
                shown_pod(view, row)
                    .map(|pod| cut(&pod_summary(pod), pod_width))
                    .unwrap_or_default(),
                view.activity(&row.record.id).label().to_string(),
            ])
        })
        .collect();
    let header = Row::new(vec!["run", "state", "target", "pod", ""]).style(theme.dim);
    // Wide enough for the longest ID, and never narrower than an ID of the
    // older `20260922-143005-a1b2` form with a space.
    let id_width = view
        .runs
        .iter()
        .map(|row| row.record.id.len() + 1)
        .fold(21, usize::max);
    let table = Table::new(
        rows,
        [
            Constraint::Length(u16::try_from(id_width).unwrap_or(u16::MAX)),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Fill(1),
            Constraint::Length(10),
        ],
    )
    .header(header)
    .row_highlight_style(theme.selected)
    .block(runs_block(
        if view.error.is_some() {
            " runs (stale: cannot list the runs) "
        } else {
            " runs "
        },
        theme,
    ));
    let mut state = TableState::default().with_selected(Some(view.selected));
    frame.render_stateful_widget(table, area, &mut state);
}

/// The pod record of `row`, unless its pod was dismissed with `p`.
fn shown_pod<'a>(view: &TrainingView, row: &'a RunRow) -> Option<&'a PodRecord> {
    row.pod
        .as_ref()
        .filter(|_| !view.dismissed.contains(&row.record.id))
}

/// What `pod.json` says of a run's pod, as `runs ls` prints it, less the leading
/// "pod" the column's header already says.
fn pod_summary(record: &PodRecord) -> String {
    let summary = record.summary();
    summary
        .strip_prefix("pod ")
        .map_or_else(|| summary.clone(), str::to_string)
}

/// The selected run's first status line: `●` when a task follows it, its ID,
/// its step with a bar filled to `shown` and a percentage, and its ETA while
/// its job runs.
fn head_line(
    record: &RunRecord,
    series: &[TrainMetric],
    (activity, shown): (RunActivity, f64),
    theme: &Theme,
) -> Line<'static> {
    // No stand-in for the marker: the ID of a run nothing follows starts in
    // the column of the lines under it.
    let mut head = Vec::new();
    if activity == RunActivity::Followed {
        head.push(Span::styled("● ", theme.title));
    }
    head.push(Span::styled(record.id.clone(), theme.title));
    let Some(now) = progress(series) else {
        return Line::from(head);
    };
    match now.max_steps {
        Some(max) if max > 0 => {
            head.push(Span::raw(format!("  step {}/{max} ", now.step)));
            head.push(Span::styled(bar(shown, STEP_BAR), theme.gauge));
            head.push(Span::raw(format!(
                " {}%",
                (now.step.saturating_mul(100) / max).min(100)
            )));
        },
        _ => head.push(Span::raw(format!("  step {}", now.step))),
    }
    if matches!(record.state, RunState::Preparing | RunState::Running) {
        head.push(Span::raw(match now.eta {
            Some(eta) => format!("  ETA {}", duration(eta)),
            None => "  ETA unknown".to_string(),
        }));
    }
    Line::from(head)
}

/// The selected run's second status line: its epoch, the run it resumed from,
/// the snapshot of a stopped run (the model in its `output/` is partial), whether a task starts (with the spinner),
/// cancels or stops it, and the points its forwarder skipped; empty when there
/// is none of them.
fn facts_line(
    (series, record): (&[TrainMetric], &RunRecord),
    (follow, ended): (Option<&Follow>, Option<&Ended>),
    (activity, spinner): (RunActivity, &str),
    theme: &Theme,
) -> Line<'static> {
    let mut facts: Vec<Span<'static>> = Vec::new();
    if let Some(epoch) = progress(series).and_then(|now| now.epoch) {
        facts.push(Span::raw(format!("epoch {epoch:.2}")));
    }
    if let Some(from) = &record.resumed_from {
        facts.push(Span::styled(format!("resumed from {from}"), theme.dim));
    }
    if let Some(snapshot) = record
        .snapshot
        .as_ref()
        .filter(|_| record.state == RunState::Stopped)
    {
        facts.push(Span::styled(
            format!(
                "snapshot at step {} ({}), output/ partial: T resumes it",
                snapshot.step,
                snapshot.reason.name()
            ),
            theme.accent,
        ));
    }
    match activity {
        RunActivity::Starting { .. } | RunActivity::Abandoning { .. } => facts.push(Span::styled(
            format!("{spinner} {}", activity.label()),
            theme.accent,
        )),
        RunActivity::Cancelling | RunActivity::Stopping => {
            facts.push(Span::styled(activity.label(), theme.accent));
        },
        RunActivity::None | RunActivity::Followed => {},
    }
    let missing = match (follow, ended) {
        (Some(follow), _) => follow.skipped,
        (None, Some(ended)) if !ended.healed => ended.skipped,
        (None, _) => 0,
    };
    if missing > 0 {
        let until = if follow.is_some() {
            "until the run ends"
        } else {
            "(no local metrics file to read them from)"
        };
        facts.push(Span::styled(
            format!("{missing} points missing {until}"),
            theme.warn,
        ));
    }
    let mut spaced = Vec::new();
    for (index, span) in facts.into_iter().enumerate() {
        if index > 0 {
            spaced.push(Span::raw("  "));
        }
        spaced.push(span);
    }
    Line::from(spaced)
}

/// The pod line: its ID and state, rate, uptime and spend estimate, and what
/// bounds it. A deleted pod shows what it cost and when it went, as recorded; a
/// kept one has no time limit (decision 39).
fn pod_line(record: &PodRecord, latest: Option<&PodStatus>, app: &App) -> Line<'static> {
    let id = record
        .pod_id
        .as_ref()
        .map_or_else(|| "(no pod yet)".to_string(), ToString::to_string);
    let spend = pod_spend(record, latest, app.now);
    let text = match latest {
        Some(PodStatus::Deleted { uptime, .. }) => deleted_line(&id, spend, *uptime, record),
        _ if record.state == PodState::Deleted => deleted_line(&id, spend, None, record),
        _ => live_line(&id, record, latest, spend, app),
    };
    Line::from(Span::styled(text, app.theme.dim))
}

/// The pod line of a deleted pod: its deletion time, its final uptime (as its
/// last event gave it, else from `pod.json`) and its recorded spend.
fn deleted_line(
    id: &str,
    spend: Option<f64>,
    uptime: Option<std::time::Duration>,
    record: &PodRecord,
) -> String {
    let mut text = format!("pod {id} deleted");
    if let Some(at) = &record.deleted_at {
        write!(text, " at {at}").ok();
    }
    if let Some(up) = uptime.or_else(|| record.final_uptime()) {
        write!(text, "  up {}", duration(up)).ok();
    }
    match spend {
        Some(spend) => write!(text, "  spent about ${spend:.2}").ok(),
        None => write!(text, "  spend unknown").ok(),
    };
    text
}

/// The pod line of a pod that exists: its state, rate, uptime and spend so far,
/// and the watchdog's deadline, or no time limit for a kept pod. The state is
/// the one `pod.json` records: no event marks the job's start, so the last
/// event would read `ready` for as long as the job runs. Only while the pod is
/// being created does the last event say more (the GPU type being tried).
fn live_line(
    id: &str,
    record: &PodRecord,
    latest: Option<&PodStatus>,
    spend: Option<f64>,
    app: &App,
) -> String {
    let state = match (record.state, latest) {
        (PodState::Creating, Some(status)) => status_name(status),
        (state, _) => state.name(),
    };
    let mut text = format!("pod {id} {state}");
    if let Some(rate) = pod_rate(record, latest) {
        write!(text, " ${rate:.2}/h").ok();
    }
    if let Some(up) = record.uptime(app.now) {
        let spend = spend.map_or_else(String::new, |spend| format!(" (about ${spend:.2})"));
        write!(text, "  up {}{spend}", duration(up)).ok();
    }
    // `--keep-pod` holds only once the job started: until then a failed start
    // or the boot grace can still delete the pod (decision 39).
    let kept = record.state == PodState::Kept
        || (record.keep && record.state == PodState::Running)
        || matches!(latest, Some(PodStatus::Kept { .. }));
    if kept {
        text.push_str("  kept, no time limit");
        return text;
    }
    if record.keep {
        text.push_str("  kept once its job starts");
    }
    if let Some(at) = &record.deadline {
        write!(text, "  the watchdog deletes it by {at}").ok();
    }
    text
}

/// A pod status in one word.
fn status_name(status: &PodStatus) -> &'static str {
    match status {
        PodStatus::Creating { .. } => "creating",
        PodStatus::Unavailable { .. } => "trying the next GPU type",
        PodStatus::Created { .. } => "created",
        PodStatus::Ready { .. } => "ready",
        PodStatus::Deleting { .. } => "deleting",
        PodStatus::Deleted { .. } => "deleted",
        PodStatus::Kept { .. } => "kept",
    }
}

/// Points of `values`, at most four per column of `width`, the last always kept.
fn downsample(points: Vec<(f64, f64)>, width: u16) -> Vec<(f64, f64)> {
    let most = usize::from(width.max(1)) * 4;
    if points.len() <= most {
        return points;
    }
    let every = points.len().div_ceil(most);
    let last = points.len() - 1;
    points
        .into_iter()
        .enumerate()
        .filter(|(index, _)| index % every == 0 || *index == last)
        .map(|(_, point)| point)
        .collect()
}

/// The chart's label on the left, and what its marks are on the right.
fn render_legend(frame: &mut Frame, area: Rect, theme: &Theme) {
    frame.render_widget(Paragraph::new(Span::styled("loss", theme.dim)), area);
    let legend = Line::from(vec![
        Span::styled("─", theme.loss),
        Span::styled(" train  ", theme.dim),
        Span::styled("•", theme.eval_loss),
        Span::styled(" eval", theme.dim),
    ])
    .right_aligned();
    frame.render_widget(Paragraph::new(legend), area);
}

/// The x axis of the loss chart: from 0 to the run's `max_steps` when the
/// metrics carry it (or to the last step, if the run went past it), else over
/// the steps seen, one step wide at least.
fn x_bounds(series: &[TrainMetric]) -> (f64, f64) {
    let steps = series.iter().map(|m| float(m.step));
    let (x0, x1) = steps.fold((f64::MAX, f64::MIN), |(lo, hi), x| (lo.min(x), hi.max(x)));
    if let Some(max) = progress(series)
        .and_then(|p| p.max_steps)
        .filter(|max| *max > 0)
    {
        return (0.0, float(max).max(x1));
    }
    (x0, if x1 > x0 { x1 } else { x0 + 1.0 })
}

fn render_chart(frame: &mut Frame, area: Rect, series: &[TrainMetric], theme: &Theme) {
    let width = area.width.saturating_sub(8);
    let points = |value: fn(&TrainMetric) -> Option<f64>| {
        let all: Vec<(f64, f64)> = series
            .iter()
            .filter_map(|m| value(m).map(|v| (float(m.step), v)))
            .collect();
        downsample(all, width)
    };
    let loss = points(|m| m.loss);
    let eval = points(|m| m.eval_loss);
    let (x0, x1) = x_bounds(series);
    let values = loss.iter().chain(&eval).map(|(_, y)| *y);
    let (y0, y1) = values.fold((f64::MAX, f64::MIN), |(lo, hi), y| (lo.min(y), hi.max(y)));
    let (y0, y1) = if y0 > y1 { (0.0, 1.0) } else { (y0, y1) };
    let pad = ((y1 - y0) * 0.05).max(0.01);
    let datasets = vec![
        Dataset::default()
            .marker(Marker::HalfBlock)
            .graph_type(GraphType::Line)
            .style(theme.loss)
            .data(&loss),
        Dataset::default()
            .marker(Marker::Dot)
            .graph_type(GraphType::Scatter)
            .style(theme.eval_loss)
            .data(&eval),
    ];
    let chart = Chart::new(datasets)
        .x_axis(
            Axis::default()
                .bounds([x0, x1])
                .labels([format!("step {x0}"), format!("{x1}")])
                .style(theme.dim),
        )
        .y_axis(
            Axis::default()
                .bounds([y0 - pad, y1 + pad])
                .labels([format!("{y0:.2}"), format!("{y1:.2}")])
                .style(theme.dim),
        );
    frame.render_widget(chart, area);
}

/// A sparkline of the last values of `value` over the width, scaled between the
/// shown minimum and maximum (a flat series draws at mid height), then the
/// latest raw value.
fn render_sparkline(
    frame: &mut Frame,
    area: Rect,
    (label, value): (&'static str, fn(&TrainMetric) -> Option<f64>),
    series: &[TrainMetric],
    style: Style,
) {
    let [name, line, latest] = Layout::horizontal([
        Constraint::Length(10),
        Constraint::Fill(1),
        Constraint::Length(10),
    ])
    .areas(area);
    let values: Vec<f64> = series.iter().filter_map(value).collect();
    let shown = &values[values.len().saturating_sub(usize::from(line.width))..];
    let (lo, hi) = shown
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    let scaled: Vec<u64> = shown
        .iter()
        .map(|v| {
            if hi > lo {
                scale((v - lo) / (hi - lo))
            } else {
                500
            }
        })
        .collect();
    frame.render_widget(Paragraph::new(Span::styled(label, style)), name);
    frame.render_widget(
        Sparkline::default().data(&scaled).max(1000).style(style),
        line,
    );
    let text = match (values.last(), label) {
        (Some(v), "lr") => format!(" {v:.2e}"),
        (Some(v), _) => format!(" {v:.3}"),
        (None, _) => String::new(),
    };
    frame.render_widget(Paragraph::new(text), latest);
}

/// `fraction` of the way from 0 to 1000, rounded, without a float cast.
fn scale(fraction: f64) -> u64 {
    let target = (fraction * 1000.0).round();
    let (mut low, mut high) = (0_u64, 1000_u64);
    while low < high {
        let middle = u64::midpoint(low, high);
        if float(middle) < target {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}

/// The lines a task reported, then how the run's last task ended.
fn messages(follow: Option<&Follow>, ended: Option<&Ended>, theme: &Theme) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let (reported, error) = match (follow, ended) {
        (Some(follow), _) => (&follow.lines, None),
        (None, Some(ended)) => (&ended.lines, ended.error.as_ref()),
        (None, None) => return lines,
    };
    lines.extend(reported.iter().map(|line| Line::from(line.clone())));
    if let Some(error) = error {
        lines.push(Line::from(Span::styled(error.clone(), theme.warn)));
    }
    let skip = lines.len().saturating_sub(3);
    lines.split_off(skip)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(steps: &[u64], max_steps: Option<u64>) -> Vec<TrainMetric> {
        steps
            .iter()
            .map(|step| TrainMetric {
                time: float(*step),
                step: *step,
                epoch: None,
                max_steps,
                loss: Some(1.0),
                eval_loss: None,
                learning_rate: None,
                grad_norm: None,
            })
            .collect()
    }

    #[test]
    fn a_known_max_steps_spans_the_axis_from_zero() {
        assert_eq!(
            x_bounds(&metrics(&[10, 600, 1200], Some(4000))),
            (0.0, 4000.0)
        );
    }

    #[test]
    fn an_unknown_max_steps_keeps_the_axis_over_the_steps_seen() {
        assert_eq!(x_bounds(&metrics(&[10, 50], None)), (10.0, 50.0));
        assert_eq!(x_bounds(&metrics(&[10], None)), (10.0, 11.0));
        assert_eq!(x_bounds(&metrics(&[10, 50], Some(0))), (10.0, 50.0));
    }

    #[test]
    fn a_step_past_max_steps_is_not_clipped() {
        assert_eq!(x_bounds(&metrics(&[3900, 4100], Some(4000))), (0.0, 4100.0));
    }
}
