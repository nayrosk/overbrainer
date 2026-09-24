//! The Training view: the runs of `runs/` on top, the selected run below with its
//! progress, pod, loss chart and sparklines.

use std::fmt::Write as _;

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, Chart, Dataset, GraphType, Paragraph, Row, Sparkline, Table, TableState, Wrap,
};

use crate::runpod::{PodRecord, PodState, PodStatus};
use crate::runs::{RunRecord, RunState};
use crate::train::TrainMetric;
use crate::tui::app::App;
use crate::tui::format::duration;
use crate::tui::theme::Theme;
use crate::tui::training::{Ended, Follow, Job, RunRow, float, progress};

/// Rows the pod line may wrap to.
const POD_ROWS: u16 = 2;
/// Rows the messages under the chart may take.
const MESSAGE_ROWS: u16 = 4;

/// Draws the Training view in `area`.
pub(in crate::tui) fn render(frame: &mut Frame, area: Rect, app: &App) {
    let view = &app.training;
    let theme = &app.theme;
    let shown = u16::try_from(view.runs.len().clamp(1, 5)).unwrap_or(5);
    let [list, detail] =
        Layout::vertical([Constraint::Length(shown + 3), Constraint::Fill(1)]).areas(area);
    render_runs(frame, list, app);
    let Some(row) = view.selected_run() else {
        let text = view
            .error
            .clone()
            .unwrap_or_else(|| "no run yet".to_string());
        let block = Block::bordered()
            .title(Span::styled(" run ", theme.title))
            .border_style(theme.dim);
        frame.render_widget(
            Paragraph::new(Span::styled(text, theme.dim)).block(block),
            detail,
        );
        return;
    };
    let follow = view.task_of(&row.record.id).map(|(_, follow)| follow);
    let block = Block::bordered()
        .title(Span::styled(format!(" {} ", row.record.id), theme.title))
        .border_style(theme.dim);
    let inner = block.inner(detail);
    frame.render_widget(block, detail);
    let series = view
        .series
        .get(&row.record.id)
        .map_or(&[][..], Vec::as_slice);
    let ended = view.ended.get(&row.record.id);
    let messages = Paragraph::new(messages(follow, ended, theme)).wrap(Wrap { trim: true });
    let pod = row.pod.as_ref().map(|record| {
        Paragraph::new(pod_line(record, follow.and_then(|f| f.pod.as_ref()), app))
            .wrap(Wrap { trim: true })
    });
    let rows = |paragraph: &Paragraph, most: u16| {
        u16::try_from(paragraph.line_count(inner.width)).map_or(most, |rows| rows.min(most))
    };
    let pod_rows = pod.as_ref().map_or(0, |pod| rows(pod, POD_ROWS));
    let message_rows = rows(&messages, MESSAGE_ROWS);
    let [status, pod_area, chart, lr, grad, notes] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(pod_rows),
        Constraint::Fill(1),
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(message_rows),
    ])
    .areas(inner);
    frame.render_widget(
        Paragraph::new(status_line(&row.record, series, (follow, ended), theme)),
        status,
    );
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
        render_chart(frame, chart, series, theme);
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
}

fn render_runs(frame: &mut Frame, area: Rect, app: &App) {
    let view = &app.training;
    let theme = &app.theme;
    let rows: Vec<Row> = view
        .runs
        .iter()
        .map(|row: &RunRow| {
            let followed = match view.task_of(&row.record.id) {
                Some((_, follow)) if follow.job == Job::Cancel || follow.cancel_after => {
                    "cancelling"
                },
                Some((_, follow)) if follow.starting() => "starting",
                Some(_) => "followed",
                None => "",
            };
            Row::new(vec![
                row.record.id.clone(),
                row.record.state.name().to_string(),
                row.record.target.clone(),
                row.pod.as_ref().map(pod_summary).unwrap_or_default(),
                followed.to_string(),
            ])
        })
        .collect();
    let header = Row::new(vec!["run", "state", "target", "pod", ""]).style(theme.dim);
    let table = Table::new(
        rows,
        [
            Constraint::Length(21),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Fill(1),
            Constraint::Length(10),
        ],
    )
    .header(header)
    .row_highlight_style(theme.selected)
    .block(
        Block::bordered()
            .title(Span::styled(
                if view.error.is_some() && !view.runs.is_empty() {
                    " runs (stale: cannot list the runs) "
                } else {
                    " runs "
                },
                theme.title,
            ))
            .border_style(theme.dim),
    );
    let mut state =
        TableState::default().with_selected((!view.runs.is_empty()).then_some(view.selected));
    frame.render_stateful_widget(table, area, &mut state);
}

/// What `pod.json` says of a run's pod, as `runs ls` prints it, less the leading
/// "pod" the column's header already says.
fn pod_summary(record: &PodRecord) -> String {
    let summary = record.summary();
    summary
        .strip_prefix("pod ")
        .map_or_else(|| summary.clone(), str::to_string)
}

/// The run's step, epoch and, while its job runs, ETA; then whether a task
/// follows or cancels it, and the points its forwarder skipped.
fn status_line(
    record: &RunRecord,
    series: &[TrainMetric],
    (follow, ended): (Option<&Follow>, Option<&Ended>),
    theme: &Theme,
) -> Line<'static> {
    let running = matches!(record.state, RunState::Preparing | RunState::Running);
    let mut spans = Vec::new();
    if let Some(now) = progress(series) {
        let step = match now.max_steps {
            Some(max) if max > 0 => format!(
                "step {}/{max} ({}%)",
                now.step,
                now.step.saturating_mul(100) / max
            ),
            _ => format!("step {}", now.step),
        };
        spans.push(Span::raw(step));
        if let Some(epoch) = now.epoch {
            spans.push(Span::raw(format!("  epoch {epoch:.2}")));
        }
        if running {
            spans.push(Span::raw(match now.eta {
                Some(eta) => format!("  ETA {}", duration(eta)),
                None => "  ETA unknown".to_string(),
            }));
        }
    }
    let state = match follow {
        Some(follow) if follow.job == Job::Cancel || follow.cancel_after => "   cancelling",
        Some(follow) if follow.starting() => "   starting",
        Some(_) => "   followed",
        None => "",
    };
    spans.push(Span::styled(state, theme.ok));
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
        spans.push(Span::styled(
            format!("   {missing} points missing {until}"),
            theme.warn,
        ));
    }
    Line::from(spans)
}

/// The pod line: its ID and state, rate, uptime and spend estimate, and what
/// bounds it. A deleted pod shows what it cost and when it went, as recorded; a
/// kept one has no time limit (decision 39).
fn pod_line(record: &PodRecord, latest: Option<&PodStatus>, app: &App) -> Line<'static> {
    let id = record
        .pod_id
        .as_ref()
        .map_or_else(|| "(no pod yet)".to_string(), ToString::to_string);
    let text = match latest {
        Some(PodStatus::Deleted {
            estimated_spend,
            uptime,
            ..
        }) => deleted_line(
            &id,
            estimated_spend.or(record.estimated_spend),
            *uptime,
            record,
        ),
        _ if record.state == PodState::Deleted => {
            deleted_line(&id, record.estimated_spend, None, record)
        },
        _ => live_line(&id, record, latest, app),
    };
    Line::from(Span::styled(text, app.theme.dim))
}

/// The pod line of a deleted pod: its recorded spend, uptime and deletion time.
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
    if let Some(up) = uptime {
        write!(text, "  up {}", duration(up)).ok();
    }
    match spend {
        Some(spend) => write!(text, "  spent about ${spend:.2}").ok(),
        None => write!(text, "  spend unknown").ok(),
    };
    text
}

/// The pod line of a pod that exists: its state, rate, uptime and spend so far,
/// and the watchdog's deadline, or no time limit for a kept pod.
fn live_line(id: &str, record: &PodRecord, latest: Option<&PodStatus>, app: &App) -> String {
    let state = latest.map_or_else(|| record.state.name(), status_name);
    let rate = record.cost_per_hour.or(match latest {
        Some(PodStatus::Created { cost_per_hour, .. }) => *cost_per_hour,
        _ => None,
    });
    let mut text = format!("pod {id} {state}");
    if let Some(rate) = rate {
        write!(text, " ${rate:.2}/h").ok();
    }
    if let Some(up) = record.uptime(app.now) {
        let spend = rate.map_or_else(String::new, |rate| {
            format!(" (about ${:.2})", rate * up.as_secs_f64() / 3600.0)
        });
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
        write!(text, "  deleted by the watchdog by {at}").ok();
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
    let steps = series.iter().map(|m| float(m.step));
    let (x0, x1) = steps.fold((f64::MAX, f64::MIN), |(lo, hi), x| (lo.min(x), hi.max(x)));
    let values = loss.iter().chain(&eval).map(|(_, y)| *y);
    let (y0, y1) = values.fold((f64::MAX, f64::MIN), |(lo, hi), y| (lo.min(y), hi.max(y)));
    let (y0, y1) = if y0 > y1 { (0.0, 1.0) } else { (y0, y1) };
    let pad = ((y1 - y0) * 0.05).max(0.01);
    let x1 = if x1 > x0 { x1 } else { x0 + 1.0 };
    let datasets = vec![
        Dataset::default()
            .name("loss")
            .marker(Marker::HalfBlock)
            .graph_type(GraphType::Line)
            .style(theme.loss)
            .data(&loss),
        Dataset::default()
            .name("eval_loss")
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
