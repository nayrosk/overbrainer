//! The system panel of the Training view: the machine of the selected run, as
//! its last samples show it. One row per disk, then the CPU, the memory and
//! each GPU: a gauge, a percentage, a sparkline of the last samples, and the
//! figures behind them.

use std::collections::VecDeque;
use std::time::SystemTime;

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Padding, Paragraph, Sparkline};

use crate::system::{Gpu, SystemSample, float};
use crate::tui::format::duration;
use crate::tui::theme::Theme;
use crate::tui::widgets::bar::{bar, round_to};

/// Columns of the panel, borders included.
pub(super) const PANEL_WIDTH: u16 = 44;
/// Columns of the view from which the panel shows.
pub(super) const PANEL_FROM: u16 = 100;
/// GPU rows at most: more GPUs than this show the first ones, then one row
/// for the average of the rest.
const GPU_ROWS: usize = 4;
/// A gauge at or above this share turns to the warning style.
const WARN_AT: f64 = 0.85;
/// A gauge at or above this share turns to the error style.
const ERROR_AT: f64 = 0.95;
/// Cells of a row's name, gauge, percentage, sparkline.
const NAME: u16 = 5;
const GAUGE: u16 = 6;
const PERCENT: u16 = 5;
const SPARK: u16 = 7;

/// What a row shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Row {
    /// The file system at this position in the sample.
    Disk(usize),
    /// The CPU.
    Cpu,
    /// The memory.
    Memory,
    /// The GPU of this index.
    Gpu(u32),
    /// The average of the GPUs from this position on.
    Rest(usize),
}

/// The rows the panel shows for `sample`.
fn rows(sample: &SystemSample) -> Vec<Row> {
    let mut rows: Vec<Row> = (0..sample.disks.len()).map(Row::Disk).collect();
    rows.extend([Row::Cpu, Row::Memory]);
    if sample.gpus.len() > GPU_ROWS {
        let shown = GPU_ROWS - 1;
        rows.extend(sample.gpus[..shown].iter().map(|gpu| Row::Gpu(gpu.index)));
        rows.push(Row::Rest(shown));
    } else {
        rows.extend(sample.gpus.iter().map(|gpu| Row::Gpu(gpu.index)));
    }
    rows
}

/// Rows of content the panel needs for `samples`: one per row of the last
/// sample, or one for its note when there is none.
pub(super) fn height(samples: Option<&VecDeque<SystemSample>>) -> u16 {
    let count = samples
        .and_then(VecDeque::back)
        .map_or(1, |sample| rows(sample).len());
    u16::try_from(count).unwrap_or(u16::MAX)
}

/// Draws the panel in `area` from `samples`, oldest first. `followed` says
/// whether a task follows the run: a run nothing follows shows the age of its
/// last sample, at `now`.
pub(super) fn render(
    frame: &mut Frame,
    area: Rect,
    samples: Option<&VecDeque<SystemSample>>,
    (followed, now): (bool, SystemTime),
    theme: &Theme,
) {
    let last = samples.and_then(VecDeque::back);
    let title = match last {
        Some(sample) if !followed => {
            let age = now.duration_since(sample.at).unwrap_or_default();
            format!(" system · {} ago ", duration(age))
        },
        _ => " system ".to_string(),
    };
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme.border)
        .padding(Padding::horizontal(1))
        .title(Span::styled(title, theme.title));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let (Some(samples), Some(last)) = (samples, last) else {
        let note = if followed {
            "waiting for the first sample"
        } else {
            "no sample: not followed"
        };
        frame.render_widget(Paragraph::new(Span::styled(note, theme.dim)), inner);
        return;
    };
    let shown = rows(last);
    let areas = Layout::vertical(shown.iter().map(|_| Constraint::Length(1))).split(inner);
    for (row, area) in shown.iter().zip(areas.iter()) {
        render_row(frame, *area, *row, samples, theme);
    }
}

/// Draws `row` of the last of `samples` in the one-line `area`.
fn render_row(
    frame: &mut Frame,
    area: Rect,
    row: Row,
    samples: &VecDeque<SystemSample>,
    theme: &Theme,
) {
    let [head, _, spark, _, detail] = Layout::horizontal([
        Constraint::Length(NAME + GAUGE + PERCENT),
        Constraint::Length(1),
        Constraint::Length(SPARK),
        Constraint::Length(1),
        Constraint::Fill(1),
    ])
    .areas(area);
    let Some(last) = samples.back() else {
        return;
    };
    let now = share(last, row);
    let style = level(now, theme.gauge, theme);
    let percent = now.map_or_else(
        || "  --%".to_string(),
        |ratio| format!(" {:>3}%", round_to(ratio * 100.0, 100)),
    );
    let line = Line::from(vec![
        Span::styled(format!("{:<5}", name(row)), theme.dim),
        Span::styled(bar(now.unwrap_or(0.0), GAUGE), style),
        Span::styled(percent, style),
    ]);
    frame.render_widget(Paragraph::new(line), head);
    let history: Vec<u64> = samples
        .iter()
        .skip(samples.len().saturating_sub(usize::from(SPARK)))
        .map(|sample| share(sample, row).map_or(0, |ratio| u64::from(round_to(ratio * 100.0, 100))))
        .collect();
    frame.render_widget(
        Sparkline::default()
            .data(&history)
            .max(100)
            .style(theme.info),
        spark,
    );
    let (text, detail_style) = figures(last, row, theme);
    frame.render_widget(Paragraph::new(Span::styled(text, detail_style)), detail);
}

/// The name of `row`, as its first column shows it.
fn name(row: Row) -> String {
    match row {
        Row::Disk(0) => "disk".to_string(),
        Row::Disk(_) => "root".to_string(),
        Row::Cpu => "cpu".to_string(),
        Row::Memory => "mem".to_string(),
        Row::Gpu(index) => format!("gpu{index}"),
        Row::Rest(_) => "rest".to_string(),
    }
}

/// The GPU of index `index` in `sample`.
fn gpu(sample: &SystemSample, index: u32) -> Option<&Gpu> {
    sample.gpus.iter().find(|gpu| gpu.index == index)
}

/// The share `row` shows in `sample`: disk space used, CPU busy, memory used,
/// GPU utilisation.
fn share(sample: &SystemSample, row: Row) -> Option<f64> {
    match row {
        Row::Disk(position) => sample.disks.get(position)?.used_ratio(),
        Row::Cpu => sample.cpu.as_ref()?.usage,
        Row::Memory => sample.memory?.ratio(),
        Row::Gpu(index) => gpu(sample, index)?.utilization,
        Row::Rest(from) => {
            let known: Vec<f64> = sample
                .gpus
                .get(from..)?
                .iter()
                .filter_map(|gpu| gpu.utilization)
                .collect();
            let count = u32::try_from(known.len()).ok().filter(|count| *count > 0)?;
            Some(known.iter().sum::<f64>() / f64::from(count))
        },
    }
}

/// `normal` below [`WARN_AT`], the warning style from it, the error style
/// from [`ERROR_AT`].
fn level(ratio: Option<f64>, normal: Style, theme: &Theme) -> Style {
    match ratio {
        Some(ratio) if ratio >= ERROR_AT => theme.error,
        Some(ratio) if ratio >= WARN_AT => theme.warn,
        _ => normal,
    }
}

/// The figures behind `row` in `sample`, and their style: a GPU's turn to
/// the warning and error styles with its memory.
fn figures(sample: &SystemSample, row: Row, theme: &Theme) -> (String, Style) {
    let text = match row {
        Row::Disk(position) => sample
            .disks
            .get(position)
            .map(|disk| pair(disk.used_bytes, disk.size_bytes)),
        Row::Cpu => sample.cpu.as_ref().and_then(|cpu| {
            let load = cpu.load1?;
            Some(match cpu.cpus {
                Some(cpus) => format!("load {}/{}", number(load), number(cpus)),
                None => format!("load {}", number(load)),
            })
        }),
        Row::Memory => sample
            .memory
            .map(|memory| pair(memory.used_bytes, memory.limit_bytes)),
        Row::Gpu(index) => gpu(sample, index).map(gpu_figures),
        Row::Rest(from) => Some(format!("avg of {}", sample.gpus.len().saturating_sub(from))),
    };
    let style = match row {
        Row::Gpu(index) => level(
            gpu(sample, index).and_then(Gpu::memory_ratio),
            theme.dim,
            theme,
        ),
        _ => theme.dim,
    };
    (text.unwrap_or_default(), style)
}

/// A GPU's memory, temperature and power: `40/80G 64C 312W`, each left out
/// when unknown.
fn gpu_figures(gpu: &Gpu) -> String {
    let mut parts = Vec::new();
    if let (Some(used), Some(total)) = (gpu.memory_used_bytes, gpu.memory_total_bytes) {
        parts.push(pair(used, total));
    }
    if let Some(celsius) = gpu.temperature_celsius {
        parts.push(format!("{celsius:.0}C"));
    }
    if let Some(watts) = gpu.power_watts {
        parts.push(format!("{watts:.0}W"));
    }
    parts.join(" ")
}

/// `used` and `total` bytes in the unit of `total`: `31/64G`.
fn pair(used: u64, total: u64) -> String {
    let units = ["B", "K", "M", "G", "T", "P"];
    let mut scale = 1.0;
    let mut unit = units[0];
    for next in &units[1..] {
        if float(total) < scale * 1024.0 {
            break;
        }
        scale *= 1024.0;
        unit = next;
    }
    format!(
        "{}/{}{unit}",
        number(float(used) / scale),
        number(float(total) / scale)
    )
}

/// `value` with one decimal below 10, none from 10 on.
fn number(value: f64) -> String {
    if value < 9.95 {
        format!("{value:.1}")
    } else {
        format!("{value:.0}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn figures_read_in_the_unit_of_their_total() {
        assert_eq!(
            pair(40 * 1024 * 1024 * 1024, 80 * 1024 * 1024 * 1024),
            "40/80G"
        );
        assert_eq!(pair(512 * 1024 * 1024, 8 * 1024 * 1024 * 1024), "0.5/8.0G");
        assert_eq!(pair(3 * 1024_u64.pow(4), 4 * 1024_u64.pow(4)), "3.0/4.0T");
        assert_eq!(pair(10, 100), "10/100B");
        assert_eq!(number(9.96), "10");
    }

    #[test]
    fn more_than_four_gpus_fold_into_an_average() {
        let gpu = |index, utilization| Gpu {
            index,
            name: "H100".into(),
            utilization: Some(utilization),
            memory_used_bytes: None,
            memory_total_bytes: None,
            temperature_celsius: None,
            power_watts: None,
            power_limit_watts: None,
        };
        let mut sample = SystemSample {
            at: SystemTime::UNIX_EPOCH,
            disks: Vec::new(),
            cpu: None,
            memory: None,
            gpus: (0..6).map(|index| gpu(index, 0.1)).collect(),
        };
        sample.gpus[4].utilization = Some(0.5);
        sample.gpus[5].utilization = None;
        assert_eq!(
            rows(&sample),
            [
                Row::Cpu,
                Row::Memory,
                Row::Gpu(0),
                Row::Gpu(1),
                Row::Gpu(2),
                Row::Rest(3)
            ]
        );
        let average = share(&sample, Row::Rest(3)).unwrap_or_default();
        assert!((average - 0.3).abs() < 1e-9, "{average}");
        sample.gpus.truncate(4);
        assert_eq!(rows(&sample).len(), 6);
        assert_eq!(rows(&sample).last(), Some(&Row::Gpu(3)));
    }
}
