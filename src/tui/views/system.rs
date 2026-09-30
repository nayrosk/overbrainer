//! The system panel of the Training view: the machine of the selected run, as
//! its last samples show it. One row per disk, then the CPU, the memory and
//! each GPU: a gauge, a percentage, a sparkline of the kept samples, and the
//! figures behind them.

use std::collections::VecDeque;
use std::time::{Duration, SystemTime};

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Padding, Paragraph, Sparkline};

use crate::system::{Cpu, Gpu, SystemSample, float};
use crate::tui::format::{cut, duration};
use crate::tui::theme::Theme;
use crate::tui::widgets::bar::{bar, round_to};

/// Columns of the panel, borders included.
pub(super) const PANEL_WIDTH: u16 = 44;
/// Columns of the view from which the panel shows: below, the runs table
/// left of it would lose its columns.
pub(super) const PANEL_FROM: u16 = 120;
/// Rows the selected run's detail keeps under the runs and the panel: a
/// panel that would leave fewer is not shown.
pub(super) const MIN_DETAIL: u16 = 14;
/// Age of the last sample from which the panel's title shows it: two missed
/// samples.
const STALE_AFTER: Duration = Duration::from_secs(2 * crate::runs::PROBE_EVERY.as_secs());
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
    let folded = sample.gpus.len() > GPU_ROWS;
    let shown = if folded { GPU_ROWS - 1 } else { GPU_ROWS };
    rows.extend(
        sample
            .gpus
            .iter()
            .take(shown)
            .map(|gpu| Row::Gpu(gpu.index)),
    );
    if folded {
        rows.push(Row::Rest(shown));
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
/// whether a task follows the run. A last sample older than [`STALE_AFTER`]
/// at `now` shows its age, followed or not.
pub(super) fn render(
    frame: &mut Frame,
    area: Rect,
    samples: Option<&VecDeque<SystemSample>>,
    (followed, now): (bool, SystemTime),
    theme: &Theme,
) {
    let last = samples.and_then(VecDeque::back);
    let age = last.map(|sample| now.duration_since(sample.at).unwrap_or_default());
    let title = match age {
        Some(age) if age > STALE_AFTER => format!(" system · {} ago ", duration(age)),
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
    // A shared disk's share is the whole cluster's: shown, never a warning.
    let style = if shared(last, row) {
        theme.dim
    } else {
        level(now, theme.gauge, theme)
    };
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
    let history = history(samples, row, usize::from(spark.width));
    frame.render_widget(
        Sparkline::default()
            .data(&history)
            .max(100)
            .style(theme.info),
        spark,
    );
    let (parts, detail_style) = figures(last, row, theme);
    let text = fit(parts, usize::from(detail.width));
    frame.render_widget(Paragraph::new(Span::styled(text, detail_style)), detail);
}

/// The sparkline of `row` over `samples` in `width` cells, in percent: each
/// cell the peak of its share of the samples, so the whole kept history fits
/// (ten minutes in the few cells of the panel). A cell whose samples all miss
/// the figure is left blank rather than drawn at zero.
fn history(samples: &VecDeque<SystemSample>, row: Row, width: usize) -> Vec<Option<u64>> {
    let per_cell = samples.len().div_ceil(width.max(1)).max(1);
    let shares: Vec<Option<u64>> = samples
        .iter()
        .map(|sample| share(sample, row).map(|ratio| u64::from(round_to(ratio * 100.0, 100))))
        .collect();
    shares
        .chunks(per_cell)
        .map(|cell| cell.iter().flatten().copied().max())
        .collect()
}

/// Whether `row` is a shared (network) file system in `sample`.
fn shared(sample: &SystemSample, row: Row) -> bool {
    matches!(row, Row::Disk(position) if sample.disks.get(position).is_some_and(|disk| disk.shared))
}

/// `parts` joined with spaces, the last ones dropped whole until they fit
/// `width`; the first part alone is cut when even it does not.
fn fit(mut parts: Vec<String>, width: usize) -> String {
    while parts.len() > 1 && parts.join(" ").chars().count() > width {
        parts.pop();
    }
    cut(&parts.join(" "), width)
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

/// The figures behind `row` in `sample`, most telling first, and their
/// style: a GPU's turn to the warning and error styles with its memory.
fn figures(sample: &SystemSample, row: Row, theme: &Theme) -> (Vec<String>, Style) {
    let parts = match row {
        Row::Disk(position) => sample.disks.get(position).map_or_else(Vec::new, |disk| {
            let figures = pair(disk.used_bytes, disk.size_bytes);
            if disk.shared {
                vec!["shared".to_string(), figures]
            } else {
                vec![figures]
            }
        }),
        Row::Cpu => sample.cpu.as_ref().map_or_else(Vec::new, cpu_figures),
        Row::Memory => sample
            .memory
            .map(|memory| vec![pair(memory.used_bytes, memory.limit_bytes)])
            .unwrap_or_default(),
        Row::Gpu(index) => gpu(sample, index).map(gpu_figures).unwrap_or_default(),
        Row::Rest(from) => vec![format!("avg of {}", sample.gpus.len().saturating_sub(from))],
    };
    let style = match row {
        Row::Gpu(index) => level(
            gpu(sample, index).and_then(Gpu::memory_ratio),
            theme.dim,
            theme,
        ),
        _ => theme.dim,
    };
    (parts, style)
}

/// A CPU's figures. Inside a container the load average is the host's, so
/// the cores busy (usage times CPUs) show instead: `3.2/8 cores`; on a host,
/// the load over the CPU count: `load 6.4/16`.
fn cpu_figures(cpu: &Cpu) -> Vec<String> {
    let text = match (cpu.container, cpu.usage, cpu.load1, cpu.cpus) {
        (true, Some(usage), _, Some(cpus)) => {
            format!("{}/{} cores", number(usage * cpus), number(cpus))
        },
        (true, None, _, Some(cpus)) => format!("{} cores", number(cpus)),
        (false, _, Some(load), Some(cpus)) => format!("load {}/{}", number(load), number(cpus)),
        (false, _, Some(load), None) => format!("load {}", number(load)),
        _ => return Vec::new(),
    };
    vec![text]
}

/// A GPU's memory, temperature and power: `40/80G 64C 312W`, each left out
/// when unknown.
fn gpu_figures(gpu: &Gpu) -> Vec<String> {
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
    parts
}

/// `used` and `total` bytes in the unit of `total`: `31/64G`.
fn pair(used: u64, total: u64) -> String {
    let mut scale = 1.0;
    let mut unit = "B";
    for next in ["K", "M", "G", "T", "P"] {
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
    fn gpu_figures_drop_whole_parts_to_fit() {
        let h200 = Gpu {
            index: 0,
            name: "NVIDIA H200".into(),
            utilization: Some(0.9),
            memory_used_bytes: Some(141 * 1024 * 1024 * 1024),
            memory_total_bytes: Some(141 * 1024 * 1024 * 1024),
            temperature_celsius: Some(60.0),
            power_watts: Some(700.0),
            power_limit_watts: Some(700.0),
        };
        let parts = gpu_figures(&h200);
        assert_eq!(fit(parts.clone(), 20), "141/141G 60C 700W");
        // Power goes first, then the temperature; never half a figure.
        assert_eq!(fit(parts.clone(), 15), "141/141G 60C");
        assert_eq!(fit(parts.clone(), 11), "141/141G");
        assert_eq!(fit(parts, 5), "141/…");
    }

    #[test]
    fn a_container_shows_its_cores_busy_not_the_host_load() {
        let cpu = |container, usage| Cpu {
            usage,
            load1: Some(40.0),
            cpus: Some(8.0),
            container,
            times: None,
        };
        assert_eq!(cpu_figures(&cpu(true, Some(0.4))), ["3.2/8.0 cores"]);
        assert_eq!(cpu_figures(&cpu(true, None)), ["8.0 cores"]);
        assert_eq!(cpu_figures(&cpu(false, Some(0.4))), ["load 40/8.0"]);
    }

    #[test]
    fn the_sparkline_keeps_the_peaks_and_skips_what_is_missing() {
        let sample = |usage| SystemSample {
            at: SystemTime::UNIX_EPOCH,
            disks: Vec::new(),
            cpu: Some(Cpu {
                usage,
                load1: None,
                cpus: None,
                container: false,
                times: None,
            }),
            memory: None,
            gpus: Vec::new(),
        };
        let few: VecDeque<SystemSample> = [None, Some(0.2)].map(sample).into();
        assert_eq!(history(&few, Row::Cpu, 7), [None, Some(20)]);
        let many: VecDeque<SystemSample> = (0..60)
            .map(|index| sample((index == 10).then_some(0.9).or(Some(0.1))))
            .collect();
        let cells = history(&many, Row::Cpu, 7);
        assert_eq!(cells.len(), 7);
        assert_eq!(cells.get(1), Some(&Some(90)));
        assert_eq!(cells.first(), Some(&Some(10)));
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
        if let Some(gpu) = sample.gpus.get_mut(4) {
            gpu.utilization = Some(0.5);
        }
        if let Some(gpu) = sample.gpus.get_mut(5) {
            gpu.utilization = None;
        }
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
