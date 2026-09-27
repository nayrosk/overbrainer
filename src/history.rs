//! The project's stage history: `.overbrainer/history.jsonl`, one line per stage
//! execution, appended when the stage ends. Never rewritten.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::OpenOptions;
use std::io::{self, BufRead as _, BufReader, ErrorKind, Write as _};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::RoleModel;
use crate::events::{Stage, StageStats};
use crate::project_lock::STATE_DIR;

/// File name of the history in the state directory.
pub const HISTORY_FILE: &str = "history.jsonl";

/// Path of the history of `project_dir`.
#[must_use]
pub fn path(project_dir: &Path) -> PathBuf {
    project_dir.join(STATE_DIR).join(HISTORY_FILE)
}

/// How a stage execution ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    /// Every item done or skipped.
    Ok,
    /// Items failed, or a provider error stopped the stage.
    Failed,
    /// Ctrl-C or the TUI stopped it.
    Interrupted,
}

/// What `split` wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitCounts {
    /// Examples in `data/train.jsonl`.
    pub train: usize,
    /// Examples in `data/eval.jsonl`.
    pub eval: usize,
    /// Examples left out as orphaned.
    pub orphaned: usize,
}

/// One stage execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    /// The stage.
    pub stage: Stage,
    /// When it started, RFC 3339 UTC.
    pub started_at: String,
    /// When it ended, RFC 3339 UTC.
    pub ended_at: String,
    /// How it ended.
    pub status: Status,
    /// Provider of the role it used; none for `split`.
    pub provider: Option<String>,
    /// Model of the role it used; none for `split`.
    pub model: Option<String>,
    /// Items produced.
    pub done: usize,
    /// Items already present.
    pub skipped: usize,
    /// Items failed after retries.
    pub failed: usize,
    /// Items produced but not usable for training.
    pub excluded: usize,
    /// Input tokens, retries included.
    pub input_tokens: u64,
    /// Output tokens, reasoning included.
    pub output_tokens: u64,
    /// Cost in USD, when the price was known.
    pub cost: Option<f64>,
    /// What `split` wrote; only on `split` lines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<SplitCounts>,
}

impl Entry {
    /// An entry from a stage's counters.
    #[must_use]
    // The fixed history-entry shape needs all six inputs; `status` and `stats` are
    // both part of the interface and not worth renaming to please the lint.
    #[allow(clippy::too_many_arguments, clippy::similar_names)]
    pub fn from_stats(
        stage: Stage,
        started_at: String,
        ended_at: String,
        status: Status,
        role: Option<&RoleModel>,
        stats: &StageStats,
    ) -> Self {
        Self {
            stage,
            started_at,
            ended_at,
            status,
            provider: role.map(|role| role.provider.clone()),
            model: role.map(|role| role.model.clone()),
            done: stats.done,
            skipped: stats.skipped,
            failed: stats.failed,
            excluded: stats.excluded,
            input_tokens: stats.usage.input_tokens,
            output_tokens: stats.usage.output_tokens,
            cost: stats.cost,
            split: None,
        }
    }
}

/// Appends `entry` to the history of `project_dir`, creating it when needed.
///
/// # Errors
///
/// Returns an error when the file cannot be created or written.
pub fn append(project_dir: &Path, entry: &Entry) -> io::Result<()> {
    let path = path(project_dir);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut line = serde_json::to_string(entry).map_err(io::Error::other)?;
    line.push('\n');
    let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
    // One write per line, so a crash leaves at most a truncated last line.
    file.write_all(line.as_bytes())
}

/// Every entry of the history of `project_dir`, oldest first. A missing file is an
/// empty history; a line that does not parse is logged and skipped.
///
/// # Errors
///
/// Returns an error when the file exists but cannot be read.
pub fn read(project_dir: &Path) -> io::Result<Vec<Entry>> {
    let path = path(project_dir);
    let file = match std::fs::File::open(&path) {
        Ok(file) => file,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut entries = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str(&line) {
            Ok(entry) => entries.push(entry),
            Err(error) => {
                tracing::warn!("skipping line {} of {}: {error}", index + 1, path.display());
            },
        }
    }
    Ok(entries)
}

/// A sum of costs, some of which may be unknown.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Cost {
    /// No cost known.
    #[default]
    Unknown,
    /// Known for some entries only: at least this much.
    Partial(f64),
    /// Known for every entry.
    Known(f64),
}

impl Cost {
    fn add(self, cost: Option<f64>) -> Self {
        match (self, cost) {
            (Self::Unknown, None) => Self::Unknown,
            (Self::Unknown, Some(c)) => Self::Partial(c),
            (Self::Partial(a), c) => Self::Partial(a + c.unwrap_or(0.0)),
            (Self::Known(a), Some(c)) => Self::Known(a + c),
            (Self::Known(a), None) => Self::Partial(a),
        }
    }

    fn first(cost: Option<f64>) -> Self {
        cost.map_or(Self::Unknown, Self::Known)
    }

    fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unknown, x) | (x, Self::Unknown) => match x {
                Self::Known(c) | Self::Partial(c) => Self::Partial(c),
                Self::Unknown => Self::Unknown,
            },
            (Self::Known(a), Self::Known(b)) => Self::Known(a + b),
            (Self::Known(a) | Self::Partial(a), Self::Known(b) | Self::Partial(b)) => {
                Self::Partial(a + b)
            },
        }
    }
}

impl fmt::Display for Cost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown => f.write_str("cost unknown"),
            Self::Partial(c) => write!(f, "${c:.4}+"),
            Self::Known(c) => write!(f, "${c:.4}"),
        }
    }
}

/// The sum of some entries.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Total {
    /// Executions.
    pub runs: usize,
    /// Items produced.
    pub done: usize,
    /// Items failed.
    pub failed: usize,
    /// Input tokens.
    pub input_tokens: u64,
    /// Output tokens.
    pub output_tokens: u64,
    /// Cost.
    pub cost: Cost,
}

impl Total {
    fn add(&mut self, entry: &Entry) {
        self.cost = if self.runs == 0 {
            Cost::first(entry.cost)
        } else {
            self.cost.add(entry.cost)
        };
        self.runs += 1;
        self.done += entry.done;
        self.failed += entry.failed;
        self.input_tokens += entry.input_tokens;
        self.output_tokens += entry.output_tokens;
    }
}

/// Totals per stage, in stage order, and over every entry.
#[must_use]
pub fn totals(entries: &[Entry]) -> (BTreeMap<Stage, Total>, Total) {
    let mut per_stage: BTreeMap<Stage, Total> = BTreeMap::new();
    for entry in entries {
        per_stage.entry(entry.stage).or_default().add(entry);
    }
    let mut all = Total::default();
    for total in per_stage.values() {
        all.cost = if all.runs == 0 {
            total.cost
        } else {
            all.cost.merge(total.cost)
        };
        all.runs += total.runs;
        all.done += total.done;
        all.failed += total.failed;
        all.input_tokens += total.input_tokens;
        all.output_tokens += total.output_tokens;
    }
    (per_stage, all)
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn entry(stage: Stage, cost: Option<f64>) -> Entry {
        let stats = StageStats {
            done: 2,
            failed: 1,
            usage: crate::llm::Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
            cost,
            ..StageStats::default()
        };
        Entry::from_stats(
            stage,
            "2026-09-27T10:00:00Z".into(),
            "2026-09-27T10:01:00Z".into(),
            Status::Ok,
            None,
            &stats,
        )
    }

    #[test]
    fn appended_entries_read_back_in_order() -> TestResult {
        let dir = tempfile::tempdir()?;
        assert!(read(dir.path())?.is_empty());
        let first = entry(Stage::Subtopics, Some(0.5));
        let second = entry(Stage::Answers, None);
        append(dir.path(), &first)?;
        append(dir.path(), &second)?;
        assert_eq!(read(dir.path())?, vec![first, second]);
        Ok(())
    }

    #[test]
    fn a_malformed_line_is_skipped() -> TestResult {
        let dir = tempfile::tempdir()?;
        let good = entry(Stage::Questions, Some(1.0));
        append(dir.path(), &good)?;
        let mut file = OpenOptions::new().append(true).open(path(dir.path()))?;
        writeln!(file, "not json")?;
        append(dir.path(), &good)?;
        assert_eq!(read(dir.path())?.len(), 2);
        Ok(())
    }

    #[test]
    fn totals_mark_a_partly_unknown_cost() {
        let entries = [
            entry(Stage::Answers, Some(1.0)),
            entry(Stage::Answers, None),
            entry(Stage::Questions, Some(0.25)),
            entry(Stage::Subtopics, None),
        ];
        let (per_stage, all) = totals(&entries);
        assert_eq!(per_stage[&Stage::Answers].runs, 2);
        assert_eq!(per_stage[&Stage::Answers].input_tokens, 20);
        assert_eq!(per_stage[&Stage::Answers].cost, Cost::Partial(1.0));
        assert_eq!(per_stage[&Stage::Questions].cost, Cost::Known(0.25));
        assert_eq!(per_stage[&Stage::Subtopics].cost, Cost::Unknown);
        assert_eq!(all.cost, Cost::Partial(1.25));
        assert_eq!(all.runs, 4);
    }

    #[test]
    fn costs_display_with_their_certainty() {
        assert_eq!(Cost::Unknown.to_string(), "cost unknown");
        assert_eq!(Cost::Known(1.5).to_string(), "$1.5000");
        assert_eq!(Cost::Partial(1.5).to_string(), "$1.5000+");
    }
}
