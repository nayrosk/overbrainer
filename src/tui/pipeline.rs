//! The Pipeline view's model: what the events of the current pipeline task say
//! of each stage, counted like the command line's progress lines.

use std::collections::VecDeque;

use crate::cli::data::Command;
use crate::events::{Event, Stage, StageStats};
use crate::llm::Usage;

/// The stages, in the order they run.
pub(super) const STAGES: [Stage; 4] = [
    Stage::Subtopics,
    Stage::Questions,
    Stage::Answers,
    Stage::Split,
];

/// Item failures kept for the view.
const ERRORS: usize = 5;

/// Where a stage is in the current task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StageState {
    /// Not part of the task, or not reached.
    Idle,
    /// Part of `run`, not started yet.
    Pending,
    /// Started.
    Running,
    /// Finished.
    Done,
}

/// One stage's counters.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Row {
    /// Where it is.
    pub(super) state: StageState,
    /// Items to process, from `StageStarted`.
    pub(super) total: usize,
    /// Items done or failed for good, at most `total`.
    pub(super) finished: usize,
    /// Failed attempts that are tried again.
    pub(super) retries: usize,
    /// Items that failed for good.
    pub(super) failed: usize,
    /// Tokens so far, then the stage's own count once it finished.
    pub(super) usage: Usage,
    /// The stage's final counters.
    pub(super) stats: Option<StageStats>,
}

impl Row {
    fn new(state: StageState) -> Self {
        Self {
            state,
            total: 0,
            finished: 0,
            retries: 0,
            failed: 0,
            usage: Usage::default(),
            stats: None,
        }
    }
}

/// A recent item failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Failure {
    /// Its stage.
    pub(super) stage: Stage,
    /// Its item.
    pub(super) id: String,
    /// What happened.
    pub(super) error: String,
}

/// The Pipeline view's state: the current or last pipeline task.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PipelineView {
    /// The command of the current or last task, if any ran.
    pub(super) command: Option<Command>,
    /// Whether it still runs.
    pub(super) running: bool,
    /// `pipeline.concurrency` when it started.
    pub(super) concurrency: usize,
    /// One row per stage of [`STAGES`].
    pub(super) rows: [Row; 4],
    /// The newest item failures, oldest first.
    pub(super) errors: VecDeque<Failure>,
    /// The summary lines the command line would print.
    pub(super) results: Vec<String>,
    /// How the task ended.
    pub(super) outcome: Option<Result<(), String>>,
    /// Events its forwarder skipped.
    pub(super) skipped: u64,
}

impl Default for PipelineView {
    fn default() -> Self {
        Self {
            command: None,
            running: false,
            concurrency: 1,
            rows: std::array::from_fn(|_| Row::new(StageState::Idle)),
            errors: VecDeque::new(),
            results: Vec::new(),
            outcome: None,
            skipped: 0,
        }
    }
}

/// The row of `stage` in [`STAGES`].
fn index(stage: Stage) -> usize {
    match stage {
        Stage::Subtopics => 0,
        Stage::Questions => 1,
        Stage::Answers => 2,
        Stage::Split => 3,
    }
}

/// The name `command` has in the `r` menu.
pub(super) fn command_name(command: Command) -> &'static str {
    match command {
        Command::Subtopics => "subtopics",
        Command::Questions => "questions",
        Command::Answers => "answers",
        Command::Split => "split",
        Command::Run => "run",
    }
}

impl PipelineView {
    /// A new task running `command` with `concurrency` requests at a time.
    pub(super) fn started(&mut self, command: Command, concurrency: usize) {
        let pending = if command == Command::Run {
            StageState::Pending
        } else {
            StageState::Idle
        };
        *self = Self {
            command: Some(command),
            running: true,
            concurrency: concurrency.max(1),
            rows: std::array::from_fn(|_| Row::new(pending)),
            ..Self::default()
        };
    }

    /// Counts `event`, of the task's bus.
    pub(super) fn event(&mut self, event: &Event) {
        match event {
            Event::StageStarted { stage, total } => {
                let row = &mut self.rows[index(*stage)];
                *row = Row::new(StageState::Running);
                row.total = *total;
            },
            Event::ItemDone { stage, usage, .. } => {
                let row = &mut self.rows[index(*stage)];
                row.finished = (row.finished + 1).min(row.total);
                if let Some(usage) = usage {
                    row.usage += *usage;
                }
            },
            Event::ItemFailed {
                stage,
                id,
                error,
                retryable,
            } => self.failed(*stage, id, error, *retryable),
            Event::StageFinished { stage, stats } => {
                let row = &mut self.rows[index(*stage)];
                row.state = StageState::Done;
                row.usage = stats.usage;
                row.stats = Some(stats.clone());
            },
            _ => {},
        }
    }

    fn failed(&mut self, stage: Stage, id: &str, error: &str, retryable: bool) {
        let row = &mut self.rows[index(stage)];
        if retryable {
            row.retries += 1;
        } else {
            row.failed += 1;
            row.finished = (row.finished + 1).min(row.total);
        }
        if self.errors.len() == ERRORS {
            self.errors.pop_front();
        }
        self.errors.push_back(Failure {
            stage,
            id: id.to_string(),
            error: error.to_string(),
        });
    }

    /// Requests in flight in `stage`: the answers stage holds up to
    /// `pipeline.concurrency` requests, retries included; the others one.
    pub(super) fn in_flight(&self, stage: Stage) -> usize {
        let row = &self.rows[index(stage)];
        match (row.state, stage) {
            (StageState::Running, Stage::Answers) => self.concurrency.min(row.total - row.finished),
            (StageState::Running, Stage::Subtopics | Stage::Questions) => {
                usize::from(row.finished < row.total)
            },
            _ => 0,
        }
    }

    /// Tokens of the whole task.
    pub(super) fn usage(&self) -> Usage {
        let mut usage = Usage::default();
        for row in &self.rows {
            usage += row.usage;
        }
        usage
    }

    /// The row of `stage`.
    pub(super) fn row(&self, stage: Stage) -> &Row {
        &self.rows[index(stage)]
    }

    /// The running stage and its progress, for the status line.
    pub(super) fn progress(&self) -> Option<String> {
        let command = self.command?;
        if !self.running {
            return None;
        }
        let running = STAGES
            .iter()
            .find(|stage| self.row(**stage).state == StageState::Running);
        Some(running.map_or_else(
            || command_name(command).to_string(),
            |stage| {
                let row = self.row(*stage);
                format!("{stage} {}/{}", row.finished, row.total)
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn done(stage: Stage, output_tokens: u64) -> Event {
        Event::ItemDone {
            stage,
            id: "x".into(),
            usage: Some(Usage {
                input_tokens: 1,
                output_tokens,
            }),
        }
    }

    fn failed(stage: Stage, retryable: bool) -> Event {
        Event::ItemFailed {
            stage,
            id: "3f1c".into(),
            error: "429 Too Many Requests".into(),
            retryable,
        }
    }

    #[test]
    fn counters_follow_the_command_line_rules() {
        let mut view = PipelineView::default();
        view.started(Command::Run, 8);
        assert_eq!(view.row(Stage::Split).state, StageState::Pending);
        view.event(&Event::StageStarted {
            stage: Stage::Answers,
            total: 10,
        });
        for event in [
            done(Stage::Answers, 5),
            failed(Stage::Answers, true),
            failed(Stage::Answers, true),
            failed(Stage::Answers, false),
            done(Stage::Answers, 7),
        ] {
            view.event(&event);
        }
        let row = view.row(Stage::Answers);
        assert_eq!((row.finished, row.retries, row.failed), (3, 2, 1));
        assert_eq!(view.in_flight(Stage::Answers), 7);
        assert_eq!(view.usage().output_tokens, 12);
        assert_eq!(view.errors.len(), 3);
        assert_eq!(view.progress().as_deref(), Some("answers 3/10"));
        let stats = StageStats {
            done: 9,
            usage: Usage {
                input_tokens: 100,
                output_tokens: 200,
            },
            cost: Some(0.5),
            ..StageStats::default()
        };
        view.event(&Event::StageFinished {
            stage: Stage::Answers,
            stats,
        });
        assert_eq!(
            view.usage().output_tokens,
            200,
            "final stats replace live ones"
        );
        assert_eq!(view.in_flight(Stage::Answers), 0);
    }

    #[test]
    fn finished_never_passes_the_total_and_errors_keep_the_newest_five() {
        let mut view = PipelineView::default();
        view.started(Command::Answers, 2);
        view.event(&Event::StageStarted {
            stage: Stage::Answers,
            total: 1,
        });
        for _ in 0..7 {
            view.event(&failed(Stage::Answers, false));
        }
        assert_eq!(view.row(Stage::Answers).finished, 1);
        assert_eq!(view.errors.len(), 5);
        assert_eq!(view.row(Stage::Subtopics).state, StageState::Idle);
    }
}
