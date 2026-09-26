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
    /// Started, then its task ended first: interrupted, failed or panicked.
    Stopped,
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

    /// The share of its items finished, 1 when it has none.
    pub(super) fn ratio(&self) -> f64 {
        let count = |n: usize| super::training::float(u64::try_from(n).unwrap_or(u64::MAX));
        if self.total == 0 {
            1.0
        } else {
            count(self.finished) / count(self.total)
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

/// The single stage a stage command runs, or `None` for `run`, which runs them all.
fn command_stage(command: Command) -> Option<Stage> {
    match command {
        Command::Subtopics => Some(Stage::Subtopics),
        Command::Questions => Some(Stage::Questions),
        Command::Answers => Some(Stage::Answers),
        Command::Split => Some(Stage::Split),
        Command::Run => None,
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
    /// A new task running `command` with `concurrency` requests at a time. A stage
    /// command keeps the rows of the finished stages it does not run, so their tokens
    /// and cost stay on screen when stages are run one at a time; `run` starts every
    /// stage afresh.
    pub(super) fn started(&mut self, command: Command, concurrency: usize) {
        let rows = std::array::from_fn(|i| match command_stage(command) {
            None => Row::new(StageState::Pending),
            Some(stage) if stage == STAGES[i] => Row::new(StageState::Idle),
            Some(_) => self.rows[i].clone(),
        });
        *self = Self {
            command: Some(command),
            running: true,
            concurrency: concurrency.max(1),
            rows,
            ..Self::default()
        };
    }

    /// Counts `event`, of the task's bus.
    pub(super) fn event(&mut self, event: &Event) {
        match event {
            Event::StageStarted { stage, total } => {
                let state = if self.running {
                    StageState::Running
                } else {
                    StageState::Stopped
                };
                let row = &mut self.rows[index(*stage)];
                *row = Row::new(state);
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
                row.finished = row.total;
                row.failed = stats.failed;
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
        if !self.running {
            return 0;
        }
        let row = &self.rows[index(stage)];
        match (row.state, stage) {
            (StageState::Running, Stage::Answers) => self.concurrency.min(row.total - row.finished),
            (StageState::Running, Stage::Subtopics | Stage::Questions) => {
                usize::from(row.finished < row.total)
            },
            _ => 0,
        }
    }

    /// The task ended: a stage it was running is stopped, and a stage it had
    /// not reached is no longer pending.
    pub(super) fn stopped(&mut self) {
        self.running = false;
        for row in &mut self.rows {
            row.state = match row.state {
                StageState::Running => StageState::Stopped,
                StageState::Pending => StageState::Idle,
                state => state,
            };
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

    #[test]
    fn a_finished_stage_counts_every_item_even_without_item_events() {
        let mut view = PipelineView::default();
        view.started(Command::Split, 1);
        view.event(&Event::StageStarted {
            stage: Stage::Split,
            total: 12,
        });
        view.event(&Event::StageFinished {
            stage: Stage::Split,
            stats: StageStats {
                done: 11,
                failed: 1,
                ..StageStats::default()
            },
        });
        let row = view.row(Stage::Split);
        assert_eq!(
            (row.state, row.finished, row.total, row.failed),
            (StageState::Done, 12, 12, 1)
        );
    }

    #[test]
    fn running_a_later_stage_keeps_the_finished_stage_tokens_and_cost() {
        let mut view = PipelineView::default();
        view.started(Command::Subtopics, 1);
        view.event(&Event::StageStarted {
            stage: Stage::Subtopics,
            total: 1,
        });
        view.event(&Event::StageFinished {
            stage: Stage::Subtopics,
            stats: StageStats {
                done: 1,
                usage: Usage {
                    input_tokens: 40,
                    output_tokens: 60,
                },
                cost: Some(0.25),
                ..StageStats::default()
            },
        });
        view.stopped();
        // Running questions on its own must not wipe the subtopics totals.
        view.started(Command::Questions, 1);
        let subtopics = view.row(Stage::Subtopics);
        assert_eq!(subtopics.state, StageState::Done);
        assert_eq!(subtopics.usage.output_tokens, 60);
        assert_eq!(
            subtopics.stats.as_ref().and_then(|stats| stats.cost),
            Some(0.25)
        );
        assert_eq!(view.usage().output_tokens, 60);
        assert_eq!(
            view.row(Stage::Questions).state,
            StageState::Idle,
            "the stage being run starts fresh"
        );
    }

    #[test]
    fn a_full_run_starts_every_stage_afresh() {
        let mut view = PipelineView::default();
        view.started(Command::Subtopics, 1);
        view.event(&Event::StageFinished {
            stage: Stage::Subtopics,
            stats: StageStats {
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 20,
                },
                ..StageStats::default()
            },
        });
        view.started(Command::Run, 4);
        assert_eq!(view.row(Stage::Subtopics).state, StageState::Pending);
        assert_eq!(
            view.usage().output_tokens,
            0,
            "run clears the earlier totals"
        );
    }

    #[test]
    fn a_stopped_task_shows_no_running_row_and_nothing_in_flight() {
        let mut view = PipelineView::default();
        view.started(Command::Run, 8);
        view.event(&Event::StageStarted {
            stage: Stage::Answers,
            total: 10,
        });
        view.event(&done(Stage::Answers, 5));
        assert_eq!(view.in_flight(Stage::Answers), 8);
        view.stopped();
        assert_eq!(view.row(Stage::Answers).state, StageState::Stopped);
        assert_eq!(view.row(Stage::Split).state, StageState::Idle, "never ran");
        assert_eq!(view.in_flight(Stage::Answers), 0);
        assert_eq!(view.progress(), None);
        view.event(&Event::StageStarted {
            stage: Stage::Questions,
            total: 3,
        });
        assert_eq!(
            view.row(Stage::Questions).state,
            StageState::Stopped,
            "a late start is not running"
        );
        assert_eq!(view.in_flight(Stage::Questions), 0);
    }
}
