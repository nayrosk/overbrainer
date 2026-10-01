//! What a running job is doing, from its metric lines: training, evaluating, or
//! the work after the last step that a step count cannot show.

use super::metrics::{JobStage, Mark, MetricLine, TrainMetric};

/// What a run's job is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Training steps.
    Training,
    /// An evaluation, during the training or after its last step.
    Evaluating {
        /// Prediction steps done.
        step: u64,
        /// Prediction steps of the whole evaluation, when known.
        total: Option<u64>,
        /// Whether the training loop is over: the final evaluation.
        last: bool,
    },
    /// The training loop is over; the trainer saves the model.
    Finalizing,
    /// The adapter is merged into the base model.
    Merging,
    /// The model is exported to GGUF.
    Exporting,
    /// The job exited; its results are being retrieved.
    Retrieving,
}

impl Phase {
    /// The phase as the TUI and the command line show it, for example
    /// `evaluating 340/1200` or `finalizing (saving model)`.
    #[must_use]
    pub fn label(self) -> String {
        match self {
            Self::Training => "training".to_string(),
            Self::Evaluating { step, total, last } => {
                let done = match total {
                    Some(total) => format!("{step}/{total}"),
                    None => step.to_string(),
                };
                if last {
                    format!("finalizing (evaluation {done})")
                } else {
                    format!("evaluating {done}")
                }
            },
            Self::Finalizing => "finalizing (saving model)".to_string(),
            Self::Merging => "merging adapter".to_string(),
            Self::Exporting => "exporting GGUF".to_string(),
            Self::Retrieving => "retrieving results".to_string(),
        }
    }

    /// One lowercase word, as the `phase` label of Prometheus has it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Training => "training",
            Self::Evaluating { .. } => "evaluating",
            Self::Finalizing => "finalizing",
            Self::Merging => "merging",
            Self::Exporting => "exporting",
            Self::Retrieving => "retrieving",
        }
    }

    /// Every [`Phase::name`].
    pub const NAMES: [&'static str; 6] = [
        "training",
        "evaluating",
        "finalizing",
        "merging",
        "exporting",
        "retrieving",
    ];
}

/// What the lines of a run said so far that tells its phase.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Phases {
    /// The latest stage started.
    stage: Option<JobStage>,
    /// The latest evaluation step, when no log came after it.
    eval: Option<(u64, Option<u64>)>,
    /// Whether the training loop said it ended.
    ended: bool,
    /// Latest optimizer step.
    step: u64,
    /// Total steps, when known.
    max_steps: Option<u64>,
}

impl Phases {
    /// Takes the next line of `metrics.jsonl` into account.
    pub fn line(&mut self, line: &MetricLine) {
        match line {
            MetricLine::Begin { max_steps, .. } => self.max_steps = max_steps.or(self.max_steps),
            MetricLine::Log(metric) => self.metric(metric),
            other => {
                if let Some(mark) = other.mark() {
                    self.mark(mark);
                }
            },
        }
    }

    /// Takes the next log into account: an evaluation it follows is over.
    pub fn metric(&mut self, metric: &TrainMetric) {
        self.step = self.step.max(metric.step);
        self.max_steps = metric.max_steps.or(self.max_steps);
        self.eval = None;
    }

    /// Takes the next mark into account.
    pub fn mark(&mut self, mark: Mark) {
        match mark {
            Mark::Stage(stage) => self.stage = Some(stage),
            Mark::Eval { step, total } => self.eval = Some((step, total)),
            Mark::End { step } => {
                self.ended = true;
                self.step = self.step.max(step);
            },
        }
    }

    /// The phase of a job still running, or one that `exited` while its run is
    /// still recorded running. A merge or export stage wins; then an
    /// evaluation no log followed; then the end of the training loop, said by
    /// the plugin or read from the step count (runs from before these events);
    /// else training.
    #[must_use]
    pub fn phase(&self, exited: bool) -> Phase {
        if exited {
            return Phase::Retrieving;
        }
        match self.stage {
            Some(JobStage::Merge) => return Phase::Merging,
            Some(JobStage::Export) => return Phase::Exporting,
            Some(JobStage::Train) | None => {},
        }
        let last = self.ended
            || self
                .max_steps
                .is_some_and(|max| max > 0 && self.step >= max);
        match self.eval {
            Some((step, total)) => Phase::Evaluating { step, total, last },
            None if last => Phase::Finalizing,
            None => Phase::Training,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::parse_line;

    fn phases(lines: &[&str]) -> Result<Phases, serde_json::Error> {
        let mut phases = Phases::default();
        for line in lines {
            phases.line(&parse_line(line)?);
        }
        Ok(phases)
    }

    const BEGIN: &str = r#"{"event":"begin","time":1,"max_steps":100}"#;
    const STAGE_TRAIN: &str = r#"{"event":"stage","name":"train","time":1}"#;
    const STEP_50: &str = r#"{"event":"log","time":2,"step":50,"max_steps":100,"loss":1.0}"#;
    const EVAL_50: &str = r#"{"event":"log","time":3,"step":50,"max_steps":100,"eval_loss":1.0}"#;
    const STEP_100: &str = r#"{"event":"log","time":4,"step":100,"max_steps":100,"loss":0.9}"#;
    const EVAL: &str = r#"{"event":"eval","time":3,"step":340,"total":1200}"#;
    const END: &str = r#"{"event":"end","time":5,"step":100}"#;

    #[test]
    fn the_phase_follows_the_rule() -> Result<(), serde_json::Error> {
        let mid_eval = Phase::Evaluating {
            step: 340,
            total: Some(1200),
            last: false,
        };
        let final_eval = Phase::Evaluating {
            step: 340,
            total: Some(1200),
            last: true,
        };
        let cases: [(&[&str], Phase); 9] = [
            (&[], Phase::Training),
            (&[BEGIN, STAGE_TRAIN, STEP_50], Phase::Training),
            (&[BEGIN, STEP_50, EVAL], mid_eval),
            (&[BEGIN, STEP_50, EVAL, EVAL_50], Phase::Training),
            (&[BEGIN, STEP_100, EVAL], final_eval),
            (&[BEGIN, STEP_50, END], Phase::Finalizing),
            // A run from before these events: the step count alone.
            (&[BEGIN, STEP_100], Phase::Finalizing),
            (
                &[
                    STAGE_TRAIN,
                    STEP_100,
                    END,
                    r#"{"event":"stage","name":"merge","time":6}"#,
                ],
                Phase::Merging,
            ),
            (
                &[STEP_100, r#"{"event":"stage","name":"export","time":6}"#],
                Phase::Exporting,
            ),
        ];
        for (lines, expected) in cases {
            assert_eq!(phases(lines)?.phase(false), expected, "{lines:?}");
        }
        assert_eq!(phases(&[BEGIN, STEP_50])?.phase(true), Phase::Retrieving);
        Ok(())
    }

    #[test]
    fn labels_say_what_the_job_does() {
        let labels = [
            Phase::Training,
            Phase::Evaluating {
                step: 340,
                total: Some(1200),
                last: false,
            },
            Phase::Evaluating {
                step: 340,
                total: None,
                last: true,
            },
            Phase::Finalizing,
            Phase::Merging,
            Phase::Exporting,
            Phase::Retrieving,
        ]
        .map(Phase::label);
        assert_eq!(
            labels,
            [
                "training",
                "evaluating 340/1200",
                "finalizing (evaluation 340)",
                "finalizing (saving model)",
                "merging adapter",
                "exporting GGUF",
                "retrieving results",
            ]
        );
        assert!(Phase::NAMES.contains(&Phase::Exporting.name()));
    }
}
