//! The metrics plugin shipped with each run, and the lines it writes.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Source of the Axolotl plugin that writes `metrics.jsonl`. It is uploaded as
/// [`PLUGIN_FILE`] and found through `PYTHONPATH`.
pub const METRICS_PLUGIN: &str = include_str!("overbrainer_metrics.py");

/// File name of the plugin module, inside the run's `plugin/` directory.
pub const PLUGIN_FILE: &str = "overbrainer_metrics.py";

/// The plugin as Axolotl's `plugins` list names it: module, then class.
pub const PLUGIN_CLASS: &str = "overbrainer_metrics.OverbrainerMetricsPlugin";

/// Env variable naming the file the plugin appends to.
pub const METRICS_ENV: &str = "OVERBRAINER_METRICS";

/// Env variable naming the snapshot request the plugin watches for.
pub const SNAPSHOT_ENV: &str = "OVERBRAINER_SNAPSHOT";

/// The snapshot request, relative to the run directory. Once it exists, the
/// plugin saves a checkpoint at the end of the current step and stops training.
/// It holds the reason, one word (see `runs::SnapshotReason`), or nothing.
pub const SNAPSHOT_REQUEST: &str = "snapshot.request";

/// The proof of a snapshot, relative to the run directory, written by the plugin
/// once the checkpoint is saved: `{"checkpoint", "step", "time", "reason"}`, the
/// checkpoint relative to the run directory.
pub const SNAPSHOT_FILE: &str = "snapshot.json";

/// One training log line: the trainer's step and the values it logged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainMetric {
    /// Unix time of the log, in seconds.
    pub time: f64,
    /// Optimizer steps done.
    pub step: u64,
    /// Fractional epoch, when known.
    #[serde(default)]
    pub epoch: Option<f64>,
    /// Total steps of the run, when known.
    #[serde(default)]
    pub max_steps: Option<u64>,
    /// Training loss.
    #[serde(default)]
    pub loss: Option<f64>,
    /// Loss on `data/eval.jsonl`, on evaluation lines.
    #[serde(default)]
    pub eval_loss: Option<f64>,
    /// Learning rate.
    #[serde(default)]
    pub learning_rate: Option<f64>,
    /// Gradient norm.
    #[serde(default)]
    pub grad_norm: Option<f64>,
}

/// A line of `metrics.jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "lowercase")]
pub enum MetricLine {
    /// Training started: the plugin is loaded.
    Begin {
        /// Unix time, in seconds.
        time: f64,
        /// Total steps of the run, when known.
        #[serde(default)]
        max_steps: Option<u64>,
    },
    /// A training or evaluation log.
    Log(TrainMetric),
    /// The job starts one of its commands; written by the job itself, not the
    /// plugin (see [`stage_command`](crate::exec::stage_command)).
    Stage {
        /// Unix time, in seconds.
        time: f64,
        /// The command starting.
        name: JobStage,
    },
    /// An evaluation is under way: at most one line every 2 seconds.
    Eval {
        /// Unix time, in seconds.
        time: f64,
        /// Prediction steps done in this evaluation, from 1.
        step: u64,
        /// Prediction steps of the whole evaluation, when known.
        #[serde(default)]
        total: Option<u64>,
    },
    /// The training loop ended: what follows is the trainer's final work.
    End {
        /// Unix time, in seconds.
        time: f64,
        /// Optimizer steps done.
        step: u64,
    },
}

/// A command of a training job, as its stage event names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStage {
    /// The training itself.
    Train,
    /// The merge of the adapter into the base model.
    Merge,
    /// The export of the model to GGUF.
    Export,
}

impl JobStage {
    /// The name its stage event carries.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Merge => "merge",
            Self::Export => "export",
        }
    }
}

/// What a line other than a log says of the job's progress: the stage, the
/// evaluation and the end lines, without their time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mark {
    /// A command starts.
    Stage(JobStage),
    /// Step `step` of an evaluation of `total` steps, when known.
    Eval {
        /// Prediction steps done.
        step: u64,
        /// Prediction steps of the whole evaluation, when known.
        total: Option<u64>,
    },
    /// The training loop ended at `step`.
    End {
        /// Optimizer steps done.
        step: u64,
    },
}

impl MetricLine {
    /// The mark of a stage, evaluation or end line; `None` for the others.
    #[must_use]
    pub const fn mark(&self) -> Option<Mark> {
        match *self {
            Self::Stage { name, .. } => Some(Mark::Stage(name)),
            Self::Eval { step, total, .. } => Some(Mark::Eval { step, total }),
            Self::End { step, .. } => Some(Mark::End { step }),
            Self::Begin { .. } | Self::Log(_) => None,
        }
    }
}

/// The pace of a training, from its logs in order: steps per second between
/// the first and the latest training log (the ones with a loss), measured on the
/// plugin's own clock, and the time left at that pace.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Pace {
    /// Step and time of the first training log.
    first: Option<(u64, f64)>,
    /// Step and time of the latest training log.
    last: Option<(u64, f64)>,
    /// Latest step, of any log.
    step: u64,
    /// Total steps of the run, as last given.
    max_steps: Option<u64>,
}

impl Pace {
    /// Takes the next log `metric` into account.
    pub fn add(&mut self, metric: &TrainMetric) {
        self.step = metric.step;
        self.max_steps = metric.max_steps.or(self.max_steps);
        if metric.loss.is_some() {
            let point = (metric.step, metric.time);
            self.first.get_or_insert(point);
            self.last = Some(point);
        }
    }

    /// Latest step.
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.step
    }

    /// Total steps of the run, when known.
    #[must_use]
    pub const fn max_steps(&self) -> Option<u64> {
        self.max_steps
    }

    /// Time left to reach the last step at the pace seen so far; `None` until
    /// two training logs apart in steps and time, or without a step count.
    #[must_use]
    pub fn eta(&self) -> Option<Duration> {
        let ((first_step, first_time), (last_step, last_time)) = self.first.zip(self.last)?;
        if last_step <= first_step || last_time <= first_time {
            return None;
        }
        let rate = float(last_step - first_step) / (last_time - first_time);
        let left = float(self.max_steps?.saturating_sub(self.step));
        Duration::try_from_secs_f64(left / rate).ok()
    }
}

/// `count` as a float, saturating at `u32::MAX`.
fn float(count: u64) -> f64 {
    f64::from(u32::try_from(count).unwrap_or(u32::MAX))
}

/// Parses one line of `metrics.jsonl`.
///
/// # Errors
///
/// Returns the JSON error when the line is not a metrics record.
pub fn parse_line(line: &str) -> Result<MetricLine, serde_json::Error> {
    serde_json::from_str(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log(time: f64, step: u64, loss: Option<f64>) -> TrainMetric {
        TrainMetric {
            time,
            step,
            epoch: None,
            max_steps: Some(400),
            loss,
            eval_loss: loss.is_none().then_some(1.0),
            learning_rate: None,
            grad_norm: None,
        }
    }

    #[test]
    fn the_pace_follows_the_training_logs_only() {
        let mut pace = Pace::default();
        pace.add(&log(100.0, 10, Some(2.0)));
        assert_eq!(pace.eta(), None);
        pace.add(&log(150.0, 60, None));
        assert_eq!(pace.eta(), None);
        pace.add(&log(200.0, 110, Some(1.5)));
        assert_eq!((pace.step(), pace.max_steps()), (110, Some(400)));
        assert_eq!(pace.eta(), Some(Duration::from_secs(290)));
    }

    #[test]
    fn the_pace_has_no_eta_without_a_step_count() {
        let mut pace = Pace::default();
        for (time, step) in [(0.0, 1), (10.0, 2)] {
            pace.add(&TrainMetric {
                max_steps: None,
                ..log(time, step, Some(1.0))
            });
        }
        assert_eq!(pace.eta(), None);
    }

    #[test]
    fn log_lines_parse_with_missing_values() -> Result<(), serde_json::Error> {
        let line = r#"{"event": "log", "time": 1.5, "step": 3, "epoch": 0.25, "max_steps": 12, "loss": 1.2, "grad_norm": null, "extra": 1}"#;
        assert_eq!(
            parse_line(line)?,
            MetricLine::Log(TrainMetric {
                time: 1.5,
                step: 3,
                epoch: Some(0.25),
                max_steps: Some(12),
                loss: Some(1.2),
                eval_loss: None,
                learning_rate: None,
                grad_norm: None,
            })
        );
        Ok(())
    }

    #[test]
    fn begin_lines_parse() -> Result<(), serde_json::Error> {
        assert_eq!(
            parse_line(r#"{"event": "begin", "time": 1.0, "max_steps": 40}"#)?,
            MetricLine::Begin {
                time: 1.0,
                max_steps: Some(40)
            }
        );
        Ok(())
    }

    #[test]
    fn stage_eval_and_end_lines_parse_to_marks() -> Result<(), serde_json::Error> {
        let marks = [
            r#"{"event":"stage","name":"train","time":1700000000}"#,
            r#"{"event":"stage","name":"merge","time":1700000000}"#,
            r#"{"event":"stage","name":"export","time":1700000000}"#,
            r#"{"event": "eval", "step": 340, "total": 1200, "time": 1.5}"#,
            r#"{"event": "eval", "step": 3, "time": 1.5}"#,
            r#"{"event": "end", "step": 7206, "time": 2.0}"#,
        ]
        .map(|line| parse_line(line).map(|line| line.mark()));
        let marks: Vec<Option<Mark>> = marks.into_iter().collect::<Result<_, _>>()?;
        assert_eq!(
            marks,
            [
                Some(Mark::Stage(JobStage::Train)),
                Some(Mark::Stage(JobStage::Merge)),
                Some(Mark::Stage(JobStage::Export)),
                Some(Mark::Eval {
                    step: 340,
                    total: Some(1200)
                }),
                Some(Mark::Eval {
                    step: 3,
                    total: None
                }),
                Some(Mark::End { step: 7206 }),
            ]
        );
        assert_eq!(
            parse_line(r#"{"event": "begin", "time": 1.0}"#)?.mark(),
            None
        );
        Ok(())
    }

    #[test]
    fn other_lines_are_errors() {
        assert!(parse_line(r#"{"event": "other"}"#).is_err());
        assert!(parse_line(r#"{"event": "stage", "name": "deploy", "time": 1}"#).is_err());
        assert!(parse_line("not json").is_err());
    }
}
