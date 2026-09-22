//! The metrics plugin shipped with each run, and the lines it writes.

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
    fn other_lines_are_errors() {
        assert!(parse_line(r#"{"event": "other"}"#).is_err());
        assert!(parse_line("not json").is_err());
    }
}
