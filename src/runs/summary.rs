use crate::train::{MetricLine, TrainMetric, parse_line};

/// What the metric lines of a run add up to.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetricsSummary {
    /// Whether the plugin wrote its start line, proof that it was loaded.
    pub begun: bool,
    /// Metric lines read, start line included.
    pub lines: usize,
    /// Lines that are not metric records, skipped.
    pub malformed: usize,
    /// The latest training log (one carrying a loss).
    pub last_train: Option<TrainMetric>,
    /// The latest evaluation loss.
    pub eval_loss: Option<f64>,
    /// The latest step seen, on any line.
    pub step: u64,
    /// Total steps, when known.
    pub max_steps: Option<u64>,
}

impl MetricsSummary {
    /// Adds one line of `metrics.jsonl`. Returns the metric it holds, if any; a
    /// malformed line is counted and logged.
    pub fn add(&mut self, line: &str) -> Option<TrainMetric> {
        match parse_line(line) {
            Ok(MetricLine::Begin { max_steps, .. }) => {
                self.begun = true;
                self.lines += 1;
                self.max_steps = max_steps.or(self.max_steps);
                None
            },
            Ok(MetricLine::Log(metric)) => {
                self.lines += 1;
                self.step = self.step.max(metric.step);
                self.max_steps = metric.max_steps.or(self.max_steps);
                if metric.eval_loss.is_some() {
                    self.eval_loss = metric.eval_loss;
                }
                if metric.loss.is_some() {
                    self.last_train = Some(metric.clone());
                }
                Some(metric)
            },
            Err(error) => {
                self.malformed += 1;
                malformed(&error);
                None
            },
        }
    }

    /// One line: step, epoch, last loss and last evaluation loss.
    #[must_use]
    pub fn describe(&self) -> String {
        let mut parts = vec![match self.max_steps {
            Some(max) => format!("step {}/{max}", self.step),
            None => format!("step {}", self.step),
        }];
        if let Some(train) = &self.last_train {
            if let Some(epoch) = train.epoch {
                parts.push(format!("epoch {epoch:.2}"));
            }
            if let Some(loss) = train.loss {
                parts.push(format!("loss {loss:.4}"));
            }
        }
        if let Some(eval_loss) = self.eval_loss {
            parts.push(format!("eval_loss {eval_loss:.4}"));
        }
        parts.join(", ")
    }
}

fn malformed(error: &serde_json::Error) {
    tracing::warn!("skipping a malformed metrics line: {error}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_add_up() {
        let mut summary = MetricsSummary::default();
        summary.add(r#"{"event": "begin", "time": 1, "max_steps": 10}"#);
        let metric = summary.add(
            r#"{"event": "log", "time": 2, "step": 4, "epoch": 0.4, "max_steps": 10, "loss": 1.5}"#,
        );
        assert_eq!(metric.map(|metric| metric.step), Some(4));
        summary.add(r#"{"event": "log", "time": 3, "step": 5, "eval_loss": 1.25}"#);
        summary.add("garbage");
        summary.add(r#"{"event": "log", "time": 4, "step": 10, "epoch": 1.0}"#);
        assert!(summary.begun);
        assert_eq!((summary.lines, summary.malformed), (4, 1));
        assert_eq!(summary.step, 10);
        assert_eq!(
            summary.describe(),
            "step 10/10, epoch 0.40, loss 1.5000, eval_loss 1.2500"
        );
        assert_eq!(MetricsSummary::default().describe(), "step 0");
    }
}
