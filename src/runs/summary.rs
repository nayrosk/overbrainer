use crate::events::Event;
use crate::train::{MetricLine, TrainMetric, parse_line};

/// What the metric lines of a run add up to.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetricsSummary {
    /// Whether the plugin wrote its start line, proof that it was loaded.
    pub begun: bool,
    /// Lines the plugin wrote, start line included; the stage lines the job
    /// writes itself are not counted.
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
    /// Adds one line of `metrics.jsonl`. Returns the event to publish for it, a
    /// [`Event::Metric`] or an [`Event::Mark`], if any; a malformed line is
    /// counted and logged.
    pub fn add(&mut self, line: &str) -> Option<Event> {
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
                Some(Event::Metric(metric))
            },
            Ok(line @ MetricLine::Stage { .. }) => line.mark().map(Event::Mark),
            Ok(line @ (MetricLine::Eval { .. } | MetricLine::End { .. })) => {
                self.lines += 1;
                line.mark().map(Event::Mark)
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
    use crate::train::{JobStage, Mark};

    #[test]
    fn lines_add_up() {
        let mut summary = MetricsSummary::default();
        summary.add(r#"{"event": "begin", "time": 1, "max_steps": 10}"#);
        let metric = summary.add(
            r#"{"event": "log", "time": 2, "step": 4, "epoch": 0.4, "max_steps": 10, "loss": 1.5}"#,
        );
        assert!(matches!(metric, Some(Event::Metric(m)) if m.step == 4));
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

    #[test]
    fn stage_lines_are_marks_the_plugin_did_not_write() {
        let mut summary = MetricsSummary::default();
        let stage = summary.add(r#"{"event":"stage","name":"train","time":1700000000}"#);
        assert_eq!(stage, Some(Event::Mark(Mark::Stage(JobStage::Train))));
        // Without the plugin's own lines, the job wrote no metrics.
        assert_eq!((summary.lines, summary.malformed), (0, 0));
        let end = summary.add(r#"{"event":"end","time":2,"step":9}"#);
        assert_eq!(end, Some(Event::Mark(Mark::End { step: 9 })));
        assert_eq!(summary.lines, 1);
    }
}
