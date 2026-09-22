//! Renders pipeline and training events as log lines on stderr.

use std::time::{Duration, Instant};

use tokio::sync::broadcast::Receiver;
use tokio::sync::broadcast::error::RecvError;

use crate::events::{Event, Stage, StageStats};
use crate::exec::JobStatus;
use crate::train::TrainMetric;

/// Training logs come at every step: at most one is shown per interval, plus every
/// evaluation.
const METRIC_INTERVAL: Duration = Duration::from_secs(10);

/// Logs events until every sender is dropped.
pub(crate) async fn render(mut receiver: Receiver<Event>) {
    let mut progress = Progress::default();
    loop {
        match receiver.recv().await {
            Ok(event) => progress.log(&event),
            Err(RecvError::Lagged(skipped)) => lagged(skipped),
            Err(RecvError::Closed) => break,
        }
    }
}

fn lagged(skipped: u64) {
    tracing::debug!("{skipped} progress events skipped");
}

/// Counts finished items to log progress every tenth of the stage.
///
/// `StageStarted.total` counts only the items processed in this run (skipped items
/// publish nothing), and each processed item ends with one `ItemDone` or one final
/// `ItemFailed { retryable: false }`. Retryable failures are progress noise: they are
/// logged but never counted. A failure that covers several items (for example a
/// topic whose deduplicator cannot be seeded) counts once, so a stage may finish
/// below its total.
#[derive(Debug, Default)]
struct Progress {
    total: usize,
    finished: usize,
    next_tenth: usize,
    last_metric: Option<Instant>,
}

impl Progress {
    fn log(&mut self, event: &Event) {
        match event {
            Event::StageStarted { stage, total } => self.started(*stage, *total),
            Event::ItemDone { stage, id, .. } => self.done(*stage, id),
            Event::ItemFailed {
                stage,
                id,
                error,
                retryable,
            } => self.failed(*stage, id, error, *retryable),
            Event::StageFinished { stage, stats } => finished(*stage, stats),
            Event::Metric(metric) => {
                if self.show_metric(metric, Instant::now()) {
                    training(metric);
                }
            },
            Event::JobStatus(status) => job(*status),
        }
    }

    fn started(&mut self, stage: Stage, total: usize) {
        *self = Self {
            total,
            finished: 0,
            next_tenth: 1,
            last_metric: None,
        };
        tracing::info!("{stage}: {total} to process");
    }

    fn done(&mut self, stage: Stage, id: &str) {
        tracing::debug!("{stage}: {id} done");
        self.report(stage);
    }

    fn failed(&mut self, stage: Stage, id: &str, error: &str, retryable: bool) {
        if retryable {
            retrying(stage, id, error);
        } else {
            gave_up(stage, id, error);
            self.report(stage);
        }
    }

    fn report(&mut self, stage: Stage) {
        if let Some(finished) = self.advance() {
            step(stage, finished, self.total);
        }
    }

    /// Whether to show `metric`, received at `now`: evaluations always, training logs
    /// at most once per [`METRIC_INTERVAL`].
    fn show_metric(&mut self, metric: &TrainMetric, now: Instant) -> bool {
        if metric.eval_loss.is_none()
            && self
                .last_metric
                .is_some_and(|last| now.duration_since(last) < METRIC_INTERVAL)
        {
            return false;
        }
        self.last_metric = Some(now);
        true
    }

    /// Counts one finished item. Returns the count to log when it reaches the next
    /// tenth of the total, never more than the total.
    fn advance(&mut self) -> Option<usize> {
        if self.finished >= self.total {
            return None;
        }
        self.finished += 1;
        if self.finished * 10 < self.next_tenth * self.total {
            return None;
        }
        self.next_tenth = self.finished * 10 / self.total + 1;
        Some(self.finished)
    }
}

fn step(stage: Stage, finished: usize, total: usize) {
    tracing::info!("{stage}: {finished}/{total}");
}

fn retrying(stage: Stage, id: &str, error: &str) {
    tracing::warn!("{stage}: {id}: {error}");
}

fn gave_up(stage: Stage, id: &str, error: &str) {
    tracing::error!("{stage}: {id} failed: {error}");
}

fn finished(stage: Stage, stats: &StageStats) {
    tracing::info!("{stage}: finished");
    tracing::debug!("{stage}: {stats:?}");
}

fn training(metric: &TrainMetric) {
    tracing::info!("train: {}", metric_line(metric));
}

fn job(status: JobStatus) {
    tracing::info!("train: job {}", status_name(status));
}

/// A job status in words.
#[must_use]
pub fn status_name(status: JobStatus) -> String {
    match status {
        JobStatus::Running => "running".to_string(),
        JobStatus::Exited(code) => format!("exited with code {code}"),
        JobStatus::Cancelled => "cancelled".to_string(),
        JobStatus::Lost => "lost (stopped without an exit code)".to_string(),
    }
}

/// Step, epoch and the logged values of `metric`, for example
/// `step 12/40 (30%), epoch 0.90, loss 1.2345, lr 2.00e-4, grad_norm 0.812`.
#[must_use]
pub fn metric_line(metric: &TrainMetric) -> String {
    let mut parts = vec![match metric.max_steps.filter(|max| *max > 0) {
        Some(max) => format!("step {}/{max} ({}%)", metric.step, metric.step * 100 / max),
        None => format!("step {}", metric.step),
    }];
    if let Some(epoch) = metric.epoch {
        parts.push(format!("epoch {epoch:.2}"));
    }
    let values = [
        ("loss", metric.loss, false),
        ("eval_loss", metric.eval_loss, false),
        ("lr", metric.learning_rate, true),
        ("grad_norm", metric.grad_norm, false),
    ];
    for (name, value, scientific) in values {
        match value {
            Some(value) if scientific => parts.push(format!("{name} {value:.2e}")),
            Some(value) => parts.push(format!("{name} {value:.4}")),
            None => {},
        }
    }
    parts.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn started(total: usize) -> Progress {
        let mut progress = Progress::default();
        progress.log(&Event::StageStarted {
            stage: Stage::Answers,
            total,
        });
        progress
    }

    fn failure(retryable: bool) -> Event {
        Event::ItemFailed {
            stage: Stage::Answers,
            id: "q".to_string(),
            error: "boom".to_string(),
            retryable,
        }
    }

    #[test]
    fn logs_each_tenth_once() {
        let mut progress = started(20);
        let logged: Vec<usize> = (0..20).filter_map(|_| progress.advance()).collect();
        assert_eq!(logged, vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);
    }

    #[test]
    fn a_small_stage_logs_every_item() {
        let mut progress = started(3);
        let logged: Vec<usize> = (0..3).filter_map(|_| progress.advance()).collect();
        assert_eq!(logged, vec![1, 2, 3]);
    }

    #[test]
    fn only_final_failures_count() {
        let mut progress = started(4);
        progress.log(&failure(true));
        progress.log(&failure(true));
        assert_eq!(progress.finished, 0);
        progress.log(&failure(false));
        assert_eq!(progress.finished, 1);
    }

    fn metric(step: u64, eval_loss: Option<f64>) -> TrainMetric {
        TrainMetric {
            time: 0.0,
            step,
            epoch: Some(0.5),
            max_steps: Some(40),
            loss: eval_loss.is_none().then_some(1.234_56),
            eval_loss,
            learning_rate: eval_loss.is_none().then_some(2e-4),
            grad_norm: None,
        }
    }

    #[test]
    fn training_logs_are_throttled_but_evaluations_are_not() {
        let mut progress = Progress::default();
        let start = Instant::now();
        assert!(progress.show_metric(&metric(1, None), start));
        assert!(!progress.show_metric(&metric(2, None), start + Duration::from_secs(5)));
        assert!(progress.show_metric(&metric(3, Some(1.1)), start + Duration::from_secs(6)));
        assert!(!progress.show_metric(&metric(4, None), start + Duration::from_secs(15)));
        assert!(progress.show_metric(&metric(5, None), start + Duration::from_secs(17)));
    }

    #[test]
    fn metric_lines_show_progress_and_values() {
        assert_eq!(
            metric_line(&metric(12, None)),
            "step 12/40 (30%), epoch 0.50, loss 1.2346, lr 2.00e-4"
        );
        assert_eq!(
            metric_line(&metric(40, Some(0.9))),
            "step 40/40 (100%), epoch 0.50, eval_loss 0.9000"
        );
        let bare = TrainMetric {
            max_steps: None,
            epoch: None,
            ..metric(3, Some(1.0))
        };
        assert_eq!(metric_line(&bare), "step 3, eval_loss 1.0000");
    }

    #[test]
    fn never_counts_past_the_total() {
        let mut progress = started(1);
        assert_eq!(progress.advance(), Some(1));
        assert_eq!(progress.advance(), None);
        assert_eq!(progress.finished, 1);
        let mut empty = started(0);
        assert_eq!(empty.advance(), None);
    }
}
