//! Renders pipeline events as log lines on stderr.

use tokio::sync::broadcast::Receiver;
use tokio::sync::broadcast::error::RecvError;

use crate::events::{Event, Stage, StageStats};

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
        }
    }

    fn started(&mut self, stage: Stage, total: usize) {
        *self = Self {
            total,
            finished: 0,
            next_tenth: 1,
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
