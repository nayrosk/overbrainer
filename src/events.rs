//! Progress events published by pipeline stages and training runs. The core never
//! renders: the CLI turns events into log lines, the TUI (later) into views.

use std::fmt;

use tokio::sync::broadcast;

use crate::exec::JobStatus;
use crate::llm::Usage;
use crate::pricing::Price;
use crate::runpod::PodStatus;
use crate::train::TrainMetric;

/// A pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Subtopics of each topic.
    Subtopics,
    /// Questions of each subtopic.
    Questions,
    /// Parent answers to each question.
    Answers,
    /// Train and eval split.
    Split,
}

impl Stage {
    /// Lowercase name, as used by the CLI command.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Subtopics => "subtopics",
            Self::Questions => "questions",
            Self::Answers => "answers",
            Self::Split => "split",
        }
    }
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Counters of a stage run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StageStats {
    /// Items produced.
    pub done: usize,
    /// Items already present and left untouched.
    pub skipped: usize,
    /// Items that failed after retries.
    pub failed: usize,
    /// Items produced but marked as not usable for training.
    pub excluded: usize,
    /// Tokens used by every request of the stage, retries included.
    pub usage: Usage,
    /// Cost in USD, when the price of the model is known.
    pub cost: Option<f64>,
}

impl StageStats {
    /// Adds `usage` to the totals, and its cost when `price` is known.
    pub fn add_usage(&mut self, usage: Usage, price: Option<&Price>) {
        self.usage += usage;
        if let Some(price) = price {
            *self.cost.get_or_insert(0.0) += price.cost(usage);
        }
    }
}

/// Something that happened in a stage, a training run or its Runpod pod.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Event {
    /// A stage started with `total` items to process.
    StageStarted {
        /// The stage.
        stage: Stage,
        /// Items to process in this run.
        total: usize,
    },
    /// One item was produced.
    ItemDone {
        /// The stage.
        stage: Stage,
        /// ID of the item, or the topic name for the subtopics stage.
        id: String,
        /// Tokens used for this item, when known.
        usage: Option<Usage>,
    },
    /// One attempt or one item failed.
    ItemFailed {
        /// The stage.
        stage: Stage,
        /// ID of the item, or the topic name for the subtopics stage.
        id: String,
        /// Error message. Never contains secrets.
        error: String,
        /// True when the item will be tried again in this run.
        retryable: bool,
    },
    /// A stage finished.
    StageFinished {
        /// The stage.
        stage: Stage,
        /// Final counters.
        stats: StageStats,
    },
    /// A training or evaluation log of the running job.
    Metric(TrainMetric),
    /// The training job is in a new state.
    JobStatus(JobStatus),
    /// The Runpod pod of a run changed.
    PodStatus(PodStatus),
}

/// Broadcast channel of [`Event`]s. Publishing never blocks and never fails: events
/// without subscribers are dropped, and a slow subscriber skips old events.
#[derive(Debug, Clone)]
pub struct EventBus {
    sender: broadcast::Sender<Event>,
}

impl EventBus {
    /// Events kept for a subscriber that falls behind.
    pub const CAPACITY: usize = 1024;

    /// A bus with no subscriber yet, keeping [`Self::CAPACITY`] events.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(Self::CAPACITY)
    }

    /// A bus with no subscriber yet, keeping `capacity` events (at least 1) for a
    /// subscriber that falls behind.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity.max(1));
        Self { sender }
    }

    /// Sends `event` to every current subscriber.
    pub fn publish(&self, event: Event) {
        self.sender.send(event).ok();
    }

    /// Receives every event published from now on.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.sender.subscribe()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn subscribers_receive_published_events() -> Result<(), broadcast::error::RecvError> {
        let bus = EventBus::new();
        bus.publish(Event::StageStarted {
            stage: Stage::Split,
            total: 0,
        });
        let mut receiver = bus.subscribe();
        bus.publish(Event::StageStarted {
            stage: Stage::Answers,
            total: 3,
        });
        assert_eq!(
            receiver.recv().await?,
            Event::StageStarted {
                stage: Stage::Answers,
                total: 3
            }
        );
        Ok(())
    }

    #[test]
    fn cost_is_known_only_with_a_price() {
        let usage = Usage {
            input_tokens: 1_000_000,
            output_tokens: 0,
        };
        let mut stats = StageStats::default();
        stats.add_usage(usage, None);
        assert_eq!(stats.cost, None);
        let price = Price {
            input_per_million: 3.0,
            output_per_million: 15.0,
        };
        stats.add_usage(usage, Some(&price));
        assert_eq!(stats.usage.input_tokens, 2_000_000);
        assert!(stats.cost.is_some_and(|cost| (cost - 3.0).abs() < 1e-9));
    }

    #[tokio::test]
    async fn a_larger_bus_keeps_more_events_for_a_slow_subscriber()
    -> Result<(), broadcast::error::RecvError> {
        let small = EventBus::new();
        let large = EventBus::with_capacity(EventBus::CAPACITY * 2);
        let (mut behind, mut kept) = (small.subscribe(), large.subscribe());
        for total in 0..EventBus::CAPACITY * 2 {
            let event = Event::StageStarted {
                stage: Stage::Answers,
                total,
            };
            small.publish(event.clone());
            large.publish(event);
        }
        assert!(matches!(
            behind.recv().await,
            Err(broadcast::error::RecvError::Lagged(1024))
        ));
        assert_eq!(
            kept.recv().await?,
            Event::StageStarted {
                stage: Stage::Answers,
                total: 0
            }
        );
        Ok(())
    }
}
