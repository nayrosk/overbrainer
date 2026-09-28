//! Progress events published by pipeline stages and training runs. The core never
//! renders: the CLI turns events into log lines, the TUI (later) into views.

use std::fmt;
use std::sync::Arc;

use tokio::sync::broadcast;

use crate::exec::JobStatus;
use crate::llm::Usage;
use crate::pricing::Price;
use crate::runpod::PodStatus;
use crate::train::TrainMetric;

/// A pipeline stage.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
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

    /// Adds the counts, tokens and cost of `other`.
    pub(crate) fn merge(&mut self, other: &Self) {
        self.done += other.done;
        self.skipped += other.skipped;
        self.failed += other.failed;
        self.excluded += other.excluded;
        self.usage += other.usage;
        if let Some(cost) = other.cost {
            *self.cost.get_or_insert(0.0) += cost;
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
        /// Cost of this item in USD, when the model price is known.
        cost: Option<f64>,
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
    /// The model a stage asks, published right after its [`Event::StageStarted`];
    /// never for `split`.
    StageModel {
        /// The stage.
        stage: Stage,
        /// The model of the stage's role.
        model: String,
    },
    /// A stage finished.
    StageFinished {
        /// The stage.
        stage: Stage,
        /// Final counters.
        stats: StageStats,
    },
    /// The flow follows the job of run `run_id`: the [`Event::Metric`]s that come
    /// next on this bus are its own.
    RunWatched {
        /// ID of the run.
        run_id: String,
    },
    /// A training or evaluation log of the running job.
    Metric(TrainMetric),
    /// The training job is in a new state.
    JobStatus(JobStatus),
    /// The Runpod pod of a run changed.
    PodStatus(PodStatus),
}

/// How an item ended, as its event says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemResult {
    /// Produced: an [`Event::ItemDone`] with its usage.
    Done,
    /// Already present: an [`Event::ItemDone`] without usage.
    Skipped,
    /// Failed for good: an [`Event::ItemFailed`] not retried.
    Failed,
}

impl ItemResult {
    /// Lowercase name, as the history counts it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Done => "done",
            Self::Skipped => "skipped",
            Self::Failed => "failed",
        }
    }
}

impl Event {
    /// The stage and result of the item this event ends; `None` for any other
    /// event, a failed attempt that is retried included.
    #[must_use]
    pub fn item_result(&self) -> Option<(Stage, ItemResult)> {
        match self {
            Self::ItemDone {
                stage,
                usage: Some(_),
                ..
            } => Some((*stage, ItemResult::Done)),
            Self::ItemDone {
                stage, usage: None, ..
            } => Some((*stage, ItemResult::Skipped)),
            Self::ItemFailed {
                stage,
                retryable: false,
                ..
            } => Some((*stage, ItemResult::Failed)),
            _ => None,
        }
    }
}

/// Sees every event of the buses built with it, as they are published: nothing is
/// ever skipped, however many come at once.
pub trait Observer: Send + Sync {
    /// `event` was published on bus `bus`, an ID unique among the open buses.
    fn event(&self, bus: usize, event: &Event);
    /// Every handle of bus `bus` was dropped: nothing more comes from it, and a
    /// later bus may get its ID.
    fn closed(&self, bus: usize);
}

/// Broadcast channel of [`Event`]s. Publishing never blocks and never fails: events
/// without subscribers are dropped, and a slow subscriber skips old events. An
/// [`Observer`] given at creation sees each event before the subscribers.
#[derive(Debug, Clone)]
pub struct EventBus {
    sender: broadcast::Sender<Event>,
    /// Shared by the clones: the last one dropped tells the observer.
    id: Arc<BusId>,
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
        Self::observed(capacity, None)
    }

    /// [`EventBus::with_capacity`], whose events `observer` also sees.
    #[must_use]
    pub fn observed(capacity: usize, observer: Option<Arc<dyn Observer>>) -> Self {
        let (sender, _) = broadcast::channel(capacity.max(1));
        Self {
            sender,
            id: Arc::new(BusId { observer }),
        }
    }

    /// Sends `event` to the observer, then to every current subscriber.
    pub fn publish(&self, event: Event) {
        if let Some(observer) = &self.id.observer {
            observer.event(self.id.key(), &event);
        }
        self.sender.send(event).ok();
    }

    /// Receives every event published from now on.
    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.sender.subscribe()
    }

    /// How many subscribers are listening.
    #[cfg(test)]
    pub(crate) fn receiver_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

/// The identity of a bus and its clones: its address, while it lives.
struct BusId {
    observer: Option<Arc<dyn Observer>>,
}

impl BusId {
    fn key(&self) -> usize {
        std::ptr::from_ref(self).addr()
    }
}

impl fmt::Debug for BusId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BusId")
            .field("observed", &self.observer.is_some())
            .finish()
    }
}

impl Drop for BusId {
    fn drop(&mut self) {
        if let Some(observer) = &self.observer {
            observer.closed(self.key());
        }
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

    /// What an observer saw: `Ok` events and `Err` ends, by bus.
    #[derive(Default)]
    struct Seen(std::sync::Mutex<Vec<Result<(usize, Event), usize>>>);

    impl Observer for Seen {
        fn event(&self, bus: usize, event: &Event) {
            if let Ok(mut seen) = self.0.lock() {
                seen.push(Ok((bus, event.clone())));
            }
        }

        fn closed(&self, bus: usize) {
            if let Ok(mut seen) = self.0.lock() {
                seen.push(Err(bus));
            }
        }
    }

    #[test]
    fn an_observer_sees_every_event_of_its_buses_and_their_end() -> Result<(), String> {
        let seen = Arc::new(Seen::default());
        let observer: Arc<dyn Observer> = seen.clone();
        let first = EventBus::observed(1, Some(Arc::clone(&observer)));
        let second = EventBus::observed(1, Some(observer));
        let clone = first.clone();
        let (a, b) = (first.id.key(), second.id.key());
        assert_ne!(a, b);
        let started = |total| Event::StageStarted {
            stage: Stage::Answers,
            total,
        };
        // Far more than the channel keeps, with no subscriber at all.
        for total in 0..3 {
            clone.publish(started(total));
        }
        second.publish(started(9));
        EventBus::new().publish(started(5));
        drop(first);
        drop(clone);
        let seen = seen.0.lock().map_err(|e| e.to_string())?;
        assert_eq!(
            *seen,
            [
                Ok((a, started(0))),
                Ok((a, started(1))),
                Ok((a, started(2))),
                Ok((b, started(9))),
                Err(a),
            ]
        );
        Ok(())
    }

    #[test]
    fn item_events_say_how_their_item_ended() {
        let done = |usage| Event::ItemDone {
            stage: Stage::Answers,
            id: "a".into(),
            usage,
            cost: None,
        };
        let failed = |retryable| Event::ItemFailed {
            stage: Stage::Questions,
            id: "q".into(),
            error: "no".into(),
            retryable,
        };
        assert_eq!(
            done(Some(Usage::default())).item_result(),
            Some((Stage::Answers, ItemResult::Done))
        );
        assert_eq!(
            done(None).item_result(),
            Some((Stage::Answers, ItemResult::Skipped))
        );
        assert_eq!(
            failed(false).item_result(),
            Some((Stage::Questions, ItemResult::Failed))
        );
        assert_eq!(
            failed(true).item_result(),
            None,
            "a retried attempt ends nothing"
        );
        assert_eq!(
            Event::StageStarted {
                stage: Stage::Split,
                total: 1
            }
            .item_result(),
            None
        );
        assert_eq!(
            [ItemResult::Done, ItemResult::Skipped, ItemResult::Failed].map(ItemResult::name),
            ["done", "skipped", "failed"]
        );
    }

    #[test]
    fn stage_serializes_as_its_name() -> Result<(), serde_json::Error> {
        for stage in [
            Stage::Subtopics,
            Stage::Questions,
            Stage::Answers,
            Stage::Split,
        ] {
            assert_eq!(
                serde_json::to_string(&stage)?,
                format!("\"{}\"", stage.name())
            );
            assert_eq!(
                serde_json::from_str::<Stage>(&format!("\"{}\"", stage.name()))?,
                stage
            );
        }
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

    #[test]
    fn merge_adds_counts_tokens_and_cost() {
        let mut total = StageStats {
            done: 1,
            cost: None,
            ..StageStats::default()
        };
        let other = StageStats {
            done: 2,
            skipped: 1,
            failed: 1,
            excluded: 1,
            usage: Usage {
                input_tokens: 3,
                output_tokens: 4,
            },
            cost: Some(0.5),
        };
        total.merge(&other);
        total.merge(&other);
        assert_eq!(
            (total.done, total.skipped, total.failed, total.excluded),
            (5, 2, 2, 2)
        );
        assert_eq!(
            total.usage,
            Usage {
                input_tokens: 6,
                output_tokens: 8
            }
        );
        assert_eq!(total.cost, Some(1.0));
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
