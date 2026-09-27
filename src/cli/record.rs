//! The stage history of the pipeline commands: one line per stage a command started,
//! written when the stage ends, fails or is interrupted.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::SystemTime;

use tokio::sync::broadcast::Receiver;
use tokio::sync::broadcast::error::{RecvError, TryRecvError};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::config::RoleModel;
use crate::events::{Event, EventBus, Stage, StageStats};
use crate::history::{self, Entry, Span, SplitCounts, Status};
use crate::pipeline::SplitReport;
use crate::runs::rfc3339;

/// Writes the history lines of one command's stages.
pub(super) struct Recorder {
    project_dir: PathBuf,
    tally: Tally,
    begun: Option<Begun>,
}

/// The stage that began and has not ended yet.
struct Begun {
    stage: Stage,
    started_at: String,
    role: Option<RoleModel>,
}

impl Recorder {
    /// A recorder for `project_dir`, counting the items published on `bus` from now on.
    pub(super) fn new(project_dir: &Path, bus: &EventBus) -> Self {
        Self {
            project_dir: project_dir.to_path_buf(),
            tally: Tally::start(bus.subscribe()),
            begun: None,
        }
    }

    /// Notes that `stage` begins now, using `role` (none for `split`).
    pub(super) fn begin(&mut self, stage: Stage, role: Option<&RoleModel>) {
        self.begun = Some(Begun {
            stage,
            started_at: rfc3339(SystemTime::now()),
            role: role.cloned(),
        });
    }

    /// Records the stage that began as ended with `outcome` and `stats`.
    pub(super) fn end(&mut self, outcome: Status, stats: &StageStats) {
        self.write(outcome, stats, None);
    }

    /// Records the stage that began as failed, with what the events published
    /// before the failure counted. Counting stops: the command ends with the stage.
    pub(super) async fn fail(&mut self) {
        let Some(stage) = self.begun.as_ref().map(|begun| begun.stage) else {
            return;
        };
        self.tally.settle().await;
        let stats = self.tally.of(stage);
        self.write(Status::Failed, &stats, None);
    }

    /// Records `split` as ended with `report`.
    pub(super) fn split(&mut self, report: &SplitReport) {
        let stats = StageStats {
            excluded: report.excluded.values().sum(),
            ..StageStats::default()
        };
        let counts = SplitCounts {
            train: report.train,
            eval: report.eval,
            orphaned: report.orphaned,
        };
        self.write(Status::Ok, &stats, Some(counts));
    }

    /// Records the stage that began, if any, as interrupted, with what the events
    /// published before the interruption counted.
    pub(super) async fn interrupted(&mut self) {
        let Some(stage) = self.begun.as_ref().map(|begun| begun.stage) else {
            return;
        };
        self.tally.settle().await;
        let stats = self.tally.of(stage);
        self.write(Status::Interrupted, &stats, None);
    }

    /// Appends the line of the stage that began. A write error is only logged: the
    /// history never fails a command.
    fn write(&mut self, outcome: Status, stats: &StageStats, split: Option<SplitCounts>) {
        let Some(begun) = self.begun.take() else {
            return;
        };
        let mut entry = Entry::from_stats(
            begun.stage,
            Span {
                started_at: begun.started_at,
                ended_at: rfc3339(SystemTime::now()),
            },
            outcome,
            begun.role.as_ref(),
            stats,
        );
        entry.split = split;
        if let Err(error) = history::append(&self.project_dir, &entry) {
            tracing::warn!(
                "cannot record the {} stage in {}: {error}",
                begun.stage,
                history::path(&self.project_dir).display()
            );
        }
    }
}

type Counts = Arc<Mutex<BTreeMap<Stage, StageStats>>>;

/// Counts per stage the items done, skipped and failed that a bus publishes, for a
/// stage that stops before it returns its own counters.
struct Tally {
    counts: Counts,
    stop: CancellationToken,
    task: JoinHandle<()>,
}

impl Tally {
    fn start(mut events: Receiver<Event>) -> Self {
        let counts = Counts::default();
        let stop = CancellationToken::new();
        let task = tokio::spawn({
            let counts = Arc::clone(&counts);
            let stop = stop.clone();
            async move {
                loop {
                    tokio::select! {
                        () = stop.cancelled() => break,
                        event = events.recv() => match event {
                            Ok(event) => count(&counts, &event),
                            Err(RecvError::Lagged(skipped)) => lagged(skipped),
                            Err(RecvError::Closed) => return,
                        },
                    }
                }
                // Stopped: count what was published before.
                loop {
                    match events.try_recv() {
                        Ok(event) => count(&counts, &event),
                        Err(TryRecvError::Lagged(skipped)) => lagged(skipped),
                        Err(TryRecvError::Empty | TryRecvError::Closed) => break,
                    }
                }
            }
        });
        Self { counts, stop, task }
    }

    /// What was counted so far for `stage`.
    fn of(&self, stage: Stage) -> StageStats {
        self.counts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&stage)
            .cloned()
            .unwrap_or_default()
    }

    /// Stops counting once every event published so far is counted.
    async fn settle(&mut self) {
        self.stop.cancel();
        if let Err(error) = (&mut self.task).await {
            tracing::debug!("the stage tally ended early: {error}");
        }
    }
}

impl Drop for Tally {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

fn count(counts: &Counts, event: &Event) {
    let mut counts = counts.lock().unwrap_or_else(PoisonError::into_inner);
    match event {
        Event::ItemDone {
            stage, usage, cost, ..
        } => {
            let stats = counts.entry(*stage).or_default();
            match usage {
                Some(usage) => {
                    stats.done += 1;
                    stats.usage += *usage;
                },
                None => stats.skipped += 1,
            }
            if let Some(cost) = cost {
                *stats.cost.get_or_insert(0.0) += cost;
            }
        },
        Event::ItemFailed {
            stage,
            retryable: false,
            ..
        } => counts.entry(*stage).or_default().failed += 1,
        _ => {},
    }
}

fn lagged(skipped: u64) {
    tracing::debug!("the stage tally missed {skipped} event(s); its counts are low");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Usage;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn done(stage: Stage, usage: Option<Usage>, cost: Option<f64>) -> Event {
        Event::ItemDone {
            stage,
            id: "x".into(),
            usage,
            cost,
        }
    }

    fn failed(stage: Stage, retryable: bool) -> Event {
        Event::ItemFailed {
            stage,
            id: "x".into(),
            error: "no".into(),
            retryable,
        }
    }

    #[tokio::test]
    async fn an_interrupted_stage_is_recorded_with_its_tally() -> TestResult {
        let dir = tempfile::tempdir()?;
        let bus = EventBus::new();
        let mut recorder = Recorder::new(dir.path(), &bus);
        let role = RoleModel {
            provider: "mock".into(),
            model: "parent".into(),
            reasoning: false,
            max_tokens: 100,
            temperature: None,
            reasoning_effort: None,
            thinking_budget: None,
        };
        recorder.begin(Stage::Answers, Some(&role));
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 5,
        };
        bus.publish(done(Stage::Answers, Some(usage), Some(0.5)));
        bus.publish(done(Stage::Answers, Some(usage), Some(0.25)));
        bus.publish(done(Stage::Answers, None, None));
        bus.publish(failed(Stage::Answers, true));
        bus.publish(failed(Stage::Answers, false));
        bus.publish(done(Stage::Questions, Some(usage), None));
        recorder.interrupted().await;
        let entries = history::read(dir.path())?;
        let [entry] = entries.as_slice() else {
            return Err(format!("expected one entry, got {entries:?}").into());
        };
        assert_eq!(entry.stage, Stage::Answers);
        assert_eq!(entry.status, Status::Interrupted);
        assert_eq!(entry.model.as_deref(), Some("parent"));
        assert_eq!(
            (entry.done, entry.skipped, entry.failed),
            (2, 1, 1),
            "a retryable failure is not counted"
        );
        assert_eq!((entry.input_tokens, entry.output_tokens), (20, 10));
        assert_eq!(entry.cost, Some(0.75));
        Ok(())
    }

    #[tokio::test]
    async fn a_failed_stage_is_recorded_with_its_tally() -> TestResult {
        let dir = tempfile::tempdir()?;
        let bus = EventBus::new();
        let mut recorder = Recorder::new(dir.path(), &bus);
        recorder.begin(Stage::Questions, None);
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 5,
        };
        bus.publish(done(Stage::Questions, Some(usage), None));
        bus.publish(done(Stage::Questions, None, None));
        bus.publish(failed(Stage::Questions, false));
        recorder.fail().await;
        let entries = history::read(dir.path())?;
        let [entry] = entries.as_slice() else {
            return Err(format!("expected one entry, got {entries:?}").into());
        };
        assert_eq!(entry.stage, Stage::Questions);
        assert_eq!(entry.status, Status::Failed);
        assert_eq!((entry.done, entry.skipped, entry.failed), (1, 1, 1));
        assert_eq!((entry.input_tokens, entry.output_tokens), (10, 5));
        Ok(())
    }

    #[tokio::test]
    async fn nothing_is_recorded_without_a_begun_stage() -> TestResult {
        let dir = tempfile::tempdir()?;
        let bus = EventBus::new();
        let mut recorder = Recorder::new(dir.path(), &bus);
        recorder.interrupted().await;
        recorder.fail().await;
        recorder.begin(Stage::Split, None);
        recorder.split(&SplitReport {
            train: 3,
            eval: 1,
            excluded: BTreeMap::from([(crate::dataset::Exclusion::Truncated, 2)]),
            orphaned: 0,
        });
        // The stage ended: a later interruption records nothing more.
        recorder.interrupted().await;
        let entries = history::read(dir.path())?;
        let [entry] = entries.as_slice() else {
            return Err(format!("expected one entry, got {entries:?}").into());
        };
        assert_eq!(entry.status, Status::Ok);
        assert_eq!(entry.excluded, 2);
        assert_eq!(entry.provider, None);
        assert_eq!(
            entry.split,
            Some(SplitCounts {
                train: 3,
                eval: 1,
                orphaned: 0
            })
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_history_that_cannot_be_written_never_fails() -> TestResult {
        let dir = tempfile::tempdir()?;
        // The state directory is a file: the history cannot be created.
        std::fs::write(dir.path().join(crate::project_lock::STATE_DIR), "")?;
        let bus = EventBus::new();
        let mut recorder = Recorder::new(dir.path(), &bus);
        recorder.begin(Stage::Subtopics, None);
        recorder.end(Status::Ok, &StageStats::default());
        assert!(!history::path(dir.path()).exists());
        assert!(history::read(dir.path()).is_err());
        Ok(())
    }
}
