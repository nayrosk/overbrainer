//! The TUI's background work: each task runs on its own, owns its inputs and
//! returns what the app needs when it ends. The app only ever sees results.

use std::collections::{BTreeMap, HashMap};
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use tokio::sync::broadcast::Receiver;
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::{AbortHandle, JoinHandle, JoinSet};
use tokio_util::sync::CancellationToken;

use super::catalog::{Listed, Query, fetch};
use super::cost::history_cost;
use super::editor::Edited;
use super::project::ProjectConfig;
use super::project_edit::{SaveRefusal, save_config};
use super::start::{AutoPlan, Catalog, StartPlan, look_up, prepare, prepare_auto};
use super::training::{Listing, list_runs, read_series};
use crate::cli::data::{Command, Load};
use crate::cli::front::{Frontend, Report};
use crate::cli::{StageArgs, TrainArgs, TrainCommand};
use crate::config::{DotenvKeys, EnvSource, ReloadError, Source, Stamp, reload, stamp};
use crate::dataset::{Counts, DataFiles, Dataset, Deletion};
use crate::events::{Event, EventBus, Observer, Stage};
use crate::history::{self, Cost, Entry, Total};
use crate::pipeline::{Ctx, SplitReport};
use crate::prompts::Prompts;
use crate::train::TrainMetric;

/// Events kept for a TUI task's forwarder when it falls behind. `watch` publishes
/// every line of one tail read at once, at most 1 MiB, and a metrics line is at
/// least about 40 bytes: about 26 000 events, then an await lets the forwarder
/// drain.
pub(super) const BUS_CAPACITY: usize = 32_768;

/// How long a pipeline or training task waits for its forwarder once its flow is dropped:
/// only a sender kept by a task the flow spawned and has not yet dropped can
/// hold it that long.
const FORWARD_GRACE: Duration = Duration::from_secs(5);

/// Identifies a task for the app.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct TaskId(pub(super) u64);

/// Work the app asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Task {
    /// Reads the dataset files.
    Load,
    /// Changes the dataset files, then runs `split`.
    Edit(Edit),
    /// A pipeline command, on every topic, without `--force`.
    Pipeline(Command),
    /// A training flow of the command line.
    Train(TrainJob),
    /// Reads `runs/`, with the pod records.
    Runs,
    /// Reads the local metrics file of a run.
    Series(String),
    /// What a training run started now would use.
    Prepare,
    /// What auto mode run now would do after split.
    PrepareAuto,
    /// The Runpod GPU catalog for this many GPUs per pod, for the start
    /// dialog (list prices, VRAM and stock), and the VRAM the run needs.
    StartCatalog(u32),
    /// What a picker lists, or the GPU types a field hint needs.
    Catalog(Query),
    /// Reads `overbrainer.toml` and `.env` again when their stamp is no
    /// longer `seen`, with the process environment but the keys `.env` set at
    /// start, `dotenv`.
    CheckConfig {
        /// The stamp of the files the configuration shown was read from.
        seen: Stamp,
        /// The keys `.env` set at start.
        dotenv: DotenvKeys,
    },
    /// Validates `text` with `env` and writes it to `overbrainer.toml`,
    /// unless the file no longer holds `base`.
    SaveConfig {
        /// The new text.
        text: String,
        /// The text the file held when it was read.
        base: String,
        /// The environment it is validated with.
        env: EnvSource,
    },
}

/// A training flow, run exactly as `overbrainer train` runs it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum TrainJob {
    /// `train`, on `training.target`, without `--keep-pod`.
    Start,
    /// `train attach <run-id>`.
    Attach(String),
    /// `train cancel <run-id>`.
    Cancel(String),
}

impl TrainJob {
    /// The arguments `overbrainer train` would get.
    fn args(&self) -> TrainArgs {
        let command = match self {
            Self::Start => return TrainArgs::default(),
            Self::Attach(run_id) => TrainCommand::Attach {
                run_id: run_id.clone(),
            },
            Self::Cancel(run_id) => TrainCommand::Cancel {
                run_id: run_id.clone(),
            },
        };
        TrainArgs {
            command: Some(command),
            ..TrainArgs::default()
        }
    }
}

/// A change to the dataset files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Edit {
    /// An edit made in the editor.
    Change(Edited),
    /// A confirmed deletion, which must still remove `counts`.
    Delete {
        /// What is deleted.
        deletion: Deletion,
        /// What the confirmation said it removes.
        counts: Counts,
    },
}

/// What the history records, as the TUI shows it.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct History {
    /// The cost of every stage, `None` while none spent anything.
    pub(super) cost: Option<Cost>,
    /// Totals per stage.
    pub(super) stages: BTreeMap<Stage, Total>,
    /// The total over every stage.
    pub(super) all: Total,
    /// Totals per model.
    pub(super) models: BTreeMap<String, Total>,
}

impl History {
    /// The history of `entries`.
    pub(super) fn of(entries: &[Entry]) -> Self {
        let (stages, all) = history::totals(entries);
        Self {
            cost: history_cost(&all),
            stages,
            all,
            models: history::per_model(entries),
        }
    }

    /// A history whose stages cost `cost`, and nothing else.
    #[cfg(test)]
    pub(super) fn costing(cost: Cost) -> Self {
        Self {
            cost: Some(cost),
            ..Self::default()
        }
    }
}

/// What a task gives back.
#[derive(Debug)]
pub(super) enum Done {
    /// The dataset files, or why they cannot be read; then what the history
    /// records, or why it cannot be read.
    Loaded(Result<Dataset, String>, Result<History, String>),
    /// What an edit did, or why nothing changed.
    Saved(Result<Saved, String>),
    /// How a pipeline command ended.
    Pipeline(Result<(), String>),
    /// How a training flow ended.
    Trained(Result<(), String>),
    /// What `runs/` holds.
    Runs(Listing),
    /// The local metrics of run `run`, `None` when its file cannot be read.
    Series {
        /// The run.
        run: String,
        /// Its metrics.
        series: Option<Vec<TrainMetric>>,
    },
    /// What a training run started now would use, or why none can start.
    Prepared(Result<StartPlan, String>),
    /// What auto mode would do after split, or why it cannot run.
    PreparedAuto(Result<AutoPlan, String>),
    /// The GPU catalog of the start dialog and the VRAM the run needs, or
    /// why they cannot be read.
    StartCatalog(Catalog),
    /// A picker's entries, or why they cannot be read.
    Catalog(Result<Listed, String>),
    /// The configuration written to `overbrainer.toml`, or why nothing was.
    ConfigSaved(Result<Box<ProjectConfig>, SaveRefusal>),
    /// What a look at the configuration files found.
    ConfigChecked(Box<Checked>),
}

/// What a look at `overbrainer.toml` and `.env` found.
#[derive(Debug)]
pub(super) struct Checked {
    /// The stamp of the files, taken before they were read.
    pub(super) stamp: Stamp,
    /// When the stamp changed: the configuration read again, or why it
    /// cannot be used.
    pub(super) read: Option<Result<Reread, ReloadError>>,
}

/// The configuration read again, and the environment it was read with.
#[derive(Debug)]
pub(super) struct Reread {
    /// The configuration.
    pub(super) config: ProjectConfig,
    /// Its environment, which the next tasks read the settings with.
    pub(super) env: EnvSource,
}

/// Reads the configuration of the project in `dir` again unless the stamp of
/// its files is still `seen`; `dotenv` are the keys `.env` set at start.
pub(super) fn check_config(dir: &Path, seen: Stamp, dotenv: &DotenvKeys) -> Checked {
    let now = stamp(dir);
    if now == seen {
        return Checked {
            stamp: now,
            read: None,
        };
    }
    let read = reload(dir, dotenv).and_then(|reloaded| {
        let mut config = ProjectConfig::new(&reloaded.text, &reloaded.env)?;
        config.stamp = Some(now);
        Ok(Reread {
            config,
            env: reloaded.env,
        })
    });
    Checked {
        stamp: now,
        read: Some(read),
    }
}

/// A saved edit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Saved {
    /// What was changed, for the status line.
    pub(super) message: String,
    /// The `split` run after it, or why it failed (the edit stays saved).
    pub(super) split: Result<SplitReport, String>,
}

/// What arrives from tasks while they run.
#[derive(Debug)]
pub(super) enum Msg {
    /// The editor ended, or could not be started.
    EditorExited(io::Result<ExitStatus>),
    /// No browser could be started on this URL.
    BrowserFailed(String),
    /// The Logs export ended: the file name and how many lines were written, or
    /// why it could not be written.
    LogsExported(Result<(String, usize), String>),
    /// crates.io has this release, newer than the one running.
    NewerRelease(String),
    /// An event of a task's bus.
    Event(TaskId, Event),
    /// A task's forwarder fell behind and skipped this many events.
    Lagged(TaskId, u64),
    /// A line a task's flow reported.
    Report(TaskId, Report),
}

/// Forwards the events of task `id` from `events` as messages, until every
/// sender of its bus is dropped (the task ended). Lag is reported, never fatal.
/// The handle ends once the last event is sent.
pub(super) fn forward(
    id: TaskId,
    mut events: Receiver<Event>,
    messages: UnboundedSender<Msg>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let message = match events.recv().await {
                Ok(event) => Msg::Event(id, event),
                Err(RecvError::Lagged(skipped)) => Msg::Lagged(id, skipped),
                Err(RecvError::Closed) => break,
            };
            if messages.send(message).is_err() {
                break;
            }
        }
    })
}

/// Drops `front`, which closes its bus, then waits for `forwarder` to send what
/// is left and end, so every message of the task is in the app's inbox before
/// its end is. `what` names the task in the warning of a forwarder that does not
/// end within [`FORWARD_GRACE`].
async fn forwarded(front: Frontend, forwarder: JoinHandle<()>, what: &str) {
    drop(front);
    if tokio::time::timeout(FORWARD_GRACE, forwarder)
        .await
        .is_err()
    {
        tracing::warn!("the events of the {what} were not all forwarded");
    }
}

/// A TUI front end for task `id`: its own bus, seen by `observer` if any and
/// forwarded to `messages` by the task returned with it, its detach token and
/// abandon flag, and its reports sent as messages.
fn front_end(
    id: TaskId,
    messages: &UnboundedSender<Msg>,
    detach: &CancellationToken,
    abandon: &Arc<AtomicBool>,
    observer: Option<Arc<dyn Observer>>,
) -> (Frontend, JoinHandle<()>) {
    let bus = EventBus::observed(BUS_CAPACITY, observer);
    let forwarder = forward(id, bus.subscribe(), messages.clone());
    let sink = messages.clone();
    let front = Frontend::Tui {
        bus,
        detach: detach.clone(),
        abandon: Arc::clone(abandon),
        report: Arc::new(move |report| {
            sink.send(Msg::Report(id, report)).ok();
        }),
    };
    (front, forwarder)
}

/// Applies `edit` to the files of the project in `project_dir`, freshly read,
/// then rewrites train and eval with `split`, with the settings of `source` (a
/// configuration error refuses the edit before anything is written).
///
/// # Errors
///
/// Returns why nothing changed: the settings cannot be loaded, the files cannot
/// be read or written, or the edit is refused.
pub(super) fn save(project_dir: &Path, edit: &Edit, source: &Source) -> Result<Saved, String> {
    let error = |error: anyhow::Error| format!("{error:#}");
    let settings = source.load(project_dir).map_err(|e| error(e.into()))?;
    let files = DataFiles::new(project_dir);
    let mut data = Dataset::read(&files).map_err(|e| error(e.into()))?;
    let change = match edit {
        Edit::Change(Edited::Question { id, before, text }) => data.edit_question(id, before, text),
        Edit::Change(Edited::Answer { id, before, after }) => {
            data.edit_answer(id, before, after.clone())
        },
        Edit::Change(Edited::Subtopic { id, before, name }) => {
            data.rename_subtopic(id, before, name)
        },
        Edit::Delete { deletion, counts } => data.delete(deletion, *counts),
    }
    .map_err(|e| error(e.into()))?;
    data.save(&files, &change).map_err(|e| error(e.into()))?;
    let ctx = Ctx {
        settings: &settings,
        files: &files,
        prompts: &Prompts::empty(),
        bus: &EventBus::new(),
        topic: None,
        force: false,
    };
    let split = crate::pipeline::split(&ctx).map_err(|e| error(e.into()));
    Ok(Saved {
        message: change.message,
        split,
    })
}

/// The running tasks.
pub(super) struct Tasks {
    project_dir: PathBuf,
    messages: UnboundedSender<Msg>,
    set: JoinSet<Done>,
    ids: HashMap<tokio::task::Id, TaskId>,
    tokens: HashMap<TaskId, CancellationToken>,
    abandons: HashMap<TaskId, Arc<AtomicBool>>,
    /// The catalog lookups running: only reads, aborted when
    /// the TUI ends.
    lookups: HashMap<TaskId, AbortHandle>,
    /// Where the tasks started from now read the settings: the configuration
    /// the app keeps, never a file it has not validated.
    source: Source,
    /// Sees the events of every pipeline and training task.
    observer: Option<Arc<dyn Observer>>,
}

impl Tasks {
    /// No task yet, for the project in `project_dir`; running tasks send their
    /// messages to `messages`.
    pub(super) fn new(project_dir: &Path, messages: UnboundedSender<Msg>) -> Self {
        Self {
            project_dir: project_dir.to_path_buf(),
            messages,
            set: JoinSet::new(),
            ids: HashMap::new(),
            tokens: HashMap::new(),
            abandons: HashMap::new(),
            lookups: HashMap::new(),
            source: Source::from(EnvSource::Process),
            observer: None,
        }
    }

    /// Has the tasks started from now read the settings from `source`; those
    /// running keep the settings they read.
    pub(super) fn use_config(&mut self, source: Source) {
        self.source = source;
    }

    /// Has `observer` see the events of the tasks started from now on.
    pub(super) fn observe(&mut self, observer: Option<Arc<dyn Observer>>) {
        self.observer = observer;
    }

    /// The project directory tasks are spawned for.
    pub(super) fn project_dir(&self) -> &Path {
        &self.project_dir
    }

    /// Aborts the catalog lookups: they only read, so the TUI
    /// never waits for them to end.
    pub(super) fn abort_lookups(&mut self) {
        for (_, lookup) in self.lookups.drain() {
            lookup.abort();
        }
    }

    /// Sets the abandon flag of task `id` and cancels its token: a Runpod run
    /// still provisioning deletes its pod and fails, as Ctrl-C does on the
    /// command line; any other flow detaches.
    pub(super) fn abandon(&self, id: TaskId) {
        if let Some(abandon) = self.abandons.get(&id) {
            abandon.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        self.cancel(id);
    }

    /// Cancels the token of task `id`: a pipeline stops, a training detaches.
    pub(super) fn cancel(&self, id: TaskId) {
        if let Some(token) = self.tokens.get(&id) {
            token.cancel();
        }
    }

    /// Whether no task runs.
    pub(super) fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    /// Prepares a training start, or auto mode when `auto`, off the UI
    /// thread, on the configuration kept.
    fn spawn_prepare(&mut self, auto: bool) -> AbortHandle {
        let dir = self.project_dir.clone();
        let source = self.source.clone();
        self.set.spawn(async move {
            if auto {
                let plan = tokio::task::spawn_blocking(move || prepare_auto(&dir, &source)).await;
                return Done::PreparedAuto(match plan {
                    Ok(plan) => plan,
                    Err(error) => Err(format!("cannot prepare auto mode: {error}")),
                });
            }
            let plan = tokio::task::spawn_blocking(move || prepare(&dir, &source)).await;
            Done::Prepared(match plan {
                Ok(plan) => plan,
                Err(error) => Err(format!("cannot prepare the run: {error}")),
            })
        })
    }

    /// Starts `task` as `id`.
    pub(super) fn spawn(&mut self, id: TaskId, task: Task) {
        let files = DataFiles::new(&self.project_dir);
        let handle = match task {
            Task::Load => {
                let dir = self.project_dir.clone();
                self.set.spawn(async move {
                    let read = tokio::task::spawn_blocking(move || {
                        let history = history::read(&dir)
                            .map(|entries| History::of(&entries))
                            .map_err(|error| {
                                format!("cannot read {}: {error}", history::path(&dir).display())
                            });
                        (Dataset::read(&files), history)
                    })
                    .await;
                    match read {
                        Ok((Ok(data), history)) => Done::Loaded(Ok(data), history),
                        Ok((Err(error), history)) => {
                            Done::Loaded(Err(format!("{:#}", anyhow::Error::from(error))), history)
                        },
                        Err(error) => Done::Loaded(
                            Err(format!("cannot read the data files: {error}")),
                            Err(format!("cannot read the history: {error}")),
                        ),
                    }
                })
            },
            Task::Edit(edit) => {
                let dir = self.project_dir.clone();
                let source = self.source.clone();
                self.set.spawn(async move {
                    let saved =
                        tokio::task::spawn_blocking(move || save(&dir, &edit, &source)).await;
                    Done::Saved(match saved {
                        Ok(saved) => saved,
                        Err(error) => Err(format!("the edit failed: {error}")),
                    })
                })
            },
            Task::CheckConfig { seen, dotenv } => self.spawn_check(seen, dotenv),
            Task::SaveConfig { text, base, env } => self.spawn_save(text, base, env),
            Task::Pipeline(command) => self.spawn_pipeline(id, command),
            Task::Train(job) => self.spawn_train(id, job),
            Task::Runs => {
                let dir = self.project_dir.clone();
                self.set.spawn(async move {
                    let listing = tokio::task::spawn_blocking(move || list_runs(&dir)).await;
                    Done::Runs(listing.unwrap_or_else(|error| Listing {
                        runs: Err(format!("cannot list the runs: {error}")),
                        skipped: Vec::new(),
                    }))
                })
            },
            Task::Series(run) => {
                let dir = self.project_dir.clone();
                self.set.spawn(async move {
                    let id = run.clone();
                    let series = tokio::task::spawn_blocking(move || read_series(&dir, &id)).await;
                    Done::Series {
                        run,
                        series: series.ok().flatten(),
                    }
                })
            },
            Task::Prepare => self.spawn_prepare(false),
            Task::PrepareAuto => self.spawn_prepare(true),
            Task::StartCatalog(gpu_count) => {
                let dir = self.project_dir.clone();
                let source = self.source.clone();
                self.spawn_lookup(id, async move {
                    Done::StartCatalog(look_up(&dir, source, gpu_count).await)
                })
            },
            Task::Catalog(query) => {
                let dir = self.project_dir.clone();
                let source = self.source.clone();
                self.spawn_lookup(id, async move {
                    Done::Catalog(fetch(&dir, source, query).await)
                })
            },
        };
        self.ids.insert(handle.id(), id);
    }

    /// Starts writing `text` to `overbrainer.toml`, validated with `env`,
    /// unless the file no longer holds `base`.
    fn spawn_save(&mut self, text: String, base: String, env: EnvSource) -> AbortHandle {
        let dir = self.project_dir.clone();
        self.set.spawn(async move {
            let saved = tokio::task::spawn_blocking(move || {
                save_config(&dir, &text, &base, &env).map(Box::new)
            })
            .await;
            Done::ConfigSaved(saved.unwrap_or_else(|error| {
                Err(SaveRefusal::Failed(format!("the save failed: {error}")))
            }))
        })
    }

    /// Starts a look at the configuration files, whose stamp was `seen`; a
    /// look that fails finds nothing new, and the next one looks again.
    fn spawn_check(&mut self, seen: Stamp, dotenv: DotenvKeys) -> AbortHandle {
        let dir = self.project_dir.clone();
        self.set.spawn(async move {
            let checked =
                tokio::task::spawn_blocking(move || check_config(&dir, seen, &dotenv)).await;
            Done::ConfigChecked(Box::new(checked.unwrap_or_else(|error| {
                tracing::warn!("cannot check the configuration files: {error}");
                Checked {
                    stamp: seen,
                    read: None,
                }
            })))
        })
    }

    /// Starts the lookup `read` as `id`: it only reads, so it is aborted when
    /// the TUI ends.
    fn spawn_lookup(
        &mut self,
        id: TaskId,
        read: impl Future<Output = Done> + Send + 'static,
    ) -> AbortHandle {
        let handle = self.set.spawn(read);
        self.lookups.insert(id, handle.clone());
        handle
    }

    /// Starts the pipeline `command` as `id`: its token interrupts the stage.
    fn spawn_pipeline(&mut self, id: TaskId, command: Command) -> AbortHandle {
        let dir = self.project_dir.clone();
        let source = self.source.clone();
        let token = CancellationToken::new();
        let abandon = Arc::new(AtomicBool::new(false));
        let (front, forwarder) =
            front_end(id, &self.messages, &token, &abandon, self.observer.clone());
        self.tokens.insert(id, token.clone());
        self.set.spawn(async move {
            let args = StageArgs::default();
            // The flow ends itself when the token is cancelled, recording the stage.
            let outcome =
                crate::cli::data::run(&dir, command, &args, &front, Load::Source(&source))
                    .await
                    .map_err(|error| format!("{error:#}"));
            forwarded(front, forwarder, "stage").await;
            Done::Pipeline(outcome)
        })
    }

    /// Starts the training flow `job` as `id`: its token detaches the flow, and
    /// its abandon flag makes a Runpod provisioning delete its pod.
    fn spawn_train(&mut self, id: TaskId, job: TrainJob) -> AbortHandle {
        let dir = self.project_dir.clone();
        let source = self.source.clone();
        let token = CancellationToken::new();
        let abandon = Arc::new(AtomicBool::new(false));
        let (front, forwarder) =
            front_end(id, &self.messages, &token, &abandon, self.observer.clone());
        self.tokens.insert(id, token);
        self.abandons.insert(id, abandon);
        self.set.spawn(async move {
            // Never aborted: the flow shields its starts and cancels, and only
            // its token detaches it.
            let result = crate::cli::train::run(&dir, &job.args(), &front, &source).await;
            forwarded(front, forwarder, "training").await;
            Done::Trained(result.map_err(|error| format!("{error:#}")))
        })
    }

    /// The next task to end, with what it gave back or how it failed (a panic).
    /// `None` when no task runs.
    pub(super) async fn next(&mut self) -> Option<(TaskId, Result<Done, String>)> {
        loop {
            let joined = self.set.join_next_with_id().await?;
            let (task, result) = match joined {
                Ok((task, done)) => (task, Ok(done)),
                Err(error) => (
                    error.id(),
                    Err(format!("a background task failed: {error}")),
                ),
            };
            if let Some(id) = self.ids.remove(&task) {
                self.tokens.remove(&id);
                self.abandons.remove(&id);
                self.lookups.remove(&id);
                return Some((id, result));
            }
        }
    }
}

#[cfg(test)]
impl Tasks {
    /// A task `id` standing for a training start: it holds a detach token and
    /// an abandon flag, as a start does, and ends once `release` or its token
    /// is cancelled. Returns its abandon flag.
    pub(super) fn park(&mut self, id: TaskId, release: CancellationToken) -> Arc<AtomicBool> {
        let token = CancellationToken::new();
        let abandon = Arc::new(AtomicBool::new(false));
        self.tokens.insert(id, token.clone());
        self.abandons.insert(id, Arc::clone(&abandon));
        let handle = self.set.spawn(async move {
            tokio::select! {
                () = release.cancelled() => {},
                () = token.cancelled() => {},
            }
            Done::Trained(Err("parked".to_string()))
        });
        self.ids.insert(handle.id(), id);
        abandon
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::dataset::{Example, Id, Question, rewrite};
    use crate::tui::snapshots::{MOVED, dataset, project};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// An upper bound only: a shared CI runner can be far slower than a laptop.
    const LIMIT: Duration = Duration::from_secs(30);

    #[tokio::test]
    async fn a_save_task_writes_the_configuration_and_ends_with_it() -> TestResult {
        use crate::tui::snapshots::{PROJECT_CONFIG, project_env};
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join("overbrainer.toml"), PROJECT_CONFIG)?;
        let text = PROJECT_CONFIG.replace("rust_expert", "rust_pro");
        let mut tasks = Tasks::new(dir.path(), tokio::sync::mpsc::unbounded_channel().0);
        tasks.spawn(
            TaskId(5),
            Task::SaveConfig {
                text: text.clone(),
                base: PROJECT_CONFIG.to_string(),
                env: project_env(),
            },
        );
        let next = tokio::time::timeout(LIMIT, tasks.next()).await?;
        let Some((TaskId(5), Ok(Done::ConfigSaved(Ok(config))))) = next else {
            return Err(format!("unexpected end: {next:?}").into());
        };
        assert_eq!(config.settings.project.name, "rust_pro");
        assert_eq!(config.text, text);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("overbrainer.toml"))?,
            text
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_task_started_after_a_reload_reads_the_settings_with_its_env() -> TestResult {
        let dir = crate::tui::snapshots::project()?;
        let config = format!(
            "{}\n[training]\ntarget = \"homelab\"\nbase_model = \"Qwen/Qwen3-4B\"\n\
             adapter = \"qlora\"\nhub_model_id = \"me/model\"\n\n[targets.homelab]\n\
             kind = \"ssh\"\nruntime = \"docker\"\n",
            crate::tui::snapshots::CONFIG
        );
        std::fs::write(dir.path().join("overbrainer.toml"), config)?;
        let mut tasks = Tasks::new(dir.path(), tokio::sync::mpsc::unbounded_channel().0);
        let mut warned = Vec::new();
        for (id, env) in [
            (TaskId(1), Vec::new()),
            (
                TaskId(2),
                vec![("OVERBRAINER_HF_TOKEN".to_string(), "hf_x".to_string())],
            ),
        ] {
            tasks.use_config(Source::from(EnvSource::Vars(env)));
            tasks.spawn(id, Task::Prepare);
            let next = tokio::time::timeout(LIMIT, tasks.next()).await?;
            let Some((ended, Ok(Done::Prepared(Ok(plan))))) = next else {
                return Err(format!("unexpected end: {next:?}").into());
            };
            assert_eq!(ended, id);
            warned.push(
                plan.warnings
                    .iter()
                    .any(|w| w.contains("OVERBRAINER_HF_TOKEN")),
            );
        }
        assert_eq!(warned, [true, false], "the token set by the new env");
        Ok(())
    }

    #[test]
    fn a_look_reads_the_files_again_only_once_they_changed() -> TestResult {
        let dir = crate::tui::snapshots::project()?;
        let seen = stamp(dir.path());
        let none = check_config(dir.path(), seen, &DotenvKeys::default());
        assert!(none.read.is_none());
        assert_eq!(none.stamp, seen);
        std::fs::write(
            dir.path().join(crate::config::DOTENV_FILE),
            "OVERBRAINER_PROJECT__NAME=renamed\n",
        )?;
        let checked = check_config(dir.path(), seen, &DotenvKeys::default());
        assert_ne!(checked.stamp, seen);
        let Some(Ok(reread)) = checked.read else {
            return Err(format!("not read again: {:?}", checked.read).into());
        };
        assert_eq!(reread.config.settings.project.name, "renamed");
        assert_eq!(reread.config.stamp, Some(checked.stamp));
        assert!(reread.config.env.contains("project.name"));
        Ok(())
    }

    #[tokio::test]
    async fn a_load_reads_the_data_files_and_the_history_cost_and_ends_with_its_id() -> TestResult {
        let dir = tempfile::tempdir()?;
        let files = DataFiles::new(dir.path());
        rewrite(&files.subtopics, &dataset().subtopics)?;
        let stats = crate::events::StageStats {
            cost: Some(0.5),
            ..crate::events::StageStats::default()
        };
        let span = history::Span {
            started_at: "2026-09-27T10:00:00Z".into(),
            ended_at: "2026-09-27T10:01:00Z".into(),
        };
        let entry = history::Entry::from_stats(
            crate::events::Stage::Answers,
            span,
            history::Status::Ok,
            None,
            &stats,
        );
        history::append(dir.path(), &entry)?;
        let mut tasks = Tasks::new(dir.path(), tokio::sync::mpsc::unbounded_channel().0);
        assert!(tasks.is_empty());
        tasks.spawn(TaskId(7), Task::Load);
        assert!(!tasks.is_empty());
        let next = tokio::time::timeout(LIMIT, tasks.next()).await?;
        let Some((TaskId(7), Ok(Done::Loaded(Ok(data), Ok(history))))) = next else {
            return Err(format!("unexpected end: {next:?}").into());
        };
        let Some(Cost::Known(cost)) = history.cost else {
            return Err(format!("unexpected history: {history:?}").into());
        };
        assert_eq!(data.subtopics.len(), 3);
        let answers = history.stages.get(&crate::events::Stage::Answers);
        assert_eq!(answers.map(|total| total.cost), Some(Cost::Known(0.5)));
        assert!(history.models.is_empty(), "no model recorded");
        assert!((cost - 0.5).abs() < f64::EPSILON, "{cost}");
        assert!(tasks.is_empty());
        assert!(tasks.next().await.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn a_load_of_a_broken_file_ends_with_its_error() -> TestResult {
        let dir = tempfile::tempdir()?;
        let files = DataFiles::new(dir.path());
        if let Some(parent) = files.answers.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&files.answers, "not json\n")?;
        let mut tasks = Tasks::new(dir.path(), tokio::sync::mpsc::unbounded_channel().0);
        tasks.spawn(TaskId(1), Task::Load);
        let next = tokio::time::timeout(LIMIT, tasks.next()).await?;
        let Some((TaskId(1), Ok(Done::Loaded(Err(error), Ok(history))))) = next else {
            return Err(format!("unexpected end: {next:?}").into());
        };
        assert!(error.contains("answers.jsonl"), "{error}");
        assert_eq!(history, History::default());
        Ok(())
    }

    #[test]
    fn a_saved_edit_rewrites_the_files_then_train_and_eval() -> TestResult {
        let dir = project()?;
        let files = DataFiles::new(dir.path());
        let subtopic = Id::subtopic("ownership", "Borrowing");
        let edit = Edit::Change(Edited::Question {
            id: Id::question(&subtopic, MOVED),
            before: MOVED.into(),
            text: "What happens to a borrow after a move?".into(),
        });
        let saved = save(
            dir.path(),
            &edit,
            &Source::from(EnvSource::Vars(Vec::new())),
        )?;
        assert_eq!(
            saved.message,
            "question saved; its old answer was deleted: 1 question unanswered, run answers to answer it"
        );
        let report = saved.split?;
        assert_eq!((report.train, report.eval, report.orphaned), (1, 1, 1));
        let questions: Vec<Question> = crate::dataset::read(&files.questions)?;
        assert!(
            questions
                .iter()
                .any(|q| q.text == "What happens to a borrow after a move?")
        );
        let train: Vec<Example> = crate::dataset::read(&files.train)?;
        assert_eq!(train.len(), 1);
        Ok(())
    }

    #[test]
    fn a_broken_config_refuses_the_edit_before_writing() -> TestResult {
        let dir = project()?;
        let files = DataFiles::new(dir.path());
        let before = std::fs::read(&files.answers)?;
        std::fs::write(dir.path().join("overbrainer.toml"), "[project\n")?;
        let edit = Edit::Delete {
            deletion: Deletion::Answer(Id::question(
                &Id::subtopic("ownership", "Borrowing"),
                MOVED,
            )),
            counts: Counts {
                questions: 0,
                answers: 1,
            },
        };
        assert!(
            save(
                dir.path(),
                &edit,
                &Source::from(EnvSource::Vars(Vec::new()))
            )
            .is_err()
        );
        assert_eq!(std::fs::read(&files.answers)?, before);
        assert!(!files.train.exists());
        Ok(())
    }

    /// A real `train cancel` task on a run that has no job: the command line's
    /// flow refuses it, and its error comes back as the task's end.
    #[tokio::test]
    async fn a_cancel_task_runs_the_command_line_flow() -> TestResult {
        let dir = project()?;
        let run = "20260921-133200-a1b2";
        let mut record = crate::tui::snapshots::run(run, "homelab", crate::runs::RunState::Failed);
        record.job = None;
        crate::runs::Runs::new(dir.path()).save(&record)?;
        let mut tasks = Tasks::new(dir.path(), tokio::sync::mpsc::unbounded_channel().0);
        tasks.spawn(TaskId(3), Task::Train(TrainJob::Cancel(run.into())));
        let next = tokio::time::timeout(LIMIT, tasks.next()).await?;
        let Some((TaskId(3), Ok(Done::Trained(Err(error))))) = next else {
            return Err(format!("unexpected end: {next:?}").into());
        };
        assert_eq!(error, format!("run {run} has not started"));
        assert!(tasks.is_empty());
        Ok(())
    }

    /// The runs and a run's metrics are read in tasks, off the loop's thread.
    #[tokio::test]
    async fn runs_and_metrics_are_read_in_tasks() -> TestResult {
        let dir = tempfile::tempdir()?;
        let run = "20260921-133200-a1b2";
        let runs = crate::runs::Runs::new(dir.path());
        runs.save(&crate::tui::snapshots::run(
            run,
            "homelab",
            crate::runs::RunState::Running,
        ))?;
        std::fs::write(
            runs.run_dir(run)?.join(crate::train::METRICS_FILE),
            "{\"event\": \"log\", \"time\": 2, \"step\": 1, \"loss\": 1.5}\n",
        )?;
        let mut tasks = Tasks::new(dir.path(), tokio::sync::mpsc::unbounded_channel().0);
        tasks.spawn(TaskId(1), Task::Runs);
        let next = tokio::time::timeout(LIMIT, tasks.next()).await?;
        let Some((TaskId(1), Ok(Done::Runs(Listing { runs: Ok(rows), .. })))) = next else {
            return Err(format!("unexpected end: {next:?}").into());
        };
        assert_eq!(rows.len(), 1);
        tasks.spawn(TaskId(2), Task::Series(run.into()));
        let next = tokio::time::timeout(LIMIT, tasks.next()).await?;
        let Some((TaskId(2), Ok(Done::Series { series, .. }))) = next else {
            return Err(format!("unexpected end: {next:?}").into());
        };
        assert_eq!(series.map(|s| s.len()), Some(1));
        Ok(())
    }

    /// Abandoning a task sets its flag and cancels its token; a task already
    /// ended is ignored.
    #[test]
    fn abandon_sets_the_flag_then_cancels_the_token() {
        let mut tasks = Tasks::new(
            Path::new("/nonexistent"),
            tokio::sync::mpsc::unbounded_channel().0,
        );
        let token = CancellationToken::new();
        let flag = Arc::new(AtomicBool::new(false));
        tasks.tokens.insert(TaskId(1), token.clone());
        tasks.abandons.insert(TaskId(1), Arc::clone(&flag));
        tasks.abandon(TaskId(2));
        assert!(!flag.load(std::sync::atomic::Ordering::SeqCst));
        tasks.abandon(TaskId(1));
        assert!(flag.load(std::sync::atomic::Ordering::SeqCst));
        assert!(token.is_cancelled());
    }

    /// A copy of a message of the pipeline task, which [`Msg`] cannot clone.
    fn copy(message: &Msg) -> Option<Msg> {
        match message {
            Msg::Event(id, event) => Some(Msg::Event(*id, event.clone())),
            Msg::Lagged(id, skipped) => Some(Msg::Lagged(*id, *skipped)),
            Msg::Report(id, report) => Some(Msg::Report(*id, report.clone())),
            Msg::EditorExited(_)
            | Msg::BrowserFailed(_)
            | Msg::LogsExported(_)
            | Msg::NewerRelease(_) => None,
        }
    }

    /// Records the events it sees.
    #[derive(Default)]
    struct Seen(std::sync::Mutex<Vec<Event>>);

    impl Observer for Seen {
        fn event(&self, _bus: usize, event: &Event) {
            if let Ok(mut seen) = self.0.lock() {
                seen.push(event.clone());
            }
        }

        fn closed(&self, _bus: usize) {}
    }

    /// The metrics see the tasks started after the configuration was reloaded.
    #[tokio::test]
    async fn the_observer_sees_tasks_started_with_a_reloaded_configuration() -> TestResult {
        let dir = project()?;
        let seen = Arc::new(Seen::default());
        let mut tasks = Tasks::new(dir.path(), tokio::sync::mpsc::unbounded_channel().0);
        tasks.observe(Some(Arc::clone(&seen) as Arc<dyn Observer>));
        tasks.use_config(Source::from(EnvSource::Vars(Vec::new())));
        tasks.spawn(TaskId(1), Task::Pipeline(Command::Split));
        let next = tokio::time::timeout(LIMIT, tasks.next()).await?;
        let Some((TaskId(1), Ok(Done::Pipeline(Ok(()))))) = next else {
            return Err(format!("unexpected end: {next:?}").into());
        };
        let seen = seen.0.lock().map_err(|e| e.to_string())?;
        assert!(
            seen.iter().any(|event| matches!(
                event,
                Event::StageFinished {
                    stage: Stage::Split,
                    ..
                }
            )),
            "{seen:?}"
        );
        Ok(())
    }

    /// A real `split` task: every message is sent before it ends, and the app
    /// keeps them whether they are handled before or after its end.
    #[tokio::test]
    async fn a_split_task_sends_every_message_before_it_ends() -> TestResult {
        use crossterm::event::KeyCode;

        use crate::events::Stage;
        use crate::tui::app::Effect;
        use crate::tui::pipeline::StageState;
        use crate::tui::snapshots::{app, key};

        let dir = project()?;
        let (messages, mut inbox) = tokio::sync::mpsc::unbounded_channel();
        let mut tasks = Tasks::new(dir.path(), messages);
        tasks.spawn(TaskId(1), Task::Pipeline(Command::Split));
        let next = tokio::time::timeout(LIMIT, tasks.next()).await?;
        let Some((TaskId(1), Ok(Done::Pipeline(Ok(()))))) = next else {
            return Err(format!("unexpected end: {next:?}").into());
        };
        let mut received = Vec::new();
        while let Ok(message) = inbox.try_recv() {
            received.push(message);
        }
        let started = received.iter().any(|message| {
            matches!(
                message,
                Msg::Event(
                    TaskId(1),
                    Event::StageStarted {
                        stage: Stage::Split,
                        ..
                    }
                )
            )
        });
        let finished = received.iter().any(|message| {
            matches!(
                message,
                Msg::Event(
                    TaskId(1),
                    Event::StageFinished {
                        stage: Stage::Split,
                        ..
                    }
                )
            )
        });
        assert!(started && finished, "{received:?}");
        let line = received.iter().find_map(|message| match message {
            Msg::Report(TaskId(1), Report::Line(line)) => Some(line.clone()),
            _ => None,
        });
        let Some(line) = line else {
            return Err(format!("no split line: {received:?}").into());
        };
        assert!(line.starts_with("split: "), "{line}");
        for done_first in [false, true] {
            let mut app = app();
            let mut effects = Vec::new();
            for code in [
                KeyCode::Char('r'),
                KeyCode::Down,
                KeyCode::Down,
                KeyCode::Down,
                KeyCode::Down,
                KeyCode::Enter,
            ] {
                effects.extend(app.on_input(&key(code)));
            }
            assert_eq!(
                effects,
                [Effect::Spawn(TaskId(1), Task::Pipeline(Command::Split))]
            );
            if done_first {
                app.on_done(TaskId(1), Ok(Done::Pipeline(Ok(()))));
            }
            for message in received.iter().filter_map(copy) {
                app.on_message(message);
            }
            if !done_first {
                app.on_done(TaskId(1), Ok(Done::Pipeline(Ok(()))));
            }
            assert_eq!(
                app.pipeline.results,
                std::slice::from_ref(&line),
                "done first: {done_first}"
            );
            let row = app.pipeline.row(Stage::Split);
            assert_eq!(row.state, StageState::Done, "done first: {done_first}");
            assert_eq!(row.finished, row.total, "done first: {done_first}");
            assert!(row.total > 0);
        }
        Ok(())
    }

    /// A fake `axolotl` (from `tests/cli_train.rs`): `train` waits for a
    /// `release` file next to `bin/`, then writes three metrics lines and an
    /// adapter.
    const FAKE_AXOLOTL: &str = r#"#!/bin/sh
here="$(dirname "$0")/.."
[ "$1" = train ] || exit 0
i=0
until [ -f "$here/release" ]; do
  i=$((i + 1))
  [ "$i" -gt 600 ] && exit 3
  sleep 0.1
done
printf '{"event": "begin", "time": 1, "max_steps": 2}\n' >> "$OVERBRAINER_METRICS"
printf '{"event": "log", "time": 2, "step": 1, "epoch": 0.5, "max_steps": 2, "loss": 1.5}\n' >> "$OVERBRAINER_METRICS"
printf '{"event": "log", "time": 3, "step": 2, "epoch": 1.0, "max_steps": 2, "eval_loss": 1.25}\n' >> "$OVERBRAINER_METRICS"
mkdir -p output && echo adapter > output/adapter_model.safetensors
exit 0
"#;

    /// A project training on a local target with [`FAKE_AXOLOTL`].
    fn local_project() -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
        use std::os::unix::fs::PermissionsExt;

        let dir = project()?;
        let venv = dir.path().join("venv");
        std::fs::create_dir_all(venv.join("bin"))?;
        std::fs::write(venv.join("bin/axolotl"), FAKE_AXOLOTL)?;
        std::fs::set_permissions(
            venv.join("bin/axolotl"),
            std::fs::Permissions::from_mode(0o755),
        )?;
        let files = DataFiles::new(dir.path());
        std::fs::write(&files.train, "{\"id\": 1}\n")?;
        std::fs::write(&files.eval, "{\"id\": 2}\n")?;
        let config = format!(
            "{}\n[training]\ntarget = \"here\"\nbase_model = \"Qwen/Qwen3-4B\"\n\
             adapter = \"qlora\"\n\n[targets.here]\nkind = \"local\"\nruntime = \"native\"\n\
             venv = \"{}\"\n",
            crate::tui::snapshots::CONFIG,
            venv.display()
        );
        std::fs::write(dir.path().join("overbrainer.toml"), config)?;
        Ok(dir)
    }

    /// Waits for the next message, at most a minute.
    async fn next_message(
        inbox: &mut tokio::sync::mpsc::UnboundedReceiver<Msg>,
    ) -> Result<Option<Msg>, tokio::time::error::Elapsed> {
        tokio::time::timeout(Duration::from_secs(60), inbox.recv()).await
    }

    /// A real start on a local target: it reports its run, its token detaches
    /// it once its job started (the run keeps running), and an attach then
    /// forwards every metric and the command line's own outcome line.
    #[tokio::test]
    async fn a_training_task_reports_its_run_and_detaches_without_cancelling_it() -> TestResult {
        let dir = local_project()?;
        let (messages, mut inbox) = tokio::sync::mpsc::unbounded_channel();
        let mut tasks = Tasks::new(dir.path(), messages);
        tasks.spawn(TaskId(1), Task::Train(TrainJob::Start));
        let mut created = None;
        loop {
            match next_message(&mut inbox).await? {
                Some(Msg::Report(TaskId(1), Report::RunCreated(id))) => created = Some(id),
                Some(Msg::Event(TaskId(1), Event::JobStatus(_))) => break,
                Some(_) => {},
                None => return Err("the task ended before its job started".into()),
            }
        }
        let run = created.ok_or("no RunCreated report before the job started")?;
        tasks.cancel(TaskId(1));
        let ended = tokio::time::timeout(Duration::from_secs(60), tasks.next()).await?;
        let Some((TaskId(1), Ok(Done::Trained(Err(error))))) = ended else {
            return Err(format!("{ended:?}").into());
        };
        assert_eq!(
            error,
            format!(
                "interrupted: run {run} keeps running on target `here`; follow it again with \
                 `overbrainer train attach {run}`"
            )
        );
        let record = crate::runs::Runs::new(dir.path()).load(&run)?;
        assert_eq!(
            record.state,
            crate::runs::RunState::Running,
            "detached, not cancelled"
        );

        std::fs::write(dir.path().join("venv/release"), "")?;
        tasks.spawn(TaskId(2), Task::Train(TrainJob::Attach(run.clone())));
        let ended = tokio::time::timeout(Duration::from_secs(60), tasks.next()).await?;
        assert!(
            matches!(ended, Some((TaskId(2), Ok(Done::Trained(Ok(())))))),
            "{ended:?}"
        );
        drop(tasks);
        let (mut metrics, mut lines) = (0, Vec::new());
        while let Some(message) = next_message(&mut inbox).await? {
            match message {
                Msg::Event(TaskId(2), Event::Metric(_)) => metrics += 1,
                Msg::Report(TaskId(2), Report::Line(line)) => lines.push(line),
                _ => {},
            }
        }
        assert_eq!(metrics, 2, "every metric of the run, from its first");
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(
            lines[0].starts_with(&format!("train: run {run} succeeded; ")),
            "the command line's own outcome line: {lines:?}"
        );
        assert!(
            lines[1].ends_with(&format!(" in runs/{run}/output")),
            "then where the model is: {lines:?}"
        );
        Ok(())
    }

    /// A start catalog lookup is aborted when the TUI ends: it ends at once.
    #[tokio::test]
    async fn a_catalog_lookup_is_aborted_when_the_tui_ends() -> TestResult {
        let dir = tempfile::tempdir()?;
        let mut tasks = Tasks::new(dir.path(), tokio::sync::mpsc::unbounded_channel().0);
        let handle = tasks.set.spawn(std::future::pending::<Done>());
        tasks.ids.insert(handle.id(), TaskId(9));
        tasks.lookups.insert(TaskId(9), handle);
        tasks.abort_lookups();
        let next = tokio::time::timeout(LIMIT, tasks.next()).await?;
        assert!(matches!(next, Some((TaskId(9), Err(_)))), "{next:?}");
        Ok(())
    }
}
