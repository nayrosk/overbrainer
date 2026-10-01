//! What the app does with training runs: listing them, attaching, cancelling, and
//! what quitting and signals do to the training tasks.
//!
//! A training task's token is cancelled only once its job's first status arrived
//! (its watch began), so a start is never cut: asked to detach before, it is
//! marked and detached then. Only a process signal abandons a task at once. A
//! confirmed cancel always runs, quitting included; only a signal drops one not
//! started yet, and says how to run it.
//!
//! `s` stops a running run with a snapshot (`train stop`); `T` prepares a new
//! run resuming a stopped one (`train --resume-from`), as `t` prepares a start.
//!
//! A start (`t`) holds the data lock until its job started. Quitting while a
//! Runpod run still provisions offers to abandon it instead of waiting: its pod
//! is deleted and the run fails, as Ctrl-C does on the command line. `c` on such
//! a run offers the same for that run alone, and the TUI stays open.
//!
//! Files are read by tasks, never on the loop's thread nor while drawing.

use std::time::{Duration, SystemTime};

use crossterm::event::KeyCode;

use super::app::{Action, App, Confirm, Effect, Exit, NoteOf, Overlay, Severity, View};
use super::auto;
use super::start::{self, Catalog, StartPlan};
use super::tasks::{Msg, Task, TaskId, TrainJob};
use super::training::{Detach, Ended, Follow, Job, Listing, RunActivity};
use crate::cli::front::Report;
use crate::events::Event;
use crate::runs::RunState;
use crate::train::sizing::{Estimate, VramFloor};
use crate::train::{Phases, TrainMetric};

/// Time between two reads of `runs/` while the Training view is shown, and of
/// the data files while the Dataset view is shown during a stage.
pub(super) const REFRESH: Duration = Duration::from_secs(2);

/// What abandoning a Runpod start does, for the dialogs that offer it.
const ABANDONED: &str = "If its pod is still being prepared, it is deleted and the run fails, \
                         as Ctrl-C does on the command line; once its job is being sent, the \
                         run is detached instead.";

impl App {
    /// Reads `runs/` again, in a task; while one reads, once more after it.
    pub(super) fn refresh_runs(&mut self) -> Vec<Effect> {
        self.refreshed = self.now;
        if self.training.listing.is_some() {
            self.training.list_again = true;
            return Vec::new();
        }
        let id = self.task_id();
        self.training.listing = Some(id);
        vec![Effect::Spawn(id, Task::Runs)]
    }

    /// Reads `runs/` again when the Training view is shown and it is time (or
    /// the clock went back).
    pub(super) fn refresh_when_due(&mut self) -> Vec<Effect> {
        let recent = matches!(
            self.now.duration_since(self.refreshed),
            Ok(since) if since < REFRESH
        );
        if self.view == View::Training && !recent {
            return self.refresh_runs();
        }
        Vec::new()
    }

    /// Read `id` of `runs/` found `listing`: shown when it is the latest read,
    /// then the selected run's metrics are read, and a read asked for meanwhile
    /// starts. Each new warning is logged once.
    pub(super) fn listed(&mut self, id: TaskId, listing: Listing) -> Vec<Effect> {
        if self.training.listing != Some(id) {
            return Vec::new();
        }
        self.training.listing = None;
        self.dirty = true;
        for warning in self.training.listed(listing) {
            tracing::warn!("{warning}");
        }
        let mut effects = self.read_selected();
        if std::mem::take(&mut self.training.list_again) {
            effects.extend(self.refresh_runs());
        }
        effects
    }

    /// Reads the local metrics of the selected run, in a task, once, unless a
    /// task follows it.
    fn read_selected(&mut self) -> Vec<Effect> {
        match self.training.to_read() {
            Some(run) => self.read_series(&run),
            None => Vec::new(),
        }
    }

    /// Reads the local metrics of run `run` in a task; an earlier read of it no
    /// longer counts.
    fn read_series(&mut self, run: &str) -> Vec<Effect> {
        let id = self.task_id();
        self.training.reading.insert(run.to_string(), id);
        vec![Effect::Spawn(id, Task::Series(run.to_string()))]
    }

    /// Read `id` of run `run`'s local metrics found `series`: kept when it is
    /// that run's latest read. Read after a task of the run ended, they hold
    /// every metric, late or skipped.
    pub(super) fn series_read(
        &mut self,
        id: TaskId,
        run: String,
        series: Option<(Vec<TrainMetric>, Phases)>,
    ) -> Vec<Effect> {
        if self.training.reading.get(&run) != Some(&id) {
            return Vec::new();
        }
        self.training.reading.remove(&run);
        if let Some((series, phases)) = series {
            if let Some(ended) = self.training.ended.get_mut(&run) {
                ended.healed = true;
            }
            self.training.phases.insert(run.clone(), phases);
            self.training.series.insert(run, series);
            self.dirty = true;
        }
        Vec::new()
    }

    /// Read task `id` failed (a panic): it no longer counts.
    pub(super) fn read_failed(&mut self, id: TaskId, error: &str) -> bool {
        if self.training.listing == Some(id) {
            self.training.listing = None;
            self.training.error = Some(format!("cannot list the runs: {error}"));
            return true;
        }
        let before = self.training.reading.len();
        self.training.reading.retain(|_, read| *read != id);
        before != self.training.reading.len()
    }

    /// A key in the Training view.
    pub(super) fn on_training_key(&mut self, code: KeyCode) -> Vec<Effect> {
        let view = &mut self.training;
        match code {
            KeyCode::Up | KeyCode::Char('k') => view.selected = view.selected.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => {
                view.selected = (view.selected + 1).min(view.runs.len().saturating_sub(1));
            },
            KeyCode::Char('a') => return self.attach_selected(),
            KeyCode::Char('c') => {
                self.ask_cancel();
                return Vec::new();
            },
            KeyCode::Char('t') => return self.prepare_start(None),
            KeyCode::Char('T') => return self.prepare_resume(),
            KeyCode::Char('s') => {
                self.ask_stop();
                return Vec::new();
            },
            KeyCode::Char('x') => {
                self.ask_clear_failed();
                return Vec::new();
            },
            KeyCode::Char('p') => {
                self.toggle_pod();
                return Vec::new();
            },
            _ => return Vec::new(),
        }
        self.read_selected()
    }

    /// `x`: asks to leave the failed runs no task is on out of the list until
    /// the TUI restarts. Nothing is deleted.
    fn ask_clear_failed(&mut self) {
        let failed = self.training.failed_runs();
        let (runs, no) = match failed.len() {
            0 => {
                self.say(Severity::Info, "no failed run to clear");
                return;
            },
            1 => ("the failed run".to_string(), "keep it"),
            count => (format!("the {count} failed runs"), "keep them"),
        };
        self.overlay = Some(Overlay::Confirm(Confirm {
            title: " Clear failed runs? ".to_string(),
            text: vec![format!(
                "Hide {runs} from this list until the TUI restarts? Nothing is deleted: the \
                 files stay in runs/, and `overbrainer runs ls` still lists every run."
            )],
            yes: "clear",
            no,
            action: Action::ClearFailed(failed),
        }));
    }

    /// Leaves runs `runs`, failed, out of the list until the TUI restarts; the
    /// metrics of the run then selected are read.
    pub(super) fn clear_failed(&mut self, runs: &[String]) -> Vec<Effect> {
        self.training.hide(runs);
        let cleared = match runs.len() {
            1 => "1 failed run cleared".to_string(),
            count => format!("{count} failed runs cleared"),
        };
        self.say(
            Severity::Info,
            format!("{cleared} from the list until the TUI restarts"),
        );
        self.read_selected()
    }

    /// `p`: dismisses the selected run's pod, whose line and column then show
    /// nothing, or shows it again. Refused while a task follows the run: its
    /// pod is live.
    fn toggle_pod(&mut self) {
        let Some(row) = self.training.selected_run() else {
            return;
        };
        let id = row.record.id.clone();
        if row.pod.is_none() {
            self.say(Severity::Info, format!("run {id} has no pod"));
            return;
        }
        if self.training.dismissed.remove(&id) {
            self.say(Severity::Info, format!("the pod of run {id} shows again"));
            return;
        }
        if self.training.task_of(&id).is_some() {
            self.say(
                Severity::Info,
                format!("run {id} is followed: its pod stays in view"),
            );
            return;
        }
        self.training.dismissed.insert(id.clone());
        self.say(
            Severity::Info,
            format!("the pod of run {id} is dismissed; p shows it again"),
        );
    }

    /// `a`: follows the selected run again, from its first metric; refused
    /// while the TUI is quitting.
    fn attach_selected(&mut self) -> Vec<Effect> {
        let Some(id) = self
            .training
            .selected_run()
            .map(|row| row.record.id.clone())
        else {
            return Vec::new();
        };
        if self.leaving.is_some() {
            self.say(Severity::Warn, "refused: quitting; nothing new is followed");
            return Vec::new();
        }
        if self.training.cancelling(&id) {
            self.say(Severity::Info, format!("run {id} is being cancelled"));
            return Vec::new();
        }
        if self.training.task_of(&id).is_some() {
            self.say(Severity::Info, format!("run {id} is already followed"));
            return Vec::new();
        }
        self.training.series.remove(&id);
        self.training.phases.remove(&id);
        self.training.ended.remove(&id);
        self.train(Job::Attach, &id)
    }

    /// Refuses a new run while the data is locked (a stage, an edit or another
    /// start runs) or the TUI is quitting; says why.
    fn start_refused(&mut self) -> bool {
        self.refuse_new("one task at a time", "run")
    }

    /// `t`: prepares the confirmation of a new run, in a task; with
    /// `resume`, of a run resuming that stopped run.
    fn prepare_start(&mut self, resume: Option<String>) -> Vec<Effect> {
        if self.start_refused() {
            return Vec::new();
        }
        if self.prepare.is_some() {
            self.say(Severity::Info, "already preparing a run");
            return Vec::new();
        }
        let id = self.task_id();
        self.prepare = Some(id);
        let task = resume.map_or(Task::Prepare, Task::PrepareResume);
        vec![Effect::Spawn(id, task)]
    }

    /// `T`: prepares the confirmation of a new run resuming the selected run,
    /// which must be stopped with a snapshot.
    fn prepare_resume(&mut self) -> Vec<Effect> {
        let Some(row) = self.training.selected_run() else {
            return Vec::new();
        };
        let id = row.record.id.clone();
        if row.record.state != RunState::Stopped || row.record.snapshot.is_none() {
            self.say(
                Severity::Info,
                format!(
                    "run {id} is {}: T resumes a run stopped with a snapshot (s)",
                    row.record.state.name()
                ),
            );
            return Vec::new();
        }
        self.prepare_start(Some(id))
    }

    /// `s`: asks to stop the selected run's job with a snapshot; only a
    /// running job that can save one can be.
    fn ask_stop(&mut self) {
        let Some(row) = self.training.selected_run() else {
            return;
        };
        let (id, target) = (row.record.id.clone(), row.record.target.clone());
        let running = row.record.state == RunState::Running && row.record.job.is_some();
        let snapshots = row.record.snapshots;
        if self.training.stopping(&id) {
            self.say(Severity::Info, format!("run {id} is being stopped"));
            return;
        }
        if self.training.cancelling(&id) {
            self.say(Severity::Info, format!("run {id} is being cancelled"));
            return;
        }
        if !running || self.training.activity(&id).abandons() {
            self.say(
                Severity::Info,
                format!("run {id} has no running job to stop with a snapshot"),
            );
            return;
        }
        if !snapshots {
            let refused = crate::runs::RunError::NoSnapshots(id);
            self.say(Severity::Warn, format!("refused: {refused}"));
            return;
        }
        if self.leaving.is_some() {
            self.say(Severity::Warn, "refused: quitting; nothing new is followed");
            return;
        }
        self.overlay = Some(Overlay::Confirm(Confirm {
            title: " Stop a run with a snapshot? ".to_string(),
            text: vec![format!(
                "Stop run {id} on {target} with a snapshot? Its job saves a checkpoint at the end \
                 of its current step and stops; the checkpoint is retrieved with its results, and \
                 for Runpod the pod is then ended. T then starts a new run from it. A job that \
                 gives no snapshot within 30 minutes is cancelled."
            )],
            yes: "stop the run",
            no: "keep it",
            action: Action::Stop(id),
        }));
    }

    /// Stops run `run_id` with a snapshot: a task following it is detached
    /// first, and the stop starts once it ended, so two flows never follow
    /// the same run at once.
    pub(super) fn stop_run(&mut self, run_id: &str) -> Vec<Effect> {
        if self.training.stopping(run_id) || self.training.cancelling(run_id) {
            return Vec::new();
        }
        let Some((task, _)) = self.training.task_of(run_id) else {
            return self.train(Job::Stop, run_id);
        };
        let Some(follow) = self.training.tasks.get_mut(&task) else {
            return Vec::new();
        };
        follow.stop_after = true;
        self.say(
            Severity::Info,
            format!("detaching run {run_id}, then stopping it with a snapshot"),
        );
        self.detach(task)
    }

    /// The plan of a new run is ready: asks to start it, and reads the GPU
    /// catalog of a Runpod run meanwhile (an earlier lookup no longer counts).
    /// Dropped once the TUI is quitting, and while a dialog, the help, the menu
    /// or the filter is open (it never replaces what the user is answering):
    /// the status line then says to press `t` again.
    pub(super) fn prepared(&mut self, plan: Result<StartPlan, String>) -> Vec<Effect> {
        self.prepare = None;
        if self.leaving.is_some() {
            return Vec::new();
        }
        let plan = match plan {
            Ok(plan) => plan,
            Err(error) => {
                self.say(Severity::Warn, format!("refused: {error}"));
                return Vec::new();
            },
        };
        if self.overlay.is_some() || self.dataset.input.is_some() {
            self.say(
                Severity::Info,
                "a run is prepared: press t again to start it",
            );
            return Vec::new();
        }
        let gpu_count = plan.runpod.as_ref().map(|runpod| runpod.spec.gpu_count);
        self.overlay = Some(Overlay::Confirm(start_dialog(Box::new(plan), None)));
        self.start_catalog = None;
        self.start_gpus = None;
        let Some(gpu_count) = gpu_count else {
            return Vec::new();
        };
        let id = self.task_id();
        self.start_catalog = Some(id);
        vec![Effect::Spawn(id, Task::StartCatalog(gpu_count))]
    }

    /// The GPU catalog of the start dialog arrived: kept while the dialog or
    /// its picker is open, and shown in the dialog.
    pub(super) fn start_catalog_read(&mut self, gpus: Catalog) {
        self.start_catalog = None;
        // The plan keeps the estimate: the run and the dialog's picker use it.
        let held = self.start_held.as_deref_mut();
        let shown = match &mut self.overlay {
            Some(Overlay::Confirm(Confirm {
                action: Action::Start(plan),
                ..
            })) => Some(&mut **plan),
            _ => None,
        };
        for plan in [held, shown].into_iter().flatten() {
            if let Some(runpod) = plan.runpod.as_deref_mut() {
                runpod.need = Some(gpus.need.clone());
            }
        }
        self.start_gpus = Some(gpus);
        if let Some(Overlay::Confirm(confirm)) = &mut self.overlay {
            match &confirm.action {
                Action::Start(plan) => confirm.text = start::text(plan, self.start_gpus.as_ref()),
                Action::Auto(plan) => confirm.text = auto::text(plan, self.start_gpus.as_ref()),
                _ => {},
            }
        }
    }

    /// Starts the run of `plan`, on `training.target` and without keeping its
    /// pod, unless something started meanwhile. The VRAM floor its
    /// confirmation showed, once estimated, is the one the run uses.
    pub(super) fn start_run(&mut self, plan: &StartPlan) -> Vec<Effect> {
        if self.start_refused() {
            return Vec::new();
        }
        let runpod = plan.runpod.is_some();
        let vram_floor = plan
            .runpod
            .as_ref()
            .and_then(|runpod| runpod.need.as_ref())
            .map_or(VramFloor::ToEstimate, |need| {
                VramFloor::Known(need.as_ref().ok().map(Estimate::floor_gb))
            });
        let task = plan
            .resume
            .as_ref()
            .map_or(TrainJob::Start { vram_floor }, |resume| {
                TrainJob::Resume(resume.run_id.clone())
            });
        let effects = self.spawn_train(Job::Start { runpod }, "", task);
        let resuming = plan.resume.as_ref().map_or_else(String::new, |resume| {
            format!(", resuming run {}", resume.run_id)
        });
        self.say(
            Severity::Info,
            format!("starting a run on {}{resuming}", plan.target),
        );
        effects
    }

    /// Start task `id` was never spawned (the loop ended first): it is
    /// forgotten, and the exit notes say so.
    pub(super) fn start_dropped(&mut self, id: TaskId) {
        self.training.tasks.remove(&id);
        self.exit_notes.push(NOT_STARTED.to_string());
        self.leave_when_idle();
    }

    /// Starts a training task doing `job` on run `run_id`. The late messages of
    /// the run's earlier tasks, and a read of its metrics, no longer count.
    fn train(&mut self, job: Job, run_id: &str) -> Vec<Effect> {
        let task = match job {
            Job::Start { .. } => TrainJob::Start {
                vram_floor: VramFloor::ToEstimate,
            },
            Job::Attach => TrainJob::Attach(run_id.to_string()),
            Job::Cancel => TrainJob::Cancel(run_id.to_string()),
            Job::Stop => TrainJob::Stop(run_id.to_string()),
        };
        self.spawn_train(job, run_id, task)
    }

    /// [`Self::train`] with its task given.
    fn spawn_train(&mut self, job: Job, run_id: &str, task: TrainJob) -> Vec<Effect> {
        let id = self.task_id();
        if !run_id.is_empty() {
            self.forget_notes(&NoteOf::Run(run_id.to_string()));
        }
        self.training.last.retain(|_, run| run != run_id);
        self.training.reading.remove(run_id);
        // A followed run's pod is live: it shows again.
        self.training.dismissed.remove(run_id);
        self.training.tasks.insert(id, Follow::new(job, run_id));
        vec![Effect::Spawn(id, Task::Train(task))]
    }

    /// `c`: asks to cancel the selected run's job, or to abandon it while it is
    /// a Runpod start still starting; refused after a signal.
    fn ask_cancel(&mut self) {
        let Some(row) = self.training.selected_run() else {
            return;
        };
        let (id, target) = (row.record.id.clone(), row.record.target.clone());
        if self.training.cancelling(&id) {
            self.say(Severity::Info, format!("run {id} is being cancelled"));
            return;
        }
        if self.leaving == Some(Exit::Signal) {
            self.say(Severity::Warn, "refused: interrupted, exiting");
            return;
        }
        let activity = self.training.activity(&id);
        let starting = self
            .training
            .task_of(&id)
            .filter(|_| activity.abandons())
            .map(|(task, _)| (task, matches!(activity, RunActivity::Abandoning { .. })));
        if let Some((task, abandoned)) = starting {
            // A Runpod start has no job to cancel yet: it is abandoned instead.
            if abandoned {
                self.say(Severity::Info, format!("run {id} is being abandoned"));
                return;
            }
            let text = format!(
                "Run {id} is still starting, so it has no job to cancel yet. Abandon it \
                 instead? {ABANDONED}"
            );
            self.overlay = Some(Overlay::Confirm(Confirm {
                title: " Abandon a starting run? ".to_string(),
                text: vec![text],
                yes: "abandon",
                no: "keep it",
                action: Action::AbandonStart(task),
            }));
            return;
        }
        self.overlay = Some(Overlay::Confirm(Confirm {
            title: " Cancel a run? ".to_string(),
            text: vec![format!(
                "Cancel run {id} on {target}? Its job is stopped (its container first), what it \
                 produced is retrieved best effort, and for Runpod the pod is then ended. This \
                 cannot be undone."
            )],
            yes: "cancel the run",
            no: "keep it",
            action: Action::Cancel(id),
        }));
    }

    /// Cancels run `run_id`: a task following it is detached first, and the
    /// cancel starts once it ended, so two flows never end the same pod at once.
    pub(super) fn cancel_run(&mut self, run_id: &str) -> Vec<Effect> {
        if self.training.cancelling(run_id) {
            self.say(Severity::Info, format!("run {run_id} is being cancelled"));
            return Vec::new();
        }
        let Some((task, _)) = self.training.task_of(run_id) else {
            return self.train(Job::Cancel, run_id);
        };
        let Some(follow) = self.training.tasks.get_mut(&task) else {
            return Vec::new();
        };
        follow.cancel_after = true;
        let said = if follow.starting() {
            // Interrupting a start would abandon it: its job is waited for.
            format!("run {run_id} is starting: it is cancelled once its job started")
        } else {
            format!("detaching run {run_id}, then cancelling it")
        };
        self.say(Severity::Info, said);
        self.detach(task)
    }

    /// Detaches task `id`: at once when its watch began, else once its job's
    /// first status arrives.
    fn detach(&mut self, id: TaskId) -> Vec<Effect> {
        let Some(follow) = self.training.tasks.get_mut(&id) else {
            return Vec::new();
        };
        match (follow.detach, follow.watching) {
            (Detach::Done, _) => Vec::new(),
            (_, true) => {
                follow.detach = Detach::Done;
                vec![Effect::Cancel(id)]
            },
            (_, false) => {
                follow.detach = Detach::OnStart;
                Vec::new()
            },
        }
    }

    /// A message of training task `id`, running or ended.
    pub(super) fn on_training_message(&mut self, id: TaskId, message: Msg) -> Vec<Effect> {
        if !self.training.tasks.contains_key(&id) {
            self.late_training_message(id, message);
            return Vec::new();
        }
        match message {
            Msg::Event(_, event) => {
                let started = self.training.event(id, event);
                let waiting = self
                    .training
                    .tasks
                    .get(&id)
                    .is_some_and(|follow| follow.detach == Detach::OnStart);
                if started && waiting {
                    return self.detach(id);
                }
            },
            Msg::Lagged(_, skipped) => {
                if let Some(follow) = self.training.tasks.get_mut(&id) {
                    follow.skipped += skipped;
                }
            },
            Msg::Report(_, Report::Line(line)) => {
                if let Some(follow) = self.training.tasks.get_mut(&id) {
                    follow.lines.push(line);
                }
            },
            Msg::Report(_, Report::RunStopped(step)) => {
                if let Some(follow) = self.training.tasks.get_mut(&id) {
                    follow.stopped_at = Some(step);
                }
            },
            Msg::Report(_, Report::RunCreated(run_id)) => {
                if let Some(follow) = self.training.tasks.get_mut(&id) {
                    follow.run_id.clone_from(&run_id);
                }
                self.training.wanted = Some(run_id);
                return self.refresh_runs();
            },
            Msg::EditorExited(_)
            | Msg::BrowserFailed(_)
            | Msg::LogsExported(_)
            | Msg::NewerRelease(_) => {},
        }
        Vec::new()
    }

    /// A message of training task `id` handled after its end: a line joins how
    /// its run ended, and a metric its series, unless the series was read again
    /// from the local file (which holds it).
    fn late_training_message(&mut self, id: TaskId, message: Msg) {
        let Some(run) = self.training.last.get(&id).cloned() else {
            return;
        };
        let healed = self.training.ended.get(&run).is_some_and(|e| e.healed);
        match message {
            Msg::Event(_, Event::Metric(metric)) if !healed => {
                self.training
                    .phases
                    .entry(run.clone())
                    .or_default()
                    .metric(&metric);
                self.training.series.entry(run).or_default().push(metric);
            },
            Msg::Event(_, Event::Mark(mark)) if !healed => {
                self.training.phases.entry(run).or_default().mark(mark);
            },
            Msg::Lagged(_, skipped) => {
                if let Some(ended) = self.training.ended.get_mut(&run) {
                    ended.skipped += skipped;
                }
            },
            Msg::Report(_, Report::Line(line)) => {
                if self.leaving.is_some() {
                    // A flow's line (a pod warning) always stays.
                    self.exit_notes.push(line.clone());
                }
                if let Some(ended) = self.training.ended.get_mut(&run) {
                    ended.lines.push(line);
                }
            },
            _ => {},
        }
    }

    /// Training task `id` ended with `result` (see [`App::run_ended`]); auto
    /// mode ends when it was its run.
    pub(super) fn trained(&mut self, id: TaskId, result: Result<(), String>) -> Vec<Effect> {
        let ended = self
            .training
            .tasks
            .get(&id)
            .map(|follow| (follow.run_id.clone(), follow.stopped_at));
        let error = result.as_ref().err().cloned();
        let effects = self.run_ended(id, result);
        if let Some((run, stopped_at)) = ended {
            self.auto_trained(id, &run, error.as_deref(), stopped_at);
        }
        effects
    }

    /// Training task `id` ended with `result`. A cancel waiting for it starts,
    /// unless a signal came (an exit note then says how to cancel). While the
    /// TUI is leaving, what it reported and its error (a detached run's "keeps
    /// running ... attach") are kept for the exit.
    fn run_ended(&mut self, id: TaskId, result: Result<(), String>) -> Vec<Effect> {
        let Some(follow) = self.training.tasks.remove(&id) else {
            return Vec::new();
        };
        let run = follow.run_id.clone();
        let name = follow.run();
        let error = result.err();
        let cancel = follow.cancel_after && self.leaving != Some(Exit::Signal);
        let stop = follow.stop_after && self.leaving.is_none();
        if self.leaving.is_some() {
            // The flow's lines (pod warnings) always stay.
            self.exit_notes.extend(follow.lines.iter().cloned());
            if !cancel && let Some(error) = &error {
                self.note_leaving(error.clone(), NoteOf::Run(run.clone()));
            }
            if follow.stop_after {
                self.note_leaving(
                    format!(
                        "run {run} was not stopped: the TUI ended before its stop started; stop \
                         it with `overbrainer train stop {run}`"
                    ),
                    NoteOf::Run(run.clone()),
                );
            }
            if follow.cancel_after && !cancel {
                // A start abandoned may have failed with no job, or detached
                // once its job started (a local or SSH start, a Runpod job
                // already being sent), whose pod then bills until `max_hours`.
                let note = if follow.starting() {
                    format!(
                        "run {run} was not cancelled (a signal came during its start); if it \
                         is running, cancel it with `overbrainer train cancel {run}`"
                    )
                } else {
                    format!(
                        "run {run} was not cancelled: interrupted before its cancel started; \
                         cancel it with `overbrainer train cancel {run}`"
                    )
                };
                self.note_leaving(note, NoteOf::Run(run.clone()));
            }
        }
        match &error {
            Some(error) if !follow.cancel_after && !follow.stop_after => {
                self.say(Severity::Warn, format!("{name}: {error}"));
            },
            Some(_) => {},
            None => self.say(Severity::Info, format!("{name}: done")),
        }
        if run.is_empty() {
            // A start that failed before its run was created: no run to show.
            self.leave_when_idle();
            return Vec::new();
        }
        self.training.ended.insert(
            run.clone(),
            Ended {
                lines: follow.lines,
                error,
                skipped: follow.skipped,
                healed: false,
            },
        );
        self.training.last.insert(id, run.clone());
        let mut effects = Vec::new();
        if self.leaving.is_none() {
            // The local metrics, retrieved with the results, heal any missing
            // point.
            effects.extend(self.read_series(&run));
            effects.extend(self.refresh_runs());
        }
        if cancel {
            effects.extend(self.train(Job::Cancel, &run));
        } else if stop {
            effects.extend(self.train(Job::Stop, &run));
        }
        self.leave_when_idle();
        effects
    }

    /// What quitting does to each training task, for the quit dialog.
    pub(super) fn training_quit_text(&self) -> Vec<String> {
        self.training
            .tasks
            .values()
            .map(|follow| {
                let run = &follow.run_id;
                let pod = match follow.job {
                    Job::Start { runpod: true } => " its pod",
                    _ => "",
                };
                match follow.job {
                    Job::Cancel => format!("Run {run}: cancel in progress, quitting waits for it."),
                    Job::Stop => format!(
                        "Run {run}: its snapshot was asked for and it keeps running until it saves \
                         it; attach again here or with `overbrainer train attach {run}`."
                    ),
                    _ if follow.cancel_after => format!(
                        "Run {run}: cancel pending, it starts once the run is detached; \
                         quitting waits for it."
                    ),
                    Job::Start { .. } if follow.starting() => format!(
                        "{} is starting{pod}: quitting waits until its job has started, then \
                         leaves it running.",
                        capitalized(&follow.run())
                    ),
                    Job::Start { .. } | Job::Attach => format!(
                        "Run {run} keeps running{}; attach again from here or with \
                         `overbrainer train attach {run}`.",
                        self.where_it_runs(run)
                    ),
                }
            })
            .collect()
    }

    /// Where run `run` keeps running: its target, or its pod and rate for a Runpod
    /// run with a pod.
    fn where_it_runs(&self, run: &str) -> String {
        let Some(row) = self.training.runs.iter().find(|row| row.record.id == run) else {
            return String::new();
        };
        match row
            .pod
            .as_ref()
            .and_then(|pod| pod.pod_id.as_ref().map(|id| (id, pod)))
        {
            Some((pod_id, pod)) => match pod.cost_per_hour {
                Some(rate) => format!(" on pod {pod_id} (${rate:.2}/h)"),
                None => format!(" on pod {pod_id}"),
            },
            None => format!(" on target `{}`", row.record.target),
        }
    }

    /// Detaches every task that follows a run, each once its watch began;
    /// cancels are waited for.
    pub(super) fn detach_all(&mut self) -> Vec<Effect> {
        let following: Vec<TaskId> = self
            .training
            .tasks
            .iter()
            .filter(|(_, follow)| follow.job != Job::Cancel)
            .map(|(id, _)| *id)
            .collect();
        following
            .into_iter()
            .flat_map(|id| self.detach(id))
            .collect()
    }

    /// Quitting while Runpod runs still provision (and no signal abandoned them
    /// yet): offers to abandon them rather than wait for their jobs.
    pub(super) fn offer_abandon(&mut self) {
        if self.leaving != Some(Exit::Quit) {
            return;
        }
        let starting: Vec<(TaskId, String)> = self
            .training
            .tasks
            .iter()
            .filter(|(_, follow)| {
                follow.job == Job::Start { runpod: true }
                    && follow.starting()
                    && follow.detach != Detach::Done
            })
            .map(|(id, follow)| (*id, follow.run()))
            .collect();
        if starting.is_empty() {
            return;
        }
        let runs: Vec<&str> = starting.iter().map(|(_, run)| run.as_str()).collect();
        let text = format!(
            "Quitting waits until the job of {} has started, which can take minutes. Abandon \
             it instead? {ABANDONED}",
            runs.join(", ")
        );
        self.overlay = Some(Overlay::Confirm(Confirm {
            title: " Abandon a starting run? ".to_string(),
            text: vec![text],
            yes: "abandon",
            no: "wait",
            action: Action::Abandon(starting.into_iter().map(|(id, _)| id).collect()),
        }));
    }

    /// Abandons the start tasks `ids` that still provision, once confirmed: as
    /// a signal would, their pods are deleted and their runs fail. A cancel
    /// asked for meanwhile is dropped: an abandoned run has no job to cancel,
    /// and its flow's error says what became of it.
    pub(super) fn abandon(&mut self, ids: &[TaskId]) -> Vec<Effect> {
        let mut effects = Vec::new();
        for id in ids {
            if let Some(follow) = self.training.tasks.get_mut(id)
                && follow.starting()
                && follow.detach != Detach::Done
            {
                follow.detach = Detach::Done;
                follow.cancel_after = false;
                effects.push(Effect::Abandon(*id));
            }
        }
        effects
    }

    /// Abandons start task `id`, once confirmed after `c`, as [`App::abandon`]
    /// does, and says so, quitting or not. Its job may have started since the
    /// dialog opened: it is then not abandoned, and the status line says how
    /// to cancel it, since its pod keeps billing.
    pub(super) fn abandon_start(&mut self, id: TaskId) -> Vec<Effect> {
        let Some(follow) = self.training.tasks.get(&id) else {
            return Vec::new();
        };
        let run = follow.run();
        if !follow.starting() {
            self.say(
                Severity::Warn,
                format!(
                    "{run}: its job started, so it was not abandoned and its pod keeps billing; \
                     press c to cancel it"
                ),
            );
            return Vec::new();
        }
        if follow.detach == Detach::Done {
            self.say(Severity::Info, format!("{run} is being abandoned"));
            return Vec::new();
        }
        self.say(Severity::Info, format!("abandoning {run}"));
        self.abandon(&[id])
    }

    /// On a process signal: every task that follows a run is abandoned at once,
    /// as Ctrl-C does on the command line (a Runpod run still provisioning
    /// deletes its pod and fails); cancels are waited for. A cancel waiting
    /// for a task never starts after a signal; the exit notes say how to run
    /// it (see [`App::trained`]).
    pub(super) fn abandon_all(&mut self) -> Vec<Effect> {
        self.training
            .tasks
            .iter_mut()
            .filter(|(_, follow)| follow.job != Job::Cancel)
            .map(|(id, follow)| {
                follow.detach = Detach::Done;
                Effect::Abandon(*id)
            })
            .collect()
    }

    /// When quitting is called off: tasks still waiting for their job to detach
    /// keep following, unless a cancel waits for them. Returns how many runs
    /// are detached all the same (their tokens are cancelled), and how many
    /// starts are abandoned all the same.
    pub(super) fn keep_following(&mut self) -> (usize, usize) {
        let (mut detached, mut abandoned) = (0, 0);
        for follow in self.training.tasks.values_mut() {
            if follow.detach == Detach::OnStart && !follow.cancel_after {
                follow.detach = Detach::No;
            }
            if follow.detach != Detach::Done || follow.job == Job::Cancel {
                continue;
            }
            if follow.starting() {
                abandoned += 1;
            } else {
                detached += 1;
            }
        }
        (detached, abandoned)
    }

    /// When the runs were last read: never, for a new app.
    pub(super) fn never() -> SystemTime {
        SystemTime::UNIX_EPOCH
    }
}

/// `text` with its first letter in upper case.
fn capitalized(text: &str) -> String {
    let mut chars = text.chars();
    chars.next().map_or_else(String::new, |first| {
        first.to_uppercase().chain(chars).collect()
    })
}

/// The exit note of a run asked for but not started because the TUI ended.
pub(super) const NOT_STARTED: &str = "a new training run was not started: the TUI ended first; \
                                       start it again with `overbrainer train` or t";

/// The dialog asking to start the run of `plan`, with the GPU catalog `gpus`
/// once read.
pub(super) fn start_dialog(plan: Box<StartPlan>, gpus: Option<&Catalog>) -> Confirm {
    let title = if plan.resume.is_some() {
        " Resume a training run? "
    } else {
        " Start a training run? "
    };
    Confirm {
        title: title.to_string(),
        text: start::text(&plan, gpus),
        yes: "start",
        no: "cancel",
        action: Action::Start(plan),
    }
}
