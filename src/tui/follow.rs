//! What the app does with training runs: listing them, attaching, cancelling, and
//! what quitting and signals do to the training tasks.
//!
//! A training task's token is cancelled only once its job's first status arrived
//! (its watch began), so a start is never cut: asked to detach before, it is
//! marked and detached then. Only a process signal abandons a task at once.

use std::time::{Duration, SystemTime};

use crossterm::event::KeyCode;

use super::app::{Action, App, Confirm, Effect, Overlay, Severity, View};
use super::tasks::{Msg, Task, TaskId, TrainJob};
use super::training::{Detach, Ended, Follow, Job, Last, read_series};
use crate::cli::front::Report;
use crate::events::Event;

/// Time between two reads of `runs/` while the Training view is shown.
const REFRESH: Duration = Duration::from_secs(2);

impl App {
    /// Reads `runs/` again, and the local metrics of the selected run.
    pub(super) fn refresh_runs(&mut self) {
        self.training.refresh(&self.project.dir);
        self.training.load_selected(&self.project.dir);
        self.refreshed = self.now;
        self.dirty = true;
    }

    /// Reads `runs/` again when the Training view is shown and it is time (or
    /// the clock went back).
    pub(super) fn refresh_when_due(&mut self) {
        let recent = matches!(
            self.now.duration_since(self.refreshed),
            Ok(since) if since < REFRESH
        );
        if self.view == View::Training && !recent {
            self.refresh_runs();
        }
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
            KeyCode::Char('c') => self.ask_cancel(),
            _ => return Vec::new(),
        }
        self.training.load_selected(&self.project.dir);
        Vec::new()
    }

    /// `a`: follows the selected run again, from its first metric.
    fn attach_selected(&mut self) -> Vec<Effect> {
        let Some(id) = self
            .training
            .selected_run()
            .map(|row| row.record.id.clone())
        else {
            return Vec::new();
        };
        if let Some((_, follow)) = self.training.task_of(&id) {
            let said = match follow.job {
                Job::Attach => format!("run {id} is already followed"),
                Job::Cancel => format!("run {id} is being cancelled"),
            };
            self.say(Severity::Info, said);
            return Vec::new();
        }
        self.training.series.remove(&id);
        self.training.ended.remove(&id);
        self.train(Job::Attach, &id)
    }

    /// Starts a training task doing `job` on run `run_id`. The late messages of
    /// the run's earlier tasks no longer count.
    fn train(&mut self, job: Job, run_id: &str) -> Vec<Effect> {
        let id = self.task_id();
        let task = match job {
            Job::Attach => TrainJob::Attach(run_id.to_string()),
            Job::Cancel => TrainJob::Cancel(run_id.to_string()),
        };
        self.training.last.retain(|_, last| last.run_id != run_id);
        self.training.tasks.insert(id, Follow::new(job, run_id));
        vec![Effect::Spawn(id, Task::Train(task))]
    }

    /// `c`: asks to cancel the selected run's job.
    fn ask_cancel(&mut self) {
        let Some(row) = self.training.selected_run() else {
            return;
        };
        let (id, target) = (row.record.id.clone(), row.record.target.clone());
        if matches!(self.training.task_of(&id), Some((_, follow)) if follow.job == Job::Cancel) {
            self.say(
                Severity::Info,
                format!("run {id} is already being cancelled"),
            );
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
        let Some((task, follow)) = self.training.task_of(run_id) else {
            return self.train(Job::Cancel, run_id);
        };
        if follow.job == Job::Cancel {
            self.say(
                Severity::Info,
                format!("run {run_id} is already being cancelled"),
            );
            return Vec::new();
        }
        if let Some(follow) = self.training.tasks.get_mut(&task) {
            follow.cancel_after = true;
        }
        self.say(
            Severity::Info,
            format!("detaching run {run_id}, then cancelling it"),
        );
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
            Msg::Report(_, Report::RunCreated(run_id)) => {
                if let Some(follow) = self.training.tasks.get_mut(&id) {
                    follow.run_id.clone_from(&run_id);
                }
                self.refresh_runs();
                self.training.select(&run_id);
            },
            Msg::EditorExited(_) => {},
        }
        Vec::new()
    }

    /// A message of training task `id` handled after its end: a line joins how
    /// its run ended, and a metric its series, unless the series was read again
    /// from the local file (which holds it).
    fn late_training_message(&mut self, id: TaskId, message: Msg) {
        let Some(last) = self.training.last.get(&id) else {
            return;
        };
        let run = last.run_id.clone();
        match message {
            Msg::Event(_, Event::Metric(metric)) if !last.healed => {
                self.training.series.entry(run).or_default().push(metric);
            },
            Msg::Report(_, Report::Line(line)) => {
                if self.leaving.is_some() {
                    self.exit_notes.push(line.clone());
                }
                if let Some(ended) = self.training.ended.get_mut(&run) {
                    ended.lines.push(line);
                }
            },
            _ => {},
        }
    }

    /// Training task `id` ended with `result`. While the TUI is leaving, what it
    /// reported and its error (a detached run's "keeps running ... attach") are
    /// kept for the exit.
    pub(super) fn trained(&mut self, id: TaskId, result: Result<(), String>) -> Vec<Effect> {
        let Some(follow) = self.training.tasks.remove(&id) else {
            return Vec::new();
        };
        let run = follow.run_id.clone();
        let error = result.err();
        if self.leaving.is_some() {
            self.exit_notes.extend(follow.lines.iter().cloned());
            self.exit_notes.extend(error.iter().cloned());
        }
        match &error {
            Some(error) if !follow.cancel_after => {
                self.say(Severity::Warn, format!("run {run}: {error}"));
            },
            Some(_) => {},
            None => self.say(Severity::Info, format!("run {run}: done")),
        }
        self.training.ended.insert(
            run.clone(),
            Ended {
                lines: follow.lines,
                error,
            },
        );
        // The local metrics, retrieved with the results, heal any missing point.
        let healed = match read_series(&self.project.dir, &run) {
            Some(series) => {
                self.training.series.insert(run.clone(), series);
                true
            },
            None => false,
        };
        self.training.last.insert(
            id,
            Last {
                run_id: run.clone(),
                healed,
            },
        );
        self.refresh_runs();
        let effects = if follow.cancel_after && self.leaving.is_none() {
            self.train(Job::Cancel, &run)
        } else {
            Vec::new()
        };
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
                match follow.job {
                    Job::Cancel => format!("Run {run}: cancel in progress, quitting waits for it."),
                    Job::Attach => format!(
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

    /// On a process signal: every task that follows a run is abandoned at once,
    /// as Ctrl-C does on the command line (a Runpod run still provisioning
    /// deletes its pod and fails); cancels are waited for.
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
    /// are detached all the same (their tokens are cancelled).
    pub(super) fn keep_following(&mut self) -> usize {
        let mut detached = 0;
        for follow in self.training.tasks.values_mut() {
            if follow.detach == Detach::OnStart && !follow.cancel_after {
                follow.detach = Detach::No;
            }
            if follow.detach == Detach::Done && follow.job == Job::Attach {
                detached += 1;
            }
        }
        detached
    }

    /// When the runs were last read: never, for a new app.
    pub(super) fn never() -> SystemTime {
        SystemTime::UNIX_EPOCH
    }
}
