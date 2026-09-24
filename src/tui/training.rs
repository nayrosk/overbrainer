//! The Training view's model: the runs of `runs/`, their metrics, and the
//! training tasks the TUI follows.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Duration;

use super::tasks::TaskId;
use crate::events::Event;
use crate::runpod::{PodRecord, PodStatus};
use crate::runs::{RunRecord, Runs};
use crate::train::{METRICS_FILE, MetricLine, TrainMetric, parse_line};

/// A run of `runs/`, with its pod record when it has one.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct RunRow {
    /// `run.json`.
    pub(super) record: RunRecord,
    /// `pod.json`, for a Runpod run.
    pub(super) pod: Option<PodRecord>,
}

/// What a training task does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Job {
    /// Starts a run, then follows it.
    Start {
        /// Whether its target is a Runpod one.
        runpod: bool,
    },
    /// Follows a run again.
    Attach,
    /// Cancels a run's job.
    Cancel,
}

/// Whether a training task is detached: its token cancelled, so its raced watch
/// ends and the flow reports that the run keeps running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Detach {
    /// Not asked to.
    No,
    /// Asked to before its job's first status: its token is cancelled only then,
    /// so a start is never cut.
    OnStart,
    /// Its token is cancelled.
    Done,
}

/// A training task the TUI runs.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Follow {
    /// What it does.
    pub(super) job: Job,
    /// Its run; empty until a new run is created.
    pub(super) run_id: String,
    /// Whether its job's status was seen: its watch began.
    pub(super) watching: bool,
    /// The latest status of its pod.
    pub(super) pod: Option<PodStatus>,
    /// The lines the command line would print.
    pub(super) lines: Vec<String>,
    /// Metrics its forwarder skipped.
    pub(super) skipped: u64,
    /// Whether the run is cancelled once this task, detached, ends.
    pub(super) cancel_after: bool,
    /// Whether it is detached, or to be once its job's first status arrives.
    pub(super) detach: Detach,
}

impl Follow {
    /// A task doing `job` on run `run_id`.
    pub(super) fn new(job: Job, run_id: &str) -> Self {
        Self {
            job,
            run_id: run_id.to_string(),
            watching: false,
            pod: None,
            lines: Vec::new(),
            skipped: 0,
            cancel_after: false,
            detach: Detach::No,
        }
    }

    /// Whether it is a start whose job has not started yet: it must not be
    /// interrupted, and it holds the data lock.
    pub(super) fn starting(&self) -> bool {
        matches!(self.job, Job::Start { .. }) && !self.watching
    }

    /// Its run, or `a new run` before it is created.
    pub(super) fn run(&self) -> String {
        if self.run_id.is_empty() {
            "a new run".to_string()
        } else {
            format!("run {}", self.run_id)
        }
    }
}

/// What a training task does to its run, as the Training view shows it:
/// computed in one place, [`RunActivity::of`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RunActivity {
    /// No task on the run.
    None,
    /// A task follows the run: its job started, or it attached to it.
    Followed,
    /// A start whose job has not started yet; `c` abandons a Runpod one.
    Starting {
        /// Whether its target is a Runpod one.
        runpod: bool,
    },
    /// A start whose job has not started yet, being abandoned: its token is
    /// cancelled.
    Abandoning {
        /// Whether its target is a Runpod one.
        runpod: bool,
    },
    /// A cancel task, or a task whose run is cancelled once it ended.
    Cancelling,
}

impl RunActivity {
    /// What `follow`, the task on a run if any, does to it.
    pub(super) fn of(follow: Option<&Follow>) -> Self {
        let Some(follow) = follow else {
            return Self::None;
        };
        if follow.job == Job::Cancel || follow.cancel_after {
            return Self::Cancelling;
        }
        let runpod = follow.job == (Job::Start { runpod: true });
        match (follow.starting(), follow.detach) {
            (true, Detach::Done) => Self::Abandoning { runpod },
            (true, _) => Self::Starting { runpod },
            (false, _) => Self::Followed,
        }
    }

    /// The activity in one word, as the runs table shows it; empty for none.
    /// A start being abandoned still reads `starting`: it is, until its
    /// flow ends.
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Followed => "followed",
            Self::Starting { .. } | Self::Abandoning { .. } => "starting",
            Self::Cancelling => "cancelling",
        }
    }

    /// Whether `c` abandons the run rather than cancel it: a Runpod start
    /// still starting, which has no job to cancel yet.
    pub(super) fn abandons(self) -> bool {
        matches!(
            self,
            Self::Starting { runpod: true } | Self::Abandoning { runpod: true }
        )
    }
}

/// How the last task of a run ended.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct Ended {
    /// The lines it reported.
    pub(super) lines: Vec<String>,
    /// Its error, if it failed or was detached.
    pub(super) error: Option<String>,
    /// Metrics its forwarder skipped.
    pub(super) skipped: u64,
    /// Whether the run's metrics were read again from its local file since: they
    /// then hold every metric, late or skipped.
    pub(super) healed: bool,
}

/// What a read of `runs/` found: the runs newest first, or why they cannot be
/// listed, and the warnings of the records it skipped.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Listing {
    /// The runs, newest first.
    pub(super) runs: Result<Vec<RunRow>, String>,
    /// Why a run record or a pod record was skipped, one line each.
    pub(super) skipped: Vec<String>,
}

/// Reads the runs of the project in `dir`, with their pod records: local files
/// only. Blocking.
pub(super) fn list_runs(dir: &Path) -> Listing {
    let runs = Runs::new(dir);
    let mut skipped = Vec::new();
    let listed = runs.list_with(|id, error| {
        skipped.push(format!("skipping unreadable run record for {id}: {error}"));
    });
    let runs = match listed {
        Ok(records) => Ok(records
            .into_iter()
            .rev()
            .map(|record| {
                let pod = match PodRecord::load(&runs, &record.id) {
                    Ok(pod) => pod,
                    Err(error) => {
                        skipped.push(format!(
                            "cannot read the pod record of run {}: {error}",
                            record.id
                        ));
                        None
                    },
                };
                RunRow { record, pod }
            })
            .collect()),
        Err(error) => Err(format!("cannot list the runs: {error}")),
    };
    Listing { runs, skipped }
}

/// The Training view's state.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct TrainingView {
    /// The runs, newest first.
    pub(super) runs: Vec<RunRow>,
    /// The selected run, by position in `runs`.
    pub(super) selected: usize,
    /// Why the runs could not be listed at the last read; the runs shown are
    /// then those of an earlier read.
    pub(super) error: Option<String>,
    /// Metrics of each run shown, by run ID.
    pub(super) series: BTreeMap<String, Vec<TrainMetric>>,
    /// The training tasks running.
    pub(super) tasks: BTreeMap<TaskId, Follow>,
    /// How the last task of each run ended.
    pub(super) ended: BTreeMap<String, Ended>,
    /// The training tasks that ended, with their run, until another task
    /// starts on it: their messages handled after their end still count.
    pub(super) last: BTreeMap<TaskId, String>,
    /// The read of `runs/` running, if any.
    pub(super) listing: Option<TaskId>,
    /// Whether another read of `runs/` was asked for while one ran.
    pub(super) list_again: bool,
    /// The run to select once it is listed (a run just created).
    pub(super) wanted: Option<String>,
    /// The latest read of each run's local metrics, by run ID; an earlier one's
    /// result is ignored.
    pub(super) reading: BTreeMap<String, TaskId>,
    /// The listing warnings already logged: each is logged once.
    pub(super) warned: BTreeSet<String>,
}

/// The progress a series of metrics shows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Progress {
    /// Steps done.
    pub(super) step: u64,
    /// Steps of the whole run.
    pub(super) max_steps: Option<u64>,
    /// Fractional epoch.
    pub(super) epoch: Option<f64>,
    /// Time left at the rate seen so far.
    pub(super) eta: Option<Duration>,
}

/// The progress of `series`: from its latest metric, with an ETA from the steps
/// per second between the first and the latest training log, measured on the
/// plugin's own clock.
pub(super) fn progress(series: &[TrainMetric]) -> Option<Progress> {
    let latest = series.last()?;
    let logs: Vec<&TrainMetric> = series.iter().filter(|m| m.loss.is_some()).collect();
    let max_steps = series.iter().rev().find_map(|m| m.max_steps);
    let eta = match (logs.first(), logs.last(), max_steps) {
        (Some(first), Some(last), Some(max))
            if last.step > first.step && last.time > first.time =>
        {
            let rate = float(last.step - first.step) / (last.time - first.time);
            Duration::try_from_secs_f64(float(max.saturating_sub(latest.step)) / rate).ok()
        },
        _ => None,
    };
    Some(Progress {
        step: latest.step,
        max_steps,
        epoch: series.iter().rev().find_map(|m| m.epoch),
        eta,
    })
}

/// `count` as a float, saturating at `u32::MAX`.
pub(super) fn float(count: u64) -> f64 {
    f64::from(u32::try_from(count).unwrap_or(u32::MAX))
}

/// The metrics in `runs/<id>/metrics.jsonl` of the project in `dir`, malformed
/// lines skipped; `None` when the file cannot be read.
pub(super) fn read_series(dir: &Path, id: &str) -> Option<Vec<TrainMetric>> {
    let path = Runs::new(dir).run_dir(id).ok()?.join(METRICS_FILE);
    let content = std::fs::read_to_string(path).ok()?;
    Some(
        content
            .lines()
            .filter_map(|line| match parse_line(line) {
                Ok(MetricLine::Log(metric)) => Some(metric),
                _ => None,
            })
            .collect(),
    )
}

impl TrainingView {
    /// Shows the runs of `listing`, keeping the selected run, or selecting the
    /// run wanted once it is listed. When it failed, the earlier runs stay, with
    /// its error. Returns the warnings not logged yet.
    pub(super) fn listed(&mut self, listing: Listing) -> Vec<String> {
        let selected = self.selected_run().map(|row| row.record.id.clone());
        match listing.runs {
            Ok(runs) => {
                self.runs = runs;
                self.error = None;
            },
            Err(error) => self.error = Some(error),
        }
        if let Some(id) = selected {
            self.select(&id);
        }
        if let Some(id) = self.wanted.take() {
            if self.runs.iter().any(|row| row.record.id == id) {
                self.select(&id);
            } else {
                self.wanted = Some(id);
            }
        }
        self.selected = self.selected.min(self.runs.len().saturating_sub(1));
        listing
            .skipped
            .into_iter()
            .filter(|warning| self.warned.insert(warning.clone()))
            .collect()
    }

    /// Selects run `id`, when listed.
    pub(super) fn select(&mut self, id: &str) {
        if let Some(index) = self.runs.iter().position(|row| row.record.id == id) {
            self.selected = index;
        }
    }

    /// The selected run.
    pub(super) fn selected_run(&self) -> Option<&RunRow> {
        self.runs.get(self.selected)
    }

    /// The share of its steps the selected run did, when its metrics say how
    /// many it has.
    pub(super) fn selected_ratio(&self) -> Option<f64> {
        let row = self.selected_run()?;
        let now = progress(self.series.get(&row.record.id)?)?;
        let max = now.max_steps.filter(|max| *max > 0)?;
        Some((float(now.step) / float(max)).min(1.0))
    }

    /// What the task on run `id`, if any, does to it.
    pub(super) fn activity(&self, id: &str) -> RunActivity {
        RunActivity::of(self.task_of(id).map(|(_, follow)| follow))
    }

    /// What the task on the selected run, if any, does to it.
    pub(super) fn selected_activity(&self) -> RunActivity {
        self.selected_run()
            .map_or(RunActivity::None, |row| self.activity(&row.record.id))
    }

    /// The task following or cancelling run `id`, if any.
    pub(super) fn task_of(&self, id: &str) -> Option<(TaskId, &Follow)> {
        self.tasks
            .iter()
            .find(|(_, follow)| follow.run_id == id)
            .map(|(task, follow)| (*task, follow))
    }

    /// Whether task `id` is a training task, running or ended.
    pub(super) fn is_training(&self, id: TaskId) -> bool {
        self.tasks.contains_key(&id) || self.last.contains_key(&id)
    }

    /// The selected run, when its local metrics are to be read: not read yet,
    /// not being read, and followed by no task.
    pub(super) fn to_read(&self) -> Option<String> {
        let id = &self.selected_run()?.record.id;
        let skip = self.series.contains_key(id)
            || self.reading.contains_key(id)
            || self.task_of(id).is_some();
        (!skip).then(|| id.clone())
    }

    /// Whether run `id` is being cancelled: by a cancel task, or by one that
    /// detaches its task first.
    pub(super) fn cancelling(&self, id: &str) -> bool {
        self.activity(id) == RunActivity::Cancelling
    }

    /// Records `event` of task `id`. Returns whether it is the first job status
    /// of that task: its watch began.
    pub(super) fn event(&mut self, id: TaskId, event: Event) -> bool {
        let Some(follow) = self.tasks.get_mut(&id) else {
            return false;
        };
        match event {
            Event::Metric(metric) => {
                self.series
                    .entry(follow.run_id.clone())
                    .or_default()
                    .push(metric);
            },
            Event::JobStatus(_) if !follow.watching => {
                follow.watching = true;
                return true;
            },
            Event::PodStatus(status) => follow.pod = Some(status),
            _ => {},
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log(time: f64, step: u64, loss: Option<f64>) -> TrainMetric {
        TrainMetric {
            time,
            step,
            epoch: Some(0.5),
            max_steps: Some(400),
            loss,
            eval_loss: loss.is_none().then_some(1.0),
            learning_rate: Some(2e-4),
            grad_norm: Some(0.8),
        }
    }

    #[test]
    fn the_eta_follows_the_training_logs_on_the_plugins_clock() -> Result<(), String> {
        let series = [
            log(100.0, 10, Some(2.0)),
            log(150.0, 60, None),
            log(200.0, 110, Some(1.5)),
        ];
        let now = progress(&series).ok_or("no progress")?;
        assert_eq!((now.step, now.max_steps), (110, Some(400)));
        assert_eq!(now.eta, Some(Duration::from_secs(290)));
        assert_eq!(progress(&series[..1]).and_then(|p| p.eta), None);
        assert_eq!(progress(&[]), None);
        Ok(())
    }

    #[test]
    fn one_activity_says_what_a_task_does_to_its_run() {
        let runpod = Job::Start { runpod: true };
        let task = |job: &Job, change: &dyn Fn(&mut Follow)| {
            let mut follow = Follow::new(job.clone(), "run");
            change(&mut follow);
            RunActivity::of(Some(&follow))
        };
        let keep = |_: &mut Follow| {};
        assert_eq!(RunActivity::of(None), RunActivity::None);
        assert_eq!(task(&Job::Attach, &keep), RunActivity::Followed);
        assert_eq!(task(&runpod, &|f| f.watching = true), RunActivity::Followed);
        assert_eq!(task(&runpod, &keep), RunActivity::Starting { runpod: true });
        assert_eq!(
            task(&Job::Start { runpod: false }, &keep),
            RunActivity::Starting { runpod: false }
        );
        assert_eq!(
            task(&runpod, &|f| f.detach = Detach::Done),
            RunActivity::Abandoning { runpod: true }
        );
        assert_eq!(task(&Job::Cancel, &keep), RunActivity::Cancelling);
        assert_eq!(
            task(&runpod, &|f| f.cancel_after = true),
            RunActivity::Cancelling
        );
        assert!(task(&runpod, &keep).abandons());
        assert!(task(&runpod, &|f| f.detach = Detach::Done).abandons());
        assert!(!task(&Job::Start { runpod: false }, &keep).abandons());
        assert!(!task(&Job::Attach, &keep).abandons());
        let labels: Vec<&str> = [
            RunActivity::None,
            RunActivity::Followed,
            RunActivity::Starting { runpod: true },
            RunActivity::Abandoning { runpod: true },
            RunActivity::Cancelling,
        ]
        .map(RunActivity::label)
        .to_vec();
        assert_eq!(
            labels,
            ["", "followed", "starting", "starting", "cancelling"]
        );
    }

    #[test]
    fn a_series_is_read_from_the_local_metrics_file() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let run = dir.path().join("runs/20260921-140000-a1b2");
        std::fs::create_dir_all(&run)?;
        std::fs::write(
            run.join(METRICS_FILE),
            "{\"event\": \"begin\", \"time\": 1, \"max_steps\": 2}\nnot json\n\
             {\"event\": \"log\", \"time\": 2, \"step\": 1, \"loss\": 1.5}\n",
        )?;
        let series = read_series(dir.path(), "20260921-140000-a1b2");
        assert_eq!(series.map(|s| s.len()), Some(1));
        assert_eq!(read_series(dir.path(), "20260921-140000-ffff"), None);
        Ok(())
    }
}
