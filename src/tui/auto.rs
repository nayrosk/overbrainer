//! Auto mode: `A`, or `auto` in the `r` menu, asks once, then runs every
//! stage, each once the one before ended without failed items, then starts a
//! training run as `t` does and follows it. A failure stops the chain at its
//! link; running auto again resumes it, as the stages skip what is done.

use crossterm::event::KeyCode;

use super::app::{Action, App, Confirm, Effect, Overlay, Severity, View};
use super::pipeline::command_name;
use super::start::{self, AutoPlan, Catalog};
use super::tasks::{Task, TaskId, TrainJob};
use crate::cli::data::Command;
use crate::config::CONFIG_FILE;

/// The stages auto mode runs, in order, before training.
pub(super) const STAGES: [Command; 4] = [
    Command::Subtopics,
    Command::Questions,
    Command::Answers,
    Command::Split,
];

/// The name of the chain's last link.
const TRAIN: &str = "train";

/// What `c` in the Pipeline view says once the chain trains.
const TRAINING_CANCEL: &str = "auto mode is training: c in the Training view cancels the run";

/// Where a link of the chain is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Mark {
    /// Not reached.
    Pending,
    /// Running.
    Running,
    /// Ended well.
    Done,
    /// Failed or cancelled: the chain stopped there.
    Stopped,
}

/// How the chain is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ChainState {
    /// A link runs.
    Running,
    /// Every link ended well.
    Done,
    /// A link failed, or was cancelled: the chain stopped there.
    Stopped,
}

/// Auto mode running, or how it last ended.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Chain {
    /// What runs after split.
    pub(super) plan: AutoPlan,
    /// The link running, or where the chain stopped: a position in
    /// [`STAGES`], then training.
    pub(super) at: usize,
    /// How it is.
    pub(super) state: ChainState,
    /// The training task, once started.
    pub(super) train_task: Option<TaskId>,
}

impl Chain {
    fn new(plan: AutoPlan) -> Self {
        Self {
            plan,
            at: 0,
            state: ChainState::Running,
            train_task: None,
        }
    }

    /// Whether a link still runs.
    pub(super) fn running(&self) -> bool {
        self.state == ChainState::Running
    }

    /// Whether it runs a stage, not training.
    fn in_stages(&self) -> bool {
        self.running() && self.at < STAGES.len()
    }

    /// The links, each with where it is: the stages, then `train` when
    /// training is configured.
    pub(super) fn links(&self) -> Vec<(&'static str, Mark)> {
        let mut names: Vec<&'static str> = STAGES.iter().map(|c| command_name(*c)).collect();
        if self.plan.run.is_some() {
            names.push(TRAIN);
        }
        names
            .into_iter()
            .enumerate()
            .map(|(at, name)| {
                let mark = match (self.state, at.cmp(&self.at)) {
                    (ChainState::Done, _) | (_, std::cmp::Ordering::Less) => Mark::Done,
                    (_, std::cmp::Ordering::Greater) => Mark::Pending,
                    (ChainState::Running, _) => Mark::Running,
                    (ChainState::Stopped, _) => Mark::Stopped,
                };
                (name, mark)
            })
            .collect()
    }
}

/// Auto mode's state in the app.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct Auto {
    /// The task preparing its confirmation, if any.
    pub(super) prepare: Option<TaskId>,
    /// The chain running, or the last one.
    pub(super) chain: Option<Chain>,
}

impl Auto {
    /// Whether a chain runs.
    pub(super) fn running(&self) -> bool {
        self.chain.as_ref().is_some_and(Chain::running)
    }

    /// The chain's training task, while it runs.
    pub(super) fn train_task(&self) -> Option<TaskId> {
        self.chain
            .as_ref()
            .filter(|chain| chain.running())
            .and_then(|chain| chain.train_task)
    }
}

/// The confirmation text of `plan`: the stages, then the run's, as `t`
/// shows it, with the GPU types of the catalog once looked up.
pub(super) fn text(plan: &AutoPlan, gpus: Option<&Catalog>) -> Vec<String> {
    let mut text = vec![
        "stages      subtopics, questions, answers, split: each starts once the one before \
         ended without failed items; what is done is skipped"
            .to_string(),
    ];
    match &plan.run {
        None => text.push(format!(
            "training    none: no [training] in {CONFIG_FILE}, so auto mode stops after split"
        )),
        Some((run, _)) => {
            text.push("training    then a run starts, as t starts it:".to_string());
            text.extend(start::text_after_split(run, gpus));
        },
    }
    text
}

/// The dialog asking to run `plan`.
pub(super) fn dialog(plan: Box<AutoPlan>, gpus: Option<&Catalog>) -> Confirm {
    Confirm {
        title: " Run auto mode? ".to_string(),
        text: text(&plan, gpus),
        yes: "run",
        no: "cancel",
        action: Action::Auto(plan),
    }
}

impl App {
    /// `A`, or `auto` in the `r` menu: prepares the confirmation of auto
    /// mode, in a task; refused while a chain, a stage, an edit or a start
    /// runs, or while the TUI quits.
    pub(super) fn ask_auto(&mut self) -> Vec<Effect> {
        if self.auto.running() {
            self.say(Severity::Info, "auto mode is already running");
            return Vec::new();
        }
        if self.refuse_new("one task at a time", "stage") {
            return Vec::new();
        }
        if self.auto.prepare.is_some() {
            self.say(Severity::Info, "already preparing auto mode");
            return Vec::new();
        }
        let id = self.task_id();
        self.auto.prepare = Some(id);
        vec![Effect::Spawn(id, Task::PrepareAuto)]
    }

    /// The confirmation of auto mode is ready: it opens, and the GPU catalog
    /// of a Runpod run is read meanwhile. Dropped once the TUI quits, and
    /// while something else is open (the status line then says to press `A`
    /// again).
    pub(super) fn auto_prepared(&mut self, plan: Result<AutoPlan, String>) -> Vec<Effect> {
        self.auto.prepare = None;
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
                "auto mode is prepared: press A again to run it",
            );
            return Vec::new();
        }
        let gpu_count = plan
            .run
            .as_ref()
            .and_then(|(run, _)| run.runpod.as_ref())
            .map(|runpod| runpod.spec.gpu_count);
        self.overlay = Some(Overlay::Confirm(dialog(Box::new(plan), None)));
        self.start_catalog = None;
        self.start_gpus = None;
        let Some(gpu_count) = gpu_count else {
            return Vec::new();
        };
        let id = self.task_id();
        self.start_catalog = Some(id);
        vec![Effect::Spawn(id, Task::StartCatalog(gpu_count))]
    }

    /// `y` on the confirmation: the chain starts with its first stage.
    pub(super) fn confirm_auto(&mut self, plan: AutoPlan) -> Vec<Effect> {
        let effects = self.run_pipeline(STAGES[0]);
        if effects.is_empty() {
            return effects;
        }
        self.auto.chain = Some(Chain::new(plan));
        effects
    }

    /// A pipeline task ended, its outcome in the Pipeline view: the chain's
    /// next link starts when its stage ended well; a failure, or the TUI
    /// quitting, stops it. The caller has reloaded the data.
    pub(super) fn auto_stage_ended(&mut self) -> Vec<Effect> {
        let Some(chain) = self.auto.chain.as_ref().filter(|chain| chain.in_stages()) else {
            return Vec::new();
        };
        let (at, run) = (chain.at, chain.plan.run.clone());
        let stage = command_name(STAGES[at]);
        let error = match &self.pipeline.outcome {
            Some(Ok(())) if self.leaving.is_none() => None,
            Some(Err(error)) => Some(error.clone()),
            _ => Some("the TUI is quitting".to_string()),
        };
        if let Some(error) = error {
            self.chain_ended(ChainState::Stopped);
            if self.leaving.is_none() {
                self.say(
                    Severity::Error,
                    format!("auto stopped at {stage}: {error}; A resumes it"),
                );
            }
            return Vec::new();
        }
        if let Some(chain) = self.auto.chain.as_mut() {
            chain.at = at + 1;
        }
        let effects = match (STAGES.get(at + 1), run) {
            (Some(next), _) => self.run_pipeline(*next),
            (None, None) => {
                self.chain_ended(ChainState::Done);
                self.say(
                    Severity::Info,
                    format!("✓ auto done after split: no [training] in {CONFIG_FILE}"),
                );
                return Vec::new();
            },
            (None, Some((run, _))) => self.start_run(&run),
        };
        let task = effects.iter().find_map(|effect| match effect {
            Effect::Spawn(id, Task::Train(TrainJob::Start { .. }) | Task::Pipeline(_)) => Some(*id),
            _ => None,
        });
        let Some(task) = task else {
            // Refused: the status line already says why.
            self.chain_ended(ChainState::Stopped);
            return effects;
        };
        if at + 1 < STAGES.len() {
            return effects;
        }
        if let Some(chain) = self.auto.chain.as_mut() {
            chain.train_task = Some(task);
        }
        let mut effects = effects;
        effects.extend(self.show(View::Training));
        effects
    }

    /// The chain ends in `state`.
    fn chain_ended(&mut self, state: ChainState) {
        if let Some(chain) = self.auto.chain.as_mut() {
            chain.state = state;
        }
    }

    /// Training task `id` ended on run `run`, failing with `error`, or
    /// stopped with a snapshot at step `stopped_at`: the chain ends there when
    /// it is its run, saying where the model is, or how to resume it.
    pub(super) fn auto_trained(
        &mut self,
        id: TaskId,
        run: &str,
        error: Option<&str>,
        stopped_at: Option<u64>,
    ) {
        if self.auto.train_task() != Some(id) {
            return;
        }
        let Some(chain) = self.auto.chain.as_mut() else {
            return;
        };
        let leaving = self.leaving.is_some();
        let said = match (error, &chain.plan.run) {
            // Stopped with a snapshot: the model is not finished.
            (None, Some(_)) if !leaving && stopped_at.is_some() => {
                chain.state = ChainState::Stopped;
                (
                    Severity::Info,
                    format!(
                        "auto stopped at {TRAIN}: snapshot at step {}; T resumes it",
                        stopped_at.unwrap_or_default()
                    ),
                )
            },
            (None, Some((_, outputs))) if !leaving => {
                chain.state = ChainState::Done;
                let paths: Vec<String> = outputs
                    .paths(run)
                    .into_iter()
                    .enumerate()
                    .map(|(at, (what, path))| {
                        if at == 0 {
                            path
                        } else {
                            format!("{what} {path}")
                        }
                    })
                    .collect();
                (Severity::Info, format!("✓ auto done: {}", paths.join(", ")))
            },
            (Some(error), _) => {
                chain.state = ChainState::Stopped;
                (Severity::Warn, format!("auto stopped at {TRAIN}: {error}"))
            },
            _ => {
                chain.state = ChainState::Stopped;
                return;
            },
        };
        if !leaving {
            self.say(said.0, said.1);
        }
    }

    /// A key in the Pipeline view: `c` asks to cancel the chain while it runs
    /// a stage.
    pub(super) fn on_pipeline_key(&mut self, code: KeyCode) {
        if code != KeyCode::Char('c') {
            return;
        }
        let Some(chain) = self.auto.chain.as_ref().filter(|chain| chain.running()) else {
            return;
        };
        if !chain.in_stages() {
            self.say(Severity::Info, TRAINING_CANCEL);
            return;
        }
        let stage = command_name(STAGES[chain.at]);
        self.overlay = Some(Overlay::Confirm(Confirm {
            title: " Cancel auto mode? ".to_string(),
            text: vec![format!(
                "{stage} stops now (what it finished is kept) and the rest of the chain does \
                 not run. A runs it again from where it stopped."
            )],
            yes: "cancel it",
            no: "keep going",
            action: Action::CancelAuto,
        }));
    }

    /// `y` on the cancel dialog: the chain stops, and its stage with it.
    pub(super) fn cancel_auto(&mut self) -> Vec<Effect> {
        let Some(chain) = self.auto.chain.as_mut().filter(|chain| chain.in_stages()) else {
            // The chain reached training while the dialog was open.
            if self.auto.running() {
                self.say(Severity::Info, TRAINING_CANCEL);
            }
            return Vec::new();
        };
        chain.state = ChainState::Stopped;
        let stage = command_name(STAGES[chain.at]);
        self.say(
            Severity::Info,
            format!("auto cancelled: {stage} stops; A resumes it"),
        );
        self.pipeline_task
            .map(|id| vec![Effect::Cancel(id)])
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::Outputs;
    use crate::tui::snapshots::{app, key, runpod_plan};
    use crate::tui::tasks::Done;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn press(app: &mut App, codes: &[KeyCode]) -> Vec<Effect> {
        codes
            .iter()
            .flat_map(|code| app.on_input(&key(*code)))
            .collect()
    }

    fn spawned(effects: &[Effect]) -> Vec<(TaskId, Task)> {
        effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Spawn(id, task) if !matches!(task, Task::Load | Task::Runs) => {
                    Some((*id, task.clone()))
                },
                _ => None,
            })
            .collect()
    }

    /// A plan with a local run, or none.
    fn plan(training: bool) -> AutoPlan {
        let mut run = runpod_plan();
        run.runpod = None;
        run.target = "local".into();
        run.kind = "local, docker".into();
        AutoPlan {
            run: training.then(|| {
                (
                    Box::new(run),
                    Outputs {
                        adapter: crate::config::Adapter::Qlora,
                        merge: true,
                    },
                )
            }),
        }
    }

    /// An app whose `A` was answered with `plan` and `y`: the first stage is
    /// spawned. Returns it.
    fn confirmed(app: &mut App, plan: AutoPlan) -> Result<TaskId, String> {
        app.view = View::Pipeline;
        let effects = press(app, &[KeyCode::Char('A')]);
        let [(prepare, Task::PrepareAuto)] = spawned(&effects)[..] else {
            return Err(format!("{effects:?}"));
        };
        app.on_done(prepare, Ok(Done::PreparedAuto(Ok(plan))));
        let effects = press(app, &[KeyCode::Char('y')]);
        match spawned(&effects)[..] {
            [(id, Task::Pipeline(Command::Subtopics))] => Ok(id),
            _ => Err(format!("{effects:?}")),
        }
    }

    /// Ends pipeline task `id` with `outcome`; returns the task spawned next.
    fn end(app: &mut App, id: TaskId, outcome: Result<(), String>) -> Vec<(TaskId, Task)> {
        spawned(&app.on_done(id, Ok(Done::Pipeline(outcome))))
    }

    fn status(app: &App) -> String {
        app.status
            .as_ref()
            .map(|s| s.text.clone())
            .unwrap_or_default()
    }

    #[test]
    fn the_confirmation_lists_the_stages_then_the_run() {
        let with = text(&plan(true), None);
        assert!(with[0].starts_with("stages      subtopics, questions, answers, split"));
        assert_eq!(with[1], "training    then a run starts, as t starts it:");
        assert!(
            with.iter()
                .any(|line| line.starts_with("target      local"))
        );
        assert!(with.iter().any(|line| line.contains("rebuilt by split")));
        let without = text(&plan(false), None);
        assert_eq!(
            without[1],
            "training    none: no [training] in overbrainer.toml, so auto mode stops after split"
        );
    }

    #[test]
    fn the_chain_runs_each_stage_after_the_last_then_trains() -> TestResult {
        let mut app = app();
        let mut id = confirmed(&mut app, plan(true))?;
        for next in [Command::Questions, Command::Answers, Command::Split] {
            let spawned = end(&mut app, id, Ok(()));
            let [(next_id, Task::Pipeline(command))] = spawned[..] else {
                return Err(format!("{spawned:?}").into());
            };
            assert_eq!(command, next);
            id = next_id;
        }
        let spawned = end(&mut app, id, Ok(()));
        let [(train, Task::Train(TrainJob::Start { .. }))] = spawned[..] else {
            return Err(format!("{spawned:?}").into());
        };
        assert_eq!(
            app.view,
            View::Training,
            "training start shows the Training view"
        );
        app.on_message(crate::tui::tasks::Msg::Report(
            train,
            crate::cli::front::Report::RunCreated("r7".into()),
        ));
        app.on_done(train, Ok(Done::Trained(Ok(()))));
        assert_eq!(
            status(&app),
            "✓ auto done: runs/r7/output, merged model runs/r7/output/merged"
        );
        assert!(!app.auto.running());
        Ok(())
    }

    #[test]
    fn a_run_stopped_with_a_snapshot_stops_the_chain_and_says_how_to_resume() -> TestResult {
        let mut app = app();
        let mut id = confirmed(&mut app, plan(true))?;
        for _ in 0..3 {
            let spawned = end(&mut app, id, Ok(()));
            let [(next, Task::Pipeline(_))] = spawned[..] else {
                return Err(format!("{spawned:?}").into());
            };
            id = next;
        }
        let spawned = end(&mut app, id, Ok(()));
        let [(train, Task::Train(TrainJob::Start { .. }))] = spawned[..] else {
            return Err(format!("{spawned:?}").into());
        };
        app.on_message(crate::tui::tasks::Msg::Report(
            train,
            crate::cli::front::Report::RunCreated("r7".into()),
        ));
        app.on_message(crate::tui::tasks::Msg::Report(
            train,
            crate::cli::front::Report::RunStopped(1240),
        ));
        app.on_done(train, Ok(Done::Trained(Ok(()))));
        assert_eq!(
            status(&app),
            "auto stopped at train: snapshot at step 1240; T resumes it"
        );
        assert!(!app.auto.running());
        let chain = app.auto.chain.as_ref().ok_or("no chain")?;
        assert_eq!(chain.state, ChainState::Stopped);
        Ok(())
    }

    #[test]
    fn a_stage_that_failed_stops_the_chain_with_its_error() -> TestResult {
        let mut app = app();
        let id = confirmed(&mut app, plan(true))?;
        let spawned = end(&mut app, id, Err("2 items failed".into()));
        assert!(spawned.is_empty(), "{spawned:?}");
        assert_eq!(
            status(&app),
            "auto stopped at subtopics: 2 items failed; A resumes it"
        );
        let chain = app.auto.chain.as_ref().ok_or("no chain")?;
        assert_eq!(chain.links()[0], ("subtopics", Mark::Stopped));
        assert_eq!(chain.links()[1], ("questions", Mark::Pending));
        Ok(())
    }

    #[test]
    fn without_training_the_chain_stops_after_split_and_says_so() -> TestResult {
        let mut app = app();
        let mut id = confirmed(&mut app, plan(false))?;
        for _ in 0..3 {
            let spawned = end(&mut app, id, Ok(()));
            let [(next, Task::Pipeline(_))] = spawned[..] else {
                return Err(format!("{spawned:?}").into());
            };
            id = next;
        }
        assert!(end(&mut app, id, Ok(())).is_empty());
        assert_eq!(
            status(&app),
            "✓ auto done after split: no [training] in overbrainer.toml"
        );
        let chain = app.auto.chain.as_ref().ok_or("no chain")?;
        assert_eq!(chain.links().len(), 4);
        assert!(chain.links().iter().all(|(_, mark)| *mark == Mark::Done));
        Ok(())
    }

    #[test]
    fn c_cancels_the_running_stage_and_the_rest_after_asking() -> TestResult {
        let mut app = app();
        let id = confirmed(&mut app, plan(true))?;
        assert_eq!(
            press(&mut app, &[KeyCode::Char('c'), KeyCode::Char('n')]),
            []
        );
        assert!(app.auto.running(), "n keeps it going");
        let effects = press(&mut app, &[KeyCode::Char('c'), KeyCode::Char('y')]);
        assert_eq!(effects, [Effect::Cancel(id)]);
        let spawned = end(&mut app, id, Err("interrupted".into()));
        assert!(spawned.is_empty(), "{spawned:?}");
        assert!(!app.auto.running());
        Ok(())
    }

    #[test]
    fn a_cancel_confirmed_once_training_started_says_where_to_cancel() -> TestResult {
        let mut app = app();
        confirmed(&mut app, plan(true))?;
        press(&mut app, &[KeyCode::Char('c')]);
        let chain = app.auto.chain.as_mut().ok_or("no chain")?;
        chain.at = STAGES.len();
        chain.train_task = Some(TaskId(99));
        assert_eq!(press(&mut app, &[KeyCode::Char('y')]), []);
        assert_eq!(status(&app), TRAINING_CANCEL);
        assert!(app.auto.running());
        Ok(())
    }

    #[test]
    fn a_is_refused_while_a_stage_runs_and_its_failure_is_said() -> TestResult {
        let mut app = app();
        confirmed(&mut app, plan(true))?;
        assert_eq!(press(&mut app, &[KeyCode::Char('A')]), []);
        assert_eq!(status(&app), "auto mode is already running");

        let mut app = crate::tui::snapshots::app();
        app.view = View::Project;
        let effects = press(&mut app, &[KeyCode::Char('A')]);
        let [(prepare, Task::PrepareAuto)] = spawned(&effects)[..] else {
            return Err(format!("{effects:?}").into());
        };
        app.on_done(
            prepare,
            Ok(Done::PreparedAuto(Err("invalid configuration".into()))),
        );
        assert_eq!(status(&app), "refused: invalid configuration");
        assert!(app.overlay.is_none());
        Ok(())
    }

    #[test]
    fn the_pipeline_view_draws_the_chain() -> TestResult {
        let mut app = app();
        app.view = View::Pipeline;
        // The fixture runs answers, after subtopics and questions.
        crate::tui::snapshots::pipeline_running(&mut app);
        app.auto.chain = Some(Chain {
            at: 2,
            ..Chain::new(plan(true))
        });
        crate::tui::snapshots::snapshot("pipeline_auto_running", &mut app)?;
        let task = app.pipeline_task.ok_or("no stage runs")?;
        end(&mut app, task, Err("1 item failed".into()));
        assert_eq!(
            status(&app),
            "auto stopped at answers: 1 item failed; A resumes it"
        );
        crate::tui::snapshots::snapshot("pipeline_auto_stopped", &mut app)?;
        Ok(())
    }

    #[test]
    fn the_confirmation_is_drawn_with_the_run_and_its_cost() -> TestResult {
        let mut app = app();
        app.view = View::Pipeline;
        let effects = press(&mut app, &[KeyCode::Char('A')]);
        let [(prepare, Task::PrepareAuto)] = spawned(&effects)[..] else {
            return Err(format!("{effects:?}").into());
        };
        let plan = AutoPlan {
            run: Some((
                Box::new(runpod_plan()),
                Outputs {
                    adapter: crate::config::Adapter::Lora,
                    merge: false,
                },
            )),
        };
        let effects = app.on_done(prepare, Ok(Done::PreparedAuto(Ok(plan))));
        let [(catalog, Task::StartCatalog(1))] = spawned(&effects)[..] else {
            return Err(format!("{effects:?}").into());
        };
        app.on_done(
            catalog,
            Ok(Done::StartCatalog(crate::tui::snapshots::looked_up(Ok(
                crate::tui::snapshots::gpu_types()?,
            )))),
        );
        crate::tui::snapshots::snapshot("auto_dialog", &mut app)?;
        Ok(())
    }

    #[test]
    fn opening_in_auto_mode_prepares_its_confirmation() {
        let mut app = app();
        app.opening = crate::tui::app::Opening::Auto;
        let effects = app.start();
        assert!(
            spawned(&effects)
                .iter()
                .any(|(_, task)| *task == Task::PrepareAuto)
        );
        let mut app = crate::tui::snapshots::app();
        assert!(
            spawned(&app.start()).is_empty(),
            "the Project view: nothing more"
        );
    }
}
