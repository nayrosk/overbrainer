//! The Compare view's state: the compares of every run, read from `runs/`,
//! the selected one's questions, and the compare running, if any.
//!
//! `C` (here, or in the Training view on the selected run) prepares a
//! compare in a task, asks with what it would do, then runs `overbrainer
//! compare` as a task; `J` judges the selected compare again; `c` cancels
//! the compare running. One compare at a time; `C` and `J` are refused while
//! the data is locked, as a push is.

use std::collections::BTreeSet;
use std::path::Path;

use crossterm::event::KeyCode;

use super::app::{Action, App, Confirm, Effect, Overlay, Severity, View};
use super::tasks::{Msg, Task, TaskId};
use crate::cli::CompareArgs;
use crate::cli::compare::Preview;
use crate::cli::front::Report as FrontReport;
use crate::compare::{CompareEntry, QuestionResult, Verdict, list_compares};
use crate::events::Event;
use crate::runs::Runs;
use crate::train::{Phase, Phases};

/// Lines a page key scrolls the detail by.
const PAGE: u16 = 10;

/// What a read of the compares found, newest first, or why they cannot be
/// listed, and the warnings of the records it skipped.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct CompareListing {
    /// The compares, newest first.
    pub(super) rows: Result<Vec<CompareEntry>, String>,
    /// Why a compare record was skipped, one line each.
    pub(super) skipped: Vec<String>,
}

/// Reads the compares of the project in `dir`. Blocking.
pub(super) fn list(dir: &Path) -> CompareListing {
    match list_compares(&Runs::new(dir)) {
        Ok((rows, skipped)) => CompareListing {
            rows: Ok(rows),
            skipped,
        },
        Err(error) => CompareListing {
            rows: Err(format!("cannot list the compares: {error}")),
            skipped: Vec::new(),
        },
    }
}

/// Which list the keys move in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum Focus {
    /// The compares, on top.
    #[default]
    Compares,
    /// The selected compare's questions, below.
    Questions,
}

/// Which questions the list shows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum Filter {
    /// Every question.
    #[default]
    All,
    /// Losses and child errors.
    Losses,
    /// Ties.
    Ties,
    /// Wins.
    Wins,
    /// Child errors and unparsed verdicts.
    Errors,
}

impl Filter {
    /// The next filter, as `f` cycles them.
    pub(super) fn next(self) -> Self {
        match self {
            Self::All => Self::Losses,
            Self::Losses => Self::Ties,
            Self::Ties => Self::Wins,
            Self::Wins => Self::Errors,
            Self::Errors => Self::All,
        }
    }

    /// Whether a question with `verdict` is shown.
    pub(super) fn keeps(self, verdict: Verdict) -> bool {
        match self {
            Self::All => true,
            Self::Losses => matches!(verdict, Verdict::Loss | Verdict::Error),
            Self::Ties => verdict == Verdict::Tie,
            Self::Wins => verdict == Verdict::Win,
            Self::Errors => matches!(verdict, Verdict::Error | Verdict::Unparsed),
        }
    }
}

/// The compare running, started with `C` or `J`.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Comparing {
    /// Its task.
    pub(super) task: TaskId,
    /// The run compared, when chosen.
    pub(super) run: Option<String>,
    /// The compare's ID, once its job is followed.
    pub(super) compare: Option<String>,
    /// What its job's lines say of its phase.
    pub(super) phases: Phases,
    /// Whether its job exited.
    pub(super) exited: bool,
    /// Questions judged and to judge, once the judge runs.
    pub(super) judged: Option<(u64, u64)>,
    /// Whether it only judges again.
    pub(super) rejudge: bool,
}

impl Comparing {
    /// A compare of run `run` (else of the newest run with a GGUF) started
    /// as `task`; `rejudge` when it only judges again.
    pub(super) fn new(task: TaskId, run: Option<String>, rejudge: bool) -> Self {
        Self {
            task,
            run,
            compare: None,
            phases: Phases::default(),
            exited: false,
            judged: None,
            rejudge,
        }
    }

    /// Where it is: `preparing`, `starting llama-server`, `answering
    /// 34/100`, `retrieving results`, `judging 12/100`.
    pub(super) fn label(&self) -> String {
        if let Some((done, total)) = self.judged {
            return format!("judging {done}/{total}");
        }
        if self.compare.is_none() {
            return if self.rejudge { "judging" } else { "preparing" }.to_string();
        }
        match self.phases.phase(self.exited) {
            // No stage said yet: the job starts its server.
            Phase::Training => Phase::Comparing {
                step: 0,
                total: None,
            }
            .label(),
            phase => phase.label(),
        }
    }
}

/// The Compare view's state.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct CompareView {
    /// The compares, newest first.
    pub(super) rows: Vec<CompareEntry>,
    /// The selected compare, by position in `rows`.
    pub(super) selected: usize,
    /// The selected question, by position in the shown questions.
    pub(super) question: usize,
    /// Which list the keys move in.
    pub(super) focus: Focus,
    /// Which questions show.
    pub(super) filter: Filter,
    /// Scroll of the detail pane, in lines.
    pub(super) scroll: u16,
    /// Where the detail's parts start, for `[` and `]`.
    pub(super) sections: Vec<u16>,
    /// Why the compares could not be listed at the last read.
    pub(super) error: Option<String>,
    /// The read of the compares running, if any.
    pub(super) listing: Option<TaskId>,
    /// Whether another read was asked for while one ran.
    pub(super) list_again: bool,
    /// The preparation of a compare running (`C`), if any.
    pub(super) prepare: Option<TaskId>,
    /// The compare running, if any: one at a time.
    pub(super) running: Option<Comparing>,
    /// The listing warnings already logged.
    pub(super) warned: BTreeSet<String>,
}

impl CompareView {
    /// The selected compare.
    pub(super) fn selected_row(&self) -> Option<&CompareEntry> {
        self.rows.get(self.selected)
    }

    /// The selected compare's questions the filter keeps.
    pub(super) fn shown_questions(&self) -> Vec<&QuestionResult> {
        self.selected_row()
            .and_then(|row| row.report.as_ref())
            .map(|report| {
                report
                    .questions
                    .iter()
                    .filter(|question| self.filter.keeps(question.verdict))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Moves the focused list's selection by one, down or up; a new
    /// compare or question starts its detail at the top.
    fn step(&mut self, down: bool) {
        let questions = self.shown_questions().len();
        let (position, len) = match self.focus {
            Focus::Compares => (&mut self.selected, self.rows.len()),
            Focus::Questions => (&mut self.question, questions),
        };
        *position = if down {
            (*position + 1).min(len.saturating_sub(1))
        } else {
            position.saturating_sub(1)
        };
        if self.focus == Focus::Compares {
            self.question = 0;
        }
        self.scroll = 0;
    }

    /// The selected question kept in the list after the rows or the filter
    /// changed.
    fn clamp(&mut self) {
        let questions = self.shown_questions().len();
        self.question = self.question.min(questions.saturating_sub(1));
    }
}

impl App {
    /// Reads the compares again, in a task; while one reads, once more after it.
    pub(super) fn refresh_compares(&mut self) -> Vec<Effect> {
        self.compares_refreshed = self.now;
        if self.compare.listing.is_some() {
            self.compare.list_again = true;
            return Vec::new();
        }
        let id = self.task_id();
        self.compare.listing = Some(id);
        vec![Effect::Spawn(id, Task::Compares)]
    }

    /// Reads the compares again when the Compare or Training view is shown
    /// and it is time (or the clock went back).
    pub(super) fn compares_when_due(&mut self) -> Vec<Effect> {
        let recent = matches!(
            self.now.duration_since(self.compares_refreshed),
            Ok(since) if since < super::follow::REFRESH
        );
        if matches!(self.view, View::Compare | View::Training) && !recent {
            return self.refresh_compares();
        }
        Vec::new()
    }

    /// Read `id` of the compares found `listing`: shown when it is the
    /// latest read; the selection stays on the same compare when it can.
    /// Each new warning is logged once.
    pub(super) fn compares_listed(&mut self, id: TaskId, listing: CompareListing) -> Vec<Effect> {
        if self.compare.listing != Some(id) {
            return Vec::new();
        }
        self.compare.listing = None;
        self.dirty = true;
        for warning in listing.skipped {
            if self.compare.warned.insert(warning.clone()) {
                tracing::warn!("{warning}");
            }
        }
        match listing.rows {
            Ok(rows) => {
                let selected = self.compare.selected_row().map(|row| row.record.id.clone());
                self.compare.rows = rows;
                self.compare.error = None;
                self.compare.selected = selected
                    .and_then(|id| self.compare.rows.iter().position(|row| row.record.id == id))
                    .unwrap_or(0);
                self.compare.clamp();
            },
            Err(error) => self.compare.error = Some(error),
        }
        if std::mem::take(&mut self.compare.list_again) {
            return self.refresh_compares();
        }
        Vec::new()
    }

    /// Task `id` failed (a panic): the compare, its preparation or a read of
    /// the compares, which then no longer counts. `None` when `id` is none of
    /// them.
    pub(super) fn compare_failed(&mut self, id: TaskId, error: &str) -> Option<Vec<Effect>> {
        if self.is_compare(id) {
            return Some(self.compared(id, Err(error.to_string())));
        }
        if self.compare.prepare == Some(id) {
            return Some(self.compare_prepared(id, Err(error.to_string())));
        }
        if self.compare.listing == Some(id) {
            self.compare.listing = None;
            self.compare.error = Some(format!("cannot list the compares: {error}"));
            return Some(Vec::new());
        }
        None
    }

    /// A key in the Compare view.
    pub(super) fn on_compare_key(&mut self, code: KeyCode) -> Vec<Effect> {
        let view = &mut self.compare;
        match code {
            KeyCode::Up | KeyCode::Char('k') => view.step(false),
            KeyCode::Down | KeyCode::Char('j') => view.step(true),
            KeyCode::Left | KeyCode::Char('h') => view.focus = Focus::Compares,
            KeyCode::Right | KeyCode::Char('l') => view.focus = Focus::Questions,
            KeyCode::PageDown => view.scroll = view.scroll.saturating_add(PAGE),
            KeyCode::PageUp => view.scroll = view.scroll.saturating_sub(PAGE),
            KeyCode::Char(']') => {
                if let Some(&next) = view.sections.iter().find(|&&at| at > view.scroll) {
                    view.scroll = next;
                }
            },
            KeyCode::Char('[') => {
                if let Some(&previous) = view.sections.iter().rfind(|&&at| at < view.scroll) {
                    view.scroll = previous;
                }
            },
            KeyCode::Char('f') => {
                view.filter = view.filter.next();
                view.question = 0;
                view.scroll = 0;
            },
            KeyCode::Char('C') => return self.ask_compare(None),
            KeyCode::Char('J') => self.ask_rejudge(),
            KeyCode::Char('c') => self.ask_cancel_compare(),
            _ => {},
        }
        Vec::new()
    }

    /// Refuses a new compare while the data is locked, the TUI is quitting,
    /// or a compare runs or is being prepared; says why.
    fn compare_refused(&mut self) -> bool {
        if self.refuse_new("one task at a time", "compare") {
            return true;
        }
        if self.compare.running.is_some() || self.compare.prepare.is_some() {
            self.say(Severity::Warn, "a compare is running: one at a time");
            return true;
        }
        false
    }

    /// `C`: prepares, in a task, the confirmation of a compare of run `run`,
    /// else of the newest run with a GGUF.
    pub(super) fn ask_compare(&mut self, run: Option<String>) -> Vec<Effect> {
        if self.compare_refused() {
            return Vec::new();
        }
        let id = self.task_id();
        self.compare.prepare = Some(id);
        vec![Effect::Spawn(id, Task::PrepareCompare(run))]
    }

    /// Preparation `id` found `preview`: asks to compare, or says why not.
    pub(super) fn compare_prepared(
        &mut self,
        id: TaskId,
        preview: Result<Preview, String>,
    ) -> Vec<Effect> {
        if self.compare.prepare != Some(id) {
            return Vec::new();
        }
        self.compare.prepare = None;
        if self.leaving.is_some() {
            return Vec::new();
        }
        let preview = match preview {
            Ok(preview) => preview,
            Err(error) => {
                self.say(Severity::Warn, error);
                return Vec::new();
            },
        };
        if self.overlay.is_some() || self.dataset.input.is_some() {
            self.say(
                Severity::Info,
                "a compare is prepared: press C again to compare",
            );
            return Vec::new();
        }
        let mut text = vec![
            format!("run {}, {}", preview.run, preview.gguf),
            format!(
                "{} questions of data/eval.jsonl, on target `{}`",
                preview.questions, preview.target
            ),
        ];
        text.extend(preview.runpod);
        text.push(
            "The judge (roles.judge, else the parent) is called once per question.".to_string(),
        );
        self.overlay = Some(Overlay::Confirm(Confirm {
            title: format!(" Compare run {}? ", preview.run),
            text,
            yes: "compare",
            no: "cancel",
            action: Action::Compare(CompareArgs {
                run: Some(preview.run),
                ..CompareArgs::default()
            }),
        }));
        Vec::new()
    }

    /// A confirmed compare with `args`: starts it, unless refused now.
    pub(super) fn confirm_compare(&mut self, args: CompareArgs) -> Vec<Effect> {
        if self.compare_refused() {
            return Vec::new();
        }
        let task = self.task_id();
        self.compare.running = Some(Comparing::new(
            task,
            args.run.clone(),
            args.rejudge.is_some(),
        ));
        vec![Effect::Spawn(task, Task::Compare(args))]
    }

    /// `J`: asks to judge the selected compare again.
    fn ask_rejudge(&mut self) {
        if self.compare_refused() {
            return;
        }
        let Some(row) = self.compare.selected_row() else {
            self.say(Severity::Info, "no compare to judge again");
            return;
        };
        let (id, run) = (row.record.id.clone(), row.run.clone());
        self.overlay = Some(Overlay::Confirm(Confirm {
            title: format!(" Judge compare {id} again? "),
            text: vec![
                "The child is not asked again. With the judge and prompt it already used, only \
                 the questions without a verdict are judged; with another, every question is."
                    .to_string(),
            ],
            yes: "judge",
            no: "cancel",
            action: Action::Compare(CompareArgs {
                run: Some(run),
                rejudge: Some(id),
                ..CompareArgs::default()
            }),
        }));
    }

    /// `c`: asks to cancel the compare running.
    fn ask_cancel_compare(&mut self) {
        let Some(running) = &self.compare.running else {
            self.say(Severity::Info, "no compare is running");
            return;
        };
        let task = running.task;
        self.overlay = Some(Overlay::Confirm(Confirm {
            title: " Cancel the compare? ".to_string(),
            text: vec![
                "Its job is cancelled and its pod deleted, as Ctrl-C does on the command line; \
                 while it judges, the verdicts so far are kept for `J`."
                    .to_string(),
            ],
            yes: "cancel it",
            no: "keep it",
            action: Action::CancelCompare(task),
        }));
    }

    /// Whether task `id` is the compare running.
    pub(super) fn is_compare(&self, id: TaskId) -> bool {
        self.compare
            .running
            .as_ref()
            .is_some_and(|running| running.task == id)
    }

    /// A message of the compare task: its job's lines and the judge's
    /// progress; its lines go to the log.
    pub(super) fn on_compare_message(&mut self, message: Msg) -> Vec<Effect> {
        let Some(running) = &mut self.compare.running else {
            return Vec::new();
        };
        match message {
            Msg::Event(_, Event::RunWatched { run_id, .. }) => {
                running.compare = Some(run_id);
                running.phases = Phases::default();
                running.exited = false;
            },
            Msg::Event(_, Event::Mark(mark)) => running.phases.mark(mark),
            Msg::Event(_, Event::Metric(metric)) => running.phases.metric(&metric),
            Msg::Event(_, Event::JobStatus(status)) => running.exited = status.is_finished(),
            Msg::Event(_, Event::Judged { done, total, .. }) => {
                running.judged = Some((done, total));
            },
            Msg::Report(_, FrontReport::Line(line)) => tracing::info!("{line}"),
            _ => {},
        }
        Vec::new()
    }

    /// Compare task `id` ended: says how, and reads the compares again.
    pub(super) fn compared(&mut self, id: TaskId, result: Result<(), String>) -> Vec<Effect> {
        if !self.is_compare(id) {
            return Vec::new();
        }
        self.compare.running = None;
        let (severity, said) = match result {
            Ok(()) => (
                Severity::Info,
                "compare written: see the Compare view".to_string(),
            ),
            Err(error) => (Severity::Error, format!("compare failed: {error}")),
        };
        if self.leaving.is_some() {
            self.exit_notes.push(said.clone());
        }
        self.say(severity, said);
        self.leave_when_idle();
        if self.exit.is_some() {
            return Vec::new();
        }
        self.refresh_compares()
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::KeyCode;

    use super::*;
    use crate::tui::app::{Action, Effect, Overlay, View};
    use crate::tui::snapshots::{app, compare_entry, key};
    use crate::tui::tasks::{Done, Task};

    /// Result type of the tests.
    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// Presses `codes` on `app` and gathers their effects.
    fn press(app: &mut App, codes: &[KeyCode]) -> Vec<Effect> {
        codes
            .iter()
            .flat_map(|code| app.on_input(&key(*code)))
            .collect()
    }

    /// A preview of a compare of run `demo_20261006-100000` on a Runpod target.
    fn preview() -> crate::cli::compare::Preview {
        crate::cli::compare::Preview {
            run: "demo_20261006-100000".into(),
            gguf: "output/gguf/demo-Q4_K_M.gguf".into(),
            questions: 100,
            target: "gpu".into(),
            runpod: Some("a new Runpod pod, at most $0.50/h".into()),
        }
    }

    /// `6` shows the view and lists the compares; their listing fills it.
    #[test]
    fn six_shows_the_compares() -> TestResult {
        let mut app = app();
        let effects = press(&mut app, &[KeyCode::Char('6')]);
        assert_eq!(app.view, View::Compare);
        let Some(Effect::Spawn(id, Task::Compares)) = effects.first().cloned() else {
            return Err(format!("no listing: {effects:?}").into());
        };
        app.on_done(
            id,
            Ok(Done::Compares(CompareListing {
                rows: Ok(vec![compare_entry()?]),
                skipped: Vec::new(),
            })),
        );
        assert_eq!(app.compare.rows.len(), 1);
        Ok(())
    }

    /// `f` cycles the filter; `l` moves to the questions; `j` moves in them.
    #[test]
    fn the_filter_and_focus_move_the_questions() -> TestResult {
        let mut app = app();
        app.view = View::Compare;
        app.compare.rows = vec![compare_entry()?];
        let all = app.compare.shown_questions().len();
        press(&mut app, &[KeyCode::Char('f')]);
        assert_eq!(app.compare.filter, Filter::Losses);
        assert!(app.compare.shown_questions().len() < all);
        assert!(
            app.compare
                .shown_questions()
                .iter()
                .all(|question| matches!(question.verdict, Verdict::Loss | Verdict::Error))
        );
        press(&mut app, &[KeyCode::Char('l'), KeyCode::Char('j')]);
        assert_eq!(app.compare.focus, Focus::Questions);
        assert_eq!(app.compare.question, 1);
        Ok(())
    }

    /// `C` prepares, its preview asks, `y` starts the compare; a second `C`
    /// while it runs is refused.
    #[test]
    fn c_asks_then_starts_one_compare_at_a_time() -> TestResult {
        let mut app = app();
        app.view = View::Compare;
        let effects = press(&mut app, &[KeyCode::Char('C')]);
        let Some(Effect::Spawn(prepare, Task::PrepareCompare(None))) = effects.first().cloned()
        else {
            return Err(format!("no preparation: {effects:?}").into());
        };
        app.on_done(prepare, Ok(Done::ComparePrepared(Ok(preview()))));
        let Some(Overlay::Confirm(confirm)) = &app.overlay else {
            return Err("no confirmation".into());
        };
        assert!(
            confirm
                .text
                .iter()
                .any(|line| line.contains("100 questions"))
        );
        assert!(
            confirm
                .text
                .iter()
                .any(|line| line.contains("at most $0.50/h"))
        );
        assert!(matches!(confirm.action, Action::Compare(_)));
        let effects = press(&mut app, &[KeyCode::Char('y')]);
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, Effect::Spawn(_, Task::Compare(_)))),
            "{effects:?}"
        );
        assert!(app.compare.running.is_some());
        assert_eq!(press(&mut app, &[KeyCode::Char('C')]), Vec::new());
        Ok(())
    }

    /// `C` in the Training view prepares a compare of the selected run.
    #[test]
    fn c_in_the_training_view_compares_the_selected_run() {
        let mut app = app();
        app.view = View::Training;
        let run = crate::tui::snapshots::run(
            "demo_20261006-100000",
            "gpu",
            crate::runs::RunState::Succeeded,
        );
        app.training.runs = vec![crate::tui::training::RunRow {
            record: run,
            pod: None,
        }];
        let effects = press(&mut app, &[KeyCode::Char('C')]);
        assert!(
            matches!(
                effects.as_slice(),
                [Effect::Spawn(_, Task::PrepareCompare(Some(run)))]
                    if run == "demo_20261006-100000"
            ),
            "{effects:?}"
        );
    }

    /// `J` asks to judge the selected compare again; `y` starts it.
    #[test]
    fn j_asks_to_judge_again() -> TestResult {
        let mut app = app();
        app.view = View::Compare;
        app.compare.rows = vec![compare_entry()?];
        press(&mut app, &[KeyCode::Char('J')]);
        let Some(Overlay::Confirm(confirm)) = &app.overlay else {
            return Err("no confirmation".into());
        };
        let Action::Compare(args) = &confirm.action else {
            return Err(format!("not a compare: {:?}", confirm.action).into());
        };
        assert_eq!(args.rejudge.as_deref(), Some("compare_20261006-120000"));
        assert_eq!(args.run.as_deref(), Some("demo_20261006-100000"));
        press(&mut app, &[KeyCode::Char('y')]);
        let label = app.compare.running.as_ref().map(Comparing::label);
        assert_eq!(label.as_deref(), Some("judging"));
        Ok(())
    }

    /// `c` asks to cancel the compare running; `y` abandons its task.
    #[test]
    fn c_asks_then_cancels_the_compare() {
        let mut app = app();
        app.view = View::Compare;
        assert_eq!(press(&mut app, &[KeyCode::Char('c')]), Vec::new());
        assert!(app.overlay.is_none(), "nothing to cancel");
        let task = app.task_id();
        app.compare.running = Some(Comparing::new(task, None, false));
        press(&mut app, &[KeyCode::Char('c')]);
        assert!(matches!(
            &app.overlay,
            Some(Overlay::Confirm(confirm)) if confirm.action == Action::CancelCompare(task)
        ));
        let effects = press(&mut app, &[KeyCode::Char('y')]);
        assert_eq!(effects, vec![Effect::Abandon(task)]);
    }

    /// The progress reads the job's lines, then the judge's.
    #[test]
    fn the_progress_follows_the_job_then_the_judge() {
        use crate::events::Event;
        use crate::train::{JobStage, Mark};
        let mut app = app();
        let task = app.task_id();
        app.compare.running = Some(Comparing::new(task, None, false));
        let message = |event| crate::tui::tasks::Msg::Event(task, event);
        app.on_message(message(Event::RunWatched {
            run_id: "compare_1".into(),
            job: Some(crate::runs::JobKind::Compare),
        }));
        app.on_message(message(Event::Mark(Mark::Stage(JobStage::Compare))));
        app.on_message(message(Event::Mark(Mark::Eval {
            step: 34,
            total: Some(100),
        })));
        let label = app.compare.running.as_ref().map(Comparing::label);
        assert_eq!(label.as_deref(), Some("answering 34/100"));
        app.on_message(message(Event::Judged {
            compare_id: "compare_1".into(),
            done: 12,
            total: 100,
        }));
        let label = app.compare.running.as_ref().map(Comparing::label);
        assert_eq!(label.as_deref(), Some("judging 12/100"));
        app.on_done(task, Ok(Done::Compared(Ok(()))));
        assert!(app.compare.running.is_none());
    }

    /// Quitting while a compare runs asks first, then abandons it and waits.
    #[test]
    fn quitting_abandons_the_compare_and_waits() {
        let mut app = app();
        let task = app.task_id();
        app.compare.running = Some(Comparing::new(task, None, false));
        press(&mut app, &[KeyCode::Char('q')]);
        assert!(matches!(
            &app.overlay,
            Some(Overlay::Confirm(confirm)) if confirm.action == Action::Quit
        ));
        let effects = press(&mut app, &[KeyCode::Char('y')]);
        assert!(effects.contains(&Effect::Abandon(task)), "{effects:?}");
        assert_eq!(app.exit, None, "the compare is waited for");
        assert_eq!(
            app.waiting_for(),
            vec!["waiting for the compare to stop...".to_string()]
        );
        app.on_done(task, Ok(Done::Compared(Err("interrupted".into()))));
        assert!(app.exit.is_some());
    }
}
