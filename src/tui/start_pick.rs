//! `g` and `c` in the dialog starting a run on a Runpod target: the GPU types
//! and data centers picked from the catalog are used for the run and saved to
//! `overbrainer.toml` on `y`, validated and written atomically like a save of
//! the Project view, before the run starts. A save refused (validation, the
//! file changed, a lock) starts nothing. Pending Project changes are never
//! saved along: `g`, `c` and such a `y` are refused until they are saved or
//! dropped.

use super::app::{App, Effect, Origin, Overlay, Picked, Severity};
use super::catalog::{CatalogKind, Query};
use super::follow::start_dialog;
use super::start::{DATA_CENTER_IDS, GPU_TYPES, RunpodPlan, StartPlan};
use super::tasks::Task;
use super::widgets::picker::Choice;
use crate::config::edit::{ConfigDoc, FieldPath};
use crate::config::fields::FieldValue;
use crate::config::{CONFIG_FILE, ListOrAuto};

/// The text of `overbrainer.toml`, read as `base`, with the fields of the
/// Runpod target `target` that `runpod` changed set to its choices: `auto`
/// as such, an empty data center list unset (any).
///
/// # Errors
///
/// Returns why the text cannot be edited.
fn saved_text(base: &str, target: &str, runpod: &RunpodPlan) -> Result<String, String> {
    let mut doc = ConfigDoc::parse(base).map_err(|error| error.to_string())?;
    for field in runpod.changed() {
        let path = FieldPath::Target {
            name: target.to_string(),
            field,
        };
        let value = match field {
            GPU_TYPES => &runpod.spec.gpu_types,
            DATA_CENTER_IDS => &runpod.spec.data_center_ids,
            // An `auto` limit dropped with listed GPU types.
            _ => &ListOrAuto::default(),
        };
        let edited = match value {
            ListOrAuto::Auto => doc.set(&path, FieldValue::Text(ListOrAuto::AUTO.to_string())),
            ListOrAuto::List(ids) if ids.is_empty() => doc.unset(&path).map(drop),
            ListOrAuto::List(ids) => doc.set(&path, FieldValue::List(ids.clone())),
        };
        edited.map_err(|error| error.to_string())?;
    }
    Ok(doc.text())
}

impl App {
    /// `g` (`gpus`) or `c` in the start dialog of a Runpod run: opens the GPU
    /// type or data center picker on what the run would use, unless the
    /// choice could not be saved; the dialog then stays and the status line
    /// says why.
    pub(super) fn choose_for_start(&mut self, gpus: bool) -> Vec<Effect> {
        let field = if gpus { GPU_TYPES } else { DATA_CENTER_IDS };
        let Some(Overlay::Confirm(confirm)) = &self.overlay else {
            return Vec::new();
        };
        let super::app::Action::Start(plan) = &confirm.action else {
            return Vec::new();
        };
        let (target, Some(runpod)) = (plan.target.clone(), plan.runpod.clone()) else {
            return Vec::new();
        };
        if let Some(reason) = self.start_choice_refusal(&target, field, &runpod) {
            self.say(Severity::Warn, format!("refused: {reason}"));
            return Vec::new();
        }
        let Some(Overlay::Confirm(confirm)) = self.overlay.take() else {
            return Vec::new();
        };
        let super::app::Action::Start(plan) = confirm.action else {
            return Vec::new();
        };
        self.start_held = Some(plan);
        let spec = &runpod.spec;
        let (kind, preselected) = if gpus {
            (CatalogKind::Gpus, Choice::from(&spec.gpu_types))
        } else {
            (
                CatalogKind::DataCenters,
                Choice::from(&spec.data_center_ids),
            )
        };
        let query = Query {
            kind,
            gpu_count: spec.gpu_count,
            gpu_types: spec.gpu_types.list().to_vec(),
        };
        self.open_picker(query, preselected, Origin::Start(target))
    }

    /// Why the field `field` of the Runpod target `target` (whose plan is
    /// `runpod`) cannot be chosen at start and saved now, if it cannot: no
    /// configuration read, a save running, pending Project changes, the TUI
    /// quitting, the data center of a network volume, or the field set by
    /// the environment or used by a run, as the Project view refuses it.
    fn start_choice_refusal(
        &mut self,
        target: &str,
        field: &'static str,
        runpod: &RunpodPlan,
    ) -> Option<String> {
        if self.config.is_none() {
            return Some(format!("{CONFIG_FILE} could not be read"));
        }
        if self.project_view.save.is_some() {
            return Some(format!("{CONFIG_FILE} is being saved"));
        }
        if self.project_view.pending.is_some() {
            return Some(
                "save (s) or drop (u) the pending Project changes first; they are never \
                 saved along"
                    .to_string(),
            );
        }
        if self.leaving.is_some() {
            return Some("quitting; nothing is saved".to_string());
        }
        let key = format!("targets.{target}.{field}");
        if field == DATA_CENTER_IDS && runpod.spec.network_volume_id.is_some() {
            return Some(format!(
                "{key} is the network volume's data center; pick the volume in the Project view"
            ));
        }
        let listing = self.project_listing();
        let row = listing
            .find(&key)
            .and_then(|index| listing.field(index))
            .filter(|row| row.key == key)?;
        if let Some(user) = &row.lock {
            return Some(format!("{key} is used by {user}; read-only until it ends"));
        }
        row.env_note()
    }

    /// The picker opened from the start dialog of the Runpod target `target`
    /// kept `picked`: the plan uses it, and the dialog shows again. No GPU
    /// type chosen changes nothing.
    pub(super) fn picked_start(&mut self, target: &str, picked: Picked) {
        if let Some(runpod) = self
            .start_held
            .as_mut()
            .filter(|plan| plan.target == target)
            .and_then(|plan| plan.runpod.as_mut())
        {
            match (picked.kind, picked.choice) {
                (CatalogKind::Gpus, Choice::List(ids)) if ids.is_empty() => {
                    self.say(Severity::Warn, "choose a GPU type or auto; nothing changed");
                },
                (CatalogKind::Gpus, Choice::Auto) => runpod.choose_gpus(ListOrAuto::Auto),
                (CatalogKind::Gpus, Choice::List(ids)) => {
                    runpod.choose_gpus(ListOrAuto::List(ids));
                },
                (CatalogKind::DataCenters, Choice::Auto) => {
                    runpod.spec.data_center_ids = ListOrAuto::Auto;
                },
                (CatalogKind::DataCenters, Choice::List(ids)) => {
                    runpod.spec.data_center_ids = ListOrAuto::List(ids);
                },
                (CatalogKind::Volumes | CatalogKind::Templates, _) => {},
            }
        }
        self.reopen_start();
    }

    /// Shows the start dialog again, on its plan held while its picker was
    /// open.
    pub(super) fn reopen_start(&mut self) {
        if let Some(plan) = self.start_held.take() {
            self.overlay = Some(Overlay::Confirm(start_dialog(
                plan,
                self.start_gpus.as_ref(),
            )));
        }
    }

    /// `y` in the start dialog: starts the run of `plan`, after saving the
    /// choices made with `g` and `c`, if any; refused when either cannot
    /// happen now.
    pub(super) fn confirm_start(&mut self, plan: Box<StartPlan>) -> Vec<Effect> {
        let changed = plan
            .runpod
            .as_ref()
            .map(|runpod| runpod.changed())
            .unwrap_or_default();
        let Some(runpod) = plan.runpod.as_ref().filter(|_| !changed.is_empty()) else {
            return self.start_run(&plan);
        };
        if self.refuse_new("one task at a time", "run") {
            return Vec::new();
        }
        let refusal = changed
            .iter()
            .find_map(|field| self.start_choice_refusal(&plan.target, field, runpod));
        let base = self.config.as_ref().map(|config| config.text.clone());
        let text = match (refusal, base) {
            (Some(reason), _) => Err(reason),
            (None, None) => Err(format!("{CONFIG_FILE} could not be read")),
            (None, Some(base)) => saved_text(&base, &plan.target, runpod).map(|text| (text, base)),
        };
        let (text, base) = match text {
            Ok(saved) => saved,
            Err(reason) => {
                self.say(
                    Severity::Warn,
                    format!("refused: {reason}; run not started"),
                );
                return Vec::new();
            },
        };
        let id = self.task_id();
        self.project_view.save = Some(id);
        self.say(
            Severity::Info,
            format!(
                "saving {CONFIG_FILE}, then starting a run on {}",
                plan.target
            ),
        );
        self.start_after_save = Some(plan);
        vec![Effect::Spawn(
            id,
            Task::SaveConfig {
                text,
                base,
                env: self.env.clone(),
            },
        )]
    }

    /// The choices of `plan` were saved: its run starts, with the settings
    /// written.
    pub(super) fn started_after_save(&mut self, plan: &StartPlan) -> Vec<Effect> {
        let effects = self.start_run(plan);
        if !effects.is_empty() {
            self.say(
                Severity::Info,
                format!("✓ saved {CONFIG_FILE}; starting a run on {}", plan.target),
            );
        }
        effects
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crossterm::event::KeyCode;

    use super::*;
    use crate::tui::app::{Action, Confirm};
    use crate::tui::catalog::Listed;
    use crate::tui::project_edit::save_config;
    use crate::tui::snapshots::{
        PROJECT_CONFIG, draw, gpu_catalog, gpu_types, key, project_app, project_env, runpod_plan,
        text as screen,
    };
    use crate::tui::tasks::{Done, TaskId, TrainJob};
    use crate::tui::widgets::picker::Entry;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// The Project app on a directory holding `config`, with the start
    /// dialog of a run on `gpu_cloud` (GPU types `["NVIDIA A40"]`) open and
    /// its GPU catalog read.
    fn starting(config: &str) -> Result<(tempfile::TempDir, App), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join(CONFIG_FILE), config)?;
        let mut app = project_app()?;
        app.project.dir = dir.path().to_path_buf();
        app.env = project_env();
        let read = crate::tui::project::ProjectConfig::new(config, &app.env)?;
        let spec = read
            .settings
            .targets
            .get("gpu_cloud")
            .and_then(crate::runpod::RunpodTarget::from_target)
            .ok_or("no Runpod target gpu_cloud")?;
        app.config = Some(read);
        let plan = StartPlan {
            runpod: Some(Box::new(RunpodPlan::new(spec))),
            ..runpod_plan()
        };
        let effects = app.prepared(Ok(plan));
        let [Effect::Spawn(lookup, Task::StartCatalog(1))] = effects.as_slice() else {
            return Err(format!("{effects:?}").into());
        };
        app.on_done(*lookup, Ok(Done::StartCatalog(Ok(gpu_types()?))));
        Ok((dir, app))
    }

    fn press(app: &mut App, codes: &[KeyCode]) -> Vec<Effect> {
        codes
            .iter()
            .flat_map(|code| app.on_input(&key(*code)))
            .collect()
    }

    fn status(app: &App) -> &str {
        app.status
            .as_ref()
            .map_or("", |status| status.text.as_str())
    }

    /// The text of the dialog open, empty without one.
    fn dialog(app: &App) -> String {
        match &app.overlay {
            Some(Overlay::Confirm(confirm)) => confirm.text.join("\n"),
            _ => String::new(),
        }
    }

    /// `key` in the start dialog: the listing of the picker it opens.
    fn open(app: &mut App, key: char) -> Result<(TaskId, Query), String> {
        let effects = press(app, &[KeyCode::Char(key)]);
        match effects.as_slice() {
            [Effect::Spawn(id, Task::Catalog(query))] => Ok((*id, query.clone())),
            _ => Err(format!("no listing: {effects:?} ({})", status(app))),
        }
    }

    fn listed(app: &mut App, id: TaskId, entries: Vec<Entry>) {
        app.on_done(
            id,
            Ok(Done::Catalog(Ok(Listed {
                entries,
                gpus: Vec::new(),
            }))),
        );
    }

    fn center(id: &str) -> Entry {
        Entry {
            id: id.into(),
            columns: vec![id.into(), String::new(), String::new(), "HIGH".into()],
            selectable: true,
            ranks: Vec::new(),
        }
    }

    /// Picks the RTX 2000 before the A40 with `g`.
    fn pick_gpus(app: &mut App) -> Result<(), Box<dyn std::error::Error>> {
        let (id, _) = open(app, 'g')?;
        listed(app, id, gpu_catalog(1)?);
        press(
            app,
            &[
                KeyCode::Down,
                KeyCode::Char(' '),
                KeyCode::Char('K'),
                KeyCode::Enter,
            ],
        );
        Ok(())
    }

    /// `y`, then the save it asks for run in `dir`, its end handed back.
    fn confirm_and_save(app: &mut App, dir: &Path) -> Result<Vec<Effect>, String> {
        let effects = press(app, &[KeyCode::Char('y')]);
        let [Effect::Spawn(id, Task::SaveConfig { text, base, env })] = effects.as_slice() else {
            return Err(format!("no save: {effects:?} ({})", status(app)));
        };
        assert_eq!(app.project_view.save, Some(*id));
        assert_eq!(app.lock().as_deref(), Some("a new run is starting"));
        let saved = save_config(dir, text, base, env).map(Box::new);
        Ok(app.on_done(*id, Ok(Done::ConfigSaved(saved))))
    }

    fn starts(effects: &[Effect]) -> bool {
        effects
            .iter()
            .any(|effect| matches!(effect, Effect::Spawn(_, Task::Train(TrainJob::Start))))
    }

    fn written(dir: &Path) -> std::io::Result<String> {
        std::fs::read_to_string(dir.join(CONFIG_FILE))
    }

    #[test]
    fn g_and_c_open_untyped_pickers_on_what_the_run_would_use() -> TestResult {
        let (_dir, mut app) = starting(PROJECT_CONFIG)?;
        assert!(
            dialog(&app).contains("- NVIDIA A40  $0.40/h        48 GB  HIGH"),
            "price, VRAM and stock: {}",
            dialog(&app)
        );
        let footer = screen(&draw(&mut app, 80, 24)?).join("\n");
        assert!(footer.contains("g GPU types · c data centers"), "{footer}");
        let (_, query) = open(&mut app, 'g')?;
        assert_eq!(
            query,
            Query {
                kind: CatalogKind::Gpus,
                gpu_count: 1,
                gpu_types: vec!["NVIDIA A40".into()],
            }
        );
        let Some(Overlay::Picker(picking)) = &app.overlay else {
            return Err("no picker".into());
        };
        assert_eq!(picking.origin, Origin::Start("gpu_cloud".into()));
        assert!(!picking.picker.typed(), "t is not offered at start");
        let rows = screen(&draw(&mut app, 80, 24)?).join("\n");
        assert!(!rows.contains("t type"), "{rows}");
        assert_eq!(press(&mut app, &[KeyCode::Char('t')]), []);
        press(&mut app, &[KeyCode::Esc]);
        assert!(
            dialog(&app).contains("NVIDIA A40"),
            "Esc shows the dialog again"
        );
        let (_, query) = open(&mut app, 'c')?;
        assert_eq!(query.kind, CatalogKind::DataCenters);
        assert_eq!(query.gpu_types, ["NVIDIA A40"]);
        Ok(())
    }

    #[test]
    fn a_gpu_id_from_the_api_reaches_the_dialog_without_control_characters() -> TestResult {
        let (_dir, mut app) = starting(PROJECT_CONFIG)?;
        let gpus: Vec<crate::runpod::GpuType> = serde_json::from_value(serde_json::json!([
            {"id": "odd\u{1b}[2J\ngpu", "memory": 48, "price": {"secure": 0.3},
             "maxCount": {"secure": 8}, "availability": "HIGH"}
        ]))?;
        let (id, _) = open(&mut app, 'g')?;
        listed(&mut app, id, crate::tui::catalog::gpu_entries(&gpus, 1));
        press(
            &mut app,
            &[KeyCode::End, KeyCode::Char(' '), KeyCode::Enter],
        );
        let shown = dialog(&app);
        assert!(shown.contains("odd gpu"), "{shown}");
        assert!(!shown.contains('\u{1b}'), "{shown:?}");
        Ok(())
    }

    #[test]
    fn a_choice_is_shown_then_saved_before_the_run_starts() -> TestResult {
        let (dir, mut app) = starting(PROJECT_CONFIG)?;
        pick_gpus(&mut app)?;
        let (id, _) = open(&mut app, 'c')?;
        listed(&mut app, id, vec![center("EU-RO-1"), center("US-KS-2")]);
        press(
            &mut app,
            &[KeyCode::Home, KeyCode::Char(' '), KeyCode::Enter],
        );
        let shown = dialog(&app);
        assert!(
            shown.contains(
                "changed     gpu_types, data_center_ids: saved to overbrainer.toml on y, then \
                 the run starts"
            ),
            "{shown}"
        );
        assert!(
            shown.contains("- NVIDIA RTX 2000 Ada Generation  $0.24/h        16 GB  HIGH"),
            "{shown}"
        );
        let rows = screen(&draw(&mut app, 80, 24)?).join("\n");
        for line in [
            "changed     gpu_types",
            "max_hours   6",
            "y start",
            "datacenters auto",
        ] {
            assert!(rows.contains(line), "fits 80x24: {line}\n{rows}");
        }
        assert_eq!(written(dir.path())?, PROJECT_CONFIG, "nothing written yet");
        let effects = confirm_and_save(&mut app, dir.path())?;
        assert!(starts(&effects), "{effects:?}");
        let text = written(dir.path())?;
        assert!(
            text.contains(r#"gpu_types = ["NVIDIA RTX 2000 Ada Generation", "NVIDIA A40"]"#),
            "{text}"
        );
        assert!(text.contains(r#"data_center_ids = "auto""#), "{text}");
        assert_eq!(
            status(&app),
            "✓ saved overbrainer.toml; starting a run on gpu_cloud"
        );
        assert_eq!(app.start_after_save, None);
        Ok(())
    }

    #[test]
    fn choices_left_as_they_were_start_without_a_save() -> TestResult {
        let (dir, mut app) = starting(PROJECT_CONFIG)?;
        let (id, _) = open(&mut app, 'g')?;
        listed(&mut app, id, gpu_catalog(1)?);
        press(&mut app, &[KeyCode::Enter]);
        assert!(!dialog(&app).contains("changed"), "{}", dialog(&app));
        let effects = press(&mut app, &[KeyCode::Char('y')]);
        assert!(starts(&effects), "{effects:?}");
        assert_eq!(app.project_view.save, None);
        assert_eq!(written(dir.path())?, PROJECT_CONFIG);
        Ok(())
    }

    #[test]
    fn a_refused_save_starts_nothing_and_says_why() -> TestResult {
        let (dir, mut app) = starting(PROJECT_CONFIG)?;
        pick_gpus(&mut app)?;
        std::fs::write(dir.path().join(CONFIG_FILE), "# changed elsewhere\n")?;
        let effects = confirm_and_save(&mut app, dir.path())?;
        assert!(!starts(&effects), "{effects:?}");
        assert_eq!(written(dir.path())?, "# changed elsewhere\n");
        assert!(
            status(&app).starts_with(
                "run not started: overbrainer.toml not saved: overbrainer.toml changed"
            ),
            "{}",
            status(&app)
        );
        assert_eq!(app.lock(), None);
        assert!(app.project_view.errors.is_empty(), "the file is fine");
        Ok(())
    }

    /// `gpu_types = "auto"` with both limits.
    fn auto_config() -> String {
        PROJECT_CONFIG.replace(
            "gpu_types = [\"NVIDIA A40\"]",
            "gpu_types = \"auto\"\nmin_vram_gb = 24\nmax_price_per_hour = 1.5",
        )
    }

    /// `g`, then the RTX 2000 alone instead of `auto`.
    fn pick_a_list(app: &mut App) -> Result<(), Box<dyn std::error::Error>> {
        let (id, _) = open(app, 'g')?;
        listed(app, id, gpu_catalog(1)?);
        press(app, &[KeyCode::Down, KeyCode::Char(' '), KeyCode::Enter]);
        Ok(())
    }

    #[test]
    fn listed_gpu_types_drop_the_auto_limits_and_the_save_succeeds() -> TestResult {
        let auto = auto_config();
        let (dir, mut app) = starting(&auto)?;
        pick_a_list(&mut app)?;
        assert!(
            dialog(&app).contains(
                "changed     gpu_types (min_vram_gb and max_price_per_hour removed): saved to \
                 overbrainer.toml on y, then the run starts"
            ),
            "{}",
            dialog(&app)
        );
        // Back to auto: the limits come back, nothing is changed.
        let (id, _) = open(&mut app, 'g')?;
        listed(&mut app, id, gpu_catalog(1)?);
        press(
            &mut app,
            &[KeyCode::Home, KeyCode::Char(' '), KeyCode::Enter],
        );
        assert!(!dialog(&app).contains("changed"), "{}", dialog(&app));
        pick_a_list(&mut app)?;
        let effects = confirm_and_save(&mut app, dir.path())?;
        assert!(starts(&effects), "{effects:?} ({})", status(&app));
        let text = written(dir.path())?;
        assert!(
            text.contains(r#"gpu_types = ["NVIDIA RTX 2000 Ada Generation"]"#),
            "{text}"
        );
        assert!(
            !text.contains("min_vram_gb") && !text.contains("max_price_per_hour"),
            "{text}"
        );
        Ok(())
    }

    #[test]
    fn quitting_while_the_choice_is_saved_starts_nothing_and_notes_it() -> TestResult {
        for (signal, moved) in [(false, false), (true, false), (false, true)] {
            let (dir, mut app) = starting(PROJECT_CONFIG)?;
            pick_gpus(&mut app)?;
            let effects = press(&mut app, &[KeyCode::Char('y')]);
            let [Effect::Spawn(id, Task::SaveConfig { text, base, env })] = effects.as_slice()
            else {
                return Err(format!("{effects:?}").into());
            };
            if signal {
                app.on_signal();
            } else {
                press(&mut app, &[KeyCode::Char('q'), KeyCode::Char('y')]);
            }
            assert_eq!(app.exit, None, "waits for the save");
            if moved {
                std::fs::write(dir.path().join(CONFIG_FILE), "# changed elsewhere\n")?;
            }
            let saved = save_config(dir.path(), text, base, env).map(Box::new);
            let effects = app.on_done(*id, Ok(Done::ConfigSaved(saved)));
            assert!(!starts(&effects), "{effects:?}");
            assert!(app.exit.is_some());
            let note = if moved {
                "a new training run was not started: overbrainer.toml not saved: \
                 overbrainer.toml changed"
            } else {
                crate::tui::follow::NOT_STARTED
            };
            assert!(
                app.exit_notes.iter().any(|said| said.starts_with(note)),
                "{:?}",
                app.exit_notes
            );
        }
        Ok(())
    }

    #[test]
    fn pending_changes_a_lock_or_a_volume_refuse_the_choice() -> TestResult {
        let (_dir, mut app) = starting(PROJECT_CONFIG)?;
        app.project_view.pending = Some(crate::tui::project::Pending::new(
            &crate::tui::project::ProjectConfig::new(PROJECT_CONFIG, &app.env)?,
        ));
        assert_eq!(press(&mut app, &[KeyCode::Char('g')]), []);
        assert_eq!(
            status(&app),
            "refused: save (s) or drop (u) the pending Project changes first; they are never \
             saved along"
        );
        assert!(dialog(&app).contains("NVIDIA A40"), "the dialog stays");
        app.project_view.pending = None;
        let mut follow =
            crate::tui::training::Follow::new(crate::tui::training::Job::Attach, "20260921-a1");
        follow.watching = true;
        app.training.tasks.insert(TaskId(90), follow);
        assert_eq!(press(&mut app, &[KeyCode::Char('c')]), []);
        assert_eq!(
            status(&app),
            "refused: targets.gpu_cloud.data_center_ids is used by run 20260921-a1; read-only \
             until it ends"
        );
        // Chosen first, then the lock: `y` saves nothing, starts nothing.
        app.training.tasks.clear();
        pick_gpus(&mut app)?;
        let mut follow =
            crate::tui::training::Follow::new(crate::tui::training::Job::Attach, "20260921-a1");
        follow.watching = true;
        app.training.tasks.insert(TaskId(91), follow);
        assert_eq!(press(&mut app, &[KeyCode::Char('y')]), []);
        assert!(status(&app).ends_with("read-only until it ends; run not started"));
        let volume = PROJECT_CONFIG.replace(
            "max_hours = 6",
            "max_hours = 6\nnetwork_volume_id = \"vol1\"\ndata_center_ids = [\"EU-RO-1\"]",
        );
        let (_dir, mut app) = starting(&volume)?;
        if let Some(Overlay::Confirm(Confirm {
            action: Action::Start(plan),
            ..
        })) = &mut app.overlay
            && let Some(runpod) = &mut plan.runpod
        {
            runpod.spec.network_volume_id = Some("vol1".into());
        }
        assert_eq!(press(&mut app, &[KeyCode::Char('c')]), []);
        assert!(
            status(&app).contains("is the network volume's data center"),
            "{}",
            status(&app)
        );
        Ok(())
    }

    #[test]
    fn g_and_c_cancel_a_start_on_another_target_like_any_key() -> TestResult {
        let (_dir, mut app) = starting(PROJECT_CONFIG)?;
        for code in ['g', 'c'] {
            let plan = StartPlan {
                target: "homelab".into(),
                kind: "ssh, docker".into(),
                runpod: None,
                ..runpod_plan()
            };
            app.overlay = Some(Overlay::Confirm(crate::tui::follow::start_dialog(
                Box::new(plan),
                None,
            )));
            // Like any key but `y`: the dialog closes, nothing opens or starts.
            assert_eq!(press(&mut app, &[KeyCode::Char(code)]), [], "{code}");
            assert_eq!(app.overlay, None, "{code}");
            assert_eq!(app.start_held, None);
        }
        Ok(())
    }

    #[test]
    fn ctrl_c_in_a_start_picker_forgets_the_plan() -> TestResult {
        let (_dir, mut app) = starting(PROJECT_CONFIG)?;
        open(&mut app, 'g')?;
        assert!(app.start_held.is_some());
        app.on_input(&crate::tui::snapshots::ctrl_c());
        assert_eq!(app.start_held, None);
        assert_eq!(app.start_gpus, None);
        Ok(())
    }
}
