//! The look at `overbrainer.toml` and `.env` every few seconds: when their
//! stamp changed, they are read again off the UI thread, and a valid
//! configuration is applied as a save from the Project view applies it. A
//! stage or a run already started keeps the settings it read.

use std::time::{Duration, SystemTime};

use super::app::{App, Effect, Severity};
use super::project_edit::first_of;
use super::tasks::{Checked, Reread, Task, TaskId};
use crate::config::{CONFIG_FILE, ConfigError, DotenvKeys, ReloadError, Stamp};

/// Time between two looks at the files.
pub(super) const CHECK_EVERY: Duration = Duration::from_secs(2);

/// What the look at the configuration files knows.
#[derive(Debug)]
pub(super) struct ConfigWatch {
    /// The keys `.env` set at start.
    dotenv: DotenvKeys,
    /// The stamp of the files as last read, valid or not.
    stamp: Stamp,
    /// When the last look started.
    checked: Option<SystemTime>,
    /// The look running, if any: the next one waits for its end.
    check: Option<TaskId>,
}

impl ConfigWatch {
    /// Watches files read at `stamp`; `dotenv` are the keys `.env` set at
    /// start.
    pub(super) fn new(dotenv: DotenvKeys, stamp: Stamp) -> Self {
        Self {
            dotenv,
            stamp,
            checked: None,
            check: None,
        }
    }

    /// `overbrainer.toml` was just read at `stamp` (a save, the editor, a
    /// reload): the look running, which may have read it before, is ignored.
    /// Only its half of the stamp is taken: a `.env` changed meanwhile is read
    /// by the next look.
    pub(super) fn seen(&mut self, stamp: Stamp) {
        self.stamp = self.stamp.with_config(stamp);
        self.check = None;
    }
}

impl App {
    /// Looks at the files when it is time: not while a look runs, a save
    /// runs, the editor has `overbrainer.toml`, or the TUI is leaving.
    pub(super) fn check_config(&mut self) -> Vec<Effect> {
        let now = self.now;
        let busy =
            self.leaving.is_some() || self.project_view.save.is_some() || self.project_view.editing;
        let due = self.watch.as_ref().is_some_and(|watch| {
            watch.check.is_none()
                && watch.checked.is_none_or(
                    |at| !matches!(now.duration_since(at), Ok(since) if since < CHECK_EVERY),
                )
        });
        if busy || !due {
            return Vec::new();
        }
        let id = self.task_id();
        let Some(watch) = self.watch.as_mut() else {
            return Vec::new();
        };
        watch.checked = Some(now);
        watch.check = Some(id);
        let task = Task::CheckConfig {
            seen: watch.stamp,
            dotenv: watch.dotenv.clone(),
        };
        vec![Effect::Spawn(id, task)]
    }

    /// Look `id` ended with `checked`: a new valid configuration is applied, an
    /// invalid one is said and the previous one kept. Ignored when it is not
    /// the look running, or found nothing new.
    pub(super) fn config_checked(&mut self, id: TaskId, checked: Checked) -> Vec<Effect> {
        let Some(watch) = self.watch.as_mut().filter(|watch| watch.check == Some(id)) else {
            return Vec::new();
        };
        watch.check = None;
        if checked.stamp == watch.stamp {
            return Vec::new();
        }
        watch.stamp = checked.stamp;
        match checked.read {
            None => Vec::new(),
            Some(Ok(reread)) => self.reloaded(reread),
            Some(Err(error)) => {
                self.reload_refused(&error);
                Vec::new()
            },
        }
    }

    /// Whether `id` is the look running, which failed: the next tick looks
    /// again.
    pub(super) fn check_failed(&mut self, id: TaskId) -> bool {
        match self.watch.as_mut() {
            Some(watch) if watch.check == Some(id) => {
                watch.check = None;
                true
            },
            _ => false,
        }
    }

    /// Applies the configuration read again, as a save does; `u` then has
    /// nothing to undo.
    fn reloaded(&mut self, reread: Reread) -> Vec<Effect> {
        let Reread { config, env } = reread;
        self.env = env;
        let used = self.adopt(config);
        self.project_view.undo = None;
        self.say(Severity::Info, "✓ config reloaded");
        let mut effects = vec![used];
        effects.extend(self.reload());
        effects
    }

    /// Says why the files read again cannot be used; the Project view marks
    /// the fields the problems name.
    fn reload_refused(&mut self, error: &ReloadError) {
        let said = match error {
            ReloadError::Config(ConfigError::Invalid(problems)) => {
                self.note_errors(problems);
                format!("✗ {CONFIG_FILE}: {}", first_of(problems))
            },
            // A syntax error already names the file.
            ReloadError::Config(ConfigError::Parse(problem)) if problem.contains(CONFIG_FILE) => {
                format!("✗ {problem}")
            },
            ReloadError::Config(ConfigError::Parse(problem)) => {
                format!("✗ {CONFIG_FILE}: {problem}")
            },
            ReloadError::Config(error @ ConfigError::Read { .. }) => format!("✗ {error}"),
            ReloadError::Dotenv(error) => format!("✗ {error}"),
        };
        self.say(Severity::Error, said);
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::process::ExitStatusExt as _;
    use std::path::Path;

    use crossterm::event::KeyCode;

    use super::*;
    use crate::config::{DOTENV_FILE, EnvSource, Source, stamp};
    use crate::tui::app::View;
    use crate::tui::project::{ProjectConfig, Undo};
    use crate::tui::project_edit::save_config;
    use crate::tui::snapshots::{
        NOW, PROJECT_CONFIG, SECRET, at, draw, key, project_app, project_env, text,
    };
    use crate::tui::tasks::{Done, Tasks, check_config};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// The Project view on [`PROJECT_CONFIG`], written in a project directory
    /// whose files are watched.
    fn watched() -> Result<(tempfile::TempDir, App), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        fs::write(dir.path().join(CONFIG_FILE), PROJECT_CONFIG)?;
        let mut app = project_app()?;
        app.project.dir = dir.path().to_path_buf();
        app.env = project_env();
        app.now = at(NOW);
        app.watch = Some(ConfigWatch::new(DotenvKeys::default(), stamp(dir.path())));
        Ok((dir, app))
    }

    /// The look the tick at `seconds` starts: its ID and the stamp it knows.
    fn tick(app: &mut App, seconds: u64) -> Result<(TaskId, Stamp), String> {
        let effects = app.on_tick(at(seconds));
        match effects.as_slice() {
            [Effect::Spawn(id, Task::CheckConfig { seen, .. })] => Ok((*id, *seen)),
            _ => Err(format!("no look: {effects:?}")),
        }
    }

    /// What a look finds in `dir` once `text` replaced `overbrainer.toml`,
    /// read with the environment of the fixtures.
    fn rewritten(dir: &Path, text: &str) -> Result<Checked, Box<dyn std::error::Error>> {
        rewritten_with(dir, text, &project_env())
    }

    /// [`rewritten`], read with `env`.
    fn rewritten_with(
        dir: &Path,
        text: &str,
        env: &EnvSource,
    ) -> Result<Checked, Box<dyn std::error::Error>> {
        fs::write(dir.join(CONFIG_FILE), text)?;
        let now = stamp(dir);
        let read = ProjectConfig::new(text, env)
            .map(|mut config| {
                config.stamp = Some(now);
                Reread {
                    config,
                    env: env.clone(),
                }
            })
            .map_err(ReloadError::from);
        Ok(Checked {
            stamp: now,
            read: Some(read),
        })
    }

    fn status(app: &App) -> &str {
        app.status
            .as_ref()
            .map_or("", |status| status.text.as_str())
    }

    #[test]
    fn a_changed_file_is_read_once_and_applied() -> TestResult {
        let (dir, mut app) = watched()?;
        let (id, _) = tick(&mut app, NOW)?;
        assert_eq!(app.on_tick(at(NOW + 1)), [], "one look at a time");
        let renamed = PROJECT_CONFIG.replace("rust_expert", "rust_pro");
        let effects = app.on_done(
            id,
            Ok(Done::ConfigChecked(Box::new(rewritten(
                dir.path(),
                &renamed,
            )?))),
        );
        assert!(
            matches!(
                effects.as_slice(),
                [Effect::UseConfig(Source { text: Some(text), env }), Effect::Spawn(_, Task::Load)]
                    if *text == renamed && *env == project_env()
            ),
            "{effects:?}"
        );
        assert_eq!(status(&app), "✓ config reloaded");
        assert_eq!(app.project.name, "rust_pro");
        assert_eq!(
            app.config.as_ref().map(|config| config.text.as_str()),
            Some(renamed.as_str())
        );
        // The next look knows the new stamp: nothing to read again.
        let (_, seen) = tick(&mut app, NOW + 3)?;
        assert!(
            check_config(dir.path(), seen, &DotenvKeys::default())
                .read
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn an_invalid_file_keeps_the_settings_and_marks_the_field() -> TestResult {
        let (dir, mut app) = watched()?;
        let (id, _) = tick(&mut app, NOW)?;
        let invalid = PROJECT_CONFIG.replace("subtopics = 2", "subtopics = 0");
        app.on_done(
            id,
            Ok(Done::ConfigChecked(Box::new(rewritten(
                dir.path(),
                &invalid,
            )?))),
        );
        assert!(
            status(&app).starts_with("✗ overbrainer.toml: "),
            "{}",
            status(&app)
        );
        assert_eq!(
            app.config.as_ref().map(|config| config.text.as_str()),
            Some(PROJECT_CONFIG)
        );
        assert_eq!(app.project.name, "rust_expert");
        assert!(
            app.project_view
                .errors
                .keys()
                .any(|key| key.starts_with("topics.ownership")),
            "{:?}",
            app.project_view.errors
        );
        let screen = text(&draw(&mut app, 120, 40)?).join("\n");
        assert!(screen.contains("✗ overbrainer.toml"), "{screen}");
        assert!(!screen.contains(SECRET));
        // The same invalid file is not said again.
        let (id, _) = tick(&mut app, NOW + 2)?;
        let again = Checked {
            stamp: stamp(dir.path()),
            read: None,
        };
        assert_eq!(
            app.on_done(id, Ok(Done::ConfigChecked(Box::new(again)))),
            []
        );
        Ok(())
    }

    #[tokio::test]
    async fn after_a_refused_reload_the_next_task_runs_on_the_kept_configuration() -> TestResult {
        let (dir, mut app) = watched()?;
        let mut tasks = Tasks::new(dir.path(), tokio::sync::mpsc::unbounded_channel().0);
        let mut apply = |effects: Vec<Effect>| {
            for effect in effects {
                if let Effect::UseConfig(source) = effect {
                    tasks.use_config(source);
                }
            }
        };
        apply(app.start());
        let (id, _) = tick(&mut app, NOW)?;
        let invalid = PROJECT_CONFIG.replace("subtopics = 2", "subtopics = 0");
        let checked = rewritten(dir.path(), &invalid)?;
        apply(app.on_done(id, Ok(Done::ConfigChecked(Box::new(checked)))));
        assert!(status(&app).starts_with('✗'), "{}", status(&app));
        tasks.spawn(TaskId(500), Task::Prepare);
        let next = tokio::time::timeout(std::time::Duration::from_secs(30), tasks.next()).await?;
        let Some((TaskId(500), Ok(Done::Prepared(Ok(plan))))) = next else {
            return Err(format!("not run on the kept configuration: {next:?}").into());
        };
        assert_eq!(plan.target, "gpu_cloud");
        Ok(())
    }

    #[test]
    fn a_broken_env_file_is_said_by_its_line_only() -> TestResult {
        let (dir, mut app) = watched()?;
        let (id, seen) = tick(&mut app, NOW)?;
        fs::write(
            dir.path().join(DOTENV_FILE),
            format!("OVERBRAINER_PIPELINE__CONCURRENCY=8\nBROKEN {SECRET}\n"),
        )?;
        let checked = check_config(dir.path(), seen, &DotenvKeys::default());
        app.on_done(id, Ok(Done::ConfigChecked(Box::new(checked))));
        assert_eq!(status(&app), "✗ cannot parse .env (syntax error at line 2)");
        let screen = text(&draw(&mut app, 120, 40)?).join("\n");
        assert!(!screen.contains(SECRET));
        assert_eq!(app.env, project_env(), "the previous environment is kept");
        Ok(())
    }

    #[test]
    fn a_running_stage_is_left_alone_and_the_next_tasks_get_the_new_env() -> TestResult {
        let (dir, mut app) = watched()?;
        app.pipeline_task = Some(TaskId(90));
        let (id, _) = tick(&mut app, NOW)?;
        let changed = PROJECT_CONFIG.replace("concurrency = 16", "concurrency = 12");
        let EnvSource::Vars(mut vars) = project_env() else {
            return Err("the fixtures' env is a list".into());
        };
        vars.push(("OVERBRAINER_PIPELINE__SEED".into(), "99".into()));
        let env = EnvSource::Vars(vars);
        let effects = app.on_done(
            id,
            Ok(Done::ConfigChecked(Box::new(rewritten_with(
                dir.path(),
                &changed,
                &env,
            )?))),
        );
        assert!(
            !effects
                .iter()
                .any(|effect| matches!(effect, Effect::Cancel(_) | Effect::Abandon(_)))
        );
        let used = Source {
            text: Some(changed),
            env: env.clone(),
        };
        assert!(effects.contains(&Effect::UseConfig(used)), "{effects:?}");
        assert_eq!(app.env, env);
        assert_eq!(app.pipeline_task, Some(TaskId(90)));
        Ok(())
    }

    #[test]
    fn a_reload_leaves_nothing_to_undo() -> TestResult {
        let (dir, mut app) = watched()?;
        app.project_view.undo = Some(Undo {
            before: PROJECT_CONFIG.replace("rust_expert", "rust_old"),
            after: PROJECT_CONFIG.to_string(),
        });
        let (id, _) = tick(&mut app, NOW)?;
        let renamed = PROJECT_CONFIG.replace("rust_expert", "rust_pro");
        app.on_done(
            id,
            Ok(Done::ConfigChecked(Box::new(rewritten(
                dir.path(),
                &renamed,
            )?))),
        );
        assert_eq!(status(&app), "✓ config reloaded");
        assert_eq!(app.project_view.undo, None);
        app.view = View::Project;
        assert_eq!(app.on_input(&key(KeyCode::Char('u'))), []);
        assert_eq!(status(&app), "nothing to undo");
        assert_eq!(fs::read_to_string(dir.path().join(CONFIG_FILE))?, renamed);
        Ok(())
    }

    #[test]
    fn a_save_takes_the_new_stamp_and_is_not_read_again() -> TestResult {
        let (dir, mut app) = watched()?;
        let (running, _) = tick(&mut app, NOW)?;
        let saved = save_config(
            dir.path(),
            &PROJECT_CONFIG.replace("rust_expert", "rust_pro"),
            PROJECT_CONFIG,
            &project_env(),
        )
        .map_err(|refusal| format!("{refusal:?}"))?;
        let id = app.task_id();
        app.project_view.save = Some(id);
        app.on_done(id, Ok(Done::ConfigSaved(Ok(Box::new(saved)))));
        assert_eq!(status(&app), "✓ saved overbrainer.toml");
        // The look started before the save is ignored.
        let stale = Checked {
            stamp: stamp(dir.path()),
            read: None,
        };
        assert_eq!(
            app.on_done(running, Ok(Done::ConfigChecked(Box::new(stale)))),
            []
        );
        let (_, seen) = tick(&mut app, NOW + 2)?;
        assert!(
            check_config(dir.path(), seen, &DotenvKeys::default())
                .read
                .is_none()
        );
        assert_eq!(status(&app), "✓ saved overbrainer.toml");
        Ok(())
    }

    #[test]
    fn an_env_file_changed_during_a_save_is_still_read() -> TestResult {
        let (dir, mut app) = watched()?;
        fs::write(
            dir.path().join(DOTENV_FILE),
            "OVERBRAINER_PIPELINE__SEED=7\n",
        )?;
        let saved = save_config(
            dir.path(),
            &PROJECT_CONFIG.replace("rust_expert", "rust_pro"),
            PROJECT_CONFIG,
            &project_env(),
        )
        .map_err(|refusal| format!("{refusal:?}"))?;
        let id = app.task_id();
        app.project_view.save = Some(id);
        app.on_done(id, Ok(Done::ConfigSaved(Ok(Box::new(saved)))));
        let (_, seen) = tick(&mut app, NOW)?;
        let checked = check_config(dir.path(), seen, &DotenvKeys::default());
        assert!(
            matches!(&checked.read, Some(Ok(reread)) if reread.config.settings.pipeline.seed == 7),
            "{:?}",
            checked.read.as_ref().map(Result::is_ok)
        );
        Ok(())
    }

    #[test]
    fn an_invalid_file_left_by_the_editor_is_said_once() -> TestResult {
        let (dir, mut app) = watched()?;
        app.project_view.editing = true;
        fs::write(
            dir.path().join(CONFIG_FILE),
            PROJECT_CONFIG.replace("subtopics = 2", "subtopics = 0"),
        )?;
        let exited = std::process::ExitStatus::from_raw(0);
        assert_eq!(app.on_editor_exit(Ok(exited)), []);
        assert!(status(&app).contains("subtopics"), "{}", status(&app));
        assert!(
            app.project_view
                .errors
                .keys()
                .any(|key| key.starts_with("topics.ownership")),
            "{:?}",
            app.project_view.errors
        );
        let (_, seen) = tick(&mut app, NOW)?;
        assert!(
            check_config(dir.path(), seen, &DotenvKeys::default())
                .read
                .is_none(),
            "not read again"
        );
        Ok(())
    }

    #[test]
    fn a_toml_syntax_error_is_said_without_the_file_name_twice() -> TestResult {
        let (dir, mut app) = watched()?;
        let (id, _) = tick(&mut app, NOW)?;
        let checked = rewritten(dir.path(), "[project\nname = 1\n")?;
        app.on_done(id, Ok(Done::ConfigChecked(Box::new(checked))));
        assert_eq!(
            status(&app).matches(CONFIG_FILE).count(),
            1,
            "{}",
            status(&app)
        );
        assert!(status(&app).starts_with("✗ TOML syntax error in overbrainer.toml"));
        Ok(())
    }

    #[test]
    fn no_look_while_a_save_runs_or_the_editor_has_the_file() -> TestResult {
        let (_dir, mut app) = watched()?;
        app.project_view.save = Some(TaskId(50));
        assert_eq!(app.on_tick(at(NOW)), []);
        app.project_view.save = None;
        app.project_view.editing = true;
        assert_eq!(app.on_tick(at(NOW + 5)), []);
        app.project_view.editing = false;
        tick(&mut app, NOW + 6)?;
        Ok(())
    }
}
