//! The terminal UI, `overbrainer tui`: the configuration and stats of the
//! project, the dataset, the pipeline stages, training runs and the logs in five
//! views. It runs the flows the command line runs, and writes nothing to stdout
//! or stderr while the terminal shows it.

mod app;
mod auto;
mod catalog;
mod config_watch;
mod cost;
mod dataset;
mod editor;
mod event_loop;
mod follow;
mod format;
mod keys;
mod motion;
mod pipeline;
mod project;
mod project_edit;
mod project_save;
#[cfg(test)]
mod snapshots;
mod start;
mod start_pick;
mod tasks;
mod terminal;
mod theme;
mod training;
mod ui;
mod views;
mod widgets;
mod wizard;

use std::io::{self, IsTerminal, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::{Context, bail};
use tokio::task::JoinHandle;

use self::app::{App, Project};
use self::config_watch::ConfigWatch;
use self::motion::{Motion, MotionLevel};
use self::project::ProjectConfig;
use self::terminal::TerminalGuard;
use self::theme::{ColorLevel, LookEnv, Theme};
use crate::config::{CONFIG_FILE, ConfigError, DotenvKeys, EnvSource, stamp};
use crate::events::Observer;
use crate::logging::LogBuffer;
use crate::update::Newer;

pub use self::wizard::Ended as WizardEnded;

/// How [`run`] starts the TUI.
#[derive(Debug, Default)]
pub struct Start {
    /// The keys `.env` set at start, which a reload replaces.
    pub dotenv: DotenvKeys,
    /// Whether auto mode's confirmation opens at once: the init wizard was
    /// answered yes.
    pub auto: bool,
}

/// Runs the init wizard for the project in `project_dir`, which has no
/// `overbrainer.toml`: it asks what the project needs, screen by screen, then
/// writes it. It starts no thread, so `.env` can be loaded after it.
///
/// # Errors
///
/// Returns an error when stdout is not a terminal, or the terminal cannot be
/// set up, read or drawn on.
pub fn wizard(project_dir: &Path) -> anyhow::Result<WizardEnded> {
    if !std::io::stdout().is_terminal() {
        bail!("overbrainer tui needs a terminal: stdout is not a TTY");
    }
    wizard::run(project_dir)
}

/// Why the init wizard cannot run in `project_dir`, when a file it would
/// write exists there: it never overwrites one, so it says so before asking
/// anything.
#[must_use]
pub fn wizard_refusal(project_dir: &Path) -> Option<String> {
    wizard::existing(project_dir).map(|path| {
        format!(
            "{} exists, and the init wizard never overwrites a file: move it away and run \
             `overbrainer tui` again, or write the project with `overbrainer init`",
            path.display()
        )
    })
}

/// Runs the terminal UI on the project in `project_dir`; `logs` holds the log
/// lines the Logs view shows; the answer of `check`, when newer, shows in the
/// footer; `start` holds the keys `.env` set at start and whether auto
/// mode's confirmation opens at once; `observer` sees the events of the
/// stages and trainings it runs, auto mode's included.
///
/// # Errors
///
/// Returns an error when stdout is not a terminal, `overbrainer.toml` cannot be
/// loaded, the terminal cannot be set up, or a process signal ended the TUI.
pub async fn run(
    project_dir: &Path,
    logs: LogBuffer,
    check: Option<JoinHandle<Option<Newer>>>,
    start: Start,
    observer: Option<Arc<dyn Observer>>,
) -> anyhow::Result<()> {
    if !std::io::stdout().is_terminal() {
        bail!("overbrainer tui needs a terminal: stdout is not a TTY");
    }
    // The Project view shows this text and the environment on it; the files are
    // read again once their stamp, taken first, changes.
    let path = project_dir.join(CONFIG_FILE);
    let read_at = stamp(project_dir);
    let text = std::fs::read_to_string(&path).map_err(|error| ConfigError::Read { path, error })?;
    let mut config = ProjectConfig::new(&text, &EnvSource::Process)?;
    config.stamp = Some(read_at);
    let project = Project::new(project_dir, &config.settings);
    let env = LookEnv::from_process();
    let color = ColorLevel::detect(&env);
    let theme = Theme::new(color);
    let mut app = App::new(project, logs, &theme, SystemTime::now());
    app.set_config(config);
    app.watch = Some(ConfigWatch::new(start.dotenv, read_at));
    if start.auto {
        app.opening = app::Opening::Auto;
    }
    app.motion = Motion::new(MotionLevel::detect(&env, color)).colored(&theme);
    // Logged after `App::new`, so the status line shows them on start.
    warn_on_start(&env, project_dir);
    app.editor = editor::command(std::env::var_os("VISUAL"), std::env::var_os("EDITOR"));
    let guard = TerminalGuard::enter();
    let mut terminal = terminal::init().context("cannot set up the terminal")?;
    let result = event_loop::run(&mut terminal, &mut app, check, observer).await;
    drop(guard);
    app.abandon_edit();
    for note in &app.exit_notes {
        writeln!(io::stderr(), "{note}").ok();
    }
    result
}

/// Logs what the user should know on start: the look settings ignored, and
/// the need to run `overbrainer migrate` on a project from before 0.4.0.
fn warn_on_start(env: &LookEnv, project_dir: &Path) {
    let mut warnings = env.warnings();
    if crate::project_format::predates_versions(project_dir) {
        warnings.push(crate::cli::migrate::HINT.to_string());
    }
    for warning in warnings {
        tracing::warn!("{warning}");
    }
}
