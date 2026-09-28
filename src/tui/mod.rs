//! The terminal UI, `overbrainer tui`: the configuration and stats of the
//! project, the dataset, the pipeline stages, training runs and the logs in five
//! views. It runs the flows the command line runs, and writes nothing to stdout
//! or stderr while the terminal shows it.

mod app;
mod catalog;
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

use std::io::{self, IsTerminal, Write};
use std::path::Path;
use std::time::SystemTime;

use anyhow::{Context, bail};
use tokio::task::JoinHandle;

use self::app::{App, Project};
use self::motion::{Motion, MotionLevel};
use self::project::ProjectConfig;
use self::terminal::TerminalGuard;
use self::theme::{ColorLevel, LookEnv, Theme};
use crate::config::{CONFIG_FILE, ConfigError, DotenvKeys, EnvSource};
use crate::logging::LogBuffer;
use crate::update::Newer;

/// Runs the terminal UI on the project in `project_dir`; `logs` holds the log
/// lines the Logs view shows; the answer of `check`, when newer, shows in the
/// footer; `dotenv` are the keys `.env` set at start.
///
/// # Errors
///
/// Returns an error when stdout is not a terminal, `overbrainer.toml` cannot be
/// loaded, the terminal cannot be set up, or a process signal ended the TUI.
pub async fn run(
    project_dir: &Path,
    logs: LogBuffer,
    check: Option<JoinHandle<Option<Newer>>>,
    _dotenv: DotenvKeys,
) -> anyhow::Result<()> {
    if !std::io::stdout().is_terminal() {
        bail!("overbrainer tui needs a terminal: stdout is not a TTY");
    }
    // Read once: the Project view shows this text and the environment on it.
    let path = project_dir.join(CONFIG_FILE);
    let text = std::fs::read_to_string(&path).map_err(|error| ConfigError::Read { path, error })?;
    let config = ProjectConfig::new(&text, &EnvSource::Process)?;
    let project = Project::new(project_dir, &config.settings);
    let env = LookEnv::from_process();
    let color = ColorLevel::detect(&env);
    let theme = Theme::new(color);
    let mut app = App::new(project, logs, &theme, SystemTime::now());
    app.set_config(config);
    app.motion = Motion::new(MotionLevel::detect(&env, color)).colored(&theme);
    for warning in env.warnings() {
        tracing::warn!("{warning}");
    }
    app.editor = editor::command(std::env::var_os("VISUAL"), std::env::var_os("EDITOR"));
    let guard = TerminalGuard::enter();
    let mut terminal = terminal::init().context("cannot set up the terminal")?;
    let result = event_loop::run(&mut terminal, &mut app, check).await;
    drop(guard);
    app.abandon_edit();
    if app.project_view.pending.is_some() {
        app.exit_notes
            .push("the pending changes to overbrainer.toml were not saved".to_string());
    }
    for note in &app.exit_notes {
        writeln!(io::stderr(), "{note}").ok();
    }
    result
}
