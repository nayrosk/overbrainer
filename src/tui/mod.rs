//! The terminal UI, `overbrainer tui`: the dataset, the pipeline stages, training
//! runs and the logs in four views. It runs the flows the command line runs, and
//! writes nothing to stdout or stderr while the terminal shows it.

mod app;
mod dataset;
mod editor;
mod event_loop;
mod follow;
mod format;
mod keys;
mod pipeline;
#[cfg(test)]
mod snapshots;
mod start;
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

use self::app::{App, Project};
use self::terminal::TerminalGuard;
use self::theme::{ColorLevel, LookEnv, Theme};
use crate::config::EnvSource;
use crate::logging::LogBuffer;

/// Runs the terminal UI on the project in `project_dir`; `logs` holds the log
/// lines the Logs view shows.
///
/// # Errors
///
/// Returns an error when stdout is not a terminal, `overbrainer.toml` cannot be
/// loaded, the terminal cannot be set up, or a process signal ended the TUI.
pub async fn run(project_dir: &Path, logs: LogBuffer) -> anyhow::Result<()> {
    if !std::io::stdout().is_terminal() {
        bail!("overbrainer tui needs a terminal: stdout is not a TTY");
    }
    let settings = crate::config::load(project_dir, EnvSource::Process)?;
    let project = Project::new(project_dir, &settings);
    let env = LookEnv::from_process();
    let theme = Theme::new(ColorLevel::detect(&env));
    let mut app = App::new(project, logs, &theme, SystemTime::now());
    for warning in env.warnings() {
        tracing::warn!("{warning}");
    }
    app.editor = editor::command(std::env::var_os("VISUAL"), std::env::var_os("EDITOR"));
    let guard = TerminalGuard::enter();
    let mut terminal = terminal::init().context("cannot set up the terminal")?;
    let result = event_loop::run(&mut terminal, &mut app).await;
    drop(guard);
    app.abandon_edit();
    for note in &app.exit_notes {
        writeln!(io::stderr(), "{note}").ok();
    }
    result
}
