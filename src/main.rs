//! Command line entry point.

use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

use clap::{CommandFactory, Parser};
use clap_complete::env::{Bash, CompleteEnv, Fish, Shells, Zsh};
use overbrainer::cli::{self, Cli};
use overbrainer::config::{DOTENV_FILE, DotenvKeys};

fn main() -> ExitCode {
    // Answer a shell's completion request (`COMPLETE=<shell>`) and exit, before
    // anything else: completion needs neither .env nor logs, and nothing else may
    // write to stdout. Without `COMPLETE` this returns at once.
    CompleteEnv::with_factory(Cli::command)
        .shells(Shells(&[&Bash, &Zsh, &Fish]))
        .complete();

    let cli = Cli::parse();

    // Load .env before any thread exists: dotenvy writes to the process environment.
    // A missing .env is normal; any other error is reported. `skill` needs no
    // configuration, so a broken .env does not stop it.
    // The keys it sets are recorded first: a reload of the configuration
    // replaces them with what .env holds then.
    let needs_env = !matches!(cli.command, cli::Command::Skill { .. });
    let dotenv = if needs_env {
        let path = cli.project_dir.join(DOTENV_FILE);
        let loaded = DotenvKeys::record(&path, process_vars())
            .map_err(|error| error.to_string())
            .and_then(|keys| load_dotenv(&path).map(|()| keys));
        match loaded {
            Ok(keys) => keys,
            Err(message) => {
                eprintln!("error: {message}");
                return ExitCode::FAILURE;
            },
        }
    } else {
        DotenvKeys::default()
    };

    let logs = cli.command.log_mode();
    overbrainer::logging::init(&logs);

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("error: cannot start async runtime: {e}");
            return ExitCode::FAILURE;
        },
    };

    let result = runtime.block_on(cli::run(cli, logs, dotenv));
    shut_down(runtime);
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        },
    }
}

/// How long the exit waits for blocking work still running.
const SHUTDOWN_GRACE: Duration = Duration::from_millis(500);

/// Drops `runtime`, waiting at most [`SHUTDOWN_GRACE`] for its blocking tasks.
/// A plain drop waits for them without limit, and the update check's DNS lookup
/// runs in one that aborting its task does not stop: a slow resolver would hold
/// every exit. The grace still lets short work, such as a history write, end.
fn shut_down(runtime: tokio::runtime::Runtime) {
    runtime.shutdown_timeout(SHUTDOWN_GRACE);
}

/// The names of the process environment, before .env is loaded; values that
/// are not UTF-8 are kept lossily, only the names count.
fn process_vars() -> impl Iterator<Item = (String, String)> {
    std::env::vars_os().filter_map(|(key, value)| {
        Some((
            key.into_string().ok()?,
            value.to_string_lossy().into_owned(),
        ))
    })
}

/// Loads `path` into the process environment. A missing file is not an error.
///
/// A syntax error was already reported, with its line, by [`DotenvKeys::record`],
/// which parses the file first. The returned message never contains file content:
/// dotenvy's parse error displays the offending text, which usually holds a
/// secret. I/O errors carry no file content and are shown as is.
fn load_dotenv(path: &Path) -> Result<(), String> {
    match dotenvy::from_path(path) {
        Ok(()) => Ok(()),
        Err(dotenvy::Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(dotenvy::Error::Io(e)) => Err(format!("cannot load .env: {e}")),
        Err(_) => Err("cannot load .env".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    #[test]
    fn a_stuck_blocking_task_holds_the_exit_at_most_the_grace() -> std::io::Result<()> {
        let runtime = tokio::runtime::Builder::new_multi_thread().build()?;
        let (started, running) = std::sync::mpsc::channel();
        runtime.spawn_blocking(move || {
            started.send(()).ok();
            std::thread::sleep(Duration::from_secs(10));
        });
        running
            .recv_timeout(Duration::from_secs(5))
            .map_err(std::io::Error::other)?;
        let start = Instant::now();
        shut_down(runtime);
        let took = start.elapsed();
        assert!(
            took >= SHUTDOWN_GRACE && took < SHUTDOWN_GRACE + Duration::from_secs(3),
            "{took:?}"
        );
        Ok(())
    }
}
