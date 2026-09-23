//! Command line entry point.

use std::path::Path;
use std::process::ExitCode;

use clap::Parser;
use overbrainer::cli::{self, Cli};

fn main() -> ExitCode {
    let cli = Cli::parse();

    // Load .env before any thread exists: dotenvy writes to the process environment.
    // A missing .env is normal; any other error is reported.
    if let Err(message) = load_dotenv(&cli.project_dir.join(".env")) {
        eprintln!("error: {message}");
        return ExitCode::FAILURE;
    }

    overbrainer::logging::init(&overbrainer::logging::LogMode::Stderr);

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

    match runtime.block_on(cli::run(cli)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        },
    }
}

/// Loads `path` into the process environment. A missing file is not an error.
///
/// The returned message never contains file content: dotenvy's parse error displays
/// the whole offending line, which usually holds a secret, so only its position is
/// reported. I/O errors carry no file content and are shown as is.
fn load_dotenv(path: &Path) -> Result<(), String> {
    match dotenvy::from_path(path) {
        Ok(()) => Ok(()),
        Err(dotenvy::Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(dotenvy::Error::LineParse(_, index)) => {
            Err(format!("cannot parse .env (syntax error at index {index})"))
        },
        Err(dotenvy::Error::Io(e)) => Err(format!("cannot load .env: {e}")),
        Err(_) => Err("cannot load .env".to_string()),
    }
}
