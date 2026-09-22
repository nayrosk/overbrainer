//! Command line entry point.

use std::process::ExitCode;

use clap::Parser;
use overbrainer::cli::{self, Cli};

fn main() -> ExitCode {
    let cli = Cli::parse();

    // Load .env before any thread exists: dotenvy writes to the process environment.
    // A missing .env is normal; any other error is reported.
    match dotenvy::from_path(cli.project_dir.join(".env")) {
        Ok(()) => {}
        Err(dotenvy::Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            eprintln!("error: cannot load .env: {e}");
            return ExitCode::FAILURE;
        }
    }

    overbrainer::logging::init();

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("error: cannot start async runtime: {e}");
            return ExitCode::FAILURE;
        }
    };

    match runtime.block_on(cli::run(cli)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
