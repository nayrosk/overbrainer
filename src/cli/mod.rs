//! Command line interface.

mod config_check;
mod init;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// The `overbrainer` command line interface.
#[derive(Debug, Parser)]
#[command(name = "overbrainer", version, about)]
pub struct Cli {
    /// Project directory containing overbrainer.toml.
    #[arg(short = 'C', long, global = true, default_value = ".")]
    pub project_dir: PathBuf,

    /// The subcommand to run.
    #[command(subcommand)]
    pub command: Command,
}

/// Top-level subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Create an example project (overbrainer.toml, .env.example, .gitignore).
    Init {
        /// Target directory. Defaults to the project directory (`-C`).
        dir: Option<PathBuf>,
    },
    /// Inspect the configuration.
    Config {
        /// The configuration subcommand to run.
        #[command(subcommand)]
        command: ConfigCommand,
    },
}

/// Subcommands of `overbrainer config`.
#[derive(Debug, Subcommand)]
pub enum ConfigCommand {
    /// Print the resolved configuration with secrets masked.
    Check {
        /// Also resolve every secret, testing Vault access.
        #[arg(long)]
        resolve: bool,
    },
}

/// Runs the parsed command line.
///
/// # Errors
///
/// Returns an error if the selected subcommand fails.
pub async fn run(cli: Cli) -> anyhow::Result<()> {
    match cli.command {
        Command::Init { dir } => init::run(dir.as_deref().unwrap_or(&cli.project_dir)),
        Command::Config {
            command: ConfigCommand::Check { resolve },
        } => config_check::run(&cli.project_dir, resolve).await,
    }
}
