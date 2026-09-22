//! Command line interface.

mod config_check;
mod data;
mod init;
mod progress;

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use secrecy::SecretString;
use tokio::sync::OnceCell;

use crate::secrets::{Resolver, SecretError, SecretSource, VaultRef, VaultSettings, VaultSource};

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
    /// Create an example project (overbrainer.toml, .env.example, prompts/, .gitignore).
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
    /// Generate the subtopics of each topic into data/subtopics.jsonl.
    Subtopics(StageArgs),
    /// Generate questions for each subtopic into data/questions.jsonl.
    ///
    /// Topics without subtopics get them first, as `overbrainer subtopics` would.
    Questions(StageArgs),
    /// Ask the parent model to answer each question into data/answers.jsonl.
    ///
    /// Resuming skips every question that already has an answer, including answers
    /// excluded from training (truncated, refused, empty, no raw reasoning). Only
    /// `--force` asks the parent again for them.
    Answers(StageArgs),
    /// Split usable answers into data/train.jsonl and data/eval.jsonl.
    ///
    /// Both files are always rewritten from scratch with the usable examples of every
    /// topic: `--topic` only limits the counts printed. Answers whose topic is no longer
    /// in overbrainer.toml, or whose question is no longer in data/questions.jsonl (for
    /// example after the questions were regenerated), are left out and counted as
    /// orphaned; they stay in data/answers.jsonl.
    Split(SplitArgs),
    /// Run subtopics, questions, answers and split in order.
    Run,
}

/// Options shared by the pipeline stage commands.
#[derive(Debug, Default, Args)]
pub struct StageArgs {
    /// Only process this topic.
    #[arg(long)]
    pub topic: Option<String>,
    /// Regenerate this stage's output for the selected topics instead of resuming.
    #[arg(long)]
    pub force: bool,
}

/// Options of `overbrainer split`.
#[derive(Debug, Default, Args)]
pub struct SplitArgs {
    /// Only count this topic in the printed report. Both files still hold every topic.
    #[arg(long)]
    pub topic: Option<String>,
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
    let dir = &cli.project_dir;
    match cli.command {
        Command::Init { dir: target } => init::run(target.as_deref().unwrap_or(dir)),
        Command::Config {
            command: ConfigCommand::Check { resolve },
        } => config_check::run(dir, resolve).await,
        Command::Subtopics(args) => data::run(dir, data::Command::Subtopics, &args).await,
        Command::Questions(args) => data::run(dir, data::Command::Questions, &args).await,
        Command::Answers(args) => data::run(dir, data::Command::Answers, &args).await,
        Command::Split(SplitArgs { topic }) => {
            let args = StageArgs {
                topic,
                force: false,
            };
            data::run(dir, data::Command::Split, &args).await
        },
        Command::Run => data::run(dir, data::Command::Run, &StageArgs::default()).await,
    }
}

/// Secret resolver from `VAULT_ADDR`, `VAULT_TOKEN` and `~/.vault-token`. Vault is
/// only set up when a `vault:` reference is first resolved, so literal secrets work
/// even when Vault settings are incomplete. Without `VAULT_ADDR`, references fail
/// when used.
fn resolver() -> Resolver<LazyVault> {
    Resolver::new(Some(LazyVault::default()))
}

/// A [`VaultSource`] built from the environment on the first fetch.
#[derive(Default)]
struct LazyVault {
    source: OnceCell<Option<VaultSource>>,
}

impl LazyVault {
    async fn source(&self) -> Result<Option<&VaultSource>, SecretError> {
        let source = self.source.get_or_try_init(|| async { vault() }).await?;
        Ok(source.as_ref())
    }
}

impl SecretSource for LazyVault {
    async fn fetch(&self, reference: &VaultRef) -> Result<SecretString, SecretError> {
        match self.source().await? {
            Some(source) => source.fetch(reference).await,
            None => Err(SecretError::VaultNotConfigured),
        }
    }
}

fn vault() -> Result<Option<VaultSource>, SecretError> {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    VaultSettings::from_env(|key| std::env::var(key).ok(), home.as_deref())?
        .map(|settings| VaultSource::new(&settings))
        .transpose()
}
