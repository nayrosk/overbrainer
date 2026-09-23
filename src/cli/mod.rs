//! Command line interface.

mod config_check;
pub(crate) mod data;
pub(crate) mod front;
mod init;
mod pod;
mod progress;
mod runpod_train;
mod train;

use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};
use secrecy::SecretString;
use tokio::sync::OnceCell;

use self::front::Frontend;
use crate::logging::{LOG_LINES, LogBuffer, LogMode};
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
    /// Run subtopics, questions, answers and split in order, then train when
    /// overbrainer.toml has a [training] section.
    Run,
    /// Fine-tune the child model on data/train.jsonl, evaluating on data/eval.jsonl.
    ///
    /// The job runs detached on the target and writes to runs/<run-id>/. Ctrl-C stops
    /// following it but leaves it running: `overbrainer train attach <run-id>` follows
    /// it again, `overbrainer train cancel <run-id>` stops it.
    Train(TrainArgs),
    /// Inspect training runs.
    Runs {
        /// The runs subcommand to run.
        #[command(subcommand)]
        command: RunsCommand,
    },
    /// Find and remove the Runpod pods overbrainer created.
    Pod {
        /// The pod subcommand to run.
        #[command(subcommand)]
        command: PodCommand,
    },
    /// Browse the dataset, run stages and follow training runs in a terminal UI.
    Tui,
}

impl Command {
    /// Where this command's logs go: an in-memory buffer for `tui`, which owns the
    /// terminal, stderr for every other command.
    #[must_use]
    pub fn log_mode(&self) -> LogMode {
        match self {
            Self::Tui => LogMode::Tui(LogBuffer::new(LOG_LINES)),
            _ => LogMode::Stderr,
        }
    }
}

/// Subcommands of `overbrainer pod`.
#[derive(Debug, Subcommand)]
pub enum PodCommand {
    /// List the pods overbrainer created, with their run and what is known of it.
    Ls,
    /// Delete every pod of a run, and wait until Runpod no longer shows them.
    Rm {
        /// ID of the run, as shown by `overbrainer runs ls` or `overbrainer pod ls`.
        run_id: String,
        /// Delete even when the run's job is still running (the run is then failed).
        #[arg(long)]
        force: bool,
    },
}

/// Options and subcommands of `overbrainer train`.
#[derive(Debug, Default, Args)]
#[command(args_conflicts_with_subcommands = true)]
pub struct TrainArgs {
    /// Train on this target instead of training.target.
    #[arg(long)]
    pub target: Option<String>,
    /// Runpod target only: keep the pod once the run ends, with no time limit.
    /// Nothing deletes it then but `overbrainer pod rm <run-id>`.
    #[arg(long)]
    pub keep_pod: bool,
    /// Follow or stop an existing run instead of starting one.
    #[command(subcommand)]
    pub command: Option<TrainCommand>,
}

/// Subcommands of `overbrainer train`.
#[derive(Debug, Subcommand)]
pub enum TrainCommand {
    /// Follow a run again after Ctrl-C or a lost connection, then retrieve its results.
    Attach {
        /// ID of the run, as shown by `overbrainer runs ls`.
        run_id: String,
    },
    /// Stop the job of a run.
    Cancel {
        /// ID of the run, as shown by `overbrainer runs ls`.
        run_id: String,
    },
}

/// Subcommands of `overbrainer runs`.
#[derive(Debug, Subcommand)]
pub enum RunsCommand {
    /// List the runs in runs/, oldest first: ID, state, target, creation time, and
    /// the pod of a Runpod run.
    Ls,
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

/// Runs the parsed command line, whose logs were set up with `logs` (see
/// [`Command::log_mode`]).
///
/// # Errors
///
/// Returns an error if the selected subcommand fails, or if `tui` is not given the
/// [`LogMode::Tui`] its logs need.
pub async fn run(cli: Cli, logs: LogMode) -> anyhow::Result<()> {
    let dir = &cli.project_dir;
    match cli.command {
        Command::Init { dir: target } => init::run(target.as_deref().unwrap_or(dir)),
        Command::Config {
            command: ConfigCommand::Check { resolve },
        } => config_check::run(dir, resolve).await,
        Command::Subtopics(args) => stage(dir, data::Command::Subtopics, &args).await,
        Command::Questions(args) => stage(dir, data::Command::Questions, &args).await,
        Command::Answers(args) => stage(dir, data::Command::Answers, &args).await,
        Command::Split(SplitArgs { topic }) => {
            let args = StageArgs {
                topic,
                force: false,
            };
            stage(dir, data::Command::Split, &args).await
        },
        Command::Run => {
            stage(dir, data::Command::Run, &StageArgs::default()).await?;
            train::after_run(dir, &Frontend::Cli).await
        },
        Command::Train(args) => train::run(dir, &args, &Frontend::Cli).await,
        Command::Runs {
            command: RunsCommand::Ls,
        } => train::list(dir),
        Command::Pod { command } => pod::run(dir, &command).await,
        Command::Tui => match logs {
            LogMode::Tui(buffer) => crate::tui::run(dir, buffer).await,
            LogMode::Stderr => anyhow::bail!("overbrainer tui needs the TUI log mode"),
        },
    }
}

/// Runs a pipeline command on the command line.
async fn stage(dir: &Path, command: data::Command, args: &StageArgs) -> anyhow::Result<()> {
    data::run(dir, command, args, &Frontend::Cli).await
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
