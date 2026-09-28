//! Command line interface.

mod complete;
mod config_check;
pub(crate) mod data;
pub(crate) mod front;
mod history;
mod init;
pub(crate) mod pod;
mod progress;
mod record;
mod runpod_train;
mod skill;
pub(crate) mod train;

use std::io::IsTerminal as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use clap::{Args, Parser, Subcommand, ValueHint};
use clap_complete::engine::ArgValueCandidates;
use secrecy::SecretString;
use tokio::sync::OnceCell;
use tokio::task::JoinHandle;

use self::front::Frontend;
use crate::logging::{LOG_LINES, LogBuffer, LogMode};
use crate::secrets::{Resolver, SecretError, SecretSource, VaultRef, VaultSettings, VaultSource};
use crate::update::{self, CheckEnv, Newer};

/// How long a command that ended waits for the update check still running.
const CHECK_GRACE: Duration = Duration::from_millis(500);

/// The `overbrainer` command line interface.
#[derive(Debug, Parser)]
#[command(name = "overbrainer", version, about)]
pub struct Cli {
    /// Project directory containing overbrainer.toml.
    #[arg(short = 'C', long, global = true, default_value = ".", value_hint = ValueHint::DirPath)]
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
        #[arg(value_hint = ValueHint::DirPath)]
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
    /// overbrainer.toml has a `[training]` section.
    Run,
    /// Fine-tune the child model on data/train.jsonl, evaluating on data/eval.jsonl.
    ///
    /// The job runs detached on the target and writes to `runs/<run-id>/`. Ctrl-C stops
    /// following it but leaves it running: `overbrainer train attach <run-id>` follows
    /// it again, `overbrainer train cancel <run-id>` stops it.
    Train(TrainArgs),
    /// Inspect training runs.
    Runs {
        /// The runs subcommand to run.
        #[command(subcommand)]
        command: RunsCommand,
    },
    /// Show what the pipeline stages did and spent, from .overbrainer/history.jsonl.
    History(HistoryArgs),
    /// Find and remove the Runpod pods overbrainer created.
    Pod {
        /// The pod subcommand to run.
        #[command(subcommand)]
        command: PodCommand,
    },
    /// Install the agent skill that teaches AI coding agents to drive overbrainer.
    Skill {
        /// The skill subcommand to run.
        #[command(subcommand)]
        command: SkillCommand,
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

    /// Whether this command writes to the project, and so must hold the project
    /// lock: at most one such overbrainer process per project.
    #[must_use]
    pub fn writes_project(&self) -> bool {
        match self {
            Self::Tui
            | Self::Subtopics(_)
            | Self::Questions(_)
            | Self::Answers(_)
            | Self::Split(_)
            | Self::Run
            | Self::Train(_) => true,
            Self::Pod { command } => matches!(command, PodCommand::Rm { .. }),
            Self::Init { .. }
            | Self::Config { .. }
            | Self::Runs { .. }
            | Self::History(_)
            | Self::Skill { .. } => false,
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
        #[arg(add = ArgValueCandidates::new(complete::run_ids))]
        run_id: String,
        /// Delete even when the run's job is still running (the run is then failed).
        #[arg(long)]
        force: bool,
    },
}

/// Subcommands of `overbrainer skill`.
#[derive(Debug, Subcommand)]
pub enum SkillCommand {
    /// Write the skill to `<project>/.claude/skills/overbrainer/SKILL.md`, or with
    /// --global to ~/.claude/skills, or with --dir to another agent's skills directory.
    Install(SkillInstallArgs),
}

/// Options of `overbrainer skill install`.
#[derive(Debug, Args)]
pub struct SkillInstallArgs {
    /// Install for every project, in ~/.claude/skills.
    #[arg(long, conflicts_with = "dir")]
    pub global: bool,
    /// Install into this skills directory instead.
    #[arg(long, value_hint = ValueHint::DirPath)]
    pub dir: Option<PathBuf>,
    /// Replace an installed skill that differs from this version's.
    #[arg(long)]
    pub force: bool,
}

/// Options and subcommands of `overbrainer train`.
#[derive(Debug, Default, Args)]
#[command(args_conflicts_with_subcommands = true)]
pub struct TrainArgs {
    /// Train on this target instead of training.target.
    #[arg(long, add = ArgValueCandidates::new(complete::targets))]
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
        #[arg(add = ArgValueCandidates::new(complete::run_ids))]
        run_id: String,
    },
    /// Stop the job of a run.
    Cancel {
        /// ID of the run, as shown by `overbrainer runs ls`.
        #[arg(add = ArgValueCandidates::new(complete::run_ids))]
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

/// Options of `overbrainer history`.
#[derive(Debug, Default, Args)]
pub struct HistoryArgs {
    /// List every execution, oldest first, instead of the totals per stage.
    #[arg(long)]
    pub all: bool,
}

/// Options shared by the pipeline stage commands.
#[derive(Debug, Default, Args)]
pub struct StageArgs {
    /// Only process this topic.
    #[arg(long, add = ArgValueCandidates::new(complete::topics))]
    pub topic: Option<String>,
    /// Regenerate this stage's output for the selected topics instead of resuming.
    #[arg(long)]
    pub force: bool,
}

/// Options of `overbrainer split`.
#[derive(Debug, Default, Args)]
pub struct SplitArgs {
    /// Only count this topic in the printed report. Both files still hold every topic.
    #[arg(long, add = ArgValueCandidates::new(complete::topics))]
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
/// [`Command::log_mode`]). Meanwhile it looks for a newer release, told on
/// stderr once the command ended, even with an error (`tui` shows it itself).
///
/// # Errors
///
/// Returns an error if the selected subcommand fails, or if `tui` is not given the
/// [`LogMode::Tui`] its logs need.
pub async fn run(cli: Cli, logs: LogMode) -> anyhow::Result<()> {
    let stderr = std::io::stderr().is_terminal();
    let stdout = std::io::stdout().is_terminal();
    // `skill` never checks: it does not even read the environment.
    let env = (!matches!(cli.command, Command::Skill { .. })).then(CheckEnv::from_process);
    let mut check = env
        .filter(|env| should_check(&cli.command, stderr, stdout, env))
        .map(|env| {
            tokio::spawn(async move {
                update::check(&env, update::CRATES_IO_URL, SystemTime::now()).await
            })
        });
    let result = dispatch(cli, logs, &mut check).await;
    // `tui` took the check, and shows its answer itself.
    if let Some(check) = check
        && let Some(newer) = settle(check).await
    {
        eprintln!("{newer}");
    }
    result
}

/// Whether to look for a newer release beside `command`: not when turned off,
/// not for `skill`, and not when stderr, where the notice goes, is not a
/// terminal, except for `tui`, which shows it on its own screen, and so needs
/// stdout a terminal instead (it refuses to start otherwise).
fn should_check(
    command: &Command,
    stderr_is_terminal: bool,
    stdout_is_terminal: bool,
    env: &CheckEnv,
) -> bool {
    !env.disabled
        && match command {
            Command::Skill { .. } => false,
            Command::Tui => stdout_is_terminal,
            _ => stderr_is_terminal,
        }
}

/// The answer of `check`, waiting for it at most [`CHECK_GRACE`]; a check
/// still running then is dropped.
async fn settle(mut check: JoinHandle<Option<Newer>>) -> Option<Newer> {
    let answer = tokio::time::timeout(CHECK_GRACE, &mut check).await;
    check.abort();
    answer.ok()?.ok()?
}

/// Runs `cli`'s command; `tui` takes `check` to show its answer.
async fn dispatch(
    cli: Cli,
    logs: LogMode,
    check: &mut Option<JoinHandle<Option<Newer>>>,
) -> anyhow::Result<()> {
    let dir = &cli.project_dir;
    // Only a project takes the lock: without `overbrainer.toml` the command fails
    // with its usual error and leaves nothing behind.
    let _lock = if cli.command.writes_project() && dir.join(crate::config::CONFIG_FILE).is_file() {
        Some(crate::project_lock::ProjectLock::acquire(dir)?)
    } else {
        None
    };
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
        Command::History(args) => history::run(dir, &args),
        Command::Pod { command } => pod::run(dir, &command).await,
        Command::Skill { command } => skill::run(dir, &command),
        Command::Tui => match logs {
            LogMode::Tui(buffer) => crate::tui::run(dir, buffer, check.take()).await,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn command(args: &[&str]) -> Result<Command, clap::Error> {
        let cli = Cli::try_parse_from(std::iter::once("overbrainer").chain(args.iter().copied()))?;
        Ok(cli.command)
    }

    #[test]
    fn the_update_check_runs_on_a_terminal_unless_turned_off() -> Result<(), clap::Error> {
        let on = CheckEnv::default();
        let off = CheckEnv {
            disabled: true,
            cache_dir: None,
        };
        // (args, stderr a terminal, stdout a terminal, env, expected)
        for (args, stderr, stdout, env, expected) in [
            (&["run"][..], true, true, &on, true),
            (&["history"], true, false, &on, true),
            (&["run"], false, true, &on, false),
            (&["run"], true, true, &off, false),
            (&["skill", "install"], true, true, &on, false),
            (&["tui"], true, true, &on, true),
            (&["tui"], false, true, &on, true),
            (&["tui"], true, false, &on, false),
            (&["tui"], true, true, &off, false),
        ] {
            assert_eq!(
                should_check(&command(args)?, stderr, stdout, env),
                expected,
                "{args:?}, stderr a terminal: {stderr}, stdout: {stdout}, disabled: {}",
                env.disabled
            );
        }
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn the_notice_waits_at_most_the_grace_period() {
        let check = tokio::spawn(std::future::pending::<Option<Newer>>());
        let start = tokio::time::Instant::now();
        assert_eq!(settle(check).await, None);
        assert_eq!(start.elapsed(), CHECK_GRACE);
    }

    #[tokio::test(start_paused = true)]
    async fn a_finished_check_gives_its_answer() {
        let newer = Newer {
            latest: "9.0.0".into(),
            current: "0.4.0",
        };
        let answer = newer.clone();
        let check = tokio::spawn(async move { Some(answer) });
        assert_eq!(settle(check).await, Some(newer));
    }

    #[test]
    fn only_commands_that_write_take_the_lock() -> Result<(), clap::Error> {
        let writes = |args: &[&str]| -> Result<bool, clap::Error> {
            let cli =
                Cli::try_parse_from(std::iter::once("overbrainer").chain(args.iter().copied()))?;
            Ok(cli.command.writes_project())
        };
        for args in [
            &["tui"][..],
            &["subtopics"],
            &["questions"],
            &["answers"],
            &["split"],
            &["run"],
            &["train"],
            &["train", "attach", "x"],
            &["train", "cancel", "x"],
            &["pod", "rm", "x"],
        ] {
            assert!(writes(args)?, "{args:?} should lock");
        }
        for args in [
            &["init"][..],
            &["config", "check"],
            &["runs", "ls"],
            &["history"],
            &["pod", "ls"],
            &["skill", "install"],
        ] {
            assert!(!writes(args)?, "{args:?} should not lock");
        }
        Ok(())
    }
}
