//! Command line interface.

mod complete;
mod config_check;
pub(crate) mod data;
mod export;
pub(crate) mod front;
mod history;
pub(crate) mod init;
mod logs;
pub(crate) mod migrate;
pub(crate) mod pod;
mod progress;
pub(crate) mod push;
mod record;
mod reload;
mod runpod_train;
mod skill;
pub(crate) mod train;

use std::io::IsTerminal as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use clap::builder::PossibleValuesParser;
use clap::{Args, Parser, Subcommand, ValueHint};
use clap_complete::engine::ArgValueCandidates;
use secrecy::SecretString;
use tokio::sync::OnceCell;
use tokio::task::JoinHandle;

use self::front::Frontend;
use self::reload::Reloader;
use crate::config::{DotenvKeys, EnvSource, QUANTIZE_TYPES, Source};
use crate::events::Observer;
use crate::logging::{LOG_LINES, LogBuffer, LogMode};
use crate::metrics::{Metrics, MetricsServer};
use crate::project_lock::{LockError, ProjectLock};
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

    /// Whether `tui` starts in auto mode: set by the init wizard, never
    /// parsed.
    #[arg(skip)]
    pub auto: bool,
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
    /// Export the model of a finished run to GGUF, with an Ollama Modelfile.
    ///
    /// The export runs on the run's target (on Runpod, a new pod): the adapter
    /// is merged into its base model unless the run merged it, converted with
    /// llama.cpp and quantized, and the GGUF and its Modelfile land in
    /// `runs/<run-id>/output/gguf/`. Ctrl-C cancels it.
    Export(ExportArgs),
    /// Push the model of a finished run to a Hugging Face model repo, with a
    /// generated model card.
    ///
    /// Everything the run left in `runs/<run-id>/output/` goes in one commit,
    /// checkpoints, `debug.log` and Axolotl's README.md aside. The repo is
    /// created private unless `--public` or `[hub] private = false`. Needs
    /// `OVERBRAINER_HF_TOKEN`, a token with write access. Ctrl-C cancels it:
    /// nothing is committed, and a rerun resumes.
    Push(PushArgs),
    /// Inspect training runs.
    Runs {
        /// The runs subcommand to run.
        #[command(subcommand)]
        command: RunsCommand,
    },
    /// Show what the pipeline stages did and spent, from .overbrainer/history.jsonl.
    History(HistoryArgs),
    /// Bring a project from an older overbrainer up to this one's format.
    ///
    /// Adds /.overbrainer/ to .gitignore, rebuilds the answers history from
    /// data/answers.jsonl (tokens per model, cost unknown) unless the history
    /// already has answers, and writes .overbrainer/version. Running it again
    /// changes nothing.
    Migrate(MigrateArgs),
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

    /// Whether this command serves the metrics while it holds the project lock:
    /// every command that writes, but `migrate`, which spends nothing.
    #[must_use]
    pub fn serves_metrics(&self) -> bool {
        self.writes_project() && !matches!(self, Self::Migrate(_))
    }

    /// The run `train stop` asks a snapshot of; `None` for any other command.
    #[must_use]
    pub fn stopped_run(&self) -> Option<&str> {
        match self {
            Self::Train(TrainArgs {
                command: Some(TrainCommand::Stop { run_id }),
                ..
            }) => Some(run_id),
            _ => None,
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
            | Self::Migrate(_)
            | Self::Train(_)
            | Self::Export(_)
            | Self::Push(_) => true,
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
    /// Delete every pod of a run, or of an export, and wait until Runpod no
    /// longer shows them.
    Rm {
        /// ID of the run, as shown by `overbrainer runs ls` or `overbrainer pod
        /// ls`, or of an export (`export_<date>-<time>`).
        #[arg(add = ArgValueCandidates::new(complete::run_ids))]
        run_id: String,
        /// Delete even when the run's job is still running (the run is then failed).
        #[arg(long)]
        force: bool,
    },
    /// List the Runpod catalog's Secure Cloud GPU types, cheapest first.
    Gpus(GpuArgs),
    /// List the Runpod catalog's data centers, by ID.
    Datacenters,
    /// List the account's network volumes, by name.
    Volumes,
    /// List the account's pod templates, by name.
    Templates,
}

/// Options of `overbrainer pod gpus`.
#[derive(Debug, Default, Args)]
pub struct GpuArgs {
    /// Only GPU types with at least this much VRAM, in GB.
    #[arg(long)]
    pub min_vram: Option<u32>,
    /// Only GPU types priced at or under this, in USD per hour.
    #[arg(long)]
    pub max_price: Option<f64>,
    /// Only GPU types offered in this data center, whose stock there is shown
    /// instead of the overall stock.
    #[arg(long)]
    pub data_center: Option<String>,
    /// Only GPU types with some stock.
    #[arg(long)]
    pub in_stock: bool,
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
    /// Start from the snapshot of this stopped run (`overbrainer train stop`):
    /// its checkpoint and its data, with the same training settings.
    #[arg(long, value_name = "RUN_ID", add = ArgValueCandidates::new(complete::run_ids))]
    pub resume_from: Option<String>,
    /// Follow or stop an existing run instead of starting one.
    #[command(subcommand)]
    pub command: Option<TrainCommand>,
    /// Never on the command line: the VRAM floor of `auto` GPU types, which
    /// the TUI's start confirmation may have estimated already, so the run
    /// uses what it showed.
    #[arg(skip)]
    pub vram_floor: crate::train::sizing::VramFloor,
}

/// Options of `overbrainer export`.
#[derive(Debug, Args)]
pub struct ExportArgs {
    /// ID of the run, as shown by `overbrainer runs ls`.
    #[arg(add = ArgValueCandidates::new(complete::run_ids))]
    pub run_id: String,
    /// llama-quantize type of the GGUF; F16 and BF16 keep 16-bit weights.
    /// Defaults to `export.quantize`.
    #[arg(long, value_name = "TYPE", value_parser = PossibleValuesParser::new(QUANTIZE_TYPES))]
    pub quantize: Option<String>,
    /// Create this Ollama model from the Modelfile once the GGUF is back, when
    /// `ollama` is on PATH. Defaults to `export.ollama_name`.
    #[arg(long, value_name = "NAME", value_parser = ollama_name)]
    pub ollama: Option<String>,
    /// Runpod target only: keep the export's pod once it ends, with no time
    /// limit. Nothing deletes it then but `overbrainer pod rm <export-id>`.
    #[arg(long)]
    pub keep_pod: bool,
}

/// Options of `overbrainer push`.
#[derive(Debug, Args)]
pub struct PushArgs {
    /// Run to push.
    #[arg(add = ArgValueCandidates::new(complete::run_ids))]
    pub run_id: String,
    /// Repo, NAMESPACE/NAME (default `[hub] repo`, else `<you>/<project>`).
    #[arg(long)]
    pub repo: Option<String>,
    /// Create the repo public.
    #[arg(long)]
    pub public: bool,
    /// Replace a README.md the repo already has, even one overbrainer did not write.
    #[arg(long)]
    pub overwrite_card: bool,
    /// Show what would be pushed and write the card to `runs/<id>/hub/README.md`, push nothing.
    #[arg(long)]
    pub dry_run: bool,
}

/// Parses an Ollama model name, as `--ollama` takes it.
fn ollama_name(text: &str) -> Result<String, String> {
    if crate::config::is_ollama_name(text) {
        Ok(text.to_string())
    } else {
        Err(
            "not an Ollama model name ([host/][namespace/]model[:tag], each part of letters, \
             digits, '_', '-' and '.')"
                .to_string(),
        )
    }
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
    /// Stop the job of a run with a snapshot: it saves a checkpoint at the end of
    /// its current step, stops, and the checkpoint is retrieved with its results.
    /// `overbrainer train --resume-from <run-id>` then starts a new run from it.
    Stop {
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
    /// Print the log of a run's job, or with --pod the logs of its Runpod pod.
    ///
    /// The pod's logs are kept in `runs/<run-id>/.pod/` while overbrainer follows
    /// the run and before every delete, so they outlive the pod; while the pod
    /// exists, the lines not kept yet are read from Runpod.
    Logs(LogsArgs),
}

/// Options of `overbrainer runs logs`.
#[derive(Debug, Args)]
pub struct LogsArgs {
    /// ID of the run, as shown by `overbrainer runs ls`.
    #[arg(add = ArgValueCandidates::new(complete::run_ids))]
    pub run_id: String,
    /// Print the logs of the run's Runpod pod (container and system lines, and
    /// the bootstrap's and watchdog's own logs) instead of the job's.
    #[arg(long)]
    pub pod: bool,
    /// Only the pod's lines from this source.
    #[arg(long, value_enum, requires = "pod")]
    pub source: Option<crate::runpod::LogSource>,
    /// Keep printing the pod's new lines until Ctrl-C.
    #[arg(long, requires = "pod")]
    pub follow: bool,
    /// Only the last N lines of each log (at most 5000 from Runpod).
    #[arg(long, value_name = "N")]
    pub tail: Option<u32>,
}

/// Options of `overbrainer history`.
#[derive(Debug, Default, Args)]
pub struct HistoryArgs {
    /// List every execution, oldest first, instead of the totals per stage.
    #[arg(long)]
    pub all: bool,
}

/// Options of `overbrainer migrate`.
#[derive(Debug, Default, Args)]
pub struct MigrateArgs {
    /// Print what would change, and change no project file (the lock is still
    /// taken).
    #[arg(long)]
    pub dry_run: bool,
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

/// Runs the init wizard when `cli` is `tui` on a terminal and the project
/// has no `overbrainer.toml`, before anything else: `.env` is loaded after
/// it, which it may write, and the project lock is taken once the files
/// exist. Returns whether the command goes on: not when the wizard was quit.
///
/// # Errors
///
/// Returns an error when the wizard cannot use the terminal.
pub fn wizard(cli: &mut Cli) -> anyhow::Result<bool> {
    if !wants_wizard(cli, std::io::stdout().is_terminal()) {
        return Ok(true);
    }
    // Refused before the first screen, not after the seventh.
    if let Some(refusal) = crate::tui::wizard_refusal(&cli.project_dir) {
        anyhow::bail!(refusal);
    }
    match crate::tui::wizard(&cli.project_dir)? {
        crate::tui::WizardEnded::Open { auto } => {
            cli.auto = auto;
            Ok(true)
        },
        crate::tui::WizardEnded::Quit { written: false } => {
            eprintln!("init wizard quit: nothing was written");
            Ok(false)
        },
        crate::tui::WizardEnded::Quit { written: true } => {
            eprintln!("the project is written: `overbrainer tui` opens it");
            Ok(false)
        },
    }
}

/// Whether `cli` opens the init wizard: `tui`, on a terminal, in a project
/// directory that exists and has no `overbrainer.toml`. A missing directory
/// fails as it did, never created.
fn wants_wizard(cli: &Cli, stdout_is_terminal: bool) -> bool {
    matches!(cli.command, Command::Tui)
        && stdout_is_terminal
        && cli.project_dir.is_dir()
        && cli
            .project_dir
            .join(crate::config::CONFIG_FILE)
            .symlink_metadata()
            .is_err()
}

/// Runs the parsed command line, whose logs were set up with `logs` (see
/// [`Command::log_mode`]); `dotenv` are the keys `.env` set at start, which a
/// reload of the configuration replaces (`tui`, and `run` between stages). Meanwhile it looks for a newer release, told on
/// stderr once the command ended, even with an error (`tui` shows it itself),
/// as is the need to run `overbrainer migrate` on a project from before 0.4.0.
///
/// # Errors
///
/// Returns an error if the selected subcommand fails, or if `tui` is not given the
/// [`LogMode::Tui`] its logs need.
pub async fn run(cli: Cli, logs: LogMode, dotenv: DotenvKeys) -> anyhow::Result<()> {
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
    let hint = migrate::hints(&cli.command, &cli.project_dir);
    let result = dispatch(cli, logs, &mut check, dotenv).await;
    if hint {
        eprintln!("{}", migrate::HINT);
    }
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
// One arm per command: splitting the table would only hide it.
#[allow(clippy::cognitive_complexity)]
async fn dispatch(
    cli: Cli,
    logs: LogMode,
    check: &mut Option<JoinHandle<Option<Newer>>>,
    dotenv: DotenvKeys,
) -> anyhow::Result<()> {
    let dir = &cli.project_dir;
    let auto = cli.auto;
    // Only a project takes the lock: without `overbrainer.toml` the command fails
    // with its usual error and leaves nothing behind.
    let lock = if cli.command.writes_project() && dir.join(crate::config::CONFIG_FILE).is_file() {
        match ProjectLock::acquire(dir) {
            Ok(lock) => Some(lock),
            // `train stop` only asks for the snapshot then: the process holding
            // the project, most likely following the run, collects it.
            Err(LockError::Held { pid }) => match cli.command.stopped_run() {
                Some(run_id) => {
                    let source = Source::from(EnvSource::Process);
                    let front = Frontend::Cli(None);
                    return train::request_stop(dir, run_id, pid, &front, &source).await;
                },
                None => return Err(LockError::Held { pid }.into()),
            },
            Err(error) => return Err(error.into()),
        }
    } else {
        None
    };
    // Served while the lock is held: dropped before it.
    let served = served_metrics(&cli.command, dir, lock.is_some()).await;
    // The buses of the command count into the served metrics, if any.
    let observer = served
        .as_ref()
        .map(|(_, metrics)| Arc::clone(metrics) as Arc<dyn Observer>);
    let front = Frontend::Cli(observer.clone());
    match cli.command {
        Command::Init { dir: target } => init::run(target.as_deref().unwrap_or(dir)),
        Command::Config {
            command: ConfigCommand::Check { resolve },
        } => config_check::run(dir, resolve).await,
        Command::Subtopics(args) => stage(dir, &front, data::Command::Subtopics, &args).await,
        Command::Questions(args) => stage(dir, &front, data::Command::Questions, &args).await,
        Command::Answers(args) => stage(dir, &front, data::Command::Answers, &args).await,
        Command::Split(SplitArgs { topic }) => {
            let args = StageArgs {
                topic,
                force: false,
            };
            stage(dir, &front, data::Command::Split, &args).await
        },
        Command::Run => run_all(dir, &front, dotenv).await,
        Command::Train(args) => {
            let source = Source::from(EnvSource::Process);
            Box::pin(train::run(dir, &args, &front, &source)).await
        },
        Command::Export(args) => Box::pin(export::run(dir, &args, &front)).await,
        Command::Push(args) => Box::pin(push::run(dir, &args, &front)).await,
        Command::Runs {
            command: RunsCommand::Ls,
        } => train::list(dir),
        Command::Runs {
            command: RunsCommand::Logs(args),
        } => logs::run(dir, &args).await,
        Command::History(args) => history::run(dir, &args),
        Command::Migrate(args) => migrate::run(dir, args.dry_run),
        Command::Pod { command } => pod::run(dir, &command).await,
        Command::Skill { command } => skill::run(dir, &command),
        Command::Tui => match logs {
            LogMode::Tui(buffer) => {
                let start = crate::tui::Start { dotenv, auto };
                crate::tui::run(dir, buffer, check.take(), start, observer).await
            },
            LogMode::Stderr => anyhow::bail!("overbrainer tui needs the TUI log mode"),
        },
    }
}

/// `overbrainer run`: every pipeline stage, then training when `[training]`
/// is set, reloading the settings whose files changed between them.
async fn run_all(dir: &Path, front: &Frontend, dotenv: DotenvKeys) -> anyhow::Result<()> {
    let mut reloader = Reloader::new(dir, dotenv);
    let args = StageArgs::default();
    let load = data::Load::Reload(&mut reloader);
    data::run(dir, data::Command::Run, &args, front, load).await?;
    // Boxed: the training flows would otherwise weigh on every command's
    // future.
    Box::pin(train::after_run(dir, front, &mut reloader)).await
}

/// Runs a pipeline command on the command line, through `front`.
async fn stage(
    dir: &Path,
    front: &Frontend,
    command: data::Command,
    args: &StageArgs,
) -> anyhow::Result<()> {
    let load = data::Load::Source(&Source::from(EnvSource::Process));
    data::run(dir, command, args, front, load).await
}

/// [`serve_metrics`] when `command` holds the lock (`locked`) and serves them.
async fn served_metrics(
    command: &Command,
    dir: &Path,
    locked: bool,
) -> Option<(MetricsServer, Arc<Metrics>)> {
    if locked && command.serves_metrics() {
        serve_metrics(dir).await
    } else {
        None
    }
}

/// Serves the Prometheus metrics of the project in `dir` when `metrics.listen` is
/// set, with the metrics it serves. Nothing here fails the command: a
/// configuration error is the command's to report, and an address that cannot be
/// bound only warns.
async fn serve_metrics(dir: &Path) -> Option<(MetricsServer, Arc<Metrics>)> {
    let settings = crate::config::load(dir, EnvSource::Process).ok()?;
    let address = settings.metrics.listen?;
    if !address.ip().is_loopback() {
        tracing::warn!("metrics on {address} have no authentication");
    }
    let metrics = Arc::new(project_metrics(dir));
    crate::metrics::serve(address, Arc::clone(&metrics))
        .await
        .inspect(|server| tracing::info!("metrics at http://{}/metrics", server.address()))
        .inspect_err(|error| tracing::warn!("cannot serve the metrics on {address}: {error}"))
        .ok()
        .map(|server| (server, metrics))
}

/// The metrics of the project in `dir`, from its stage history and its runs.
fn project_metrics(dir: &Path) -> Metrics {
    let history = crate::history::read(dir).unwrap_or_else(|error| {
        tracing::warn!("cannot read the stage history, the metrics start from zero: {error}");
        Vec::new()
    });
    Metrics::new(&history).with_runs(crate::runs::Runs::new(dir))
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
    fn the_wizard_opens_only_for_tui_in_an_existing_directory_without_a_config()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().to_string_lossy().into_owned();
        let missing = dir.path().join("missing").to_string_lossy().into_owned();
        let cli = |args: &[&str]| Cli::try_parse_from(args);
        assert!(wants_wizard(
            &cli(&["overbrainer", "-C", &path, "tui"])?,
            true
        ));
        assert!(!wants_wizard(
            &cli(&["overbrainer", "-C", &path, "tui"])?,
            false
        ));
        assert!(!wants_wizard(
            &cli(&["overbrainer", "-C", &path, "run"])?,
            true
        ));
        assert!(!wants_wizard(
            &cli(&["overbrainer", "-C", &missing, "tui"])?,
            true
        ));
        assert!(!dir.path().join("missing").exists());
        std::fs::write(dir.path().join(crate::config::CONFIG_FILE), "")?;
        assert!(!wants_wizard(
            &cli(&["overbrainer", "-C", &path, "tui"])?,
            true
        ));
        Ok(())
    }

    #[test]
    fn export_takes_a_known_type_and_an_ollama_name() -> Result<(), clap::Error> {
        let Command::Export(args) = command(&[
            "export",
            "r1",
            "--quantize",
            "Q8_0",
            "--ollama",
            "me/mentor:q8",
            "--keep-pod",
        ])?
        else {
            return Err(clap::Error::new(clap::error::ErrorKind::InvalidSubcommand));
        };
        assert_eq!(args.run_id, "r1");
        assert_eq!(args.quantize.as_deref(), Some("Q8_0"));
        assert_eq!(args.ollama.as_deref(), Some("me/mentor:q8"));
        assert!(args.keep_pod);
        let Command::Export(defaults) = command(&["export", "r1"])? else {
            return Err(clap::Error::new(clap::error::ErrorKind::InvalidSubcommand));
        };
        assert_eq!((defaults.quantize, defaults.ollama), (None, None));
        assert!(command(&["export", "r1", "--quantize", "Q4"]).is_err());
        assert!(command(&["export", "r1", "--ollama", "bad name"]).is_err());
        assert!(command(&["export", "r1", "--ollama", "a//b"]).is_err());
        assert!(command(&["export"]).is_err());
        Ok(())
    }

    #[test]
    fn push_takes_a_repo_and_its_flags() -> Result<(), clap::Error> {
        let Command::Push(args) = command(&[
            "push",
            "r1",
            "--repo",
            "me/model",
            "--public",
            "--overwrite-card",
            "--dry-run",
        ])?
        else {
            return Err(clap::Error::new(clap::error::ErrorKind::InvalidSubcommand));
        };
        assert_eq!(args.run_id, "r1");
        assert_eq!(args.repo.as_deref(), Some("me/model"));
        assert!(args.public && args.overwrite_card && args.dry_run);
        let Command::Push(defaults) = command(&["push", "r1"])? else {
            return Err(clap::Error::new(clap::error::ErrorKind::InvalidSubcommand));
        };
        assert_eq!(defaults.repo, None);
        assert!(!defaults.public && !defaults.overwrite_card && !defaults.dry_run);
        assert!(command(&["push"]).is_err());
        Ok(())
    }

    #[test]
    fn only_train_stop_names_a_stopped_run() -> Result<(), clap::Error> {
        let stopped = |args: &[&str]| -> Result<Option<String>, clap::Error> {
            let cli =
                Cli::try_parse_from(std::iter::once("overbrainer").chain(args.iter().copied()))?;
            Ok(cli.command.stopped_run().map(str::to_string))
        };
        assert_eq!(stopped(&["train", "stop", "x"])?.as_deref(), Some("x"));
        for args in [
            &["train"][..],
            &["train", "attach", "x"],
            &["train", "cancel", "x"],
        ] {
            assert_eq!(stopped(args)?, None, "{args:?}");
        }
        Ok(())
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
            &["train", "stop", "x"],
            &["export", "x"],
            &["push", "x"],
            &["push", "x", "--dry-run"],
            &["pod", "rm", "x"],
            &["migrate"],
            &["migrate", "--dry-run"],
        ] {
            assert!(writes(args)?, "{args:?} should lock");
        }
        let serves = |args: &[&str]| -> Result<bool, clap::Error> {
            let cli =
                Cli::try_parse_from(std::iter::once("overbrainer").chain(args.iter().copied()))?;
            Ok(cli.command.serves_metrics())
        };
        assert!(serves(&["run"])? && serves(&["tui"])?);
        assert!(!serves(&["migrate"])? && !serves(&["history"])?);
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
