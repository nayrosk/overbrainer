//! The pipeline commands: `subtopics`, `questions`, `answers`, `split` and `run`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, bail};
use serde_json::Value;
use tokio::sync::{Mutex, OnceCell};

use super::{LazyVault, StageArgs};
use crate::config::{EnvSource, RoleModel, Settings};
use crate::dataset::{DataFiles, Subtopic, read};
use crate::dedup::{Embedding, Layered, Lexical};
use crate::events::{EventBus, Stage, StageStats};
use crate::llm::{ProtocolClient, connect};
use crate::pipeline::{self, Ctx, PipelineError, RoleClient, SplitReport};
use crate::pricing::{LISTING_TIMEOUT, Price, fetch_listing, listed_price};
use crate::prompts::Prompts;
use crate::secrets::Resolver;

/// Which pipeline command to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    /// `overbrainer subtopics`
    Subtopics,
    /// `overbrainer questions`
    Questions,
    /// `overbrainer answers`
    Answers,
    /// `overbrainer split`
    Split,
    /// `overbrainer run`: the four stages in order.
    Run,
}

type Client = RoleClient<ProtocolClient>;

/// Everything a pipeline command needs. Role clients are built on first use and at
/// most once per command, and each provider's model listing is read at most once.
struct Session {
    project_dir: PathBuf,
    settings: Settings,
    files: DataFiles,
    resolver: Resolver<LazyVault>,
    generator: OnceCell<Client>,
    parent: OnceCell<Arc<Client>>,
    embedder: OnceCell<Option<ProtocolClient>>,
    listings: Mutex<BTreeMap<String, Option<Value>>>,
}

impl Session {
    fn open(project_dir: &Path) -> anyhow::Result<Self> {
        Ok(Self {
            project_dir: project_dir.to_path_buf(),
            settings: crate::config::load(project_dir, EnvSource::Process)?,
            files: DataFiles::new(project_dir),
            resolver: super::resolver(),
            generator: OnceCell::new(),
            parent: OnceCell::new(),
            embedder: OnceCell::new(),
            listings: Mutex::new(BTreeMap::new()),
        })
    }

    fn ctx<'a>(&'a self, bus: &'a EventBus, args: &'a StageArgs, prompts: &'a Prompts) -> Ctx<'a> {
        Ctx {
            settings: &self.settings,
            files: &self.files,
            prompts,
            bus,
            topic: args.topic.as_deref(),
            force: args.force,
        }
    }

    async fn generator(&self) -> anyhow::Result<&Client> {
        self.generator
            .get_or_try_init(|| self.role(&self.settings.roles.generator))
            .await
    }

    async fn parent(&self) -> anyhow::Result<Arc<Client>> {
        let parent = self
            .parent
            .get_or_try_init(|| async {
                Ok::<_, anyhow::Error>(Arc::new(self.role(&self.settings.roles.parent).await?))
            })
            .await?;
        Ok(Arc::clone(parent))
    }

    async fn embedder(&self) -> anyhow::Result<Option<ProtocolClient>> {
        let embedder = self
            .embedder
            .get_or_try_init(|| async {
                match &self.settings.roles.embedder {
                    Some(role) => connect(&self.settings, role, &self.resolver)
                        .await
                        .map(Some)
                        .context("cannot set up the embedder"),
                    None => Ok(None),
                }
            })
            .await?;
        Ok(embedder.clone())
    }

    /// Connects to the provider of `role` and looks up the model price.
    async fn role(&self, role: &RoleModel) -> anyhow::Result<Client> {
        let client = connect(&self.settings, role, &self.resolver).await?;
        let price = self.price(role, &client).await;
        Ok(RoleClient {
            client,
            model: role.clone(),
            price,
        })
    }

    /// Price of `role`'s model, from its provider's listing read once per command.
    async fn price(&self, role: &RoleModel, client: &ProtocolClient) -> Option<Price> {
        let mut listings = self.listings.lock().await;
        if !listings.contains_key(&role.provider) {
            let listing = fetch_listing(client, LISTING_TIMEOUT).await;
            listings.insert(role.provider.clone(), listing);
        }
        let listing = listings.get(&role.provider)?.as_ref()?;
        listed_price(listing, &role.model)
    }
}

/// Runs `command` in `project_dir`, logging progress to stderr and printing a usage
/// summary per stage to stdout.
///
/// # Errors
///
/// Returns an error if the configuration, a template or a data file cannot be loaded,
/// if `--topic` names no configured topic, if a provider cannot be reached, or if
/// items failed (they are retried on the next run).
pub async fn run(project_dir: &Path, command: Command, args: &StageArgs) -> anyhow::Result<()> {
    let session = Session::open(project_dir)?;
    let bus = EventBus::new();
    let renderer = tokio::spawn(super::progress::render(bus.subscribe()));
    let result = execute(&session, &bus, command, args).await;
    drop(bus);
    renderer.await.ok();
    result
}

async fn execute(
    session: &Session,
    bus: &EventBus,
    command: Command,
    args: &StageArgs,
) -> anyhow::Result<()> {
    let empty = Prompts::empty();
    session.ctx(bus, args, &empty).topics()?;
    let prompts = if command == Command::Split {
        empty
    } else {
        Prompts::load(&session.project_dir)?
    };
    let ctx = session.ctx(bus, args, &prompts);
    match command {
        Command::Subtopics => report(Stage::Subtopics, subtopics(session, &ctx).await),
        Command::Questions => {
            if missing_subtopics(&ctx)? {
                let first = Ctx {
                    force: false,
                    ..ctx
                };
                report(Stage::Subtopics, subtopics(session, &first).await)?;
            }
            report(Stage::Questions, questions(session, &ctx).await)
        },
        Command::Answers => report(Stage::Answers, answers(session, &ctx).await),
        Command::Split => {
            print_split(&pipeline::split(&ctx)?);
            Ok(())
        },
        Command::Run => run_all(session, &ctx).await,
    }
}

async fn run_all(session: &Session, ctx: &Ctx<'_>) -> anyhow::Result<()> {
    report(Stage::Subtopics, subtopics(session, ctx).await)?;
    report(Stage::Questions, questions(session, ctx).await)?;
    report(Stage::Answers, answers(session, ctx).await)?;
    print_split(&pipeline::split(ctx)?);
    tracing::info!("training is not available yet: run stops after split");
    Ok(())
}

/// Whether a selected topic has no subtopic yet (`questions` then generates them).
fn missing_subtopics(ctx: &Ctx<'_>) -> anyhow::Result<bool> {
    let existing: Vec<Subtopic> = read(&ctx.files.subtopics)?;
    Ok(ctx
        .topics()?
        .iter()
        .any(|topic| !existing.iter().any(|subtopic| subtopic.topic == topic.name)))
}

async fn subtopics(session: &Session, ctx: &Ctx<'_>) -> anyhow::Result<StageStats> {
    let generator = session.generator().await?;
    Ok(pipeline::subtopics(ctx, generator).await?)
}

async fn questions(session: &Session, ctx: &Ctx<'_>) -> anyhow::Result<StageStats> {
    let generator = session.generator().await?;
    let embedder = session.embedder().await?;
    let pipeline = &session.settings.pipeline;
    let policy = ctx.policy();
    let new_dedup = || {
        Layered::new(
            Lexical::new(pipeline.dedup_threshold),
            embedder
                .clone()
                .map(|client| Embedding::new(client, pipeline.embedding_threshold, policy)),
        )
    };
    Ok(pipeline::questions(ctx, generator, new_dedup).await?)
}

async fn answers(session: &Session, ctx: &Ctx<'_>) -> anyhow::Result<StageStats> {
    Ok(pipeline::answers(ctx, session.parent().await?).await?)
}

/// Prints the stage summary on stdout and fails when items failed. A stage stopped by
/// a fatal provider error still prints what it produced and spent before stopping.
fn report(stage: Stage, result: anyhow::Result<StageStats>) -> anyhow::Result<()> {
    let stats = match result {
        Ok(stats) => stats,
        Err(error) => {
            if let Some(PipelineError::Llm { stage, spent, .. }) = error.downcast_ref() {
                println!("{}", summary(*stage, spent));
            }
            return Err(error);
        },
    };
    println!("{}", summary(stage, &stats));
    if stats.failed > 0 {
        bail!(
            "{} {stage} item(s) failed; run the command again to retry them",
            stats.failed
        );
    }
    Ok(())
}

fn print_split(report: &SplitReport) {
    println!("{}", split_summary(report));
}

/// One line: counts, tokens, and cost when the price is known.
#[must_use]
pub fn summary(stage: Stage, stats: &StageStats) -> String {
    let cost = stats.cost.map_or_else(
        || "cost unknown".to_string(),
        |cost| format!("cost ${cost:.4}"),
    );
    format!(
        "{stage}: {} done, {} skipped, {} failed, {} excluded; tokens {} in, {} out; {cost}",
        stats.done,
        stats.skipped,
        stats.failed,
        stats.excluded,
        stats.usage.input_tokens,
        stats.usage.output_tokens,
    )
}

/// One line: train and eval sizes, exclusions by reason, then orphaned examples.
#[must_use]
pub fn split_summary(report: &SplitReport) -> String {
    let excluded: usize = report.excluded.values().sum();
    let reasons: Vec<String> = report
        .excluded
        .iter()
        .map(|(reason, count)| {
            let name = serde_json::to_value(reason)
                .ok()
                .and_then(|value| value.as_str().map(str::to_string))
                .unwrap_or_default();
            format!("{name} {count}")
        })
        .collect();
    let detail = if reasons.is_empty() {
        String::new()
    } else {
        format!(" ({})", reasons.join(", "))
    };
    format!(
        "split: {} train, {} eval, {excluded} excluded{detail}, {} orphaned",
        report.train, report.eval, report.orphaned
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Exclusion;
    use crate::llm::Usage;

    #[test]
    fn summary_shows_cost_only_when_known() {
        let mut stats = StageStats {
            done: 3,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 20,
            },
            ..StageStats::default()
        };
        assert_eq!(
            summary(Stage::Answers, &stats),
            "answers: 3 done, 0 skipped, 0 failed, 0 excluded; tokens 10 in, 20 out; cost unknown"
        );
        stats.cost = Some(0.123_45);
        assert!(summary(Stage::Answers, &stats).ends_with("cost $0.1235"));
    }

    #[test]
    fn split_summary_lists_reasons() {
        let report = SplitReport {
            train: 9,
            eval: 1,
            excluded: BTreeMap::from([(Exclusion::Truncated, 2), (Exclusion::NoRawReasoning, 1)]),
            orphaned: 4,
        };
        assert_eq!(
            split_summary(&report),
            "split: 9 train, 1 eval, 3 excluded (truncated 2, no_raw_reasoning 1), 4 orphaned"
        );
    }
}
