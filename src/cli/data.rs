//! The pipeline commands: `subtopics`, `questions`, `answers`, `split` and `run`.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, bail};

use super::StageArgs;
use crate::config::{EnvSource, RoleModel, Settings};
use crate::dataset::{DataFiles, Subtopic, read};
use crate::dedup::{Embedding, Layered, Lexical};
use crate::events::{EventBus, Stage, StageStats};
use crate::llm::{ProtocolClient, connect};
use crate::pipeline::{self, Ctx, RoleClient, SplitReport};
use crate::pricing::fetch_price;
use crate::prompts::Prompts;
use crate::secrets::{Resolver, VaultSource};

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

/// Everything a pipeline command needs, loaded once.
struct Session {
    settings: Settings,
    files: DataFiles,
    prompts: Prompts,
    resolver: Resolver<VaultSource>,
}

impl Session {
    fn open(project_dir: &Path) -> anyhow::Result<Self> {
        Ok(Self {
            settings: crate::config::load(project_dir, EnvSource::Process)?,
            files: DataFiles::new(project_dir),
            prompts: Prompts::load(project_dir)?,
            resolver: super::resolver()?,
        })
    }

    fn ctx<'a>(&'a self, bus: &'a EventBus, args: &'a StageArgs) -> Ctx<'a> {
        Ctx {
            settings: &self.settings,
            files: &self.files,
            prompts: &self.prompts,
            bus,
            topic: args.topic.as_deref(),
            force: args.force,
        }
    }

    /// Connects to the provider of `role` and looks up the model price.
    async fn role(&self, role: &RoleModel) -> anyhow::Result<RoleClient<ProtocolClient>> {
        let client = connect(&self.settings, role, &self.resolver).await?;
        let price = fetch_price(&client, &role.model).await;
        Ok(RoleClient {
            client,
            model: role.clone(),
            price,
        })
    }
}

/// Runs `command` in `project_dir`, logging progress to stderr and printing a usage
/// summary per stage to stdout.
///
/// # Errors
///
/// Returns an error if the configuration, a template or a data file cannot be loaded,
/// if a provider cannot be reached, or if items failed (they are retried on the next run).
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
    let ctx = session.ctx(bus, args);
    match command {
        Command::Subtopics => report(Stage::Subtopics, &subtopics(session, &ctx).await?),
        Command::Questions => report(Stage::Questions, &questions(session, &ctx).await?),
        Command::Answers => report(Stage::Answers, &answers(session, &ctx).await?),
        Command::Split => {
            print_split(&pipeline::split(&ctx)?);
            Ok(())
        },
        Command::Run => run_all(session, &ctx).await,
    }
}

async fn run_all(session: &Session, ctx: &Ctx<'_>) -> anyhow::Result<()> {
    report(Stage::Subtopics, &subtopics(session, ctx).await?)?;
    report(Stage::Questions, &questions(session, ctx).await?)?;
    report(Stage::Answers, &answers(session, ctx).await?)?;
    print_split(&pipeline::split(ctx)?);
    tracing::info!("training is not available yet: run stops after split");
    Ok(())
}

async fn subtopics(session: &Session, ctx: &Ctx<'_>) -> anyhow::Result<StageStats> {
    let generator = session.role(&session.settings.roles.generator).await?;
    Ok(pipeline::subtopics(ctx, &generator).await?)
}

/// Runs `subtopics` first when a selected topic has none, then generates questions.
async fn questions(session: &Session, ctx: &Ctx<'_>) -> anyhow::Result<StageStats> {
    let existing: Vec<Subtopic> = read(&session.files.subtopics)?;
    let missing = ctx
        .topics()?
        .iter()
        .any(|topic| !existing.iter().any(|subtopic| subtopic.topic == topic.name));
    if missing {
        let first = Ctx {
            force: false,
            ..*ctx
        };
        report(Stage::Subtopics, &subtopics(session, &first).await?)?;
    }
    let generator = session.role(&session.settings.roles.generator).await?;
    let embedder = match &session.settings.roles.embedder {
        Some(role) => Some(
            connect(&session.settings, role, &session.resolver)
                .await
                .context("cannot set up the embedder")?,
        ),
        None => None,
    };
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
    Ok(pipeline::questions(ctx, &generator, new_dedup).await?)
}

async fn answers(session: &Session, ctx: &Ctx<'_>) -> anyhow::Result<StageStats> {
    let parent = Arc::new(session.role(&session.settings.roles.parent).await?);
    Ok(pipeline::answers(ctx, parent).await?)
}

/// Prints the stage summary on stdout and fails when items failed.
fn report(stage: Stage, stats: &StageStats) -> anyhow::Result<()> {
    println!("{}", summary(stage, stats));
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

/// One line: train and eval sizes, then exclusions by reason.
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
        "split: {} train, {} eval, {excluded} excluded{detail}",
        report.train, report.eval
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

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
        };
        assert_eq!(
            split_summary(&report),
            "split: 9 train, 1 eval, 3 excluded (truncated 2, no_raw_reasoning 1)"
        );
    }
}
