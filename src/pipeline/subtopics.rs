use std::collections::BTreeSet;

use serde::Serialize;

use super::{Ctx, Item, PipelineError, RoleClient, ask_list, item_error, without_topics};
use crate::config::Topic;
use crate::dataset::{Appender, Id, Subtopic, read, rewrite};
use crate::events::{Event, Stage, StageStats};
use crate::llm::{LlmClient, Usage};
use crate::prompts;

#[derive(Serialize)]
struct Context<'a> {
    topic: &'a str,
    description: Option<&'a str>,
    count: u32,
}

/// One topic's remaining subtopics to generate: how many are still missing and which
/// IDs (stored or already generated in this run) must not be repeated.
struct Plan<'a> {
    topic: &'a Topic,
    existing_ids: BTreeSet<Id>,
    missing: usize,
    item: Item,
}

/// Generates the subtopics of each selected topic into `data/subtopics.jsonl`.
///
/// A topic is skipped once it has `topic.subtopics` stored subtopics; `--force` first
/// removes the topic's subtopics. A topic with fewer stored subtopics than configured,
/// for example after a crash mid-topic, is resumed: only the missing count is
/// requested, and generated names that normalize to an ID already stored or already
/// generated in this batch are dropped. An unparseable answer is asked again.
///
/// # Errors
///
/// Returns a [`PipelineError`] when a file or template fails, or when the provider
/// rejects the requests (for example a bad API key).
pub async fn subtopics<C: LlmClient>(
    ctx: &Ctx<'_>,
    generator: &RoleClient<C>,
) -> Result<StageStats, PipelineError> {
    let topics = ctx.topics()?;
    let mut existing: Vec<Subtopic> = read(&ctx.files.subtopics)?;
    if ctx.force {
        existing = without_topics(existing, &topics, |subtopic| &subtopic.topic);
        rewrite(&ctx.files.subtopics, &existing)?;
    }
    let mut out = Appender::open(&ctx.files.subtopics)?;
    let total = topics
        .iter()
        .filter(|topic| stored_count(&existing, &topic.name) < target_count(topic))
        .count();
    ctx.bus.publish(Event::StageStarted {
        stage: Stage::Subtopics,
        total,
    });
    let mut stats = StageStats::default();
    for topic in topics {
        let have = stored_count(&existing, &topic.name);
        let target = target_count(topic);
        if have >= target {
            stats.skipped += 1;
            continue;
        }
        let plan = Plan {
            existing_ids: existing
                .iter()
                .filter(|subtopic| subtopic.topic == topic.name)
                .map(|subtopic| subtopic.id.clone())
                .collect(),
            missing: target - have,
            item: Item {
                stage: Stage::Subtopics,
                id: topic.name.clone(),
            },
            topic,
        };
        match generate(ctx, generator, &plan, &mut stats).await {
            Ok((new_subtopics, usage)) => {
                for subtopic in &new_subtopics {
                    out.append(subtopic)?;
                }
                stats.done += 1;
                ctx.bus.publish(Event::ItemDone {
                    stage: Stage::Subtopics,
                    id: plan.item.id,
                    usage: Some(usage),
                });
            },
            Err(PipelineError::Llm { source, .. }) => {
                item_error(ctx, &plan.item, source, &mut stats)?;
            },
            Err(error) => return Err(error),
        }
    }
    ctx.bus.publish(Event::StageFinished {
        stage: Stage::Subtopics,
        stats: stats.clone(),
    });
    Ok(stats)
}

/// Subtopics of `topic_name` already stored.
fn stored_count(existing: &[Subtopic], topic_name: &str) -> usize {
    existing
        .iter()
        .filter(|subtopic| subtopic.topic == topic_name)
        .count()
}

/// Configured subtopic count of `topic`.
fn target_count(topic: &Topic) -> usize {
    usize::try_from(topic.subtopics).unwrap_or(usize::MAX)
}

/// Asks for `plan.missing` names and turns them into records, dropping names that
/// normalize to an ID already stored or already generated earlier in this batch.
async fn generate<C: LlmClient>(
    ctx: &Ctx<'_>,
    generator: &RoleClient<C>,
    plan: &Plan<'_>,
    stats: &mut StageStats,
) -> Result<(Vec<Subtopic>, Usage), PipelineError> {
    let prompt = ctx.prompts.render(
        prompts::SUBTOPICS,
        Context {
            topic: &plan.topic.name,
            description: plan.topic.description.as_deref(),
            count: u32::try_from(plan.missing).unwrap_or(u32::MAX),
        },
    )?;
    let (names, usage) = ask_list(ctx, generator, prompt, &plan.item, stats)
        .await
        .map_err(|source| PipelineError::Llm {
            stage: Stage::Subtopics,
            source,
        })?;
    let mut seen = plan.existing_ids.clone();
    let subtopics = names
        .into_iter()
        .map(|name| Subtopic {
            id: Id::subtopic(&plan.topic.name, &name),
            topic: plan.topic.name.clone(),
            name,
        })
        .filter(|subtopic| seen.insert(subtopic.id.clone()))
        .take(plan.missing)
        .collect();
    Ok((subtopics, usage))
}
