use std::collections::BTreeSet;

use serde::Serialize;

use super::{Ctx, Item, PipelineError, RoleClient, ask_list, item_error, without_topics};
use crate::config::Topic;
use crate::dataset::{Appender, Id, Subtopic, read, rewrite};
use crate::events::{Event, Stage, StageStats};
use crate::llm::LlmClient;
use crate::prompts;

#[derive(Serialize)]
struct Context<'a> {
    topic: &'a str,
    description: Option<&'a str>,
    count: u32,
}

/// Generates the subtopics of each selected topic into `data/subtopics.jsonl`.
///
/// A topic that already has subtopics is skipped unless `--force` is set, which first
/// removes the topic's subtopics. An unparseable answer is asked again.
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
    ctx.bus.publish(Event::StageStarted {
        stage: Stage::Subtopics,
        total: topics.len(),
    });
    let mut stats = StageStats::default();
    for topic in topics {
        if existing.iter().any(|subtopic| subtopic.topic == topic.name) {
            stats.skipped += 1;
            continue;
        }
        let item = Item {
            stage: Stage::Subtopics,
            id: topic.name.clone(),
        };
        match generate(ctx, generator, topic, &item, &mut stats).await {
            Ok(subtopics) => {
                for subtopic in &subtopics {
                    out.append(subtopic)?;
                }
                stats.done += 1;
                ctx.bus.publish(Event::ItemDone {
                    stage: Stage::Subtopics,
                    id: item.id,
                    usage: None,
                });
            },
            Err(PipelineError::Llm { source, .. }) => {
                item_error(ctx, &item, source, &mut stats)?;
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

/// Asks for `topic.subtopics` names and turns them into records, dropping names that
/// normalize to the same ID and anything beyond the requested count.
async fn generate<C: LlmClient>(
    ctx: &Ctx<'_>,
    generator: &RoleClient<C>,
    topic: &Topic,
    item: &Item,
    stats: &mut StageStats,
) -> Result<Vec<Subtopic>, PipelineError> {
    let prompt = ctx.prompts.render(
        prompts::SUBTOPICS,
        Context {
            topic: &topic.name,
            description: topic.description.as_deref(),
            count: topic.subtopics,
        },
    )?;
    let names = ask_list(ctx, generator, prompt, item, stats)
        .await
        .map_err(|source| PipelineError::Llm {
            stage: Stage::Subtopics,
            source,
        })?;
    let limit = usize::try_from(topic.subtopics).unwrap_or(usize::MAX);
    let mut seen = BTreeSet::new();
    Ok(names
        .into_iter()
        .map(|name| Subtopic {
            id: Id::subtopic(&topic.name, &name),
            topic: topic.name.clone(),
            name,
        })
        .filter(|subtopic| seen.insert(subtopic.id.clone()))
        .take(limit)
        .collect())
}
