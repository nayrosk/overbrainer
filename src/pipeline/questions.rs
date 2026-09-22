use serde::Serialize;

use super::{Ctx, Item, PipelineError, RoleClient, ask_list, item_error, without_topics};
use crate::config::Topic;
use crate::dataset::{Appender, Id, Question, Subtopic, read, rewrite};
use crate::dedup::Deduplicator;
use crate::events::{Event, Stage, StageStats};
use crate::llm::{LlmClient, Usage};
use crate::prompts;

#[derive(Serialize)]
struct Context<'a> {
    topic: &'a str,
    description: Option<&'a str>,
    subtopic: &'a str,
    count: usize,
    accepted: &'a [String],
}

/// Generates questions for every subtopic of the selected topics into
/// `data/questions.jsonl`.
///
/// Subtopics are filled one batch at a time (`pipeline.question_batch_size`), each
/// prompt listing the questions already accepted for the subtopic. Every batch goes
/// through a deduplicator shared by the whole topic, created by `new_dedup` and seeded
/// with the questions already on disk. A subtopic stops at `questions_per_subtopic`
/// or after `pipeline.max_retries` batches without a new question (at least one).
/// Batches run one after the other because each depends on the previous ones. A
/// deduplicator error that survives its own retries stops the whole stage when it is
/// fatal; otherwise it fails only the subtopic being filled, or, when it happens while
/// seeding a topic's deduplicator from the questions already on disk, the whole topic.
///
/// # Errors
///
/// Returns a [`PipelineError`] when a file or template fails, when deduplication
/// fails fatally, or when the provider rejects the requests.
pub async fn questions<C, D, F>(
    ctx: &Ctx<'_>,
    generator: &RoleClient<C>,
    mut new_dedup: F,
) -> Result<StageStats, PipelineError>
where
    C: LlmClient,
    D: Deduplicator,
    F: FnMut() -> D,
{
    let topics = ctx.topics()?;
    let subtopics: Vec<Subtopic> = read(&ctx.files.subtopics)?;
    let mut existing: Vec<Question> = read(&ctx.files.questions)?;
    if ctx.force {
        existing = without_topics(existing, &topics, |question| &question.topic);
        rewrite(&ctx.files.questions, &existing)?;
    }
    ctx.bus.publish(Event::StageStarted {
        stage: Stage::Questions,
        total: remaining_subtopics(&subtopics, &existing, &topics),
    });
    let mut filler = Filler {
        ctx,
        generator,
        out: Appender::open(&ctx.files.questions)?,
        stats: StageStats::default(),
    };
    for topic in topics {
        let mut dedup = new_dedup();
        let known: Vec<String> = existing
            .iter()
            .filter(|question| question.topic == topic.name)
            .map(|question| question.text.clone())
            .collect();
        let seed = Item {
            stage: Stage::Questions,
            id: topic.name.clone(),
        };
        if let Err(error) = dedup.record(&known).await {
            item_error(ctx, &seed, error, &mut filler.stats)?;
            continue;
        }
        for subtopic in subtopics.iter().filter(|s| s.topic == topic.name) {
            let accepted: Vec<String> = existing
                .iter()
                .filter(|question| question.subtopic_id == subtopic.id)
                .map(|question| question.text.clone())
                .collect();
            filler
                .fill(Slot::new(topic, subtopic, accepted), &mut dedup)
                .await?;
        }
    }
    ctx.bus.publish(Event::StageFinished {
        stage: Stage::Questions,
        stats: filler.stats.clone(),
    });
    Ok(filler.stats)
}

/// Subtopics of `topics` not yet filled to their `questions_per_subtopic` target.
fn remaining_subtopics(subtopics: &[Subtopic], existing: &[Question], topics: &[&Topic]) -> usize {
    subtopics
        .iter()
        .filter(|subtopic| needs_filling(subtopic, existing, topics))
        .count()
}

/// Whether `subtopic` still needs questions, given what is already accepted for it.
fn needs_filling(subtopic: &Subtopic, existing: &[Question], topics: &[&Topic]) -> bool {
    let Some(topic) = topics.iter().find(|topic| topic.name == subtopic.topic) else {
        return false;
    };
    let accepted = existing
        .iter()
        .filter(|question| question.subtopic_id == subtopic.id)
        .count();
    accepted < usize::try_from(topic.questions_per_subtopic).unwrap_or(usize::MAX)
}

/// One subtopic being filled.
struct Slot<'a> {
    topic: &'a Topic,
    subtopic: &'a Subtopic,
    item: Item,
    target: usize,
    accepted: Vec<String>,
}

impl<'a> Slot<'a> {
    fn new(topic: &'a Topic, subtopic: &'a Subtopic, accepted: Vec<String>) -> Self {
        Self {
            topic,
            subtopic,
            item: Item {
                stage: Stage::Questions,
                id: subtopic.id.to_string(),
            },
            target: usize::try_from(topic.questions_per_subtopic).unwrap_or(usize::MAX),
            accepted,
        }
    }
}

struct Filler<'a, C> {
    ctx: &'a Ctx<'a>,
    generator: &'a RoleClient<C>,
    out: Appender,
    stats: StageStats,
}

impl<C: LlmClient> Filler<'_, C> {
    async fn fill<D: Deduplicator>(
        &mut self,
        mut slot: Slot<'_>,
        dedup: &mut D,
    ) -> Result<(), PipelineError> {
        if slot.accepted.len() >= slot.target {
            self.stats.skipped += 1;
            return Ok(());
        }
        let patience = self.ctx.settings.pipeline.max_retries.max(1);
        let mut stalls = 0;
        let mut usage = Usage::default();
        while slot.accepted.len() < slot.target && stalls < patience {
            let Some((added, batch_usage)) = self.batch(&slot, dedup).await? else {
                return Ok(());
            };
            usage += batch_usage;
            stalls = if added.is_empty() { stalls + 1 } else { 0 };
            slot.accepted.extend(added);
        }
        if slot.accepted.len() < slot.target {
            short(&slot);
        }
        self.stats.done += 1;
        self.ctx.bus.publish(Event::ItemDone {
            stage: Stage::Questions,
            id: slot.item.id,
            usage: Some(usage),
        });
        Ok(())
    }

    /// Generates, deduplicates and writes one batch. Returns the accepted texts with
    /// their token usage, or `None` when the subtopic failed and was reported.
    async fn batch<D: Deduplicator>(
        &mut self,
        slot: &Slot<'_>,
        dedup: &mut D,
    ) -> Result<Option<(Vec<String>, Usage)>, PipelineError> {
        let batch_size =
            usize::try_from(self.ctx.settings.pipeline.question_batch_size).unwrap_or(usize::MAX);
        let count = batch_size.min(slot.target - slot.accepted.len());
        let prompt = self.ctx.prompts.render(
            prompts::QUESTIONS,
            Context {
                topic: &slot.topic.name,
                description: slot.topic.description.as_deref(),
                subtopic: &slot.subtopic.name,
                count,
                accepted: &slot.accepted,
            },
        )?;
        let asked = ask_list(
            self.ctx,
            self.generator,
            prompt,
            &slot.item,
            &mut self.stats,
        )
        .await;
        let (candidates, usage) = match asked {
            Ok(asked) => asked,
            Err(error) => {
                item_error(self.ctx, &slot.item, error, &mut self.stats)?;
                return Ok(None);
            },
        };
        let candidates = candidates.into_iter().take(count).collect();
        let added = match dedup.admit(candidates).await {
            Ok(added) => added,
            Err(error) => {
                item_error(self.ctx, &slot.item, error, &mut self.stats)?;
                return Ok(None);
            },
        };
        for text in &added {
            self.out.append(&Question {
                id: Id::question(&slot.subtopic.id, text),
                topic: slot.topic.name.clone(),
                subtopic_id: slot.subtopic.id.clone(),
                subtopic: slot.subtopic.name.clone(),
                text: text.clone(),
            })?;
        }
        Ok(Some((added, usage)))
    }
}

fn short(slot: &Slot<'_>) {
    tracing::warn!(
        "subtopic `{}`: stopped at {} of {} questions, the generator kept repeating itself",
        slot.subtopic.name,
        slot.accepted.len(),
        slot.target
    );
}
