use serde::Serialize;

use super::{Ctx, Item, PipelineError, RoleClient, item_error, without_topics};
use crate::config::Topic;
use crate::dataset::{Appender, Id, Question, Rejected, Subtopic, read, rewrite};
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
/// with the questions already on disk and the topic's questions recorded in
/// `data/rejected.jsonl` (deleted questions), so neither comes back; a topic with
/// nothing left to fill is not seeded, so its stored questions are not embedded
/// again. A subtopic stops at `questions_per_subtopic` or after
/// `pipeline.max_retries` batches without a new question (at least one).
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
    let rejected: Vec<Rejected> = read(&ctx.files.rejected)?;
    if ctx.force {
        existing = without_topics(existing, &topics, |question| &question.topic);
        rewrite(&ctx.files.questions, &existing)?;
    }
    // Every selected subtopic is one item; a subtopic already at its target is reported
    // done at once, so progress across runs counts what earlier runs finished.
    ctx.bus.publish(Event::StageStarted {
        stage: Stage::Questions,
        total: selected_subtopics(&subtopics, &topics),
    });
    let mut filler = Filler {
        ctx,
        generator,
        out: Appender::open(&ctx.files.questions)?,
        stats: StageStats::default(),
    };
    for topic in topics {
        let mut dedup = new_dedup();
        let pending = subtopics.iter().any(|subtopic| {
            subtopic.topic == topic.name && needs_filling(subtopic, &existing, &[topic])
        });
        if pending {
            let known = seed_texts(&topic.name, &existing, &rejected);
            if !seed(ctx, topic, &known, &mut dedup, &mut filler.stats).await? {
                continue;
            }
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

/// The texts of `topic_name`'s questions already on disk, then of its rejected ones.
fn seed_texts(topic_name: &str, existing: &[Question], rejected: &[Rejected]) -> Vec<String> {
    let stored = existing
        .iter()
        .filter(|question| question.topic == topic_name)
        .map(|question| question.text.clone());
    let deleted = rejected.iter().filter_map(|record| match record {
        Rejected::Question { topic, text, .. } if topic == topic_name => Some(text.clone()),
        _ => None,
    });
    stored.chain(deleted).collect()
}

/// Records `known`, the texts `topic` must not generate again, in `dedup`. Returns
/// `false` when that failed without stopping the stage (the topic is then reported
/// as failed).
async fn seed<D: Deduplicator>(
    ctx: &Ctx<'_>,
    topic: &Topic,
    known: &[String],
    dedup: &mut D,
    stats: &mut StageStats,
) -> Result<bool, PipelineError> {
    let Err(error) = dedup.record(known).await else {
        return Ok(true);
    };
    let item = Item {
        stage: Stage::Questions,
        id: topic.name.clone(),
    };
    item_error(ctx, &item, error, stats)?;
    Ok(false)
}

/// Subtopics belonging to one of `topics`.
fn selected_subtopics(subtopics: &[Subtopic], topics: &[&Topic]) -> usize {
    subtopics
        .iter()
        .filter(|subtopic| topics.iter().any(|topic| topic.name == subtopic.topic))
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
            self.ctx.bus.publish(Event::ItemDone {
                stage: Stage::Questions,
                id: slot.item.id,
                usage: None,
            });
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
        let asked = self
            .generator
            .ask_list(&self.ctx.asking(), prompt, &slot.item, &mut self.stats)
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
