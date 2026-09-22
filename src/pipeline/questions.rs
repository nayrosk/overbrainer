use serde::Serialize;

use super::{Ctx, Item, PipelineError, RoleClient, ask_list, item_error, without_topics};
use crate::config::Topic;
use crate::dataset::{Appender, Id, Question, Subtopic, read, rewrite};
use crate::dedup::Deduplicator;
use crate::events::{Event, Stage, StageStats};
use crate::llm::{LlmClient, LlmError};
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
/// Batches run one after the other because each depends on the previous ones.
///
/// # Errors
///
/// Returns a [`PipelineError`] when a file or template fails, when deduplication
/// fails, or when the provider rejects the requests.
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
    let total = subtopics
        .iter()
        .filter(|subtopic| topics.iter().any(|topic| topic.name == subtopic.topic))
        .count();
    ctx.bus.publish(Event::StageStarted {
        stage: Stage::Questions,
        total,
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
        dedup.record(&known).await.map_err(stage_error)?;
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

fn stage_error(source: LlmError) -> PipelineError {
    PipelineError::Llm {
        stage: Stage::Questions,
        source,
    }
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
        while slot.accepted.len() < slot.target && stalls < patience {
            let Some(added) = self.batch(&slot, dedup).await? else {
                return Ok(());
            };
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
            usage: None,
        });
        Ok(())
    }

    /// Generates, deduplicates and writes one batch. Returns the accepted texts, or
    /// `None` when the subtopic failed and was reported.
    async fn batch<D: Deduplicator>(
        &mut self,
        slot: &Slot<'_>,
        dedup: &mut D,
    ) -> Result<Option<Vec<String>>, PipelineError> {
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
        let candidates = match asked {
            Ok(candidates) => candidates,
            Err(error) => {
                item_error(self.ctx, &slot.item, error, &mut self.stats)?;
                return Ok(None);
            },
        };
        let candidates = candidates.into_iter().take(count).collect();
        let added = dedup.admit(candidates).await.map_err(stage_error)?;
        for text in &added {
            self.out.append(&Question {
                id: Id::question(&slot.subtopic.id, text),
                topic: slot.topic.name.clone(),
                subtopic_id: slot.subtopic.id.clone(),
                subtopic: slot.subtopic.name.clone(),
                text: text.clone(),
            })?;
        }
        Ok(Some(added))
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
