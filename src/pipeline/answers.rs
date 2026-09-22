use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::Serialize;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use super::{
    Ctx, Item, PipelineError, RoleClient, complete_with_retry, item_error, without_topics,
};
use crate::config::{Protocol, Topic};
use crate::dataset::{
    Appender, Example, Exclusion, FinishReason, Id, Message, Meta, Question, ReasoningKind, Role,
    read, rewrite,
};
use crate::events::{Event, EventBus, Stage, StageStats};
use crate::llm::{Completion, LlmClient, LlmError, RetryPolicy};
use crate::prompts;

#[derive(Serialize)]
struct Context<'a> {
    topic: &'a str,
    description: Option<&'a str>,
}

/// Asks the parent to answer every question of the selected topics that has no answer
/// yet, writing `data/answers.jsonl` one line per completed request.
///
/// Requests run concurrently, at most `pipeline.concurrency` at a time, each with
/// retries. Unusable answers are kept with `meta.excluded` set (see [`classify`]).
/// A failed question is reported and left for the next run; an error that would fail
/// every request (such as a rejected key) stops the stage.
///
/// # Errors
///
/// Returns a [`PipelineError`] when a file or template fails, when a task panics, or
/// when the provider rejects the requests.
pub async fn answers<C: LlmClient + 'static>(
    ctx: &Ctx<'_>,
    parent: Arc<RoleClient<C>>,
) -> Result<StageStats, PipelineError> {
    let topics = ctx.topics()?;
    let questions: Vec<Question> = read(&ctx.files.questions)?;
    let mut existing: Vec<Example> = read(&ctx.files.answers)?;
    if ctx.force {
        existing = without_topics(existing, &topics, |example| &example.topic);
        rewrite(&ctx.files.answers, &existing)?;
    }
    warn_about_reasoning(ctx, &parent);
    let answered: BTreeSet<&Id> = existing.iter().map(|example| &example.id).collect();
    let selected: Vec<&Question> = questions
        .iter()
        .filter(|question| topics.iter().any(|topic| topic.name == question.topic))
        .collect();
    let pending: Vec<Question> = selected
        .iter()
        .filter(|question| !answered.contains(&question.id))
        .map(|question| (*question).clone())
        .collect();
    let mut stats = StageStats {
        skipped: selected.len() - pending.len(),
        ..StageStats::default()
    };
    ctx.bus.publish(Event::StageStarted {
        stage: Stage::Answers,
        total: pending.len(),
    });
    let systems = system_prompts(ctx, &topics)?;
    let mut tasks = spawn_all(ctx, &parent, pending, &systems);
    let mut out = Appender::open(&ctx.files.answers)?;
    while let Some(joined) = tasks.join_next().await {
        let (question, system, result) = joined.map_err(PipelineError::Task)?;
        let item = Item {
            stage: Stage::Answers,
            id: question.id.to_string(),
        };
        match result {
            Ok(completion) => {
                let example = example(ctx, &parent, question, system, completion);
                record(ctx, &parent, &example, &mut stats);
                out.append(&example)?;
            },
            Err(error) => {
                if let Err(stop) = item_error(ctx, &item, error, &mut stats) {
                    tasks.abort_all();
                    return Err(stop);
                }
            },
        }
    }
    ctx.bus.publish(Event::StageFinished {
        stage: Stage::Answers,
        stats: stats.clone(),
    });
    Ok(stats)
}

type Outcome = (Question, String, Result<Completion, LlmError>);

/// Spawns one task per question. Each task waits for a semaphore permit, so at most
/// `pipeline.concurrency` requests are in flight.
fn spawn_all<C: LlmClient + 'static>(
    ctx: &Ctx<'_>,
    parent: &Arc<RoleClient<C>>,
    pending: Vec<Question>,
    systems: &BTreeMap<String, String>,
) -> JoinSet<Outcome> {
    let semaphore = Arc::new(Semaphore::new(ctx.settings.pipeline.concurrency));
    let policy = ctx.policy();
    let mut tasks = JoinSet::new();
    for question in pending {
        let system = systems.get(&question.topic).cloned().unwrap_or_default();
        tasks.spawn(ask(
            Arc::clone(parent),
            Arc::clone(&semaphore),
            policy,
            ctx.bus.clone(),
            (question, system),
        ));
    }
    tasks
}

async fn ask<C: LlmClient>(
    parent: Arc<RoleClient<C>>,
    semaphore: Arc<Semaphore>,
    policy: RetryPolicy,
    bus: EventBus,
    (question, system): (Question, String),
) -> Outcome {
    let item = Item {
        stage: Stage::Answers,
        id: question.id.to_string(),
    };
    let Ok(_permit) = semaphore.acquire_owned().await else {
        let error = LlmError::InvalidResponse("the request queue was closed".to_string());
        return (question, system, Err(error));
    };
    let request = parent.request(Some(system.clone()), question.text.clone());
    let result = complete_with_retry(&parent.client, &policy, &request, &item, &bus).await;
    (question, system, result)
}

/// The system prompt of each topic, rendered once.
fn system_prompts(
    ctx: &Ctx<'_>,
    topics: &[&Topic],
) -> Result<BTreeMap<String, String>, PipelineError> {
    topics
        .iter()
        .map(|topic| {
            let system = ctx.prompts.render(
                prompts::ANSWER_SYSTEM,
                Context {
                    topic: &topic.name,
                    description: topic.description.as_deref(),
                },
            )?;
            Ok((topic.name.clone(), system))
        })
        .collect()
}

fn example<C: LlmClient>(
    ctx: &Ctx<'_>,
    parent: &RoleClient<C>,
    question: Question,
    system: String,
    completion: Completion,
) -> Example {
    let excluded = classify(&completion, parent.model.reasoning);
    let mut messages = Vec::with_capacity(3);
    if ctx.settings.pipeline.include_system_prompt {
        messages.push(Message {
            role: Role::System,
            content: system,
            reasoning_content: None,
        });
    }
    messages.push(Message {
        role: Role::User,
        content: question.text,
        reasoning_content: None,
    });
    messages.push(Message {
        role: Role::Assistant,
        content: completion.content,
        reasoning_content: completion.reasoning.text,
    });
    Example {
        id: question.id,
        topic: question.topic,
        subtopic: question.subtopic,
        messages,
        meta: Meta {
            model: parent.model.model.clone(),
            input_tokens: completion.usage.input_tokens,
            output_tokens: completion.usage.output_tokens,
            finish_reason: completion.finish,
            reasoning_kind: completion.reasoning.kind,
            excluded,
        },
    }
}

fn record<C: LlmClient>(
    ctx: &Ctx<'_>,
    parent: &RoleClient<C>,
    example: &Example,
    stats: &mut StageStats,
) {
    let usage = crate::llm::Usage {
        input_tokens: example.meta.input_tokens,
        output_tokens: example.meta.output_tokens,
    };
    stats.add_usage(usage, parent.price.as_ref());
    if example.meta.excluded.is_some() {
        stats.excluded += 1;
    } else {
        stats.done += 1;
    }
    ctx.bus.publish(Event::ItemDone {
        stage: Stage::Answers,
        id: example.id.to_string(),
        usage: Some(usage),
    });
}

/// Why an answer cannot be used for training, if it cannot: hit the token limit,
/// refused, empty, or (when reasoning was requested) without raw reasoning.
#[must_use]
pub fn classify(completion: &Completion, reasoning_requested: bool) -> Option<Exclusion> {
    match completion.finish {
        FinishReason::Length => return Some(Exclusion::Truncated),
        FinishReason::Refusal | FinishReason::ContentFilter => return Some(Exclusion::Refused),
        FinishReason::Stop | FinishReason::Other => {},
    }
    if completion.content.trim().is_empty() {
        return Some(Exclusion::Empty);
    }
    (reasoning_requested && completion.reasoning.kind != ReasoningKind::Raw)
        .then_some(Exclusion::NoRawReasoning)
}

/// Warning shown when reasoning is requested from a parent known not to return raw
/// reasoning: the `anthropic` protocol always, and `OpenAI` or Claude models on the
/// `openai` protocol, which hide or summarize it.
#[must_use]
pub fn raw_reasoning_warning(protocol: Protocol, model: &str) -> Option<&'static str> {
    if protocol == Protocol::Anthropic {
        return Some(
            "the anthropic protocol never returns raw reasoning: every answer will be excluded from training",
        );
    }
    let model = model.to_ascii_lowercase();
    let name = model.rsplit('/').next().unwrap_or(&model);
    let mut chars = name.chars();
    let o_series = chars.next() == Some('o') && chars.next().is_some_and(|c| c.is_ascii_digit());
    let hidden = model.starts_with("openai/")
        || model.starts_with("anthropic/")
        || name.starts_with("gpt-")
        || name.starts_with("claude")
        || o_series;
    hidden.then_some(
        "this model hides or summarizes its reasoning: answers without raw reasoning will be excluded from training",
    )
}

fn warn_about_reasoning<C>(ctx: &Ctx<'_>, parent: &RoleClient<C>) {
    let Some(provider) = ctx.settings.providers.get(&parent.model.provider) else {
        return;
    };
    if parent.model.reasoning
        && let Some(warning) = raw_reasoning_warning(provider.protocol, &parent.model.model)
    {
        tracing::warn!("parent {}: {warning}", parent.model.model);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{Reasoning, Usage};

    fn completion(content: &str, kind: ReasoningKind, finish: FinishReason) -> Completion {
        Completion {
            content: content.to_string(),
            reasoning: Reasoning {
                text: (kind != ReasoningKind::None).then(|| "r".to_string()),
                kind,
            },
            usage: Usage::default(),
            finish,
        }
    }

    #[test]
    fn exclusion_reasons() {
        use FinishReason::{ContentFilter, Length, Refusal, Stop};
        use ReasoningKind::{None, Raw, Summary};
        assert_eq!(classify(&completion("a", Raw, Stop), true), Option::None);
        assert_eq!(
            classify(&completion("a", Raw, Length), true),
            Some(Exclusion::Truncated)
        );
        assert_eq!(
            classify(&completion("", None, Refusal), true),
            Some(Exclusion::Refused)
        );
        assert_eq!(
            classify(&completion("a", None, ContentFilter), false),
            Some(Exclusion::Refused)
        );
        assert_eq!(
            classify(&completion("  ", Raw, Stop), true),
            Some(Exclusion::Empty)
        );
        assert_eq!(
            classify(&completion("a", Summary, Stop), true),
            Some(Exclusion::NoRawReasoning)
        );
        assert_eq!(classify(&completion("a", None, Stop), false), Option::None);
    }

    #[test]
    fn warnings_for_models_without_raw_reasoning() {
        assert!(raw_reasoning_warning(Protocol::Anthropic, "claude-opus-5").is_some());
        for model in [
            "openai/gpt-6",
            "gpt-5-mini",
            "o3",
            "anthropic/claude-sonnet-5",
        ] {
            assert!(
                raw_reasoning_warning(Protocol::Openai, model).is_some(),
                "{model}"
            );
        }
        for model in ["deepseek-r1", "qwen/qwen3-235b-a22b", "ollama3"] {
            assert!(
                raw_reasoning_warning(Protocol::Openai, model).is_none(),
                "{model}"
            );
        }
    }
}
