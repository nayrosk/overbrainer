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
        match result {
            Ok(completion) => save(
                ctx,
                &parent,
                &mut out,
                &mut stats,
                (question, system, completion),
            )?,
            Err(error) => {
                let item = Item {
                    stage: Stage::Answers,
                    id: question.id.to_string(),
                };
                if let Err(stop) = item_error(ctx, &item, error, &mut stats) {
                    tasks.abort_all();
                    let kept = keep_finished(ctx, &parent, &mut tasks, &mut out, &mut stats).await;
                    return Err(stopped(stop, kept, stats));
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

/// Builds the example of an answered question, records it and appends it.
fn save<C: LlmClient>(
    ctx: &Ctx<'_>,
    parent: &RoleClient<C>,
    out: &mut Appender,
    stats: &mut StageStats,
    (question, system, completion): (Question, String, Completion),
) -> Result<(), PipelineError> {
    let example = example(ctx, parent, question, system, completion);
    record(ctx, parent, &example, stats);
    out.append(&example)?;
    Ok(())
}

/// After a fatal error, waits for the aborted tasks and saves the answers that had
/// already finished, so they are not paid for twice. Cancelled tasks and failed
/// requests are left for the next run.
async fn keep_finished<C: LlmClient>(
    ctx: &Ctx<'_>,
    parent: &RoleClient<C>,
    tasks: &mut JoinSet<Outcome>,
    out: &mut Appender,
    stats: &mut StageStats,
) -> Result<(), PipelineError> {
    while let Some(joined) = tasks.join_next().await {
        if let Ok((question, system, Ok(completion))) = joined {
            save(ctx, parent, out, stats, (question, system, completion))?;
        }
    }
    Ok(())
}

/// The error of a stage stopped by the fatal provider error `stop`, once the answers
/// that had finished were saved (`kept`). When saving them failed, that disk error is
/// returned, and the provider error, which would otherwise be lost, is logged.
fn stopped(
    mut stop: PipelineError,
    kept: Result<(), PipelineError>,
    stats: StageStats,
) -> PipelineError {
    if let Err(disk) = kept {
        log_lost(&stop);
        return disk;
    }
    if let PipelineError::Llm { spent, .. } = &mut stop {
        **spent = stats;
    }
    stop
}

fn log_lost(stop: &PipelineError) {
    let cause = std::error::Error::source(stop)
        .map(|source| format!(": {source}"))
        .unwrap_or_default();
    tracing::error!("{stop}{cause}; saving the answers that had finished then failed too");
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
    mut completion: Completion,
) -> Example {
    if completion.reasoning.kind == ReasoningKind::Raw && hides_raw_reasoning(ctx, parent) {
        completion.reasoning.kind = ReasoningKind::Summary;
    }
    let mut meta = Meta {
        model: parent.model.model.clone(),
        input_tokens: completion.usage.input_tokens,
        output_tokens: completion.usage.output_tokens,
        finish_reason: completion.finish,
        reasoning_kind: completion.reasoning.kind,
        excluded: None,
    };
    let (reply, excluded) = assistant(completion, parent.model.reasoning);
    meta.excluded = excluded;
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
    messages.push(reply);
    Example {
        id: question.id,
        topic: question.topic,
        subtopic: question.subtopic,
        messages,
        meta,
    }
}

/// The assistant message of `completion` and why it cannot be trained on, if it
/// cannot. A usable answer keeps its reasoning only when it is raw, so a summary is
/// never trained on; an excluded answer keeps whatever it got, for inspection.
fn assistant(completion: Completion, reasoning_requested: bool) -> (Message, Option<Exclusion>) {
    let excluded = classify(&completion, reasoning_requested);
    let raw = completion.reasoning.kind == ReasoningKind::Raw;
    let reasoning_content = completion
        .reasoning
        .text
        .filter(|_| raw || excluded.is_some());
    let reply = Message {
        role: Role::Assistant,
        content: completion.content,
        reasoning_content,
    };
    (reply, excluded)
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
/// reasoning: the `anthropic` protocol always, and `OpenAI`, Claude or Gemini models on
/// the `openai` protocol, which hide or summarize it. The open-weight `gpt-oss` models
/// return raw reasoning and get no warning.
///
/// The answers stage never trusts reasoning from such a parent: what it reports as raw
/// is stored as a summary, so it is never trained on.
#[must_use]
pub fn raw_reasoning_warning(protocol: Protocol, model: &str) -> Option<&'static str> {
    if protocol == Protocol::Anthropic {
        return Some(
            "the anthropic protocol never returns raw reasoning: every answer will be excluded from training",
        );
    }
    let model = model.to_ascii_lowercase();
    let name = model.rsplit('/').next().unwrap_or(&model);
    if name.starts_with("gpt-oss") {
        return None;
    }
    let mut chars = name.chars();
    let o_series = chars.next() == Some('o') && chars.next().is_some_and(|c| c.is_ascii_digit());
    let hidden = model.starts_with("openai/")
        || model.starts_with("anthropic/")
        || name.starts_with("gpt-")
        || name.starts_with("claude")
        || name.starts_with("gemini")
        || o_series;
    hidden.then_some(
        "this model hides or summarizes its reasoning, so none of it is treated as raw: every answer will be excluded from training",
    )
}

/// Whether the parent is known not to return raw reasoning (see
/// [`raw_reasoning_warning`]), whatever the response claims.
fn hides_raw_reasoning<C>(ctx: &Ctx<'_>, parent: &RoleClient<C>) -> bool {
    ctx.settings
        .providers
        .get(&parent.model.provider)
        .is_some_and(|provider| {
            raw_reasoning_warning(provider.protocol, &parent.model.model).is_some()
        })
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
    use std::io;
    use std::sync::Mutex;

    use tracing_subscriber::fmt::MakeWriter;

    use super::*;
    use crate::dataset::DatasetError;
    use crate::llm::{Reasoning, Usage};

    /// Collects formatted log lines.
    #[derive(Clone, Default)]
    struct Logs(Arc<Mutex<Vec<u8>>>);

    impl Logs {
        fn text(&self) -> String {
            self.0
                .lock()
                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                .unwrap_or_default()
        }
    }

    impl io::Write for Logs {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if let Ok(mut bytes) = self.0.lock() {
                bytes.extend_from_slice(buf);
            }
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl MakeWriter<'_> for Logs {
        type Writer = Self;

        fn make_writer(&self) -> Self::Writer {
            self.clone()
        }
    }

    #[test]
    fn a_disk_error_after_a_fatal_stop_still_logs_the_provider_error() {
        let logs = Logs::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(logs.clone())
            .with_ansi(false)
            .finish();
        let stop = PipelineError::Llm {
            stage: Stage::Answers,
            source: LlmError::Status {
                status: 401,
                message: "authentication failed".to_string(),
                retry_after: None,
            },
            spent: Box::default(),
        };
        let disk = PipelineError::Dataset(DatasetError::Io {
            path: "data/answers.jsonl".into(),
            source: io::Error::other("no space left"),
        });
        let returned = tracing::subscriber::with_default(subscriber, || {
            stopped(stop, Err(disk), StageStats::default())
        });
        assert!(
            matches!(returned, PipelineError::Dataset(_)),
            "the disk error is returned"
        );
        let text = logs.text();
        assert!(text.contains("ERROR"), "{text}");
        assert!(text.contains("answers stopped"), "{text}");
        assert!(text.contains("401"), "{text}");
    }

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
            "google/gemini-3-pro",
            "gemini-2.5-flash",
        ] {
            assert!(
                raw_reasoning_warning(Protocol::Openai, model).is_some(),
                "{model}"
            );
        }
        for model in [
            "deepseek-r1",
            "qwen/qwen3-235b-a22b",
            "ollama3",
            "google/gemma-3-27b-it",
        ] {
            assert!(
                raw_reasoning_warning(Protocol::Openai, model).is_none(),
                "{model}"
            );
        }
    }

    #[test]
    fn gpt_oss_models_return_raw_reasoning() {
        for model in [
            "openai/gpt-oss-120b",
            "gpt-oss-20b",
            "groq/openai/gpt-oss-20b",
        ] {
            assert!(
                raw_reasoning_warning(Protocol::Openai, model).is_none(),
                "{model}"
            );
        }
    }

    #[test]
    fn only_raw_reasoning_is_kept_on_usable_answers() {
        use FinishReason::Stop;
        use ReasoningKind::{Raw, Redacted, Summary};
        for kind in [Summary, Redacted] {
            let (reply, excluded) = assistant(completion("a", kind, Stop), false);
            assert_eq!(excluded, Option::None, "{kind:?} stays usable");
            assert_eq!(
                reply.reasoning_content,
                Option::None,
                "{kind:?} is not trained"
            );
        }
        let (reply, excluded) = assistant(completion("a", Raw, Stop), false);
        assert_eq!(excluded, Option::None);
        assert_eq!(reply.reasoning_content.as_deref(), Some("r"));
        assert_eq!(reply.content, "a");
        for kind in [Summary, Redacted] {
            let (reply, excluded) = assistant(completion("a", kind, Stop), true);
            assert_eq!(excluded, Some(Exclusion::NoRawReasoning));
            assert_eq!(
                reply.reasoning_content.as_deref(),
                Some("r"),
                "an excluded answer keeps its reasoning for inspection"
            );
        }
    }
}
