//! Pipeline stages: subtopics, questions, answers, split.

mod answers;
mod parse;
mod questions;
mod split;
mod subtopics;

pub use answers::{answers, classify, raw_reasoning_warning};
pub use parse::string_array;
pub use questions::questions;
pub use split::{SplitClass, SplitReport, eval_size, split, split_class, stratify};
pub use subtopics::subtopics;

use crate::config::{RoleModel, Settings, Topic};
use crate::dataset::{DataFiles, DatasetError, FinishReason};
use crate::events::{Event, EventBus, Stage, StageStats};
use crate::llm::{
    Completion, CompletionRequest, LlmClient, LlmError, RetryPolicy, Usage, with_retry,
};
use crate::pricing::Price;
use crate::prompts::{PromptError, Prompts};

/// Errors that stop a stage.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// A data file cannot be read or written.
    #[error(transparent)]
    Dataset(#[from] DatasetError),
    /// A prompt template cannot be rendered.
    #[error(transparent)]
    Prompt(#[from] PromptError),
    /// A provider error that would fail every item, such as a rejected API key.
    #[error("{stage} stopped")]
    Llm {
        /// The stage that stopped.
        stage: Stage,
        /// The provider error.
        #[source]
        source: LlmError,
        /// What the stage produced and spent before it stopped.
        spent: Box<StageStats>,
    },
    /// `--topic` names a topic that is not configured.
    #[error("unknown topic `{0}`")]
    UnknownTopic(String),
    /// A request task panicked or was cancelled.
    #[error("a request task failed")]
    Task(#[source] tokio::task::JoinError),
}

/// A client bound to a role, with the role's request parameters and the model price.
#[derive(Debug)]
pub struct RoleClient<C> {
    /// The client.
    pub client: C,
    /// The role configuration (model, `max_tokens`, reasoning).
    pub model: RoleModel,
    /// Price of the model, when the provider lists it.
    pub price: Option<Price>,
}

impl<C: LlmClient> RoleClient<C> {
    /// A request carrying this role's parameters.
    #[must_use]
    pub fn request(&self, system: Option<String>, prompt: String) -> CompletionRequest {
        CompletionRequest::for_role(&self.model, system, prompt)
    }
}

/// What every stage needs: settings, files, templates, the event bus and CLI options.
#[derive(Debug, Clone, Copy)]
pub struct Ctx<'a> {
    /// Loaded settings.
    pub settings: &'a Settings,
    /// Data file paths.
    pub files: &'a DataFiles,
    /// Prompt templates.
    pub prompts: &'a Prompts,
    /// Where progress events go.
    pub bus: &'a EventBus,
    /// Only process this topic (`--topic`).
    pub topic: Option<&'a str>,
    /// Regenerate the stage output of the selected topics (`--force`).
    pub force: bool,
}

impl Ctx<'_> {
    /// The configured topics, or only the one named by `--topic`.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::UnknownTopic`] when `--topic` names no configured topic.
    pub fn topics(&self) -> Result<Vec<&Topic>, PipelineError> {
        let topics: Vec<&Topic> = self
            .settings
            .topics
            .iter()
            .filter(|topic| self.topic.is_none_or(|name| topic.name == name))
            .collect();
        match self.topic {
            Some(name) if topics.is_empty() => Err(PipelineError::UnknownTopic(name.to_string())),
            _ => Ok(topics),
        }
    }

    /// Retry policy from `pipeline.max_retries`.
    #[must_use]
    pub fn policy(&self) -> RetryPolicy {
        RetryPolicy::new(self.settings.pipeline.max_retries)
    }

    /// The retry policy and event bus a stage's item requests share.
    pub(crate) fn asking(&self) -> Asking<'_> {
        Asking {
            policy: self.policy(),
            bus: self.bus,
        }
    }
}

/// The item a request belongs to, for events.
#[derive(Debug, Clone)]
pub(crate) struct Item {
    pub(crate) stage: Stage,
    pub(crate) id: String,
}

impl Item {
    pub(crate) fn failed(&self, bus: &EventBus, error: String, retryable: bool) {
        bus.publish(Event::ItemFailed {
            stage: self.stage,
            id: self.id.clone(),
            error,
            retryable,
        });
    }
}

/// Sends `request` with retries, publishing a retryable `ItemFailed` before each wait.
pub(crate) async fn complete_with_retry<C: LlmClient>(
    client: &C,
    policy: &RetryPolicy,
    request: &CompletionRequest,
    item: &Item,
    bus: &EventBus,
) -> Result<Completion, LlmError> {
    with_retry(
        policy,
        || client.complete(request.clone()),
        |error, wait| {
            item.failed(
                bus,
                format!("{error}; retrying in {:.1}s", wait.as_secs_f64()),
                true,
            );
        },
    )
    .await
}

/// How much the token limit is raised, at most, when an answer is cut short: four
/// times the role's configured `max_tokens`.
const MAX_TOKEN_GROWTH: u32 = 4;

/// The retry policy and event bus a stage's item requests share. Built from a
/// [`Ctx`], or directly in tests.
#[derive(Clone, Copy)]
pub(crate) struct Asking<'a> {
    /// The retry policy for the request's transient failures.
    pub(crate) policy: RetryPolicy,
    /// Where failed attempts are reported.
    pub(crate) bus: &'a EventBus,
}

impl<C: LlmClient> RoleClient<C> {
    /// Asks the role for a list of strings, retrying unparseable answers up to
    /// `pipeline.max_retries` times. It asks for structured output (a JSON schema);
    /// should the provider reject that, it drops it and retries in plain text. When an
    /// answer is cut short by the token limit and nothing parses, the limit is raised
    /// for the next attempt, up to [`MAX_TOKEN_GROWTH`] times the configured one. The
    /// answer is parsed leniently by [`string_array`], which reads a JSON array (even
    /// one the limit truncated), an array wrapped in an object, or a numbered or
    /// bulleted list.
    ///
    /// Tokens of every attempt are added to `stats` and also returned, summed, as the
    /// item's own usage. Every failed attempt is reported as a retryable `ItemFailed`;
    /// the caller reports the final, non-retryable failure through [`item_error`], so a
    /// failing item is reported exactly once as final.
    pub(crate) async fn ask_list(
        &self,
        asking: &Asking<'_>,
        prompt: String,
        item: &Item,
        stats: &mut StageStats,
    ) -> Result<(Vec<String>, Usage), LlmError> {
        let bus = asking.bus;
        let base_tokens = self.model.max_tokens;
        let cap = base_tokens.saturating_mul(MAX_TOKEN_GROWTH);
        let mut max_tokens = base_tokens;
        let mut structured = true;
        let mut usage = Usage::default();
        for _ in 0..=asking.policy.max_retries {
            let mut request = self.request(None, prompt.clone());
            request.json_list = structured;
            request.max_tokens = max_tokens;
            let completion = match complete_with_retry(
                &self.client,
                &asking.policy,
                &request,
                item,
                bus,
            )
            .await
            {
                Ok(completion) => completion,
                Err(error) if structured => {
                    // The provider may not support structured output; drop it and
                    // try again in plain text before giving up on the error.
                    structured = false;
                    item.failed(
                        bus,
                        format!("{error}; retrying without structured output"),
                        true,
                    );
                    continue;
                },
                Err(error) => return Err(error),
            };
            stats.add_usage(completion.usage, self.price.as_ref());
            usage += completion.usage;
            if let Some(items) = string_array(&completion.content) {
                return Ok((items, usage));
            }
            if completion.finish == FinishReason::Length && max_tokens < cap {
                max_tokens = max_tokens.saturating_mul(2).min(cap);
                item.failed(
                    bus,
                    "the answer was cut short by the token limit; retrying with a higher limit"
                        .to_string(),
                    true,
                );
            } else {
                item.failed(bus, "the answer is not a list of strings".to_string(), true);
            }
        }
        Err(LlmError::InvalidResponse(
            "no list of strings after all retries".to_string(),
        ))
    }
}

/// Handles an item error: fatal errors stop the stage, others count as a failed item.
pub(crate) fn item_error(
    ctx: &Ctx<'_>,
    item: &Item,
    error: LlmError,
    stats: &mut StageStats,
) -> Result<(), PipelineError> {
    if error.is_fatal_for_stage() {
        return Err(PipelineError::Llm {
            stage: item.stage,
            source: error,
            spent: Box::new(stats.clone()),
        });
    }
    stats.failed += 1;
    item.failed(ctx.bus, error.to_string(), false);
    Ok(())
}

/// Removes the items of `topics` from `items` (used by `--force`).
pub(crate) fn without_topics<T>(
    items: Vec<T>,
    topics: &[&Topic],
    topic_of: impl Fn(&T) -> &str,
) -> Vec<T> {
    items
        .into_iter()
        .filter(|item| !topics.iter().any(|topic| topic.name == topic_of(item)))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::{Mutex, MutexGuard, PoisonError};

    use super::*;
    use crate::config::RoleModel;
    use crate::events::EventBus;
    use crate::llm::{Completion, LlmError, Reasoning};
    use crate::retry::RetryPolicy;

    /// Locks `mutex`, keeping the guard even if a thread panicked while holding it.
    fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
        mutex.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A client that returns scripted results and records every request it received.
    struct ScriptedClient {
        replies: Mutex<VecDeque<Result<Completion, LlmError>>>,
        seen: Mutex<Vec<CompletionRequest>>,
    }

    impl ScriptedClient {
        fn new(replies: Vec<Result<Completion, LlmError>>) -> Self {
            Self {
                replies: Mutex::new(replies.into_iter().collect()),
                seen: Mutex::new(Vec::new()),
            }
        }
    }

    impl LlmClient for ScriptedClient {
        fn complete(
            &self,
            request: CompletionRequest,
        ) -> impl std::future::Future<Output = Result<Completion, LlmError>> + Send {
            lock(&self.seen).push(request);
            let reply = lock(&self.replies)
                .pop_front()
                .unwrap_or_else(|| Err(LlmError::InvalidResponse("no more replies".into())));
            std::future::ready(reply)
        }

        fn embed(
            &self,
            _inputs: &[String],
        ) -> impl std::future::Future<Output = Result<Vec<Vec<f32>>, LlmError>> + Send {
            std::future::ready(Err(LlmError::Unsupported("embeddings")))
        }
    }

    fn reply(content: &str, finish: FinishReason) -> Completion {
        Completion {
            content: content.to_string(),
            reasoning: Reasoning::none(),
            usage: Usage::default(),
            finish,
        }
    }

    fn role(client: ScriptedClient, max_tokens: u32) -> RoleClient<ScriptedClient> {
        RoleClient {
            client,
            model: RoleModel {
                provider: "p".into(),
                model: "m".into(),
                reasoning: false,
                max_tokens,
                temperature: None,
                reasoning_effort: None,
                thinking_budget: None,
            },
            price: None,
        }
    }

    fn item() -> Item {
        Item {
            stage: Stage::Subtopics,
            id: "t".into(),
        }
    }

    #[tokio::test]
    async fn structured_output_is_dropped_when_the_provider_rejects_it() -> Result<(), LlmError> {
        let bus = EventBus::new();
        let asking = Asking {
            policy: RetryPolicy::new(5),
            bus: &bus,
        };
        let client = ScriptedClient::new(vec![
            Err(LlmError::Status {
                status: 400,
                message: "response_format not supported".into(),
                retry_after: None,
            }),
            Ok(reply(r#"["a", "b"]"#, FinishReason::Stop)),
        ]);
        let role = role(client, 16_384);
        let mut stats = StageStats::default();
        let (items, _usage) = role
            .ask_list(&asking, "p".into(), &item(), &mut stats)
            .await?;
        assert_eq!(items, vec!["a".to_string(), "b".to_string()]);
        let seen = lock(&role.client.seen);
        assert!(
            seen[0].json_list,
            "the first attempt asks for structured output"
        );
        assert!(!seen[1].json_list, "the retry drops it");
        Ok(())
    }

    #[tokio::test]
    async fn a_truncated_answer_raises_the_token_limit_for_the_next_attempt() -> Result<(), LlmError>
    {
        let bus = EventBus::new();
        let asking = Asking {
            policy: RetryPolicy::new(5),
            bus: &bus,
        };
        let client = ScriptedClient::new(vec![
            Ok(reply(r#"["unterminated"#, FinishReason::Length)),
            Ok(reply(r#"["a"]"#, FinishReason::Stop)),
        ]);
        let role = role(client, 100);
        let mut stats = StageStats::default();
        let (items, _usage) = role
            .ask_list(&asking, "p".into(), &item(), &mut stats)
            .await?;
        assert_eq!(items, vec!["a".to_string()]);
        let seen = lock(&role.client.seen);
        assert_eq!(seen[0].max_tokens, 100);
        assert_eq!(
            seen[1].max_tokens, 200,
            "the limit doubled after the cut-off"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_numbered_list_answer_is_accepted() -> Result<(), LlmError> {
        let bus = EventBus::new();
        let asking = Asking {
            policy: RetryPolicy::new(5),
            bus: &bus,
        };
        let client = ScriptedClient::new(vec![Ok(reply("1. foo\n2. bar", FinishReason::Stop))]);
        let role = role(client, 16_384);
        let mut stats = StageStats::default();
        let (items, _usage) = role
            .ask_list(&asking, "p".into(), &item(), &mut stats)
            .await?;
        assert_eq!(items, vec!["foo".to_string(), "bar".to_string()]);
        Ok(())
    }
}
