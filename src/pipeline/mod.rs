//! Pipeline stages: subtopics, questions, answers, split.

mod answers;
mod parse;
mod questions;
mod split;
mod subtopics;

pub use answers::{answers, classify, raw_reasoning_warning};
pub use parse::string_array;
pub use questions::questions;
pub use split::{SplitReport, split, stratify};
pub use subtopics::subtopics;

use crate::config::{RoleModel, Settings, Topic};
use crate::dataset::{DataFiles, DatasetError};
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

/// Asks `role` for a JSON array of strings, retrying unparseable answers up to
/// `pipeline.max_retries` times. Tokens of every attempt are added to `stats` and also
/// returned, summed, as the item's own usage. Every parse failure is reported as a
/// retryable `ItemFailed`; the caller reports the final, non-retryable failure through
/// [`item_error`], so a failing item is reported exactly once as final.
pub(crate) async fn ask_list<C: LlmClient>(
    ctx: &Ctx<'_>,
    role: &RoleClient<C>,
    prompt: String,
    item: &Item,
    stats: &mut StageStats,
) -> Result<(Vec<String>, Usage), LlmError> {
    let policy = ctx.policy();
    let request = role.request(None, prompt);
    let mut usage = Usage::default();
    for _ in 0..=policy.max_retries {
        let completion =
            complete_with_retry(&role.client, &policy, &request, item, ctx.bus).await?;
        stats.add_usage(completion.usage, role.price.as_ref());
        usage += completion.usage;
        if let Some(items) = string_array(&completion.content) {
            return Ok((items, usage));
        }
        item.failed(
            ctx.bus,
            "the answer is not a JSON array of strings".to_string(),
            true,
        );
    }
    Err(LlmError::InvalidResponse(
        "no JSON array of strings after all retries".to_string(),
    ))
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
