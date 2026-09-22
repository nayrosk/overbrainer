//! LLM clients for the `openai` and `anthropic` wire protocols.

mod anthropic;
mod client;
mod error;
mod http;
mod openai;
mod retry;
mod setup;
mod types;

pub use anthropic::AnthropicClient;
pub use client::ProtocolClient;
pub use error::LlmError;
pub use openai::OpenAiClient;
pub use retry::{RetryPolicy, with_retry};
pub use setup::{SetupError, connect};
pub use types::{Completion, CompletionRequest, Reasoning, Usage};

use std::future::Future;

/// A chat model bound to one provider and one model ID.
///
/// Futures are `Send` so calls can run in spawned tasks.
pub trait LlmClient: Send + Sync {
    /// Sends one completion request.
    ///
    /// # Errors
    ///
    /// Returns an [`LlmError`] when the request fails or the response is invalid.
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<Completion, LlmError>> + Send;

    /// Embeds `inputs`, returning one vector per input, in order.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::Unsupported`] when the protocol has no embeddings, or
    /// another [`LlmError`] when the request fails.
    fn embed(
        &self,
        inputs: &[String],
    ) -> impl Future<Output = Result<Vec<Vec<f32>>, LlmError>> + Send;
}
