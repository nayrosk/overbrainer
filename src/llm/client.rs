use std::time::Duration;

use secrecy::SecretString;

use super::{AnthropicClient, Completion, CompletionRequest, LlmClient, LlmError, OpenAiClient};
use crate::config::Protocol;

/// A client for one of the two wire protocols.
#[derive(Debug, Clone)]
pub enum ProtocolClient {
    /// `openai` protocol.
    OpenAi(OpenAiClient),
    /// `anthropic` protocol.
    Anthropic(AnthropicClient),
}

impl ProtocolClient {
    /// Builds the client matching `protocol`.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::InvalidApiKey`] if the key cannot be sent as a header, or
    /// [`LlmError::Client`] if the HTTP client cannot be built.
    pub fn new(
        protocol: Protocol,
        base_url: &str,
        api_key: Option<&SecretString>,
        model: &str,
        timeout: Duration,
    ) -> Result<Self, LlmError> {
        Ok(match protocol {
            Protocol::Openai => Self::OpenAi(OpenAiClient::new(base_url, api_key, model, timeout)?),
            Protocol::Anthropic => {
                Self::Anthropic(AnthropicClient::new(base_url, api_key, model, timeout)?)
            },
        })
    }

    /// Raw `GET /models?detailed=true` answer, used for pricing.
    ///
    /// # Errors
    ///
    /// Returns an [`LlmError`] when the request fails.
    pub async fn models(&self) -> Result<serde_json::Value, LlmError> {
        match self {
            Self::OpenAi(client) => client.models().await,
            Self::Anthropic(client) => client.models().await,
        }
    }
}

impl LlmClient for ProtocolClient {
    async fn complete(&self, request: CompletionRequest) -> Result<Completion, LlmError> {
        match self {
            Self::OpenAi(client) => client.complete(&request).await,
            Self::Anthropic(client) => client.complete(&request).await,
        }
    }

    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        match self {
            Self::OpenAi(client) => client.embed(inputs).await,
            Self::Anthropic(_) => Err(LlmError::Unsupported("embeddings")),
        }
    }
}
