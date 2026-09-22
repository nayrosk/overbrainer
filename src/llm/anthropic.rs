use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use super::http::{Endpoint, sensitive};
use super::{Completion, CompletionRequest, LlmError, Reasoning, Usage};
use crate::dataset::{FinishReason, ReasoningKind};

/// Version header required by the Messages API.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Client for the `anthropic` protocol (`/messages`).
#[derive(Debug, Clone)]
pub struct AnthropicClient {
    endpoint: Endpoint,
    model: String,
}

impl AnthropicClient {
    /// Builds a client for `model` at `base_url`, sending `x-api-key` when `api_key` is
    /// set and `anthropic-version: 2023-06-01` always.
    ///
    /// # Errors
    ///
    /// Returns [`LlmError::InvalidApiKey`] if the key cannot be sent as a header, or
    /// [`LlmError::Client`] if the HTTP client cannot be built.
    pub fn new(
        base_url: &str,
        api_key: Option<&SecretString>,
        model: &str,
        timeout: Duration,
    ) -> Result<Self, LlmError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
        if let Some(key) = api_key {
            headers.insert(
                HeaderName::from_static("x-api-key"),
                sensitive(key.expose_secret())?,
            );
        }
        Ok(Self {
            endpoint: Endpoint::new(base_url, headers, timeout)?,
            model: model.to_string(),
        })
    }

    /// Sends one message request. With reasoning, asks for adaptive thinking and, when
    /// an effort is configured, sets `output_config.effort`.
    ///
    /// # Errors
    ///
    /// Returns an [`LlmError`] when the request fails or the response is invalid.
    pub async fn complete(&self, request: &CompletionRequest) -> Result<Completion, LlmError> {
        let body = MessagesRequest {
            model: &self.model,
            max_tokens: request.max_tokens,
            system: request.system.as_deref(),
            messages: vec![WireMessage {
                role: "user",
                content: &request.prompt,
            }],
            temperature: request.temperature,
            thinking: request.reasoning.then_some(Thinking { kind: "adaptive" }),
            output_config: request.effort.filter(|_| request.reasoning).map(|effort| {
                OutputConfig {
                    effort: effort.as_str(),
                }
            }),
        };
        let response: MessagesResponse = self.endpoint.post("messages", &body).await?;
        let usage = response.usage.map_or_else(Usage::default, |usage| Usage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
        });
        let finish = stop_reason(response.stop_reason.as_deref());
        let (content, reasoning) = split_blocks(response.content);
        Ok(Completion {
            content,
            reasoning,
            usage,
            finish,
        })
    }

    /// Fetches `GET /models?detailed=true` as raw JSON.
    ///
    /// # Errors
    ///
    /// Returns an [`LlmError`] when the request fails.
    pub async fn models(&self) -> Result<serde_json::Value, LlmError> {
        self.endpoint.get("models?detailed=true").await
    }
}

#[derive(Serialize)]
struct MessagesRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    messages: Vec<WireMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<Thinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<OutputConfig>,
}

#[derive(Serialize)]
struct WireMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Serialize)]
struct Thinking {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Serialize)]
struct OutputConfig {
    effort: &'static str,
}

#[derive(Deserialize)]
struct MessagesResponse {
    content: Vec<Block>,
    stop_reason: Option<String>,
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Block {
    Text {
        text: String,
    },
    Thinking {
        #[serde(default)]
        thinking: String,
    },
    RedactedThinking,
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct WireUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
}

fn stop_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("end_turn" | "stop_sequence") => FinishReason::Stop,
        Some("max_tokens") => FinishReason::Length,
        Some("refusal") => FinishReason::Refusal,
        _ => FinishReason::Other,
    }
}

/// Joins text blocks into the answer. Thinking text is a provider summary, never the
/// raw chain of thought; empty thinking or `redacted_thinking` means `Redacted`.
fn split_blocks(blocks: Vec<Block>) -> (String, Reasoning) {
    let mut content = String::new();
    let mut thoughts = Vec::new();
    let mut hidden = false;
    for block in blocks {
        match block {
            Block::Text { text } => content.push_str(&text),
            Block::Thinking { thinking } if thinking.trim().is_empty() => hidden = true,
            Block::Thinking { thinking } => thoughts.push(thinking),
            Block::RedactedThinking => hidden = true,
            Block::Other => {},
        }
    }
    let reasoning = if !thoughts.is_empty() {
        Reasoning {
            text: Some(thoughts.join("\n")),
            kind: ReasoningKind::Summary,
        }
    } else if hidden {
        Reasoning {
            text: None,
            kind: ReasoningKind::Redacted,
        }
    } else {
        Reasoning::none()
    };
    (content, reasoning)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn blocks(value: &serde_json::Value) -> Result<(String, Reasoning), serde_json::Error> {
        Ok(split_blocks(serde_json::from_value(value.clone())?))
    }

    #[test]
    fn thinking_text_is_a_summary() -> Result<(), serde_json::Error> {
        let (content, reasoning) = blocks(&json!([
            {"type": "thinking", "thinking": "Consider moves.", "signature": "sig"},
            {"type": "text", "text": "Borrow it."}
        ]))?;
        assert_eq!(content, "Borrow it.");
        assert_eq!(reasoning.text.as_deref(), Some("Consider moves."));
        assert_eq!(reasoning.kind, ReasoningKind::Summary);
        Ok(())
    }

    #[test]
    fn empty_or_redacted_thinking_is_redacted() -> Result<(), serde_json::Error> {
        for value in [
            json!([{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": "a"}]),
            json!([{"type": "redacted_thinking", "data": "opaque"}, {"type": "text", "text": "a"}]),
        ] {
            let (_, reasoning) = blocks(&value)?;
            assert_eq!(
                reasoning,
                Reasoning {
                    text: None,
                    kind: ReasoningKind::Redacted
                }
            );
        }
        Ok(())
    }

    #[test]
    fn text_only_has_no_reasoning() -> Result<(), serde_json::Error> {
        let (content, reasoning) = blocks(&json!([
            {"type": "text", "text": "a"},
            {"type": "tool_use", "id": "t", "name": "n", "input": {}},
            {"type": "text", "text": "b"}
        ]))?;
        assert_eq!(content, "ab");
        assert_eq!(reasoning, Reasoning::none());
        Ok(())
    }

    #[test]
    fn stop_reasons_map() {
        assert_eq!(stop_reason(Some("end_turn")), FinishReason::Stop);
        assert_eq!(stop_reason(Some("stop_sequence")), FinishReason::Stop);
        assert_eq!(stop_reason(Some("max_tokens")), FinishReason::Length);
        assert_eq!(stop_reason(Some("refusal")), FinishReason::Refusal);
        assert_eq!(stop_reason(Some("pause_turn")), FinishReason::Other);
    }
}
