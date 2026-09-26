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
    /// an effort is configured, sets `output_config.effort`. If the role sets a
    /// `thinking_budget`, it asks for a fixed thinking budget instead and
    /// omits `output_config` entirely.
    ///
    /// # Errors
    ///
    /// Returns an [`LlmError`] when the request fails or the response is invalid.
    pub async fn complete(&self, request: &CompletionRequest) -> Result<Completion, LlmError> {
        let (thinking, output_config) = thinking_config(request);
        let body = MessagesRequest {
            model: &self.model,
            max_tokens: request.max_tokens,
            system: request.system.as_deref(),
            messages: vec![WireMessage {
                role: "user",
                content: &request.prompt,
            }],
            temperature: request.temperature,
            thinking,
            output_config,
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

/// The `thinking` request block.
///
/// `Adaptive` lets the model decide when and how much to think and is required by
/// Claude Sonnet 5, Opus 5, Opus 4.8, Opus 4.7 and Fable 5.x, which reject a fixed
/// budget. `Enabled` asks for a fixed token budget and is required by Claude Opus 4.5,
/// Sonnet 4.5 and Haiku 4.5, which reject adaptive thinking.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Thinking {
    Adaptive,
    Enabled { budget_tokens: u32 },
}

#[derive(Serialize)]
struct OutputConfig {
    effort: &'static str,
}

/// Builds the `thinking` and `output_config` fields of a message request from
/// `request.reasoning`, `request.effort` and `request.thinking_budget`.
///
/// Without reasoning, both are `None`. With reasoning and no `thinking_budget`, sends
/// adaptive thinking plus `output_config.effort` when an effort is configured. With a
/// `thinking_budget`, sends a fixed-budget `enabled` thinking block instead and always
/// omits `output_config`, since the two are mutually exclusive on the wire.
fn thinking_config(request: &CompletionRequest) -> (Option<Thinking>, Option<OutputConfig>) {
    if !request.reasoning {
        return (None, None);
    }
    match request.thinking_budget {
        Some(budget_tokens) => (Some(Thinking::Enabled { budget_tokens }), None),
        None => (
            Some(Thinking::Adaptive),
            request.effort.map(|effort| OutputConfig {
                effort: effort.as_str(),
            }),
        ),
    }
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
    use crate::config::Effort;

    fn blocks(value: &serde_json::Value) -> Result<(String, Reasoning), serde_json::Error> {
        Ok(split_blocks(serde_json::from_value(value.clone())?))
    }

    fn reasoning_request(
        effort: Option<Effort>,
        thinking_budget: Option<u32>,
    ) -> CompletionRequest {
        CompletionRequest {
            system: None,
            prompt: "p".into(),
            max_tokens: 4096,
            temperature: None,
            reasoning: true,
            effort,
            thinking_budget,
        }
    }

    #[test]
    fn adaptive_thinking_is_the_default() -> Result<(), serde_json::Error> {
        let request = reasoning_request(Some(Effort::High), None);
        let (thinking, output_config) = thinking_config(&request);
        assert_eq!(serde_json::to_value(thinking)?, json!({"type": "adaptive"}));
        assert_eq!(
            serde_json::to_value(output_config)?,
            json!({"effort": "high"})
        );
        Ok(())
    }

    #[test]
    fn a_thinking_budget_sends_an_enabled_block_and_no_output_config()
    -> Result<(), serde_json::Error> {
        let request = reasoning_request(None, Some(2048));
        let (thinking, output_config) = thinking_config(&request);
        assert_eq!(
            serde_json::to_value(thinking)?,
            json!({"type": "enabled", "budget_tokens": 2048})
        );
        assert!(output_config.is_none());
        Ok(())
    }

    #[test]
    fn a_thinking_budget_wins_over_an_effort() -> Result<(), serde_json::Error> {
        let request = reasoning_request(Some(Effort::High), Some(2048));
        let (thinking, output_config) = thinking_config(&request);
        assert_eq!(
            serde_json::to_value(thinking)?,
            json!({"type": "enabled", "budget_tokens": 2048})
        );
        assert!(output_config.is_none());
        Ok(())
    }

    #[test]
    fn no_reasoning_means_no_thinking_block() {
        let request = CompletionRequest {
            system: None,
            prompt: "p".into(),
            max_tokens: 4096,
            temperature: None,
            reasoning: false,
            effort: None,
            thinking_budget: None,
        };
        let (thinking, output_config) = thinking_config(&request);
        assert!(thinking.is_none());
        assert!(output_config.is_none());
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
