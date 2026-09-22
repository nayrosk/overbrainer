use std::time::Duration;

use reqwest::header::{AUTHORIZATION, HeaderMap};
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use super::http::{Endpoint, sensitive};
use super::{Completion, CompletionRequest, LlmError, Reasoning, Usage};
use crate::config::Effort;
use crate::dataset::{FinishReason, ReasoningKind};

/// Client for the `openai` protocol (`/chat/completions`, `/embeddings`).
#[derive(Debug, Clone)]
pub struct OpenAiClient {
    endpoint: Endpoint,
    model: String,
}

impl OpenAiClient {
    /// Builds a client for `model` at `base_url`, sending `Authorization: Bearer` when
    /// `api_key` is set.
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
        if let Some(key) = api_key {
            headers.insert(
                AUTHORIZATION,
                sensitive(&format!("Bearer {}", key.expose_secret()))?,
            );
        }
        Ok(Self {
            endpoint: Endpoint::new(base_url, headers, timeout)?,
            model: model.to_string(),
        })
    }

    /// Sends one chat completion.
    ///
    /// # Errors
    ///
    /// Returns an [`LlmError`] when the request fails or the response has no choice.
    pub async fn complete(&self, request: &CompletionRequest) -> Result<Completion, LlmError> {
        let mut messages = Vec::with_capacity(2);
        if let Some(system) = &request.system {
            messages.push(WireMessage {
                role: "system",
                content: system,
            });
        }
        messages.push(WireMessage {
            role: "user",
            content: &request.prompt,
        });
        let body = ChatRequest {
            model: &self.model,
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            reasoning: request.reasoning.then(|| ReasoningParam {
                effort: request.effort.unwrap_or(Effort::Medium).as_str(),
            }),
        };
        let response: ChatResponse = self.endpoint.post("chat/completions", &body).await?;
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::InvalidResponse("no choices".to_string()))?;
        let refused = choice
            .message
            .refusal
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty());
        let finish = if refused {
            FinishReason::Refusal
        } else {
            finish_reason(choice.finish_reason.as_deref())
        };
        let (content, reasoning) = split_message(choice.message);
        let usage = response.usage.map_or_else(Usage::default, |usage| Usage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        });
        Ok(Completion {
            content,
            reasoning,
            usage,
            finish,
        })
    }

    /// Embeds `inputs` with `POST /embeddings`.
    ///
    /// # Errors
    ///
    /// Returns an [`LlmError`] when the request fails or the response does not hold one
    /// vector per input.
    pub async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let body = EmbeddingRequest {
            model: &self.model,
            input: inputs,
            encoding_format: "float",
        };
        let mut response: EmbeddingResponse = self.endpoint.post("embeddings", &body).await?;
        if response.data.len() != inputs.len() {
            return Err(LlmError::InvalidResponse(format!(
                "{} embeddings for {} inputs",
                response.data.len(),
                inputs.len()
            )));
        }
        response.data.sort_by_key(|item| item.index);
        Ok(response
            .data
            .into_iter()
            .map(|item| item.embedding)
            .collect())
    }

    /// Fetches `GET /models?detailed=true` as raw JSON (the query adds prices on
    /// `NanoGPT` and is ignored elsewhere).
    ///
    /// # Errors
    ///
    /// Returns an [`LlmError`] when the request fails.
    pub async fn models(&self) -> Result<serde_json::Value, LlmError> {
        self.endpoint.get("models?detailed=true").await
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<WireMessage<'a>>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ReasoningParam>,
}

#[derive(Serialize)]
struct WireMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Serialize)]
struct ReasoningParam {
    effort: &'static str,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: Option<String>,
    refusal: Option<String>,
    reasoning: Option<String>,
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning_details: Vec<ReasoningDetail>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ReasoningDetail {
    #[serde(rename = "reasoning.text")]
    Text { text: Option<String> },
    #[serde(rename = "reasoning.summary")]
    Summary { summary: Option<String> },
    #[serde(rename = "reasoning.encrypted")]
    Encrypted,
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    encoding_format: &'static str,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    index: usize,
    embedding: Vec<f32>,
}

fn finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::Length,
        Some("content_filter") => FinishReason::ContentFilter,
        _ => FinishReason::Other,
    }
}

/// Separates the answer from the reasoning, in this order: `message.reasoning`,
/// `message.reasoning_content`, `message.reasoning_details[]`, then an inline
/// `<think>...</think>` block at the start of the content. When typed details are
/// present they decide the kind, even if a plain `reasoning` string is also sent.
fn split_message(message: ResponseMessage) -> (String, Reasoning) {
    let content = message.content.unwrap_or_default();
    let from_details = details_reasoning(&message.reasoning_details);
    let direct = [message.reasoning, message.reasoning_content]
        .into_iter()
        .flatten()
        .find(|text| !text.trim().is_empty());
    if let Some(text) = direct {
        let kind = from_details
            .as_ref()
            .map_or(ReasoningKind::Raw, |reasoning| reasoning.kind);
        return (
            content,
            Reasoning {
                text: Some(text),
                kind,
            },
        );
    }
    if let Some(reasoning) = from_details {
        return (content, reasoning);
    }
    match split_think(&content) {
        Some((thought, answer)) if thought.is_empty() => (answer, Reasoning::none()),
        Some((thought, answer)) => (
            answer,
            Reasoning {
                text: Some(thought),
                kind: ReasoningKind::Raw,
            },
        ),
        None => (content, Reasoning::none()),
    }
}

/// Kind and text from typed reasoning details: any text means `Raw`, else any summary
/// means `Summary`, else encrypted or empty entries mean `Redacted`.
fn details_reasoning(details: &[ReasoningDetail]) -> Option<Reasoning> {
    let non_empty = |text: &Option<String>| text.clone().filter(|t| !t.trim().is_empty());
    let texts: Vec<String> = details
        .iter()
        .filter_map(|detail| match detail {
            ReasoningDetail::Text { text } => non_empty(text),
            _ => None,
        })
        .collect();
    if !texts.is_empty() {
        return Some(Reasoning {
            text: Some(texts.join("\n")),
            kind: ReasoningKind::Raw,
        });
    }
    let summaries: Vec<String> = details
        .iter()
        .filter_map(|detail| match detail {
            ReasoningDetail::Summary { summary } => non_empty(summary),
            _ => None,
        })
        .collect();
    if !summaries.is_empty() {
        return Some(Reasoning {
            text: Some(summaries.join("\n")),
            kind: ReasoningKind::Summary,
        });
    }
    let hidden = details
        .iter()
        .any(|detail| !matches!(detail, ReasoningDetail::Other));
    hidden.then_some(Reasoning {
        text: None,
        kind: ReasoningKind::Redacted,
    })
}

/// Splits `<think>reasoning</think>answer`. The opening tag is optional because some
/// models only emit the closing one. Returns `(reasoning, answer)`, both trimmed.
fn split_think(content: &str) -> Option<(String, String)> {
    let (before, after) = content.split_once("</think>")?;
    let before = before.trim();
    let thought = before.strip_prefix("<think>").unwrap_or(before).trim();
    Some((thought.to_string(), after.trim().to_string()))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn split(message: &serde_json::Value) -> Result<(String, Reasoning), serde_json::Error> {
        Ok(split_message(serde_json::from_value(message.clone())?))
    }

    #[test]
    fn plain_reasoning_field_is_raw() -> Result<(), serde_json::Error> {
        let (content, reasoning) = split(&json!({
            "role": "assistant", "content": "42", "reasoning": "6 times 7"
        }))?;
        assert_eq!(content, "42");
        assert_eq!(reasoning.text.as_deref(), Some("6 times 7"));
        assert_eq!(reasoning.kind, ReasoningKind::Raw);
        Ok(())
    }

    #[test]
    fn reasoning_content_is_raw() -> Result<(), serde_json::Error> {
        let (_, reasoning) = split(&json!({
            "content": "42", "reasoning": null, "reasoning_content": "thinking"
        }))?;
        assert_eq!(reasoning.text.as_deref(), Some("thinking"));
        assert_eq!(reasoning.kind, ReasoningKind::Raw);
        Ok(())
    }

    #[test]
    fn typed_details_decide_the_kind() -> Result<(), serde_json::Error> {
        let text = json!({"type": "reasoning.text", "text": "step 1", "signature": "sig", "id": "r1", "format": "anthropic-claude-v1", "index": 0});
        let summary = json!({"type": "reasoning.summary", "summary": "overview", "id": "r2", "format": "openai-responses-v1", "index": 0});
        let encrypted = json!({"type": "reasoning.encrypted", "data": "opaque", "id": "r3", "format": "openai-responses-v1", "index": 1});

        let (_, reasoning) = split(&json!({"content": "a", "reasoning_details": [text]}))?;
        assert_eq!(
            reasoning,
            Reasoning {
                text: Some("step 1".into()),
                kind: ReasoningKind::Raw
            }
        );

        let (_, reasoning) = split(
            &json!({"content": "a", "reasoning": "overview", "reasoning_details": [summary, encrypted]}),
        )?;
        assert_eq!(
            reasoning,
            Reasoning {
                text: Some("overview".into()),
                kind: ReasoningKind::Summary
            }
        );

        let (_, reasoning) = split(&json!({"content": "a", "reasoning_details": [encrypted]}))?;
        assert_eq!(
            reasoning,
            Reasoning {
                text: None,
                kind: ReasoningKind::Redacted
            }
        );
        Ok(())
    }

    #[test]
    fn unknown_detail_types_are_ignored() -> Result<(), serde_json::Error> {
        let (_, reasoning) = split(&json!({
            "content": "a",
            "reasoning_details": [{"type": "reasoning.server_tool_call", "id": "x"}]
        }))?;
        assert_eq!(reasoning, Reasoning::none());
        Ok(())
    }

    #[test]
    fn inline_think_block_is_split_out() -> Result<(), serde_json::Error> {
        let (content, reasoning) = split(&json!({
            "content": "<think>\nfirst, borrow\n</think>\n\nUse a reference."
        }))?;
        assert_eq!(content, "Use a reference.");
        assert_eq!(reasoning.text.as_deref(), Some("first, borrow"));
        assert_eq!(reasoning.kind, ReasoningKind::Raw);

        let (content, reasoning) = split(&json!({"content": "only closing</think>answer"}))?;
        assert_eq!(content, "answer");
        assert_eq!(reasoning.text.as_deref(), Some("only closing"));

        let (content, reasoning) = split(&json!({"content": "<think></think>answer"}))?;
        assert_eq!(content, "answer");
        assert_eq!(reasoning, Reasoning::none());
        Ok(())
    }

    #[test]
    fn no_reasoning_at_all() -> Result<(), serde_json::Error> {
        let (content, reasoning) = split(&json!({"content": "plain", "refusal": null}))?;
        assert_eq!(content, "plain");
        assert_eq!(reasoning, Reasoning::none());
        Ok(())
    }

    #[test]
    fn finish_reasons_map() {
        assert_eq!(finish_reason(Some("stop")), FinishReason::Stop);
        assert_eq!(finish_reason(Some("length")), FinishReason::Length);
        assert_eq!(
            finish_reason(Some("content_filter")),
            FinishReason::ContentFilter
        );
        assert_eq!(finish_reason(Some("error")), FinishReason::Other);
        assert_eq!(finish_reason(None), FinishReason::Other);
    }
}
