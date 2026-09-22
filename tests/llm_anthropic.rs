use std::time::Duration;

use overbrainer::config::{Effort, EnvSource, Protocol, load};
use overbrainer::dataset::{FinishReason, ReasoningKind};
use overbrainer::llm::{
    CompletionRequest, LlmClient, LlmError, ProtocolClient, SetupError, connect,
};
use overbrainer::secrets::{Resolver, VaultSource};
use secrecy::SecretString;
use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const KEY: &str = "sk-ant-test-3";

fn client(server: &MockServer) -> Result<ProtocolClient, LlmError> {
    ProtocolClient::new(
        Protocol::Anthropic,
        &format!("{}/v1", server.uri()),
        Some(&SecretString::from(KEY)),
        "claude-sonnet-5",
        Duration::from_secs(5),
    )
}

fn request(effort: Option<Effort>) -> CompletionRequest {
    CompletionRequest {
        system: Some("Be precise.".into()),
        prompt: "Why borrow?".into(),
        max_tokens: 2048,
        temperature: None,
        reasoning: true,
        effort,
    }
}

fn message_body(content: &Value, stop_reason: &str) -> Value {
    json!({
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {"input_tokens": 21, "output_tokens": 55, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    })
}

#[tokio::test]
async fn sends_adaptive_thinking_and_reads_a_summary() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", KEY))
        .and(header("anthropic-version", "2023-06-01"))
        .and(body_partial_json(json!({
            "model": "claude-sonnet-5",
            "max_tokens": 2048,
            "system": "Be precise.",
            "messages": [{"role": "user", "content": "Why borrow?"}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"}
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(message_body(
            &json!([
                {"type": "thinking", "thinking": "Consider ownership.", "signature": "sig"},
                {"type": "text", "text": "Borrow to avoid a move."}
            ]),
            "end_turn",
        )))
        .expect(1)
        .mount(&server)
        .await;

    let completion = client(&server)?
        .complete(request(Some(Effort::High)))
        .await?;
    assert_eq!(completion.content, "Borrow to avoid a move.");
    assert_eq!(completion.reasoning.kind, ReasoningKind::Summary);
    assert_eq!(completion.usage.input_tokens, 21);
    assert_eq!(completion.usage.output_tokens, 55);
    assert_eq!(completion.finish, FinishReason::Stop);
    Ok(())
}

#[tokio::test]
async fn no_output_config_without_effort() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(message_body(
            &json!([{"type": "redacted_thinking", "data": "opaque"}, {"type": "text", "text": "a"}]),
            "max_tokens",
        )))
        .mount(&server)
        .await;
    let completion = client(&server)?.complete(request(None)).await?;
    assert_eq!(completion.reasoning.kind, ReasoningKind::Redacted);
    assert_eq!(completion.finish, FinishReason::Length);
    let requests = server.received_requests().await.ok_or("recording off")?;
    let body: Value = serde_json::from_slice(&requests.first().ok_or("no request")?.body)?;
    assert!(body.get("output_config").is_none(), "{body}");
    Ok(())
}

#[tokio::test]
async fn overloaded_is_retryable() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(529).set_body_json(json!({
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Overloaded"},
            "request_id": "req_1"
        })))
        .mount(&server)
        .await;
    match client(&server)?.complete(request(None)).await {
        Err(error) => {
            assert!(error.is_retryable());
            assert_eq!(error.to_string(), "HTTP 529: Overloaded");
            Ok(())
        },
        Ok(completion) => Err(format!("expected an error, got {completion:?}").into()),
    }
}

#[tokio::test]
async fn embeddings_are_unsupported() -> TestResult {
    let server = MockServer::start().await;
    let result = client(&server)?.embed(&["a".to_string()]).await;
    assert!(matches!(result, Err(LlmError::Unsupported("embeddings"))));
    Ok(())
}

const PROJECT: &str = r#"
[project]
name = "demo"
[providers.local]
protocol = "openai"
[roles]
generator = { provider = "local", model = "m1" }
parent = { provider = "local", model = "m2" }
"#;

#[tokio::test]
async fn connect_builds_a_client_from_settings() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer literal-key"))
        .and(body_partial_json(json!({"model": "m2"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"index": 0, "message": {"content": "hi"}, "finish_reason": "stop"}]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let dir = tempfile::tempdir()?;
    std::fs::write(dir.path().join("overbrainer.toml"), PROJECT)?;
    let settings = load(
        dir.path(),
        EnvSource::Vars(vec![
            (
                "OVERBRAINER_PROVIDERS__LOCAL__BASE_URL".into(),
                format!("{}/v1", server.uri()),
            ),
            (
                "OVERBRAINER_PROVIDERS__LOCAL__API_KEY".into(),
                "literal-key".into(),
            ),
        ]),
    )?;
    let resolver: Resolver<VaultSource> = Resolver::new(None);
    let client = connect(&settings, &settings.roles.parent, &resolver).await?;
    let completion = client
        .complete(CompletionRequest::for_role(
            &settings.roles.parent,
            None,
            "hello".into(),
        ))
        .await?;
    assert_eq!(completion.content, "hi");
    Ok(())
}

#[tokio::test]
async fn connect_requires_a_base_url() -> TestResult {
    let dir = tempfile::tempdir()?;
    std::fs::write(dir.path().join("overbrainer.toml"), PROJECT)?;
    let settings = load(dir.path(), EnvSource::Vars(Vec::new()))?;
    let resolver: Resolver<VaultSource> = Resolver::new(None);
    match connect(&settings, &settings.roles.parent, &resolver).await {
        Err(error @ SetupError::MissingBaseUrl { .. }) => {
            assert!(
                error
                    .to_string()
                    .contains("OVERBRAINER_PROVIDERS__LOCAL__BASE_URL")
            );
            Ok(())
        },
        other => Err(format!("expected MissingBaseUrl, got {other:?}").into()),
    }
}
