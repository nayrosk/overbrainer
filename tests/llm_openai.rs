use std::time::Duration;

use overbrainer::dataset::{FinishReason, ReasoningKind};
use overbrainer::llm::{CompletionRequest, LlmError, OpenAiClient, RetryPolicy, with_retry};
use secrecy::SecretString;
use serde_json::{Value, json};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const KEY: &str = "sk-test-key-9";

fn client(server: &MockServer) -> Result<OpenAiClient, LlmError> {
    OpenAiClient::new(
        &format!("{}/api/v1/", server.uri()),
        Some(&SecretString::from(KEY)),
        "deepseek-r1",
        Duration::from_secs(5),
    )
}

fn request(reasoning: bool) -> CompletionRequest {
    CompletionRequest {
        system: Some("Be precise.".into()),
        prompt: "Why borrow?".into(),
        max_tokens: 512,
        temperature: Some(0.2),
        reasoning,
        effort: None,
        thinking_budget: None,
    }
}

/// Response shaped like the `OpenAI` spec example, with `OpenRouter` reasoning fields.
fn chat_body(message: &Value, finish: &str) -> Value {
    json!({
        "id": "gen-1",
        "object": "chat.completion",
        "created": 1_741_570_283,
        "model": "deepseek-r1",
        "choices": [{"index": 0, "message": message, "logprobs": null, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 34,
            "total_tokens": 46,
            "completion_tokens_details": {"reasoning_tokens": 20}
        }
    })
}

async fn last_body(server: &MockServer) -> Result<Value, Box<dyn std::error::Error>> {
    let requests = server
        .received_requests()
        .await
        .ok_or("request recording is disabled")?;
    let last = requests.last().ok_or("no request received")?;
    Ok(serde_json::from_slice(&last.body)?)
}

#[tokio::test]
async fn sends_the_request_and_reads_reasoning() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/chat/completions"))
        .and(header("authorization", format!("Bearer {KEY}").as_str()))
        .and(body_partial_json(json!({
            "model": "deepseek-r1",
            "max_tokens": 512,
            "temperature": 0.2,
            "reasoning": {"effort": "medium"},
            "messages": [
                {"role": "system", "content": "Be precise."},
                {"role": "user", "content": "Why borrow?"}
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_body(
            &json!({"role": "assistant", "content": "To avoid moves.", "refusal": null, "reasoning": "The user asks why."}),
            "stop",
        )))
        .expect(1)
        .mount(&server)
        .await;

    let completion = client(&server)?.complete(&request(true)).await?;
    assert_eq!(completion.content, "To avoid moves.");
    assert_eq!(completion.reasoning.kind, ReasoningKind::Raw);
    assert_eq!(
        completion.reasoning.text.as_deref(),
        Some("The user asks why.")
    );
    assert_eq!(completion.usage.input_tokens, 12);
    assert_eq!(completion.usage.output_tokens, 34);
    assert_eq!(completion.finish, FinishReason::Stop);
    Ok(())
}

#[tokio::test]
async fn no_reasoning_parameter_when_not_requested() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(chat_body(&json!({"content": "ok"}), "stop")),
        )
        .mount(&server)
        .await;
    client(&server)?.complete(&request(false)).await?;
    let body = last_body(&server).await?;
    assert!(body.get("reasoning").is_none(), "{body}");
    Ok(())
}

#[tokio::test]
async fn refusal_and_length_are_reported() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_body(
            &json!({"content": null, "refusal": "I can't help with that."}),
            "stop",
        )))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(chat_body(&json!({"content": "cut"}), "length")),
        )
        .mount(&server)
        .await;
    let client = client(&server)?;
    let refused = client.complete(&request(false)).await?;
    assert_eq!(refused.finish, FinishReason::Refusal);
    assert_eq!(refused.content, "");
    let truncated = client.complete(&request(false)).await?;
    assert_eq!(truncated.finish, FinishReason::Length);
    Ok(())
}

#[tokio::test]
async fn rate_limit_carries_retry_after() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "7")
                .set_body_json(json!({"error": {"message": "slow down", "type": "rate_limit_error", "param": null, "code": "slow_down"}})),
        )
        .mount(&server)
        .await;
    match client(&server)?.complete(&request(false)).await {
        Err(error) => {
            assert!(error.is_retryable());
            assert_eq!(error.retry_after(), Some(Duration::from_secs(7)));
            assert_eq!(error.to_string(), "HTTP 429: slow down");
            Ok(())
        },
        Ok(completion) => Err(format!("expected an error, got {completion:?}").into()),
    }
}

#[tokio::test]
async fn unauthorized_is_fatal_and_does_not_echo_the_key() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(401).set_body_json(
            json!({"error": {"message": format!("Incorrect API key provided: {KEY}")}}),
        ))
        .mount(&server)
        .await;
    match client(&server)?.complete(&request(false)).await {
        Err(error) => {
            assert!(!error.is_retryable());
            assert!(error.is_fatal_for_stage());
            assert!(!error.to_string().contains(KEY));
            assert!(!format!("{error:?}").contains(KEY));
            Ok(())
        },
        Ok(completion) => Err(format!("expected an error, got {completion:?}").into()),
    }
}

#[tokio::test]
async fn retry_recovers_from_server_errors() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503).insert_header("retry-after", "0"))
        .up_to_n_times(2)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(chat_body(&json!({"content": "ok"}), "stop")),
        )
        .mount(&server)
        .await;
    let client = client(&server)?;
    let policy = RetryPolicy {
        max_retries: 3,
        base: Duration::from_millis(1),
        cap: Duration::from_millis(2),
    };
    let request = request(false);
    let mut waits = Vec::new();
    let completion = with_retry(
        &policy,
        || client.complete(&request),
        |_, wait| waits.push(wait),
    )
    .await?;
    assert_eq!(completion.content, "ok");
    assert_eq!(waits, vec![Duration::ZERO, Duration::ZERO]);
    Ok(())
}

#[tokio::test]
async fn embeddings_are_returned_in_input_order() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/v1/embeddings"))
        .and(body_partial_json(
            json!({"model": "deepseek-r1", "input": ["a", "b"], "encoding_format": "float"}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [
                {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
                {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}
            ],
            "model": "deepseek-r1",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
        })))
        .mount(&server)
        .await;
    let vectors = client(&server)?
        .embed(&["a".to_string(), "b".to_string()])
        .await?;
    assert_eq!(vectors, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    Ok(())
}

#[tokio::test]
async fn debug_output_never_shows_the_key() -> TestResult {
    let server = MockServer::start().await;
    let rendered = format!("{:?}", client(&server)?);
    assert!(!rendered.contains(KEY), "{rendered}");
    Ok(())
}
