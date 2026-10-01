use std::time::Duration;

use overbrainer::config::Protocol;
use overbrainer::llm::ProtocolClient;
use overbrainer::pricing::{fetch_listing, find_price, listed_price};
use serde_json::json;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn client(server: &MockServer) -> Result<ProtocolClient, Box<dyn std::error::Error>> {
    Ok(ProtocolClient::new(
        Protocol::Openai,
        &format!("{}/api/v1", server.uri()),
        None,
        "anthropic/claude-sonnet-5",
        Duration::from_secs(5),
    )?)
}

#[tokio::test]
async fn reads_the_detailed_nanogpt_listing() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .and(query_param("detailed", "true"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{
                "id": "anthropic/claude-sonnet-5",
                "object": "model",
                "owned_by": "anthropic",
                "pricing": {"prompt": 2, "completion": 10, "currency": "USD", "unit": "per_million_tokens"}
            }]
        })))
        .expect(1)
        .mount(&server)
        .await;
    let listing = fetch_listing(&client(&server)?, Duration::from_secs(5))
        .await
        .ok_or("listing expected")?;
    let price = listed_price(&listing, "anthropic/claude-sonnet-5").ok_or("price expected")?;
    assert!((price.output_per_million - 10.0).abs() < 1e-9);
    Ok(())
}

#[tokio::test]
async fn an_unavailable_listing_means_no_price() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    assert_eq!(
        fetch_listing(&client(&server)?, Duration::from_secs(5)).await,
        None
    );
    Ok(())
}

#[tokio::test]
async fn a_slow_listing_gives_up_at_the_cap() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"data": []}))
                .set_delay(Duration::from_secs(5)),
        )
        .mount(&server)
        .await;
    let started = std::time::Instant::now();
    assert_eq!(
        fetch_listing(&client(&server)?, Duration::from_millis(50)).await,
        None
    );
    assert!(started.elapsed() < Duration::from_secs(2));
    Ok(())
}

#[tokio::test]
async fn a_listing_without_the_model_has_no_price() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": []})))
        .mount(&server)
        .await;
    let listing = fetch_listing(&client(&server)?, Duration::from_secs(5))
        .await
        .ok_or("listing expected")?;
    assert_eq!(listing, json!({"data": []}));
    assert_eq!(listed_price(&listing, "anthropic/claude-sonnet-5"), None);
    assert_eq!(find_price(&listing, "anthropic/claude-sonnet-5"), None);
    Ok(())
}
