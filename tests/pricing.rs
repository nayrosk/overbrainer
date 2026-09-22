use std::time::Duration;

use overbrainer::config::Protocol;
use overbrainer::llm::ProtocolClient;
use overbrainer::pricing::fetch_price;
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
    let price = fetch_price(&client(&server)?, "anthropic/claude-sonnet-5")
        .await
        .ok_or("price expected")?;
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
        fetch_price(&client(&server)?, "anthropic/claude-sonnet-5").await,
        None
    );
    Ok(())
}
