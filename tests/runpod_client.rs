//! `RunpodClient` against a local HTTP stub: authentication, user agent, retries,
//! error messages without secrets, pagination.

use std::collections::BTreeMap;
use std::time::Duration;

use overbrainer::retry::RetryPolicy;
use overbrainer::runpod::{
    ApiError, CreateEnv, CreatePod, GpuRequest, PodId, RemoteStatus, RunpodClient, USER_AGENT,
};
use secrecy::SecretString;
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const KEY: &str = "rp_test_key_5f1d";
const HOST_KEY: &str = "b3BlbnNzaC1ob3N0LWtleQ";

fn client(server: &MockServer) -> Result<RunpodClient, ApiError> {
    Ok(
        RunpodClient::new(&format!("{}/v2/", server.uri()), &SecretString::from(KEY))?.with_policy(
            RetryPolicy {
                max_retries: 2,
                base: Duration::from_millis(1),
                cap: Duration::from_millis(2),
            },
        ),
    )
}

fn pod(id: &str, run_id: &str) -> serde_json::Value {
    json!({
        "id": id,
        "name": format!("overbrainer-{run_id}-1"),
        "status": "RUNNING",
        "cost": 0.53,
        "env": {"OVERBRAINER_RUN_ID": run_id, "OVERBRAINER_HOST_KEY": HOST_KEY},
        "ssh": {"direct": null}
    })
}

fn request() -> CreatePod {
    CreatePod {
        name: "overbrainer-r1-1".into(),
        image: "img@sha256:abc".into(),
        cloud: "SECURE",
        gpu: GpuRequest {
            id: "NVIDIA A40".into(),
            count: 1,
            min_cuda_version: "13.0",
        },
        disk: 50,
        ports: vec!["22/tcp".into()],
        start_ssh: false,
        data_center_ids: None,
        mounts: None,
        env: CreateEnv {
            plain: BTreeMap::from([("OVERBRAINER_RUN_ID".to_string(), "r1".to_string())]),
            host_key_name: "OVERBRAINER_HOST_KEY",
            host_key: SecretString::from(HOST_KEY),
        },
        cmd: vec!["bash".into(), "-c".into(), "true".into()],
    }
}

#[tokio::test]
async fn every_request_carries_the_key_and_the_user_agent() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v2/pods/p1"))
        .and(header("authorization", format!("Bearer {KEY}").as_str()))
        .and(header("user-agent", USER_AGENT))
        .respond_with(ResponseTemplate::new(200).set_body_json(pod("p1", "r1")))
        .expect(1)
        .mount(&server)
        .await;
    let found = client(&server)?.get_pod(&PodId::new("p1")?).await?;
    let found = found.ok_or("no pod")?;
    assert_eq!(found.status, RemoteStatus::Running);
    assert_eq!(found.run_id(), Some("r1"));
    assert!(USER_AGENT.starts_with("overbrainer/"));
    Ok(())
}

#[tokio::test]
async fn reads_are_retried_on_server_errors_and_rate_limits() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v2/pods/p1"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v2/pods/p1"))
        .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "0"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v2/pods/p1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(pod("p1", "r1")))
        .mount(&server)
        .await;
    assert!(
        client(&server)?
            .get_pod(&PodId::new("p1")?)
            .await?
            .is_some()
    );
    assert_eq!(
        server.received_requests().await.unwrap_or_default().len(),
        3
    );
    Ok(())
}

#[tokio::test]
async fn a_missing_pod_is_none_and_its_deletion_succeeds() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(path("/v2/pods/gone"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({"title": "Not Found"})))
        .mount(&server)
        .await;
    let client = client(&server)?;
    let id = PodId::new("gone")?;
    assert!(client.get_pod(&id).await?.is_none());
    client.delete_pod(&id).await?;
    Ok(())
}

#[tokio::test]
async fn a_delete_is_retried_until_it_succeeds() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/v2/pods/p1"))
        .respond_with(ResponseTemplate::new(500))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v2/pods/p1"))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;
    client(&server)?.delete_pod(&PodId::new("p1")?).await?;
    Ok(())
}

#[tokio::test]
async fn a_rejected_key_is_named_without_its_value() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(401).set_body_json(json!({"detail": format!("bad key {KEY}")})),
        )
        .expect(1)
        .mount(&server)
        .await;
    let error = client(&server)?
        .get_pod(&PodId::new("p1")?)
        .await
        .err()
        .ok_or("no error")?
        .to_string();
    assert_eq!(
        error,
        "Runpod answered 401: Runpod rejected the API key, check OVERBRAINER_RUNPOD__API_KEY"
    );
    Ok(())
}

#[tokio::test]
async fn a_cloudflare_refusal_keeps_its_body() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(403).set_body_string("error code: 1010"))
        .mount(&server)
        .await;
    let error = client(&server)?.list_pods().await.err().ok_or("no error")?;
    assert_eq!(
        error.to_string(),
        "Runpod answered 403: permission denied by Runpod: error code: 1010"
    );
    Ok(())
}

#[tokio::test]
async fn the_list_follows_every_page() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v2/pods"))
        .and(query_param("cursor", "c2"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "pods": [pod("p2", "r2")],
            "pagination": {"hasNextPage": false, "nextCursor": null}
        })))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v2/pods"))
        .and(query_param("limit", "1000"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "pods": [pod("p1", "r1")],
            "pagination": {"hasNextPage": true, "nextCursor": "c2"}
        })))
        .mount(&server)
        .await;
    let pods = client(&server)?.list_pods().await?;
    let ids: Vec<&str> = pods.iter().map(|pod| pod.id.as_str()).collect();
    assert_eq!(ids, vec!["p1", "p2"]);
    Ok(())
}

#[tokio::test]
async fn a_create_sends_the_v2_body_and_reads_the_pod() -> TestResult {
    let server = MockServer::start().await;
    let expected = json!({
        "name": "overbrainer-r1-1",
        "image": "img@sha256:abc",
        "cloud": "SECURE",
        "gpu": {"id": "NVIDIA A40", "count": 1, "minCudaVersion": "13.0"},
        "disk": 50,
        "ports": ["22/tcp"],
        "startSsh": false,
        "env": {"OVERBRAINER_RUN_ID": "r1", "OVERBRAINER_HOST_KEY": HOST_KEY},
        "cmd": ["bash", "-c", "true"]
    });
    Mock::given(method("POST"))
        .and(path("/v2/pods"))
        .and(body_json(&expected))
        .respond_with(ResponseTemplate::new(201).set_body_json(pod("k3x9abc", "r1")))
        .expect(1)
        .mount(&server)
        .await;
    let created = client(&server)?.create_pod(&request()).await?;
    assert_eq!(created.id.as_str(), "k3x9abc");
    assert_eq!(created.rate(), Some(0.53));
    Ok(())
}

#[tokio::test]
async fn a_create_error_never_quotes_a_key() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(422).set_body_json(json!({
            "title": "Unprocessable",
            "errors": [format!("env.OVERBRAINER_HOST_KEY {HOST_KEY} too long"), format!("key {KEY}")]
        })))
        .mount(&server)
        .await;
    let error = client(&server)?
        .create_pod(&request())
        .await
        .err()
        .ok_or("no error")?;
    let text = format!("{error} {error:?}");
    assert!(!text.contains(HOST_KEY) && !text.contains(KEY), "{text}");
    assert!(
        text.contains("env.OVERBRAINER_HOST_KEY *** too long; key ***"),
        "{text}"
    );
    Ok(())
}

#[tokio::test]
async fn a_create_is_retried_on_a_rate_limit_only() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "0"))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;
    let error = client(&server)?
        .create_pod(&request())
        .await
        .err()
        .ok_or("no error")?;
    assert_eq!(error.status(), Some(503));
    assert!(error.is_ambiguous());
    assert_eq!(
        server.received_requests().await.unwrap_or_default().len(),
        2
    );
    Ok(())
}

#[test]
fn the_client_debug_output_hides_the_key() -> TestResult {
    let client = RunpodClient::new("https://api.runpod.io/v2", &SecretString::from(KEY))?;
    let text = format!("{client:?}");
    assert!(!text.contains(KEY), "{text}");
    assert!(matches!(
        RunpodClient::new("https://x", &SecretString::from("bad\nkey")),
        Err(ApiError::InvalidApiKey)
    ));
    Ok(())
}
