//! `RunpodClient` against a local HTTP stub: authentication, user agent, retries,
//! error messages without secrets, pagination, and the fixed, status-only
//! messages a create call's error ever shows.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use overbrainer::retry::RetryPolicy;
use overbrainer::runpod::{
    ApiError, CreateEnv, CreatePod, GpuRequest, PodId, RemoteStatus, RunpodClient, USER_AGENT,
};
use secrecy::SecretString;
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path, query_param, query_param_is_missing};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const KEY: &str = "rp_test_key_5f1d";
const HOST_KEY: &str = "b3BlbnNzaC1ob3N0LWtleQ";

/// The fixed messages `client.rs` shows for a create call's failed answer.
/// These must match its private constants of the same name exactly.
const CAPACITY_MESSAGE: &str = "no capacity for this GPU type";
const AUTH_REJECTED_MESSAGE: &str = "Runpod refused the API key";
const FORBIDDEN_MESSAGE: &str =
    "Runpod refused the request (403): check the API key and that the GPU type is allowed";
const RATE_LIMITED_MESSAGE: &str = "Runpod is rate limiting requests; try again later";
const TIMEOUT_MESSAGE: &str =
    "Runpod timed out on the create request (the pod may have been created anyway)";
const INSUFFICIENT_BALANCE_MESSAGE: &str = "the Runpod account balance is insufficient";
const INVALID_REQUEST_MESSAGE: &str =
    "Runpod rejected the create request (please report it: overbrainer built an invalid request)";
const SERVER_ERROR_MESSAGE: &str = "Runpod failed to process the create request";
const INVALID_CREATE_ANSWER_MESSAGE: &str = "Runpod's answer to the create call could not be read (please report it: overbrainer built an invalid request)";

/// A realistic-looking base64 host key, `len` characters, deterministic so the
/// test is reproducible.
fn generated_host_key(len: usize) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut rng = fastrand::Rng::with_seed(1_234_567_890);
    (0..len)
        .map(|_| ALPHABET[rng.usize(0..ALPHABET.len())] as char)
        .collect()
}

/// `text` with every character percent-encoded.
fn percent_encode(text: &str) -> String {
    use std::fmt::Write;

    text.chars().fold(String::new(), |mut acc, c| {
        let _ = write!(acc, "%{:02X}", c as u32);
        acc
    })
}

/// `text` split into `width`-character chunks and rejoined with `sep`.
fn chunk_join(text: &str, width: usize, sep: &str) -> String {
    text.chars()
        .collect::<Vec<_>>()
        .chunks(width)
        .map(|chunk| chunk.iter().collect::<String>())
        .collect::<Vec<_>>()
        .join(sep)
}

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

/// Runs `create_pod` with its host key set to `key`, against a mocked answer of
/// `status` with `body`, and returns the resulting error.
async fn create_status_error(
    status: u16,
    key: &str,
    body: serde_json::Value,
) -> Result<ApiError, Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(status).set_body_json(body))
        .mount(&server)
        .await;
    let mut create = request();
    create.env.host_key = SecretString::from(key.to_string());
    client(&server)?
        .create_pod(&create)
        .await
        .err()
        .ok_or_else(|| "no error".into())
}

/// Asserts that `error`'s `Display` and `Debug` are built only from `status`
/// and `expected`: the exact fixed text for its class, with no server byte.
fn assert_exact_create_message(error: &ApiError, status: u16, expected: &str) {
    assert_eq!(error.status(), Some(status));
    assert_eq!(
        error.to_string(),
        format!("Runpod answered {status}: {expected}")
    );
    let debug = format!("{error:?}");
    assert!(debug.contains(&format!("message: {expected:?}")), "{debug}");
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
        .expect(2)
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
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v2/pods/p1"))
        .respond_with(ResponseTemplate::new(204))
        .expect(1)
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
async fn a_create_error_shows_no_server_text() -> TestResult {
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
    assert_exact_create_message(&error, 422, INVALID_REQUEST_MESSAGE);
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

// The attack shapes: however Runpod's answer echoes the request (plain,
// nested inside another object, JSON-escaped, percent-encoded, chunked at any
// width, joined with any separator), a create error's `Display` and `Debug`
// are built only from the fixed message for its class and the status: never
// from anything the server said.

#[tokio::test]
async fn a_plain_echo_never_appears_in_a_create_error() -> TestResult {
    let key = generated_host_key(200);
    let error =
        create_status_error(422, &key, json!({"errors": [format!("bad key {key}")]})).await?;
    assert_exact_create_message(&error, 422, INVALID_REQUEST_MESSAGE);
    Ok(())
}

#[tokio::test]
async fn a_nested_echo_never_appears_in_a_create_error() -> TestResult {
    let key = generated_host_key(200);
    let error = create_status_error(
        422,
        &key,
        json!({
            "errors": [{
                "message": "rejected",
                "context": {"env": {"OVERBRAINER_HOST_KEY": key}}
            }]
        }),
    )
    .await?;
    assert_exact_create_message(&error, 422, INVALID_REQUEST_MESSAGE);
    Ok(())
}

#[tokio::test]
async fn a_json_escaped_echo_never_appears_in_a_create_error() -> TestResult {
    let key = generated_host_key(200);
    let escaped = key.replace('/', "\\/");
    let error =
        create_status_error(422, &key, json!({"detail": format!("bad key {escaped}")})).await?;
    assert_exact_create_message(&error, 422, INVALID_REQUEST_MESSAGE);
    Ok(())
}

#[tokio::test]
async fn a_plus_escaped_echo_never_appears_in_a_create_error() -> TestResult {
    let key = generated_host_key(200);
    let escaped = key.replace('+', "\\u002B");
    let error =
        create_status_error(422, &key, json!({"detail": format!("bad key {escaped}")})).await?;
    assert_exact_create_message(&error, 422, INVALID_REQUEST_MESSAGE);
    Ok(())
}

#[tokio::test]
async fn a_percent_encoded_echo_never_appears_in_a_create_error() -> TestResult {
    let key = generated_host_key(200);
    let encoded = percent_encode(&key);
    let error =
        create_status_error(422, &key, json!({"detail": format!("bad key {encoded}")})).await?;
    assert_exact_create_message(&error, 422, INVALID_REQUEST_MESSAGE);
    Ok(())
}

#[tokio::test]
async fn a_key_wrapped_every_8_characters_never_appears_in_a_create_error() -> TestResult {
    let key = generated_host_key(200);
    let wrapped = chunk_join(&key, 8, "\n");
    let error =
        create_status_error(422, &key, json!({"detail": format!("bad key {wrapped}")})).await?;
    assert_exact_create_message(&error, 422, INVALID_REQUEST_MESSAGE);
    Ok(())
}

#[tokio::test]
async fn a_key_joined_with_dashes_every_11_characters_never_appears_in_a_create_error() -> TestResult
{
    let key = generated_host_key(200);
    let joined = chunk_join(&key, 11, "-");
    let error =
        create_status_error(422, &key, json!({"detail": format!("bad key {joined}")})).await?;
    assert_exact_create_message(&error, 422, INVALID_REQUEST_MESSAGE);
    Ok(())
}

#[tokio::test]
async fn a_capacity_400_whose_detail_echoes_the_key_shows_only_the_capacity_message() -> TestResult
{
    let key = generated_host_key(200);
    let detail = format!(
        "There are no longer any instances available with the requested specifications. \
         Please refresh and try again. (offending value: {key})"
    );
    let error = create_status_error(400, &key, json!({"detail": detail})).await?;
    assert_exact_create_message(&error, 400, CAPACITY_MESSAGE);
    Ok(())
}

// Classification: every class is reachable and shows only its own fixed text.

#[tokio::test]
async fn a_capacity_failure_shows_only_the_fixed_message() -> TestResult {
    let detail = "There are no longer any instances available with the requested \
                   specifications. Please refresh and try again.";
    let error = create_status_error(400, "irrelevant", json!({ "detail": detail })).await?;
    assert_exact_create_message(&error, 400, CAPACITY_MESSAGE);
    Ok(())
}

#[tokio::test]
async fn a_non_capacity_400_is_classified_as_invalid_request() -> TestResult {
    let error =
        create_status_error(400, "irrelevant", json!({"detail": "unrelated bad body"})).await?;
    assert_exact_create_message(&error, 400, INVALID_REQUEST_MESSAGE);
    Ok(())
}

#[tokio::test]
async fn a_401_create_failure_is_classified_as_auth_rejected() -> TestResult {
    let error = create_status_error(401, "irrelevant", json!({"detail": "irrelevant"})).await?;
    assert_exact_create_message(&error, 401, AUTH_REJECTED_MESSAGE);
    Ok(())
}

#[tokio::test]
async fn a_403_create_failure_names_the_key_and_the_gpu_type() -> TestResult {
    let error = create_status_error(403, "irrelevant", json!({"detail": "irrelevant"})).await?;
    assert_exact_create_message(&error, 403, FORBIDDEN_MESSAGE);
    Ok(())
}

#[tokio::test]
async fn a_408_create_failure_is_a_timeout_and_ambiguous() -> TestResult {
    let error = create_status_error(408, "irrelevant", json!({"detail": "irrelevant"})).await?;
    assert_exact_create_message(&error, 408, TIMEOUT_MESSAGE);
    assert!(error.is_ambiguous());
    Ok(())
}

#[tokio::test]
async fn a_429_left_after_retries_is_named_a_rate_limit() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "0")
                .set_body_json(json!({"detail": "irrelevant"})),
        )
        .mount(&server)
        .await;
    let error = client(&server)?
        .create_pod(&request())
        .await
        .err()
        .ok_or("no error")?;
    assert_exact_create_message(&error, 429, RATE_LIMITED_MESSAGE);
    assert!(!error.is_ambiguous());
    assert_eq!(
        server.received_requests().await.unwrap_or_default().len(),
        3
    );
    Ok(())
}

#[tokio::test]
async fn a_402_create_failure_is_classified_as_insufficient_balance() -> TestResult {
    let error = create_status_error(402, "irrelevant", json!({"detail": "irrelevant"})).await?;
    assert_exact_create_message(&error, 402, INSUFFICIENT_BALANCE_MESSAGE);
    Ok(())
}

#[tokio::test]
async fn a_500_create_failure_is_classified_as_a_server_error() -> TestResult {
    let error = create_status_error(500, "irrelevant", json!({"detail": "irrelevant"})).await?;
    assert_exact_create_message(&error, 500, SERVER_ERROR_MESSAGE);
    Ok(())
}

#[tokio::test]
async fn a_create_decode_failure_shows_no_server_text() -> TestResult {
    let server = MockServer::start().await;
    let key = generated_host_key(400);
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(201).set_body_json(json!({
            "id": key,
            "status": "RUNNING"
        })))
        .mount(&server)
        .await;
    let mut create = request();
    create.env.host_key = SecretString::from(key.clone());
    let error = client(&server)?
        .create_pod(&create)
        .await
        .err()
        .ok_or("no error")?;
    assert!(
        matches!(
            &error,
            ApiError::InvalidResponse(message) if message == INVALID_CREATE_ANSWER_MESSAGE
        ),
        "{error:?}"
    );
    Ok(())
}

/// Asserts that no 8-character piece of `secret` appears in `error`'s `Display`
/// or `Debug`.
fn assert_no_piece_of(error: &ApiError, secret: &str) {
    let display = error.to_string();
    let debug = format!("{error:?}");
    let chars: Vec<char> = secret.chars().collect();
    for window in chars.windows(8) {
        let piece: String = window.iter().collect();
        assert!(!display.contains(&piece), "{piece} in {display}");
        assert!(!debug.contains(&piece), "{piece} in {debug}");
    }
}

/// A pod whose `env` is a string holding the host key, which `PodEnv` cannot
/// read: serde's own message would quote the whole string.
fn pod_with_a_string_env(host_key: &str) -> serde_json::Value {
    json!({
        "id": "p1",
        "status": "RUNNING",
        "env": format!("OVERBRAINER_HOST_KEY={host_key}")
    })
}

#[tokio::test]
async fn a_get_decode_error_never_quotes_the_body() -> TestResult {
    let server = MockServer::start().await;
    let host_key = generated_host_key(500);
    Mock::given(method("GET"))
        .and(path("/v2/pods/p1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(pod_with_a_string_env(&host_key)))
        .mount(&server)
        .await;
    let error = client(&server)?
        .get_pod(&PodId::new("p1")?)
        .await
        .err()
        .ok_or("no error")?;
    assert!(matches!(error, ApiError::InvalidResponse(_)), "{error:?}");
    assert!(error.to_string().contains("line 1, column"), "{error}");
    assert_no_piece_of(&error, &host_key);
    Ok(())
}

#[tokio::test]
async fn a_list_decode_error_never_quotes_the_body() -> TestResult {
    let server = MockServer::start().await;
    let host_key = generated_host_key(500);
    Mock::given(method("GET"))
        .and(path("/v2/pods"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"pods": [pod_with_a_string_env(&host_key)]})),
        )
        .mount(&server)
        .await;
    let error = client(&server)?.list_pods().await.err().ok_or("no error")?;
    assert!(matches!(error, ApiError::InvalidResponse(_)), "{error:?}");
    assert_no_piece_of(&error, &host_key);
    Ok(())
}

#[tokio::test]
async fn a_repeated_pagination_cursor_is_an_error() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v2/pods"))
        .and(query_param_is_missing("cursor"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "pods": [],
            "pagination": {"hasNextPage": true, "nextCursor": "c2"}
        })))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v2/pods"))
        .and(query_param("cursor", "c2"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "pods": [],
            "pagination": {"hasNextPage": true, "nextCursor": "c2"}
        })))
        .expect(1)
        .mount(&server)
        .await;
    let error = client(&server)?.list_pods().await.err().ok_or("no error")?;
    assert!(matches!(error, ApiError::InvalidResponse(_)), "{error:?}");
    Ok(())
}

/// Always answers with another, never-repeating page, so `list_pods` can only
/// stop by hitting its page cap.
struct EndlessPager;

impl Respond for EndlessPager {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let current: u64 = request
            .url
            .query_pairs()
            .find(|(key, _)| key == "cursor")
            .and_then(|(_, value)| value.parse().ok())
            .unwrap_or(0);
        ResponseTemplate::new(200).set_body_json(json!({
            "pods": [],
            "pagination": {"hasNextPage": true, "nextCursor": (current + 1).to_string()}
        }))
    }
}

#[tokio::test]
async fn pagination_gives_up_after_its_page_cap() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v2/pods"))
        .respond_with(EndlessPager)
        .mount(&server)
        .await;
    let error = client(&server)?.list_pods().await.err().ok_or("no error")?;
    assert!(matches!(error, ApiError::InvalidResponse(_)), "{error:?}");
    Ok(())
}

#[tokio::test]
async fn a_404_with_an_html_body_is_a_real_error_not_a_missing_pod() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v2/pods/p1"))
        .respond_with(ResponseTemplate::new(404).set_body_string("<html>not found</html>"))
        .expect(1)
        .mount(&server)
        .await;
    let error = client(&server)?
        .get_pod(&PodId::new("p1")?)
        .await
        .err()
        .ok_or("no error")?;
    assert_eq!(error.status(), Some(404));
    Ok(())
}

#[tokio::test]
async fn a_404_with_an_empty_body_is_a_real_error_not_a_successful_delete() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/v2/pods/p1"))
        .respond_with(ResponseTemplate::new(404).set_body_string(""))
        .expect(1)
        .mount(&server)
        .await;
    let error = client(&server)?
        .delete_pod(&PodId::new("p1")?)
        .await
        .err()
        .ok_or("no error")?;
    assert_eq!(error.status(), Some(404));
    Ok(())
}

#[tokio::test]
async fn a_create_transport_failure_is_not_retried_and_is_ambiguous() -> TestResult {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;
    let attempts = Arc::new(AtomicU32::new(0));
    let counted = Arc::clone(&attempts);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { break };
            counted.fetch_add(1, Ordering::SeqCst);
            drop(stream);
        }
    });
    let broken = RunpodClient::new(&format!("http://{addr}/v2/"), &SecretString::from(KEY))?
        .with_policy(RetryPolicy {
            max_retries: 2,
            base: Duration::from_millis(1),
            cap: Duration::from_millis(2),
        });
    let error = broken
        .create_pod(&request())
        .await
        .err()
        .ok_or("no error")?;
    assert!(matches!(error, ApiError::Transport(_)), "{error:?}");
    assert!(error.is_ambiguous());
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
    Ok(())
}

#[tokio::test]
async fn a_redirect_is_not_followed() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v2/pods/p1"))
        .respond_with(
            ResponseTemplate::new(302).insert_header("location", "http://example.invalid/"),
        )
        .expect(1)
        .mount(&server)
        .await;
    let error = client(&server)?
        .get_pod(&PodId::new("p1")?)
        .await
        .err()
        .ok_or("no error")?;
    assert_eq!(error.status(), Some(302));
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
