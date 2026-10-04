//! `overbrainer push`: a run to a local stub of the Hugging Face Hub, a dry
//! run without a token, and a push refused without one. The token never shows.

use std::path::Path;

use assert_cmd::Command;
use overbrainer::runs::{RunRecord, RunState, Runs};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const RUN: &str = "20261004-120000-a1b2";

const TOKEN: &str = "hf_test";

const CONFIG: &str = r#"[project]
name = "my_demo"

[providers.mock]
protocol = "openai"

[roles]
generator = { provider = "mock", model = "gen" }
parent = { provider = "mock", model = "parent" }
"#;

/// A project with the succeeded run [`RUN`] and its adapter.
fn project() -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    std::fs::write(dir.path().join("overbrainer.toml"), CONFIG)?;
    overbrainer::project_format::write_current(dir.path())?;
    let runs = Runs::new(dir.path());
    runs.save(&RunRecord {
        id: RUN.to_string(),
        target: "box".into(),
        created: "2026-10-04T12:00:00Z".into(),
        remote_dir: format!("/w/{RUN}"),
        job: None,
        state: RunState::Succeeded,
        message: None,
        snapshot: None,
        resumed_from: None,
        snapshots: true,
    })?;
    let run_dir = runs.run_dir(RUN)?;
    std::fs::create_dir_all(run_dir.join("output"))?;
    std::fs::write(
        run_dir.join("axolotl.yaml"),
        "base_model: Qwen/Qwen3-0.6B\nadapter: qlora\n",
    )?;
    std::fs::write(run_dir.join("output/adapter_config.json"), "{}")?;
    std::fs::write(run_dir.join("output/adapter_model.safetensors"), "weights")?;
    std::fs::write(run_dir.join("output/README.md"), "axolotl")?;
    Ok(dir)
}

fn overbrainer(dir: &Path) -> Result<Command, Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("overbrainer")?;
    cmd.env_clear()
        .env("NO_COLOR", "1")
        .env("HOME", "/nonexistent")
        .arg("-C")
        .arg(dir);
    Ok(cmd)
}

fn not_found() -> ResponseTemplate {
    ResponseTemplate::new(404).set_body_json(json!({"error": "Repository not found"}))
}

/// A Hub where `me` has no repo `my-demo` yet.
async fn hub() -> MockServer {
    let server = MockServer::start().await;
    let mocks = [
        Mock::given(method("GET"))
            .and(path("/api/whoami-v2"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"name": "me", "type": "user"})),
            ),
        Mock::given(method("GET"))
            .and(path("/api/models/me/my-demo"))
            .respond_with(not_found()),
        Mock::given(method("POST"))
            .and(path("/api/repos/create"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({"url": "https://hf.co/me/my-demo"})),
            ),
        Mock::given(method("HEAD"))
            .and(path("/me/my-demo/resolve/main/README.md"))
            .respond_with(
                ResponseTemplate::new(404).insert_header("x-error-code", "EntryNotFound"),
            ),
        Mock::given(method("GET"))
            .and(path("/me/my-demo/resolve/main/README.md"))
            .respond_with(
                ResponseTemplate::new(404).insert_header("x-error-code", "EntryNotFound"),
            ),
        Mock::given(method("GET"))
            .and(path("/api/models/Qwen/Qwen3-0.6B"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                json!({"id": "Qwen/Qwen3-0.6B", "cardData": {"license": "apache-2.0"}}),
            )),
        Mock::given(method("POST"))
            .and(path("/api/models/me/my-demo/preupload/main"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"files": [
                {"path": "adapter_config.json", "uploadMode": "regular"},
                {"path": "adapter_model.safetensors", "uploadMode": "regular"},
                {"path": "README.md", "uploadMode": "regular"}
            ]}))),
        Mock::given(method("POST"))
            .and(path("/api/models/me/my-demo/commit/main"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "commitUrl": "https://hf.co/me/my-demo/commit/abc",
                "commitOid": "abc"
            }))),
    ];
    for mock in mocks {
        mock.mount(&server).await;
    }
    server
}

/// Whether `bytes` holds the test token.
fn leaks(bytes: &[u8]) -> bool {
    String::from_utf8_lossy(bytes).contains(TOKEN)
}

#[tokio::test]
async fn a_run_is_pushed_to_a_new_repo_and_recorded() -> TestResult {
    let dir = project()?;
    let server = hub().await;
    let mut cmd = overbrainer(dir.path())?;
    cmd.env("OVERBRAINER_HUB__BASE_URL", server.uri())
        .env("OVERBRAINER_HF_TOKEN", TOKEN)
        .args(["push", RUN]);
    let output = tokio::task::spawn_blocking(move || cmd.output()).await??;
    assert!(!leaks(&output.stdout), "stdout must not hold the token");
    assert!(!leaks(&output.stderr), "stderr must not hold the token");
    let stdout = String::from_utf8(output.stdout)?;
    assert!(output.status.success(), "{stdout}");
    assert_eq!(
        stdout,
        "push: https://hf.co/me/my-demo/commit/abc (2 files, 0.0 MB)\n"
    );
    let record: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        dir.path().join("runs").join(RUN).join("hub/push.json"),
    )?)?;
    assert_eq!(record["repo"], json!("me/my-demo"));
    assert_eq!(record["commit"], json!("abc"));
    assert_eq!(record["private"], json!(true));

    let requests = server.received_requests().await.ok_or("no request log")?;
    let create = requests
        .iter()
        .find(|request| request.url.path() == "/api/repos/create")
        .ok_or("no create request")?;
    let body: serde_json::Value = serde_json::from_slice(&create.body)?;
    assert_eq!(body["private"], json!(true), "private by default");
    let commit = requests
        .iter()
        .find(|request| request.url.path() == "/api/models/me/my-demo/commit/main")
        .ok_or("no commit request")?;
    let body = String::from_utf8_lossy(&commit.body);
    assert!(body.contains(&format!("Upload run {RUN} with overbrainer")));
    assert!(body.contains("adapter_model.safetensors") && body.contains("README.md"));
    Ok(())
}

#[tokio::test]
async fn a_dry_run_needs_no_token_and_sends_nothing() -> TestResult {
    let dir = project()?;
    let server = hub().await;
    let mut cmd = overbrainer(dir.path())?;
    cmd.env("OVERBRAINER_HUB__BASE_URL", server.uri())
        .args(["push", RUN, "--dry-run"]);
    let output = tokio::task::spawn_blocking(move || cmd.output()).await??;
    let stdout = String::from_utf8(output.stdout)?;
    assert!(output.status.success(), "{stdout}");
    assert_eq!(
        stdout,
        format!(
            "push: <you>/my-demo, private when created; nothing is sent (--dry-run)\n\
             adapter_config.json  0.0 MB\n\
             adapter_model.safetensors  0.0 MB\n\
             push: 2 files, 0.0 MB; card written to runs/{RUN}/hub/README.md\n"
        )
    );
    let card = std::fs::read_to_string(dir.path().join("runs").join(RUN).join("hub/README.md"))?;
    assert!(card.contains("<!-- overbrainer:card -->"));
    let requests = server.received_requests().await.ok_or("no request log")?;
    assert!(
        requests.is_empty(),
        "a dry run without a token sends nothing"
    );
    Ok(())
}

#[tokio::test]
async fn a_push_without_a_token_is_refused_before_any_request() -> TestResult {
    let dir = project()?;
    let server = hub().await;
    let mut cmd = overbrainer(dir.path())?;
    cmd.env("OVERBRAINER_HUB__BASE_URL", server.uri())
        .args(["push", RUN]);
    let output = tokio::task::spawn_blocking(move || cmd.output()).await??;
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains(
            "OVERBRAINER_HF_TOKEN is not set: a Hugging Face token with write access is needed \
             to push"
        ),
        "{stderr}"
    );
    let requests = server.received_requests().await.ok_or("no request log")?;
    assert!(requests.is_empty());
    Ok(())
}

#[tokio::test]
async fn a_refused_token_is_named_never_shown() -> TestResult {
    let dir = project()?;
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/whoami-v2"))
        .respond_with(ResponseTemplate::new(401).set_body_string(TOKEN))
        .mount(&server)
        .await;
    let mut cmd = overbrainer(dir.path())?;
    cmd.env("OVERBRAINER_HUB__BASE_URL", server.uri())
        .env("OVERBRAINER_HF_TOKEN", TOKEN)
        .args(["push", RUN]);
    let output = tokio::task::spawn_blocking(move || cmd.output()).await??;
    assert!(!output.status.success());
    assert!(!leaks(&output.stdout), "stdout must not hold the token");
    assert!(!leaks(&output.stderr), "stderr must not hold the token");
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains("OVERBRAINER_HF_TOKEN"),
        "the error must name the variable"
    );
    Ok(())
}
