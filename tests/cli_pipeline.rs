use std::sync::atomic::{AtomicUsize, Ordering};

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const KEY: &str = "sk-e2e-secret-5";

const PROJECT: &str = r#"
[project]
name = "demo"

[[topics]]
name = "ownership"
subtopics = 2
questions_per_subtopic = 2

[providers.mock]
protocol = "openai"

[roles]
generator = { provider = "mock", model = "gen" }
parent = { provider = "mock", model = "parent", reasoning = true }

[pipeline]
concurrency = 2
max_retries = 1
eval_ratio = 0.25
"#;

/// Plays generator and parent: subtopics, then unique questions, then answers with
/// raw reasoning.
struct Script {
    calls: AtomicUsize,
}

impl Respond for Script {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
        let prompt = body
            .pointer("/messages/0/content")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let message = if body["model"] == "parent" {
            json!({"content": "Borrow it.", "reasoning": "The caller keeps ownership."})
        } else if prompt.contains("subtopics that together cover") {
            json!({"content": r#"["Borrowing", "Lifetimes"]"#})
        } else {
            json!({"content": format!(r#"["Unique question {call} one?", "Different topic {call} two?"]"#)})
        };
        ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50}
        }))
    }
}

async fn provider() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": [
            {"id": "gen", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
            {"id": "parent", "pricing": {"prompt": "0.000003", "completion": "0.000015"}}
        ]})))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(Script {
            calls: AtomicUsize::new(0),
        })
        .mount(&server)
        .await;
    server
}

fn overbrainer(
    dir: &std::path::Path,
    server: &MockServer,
) -> Result<Command, Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("overbrainer")?;
    cmd.env_clear()
        .env("NO_COLOR", "1")
        .env("HOME", "/nonexistent")
        .env(
            "OVERBRAINER_PROVIDERS__MOCK__BASE_URL",
            format!("{}/v1", server.uri()),
        )
        .env("OVERBRAINER_PROVIDERS__MOCK__API_KEY", KEY)
        .arg("-C")
        .arg(dir);
    Ok(cmd)
}

fn project() -> Result<tempfile::TempDir, std::io::Error> {
    let dir = tempfile::tempdir()?;
    std::fs::write(dir.path().join("overbrainer.toml"), PROJECT)?;
    Ok(dir)
}

#[tokio::test(flavor = "multi_thread")]
async fn run_chains_the_data_stages_and_prints_costs() -> TestResult {
    let server = provider().await;
    let dir = project()?;
    overbrainer(dir.path(), &server)?
        .arg("run")
        .assert()
        .success()
        .stdout(predicate::str::contains("subtopics: 1 done"))
        .stdout(predicate::str::contains("questions: 2 done"))
        .stdout(predicate::str::contains(
            "answers: 4 done, 0 skipped, 0 failed",
        ))
        .stdout(predicate::str::contains(
            "split: 2 train, 2 eval, 0 excluded",
        ))
        .stdout(predicate::str::contains("cost $"))
        .stdout(predicate::str::contains(KEY).not())
        .stderr(predicate::str::contains("training is not available yet"))
        .stderr(predicate::str::contains(KEY).not());
    for file in ["subtopics", "questions", "answers", "train", "eval"] {
        assert!(
            dir.path().join(format!("data/{file}.jsonl")).is_file(),
            "{file}.jsonl missing"
        );
    }

    overbrainer(dir.path(), &server)?
        .arg("answers")
        .assert()
        .success()
        .stdout(predicate::str::contains("answers: 0 done, 4 skipped"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn questions_generates_missing_subtopics_first() -> TestResult {
    let server = provider().await;
    let dir = project()?;
    overbrainer(dir.path(), &server)?
        .args(["questions", "--topic", "ownership"])
        .assert()
        .success()
        .stdout(predicate::str::contains("subtopics: 1 done"))
        .stdout(predicate::str::contains("questions: 2 done"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn unknown_topic_is_an_error() -> TestResult {
    let server = provider().await;
    let dir = project()?;
    overbrainer(dir.path(), &server)?
        .args(["subtopics", "--topic", "nope"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown topic `nope`"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_rejected_key_fails_without_printing_it() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(401).set_body_json(
            json!({"error": {"message": format!("Incorrect API key provided: {KEY}")}}),
        ))
        .mount(&server)
        .await;
    let dir = project()?;
    overbrainer(dir.path(), &server)?
        .arg("subtopics")
        .assert()
        .failure()
        .stderr(predicate::str::contains("authentication failed"))
        .stderr(predicate::str::contains(KEY).not())
        .stdout(predicate::str::contains(KEY).not());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn failed_items_exit_non_zero_after_the_summary() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"index": 0, "message": {"content": "no list here"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        })))
        .mount(&server)
        .await;
    let dir = project()?;
    let output = overbrainer(dir.path(), &server)?
        .arg("subtopics")
        .assert()
        .failure()
        .stdout(predicate::str::contains(
            "subtopics: 0 done, 0 skipped, 1 failed",
        ))
        .stderr(predicate::str::contains("item(s) failed"))
        .get_output()
        .clone();
    let stdout = String::from_utf8(output.stdout)?;
    assert_eq!(stdout.lines().count(), 1, "stdout holds only the summary");
    Ok(())
}
