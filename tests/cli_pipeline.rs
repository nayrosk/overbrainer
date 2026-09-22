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
            "split: 3 train, 1 eval, 0 excluded, 0 orphaned",
        ))
        .stdout(predicate::str::contains("cost $"))
        .stdout(predicate::str::contains(KEY).not())
        .stderr(predicate::str::contains("run stops after split"))
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

#[tokio::test(flavor = "multi_thread")]
async fn literal_keys_need_no_vault_token() -> TestResult {
    let server = provider().await;
    let dir = project()?;
    overbrainer(dir.path(), &server)?
        .env("VAULT_ADDR", "http://127.0.0.1:1")
        .arg("split")
        .assert()
        .success()
        .stdout(predicate::str::contains("split: 0 train, 0 eval"));
    overbrainer(dir.path(), &server)?
        .env("VAULT_ADDR", "http://127.0.0.1:1")
        .arg("subtopics")
        .assert()
        .success()
        .stdout(predicate::str::contains("subtopics: 1 done"));
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn run_lists_models_once_per_provider() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": [
            {"id": "gen", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
            {"id": "parent", "pricing": {"prompt": "0.000003", "completion": "0.000015"}}
        ]})))
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(Script {
            calls: AtomicUsize::new(0),
        })
        .mount(&server)
        .await;
    let dir = project()?;
    overbrainer(dir.path(), &server)?
        .arg("run")
        .assert()
        .success()
        .stdout(predicate::str::contains("answers: 4 done"));
    server.verify().await;
    Ok(())
}

#[test]
fn unknown_topic_is_reported_before_connecting() -> TestResult {
    let dir = project()?;
    Command::cargo_bin("overbrainer")?
        .env_clear()
        .env("HOME", "/nonexistent")
        .arg("-C")
        .arg(dir.path())
        .args(["subtopics", "--topic", "nope"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown topic `nope`"));
    Ok(())
}

/// Parent that answers the first request, then rejects the key after a delay, so the
/// first answer is saved before the stage stops.
struct RejectAfterFirst {
    calls: AtomicUsize,
}

impl Respond for RejectAfterFirst {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"index": 0, "message": {"content": "Borrow it.", "reasoning": "Why."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50}
            }))
        } else {
            ResponseTemplate::new(401)
                .set_delay(std::time::Duration::from_millis(300))
                .set_body_json(json!({"error": {"message": format!("bad key {KEY}")}}))
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_fatal_stop_still_prints_what_was_spent() -> TestResult {
    let server = provider().await;
    let dir = project()?;
    overbrainer(dir.path(), &server)?
        .arg("questions")
        .assert()
        .success();
    let rejecting = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(RejectAfterFirst {
            calls: AtomicUsize::new(0),
        })
        .mount(&rejecting)
        .await;
    let output = overbrainer(dir.path(), &rejecting)?
        .arg("answers")
        .assert()
        .failure()
        .stdout(predicate::str::contains(
            "answers: 1 done, 0 skipped, 0 failed, 0 excluded; tokens 100 in, 50 out",
        ))
        .stderr(predicate::str::contains("authentication failed"))
        .stderr(predicate::str::contains(KEY).not())
        .get_output()
        .clone();
    let stdout = String::from_utf8(output.stdout)?;
    assert!(!stdout.contains(KEY));
    assert_eq!(stdout.lines().count(), 1, "stdout holds only the summary");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn split_ignores_prompt_templates() -> TestResult {
    let server = provider().await;
    let dir = project()?;
    std::fs::create_dir_all(dir.path().join("prompts"))?;
    std::fs::write(dir.path().join("prompts/questions.txt"), "{% if broken")?;
    overbrainer(dir.path(), &server)?
        .arg("split")
        .assert()
        .success()
        .stdout(predicate::str::contains("split: 0 train"));
    Ok(())
}

#[test]
fn split_offers_only_topic() -> TestResult {
    Command::cargo_bin("overbrainer")?
        .env_clear()
        .args(["split", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--topic"))
        .stdout(predicate::str::contains("orphaned"))
        .stdout(predicate::str::contains("--force").not());
    Command::cargo_bin("overbrainer")?
        .env_clear()
        .args(["split", "--force"])
        .assert()
        .failure();
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn split_warns_when_every_usable_answer_is_orphaned() -> TestResult {
    let server = provider().await;
    let dir = project()?;
    std::fs::create_dir_all(dir.path().join("data"))?;
    let answer = json!({
        "id": "0123456789abcdef0123456789abcdef",
        "topic": "ownership",
        "subtopic": "Borrowing",
        "messages": [
            {"role": "user", "content": "What is a borrow?"},
            {"role": "assistant", "content": "A reference.", "reasoning_content": "r"}
        ],
        "meta": {"model": "parent", "input_tokens": 1, "output_tokens": 1,
                 "finish_reason": "stop", "reasoning_kind": "raw", "excluded": null}
    });
    std::fs::write(dir.path().join("data/answers.jsonl"), format!("{answer}\n"))?;
    overbrainer(dir.path(), &server)?
        .arg("split")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "split: 0 train, 0 eval, 0 excluded, 1 orphaned",
        ))
        .stderr(predicate::str::contains("usable answers are orphaned"));
    Ok(())
}

const FAKE_AXOLOTL: &str = r#"#!/bin/sh
if [ "$1" = train ]; then
  printf '{"event": "begin", "time": 1, "max_steps": 1}\n' >> "$OVERBRAINER_METRICS"
  printf '{"event": "log", "time": 2, "step": 1, "epoch": 1.0, "max_steps": 1, "loss": 0.75}\n' >> "$OVERBRAINER_METRICS"
  mkdir -p output && echo adapter > output/adapter_model.safetensors
fi
"#;

#[tokio::test(flavor = "multi_thread")]
async fn run_trains_after_split_when_training_is_configured() -> TestResult {
    use std::os::unix::fs::PermissionsExt;

    let server = provider().await;
    let dir = project()?;
    let venv = dir.path().join("venv");
    std::fs::create_dir_all(venv.join("bin"))?;
    std::fs::write(venv.join("bin/axolotl"), FAKE_AXOLOTL)?;
    std::fs::set_permissions(
        venv.join("bin/axolotl"),
        std::fs::Permissions::from_mode(0o755),
    )?;
    let config = format!(
        "{PROJECT}\n[training]\ntarget = \"here\"\nbase_model = \"Qwen/Qwen3-4B\"\nadapter = \"lora\"\n\n[targets.here]\nkind = \"local\"\nruntime = \"native\"\nvenv = \"{}\"\n",
        venv.display()
    );
    std::fs::write(dir.path().join("overbrainer.toml"), config)?;
    overbrainer(dir.path(), &server)?
        // The fake axolotl script calls `mkdir`; without PATH its lookup would be
        // implementation-defined. A fixed PATH keeps the test deterministic and
        // independent of the developer's own PATH.
        .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
        .arg("run")
        .assert()
        .success()
        .stdout(predicate::str::contains("split: 3 train, 1 eval"))
        .stdout(predicate::str::contains(
            "succeeded; step 1/1, epoch 1.00, loss 0.7500; output in runs/",
        ));
    Ok(())
}
