//! `overbrainer compare` on a local target: a fake `llama-server` in the
//! llama.cpp cache of the target, a wiremock judge, a run with a GGUF.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use assert_cmd::Command;
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

use crate::common::{FAKE_LLAMA_SERVER, LLAMA_ASSETS};

/// Result type of the tests.
type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Fixed `PATH` of every spawned `overbrainer`: never the developer's.
const PATH: &str = "/usr/bin:/bin";
/// The run compared.
const RUN: &str = "demo_20261006-100000";
/// The judge's API key, which must never show.
const KEY: &str = "sk-judge-secret-41";
/// The llama.cpp release the compare serves the GGUF with.
const TAG: &str = overbrainer::export::LLAMA_CPP_TAG;

/// Whether python3 runs under the fixed `PATH`.
fn python_available() -> bool {
    std::process::Command::new("/usr/bin/env")
        .env_clear()
        .env("PATH", PATH)
        .args(["python3", "--version"])
        .output()
        .is_ok_and(|output| output.status.success())
}

/// The prompt of a judge request.
fn prompt_of(request: &Request) -> String {
    let body: Value = serde_json::from_slice(&request.body).unwrap_or(Value::Null);
    body.pointer("/messages/0/content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// The judge: the answer starting `Child:` wins, else the parent's. With
/// `fail_on`, the first request about that question answers 400, once.
#[derive(Default)]
struct Judge {
    /// A question whose first judging fails.
    fail_on: Option<&'static str>,
    /// Whether that failure happened already.
    failed: AtomicBool,
}

impl Respond for Judge {
    /// Answers a judge request: a 400 for the first judging of `fail_on`,
    /// else the child's answer wins unless the child does not know.
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let prompt = prompt_of(request);
        if let Some(question) = self.fail_on
            && prompt.contains(question)
            && !self.failed.swap(true, Ordering::SeqCst)
        {
            return ResponseTemplate::new(400)
                .set_body_json(json!({"error": {"message": "bad request"}}));
        }
        let after_a = prompt.split("Answer A:\n").nth(1).unwrap_or_default();
        let b = after_a.split("Answer B:\n").nth(1).unwrap_or_default();
        // The child's answer wins; when the child does not know, the parent's.
        let child_a = after_a.starts_with("Child:");
        let parent_a = !b.starts_with("Child:") && after_a.starts_with("Parent:");
        let pick = if child_a || parent_a { "A" } else { "B" };
        ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "content": format!(r#"{{"verdict": "{pick}", "reason": "{pick} answers."}}"#)
            }}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }))
    }
}

/// A project whose run `RUN` succeeded on the local target `here` and has a
/// GGUF, an eval set of `questions`, and the fake server in the target's
/// cache. `pipeline` is added to `overbrainer.toml` as it is.
fn project(
    questions: &[&str],
    pipeline: &str,
) -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let root = dir.path();
    fs::write(
        root.join("overbrainer.toml"),
        format!(
            r#"[project]
name = "demo"

[providers.mock]
protocol = "openai"

[roles]
generator = {{ provider = "mock", model = "gen" }}
parent = {{ provider = "mock", model = "parent" }}

[targets.here]
kind = "local"
runtime = "native"

[compare]
parent_price_in = 1.0
parent_price_out = 2.0
child_price_per_hour = 3.6
server_start_secs = 20
{pipeline}"#
        ),
    )?;
    fs::create_dir_all(root.join("data"))?;
    let mut eval = String::new();
    for (i, text) in questions.iter().enumerate() {
        eval.push_str(&serde_json::to_string(&json!({
            "id": format!("{:032x}", i + 1),
            "topic": "ownership",
            "subtopic": "borrowing",
            "messages": [
                {"role": "user", "content": text},
                {"role": "assistant", "content": format!("Parent: {text}")}
            ],
            "meta": {"model": "parent", "input_tokens": 100, "output_tokens": 300,
                     "finish_reason": "stop", "reasoning_kind": "none", "excluded": null}
        }))?);
        eval.push('\n');
    }
    fs::write(root.join("data/eval.jsonl"), eval)?;
    let run = root.join("runs").join(RUN);
    fs::create_dir_all(run.join("output/gguf"))?;
    let gguf = run.join("output/gguf").join(format!("{RUN}-Q4_K_M.gguf"));
    fs::write(&gguf, "gguf")?;
    fs::write(
        run.join("axolotl.yaml"),
        "base_model: Qwen/Qwen3-4B\nsequence_len: 2048\n",
    )?;
    fs::write(
        run.join("run.json"),
        serde_json::to_string(&json!({
            "id": RUN, "target": "here", "created": "2026-10-06T10:00:00Z",
            "remote_dir": run.to_string_lossy(), "job": null, "state": "succeeded",
            "message": null, "snapshot": null, "resumed_from": null, "snapshots": true
        }))?,
    )?;
    fs::write(
        run.join("export.json"),
        serde_json::to_string(&json!({
            "quantize": "Q4_K_M", "llama_cpp": TAG,
            "file": format!("output/gguf/{RUN}-Q4_K_M.gguf"),
            "sha256": overbrainer::exec::sha256_file(&gguf)?, "size": 4,
            "created": "2026-10-06T11:00:00Z"
        }))?,
    )?;
    // The local target's tools cache is runs/.cache: every build gets the fake server.
    let cache = root.join("runs/.cache/llama.cpp").join(TAG);
    for asset in LLAMA_ASSETS {
        let bin = cache.join(asset);
        fs::create_dir_all(&bin)?;
        fs::write(bin.join("llama-server"), FAKE_LLAMA_SERVER)?;
        fs::set_permissions(bin.join("llama-server"), fs::Permissions::from_mode(0o755))?;
    }
    for arch in ["x64", "arm64"] {
        let cudart = cache.join(format!("cudart-llama-{TAG}-bin-ubuntu-cuda-13.4-{arch}"));
        fs::create_dir_all(&cudart)?;
        fs::write(cudart.join("libcudart.so.13"), "")?;
    }
    Ok(dir)
}

/// `overbrainer -C dir` with a clean environment and the judge's provider.
fn overbrainer(dir: &Path, server: &MockServer) -> Result<Command, Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("overbrainer")?;
    cmd.env_clear()
        .env("NO_COLOR", "1")
        .env("HOME", "/nonexistent")
        .env("PATH", PATH)
        .env(
            "OVERBRAINER_PROVIDERS__MOCK__BASE_URL",
            format!("{}/v1", server.uri()),
        )
        .env("OVERBRAINER_PROVIDERS__MOCK__API_KEY", KEY)
        .arg("-C")
        .arg(dir);
    Ok(cmd)
}

/// Mounts `judge` as the chat completions endpoint of a new mock server.
async fn judge_server(judge: Judge) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(judge)
        .mount(&server)
        .await;
    server
}

/// The only compare directory of `RUN`.
fn only_compare(dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let compares: Vec<_> = fs::read_dir(dir.join("runs").join(RUN).join("compares"))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.join("run.json").is_file())
        .collect();
    match compares.as_slice() {
        [one] => Ok(one.clone()),
        other => Err(format!("expected one compare, found {other:?}").into()),
    }
}

/// The ID of the compare in `compare`, its directory.
fn compare_id(compare: &Path) -> Result<String, Box<dyn std::error::Error>> {
    Ok(compare
        .file_name()
        .ok_or("no compare id")?
        .to_string_lossy()
        .into_owned())
}

/// The prompts the judge was sent.
async fn prompts(server: &MockServer) -> Vec<String> {
    server
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .map(prompt_of)
        .collect()
}

/// A full compare: the child answers, the judge decides, the report is
/// written, its path on stdout; the key never shows.
#[tokio::test]
async fn a_compare_reports_wins_latency_and_cost() -> TestResult {
    if !python_available() {
        eprintln!("skipped: needs python3 on {PATH}");
        return Ok(());
    }
    let server = judge_server(Judge::default()).await;
    let dir = project(&["Why borrow?", "A hard one?", "What is a lifetime?"], "")?;
    let output = overbrainer(dir.path(), &server)?
        .args(["compare", "--limit", "3"])
        .output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout}{stderr}");
    let compare = only_compare(dir.path())?;
    let markdown = compare.join("compare.md");
    assert_eq!(stdout.trim(), markdown.to_string_lossy());
    let report = fs::read_to_string(&markdown)?;
    assert!(report.contains("| Win or tie | 66.7% |"), "{report}");
    assert!(
        report.contains("| Parent cost per 1,000 requests | $0.7000 |"),
        "{report}"
    );
    assert!(report.contains("The judge is the parent model"), "{report}");
    let json: Value = serde_json::from_str(&fs::read_to_string(compare.join("compare.json"))?)?;
    assert_eq!(json["summary"]["wins"], 2);
    assert_eq!(json["summary"]["losses"], 1);
    assert!(!compare.join("model.gguf").exists(), "no GGUF in the job");
    assert!(
        dir.path()
            .join("runs")
            .join(RUN)
            .join("output/gguf")
            .join(format!("{RUN}-Q4_K_M.gguf"))
            .is_file(),
        "the run's GGUF stays"
    );
    for text in [stdout.as_ref(), stderr.as_ref(), report.as_str()] {
        assert!(!text.contains(KEY), "the key leaked");
    }

    // A rejudge asks the child nothing and the judge nothing new.
    let id = compare_id(&compare)?;
    let before = prompts(&server).await.len();
    let again = overbrainer(dir.path(), &server)?
        .args(["compare", "--rejudge", &id])
        .output()?;
    assert!(
        again.status.success(),
        "{}",
        String::from_utf8_lossy(&again.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&again.stdout).trim(),
        markdown.to_string_lossy()
    );
    assert_eq!(prompts(&server).await.len(), before);
    Ok(())
}

/// A judge failing on a question keeps the verdicts so far and says how to
/// resume; `--rejudge` then asks the judge only the missing question.
#[tokio::test]
async fn a_failed_judge_keeps_its_verdicts_and_a_rejudge_resumes() -> TestResult {
    if !python_available() {
        eprintln!("skipped: needs python3 on {PATH}");
        return Ok(());
    }
    let judge = Judge {
        fail_on: Some("What is a lifetime?"),
        ..Judge::default()
    };
    let server = judge_server(judge).await;
    // One request at a time: the failing question is judged last.
    let dir = project(
        &["Why borrow?", "A hard one?", "What is a lifetime?"],
        "\n[pipeline]\nconcurrency = 1\n",
    )?;
    let output = overbrainer(dir.path(), &server)?.arg("compare").output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{stderr}");
    assert_eq!(String::from_utf8_lossy(&output.stdout), "");
    let compare = only_compare(dir.path())?;
    let id = compare_id(&compare)?;
    assert!(
        stderr.contains(&format!(
            "the verdicts so far are kept; resume with `overbrainer compare --rejudge {id}`"
        )),
        "{stderr}"
    );
    assert!(
        !compare.join("compare.md").exists(),
        "a report without every verdict"
    );
    let verdicts: Vec<PathBuf> = fs::read_dir(&compare)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.to_string_lossy().contains("verdicts-"))
        .collect();
    let [verdicts] = verdicts.as_slice() else {
        return Err(format!("expected one verdicts file, found {verdicts:?}").into());
    };
    assert_eq!(fs::read_to_string(verdicts)?.lines().count(), 2);

    let before = prompts(&server).await.len();
    let again = overbrainer(dir.path(), &server)?
        .args(["compare", "--rejudge", &id])
        .output()?;
    assert!(
        again.status.success(),
        "{}",
        String::from_utf8_lossy(&again.stderr)
    );
    let asked = prompts(&server).await;
    let resumed = asked.get(before..).unwrap_or_default();
    assert_eq!(resumed.len(), 1, "{resumed:?}");
    let first = resumed.first().ok_or("no prompt resumed")?;
    assert!(first.contains("What is a lifetime?"), "{resumed:?}");
    let json: Value = serde_json::from_str(&fs::read_to_string(compare.join("compare.json"))?)?;
    assert_eq!(json["summary"]["wins"], 2);
    assert_eq!(json["summary"]["losses"], 1);
    for text in [
        String::from_utf8_lossy(&again.stdout),
        String::from_utf8_lossy(&again.stderr),
        stderr,
    ] {
        assert!(!text.contains(KEY), "the key leaked");
    }
    Ok(())
}

/// Without a GGUF, compare says to export first; --keep-pod is refused off Runpod.
#[tokio::test]
async fn a_run_without_a_gguf_is_refused() -> TestResult {
    let server = MockServer::start().await;
    let dir = project(&["Why?"], "")?;
    fs::remove_file(dir.path().join("runs").join(RUN).join("export.json"))?;
    let output = overbrainer(dir.path(), &server)?
        .args(["compare", "--run", RUN])
        .output()?;
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(&format!(
            "run {RUN} has no GGUF; run `overbrainer export {RUN}` first"
        )),
        "{stderr}"
    );
    let dir = project(&["Why?"], "")?;
    let output = overbrainer(dir.path(), &server)?
        .args(["compare", "--keep-pod"])
        .output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        stderr.contains("--keep-pod only applies to a runpod target"),
        "{stderr}"
    );
    assert!(
        !dir.path().join("runs").join(RUN).join("compares").exists(),
        "a refused compare leaves no record"
    );
    Ok(())
}
