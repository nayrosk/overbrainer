//! `overbrainer train`, `train attach` and `train cancel` on a Runpod target,
//! against a local stub of the Runpod API. A full run on a pod is not covered
//! here: it needs an sshd (see `tests/runpod_ssh.rs`) and, for real, a paid pod.

use std::path::Path;
use std::process::{Output, Stdio};
use std::time::SystemTime;

use assert_cmd::Command;
use overbrainer::runpod::{AttemptResult, Pod, PodRecord, PodState};
use overbrainer::runs::{RunRecord, RunState, Runs};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const KEY: &str = "rp_cli_train_key_7731";
const RUN: &str = "20260922-143005-a1b2";
const ENDED: &str = "20260921-090000-ffff";

/// Runpod's answer to a create for a GPU type without capacity.
const CAPACITY: &str = "There are no longer any instances available with the requested specifications. Please refresh and try again.";

fn project() -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    std::fs::create_dir_all(dir.path().join("data"))?;
    std::fs::write(dir.path().join("data/train.jsonl"), "{\"id\": 1}\n")?;
    std::fs::write(dir.path().join("data/eval.jsonl"), "{\"id\": 2}\n")?;
    std::fs::write(
        dir.path().join("overbrainer.toml"),
        r#"[project]
name = "demo"

[providers.mock]
protocol = "openai"

[roles]
generator = { provider = "mock", model = "gen" }
parent = { provider = "mock", model = "parent" }

[training]
target = "gpu_cloud"
base_model = "Qwen/Qwen3-4B"
adapter = "qlora"

[targets.gpu_cloud]
kind = "runpod"
gpu_types = ["NVIDIA GeForce RTX 4090", "NVIDIA A40"]
max_hours = 6

[targets.here]
kind = "local"
runtime = "native"
"#,
    )?;
    Ok(dir)
}

fn overbrainer(dir: &Path, server: &MockServer) -> Result<Command, Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("overbrainer")?;
    cmd.env_clear()
        .env("NO_COLOR", "1")
        .env("HOME", "/nonexistent")
        .env("PATH", "/usr/bin:/bin")
        .env("OVERBRAINER_RUNPOD__API_KEY", KEY)
        .env(
            "OVERBRAINER_RUNPOD__BASE_URL",
            format!("{}/v2", server.uri()),
        )
        .arg("-C")
        .arg(dir);
    Ok(cmd)
}

async fn output(mut cmd: Command) -> Result<Output, Box<dyn std::error::Error>> {
    Ok(tokio::task::spawn_blocking(move || cmd.output()).await??)
}

fn keygen_available() -> bool {
    let available = std::process::Command::new("ssh-keygen")
        .arg("-?")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok();
    if !available {
        eprintln!("skipped: ssh-keygen is not installed");
    }
    available
}

/// Runpod's own 404.
fn not_found() -> ResponseTemplate {
    ResponseTemplate::new(404).set_body_json(json!({
        "detail": "pod not found",
        "status": 404,
        "title": "Not Found"
    }))
}

/// The run `id` recorded in `state` on pod `pod_id`, whose pod is in `pod_state`.
fn recorded_run(
    dir: &Path,
    id: &str,
    state: RunState,
    pod_id: &str,
    pod_state: PodState,
) -> Result<(), Box<dyn std::error::Error>> {
    let runs = Runs::new(dir);
    runs.save(&RunRecord {
        id: id.to_string(),
        target: "gpu_cloud".into(),
        created: "2026-09-22T14:30:05Z".into(),
        remote_dir: format!("/workspace/overbrainer/{id}"),
        job: Some(serde_json::from_value(
            json!({"dir": format!("/workspace/overbrainer/{id}"), "pid": 42, "container": null}),
        )?),
        state,
        message: None,
    })?;
    let mut pod = PodRecord::new(id, false, 1, "ssh-ed25519 AAAAhost");
    let remote: Pod =
        serde_json::from_value(json!({"id": pod_id, "status": "RUNNING", "cost": 0.5}))?;
    pod.begin_attempt("NVIDIA A40", SystemTime::now(), 6.0);
    pod.created(&remote, AttemptResult::Created, SystemTime::now());
    pod.state = pod_state;
    pod.save(&runs)?;
    Ok(())
}

#[tokio::test]
async fn a_run_without_capacity_fails_with_its_attempts_recorded() -> TestResult {
    if !keygen_available() {
        return Ok(());
    }
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v2/pods"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"pods": [{
            "id": "old1",
            "name": "overbrainer-20260101-000000-dead-1",
            "status": "RUNNING",
            "cost": 0.49,
            "env": {"OVERBRAINER_RUN_ID": "20260101-000000-dead"}
        }]})))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v2/pods"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "detail": CAPACITY,
            "status": 400,
            "title": "Bad Request"
        })))
        .mount(&server)
        .await;
    let dir = project()?;
    // An ended run whose pod the list does not show: the orphan warning never
    // looks it up.
    recorded_run(
        dir.path(),
        ENDED,
        RunState::Succeeded,
        "gone1",
        PodState::Running,
    )?;
    let mut cmd = overbrainer(dir.path(), &server)?;
    cmd.arg("train");
    let output = output(cmd).await?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!output.status.success(), "{stderr}");
    assert!(
        stderr.contains(
            "pod old1 is still on Runpod at $0.49/h: not in this project's runs/; \
             if no other checkout owns it, `overbrainer pod rm 20260101-000000-dead --force`"
        ),
        "{stderr}"
    );
    assert!(stderr.contains("pod: creating overbrainer-"), "{stderr}");
    assert!(
        stderr.contains(
            "pod: NVIDIA A40 unavailable (Runpod answered 400: no capacity for this GPU type)"
        ),
        "{stderr}"
    );
    assert!(
        stderr.contains("error: no gpu_types entry could be placed"),
        "{stderr}"
    );
    assert!(!stderr.contains(CAPACITY), "{stderr}");
    assert!(!stderr.contains(KEY) && !stdout.contains(KEY));
    assert_eq!(stdout, "");
    let looked_up = server
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .filter(|request| request.method.as_str() == "GET" && request.url.path() != "/v2/pods")
        .count();
    assert_eq!(looked_up, 0, "the orphan warning looked a pod up");
    let runs = Runs::new(dir.path());
    let run = runs
        .list()?
        .into_iter()
        .find(|run| run.id != ENDED)
        .ok_or("no run")?;
    assert_eq!(run.state, RunState::Failed);
    let message = run.message.unwrap_or_default();
    assert!(
        message.starts_with("no gpu_types entry could be placed"),
        "{message}"
    );
    let pod = PodRecord::load(&runs, &run.id)?.ok_or("no pod.json")?;
    let gpus: Vec<&str> = pod.attempts.iter().map(|a| a.gpu_type.as_str()).collect();
    assert_eq!(gpus, vec!["NVIDIA GeForce RTX 4090", "NVIDIA A40"]);
    assert!(
        pod.attempts
            .iter()
            .all(|a| a.result == AttemptResult::Unavailable)
    );
    assert_eq!(pod.pod_id, None);
    Ok(())
}

#[tokio::test]
async fn keep_pod_is_refused_off_runpod_and_a_missing_key_is_named() -> TestResult {
    let server = MockServer::start().await;
    let dir = project()?;
    let mut cmd = overbrainer(dir.path(), &server)?;
    cmd.args(["train", "--target", "here", "--keep-pod"]);
    let refused = output(cmd).await?;
    assert!(!refused.status.success());
    assert!(
        String::from_utf8_lossy(&refused.stderr)
            .contains("--keep-pod only applies to a runpod target")
    );
    let mut cmd = overbrainer(dir.path(), &server)?;
    cmd.env_remove("OVERBRAINER_RUNPOD__API_KEY").arg("train");
    let keyless = output(cmd).await?;
    assert!(!keyless.status.success());
    assert!(
        String::from_utf8_lossy(&keyless.stderr)
            .contains("no Runpod API key: set OVERBRAINER_RUNPOD__API_KEY")
    );
    assert!(!dir.path().join("runs").exists());
    assert!(
        server
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty()
    );
    Ok(())
}

#[tokio::test]
async fn a_run_whose_pod_is_gone_has_nothing_to_cancel_and_fails_on_attach() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v2/pods/p1"))
        .respond_with(not_found())
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v2/pods"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"pods": []})))
        .mount(&server)
        .await;
    let dir = project()?;
    recorded_run(dir.path(), RUN, RunState::Running, "p1", PodState::Running)?;
    // Three 404s in a row and a list without it: the pod is gone.
    let mut cmd = overbrainer(dir.path(), &server)?;
    cmd.args(["train", "cancel", RUN]);
    let cancelled = output(cmd).await?;
    let stderr = String::from_utf8_lossy(&cancelled.stderr);
    assert!(!cancelled.status.success(), "{stderr}");
    assert!(
        stderr.contains(&format!("run {RUN} has no pod left: nothing to cancel")),
        "{stderr}"
    );
    let runs = Runs::new(dir.path());
    let pod = PodRecord::load(&runs, RUN)?.ok_or("no pod.json")?;
    assert_eq!(pod.state, PodState::Deleted);
    let requests = server.received_requests().await.unwrap_or_default();
    assert!(
        requests
            .iter()
            .all(|request| request.method.as_str() == "GET")
    );
    assert_eq!(requests.len(), 4, "3 looks at the pod and 1 list");

    // The pod is recorded deleted: attach fails the run from its local files.
    let mut cmd = overbrainer(dir.path(), &server)?;
    cmd.args(["train", "attach", RUN]);
    let attached = output(cmd).await?;
    let stderr = String::from_utf8_lossy(&attached.stderr);
    assert!(!attached.status.success(), "{stderr}");
    assert!(
        stderr.contains("error: pod p1 no longer exists"),
        "{stderr}"
    );
    assert!(
        String::from_utf8_lossy(&attached.stdout).starts_with(&format!("train: run {RUN} failed")),
        "{}",
        String::from_utf8_lossy(&attached.stdout)
    );
    assert_eq!(runs.load(RUN)?.state, RunState::Failed);
    assert_eq!(
        server.received_requests().await.unwrap_or_default().len(),
        4,
        "attach looked at a pod recorded deleted"
    );
    Ok(())
}

#[tokio::test]
async fn attach_refuses_a_run_that_has_no_job_yet() -> TestResult {
    let server = MockServer::start().await;
    let dir = project()?;
    recorded_run(dir.path(), RUN, RunState::Preparing, "p1", PodState::Ready)?;
    let runs = Runs::new(dir.path());
    let mut run = runs.load(RUN)?;
    run.job = None;
    runs.save(&run)?;
    let mut cmd = overbrainer(dir.path(), &server)?;
    cmd.args(["train", "attach", RUN]);
    let attached = output(cmd).await?;
    let stderr = String::from_utf8_lossy(&attached.stderr);
    assert!(!attached.status.success(), "{stderr}");
    assert!(
        stderr.contains(&format!("run {RUN} has no job yet")),
        "{stderr}"
    );
    assert_eq!(runs.load(RUN)?.state, RunState::Preparing);
    assert!(
        server
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty()
    );
    Ok(())
}
