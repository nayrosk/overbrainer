//! `overbrainer pod ls`, `pod rm` and the pod column of `runs ls`, against a local
//! stub of the Runpod API.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use assert_cmd::Command;
use overbrainer::runpod::{AttemptResult, DeletedBy, Pod, PodId, PodRecord, PodState};
use overbrainer::runs::{RunRecord, RunState, Runs};
use predicates::prelude::*;
use serde_json::{Value, json};
use wiremock::matchers::any;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const RUNNING: &str = "20260922-143005-a1b2";
const ENDED: &str = "20260921-090000-ffff";
const FORGOTTEN: &str = "20260920-120000-0a0a";
const KEPT: &str = "20260919-080000-beef";

const CONFIG: &str = r#"[project]
name = "demo"

[providers.mock]
protocol = "openai"

[roles]
generator = { provider = "mock", model = "gen" }
parent = { provider = "mock", model = "parent" }

[targets.gpu_cloud]
kind = "runpod"
gpu_types = ["NVIDIA A40"]
max_hours = 6
"#;

fn pod(id: &str, run_id: &str) -> Value {
    json!({
        "id": id,
        "name": format!("overbrainer-{run_id}-1"),
        "status": "RUNNING",
        "cost": 0.53,
        "gpu": {"id": "NVIDIA A40", "count": 1},
        "createdAt": "2026-09-22T14:30:08Z",
        "env": {"OVERBRAINER_RUN_ID": run_id}
    })
}

/// Runpod's own 404.
fn not_found() -> ResponseTemplate {
    ResponseTemplate::new(404).set_body_json(json!({
        "detail": "pod not found",
        "status": 404,
        "title": "Not Found"
    }))
}

/// A stub of the account: `pods` exist (GET answers them) until deleted, the
/// `listed` ones show in the list, and a DELETE of a `forbidden` one is refused.
#[derive(Default)]
struct Account {
    pods: HashMap<String, Value>,
    listed: Vec<String>,
    forbidden: HashSet<String>,
    deleted: Mutex<HashSet<String>>,
}

impl Account {
    fn with(mut self, id: &str, run_id: &str, listed: bool) -> Self {
        self.pods.insert(id.to_string(), pod(id, run_id));
        if listed {
            self.listed.push(id.to_string());
        }
        self
    }

    fn deleted(&self, id: &str) -> bool {
        self.deleted
            .lock()
            .is_ok_and(|deleted| deleted.contains(id))
    }
}

struct Api(Arc<Account>);

impl Respond for Api {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let account = &self.0;
        let path = request.url.path();
        if path == "/v2/pods" {
            let pods: Vec<&Value> = account
                .listed
                .iter()
                .filter(|id| !account.deleted(id))
                .filter_map(|id| account.pods.get(id))
                .collect();
            return ResponseTemplate::new(200).set_body_json(json!({ "pods": pods }));
        }
        let id = path.strip_prefix("/v2/pods/").unwrap_or_default();
        match request.method.as_str() {
            "DELETE" if account.forbidden.contains(id) => {
                ResponseTemplate::new(403).set_body_json(json!({"status": 403}))
            },
            "DELETE" => {
                let known = account.pods.contains_key(id) && !account.deleted(id);
                if let Ok(mut deleted) = account.deleted.lock() {
                    deleted.insert(id.to_string());
                }
                if known {
                    ResponseTemplate::new(204)
                } else {
                    not_found()
                }
            },
            _ => match account.pods.get(id) {
                Some(body) if !account.deleted(id) => {
                    ResponseTemplate::new(200).set_body_json(body)
                },
                _ => not_found(),
            },
        }
    }
}

async fn serve(account: Account) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(any())
        .respond_with(Api(Arc::new(account)))
        .mount(&server)
        .await;
    server
}

/// Pod `p1` of the running run and `p9` of a run unknown here.
async fn stub() -> MockServer {
    serve(
        Account::default()
            .with("p1", RUNNING, true)
            .with("p9", "20260101-000000-dead", true),
    )
    .await
}

fn run_record(id: &str, state: RunState, created: &str) -> RunRecord {
    RunRecord {
        id: id.to_string(),
        target: "gpu_cloud".into(),
        created: created.into(),
        remote_dir: format!("/workspace/overbrainer/{id}"),
        job: None,
        state,
        message: None,
    }
}

/// A run record and its pod record, in `pod_state`.
fn recorded(
    runs: &Runs,
    id: &str,
    state: RunState,
    pod_id: &str,
    pod_state: PodState,
) -> Result<PodRecord, Box<dyn std::error::Error>> {
    runs.save(&run_record(id, state, "2026-09-22T14:30:05Z"))?;
    let mut record = PodRecord::new(id, pod_state == PodState::Kept, 1, "ssh-ed25519 AAAAhost");
    let remote: Pod = serde_json::from_value(pod(pod_id, id))?;
    record.begin_attempt("NVIDIA A40", SystemTime::now(), 6.0);
    record.created(&remote, AttemptResult::Created, SystemTime::now());
    record.state = pod_state;
    record.save(runs)?;
    Ok(record)
}

fn project() -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    std::fs::write(dir.path().join("overbrainer.toml"), CONFIG)?;
    let runs = Runs::new(dir.path());
    recorded(&runs, RUNNING, RunState::Running, "p1", PodState::Running)?;
    recorded(&runs, ENDED, RunState::Succeeded, "p3", PodState::Running)?;
    runs.save(&run_record(
        FORGOTTEN,
        RunState::Failed,
        "2026-09-20T12:00:00Z",
    ))?;
    Ok(dir)
}

/// Adds `strays` to the pod record of the run `id`.
fn add_strays(runs: &Runs, id: &str, strays: &[&str]) -> TestResult {
    let mut record = PodRecord::load(runs, id)?.ok_or("no pod.json")?;
    for stray in strays {
        record.note_stray(PodId::new(stray)?);
    }
    record.save(runs)?;
    Ok(())
}

fn strays(runs: &Runs, id: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let record = PodRecord::load(runs, id)?.ok_or("no pod.json")?;
    Ok(record.stray_pods.iter().map(ToString::to_string).collect())
}

fn overbrainer(dir: &Path, server: &MockServer) -> Result<Command, Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("overbrainer")?;
    cmd.env_clear()
        .env("NO_COLOR", "1")
        .env("HOME", "/nonexistent")
        .env("OVERBRAINER_RUNPOD__API_KEY", "rp_cli_key_4411")
        .env(
            "OVERBRAINER_RUNPOD__BASE_URL",
            format!("{}/v2", server.uri()),
        )
        .arg("-C")
        .arg(dir);
    Ok(cmd)
}

/// Runs `cmd` off the async runtime, which serves the stub meanwhile.
async fn output(mut cmd: Command) -> Result<std::process::Output, Box<dyn std::error::Error>> {
    Ok(tokio::task::spawn_blocking(move || cmd.output()).await??)
}

/// The DELETE calls the stub received, by pod.
async fn deletes(server: &MockServer) -> Vec<String> {
    server
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .filter(|request| request.method.as_str() == "DELETE")
        .map(|request| {
            request
                .url
                .path()
                .rsplit('/')
                .next()
                .unwrap_or_default()
                .to_string()
        })
        .collect()
}

#[tokio::test]
async fn pod_ls_prints_the_table_on_stdout() -> TestResult {
    let server = serve(
        Account::default()
            .with("p1", RUNNING, true)
            .with("p3", ENDED, true)
            .with("p9", "20260101-000000-dead", true),
    )
    .await;
    let dir = project()?;
    let mut cmd = overbrainer(dir.path(), &server)?;
    cmd.args(["pod", "ls"]);
    let output = output(cmd).await?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout)?;
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 4, "{stdout}");
    assert!(lines[0].starts_with("RUN "), "{stdout}");
    assert!(lines[0].ends_with("NOTE"), "{stdout}");
    let row = |run: &str| lines.iter().find(|line| line.starts_with(run)).copied();
    let elsewhere = row("20260101-000000-dead").ok_or("no orphan row")?;
    assert!(
        elsewhere.contains("p9")
            && elsewhere.ends_with(
                "not in this project's runs/; if no other checkout owns it, `overbrainer pod rm 20260101-000000-dead --force`"
            ),
        "{elsewhere}"
    );
    let running = row(RUNNING).ok_or("no running row")?;
    assert!(
        running.contains("RUNNING") && running.contains("0.53") && running.ends_with("run running"),
        "{running}"
    );
    let ended = row(ENDED).ok_or("no ended row")?;
    assert!(ended.ends_with("run succeeded, not deleted"), "{ended}");
    assert!(!stdout.contains("rp_cli_key_4411"));
    assert!(deletes(&server).await.is_empty());
    Ok(())
}

#[tokio::test]
async fn pod_rm_keeps_the_training_pod_of_a_running_run_unless_forced() -> TestResult {
    let server = stub().await;
    let dir = project()?;
    let mut cmd = overbrainer(dir.path(), &server)?;
    cmd.args(["pod", "rm", RUNNING]);
    let refused = output(cmd).await?;
    assert!(!refused.status.success());
    assert!(
        String::from_utf8_lossy(&refused.stderr).contains(&format!(
            "run {RUNNING} is still running: its training pod p1 was kept; stop it with `overbrainer train cancel {RUNNING}` first, or pass --force"
        )),
        "{}",
        String::from_utf8_lossy(&refused.stderr)
    );
    assert!(refused.stdout.is_empty());
    assert!(deletes(&server).await.is_empty());

    let mut cmd = overbrainer(dir.path(), &server)?;
    cmd.args(["pod", "rm", RUNNING, "--force"]);
    let forced = output(cmd).await?;
    assert!(
        forced.status.success(),
        "{}",
        String::from_utf8_lossy(&forced.stderr)
    );
    let stdout = String::from_utf8(forced.stdout)?;
    assert!(stdout.starts_with("pod: p1 deleted after "), "{stdout}");
    assert_eq!(stdout.lines().count(), 1, "{stdout}");
    let runs = Runs::new(dir.path());
    let run = runs.load(RUNNING)?;
    assert_eq!(run.state, RunState::Failed);
    assert_eq!(
        run.message.as_deref(),
        Some("pod deleted by `pod rm` before its results were retrieved")
    );
    let pod = PodRecord::load(&runs, RUNNING)?.ok_or("no pod.json")?;
    assert_eq!(pod.state, PodState::Deleted);
    assert_eq!(pod.deleted_by, Some(DeletedBy::PodRm));
    assert_eq!(deletes(&server).await, vec!["p1"]);

    let mut cmd = overbrainer(dir.path(), &server)?;
    cmd.args(["pod", "rm", FORGOTTEN]);
    let nothing = output(cmd).await?;
    assert!(nothing.status.success());
    assert_eq!(
        String::from_utf8(nothing.stdout)?,
        format!("pod: no pod for run {FORGOTTEN}\n")
    );
    Ok(())
}

#[tokio::test]
async fn pod_rm_prints_the_pods_it_deleted_even_when_it_fails() -> TestResult {
    let mut account = Account::default()
        .with("p3", ENDED, true)
        .with("s1", ENDED, true)
        .with("s3", ENDED, true);
    account.forbidden.insert("s3".to_string());
    let server = serve(account).await;
    let dir = project()?;
    let runs = Runs::new(dir.path());
    add_strays(&runs, ENDED, &["s1", "s3"])?;
    let mut cmd = overbrainer(dir.path(), &server)?;
    cmd.args(["pod", "rm", ENDED]);
    let failed = output(cmd).await?;
    assert!(!failed.status.success());
    let stdout = String::from_utf8(failed.stdout)?;
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 2, "{stdout}");
    assert!(lines[0].starts_with("pod: p3 deleted after "), "{stdout}");
    assert_eq!(lines[1], "pod: s1 deleted");
    assert_eq!(strays(&runs, ENDED)?, vec!["s3"]);
    assert!(!stdout.contains("rp_cli_key_4411"));
    Ok(())
}

#[tokio::test]
async fn pod_rm_removes_a_kept_pod() -> TestResult {
    let server = serve(Account::default().with("p5", KEPT, true)).await;
    let dir = project()?;
    let runs = Runs::new(dir.path());
    recorded(&runs, KEPT, RunState::Succeeded, "p5", PodState::Kept)?;
    let mut cmd = overbrainer(dir.path(), &server)?;
    cmd.args(["pod", "rm", KEPT]);
    let removed = output(cmd).await?;
    assert!(
        removed.status.success(),
        "{}",
        String::from_utf8_lossy(&removed.stderr)
    );
    let stdout = String::from_utf8(removed.stdout)?;
    assert!(stdout.starts_with("pod: p5 deleted after "), "{stdout}");
    let pod = PodRecord::load(&runs, KEPT)?.ok_or("no pod.json")?;
    assert_eq!(pod.state, PodState::Deleted);
    assert_eq!(pod.deleted_by, Some(DeletedBy::PodRm));
    assert_eq!(runs.load(KEPT)?.state, RunState::Succeeded);
    Ok(())
}

#[test]
fn runs_ls_shows_the_pod_of_a_runpod_run() -> TestResult {
    let dir = project()?;
    add_strays(&Runs::new(dir.path()), ENDED, &["s1", "s2"])?;
    Command::cargo_bin("overbrainer")?
        .env_clear()
        .env("HOME", "/nonexistent")
        .arg("-C")
        .arg(dir.path())
        .args(["runs", "ls"])
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "{RUNNING}  running    gpu_cloud  2026-09-22T14:30:05Z  pod running p1 $0.53/h"
        )))
        .stdout(predicate::str::contains(format!(
            "{ENDED}  succeeded  gpu_cloud  2026-09-22T14:30:05Z  pod running p3 $0.53/h +2 stray\n"
        )))
        .stdout(predicate::str::contains(format!(
            "{FORGOTTEN}  failed     gpu_cloud  2026-09-20T12:00:00Z\n"
        )));
    Ok(())
}
