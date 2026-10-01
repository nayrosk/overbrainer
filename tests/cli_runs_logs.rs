//! `overbrainer runs logs`: the job's log, and with `--pod` the pod's logs
//! kept in the run directory, then read from a local stub of Runpod while the
//! pod exists.

use std::path::Path;

use assert_cmd::Command;
use overbrainer::runpod::{PodId, PodRecord, PodState};
use overbrainer::runs::{RunRecord, RunState, Runs};
use predicates::prelude::*;
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const RUN: &str = "20260922-143005-a1b2";

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

fn pod_line(ts: &str, source: &str, line: &str) -> String {
    json!({"ts": ts, "source": source, "line": line}).to_string()
}

/// A project with the run [`RUN`], its pod `p1` recorded in `pod_state`.
fn project(pod_state: PodState) -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    std::fs::write(dir.path().join("overbrainer.toml"), CONFIG)?;
    let runs = Runs::new(dir.path());
    runs.save(&RunRecord {
        id: RUN.to_string(),
        target: "gpu_cloud".into(),
        created: "2026-09-22T14:30:05Z".into(),
        remote_dir: format!("/workspace/overbrainer/{RUN}"),
        job: None,
        state: RunState::Running,
        message: None,
        snapshot: None,
        resumed_from: None,
        snapshots: true,
    })?;
    let mut record = PodRecord::new(RUN, false, 1, "ssh-ed25519 AAAAhost");
    record.pod_id = Some(PodId::new("p1")?);
    record.state = pod_state;
    record.save(&runs)?;
    Ok(dir)
}

fn run_dir(dir: &Path) -> std::path::PathBuf {
    dir.join("runs").join(RUN)
}

/// Keeps pod lines, a cursor and the bootstrap's log in the run directory.
fn keep_pod_logs(dir: &Path) -> TestResult {
    let pod = run_dir(dir).join(".pod");
    std::fs::create_dir_all(&pod)?;
    let lines = [
        pod_line("2026-09-22T14:30:10Z", "system", "pulling image"),
        pod_line("2026-09-22T14:31:00Z", "container", "step 1\r"),
        pod_line("2026-09-22T14:32:00Z", "container", "step 2"),
    ];
    std::fs::write(pod.join("pod.log"), format!("{}\n", lines.join("\n")))?;
    std::fs::write(pod.join("pod.log.cursor"), "c/3\n")?;
    std::fs::write(
        pod.join("bootstrap.log"),
        "2026-09-22T14:30:20Z bootstrap: run\n",
    )?;
    Ok(())
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

#[test]
fn the_job_log_is_printed_redacted_and_tailed() -> TestResult {
    let dir = project(PodState::Deleted)?;
    std::fs::write(
        run_dir(dir.path()).join("job.log"),
        "first\nsecond HF_TOKEN=abc\nthird\n",
    )?;
    overbrainer(dir.path())?
        .args(["runs", "logs", RUN, "--tail", "2"])
        .assert()
        .success()
        .stdout("second HF_TOKEN=***\nthird\n");
    Ok(())
}

#[test]
fn a_run_without_a_job_log_says_so() -> TestResult {
    let dir = project(PodState::Deleted)?;
    overbrainer(dir.path())?
        .args(["runs", "logs", RUN])
        .assert()
        .success()
        .stdout(format!("runs: no job.log for run {RUN} yet\n"));
    overbrainer(dir.path())?
        .args(["runs", "logs", "20260101-000000-dead"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "no run 20260101-000000-dead in runs/",
        ));
    Ok(())
}

#[test]
fn the_kept_pod_logs_of_a_deleted_pod_are_printed_without_runpod() -> TestResult {
    let dir = project(PodState::Deleted)?;
    keep_pod_logs(dir.path())?;
    overbrainer(dir.path())?
        .args(["runs", "logs", RUN, "--pod"])
        .assert()
        .success()
        .stdout(
            "== .pod/bootstrap.log ==\n\
             2026-09-22T14:30:20Z bootstrap: run\n\
             == pod ==\n\
             2026-09-22T14:30:10Z sys pulling image\n\
             2026-09-22T14:31:00Z ctr step 1\\r\n\
             2026-09-22T14:32:00Z ctr step 2\n",
        );
    overbrainer(dir.path())?
        .args([
            "runs",
            "logs",
            RUN,
            "--pod",
            "--source",
            "container",
            "--tail",
            "1",
        ])
        .assert()
        .success()
        .stdout(
            "== .pod/bootstrap.log ==\n\
             2026-09-22T14:30:20Z bootstrap: run\n\
             == pod ==\n\
             2026-09-22T14:32:00Z ctr step 2\n",
        );
    overbrainer(dir.path())?
        .args(["runs", "logs", RUN, "--pod", "--source", "system"])
        .assert()
        .success()
        .stdout("2026-09-22T14:30:10Z sys pulling image\n");
    Ok(())
}

#[test]
fn a_run_with_nothing_kept_says_so_and_follow_needs_pod() -> TestResult {
    let dir = project(PodState::Deleted)?;
    overbrainer(dir.path())?
        .args(["runs", "logs", RUN, "--pod"])
        .assert()
        .success()
        .stdout(format!("runs: no pod log kept for run {RUN}\n"));
    overbrainer(dir.path())?
        .args(["runs", "logs", RUN, "--follow"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--pod"));
    Ok(())
}

#[tokio::test]
async fn a_live_pod_is_caught_up_from_the_kept_cursor() -> TestResult {
    let dir = project(PodState::Running)?;
    keep_pod_logs(dir.path())?;
    let server = MockServer::start().await;
    let data = pod_line("2026-09-22T14:33:00Z", "container", "step 3");
    Mock::given(method("GET"))
        .and(path("/v2/pods/p1/logs"))
        .and(header("last-event-id", "c/3"))
        .and(header("authorization", "Bearer rp_cli_key_4411"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(format!("id: c/4\ndata: {data}\n\n"), "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;
    let mut cmd = overbrainer(dir.path())?;
    cmd.env("OVERBRAINER_RUNPOD__API_KEY", "rp_cli_key_4411")
        .env(
            "OVERBRAINER_RUNPOD__BASE_URL",
            format!("{}/v2", server.uri()),
        )
        .args(["runs", "logs", RUN, "--pod", "--source", "container"]);
    let output = tokio::task::spawn_blocking(move || cmd.output()).await??;
    let stdout = String::from_utf8(output.stdout)?;
    assert!(output.status.success(), "{stdout}");
    assert!(
        stdout.ends_with("2026-09-22T14:32:00Z ctr step 2\n2026-09-22T14:33:00Z ctr step 3\n"),
        "{stdout}"
    );
    // Read-only: nothing new is kept.
    let kept = std::fs::read_to_string(run_dir(dir.path()).join(".pod/pod.log"))?;
    assert!(!kept.contains("step 3"));
    Ok(())
}

#[test]
fn terminal_controls_and_forged_fields_are_escaped() -> TestResult {
    let dir = project(PodState::Deleted)?;
    std::fs::write(run_dir(dir.path()).join("job.log"), "a\u{1b}[2Jb\n")?;
    overbrainer(dir.path())?
        .args(["runs", "logs", RUN])
        .assert()
        .success()
        .stdout("a\\u{1b}[2Jb\n");
    let pod = run_dir(dir.path()).join(".pod");
    std::fs::create_dir_all(&pod)?;
    std::fs::write(
        pod.join("pod.log"),
        format!("{}\n", pod_line("\u{1b}]0;title\u{7}", "\u{1b}[31m", "x")),
    )?;
    overbrainer(dir.path())?
        .args(["runs", "logs", RUN, "--pod"])
        .assert()
        .success()
        .stdout(" raw x\n");
    Ok(())
}

/// `runs logs | head`: the closed pipe ends the command quietly.
#[test]
fn a_closed_standard_output_ends_quietly() -> TestResult {
    let dir = project(PodState::Deleted)?;
    let text = "line of the job\n".repeat(50_000);
    std::fs::write(run_dir(dir.path()).join("job.log"), text)?;
    let mut child = std::process::Command::new(assert_cmd::cargo::cargo_bin("overbrainer"))
        .env_clear()
        .env("HOME", "/nonexistent")
        .arg("-C")
        .arg(dir.path())
        .args(["runs", "logs", RUN])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;
    drop(child.stdout.take());
    let output = child.wait_with_output()?;
    let stderr = String::from_utf8(output.stderr)?;
    assert!(output.status.success(), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
    Ok(())
}
