use std::net::TcpListener;
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error>>;

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
parent = { provider = "mock", model = "parent" }
"#;

/// A history line of an earlier `answers` run.
const HISTORY: &str = r#"{"stage":"answers","started_at":"2026-09-28T10:00:00Z","ended_at":"2026-09-28T10:01:00Z","status":"ok","provider":"mock","model":"parent","done":4,"skipped":0,"failed":0,"excluded":1,"input_tokens":400,"output_tokens":200,"cost":0.5}"#;

fn project(listen: Option<&str>) -> Result<tempfile::TempDir, std::io::Error> {
    let dir = tempfile::tempdir()?;
    let metrics = listen.map_or_else(String::new, |listen| {
        format!("\n[metrics]\nlisten = \"{listen}\"\n")
    });
    std::fs::write(
        dir.path().join("overbrainer.toml"),
        format!("{PROJECT}{metrics}"),
    )?;
    std::fs::create_dir_all(dir.path().join(".overbrainer"))?;
    std::fs::write(
        dir.path().join(".overbrainer/history.jsonl"),
        format!("{HISTORY}\n"),
    )?;
    Ok(dir)
}

fn overbrainer(dir: &Path, base_url: &str) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_overbrainer"));
    cmd.env_clear()
        .env("NO_COLOR", "1")
        .env("HOME", "/nonexistent")
        .env("OVERBRAINER_PROVIDERS__MOCK__BASE_URL", base_url)
        .env("OVERBRAINER_PROVIDERS__MOCK__API_KEY", "sk-metrics-test")
        .arg("-C")
        .arg(dir)
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    cmd
}

/// A port free a moment ago.
fn free_port() -> std::io::Result<u16> {
    Ok(TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

async fn wait(child: std::process::Child) -> Result<Output, Box<dyn std::error::Error>> {
    Ok(tokio::time::timeout(
        Duration::from_secs(20),
        tokio::task::spawn_blocking(move || child.wait_with_output()),
    )
    .await???)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_locked_command_serves_its_metrics_while_it_runs() -> TestResult {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(60)))
        .mount(&server)
        .await;
    let port = free_port()?;
    let dir = project(Some(&format!("127.0.0.1:{port}")))?;
    let mut child = overbrainer(dir.path(), &format!("{}/v1", server.uri()))
        .arg("subtopics")
        .spawn()?;
    // Wait for the stage to send its request: it runs now.
    let limit = Instant::now() + Duration::from_secs(20);
    while server
        .received_requests()
        .await
        .unwrap_or_default()
        .is_empty()
    {
        if Instant::now() > limit {
            child.kill()?;
            return Err("the stage never sent its request".into());
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let url = format!("http://127.0.0.1:{port}/metrics");
    let scraped = reqwest::get(&url).await;
    let status = Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()?;
    assert!(status.success());
    let output = wait(child).await?;
    let stderr = String::from_utf8(output.stderr)?;
    let text = scraped?.text().await?;
    for sample in [
        r#"overbrainer_stage_running{stage="subtopics"} 1"#,
        r#"overbrainer_tokens_total{stage="answers",model="parent",direction="in"} 400"#,
        r#"overbrainer_stage_items_total{stage="answers",result="excluded"} 1"#,
        r#"overbrainer_cost_usd_total{stage="answers",model="parent"} 0.5"#,
    ] {
        assert!(text.lines().any(|line| line == sample), "{sample}:\n{text}");
    }
    assert!(!text.contains("sk-metrics-test"), "{text}");
    assert!(
        stderr.contains(&format!("metrics at http://127.0.0.1:{port}/metrics")),
        "{stderr}"
    );
    // The command ended: nothing serves anymore.
    assert!(reqwest::get(&url).await.is_err());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_taken_port_warns_and_the_command_goes_on() -> TestResult {
    let taken = TcpListener::bind("127.0.0.1:0")?;
    let dir = project(Some(&taken.local_addr()?.to_string()))?;
    let output = overbrainer(dir.path(), "http://127.0.0.1:9/v1")
        .arg("split")
        .output()?;
    let stderr = String::from_utf8(output.stderr)?;
    assert!(output.status.success(), "{stderr}");
    assert!(stderr.contains("cannot serve the metrics on"), "{stderr}");
    Ok(())
}

#[test]
fn an_address_open_to_the_network_warns() -> TestResult {
    let dir = project(Some("0.0.0.0:0"))?;
    let output = overbrainer(dir.path(), "http://127.0.0.1:9/v1")
        .arg("split")
        .output()?;
    let stderr = String::from_utf8(output.stderr)?;
    assert!(output.status.success(), "{stderr}");
    assert!(
        stderr.contains("metrics on 0.0.0.0:0 have no authentication"),
        "{stderr}"
    );
    assert!(stderr.contains("metrics at http://0.0.0.0:"), "{stderr}");
    Ok(())
}

#[test]
fn without_an_address_or_the_lock_nothing_is_served() -> TestResult {
    let dir = project(None)?;
    let output = overbrainer(dir.path(), "http://127.0.0.1:9/v1")
        .arg("split")
        .output()?;
    let stderr = String::from_utf8(output.stderr)?;
    assert!(output.status.success(), "{stderr}");
    assert!(!stderr.contains("metrics"), "{stderr}");
    // `history` takes no lock: no endpoint, even with an address.
    let dir = project(Some("127.0.0.1:0"))?;
    let output = overbrainer(dir.path(), "http://127.0.0.1:9/v1")
        .arg("history")
        .output()?;
    let stderr = String::from_utf8(output.stderr)?;
    assert!(output.status.success(), "{stderr}");
    assert!(!stderr.contains("metrics"), "{stderr}");
    Ok(())
}
