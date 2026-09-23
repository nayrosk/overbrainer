//! The pod watchdog and the bootstrap's functions, run for real under `sh`, `dash`
//! and `busybox sh` (each when installed) against a local stub of the Runpod API.
//! The watchdog tests need `curl` and are skipped without it.

use std::fmt;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command as StdCommand, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use overbrainer::runpod::{PodKeys, USER_AGENT, base64, bootstrap_functions, watchdog_script};
use secrecy::ExposeSecret;
use tokio::process::{Child, Command};
use wiremock::matchers::{body_json, method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const KEY: &str = "pod-scoped-test-key-9c1e";
const POD: &str = "pod1";

/// A shell the scripts must run under.
#[derive(Clone, Copy)]
struct Shell(&'static [&'static str]);

impl fmt::Display for Shell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0.join(" "))
    }
}

/// Every shell of `sh`, `dash` and `busybox sh` installed here, probed once.
fn shells() -> &'static [Shell] {
    static SHELLS: OnceLock<Vec<Shell>> = OnceLock::new();
    SHELLS.get_or_init(|| {
        [Shell(&["sh"]), Shell(&["dash"]), Shell(&["busybox", "sh"])]
            .into_iter()
            .filter(|shell| {
                let available = StdCommand::new(shell.0[0])
                    .args(&shell.0[1..])
                    .args(["-c", ":"])
                    .status()
                    .is_ok_and(|status| status.success());
                if !available {
                    eprintln!("skipped: {shell} is not installed");
                }
                available
            })
            .collect()
    })
}

fn curl_available() -> bool {
    let available = StdCommand::new("curl")
        .arg("--version")
        .stdout(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if !available {
        eprintln!("skipped: curl is not installed");
    }
    available
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

/// A run directory and a watchdog script on disk.
struct Pod {
    root: tempfile::TempDir,
}

impl Pod {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        fs::create_dir_all(root.path().join("run/.pod"))?;
        fs::write(root.path().join("watchdog.sh"), watchdog_script())?;
        Ok(Self { root })
    }

    fn run_dir(&self) -> PathBuf {
        self.root.path().join("run")
    }

    fn file(&self, name: &str) -> PathBuf {
        self.run_dir().join(name)
    }

    fn read(&self, name: &str) -> String {
        fs::read_to_string(self.file(name)).unwrap_or_default()
    }

    /// Starts the watchdog under `shell`, polling every second, with `env` on top
    /// of the defaults (key, pod ID, stub URL, graces of an hour, no deadline).
    fn start(
        &self,
        shell: Shell,
        server: &MockServer,
        env: &[(&str, String)],
    ) -> std::io::Result<Child> {
        let mut command = Command::new(shell.0[0]);
        command
            .args(&shell.0[1..])
            .arg(self.root.path().join("watchdog.sh"))
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("RUNPOD_API_KEY", KEY)
            .env("RUNPOD_POD_ID", POD)
            .env("OVERBRAINER_RUN_DIR", self.run_dir())
            .env("OVERBRAINER_API_URL", format!("{}/v2", server.uri()))
            .env("OVERBRAINER_INTERVAL", "1")
            .env("OVERBRAINER_PROBE_WAIT", "0")
            .env("OVERBRAINER_BOOT_GRACE", "3600")
            .env("OVERBRAINER_RETRIEVE_GRACE", "3600")
            .env("OVERBRAINER_VERSION", "9.9.9")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        for (name, value) in env {
            if value.is_empty() {
                command.env_remove(name);
            } else {
                command.env(name, value);
            }
        }
        command.spawn()
    }
}

/// Waits until `done` holds, for at most `limit`.
async fn until(limit: Duration, mut done: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < limit {
        if done() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    done()
}

/// Waits for the watchdog to exit and returns its output, failing after `limit`.
async fn finished(
    child: Child,
    limit: Duration,
) -> Result<(i32, String), Box<dyn std::error::Error>> {
    let output = tokio::time::timeout(limit, child.wait_with_output()).await??;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok((output.status.code().unwrap_or(-1), text))
}

/// A stub that answers the probe with `probe` and a delete with `delete`.
async fn stub(probe: u16, delete: u16) -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(format!("/v2/pods/{POD}")))
        .respond_with(ResponseTemplate::new(probe))
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path(format!("/v2/pods/{POD}")))
        .respond_with(ResponseTemplate::new(delete))
        .mount(&server)
        .await;
    server
}

async fn requests(server: &MockServer) -> Vec<Request> {
    server.received_requests().await.unwrap_or_default()
}

async fn deletes(server: &MockServer) -> usize {
    requests(server)
        .await
        .iter()
        .filter(|request| request.method.as_str() == "DELETE")
        .count()
}

/// Every request carried the key and the watchdog's user agent, and the key
/// never reached the output nor the log.
async fn assert_clean(server: &MockServer, pod: &Pod, output: &str, shell: Shell) {
    for request in requests(server).await {
        let header = |name: &str| {
            request
                .headers
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(str::to_string)
        };
        assert_eq!(
            header("authorization"),
            Some(format!("Bearer {KEY}")),
            "{shell}"
        );
        assert_eq!(
            header("user-agent").as_deref(),
            Some("overbrainer-watchdog/9.9.9"),
            "{shell}"
        );
    }
    assert!(!output.contains(KEY), "{shell}: {output}");
    assert!(!pod.read(".pod/watchdog.log").contains(KEY), "{shell}");
    assert_ne!(USER_AGENT, "overbrainer-watchdog/9.9.9");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_readable_pod_is_ready_and_a_kept_pod_is_never_deleted() -> TestResult {
    if !curl_available() {
        return Ok(());
    }
    for &shell in shells() {
        let server = stub(200, 204).await;
        let pod = Pod::new()?;
        // Everything that would delete an ordinary pod at once.
        fs::write(pod.file("exit_code"), "0\n")?;
        fs::write(pod.file(".pod/retrieved"), "")?;
        let child = pod.start(
            shell,
            &server,
            &[
                ("OVERBRAINER_KEEP_POD", "1".into()),
                ("OVERBRAINER_DEADLINE", (unix_now() - 10).to_string()),
                ("OVERBRAINER_BOOT_GRACE", "1".into()),
                ("OVERBRAINER_RETRIEVE_GRACE", "1".into()),
            ],
        )?;
        assert!(
            until(Duration::from_secs(10), || pod.read(".pod/watchdog")
                == "ready\n")
            .await,
            "{shell}: no verdict"
        );
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert_eq!(deletes(&server).await, 0, "{shell}: a kept pod was deleted");
        drop(child);
        assert_clean(&server, &pod, "", shell).await;
        let log = pod.read(".pod/watchdog.log");
        assert!(log.contains("probe ready"), "{shell}: {log}");
        assert!(log.contains("retrieved marker seen"), "{shell}: {log}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_refused_or_missing_key_is_a_failed_verdict() -> TestResult {
    if !curl_available() {
        return Ok(());
    }
    for &shell in shells() {
        let server = stub(401, 204).await;
        let pod = Pod::new()?;
        let child = pod.start(shell, &server, &[("OVERBRAINER_KEEP_POD", "1".into())])?;
        assert!(
            until(Duration::from_secs(10), || {
                pod.read(".pod/watchdog") == "failed http_401\n"
            })
            .await,
            "{shell}: {:?}",
            pod.read(".pod/watchdog")
        );
        drop(child);

        let silent = stub(200, 204).await;
        let pod = Pod::new()?;
        let child = pod.start(
            shell,
            &silent,
            &[
                ("OVERBRAINER_KEEP_POD", "1".into()),
                ("RUNPOD_API_KEY", String::new()),
            ],
        )?;
        assert!(
            until(Duration::from_secs(10), || {
                pod.read(".pod/watchdog") == "failed missing_key\n"
            })
            .await,
            "{shell}: {:?}",
            pod.read(".pod/watchdog")
        );
        drop(child);
        assert!(requests(&silent).await.is_empty(), "{shell}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_transient_probe_failure_is_retried() -> TestResult {
    if !curl_available() {
        return Ok(());
    }
    for &shell in shells() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(503))
            .up_to_n_times(2)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;
        let pod = Pod::new()?;
        let child = pod.start(shell, &server, &[("OVERBRAINER_KEEP_POD", "1".into())])?;
        assert!(
            until(Duration::from_secs(10), || pod.read(".pod/watchdog")
                == "ready\n")
            .await,
            "{shell}"
        );
        drop(child);
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_past_deadline_deletes_at_once_and_the_watchdog_exits() -> TestResult {
    if !curl_available() {
        return Ok(());
    }
    for &shell in shells() {
        let server = stub(200, 204).await;
        let pod = Pod::new()?;
        let child = pod.start(
            shell,
            &server,
            &[("OVERBRAINER_DEADLINE", (unix_now() - 1).to_string())],
        )?;
        let (code, output) = finished(child, Duration::from_secs(15)).await?;
        assert_eq!(code, 0, "{shell}: {output}");
        assert_eq!(deletes(&server).await, 1, "{shell}");
        assert!(
            output.contains("delete reason=deadline"),
            "{shell}: {output}"
        );
        assert!(output.contains("DELETE 204"), "{shell}: {output}");
        assert_clean(&server, &pod, &output, shell).await;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_job_that_never_starts_is_deleted_after_the_boot_grace() -> TestResult {
    if !curl_available() {
        return Ok(());
    }
    for &shell in shells() {
        let server = stub(200, 404).await;
        let pod = Pod::new()?;
        let child = pod.start(shell, &server, &[("OVERBRAINER_BOOT_GRACE", "2".into())])?;
        let (code, output) = finished(child, Duration::from_secs(15)).await?;
        assert_eq!(code, 0, "{shell}: {output}");
        assert!(
            output.contains("delete reason=never_started"),
            "{shell}: {output}"
        );
        assert!(output.contains("DELETE 404"), "{shell}: {output}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn an_ended_job_is_deleted_after_the_retrieve_grace_or_at_once_when_retrieved() -> TestResult
{
    if !curl_available() {
        return Ok(());
    }
    for &shell in shells() {
        let server = stub(200, 204).await;
        let pod = Pod::new()?;
        fs::write(pod.file("job.pid"), "999999\n")?;
        fs::write(pod.file("exit_code"), "0\n")?;
        let started = Instant::now();
        // A grace of 3, not 2: `now()` is `date +%s`, whole seconds, so a grace
        // of 2 can fire after little more than 1 real second (ended_at sampled
        // just before its second ticks over, the check just after the next one
        // does), which would make the `>= 2s` assertion below flaky. A grace of
        // 3 keeps the same assertion comfortably clear of that rounding.
        let child = pod.start(
            shell,
            &server,
            &[("OVERBRAINER_RETRIEVE_GRACE", "3".into())],
        )?;
        let (code, output) = finished(child, Duration::from_secs(15)).await?;
        assert_eq!(code, 0, "{shell}: {output}");
        assert!(started.elapsed() >= Duration::from_secs(2), "{shell}");
        assert!(output.contains("job ended"), "{shell}: {output}");
        assert!(
            output.contains("delete reason=abandoned"),
            "{shell}: {output}"
        );

        let server = stub(200, 204).await;
        let pod = Pod::new()?;
        fs::write(pod.file("job.pid"), "999999\n")?;
        fs::write(pod.file("exit_code"), "0\n")?;
        fs::write(pod.file(".pod/retrieved"), "")?;
        let child = pod.start(shell, &server, &[])?;
        let (code, output) = finished(child, Duration::from_secs(15)).await?;
        assert_eq!(code, 0, "{shell}: {output}");
        assert!(
            output.contains("delete reason=retrieved"),
            "{shell}: {output}"
        );
    }
    Ok(())
}

/// Starts `sleep 60` as the leader of its own process group.
fn live_group() -> std::io::Result<std::process::Child> {
    StdCommand::new("sleep")
        .arg("60")
        .process_group(0)
        .stdout(Stdio::null())
        .spawn()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_live_job_is_never_ended_and_a_killed_one_is() -> TestResult {
    if !curl_available() {
        return Ok(());
    }
    for &shell in shells() {
        let server = stub(200, 204).await;
        let pod = Pod::new()?;
        let mut job = live_group()?;
        fs::write(pod.file("job.pid"), format!("{}\n", job.id()))?;
        let child = pod.start(
            shell,
            &server,
            &[
                ("OVERBRAINER_BOOT_GRACE", "1".into()),
                ("OVERBRAINER_RETRIEVE_GRACE", "1".into()),
            ],
        )?;
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert_eq!(
            deletes(&server).await,
            0,
            "{shell}: a running job's pod was deleted"
        );
        job.kill()?;
        job.wait()?;
        let (code, output) = finished(child, Duration::from_secs(15)).await?;
        assert_eq!(code, 0, "{shell}: {output}");
        assert!(output.contains("job started"), "{shell}: {output}");
        assert!(
            output.contains("delete reason=abandoned"),
            "{shell}: {output}"
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_failed_delete_falls_back_to_terminate_then_stop() -> TestResult {
    if !curl_available() {
        return Ok(());
    }
    for &shell in shells() {
        let server = stub(200, 500).await;
        Mock::given(method("POST"))
            .and(path(format!("/v2/pods/{POD}/action")))
            .and(body_json(serde_json::json!({"action": "terminate"})))
            .respond_with(ResponseTemplate::new(403))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(format!("/v2/pods/{POD}/action")))
            .and(body_json(serde_json::json!({"action": "stop"})))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;
        let pod = Pod::new()?;
        let child = pod.start(
            shell,
            &server,
            &[("OVERBRAINER_DEADLINE", (unix_now() - 1).to_string())],
        )?;
        let (code, output) = finished(child, Duration::from_secs(15)).await?;
        assert_eq!(code, 0, "{shell}: {output}");
        let calls: Vec<String> = requests(&server)
            .await
            .iter()
            .map(|request| {
                format!(
                    "{} {}",
                    request.method,
                    String::from_utf8_lossy(&request.body)
                )
            })
            .collect();
        assert_eq!(
            calls,
            vec![
                "GET ".to_string(),
                "DELETE ".to_string(),
                "POST {\"action\":\"terminate\"}".to_string(),
                "POST {\"action\":\"stop\"}".to_string(),
            ],
            "{shell}"
        );
        assert!(output.contains("terminate 403"), "{shell}: {output}");
        assert!(
            output.contains("stopped (the volume keeps billing until overbrainer pod rm)"),
            "{shell}: {output}"
        );
        assert!(!output.contains(" deleted\n"), "{shell}: {output}");
        assert_clean(&server, &pod, &output, shell).await;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_delete_that_fails_once_is_retried_and_succeeds() -> TestResult {
    if !curl_available() {
        return Ok(());
    }
    for &shell in shells() {
        // A fresh server, not `stub()`: it would mount its own unconditional
        // DELETE mock alongside these two, and an already-matching earlier mock
        // wins over a later, more specific one.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path(format!("/v2/pods/{POD}")))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(path(format!("/v2/pods/{POD}")))
            .respond_with(ResponseTemplate::new(500))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("DELETE"))
            .and(path(format!("/v2/pods/{POD}")))
            .respond_with(ResponseTemplate::new(204))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path(format!("/v2/pods/{POD}/action")))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let pod = Pod::new()?;
        let child = pod.start(
            shell,
            &server,
            &[("OVERBRAINER_DEADLINE", (unix_now() - 1).to_string())],
        )?;
        let (code, output) = finished(child, Duration::from_secs(15)).await?;
        assert_eq!(code, 0, "{shell}: {output}");
        assert_eq!(deletes(&server).await, 2, "{shell}: {output}");
        assert_eq!(
            output.matches("delete reason=deadline").count(),
            2,
            "{shell}: {output}"
        );
        assert!(output.contains("DELETE 500"), "{shell}: {output}");
        assert!(output.contains("DELETE 204"), "{shell}: {output}");
        assert!(output.contains(" deleted\n"), "{shell}: {output}");
        assert_clean(&server, &pod, &output, shell).await;
    }
    Ok(())
}

/// A directory on `PATH` holding a `runpodctl` that succeeds a `pod delete` and
/// fails everything else, and its full path.
fn fake_runpodctl() -> Result<(tempfile::TempDir, String), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let script = dir.path().join("runpodctl");
    fs::write(
        &script,
        "#!/bin/sh\ncase \"$1 $2\" in\n'pod delete') exit 0 ;;\n*) exit 1 ;;\nesac\n",
    )?;
    fs::set_permissions(&script, fs::Permissions::from_mode(0o700))?;
    let path = format!(
        "{}:{}",
        dir.path().display(),
        std::env::var("PATH").unwrap_or_default()
    );
    Ok((dir, path))
}

#[tokio::test(flavor = "multi_thread")]
async fn a_runpodctl_on_path_is_the_last_fallback() -> TestResult {
    if !curl_available() {
        return Ok(());
    }
    for &shell in shells() {
        let server = stub(200, 500).await;
        Mock::given(method("POST"))
            .and(path(format!("/v2/pods/{POD}/action")))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let pod = Pod::new()?;
        let (_dir, path_with_runpodctl) = fake_runpodctl()?;
        let child = pod.start(
            shell,
            &server,
            &[
                ("OVERBRAINER_DEADLINE", (unix_now() - 1).to_string()),
                ("PATH", path_with_runpodctl),
            ],
        )?;
        let (code, output) = finished(child, Duration::from_secs(15)).await?;
        assert_eq!(code, 0, "{shell}: {output}");
        assert!(output.contains("runpodctl delete ok"), "{shell}: {output}");
        assert!(output.contains(" deleted\n"), "{shell}: {output}");
        assert_clean(&server, &pod, &output, shell).await;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_cancelled_marker_ends_the_job() -> TestResult {
    if !curl_available() {
        return Ok(());
    }
    for &shell in shells() {
        let server = stub(200, 204).await;
        let pod = Pod::new()?;
        fs::write(pod.file("job.pid"), "999999\n")?;
        fs::write(pod.file("cancelled"), "")?;
        let child = pod.start(
            shell,
            &server,
            &[("OVERBRAINER_RETRIEVE_GRACE", "3".into())],
        )?;
        let started = Instant::now();
        let (code, output) = finished(child, Duration::from_secs(15)).await?;
        assert_eq!(code, 0, "{shell}: {output}");
        assert!(started.elapsed() >= Duration::from_secs(2), "{shell}");
        assert!(output.contains("job ended"), "{shell}: {output}");
        assert!(
            output.contains("delete reason=abandoned"),
            "{shell}: {output}"
        );
    }
    Ok(())
}

/// A `PATH` with every command the watchdog needs except `curl`: real copies of
/// `sh`, `dash`, `busybox`, and the plain utilities the script calls, resolved
/// through the current `PATH` (so it works whether they are external programs or,
/// for busybox, an internal applet), symlinked into a fresh directory. `curl`
/// itself is never linked in, so `command -v curl` genuinely fails.
fn path_without_curl() -> Result<(tempfile::TempDir, String), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    for name in [
        "sh", "dash", "busybox", "date", "mkdir", "cat", "mv", "sleep", "kill", "printf", "env",
    ] {
        let Ok(output) = StdCommand::new("sh")
            .arg("-c")
            .arg(format!("command -v {name}"))
            .output()
        else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        let found = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !found.is_empty() {
            let _ = std::os::unix::fs::symlink(&found, dir.path().join(name));
        }
    }
    let path = dir.path().to_string_lossy().into_owned();
    Ok((dir, path))
}

#[tokio::test(flavor = "multi_thread")]
async fn a_pod_with_no_curl_on_path_is_a_failed_verdict() -> TestResult {
    for &shell in shells() {
        let (_dir, path) = path_without_curl()?;
        let server = stub(200, 204).await;
        let pod = Pod::new()?;
        let child = pod.start(
            shell,
            &server,
            &[("OVERBRAINER_KEEP_POD", "1".into()), ("PATH", path)],
        )?;
        assert!(
            until(Duration::from_secs(10), || {
                pod.read(".pod/watchdog") == "failed no_curl\n"
            })
            .await,
            "{shell}: {:?}",
            pod.read(".pod/watchdog")
        );
        drop(child);
        assert!(requests(&server).await.is_empty(), "{shell}");
    }
    Ok(())
}

/// Runs `bootstrap_functions()` then `bootstrap_main`, under `shell`, with a
/// `write_watchdog` that copies the real watchdog content (from
/// `OVERBRAINER_TEST_WATCHDOG_SRC`) to its target instead of embedding a
/// here-document, so a test can drive the real bootstrap failure path without
/// touching the pod's actual `/etc` or `/root`. Like the real one built by
/// `pod_command()`, it copies to a sibling `.tmp` file and only `mv -f`s it into
/// place once the copy itself succeeded.
fn start_bootstrap(shell: Shell, env: &[(&str, String)]) -> std::io::Result<Child> {
    let mut command = Command::new(shell.0[0]);
    command
        .args(&shell.0[1..])
        .arg("-c")
        .arg(format!(
            "{}\nwrite_watchdog() {{\n  mkdir -p \"$(dirname \"$1\")\" || return 1\n  cp \"$OVERBRAINER_TEST_WATCHDOG_SRC\" \"$1.tmp\" || return 1\n  mv -f \"$1.tmp\" \"$1\"\n}}\nbootstrap_main\n",
            bootstrap_functions()
        ))
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    for (name, value) in env {
        command.env(name, value);
    }
    command.spawn()
}

/// Everything a `start_bootstrap` run of `bootstrap_main` needs, in one temporary
/// tree: the run's SSH keys, a stub-reachable API, and every path `bootstrap_main`
/// would otherwise hard-code redirected under `root`.
struct FailingBootstrap {
    // Held only for its `Drop`: removes the temporary tree once the test is done.
    _root: tempfile::TempDir,
    run_dir: PathBuf,
    env: Vec<(&'static str, String)>,
}

impl FailingBootstrap {
    fn new(server: &MockServer) -> Result<Self, Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let run_dir = root.path().join("run");
        let watchdog_file = root.path().join("watchdog.sh");
        let watchdog_src = root.path().join("watchdog_src.sh");
        fs::write(&watchdog_src, watchdog_script())?;
        let etc_ssh = root.path().join("etc/ssh");
        // A directory whose parent cannot be written into, so
        // install_authorized_key's own `mkdir -p` fails: the target is never
        // created, regardless of what permissions it would otherwise get.
        let readonly_root = root.path().join("readonly");
        fs::create_dir_all(&readonly_root)?;
        fs::set_permissions(&readonly_root, fs::Permissions::from_mode(0o500))?;
        let authorized_keys_dir = readonly_root.join("ssh");
        let keys = PodKeys::generate(&root.path().join("ssh-keys"), "overbrainer-r1")?;
        let env = vec![
            ("OVERBRAINER_RUN_ID", "r1".to_string()),
            (
                "OVERBRAINER_RUN_DIR",
                run_dir.to_string_lossy().into_owned(),
            ),
            (
                "OVERBRAINER_WORKDIR",
                root.path().join("workdir").to_string_lossy().into_owned(),
            ),
            (
                "OVERBRAINER_WATCHDOG_FILE",
                watchdog_file.to_string_lossy().into_owned(),
            ),
            (
                "OVERBRAINER_TEST_WATCHDOG_SRC",
                watchdog_src.to_string_lossy().into_owned(),
            ),
            (
                "OVERBRAINER_ETC_SSH_DIR",
                etc_ssh.to_string_lossy().into_owned(),
            ),
            (
                "OVERBRAINER_AUTHORIZED_KEYS_DIR",
                authorized_keys_dir.to_string_lossy().into_owned(),
            ),
            (
                "OVERBRAINER_HOST_KEY",
                keys.host_private().expose_secret().to_string(),
            ),
            ("OVERBRAINER_AUTHORIZED_KEY", keys.client_public.clone()),
            ("RUNPOD_API_KEY", KEY.to_string()),
            ("RUNPOD_POD_ID", POD.to_string()),
            ("OVERBRAINER_API_URL", format!("{}/v2", server.uri())),
            ("OVERBRAINER_INTERVAL", "1".to_string()),
            ("OVERBRAINER_PROBE_WAIT", "0".to_string()),
            ("OVERBRAINER_VERSION", "9.9.9".to_string()),
        ];
        Ok(Self {
            _root: root,
            run_dir,
            env,
        })
    }

    fn read(&self, name: &str) -> String {
        fs::read_to_string(self.run_dir.join(name)).unwrap_or_default()
    }
}

/// Whether this process's effective user is root, which can write through the
/// read-only directory `FailingBootstrap` relies on to make `install_authorized_key`
/// fail: the two bootstrap-failure tests below need to skip in that case, the same
/// way other tests skip for a missing tool.
fn running_as_root() -> bool {
    let root = StdCommand::new("id")
        .arg("-u")
        .output()
        .is_ok_and(|output| output.status.success() && output.stdout.trim_ascii() == b"0");
    if root {
        eprintln!("skipped: running as root, permissions cannot force a write to fail");
    }
    root
}

#[tokio::test(flavor = "multi_thread")]
async fn a_bootstrap_failure_still_deletes_a_guarded_pod() -> TestResult {
    if !curl_available() || !keygen_available() || running_as_root() {
        return Ok(());
    }
    for &shell in shells() {
        let server = stub(200, 204).await;
        let setup = FailingBootstrap::new(&server)?;
        let child = start_bootstrap(shell, &setup.env)?;
        let (code, output) = finished(child, Duration::from_secs(15)).await?;
        assert_eq!(code, 0, "{shell}: {output}");
        assert_eq!(
            setup.read(".pod/bootstrap_failed"),
            "cannot install the authorized key\n",
            "{shell}: {output}"
        );
        assert_eq!(
            setup.read(".pod/watchdog"),
            "failed bootstrap: cannot install the authorized key\n",
            "{shell}: {output}"
        );
        assert!(
            output.contains("delete reason=bootstrap_failed"),
            "{shell}: {output}"
        );
        assert_eq!(deletes(&server).await, 1, "{shell}: {output}");
        assert!(!output.contains(KEY), "{shell}: {output}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn a_bootstrap_failure_in_keep_mode_only_logs() -> TestResult {
    if !curl_available() || !keygen_available() || running_as_root() {
        return Ok(());
    }
    for &shell in shells() {
        let server = stub(200, 204).await;
        let setup = FailingBootstrap::new(&server)?;
        let mut env = setup.env.clone();
        env.push(("OVERBRAINER_KEEP_POD", "1".to_string()));
        let child = start_bootstrap(shell, &env)?;
        assert!(
            until(Duration::from_secs(10), || {
                setup.read(".pod/watchdog")
                    == "failed bootstrap: cannot install the authorized key\n"
            })
            .await,
            "{shell}: {:?}",
            setup.read(".pod/watchdog")
        );
        tokio::time::sleep(Duration::from_secs(2)).await;
        assert_eq!(
            deletes(&server).await,
            0,
            "{shell}: a kept, failed pod was deleted"
        );
        drop(child);
        let log = setup.read(".pod/watchdog.log");
        assert!(log.contains("bootstrap failed, kept"), "{shell}: {log}");
    }
    Ok(())
}

/// Sources the bootstrap's functions, then runs `script`, under `shell`.
fn bootstrap(
    shell: Shell,
    script: &str,
    env: &[(&str, &str)],
) -> std::io::Result<std::process::Output> {
    StdCommand::new(shell.0[0])
        .args(&shell.0[1..])
        .arg("-c")
        .arg(format!("{}\n{script}", bootstrap_functions()))
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .envs(env.iter().copied())
        .output()
}

fn keygen_available() -> bool {
    StdCommand::new("ssh-keygen")
        .arg("-?")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

#[test]
fn the_bootstrap_installs_the_run_keys() -> TestResult {
    if !keygen_available() {
        eprintln!("skipped: ssh-keygen is not installed");
        return Ok(());
    }
    let keys_dir = tempfile::tempdir()?;
    let keys = PodKeys::generate(&keys_dir.path().join("ssh"), "overbrainer-r1")?;
    for &shell in shells() {
        let root = tempfile::tempdir()?;
        let etc = root.path().join("etc/ssh");
        fs::create_dir_all(&etc)?;
        fs::write(etc.join("ssh_host_rsa_key"), "baked")?;
        let home = root.path().join("root/.ssh");
        let output = bootstrap(
            shell,
            "install_host_key \"$ETC\" && install_authorized_key \"$HOME_SSH\"",
            &[
                ("OVERBRAINER_HOST_KEY", keys.host_private().expose_secret()),
                ("OVERBRAINER_AUTHORIZED_KEY", &keys.client_public),
                ("ETC", &etc.to_string_lossy()),
                ("HOME_SSH", &home.to_string_lossy()),
            ],
        )?;
        assert!(
            output.status.success(),
            "{shell}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!etc.join("ssh_host_rsa_key").exists(), "{shell}");
        let key = etc.join("ssh_host_ed25519_key");
        assert_eq!(
            fs::metadata(&key)?.permissions().mode() & 0o777,
            0o600,
            "{shell}"
        );
        let public = fs::read_to_string(etc.join("ssh_host_ed25519_key.pub"))?;
        assert!(public.starts_with(&keys.host_public), "{shell}: {public}");
        assert_eq!(
            fs::read_to_string(home.join("authorized_keys"))?,
            format!("{}\n", keys.client_public),
            "{shell}"
        );
        assert_eq!(
            fs::metadata(&home)?.permissions().mode() & 0o777,
            0o700,
            "{shell}"
        );
    }
    Ok(())
}

#[test]
fn the_job_env_survives_quotes_and_spaces() -> TestResult {
    for &shell in shells() {
        let root = tempfile::tempdir()?;
        let cuda = root.path().join("cuda13_env.sh");
        fs::write(
            &cuda,
            "export LD_LIBRARY_PATH=\"/usr/local/cuda-13/lib64:$UNSET_BY_DESIGN\"\n",
        )?;
        let job_env = root.path().join("etc/overbrainer/job.env");
        let output = bootstrap(
            shell,
            "set -u; write_job_env \"$JOB_ENV\" \"$CUDA\" && . \"$JOB_ENV\" && printf '%s|%s|%s' \"$PATH\" \"$LD_LIBRARY_PATH\" \"$HF_HOME\"",
            &[
                ("JOB_ENV", &job_env.to_string_lossy()),
                ("CUDA", &cuda.to_string_lossy()),
                ("OVERBRAINER_WORKDIR", "/workspace/it's here"),
                ("RUNPOD_API_KEY", KEY),
            ],
        )?;
        assert!(
            output.status.success(),
            "{shell}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let printed = String::from_utf8(output.stdout)?;
        let path = std::env::var("PATH").unwrap_or_default();
        assert_eq!(
            printed,
            format!("{path}|/usr/local/cuda-13/lib64:|/workspace/it's here/.hf-cache"),
            "{shell}"
        );
        let text = fs::read_to_string(&job_env)?;
        assert!(
            !text.contains(KEY) && !text.contains("RUNPOD"),
            "{shell}: {text}"
        );
        assert!(!Path::new(&format!("{}.tmp", job_env.display())).exists());
    }
    Ok(())
}

#[test]
fn the_base64_of_a_key_is_what_base64_decodes() -> TestResult {
    let output = StdCommand::new("sh")
        .arg("-c")
        .arg("printf '%s' \"$K\" | base64 -d")
        .env("K", base64(b"\x00binary\xffkey\n"))
        .output()?;
    assert_eq!(output.stdout, b"\x00binary\xffkey\n");
    Ok(())
}
