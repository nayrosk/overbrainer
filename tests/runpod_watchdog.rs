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
        let child = pod.start(
            shell,
            &server,
            &[("OVERBRAINER_RETRIEVE_GRACE", "2".into())],
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
        assert_clean(&server, &pod, &output, shell).await;
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
