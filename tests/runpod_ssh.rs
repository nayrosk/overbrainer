//! A Runpod pod reached over SSH, with a real sshd standing in for the pod and a
//! local stub for the Runpod API: the per-run ssh config and its `HostKeyAlias`
//! pinning, readiness and the watchdog's verdict, and ending a pod. Runs only when
//! `OVERBRAINER_TEST_SSH_HOST` and `OVERBRAINER_TEST_SSH_CONFIG` are set, as in the
//! `ssh` CI job, whose config file supplies the endpoint, the client key and the
//! server's host key; skipped otherwise.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use std::sync::Mutex;

use overbrainer::events::EventBus;
use overbrainer::exec::{Executor, JobCommand, JobRuntime, JobStatus, SshExecutor};
use overbrainer::retry::RetryPolicy;
use overbrainer::runpod::{
    AttemptResult, Ending, PodCtx, PodError, PodId, PodKeys, PodPlan, PodRecord, PodState,
    RunpodClient, RunpodTarget, SshEndpoint, Timing, alias, end_pod, job_started, provision,
    reconnect, write_config,
};
use overbrainer::runs::{Launch, RunCtx, RunRecord, RunState, Runs, start};
use overbrainer::train::{Artifacts, TrainError, Trainer};
use secrecy::SecretString;
use serde_json::{Value, json};
use tokio::sync::{Semaphore, SemaphorePermit};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Tests using the test sshd at once. Its `MaxStartups` (10 by default) drops
/// handshakes beyond that many pending at once, and a test opens up to three
/// connections at the same time.
static SSHD_SLOTS: Semaphore = Semaphore::const_new(3);

/// What the CI ssh config says about the test sshd, and this test's slot on it.
struct Sshd {
    endpoint: SshEndpoint,
    identity: PathBuf,
    host_public: String,
    _slot: SemaphorePermit<'static>,
}

/// The value of `key` in the ssh config `text`.
fn value(text: &str, key: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let (name, value) = line.trim().split_once(char::is_whitespace)?;
        name.eq_ignore_ascii_case(key)
            .then(|| value.trim().trim_matches('"').to_string())
    })
}

async fn sshd() -> Result<Option<Sshd>, Box<dyn std::error::Error>> {
    if std::env::var_os("OVERBRAINER_TEST_SSH_HOST").is_none() {
        return Ok(None);
    }
    let Some(config) = std::env::var_os("OVERBRAINER_TEST_SSH_CONFIG") else {
        return Ok(None);
    };
    let text = fs::read_to_string(config)?;
    let get = |key: &str| value(&text, key).ok_or(format!("no {key} in the test ssh config"));
    let known_hosts = fs::read_to_string(get("UserKnownHostsFile")?)?;
    let mut host_public = None;
    for line in known_hosts.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.get(1) == Some(&"ssh-ed25519") {
            let pair = fields
                .get(1..3)
                .ok_or("a known_hosts line names ssh-ed25519 but has no key")?;
            host_public = Some(pair.join(" "));
            break;
        }
    }
    let host_public = host_public.ok_or("no ssh-ed25519 key in the test known_hosts")?;
    let slot = SSHD_SLOTS.acquire().await?;
    Ok(Some(Sshd {
        endpoint: SshEndpoint {
            host: get("HostName")?,
            port: get("Port")?.parse()?,
            user: get("User")?,
        },
        identity: PathBuf::from(get("IdentityFile")?),
        host_public,
        _slot: slot,
    }))
}

fn skip() {
    eprintln!("skipped: OVERBRAINER_TEST_SSH_HOST and OVERBRAINER_TEST_SSH_CONFIG are not set");
}

fn keys(sshd: &Sshd, host_public: &str) -> PodKeys {
    PodKeys::new(
        sshd.identity.clone(),
        String::new(),
        host_public.to_string(),
        SecretString::from("unused"),
    )
}

fn new_run_id() -> String {
    format!("20260922-000000-{:04x}", fastrand::u16(..))
}

async fn connect(
    dir: &Path,
    sshd: &Sshd,
    keys: &PodKeys,
) -> Result<SshExecutor, Box<dyn std::error::Error>> {
    let run_id = new_run_id();
    let alias = alias(&run_id);
    let config = write_config(dir, &alias, &sshd.endpoint, keys)?;
    let workdir = format!("overbrainer-tests/runpod-{run_id}");
    Ok(SshExecutor::connect(&alias, &workdir, Some(&config)).await?)
}

/// Whether `error`'s chain names a host key check failure, rather than some
/// other reason the connection could have failed.
fn is_host_key_failure(error: &(dyn std::error::Error + 'static)) -> bool {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(e) = current {
        let text = e.to_string().to_lowercase();
        if text.contains("host key verification failed")
            || text.contains("remote host identification has changed")
        {
            return true;
        }
        current = e.source();
    }
    false
}

#[tokio::test]
async fn the_per_run_config_reaches_the_pinned_host() -> TestResult {
    let Some(sshd) = sshd().await? else {
        skip();
        return Ok(());
    };
    let dir = tempfile::tempdir()?;
    let executor = connect(dir.path(), &sshd, &keys(&sshd, &sshd.host_public)).await?;
    assert!(executor.workdir().contains("/overbrainer-tests/runpod-"));
    Ok(())
}

#[tokio::test]
async fn another_host_key_is_refused_through_the_per_run_config() -> TestResult {
    let Some(sshd) = sshd().await? else {
        skip();
        return Ok(());
    };
    let dir = tempfile::tempdir()?;
    let other = PodKeys::generate(&dir.path().join("other"), "other")?;
    let result = connect(dir.path(), &sshd, &keys(&sshd, &other.host_public)).await;
    let Err(error) = result else {
        return Err("connected although the pinned key differs".into());
    };
    assert!(
        is_host_key_failure(error.as_ref()),
        "expected a host key verification failure, got: {error}"
    );
    Ok(())
}

/// `GET /pods/p1`: the pod with the test sshd as its endpoint, until deleted.
/// After `dies_after` looks, the pod shows as EXITED.
struct Get {
    deleted: Arc<AtomicBool>,
    body: Value,
    looks: AtomicUsize,
    dies_after: Option<usize>,
}

impl Respond for Get {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        let looks = self.looks.fetch_add(1, Ordering::SeqCst);
        if self.dies_after.is_some_and(|after| looks >= after)
            && !self.deleted.load(Ordering::SeqCst)
        {
            let mut dead = self.body.clone();
            dead["status"] = json!("EXITED");
            return ResponseTemplate::new(200).set_body_json(dead);
        }
        if self.deleted.load(Ordering::SeqCst) {
            ResponseTemplate::new(404).set_body_json(json!({
                "detail": "pod not found",
                "status": 404,
                "title": "Not Found"
            }))
        } else {
            ResponseTemplate::new(200).set_body_json(&self.body)
        }
    }
}

struct Delete(Arc<AtomicBool>);

impl Respond for Delete {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        self.0.store(true, Ordering::SeqCst);
        ResponseTemplate::new(204)
    }
}

/// A stub whose pod `p1` is the test sshd, dying after `dies_after` looks.
async fn stub(sshd: &Sshd, run_id: &str, dies_after: Option<usize>) -> MockServer {
    let server = MockServer::start().await;
    let body = json!({
        "id": "p1",
        "name": format!("overbrainer-{run_id}-1"),
        "status": "RUNNING",
        "cost": 0.25,
        "env": {"OVERBRAINER_RUN_ID": run_id},
        "ssh": {"direct": {
            "host": sshd.endpoint.host,
            "port": sshd.endpoint.port,
            "username": sshd.endpoint.user
        }}
    });
    let deleted = Arc::new(AtomicBool::new(false));
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(201).set_body_json(&body))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v2/pods/p1"))
        .respond_with(Get {
            deleted: Arc::clone(&deleted),
            body,
            looks: AtomicUsize::new(0),
            dies_after,
        })
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v2/pods/p1"))
        .respond_with(Delete(deleted))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v2/pods"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"pods": []})))
        .mount(&server)
        .await;
    server
}

/// Writes the watchdog's verdict for `run_id` under `workdir` on the test sshd.
async fn write_verdict(
    sshd: &Sshd,
    workdir: &str,
    run_id: &str,
    verdict: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let keys = keys(sshd, &sshd.host_public);
    let alias = alias(run_id);
    let config = write_config(dir.path(), &alias, &sshd.endpoint, &keys)?;
    let executor = SshExecutor::connect(&alias, workdir, Some(&config)).await?;
    let job = executor
        .spawn(&JobCommand {
            dir: format!("{}/{run_id}", executor.workdir()),
            script: format!(
                "mkdir -p .pod && echo '{verdict}' > .pod/watchdog && echo 'probe {verdict}' > .pod/watchdog.log"
            ),
            secrets: Vec::new(),
            container: None,
        })
        .await?;
    for _ in 0..100 {
        if executor.status(&job).await? == JobStatus::Exited(0) {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Err("the verdict was not written".into())
}

/// Provisions against the test sshd with `verdict` written (none when `None`),
/// the pod dying after `dies_after` looks at it.
async fn provision_against(
    sshd: &Sshd,
    verdict: Option<&str>,
    dies_after: Option<usize>,
) -> Result<(Result<(), PodError>, PodRecord, usize), Box<dyn std::error::Error>> {
    let run_id = new_run_id();
    let workdir = format!("overbrainer-tests/runpod-{run_id}");
    if let Some(verdict) = verdict {
        write_verdict(sshd, &workdir, &run_id, verdict).await?;
    }
    let server = stub(sshd, &run_id, dies_after).await;
    let project = tempfile::tempdir()?;
    let runs = Runs::new(project.path());
    let client = client(&server)?;
    let timing = Timing {
        ready_timeout: Duration::from_secs(30),
        preflight_timeout: Duration::from_secs(30),
        ..fast()
    };
    let interrupted = AtomicBool::new(false);
    let bus = EventBus::new();
    let ctx = PodCtx {
        client: &client,
        runs: &runs,
        bus: &bus,
        timing: &timing,
        interrupted: &interrupted,
    };
    let target = RunpodTarget {
        gpu_types: vec!["NVIDIA A40".into()],
        gpu_count: 1,
        image: "img".into(),
        venv: "/venv".into(),
        container_disk_gb: 50,
        max_hours: 1.0,
        boot_grace: Duration::from_secs(1800),
        retrieve_grace: Duration::from_secs(3600),
        data_center_ids: Vec::new(),
        network_volume_id: None,
    };
    let keys = keys(sshd, &sshd.host_public);
    let ssh_dir = runs.run_dir(&run_id)?.join("ssh");
    let plan = PodPlan {
        run_id: &run_id,
        target: &target,
        keys: &keys,
        ssh_dir: &ssh_dir,
        workdir: &workdir,
        api_url: client.base_url(),
    };
    let mut record = PodRecord::new(&run_id, false, 1, &keys.host_public);
    let result = provision(&ctx, &plan, &mut record).await;
    let deletes = server
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .filter(|request| request.method.as_str() == "DELETE")
        .count();
    if let Ok(provisioned) = &result {
        assert_eq!(provisioned.pod_id.as_str(), "p1");
        assert!(provisioned.executor.workdir().ends_with(&workdir));
    }
    Ok((result.map(drop), record, deletes))
}

#[tokio::test]
async fn a_pod_is_ready_once_ssh_and_its_watchdog_answer() -> TestResult {
    let Some(sshd) = sshd().await? else {
        skip();
        return Ok(());
    };
    let (result, record, deletes) = provision_against(&sshd, Some("ready"), None).await?;
    result?;
    assert_eq!(record.state, PodState::Ready);
    assert_eq!(
        record.ssh.as_ref().map(|ssh| ssh.port),
        Some(sshd.endpoint.port)
    );
    assert!(record.ready_at.is_some());
    assert_eq!(deletes, 0);
    Ok(())
}

#[tokio::test]
async fn a_watchdog_that_cannot_delete_its_pod_refuses_the_run() -> TestResult {
    let Some(sshd) = sshd().await? else {
        skip();
        return Ok(());
    };
    let (result, record, deletes) = provision_against(&sshd, Some("failed http_403"), None).await?;
    let error = result.err().ok_or("the run was not refused")?;
    assert!(
        error
            .to_string()
            .starts_with("the pod's watchdog cannot remove its own pod (http_403)"),
        "{error}"
    );
    assert_eq!(record.state, PodState::Deleted);
    assert_eq!(deletes, 1);
    Ok(())
}

#[tokio::test]
async fn a_failed_bootstrap_deletes_the_pod_and_refuses_the_run() -> TestResult {
    let Some(sshd) = sshd().await? else {
        skip();
        return Ok(());
    };
    let (result, record, deletes) = provision_against(
        &sshd,
        Some("failed bootstrap: cannot write the job environment"),
        None,
    )
    .await?;
    let Err(PodError::BootstrapFailed { pod_id, reason }) = result else {
        return Err(format!("the run was not refused: {result:?}").into());
    };
    assert_eq!(pod_id.as_str(), "p1");
    assert_eq!(reason, "cannot write the job environment");
    assert_eq!(record.state, PodState::Deleted);
    assert_eq!(deletes, 1);
    Ok(())
}

#[tokio::test]
async fn a_pod_that_dies_while_its_verdict_is_awaited_is_deleted_at_once() -> TestResult {
    let Some(sshd) = sshd().await? else {
        skip();
        return Ok(());
    };
    let started = std::time::Instant::now();
    // No verdict: SSH works on the first look, the pod is EXITED on the next.
    let (result, record, deletes) = provision_against(&sshd, None, Some(1)).await?;
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "waited {:?} for a dead pod",
        started.elapsed()
    );
    assert!(matches!(result, Err(PodError::NoCapacity(_))), "{result:?}");
    assert_eq!(record.attempts.len(), 1);
    assert_eq!(
        record.attempts[0].detail.as_deref(),
        Some("the pod is EXITED")
    );
    assert_eq!(record.pod_id, None);
    assert_eq!(deletes, 1);
    Ok(())
}

/// A client of the stub `server`, with millisecond retries.
fn client(server: &MockServer) -> Result<RunpodClient, Box<dyn std::error::Error>> {
    Ok(
        RunpodClient::new(&format!("{}/v2", server.uri()), &SecretString::from("k"))?.with_policy(
            RetryPolicy {
                max_retries: 1,
                base: Duration::from_millis(1),
                cap: Duration::from_millis(2),
            },
        ),
    )
}

fn fast() -> Timing {
    Timing {
        poll: Duration::from_millis(50),
        ready_timeout: Duration::from_secs(10),
        preflight_timeout: Duration::from_secs(5),
        reconcile_waits: [Duration::from_millis(5), Duration::from_millis(5)],
        delete_timeout: Duration::from_secs(5),
        gone_interval: Duration::from_millis(5),
    }
}

/// A run whose directory exists on the test sshd, its pod record for pod `p1`,
/// and a connection to it.
async fn started_run(
    sshd: &Sshd,
    runs: &Runs,
    keep: bool,
) -> Result<(RunRecord, PodRecord, SshExecutor), Box<dyn std::error::Error>> {
    let run_id = new_run_id();
    let workdir = format!("overbrainer-tests/runpod-{run_id}");
    write_verdict(sshd, &workdir, &run_id, "ready").await?;
    let dir = tempfile::tempdir()?;
    let alias = alias(&run_id);
    let config = write_config(
        dir.path(),
        &alias,
        &sshd.endpoint,
        &keys(sshd, &sshd.host_public),
    )?;
    let executor = SshExecutor::connect(&alias, &workdir, Some(&config)).await?;
    let run = RunRecord {
        id: run_id.clone(),
        target: "gpu_cloud".into(),
        created: "2026-09-22T00:00:00Z".into(),
        remote_dir: format!("{}/{run_id}", executor.workdir()),
        job: None,
        state: RunState::Succeeded,
        message: None,
    };
    runs.save(&run)?;
    let mut pod = PodRecord::new(&run_id, keep, 1, &sshd.host_public);
    let remote = serde_json::from_value(json!({"id": "p1", "status": "RUNNING", "cost": 0.25}))?;
    pod.begin_attempt("NVIDIA A40", std::time::SystemTime::now(), 1.0);
    pod.created(
        &remote,
        AttemptResult::Created,
        std::time::SystemTime::now(),
    );
    pod.state = PodState::Running;
    pod.save(runs)?;
    Ok((run, pod, executor))
}

async fn marker_exists(
    executor: &SshExecutor,
    run: &RunRecord,
) -> Result<bool, Box<dyn std::error::Error>> {
    let manifest = executor
        .manifest(&run.remote_dir, &[".pod".to_string()], &[])
        .await?;
    Ok(manifest.iter().any(|file| file.path == ".pod/retrieved"))
}

#[tokio::test]
async fn a_retrieved_run_marks_its_pod_then_deletes_it_unless_kept() -> TestResult {
    let Some(sshd) = sshd().await? else {
        skip();
        return Ok(());
    };
    for keep in [false, true] {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let (run, mut pod, executor) = started_run(&sshd, &runs, keep).await?;
        let key = runs.run_dir(&run.id)?.join("ssh/id_ed25519");
        fs::create_dir_all(key.parent().ok_or("no ssh dir")?)?;
        fs::write(&key, "private")?;
        let server = stub(&sshd, &run.id, None).await;
        let client = client(&server)?;
        let (bus, timing, interrupted) = (EventBus::new(), fast(), AtomicBool::new(false));
        let ctx = PodCtx {
            client: &client,
            runs: &runs,
            bus: &bus,
            timing: &timing,
            interrupted: &interrupted,
        };
        let ending = end_pod(&ctx, &mut pod, &executor, &run, true).await?;
        assert!(marker_exists(&executor, &run).await?, "keep = {keep}");
        let log = runs.run_dir(&run.id)?.join(".pod/watchdog.log");
        assert_eq!(fs::read_to_string(log)?, "probe ready\n");
        let deletes = server
            .received_requests()
            .await
            .unwrap_or_default()
            .iter()
            .filter(|request| request.method.as_str() == "DELETE")
            .count();
        if keep {
            assert_eq!(ending, Ending::Kept);
            assert_eq!(pod.state, PodState::Kept);
            assert_eq!(deletes, 0);
            assert!(key.is_file(), "the key of a kept pod is removed");
        } else {
            assert_eq!(ending, Ending::Deleted);
            assert_eq!(pod.state, PodState::Deleted);
            assert_eq!(deletes, 1);
            assert!(!key.exists(), "the key of a deleted pod is kept");
        }
        assert_eq!(PodRecord::load(&runs, &run.id)?, Some(pod));
    }
    Ok(())
}

/// `DELETE /pods/p1` that first looks, over its own ssh connection, whether the
/// retrieved marker is already on the test sshd.
struct DeleteAfterMarker {
    deleted: Arc<AtomicBool>,
    marked: Arc<AtomicBool>,
    config: PathBuf,
    alias: String,
    marker: String,
}

impl Respond for DeleteAfterMarker {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        let seen = std::process::Command::new("ssh")
            .arg("-F")
            .arg(&self.config)
            .arg(&self.alias)
            .arg(format!("test -f '{}'", self.marker))
            .status()
            .is_ok_and(|status| status.success());
        self.marked.store(seen, Ordering::SeqCst);
        self.deleted.store(true, Ordering::SeqCst);
        ResponseTemplate::new(204)
    }
}

#[tokio::test]
async fn the_retrieved_marker_is_on_the_pod_before_its_delete() -> TestResult {
    let Some(sshd) = sshd().await? else {
        skip();
        return Ok(());
    };
    let project = tempfile::tempdir()?;
    let runs = Runs::new(project.path());
    let (run, mut pod, executor) = started_run(&sshd, &runs, false).await?;
    let dir = tempfile::tempdir()?;
    let probe = format!("{}-probe", alias(&run.id));
    let config = write_config(
        dir.path(),
        &probe,
        &sshd.endpoint,
        &keys(&sshd, &sshd.host_public),
    )?;
    let server = MockServer::start().await;
    let deleted = Arc::new(AtomicBool::new(false));
    let marked = Arc::new(AtomicBool::new(false));
    Mock::given(method("GET"))
        .and(path("/v2/pods/p1"))
        .respond_with(Get {
            deleted: Arc::clone(&deleted),
            body: json!({"id": "p1", "status": "RUNNING", "cost": 0.25}),
            looks: AtomicUsize::new(0),
            dies_after: None,
        })
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v2/pods/p1"))
        .respond_with(DeleteAfterMarker {
            deleted,
            marked: Arc::clone(&marked),
            config,
            alias: probe,
            marker: format!("{}/.pod/retrieved", run.remote_dir),
        })
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v2/pods"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"pods": []})))
        .mount(&server)
        .await;
    let client = client(&server)?;
    let (bus, timing, interrupted) = (EventBus::new(), fast(), AtomicBool::new(false));
    let ctx = PodCtx {
        client: &client,
        runs: &runs,
        bus: &bus,
        timing: &timing,
        interrupted: &interrupted,
    };
    let ending = end_pod(&ctx, &mut pod, &executor, &run, true).await?;
    assert_eq!(ending, Ending::Deleted);
    assert!(
        marked.load(Ordering::SeqCst),
        "the pod was deleted before its retrieved marker was written"
    );
    Ok(())
}

#[tokio::test]
async fn a_duplicate_pod_whose_delete_fails_is_recorded_as_stray() -> TestResult {
    let Some(sshd) = sshd().await? else {
        skip();
        return Ok(());
    };
    let project = tempfile::tempdir()?;
    let runs = Runs::new(project.path());
    let (run, mut pod, executor) = started_run(&sshd, &runs, false).await?;
    let server = MockServer::start().await;
    let deleted = Arc::new(AtomicBool::new(false));
    Mock::given(method("GET"))
        .and(path("/v2/pods/p1"))
        .respond_with(Get {
            deleted: Arc::clone(&deleted),
            body: json!({"id": "p1", "status": "RUNNING", "cost": 0.25}),
            looks: AtomicUsize::new(0),
            dies_after: None,
        })
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v2/pods/p1"))
        .respond_with(Delete(deleted))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v2/pods"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"pods": [{
            "id": "dup1",
            "status": "RUNNING",
            "env": {"OVERBRAINER_RUN_ID": run.id}
        }]})))
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v2/pods/dup1"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    let client = client(&server)?;
    let (bus, timing, interrupted) = (EventBus::new(), fast(), AtomicBool::new(false));
    let ctx = PodCtx {
        client: &client,
        runs: &runs,
        bus: &bus,
        timing: &timing,
        interrupted: &interrupted,
    };
    let ending = end_pod(&ctx, &mut pod, &executor, &run, true).await?;
    assert_eq!(ending, Ending::Deleted);
    let saved = PodRecord::load(&runs, &run.id)?.ok_or("no pod.json")?;
    assert_eq!(saved.state, PodState::Deleted);
    let stray: Vec<&str> = saved.stray_pods.iter().map(PodId::as_str).collect();
    assert_eq!(stray, ["dup1"]);
    Ok(())
}

#[tokio::test]
async fn an_unretrieved_run_leaves_its_pod_to_the_watchdog() -> TestResult {
    let Some(sshd) = sshd().await? else {
        skip();
        return Ok(());
    };
    let project = tempfile::tempdir()?;
    let runs = Runs::new(project.path());
    let (run, mut pod, executor) = started_run(&sshd, &runs, false).await?;
    let server = stub(&sshd, &run.id, None).await;
    let client = client(&server)?;
    let (bus, timing, interrupted) = (EventBus::new(), fast(), AtomicBool::new(false));
    let ctx = PodCtx {
        client: &client,
        runs: &runs,
        bus: &bus,
        timing: &timing,
        interrupted: &interrupted,
    };
    let ending = end_pod(&ctx, &mut pod, &executor, &run, false).await?;
    assert_eq!(ending, Ending::AwaitingRetrieval);
    assert_eq!(pod.state, PodState::AwaitingRetrieval);
    assert!(!marker_exists(&executor, &run).await?);
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
async fn attach_reconnects_through_a_fresh_endpoint() -> TestResult {
    let Some(sshd) = sshd().await? else {
        skip();
        return Ok(());
    };
    let project = tempfile::tempdir()?;
    let runs = Runs::new(project.path());
    let (run, mut pod, _) = started_run(&sshd, &runs, false).await?;
    let ssh_dir = runs.run_dir(&run.id)?.join("ssh");
    fs::create_dir_all(&ssh_dir)?;
    fs::copy(&sshd.identity, ssh_dir.join("id_ed25519"))?;
    // The endpoint seen last is stale: the pod was reset and its port changed.
    let stale = SshEndpoint {
        port: sshd.endpoint.port.wrapping_add(1),
        ..sshd.endpoint.clone()
    };
    write_config(
        &ssh_dir,
        &alias(&run.id),
        &stale,
        &keys(&sshd, &sshd.host_public),
    )?;
    pod.ssh = Some(stale);
    pod.save(&runs)?;
    let server = stub(&sshd, &run.id, None).await;
    let client = client(&server)?;
    let (bus, timing, interrupted) = (EventBus::new(), fast(), AtomicBool::new(false));
    let ctx = PodCtx {
        client: &client,
        runs: &runs,
        bus: &bus,
        timing: &timing,
        interrupted: &interrupted,
    };
    let executor = reconnect(&ctx, &mut pod, &run).await?.ok_or("no pod")?;
    assert_eq!(format!("{}/{}", executor.workdir(), run.id), run.remote_dir);
    assert_eq!(pod.ssh.as_ref(), Some(&sshd.endpoint));
    assert_eq!(PodRecord::load(&runs, &run.id)?, Some(pod));
    let config = fs::read_to_string(ssh_dir.join("config"))?;
    assert!(
        config.contains(&format!("Port {}", sshd.endpoint.port)),
        "{config}"
    );
    Ok(())
}

/// A trainer with nothing to run, which notes what `run.json` and `pod.json`
/// say when `runs::start` prepares the run: before its job is spawned and the
/// run saved `Running`.
struct Observer {
    runs: Runs,
    run_id: String,
    seen: Mutex<Option<(RunState, Option<PodRecord>)>>,
}

impl Trainer for Observer {
    fn prepare(&self, _run_dir: &Path, _root: &str) -> Result<(), TrainError> {
        let run = self.runs.load(&self.run_id).map(|run| run.state);
        let pod = PodRecord::load(&self.runs, &self.run_id).ok().flatten();
        if let (Ok(run), Ok(mut seen)) = (run, self.seen.lock()) {
            *seen = Some((run, pod));
        }
        Ok(())
    }

    fn commands(&self) -> Vec<Vec<String>> {
        Vec::new()
    }

    fn env(&self, _root: &str) -> Vec<(String, String)> {
        Vec::new()
    }

    fn metrics_file(&self) -> &'static str {
        "metrics.jsonl"
    }

    fn artifacts(&self) -> Artifacts {
        Artifacts {
            entries: Vec::new(),
            exclude: Vec::new(),
            required: None,
        }
    }
}

/// The library order the CLI's `train` relies on: `provision` has saved the
/// pod's ID in `pod.json` when it returns, so when `runs::start` then prepares
/// the run (before it spawns the job and saves the run `Running`), `pod.json`
/// already names the pod. It does not drive the CLI itself: that one generates
/// a host key only a real pod's bootstrap installs.
#[tokio::test]
async fn provision_saves_the_pod_id_before_runs_start_saves_running() -> TestResult {
    let Some(sshd) = sshd().await? else {
        skip();
        return Ok(());
    };
    let run_id = new_run_id();
    let workdir = format!("overbrainer-tests/runpod-{run_id}");
    write_verdict(&sshd, &workdir, &run_id, "ready").await?;
    let server = stub(&sshd, &run_id, None).await;
    let project = tempfile::tempdir()?;
    let runs = Runs::new(project.path());
    let client = client(&server)?;
    let timing = Timing {
        ready_timeout: Duration::from_secs(30),
        preflight_timeout: Duration::from_secs(30),
        ..fast()
    };
    let (bus, interrupted) = (EventBus::new(), AtomicBool::new(false));
    let ctx = PodCtx {
        client: &client,
        runs: &runs,
        bus: &bus,
        timing: &timing,
        interrupted: &interrupted,
    };
    let target = RunpodTarget {
        gpu_types: vec!["NVIDIA A40".into()],
        gpu_count: 1,
        image: "img".into(),
        venv: "/venv".into(),
        container_disk_gb: 50,
        max_hours: 1.0,
        boot_grace: Duration::from_secs(1800),
        retrieve_grace: Duration::from_secs(3600),
        data_center_ids: Vec::new(),
        network_volume_id: None,
    };
    let keys = keys(&sshd, &sshd.host_public);
    let ssh_dir = runs.run_dir(&run_id)?.join("ssh");
    let plan = PodPlan {
        run_id: &run_id,
        target: &target,
        keys: &keys,
        ssh_dir: &ssh_dir,
        workdir: &workdir,
        api_url: client.base_url(),
    };
    let mut pod = PodRecord::new(&run_id, false, 1, &keys.host_public);
    let provisioned = provision(&ctx, &plan, &mut pod).await?;
    let record = RunRecord {
        id: run_id.clone(),
        target: "gpu_cloud".into(),
        created: "2026-09-22T00:00:00Z".into(),
        remote_dir: format!("{}/{run_id}", provisioned.executor.workdir()),
        job: None,
        state: RunState::Preparing,
        message: None,
    };
    runs.save(&record)?;
    let observer = Observer {
        runs: Runs::new(project.path()),
        run_id: run_id.clone(),
        seen: Mutex::new(None),
    };
    let runtime = JobRuntime::Native {
        venv: None,
        env_file: None,
    };
    let run_ctx = RunCtx {
        runs: &runs,
        executor: &provisioned.executor,
        bus: &bus,
        poll: Duration::from_millis(50),
    };
    let launch = Launch {
        runtime: &runtime,
        secrets: Vec::new(),
    };
    let started = start(&run_ctx, &observer, launch, record).await?;
    assert_eq!(started.state, RunState::Running);
    let seen = observer
        .seen
        .lock()
        .map_err(|_| "poisoned")?
        .take()
        .ok_or("start never prepared the run")?;
    let (run_state, saved) = seen;
    assert_eq!(run_state, RunState::Preparing);
    let saved = saved.ok_or("no pod.json when the job was started")?;
    assert_eq!(saved.pod_id.as_ref().map(PodId::as_str), Some("p1"));
    assert_eq!(saved.state, PodState::Ready);
    job_started(&runs, &mut pod)?;
    let saved = PodRecord::load(&runs, &run_id)?.ok_or("no pod.json")?;
    assert_eq!(saved.state, PodState::Running);
    assert_eq!(saved.pod_id.as_ref().map(PodId::as_str), Some("p1"));
    Ok(())
}
