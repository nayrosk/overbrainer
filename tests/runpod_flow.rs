//! The Runpod run steps against a local stub of the API, with the local executor
//! standing in for the pod: a failed or interrupted provisioning fails the run,
//! the client-side deadline deletes a pod still there, and a watch that loses a
//! deleted pod fails the run. Ending a pod over SSH is covered in
//! `tests/runpod_ssh.rs`.

use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

use overbrainer::events::EventBus;
use overbrainer::exec::{Executor, JobCommand, LocalExecutor};
use overbrainer::retry::RetryPolicy;
use overbrainer::runpod::{
    AttemptResult, DeletedBy, PodCtx, PodError, PodRecord, PodState, RunpodClient, RunpodTarget,
    Timing, Watched, follow, reconnect, settle_watch, start_pod, watch_on_pod,
};
use overbrainer::runs::{RunCtx, RunRecord, RunState, Runs, create};
use overbrainer::train::{Artifacts, TrainError, Trainer};
use secrecy::SecretString;
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Runpod's answer when a GPU type has no capacity left.
const CAPACITY: &str = "There are no longer any instances available with the requested specifications. Please refresh and try again.";

struct Get {
    deleted: Arc<AtomicBool>,
    body: Value,
}

impl Respond for Get {
    fn respond(&self, _: &Request) -> ResponseTemplate {
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

/// Serves pod `p1`, gone once deleted (or from the start with `gone`).
async fn serve_p1(server: &MockServer, gone: bool) {
    let deleted = Arc::new(AtomicBool::new(gone));
    Mock::given(method("GET"))
        .and(path("/v2/pods/p1"))
        .respond_with(Get {
            deleted: Arc::clone(&deleted),
            body: json!({"id": "p1", "status": "RUNNING", "cost": 0.5}),
        })
        .mount(server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v2/pods/p1"))
        .respond_with(Delete(deleted))
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v2/pods"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"pods": []})))
        .mount(server)
        .await;
}

struct Harness {
    server: MockServer,
    project: tempfile::TempDir,
    runs: Runs,
    bus: EventBus,
    timing: Timing,
    interrupted: AtomicBool,
    client: RunpodClient,
}

impl Harness {
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let client = RunpodClient::new(&format!("{}/v2", server.uri()), &SecretString::from("k"))?
            .with_policy(RetryPolicy {
                max_retries: 1,
                base: Duration::from_millis(1),
                cap: Duration::from_millis(2),
            });
        Ok(Self {
            server,
            project,
            runs,
            bus: EventBus::new(),
            timing: Timing {
                poll: Duration::from_millis(5),
                ready_timeout: Duration::from_millis(200),
                preflight_timeout: Duration::from_millis(200),
                reconcile_waits: [Duration::from_millis(5), Duration::from_millis(5)],
                delete_timeout: Duration::from_millis(300),
                gone_interval: Duration::from_millis(5),
            },
            interrupted: AtomicBool::new(false),
            client,
        })
    }

    fn ctx(&self) -> PodCtx<'_> {
        PodCtx {
            client: &self.client,
            runs: &self.runs,
            bus: &self.bus,
            timing: &self.timing,
            interrupted: &self.interrupted,
        }
    }
}

fn target() -> RunpodTarget {
    RunpodTarget {
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
    }
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

#[tokio::test]
async fn a_run_whose_pod_cannot_be_placed_is_failed() -> TestResult {
    if !keygen_available() {
        return Ok(());
    }
    let harness = Harness::new().await?;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({"detail": CAPACITY})))
        .mount(&harness.server)
        .await;
    let run = create(&harness.runs, "/workspace/overbrainer", "gpu_cloud")?;
    let result = start_pod(&harness.ctx(), &target(), run.clone(), false).await;
    assert!(matches!(result, Err(PodError::NoCapacity(_))), "{result:?}");
    let saved = harness.runs.load(&run.id)?;
    assert_eq!(saved.state, RunState::Failed);
    assert!(
        saved
            .message
            .as_deref()
            .is_some_and(|message| message.starts_with("no gpu_types entry could be placed")),
        "{:?}",
        saved.message
    );
    let pod = PodRecord::load(&harness.runs, &run.id)?.ok_or("no pod.json")?;
    assert_eq!(pod.attempts[0].result, AttemptResult::Unavailable);
    assert!(pod.host_key.starts_with("ssh-ed25519 "));
    // No pod was left: the private client key has nothing left to reach.
    let ssh = harness.runs.run_dir(&run.id)?.join("ssh");
    assert!(!ssh.join("id_ed25519").exists());
    assert!(!ssh.join("host_ed25519").exists());
    Ok(())
}

/// A pod whose deletion cannot be confirmed may still run: the client key
/// that reaches it stays.
#[tokio::test]
async fn a_failed_provisioning_keeps_the_client_key_while_a_pod_may_remain() -> TestResult {
    if !keygen_available() {
        return Ok(());
    }
    let harness = Harness::new().await?;
    Mock::given(method("POST"))
        .and(path("/v2/pods"))
        .respond_with(
            ResponseTemplate::new(201).set_body_json(json!({"id": "p1", "status": "RUNNING"})),
        )
        .mount(&harness.server)
        .await;
    // A dead pod that never goes away.
    Mock::given(method("GET"))
        .and(path("/v2/pods/p1"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"id": "p1", "status": "EXITED"})),
        )
        .mount(&harness.server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v2/pods/p1"))
        .respond_with(ResponseTemplate::new(204))
        .mount(&harness.server)
        .await;
    let run = create(&harness.runs, "/workspace/overbrainer", "gpu_cloud")?;
    let result = start_pod(&harness.ctx(), &target(), run.clone(), false).await;
    assert!(matches!(result, Err(PodError::NotDeleted(_))), "{result:?}");
    let pod = PodRecord::load(&harness.runs, &run.id)?.ok_or("no pod.json")?;
    assert_eq!(pod.state, PodState::Deleting);
    let ssh = harness.runs.run_dir(&run.id)?.join("ssh");
    assert!(ssh.join("id_ed25519").is_file());
    Ok(())
}

#[tokio::test]
async fn ctrl_c_before_the_pod_fails_the_run_as_interrupted() -> TestResult {
    if !keygen_available() {
        return Ok(());
    }
    let harness = Harness::new().await?;
    harness.interrupted.store(true, Ordering::SeqCst);
    let run = create(&harness.runs, "/workspace/overbrainer", "gpu_cloud")?;
    let result = start_pod(&harness.ctx(), &target(), run.clone(), false).await;
    assert!(matches!(result, Err(PodError::Interrupted)), "{result:?}");
    let saved = harness.runs.load(&run.id)?;
    assert_eq!(saved.state, RunState::Failed);
    assert_eq!(
        saved.message.as_deref(),
        Some("interrupted before the job started")
    );
    let key = harness.runs.run_dir(&run.id)?.join("ssh/id_ed25519");
    assert!(!key.exists());
    Ok(())
}

/// A trainer with nothing to prepare or retrieve.
struct Nothing;

impl Trainer for Nothing {
    fn prepare(&self, _run_dir: &Path, _root: &str) -> Result<(), TrainError> {
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

/// How many deletes the stub received.
async fn deletes(server: &MockServer) -> usize {
    server
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .filter(|request| request.method.as_str() == "DELETE")
        .count()
}

/// How many looks at pod `p1` the stub received.
async fn gets(server: &MockServer) -> usize {
    server
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .filter(|request| request.method.as_str() == "GET" && request.url.path() == "/v2/pods/p1")
        .count()
}

/// `GET /pods/p1` answering in turn from `present` (the last answer repeats):
/// the pod when true, Runpod's 404 when false.
struct Sequence {
    present: Vec<bool>,
    looks: AtomicUsize,
}

impl Respond for Sequence {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        let look = self.looks.fetch_add(1, Ordering::SeqCst);
        let present = self
            .present
            .get(look)
            .or(self.present.last())
            .copied()
            .unwrap_or(false);
        if present {
            ResponseTemplate::new(200)
                .set_body_json(json!({"id": "p1", "status": "RUNNING", "cost": 0.5}))
        } else {
            ResponseTemplate::new(404).set_body_json(json!({
                "detail": "pod not found",
                "status": 404,
                "title": "Not Found"
            }))
        }
    }
}

/// Serves pod `p1` answering from `present` (see [`Sequence`]), listed by
/// `GET /pods` when `listed`, and accepting deletes (which the tests count).
async fn serve_sequence(server: &MockServer, present: Vec<bool>, listed: bool) {
    Mock::given(method("GET"))
        .and(path("/v2/pods/p1"))
        .respond_with(Sequence {
            present,
            looks: AtomicUsize::new(0),
        })
        .mount(server)
        .await;
    Mock::given(method("DELETE"))
        .and(path("/v2/pods/p1"))
        .respond_with(ResponseTemplate::new(204))
        .mount(server)
        .await;
    let pods = if listed {
        json!({"pods": [{"id": "p1", "status": "RUNNING"}]})
    } else {
        json!({"pods": []})
    };
    Mock::given(method("GET"))
        .and(path("/v2/pods"))
        .respond_with(ResponseTemplate::new(200).set_body_json(pods))
        .mount(server)
        .await;
}

/// A pod record for pod `p1`, created now, whose deadline passed `ago` ago.
fn pod_record(
    run_id: &str,
    keep: bool,
    ago: Duration,
) -> Result<PodRecord, Box<dyn std::error::Error>> {
    let mut record = PodRecord::new(run_id, keep, 1, "ssh-ed25519 AAAAhost");
    let pod = serde_json::from_value(json!({"id": "p1", "status": "RUNNING", "cost": 0.5}))?;
    record.begin_attempt(
        "NVIDIA A40",
        SystemTime::now() - Duration::from_secs(3600) - ago,
        1.0,
    );
    record.created(&pod, AttemptResult::Created, SystemTime::now());
    Ok(record)
}

#[tokio::test]
async fn the_client_deletes_a_pod_past_its_deadline() -> TestResult {
    let harness = Harness::new().await?;
    serve_p1(&harness.server, false).await;
    let executor = LocalExecutor::new(&harness.project.path().join("pod"))?;
    let mut run = create(&harness.runs, executor.workdir(), "gpu_cloud")?;
    let job = executor
        .spawn(&JobCommand {
            dir: run.remote_dir.clone(),
            script: "sleep 30".into(),
            secrets: Vec::new(),
            container: None,
        })
        .await?;
    run.job = Some(job.clone());
    run.state = RunState::Running;
    harness.runs.save(&run)?;
    let mut pod = pod_record(&run.id, false, Duration::from_secs(600))?;
    let run_ctx = RunCtx {
        runs: &harness.runs,
        executor: &executor,
        bus: &harness.bus,
        poll: Duration::from_millis(50),
    };
    let result = follow(&harness.ctx(), &run_ctx, &Nothing, run.clone(), &mut pod).await;
    executor.cancel(&job).await?;
    assert!(
        matches!(result, Err(PodError::DeadlineReached)),
        "{result:?}"
    );
    assert_eq!(pod.state, PodState::Deleted);
    let saved = harness.runs.load(&run.id)?;
    assert_eq!(saved.state, RunState::Failed);
    assert_eq!(
        saved.message.as_deref(),
        Some("max_hours reached: the pod was deleted before the job ended")
    );
    assert_eq!(deletes(&harness.server).await, 1);
    Ok(())
}

/// The CLI races the watch against Ctrl-C, which may drop it: past the
/// deadline, the watch itself deletes nothing, and the delete comes from
/// `settle_watch`, which the CLI shields.
#[tokio::test]
async fn a_watch_past_its_deadline_deletes_nothing_until_settled() -> TestResult {
    let harness = Harness::new().await?;
    serve_p1(&harness.server, false).await;
    let executor = LocalExecutor::new(&harness.project.path().join("pod"))?;
    let mut run = create(&harness.runs, executor.workdir(), "gpu_cloud")?;
    let job = executor
        .spawn(&JobCommand {
            dir: run.remote_dir.clone(),
            script: "sleep 30".into(),
            secrets: Vec::new(),
            container: None,
        })
        .await?;
    run.job = Some(job.clone());
    run.state = RunState::Running;
    harness.runs.save(&run)?;
    let mut pod = pod_record(&run.id, false, Duration::from_secs(600))?;
    let run_ctx = RunCtx {
        runs: &harness.runs,
        executor: &executor,
        bus: &harness.bus,
        poll: Duration::from_millis(50),
    };
    let watched = watch_on_pod(&run_ctx, &Nothing, run.clone(), &pod).await;
    assert!(matches!(watched, Watched::DeadlinePassed));
    assert_eq!(deletes(&harness.server).await, 0);
    assert_eq!(harness.runs.load(&run.id)?.state, RunState::Running);
    let result = settle_watch(&harness.ctx(), &mut pod, &run.id, watched).await;
    executor.cancel(&job).await?;
    assert!(
        matches!(result, Err(PodError::DeadlineReached)),
        "{result:?}"
    );
    assert_eq!(pod.state, PodState::Deleted);
    assert_eq!(harness.runs.load(&run.id)?.state, RunState::Failed);
    assert_eq!(deletes(&harness.server).await, 1);
    Ok(())
}

/// A started run whose job cannot be found: its watch fails at once.
fn broken_run(runs: &Runs) -> Result<RunRecord, Box<dyn std::error::Error>> {
    let mut run = create(runs, "/nonexistent/pod", "gpu_cloud")?;
    run.state = RunState::Running;
    runs.save(&run)?;
    Ok(run)
}

#[tokio::test]
async fn a_watch_that_loses_a_deleted_pod_fails_the_run() -> TestResult {
    let harness = Harness::new().await?;
    serve_p1(&harness.server, true).await;
    let executor = LocalExecutor::new(&harness.project.path().join("pod"))?;
    let run = broken_run(&harness.runs)?;
    let mut pod = pod_record(&run.id, true, Duration::ZERO)?;
    let run_ctx = RunCtx {
        runs: &harness.runs,
        executor: &executor,
        bus: &harness.bus,
        poll: Duration::from_millis(5),
    };
    let result = follow(&harness.ctx(), &run_ctx, &Nothing, run.clone(), &mut pod).await;
    assert!(matches!(result, Err(PodError::PodGone(_))), "{result:?}");
    assert_eq!(pod.state, PodState::Deleted);
    let saved = harness.runs.load(&run.id)?;
    assert_eq!(saved.state, RunState::Failed);
    assert_eq!(saved.message.as_deref(), Some("pod p1 no longer exists"));
    assert_eq!(pod.deleted_by, Some(DeletedBy::Unknown));
    // A pod that only looks gone is never sent a delete, kept or not.
    assert_eq!(deletes(&harness.server).await, 0);
    assert_eq!(gets(&harness.server).await, 3);

    let alive = Harness::new().await?;
    serve_p1(&alive.server, false).await;
    let run = broken_run(&alive.runs)?;
    let mut pod = pod_record(&run.id, true, Duration::ZERO)?;
    let run_ctx = RunCtx {
        runs: &alive.runs,
        executor: &executor,
        bus: &alive.bus,
        poll: Duration::from_millis(5),
    };
    let result = follow(&alive.ctx(), &run_ctx, &Nothing, run.clone(), &mut pod).await;
    assert!(matches!(result, Err(PodError::Run(_))), "{result:?}");
    assert_eq!(alive.runs.load(&run.id)?.state, RunState::Running);
    assert_eq!(deletes(&alive.server).await, 0);
    Ok(())
}

#[tokio::test]
async fn a_pod_that_only_looks_gone_once_is_not_gone() -> TestResult {
    let executor = LocalExecutor::new(&tempfile::tempdir()?.path().join("pod"))?;
    // A 404 then the pod; 404s only, but the pod is still listed.
    for (present, listed) in [(vec![false, true], false), (vec![false], true)] {
        let harness = Harness::new().await?;
        serve_sequence(&harness.server, present, listed).await;
        let run = broken_run(&harness.runs)?;
        let mut pod = pod_record(&run.id, false, Duration::ZERO)?;
        pod.save(&harness.runs)?;
        let run_ctx = RunCtx {
            runs: &harness.runs,
            executor: &executor,
            bus: &harness.bus,
            poll: Duration::from_millis(5),
        };
        let result = follow(&harness.ctx(), &run_ctx, &Nothing, run.clone(), &mut pod).await;
        assert!(matches!(result, Err(PodError::Run(_))), "{result:?}");
        assert_eq!(pod.state, PodState::Provisioning);
        assert_eq!(
            PodRecord::load(&harness.runs, &run.id)?.map(|saved| saved.state),
            Some(PodState::Provisioning)
        );
        assert_eq!(harness.runs.load(&run.id)?.state, RunState::Running);
        assert_eq!(deletes(&harness.server).await, 0);
    }

    let harness = Harness::new().await?;
    Mock::given(method("GET"))
        .and(path("/v2/pods/p1"))
        .respond_with(Sequence {
            present: vec![false],
            looks: AtomicUsize::new(0),
        })
        .mount(&harness.server)
        .await;
    Mock::given(method("GET"))
        .and(path("/v2/pods"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&harness.server)
        .await;
    let run = broken_run(&harness.runs)?;
    let mut pod = pod_record(&run.id, false, Duration::ZERO)?;
    pod.save(&harness.runs)?;
    let run_ctx = RunCtx {
        runs: &harness.runs,
        executor: &executor,
        bus: &harness.bus,
        poll: Duration::from_millis(5),
    };
    let result = follow(&harness.ctx(), &run_ctx, &Nothing, run.clone(), &mut pod).await;
    assert!(matches!(result, Err(PodError::Run(_))), "{result:?}");
    assert_eq!(
        PodRecord::load(&harness.runs, &run.id)?.map(|saved| saved.state),
        Some(PodState::Provisioning)
    );
    assert_eq!(deletes(&harness.server).await, 0);
    Ok(())
}

/// A stopped pod will not run again by itself: waiting for it is pointless.
#[tokio::test]
async fn reconnect_to_a_stopped_pod_points_to_pod_rm() -> TestResult {
    let harness = Harness::new().await?;
    Mock::given(method("GET"))
        .and(path("/v2/pods/p1"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"id": "p1", "status": "EXITED"})),
        )
        .mount(&harness.server)
        .await;
    let run = broken_run(&harness.runs)?;
    let mut pod = pod_record(&run.id, true, Duration::ZERO)?;
    let error = reconnect(&harness.ctx(), &mut pod, &run)
        .await
        .err()
        .ok_or("reconnected to a stopped pod")?;
    assert!(matches!(error, PodError::PodStopped { .. }), "{error}");
    assert_eq!(
        error.to_string(),
        format!(
            "pod p1 is stopped and will not run again by itself; remove it with `overbrainer pod rm {}`",
            run.id
        )
    );
    assert_eq!(deletes(&harness.server).await, 0);
    Ok(())
}

#[tokio::test]
async fn reconnect_records_a_pod_confirmed_gone_without_deleting_it() -> TestResult {
    let harness = Harness::new().await?;
    serve_sequence(&harness.server, vec![false], false).await;
    let run = broken_run(&harness.runs)?;
    let mut pod = pod_record(&run.id, true, Duration::ZERO)?;
    let key = harness.runs.run_dir(&run.id)?.join("ssh/id_ed25519");
    std::fs::create_dir_all(key.parent().ok_or("no ssh dir")?)?;
    std::fs::write(&key, "private")?;
    let executor = reconnect(&harness.ctx(), &mut pod, &run).await?;
    assert!(executor.is_none());
    assert_eq!(pod.state, PodState::Deleted);
    assert_eq!(pod.deleted_by, Some(DeletedBy::Unknown));
    assert_eq!(PodRecord::load(&harness.runs, &run.id)?, Some(pod));
    assert!(!key.exists());
    assert_eq!(deletes(&harness.server).await, 0);
    assert_eq!(gets(&harness.server).await, 3);

    // A single 404, then the pod (without an SSH endpoint): not gone.
    let flaky = Harness::new().await?;
    serve_sequence(&flaky.server, vec![false, true], false).await;
    let run = broken_run(&flaky.runs)?;
    let mut pod = pod_record(&run.id, true, Duration::ZERO)?;
    let result = reconnect(&flaky.ctx(), &mut pod, &run).await;
    assert!(
        matches!(result, Err(PodError::NoEndpoint(..))),
        "{result:?}"
    );
    assert_eq!(pod.state, PodState::Provisioning);
    assert_eq!(deletes(&flaky.server).await, 0);
    Ok(())
}

#[tokio::test]
async fn a_kept_pod_past_a_deadline_is_never_deleted() -> TestResult {
    let harness = Harness::new().await?;
    serve_p1(&harness.server, false).await;
    let executor = LocalExecutor::new(&harness.project.path().join("pod"))?;
    let mut run = create(&harness.runs, executor.workdir(), "gpu_cloud")?;
    let job = executor
        .spawn(&JobCommand {
            dir: run.remote_dir.clone(),
            script: r#"sleep 1; printf '{"event": "log", "time": 1, "step": 1, "epoch": 1, "max_steps": 1, "loss": 1.5, "learning_rate": 0.0002}\n' >> metrics.jsonl"#.into(),
            secrets: Vec::new(),
            container: None,
        })
        .await?;
    run.job = Some(job);
    run.state = RunState::Running;
    harness.runs.save(&run)?;
    let mut pod = pod_record(&run.id, true, Duration::ZERO)?;
    // Whatever the record says, a kept pod has no deadline.
    let past = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs()
        .saturating_sub(3600);
    pod.deadline_unix = Some(past);
    let run_ctx = RunCtx {
        runs: &harness.runs,
        executor: &executor,
        bus: &harness.bus,
        poll: Duration::from_millis(50),
    };
    let outcome = follow(&harness.ctx(), &run_ctx, &Nothing, run.clone(), &mut pod).await?;
    assert_eq!(
        outcome.record.state,
        RunState::Succeeded,
        "{:?}",
        outcome.record.message
    );
    assert_ne!(pod.state, PodState::Deleted);
    assert_eq!(deletes(&harness.server).await, 0);
    Ok(())
}
