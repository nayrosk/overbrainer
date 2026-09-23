//! Provisioning a run's pod against a local stub of the Runpod API: the ordered
//! GPU list, `pod.json` written before each create, reconciliation of ambiguous
//! creates, pods that never become ready, interruption, confirmed deletes. SSH
//! readiness itself is covered against a real sshd in `tests/runpod_ssh.rs`.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime};

use overbrainer::events::{Event, EventBus};
use overbrainer::retry::RetryPolicy;
use overbrainer::runpod::{
    AttemptResult, DeleteReason, DeletedBy, PodCtx, PodError, PodKeys, PodPlan, PodRecord,
    PodState, PodStatus, RunpodClient, RunpodTarget, Timing, provision, remove,
};
use overbrainer::runs::Runs;
use secrecy::SecretString;
use serde_json::{Value, json};
use tokio::sync::broadcast::Receiver;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const RUN: &str = "20260922-143005-a1b2";
const HOST_KEY: &str = "aG9zdC1rZXk";

/// Runpod's answer to a create call when the GPU type has no capacity left.
fn no_capacity() -> ResponseTemplate {
    ResponseTemplate::new(400).set_body_json(json!({
        "detail": "There are no longer any instances available with the requested specifications. Please refresh and try again.",
        "status": 400,
        "title": "Bad Request"
    }))
}

/// Runpod's answer about a pod it no longer knows.
fn gone() -> ResponseTemplate {
    ResponseTemplate::new(404).set_body_json(json!({
        "detail": "pod not found",
        "status": 404,
        "title": "Not Found"
    }))
}

/// A pod as the API returns it.
fn pod(id: &str, name: &str, status: &str) -> Value {
    json!({
        "id": id,
        "name": name,
        "status": status,
        "cost": 0.53,
        "dataCenterId": "EU-RO-1",
        "gpu": {"id": "NVIDIA A40", "count": 1},
        "env": {"OVERBRAINER_RUN_ID": RUN},
        "ssh": {"direct": null}
    })
}

/// `GET /pods/{id}`: `body` until the pod is deleted, then 404.
struct Get {
    deleted: Arc<AtomicBool>,
    body: Value,
}

impl Respond for Get {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        if self.deleted.load(Ordering::SeqCst) {
            gone()
        } else {
            ResponseTemplate::new(200).set_body_json(&self.body)
        }
    }
}

/// `DELETE /pods/{id}`: marks the pod deleted.
struct Delete(Arc<AtomicBool>);

impl Respond for Delete {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        self.0.store(true, Ordering::SeqCst);
        ResponseTemplate::new(204)
    }
}

/// Serves pod `id` with `body` until a `DELETE` of it.
async fn serve_pod(server: &MockServer, id: &str, body: Value) {
    let deleted = Arc::new(AtomicBool::new(false));
    Mock::given(method("GET"))
        .and(path(format!("/v2/pods/{id}")))
        .respond_with(Get {
            deleted: Arc::clone(&deleted),
            body,
        })
        .mount(server)
        .await;
    Mock::given(method("DELETE"))
        .and(path(format!("/v2/pods/{id}")))
        .respond_with(Delete(deleted))
        .mount(server)
        .await;
}

async fn list(server: &MockServer, pods: Value) {
    Mock::given(method("GET"))
        .and(path("/v2/pods"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"pods": pods})))
        .mount(server)
        .await;
}

async fn create_for(server: &MockServer, gpu: &str, response: ResponseTemplate) {
    Mock::given(method("POST"))
        .and(path("/v2/pods"))
        .and(body_partial_json(json!({"gpu": {"id": gpu}})))
        .respond_with(response)
        .mount(server)
        .await;
}

fn target(gpu_types: &[&str]) -> RunpodTarget {
    RunpodTarget {
        gpu_types: gpu_types.iter().map(|gpu| (*gpu).to_string()).collect(),
        gpu_count: 1,
        image: "img@sha256:abc".into(),
        venv: "/workspace/axolotl-venv".into(),
        container_disk_gb: 50,
        max_hours: 6.0,
        boot_grace: Duration::from_secs(1800),
        retrieve_grace: Duration::from_secs(3600),
        data_center_ids: Vec::new(),
        network_volume_id: None,
    }
}

/// Everything provisioning needs, around a stub.
struct Harness {
    server: MockServer,
    _project: tempfile::TempDir,
    runs: Runs,
    ssh_dir: PathBuf,
    bus: EventBus,
    timing: Timing,
    interrupted: Arc<AtomicBool>,
    keys: PodKeys,
    client: RunpodClient,
}

impl Harness {
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let ssh_dir = runs.run_dir(RUN)?.join("ssh");
        let client = RunpodClient::new(
            &format!("{}/v2", server.uri()),
            &SecretString::from("rp_key"),
        )?
        .with_policy(RetryPolicy {
            max_retries: 1,
            base: Duration::from_millis(1),
            cap: Duration::from_millis(2),
        });
        Ok(Self {
            server,
            _project: project,
            runs,
            ssh_dir,
            bus: EventBus::new(),
            timing: Timing {
                poll: Duration::from_millis(5),
                ready_timeout: Duration::from_millis(300),
                preflight_timeout: Duration::from_millis(300),
                reconcile_waits: [Duration::from_millis(5), Duration::from_millis(5)],
                delete_timeout: Duration::from_millis(300),
            },
            interrupted: Arc::new(AtomicBool::new(false)),
            keys: PodKeys::new(
                PathBuf::from("/nonexistent/id_ed25519"),
                "ssh-ed25519 AAAAclient overbrainer".into(),
                "ssh-ed25519 AAAAhost".into(),
                SecretString::from(HOST_KEY),
            ),
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

    async fn provision(
        &self,
        target: &RunpodTarget,
    ) -> Result<(Result<(), PodError>, PodRecord), Box<dyn std::error::Error>> {
        let mut record = PodRecord::new(RUN, false, 1, "ssh-ed25519 AAAAhost");
        record.save(&self.runs)?;
        let plan = PodPlan {
            run_id: RUN,
            target,
            keys: &self.keys,
            ssh_dir: &self.ssh_dir,
            workdir: "/workspace/overbrainer",
            api_url: self.client.base_url(),
        };
        let result = provision(&self.ctx(), &plan, &mut record).await.map(drop);
        assert_eq!(PodRecord::load(&self.runs, RUN)?.as_ref(), Some(&record));
        Ok((result, record))
    }

    async fn calls(&self, verb: &str) -> Vec<Request> {
        self.server
            .received_requests()
            .await
            .unwrap_or_default()
            .into_iter()
            .filter(|request| request.method.as_str() == verb)
            .collect()
    }
}

fn statuses(receiver: &mut Receiver<Event>) -> Vec<PodStatus> {
    let mut statuses = Vec::new();
    while let Ok(event) = receiver.try_recv() {
        if let Event::PodStatus(status) = event {
            statuses.push(status);
        }
    }
    statuses
}

fn results(record: &PodRecord) -> Vec<AttemptResult> {
    record
        .attempts
        .iter()
        .map(|attempt| attempt.result)
        .collect()
}

#[tokio::test]
async fn every_gpu_type_is_tried_in_order_until_none_is_left() -> TestResult {
    let harness = Harness::new().await?;
    let mut receiver = harness.bus.subscribe();
    for gpu in ["NVIDIA GeForce RTX 4090", "NVIDIA A40"] {
        create_for(&harness.server, gpu, no_capacity()).await;
    }
    let (result, record) = harness
        .provision(&target(&["NVIDIA GeForce RTX 4090", "NVIDIA A40"]))
        .await?;
    let error = result.err().ok_or("provisioning succeeded")?;
    assert_eq!(
        error.to_string(),
        "no gpu_types entry could be placed: Runpod answered 400: no capacity for this GPU type"
    );
    assert_eq!(
        results(&record),
        vec![AttemptResult::Unavailable, AttemptResult::Unavailable]
    );
    let names: Vec<&str> = record.attempts.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "overbrainer-20260922-143005-a1b2-1",
            "overbrainer-20260922-143005-a1b2-2"
        ]
    );
    let sent: Vec<Value> = harness
        .calls("POST")
        .await
        .iter()
        .map(|request| serde_json::from_slice::<Value>(&request.body).unwrap_or_default())
        .collect();
    assert_eq!(sent[0]["gpu"]["id"], "NVIDIA GeForce RTX 4090");
    assert_eq!(sent[1]["gpu"]["id"], "NVIDIA A40");
    let events = statuses(&mut receiver);
    assert!(matches!(
        &events[..],
        [
            PodStatus::Creating { .. },
            PodStatus::Unavailable { .. },
            PodStatus::Creating { .. },
            PodStatus::Unavailable { .. }
        ]
    ));
    Ok(())
}

/// Answers 400 after noting whether `pod.json` already held a `sent` attempt.
struct Recorded {
    file: PathBuf,
    seen: Arc<AtomicBool>,
}

impl Respond for Recorded {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        let text = std::fs::read_to_string(&self.file).unwrap_or_default();
        if text.contains("\"result\": \"sent\"") {
            self.seen.store(true, Ordering::SeqCst);
        }
        ResponseTemplate::new(400)
    }
}

#[tokio::test]
async fn pod_json_holds_the_attempt_before_the_create_arrives() -> TestResult {
    let harness = Harness::new().await?;
    let seen = Arc::new(AtomicBool::new(false));
    Mock::given(method("POST"))
        .respond_with(Recorded {
            file: harness.runs.run_dir(RUN)?.join("pod.json"),
            seen: Arc::clone(&seen),
        })
        .mount(&harness.server)
        .await;
    let (result, _) = harness.provision(&target(&["NVIDIA A40"])).await?;
    assert!(result.is_err());
    assert!(seen.load(Ordering::SeqCst));
    Ok(())
}

#[tokio::test]
async fn lack_of_credits_and_a_rejected_request_stop_at_once() -> TestResult {
    for (status, expected) in [
        (402, "Runpod refused for lack of credits (402)"),
        (422, "Runpod rejected overbrainer's request"),
    ] {
        let harness = Harness::new().await?;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(status))
            .mount(&harness.server)
            .await;
        let (result, record) = harness.provision(&target(&["A", "B"])).await?;
        let error = result.err().ok_or("provisioning succeeded")?.to_string();
        assert!(error.starts_with(expected), "{error}");
        assert_eq!(harness.calls("POST").await.len(), 1);
        assert_eq!(results(&record), vec![AttemptResult::Rejected]);
    }
    Ok(())
}

#[tokio::test]
async fn a_400_that_is_not_about_capacity_stops_at_once() -> TestResult {
    let harness = Harness::new().await?;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(400).set_body_json(json!({"detail": "gpu.count is invalid"})),
        )
        .mount(&harness.server)
        .await;
    let (result, record) = harness.provision(&target(&["A", "B"])).await?;
    let error = result.err().ok_or("provisioning succeeded")?.to_string();
    assert!(
        error.starts_with("Runpod rejected overbrainer's request"),
        "{error}"
    );
    assert!(
        error.ends_with(
            "Runpod answered 400: Runpod rejected the create request (please report it: overbrainer built an invalid request)"
        ),
        "{error}"
    );
    assert!(!error.contains("gpu.count"), "{error}");
    assert_eq!(harness.calls("POST").await.len(), 1);
    assert_eq!(results(&record), vec![AttemptResult::Rejected]);
    Ok(())
}

#[tokio::test]
async fn a_403_moves_to_the_next_gpu_type() -> TestResult {
    let harness = Harness::new().await?;
    let mut receiver = harness.bus.subscribe();
    create_for(&harness.server, "A", ResponseTemplate::new(403)).await;
    create_for(&harness.server, "B", no_capacity()).await;
    let (result, record) = harness.provision(&target(&["A", "B"])).await?;
    assert!(matches!(result, Err(PodError::NoCapacity(_))), "{result:?}");
    assert_eq!(
        results(&record),
        vec![AttemptResult::Forbidden, AttemptResult::Unavailable]
    );
    let forbidden = "Runpod answered 403: Runpod refused the request (403): check the API key and that the GPU type is allowed";
    assert_eq!(record.attempts[0].detail.as_deref(), Some(forbidden));
    assert_eq!(
        record.attempts[1].detail.as_deref(),
        Some("Runpod answered 400: no capacity for this GPU type")
    );
    let reasons: Vec<String> = statuses(&mut receiver)
        .into_iter()
        .filter_map(|status| match status {
            PodStatus::Unavailable { reason, .. } => Some(reason),
            _ => None,
        })
        .collect();
    assert_eq!(
        reasons,
        vec![
            forbidden.to_string(),
            "Runpod answered 400: no capacity for this GPU type".to_string()
        ]
    );
    Ok(())
}

#[tokio::test]
async fn a_pod_gone_or_dead_before_ssh_ends_its_attempt_at_once() -> TestResult {
    let mut harness = Harness::new().await?;
    harness.timing.ready_timeout = Duration::from_secs(60);
    create_for(
        &harness.server,
        "A",
        ResponseTemplate::new(201).set_body_json(pod("p1", "n1", "RUNNING")),
    )
    .await;
    create_for(
        &harness.server,
        "B",
        ResponseTemplate::new(201).set_body_json(pod("p2", "n2", "RUNNING")),
    )
    .await;
    // p1's watchdog deleted it after a failed bootstrap: Runpod no longer knows it.
    Mock::given(path("/v2/pods/p1"))
        .respond_with(gone())
        .mount(&harness.server)
        .await;
    serve_pod(&harness.server, "p2", pod("p2", "n2", "TERMINATED")).await;
    let started = std::time::Instant::now();
    let (result, record) = harness.provision(&target(&["A", "B"])).await?;
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "waited {:?}",
        started.elapsed()
    );
    assert!(matches!(result, Err(PodError::NoCapacity(_))), "{result:?}");
    assert_eq!(
        results(&record),
        vec![AttemptResult::NotReady, AttemptResult::NotReady]
    );
    assert_eq!(
        record.attempts[0].detail.as_deref(),
        Some("the pod disappeared")
    );
    assert_eq!(
        record.attempts[1].detail.as_deref(),
        Some("the pod is TERMINATED")
    );
    let deleted: Vec<String> = harness
        .calls("DELETE")
        .await
        .iter()
        .map(|request| request.url.path().to_string())
        .collect();
    assert_eq!(deleted, vec!["/v2/pods/p1", "/v2/pods/p2"]);
    assert_eq!(record.pod_id, None);
    Ok(())
}

#[tokio::test]
async fn an_ambiguous_create_adopts_its_pod_and_deletes_a_duplicate() -> TestResult {
    let harness = Harness::new().await?;
    let mut receiver = harness.bus.subscribe();
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&harness.server)
        .await;
    let name = "overbrainer-20260922-143005-a1b2-1";
    list(
        &harness.server,
        json!([
            pod("adopted", name, "RUNNING"),
            pod("dup", "overbrainer-old", "RUNNING")
        ]),
    )
    .await;
    serve_pod(&harness.server, "adopted", pod("adopted", name, "EXITED")).await;
    serve_pod(
        &harness.server,
        "dup",
        pod("dup", "overbrainer-old", "RUNNING"),
    )
    .await;
    let (result, record) = harness.provision(&target(&["NVIDIA A40"])).await?;
    assert!(matches!(result, Err(PodError::NoCapacity(_))), "{result:?}");
    let attempt = &record.attempts[0];
    assert_eq!(attempt.result, AttemptResult::NotReady);
    assert_eq!(attempt.detail.as_deref(), Some("the pod is EXITED"));
    assert_eq!(
        attempt
            .pod_id
            .as_ref()
            .map(overbrainer::runpod::PodId::as_str),
        Some("adopted")
    );
    let deleted: Vec<String> = harness
        .calls("DELETE")
        .await
        .iter()
        .map(|request| request.url.path().to_string())
        .collect();
    assert_eq!(deleted, vec!["/v2/pods/dup", "/v2/pods/adopted"]);
    assert!(statuses(&mut receiver).contains(&PodStatus::Deleting {
        pod_id: overbrainer::runpod::PodId::new("dup")?,
        reason: DeleteReason::Duplicate,
    }));
    Ok(())
}

#[tokio::test]
async fn an_ambiguous_create_without_a_pod_is_sent_again() -> TestResult {
    let harness = Harness::new().await?;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(503))
        .up_to_n_times(1)
        .mount(&harness.server)
        .await;
    Mock::given(method("POST"))
        .respond_with(no_capacity())
        .mount(&harness.server)
        .await;
    list(&harness.server, json!([])).await;
    let (result, record) = harness.provision(&target(&["NVIDIA A40"])).await?;
    assert!(matches!(result, Err(PodError::NoCapacity(_))), "{result:?}");
    assert_eq!(
        results(&record),
        vec![AttemptResult::Ambiguous, AttemptResult::Unavailable]
    );
    assert_eq!(harness.calls("GET").await.len(), 2, "two looks for the pod");
    Ok(())
}

#[tokio::test]
async fn a_pod_that_dies_or_never_gets_ssh_is_deleted_and_the_next_type_tried() -> TestResult {
    let harness = Harness::new().await?;
    let mut receiver = harness.bus.subscribe();
    create_for(
        &harness.server,
        "A",
        ResponseTemplate::new(201).set_body_json(pod("p1", "n1", "RUNNING")),
    )
    .await;
    create_for(
        &harness.server,
        "B",
        ResponseTemplate::new(201).set_body_json(pod("p2", "n2", "RUNNING")),
    )
    .await;
    serve_pod(&harness.server, "p1", pod("p1", "n1", "ERROR")).await;
    serve_pod(&harness.server, "p2", pod("p2", "n2", "RUNNING")).await;
    list(&harness.server, json!([])).await;
    let (result, record) = harness.provision(&target(&["A", "B"])).await?;
    assert!(matches!(result, Err(PodError::NoCapacity(_))), "{result:?}");
    assert_eq!(
        results(&record),
        vec![AttemptResult::NotReady, AttemptResult::NotReady]
    );
    assert_eq!(
        record.attempts[0].detail.as_deref(),
        Some("the pod is ERROR")
    );
    let waited = record.attempts[1].detail.clone().unwrap_or_default();
    assert!(
        waited.starts_with("not reachable over SSH after 0s: no SSH endpoint yet"),
        "{waited}"
    );
    assert_eq!(harness.calls("DELETE").await.len(), 2);
    assert_eq!(record.pod_id, None);
    let deleted = statuses(&mut receiver)
        .into_iter()
        .filter(|status| matches!(status, PodStatus::Deleted { .. }))
        .count();
    assert_eq!(deleted, 2);
    Ok(())
}

#[tokio::test]
async fn nothing_is_created_after_ctrl_c() -> TestResult {
    let harness = Harness::new().await?;
    harness.interrupted.store(true, Ordering::SeqCst);
    let (result, record) = harness.provision(&target(&["A"])).await?;
    assert!(matches!(result, Err(PodError::Interrupted)), "{result:?}");
    assert!(record.attempts.is_empty());
    assert!(
        harness
            .server
            .received_requests()
            .await
            .unwrap_or_default()
            .is_empty()
    );
    Ok(())
}

/// Answers the create with pod `p1`, and presses Ctrl-C meanwhile.
struct CreateThenInterrupt(Arc<AtomicBool>);

impl Respond for CreateThenInterrupt {
    fn respond(&self, _: &Request) -> ResponseTemplate {
        self.0.store(true, Ordering::SeqCst);
        ResponseTemplate::new(201).set_body_json(pod("p1", "n1", "RUNNING"))
    }
}

#[tokio::test]
async fn ctrl_c_while_the_pod_starts_deletes_it() -> TestResult {
    let harness = Harness::new().await?;
    let mut receiver = harness.bus.subscribe();
    Mock::given(method("POST"))
        .respond_with(CreateThenInterrupt(Arc::clone(&harness.interrupted)))
        .mount(&harness.server)
        .await;
    serve_pod(&harness.server, "p1", pod("p1", "n1", "RUNNING")).await;
    let (result, record) = harness.provision(&target(&["A", "B"])).await?;
    assert!(matches!(result, Err(PodError::Interrupted)), "{result:?}");
    assert_eq!(record.state, PodState::Deleted);
    assert_eq!(record.deleted_by, Some(DeletedBy::Client));
    assert_eq!(harness.calls("POST").await.len(), 1);
    assert!(statuses(&mut receiver).contains(&PodStatus::Deleting {
        pod_id: overbrainer::runpod::PodId::new("p1")?,
        reason: DeleteReason::Interrupted,
    }));
    Ok(())
}

#[tokio::test]
async fn the_create_carries_the_target_and_the_watchdog_settings() -> TestResult {
    let harness = Harness::new().await?;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(400))
        .mount(&harness.server)
        .await;
    let mut target = target(&["NVIDIA A40"]);
    target.data_center_ids = vec!["EU-RO-1".into()];
    target.network_volume_id = Some("vol1".into());
    let before = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();
    let (result, _) = harness.provision(&target).await?;
    assert!(result.is_err());
    let requests = harness.calls("POST").await;
    let body: Value = serde_json::from_slice(&requests[0].body)?;
    assert_eq!(body["name"], "overbrainer-20260922-143005-a1b2-1");
    assert_eq!(body["image"], "img@sha256:abc");
    assert_eq!(body["cloud"], "SECURE");
    assert_eq!(
        body["gpu"],
        json!({"id": "NVIDIA A40", "count": 1, "minCudaVersion": "13.0"})
    );
    assert_eq!(body["disk"], 50);
    assert_eq!(body["ports"], json!(["22/tcp"]));
    assert_eq!(body["startSsh"], false);
    assert_eq!(body["dataCenterIds"], json!(["EU-RO-1"]));
    assert_eq!(
        body["mounts"],
        json!({"network": [{"volumeId": "vol1", "path": "/workspace/data"}]})
    );
    assert_eq!(body["cmd"][0], "bash");
    let env = &body["env"];
    assert_eq!(env["OVERBRAINER_RUN_ID"], RUN);
    assert_eq!(env["OVERBRAINER_HOST_KEY"], HOST_KEY);
    assert_eq!(
        env["OVERBRAINER_AUTHORIZED_KEY"],
        "ssh-ed25519 AAAAclient overbrainer"
    );
    assert_eq!(
        env["OVERBRAINER_API_URL"],
        format!("{}/v2", harness.server.uri())
    );
    assert_eq!(env["OVERBRAINER_KEEP_POD"], "0");
    let deadline: u64 = env["OVERBRAINER_DEADLINE"]
        .as_str()
        .unwrap_or("0")
        .parse()?;
    assert!((before + 6 * 3600..=before + 6 * 3600 + 5).contains(&deadline));
    assert!(env.get("HF_TOKEN").is_none());
    Ok(())
}

#[tokio::test]
async fn a_delete_is_confirmed_by_the_api_and_priced() -> TestResult {
    let harness = Harness::new().await?;
    serve_pod(&harness.server, "p1", pod("p1", "n1", "RUNNING")).await;
    let pod: overbrainer::runpod::Pod = serde_json::from_value(pod("p1", "n1", "RUNNING"))?;
    let mut record = PodRecord::new(RUN, false, 1, "ssh-ed25519 AAAAhost");
    let two_hours_ago = SystemTime::now() - Duration::from_secs(7200);
    record.begin_attempt("NVIDIA A40", two_hours_ago, 6.0);
    record.created(&pod, AttemptResult::Created, two_hours_ago);
    remove(
        &harness.ctx(),
        &mut record,
        DeleteReason::Retrieved,
        DeletedBy::Client,
    )
    .await?;
    assert_eq!(record.state, PodState::Deleted);
    let spend = record.estimated_spend.ok_or("no spend")?;
    assert!((spend - 1.06).abs() < 0.01, "{spend}");
    assert_eq!(PodRecord::load(&harness.runs, RUN)?, Some(record));

    let stuck = Harness::new().await?;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"id": "p1"})))
        .mount(&stuck.server)
        .await;
    Mock::given(method("DELETE"))
        .respond_with(ResponseTemplate::new(204))
        .mount(&stuck.server)
        .await;
    let mut record = PodRecord::new(RUN, false, 1, "ssh-ed25519 AAAAhost");
    record.begin_attempt("NVIDIA A40", SystemTime::now(), 6.0);
    record.created(&pod, AttemptResult::Created, SystemTime::now());
    let result = remove(
        &stuck.ctx(),
        &mut record,
        DeleteReason::Requested,
        DeletedBy::PodRm,
    )
    .await;
    assert!(matches!(result, Err(PodError::NotDeleted(_))), "{result:?}");
    assert_eq!(record.state, PodState::Deleting);
    Ok(())
}
