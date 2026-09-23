//! `runpod::pod_rows` and `runpod::remove_run_pods` against a local stub of the
//! Runpod API, with millisecond timing: which pods are orphans, when a recorded
//! pod or a stray is confirmed gone, and what `pod rm` deletes, keeps or refuses.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use overbrainer::events::EventBus;
use overbrainer::retry::RetryPolicy;
use overbrainer::runpod::{
    AttemptResult, DeletedBy, Pod, PodCtx, PodError, PodId, PodRecord, PodRow, PodState, RowKind,
    RunpodClient, Timing, orphan_warnings, pod_rows, remove_run_pods,
};
use overbrainer::runs::{RunRecord, RunState, Runs};
use secrecy::SecretString;
use serde_json::{Value, json};
use wiremock::matchers::any;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const RUN: &str = "20260922-143005-a1b2";
const ENDED: &str = "20260921-090000-ffff";
const ELSEWHERE: &str = "20260101-000000-dead";

fn pod(id: &str, run_id: Option<&str>) -> Value {
    let mut pod = json!({
        "id": id,
        "name": format!("overbrainer-{}-1", run_id.unwrap_or("x")),
        "status": "RUNNING",
        "cost": 0.53,
        "gpu": {"id": "NVIDIA A40", "count": 1},
        "createdAt": "2026-09-22T14:30:08Z",
    });
    if let Some(run_id) = run_id {
        pod["env"] = json!({"OVERBRAINER_RUN_ID": run_id});
    }
    pod
}

fn not_found() -> ResponseTemplate {
    ResponseTemplate::new(404).set_body_json(json!({
        "detail": "pod not found",
        "status": 404,
        "title": "Not Found"
    }))
}

/// The account: `pods` answer GET until deleted, `listed` ones show in the list,
/// a DELETE of a `forbidden` one and a GET of a `broken` one answer 400.
#[derive(Default)]
struct Account {
    pods: HashMap<String, Value>,
    listed: Vec<String>,
    forbidden: HashSet<String>,
    broken: HashSet<String>,
    deleted: Mutex<HashSet<String>>,
}

impl Account {
    fn with(mut self, id: &str, run_id: Option<&str>, listed: bool) -> Self {
        self.pods.insert(id.to_string(), pod(id, run_id));
        if listed {
            self.listed.push(id.to_string());
        }
        self
    }

    fn forbid(mut self, id: &str) -> Self {
        self.forbidden.insert(id.to_string());
        self
    }

    fn break_get(mut self, id: &str) -> Self {
        self.broken.insert(id.to_string());
        self
    }

    fn deleted(&self, id: &str) -> bool {
        self.deleted
            .lock()
            .is_ok_and(|deleted| deleted.contains(id))
    }

    fn list(&self) -> ResponseTemplate {
        let pods: Vec<&Value> = self
            .listed
            .iter()
            .filter(|id| !self.deleted(id))
            .filter_map(|id| self.pods.get(id))
            .collect();
        ResponseTemplate::new(200).set_body_json(json!({ "pods": pods }))
    }

    fn delete(&self, id: &str) -> ResponseTemplate {
        if self.forbidden.contains(id) {
            return ResponseTemplate::new(403).set_body_json(json!({"status": 403}));
        }
        let known = self.pods.contains_key(id) && !self.deleted(id);
        if let Ok(mut deleted) = self.deleted.lock() {
            deleted.insert(id.to_string());
        }
        if known {
            ResponseTemplate::new(204)
        } else {
            not_found()
        }
    }

    fn get(&self, id: &str) -> ResponseTemplate {
        if self.broken.contains(id) {
            return ResponseTemplate::new(400).set_body_json(json!({"status": 400}));
        }
        match self.pods.get(id) {
            Some(body) if !self.deleted(id) => ResponseTemplate::new(200).set_body_json(body),
            _ => not_found(),
        }
    }
}

struct Api(Arc<Account>);

impl Respond for Api {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let path = request.url.path();
        if path == "/v2/pods" {
            return self.0.list();
        }
        let id = path.strip_prefix("/v2/pods/").unwrap_or_default();
        if request.method.as_str() == "DELETE" {
            self.0.delete(id)
        } else {
            self.0.get(id)
        }
    }
}

struct Harness {
    server: MockServer,
    _project: tempfile::TempDir,
    runs: Runs,
    bus: EventBus,
    timing: Timing,
    interrupted: AtomicBool,
    client: RunpodClient,
}

impl Harness {
    async fn new(account: Account) -> Result<Self, Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        Mock::given(any())
            .respond_with(Api(Arc::new(account)))
            .mount(&server)
            .await;
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
            _project: project,
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

    /// A run in `state` whose `pod.json` records `pod_id` in `pod_state`.
    fn recorded(&self, id: &str, state: RunState, pod_id: &str, pod_state: PodState) -> TestResult {
        self.runs.save(&RunRecord {
            id: id.to_string(),
            target: "gpu_cloud".into(),
            created: "2026-09-22T14:30:05Z".into(),
            remote_dir: format!("/workspace/overbrainer/{id}"),
            job: None,
            state,
            message: None,
        })?;
        let mut record = PodRecord::new(id, pod_state == PodState::Kept, 1, "ssh-ed25519 AAAA");
        let remote: Pod = serde_json::from_value(pod(pod_id, Some(id)))?;
        record.begin_attempt("NVIDIA A40", SystemTime::now(), 6.0);
        record.created(&remote, AttemptResult::Created, SystemTime::now());
        record.state = pod_state;
        record.save(&self.runs)?;
        Ok(())
    }

    /// Adds `strays` to the `pod.json` of the run `id`.
    fn strays(&self, id: &str, strays: &[&str]) -> TestResult {
        let mut record = self.pod_json(id)?;
        for stray in strays {
            record.note_stray(PodId::new(stray)?);
        }
        record.save(&self.runs)?;
        Ok(())
    }

    fn pod_json(&self, id: &str) -> Result<PodRecord, Box<dyn std::error::Error>> {
        Ok(PodRecord::load(&self.runs, id)?.ok_or("no pod.json")?)
    }

    /// The calls the stub received with `method` on `path`.
    async fn calls(&self, method: &str, path: &str) -> usize {
        self.server
            .received_requests()
            .await
            .unwrap_or_default()
            .iter()
            .filter(|request| request.method.as_str() == method && request.url.path() == path)
            .count()
    }

    /// The pods the stub received a DELETE for, sorted.
    async fn deletes(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .server
            .received_requests()
            .await
            .unwrap_or_default()
            .iter()
            .filter(|request| request.method.as_str() == "DELETE")
            .filter_map(|request| request.url.path().rsplit('/').next().map(str::to_string))
            .collect();
        ids.sort();
        ids
    }
}

fn find<'a>(rows: &'a [PodRow], pod_id: &str) -> Result<&'a PodRow, String> {
    rows.iter()
        .find(|row| row.pod_id == pod_id)
        .ok_or(format!("no row for {pod_id}: {rows:?}"))
}

#[tokio::test]
async fn unlisted_recorded_pods_are_gone_only_after_three_404s_and_one_list() -> TestResult {
    let harness = Harness::new(Account::default()).await?;
    harness.recorded(RUN, RunState::Succeeded, "p1", PodState::Running)?;
    harness.recorded(ENDED, RunState::Failed, "p3", PodState::Kept)?;
    let rows = pod_rows(&harness.ctx()).await?;
    for id in ["p1", "p3"] {
        let row = find(&rows, id)?;
        assert_eq!((row.status.as_str(), row.kind), ("GONE", RowKind::Gone));
        assert_eq!(harness.calls("GET", &format!("/v2/pods/{id}")).await, 3);
    }
    // One list for the table, one for both confirmations.
    assert_eq!(harness.calls("GET", "/v2/pods").await, 2);
    for id in [RUN, ENDED] {
        let record = harness.pod_json(id)?;
        assert_eq!(record.state, PodState::Deleted);
        assert_eq!(record.deleted_by, Some(DeletedBy::Unknown));
    }
    assert!(harness.deletes().await.is_empty());
    Ok(())
}

#[tokio::test]
async fn a_recorded_pod_answering_again_is_not_gone() -> TestResult {
    // p1 is not listed but still answers: no wait, no confirmation.
    let harness = Harness::new(Account::default().with("p1", Some(RUN), false)).await?;
    harness.recorded(RUN, RunState::Running, "p1", PodState::Running)?;
    let rows = pod_rows(&harness.ctx()).await?;
    let row = find(&rows, "p1")?;
    assert_eq!(
        (row.status.as_str(), row.kind),
        ("RUNNING", RowKind::InProgress)
    );
    assert_eq!(harness.calls("GET", "/v2/pods/p1").await, 1);
    assert_eq!(harness.calls("GET", "/v2/pods").await, 1);
    assert_eq!(harness.pod_json(RUN)?.state, PodState::Running);
    Ok(())
}

#[tokio::test]
async fn strays_are_shown_and_dropped_once_confirmed_gone() -> TestResult {
    let account = Account::default()
        .with("p3", Some(ENDED), true)
        .with("s1", Some(ENDED), true)
        .with("s2", Some(ENDED), false);
    let harness = Harness::new(account).await?;
    harness.recorded(ENDED, RunState::Succeeded, "p3", PodState::Running)?;
    harness.strays(ENDED, &["s1", "s2", "s3"])?;
    let rows = pod_rows(&harness.ctx()).await?;
    for id in ["s1", "s2"] {
        let row = find(&rows, id)?;
        assert_eq!(
            (row.kind, row.note.as_str()),
            (RowKind::Stray, "stray, not deleted")
        );
        assert_eq!(row.run, ENDED);
    }
    assert_eq!(find(&rows, "p3")?.kind, RowKind::Ended);
    assert!(find(&rows, "s3").is_err());
    let strays: Vec<String> = harness
        .pod_json(ENDED)?
        .stray_pods
        .iter()
        .map(ToString::to_string)
        .collect();
    assert_eq!(strays, vec!["s1", "s2"]);
    assert!(harness.deletes().await.is_empty());
    Ok(())
}

#[tokio::test]
async fn a_failed_look_gives_an_unchecked_row_and_records_nothing() -> TestResult {
    let harness = Harness::new(Account::default().break_get("p3")).await?;
    harness.recorded(ENDED, RunState::Succeeded, "p3", PodState::Running)?;
    let rows = pod_rows(&harness.ctx()).await?;
    let row = find(&rows, "p3")?;
    assert_eq!(
        (row.status.as_str(), row.kind, row.note.as_str()),
        ("UNCHECKED", RowKind::Ended, "run succeeded, not deleted")
    );
    assert_eq!(row.created.len(), "2026-09-22T14:30:05Z".len());
    assert_eq!(harness.pod_json(ENDED)?.state, PodState::Running);
    Ok(())
}

#[tokio::test]
async fn pods_without_a_marker_or_a_local_run_get_safe_hints() -> TestResult {
    let account = Account::default()
        .with("n1", None, true)
        .with("p9", Some(ELSEWHERE), true);
    let harness = Harness::new(account).await?;
    let rows = pod_rows(&harness.ctx()).await?;
    let unmarked = find(&rows, "n1")?;
    assert_eq!(
        (unmarked.run.as_str(), unmarked.kind),
        ("-", RowKind::NoMarker)
    );
    let elsewhere = find(&rows, "p9")?;
    assert_eq!(elsewhere.kind, RowKind::NotInRuns);
    assert_eq!(
        orphan_warnings(&rows),
        vec![
            "pod n1 is named like an overbrainer pod but has no run marker; if it is yours, delete it from the Runpod console".to_string(),
            format!(
                "pod p9 is still on Runpod at $0.53/h: not in this project's runs/; if no other checkout owns it, `overbrainer pod rm {ELSEWHERE} --force`"
            ),
        ]
    );
    Ok(())
}

#[tokio::test]
async fn a_corrupt_pod_json_is_skipped_with_a_warning() -> TestResult {
    let harness = Harness::new(Account::default().with("p1", Some(RUN), true)).await?;
    harness.recorded(RUN, RunState::Running, "p1", PodState::Running)?;
    harness.recorded(ENDED, RunState::Succeeded, "p3", PodState::Running)?;
    std::fs::write(harness.runs.run_dir(ENDED)?.join("pod.json"), "{not json")?;
    let rows = pod_rows(&harness.ctx()).await?;
    assert_eq!(find(&rows, "p1")?.kind, RowKind::InProgress);
    assert!(find(&rows, "p3").is_err());
    Ok(())
}

#[tokio::test]
async fn pod_rm_of_a_run_in_progress_keeps_its_training_pod_and_deletes_the_others() -> TestResult {
    for state in [RunState::Running, RunState::Preparing] {
        let account = Account::default()
            .with("p1", Some(RUN), true)
            .with("s1", Some(RUN), true)
            .with("x2", Some(RUN), true);
        let harness = Harness::new(account).await?;
        harness.recorded(RUN, state, "p1", PodState::Running)?;
        harness.strays(RUN, &["s1"])?;
        let removal = remove_run_pods(&harness.ctx(), RUN, false).await;
        let error = removal.result.err().ok_or("expected a refusal")?;
        assert!(matches!(error, PodError::RunStillRunning { .. }), "{error}");
        assert_eq!(
            error.to_string(),
            format!(
                "run {RUN} is still running: its training pod p1 was kept, and its other pods s1, x2 were deleted; stop it with `overbrainer train cancel {RUN}` first, or pass --force"
            )
        );
        let ids: Vec<String> = removal
            .removed
            .iter()
            .map(|pod| pod.pod_id.to_string())
            .collect();
        assert_eq!(ids, vec!["s1", "x2"]);
        assert_eq!(harness.deletes().await, vec!["s1", "x2"]);
        let record = harness.pod_json(RUN)?;
        assert_eq!(record.state, PodState::Running);
        assert!(record.stray_pods.is_empty());
        assert_eq!(harness.runs.load(RUN)?.state, state);
    }
    Ok(())
}

#[tokio::test]
async fn a_forced_pod_rm_of_a_running_run_deletes_everything_and_fails_it() -> TestResult {
    let account = Account::default()
        .with("p1", Some(RUN), true)
        .with("x2", Some(RUN), true);
    let harness = Harness::new(account).await?;
    harness.recorded(RUN, RunState::Running, "p1", PodState::Running)?;
    let removal = remove_run_pods(&harness.ctx(), RUN, true).await;
    removal.result?;
    assert_eq!(harness.deletes().await, vec!["p1", "x2"]);
    let run = harness.runs.load(RUN)?;
    assert_eq!(run.state, RunState::Failed);
    let record = harness.pod_json(RUN)?;
    assert_eq!(
        (record.state, record.deleted_by),
        (PodState::Deleted, Some(DeletedBy::PodRm))
    );
    Ok(())
}

#[tokio::test]
async fn pod_rm_of_a_run_not_in_runs_needs_force() -> TestResult {
    let harness = Harness::new(Account::default().with("p9", Some(ELSEWHERE), true)).await?;
    let refused = remove_run_pods(&harness.ctx(), ELSEWHERE, false).await;
    assert!(matches!(refused.result, Err(PodError::NotInRuns(_))));
    assert!(refused.removed.is_empty());
    assert!(harness.deletes().await.is_empty());
    let forced = remove_run_pods(&harness.ctx(), ELSEWHERE, true).await;
    forced.result?;
    assert_eq!(forced.removed.len(), 1);
    assert_eq!(harness.deletes().await, vec!["p9"]);
    Ok(())
}

#[tokio::test]
async fn pod_rm_tries_every_pod_and_records_a_failed_marker_pod_as_stray() -> TestResult {
    let account = Account::default()
        .with("p3", Some(ENDED), true)
        .with("s1", Some(ENDED), true)
        .with("s3", Some(ENDED), true)
        .with("x2", Some(ENDED), true)
        .with("x4", Some(ENDED), true)
        .forbid("s3")
        .forbid("x2");
    let harness = Harness::new(account).await?;
    harness.recorded(ENDED, RunState::Succeeded, "p3", PodState::Kept)?;
    harness.strays(ENDED, &["s1", "s2", "s3"])?;
    let removal = remove_run_pods(&harness.ctx(), ENDED, false).await;
    assert!(removal.result.is_err());
    let mut ids: Vec<String> = removal
        .removed
        .iter()
        .map(|pod| pod.pod_id.to_string())
        .collect();
    ids.sort();
    // s2 is unknown to Runpod: its DELETE is still sent, then confirmed.
    assert_eq!(ids, vec!["p3", "s1", "s2", "x4"]);
    assert_eq!(
        harness.deletes().await,
        vec!["p3", "s1", "s2", "s3", "x2", "x4"]
    );
    let record = harness.pod_json(ENDED)?;
    assert_eq!(record.state, PodState::Deleted);
    let strays: Vec<String> = record.stray_pods.iter().map(ToString::to_string).collect();
    assert_eq!(strays, vec!["s3", "x2"]);
    Ok(())
}
