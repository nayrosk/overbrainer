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
    RunpodClient, Timing, host_key_secret, listed_rows, orphan_warnings, pod_rows, remove_run_pods,
    sweep_host_keys,
};
use overbrainer::runs::{RunRecord, RunState, Runs};
use secrecy::SecretString;
use serde_json::{Value, json};
use wiremock::matchers::any;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

use crate::common::SecretStore;

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
    flaky: Mutex<HashSet<String>>,
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

    /// The next GET of `id` answers Runpod's 404, and the later ones the pod.
    fn flaky(self, id: &str) -> Self {
        if let Ok(mut flaky) = self.flaky.lock() {
            flaky.insert(id.to_string());
        }
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
        if self.flaky.lock().is_ok_and(|mut flaky| flaky.remove(id)) {
            return not_found();
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
    secrets: SecretStore,
}

impl Harness {
    async fn new(account: Account) -> Result<Self, Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        let secrets = SecretStore::mount(&server).await;
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
            secrets,
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
            snapshot: None,
            resumed_from: None,
            snapshots: true,
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
            .filter(|request| {
                request.method.as_str() == "DELETE" && request.url.path().starts_with("/v2/pods/")
            })
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
    assert_eq!(harness.deletes().await, [] as [String; 0]);
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
async fn a_pod_answering_again_in_a_later_round_is_not_gone() -> TestResult {
    let harness = Harness::new(Account::default().with("p1", Some(RUN), false).flaky("p1")).await?;
    harness.recorded(RUN, RunState::Running, "p1", PodState::Running)?;
    let rows = pod_rows(&harness.ctx()).await?;
    let row = find(&rows, "p1")?;
    assert_eq!(
        (row.status.as_str(), row.kind),
        ("RUNNING", RowKind::InProgress)
    );
    assert_eq!(harness.calls("GET", "/v2/pods/p1").await, 2);
    let record = harness.pod_json(RUN)?;
    assert_eq!((record.state, record.deleted_by), (PodState::Running, None));
    Ok(())
}

#[tokio::test]
async fn the_stray_hint_of_a_kept_run_says_pod_rm_deletes_the_kept_pod_too() -> TestResult {
    for pod_state in [PodState::Kept, PodState::AwaitingRetrieval] {
        let account =
            Account::default()
                .with("p3", Some(ENDED), true)
                .with("s1", Some(ENDED), true);
        let harness = Harness::new(account).await?;
        harness.recorded(ENDED, RunState::Succeeded, "p3", pod_state)?;
        harness.strays(ENDED, &["s1"])?;
        let rows = pod_rows(&harness.ctx()).await?;
        assert_eq!(
            orphan_warnings(&rows),
            vec![format!(
                "pod s1 (stray, not deleted) is still on Runpod at $0.53/h: remove it with `overbrainer pod rm {ENDED}` (this also deletes the run's kept pod)"
            )]
        );
    }
    Ok(())
}

/// A pod carrying a run's marker that is neither the pod in its `pod.json` nor a
/// known stray (an extra pod of an unclear create) is a stray: nothing else
/// guards it, least of all beside a kept pod whose watchdog stays its hand.
#[tokio::test]
async fn an_extra_marker_pod_is_a_stray_and_an_orphan() -> TestResult {
    let cases = [
        (
            PodState::Kept,
            RunState::Succeeded,
            RowKind::StrayBesideKept,
        ),
        (PodState::Running, RunState::Running, RowKind::Stray),
    ];
    for (pod_state, run_state, kind) in cases {
        let account =
            Account::default()
                .with("p3", Some(ENDED), true)
                .with("x2", Some(ENDED), true);
        let harness = Harness::new(account).await?;
        harness.recorded(ENDED, run_state, "p3", pod_state)?;
        for rows in [
            pod_rows(&harness.ctx()).await?,
            listed_rows(&harness.ctx()).await?,
        ] {
            let extra = find(&rows, "x2")?;
            assert_eq!(
                (extra.kind, extra.note.as_str()),
                (kind, "stray, not deleted")
            );
            assert!(extra.is_orphan(), "{extra:?}");
            assert_ne!(find(&rows, "p3")?.kind, kind);
            let warnings = orphan_warnings(&rows);
            assert_eq!(warnings.len(), 1, "{warnings:?}");
            assert!(
                warnings[0].starts_with(&format!(
                    "pod x2 (stray, not deleted) is still on Runpod at $0.53/h: remove it with `overbrainer pod rm {ENDED}`"
                )),
                "{warnings:?}"
            );
        }
        assert_eq!(harness.deletes().await, [] as [String; 0]);
    }
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
    assert_eq!(harness.deletes().await, [] as [String; 0]);
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
async fn pod_rm_of_a_running_run_keeps_its_training_pod_and_deletes_the_others() -> TestResult {
    let state = RunState::Running;
    let account = Account::default()
        .with("p1", Some(RUN), true)
        .with("s1", Some(RUN), true)
        .with("x2", Some(RUN), true);
    let harness = Harness::new(account).await?;
    harness.secrets.hold("k1", &host_key_secret(RUN));
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
    assert_eq!(record.stray_pods, [] as [overbrainer::runpod::PodId; 0]);
    assert_eq!(harness.runs.load(RUN)?.state, state);
    // The training pod may restart and need its host key again.
    assert_eq!(harness.secrets.names(), [host_key_secret(RUN)]);
    Ok(())
}

#[tokio::test]
async fn pod_rm_leaves_a_run_still_starting_its_pod_alone_unless_forced() -> TestResult {
    // A preparing run, and a running run whose last create call has no pod yet.
    for (state, answered) in [(RunState::Preparing, true), (RunState::Running, false)] {
        let account = Account::default()
            .with("p1", Some(RUN), true)
            .with("s1", Some(RUN), true)
            .with("x2", Some(RUN), true);
        let harness = Harness::new(account).await?;
        harness.recorded(RUN, state, "p1", PodState::Running)?;
        harness.strays(RUN, &["s1"])?;
        if !answered {
            let mut record = harness.pod_json(RUN)?;
            record.begin_attempt("NVIDIA A40", SystemTime::now(), 6.0);
            record.save(&harness.runs)?;
        }
        let refused = remove_run_pods(&harness.ctx(), RUN, false).await;
        let error = refused.result.err().ok_or("expected a refusal")?;
        assert!(matches!(error, PodError::StillStarting(_)), "{error}");
        assert_eq!(
            error.to_string(),
            format!(
                "run {RUN} is still starting its pod; wait for it, or use `overbrainer train cancel {RUN}` once it runs, or `pod rm {RUN} --force`"
            )
        );
        assert_eq!(refused.removed, [] as [overbrainer::runpod::Removed; 0]);
        assert_eq!(harness.deletes().await, [] as [String; 0]);
        assert_eq!(harness.pod_json(RUN)?.stray_pods.len(), 1);
        assert_eq!(harness.runs.load(RUN)?.state, state);

        let forced = remove_run_pods(&harness.ctx(), RUN, true).await;
        forced.result?;
        assert_eq!(harness.deletes().await, vec!["p1", "s1", "x2"]);
        assert_eq!(harness.runs.load(RUN)?.state, RunState::Failed);
    }
    Ok(())
}

#[tokio::test]
async fn pod_rm_leaves_a_running_run_whose_pod_is_not_recorded_alone() -> TestResult {
    // No pod.json at all, then a pod.json without any pod.
    for with_record in [false, true] {
        let harness = Harness::new(Account::default().with("p1", Some(RUN), true)).await?;
        harness.recorded(RUN, RunState::Running, "p1", PodState::Running)?;
        let path = harness.runs.run_dir(RUN)?.join("pod.json");
        std::fs::remove_file(&path)?;
        if with_record {
            PodRecord::new(RUN, false, 1, "ssh-ed25519 AAAA").save(&harness.runs)?;
        }
        let refused = remove_run_pods(&harness.ctx(), RUN, false).await;
        let error = refused.result.err().ok_or("expected a refusal")?;
        assert!(matches!(error, PodError::PodNotRecorded(_)), "{error}");
        assert_eq!(
            error.to_string(),
            format!(
                "run {RUN} is in progress but its pod is not recorded, so its pods are left alone; if the run is really dead, use `overbrainer pod rm {RUN} --force`"
            )
        );
        assert_eq!(refused.removed, [] as [overbrainer::runpod::Removed; 0]);
        assert_eq!(harness.deletes().await, [] as [String; 0]);
        assert_eq!(harness.runs.load(RUN)?.state, RunState::Running);

        remove_run_pods(&harness.ctx(), RUN, true).await.result?;
        assert_eq!(harness.deletes().await, vec!["p1"]);
        assert_eq!(harness.runs.load(RUN)?.state, RunState::Failed);
    }
    Ok(())
}

#[tokio::test]
async fn a_forced_pod_rm_of_a_running_run_deletes_everything_and_fails_it() -> TestResult {
    let account = Account::default()
        .with("p1", Some(RUN), true)
        .with("x2", Some(RUN), true);
    let harness = Harness::new(account).await?;
    harness.secrets.hold("k1", &host_key_secret(RUN));
    harness.recorded(RUN, RunState::Running, "p1", PodState::Running)?;
    let removal = remove_run_pods(&harness.ctx(), RUN, true).await;
    removal.result?;
    assert_eq!(harness.deletes().await, vec!["p1", "x2"]);
    assert_eq!(harness.secrets.deleted(), [host_key_secret(RUN)]);
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
    assert_eq!(refused.removed, [] as [overbrainer::runpod::Removed; 0]);
    assert_eq!(harness.deletes().await, [] as [String; 0]);
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

/// The startup sweep deletes the host key secret of an ended run of this
/// project without a pod, keeps those a pod may still need, and only warns
/// about a run of another checkout.
#[tokio::test]
async fn the_sweep_deletes_only_the_secrets_no_pod_needs() -> TestResult {
    let account = Account::default().with("p1", Some(RUN), true);
    let harness = Harness::new(account).await?;
    harness.recorded(RUN, RunState::Succeeded, "p1", PodState::Running)?;
    harness.recorded(ENDED, RunState::Failed, "p2", PodState::Deleted)?;
    harness.recorded(
        "20260923-100000-beef",
        RunState::Running,
        "p3",
        PodState::Running,
    )?;
    for (id, run) in [
        ("k1", RUN),
        ("k2", ENDED),
        ("k3", "20260923-100000-beef"),
        ("k4", ELSEWHERE),
    ] {
        harness.secrets.hold(id, &host_key_secret(run));
    }
    harness.secrets.hold("k5", "hf-token");
    let rows = listed_rows(&harness.ctx()).await?;
    let warnings = sweep_host_keys(&harness.ctx(), &rows).await?;
    assert_eq!(harness.secrets.deleted(), [host_key_secret(ENDED)]);
    assert_eq!(
        warnings,
        [format!(
            "the Runpod secret {} holds the pod host key of run {ELSEWHERE}, which is not in this project's runs/; if no other checkout owns it, `overbrainer pod rm {ELSEWHERE} --force`",
            host_key_secret(ELSEWHERE)
        )]
    );
    assert_eq!(harness.deletes().await, Vec::<String>::new());
    Ok(())
}

/// The private client key of the run `id`, written so its removal shows.
fn client_key(
    harness: &Harness,
    id: &str,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let key = harness.runs.run_dir(id)?.join("ssh/id_ed25519");
    std::fs::create_dir_all(key.parent().ok_or("no ssh dir")?)?;
    std::fs::write(&key, "private")?;
    Ok(key)
}

/// A run whose recorded pod is confirmed gone keeps its keys while a stray of
/// it remains, since a restarted stray boots with the same secret; `pod rm`
/// deleting the stray then forgets them.
#[tokio::test]
async fn a_stray_keeps_the_keys_of_a_run_whose_pod_is_gone_until_pod_rm() -> TestResult {
    let account = Account::default().with("s1", Some(RUN), true);
    let harness = Harness::new(account).await?;
    harness.recorded(RUN, RunState::Failed, "p1", PodState::Running)?;
    harness.strays(RUN, &["s1"])?;
    harness.secrets.hold("k1", &host_key_secret(RUN));
    let key = client_key(&harness, RUN)?;
    pod_rows(&harness.ctx()).await?;
    assert_eq!(harness.pod_json(RUN)?.state, PodState::Deleted);
    assert_eq!(harness.secrets.names(), [host_key_secret(RUN)]);
    assert_eq!(harness.secrets.deleted(), Vec::<String>::new());
    assert!(key.is_file());
    let removal = remove_run_pods(&harness.ctx(), RUN, false).await;
    removal.result?;
    assert_eq!(harness.deletes().await, vec!["s1"]);
    assert_eq!(harness.secrets.deleted(), [host_key_secret(RUN)]);
    assert!(!key.exists());
    Ok(())
}

/// The startup sweep keeps the secret of an ended run while a stray of it is
/// listed, and deletes it once that stray is gone.
#[tokio::test]
async fn the_sweep_deletes_the_secret_once_the_last_stray_is_gone() -> TestResult {
    let account = Account::default().with("s1", Some(ENDED), true);
    let harness = Harness::new(account).await?;
    harness.recorded(ENDED, RunState::Failed, "p2", PodState::Deleted)?;
    harness.strays(ENDED, &["s1"])?;
    harness.secrets.hold("k2", &host_key_secret(ENDED));
    let rows = listed_rows(&harness.ctx()).await?;
    sweep_host_keys(&harness.ctx(), &rows).await?;
    assert_eq!(harness.secrets.deleted(), Vec::<String>::new());
    // Deleted from elsewhere (its watchdog, the console).
    harness.client.delete_pod(&PodId::new("s1")?).await?;
    let rows = listed_rows(&harness.ctx()).await?;
    sweep_host_keys(&harness.ctx(), &rows).await?;
    assert_eq!(harness.secrets.deleted(), [host_key_secret(ENDED)]);
    Ok(())
}

const EXPORT: &str = "export_20261001-120000";
const GONE_EXPORT: &str = "export_20261001-130000";

/// The export `id` of `RUN`, in `state`, whose `pod.json` (in the export's own
/// directory) records `pod_id` in `pod_state`.
fn export_recorded(
    exports: &Runs,
    id: &str,
    state: RunState,
    pod_id: &str,
    pod_state: PodState,
) -> TestResult {
    exports.save(&RunRecord {
        id: id.to_string(),
        target: "gpu_cloud".into(),
        created: "2026-10-01T12:00:00Z".into(),
        remote_dir: format!("/workspace/overbrainer/{id}"),
        job: None,
        state,
        message: None,
        snapshot: None,
        resumed_from: None,
        snapshots: false,
    })?;
    let mut record = PodRecord::new(id, pod_state == PodState::Kept, 1, "ssh-ed25519 AAAA");
    let remote: Pod = serde_json::from_value(pod(pod_id, Some(id)))?;
    record.begin_attempt("NVIDIA A40", SystemTime::now(), 6.0);
    record.created(&remote, AttemptResult::Created, SystemTime::now());
    record.state = pod_state;
    record.save(exports)?;
    Ok(())
}

/// Export pods are listed with their export, confirmed gone in the export's own
/// `pod.json`, and removed by the export's ID through `find_job`.
#[tokio::test]
async fn export_pods_are_listed_with_their_export_and_removed_by_its_id() -> TestResult {
    let harness = Harness::new(Account::default().with("x1", Some(EXPORT), true)).await?;
    harness.recorded(RUN, RunState::Succeeded, "p1", PodState::Deleted)?;
    let exports = harness.runs.exports(RUN)?;
    export_recorded(
        &exports,
        EXPORT,
        RunState::Succeeded,
        "x1",
        PodState::Running,
    )?;
    export_recorded(
        &exports,
        GONE_EXPORT,
        RunState::Running,
        "x2",
        PodState::Running,
    )?;
    let rows = pod_rows(&harness.ctx()).await?;
    let ended = find(&rows, "x1")?;
    assert_eq!((ended.run.as_str(), ended.kind), (EXPORT, RowKind::Ended));
    assert_eq!(
        ended.note,
        format!("export of run {RUN}: succeeded, not deleted")
    );
    assert_eq!(
        orphan_warnings(&rows),
        vec![format!(
            "pod x1 (export of run {RUN}: succeeded, not deleted) is still on Runpod at \
             $0.53/h: remove it with `overbrainer pod rm {EXPORT}`"
        )]
    );
    // The unlisted export pod is confirmed gone in the export's own pod.json.
    let gone = find(&rows, "x2")?;
    assert_eq!((gone.run.as_str(), gone.kind), (GONE_EXPORT, RowKind::Gone));
    let record = PodRecord::load(&exports, GONE_EXPORT)?.ok_or("no export pod.json")?;
    assert_eq!(record.state, PodState::Deleted);
    assert!(PodRecord::load(&harness.runs, GONE_EXPORT)?.is_none());

    // `pod rm <export-id>` works on the export's own runs.
    let found = harness.runs.find_job(EXPORT).ok_or("export not found")?;
    let ctx = PodCtx {
        runs: &found,
        ..harness.ctx()
    };
    let removal = remove_run_pods(&ctx, EXPORT, false).await;
    removal.result?;
    assert_eq!(harness.deletes().await, ["x1"]);
    let record = PodRecord::load(&exports, EXPORT)?.ok_or("no export pod.json")?;
    assert_eq!(record.state, PodState::Deleted);
    Ok(())
}

const SWEPT_EXPORT: &str = "export_20261001-140000";

/// Each export pod's secret is named after its export and goes with it: once
/// its pod is confirmed gone, by the startup sweep once the export ended
/// without a pod, and by `pod rm <export-id>`; a listed pod keeps it, and the
/// run's own secret is never touched by an export.
#[tokio::test]
async fn export_pod_secrets_follow_their_export() -> TestResult {
    let account = Account::default()
        .with("x1", Some(EXPORT), true)
        .with("p1", Some(RUN), false);
    let harness = Harness::new(account).await?;
    harness.recorded(RUN, RunState::Running, "p1", PodState::Running)?;
    let exports = harness.runs.exports(RUN)?;
    export_recorded(
        &exports,
        EXPORT,
        RunState::Succeeded,
        "x1",
        PodState::Running,
    )?;
    export_recorded(
        &exports,
        GONE_EXPORT,
        RunState::Running,
        "x2",
        PodState::Running,
    )?;
    export_recorded(
        &exports,
        SWEPT_EXPORT,
        RunState::Failed,
        "x3",
        PodState::Deleted,
    )?;
    for (id, owner) in [
        ("k0", RUN),
        ("k1", EXPORT),
        ("k2", GONE_EXPORT),
        ("k3", SWEPT_EXPORT),
    ] {
        harness.secrets.hold(id, &host_key_secret(owner));
    }
    // p1 is not listed but answers: the run keeps its secret.
    pod_rows(&harness.ctx()).await?;
    assert_eq!(harness.secrets.deleted(), [host_key_secret(GONE_EXPORT)]);
    let rows = listed_rows(&harness.ctx()).await?;
    let warnings = sweep_host_keys(&harness.ctx(), &rows).await?;
    assert_eq!(warnings, Vec::<String>::new());
    assert_eq!(
        harness.secrets.deleted(),
        [host_key_secret(GONE_EXPORT), host_key_secret(SWEPT_EXPORT)]
    );
    let found = harness.runs.holding(EXPORT);
    let ctx = PodCtx {
        runs: &found,
        ..harness.ctx()
    };
    remove_run_pods(&ctx, EXPORT, false).await.result?;
    assert_eq!(
        harness.secrets.names(),
        [host_key_secret(RUN)],
        "only the run's secret is left"
    );
    Ok(())
}
