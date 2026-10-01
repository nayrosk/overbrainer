//! The disk policy of a Runpod run against a local stub of the API, with the
//! local executor standing in for the pod: a full disk grows the network volume
//! when `max_volume_gb` allows it, and stops the job with a snapshot otherwise,
//! when Runpod refuses the grow, or when the grow does not show in time.

use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, SystemTime};

use overbrainer::events::{Event, EventBus};
use overbrainer::exec::{Executor, LocalExecutor};
use overbrainer::retry::RetryPolicy;
use overbrainer::runpod::{DiskWatch, RunpodClient, Usage, VOLUME_SIZE_FILE, VolumeDisk};
use overbrainer::runs::{RunRecord, RunState, Runs, create};
use overbrainer::system::{Disk, SystemSample};
use overbrainer::train::SNAPSHOT_REQUEST;
use secrecy::SecretString;
use serde_json::json;
use wiremock::matchers::path;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const GB: u64 = 1_000_000_000;

/// What the stub of network volume `vol1` answers.
#[derive(Debug, Clone, Copy)]
struct State {
    size: u32,
    /// Whether a PATCH changes the size (and answers with the new one).
    grows: bool,
    /// Status of a PATCH.
    patch: u16,
    /// Status of a GET.
    get: u16,
}

#[derive(Clone)]
struct Volume(Arc<Mutex<State>>);

impl Volume {
    fn with(&self, change: impl FnOnce(&mut State)) {
        change(&mut self.0.lock().unwrap_or_else(PoisonError::into_inner));
    }

    fn state(&self) -> State {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Respond for Volume {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let mut state = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let status = if request.method.as_str() == "PATCH" {
            state.patch
        } else {
            state.get
        };
        if status != 200 {
            return ResponseTemplate::new(status).set_body_json(json!({"error": "no"}));
        }
        if request.method.as_str() == "PATCH" && state.grows {
            state.size = serde_json::from_slice::<serde_json::Value>(&request.body)
                .ok()
                .and_then(|body| body["size"].as_u64())
                .and_then(|size| u32::try_from(size).ok())
                .unwrap_or(0);
        }
        ResponseTemplate::new(200)
            .set_body_json(json!({"id": "vol1", "name": "data", "size": state.size}))
    }
}

struct Setup {
    server: MockServer,
    client: RunpodClient,
    _project: tempfile::TempDir,
    executor: LocalExecutor,
    run: RunRecord,
    volume: Volume,
}

impl Setup {
    async fn new(grows: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        let volume = Volume(Arc::new(Mutex::new(State {
            size: 200,
            grows,
            patch: 200,
            get: 200,
        })));
        Mock::given(path("/v2/network-volumes/vol1"))
            .respond_with(volume.clone())
            .mount(&server)
            .await;
        let client = RunpodClient::new(&format!("{}/v2", server.uri()), &SecretString::from("k"))?
            .with_policy(RetryPolicy {
                max_retries: 0,
                base: Duration::from_millis(1),
                cap: Duration::from_millis(2),
            });
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let executor = LocalExecutor::new(&project.path().join("pod"))?;
        let mut run = create(&runs, "demo", executor.workdir(), "gpu_cloud")?;
        std::fs::create_dir_all(Path::new(&run.remote_dir).join(".pod"))?;
        run.job = Some(serde_json::from_value(
            json!({"dir": run.remote_dir, "pid": 424_242}),
        )?);
        run.state = RunState::Running;
        run.snapshots = true;
        Ok(Self {
            server,
            client,
            _project: project,
            executor,
            run,
            volume,
        })
    }

    fn watch(&self, max_gb: Option<u32>) -> DiskWatch<'_, LocalExecutor> {
        DiskWatch::new(
            &self.executor,
            &self.client,
            &self.run,
            Some(VolumeDisk {
                id: "vol1",
                mount: "/nonexistent-volume",
                max_gb,
            }),
        )
    }

    fn read(&self, name: &str) -> String {
        std::fs::read_to_string(Path::new(&self.run.remote_dir).join(name)).unwrap_or_default()
    }

    async fn patches(&self) -> usize {
        self.server
            .received_requests()
            .await
            .unwrap_or_default()
            .iter()
            .filter(|request| request.method.as_str() == "PATCH")
            .count()
    }
}

/// A volume of `size_gb` holding `percent` of what it can hold.
fn full(size_gb: u32, percent: u64) -> Usage {
    Usage::of_volume(u64::from(size_gb) * GB / 100 * 94 / 100 * percent, size_gb)
}

#[tokio::test]
async fn a_full_volume_grows_within_max_volume_gb() -> TestResult {
    let setup = Setup::new(true).await?;
    let mut disk = setup.watch(Some(1000));
    assert_eq!(disk.read_size().await, Some(200));
    assert_eq!(setup.read(VOLUME_SIZE_FILE), "200");
    disk.check(full(200, 90), None).await;
    assert_eq!(setup.patches().await, 0, "90% only warns");
    disk.check(full(200, 93), None).await;
    assert_eq!(setup.patches().await, 1);
    assert_eq!(setup.volume.state().size, 300);
    // Runpod's answer shows the new size: it counts at once, for the watchdog too.
    assert_eq!(setup.read(VOLUME_SIZE_FILE), "300");
    disk.check(full(300, 50), None).await;
    assert_eq!(setup.read(SNAPSHOT_REQUEST), "");
    Ok(())
}

#[tokio::test]
async fn repeated_grows_reach_the_cap_then_the_job_stops() -> TestResult {
    let setup = Setup::new(true).await?;
    let mut disk = setup.watch(Some(400));
    disk.read_size().await;
    disk.check(full(200, 93), None).await;
    disk.check(full(300, 93), None).await;
    assert_eq!(setup.volume.state().size, 400);
    assert_eq!(setup.read(SNAPSHOT_REQUEST), "");
    disk.check(full(400, 93), None).await;
    assert_eq!(setup.patches().await, 2);
    assert_eq!(setup.read(SNAPSHOT_REQUEST), "disk");
    Ok(())
}

#[tokio::test]
async fn a_checkpoint_that_would_not_fit_grows_the_volume() -> TestResult {
    let setup = Setup::new(true).await?;
    let mut disk = setup.watch(Some(1000));
    disk.read_size().await;
    // 60% used of 188 GB leaves 75 GB: a 60 GB checkpoint needs 90.
    disk.check(full(200, 60), Some(60 * GB)).await;
    assert_eq!(setup.patches().await, 1);
    assert_eq!(setup.read(SNAPSHOT_REQUEST), "");
    Ok(())
}

#[tokio::test]
async fn a_grow_that_does_not_show_stops_the_job() -> TestResult {
    let setup = Setup::new(false).await?;
    let mut disk = setup.watch(Some(1000)).with_grow_window(Duration::ZERO);
    disk.read_size().await;
    disk.check(full(200, 93), None).await;
    assert_eq!(setup.patches().await, 1);
    // The size asked for reaches the watchdog before any sample, so its own
    // rule never races the grow.
    assert_eq!(setup.read(VOLUME_SIZE_FILE), "300");
    assert_eq!(setup.read(SNAPSHOT_REQUEST), "");
    disk.check(full(200, 93), None).await;
    assert_eq!(setup.read(SNAPSHOT_REQUEST), "disk");
    assert_eq!(setup.patches().await, 1);
    Ok(())
}

#[tokio::test]
async fn a_failed_read_after_a_grow_is_not_a_failed_grow() -> TestResult {
    let setup = Setup::new(false).await?;
    let mut disk = setup
        .watch(Some(1000))
        .with_grow_window(Duration::from_millis(300));
    disk.read_size().await;
    disk.check(full(200, 93), None).await;
    setup.volume.with(|state| state.get = 500);
    tokio::time::sleep(Duration::from_millis(400)).await;
    // Past the window, but no read succeeded: still waiting.
    disk.check(full(200, 93), None).await;
    disk.check(full(200, 93), None).await;
    assert_eq!(setup.read(SNAPSHOT_REQUEST), "");
    // A read that works and shows the old size ends the wait.
    setup.volume.with(|state| state.get = 200);
    disk.check(full(200, 93), None).await;
    assert_eq!(setup.read(SNAPSHOT_REQUEST), "disk");
    Ok(())
}

#[tokio::test]
async fn a_refused_grow_is_asked_once_then_the_job_stops() -> TestResult {
    for status in [400, 401, 403] {
        let setup = Setup::new(true).await?;
        setup.volume.with(|state| state.patch = status);
        let mut disk = setup.watch(Some(1000));
        disk.read_size().await;
        // The request cannot be written at first: a directory is in its way.
        let request = Path::new(&setup.run.remote_dir).join(SNAPSHOT_REQUEST);
        std::fs::create_dir_all(&request)?;
        disk.check(full(200, 93), None).await;
        disk.check(full(200, 93), None).await;
        std::fs::remove_dir(&request)?;
        disk.check(full(200, 93), None).await;
        assert_eq!(setup.patches().await, 1, "{status}");
        assert_eq!(setup.read(SNAPSHOT_REQUEST), "disk", "{status}");
    }
    Ok(())
}

#[tokio::test]
async fn a_volume_resized_elsewhere_is_weighed_again() -> TestResult {
    let setup = Setup::new(true).await?;
    let mut disk = setup.watch(Some(1000));
    disk.read_size().await;
    setup.volume.with(|state| state.size = 1000);
    disk.check(full(200, 93), None).await;
    assert_eq!(setup.patches().await, 0);
    assert_eq!(setup.read(SNAPSHOT_REQUEST), "");
    assert_eq!(setup.read(VOLUME_SIZE_FILE), "1000");
    Ok(())
}

#[tokio::test]
async fn a_size_of_zero_is_no_size() -> TestResult {
    let setup = Setup::new(true).await?;
    setup.volume.with(|state| state.size = 0);
    let mut disk = setup.watch(Some(1000));
    assert_eq!(disk.read_size().await, None);
    assert_eq!(setup.read(VOLUME_SIZE_FILE), "");
    Ok(())
}

#[tokio::test]
async fn without_max_volume_gb_a_full_volume_stops_the_job() -> TestResult {
    let setup = Setup::new(true).await?;
    let mut disk = setup.watch(None);
    disk.read_size().await;
    // The next checkpoint would not fit: 10% free of 188 GB, a 15 GB checkpoint.
    disk.check(full(200, 90), Some(15 * GB)).await;
    assert_eq!(setup.read(SNAPSHOT_REQUEST), "disk");
    assert_eq!(setup.patches().await, 0);
    Ok(())
}

#[tokio::test]
async fn an_earlier_snapshot_request_keeps_its_reason() -> TestResult {
    let setup = Setup::new(true).await?;
    std::fs::write(
        Path::new(&setup.run.remote_dir).join(SNAPSHOT_REQUEST),
        "cost",
    )?;
    let mut disk = setup.watch(None);
    disk.read_size().await;
    disk.check(full(200, 95), None).await;
    assert_eq!(setup.read(SNAPSHOT_REQUEST), "cost");
    Ok(())
}

#[tokio::test]
async fn the_full_disk_of_a_run_started_before_snapshots_only_warns() -> TestResult {
    let mut setup = Setup::new(true).await?;
    // Started by overbrainer 0.4.1: its job ignores a snapshot request.
    setup.run.snapshots = false;
    let mut disk = setup.watch(Some(300));
    disk.read_size().await;
    disk.check(full(200, 93), None).await;
    assert_eq!(setup.volume.state().size, 300, "a grow is still made");
    disk.check(full(300, 93), None).await;
    disk.check(full(300, 99), None).await;
    assert_eq!(setup.read(SNAPSHOT_REQUEST), "");
    Ok(())
}

#[tokio::test]
async fn an_idle_policy_does_nothing() -> TestResult {
    let setup = Setup::new(true).await?;
    let mut disk = setup.watch(Some(1000)).idle();
    disk.check(full(200, 99), None).await;
    assert_eq!(setup.patches().await, 0);
    assert_eq!(setup.read(SNAPSHOT_REQUEST), "");
    Ok(())
}

#[tokio::test]
async fn a_full_container_disk_sample_stops_the_job() -> TestResult {
    let setup = Setup::new(true).await?;
    let bus = EventBus::new();
    let disk = DiskWatch::new(&setup.executor, &setup.client, &setup.run, None);
    let events = bus.subscribe();
    let sample = |used: u64| SystemSample {
        at: SystemTime::now(),
        disks: vec![Disk {
            mount: "/".into(),
            fstype: None,
            shared: false,
            size_bytes: 100 * GB,
            used_bytes: used * GB,
            available_bytes: (100 - used) * GB,
        }],
        cpu: None,
        memory: None,
        gpus: Vec::new(),
    };
    let publish = async {
        bus.publish(Event::System(sample(50)));
        bus.publish(Event::System(sample(95)));
        for _ in 0..100 {
            if !setup.read(SNAPSHOT_REQUEST).is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    };
    tokio::select! {
        never = disk.run(events) => match never {},
        () = publish => {},
    }
    assert_eq!(setup.read(SNAPSHOT_REQUEST), "disk");
    assert_eq!(setup.patches().await, 0);
    Ok(())
}

#[tokio::test]
async fn a_shared_file_system_is_never_judged_by_df() -> TestResult {
    let setup = Setup::new(true).await?;
    let bus = EventBus::new();
    let disk = DiskWatch::new(&setup.executor, &setup.client, &setup.run, None);
    let events = bus.subscribe();
    // A network volume's `df` shows the whole cluster: 99% says nothing of
    // what this run may still write.
    let sample = SystemSample {
        at: SystemTime::now(),
        disks: vec![Disk {
            mount: "/workspace/data".into(),
            fstype: Some("fuse.mfs".into()),
            shared: true,
            size_bytes: 100 * GB,
            used_bytes: 99 * GB,
            available_bytes: GB,
        }],
        cpu: None,
        memory: None,
        gpus: Vec::new(),
    };
    let publish = async {
        bus.publish(Event::System(sample));
        tokio::time::sleep(Duration::from_millis(300)).await;
    };
    tokio::select! {
        never = disk.run(events) => match never {},
        () = publish => {},
    }
    assert_eq!(setup.read(SNAPSHOT_REQUEST), "");
    Ok(())
}
