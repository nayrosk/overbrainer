//! The disk policy of a Runpod run against a local stub of the API, with the
//! local executor standing in for the pod: a full disk grows the network volume
//! when `max_volume_gb` allows it, and stops the job with a snapshot otherwise,
//! or when the grow does not show.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
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

/// Network volume `vol1`, whose size a PATCH changes only when `grows`.
struct Volume {
    size: Arc<AtomicU32>,
    grows: bool,
}

impl Respond for Volume {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        if request.method.as_str() == "PATCH" {
            let asked = serde_json::from_slice::<serde_json::Value>(&request.body)
                .ok()
                .and_then(|body| body["size"].as_u64())
                .and_then(|size| u32::try_from(size).ok())
                .unwrap_or(0);
            if self.grows {
                self.size.store(asked, Ordering::SeqCst);
            }
            return ResponseTemplate::new(200)
                .set_body_json(json!({"id": "vol1", "name": "data", "size": asked}));
        }
        let size = self.size.load(Ordering::SeqCst);
        ResponseTemplate::new(200)
            .set_body_json(json!({"id": "vol1", "name": "data", "size": size}))
    }
}

struct Setup {
    server: MockServer,
    client: RunpodClient,
    _project: tempfile::TempDir,
    executor: LocalExecutor,
    run: RunRecord,
    size: Arc<AtomicU32>,
}

async fn setup(grows: bool) -> Result<Setup, Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    let size = Arc::new(AtomicU32::new(200));
    Mock::given(path("/v2/network-volumes/vol1"))
        .respond_with(Volume {
            size: Arc::clone(&size),
            grows,
        })
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
    let mut run = create(&runs, executor.workdir(), "gpu_cloud")?;
    std::fs::create_dir_all(&run.remote_dir)?;
    run.job = Some(serde_json::from_value(
        json!({"dir": run.remote_dir, "pid": 424_242}),
    )?);
    run.state = RunState::Running;
    Ok(Setup {
        server,
        client,
        _project: project,
        executor,
        run,
        size,
    })
}

fn volume(max_gb: Option<u32>) -> VolumeDisk<'static> {
    VolumeDisk {
        id: "vol1",
        mount: "/nonexistent-volume",
        max_gb,
    }
}

/// A volume of `size_gb` holding `percent` of what it can hold.
fn full(size_gb: u32, percent: u64) -> Usage {
    Usage::of_volume(u64::from(size_gb) * GB / 100 * 94 / 100 * percent, size_gb)
}

fn read(dir: &str, name: &str) -> String {
    std::fs::read_to_string(Path::new(dir).join(name)).unwrap_or_default()
}

async fn patches(server: &MockServer) -> usize {
    server
        .received_requests()
        .await
        .unwrap_or_default()
        .iter()
        .filter(|request| request.method.as_str() == "PATCH")
        .count()
}

#[tokio::test]
async fn a_full_volume_grows_within_max_volume_gb() -> TestResult {
    let setup = setup(true).await?;
    let mut disk = DiskWatch::new(
        &setup.executor,
        &setup.client,
        &setup.run,
        Some(volume(Some(1000))),
    );
    assert_eq!(disk.read_size().await, Some(200));
    assert_eq!(read(&setup.run.remote_dir, VOLUME_SIZE_FILE), "200");
    disk.check(full(200, 90), None).await;
    assert_eq!(patches(&setup.server).await, 0, "90% only warns");
    disk.check(full(200, 93), None).await;
    assert_eq!(patches(&setup.server).await, 1);
    assert_eq!(setup.size.load(Ordering::SeqCst), 300);
    // The next sample sees the new size: no snapshot, and the watchdog knows it.
    disk.check(full(200, 93), None).await;
    assert_eq!(read(&setup.run.remote_dir, VOLUME_SIZE_FILE), "300");
    assert_eq!(read(&setup.run.remote_dir, SNAPSHOT_REQUEST), "");
    // Full again at the cap: nothing left to grow, the job stops.
    setup.size.store(1000, Ordering::SeqCst);
    assert_eq!(disk.read_size().await, Some(1000));
    disk.check(full(1000, 95), None).await;
    assert_eq!(patches(&setup.server).await, 1);
    assert_eq!(read(&setup.run.remote_dir, SNAPSHOT_REQUEST), "disk");
    Ok(())
}

#[tokio::test]
async fn a_grow_that_does_not_show_stops_the_job_after_two_samples() -> TestResult {
    let setup = setup(false).await?;
    let mut disk = DiskWatch::new(
        &setup.executor,
        &setup.client,
        &setup.run,
        Some(volume(Some(1000))),
    );
    disk.read_size().await;
    disk.check(full(200, 93), None).await;
    assert_eq!(patches(&setup.server).await, 1);
    disk.check(full(200, 93), None).await;
    assert_eq!(read(&setup.run.remote_dir, SNAPSHOT_REQUEST), "");
    disk.check(full(200, 93), None).await;
    assert_eq!(read(&setup.run.remote_dir, SNAPSHOT_REQUEST), "disk");
    assert_eq!(patches(&setup.server).await, 1);
    Ok(())
}

#[tokio::test]
async fn without_max_volume_gb_a_full_volume_stops_the_job() -> TestResult {
    let setup = setup(true).await?;
    let mut disk = DiskWatch::new(
        &setup.executor,
        &setup.client,
        &setup.run,
        Some(volume(None)),
    );
    disk.read_size().await;
    // The next checkpoint would not fit: 10% free of 188 GB, a 15 GB checkpoint.
    disk.check(full(200, 90), Some(15 * GB)).await;
    assert_eq!(read(&setup.run.remote_dir, SNAPSHOT_REQUEST), "disk");
    assert_eq!(patches(&setup.server).await, 0);
    Ok(())
}

#[tokio::test]
async fn a_full_container_disk_sample_stops_the_job() -> TestResult {
    let setup = setup(true).await?;
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
            if !read(&setup.run.remote_dir, SNAPSHOT_REQUEST).is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    };
    tokio::select! {
        never = disk.run(events) => match never {},
        () = publish => {},
    }
    assert_eq!(read(&setup.run.remote_dir, SNAPSHOT_REQUEST), "disk");
    assert_eq!(patches(&setup.server).await, 0);
    Ok(())
}
