//! The train, watch and cancel flows on the local executor, with a fake `axolotl`
//! that writes metric lines and an adapter.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::time::Duration;

use overbrainer::config::{EnvSource, Settings, load};
use overbrainer::dataset::DataFiles;
use overbrainer::events::{Event, EventBus};
use overbrainer::exec::{Executor, JobId, JobRuntime, JobStatus, LocalExecutor};
use overbrainer::runs::{Launch, RunCtx, RunState, Runs, cancel, create, start, watch};
use overbrainer::train::Axolotl;
use secrecy::SecretString;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const PROJECT: &str = r#"
[project]
name = "demo"

[providers.mock]
protocol = "openai"

[roles]
generator = { provider = "mock", model = "gen" }
parent = { provider = "mock", model = "parent" }

[training]
target = "box"
base_model = "Qwen/Qwen3-4B"
adapter = "qlora"
merge = true

[targets.box]
kind = "local"
runtime = "native"
"#;

const FAKE_AXOLOTL: &str = r#"#!/bin/sh
mode=$(cat "$(dirname "$0")/../mode")
case "$1:$mode" in
  train:fail) echo "boom" >&2; exit 1 ;;
  train:silent) exit 0 ;;
  train:slow) sleep 60 ;;
  train:*)
    sleep 1
    printf '{"event": "begin", "time": 1, "max_steps": 2}\n' >> "$OVERBRAINER_METRICS"
    printf '{"event": "log", "time": 2, "step": 1, "epoch": 0.5, "max_steps": 2, "loss": 1.5, "learning_rate": 0.0002}\n' >> "$OVERBRAINER_METRICS"
    printf '{"event": "log", "time": 3, "step": 2, "epoch": 1.0, "max_steps": 2, "eval_loss": 1.25}\n' >> "$OVERBRAINER_METRICS"
    mkdir -p output/checkpoint-2
    echo adapter > output/adapter_model.safetensors
    if [ -n "$HF_TOKEN" ]; then echo "token present"; fi
    ;;
  merge-lora:*) mkdir -p output/merged && echo merged > output/merged/model.safetensors ;;
esac
"#;

struct Fixture {
    dir: tempfile::TempDir,
    settings: Settings,
    runtime: JobRuntime,
}

impl Fixture {
    fn new(mode: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        fs::write(dir.path().join("overbrainer.toml"), PROJECT)?;
        fs::create_dir_all(dir.path().join("data"))?;
        fs::write(dir.path().join("data/train.jsonl"), "{\"id\": 1}\n")?;
        fs::write(dir.path().join("data/eval.jsonl"), "{\"id\": 2}\n")?;
        let venv = dir.path().join("venv");
        fs::create_dir_all(venv.join("bin"))?;
        let axolotl = venv.join("bin/axolotl");
        fs::write(&axolotl, FAKE_AXOLOTL)?;
        fs::set_permissions(&axolotl, fs::Permissions::from_mode(0o755))?;
        fs::write(venv.join("mode"), mode)?;
        let settings = load(dir.path(), EnvSource::Vars(Vec::new()))?;
        let runtime = JobRuntime::Native {
            venv: Some(venv.to_string_lossy().into_owned()),
        };
        Ok(Self {
            dir,
            settings,
            runtime,
        })
    }

    fn project(&self) -> &Path {
        self.dir.path()
    }
}

fn secrets() -> Vec<(String, SecretString)> {
    vec![(
        "HF_TOKEN".to_string(),
        SecretString::from("hf_fake_token_42"),
    )]
}

fn events(receiver: &mut tokio::sync::broadcast::Receiver<Event>) -> Vec<Event> {
    let mut events = Vec::new();
    while let Ok(event) = receiver.try_recv() {
        events.push(event);
    }
    events
}

/// Polls `job` until it has ended, for up to 10 seconds.
async fn wait_until_ended(
    executor: &LocalExecutor,
    job: &JobId,
) -> Result<JobStatus, Box<dyn std::error::Error>> {
    for _ in 0..200 {
        let status = executor.status(job).await?;
        if status.is_finished() {
            return Ok(status);
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    Err("the job did not end".into())
}

#[tokio::test]
async fn a_run_trains_merges_and_records_its_outcome() -> TestResult {
    let fixture = Fixture::new("ok")?;
    let runs = Runs::new(fixture.project());
    let executor = LocalExecutor::new(runs.dir())?;
    let bus = EventBus::new();
    let mut receiver = bus.subscribe();
    let ctx = RunCtx {
        runs: &runs,
        executor: &executor,
        bus: &bus,
        poll: Duration::from_millis(50),
    };
    let training = fixture.settings.training.as_ref().ok_or("training")?;
    let trainer = Axolotl::new(training, &DataFiles::new(fixture.project()));
    let created = create(&runs, &executor, "box")?;
    assert_eq!(created.state, RunState::Preparing);
    let launch = Launch {
        runtime: &fixture.runtime,
        secrets: secrets(),
    };
    let record = start(&ctx, &trainer, launch, created).await?;
    assert_eq!(record.state, RunState::Running);
    assert_eq!(runs.load(&record.id)?.state, RunState::Running);

    let outcome = watch(&ctx, &trainer, record).await?;
    assert_eq!(
        outcome.record.state,
        RunState::Succeeded,
        "{:?}",
        outcome.record.message
    );
    assert_eq!(outcome.summary.step, 2);
    assert_eq!(outcome.summary.eval_loss, Some(1.25));
    let run = runs.run_dir(&outcome.record.id)?;
    assert!(run.join("output/adapter_model.safetensors").is_file());
    assert!(run.join("output/merged/model.safetensors").is_file());
    assert_eq!(fs::read_to_string(run.join("job.log"))?, "token present\n");
    assert_eq!(runs.load(&outcome.record.id)?, outcome.record);

    for entry in walk(&run)? {
        let content = fs::read(&entry)?;
        assert!(
            !String::from_utf8_lossy(&content).contains("hf_fake_token_42"),
            "the token was written to {}",
            entry.display()
        );
    }

    let events = events(&mut receiver);
    let running = events
        .iter()
        .filter(|event| **event == Event::JobStatus(JobStatus::Running))
        .count();
    assert_eq!(running, 1, "{events:?}");
    assert!(events.contains(&Event::JobStatus(JobStatus::Exited(0))));
    let steps: Vec<u64> = events
        .iter()
        .filter_map(|event| match event {
            Event::Metric(metric) => Some(metric.step),
            _ => None,
        })
        .collect();
    assert_eq!(steps, vec![1, 2]);

    // Watching an ended run again only reads the local files.
    let again = watch(&ctx, &trainer, outcome.record.clone()).await?;
    assert_eq!(again.summary.step, 2);
    Ok(())
}

fn walk(dir: &Path) -> std::io::Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            files.extend(walk(&path)?);
        } else {
            files.push(path);
        }
    }
    Ok(files)
}

async fn run_to_end(mode: &str) -> Result<overbrainer::runs::Outcome, Box<dyn std::error::Error>> {
    let fixture = Fixture::new(mode)?;
    let runs = Runs::new(fixture.project());
    let executor = LocalExecutor::new(runs.dir())?;
    let bus = EventBus::new();
    let ctx = RunCtx {
        runs: &runs,
        executor: &executor,
        bus: &bus,
        poll: Duration::from_millis(50),
    };
    let training = fixture.settings.training.as_ref().ok_or("training")?;
    let trainer = Axolotl::new(training, &DataFiles::new(fixture.project()));
    let launch = Launch {
        runtime: &fixture.runtime,
        secrets: Vec::new(),
    };
    let record = start(&ctx, &trainer, launch, create(&runs, &executor, "box")?).await?;
    Ok(watch(&ctx, &trainer, record).await?)
}

#[tokio::test]
async fn a_failing_job_fails_the_run() -> TestResult {
    let outcome = run_to_end("fail").await?;
    assert_eq!(outcome.record.state, RunState::Failed);
    let message = outcome.record.message.unwrap_or_default();
    assert!(message.contains("exited with code 1"), "{message}");
    Ok(())
}

#[tokio::test]
async fn a_job_without_metrics_fails_the_run() -> TestResult {
    let outcome = run_to_end("silent").await?;
    assert_eq!(outcome.record.state, RunState::Failed);
    let message = outcome.record.message.unwrap_or_default();
    assert!(message.contains("plugin was not loaded"), "{message}");
    Ok(())
}

#[tokio::test]
async fn a_running_job_can_be_cancelled() -> TestResult {
    let fixture = Fixture::new("slow")?;
    let runs = Runs::new(fixture.project());
    // A work directory apart from `runs/`, so the cancel's download is seen.
    let workdir = tempfile::tempdir()?;
    let executor = LocalExecutor::new(workdir.path())?;
    let bus = EventBus::new();
    let ctx = RunCtx {
        runs: &runs,
        executor: &executor,
        bus: &bus,
        poll: Duration::from_millis(50),
    };
    let training = fixture.settings.training.as_ref().ok_or("training")?;
    let trainer = Axolotl::new(training, &DataFiles::new(fixture.project()));
    let launch = Launch {
        runtime: &fixture.runtime,
        secrets: Vec::new(),
    };
    let record = start(&ctx, &trainer, launch, create(&runs, &executor, "box")?).await?;
    let local_log = runs.run_dir(&record.id)?.join("job.log");
    assert!(!local_log.exists());
    let (cancelled, status) = cancel(&runs, &executor, &trainer, record.clone()).await?;
    assert_eq!(status, JobStatus::Cancelled);
    assert_eq!(cancelled.state, RunState::Cancelled);
    assert_eq!(cancelled.message, None);
    assert_eq!(runs.load(&record.id)?.state, RunState::Cancelled);
    assert!(local_log.is_file(), "the job log was not retrieved");
    // A watch started before the cancel sees the job end as cancelled.
    let outcome = watch(&ctx, &trainer, record).await?;
    assert_eq!(outcome.record.state, RunState::Cancelled);
    Ok(())
}

#[tokio::test]
async fn cancelling_an_ended_job_leaves_the_run_running() -> TestResult {
    let fixture = Fixture::new("silent")?;
    let runs = Runs::new(fixture.project());
    let executor = LocalExecutor::new(runs.dir())?;
    let bus = EventBus::new();
    let ctx = RunCtx {
        runs: &runs,
        executor: &executor,
        bus: &bus,
        poll: Duration::from_millis(50),
    };
    let training = fixture.settings.training.as_ref().ok_or("training")?;
    let trainer = Axolotl::new(training, &DataFiles::new(fixture.project()));
    let launch = Launch {
        runtime: &fixture.runtime,
        secrets: Vec::new(),
    };
    let record = start(&ctx, &trainer, launch, create(&runs, &executor, "box")?).await?;
    let job = record.job.clone().ok_or("no job")?;
    assert_eq!(
        wait_until_ended(&executor, &job).await?,
        JobStatus::Exited(0)
    );
    let (unchanged, status) = cancel(&runs, &executor, &trainer, record.clone()).await?;
    assert_eq!(status, JobStatus::Exited(0));
    assert_eq!(unchanged, record);
    assert_eq!(runs.load(&record.id)?.state, RunState::Running);
    // A later watch records the real outcome.
    let outcome = watch(&ctx, &trainer, record).await?;
    assert_eq!(outcome.record.state, RunState::Failed);
    Ok(())
}

#[tokio::test]
async fn a_run_whose_directory_vanished_records_why_nothing_was_retrieved() -> TestResult {
    let fixture = Fixture::new("silent")?;
    let runs = Runs::new(fixture.project());
    let executor = LocalExecutor::new(runs.dir())?;
    let bus = EventBus::new();
    let ctx = RunCtx {
        runs: &runs,
        executor: &executor,
        bus: &bus,
        poll: Duration::from_millis(50),
    };
    let training = fixture.settings.training.as_ref().ok_or("training")?;
    let trainer = Axolotl::new(training, &DataFiles::new(fixture.project()));
    let launch = Launch {
        runtime: &fixture.runtime,
        secrets: Vec::new(),
    };
    let record = start(&ctx, &trainer, launch, create(&runs, &executor, "box")?).await?;
    let job = record.job.clone().ok_or("no job")?;
    wait_until_ended(&executor, &job).await?;
    // On the local executor the remote run directory is the local one.
    fs::remove_dir_all(&record.remote_dir)?;
    assert_eq!(executor.status(&job).await?, JobStatus::Lost);

    let outcome = watch(&ctx, &trainer, record).await?;
    assert_eq!(outcome.record.state, RunState::Failed);
    let message = outcome.record.message.clone().unwrap_or_default();
    assert!(
        message.contains("stopped without an exit code"),
        "{message}"
    );
    assert!(
        message.ends_with(" does not exist)")
            && message.contains(" (artifacts not retrieved: download failed: "),
        "{message}"
    );
    assert_eq!(runs.load(&outcome.record.id)?, outcome.record);
    Ok(())
}

#[tokio::test]
async fn a_start_without_data_is_recorded_as_failed() -> TestResult {
    let fixture = Fixture::new("ok")?;
    fs::remove_file(fixture.project().join("data/train.jsonl"))?;
    let runs = Runs::new(fixture.project());
    let executor = LocalExecutor::new(runs.dir())?;
    let bus = EventBus::new();
    let ctx = RunCtx {
        runs: &runs,
        executor: &executor,
        bus: &bus,
        poll: Duration::from_millis(50),
    };
    let training = fixture.settings.training.as_ref().ok_or("training")?;
    let trainer = Axolotl::new(training, &DataFiles::new(fixture.project()));
    let launch = Launch {
        runtime: &fixture.runtime,
        secrets: Vec::new(),
    };
    let error = start(&ctx, &trainer, launch, create(&runs, &executor, "box")?)
        .await
        .err()
        .ok_or("start succeeded")?;
    assert!(
        error.to_string().contains("run `overbrainer split` first"),
        "{error}"
    );
    let listed = runs.list()?;
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].state, RunState::Failed);
    Ok(())
}
