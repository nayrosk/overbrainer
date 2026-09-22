use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

use overbrainer::exec::{
    CANCEL_FILE, CANCELLING_FILE, Executor, JobCommand, JobId, JobStatus, LocalExecutor,
};
use secrecy::SecretString;

type TestResult = Result<(), Box<dyn std::error::Error>>;

async fn wait_finished(
    executor: &LocalExecutor,
    job: &JobId,
) -> Result<JobStatus, Box<dyn std::error::Error>> {
    for _ in 0..150 {
        let status = executor.status(job).await?;
        if status.is_finished() {
            return Ok(status);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Err("the job did not finish".into())
}

fn job(dir: &Path, script: &str) -> JobCommand {
    JobCommand {
        dir: dir.to_string_lossy().into_owned(),
        script: script.to_string(),
        secrets: vec![(
            "OB_TEST_SECRET".to_string(),
            SecretString::from("s3cr3t-value"),
        )],
        container: None,
    }
}

/// Whether a process with this ID exists, asked through `sh` so the test needs no
/// signal crate.
fn process_exists(pid: u32) -> Result<bool, Box<dyn std::error::Error>> {
    Ok(Command::new("sh")
        .arg("-c")
        .arg(format!("kill -s 0 {pid} 2>/dev/null"))
        .status()?
        .success())
}

#[tokio::test]
async fn a_job_runs_detached_and_reports_its_exit_code() -> TestResult {
    let root = tempfile::tempdir()?;
    let executor = LocalExecutor::new(root.path())?;
    let dir = Path::new(executor.workdir()).join("r1");
    let started = executor
        .spawn(&job(
            &dir,
            "printf 'one\\ntwo\\n' >> metrics.jsonl; [ \"$OB_TEST_SECRET\" = s3cr3t-value ] && echo secret-seen; exit 3",
        ))
        .await?;
    assert_eq!(
        wait_finished(&executor, &started).await?,
        JobStatus::Exited(3)
    );
    assert_eq!(fs::read_to_string(dir.join("job.log"))?, "secret-seen\n");
    assert_eq!(
        fs::read_to_string(dir.join("job.pid"))?.trim(),
        started.pid.get().to_string()
    );

    let metrics = dir.join("metrics.jsonl").to_string_lossy().into_owned();
    let mut tail = executor.tail(&metrics, 0);
    assert_eq!(
        tail.read().await?,
        vec!["one".to_string(), "two".to_string()]
    );
    assert_eq!(tail.offset(), 8);
    fs::write(dir.join("metrics.jsonl"), "one\ntwo\nthr")?;
    assert!(tail.read().await?.is_empty());
    fs::write(dir.join("metrics.jsonl"), "one\ntwo\nthree\n")?;
    assert_eq!(tail.read().await?, vec!["three".to_string()]);
    let mut resumed = executor.tail(&metrics, 4);
    assert_eq!(
        resumed.read().await?,
        vec!["two".to_string(), "three".to_string()]
    );
    let mut missing = executor.tail(&format!("{metrics}.missing"), 0);
    assert!(missing.read().await?.is_empty());
    Ok(())
}

#[tokio::test]
async fn a_new_job_starts_without_stale_markers() -> TestResult {
    let root = tempfile::tempdir()?;
    let executor = LocalExecutor::new(root.path())?;
    let dir = Path::new(executor.workdir()).join("r0");
    fs::create_dir_all(&dir)?;
    for stale in ["exit_code", CANCELLING_FILE, CANCEL_FILE, "job.pid"] {
        fs::write(dir.join(stale), "0\n")?;
    }
    let started = executor.spawn(&job(&dir, "sleep 30")).await?;
    assert!(!dir.join(CANCELLING_FILE).exists());
    assert!(!dir.join(CANCEL_FILE).exists());
    assert_eq!(executor.status(&started).await?, JobStatus::Running);
    executor.cancel(&started).await?;
    assert_eq!(executor.status(&started).await?, JobStatus::Cancelled);
    Ok(())
}

#[tokio::test]
async fn cancel_stops_the_whole_process_group() -> TestResult {
    let root = tempfile::tempdir()?;
    let executor = LocalExecutor::new(root.path())?;
    let dir = Path::new(executor.workdir()).join("r2");
    let started = executor
        .spawn(&job(&dir, "sleep 60 & echo $! > child.pid; wait"))
        .await?;
    for _ in 0..50 {
        if dir.join("child.pid").exists() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(executor.status(&started).await?, JobStatus::Running);
    let begun = std::time::Instant::now();
    executor.cancel(&started).await?;
    // The group obeys SIGTERM: the cancel must not wait for the 10 second SIGKILL,
    // which it would if the exited job leader were left a zombie.
    assert!(
        begun.elapsed() < Duration::from_secs(5),
        "the cancel waited for the grace period: {:?}",
        begun.elapsed()
    );
    assert_eq!(executor.status(&started).await?, JobStatus::Cancelled);
    let child: u32 = fs::read_to_string(dir.join("child.pid"))?.trim().parse()?;
    assert!(
        !process_exists(child)?,
        "the grandchild survived the cancel"
    );
    // A second cancel is a no-op.
    executor.cancel(&started).await?;
    assert_eq!(executor.status(&started).await?, JobStatus::Cancelled);
    Ok(())
}

#[tokio::test]
async fn cancel_leaves_a_finished_job_exited() -> TestResult {
    let root = tempfile::tempdir()?;
    let executor = LocalExecutor::new(root.path())?;
    let dir = Path::new(executor.workdir()).join("r5");
    let started = executor.spawn(&job(&dir, "exit 4")).await?;
    assert_eq!(
        wait_finished(&executor, &started).await?,
        JobStatus::Exited(4)
    );
    executor.cancel(&started).await?;
    assert_eq!(executor.status(&started).await?, JobStatus::Exited(4));
    assert!(!dir.join(CANCELLING_FILE).exists());
    assert!(!dir.join(CANCEL_FILE).exists());
    Ok(())
}

#[tokio::test]
async fn status_during_a_cancel_is_running_then_cancelled_never_lost() -> TestResult {
    let root = tempfile::tempdir()?;
    let executor = LocalExecutor::new(root.path())?;
    let dir = Path::new(executor.workdir()).join("r6");
    // Obeys SIGTERM, so the group dies well within the grace period: the window
    // between its death and the final marker is where `lost` could leak out.
    let started = executor.spawn(&job(&dir, "sleep 30")).await?;
    assert_eq!(executor.status(&started).await?, JobStatus::Running);

    let poll = async {
        let mut seen = Vec::new();
        for _ in 0..600 {
            let status = executor.status(&started).await?;
            if seen.last() != Some(&status) {
                seen.push(status);
            }
            if status == JobStatus::Cancelled {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        Ok::<_, overbrainer::exec::ExecError>(seen)
    };
    let (cancelled, seen) = tokio::join!(executor.cancel(&started), poll);
    cancelled?;
    let seen = seen?;
    assert_eq!(
        seen,
        vec![JobStatus::Running, JobStatus::Cancelled],
        "unexpected status sequence during the cancel"
    );
    assert_eq!(executor.status(&started).await?, JobStatus::Cancelled);
    Ok(())
}

#[tokio::test]
async fn a_job_killed_without_exit_code_is_lost() -> TestResult {
    let root = tempfile::tempdir()?;
    let executor = LocalExecutor::new(root.path())?;
    let dir = Path::new(executor.workdir()).join("r3");
    let started = executor.spawn(&job(&dir, "kill -s KILL $$")).await?;
    assert_eq!(wait_finished(&executor, &started).await?, JobStatus::Lost);
    let gone = JobId {
        dir: root.path().join("nowhere").to_string_lossy().into_owned(),
        ..started
    };
    assert_eq!(executor.status(&gone).await?, JobStatus::Lost);
    Ok(())
}

#[tokio::test]
async fn read_from_honours_the_limit() -> TestResult {
    let root = tempfile::tempdir()?;
    let executor = LocalExecutor::new(root.path())?;
    let file = Path::new(executor.workdir()).join("data.txt");
    fs::write(&file, "0123456789")?;
    let path = file.to_string_lossy().into_owned();
    assert_eq!(executor.read_from(&path, 2, 3).await?, b"234");
    assert_eq!(executor.read_from(&path, 8, 100).await?, b"89");
    assert_eq!(executor.read_from(&path, 0, 0).await?, b"");
    assert_eq!(executor.read_from(&path, 20, 5).await?, b"");
    assert!(
        executor
            .read_from(&format!("{path}.missing"), 0, 5)
            .await?
            .is_empty()
    );
    Ok(())
}

#[tokio::test]
async fn upload_and_download_copy_selected_trees() -> TestResult {
    let root = tempfile::tempdir()?;
    let executor = LocalExecutor::new(&root.path().join("work"))?;
    let local = root.path().join("local");
    fs::create_dir_all(local.join("data"))?;
    fs::write(local.join("axolotl.yaml"), "a: 1\n")?;
    fs::write(local.join("data/train.jsonl"), "{}\n")?;

    let remote = format!("{}/r4", executor.workdir());
    executor.upload(&local, &remote).await?;
    assert_eq!(
        fs::read_to_string(Path::new(&remote).join("data/train.jsonl"))?,
        "{}\n"
    );

    fs::create_dir_all(Path::new(&remote).join("output/checkpoint-10"))?;
    fs::write(
        Path::new(&remote).join("output/adapter_model.safetensors"),
        "w",
    )?;
    fs::write(Path::new(&remote).join("output/checkpoint-10/state"), "s")?;
    let back = root.path().join("back");
    executor
        .download(
            &remote,
            &back,
            &["output".to_string(), "missing.txt".to_string()],
            &["checkpoint-*".to_string()],
        )
        .await?;
    assert!(back.join("output/adapter_model.safetensors").is_file());
    assert!(!back.join("output/checkpoint-10").exists());
    assert!(!back.join("missing.txt").exists());

    let nothing = root.path().join("nothing");
    executor
        .download(&remote, &nothing, &["missing.txt".to_string()], &[])
        .await?;
    assert!(!nothing.exists());

    // Same directory on both sides: nothing to copy.
    executor.upload(Path::new(&remote), &remote).await?;
    Ok(())
}
