//! `SshExecutor` against a real sshd. Runs only when `OVERBRAINER_TEST_SSH_HOST` (a
//! host alias) and `OVERBRAINER_TEST_SSH_CONFIG` (the ssh config file defining it,
//! with its key and `known_hosts`) are set, as in the `ssh` CI job; skipped otherwise.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use overbrainer::exec::{
    ExecError, Executor, JobCommand, JobId, JobStatus, MAX_TAIL_READ, SshExecutor,
};
use secrecy::SecretString;
use tokio::sync::Semaphore;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// The secret every test job gets. Distinctive, so a scan of process command lines
/// cannot match it by accident.
const SECRET: &str = "ob-s3cr3t-7f2c 'value'";

fn target() -> Option<(String, PathBuf)> {
    let host = std::env::var("OVERBRAINER_TEST_SSH_HOST").ok()?;
    let config = std::env::var_os("OVERBRAINER_TEST_SSH_CONFIG")?;
    Some((host, PathBuf::from(config)))
}

fn skip() {
    eprintln!("skipped: OVERBRAINER_TEST_SSH_HOST and OVERBRAINER_TEST_SSH_CONFIG are not set");
}

/// Connections being set up at once. sshd's default `MaxStartups 10:30:100` starts
/// dropping unauthenticated connections past 10, and the tests run in parallel.
static HANDSHAKES: Semaphore = Semaphore::const_new(4);

/// Connects to `host` with at most [`HANDSHAKES`] connections being set up at once.
async fn open(host: &str, workdir: &str, config: &Path) -> Result<SshExecutor, ExecError> {
    let _permit = HANDSHAKES
        .acquire()
        .await
        .map_err(|_| ExecError::Protocol("the handshake limit is closed".into()))?;
    SshExecutor::connect(host, workdir, Some(config)).await
}

/// A work directory unique to the test, so tests can run in parallel.
async fn connect(name: &str) -> Result<Option<SshExecutor>, Box<dyn std::error::Error>> {
    let Some((host, config)) = target() else {
        return Ok(None);
    };
    let workdir = format!("overbrainer-tests/{name}-{}", fastrand::u32(..));
    Ok(Some(open(&host, &workdir, &config).await?))
}

async fn wait_finished(
    executor: &SshExecutor,
    job: &JobId,
) -> Result<JobStatus, Box<dyn std::error::Error>> {
    for _ in 0..100 {
        let status = executor.status(job).await?;
        if status.is_finished() {
            return Ok(status);
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    Err("the job did not finish".into())
}

fn job(dir: String, script: &str) -> JobCommand {
    JobCommand {
        dir,
        script: script.to_string(),
        secrets: vec![("OB_TEST_SECRET".to_string(), SecretString::from(SECRET))],
        container: None,
    }
}

async fn read(executor: &SshExecutor, path: &str) -> Result<String, Box<dyn std::error::Error>> {
    Ok(String::from_utf8(
        executor.read_from(path, 0, MAX_TAIL_READ).await?,
    )?)
}

/// Runs `script` as a job in `dir` and returns its output once it has exited `0`.
async fn probe(
    executor: &SshExecutor,
    dir: String,
    script: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let started = executor.spawn(&job(dir.clone(), script)).await?;
    assert_eq!(
        wait_finished(executor, &started).await?,
        JobStatus::Exited(0)
    );
    read(executor, &format!("{dir}/job.log")).await
}

#[tokio::test]
async fn connect_resolves_the_workdir_under_home() -> TestResult {
    let Some(executor) = connect("workdir").await? else {
        skip();
        return Ok(());
    };
    assert!(
        executor.workdir().starts_with('/'),
        "{}",
        executor.workdir()
    );
    assert!(executor.workdir().contains("/overbrainer-tests/workdir-"));
    Ok(())
}

#[tokio::test]
async fn a_detached_job_gets_its_secret_and_reports_its_exit_code() -> TestResult {
    let Some(executor) = connect("job").await? else {
        skip();
        return Ok(());
    };
    let dir = format!("{}/r1", executor.workdir());
    let started = executor
        .spawn(&job(
            dir.clone(),
            // The secret is compared here, not in the script: another test scans every
            // command line on the target for it.
            "sleep 1; printf 'one\\ntwo\\n' >> metrics.jsonl; printf '%s' \"$OB_TEST_SECRET\" > secret.txt; echo done; exit 3",
        ))
        .await?;
    assert_eq!(executor.status(&started).await?, JobStatus::Running);
    assert_eq!(
        wait_finished(&executor, &started).await?,
        JobStatus::Exited(3)
    );
    assert_eq!(read(&executor, &format!("{dir}/job.log")).await?, "done\n");
    assert_eq!(read(&executor, &format!("{dir}/secret.txt")).await?, SECRET);
    assert_eq!(
        read(&executor, &format!("{dir}/job.pid")).await?.trim(),
        started.pid.get().to_string()
    );
    let mut tail = executor.tail(&format!("{dir}/metrics.jsonl"), 4);
    assert_eq!(tail.read().await?, vec!["two".to_string()]);
    assert_eq!(tail.offset(), 8);
    assert!(tail.read().await?.is_empty());
    Ok(())
}

#[tokio::test]
async fn secrets_never_reach_a_command_line() -> TestResult {
    let Some(executor) = connect("argv").await? else {
        skip();
        return Ok(());
    };
    let running = executor
        .spawn(&job(
            format!("{}/holder", executor.workdir()),
            "sleep 37 & wait",
        ))
        .await?;
    // Every command line on the target, as `ps` and `/proc` show them, while the job
    // holding the secret runs.
    let seen = probe(
        &executor,
        format!("{}/scan", executor.workdir()),
        "{ ps -ef 2>/dev/null || ps; } | sed 's/^/ps: /'\nfor f in /proc/[0-9]*/cmdline; do tr '\\000' ' ' < \"$f\" 2>/dev/null; echo; done",
    )
    .await?;
    executor.cancel(&running).await?;
    assert!(seen.contains("ps: "), "{seen}");
    assert!(seen.contains("sleep 37"), "the scan missed the job: {seen}");
    assert!(!seen.contains("s3cr3t"), "a secret is on a command line");

    // Nor on this machine, where the ssh master connection runs.
    for entry in fs::read_dir("/proc")? {
        let Ok(bytes) = fs::read(entry?.path().join("cmdline")) else {
            continue;
        };
        assert!(
            !String::from_utf8_lossy(&bytes).contains("s3cr3t"),
            "a secret is on a local command line"
        );
    }
    Ok(())
}

#[tokio::test]
async fn cancel_stops_the_process_group_and_never_reads_lost() -> TestResult {
    let Some(executor) = connect("cancel").await? else {
        skip();
        return Ok(());
    };
    let dir = format!("{}/r2", executor.workdir());
    let started = executor
        .spawn(&job(dir.clone(), "sleep 60 & echo $! > child.pid; wait"))
        .await?;
    let mut child_pid = String::new();
    for _ in 0..50 {
        child_pid = read(&executor, &format!("{dir}/child.pid")).await?;
        if !child_pid.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(!child_pid.trim().is_empty(), "the job never started");
    assert_eq!(executor.status(&started).await?, JobStatus::Running);

    let poll = async {
        let mut seen = Vec::new();
        for _ in 0..300 {
            let status = executor.status(&started).await?;
            if seen.last() != Some(&status) {
                seen.push(status);
            }
            if status == JobStatus::Cancelled {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Ok::<_, ExecError>(seen)
    };
    let (cancelled, seen) = tokio::join!(executor.cancel(&started), poll);
    cancelled?;
    let seen = seen?;
    assert_eq!(seen.last(), Some(&JobStatus::Cancelled), "{seen:?}");
    assert!(
        seen.iter()
            .all(|status| matches!(status, JobStatus::Running | JobStatus::Cancelled)),
        "unexpected status during the cancel: {seen:?}"
    );
    assert_eq!(executor.status(&started).await?, JobStatus::Cancelled);

    let gone = probe(
        &executor,
        format!("{}/probe", executor.workdir()),
        &format!(
            "kill -s 0 {} 2>/dev/null && echo child-alive || echo child-gone\nkill -s 0 -{} 2>/dev/null && echo group-alive || echo group-gone",
            child_pid.trim(),
            started.pid.get()
        ),
    )
    .await?;
    assert_eq!(gone, "child-gone\ngroup-gone\n");

    // A second cancel is a no-op.
    executor.cancel(&started).await?;
    assert_eq!(executor.status(&started).await?, JobStatus::Cancelled);
    Ok(())
}

#[tokio::test]
async fn cancel_leaves_a_finished_job_exited() -> TestResult {
    let Some(executor) = connect("finished").await? else {
        skip();
        return Ok(());
    };
    let dir = format!("{}/r5", executor.workdir());
    let started = executor.spawn(&job(dir.clone(), "exit 4")).await?;
    assert_eq!(
        wait_finished(&executor, &started).await?,
        JobStatus::Exited(4)
    );
    executor.cancel(&started).await?;
    assert_eq!(executor.status(&started).await?, JobStatus::Exited(4));
    assert_eq!(read(&executor, &format!("{dir}/cancelling")).await?, "");
    assert_eq!(read(&executor, &format!("{dir}/cancelled")).await?, "");
    Ok(())
}

#[tokio::test]
async fn read_from_honours_the_limit() -> TestResult {
    let Some(executor) = connect("read").await? else {
        skip();
        return Ok(());
    };
    let dir = format!("{}/r6", executor.workdir());
    probe(&executor, dir.clone(), "printf 0123456789 > data.txt").await?;
    let path = format!("{dir}/data.txt");
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
async fn upload_and_download_move_trees_through_tar() -> TestResult {
    let Some(executor) = connect("transfer").await? else {
        skip();
        return Ok(());
    };
    let local = tempfile::tempdir()?;
    fs::create_dir_all(local.path().join("up/data"))?;
    fs::write(local.path().join("up/data/train.jsonl"), "{}\n")?;
    let remote = format!("{}/r3", executor.workdir());
    executor.upload(&local.path().join("up"), &remote).await?;
    assert_eq!(
        read(&executor, &format!("{remote}/data/train.jsonl")).await?,
        "{}\n"
    );

    let made = executor
        .spawn(&job(
            remote.clone(),
            "mkdir -p output/checkpoint-5 && echo w > output/adapter.bin && echo s > output/checkpoint-5/state",
        ))
        .await?;
    assert_eq!(wait_finished(&executor, &made).await?, JobStatus::Exited(0));
    let back = local.path().join("back");
    executor
        .download(
            &remote,
            &back,
            &[
                "output".to_string(),
                "job.log".to_string(),
                "missing".to_string(),
            ],
            &["checkpoint-*".to_string()],
        )
        .await?;
    assert_eq!(fs::read_to_string(back.join("output/adapter.bin"))?, "w\n");
    assert!(back.join("job.log").is_file());
    assert!(!back.join("output/checkpoint-5").exists());

    let nothing = local.path().join("nothing");
    executor
        .download(&remote, &nothing, &["missing".to_string()], &[])
        .await?;
    assert!(!Path::new(&nothing).exists());
    Ok(())
}

#[tokio::test]
async fn a_download_into_an_unwritable_directory_reports_tar_and_does_not_hang() -> TestResult {
    let Some(executor) = connect("unwritable").await? else {
        skip();
        return Ok(());
    };
    let dir = format!("{}/r7", executor.workdir());
    // Far more than the pipes between the two tars hold.
    probe(
        &executor,
        dir.clone(),
        "mkdir -p output && head -c 8388608 /dev/zero > output/big",
    )
    .await?;
    let local = tempfile::tempdir()?;
    let locked = local.path().join("locked");
    fs::create_dir(&locked)?;
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o000))?;
    if fs::read_dir(&locked).is_ok() {
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o755))?;
        eprintln!("skipped: permissions are not enforced for this user");
        return Ok(());
    }
    let result = tokio::time::timeout(
        Duration::from_secs(60),
        executor.download(&dir, &locked, &["output".to_string()], &[]),
    )
    .await;
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o755))?;
    match result {
        Err(_) => return Err("the download hung once the local tar died".into()),
        Ok(Err(ExecError::Command { action, message })) => {
            assert_eq!(action, "extract");
            assert!(message.contains("Permission denied"), "{message}");
        },
        Ok(other) => return Err(format!("expected tar's own error, got {other:?}").into()),
    }
    Ok(())
}

#[tokio::test]
async fn a_download_from_a_missing_run_dir_is_an_error() -> TestResult {
    let Some(executor) = connect("missing").await? else {
        skip();
        return Ok(());
    };
    let remote = format!("{}/never-created", executor.workdir());
    let local = tempfile::tempdir()?;
    let back = local.path().join("back");
    let result = executor
        .download(&remote, &back, &["output".to_string()], &[])
        .await;
    match result {
        Err(ExecError::Command { action, message }) => {
            assert_eq!(action, "download");
            assert_eq!(message, format!("{remote} does not exist"));
        },
        other => return Err(format!("expected a missing directory error, got {other:?}").into()),
    }
    assert!(!back.exists());
    Ok(())
}

#[tokio::test]
async fn a_job_outlives_its_connection() -> TestResult {
    let Some((host, config)) = target() else {
        skip();
        return Ok(());
    };
    let Some(first) = connect("reconnect").await? else {
        skip();
        return Ok(());
    };
    let workdir = first.workdir().to_string();
    let started = first
        .spawn(&job(format!("{workdir}/r8"), "sleep 60"))
        .await?;
    assert_eq!(first.status(&started).await?, JobStatus::Running);
    // Closes the master connection and its control socket.
    drop(first);

    let second = open(&host, &workdir, &config).await?;
    assert_eq!(second.workdir(), workdir);
    assert_eq!(second.status(&started).await?, JobStatus::Running);
    second.cancel(&started).await?;
    assert_eq!(second.status(&started).await?, JobStatus::Cancelled);
    Ok(())
}

#[tokio::test]
async fn an_unknown_host_key_is_refused() -> TestResult {
    let Some((host, config)) = target() else {
        skip();
        return Ok(());
    };
    let dir = tempfile::tempdir()?;
    let empty = dir.path().join("known_hosts");
    fs::write(&empty, "")?;
    // ssh keeps the first value it reads, so these win over the included file.
    let strict = dir.path().join("config");
    fs::write(
        &strict,
        format!(
            "Host *\n  UserKnownHostsFile {}\n  GlobalKnownHostsFile /dev/null\nInclude {}\n",
            empty.display(),
            config.display()
        ),
    )?;
    let result = open(&host, "overbrainer-tests", &strict).await;
    assert!(
        result.is_err(),
        "connected to a host missing from known_hosts"
    );
    Ok(())
}
