//! `SshExecutor` against a real sshd. Runs only when `OVERBRAINER_TEST_SSH_HOST` (a
//! host alias) and `OVERBRAINER_TEST_SSH_CONFIG` (the ssh config file defining it,
//! with its key and `known_hosts`) are set, as in the `ssh` CI job; skipped otherwise.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use overbrainer::config::SshClient;
use overbrainer::exec::{
    ExecError, Executor, JobCommand, JobId, JobStatus, MAX_TAIL_READ, SshDestination, SshExecutor,
    sha256_file,
};
use secrecy::SecretString;
use tokio::sync::Semaphore;

use crate::common::each_ssh_client;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// The secret every test job gets. Distinctive, so a scan of process command lines
/// cannot match it by accident.
const SECRET: &str = "ob-s3cr3t-7f2c 'value'";

fn target() -> Option<(String, PathBuf)> {
    let host = std::env::var("OVERBRAINER_TEST_SSH_HOST").ok()?;
    let config = std::env::var_os("OVERBRAINER_TEST_SSH_CONFIG")?;
    Some((host, PathBuf::from(config)))
}

/// `error` and its sources on one line.
fn chain(error: &(dyn std::error::Error + 'static)) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    text
}

fn skip() {
    eprintln!("skipped: OVERBRAINER_TEST_SSH_HOST and OVERBRAINER_TEST_SSH_CONFIG are not set");
}

/// Connections being set up at once. sshd's default `MaxStartups 10:30:100` starts
/// dropping unauthenticated connections past 10, and the tests run in parallel.
static HANDSHAKES: Semaphore = Semaphore::const_new(4);

/// Connects to `host` with `client`, with at most [`HANDSHAKES`] connections
/// being set up at once.
async fn open(
    host: &str,
    workdir: &str,
    config: &Path,
    client: SshClient,
) -> Result<SshExecutor, ExecError> {
    let _permit = HANDSHAKES
        .acquire()
        .await
        .map_err(|_| ExecError::Protocol("the handshake limit is closed".into()))?;
    let destination = SshDestination::Config {
        destination: host,
        config_file: Some(config),
    };
    SshExecutor::connect(&destination, workdir, client).await
}

/// Connects with `client` to a work directory unique to the test, so tests can
/// run in parallel.
async fn connect(
    name: &str,
    client: SshClient,
) -> Result<Option<SshExecutor>, Box<dyn std::error::Error>> {
    let Some((host, config)) = target() else {
        return Ok(None);
    };
    let workdir = format!("overbrainer-tests/{name}-{}", fastrand::u32(..));
    Ok(Some(open(&host, &workdir, &config, client).await?))
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

/// A run directory is claimed by its first owner only, with each client.
#[tokio::test]
async fn a_run_directory_is_claimed_once_over_ssh() -> TestResult {
    each_ssh_client(|client| async move {
        let Some(executor) = connect("claim", client).await? else {
            skip();
            return Ok(());
        };
        let dir = format!("{}/demo_20260930-120000", executor.workdir());
        assert!(executor.claim(&dir, "mine").await?);
        assert!(executor.claim(&dir, "mine").await?);
        assert!(!executor.claim(&dir, "other").await?);
        let marker = executor.read_from(&format!("{dir}/.claim"), 0, 16).await?;
        assert_eq!(marker, b"mine\n");
        Ok(())
    })
    .await
}

/// The system probe reads the remote disk and memory, and a failing probe is a command error.
#[tokio::test]
async fn the_system_probe_samples_the_remote_machine() -> TestResult {
    each_ssh_client(|client| async move {
        let Some(executor) = connect("system", client).await? else {
            skip();
            return Ok(());
        };
        let script = overbrainer::system::probe_script(executor.workdir());
        let output = executor.probe(&script).await?;
        let sample = overbrainer::system::parse(
            &String::from_utf8_lossy(&output),
            std::time::SystemTime::now(),
            None,
        );
        assert!(
            sample.run_disk().is_some_and(|disk| disk.size_bytes > 0),
            "{sample:?}"
        );
        assert!(sample.memory.is_some(), "{sample:?}");
        let failed = executor.probe("exit 3").await;
        assert!(
            matches!(
                failed,
                Err(ExecError::Command {
                    action: "probe",
                    ..
                })
            ),
            "{failed:?}"
        );
        Ok(())
    })
    .await
}

/// A relative work directory resolves to an absolute path under the remote home.
#[tokio::test]
async fn connect_resolves_the_workdir_under_home() -> TestResult {
    each_ssh_client(|client| async move {
        let Some(executor) = connect("workdir", client).await? else {
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
    })
    .await
}

/// A detached job reads its secret, writes its log and reports its exit code.
#[tokio::test]
async fn a_detached_job_gets_its_secret_and_reports_its_exit_code() -> TestResult {
    each_ssh_client(|client| async move {
        let Some(executor) = connect("job", client).await? else {
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
        assert_eq!(tail.read().await?, [] as [String; 0]);
        Ok(())
    })
    .await
}

/// A job's secret is on no command line, remote or local, while the job runs.
#[tokio::test]
async fn secrets_never_reach_a_command_line() -> TestResult {
    each_ssh_client(|client| async move {
        let Some(executor) = connect("argv", client).await? else {
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
    })
    .await
}

/// A cancel stops the whole process group, and the status never reads lost meanwhile.
#[tokio::test]
async fn cancel_stops_the_process_group_and_never_reads_lost() -> TestResult {
    each_ssh_client(|client| async move {
        let Some(executor) = connect("cancel", client).await? else {
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
    })
    .await
}

/// A cancel of a finished job leaves it exited with its own code.
#[tokio::test]
async fn cancel_leaves_a_finished_job_exited() -> TestResult {
    each_ssh_client(|client| async move {
        let Some(executor) = connect("finished", client).await? else {
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
    })
    .await
}

/// A read returns the bytes asked for, and nothing past the end or for a missing file.
#[tokio::test]
async fn read_from_honours_the_limit() -> TestResult {
    each_ssh_client(|client| async move {
        let Some(executor) = connect("read", client).await? else {
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
        assert_eq!(
            executor.read_from(&format!("{path}.missing"), 0, 5).await?,
            [] as [u8; 0]
        );
        Ok(())
    })
    .await
}

/// Uploads skip the listed entries, and downloads bring back the asked entries without the excluded ones.
#[tokio::test]
async fn upload_and_download_move_trees_through_tar() -> TestResult {
    each_ssh_client(|client| async move {
        let Some(executor) = connect("transfer", client).await? else {
            skip();
            return Ok(());
        };
        let local = tempfile::tempdir()?;
        fs::create_dir_all(local.path().join("up/data"))?;
        fs::write(local.path().join("up/data/train.jsonl"), "{}\n")?;
        fs::create_dir_all(local.path().join("up/ssh"))?;
        fs::write(local.path().join("up/ssh/id_ed25519"), "private\n")?;
        fs::write(local.path().join("up/pod.json"), "{}\n")?;
        let remote = format!("{}/r3", executor.workdir());
        let skip = ["ssh".to_string(), "pod.json".to_string()];
        executor
            .upload(&local.path().join("up"), &remote, &skip)
            .await?;
        assert_eq!(
            read(&executor, &format!("{remote}/data/train.jsonl")).await?,
            "{}\n"
        );
        let skipped = executor
            .spawn(&job(remote.clone(), "test ! -e ssh && test ! -e pod.json"))
            .await?;
        assert_eq!(
            wait_finished(&executor, &skipped).await?,
            JobStatus::Exited(0),
            "the skipped entries reached the target"
        );
        let empty = tempfile::tempdir()?;
        fs::write(empty.path().join("pod.json"), "{}\n")?;
        let bare = format!("{}/r3-bare", executor.workdir());
        executor.upload(empty.path(), &bare, &skip).await?;
        let made_bare = executor
            .spawn(&job(bare.clone(), "test ! -e pod.json"))
            .await?;
        assert_eq!(
            wait_finished(&executor, &made_bare).await?,
            JobStatus::Exited(0)
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
    })
    .await
}

/// A download into an unwritable directory reports the local tar's error and does not hang.
#[tokio::test]
async fn a_download_into_an_unwritable_directory_reports_tar_and_does_not_hang() -> TestResult {
    each_ssh_client(|client| async move {
        let Some(executor) = connect("unwritable", client).await? else {
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
    })
    .await
}

/// A download from a missing run directory says so and creates nothing locally.
#[tokio::test]
async fn a_download_from_a_missing_run_dir_is_an_error() -> TestResult {
    each_ssh_client(|client| async move {
        let Some(executor) = connect("missing", client).await? else {
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
            other => {
                return Err(format!("expected a missing directory error, got {other:?}").into());
            },
        }
        assert!(!back.exists());
        Ok(())
    })
    .await
}

/// A detached job keeps running after its connection closes, and a new connection cancels it.
#[tokio::test]
async fn a_job_outlives_its_connection() -> TestResult {
    each_ssh_client(|client| async move {
        let Some((host, config)) = target() else {
            skip();
            return Ok(());
        };
        let Some(first) = connect("reconnect", client).await? else {
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

        let second = open(&host, &workdir, &config, client).await?;
        assert_eq!(second.workdir(), workdir);
        assert_eq!(second.status(&started).await?, JobStatus::Running);
        second.cancel(&started).await?;
        assert_eq!(second.status(&started).await?, JobStatus::Cancelled);
        Ok(())
    })
    .await
}

/// A host missing from `known_hosts` is refused as a host key verification failure, with each client.
#[tokio::test]
async fn an_unknown_host_key_is_refused() -> TestResult {
    each_ssh_client(|client| async move {
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
        let result = open(&host, "overbrainer-tests", &strict, client).await;
        let Err(error) = result else {
            return Err("connected to a host missing from known_hosts".into());
        };
        let text = chain(&error);
        assert!(
            text.to_lowercase().contains("host key verification failed"),
            "expected a host key verification failure, got: {text}"
        );
        Ok(())
    })
    .await
}

/// Through `OVERBRAINER_TEST_SSH_JUMP_HOST`, whose `ProxyJump` goes through a
/// bastion, each client reaches the same machine as the direct alias.
#[tokio::test]
async fn proxy_jump_reaches_the_target_through_the_bastion() -> TestResult {
    each_ssh_client(|client| async move {
        let (Some((host, config)), Ok(behind)) =
            (target(), std::env::var("OVERBRAINER_TEST_SSH_JUMP_HOST"))
        else {
            eprintln!("skipped: OVERBRAINER_TEST_SSH_JUMP_HOST is not set");
            return Ok(());
        };
        let workdir = format!("overbrainer-tests/jump-{}", fastrand::u32(..));
        let direct = open(&host, &workdir, &config, client).await?;
        let jumped = open(&behind, &workdir, &config, client).await?;
        let name = direct.probe("uname -n").await?;
        assert!(!name.is_empty(), "the target has no host name");
        assert_eq!(jumped.probe("uname -n").await?, name);
        assert_eq!(jumped.workdir(), direct.workdir());
        let dir = format!("{}/demo_20261005-120000", jumped.workdir());
        assert!(jumped.claim(&dir, "jump").await?);
        assert_eq!(
            direct.read_from(&format!("{dir}/.claim"), 0, 16).await?,
            b"jump\n"
        );
        Ok(())
    })
    .await
}

/// The manifest lists the files a download brings back, with their digests.
#[tokio::test]
async fn the_manifest_of_the_target_matches_the_downloaded_files() -> TestResult {
    each_ssh_client(|client| async move {
        let Some(executor) = connect("manifest", client).await? else {
            skip();
            return Ok(());
        };
        let remote = format!("{}/r9", executor.workdir());
        let made = executor
            .spawn(&job(
                remote.clone(),
                "mkdir -p output/checkpoint-5 output/nested && echo w > output/adapter.bin && echo n > output/nested/config.json && echo s > output/checkpoint-5/state",
            ))
            .await?;
        assert_eq!(wait_finished(&executor, &made).await?, JobStatus::Exited(0));
        let entries = [
            "output".to_string(),
            "job.log".to_string(),
            "missing".to_string(),
        ];
        let exclude = ["checkpoint-*".to_string()];
        let manifest = executor.manifest(&remote, &entries, &exclude).await?;
        let paths: Vec<&str> = manifest.iter().map(|file| file.path.as_str()).collect();
        assert_eq!(
            paths,
            vec!["job.log", "output/adapter.bin", "output/nested/config.json"]
        );
        let local = tempfile::tempdir()?;
        executor
            .download(&remote, local.path(), &entries, &exclude)
            .await?;
        for file in &manifest {
            assert_eq!(
                sha256_file(&local.path().join(&file.path))?,
                file.sha256,
                "{}",
                file.path
            );
        }
        let gone = executor
            .manifest(&format!("{remote}/gone"), &entries, &exclude)
            .await;
        assert!(
            matches!(&gone, Err(ExecError::Command { action: "manifest", message }) if message.ends_with("does not exist")),
            "{gone:?}"
        );
        Ok(())
    })
    .await
}

/// A file is written through a temporary sibling renamed into place, its directory created first.
#[tokio::test]
async fn a_marker_is_written_through_a_rename() -> TestResult {
    each_ssh_client(|client| async move {
        let Some(executor) = connect("marker", client).await? else {
            skip();
            return Ok(());
        };
        let dir = format!("{}/r10/.pod", executor.workdir());
        let made = executor.spawn(&job(dir.clone(), "true")).await?;
        assert_eq!(wait_finished(&executor, &made).await?, JobStatus::Exited(0));
        let marker = format!("{dir}/retrieved");
        executor.put_file(&marker, "").await?;
        let listing = probe(&executor, dir.clone(), "ls -a").await?;
        assert!(listing.lines().any(|name| name == "retrieved"), "{listing}");
        assert!(!listing.contains("retrieved.tmp"), "{listing}");
        let request = format!("{}/r10/new/dir/snapshot.request", executor.workdir());
        executor.put_file(&request, "requested").await?;
        let read = executor.read_from(&request, 0, 100).await?;
        assert_eq!(read, b"requested");
        Ok(())
    })
    .await
}

/// Marks the child run of [`builtin_works_with_a_broken_ssh_on_path`].
#[cfg(feature = "builtin-ssh")]
const BROKEN_SSH_CHILD: &str = "OVERBRAINER_TEST_BROKEN_SSH_CHILD";

/// The built-in client never runs `ssh`: this test binary runs
/// [`builtin_child_under_a_broken_ssh`] again with a `PATH` whose first `ssh`
/// fails and leaves a mark, and that run must connect and leave no mark. Only
/// the child's environment changes, never this process's.
#[cfg(feature = "builtin-ssh")]
#[tokio::test]
async fn builtin_works_with_a_broken_ssh_on_path() -> TestResult {
    if target().is_none() {
        skip();
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let mark = dir.path().join("ssh-ran");
    let fake = dir.path().join("ssh");
    fs::write(
        &fake,
        format!(
            "#!/bin/sh\ntouch '{}'\necho 'broken ssh' >&2\nexit 255\n",
            mark.display()
        ),
    )?;
    fs::set_permissions(&fake, fs::Permissions::from_mode(0o755))?;
    let path = format!(
        "{}:{}",
        dir.path().display(),
        std::env::var("PATH").unwrap_or_default()
    );
    // The fake really is the `ssh` such a PATH finds.
    let sanity = std::process::Command::new("ssh")
        .arg("-V")
        .env("PATH", &path)
        .output()?;
    assert!(
        !sanity.status.success(),
        "the fake ssh is not first on PATH"
    );
    assert!(mark.exists(), "the fake ssh left no mark");
    fs::remove_file(&mark)?;

    let child = tokio::process::Command::new(std::env::current_exe()?)
        .args([
            "--exact",
            "exec_ssh::builtin_child_under_a_broken_ssh",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("PATH", &path)
        .env(BROKEN_SSH_CHILD, "1")
        .output()
        .await?;
    let stdout = String::from_utf8_lossy(&child.stdout);
    assert!(
        child.status.success(),
        "the built-in client failed under a broken ssh: {stdout}{}",
        String::from_utf8_lossy(&child.stderr)
    );
    assert!(
        stdout.contains("1 passed"),
        "the child test did not run: {stdout}"
    );
    assert!(!mark.exists(), "the built-in client ran ssh");
    Ok(())
}

/// The child of [`builtin_works_with_a_broken_ssh_on_path`]: connects with the
/// built-in client and runs a command. Does nothing in a normal run.
#[cfg(feature = "builtin-ssh")]
#[tokio::test]
async fn builtin_child_under_a_broken_ssh() -> TestResult {
    if std::env::var_os(BROKEN_SSH_CHILD).is_none() {
        return Ok(());
    }
    let Some(executor) = connect("broken-ssh", SshClient::Builtin).await? else {
        return Err("the child lost the test sshd settings".into());
    };
    let output = executor.probe("echo reached").await?;
    assert_eq!(output, b"reached\n");
    Ok(())
}
