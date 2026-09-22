use std::io;
use std::path::Path;
use std::process::ExitStatus;
use std::time::Duration;

use openssh::{KnownHosts, Session, SessionBuilder, Stdio};
use secrecy::{ExposeSecret, SecretString};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::process::Child;

use super::tar;
use super::{
    CANCEL_FILE, CANCELLING_FILE, EXIT_FILE, ExecError, Executor, JOB_LOG, JobCommand, JobId,
    JobStatus, PID_FILE, Pid, cancel_script, check_secrets, job_script, parse_status, quote,
    shell_path, status_script,
};

/// Runs jobs on a remote Linux machine through the user's `ssh`: `~/.ssh/config`, the
/// agent and `known_hosts` apply, and unknown host keys are refused. One master
/// connection carries every command.
///
/// A job's secrets travel on the standard input of the command starting it and are
/// read into the job's environment there: they never appear on a command line,
/// remote or local, in a file or in an error.
///
/// The user `destination` logs in as must own the jobs it checks and cancels: status
/// and cancel signal the job's process group with `kill`. When that group belongs to
/// another user (for example a job started as root, or inside a rootful container
/// whose processes run as another user on the host), `kill -s 0` fails with a
/// permission error, which the status script cannot tell apart from a group that is
/// gone. Such a job reads [`JobStatus::Lost`] while it is really running, and a
/// cancel cannot stop it.
#[derive(Debug)]
pub struct SshExecutor {
    session: Session,
    workdir: String,
}

impl SshExecutor {
    /// Connects to `destination` (`user@host` or a `~/.ssh/config` alias) and
    /// creates `workdir` there, relative to the remote home unless absolute. Tests
    /// pass `config_file` in place of `~/.ssh/config`.
    ///
    /// # Errors
    ///
    /// Returns [`ExecError::Ssh`] when the connection fails, for example on an
    /// unknown host key, and [`ExecError::Command`] when `workdir` cannot be created.
    pub async fn connect(
        destination: &str,
        workdir: &str,
        config_file: Option<&Path>,
    ) -> Result<Self, ExecError> {
        let mut builder = SessionBuilder::default();
        builder
            .known_hosts_check(KnownHosts::Strict)
            .connect_timeout(Duration::from_secs(30))
            .server_alive_interval(Duration::from_secs(15));
        if let Some(config_file) = config_file {
            builder.config_file(config_file);
        }
        let session = builder
            .connect_mux(destination)
            .await
            .map_err(ExecError::Ssh)?;
        let dir = shell_path(workdir);
        let resolved = run(
            &session,
            &format!("mkdir -p -- {dir} && cd -- {dir} && pwd -P"),
            "create the work directory",
        )
        .await?;
        let workdir = String::from_utf8_lossy(&resolved).trim().to_string();
        if !workdir.starts_with('/') {
            return Err(ExecError::Protocol(format!(
                "the work directory resolved to `{workdir}`"
            )));
        }
        Ok(Self { session, workdir })
    }

    async fn start(&self, job: &JobCommand) -> Result<JobId, ExecError> {
        let launcher = launcher(job)?;
        let mut command = self.session.shell(launcher);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn().await.map_err(ExecError::Ssh)?;
        let stdin = child.stdin().take();
        // Fed while the output is read, so a launcher failing before it has read
        // every secret reports its own error rather than a broken pipe.
        let (fed, output) =
            tokio::join!(feed_secrets(stdin, &job.secrets), child.wait_with_output());
        let output = output.map_err(ExecError::Ssh)?;
        if !output.status.success() {
            return Err(ExecError::Command {
                action: "start the job",
                message: tar::failure(&output.stderr, output.status),
            });
        }
        fed.map_err(remote_io)?;
        let text = String::from_utf8_lossy(&output.stdout);
        let raw: u32 = text
            .trim()
            .parse()
            .map_err(|_| ExecError::Protocol(format!("`{}` is not a process ID", text.trim())))?;
        Ok(JobId {
            dir: job.dir.clone(),
            pid: Pid::new(raw)?,
            container: job.container.clone(),
        })
    }

    async fn send(&self, local: &Path, remote: &str) -> Result<(), ExecError> {
        // Checked up front: a local `tar` that cannot even open the directory would
        // send an empty stream, and the remote `tar` complaining about that would hide
        // the real cause.
        std::fs::read_dir(local).map_err(|source| ExecError::Io {
            path: local.to_path_buf(),
            source,
        })?;
        let mut command = self.session.shell(format!(
            "mkdir -p -- {dir} && tar -C {dir} -xf -",
            dir = quote(remote)
        ));
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let mut child = command.spawn().await.map_err(ExecError::Ssh)?;
        let create = tar::spawn_create(local, &[".".to_string()], &[])?;
        let (copied, errors, created) =
            transfer(create, child.stdin().take(), child.stderr().take()).await;
        let received = child.wait().await.map_err(ExecError::Ssh);
        upload_outcome(received, &errors, created, copied)
    }

    async fn fetch(
        &self,
        remote: &str,
        local: &Path,
        entries: &[String],
        exclude: &[String],
    ) -> Result<(), ExecError> {
        let mut command = self.session.shell(archive_script(remote, entries, exclude));
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn().await.map_err(ExecError::Ssh)?;
        let (extracted, errors) =
            receive(child.stdout().take(), child.stderr().take(), local).await;
        let sent = child.wait().await.map_err(ExecError::Ssh);
        download_outcome(extracted, sent, &errors)
    }
}

impl Executor for SshExecutor {
    fn workdir(&self) -> &str {
        &self.workdir
    }

    async fn upload(&self, local: &Path, remote: &str) -> Result<(), ExecError> {
        self.send(local, remote).await
    }

    async fn spawn(&self, job: &JobCommand) -> Result<JobId, ExecError> {
        self.start(job).await
    }

    async fn read_from(&self, path: &str, offset: u64, limit: u64) -> Result<Vec<u8>, ExecError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        run(&self.session, &read_script(path, offset, limit), "read").await
    }

    async fn status(&self, job: &JobId) -> Result<JobStatus, ExecError> {
        let script = status_script(&job.dir, job.pid.get());
        let output = run(&self.session, &script, "status").await?;
        let text = String::from_utf8_lossy(&output);
        parse_status(&text)
            .ok_or_else(|| ExecError::Protocol(format!("unknown job status {:?}", text.trim())))
    }

    async fn cancel(&self, job: &JobId) -> Result<(), ExecError> {
        let script = cancel_script(&job.dir, job.pid.get(), job.container.as_ref());
        run(&self.session, &script, "cancel").await.map(drop)
    }

    async fn download(
        &self,
        remote: &str,
        local: &Path,
        entries: &[String],
        exclude: &[String],
    ) -> Result<(), ExecError> {
        self.fetch(remote, local, entries, exclude).await
    }
}

/// The command starting `job` detached from the SSH session, printing its process ID.
/// It first reads one line per secret from its standard input into the environment
/// variable of that name: secret values are never written here.
///
/// # Errors
///
/// Returns [`ExecError::InvalidSecret`] when a secret cannot be passed to a job
/// (see [`check_secrets`]).
fn launcher(job: &JobCommand) -> Result<String, ExecError> {
    check_secrets(&job.secrets)?;
    let mut parts = Vec::new();
    for (name, _) in &job.secrets {
        parts.push(format!("IFS= read -r {name} && export {name}"));
    }
    parts.push(format!(
        "mkdir -p -- {dir} && cd -- {dir} && rm -f {EXIT_FILE} {CANCELLING_FILE} {CANCEL_FILE} {PID_FILE} && \
         {{ setsid nohup sh -c {inner} > {JOB_LOG} 2>&1 < /dev/null & }} && echo \"$!\"",
        dir = quote(&job.dir),
        inner = quote(&job_script(&job.script)),
    ));
    Ok(parts.join(" && "))
}

/// Writes each secret on a line of its own to `stdin`, then closes it.
async fn feed_secrets<W: AsyncWrite + Unpin>(
    stdin: Option<W>,
    secrets: &[(String, SecretString)],
) -> io::Result<()> {
    let Some(mut stdin) = stdin else {
        return Err(io::Error::other("the launcher has no standard input"));
    };
    for (_, value) in secrets {
        stdin.write_all(value.expose_secret().as_bytes()).await?;
        stdin.write_all(b"\n").await?;
    }
    stdin.shutdown().await
}

/// Prints at most `limit` bytes of `path` from byte `offset`; a missing file prints
/// nothing.
fn read_script(path: &str, offset: u64, limit: u64) -> String {
    format!(
        "[ -f {path} ] || exit 0\ntail -c +{start} -- {path} | head -c {limit}\n",
        path = quote(path),
        start = offset.saturating_add(1)
    )
}

/// Writes to stdout a `tar` archive of the `entries` of `remote` that exist, without
/// the names matching an `exclude` pattern, or nothing when none exists. Fails, saying
/// so, when `remote` itself is not a directory.
fn archive_script(remote: &str, entries: &[String], exclude: &[String]) -> String {
    let candidates: Vec<String> = entries.iter().map(|entry| quote(entry)).collect();
    let args: Vec<String> = tar::create_args(".", &[], exclude)
        .iter()
        .map(|arg| quote(arg))
        .collect();
    format!(
        "[ -d {dir} ] || {{ printf '%s does not exist\\n' {dir} >&2; exit 1; }}\ncd -- {dir} || exit 1\nset --\nfor entry in {candidates}; do [ -e \"$entry\" ] && set -- \"$@\" \"$entry\"; done\n[ \"$#\" -gt 0 ] || exit 0\nexec tar {args} \"$@\"\n",
        dir = quote(remote),
        candidates = candidates.join(" "),
        args = args.join(" "),
    )
}

/// Streams the archive of the local `tar` `create` into `sink`, closing it at the
/// end, while reading the receiver's error output `errors` and `create`'s own, all at
/// once so no error pipe can fill up and stall its writer. Returns the result of the
/// copy, the receiver's error output and the result of `create`.
async fn transfer<W, E>(
    mut create: Child,
    sink: Option<W>,
    errors: Option<E>,
) -> (io::Result<()>, Vec<u8>, Result<(), ExecError>)
where
    W: AsyncWrite + Unpin,
    E: AsyncRead + Unpin,
{
    let source = create.stdout.take();
    // Both ends are dropped as soon as the copy stops, so a receiver that dies early
    // makes the local `tar` fail on a broken pipe instead of blocking forever.
    let copy = async move {
        let (Some(mut source), Some(mut sink)) = (source, sink) else {
            return Err(io::Error::other("a transfer pipe is missing"));
        };
        tokio::io::copy(&mut source, &mut sink).await?;
        sink.shutdown().await
    };
    tokio::join!(copy, drain(errors), tar::finish(create, "archive"))
}

/// Extracts the archive read from `archive` into `local` while reading the sender's
/// error output `errors`, both at once. Returns the extraction's result and that
/// output.
async fn receive<R, E>(
    archive: Option<R>,
    errors: Option<E>,
    local: &Path,
) -> (Result<(), ExecError>, Vec<u8>)
where
    R: AsyncRead + Unpin,
    E: AsyncRead + Unpin,
{
    // `archive` is dropped once the extraction ends, early or not, so a sender still
    // writing sees a broken pipe instead of blocking forever.
    let extract = async move {
        match archive {
            Some(mut archive) => tar::extract(&mut archive, local).await,
            None => Err(ExecError::Protocol("the remote tar has no output".into())),
        }
    };
    tokio::join!(extract, drain(errors))
}

/// Everything `errors` holds until its end. A failed read only loses diagnostics.
async fn drain<E: AsyncRead + Unpin>(errors: Option<E>) -> Vec<u8> {
    let mut bytes = Vec::new();
    if let Some(mut errors) = errors
        && errors.read_to_end(&mut bytes).await.is_err()
    {
        bytes.extend_from_slice(b"\n(the error output could not be read to its end)");
    }
    bytes
}

/// The result of an upload. The receiving `tar`'s own failure comes first: when it
/// dies early, the local `tar` and the copy only fail on a broken pipe. A receiver
/// killed by a signal has no exit status: its error output, when it printed any,
/// says more than openssh's bare [`openssh::Error::RemoteProcessTerminated`].
fn upload_outcome(
    received: Result<ExitStatus, ExecError>,
    errors: &[u8],
    created: Result<(), ExecError>,
    copied: io::Result<()>,
) -> Result<(), ExecError> {
    let status = match received {
        Err(ExecError::Ssh(openssh::Error::RemoteProcessTerminated)) => {
            return Err(tar::error_text(errors).map_or(
                ExecError::Ssh(openssh::Error::RemoteProcessTerminated),
                |message| ExecError::Command {
                    action: "upload",
                    message,
                },
            ));
        },
        other => other?,
    };
    if !status.success() {
        return Err(ExecError::Command {
            action: "upload",
            message: tar::failure(errors, status),
        });
    }
    created?;
    copied.map_err(remote_io)
}

/// The result of a download. The local extraction's failure comes first: when it
/// dies early, the remote `tar` only fails on a broken pipe.
fn download_outcome(
    extracted: Result<(), ExecError>,
    sent: Result<ExitStatus, ExecError>,
    errors: &[u8],
) -> Result<(), ExecError> {
    extracted?;
    let status = sent?;
    if status.success() {
        Ok(())
    } else {
        Err(ExecError::Command {
            action: "download",
            message: tar::failure(errors, status),
        })
    }
}

/// Runs `script` with `sh` on the target and returns its stdout.
async fn run(session: &Session, script: &str, action: &'static str) -> Result<Vec<u8>, ExecError> {
    let output = session
        .shell(script)
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(ExecError::Ssh)?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(ExecError::Command {
            action,
            message: tar::failure(&output.stderr, output.status),
        })
    }
}

fn remote_io(source: io::Error) -> ExecError {
    ExecError::Ssh(openssh::Error::Remote(source))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command as StdCommand;

    use tempfile::tempdir;
    use tokio::process::Command;

    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn job(secret: &str) -> JobCommand {
        JobCommand {
            dir: "/w/r1".into(),
            script: "true".into(),
            secrets: vec![("HF_TOKEN".into(), SecretString::from(secret.to_string()))],
            container: None,
        }
    }

    fn with_name(name: &str) -> JobCommand {
        JobCommand {
            secrets: vec![(name.into(), SecretString::from("v".to_string()))],
            ..job("v")
        }
    }

    #[test]
    fn the_launcher_reads_secrets_from_stdin() -> Result<(), ExecError> {
        let launcher = launcher(&job("hf_value"))?;
        assert!(
            launcher.starts_with("IFS= read -r HF_TOKEN && export HF_TOKEN && mkdir -p -- '/w/r1'")
        );
        assert!(!launcher.contains("hf_value"));
        assert!(launcher.contains("rm -f exit_code cancelling cancelled job.pid"));
        Ok(())
    }

    #[test]
    fn multi_line_secrets_are_refused_without_their_value() {
        for value in ["line\nbreak", "line\rbreak", "nul\0byte"] {
            let error = launcher(&job(value)).err().map(|error| error.to_string());
            assert_eq!(
                error.as_deref(),
                Some("HF_TOKEN cannot be passed to the job")
            );
        }
    }

    #[test]
    fn secret_names_must_be_variable_names() {
        for name in ["", "1ABC", "A-B", "A B", "A;rm -rf /", "$(x)"] {
            assert!(
                matches!(launcher(&with_name(name)), Err(ExecError::InvalidSecret(_))),
                "{name:?} was accepted"
            );
        }
        assert!(launcher(&with_name("_A1")).is_ok());
    }

    /// Whether `setsid` (util-linux or busybox), which the launcher runs, is on `PATH`.
    fn setsid_available() -> bool {
        StdCommand::new("sh")
            .arg("-c")
            .arg("command -v setsid")
            .stdout(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    /// Runs `script` with the local `sh`, feeding it `stdin`, and returns its stdout.
    fn sh(script: &str, stdin: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        use std::io::Write;
        let mut child = StdCommand::new("sh")
            .arg("-c")
            .arg(script)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()?;
        if let Some(mut input) = child.stdin.take() {
            input.write_all(stdin)?;
        }
        let output = child.wait_with_output()?;
        if !output.status.success() {
            return Err(format!("sh failed: {}", output.status).into());
        }
        Ok(output.stdout)
    }

    #[tokio::test]
    async fn the_launcher_hands_secrets_to_the_job_verbatim() -> TestResult {
        if !setsid_available() {
            eprintln!("skipped: setsid is not installed");
            return Ok(());
        }
        let root = tempdir()?;
        let dir = root.path().join("r1");
        let job = JobCommand {
            dir: dir.to_string_lossy().into_owned(),
            script: "printf '%s|%s' \"$A\" \"$B\" > seen".into(),
            secrets: vec![
                ("A".into(), SecretString::from(" s3cr3t 'value' \\n ")),
                ("B".into(), SecretString::from(String::new())),
            ],
            container: None,
        };
        let mut stdin = Vec::new();
        feed_secrets(Some(&mut stdin), &job.secrets).await?;
        let printed = sh(&launcher(&job)?, &stdin)?;
        let pid: u32 = String::from_utf8(printed)?.trim().parse()?;
        assert!(pid > 1);
        for _ in 0..100 {
            if dir.join(EXIT_FILE).exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert_eq!(fs::read_to_string(dir.join(EXIT_FILE))?.trim(), "0");
        assert_eq!(
            fs::read_to_string(dir.join(PID_FILE))?.trim(),
            pid.to_string()
        );
        assert_eq!(
            fs::read_to_string(dir.join("seen"))?,
            " s3cr3t 'value' \\n |"
        );
        Ok(())
    }

    #[test]
    fn read_script_honours_offset_and_limit() -> TestResult {
        let root = tempdir()?;
        let file = root.path().join("data.txt");
        fs::write(&file, "0123456789")?;
        let path = file.to_string_lossy().into_owned();
        assert_eq!(sh(&read_script(&path, 2, 3), b"")?, b"234");
        assert_eq!(sh(&read_script(&path, 8, 100), b"")?, b"89");
        assert_eq!(sh(&read_script(&path, 20, 5), b"")?, b"");
        assert_eq!(
            sh(&read_script(&format!("{path}.missing"), 0, 5), b"")?,
            b""
        );
        Ok(())
    }

    #[test]
    fn archive_script_skips_missing_entries_and_excludes() -> TestResult {
        let root = tempdir()?;
        fs::create_dir_all(root.path().join("output/checkpoint-5"))?;
        fs::write(root.path().join("output/adapter.bin"), "w")?;
        fs::write(root.path().join("output/checkpoint-5/state"), "s")?;
        let remote = root.path().to_string_lossy().into_owned();
        let script = archive_script(
            &remote,
            &["output".into(), "missing".into()],
            &["checkpoint-*".into()],
        );
        let archive = sh(&script, b"")?;
        let listing = sh("tar -tf -", &archive)?;
        let listing = String::from_utf8(listing)?;
        assert!(listing.contains("output/adapter.bin"), "{listing}");
        assert!(!listing.contains("checkpoint"), "{listing}");
        assert!(sh(&archive_script(&remote, &["missing".into()], &[]), b"")?.is_empty());

        let gone = format!("{remote}/never-created");
        let output = StdCommand::new("sh")
            .arg("-c")
            .arg(archive_script(&gone, &["output".into()], &[]))
            .output()?;
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        assert_eq!(
            String::from_utf8(output.stderr)?,
            format!("{gone} does not exist\n")
        );
        Ok(())
    }

    /// Whether permissions are enforced for this user (they are not for root).
    fn permissions_enforced(locked: &Path) -> bool {
        fs::read_dir(locked).is_err()
    }

    /// A local directory holding one file far larger than a pipe holds.
    fn big_source(root: &Path) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
        let source = root.join("source");
        fs::create_dir(&source)?;
        fs::write(source.join("big"), vec![b'x'; 4 * 1024 * 1024])?;
        Ok(source)
    }

    /// A local `tar -x` into `dir`, standing in for the remote receiver.
    fn receiver(dir: &Path) -> Result<Child, Box<dyn std::error::Error>> {
        Ok(Command::new("tar")
            .arg("-C")
            .arg(dir)
            .args(["-xf", "-"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()?)
    }

    async fn upload_to(
        source: &Path,
        mut receiver: Child,
    ) -> Result<Result<(), ExecError>, Box<dyn std::error::Error>> {
        let create = tar::spawn_create(source, &[".".to_string()], &[])?;
        let (copied, errors, created) =
            transfer(create, receiver.stdin.take(), receiver.stderr.take()).await;
        let status = receiver.wait().await?;
        Ok(upload_outcome(Ok(status), &errors, created, copied))
    }

    #[tokio::test]
    async fn an_upload_reports_the_receiving_tar_error_not_the_broken_pipe() -> TestResult {
        let root = tempdir()?;
        let source = big_source(root.path())?;
        let locked = root.path().join("locked");
        fs::create_dir(&locked)?;
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o000))?;
        if !permissions_enforced(&locked) {
            fs::set_permissions(&locked, fs::Permissions::from_mode(0o755))?;
            eprintln!("skipped: permissions are not enforced for this user");
            return Ok(());
        }
        let result = tokio::time::timeout(
            Duration::from_secs(30),
            upload_to(&source, receiver(&locked)?),
        )
        .await;
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o755))?;
        match result {
            Err(_) => return Err("the upload hung once the receiver died".into()),
            Ok(outcome) => match outcome? {
                Err(ExecError::Command { action, message }) => {
                    assert_eq!(action, "upload");
                    assert!(message.contains("Permission denied"), "{message}");
                    assert!(!message.contains("Broken pipe"), "{message}");
                },
                other => return Err(format!("expected tar's own error, got {other:?}").into()),
            },
        }
        Ok(())
    }

    #[tokio::test]
    async fn an_upload_survives_a_flood_of_receiver_warnings() -> TestResult {
        let root = tempdir()?;
        let source = root.path().join("source");
        fs::create_dir(&source)?;
        for index in 0..3000 {
            fs::write(source.join(format!("{index:05}-{}", "n".repeat(150))), "")?;
        }
        // Enterable but not writable: tar warns once per entry and keeps reading.
        let target = root.path().join("target");
        fs::create_dir(&target)?;
        fs::set_permissions(&target, fs::Permissions::from_mode(0o555))?;
        if fs::write(target.join("probe"), "").is_ok() {
            fs::set_permissions(&target, fs::Permissions::from_mode(0o755))?;
            eprintln!("skipped: permissions are not enforced for this user");
            return Ok(());
        }
        let result = tokio::time::timeout(
            Duration::from_secs(30),
            upload_to(&source, receiver(&target)?),
        )
        .await;
        fs::set_permissions(&target, fs::Permissions::from_mode(0o755))?;
        match result {
            Err(_) => return Err("the upload deadlocked on the receiver's warnings".into()),
            Ok(outcome) => match outcome? {
                Err(ExecError::Command { message, .. }) => {
                    assert!(message.contains("more lines)"), "{message}");
                    assert!(message.len() < 8 * 1024, "{} bytes", message.len());
                },
                other => return Err(format!("expected tar's warnings, got {other:?}").into()),
            },
        }
        Ok(())
    }

    /// A local `sh` standing in for the remote sender: runs `script` with its stdout
    /// and stderr piped.
    fn sender(script: &str) -> Result<Child, Box<dyn std::error::Error>> {
        Ok(Command::new("sh")
            .arg("-c")
            .arg(script)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?)
    }

    async fn download_from(
        mut sender: Child,
        local: &Path,
    ) -> Result<Result<(), ExecError>, Box<dyn std::error::Error>> {
        let (extracted, errors) = receive(sender.stdout.take(), sender.stderr.take(), local).await;
        let sent = sender.wait().await?;
        Ok(download_outcome(extracted, Ok(sent), &errors))
    }

    #[tokio::test]
    async fn a_download_reports_the_local_tar_error_not_the_broken_pipe() -> TestResult {
        let root = tempdir()?;
        let source = big_source(root.path())?;
        let locked = root.path().join("locked");
        fs::create_dir(&locked)?;
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o000))?;
        if !permissions_enforced(&locked) {
            fs::set_permissions(&locked, fs::Permissions::from_mode(0o755))?;
            eprintln!("skipped: permissions are not enforced for this user");
            return Ok(());
        }
        let script = archive_script(&source.to_string_lossy(), &["big".into()], &[]);
        let result = tokio::time::timeout(
            Duration::from_secs(30),
            download_from(sender(&script)?, &locked),
        )
        .await;
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o755))?;
        match result {
            Err(_) => return Err("the download hung once the extraction died".into()),
            Ok(outcome) => match outcome? {
                Err(ExecError::Command { action, message }) => {
                    assert_eq!(action, "extract");
                    assert!(message.contains("Permission denied"), "{message}");
                },
                other => return Err(format!("expected tar's own error, got {other:?}").into()),
            },
        }
        Ok(())
    }

    #[tokio::test]
    async fn a_download_survives_a_flood_of_sender_warnings() -> TestResult {
        let root = tempdir()?;
        let source = root.path().join("source");
        fs::create_dir(&source)?;
        fs::write(source.join("file"), "content\n")?;
        // 256 KiB of warnings before the archive: without draining them at the same
        // time, the sender blocks on its error pipe and the extraction on its output.
        let script = format!(
            "i=0; while [ \"$i\" -lt 4096 ]; do echo 'warning: {}' >&2; i=$((i + 1)); done\n{}",
            "w".repeat(54),
            archive_script(&source.to_string_lossy(), &["file".into()], &[])
        );
        let local = root.path().join("back");
        let result = tokio::time::timeout(
            Duration::from_secs(30),
            download_from(sender(&script)?, &local),
        )
        .await;
        match result {
            Err(_) => return Err("the download deadlocked on the sender's warnings".into()),
            Ok(outcome) => outcome??,
        }
        assert_eq!(fs::read_to_string(local.join("file"))?, "content\n");
        Ok(())
    }

    #[test]
    fn a_receiver_killed_by_a_signal_is_reported_with_its_error_output() {
        let killed = || Err(ExecError::Ssh(openssh::Error::RemoteProcessTerminated));
        let result = upload_outcome(killed(), b"tar: write error\n", Ok(()), Ok(()));
        assert!(
            matches!(
                &result,
                Err(ExecError::Command { action: "upload", message }) if message == "tar: write error"
            ),
            "{result:?}"
        );
        let silent = upload_outcome(killed(), b" \n", Ok(()), Ok(()));
        assert!(
            matches!(
                silent,
                Err(ExecError::Ssh(openssh::Error::RemoteProcessTerminated))
            ),
            "{silent:?}"
        );
    }

    #[test]
    fn a_failed_sender_is_reported_with_its_error_output() {
        let status = std::os::unix::process::ExitStatusExt::from_raw(1 << 8);
        let result = download_outcome(Ok(()), Ok(status), b"sh: cd: can't cd to /nowhere\n");
        assert!(
            matches!(
                &result,
                Err(ExecError::Command { action: "download", message })
                    if message == "sh: cd: can't cd to /nowhere"
            ),
            "{result:?}"
        );
    }
}
