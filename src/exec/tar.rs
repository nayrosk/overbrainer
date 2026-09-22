//! Directory copies as `tar` streams: local `tar` processes on this side, piped to
//! and from the target.

use std::path::Path;
use std::process::Stdio;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};

use super::ExecError;

/// Arguments of `tar` (after the program name) writing `entries` of `dir` to
/// stdout, without the names matching an `exclude` pattern.
#[must_use]
pub(crate) fn create_args(dir: &str, entries: &[String], exclude: &[String]) -> Vec<String> {
    let mut args = vec!["-C".to_string(), dir.to_string(), "-cf".into(), "-".into()];
    args.extend(exclude.iter().map(|pattern| format!("--exclude={pattern}")));
    args.push("--".into());
    args.extend(entries.iter().cloned());
    args
}

/// Starts a local `tar` writing `entries` of `dir` to its stdout.
pub(crate) fn spawn_create(
    dir: &Path,
    entries: &[String],
    exclude: &[String],
) -> Result<Child, ExecError> {
    Command::new("tar")
        .args(create_args(&dir.to_string_lossy(), entries, exclude))
        .env("COPYFILE_DISABLE", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(spawn_error)
}

/// Extracts the `tar` stream `reader` into the local directory `dir`, created if
/// needed. An empty stream (nothing to copy) extracts nothing.
pub(crate) async fn extract<R: AsyncRead + Unpin>(
    reader: &mut R,
    dir: &Path,
) -> Result<(), ExecError> {
    let mut first = vec![0_u8; 64 * 1024];
    let read = reader.read(&mut first).await.map_err(io_error(dir))?;
    if read == 0 {
        return Ok(());
    }
    first.truncate(read);
    std::fs::create_dir_all(dir).map_err(io_error(dir))?;
    let mut child = Command::new("tar")
        .arg("-C")
        .arg(dir)
        .args(["-xf", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(spawn_error)?;
    let stdin = child.stdin.take();
    // Fed while `finish` drains tar's error output, so a flood of warnings cannot
    // fill that pipe and stall tar. `stdin` is dropped when the feeding ends, which
    // is how tar sees the end of the stream.
    let feed = async move {
        let Some(mut stdin) = stdin else {
            return Ok(());
        };
        stdin.write_all(&first).await?;
        tokio::io::copy(reader, &mut stdin).await?;
        stdin.shutdown().await
    };
    let (written, finished) = tokio::join!(feed, finish(child, "extract"));
    // When tar dies early, feeding it fails with a broken pipe: its own error says why.
    finished?;
    written.map_err(io_error(dir))
}

/// Waits for a local `tar` while reading its error output, turning a failure into
/// an error carrying that output. Run it alongside whatever feeds or drains the
/// process's other pipes, not after: otherwise enough warnings fill the error pipe
/// and `tar` stalls.
pub(crate) async fn finish(child: Child, action: &'static str) -> Result<(), ExecError> {
    let output = child.wait_with_output().await.map_err(spawn_error)?;
    if output.status.success() {
        Ok(())
    } else {
        Err(ExecError::Command {
            action,
            message: failure(&output.stderr, output.status),
        })
    }
}

/// Lines of a command's error output kept in an error message.
const MAX_ERROR_LINES: usize = 20;
/// Bytes of a command's error output kept in an error message.
const MAX_ERROR_BYTES: usize = 4096;

/// The error output of a failed command, or its exit status when it printed nothing.
/// Only its first [`MAX_ERROR_LINES`] lines are kept, followed by how many more there
/// were, and at most [`MAX_ERROR_BYTES`] of them (cut at a character boundary and
/// marked with `...`), so a flood of warnings cannot bloat the error.
pub(crate) fn failure(stderr: &[u8], status: std::process::ExitStatus) -> String {
    let text = String::from_utf8_lossy(stderr);
    let text = text.trim();
    if text.is_empty() {
        return status.to_string();
    }
    let total = text.lines().count();
    let mut head = text
        .lines()
        .take(MAX_ERROR_LINES)
        .collect::<Vec<_>>()
        .join("\n");
    if head.len() > MAX_ERROR_BYTES {
        let cut = (0..=MAX_ERROR_BYTES)
            .rev()
            .find(|index| head.is_char_boundary(*index))
            .unwrap_or(0);
        head.truncate(cut);
        head.push_str("...");
    }
    match total.saturating_sub(MAX_ERROR_LINES) {
        0 => head,
        1 => format!("{head}\n(1 more line)"),
        more => format!("{head}\n({more} more lines)"),
    }
}

fn spawn_error(source: std::io::Error) -> ExecError {
    ExecError::Spawn {
        program: "tar".to_string(),
        source,
    }
}

fn io_error(path: &Path) -> impl FnOnce(std::io::Error) -> ExecError + '_ {
    move |source| ExecError::Io {
        path: path.to_path_buf(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::time::Duration;

    use tempfile::tempdir;

    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn exit(code: i32) -> std::process::ExitStatus {
        std::os::unix::process::ExitStatusExt::from_raw(code << 8)
    }

    #[test]
    fn failure_keeps_the_first_lines_and_counts_the_rest() {
        let text: Vec<String> = (1..=25).map(|n| format!("warning {n}")).collect();
        let message = failure(text.join("\n").as_bytes(), exit(2));
        let expected: Vec<String> = (1..=20).map(|n| format!("warning {n}")).collect();
        assert_eq!(message, format!("{}\n(5 more lines)", expected.join("\n")));

        let one_more: Vec<String> = (1..=21).map(|n| format!("w{n}")).collect();
        assert!(failure(one_more.join("\n").as_bytes(), exit(2)).ends_with("\nw20\n(1 more line)"));

        let exactly: Vec<String> = (1..=20).map(|n| format!("w{n}")).collect();
        assert_eq!(
            failure(exactly.join("\n").as_bytes(), exit(2)),
            exactly.join("\n")
        );
    }

    #[test]
    fn failure_caps_its_bytes_at_a_char_boundary() {
        // One huge line of 3 byte characters: the cap falls inside a character.
        let line = "\u{20ac}".repeat(10_000);
        let message = failure(line.as_bytes(), exit(2));
        assert!(
            message.len() <= MAX_ERROR_BYTES + 3,
            "{} bytes",
            message.len()
        );
        assert!(message.ends_with("..."), "{message}");
        let kept = message.trim_end_matches("...");
        assert!(kept.chars().all(|c| c == '\u{20ac}'));
    }

    #[test]
    fn failure_falls_back_to_the_exit_status() {
        assert_eq!(failure(b"  \n", exit(2)), exit(2).to_string());
        assert_eq!(failure(b" tar: boom \n", exit(2)), "tar: boom");
    }

    /// A name long enough that one `tar` warning about it takes a few hundred bytes.
    fn long_name(index: usize) -> String {
        format!("{index:05}-{}", "n".repeat(150))
    }

    /// Whether permissions are enforced for this user (they are not for root).
    fn permissions_enforced(locked: &Path) -> bool {
        fs::read_dir(locked).is_err()
    }

    #[tokio::test]
    async fn extract_reports_the_tar_error_when_tar_dies_early() -> TestResult {
        let root = tempdir()?;
        let locked = root.path().join("locked");
        fs::create_dir(&locked)?;
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o000))?;
        if !permissions_enforced(&locked) {
            fs::set_permissions(&locked, fs::Permissions::from_mode(0o755))?;
            eprintln!("skipped: permissions are not enforced for this user");
            return Ok(());
        }
        // A real archive (tar only enters the directory for its first member) far
        // larger than a pipe holds, so feeding it fails once tar has exited.
        let source = root.path().join("source");
        fs::create_dir(&source)?;
        fs::write(source.join("big"), vec![b'x'; 4 * 1024 * 1024])?;
        let entries = ["big".to_string()];
        let mut create = spawn_create(&source, &entries, &[])?;
        let mut archive = Vec::new();
        if let Some(mut stdout) = create.stdout.take() {
            stdout.read_to_end(&mut archive).await?;
        }
        finish(create, "archive").await?;
        let result = extract(&mut archive.as_slice(), &locked).await;
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o755))?;
        match result {
            Err(ExecError::Command { action, message }) => {
                assert_eq!(action, "extract");
                assert!(message.contains("Permission denied"), "{message}");
            },
            other => return Err(format!("expected tar's own error, got {other:?}").into()),
        }
        Ok(())
    }

    #[tokio::test]
    async fn extract_survives_a_flood_of_tar_warnings() -> TestResult {
        let root = tempdir()?;
        let source = root.path().join("source");
        fs::create_dir(&source)?;
        let names: Vec<String> = (0..3000).map(long_name).collect();
        for name in &names {
            fs::write(source.join(name), "")?;
        }
        let mut create = spawn_create(&source, &names, &[])?;
        let mut archive = Vec::new();
        if let Some(mut stdout) = create.stdout.take() {
            stdout.read_to_end(&mut archive).await?;
        }
        finish(create, "archive").await?;

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
            extract(&mut archive.as_slice(), &target),
        )
        .await;
        fs::set_permissions(&target, fs::Permissions::from_mode(0o755))?;
        match result {
            Err(_) => return Err("extract deadlocked on tar's error output".into()),
            Ok(Err(ExecError::Command { message, .. })) => {
                assert!(message.contains("more lines)"), "{message}");
                assert!(message.len() < 8 * 1024, "{} bytes", message.len());
            },
            Ok(other) => return Err(format!("expected tar's warnings, got {other:?}").into()),
        }
        Ok(())
    }

    #[test]
    fn create_args_exclude_before_the_entries() {
        assert_eq!(
            create_args(
                "/w",
                &["output".into(), "m.jsonl".into()],
                &["checkpoint-*".into()]
            ),
            vec![
                "-C",
                "/w",
                "-cf",
                "-",
                "--exclude=checkpoint-*",
                "--",
                "output",
                "m.jsonl"
            ]
        );
    }
}
