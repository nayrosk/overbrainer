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
    if let Some(mut stdin) = child.stdin.take() {
        let written = async {
            stdin.write_all(&first).await?;
            tokio::io::copy(reader, &mut stdin).await?;
            stdin.shutdown().await
        }
        .await;
        drop(stdin);
        written.map_err(io_error(dir))?;
    }
    finish(child, "extract").await
}

/// Waits for a local `tar`, turning a failure into an error carrying its stderr.
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

/// The error output of a failed command, or its exit status when it printed nothing.
pub(crate) fn failure(stderr: &[u8], status: std::process::ExitStatus) -> String {
    let text = String::from_utf8_lossy(stderr).trim().to_string();
    if text.is_empty() {
        status.to_string()
    } else {
        text
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
    use super::*;

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
