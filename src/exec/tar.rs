//! Directory copies as `tar` streams: local `tar` processes on this side, piped to
//! and from the target.

use std::fs;
use std::os::unix::fs::DirBuilderExt as _;
use std::path::{Path, PathBuf};
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

/// The entries of the local directory `dir` an upload sends: `.` (the whole
/// directory) when nothing is skipped, otherwise every top-level name of `dir`
/// but those in `skip`, sorted. Empty when nothing is left to send.
///
/// # Errors
///
/// Returns [`ExecError::Io`] when `dir` cannot be read, and
/// [`ExecError::Command`] for a top-level name that is not UTF-8, which could
/// not be passed to `tar` as is.
pub(crate) fn upload_entries(dir: &Path, skip: &[String]) -> Result<Vec<String>, ExecError> {
    let read = std::fs::read_dir(dir).map_err(io_error(dir))?;
    if skip.is_empty() {
        return Ok(vec![".".to_string()]);
    }
    let mut entries = Vec::new();
    for entry in read {
        let name = entry.map_err(io_error(dir))?.file_name();
        let name = name.into_string().map_err(|name| ExecError::Command {
            action: "upload",
            message: format!(
                "{} holds a name that is not UTF-8: {}",
                dir.display(),
                name.to_string_lossy()
            ),
        })?;
        if !skip.contains(&name) {
            entries.push(name);
        }
    }
    entries.sort();
    Ok(entries)
}

/// A local `tar` command in its own process group, as the local executor's
/// scripts are: a Ctrl-C typed in the terminal (SIGINT to the foreground
/// group, from the command line or from `$EDITOR` over the TUI) cannot kill a
/// copy in flight; only the flow copying decides when it stops.
fn tar() -> Command {
    let mut command = Command::new("tar");
    command.process_group(0);
    command
}

/// Starts a local `tar` writing `entries` of `dir` to its stdout, in its own
/// process group.
pub(crate) fn spawn_create(
    dir: &Path,
    entries: &[String],
    exclude: &[String],
) -> Result<Child, ExecError> {
    tar()
        .args(create_args(&dir.to_string_lossy(), entries, exclude))
        .env("COPYFILE_DISABLE", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(spawn_error)
}

/// Extracts the `tar` stream `reader`, sent by a target for the `entries` it was
/// asked for, into the local directory `dir`, created if needed. An empty
/// stream (nothing to copy) extracts nothing.
///
/// The stream goes to a new staging directory inside `dir` first. Only regular
/// files inside the `entries` (`.` for all) are then moved into place, each by a
/// rename, which replaces a link at its destination instead of writing through
/// it; their directories are created as needed, so an empty directory is not
/// kept and directories keep this side's modes. A symbolic link, a FIFO, a
/// device or a socket is skipped with a warning. A member outside the entries,
/// such as a `pod.json` the target was never asked for, or a destination that
/// cannot be replaced (a directory, a linked directory on the way), fails the
/// copy before any file moves. The staging directory is removed whatever
/// happens.
///
/// # Errors
///
/// Returns [`ExecError::Protocol`] for a member refused as above, and the
/// errors of the extraction and of the moves.
pub(crate) async fn extract<R: AsyncRead + Unpin>(
    reader: &mut R,
    dir: &Path,
    entries: &[String],
) -> Result<(), ExecError> {
    let mut first = vec![0_u8; 64 * 1024];
    let read = reader.read(&mut first).await.map_err(io_error(dir))?;
    if read == 0 {
        return Ok(());
    }
    first.truncate(read);
    fs::create_dir_all(dir).map_err(io_error(dir))?;
    let staging = staging_dir(dir)?;
    let mut stream = first.as_slice().chain(reader);
    let extracted = extract_raw(&mut stream, &staging).await;
    // A `.` member of the archive sets the staging directory's mode: back to
    // its owner only.
    let private = || {
        use std::os::unix::fs::PermissionsExt as _;
        fs::set_permissions(&staging, fs::Permissions::from_mode(0o700)).map_err(io_error(&staging))
    };
    let result = extracted
        .and_then(|()| private())
        .and_then(|()| promote(&staging, dir, entries));
    unlock_tree(&staging);
    if let Err(error) = fs::remove_dir_all(&staging) {
        tracing::warn!("cannot remove {}: {error}", staging.display());
    }
    result
}

/// Makes every directory under `root` owner-writable, as far as it can, so a
/// mode the archive set cannot stop the removal of a staging tree that a
/// failed extraction or a refusal left partly unvisited.
fn unlock_tree(root: &Path) {
    if owner_writable(root).is_err() {
        return;
    }
    let Ok(read) = fs::read_dir(root) else {
        return;
    };
    for child in read.filter_map(Result::ok) {
        if child.file_type().is_ok_and(|kind| kind.is_dir()) {
            unlock_tree(&child.path());
        }
    }
}

/// A new staging directory in `dir`, `.incoming-<16 hex>`, only its owner can
/// enter.
fn staging_dir(dir: &Path) -> Result<PathBuf, ExecError> {
    loop {
        let staging = dir.join(format!(".incoming-{:016x}", fastrand::u64(..)));
        match fs::DirBuilder::new().mode(0o700).create(&staging) {
            Ok(()) => return Ok(staging),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {},
            Err(error) => return Err(io_error(&staging)(error)),
        }
    }
}

/// What a member of an extracted archive may be, by where it lies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Place {
    /// Inside a requested entry, or the entry itself.
    Inside,
    /// A directory on the way to an entry, such as `.pod` for `.pod/state`.
    OnTheWay,
    /// Neither.
    Outside,
}

/// Where the member `relative` (with `/` separators) lies against `entries`.
fn place(relative: &str, entries: &[String]) -> Place {
    let mut on_the_way = false;
    for entry in entries {
        let entry = entry.trim_start_matches("./").trim_end_matches('/');
        if entry.is_empty() || entry == "." {
            return Place::Inside;
        }
        if relative == entry || relative.starts_with(&format!("{entry}/")) {
            return Place::Inside;
        }
        if entry.starts_with(&format!("{relative}/")) {
            on_the_way = true;
        }
    }
    if on_the_way {
        Place::OnTheWay
    } else {
        Place::Outside
    }
}

/// The members of the staging tree under `staging/relative`, checked: a file
/// inside the `entries`, a directory inside them or on the way to one. A
/// symbolic link, a FIFO, a device or a socket is skipped with a warning and
/// never kept: the manifest ignores them too. Adds the files, relative to
/// `staging`, to `files`. Each directory is made owner-writable first, so a
/// mode the archive set (a read-only `0555`, or `0000` for `.`) cannot stop the
/// walk, the moves or the cleanup.
fn checked_files(
    staging: &Path,
    relative: &str,
    entries: &[String],
    files: &mut Vec<String>,
) -> Result<(), ExecError> {
    let path = staging.join(relative);
    owner_writable(&path)?;
    for child in fs::read_dir(&path).map_err(io_error(&path))? {
        let child = child.map_err(io_error(&path))?;
        let name = child.file_name();
        let Some(name) = name.to_str() else {
            return Err(ExecError::Protocol(format!(
                "the copy sent a name that is not UTF-8 in {}: nothing was kept",
                crate::logging::one_line(if relative.is_empty() { "." } else { relative })
            )));
        };
        let member = if relative.is_empty() {
            name.to_string()
        } else {
            format!("{relative}/{name}")
        };
        let kind = child.file_type().map_err(io_error(&child.path()))?;
        let place = place(&member, entries);
        let refused = |why: &str| {
            ExecError::Protocol(format!(
                "the copy sent {}, which is {why}: nothing was kept",
                crate::logging::one_line(&member)
            ))
        };
        if kind.is_dir() {
            if place == Place::Outside {
                return Err(refused("outside the requested entries"));
            }
            checked_files(staging, &member, entries, files)?;
        } else if kind.is_file() {
            if place != Place::Inside {
                return Err(refused("outside the requested entries"));
            }
            files.push(member);
        } else {
            let what = if kind.is_symlink() {
                "a symbolic link"
            } else {
                "not a file or a directory"
            };
            tracing::warn!(
                "the copy sent {}, which is {what}: not kept",
                crate::logging::one_line(&member)
            );
        }
    }
    Ok(())
}

/// Adds the owner's read, write and search bits to the directory `dir`.
fn owner_writable(dir: &Path) -> Result<(), ExecError> {
    use std::os::unix::fs::PermissionsExt as _;

    let mode = fs::symlink_metadata(dir)
        .map_err(io_error(dir))?
        .permissions()
        .mode();
    if mode & 0o700 != 0o700 {
        fs::set_permissions(dir, fs::Permissions::from_mode(mode | 0o700))
            .map_err(io_error(dir))?;
    }
    Ok(())
}

/// Moves the checked files of `staging` into `dir`, creating their
/// directories. A directory on the way that is a symbolic link, or anything
/// but a directory, in `dir` is refused before any file moves, so no file
/// lands outside `dir`.
fn promote(staging: &Path, dir: &Path, entries: &[String]) -> Result<(), ExecError> {
    let mut files = Vec::new();
    checked_files(staging, "", entries, &mut files)?;
    for file in &files {
        for parent in parents(dir, file) {
            match fs::symlink_metadata(&parent) {
                Ok(metadata) if metadata.is_dir() => {},
                Ok(_) => return Err(not_kept(&parent, "is not a directory")),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
                Err(error) => return Err(io_error(&parent)(error)),
            }
        }
        let to = dir.join(file);
        match fs::symlink_metadata(&to) {
            Ok(metadata) if metadata.is_dir() => {
                return Err(not_kept(&to, "is a directory"));
            },
            Ok(_) => {},
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
            Err(error) => return Err(io_error(&to)(error)),
        }
    }
    for file in &files {
        for parent in parents(dir, file) {
            match fs::create_dir(&parent) {
                Ok(()) => {},
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {},
                Err(error) => return Err(io_error(&parent)(error)),
            }
        }
        move_into(&staging.join(file), &dir.join(file))?;
    }
    Ok(())
}

/// The refusal of a copy because the destination `path` of one of its files
/// `is` something it cannot replace.
fn not_kept(path: &Path, is: &str) -> ExecError {
    ExecError::Protocol(format!(
        "{} {is}: the copy was not kept",
        crate::logging::one_line(&path.display().to_string())
    ))
}

/// Moves the staged file `from` to `to` with a rename, which replaces a link at
/// `to` instead of writing through it. When `to` is on another file system (a
/// mount inside the run directory), the file is copied next to `to` under a
/// new name first, then renamed over it.
fn move_into(from: &Path, to: &Path) -> Result<(), ExecError> {
    match fs::rename(from, to) {
        Err(error) if error.raw_os_error() == Some(rustix::io::Errno::XDEV.raw_os_error()) => {
            let parent = to.parent().unwrap_or_else(|| Path::new("."));
            let name = to
                .file_name()
                .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
            let tmp = parent.join(format!(".{name}.{:016x}.tmp", fastrand::u64(..)));
            let copied = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp)
                .and_then(|mut out| std::io::copy(&mut fs::File::open(from)?, &mut out))
                .and_then(|_| fs::rename(&tmp, to));
            if copied.is_err() {
                fs::remove_file(&tmp).ok();
            }
            copied.map_err(io_error(to))
        },
        other => other.map_err(io_error(to)),
    }
}

/// The directories of `dir` on the way to its file `file` (`/` separated),
/// nearest to `dir` first: `dir/a` and `dir/a/b` for `a/b/c`.
fn parents(dir: &Path, file: &str) -> Vec<PathBuf> {
    let parts: Vec<&str> = file.split('/').collect();
    let mut parent = dir.to_path_buf();
    parts
        .iter()
        .take(parts.len().saturating_sub(1))
        .map(|part| {
            parent.push(part);
            parent.clone()
        })
        .collect()
}

/// Extracts the `tar` stream `reader` into the local directory `dir`, created if
/// needed, with a `tar` in its own process group. An empty stream (nothing to
/// copy) extracts nothing. Every member lands as sent, so use it only for a
/// stream this side made, such as an upload to a local target. A download goes
/// through [`extract`].
pub(crate) async fn extract_raw<R: AsyncRead + Unpin>(
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
    let mut child = tar()
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
    error_text(stderr).unwrap_or_else(|| status.to_string())
}

/// The error output of a command, capped as [`failure`] describes, or `None` when
/// it printed nothing but whitespace.
pub(crate) fn error_text(stderr: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(stderr);
    let text = text.trim();
    if text.is_empty() {
        return None;
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
    Some(match total.saturating_sub(MAX_ERROR_LINES) {
        0 => head,
        1 => format!("{head}\n(1 more line)"),
        more => format!("{head}\n({more} more lines)"),
    })
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

    /// The `tar` archive of the `entries` of `dir`, as a target sends it.
    async fn archive_of(
        dir: &Path,
        entries: &[&str],
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let entries: Vec<String> = entries.iter().map(|entry| (*entry).to_string()).collect();
        let mut create = spawn_create(dir, &entries, &[])?;
        let mut archive = Vec::new();
        if let Some(mut stdout) = create.stdout.take() {
            stdout.read_to_end(&mut archive).await?;
        }
        finish(create, "archive").await?;
        Ok(archive)
    }

    /// Whether `dir` holds a staging directory left behind.
    fn staging_left(dir: &Path) -> Result<bool, Box<dyn std::error::Error>> {
        Ok(fs::read_dir(dir)?.filter_map(Result::ok).any(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with(".incoming-")
        }))
    }

    /// The requested files and directories of a download land in place, and
    /// the staging directory is gone.
    #[tokio::test]
    async fn a_download_keeps_the_requested_entries() -> TestResult {
        let root = tempdir()?;
        let source = root.path().join("source");
        fs::create_dir_all(source.join("output/gguf"))?;
        fs::write(source.join("output/gguf/m.gguf"), "gguf")?;
        fs::write(source.join("job.log"), "log")?;
        fs::create_dir_all(source.join(".pod"))?;
        fs::write(source.join(".pod/state"), "s")?;
        let archive = archive_of(&source, &["output", "job.log", ".pod/state"]).await?;
        let local = root.path().join("run");
        let entries = ["output".to_string(), "job.log".into(), ".pod/state".into()];
        extract(&mut archive.as_slice(), &local, &entries).await?;
        assert_eq!(
            fs::read_to_string(local.join("output/gguf/m.gguf"))?,
            "gguf"
        );
        assert_eq!(fs::read_to_string(local.join("job.log"))?, "log");
        assert_eq!(fs::read_to_string(local.join(".pod/state"))?, "s");
        assert!(!staging_left(&local)?);
        Ok(())
    }

    /// A member the download was not asked for, such as a `pod.json`, fails it,
    /// and nothing it sent is kept, not even the requested files.
    #[tokio::test]
    async fn an_unrequested_member_fails_the_download() -> TestResult {
        let root = tempdir()?;
        let source = root.path().join("source");
        fs::create_dir_all(source.join("output"))?;
        fs::write(source.join("output/adapter.bin"), "w")?;
        fs::write(source.join("pod.json"), "{\"forged\": true}")?;
        let archive = archive_of(&source, &["output", "pod.json"]).await?;
        let local = root.path().join("run");
        fs::create_dir_all(&local)?;
        fs::write(local.join("pod.json"), "mine")?;
        let result = extract(&mut archive.as_slice(), &local, &["output".to_string()]).await;
        let Err(ExecError::Protocol(message)) = result else {
            return Err(format!("expected a refusal, got {result:?}").into());
        };
        assert!(message.contains("pod.json"), "{message}");
        assert_eq!(fs::read_to_string(local.join("pod.json"))?, "mine");
        assert!(!local.join("output").exists(), "nothing is kept");
        assert!(!staging_left(&local)?);
        Ok(())
    }

    /// A symbolic link in a download is skipped, never kept, and the rest of
    /// the download is.
    #[tokio::test]
    async fn a_link_is_skipped() -> TestResult {
        let root = tempdir()?;
        let member = prepare_member(root.path(), "link")?;
        std::os::unix::fs::symlink("/etc/passwd", &member)?;
        assert_member_skipped(root.path(), "link").await
    }

    /// A FIFO in a download is skipped, never kept, and the rest of the
    /// download is. Linux only: `mkfifoat` is not available on macOS.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn a_fifo_is_skipped() -> TestResult {
        let root = tempdir()?;
        let member = prepare_member(root.path(), "fifo")?;
        rustix::fs::mkfifoat(
            rustix::fs::CWD,
            &member,
            rustix::fs::Mode::from_raw_mode(0o600),
        )?;
        assert_member_skipped(root.path(), "fifo").await
    }

    /// Writes `source-<kind>/output/adapter.bin` under `root` and returns the
    /// path where the test plants its special `output/member`.
    fn prepare_member(root: &Path, kind: &str) -> std::io::Result<PathBuf> {
        let source = root.join(format!("source-{kind}"));
        fs::create_dir_all(source.join("output"))?;
        fs::write(source.join("output/adapter.bin"), "w")?;
        Ok(source.join("output/member"))
    }

    /// Archives `source-<kind>/output`, extracts it into `run-<kind>` and checks
    /// that the special member is gone while the regular file came through.
    async fn assert_member_skipped(root: &Path, kind: &str) -> TestResult {
        let archive = archive_of(&root.join(format!("source-{kind}")), &["output"]).await?;
        let local = root.join(format!("run-{kind}"));
        extract(&mut archive.as_slice(), &local, &["output".to_string()]).await?;
        assert!(
            fs::symlink_metadata(local.join("output/member")).is_err(),
            "{kind}"
        );
        assert_eq!(fs::read_to_string(local.join("output/adapter.bin"))?, "w");
        assert!(!staging_left(&local)?, "{kind}");
        Ok(())
    }

    /// A refused read-only directory, which the walk never enters, leaves no
    /// staging directory behind.
    #[tokio::test]
    async fn a_refused_read_only_directory_leaves_no_staging() -> TestResult {
        let root = tempdir()?;
        let source = root.path().join("source");
        fs::create_dir_all(source.join("output"))?;
        fs::write(source.join("output/f"), "f")?;
        fs::create_dir_all(source.join("extra"))?;
        fs::write(source.join("extra/f"), "f")?;
        fs::set_permissions(source.join("extra"), fs::Permissions::from_mode(0o555))?;
        let archive = archive_of(&source, &["extra", "output"]).await;
        fs::set_permissions(source.join("extra"), fs::Permissions::from_mode(0o755))?;
        let local = root.path().join("run");
        let result = extract(&mut archive?.as_slice(), &local, &["output".to_string()]).await;
        assert!(matches!(result, Err(ExecError::Protocol(_))), "{result:?}");
        assert!(!staging_left(&local)?);
        Ok(())
    }

    /// A hard link whose target climbs out of the staging directory, with or
    /// without data of its own, fails the download without touching the file
    /// it names: the local `tar` (GNU tar or bsdtar) refuses or rewrites such
    /// a target.
    #[tokio::test]
    async fn a_hard_link_out_of_the_staging_directory_touches_nothing() -> TestResult {
        let root = tempdir()?;
        let local = root.path().join("run");
        fs::create_dir_all(&local)?;
        let pod = local.join("pod.json");
        fs::write(&pod, "mine")?;
        let mut archive = Vec::new();
        archive.extend(ustar_header("output/", b'5', "", 0)?);
        archive.extend(ustar_header("output/up", b'1', "../pod.json", 0)?);
        archive.extend(ustar_header("output/data", b'1', "../pod.json", 5)?);
        archive.extend(b"PWNED".iter().copied().chain(std::iter::repeat_n(0, 507)));
        archive.extend(std::iter::repeat_n(0, 1024));
        let result = extract(&mut archive.as_slice(), &local, &["output".to_string()]).await;
        assert!(result.is_err(), "{result:?}");
        assert_eq!(fs::read_to_string(&pod)?, "mine");
        assert_eq!(
            std::os::unix::fs::MetadataExt::nlink(&fs::metadata(&pod)?),
            1
        );
        assert!(!staging_left(&local)?);
        Ok(())
    }

    /// A ustar header for the member `name` of type `kind`, linked to `link`,
    /// with `size` bytes of data following it.
    fn ustar_header(
        name: &str,
        kind: u8,
        link: &str,
        size: u64,
    ) -> Result<[u8; 512], Box<dyn std::error::Error>> {
        if name.len() > 100 || link.len() > 100 {
            return Err(format!("{name} or its link is too long for a ustar header").into());
        }
        let mut header = [0_u8; 512];
        let mut put = |at: usize, len: usize, value: &[u8]| {
            let end = at + value.len().min(len);
            header[at..end].copy_from_slice(&value[..end - at]);
        };
        put(0, 100, name.as_bytes());
        put(100, 8, b"0000644\0");
        put(108, 8, b"0000000\0");
        put(116, 8, b"0000000\0");
        put(124, 12, format!("{size:011o}\0").as_bytes());
        put(136, 12, b"00000000000\0");
        put(148, 8, b"        ");
        put(156, 1, &[kind]);
        put(157, 100, link.as_bytes());
        put(257, 8, b"ustar\x0000");
        let sum: u32 = header.iter().map(|byte| u32::from(*byte)).sum();
        header[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
        Ok(header)
    }

    /// A destination that is a directory, where the download sends a file,
    /// fails it before any file moves.
    #[tokio::test]
    async fn a_directory_in_the_way_fails_the_download_before_any_move() -> TestResult {
        let root = tempdir()?;
        let source = root.path().join("source");
        fs::create_dir_all(source.join("output"))?;
        fs::write(source.join("output/a"), "a")?;
        fs::write(source.join("output/b"), "b")?;
        let archive = archive_of(&source, &["output"]).await?;
        let local = root.path().join("run");
        fs::create_dir_all(local.join("output/b"))?;
        let result = extract(&mut archive.as_slice(), &local, &["output".to_string()]).await;
        assert!(matches!(result, Err(ExecError::Protocol(_))), "{result:?}");
        assert!(!local.join("output/a").exists(), "nothing moved");
        Ok(())
    }

    /// A download whose archive holds read-only directories, and a `.` member
    /// that locks the staging directory, is kept and leaves no staging behind.
    #[tokio::test]
    async fn read_only_directories_in_a_download_are_kept() -> TestResult {
        use std::os::unix::fs::PermissionsExt as _;

        let root = tempdir()?;
        let source = root.path().join("source");
        fs::create_dir_all(source.join("output/sub"))?;
        fs::write(source.join("output/sub/f"), "f")?;
        fs::set_permissions(source.join("output/sub"), fs::Permissions::from_mode(0o555))?;
        let archive = archive_of(&source, &["."]).await;
        fs::set_permissions(source.join("output/sub"), fs::Permissions::from_mode(0o755))?;
        let local = root.path().join("run");
        extract(&mut archive?.as_slice(), &local, &[".".to_string()]).await?;
        assert_eq!(fs::read_to_string(local.join("output/sub/f"))?, "f");
        assert!(!staging_left(&local)?);
        Ok(())
    }

    /// Hard-linked files in an archive land as regular files.
    #[tokio::test]
    async fn hard_links_land_as_regular_files() -> TestResult {
        let root = tempdir()?;
        let source = root.path().join("source");
        fs::create_dir_all(source.join("output"))?;
        fs::write(source.join("output/a"), "same")?;
        fs::hard_link(source.join("output/a"), source.join("output/b"))?;
        let archive = archive_of(&source, &["output"]).await?;
        let local = root.path().join("run");
        extract(&mut archive.as_slice(), &local, &["output".to_string()]).await?;
        assert_eq!(fs::read_to_string(local.join("output/b"))?, "same");

        Ok(())
    }

    /// A name that is not UTF-8 fails the download with nothing kept. Linux
    /// only: APFS refuses such a name before the test can make it.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn non_utf8_names_fail_the_download() -> TestResult {
        use std::os::unix::ffi::OsStrExt as _;

        let root = tempdir()?;
        let odd = root.path().join("odd");
        fs::create_dir_all(odd.join("output"))?;
        let name = std::ffi::OsStr::from_bytes(b"bad\xff");
        fs::write(odd.join("output").join(name), "x")?;
        fs::write(odd.join("output/good"), "g")?;
        let archive = archive_of(&odd, &["output"]).await?;
        let back = root.path().join("back");
        let result = extract(&mut archive.as_slice(), &back, &["output".to_string()]).await;
        assert!(matches!(result, Err(ExecError::Protocol(_))), "{result:?}");
        assert!(!back.join("output/good").exists());
        Ok(())
    }

    /// An empty stream extracts nothing and creates nothing.
    #[tokio::test]
    async fn an_empty_download_creates_nothing() -> TestResult {
        let root = tempdir()?;
        let local = root.path().join("run");
        extract(&mut [].as_slice(), &local, &["output".to_string()]).await?;
        assert!(!local.exists());
        Ok(())
    }

    /// A link already at a destination is replaced, and its target is left
    /// as it was; a linked directory on the way is refused.
    #[tokio::test]
    async fn a_link_at_the_destination_is_replaced_not_followed() -> TestResult {
        let root = tempdir()?;
        let source = root.path().join("source");
        fs::create_dir_all(source.join("output"))?;
        fs::write(source.join("output/config"), "new")?;
        let archive = archive_of(&source, &["output"]).await?;
        let outside = root.path().join("outside");
        fs::create_dir_all(&outside)?;
        fs::write(outside.join("config"), "keep")?;
        let local = root.path().join("run");
        fs::create_dir_all(local.join("output"))?;
        std::os::unix::fs::symlink(outside.join("config"), local.join("output/config"))?;
        extract(&mut archive.as_slice(), &local, &["output".to_string()]).await?;
        assert_eq!(fs::read_to_string(outside.join("config"))?, "keep");
        assert_eq!(fs::read_to_string(local.join("output/config"))?, "new");
        let linked = root.path().join("linked");
        fs::create_dir_all(&linked)?;
        std::os::unix::fs::symlink(&outside, linked.join("output"))?;
        let result = extract(&mut archive.as_slice(), &linked, &["output".to_string()]).await;
        assert!(matches!(result, Err(ExecError::Protocol(_))), "{result:?}");
        assert_eq!(fs::read_to_string(outside.join("config"))?, "keep");
        Ok(())
    }

    /// Where a member lies against the requested entries.
    #[test]
    fn members_are_placed_against_the_entries() {
        let entries = [
            "output".to_string(),
            ".pod/state".into(),
            "./job.log".into(),
        ];
        assert_eq!(place("output", &entries), Place::Inside);
        assert_eq!(place("output/a/b", &entries), Place::Inside);
        assert_eq!(place("job.log", &entries), Place::Inside);
        assert_eq!(place(".pod", &entries), Place::OnTheWay);
        assert_eq!(place(".pod/state", &entries), Place::Inside);
        assert_eq!(place(".pod/other", &entries), Place::Outside);
        assert_eq!(place("outputs", &entries), Place::Outside);
        assert_eq!(place("pod.json", &entries), Place::Outside);
        assert_eq!(place("anything", &[".".to_string()]), Place::Inside);
    }

    /// The process group of process `pid`, from `/proc`.
    fn process_group(pid: u32) -> Result<u32, Box<dyn std::error::Error>> {
        let stat = fs::read_to_string(format!("/proc/{pid}/stat"))?;
        // `pid (comm) state ppid pgrp ...`: the fields after the command name.
        let after = stat.rsplit_once(')').ok_or("no command name")?.1;
        let pgrp = after.split_whitespace().nth(2).ok_or("no process group")?;
        Ok(pgrp.parse()?)
    }

    /// A copy's `tar` leads its own process group, so a SIGINT sent to the
    /// terminal's foreground group does not reach it.
    #[tokio::test]
    async fn tar_runs_in_its_own_process_group() -> TestResult {
        let dir = tempdir()?;
        let child = spawn_create(dir.path(), &[".".to_string()], &[])?;
        let pid = child.id().ok_or("tar exited")?;
        assert_eq!(process_group(pid)?, pid);
        finish(child, "upload").await?;
        Ok(())
    }

    fn exit(code: i32) -> std::process::ExitStatus {
        std::os::unix::process::ExitStatusExt::from_raw(code << 8)
    }

    #[test]
    fn an_upload_sends_everything_but_the_skipped_top_level_names() -> TestResult {
        let dir = tempdir()?;
        fs::create_dir_all(dir.path().join("ssh"))?;
        fs::create_dir_all(dir.path().join("data/ssh"))?;
        fs::write(dir.path().join("pod.json"), "{}")?;
        fs::write(dir.path().join("run.json"), "{}")?;
        let skip = ["ssh".to_string(), "pod.json".to_string()];
        assert_eq!(upload_entries(dir.path(), &skip)?, vec!["data", "run.json"]);
        assert_eq!(upload_entries(dir.path(), &[])?, vec!["."]);
        assert!(matches!(
            upload_entries(&dir.path().join("missing"), &skip),
            Err(ExecError::Io { .. })
        ));
        Ok(())
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

    /// A tar that dies early, here on a directory it cannot enter, is reported
    /// with its own error rather than the feeder's broken pipe.
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
        let result = extract_raw(&mut archive.as_slice(), &locked).await;
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

    /// A flood of tar warnings never fills the error pipe and stalls the
    /// extraction.
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
            extract_raw(&mut archive.as_slice(), &target),
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
