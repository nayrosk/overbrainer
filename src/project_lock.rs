//! One overbrainer process per project: an exclusive lock on `.overbrainer/lock`,
//! held until the process exits. The OS releases it when the process dies.

use std::fmt;
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{self, ErrorKind, Read as _, Seek as _, SeekFrom, Write as _};
use std::os::unix::fs::MetadataExt as _;
use std::path::{Path, PathBuf};

/// Directory of the project's own state, next to `overbrainer.toml`.
pub const STATE_DIR: &str = ".overbrainer";

const LOCK_FILE: &str = "lock";

/// Why the project lock cannot be taken.
#[derive(Debug, thiserror::Error)]
pub enum LockError {
    /// Another process holds it.
    #[error("another overbrainer ({}) is using this project", holder(*.pid))]
    Held {
        /// The PID the holder wrote, when readable.
        pid: Option<u32>,
    },
    /// The state directory or the lock file is not safe to treat as this
    /// process's own lock file: a symbolic link, a hard link to another file, not
    /// a regular file, or swapped for one of those between the check and the
    /// open.
    #[error("refusing to use {}: it is not a safe lock path", path.display())]
    UnsafePath {
        /// The unsafe path.
        path: PathBuf,
    },
    /// The lock file cannot be created, read or locked.
    #[error("cannot lock {}", path.display())]
    Io {
        /// The lock file.
        path: PathBuf,
        /// The I/O error.
        source: io::Error,
    },
}

/// Why an [`io::Error`] from [`check_path`] or [`verify_opened`] was raised: the
/// path is not safe to treat as a single, owned file.
#[derive(Debug)]
struct UnsafePath(PathBuf);

impl fmt::Display for UnsafePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "refusing to use {}: it is not a safe path",
            self.0.display()
        )
    }
}

impl std::error::Error for UnsafePath {}

/// An [`io::Error`] naming `path` as unsafe: [`check_path`] and [`verify_opened`]
/// return this to signal a symbolic link, a hard link, or the like, as opposed to
/// an ordinary I/O failure.
fn unsafe_path_error(path: &Path) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, UnsafePath(path.to_path_buf()))
}

/// The path named by `error`, when `error` came from [`check_path`] or
/// [`verify_opened`] rejecting an unsafe path, rather than from an ordinary I/O
/// failure.
fn unsafe_path_of(error: &io::Error) -> Option<PathBuf> {
    if error.kind() != ErrorKind::InvalidInput {
        return None;
    }
    error
        .get_ref()?
        .downcast_ref::<UnsafePath>()
        .map(|unsafe_path| unsafe_path.0.clone())
}

/// Whether `path` is a symbolic link; a missing path is not.
fn is_symlink(path: &Path) -> io::Result<bool> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => Ok(metadata.file_type().is_symlink()),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

/// Refuses `path` when it is currently a symbolic link; a missing path is fine,
/// since `open` will create it.
///
/// This narrows the window before the file is opened but does not close it: the
/// path could still be swapped for a symbolic link between this check and the
/// `open` call. [`verify_opened`] closes that window.
pub(crate) fn check_path(path: &Path) -> io::Result<()> {
    if is_symlink(path)? {
        return Err(unsafe_path_error(path));
    }
    Ok(())
}

/// Verifies that `file`, already opened from `path` inside `dir`, is truly the
/// file at `path`: not a symbolic link, not a hard link to another file, and not
/// swapped for either between the earlier [`check_path`] calls and the `open`
/// that produced `file`. Also re-checks `dir`, which could have been swapped for
/// a symbolic link in that same window.
///
/// `open` follows symbolic links, so a check made before opening cannot by itself
/// rule out a link swapped in during that window. This closes the window by
/// comparing what was actually opened (`fstat`, via [`File::metadata`]) against
/// what is on disk right now (`lstat`, via [`std::fs::symlink_metadata`]).
pub(crate) fn verify_opened(dir: &Path, path: &Path, file: &File) -> io::Result<()> {
    if is_symlink(dir)? {
        return Err(unsafe_path_error(dir));
    }

    let opened = file.metadata()?;
    // A file with more than one hard link is also reachable through another path;
    // refusing it rules out the lock file being a second name for a file we do not
    // own, even though a hard link is not a symbolic link.
    if !opened.is_file() || opened.nlink() != 1 {
        return Err(unsafe_path_error(path));
    }

    // `lstat` the path itself: it must still be a plain file, never a symlink, and
    // it must be the very inode `file` was opened from. A mismatch means `path` was
    // replaced after the open, and `file` is not the file the caller expects to
    // manage.
    let on_disk = std::fs::symlink_metadata(path)?;
    if on_disk.file_type().is_symlink() {
        return Err(unsafe_path_error(path));
    }
    if on_disk.dev() != opened.dev() || on_disk.ino() != opened.ino() {
        return Err(unsafe_path_error(path));
    }

    Ok(())
}

fn holder(pid: Option<u32>) -> String {
    pid.map_or_else(|| "unknown pid".to_string(), |pid| format!("pid {pid}"))
}

/// The held lock. Dropping it releases the lock.
#[derive(Debug)]
pub struct ProjectLock {
    _file: File,
}

impl ProjectLock {
    /// Takes the lock of `project_dir`, creating `.overbrainer/` when needed, and
    /// writes this process's PID into the lock file.
    ///
    /// # Errors
    ///
    /// [`LockError::Held`] when another process holds it, [`LockError::UnsafePath`]
    /// when `.overbrainer/` or its lock file is not a safe path (a symbolic link, a
    /// hard link to another file, not a regular file, or swapped for one of those
    /// after the initial check), [`LockError::Io`] when the file cannot be created,
    /// locked or written.
    pub fn acquire(project_dir: &Path) -> Result<Self, LockError> {
        let dir = project_dir.join(STATE_DIR);
        let path = dir.join(LOCK_FILE);
        let io = |source| LockError::Io {
            path: path.clone(),
            source,
        };
        let unsafe_or_io = |source: io::Error| match unsafe_path_of(&source) {
            Some(path) => LockError::UnsafePath { path },
            None => io(source),
        };
        // Never the project directory itself: only its state directory.
        match std::fs::create_dir(&dir) {
            Err(e) if e.kind() != ErrorKind::AlreadyExists => return Err(io(e)),
            _ => {},
        }
        // Never follow a symbolic link: it could point the lock, and the PID written
        // into it, at any file. This first check narrows the window but does not
        // close it; `open` below follows symlinks, so the path could still be
        // swapped for one between this check and the open.
        for link in [&dir, &path] {
            check_path(link).map_err(&unsafe_or_io)?;
        }
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(io)?;
        // Close the race: verify the file just opened is really the one at `path`,
        // and only that one, before locking or writing anything into it. Without
        // this, a symlink or hard link swapped in after the check above would let
        // `set_len(0)` and the PID write land on an unrelated file.
        verify_opened(&dir, &path, &file).map_err(&unsafe_or_io)?;
        match file.try_lock() {
            Ok(()) => {},
            Err(TryLockError::WouldBlock) => {
                let mut text = String::new();
                let pid = file
                    .read_to_string(&mut text)
                    .ok()
                    .and_then(|_| text.trim().parse().ok());
                return Err(LockError::Held { pid });
            },
            Err(TryLockError::Error(source)) => return Err(io(source)),
        }
        file.set_len(0).map_err(io)?;
        file.seek(SeekFrom::Start(0)).map_err(io)?;
        write!(file, "{}", std::process::id()).map_err(io)?;
        file.flush().map_err(io)?;
        Ok(Self { _file: file })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn a_second_acquire_is_refused_with_the_holder_pid() -> TestResult {
        let dir = tempfile::tempdir()?;
        let _held = ProjectLock::acquire(dir.path())?;
        match ProjectLock::acquire(dir.path()) {
            Err(LockError::Held { pid }) => assert_eq!(pid, Some(std::process::id())),
            other => return Err(format!("expected Held, got {other:?}").into()),
        }
        Ok(())
    }

    #[test]
    fn the_lock_is_released_on_drop() -> TestResult {
        let dir = tempfile::tempdir()?;
        drop(ProjectLock::acquire(dir.path())?);
        ProjectLock::acquire(dir.path())?;
        Ok(())
    }

    #[test]
    fn a_missing_project_directory_is_never_created() -> TestResult {
        let dir = tempfile::tempdir()?;
        let missing = dir.path().join("missing");
        match ProjectLock::acquire(&missing) {
            Err(error @ LockError::Io { .. }) => {
                assert!(error.to_string().starts_with("cannot lock "), "{error}");
            },
            other => return Err(format!("expected Io, got {other:?}").into()),
        }
        assert!(!missing.exists());
        Ok(())
    }

    #[test]
    fn a_symlinked_state_directory_is_refused() -> TestResult {
        let dir = tempfile::tempdir()?;
        let elsewhere = tempfile::tempdir()?;
        let state = dir.path().join(STATE_DIR);
        std::os::unix::fs::symlink(elsewhere.path(), &state)?;
        match ProjectLock::acquire(dir.path()) {
            Err(error @ LockError::UnsafePath { .. }) => assert_eq!(
                error.to_string(),
                format!(
                    "refusing to use {}: it is not a safe lock path",
                    state.display()
                )
            ),
            other => return Err(format!("expected UnsafePath, got {other:?}").into()),
        }
        assert!(!elsewhere.path().join(LOCK_FILE).exists());
        Ok(())
    }

    #[test]
    fn a_symlinked_lock_file_is_refused_and_its_target_kept() -> TestResult {
        let dir = tempfile::tempdir()?;
        let elsewhere = tempfile::tempdir()?;
        let target = elsewhere.path().join("precious");
        std::fs::write(&target, "keep me")?;
        std::fs::create_dir(dir.path().join(STATE_DIR))?;
        let lock = dir.path().join(STATE_DIR).join(LOCK_FILE);
        std::os::unix::fs::symlink(&target, &lock)?;
        match ProjectLock::acquire(dir.path()) {
            Err(LockError::UnsafePath { path }) => assert_eq!(path, lock),
            other => return Err(format!("expected UnsafePath, got {other:?}").into()),
        }
        assert_eq!(std::fs::read_to_string(&target)?, "keep me");
        Ok(())
    }

    #[test]
    fn a_hard_linked_lock_file_is_refused_and_its_target_kept() -> TestResult {
        let dir = tempfile::tempdir()?;
        let target = dir.path().join("precious");
        std::fs::write(&target, "keep me")?;
        std::fs::create_dir(dir.path().join(STATE_DIR))?;
        let lock = dir.path().join(STATE_DIR).join(LOCK_FILE);
        std::fs::hard_link(&target, &lock)?;
        match ProjectLock::acquire(dir.path()) {
            Err(LockError::UnsafePath { path }) => assert_eq!(path, lock),
            other => return Err(format!("expected UnsafePath, got {other:?}").into()),
        }
        assert_eq!(std::fs::read_to_string(&target)?, "keep me");
        Ok(())
    }

    #[test]
    fn the_message_names_the_pid() {
        let error = LockError::Held { pid: Some(42) };
        assert_eq!(
            error.to_string(),
            "another overbrainer (pid 42) is using this project"
        );
    }
}
