//! One overbrainer process per project: an exclusive lock on `.overbrainer/`
//! itself, held until the process exits. The OS releases it when the process
//! dies. `.overbrainer/lock` only holds the holding process's PID, for the
//! "another overbrainer (pid N)" message; it plays no part in exclusivity.

use std::fmt;
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{self, ErrorKind, Seek as _, SeekFrom, Write as _};
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
    /// The state directory or its PID file is not safe to treat as this
    /// process's own: a symbolic link, a hard link to another file, not the
    /// expected file type, or swapped for one of those between the check and the
    /// open.
    #[error("refusing to use {}: it is not a safe lock path", path.display())]
    UnsafePath {
        /// The unsafe path.
        path: PathBuf,
    },
    /// The state directory cannot be created or locked, or the PID file cannot
    /// be read or written.
    #[error("cannot lock {}", path.display())]
    Io {
        /// The directory or the PID file.
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

/// Verifies that `dir_file`, already opened from `dir`, is truly the directory
/// at `dir`: not a symbolic link, still a directory, and not swapped for
/// another one between the earlier [`check_path`] call and the `open` that
/// produced `dir_file`. Mirrors [`verify_opened`], for a directory instead of a
/// regular file, so it checks the file type instead of the hard link count.
fn verify_dir(dir: &Path, dir_file: &File) -> io::Result<()> {
    let opened = dir_file.metadata()?;
    if !opened.is_dir() {
        return Err(unsafe_path_error(dir));
    }
    let on_disk = std::fs::symlink_metadata(dir)?;
    if on_disk.file_type().is_symlink() {
        return Err(unsafe_path_error(dir));
    }
    if on_disk.dev() != opened.dev() || on_disk.ino() != opened.ino() {
        return Err(unsafe_path_error(dir));
    }
    Ok(())
}

/// The PID written by the process holding the directory lock, for the "another
/// overbrainer (pid N)" message. Missing or unreadable, this gives `None`: the
/// directory lock, not this file, is what proves exclusivity, so no safety
/// check is needed to only read it.
fn read_pid(pid_path: &Path) -> Option<u32> {
    std::fs::read_to_string(pid_path)
        .ok()
        .and_then(|text| text.trim().parse().ok())
}

fn holder(pid: Option<u32>) -> String {
    pid.map_or_else(|| "unknown pid".to_string(), |pid| format!("pid {pid}"))
}

/// The held lock. Dropping it releases the lock.
#[derive(Debug)]
pub struct ProjectLock {
    _dir: File,
}

impl ProjectLock {
    /// Takes the lock on `.overbrainer/`, creating it when needed, and writes
    /// this process's PID into `.overbrainer/lock` for the "another overbrainer
    /// (pid N)" message. Exclusivity comes from the directory lock alone: the
    /// PID file only carries that diagnostic, so removing it changes nothing.
    ///
    /// # Errors
    ///
    /// [`LockError::Held`] when another process holds it, [`LockError::UnsafePath`]
    /// when `.overbrainer/` or its PID file is not a safe path (a symbolic link, a
    /// hard link to another file, not the expected file type, or swapped for one
    /// of those after the initial check), [`LockError::Io`] when the directory
    /// cannot be created or locked, or the PID file cannot be written.
    pub fn acquire(project_dir: &Path) -> Result<Self, LockError> {
        let dir = project_dir.join(STATE_DIR);
        let pid_path = dir.join(LOCK_FILE);
        let dir_io = |source| LockError::Io {
            path: dir.clone(),
            source,
        };
        let unsafe_or_dir_io = |source: io::Error| match unsafe_path_of(&source) {
            Some(path) => LockError::UnsafePath { path },
            None => dir_io(source),
        };
        // Never the project directory itself: only its state directory.
        match std::fs::create_dir(&dir) {
            Err(e) if e.kind() != ErrorKind::AlreadyExists => return Err(dir_io(e)),
            _ => {},
        }
        // Never follow a symbolic link: it could point the lock at any directory.
        // This first check narrows the window but does not close it; `open` below
        // follows symlinks, so `dir` could still be swapped for one between this
        // check and the open.
        check_path(&dir).map_err(&unsafe_or_dir_io)?;
        let dir_file = File::open(&dir).map_err(dir_io)?;
        // Close the race: verify the directory just opened is really the one at
        // `dir`, before locking it. A non-empty directory cannot be removed and
        // replaced without first removing everything inside it, so once this
        // lock is held, nothing short of that can swap `dir` out from under it.
        verify_dir(&dir, &dir_file).map_err(&unsafe_or_dir_io)?;
        match dir_file.try_lock() {
            Ok(()) => {},
            Err(TryLockError::WouldBlock) => {
                return Err(LockError::Held {
                    pid: read_pid(&pid_path),
                });
            },
            Err(TryLockError::Error(source)) => return Err(dir_io(source)),
        }

        // The directory lock alone proves exclusivity; the PID file below is
        // only written for the "another overbrainer (pid N)" message.
        let pid_io = |source| LockError::Io {
            path: pid_path.clone(),
            source,
        };
        let unsafe_or_pid_io = |source: io::Error| match unsafe_path_of(&source) {
            Some(path) => LockError::UnsafePath { path },
            None => pid_io(source),
        };
        check_path(&pid_path).map_err(&unsafe_or_pid_io)?;
        let mut pid_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&pid_path)
            .map_err(pid_io)?;
        // Same race as above, for the PID file this time: verify it before
        // truncating and writing into it.
        verify_opened(&dir, &pid_path, &pid_file).map_err(&unsafe_or_pid_io)?;
        pid_file.set_len(0).map_err(pid_io)?;
        pid_file.seek(SeekFrom::Start(0)).map_err(pid_io)?;
        write!(pid_file, "{}", std::process::id()).map_err(pid_io)?;
        pid_file.flush().map_err(pid_io)?;
        Ok(Self { _dir: dir_file })
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
    fn removing_the_pid_file_does_not_free_the_lock() -> TestResult {
        let dir = tempfile::tempdir()?;
        let _held = ProjectLock::acquire(dir.path())?;
        std::fs::remove_file(dir.path().join(STATE_DIR).join(LOCK_FILE))?;
        match ProjectLock::acquire(dir.path()) {
            Err(LockError::Held { pid }) => assert_eq!(pid, None),
            other => return Err(format!("expected Held, got {other:?}").into()),
        }
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
