//! One overbrainer process per project: an exclusive lock on the project
//! directory itself, held until the process exits. The OS releases it when the
//! process dies. `.overbrainer/lock` only holds the holding process's PID, for
//! the "another overbrainer (pid N)" message; it plays no part in exclusivity.
//!
//! The lock guards against a second overbrainer process, not against other
//! programs changing the project. Once the project directory is open, the state
//! directory and the PID file are opened relative to it, never by path, so
//! renaming or swapping `.overbrainer` cannot redirect the PID write.

use std::fmt;
use std::fs::File;
use std::io::{self, ErrorKind, Read as _, Write as _};
use std::os::fd::OwnedFd;
use std::os::unix::fs::MetadataExt as _;
use std::path::{Path, PathBuf};

use rustix::fs::{FlockOperation, Mode, OFlags};
use rustix::io::Errno;

/// Directory of the project's own state, next to `overbrainer.toml`.
pub const STATE_DIR: &str = ".overbrainer";

const LOCK_FILE: &str = "lock";

/// Longest PID file read back: a PID never needs more.
const PID_READ_LIMIT: u64 = 32;

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
    /// process's own: a symbolic link, a hard link to another file, or not the
    /// expected file type.
    #[error("refusing to use {}: it is not a safe lock path", path.display())]
    UnsafePath {
        /// The unsafe path.
        path: PathBuf,
    },
    /// The project directory cannot be opened or locked, the state directory
    /// cannot be created or opened, or the PID file cannot be written.
    #[error("cannot lock {}", path.display())]
    Io {
        /// The project directory, the state directory or the PID file.
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

/// An [`io::Error`] naming `path` as unsafe.
fn unsafe_path_error(path: &Path) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, UnsafePath(path.to_path_buf()))
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
/// This narrows the window before the file is opened but does not close it:
/// [`verify_opened`] checks what was actually opened.
pub(crate) fn check_path(path: &Path) -> io::Result<()> {
    if is_symlink(path)? {
        return Err(unsafe_path_error(path));
    }
    Ok(())
}

/// Verifies that `file`, opened from `path` inside `dir`, is a regular file
/// with a single link that is still the file at `path`, and that `dir` is not
/// a symbolic link.
///
/// `open` follows symbolic links, so this compares what was opened (`fstat`)
/// with what is on disk now (`lstat`).
pub(crate) fn verify_opened(dir: &Path, path: &Path, file: &File) -> io::Result<()> {
    if is_symlink(dir)? {
        return Err(unsafe_path_error(dir));
    }

    let opened = file.metadata()?;
    // A second hard link makes the file reachable through a path we do not own.
    if !opened.is_file() || opened.nlink() != 1 {
        return Err(unsafe_path_error(path));
    }

    // `path` must still be that very inode, and not a symbolic link.
    let on_disk = std::fs::symlink_metadata(path)?;
    if on_disk.file_type().is_symlink()
        || on_disk.dev() != opened.dev()
        || on_disk.ino() != opened.ino()
    {
        return Err(unsafe_path_error(path));
    }

    Ok(())
}

/// A [`LockError::Io`] for `path`.
fn io_error(path: &Path, errno: Errno) -> LockError {
    LockError::Io {
        path: path.to_path_buf(),
        source: errno.into(),
    }
}

/// Maps an `openat` failure with `O_NOFOLLOW`: a symbolic link (or, for a
/// directory, anything but one) is [`LockError::UnsafePath`].
fn open_error(path: &Path, errno: Errno) -> LockError {
    if errno == Errno::LOOP || errno == Errno::NOTDIR {
        LockError::UnsafePath {
            path: path.to_path_buf(),
        }
    } else {
        io_error(path, errno)
    }
}

/// Opens the state directory relative to `project`, never through a symbolic
/// link.
fn open_state_dir(project: &OwnedFd) -> Result<OwnedFd, Errno> {
    rustix::fs::openat(
        project,
        STATE_DIR,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
}

/// The PID written by the lock holder, for the "another overbrainer (pid N)"
/// message. Missing or unreadable, this gives `None`.
fn read_pid(project: &OwnedFd) -> Option<u32> {
    let state = open_state_dir(project).ok()?;
    // Non-blocking, so a FIFO planted as the PID file cannot hang the read.
    let fd = rustix::fs::openat(
        &state,
        LOCK_FILE,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .ok()?;
    let mut text = String::new();
    File::from(fd)
        .take(PID_READ_LIMIT)
        .read_to_string(&mut text)
        .ok()?;
    text.trim().parse().ok()
}

fn holder(pid: Option<u32>) -> String {
    pid.map_or_else(|| "unknown pid".to_string(), |pid| format!("pid {pid}"))
}

/// The held lock, on the project directory. Dropping it releases the lock.
#[derive(Debug)]
pub struct ProjectLock {
    _project: OwnedFd,
}

impl ProjectLock {
    /// Takes the lock on `project_dir` itself, creates `.overbrainer/` when
    /// needed, and writes this process's PID into `.overbrainer/lock` for the
    /// "another overbrainer (pid N)" message. Exclusivity comes from the lock
    /// on the project directory alone: removing or renaming `.overbrainer/`
    /// changes nothing.
    ///
    /// # Errors
    ///
    /// [`LockError::Held`] when another process holds it,
    /// [`LockError::UnsafePath`] when `.overbrainer/` or its PID file is a
    /// symbolic link, a hard link to another file, or not the expected file
    /// type, [`LockError::Io`] when the project directory cannot be opened or
    /// locked, `.overbrainer/` cannot be created, or the PID file cannot be
    /// written.
    pub fn acquire(project_dir: &Path) -> Result<Self, LockError> {
        let dir = project_dir.join(STATE_DIR);
        let pid_path = dir.join(LOCK_FILE);

        let project = rustix::fs::open(
            project_dir,
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(|e| io_error(project_dir, e))?;
        match rustix::fs::flock(&project, FlockOperation::NonBlockingLockExclusive) {
            Ok(()) => {},
            Err(e) if e == Errno::WOULDBLOCK => {
                return Err(LockError::Held {
                    pid: read_pid(&project),
                });
            },
            Err(e) => return Err(io_error(project_dir, e)),
        }

        // Everything below is relative to the locked descriptor, so a rename
        // of either directory cannot redirect it.
        match rustix::fs::mkdirat(&project, STATE_DIR, Mode::from_raw_mode(0o777)) {
            Ok(()) => {},
            Err(e) if e == Errno::EXIST => {},
            Err(e) => return Err(io_error(&dir, e)),
        }
        let state = open_state_dir(&project).map_err(|e| open_error(&dir, e))?;
        let pid_fd = rustix::fs::openat(
            &state,
            LOCK_FILE,
            OFlags::RDWR | OFlags::CREATE | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::from_raw_mode(0o644),
        )
        .map_err(|e| open_error(&pid_path, e))?;
        let mut pid_file = File::from(pid_fd);
        let pid_io = |source| LockError::Io {
            path: pid_path.clone(),
            source,
        };
        // A second hard link would make the write land in a file we do not own.
        let metadata = pid_file.metadata().map_err(pid_io)?;
        if !metadata.is_file() || metadata.nlink() != 1 {
            return Err(LockError::UnsafePath {
                path: pid_path.clone(),
            });
        }
        pid_file.set_len(0).map_err(pid_io)?;
        write!(pid_file, "{}", std::process::id()).map_err(pid_io)?;
        pid_file.flush().map_err(pid_io)?;
        Ok(Self { _project: project })
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
    fn renaming_the_state_directory_does_not_free_the_lock() -> TestResult {
        let dir = tempfile::tempdir()?;
        let _held = ProjectLock::acquire(dir.path())?;
        std::fs::rename(dir.path().join(STATE_DIR), dir.path().join("moved"))?;
        match ProjectLock::acquire(dir.path()) {
            Err(LockError::Held { pid }) => assert_eq!(pid, None),
            other => return Err(format!("expected Held, got {other:?}").into()),
        }
        assert!(!dir.path().join(STATE_DIR).exists());
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
