//! One overbrainer process per project: an exclusive lock on `.overbrainer/lock`,
//! held until the process exits. The OS releases it when the process dies.

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
    /// The state directory or the lock file is a symbolic link, a hard link to
    /// another file, or was swapped for one of those between the check and the open.
    #[error("refusing to use {}: it is a symbolic link", path.display())]
    Symlink {
        /// The link, or the path replaced by one.
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

/// Whether `path` is a symbolic link; a missing path is not.
fn is_symlink(path: &Path) -> io::Result<bool> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => Ok(metadata.file_type().is_symlink()),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

/// Verifies that `file`, already opened from `path`, is truly the file at `path`:
/// not a symbolic link, not a hard link to another file, and not swapped for either
/// between the earlier symlink check and the `open` call that produced `file`.
///
/// `open` follows symbolic links, so a check made before opening cannot by itself
/// rule out a link swapped in during that window. This closes the window by
/// comparing what was actually opened (`fstat`, via [`File::metadata`]) against what
/// is on disk right now (`lstat`, via [`std::fs::symlink_metadata`]).
fn verify_opened_file(dir: &Path, path: &Path, file: &File) -> Result<(), LockError> {
    let io = |source| LockError::Io {
        path: path.to_path_buf(),
        source,
    };
    let symlink = |bad: &Path| LockError::Symlink {
        path: bad.to_path_buf(),
    };

    // The state directory could have been swapped for a symlink after the earlier
    // check and before the lock file was opened.
    if is_symlink(dir).map_err(io)? {
        return Err(symlink(dir));
    }

    let opened = file.metadata().map_err(io)?;
    // A file with more than one hard link is also reachable through another path;
    // refusing it rules out the lock file being a second name for a file we do not
    // own, even though a hard link is not a symbolic link.
    if !opened.is_file() || opened.nlink() != 1 {
        return Err(symlink(path));
    }

    // `lstat` the path itself: it must still be a plain file, never a symlink, and
    // it must be the very inode `file` was opened from. A mismatch means `path` was
    // replaced after the open, and `file` is not the file the caller expects to
    // manage.
    let on_disk = std::fs::symlink_metadata(path).map_err(io)?;
    if on_disk.file_type().is_symlink() {
        return Err(symlink(path));
    }
    if on_disk.dev() != opened.dev() || on_disk.ino() != opened.ino() {
        return Err(symlink(path));
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
    /// [`LockError::Held`] when another process holds it, [`LockError::Symlink`] when
    /// `.overbrainer/` or its lock file is a symbolic link, is a hard link to another
    /// file, or was swapped for one of those after the initial check, [`LockError::Io`]
    /// when the file cannot be created, locked or written.
    pub fn acquire(project_dir: &Path) -> Result<Self, LockError> {
        let dir = project_dir.join(STATE_DIR);
        let path = dir.join(LOCK_FILE);
        let io = |source| LockError::Io {
            path: path.clone(),
            source,
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
            if is_symlink(link).map_err(io)? {
                return Err(LockError::Symlink { path: link.clone() });
            }
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
        verify_opened_file(&dir, &path, &file)?;
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
            Err(error @ LockError::Symlink { .. }) => assert_eq!(
                error.to_string(),
                format!("refusing to use {}: it is a symbolic link", state.display())
            ),
            other => return Err(format!("expected Symlink, got {other:?}").into()),
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
            Err(LockError::Symlink { path }) => assert_eq!(path, lock),
            other => return Err(format!("expected Symlink, got {other:?}").into()),
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
            Err(LockError::Symlink { path }) => assert_eq!(path, lock),
            other => return Err(format!("expected Symlink, got {other:?}").into()),
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
