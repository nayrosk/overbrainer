//! One overbrainer process per project: an exclusive lock on `.overbrainer/lock`,
//! held until the process exits. The OS releases it when the process dies.

use std::fs::{File, OpenOptions, TryLockError};
use std::io::{self, Read as _, Seek as _, SeekFrom, Write as _};
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
    /// The lock file cannot be created, read or locked.
    #[error("cannot lock {}: {source}", path.display())]
    Io {
        /// The lock file.
        path: PathBuf,
        /// The I/O error.
        source: io::Error,
    },
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
    /// [`LockError::Held`] when another process holds it, [`LockError::Io`] when the
    /// file cannot be created, locked or written.
    pub fn acquire(project_dir: &Path) -> Result<Self, LockError> {
        let dir = project_dir.join(STATE_DIR);
        let path = dir.join(LOCK_FILE);
        let io = |source| LockError::Io {
            path: path.clone(),
            source,
        };
        std::fs::create_dir_all(&dir).map_err(io)?;
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(io)?;
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
    fn the_message_names_the_pid() {
        let error = LockError::Held { pid: Some(42) };
        assert_eq!(
            error.to_string(),
            "another overbrainer (pid 42) is using this project"
        );
    }
}
