//! The project format: `.overbrainer/version`, one integer line naming the
//! layout of the project's files. A project without it predates 0.4.0, whose
//! format is 1; `overbrainer migrate` brings it up to date.

use std::io::{self, ErrorKind, Read as _};
use std::path::Path;

use rustix::fs::OFlags;

use crate::project_lock::{open_state_file, replace_state_file};

/// The format this overbrainer writes and reads.
pub const CURRENT: u32 = 1;

/// File name of the format version in the state directory.
pub const VERSION_FILE: &str = "version";

/// Longest version file read back: an integer never needs more.
const READ_LIMIT: u64 = 32;

/// The format of the project in `project_dir`, `None` when it has no
/// `.overbrainer/version` (a project from before 0.4.0).
///
/// # Errors
///
/// An [`io::Error`] when the file exists but cannot be read, is not a safe path
/// (a symbolic or hard link), or does not hold an integer (`InvalidData`).
pub fn read(project_dir: &Path) -> io::Result<Option<u32>> {
    let file = match open_state_file(project_dir, VERSION_FILE, OFlags::RDONLY, false) {
        Ok(file) => file,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let mut text = String::new();
    file.take(READ_LIMIT).read_to_string(&mut text)?;
    text.trim().parse().map(Some).map_err(|_| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "{} does not hold a format number",
                project_dir
                    .join(crate::project_lock::STATE_DIR)
                    .join(VERSION_FILE)
                    .display()
            ),
        )
    })
}

/// Writes [`CURRENT`] into `.overbrainer/version`, atomically.
///
/// # Errors
///
/// An [`io::Error`] when the state directory or the file cannot be written or
/// is not a safe path.
pub fn write(project_dir: &Path) -> io::Result<()> {
    replace_state_file(project_dir, VERSION_FILE, format!("{CURRENT}\n").as_bytes())
}

/// Whether `project_dir` holds a project (`overbrainer.toml`) from before
/// 0.4.0: no `.overbrainer/version`. An unreadable version says no.
#[must_use]
pub fn predates_versions(project_dir: &Path) -> bool {
    project_dir.join(crate::config::CONFIG_FILE).is_file() && matches!(read(project_dir), Ok(None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project_lock::STATE_DIR;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn a_written_version_reads_back() -> TestResult {
        let dir = tempfile::tempdir()?;
        assert_eq!(read(dir.path())?, None);
        write(dir.path())?;
        assert_eq!(read(dir.path())?, Some(CURRENT));
        assert_eq!(
            std::fs::read_to_string(dir.path().join(STATE_DIR).join(VERSION_FILE))?,
            "1\n"
        );
        Ok(())
    }

    #[test]
    fn a_version_that_is_not_a_number_is_an_error() -> TestResult {
        let dir = tempfile::tempdir()?;
        std::fs::create_dir(dir.path().join(STATE_DIR))?;
        std::fs::write(dir.path().join(STATE_DIR).join(VERSION_FILE), "one\n")?;
        match read(dir.path()) {
            Err(e) => assert_eq!(e.kind(), ErrorKind::InvalidData),
            other => return Err(format!("expected an error, got {other:?}").into()),
        }
        Ok(())
    }

    #[test]
    fn only_a_project_without_a_version_predates_versions() -> TestResult {
        let dir = tempfile::tempdir()?;
        assert!(!predates_versions(dir.path()), "not a project");
        std::fs::write(dir.path().join(crate::config::CONFIG_FILE), "")?;
        assert!(predates_versions(dir.path()));
        write(dir.path())?;
        assert!(!predates_versions(dir.path()));
        Ok(())
    }
}
