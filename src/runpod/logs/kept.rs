//! The pod's logs as a run directory keeps them, in `.pod/`: the log stream's
//! lines ([`POD_LOG`]) and cursor ([`POD_LOG_CURSOR`]), and the copies of the
//! bootstrap's and watchdog's own logs.
//!
//! A pod writes into `.pod/` (the results it sends back are extracted into
//! the run directory), so nothing here goes through a symbolic link: `.pod`
//! and each file are opened with `O_NOFOLLOW`, relative to their directory,
//! and a file with a second hard link is refused, like the project's own
//! state files.

use std::fs::File;
use std::io::{self, ErrorKind, Read, Seek, SeekFrom, Write};
use std::os::fd::OwnedFd;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use rustix::fs::{AtFlags, Mode, OFlags};
use rustix::io::Errno;

use super::{
    CAPPED_SOURCE, LogQuery, POD_LOG, POD_LOG_CAP, POD_LOG_CURSOR, PodLogLine, TAIL_MAX,
    valid_cursor,
};

/// The directory of the kept logs in a run directory.
const POD_DIR: &str = ".pod";

/// Room kept under the cap for the line saying the log was capped.
const CAP_RESERVE: u64 = 512;

/// How often, at most, the cursor is written while lines keep coming.
const CURSOR_EVERY: Duration = Duration::from_secs(1);

/// The file name, in `.pod/`, of `path` (such as [`POD_LOG`]): `None` unless
/// it is `.pod/<name>`.
fn name_in_pod(path: &str) -> Option<&str> {
    path.strip_prefix(".pod/")
        .filter(|name| !name.is_empty() && !name.contains('/'))
}

/// The error for a path that is a symbolic link, a hard link to another
/// file, or not what it should be.
fn unsafe_path(path: &Path) -> io::Error {
    io::Error::new(
        ErrorKind::InvalidInput,
        format!("refusing to use {}: it is not a safe path", path.display()),
    )
}

/// Maps the failure of an `O_NOFOLLOW` open of `path`.
fn open_error(path: &Path, errno: Errno) -> io::Error {
    if errno == Errno::LOOP || errno == Errno::NOTDIR {
        unsafe_path(path)
    } else {
        errno.into()
    }
}

/// The `.pod` directory of `run_dir`, never through a symbolic link, created
/// (mode 0700, with the run directory) when `create`.
fn pod_dir(run_dir: &Path, create: bool) -> io::Result<OwnedFd> {
    if create {
        std::fs::create_dir_all(run_dir)?;
    }
    let run = rustix::fs::open(
        run_dir,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
        Mode::empty(),
    )?;
    if create {
        match rustix::fs::mkdirat(&run, POD_DIR, Mode::from_raw_mode(0o700)) {
            Ok(()) => {},
            Err(errno) if errno == Errno::EXIST => {},
            Err(errno) => return Err(errno.into()),
        }
    }
    rustix::fs::openat(
        &run,
        POD_DIR,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map_err(|errno| open_error(&run_dir.join(POD_DIR), errno))
}

/// Opens `name` in the `.pod` directory `dir` (whose path is `dir_path`),
/// never through a symbolic link, and only when it is a regular file with a
/// single link. A file opened for writing is made mode 0600.
fn open_in(dir: &OwnedFd, dir_path: &Path, name: &str, flags: OFlags) -> io::Result<File> {
    let path = dir_path.join(name);
    let fd = rustix::fs::openat(
        dir,
        name,
        flags | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::from_raw_mode(0o600),
    )
    .map_err(|errno| open_error(&path, errno))?;
    let file = File::from(fd);
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.nlink() != 1 {
        return Err(unsafe_path(&path));
    }
    if flags.intersects(OFlags::WRONLY | OFlags::RDWR) {
        rustix::fs::fchmod(&file, Mode::from_raw_mode(0o600))?;
    }
    Ok(file)
}

/// Opens the kept file `path` (such as [`POD_LOG`] or
/// [`BOOTSTRAP_LOG`](super::BOOTSTRAP_LOG)) of `run_dir` for reading; `None`
/// when it, or `.pod`, does not exist.
///
/// # Errors
///
/// Returns an error of kind `InvalidInput` when `.pod` or the file is a
/// symbolic link, a hard link or not a regular file, and the I/O error of
/// opening it otherwise.
pub fn open_kept(run_dir: &Path, path: &str) -> io::Result<Option<File>> {
    let name = name_in_pod(path).ok_or_else(|| unsafe_path(Path::new(path)))?;
    let dir_path = run_dir.join(POD_DIR);
    let dir = match pod_dir(run_dir, false) {
        Ok(dir) => dir,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    match open_in(&dir, &dir_path, name, OFlags::RDONLY) {
        Ok(file) => Ok(Some(file)),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// The whole kept file `path` of `run_dir` (see [`open_kept`]); `None` when
/// it does not exist.
///
/// # Errors
///
/// As [`open_kept`], and the I/O error of reading it.
pub fn read_kept(run_dir: &Path, path: &str) -> io::Result<Option<Vec<u8>>> {
    let Some(mut file) = open_kept(run_dir, path)? else {
        return Ok(None);
    };
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(Some(bytes))
}

/// Replaces the kept file `path` of `run_dir` with `bytes`, whole or not at
/// all: written to a new temporary file (mode 0600, a stale one removed
/// first), then renamed over it.
///
/// # Errors
///
/// As [`open_kept`], and the I/O error of writing or renaming.
pub fn keep_file(run_dir: &Path, path: &str, bytes: &[u8]) -> io::Result<()> {
    let name = name_in_pod(path).ok_or_else(|| unsafe_path(Path::new(path)))?;
    let dir = pod_dir(run_dir, true)?;
    replace_in(&dir, &run_dir.join(POD_DIR), name, bytes)
}

/// [`keep_file`] in the open `.pod` directory `dir`.
fn replace_in(dir: &OwnedFd, dir_path: &Path, name: &str, bytes: &[u8]) -> io::Result<()> {
    let temporary = format!("{name}.tmp");
    match rustix::fs::unlinkat(dir, temporary.as_str(), AtFlags::empty()) {
        Ok(()) => {},
        Err(errno) if errno == Errno::NOENT => {},
        Err(errno) => return Err(errno.into()),
    }
    let flags = OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL;
    let mut file = open_in(dir, dir_path, &temporary, flags)?;
    file.write_all(bytes)?;
    rustix::fs::renameat(dir, temporary.as_str(), dir, name)?;
    Ok(())
}

/// The event ID the log kept in `run_dir` ends at, if any.
#[must_use]
pub fn kept_cursor(run_dir: &Path) -> Option<String> {
    let bytes = read_kept(run_dir, POD_LOG_CURSOR).ok()??;
    let cursor = String::from_utf8_lossy(&bytes).trim().to_string();
    valid_cursor(&cursor).then_some(cursor)
}

/// A line of the kept log, cleaned like a line of the stream (see
/// [`PodLogLine::cleaned`]); `None` when it is not one.
#[must_use]
pub fn parse_kept_line(text: &str) -> Option<PodLogLine> {
    serde_json::from_str::<PodLogLine>(text)
        .ok()
        .map(PodLogLine::cleaned)
}

/// The lines of the log kept in `run_dir`, oldest first; a line that cannot
/// be read is skipped. Empty when none is kept.
///
/// # Errors
///
/// As [`read_kept`].
pub fn kept_lines(run_dir: &Path) -> io::Result<Vec<PodLogLine>> {
    let Some(bytes) = read_kept(run_dir, POD_LOG)? else {
        return Ok(Vec::new());
    };
    Ok(String::from_utf8_lossy(&bytes)
        .lines()
        .filter_map(parse_kept_line)
        .collect())
}

/// The bytes of the kept log of `run_dir` from `offset`, or from its start
/// when it is now shorter, with where they start; `None` when none is kept.
///
/// # Errors
///
/// As [`open_kept`], and the I/O error of reading it.
pub fn read_kept_from(run_dir: &Path, offset: u64) -> io::Result<Option<(Vec<u8>, u64)>> {
    let Some(mut file) = open_kept(run_dir, POD_LOG)? else {
        return Ok(None);
    };
    let start = if file.metadata()?.len() < offset {
        0
    } else {
        offset
    };
    file.seek(SeekFrom::Start(start))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(Some((bytes, start)))
}

/// The pod's log kept in a run directory ([`POD_LOG`], mode 0600) and its
/// cursor ([`POD_LOG_CURSOR`]). Past [`POD_LOG_CAP`], one line says the log
/// was capped and nothing more is kept. The cursor is written when it moved,
/// at most once a second while lines keep coming, then once more at the end
/// (dropping the capture writes it too): a cursor behind the log repeats a
/// few lines on the next resume, never loses one.
#[derive(Debug)]
pub struct Capture {
    dir: OwnedFd,
    dir_path: PathBuf,
    file: File,
    size: u64,
    cap: u64,
    capped: bool,
    /// The cursor the log ends at, written or not.
    cursor: Option<String>,
    /// Whether `cursor` is not written yet.
    unsaved: bool,
    saved_at: Option<Instant>,
}

impl Capture {
    /// Opens the kept log of the run directory `run_dir`, created when absent,
    /// with its cursor. A last line left half written (a crash) is ended first.
    ///
    /// # Errors
    ///
    /// Returns an error of kind `InvalidInput` when `.pod` or the log is a
    /// symbolic link or a hard link, and the I/O error of creating or opening
    /// them otherwise.
    pub fn open(run_dir: &Path) -> io::Result<Self> {
        let dir = pod_dir(run_dir, true)?;
        let dir_path = run_dir.join(POD_DIR);
        let log = name_in_pod(POD_LOG).unwrap_or("pod.log");
        let flags = OFlags::RDWR | OFlags::APPEND | OFlags::CREATE;
        let mut file = open_in(&dir, &dir_path, log, flags)?;
        let mut size = file.metadata()?.len();
        let mut last = [0_u8];
        if size > 0 && file.read_at(&mut last, size - 1)? == 1 && last[0] != b'\n' {
            file.write_all(b"\n")?;
            size += 1;
        }
        let mut capture = Self {
            dir,
            dir_path,
            file,
            size,
            cap: POD_LOG_CAP,
            capped: false,
            cursor: kept_cursor(run_dir),
            unsaved: false,
            saved_at: None,
        };
        capture.capped = capture.full(0) || capture.ends_capped();
        Ok(capture)
    }

    /// This capture with another cap, in bytes (tests use small ones).
    #[must_use]
    pub fn with_cap(mut self, cap: u64) -> Self {
        self.cap = cap;
        self.capped = self.full(0) || self.ends_capped();
        self
    }

    /// The event ID the kept log ends at.
    #[must_use]
    pub fn cursor(&self) -> Option<&str> {
        self.cursor.as_deref()
    }

    /// Whether the cap was reached: nothing more is kept.
    #[must_use]
    pub fn capped(&self) -> bool {
        self.capped
    }

    /// Where the lines of the log stream start for this capture: after its
    /// cursor, or the longest replay without one.
    #[must_use]
    pub fn query(&self) -> LogQuery {
        LogQuery {
            source: None,
            tail: self.cursor.is_none().then_some(TAIL_MAX),
            since: None,
            cursor: self.cursor.clone(),
        }
    }

    fn full(&self, adding: u64) -> bool {
        self.size + adding > self.cap.saturating_sub(CAP_RESERVE)
    }

    /// Whether the log ends with the line saying it was capped.
    fn ends_capped(&self) -> bool {
        let from = self.size.saturating_sub(1024);
        let length = usize::try_from(self.size - from).unwrap_or(0);
        let mut tail = vec![0_u8; length];
        if self.file.read_exact_at(&mut tail, from).is_err() {
            return false;
        }
        String::from_utf8_lossy(&tail)
            .lines()
            .last()
            .and_then(|last| serde_json::from_str::<PodLogLine>(last).ok())
            .is_some_and(|last| last.source == CAPPED_SOURCE)
    }

    /// Appends `lines`, then keeps `cursor`. Reaching the cap writes one line
    /// saying so instead of the rest.
    ///
    /// # Errors
    ///
    /// Returns the I/O error of a write.
    pub fn take(&mut self, lines: &[PodLogLine], cursor: Option<&str>) -> io::Result<()> {
        if self.capped {
            return Ok(());
        }
        let mut text = String::new();
        for line in lines {
            let json = serde_json::to_string(line).map_err(io::Error::other)?;
            let adding = u64::try_from(text.len() + json.len() + 1).unwrap_or(u64::MAX);
            if self.full(adding) {
                self.capped = true;
                text.push_str(&self.marker()?);
                break;
            }
            text.push_str(&json);
            text.push('\n');
        }
        self.file.write_all(text.as_bytes())?;
        self.size += u64::try_from(text.len()).unwrap_or(u64::MAX);
        if self.capped {
            return Ok(());
        }
        if let Some(cursor) = cursor
            && self.cursor.as_deref() != Some(cursor)
        {
            self.cursor = Some(cursor.to_string());
            self.unsaved = true;
        }
        if self.saved_at.is_none_or(|at| at.elapsed() >= CURSOR_EVERY) {
            self.finish()?;
        }
        Ok(())
    }

    /// The line saying the log was capped, with its newline.
    fn marker(&self) -> io::Result<String> {
        let marker = PodLogLine {
            ts: crate::runs::rfc3339(SystemTime::now()),
            source: CAPPED_SOURCE.to_string(),
            line: format!(
                "the pod's log reached {} MiB: later lines are not kept",
                self.cap / (1024 * 1024)
            ),
        };
        let mut text = serde_json::to_string(&marker).map_err(io::Error::other)?;
        text.push('\n');
        Ok(text)
    }

    /// Writes the cursor when it moved since it was last written.
    ///
    /// # Errors
    ///
    /// Returns the I/O error of writing it.
    pub fn finish(&mut self) -> io::Result<()> {
        if !self.unsaved {
            return Ok(());
        }
        let Some(cursor) = self.cursor.clone() else {
            return Ok(());
        };
        let name = name_in_pod(POD_LOG_CURSOR).unwrap_or("pod.log.cursor");
        replace_in(&self.dir, &self.dir_path, name, cursor.as_bytes())?;
        self.unsaved = false;
        self.saved_at = Some(Instant::now());
        Ok(())
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        if let Err(error) = self.finish() {
            tracing::debug!("cannot keep the cursor of the pod's log: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::fs::{PermissionsExt, symlink};

    use super::*;

    fn line(length: usize) -> PodLogLine {
        PodLogLine {
            ts: "2026-06-01T12:02:03Z".into(),
            source: "container".into(),
            line: "x".repeat(length),
        }
    }

    fn kept(dir: &Path) -> Result<Vec<PodLogLine>, Box<dyn std::error::Error>> {
        Ok(kept_lines(dir)?)
    }

    #[test]
    fn a_capture_keeps_lines_and_its_cursor_then_stops_at_its_cap()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let mut capture = Capture::open(dir.path())?.with_cap(CAP_RESERVE + 400);
        assert_eq!(capture.query().tail, Some(TAIL_MAX));
        capture.take(std::slice::from_ref(&line(100)), Some("c/1"))?;
        assert_eq!(kept_cursor(dir.path()).as_deref(), Some("c/1"));
        let reopened = Capture::open(dir.path())?;
        assert_eq!(
            reopened.query(),
            LogQuery {
                source: None,
                tail: None,
                since: None,
                cursor: Some("c/1".into())
            }
        );
        drop(reopened);
        capture.take(&[line(100), line(100), line(100)], Some("c/4"))?;
        assert!(capture.capped());
        capture.take(&[line(1)], Some("c/5"))?;
        let lines = kept(dir.path())?;
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert_eq!(lines[2].source, CAPPED_SOURCE);
        assert!(lines[2].line.contains("later lines are not kept"));
        let mode = fs::metadata(dir.path().join(POD_LOG))?.permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
        Ok(())
    }

    /// The line past the cap can be longer than the marker written instead:
    /// the log reopened must still count as capped, with any cap.
    #[test]
    fn a_capped_log_stays_capped_when_reopened() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let cap = CAP_RESERVE + 400;
        let mut capture = Capture::open(dir.path())?.with_cap(cap);
        capture.take(&[line(100), line(300)], Some("c/2"))?;
        assert!(capture.capped());
        drop(capture);
        let size = fs::metadata(dir.path().join(POD_LOG))?.len();
        assert!(size < cap - CAP_RESERVE, "{size}");
        assert!(Capture::open(dir.path())?.with_cap(cap).capped());
        assert!(Capture::open(dir.path())?.capped());
        Ok(())
    }

    #[test]
    fn a_half_written_last_line_is_ended_on_open() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        fs::create_dir_all(dir.path().join(".pod"))?;
        let first = serde_json::to_string(&line(3))?;
        fs::write(
            dir.path().join(POD_LOG),
            format!("{first}\n{{\"ts\":\"2026"),
        )?;
        let mut capture = Capture::open(dir.path())?;
        capture.take(&[line(4)], None)?;
        let lines = kept(dir.path())?;
        assert_eq!(lines, [line(3), line(4)]);
        Ok(())
    }

    #[test]
    fn the_cursor_is_written_at_most_once_a_second_then_at_the_end()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let mut capture = Capture::open(dir.path())?;
        capture.take(&[line(1)], Some("c/1"))?;
        capture.take(&[line(1)], Some("c/2"))?;
        assert_eq!(capture.cursor(), Some("c/2"));
        assert_eq!(kept_cursor(dir.path()).as_deref(), Some("c/1"));
        drop(capture);
        assert_eq!(kept_cursor(dir.path()).as_deref(), Some("c/2"));
        Ok(())
    }

    #[test]
    fn links_planted_in_the_run_directory_are_refused() -> Result<(), Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let outside = root.path().join("outside");
        fs::write(&outside, "mine\n")?;
        // A symbolic link in place of the log.
        let run = root.path().join("r1");
        fs::create_dir_all(run.join(".pod"))?;
        symlink(&outside, run.join(POD_LOG))?;
        let refused = Capture::open(&run).map(drop);
        assert_eq!(refused.map_err(|e| e.kind()), Err(ErrorKind::InvalidInput));
        assert_eq!(
            open_kept(&run, POD_LOG).map(drop).map_err(|e| e.kind()),
            Err(ErrorKind::InvalidInput)
        );
        // A hard link to another file.
        let run = root.path().join("r2");
        fs::create_dir_all(run.join(".pod"))?;
        fs::hard_link(&outside, run.join(POD_LOG))?;
        let refused = Capture::open(&run).map(drop);
        assert_eq!(refused.map_err(|e| e.kind()), Err(ErrorKind::InvalidInput));
        // `.pod` itself a symbolic link to another directory.
        let run = root.path().join("r3");
        let elsewhere = root.path().join("elsewhere");
        fs::create_dir_all(&run)?;
        fs::create_dir_all(&elsewhere)?;
        symlink(&elsewhere, run.join(".pod"))?;
        let refused = Capture::open(&run).map(drop);
        assert_eq!(refused.map_err(|e| e.kind()), Err(ErrorKind::InvalidInput));
        assert!(fs::read_dir(&elsewhere)?.next().is_none());
        // A temporary cursor file planted as a link is replaced, not followed.
        let run = root.path().join("r4");
        fs::create_dir_all(run.join(".pod"))?;
        symlink(&outside, run.join(".pod/pod.log.cursor.tmp"))?;
        let mut capture = Capture::open(&run)?;
        capture.take(&[line(1)], Some("c/1"))?;
        assert_eq!(kept_cursor(&run).as_deref(), Some("c/1"));
        // The copies of the pod's own logs go through the same checks.
        symlink(&outside, run.join(".pod/watchdog.log.tmp"))?;
        keep_file(&run, super::super::super::WATCHDOG_LOG, b"watchdog\n")?;
        assert_eq!(fs::read(run.join(".pod/watchdog.log"))?, b"watchdog\n");
        assert_eq!(fs::read_to_string(&outside)?, "mine\n");
        Ok(())
    }
}
