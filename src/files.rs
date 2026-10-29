//! Reading and writing files a project, a remote job or another user may have
//! planted: regular files only, never more than a known size, and, where it
//! matters, never through a symbolic link.

use std::fs::File;
use std::io::{self, Read as _};
use std::os::fd::AsFd;
use std::path::Path;

use rustix::fs::{Mode, OFlags};

/// Flags of every open here: no controlling terminal, no blocking on a FIFO,
/// no descriptor leaked to a child process.
const OPEN: OFlags = OFlags::NOCTTY
    .union(OFlags::NONBLOCK)
    .union(OFlags::CLOEXEC);

/// The content of the regular file at `path`, read without following a
/// symbolic link at `path` and refused past `max` bytes.
///
/// # Errors
///
/// Returns the error of opening or reading `path`: `NotFound` when it does not
/// exist, `InvalidInput` when it is a symbolic link or not a regular file
/// (a directory, a device, a FIFO), and `FileTooLarge` when it holds more than
/// `max` bytes.
pub(crate) fn read_regular(path: &Path, max: u64) -> io::Result<Vec<u8>> {
    let fd = rustix::fs::open(
        path,
        OFlags::RDONLY | OFlags::NOFOLLOW | OPEN,
        Mode::empty(),
    )
    .map_err(|errno| open_error(path, errno))?;
    read_opened(File::from(fd), path, max)
}

/// The content of the file at `path`, a symbolic link followed, refused when
/// it is not a regular file (a device such as `/dev/zero`, a FIFO) or holds
/// more than `max` bytes. For a file its owner may keep elsewhere and link,
/// such as `overbrainer.toml`.
///
/// # Errors
///
/// Returns the error of opening or reading `path`: `NotFound` when it does not
/// exist, `InvalidInput` when it is not a regular file, and `FileTooLarge` when
/// it holds more than `max` bytes.
pub(crate) fn read_bounded(path: &Path, max: u64) -> io::Result<Vec<u8>> {
    let fd = rustix::fs::open(path, OFlags::RDONLY | OPEN, Mode::empty())
        .map_err(|errno| open_error(path, errno))?;
    read_opened(File::from(fd), path, max)
}

/// The content of the regular file `name` in the directory `dir`, which must
/// be owned by this user and closed to everyone else (no group or other
/// permission bits). Neither `dir` nor `name` is followed when it is a
/// symbolic link, so another user who can rename entries beside `dir` cannot
/// swap in a directory or a file of their own.
///
/// # Errors
///
/// Returns the error of opening or reading: `NotFound` when either is missing,
/// `InvalidInput` when `dir` is a link, not a directory, not ours or open to
/// others, or `name` is a link or not a regular file, and `FileTooLarge` when
/// it holds more than `max` bytes.
pub(crate) fn read_private(dir: &Path, name: &str, max: u64) -> io::Result<Vec<u8>> {
    let dir_fd = rustix::fs::open(
        dir,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OPEN,
        Mode::empty(),
    )
    .map_err(|errno| open_error(dir, errno))?;
    let stat = rustix::fs::fstat(&dir_fd)?;
    if stat.st_uid != rustix::process::geteuid().as_raw() || stat.st_mode & 0o077 != 0 {
        return Err(not_regular(dir));
    }
    let path = dir.join(name);
    let fd = rustix::fs::openat(
        dir_fd.as_fd(),
        name,
        OFlags::RDONLY | OFlags::NOFOLLOW | OPEN,
        Mode::empty(),
    )
    .map_err(|errno| open_error(&path, errno))?;
    read_opened(File::from(fd), &path, max)
}

/// Reads the open `file`, found at `path`, as a regular file of at most `max`
/// bytes.
fn read_opened(file: File, path: &Path, max: u64) -> io::Result<Vec<u8>> {
    if !file.metadata()?.is_file() {
        return Err(not_regular(path));
    }
    let mut bytes = Vec::new();
    file.take(max.saturating_add(1)).read_to_end(&mut bytes)?;
    if !u64::try_from(bytes.len()).is_ok_and(|len| len <= max) {
        return Err(io::Error::new(
            io::ErrorKind::FileTooLarge,
            format!("{} is larger than {max} bytes", path.display()),
        ));
    }
    Ok(bytes)
}

/// The error of opening `path`: a symbolic link refused by `NOFOLLOW` (`ELOOP`,
/// or `ENOTDIR` for a directory) reads as [`not_regular`].
fn open_error(path: &Path, errno: rustix::io::Errno) -> io::Error {
    if errno == rustix::io::Errno::LOOP || errno == rustix::io::Errno::NOTDIR {
        not_regular(path)
    } else {
        io::Error::from(errno)
    }
}

/// The regular file at `path`, opened to append, never through a symbolic
/// link at `path` and only when it has a single link, so a write cannot land in
/// a file outside the directory.
///
/// # Errors
///
/// Returns the error of opening `path`, and `InvalidInput` when it is a
/// symbolic link, not a regular file, or a file with more than one link.
pub(crate) fn open_append_regular(path: &Path) -> io::Result<File> {
    use std::os::unix::fs::MetadataExt as _;

    let fd = rustix::fs::open(
        path,
        OFlags::WRONLY | OFlags::APPEND | OFlags::NOFOLLOW | OPEN,
        Mode::empty(),
    )
    .map_err(|errno| open_error(path, errno))?;
    let file = File::from(fd);
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.nlink() != 1 {
        return Err(not_regular(path));
    }
    Ok(file)
}

/// Writes `content` to `dir/name` through [`write_atomic`](crate::runs::write_atomic):
/// a new temporary file renamed over it, so a symbolic link at `dir/name` is
/// replaced, never written through.
///
/// # Errors
///
/// Returns the error of creating, writing or renaming the temporary file.
pub(crate) fn replace(dir: &Path, name: &str, content: &[u8]) -> io::Result<()> {
    crate::runs::write_atomic(dir, name, content).map_err(|error| match error {
        crate::runs::RunsError::Io { source, .. } => source,
        other => io::Error::other(other.to_string()),
    })
}

/// [`read_regular`] as UTF-8 text.
///
/// # Errors
///
/// Returns the errors of [`read_regular`], and `InvalidData` when the file is
/// not UTF-8.
pub(crate) fn read_regular_text(path: &Path, max: u64) -> io::Result<String> {
    String::from_utf8(read_regular(path, max)?).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} is not UTF-8", path.display()),
        )
    })
}

/// The error for a `path` that is a symbolic link or not a regular file (or,
/// for a directory, not a private one).
fn not_regular(path: &Path) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "{} is a symbolic link or not a regular file",
            path.display()
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// A regular file within the limit is read whole.
    #[test]
    fn a_regular_file_is_read() -> TestResult {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("a");
        std::fs::write(&path, "abc")?;
        assert_eq!(read_regular(&path, 3)?, b"abc");
        assert_eq!(read_regular_text(&path, 10)?, "abc");
        Ok(())
    }

    /// A symbolic link, even to a regular file, is refused.
    #[test]
    fn a_symbolic_link_is_refused() -> TestResult {
        let dir = tempfile::tempdir()?;
        let target = dir.path().join("target");
        std::fs::write(&target, "secret")?;
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&target, &link)?;
        let error = read_regular(&link, 100).err().ok_or("a link was read")?;
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        Ok(())
    }

    /// A device and a directory are refused, so `/dev/zero` cannot make a
    /// read endless.
    #[test]
    fn a_device_or_a_directory_is_refused() -> TestResult {
        for path in [Path::new("/dev/zero"), Path::new("/")] {
            let error = read_regular(path, 100).err().ok_or("not refused")?;
            assert_eq!(
                error.kind(),
                io::ErrorKind::InvalidInput,
                "{}",
                path.display()
            );
        }
        Ok(())
    }

    /// Appending refuses a link and a file with a second hard link, and adds
    /// to a plain file.
    #[test]
    fn appending_refuses_links() -> TestResult {
        use std::io::Write as _;

        let dir = tempfile::tempdir()?;
        let target = dir.path().join("target");
        std::fs::write(&target, "a")?;
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&target, &link)?;
        let hard = dir.path().join("hard");
        std::fs::hard_link(&target, &hard)?;
        for path in [&link, &hard] {
            let error = open_append_regular(path).err().ok_or("not refused")?;
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        }
        let plain = dir.path().join("plain");
        std::fs::write(&plain, "a")?;
        open_append_regular(&plain)?.write_all(b"b")?;
        assert_eq!(std::fs::read_to_string(&plain)?, "ab");
        assert_eq!(std::fs::read_to_string(&target)?, "a");
        Ok(())
    }

    /// Replacing a file that is a link replaces the link, and the file it
    /// pointed to stays as it was.
    #[test]
    fn replacing_a_link_leaves_its_target() -> TestResult {
        let dir = tempfile::tempdir()?;
        let target = dir.path().join("target");
        std::fs::write(&target, "keep")?;
        std::os::unix::fs::symlink(&target, dir.path().join("name"))?;
        replace(dir.path(), "name", b"new")?;
        assert_eq!(std::fs::read_to_string(&target)?, "keep");
        assert_eq!(read_regular(&dir.path().join("name"), 10)?, b"new");
        Ok(())
    }

    /// A bounded read follows a link to a regular file, and refuses a link to
    /// a device.
    #[test]
    fn a_bounded_read_follows_a_link_to_a_regular_file() -> TestResult {
        let dir = tempfile::tempdir()?;
        let target = dir.path().join("target");
        std::fs::write(&target, "abc")?;
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&target, &link)?;
        assert_eq!(read_bounded(&link, 3)?, b"abc");
        let zero = dir.path().join("zero");
        std::os::unix::fs::symlink("/dev/zero", &zero)?;
        let error = read_bounded(&zero, 3).err().ok_or("a device was read")?;
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        Ok(())
    }

    /// A private read takes a file in a directory closed to others, and
    /// refuses a directory open to others or swapped for a link.
    #[test]
    fn a_private_read_needs_a_closed_directory() -> TestResult {
        use std::os::unix::fs::PermissionsExt as _;

        let root = tempfile::tempdir()?;
        let dir = root.path().join("session");
        std::fs::create_dir(&dir)?;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))?;
        std::fs::write(dir.join("edit.txt"), "text")?;
        assert_eq!(read_private(&dir, "edit.txt", 10)?, b"text");
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o770))?;
        let error = read_private(&dir, "edit.txt", 10)
            .err()
            .ok_or("open dir read")?;
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))?;
        let swapped = root.path().join("swapped");
        std::os::unix::fs::symlink(&dir, &swapped)?;
        let error = read_private(&swapped, "edit.txt", 10)
            .err()
            .ok_or("link read")?;
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        Ok(())
    }

    /// A file past the limit is refused, and a missing one is `NotFound`.
    #[test]
    fn a_large_or_missing_file_is_refused() -> TestResult {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("a");
        std::fs::write(&path, "abcd")?;
        let error = read_regular(&path, 3).err().ok_or("not refused")?;
        assert_eq!(error.kind(), io::ErrorKind::FileTooLarge);
        let error = read_regular(&dir.path().join("none"), 3)
            .err()
            .ok_or("not refused")?;
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        Ok(())
    }
}
