//! Replacing `overbrainer.toml` only while it still holds the text it was
//! read from. Every step goes through one descriptor of the project directory.
//!
//! On Linux the new file is swapped in with `RENAME_EXCHANGE`, then the file
//! it moved aside is read again: when it no longer holds that text, the swap
//! is undone. Elsewhere, and on a file system without the exchange, the
//! identity of the file read (device, inode, size, mtime) is checked again
//! just before the rename: a write landing between that check and the rename
//! is still lost there.

use std::fs::File;
use std::io::{self, Read as _, Write as _};
use std::os::fd::OwnedFd;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};

use rustix::fs::{AtFlags, Mode, OFlags, Stat};
use rustix::io::Errno;

use super::project_edit::SaveRefusal;
use crate::config::CONFIG_FILE;

/// The new text of `overbrainer.toml`, written and synced to a temporary file
/// next to it, not yet in place. Dropped uncommitted, the temporary file goes.
pub(super) struct Staged {
    /// The project directory.
    dir: OwnedFd,
    /// `overbrainer.toml` in it, for messages.
    path: PathBuf,
    /// Name of the temporary file in `dir`.
    tmp: String,
    /// What the file held when it was read.
    #[cfg(target_os = "linux")]
    base: String,
    /// The file as it was read.
    read: Stat,
    /// Whether the temporary file is gone or must stay.
    done: bool,
}

/// Reads `overbrainer.toml` in `dir` through one handle, never through a link
/// nor from a FIFO, checks that it still holds `base`, then writes `text` to a
/// temporary file private until it takes the file's mode (never setuid, setgid
/// or sticky), synced to disk.
///
/// # Errors
///
/// Returns [`SaveRefusal::Failed`] when the file is a symlink, cannot be read,
/// no longer holds `base`, or the temporary file cannot be written.
pub(super) fn stage(dir: &Path, text: &str, base: &str) -> Result<Staged, SaveRefusal> {
    let path = dir.join(CONFIG_FILE);
    let cannot_read =
        |error: io::Error| SaveRefusal::Failed(format!("cannot read {}: {error}", path.display()));
    let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC;
    let dir =
        rustix::fs::open(dir, flags, Mode::empty()).map_err(|errno| cannot_read(errno.into()))?;
    let (read, on_disk, mode) = match read_regular(&dir, CONFIG_FILE) {
        Ok(read) => read,
        Err(error) if error.raw_os_error() == Some(Errno::LOOP.raw_os_error()) => {
            return Err(SaveRefusal::Failed(format!(
                "{CONFIG_FILE} is a symlink; nothing written, edit it with E"
            )));
        },
        Err(error) => return Err(cannot_read(error)),
    };
    if on_disk != base {
        return Err(SaveRefusal::Failed(format!(
            "{CONFIG_FILE} changed on disk since it was read; drop the changes (u), then E"
        )));
    }
    let tmp = format!(".{CONFIG_FILE}.{:016x}.tmp", fastrand::u64(..));
    let flags = OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let file = rustix::fs::openat(&dir, tmp.as_str(), flags, Mode::RUSR | Mode::WUSR)
        .map_err(|errno| cannot_write(&path, &io::Error::from(errno)))?;
    let staged = Staged {
        dir,
        path,
        tmp,
        #[cfg(target_os = "linux")]
        base: base.to_string(),
        read,
        done: false,
    };
    let mut file = File::from(file);
    file.write_all(text.as_bytes())
        .and_then(|()| file.set_permissions(std::fs::Permissions::from_mode(mode & 0o777)))
        .and_then(|()| file.sync_all())
        .map_err(|error| cannot_write(&staged.path, &error))?;
    Ok(staged)
}

impl Staged {
    /// Puts the new text in place unless the file changed since it was read,
    /// then syncs the directory.
    ///
    /// # Errors
    ///
    /// Returns [`SaveRefusal::Failed`] when the file changed on disk while
    /// saving, or the rename fails; the file on disk is then left as it is.
    pub(super) fn commit(self) -> Result<(), SaveRefusal> {
        #[cfg(target_os = "linux")]
        {
            self.exchange()
        }
        #[cfg(not(target_os = "linux"))]
        {
            self.checked_rename()
        }
    }

    /// Swaps the new file in, then reads what it moved aside: when that is not
    /// `base`, swaps back and refuses.
    #[cfg(target_os = "linux")]
    fn exchange(mut self) -> Result<(), SaveRefusal> {
        use rustix::fs::RenameFlags;
        let swap = |staged: &Self| {
            rustix::fs::renameat_with(
                &staged.dir,
                staged.tmp.as_str(),
                &staged.dir,
                CONFIG_FILE,
                RenameFlags::EXCHANGE,
            )
        };
        match swap(&self) {
            Ok(()) => {},
            Err(Errno::INVAL | Errno::NOSYS | Errno::OPNOTSUPP) => return self.checked_rename(),
            Err(Errno::NOENT) => return Err(changed_while_saving()),
            Err(errno) => return Err(cannot_write(&self.path, &io::Error::from(errno))),
        }
        let moved = read_regular(&self.dir, self.tmp.as_str());
        if moved.is_ok_and(|(_, text, _)| text == self.base) {
            self.finish();
            return Ok(());
        }
        if let Err(errno) = swap(&self) {
            // The other writer's file is under the temporary name: keep it.
            self.done = true;
            return Err(SaveRefusal::Failed(format!(
                "{CONFIG_FILE} changed on disk while saving and cannot be put back: \
                 {errno}; its content is in {}",
                self.tmp
            )));
        }
        Err(changed_while_saving())
    }

    /// Renames the new file over the old one when the old one is still the
    /// file read: same device, inode, size and mtime.
    fn checked_rename(mut self) -> Result<(), SaveRefusal> {
        let now = rustix::fs::statat(&self.dir, CONFIG_FILE, AtFlags::SYMLINK_NOFOLLOW);
        if !now.is_ok_and(|now| identity(&now) == identity(&self.read)) {
            return Err(changed_while_saving());
        }
        rustix::fs::renameat(&self.dir, self.tmp.as_str(), &self.dir, CONFIG_FILE)
            .map_err(|errno| cannot_write(&self.path, &io::Error::from(errno)))?;
        self.done = true;
        self.sync_dir();
        Ok(())
    }

    /// Removes the temporary file, which holds the old text, and syncs the
    /// directory.
    #[cfg(target_os = "linux")]
    fn finish(&mut self) {
        self.remove_tmp();
        self.done = true;
        self.sync_dir();
    }

    /// Syncs the directory, so the rename is on disk; a failure is only
    /// logged, the new text being in place already.
    fn sync_dir(&self) {
        if let Err(errno) = rustix::fs::fsync(&self.dir) {
            tracing::warn!(
                "cannot sync the directory of {}: {errno}",
                self.path.display()
            );
        }
    }

    fn remove_tmp(&self) {
        if let Err(errno) = rustix::fs::unlinkat(&self.dir, self.tmp.as_str(), AtFlags::empty()) {
            tracing::warn!("cannot remove {}: {errno}", self.tmp);
        }
    }
}

impl Drop for Staged {
    fn drop(&mut self) {
        if !self.done {
            self.remove_tmp();
        }
    }
}

/// `name` in `dir`, opened without following a link nor blocking on a FIFO,
/// when it is a regular file: its `stat`, its text and its mode.
fn read_regular(dir: &OwnedFd, name: &str) -> io::Result<(Stat, String, u32)> {
    let flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC;
    let fd = rustix::fs::openat(dir, name, flags, Mode::empty())?;
    let stat = rustix::fs::fstat(&fd)?;
    let mut file = File::from(fd);
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(io::Error::other("not a regular file"));
    }
    let mut text = String::new();
    file.read_to_string(&mut text)?;
    Ok((stat, text, metadata.permissions().mode()))
}

/// What names one version of a file: device, inode, size, mtime.
fn identity(stat: &Stat) -> impl PartialEq + use<> {
    (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime,
        stat.st_mtime_nsec,
    )
}

fn changed_while_saving() -> SaveRefusal {
    SaveRefusal::Failed(format!(
        "{CONFIG_FILE} changed on disk while saving; drop the changes (u), then E"
    ))
}

fn cannot_write(path: &Path, error: &io::Error) -> SaveRefusal {
    SaveRefusal::Failed(format!("cannot write {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::runs::tests::leftover_temp_files;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    const BASE: &str = "# kept comment\n[project]\nname = \"a\"\n";
    const NEW: &str = "# kept comment\n[project]\nname = \"bb\"\n";

    /// A project directory whose `overbrainer.toml` holds [`BASE`], mode `0o640`.
    fn project() -> io::Result<tempfile::TempDir> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join(CONFIG_FILE);
        fs::write(&path, BASE)?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640))?;
        Ok(dir)
    }

    /// Another program writing the file at `path`.
    type Writer = fn(&Path) -> io::Result<()>;

    /// Writes to the file in place, or renames a new file over it.
    fn writers() -> [(&'static str, Writer); 2] {
        [
            ("in place", |path| fs::write(path, "# theirs, in place\n")),
            ("by rename", |path| {
                let other = path.with_extension("other");
                fs::write(&other, "# theirs, renamed\n")?;
                fs::rename(other, path)
            }),
        ]
    }

    /// `result` with its refusal as text, for `?`.
    fn ok<T>(result: Result<T, SaveRefusal>) -> Result<T, String> {
        result.map_err(|refusal| format!("{refusal:?}"))
    }

    fn refused_while_saving(result: Result<(), SaveRefusal>) -> bool {
        matches!(result, Err(SaveRefusal::Failed(message)) if message.contains("while saving"))
    }

    #[test]
    fn a_commit_keeps_the_mode_and_leaves_no_temporary_file() -> TestResult {
        let dir = project()?;
        ok(ok(stage(dir.path(), NEW, BASE))?.commit())?;
        let path = dir.path().join(CONFIG_FILE);
        assert_eq!(fs::read_to_string(&path)?, NEW, "the comment is kept");
        assert_eq!(fs::metadata(&path)?.permissions().mode() & 0o7777, 0o640);
        assert!(leftover_temp_files(dir.path())?.is_empty());
        Ok(())
    }

    #[test]
    fn a_staged_save_dropped_leaves_the_file_and_no_temporary_file() -> TestResult {
        let dir = project()?;
        drop(ok(stage(dir.path(), NEW, BASE))?);
        assert_eq!(fs::read_to_string(dir.path().join(CONFIG_FILE))?, BASE);
        assert!(leftover_temp_files(dir.path())?.is_empty());
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_write_between_the_read_and_the_swap_is_kept_and_the_save_refused() -> TestResult {
        for (name, writer) in writers() {
            let dir = project()?;
            let path = dir.path().join(CONFIG_FILE);
            let staged = ok(stage(dir.path(), NEW, BASE))?;
            writer(&path)?;
            let theirs = fs::read_to_string(&path)?;
            assert!(refused_while_saving(staged.commit()), "{name}");
            assert_eq!(fs::read_to_string(&path)?, theirs, "{name}");
            assert!(leftover_temp_files(dir.path())?.is_empty(), "{name}");
        }
        Ok(())
    }

    #[test]
    fn the_identity_check_refuses_a_file_changed_before_the_rename() -> TestResult {
        for (name, writer) in writers() {
            let dir = project()?;
            let path = dir.path().join(CONFIG_FILE);
            let staged = ok(stage(dir.path(), NEW, BASE))?;
            writer(&path)?;
            let theirs = fs::read_to_string(&path)?;
            assert!(refused_while_saving(staged.checked_rename()), "{name}");
            assert_eq!(fs::read_to_string(&path)?, theirs, "{name}");
            assert!(leftover_temp_files(dir.path())?.is_empty(), "{name}");
        }
        let dir = project()?;
        ok(ok(stage(dir.path(), NEW, BASE))?.checked_rename())?;
        assert_eq!(fs::read_to_string(dir.path().join(CONFIG_FILE))?, NEW);
        Ok(())
    }
}
