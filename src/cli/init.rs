//! The `overbrainer init` subcommand.

use std::fs::OpenOptions;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};

use crate::prompts;

/// The example `overbrainer.toml`.
pub(crate) const CONFIG_TEMPLATE: &str = include_str!("../../templates/overbrainer.toml");
/// The example `.env.example`.
const ENV_TEMPLATE: &str = include_str!("../../templates/env.example");
/// The environment file, next to `overbrainer.toml`.
pub(crate) const ENV_FILE: &str = ".env";
/// Its example, meant to be committed.
pub(crate) const ENV_EXAMPLE_FILE: &str = ".env.example";

/// Files that are never overwritten, relative to the project directory: `init`
/// refuses when any of them exists.
fn files() -> Vec<(PathBuf, &'static str)> {
    let mut files = vec![
        (PathBuf::from(crate::config::CONFIG_FILE), CONFIG_TEMPLATE),
        (PathBuf::from(ENV_EXAMPLE_FILE), ENV_TEMPLATE),
    ];
    files.extend(prompt_files());
    files
}

/// The default prompt templates, relative to the project directory.
pub(crate) fn prompt_files() -> Vec<(PathBuf, &'static str)> {
    prompts::DEFAULTS
        .iter()
        .map(|(name, content)| (Path::new(prompts::DIR).join(name), *content))
        .collect()
}

/// `.gitignore`, relative to the project directory.
pub(crate) const GITIGNORE: &str = ".gitignore";
const GITIGNORE_TEMPLATE: &str = include_str!("../../templates/gitignore");

/// Writes the example project files into `dir`: `overbrainer.toml`, `.env.example`,
/// the default prompt templates in `prompts/`, `.gitignore`, and
/// `.overbrainer/version`, so the new project is never taken for an old one.
///
/// None of these files is ever overwritten: if one exists, nothing is written. An
/// existing `.gitignore` gets only the entries it lacks.
///
/// # Errors
///
/// Returns an error if `dir` cannot be created, if one of the files already exists,
/// or if a file cannot be read or written.
pub fn run(dir: &Path) -> anyhow::Result<()> {
    let files = files();
    for (name, _) in &files {
        let path = dir.join(name);
        if path.exists() {
            bail!("{} already exists", path.display());
        }
    }
    let prompts_dir = dir.join(prompts::DIR);
    std::fs::create_dir_all(&prompts_dir)
        .with_context(|| format!("cannot create {}", prompts_dir.display()))?;
    for (name, content) in &files {
        create_new(&dir.join(name), content)?;
    }
    update_gitignore(&dir.join(GITIGNORE))?;
    crate::project_format::write_current(dir).with_context(|| {
        format!(
            "cannot write the project format into {}",
            dir.join(crate::project_lock::STATE_DIR).display()
        )
    })
}

/// Creates `path` with `content`, failing if it exists, even if it appeared after
/// the existence check.
///
/// # Errors
///
/// Returns an error when the file exists or cannot be created or written.
pub(crate) fn create_new(path: &Path, content: &str) -> anyhow::Result<()> {
    create(
        path,
        content,
        OpenOptions::new().write(true).create_new(true),
    )
}

/// [`create_new`] for a file holding secrets: readable by its owner only
/// (mode 600) from its creation, so it is never world-readable, even briefly.
///
/// # Errors
///
/// Returns an error when the file exists or cannot be created or written.
pub(crate) fn create_private(path: &Path, content: &str) -> anyhow::Result<()> {
    use std::os::unix::fs::OpenOptionsExt as _;

    create(
        path,
        content,
        OpenOptions::new().write(true).create_new(true).mode(0o600),
    )
}

fn create(path: &Path, content: &str, options: &OpenOptions) -> anyhow::Result<()> {
    let mut file = options
        .open(path)
        .with_context(|| format!("cannot create {}", path.display()))?;
    let written = file
        .write_all(content.as_bytes())
        .and_then(|()| file.sync_all());
    drop(file);
    kept_or_removed(path, written)?;
    tracing::info!("created {}", path.display());
    Ok(())
}

/// The file this call just created at `path` stays when `written` is `Ok`;
/// otherwise it is removed, so nothing empty or partial (part of a secret)
/// stays, and creating it again works.
fn kept_or_removed(path: &Path, written: std::io::Result<()>) -> anyhow::Result<()> {
    written.map_err(|error| {
        std::fs::remove_file(path).ok();
        anyhow::Error::from(error).context(format!("cannot write {}", path.display()))
    })
}

/// Creates `.gitignore` from the template, or appends the template entries that an
/// existing file lacks.
///
/// # Errors
///
/// Returns an error when the file cannot be read, created or written.
pub(crate) fn update_gitignore(path: &Path) -> anyhow::Result<()> {
    add_gitignore_entries(path, GITIGNORE_TEMPLATE)
}

/// Creates `.gitignore` at `path` with `entries`, or appends the entries
/// (non-empty lines) that an existing file lacks. Never rewrites the file.
///
/// # Errors
///
/// Returns an error when the file cannot be read, created or appended to.
pub(crate) fn add_gitignore_entries(path: &Path, entries: &str) -> anyhow::Result<()> {
    let existing = match std::fs::read_to_string(path) {
        Ok(existing) => existing,
        Err(e) if e.kind() == ErrorKind::NotFound => {
            return create_new(path, entries);
        },
        Err(e) => return Err(e).with_context(|| format!("cannot read {}", path.display())),
    };
    let missing = missing_entries(&existing, entries);
    if missing.is_empty() {
        return Ok(());
    }
    let mut addition = String::new();
    if !existing.is_empty() && !existing.ends_with('\n') {
        addition.push('\n');
    }
    for entry in &missing {
        addition.push_str(entry);
        addition.push('\n');
    }
    let mut file = OpenOptions::new()
        .append(true)
        .open(path)
        .with_context(|| format!("cannot open {}", path.display()))?;
    file.write_all(addition.as_bytes())
        .with_context(|| format!("cannot write {}", path.display()))?;
    tracing::info!("added {} to {}", missing.join(", "), path.display());
    Ok(())
}

/// Template entries (non-empty lines) not already present as a line of `existing`.
fn missing_entries<'a>(existing: &str, template: &'a str) -> Vec<&'a str> {
    let present: Vec<&str> = existing.lines().map(str::trim).collect();
    template
        .lines()
        .map(str::trim)
        .filter(|entry| !entry.is_empty() && !present.contains(entry))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::io::{self, Write as _};

    use super::{create_private, kept_or_removed, missing_entries};

    /// A file whose writes fail once opened, like a full disk.
    struct Full;

    impl io::Write for Full {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("no space left on device"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_failed_write_removes_the_file_it_created_and_a_retry_works()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join(".env");
        std::fs::File::create(&path)?;
        let error = kept_or_removed(&path, Full.write_all(b"KEY=secret\n"))
            .err()
            .ok_or("the write did not fail")?;
        assert!(
            format!("{error:#}").starts_with("cannot write "),
            "{error:#}"
        );
        assert!(!path.exists(), "nothing stays");
        create_private(&path, "KEY=secret\n")?;
        assert_eq!(std::fs::read_to_string(&path)?, "KEY=secret\n");
        Ok(())
    }

    #[test]
    fn only_absent_entries_are_missing() {
        assert_eq!(
            missing_entries("target/\n .env \n", ".env\n/data/\n\n/runs/\n"),
            vec!["/data/", "/runs/"]
        );
        assert_eq!(
            missing_entries(".env\n/data/\n/runs/\n", ".env\n/data/\n/runs/\n"),
            [] as [&str; 0]
        );
    }
}
