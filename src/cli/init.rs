//! The `overbrainer init` subcommand.

use std::fs::OpenOptions;
use std::io::{ErrorKind, Write};
use std::path::Path;

use anyhow::{Context, bail};

/// Files that are never overwritten: `init` refuses when any of them exists.
const FILES: [(&str, &str); 2] = [
    (
        "overbrainer.toml",
        include_str!("../../templates/overbrainer.toml"),
    ),
    (".env.example", include_str!("../../templates/env.example")),
];

const GITIGNORE: &str = ".gitignore";
const GITIGNORE_TEMPLATE: &str = include_str!("../../templates/gitignore");

/// Writes the example project files into `dir`.
///
/// `overbrainer.toml` and `.env.example` are never overwritten: if either exists,
/// nothing is written. An existing `.gitignore` gets only the entries it lacks.
///
/// # Errors
///
/// Returns an error if `dir` cannot be created, if `overbrainer.toml` or
/// `.env.example` already exists, or if a file cannot be read or written.
pub fn run(dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
    for (name, _) in FILES {
        let path = dir.join(name);
        if path.exists() {
            bail!("{} already exists", path.display());
        }
    }
    for (name, content) in FILES {
        create_new(&dir.join(name), content)?;
    }
    update_gitignore(&dir.join(GITIGNORE))
}

/// Creates `path` with `content`, failing if it exists, even if it appeared after
/// the existence check.
fn create_new(path: &Path, content: &str) -> anyhow::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("cannot create {}", path.display()))?;
    file.write_all(content.as_bytes())
        .with_context(|| format!("cannot write {}", path.display()))?;
    tracing::info!("created {}", path.display());
    Ok(())
}

/// Creates `.gitignore` from the template, or appends the template entries that an
/// existing file lacks.
fn update_gitignore(path: &Path) -> anyhow::Result<()> {
    let existing = match std::fs::read_to_string(path) {
        Ok(existing) => existing,
        Err(e) if e.kind() == ErrorKind::NotFound => {
            return create_new(path, GITIGNORE_TEMPLATE);
        },
        Err(e) => return Err(e).with_context(|| format!("cannot read {}", path.display())),
    };
    let missing = missing_entries(&existing, GITIGNORE_TEMPLATE);
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
    use super::missing_entries;

    #[test]
    fn only_absent_entries_are_missing() {
        assert_eq!(
            missing_entries("target/\n .env \n", ".env\n/data/\n\n/runs/\n"),
            vec!["/data/", "/runs/"]
        );
        assert!(missing_entries(".env\n/data/\n/runs/\n", ".env\n/data/\n/runs/\n").is_empty());
    }
}
