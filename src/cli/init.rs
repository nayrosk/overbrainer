//! The `overbrainer init` subcommand.

use std::path::Path;

use anyhow::{Context, bail};

const FILES: [(&str, &str); 3] = [
    (
        "overbrainer.toml",
        include_str!("../../templates/overbrainer.toml"),
    ),
    (".env.example", include_str!("../../templates/env.example")),
    (".gitignore", include_str!("../../templates/gitignore")),
];

/// Writes the example project files into `dir`, refusing to overwrite anything.
///
/// # Errors
///
/// Returns an error if `dir` cannot be created, if any of the target files already
/// exists, or if a file cannot be written.
pub fn run(dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
    for (name, _) in FILES {
        let path = dir.join(name);
        if path.exists() {
            bail!("{} already exists", path.display());
        }
    }
    for (name, content) in FILES {
        let path = dir.join(name);
        std::fs::write(&path, content)
            .with_context(|| format!("cannot write {}", path.display()))?;
        tracing::info!("created {}", path.display());
    }
    Ok(())
}
