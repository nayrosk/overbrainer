//! Choosing which files of a run's `output/` go to the Hub.

use std::path::{Path, PathBuf};
use std::{collections::BTreeMap, fs, io};

use super::UploadFile;

/// Errors choosing the files of a run.
#[derive(Debug, thiserror::Error)]
pub enum SelectError {
    /// A directory or a file's metadata could not be read.
    #[error("cannot read {}", path.display())]
    Read {
        /// The directory or the file.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// A path is not valid UTF-8, so it cannot name a file in the repo.
    #[error("{} is not valid UTF-8", path.display())]
    NotUtf8 {
        /// The file.
        path: PathBuf,
    },
    /// A file found outside the directory walked.
    #[error("{} is outside {}", path.display(), root.display())]
    Outside {
        /// The file.
        path: PathBuf,
        /// The directory walked.
        root: PathBuf,
    },
    /// Two files land on the same path in the repo.
    #[error("{} and {} both map to {path_in_repo} in the repo", first.display(), second.display())]
    Clash {
        /// The file found first.
        first: PathBuf,
        /// The file found second.
        second: PathBuf,
        /// The path in the repo they both take.
        path_in_repo: String,
    },
}

/// The repo path of the card, which no file of the run may take.
const CARD: &str = "README.md";

/// The files of a run's `output/` to push, with their place in the repo.
/// Skips `debug.log`, `checkpoint-*` directories, hidden entries and any file
/// that would land on the repo's `README.md`, which is the card.
/// `merged/**` keeps its prefix; `gguf/**` goes to the repo root.
///
/// # Errors
///
/// Returns a [`SelectError`] when a directory cannot be read, a path is not
/// valid UTF-8, or two files land on the same repo path.
pub fn select(output: &Path) -> Result<Vec<UploadFile>, SelectError> {
    let mut chosen: BTreeMap<String, UploadFile> = BTreeMap::new();
    walk(output, output, &mut chosen)?;
    Ok(chosen.into_values().collect())
}

/// Walks `dir` and collects the files to upload into `chosen`, keyed by their path in the repo.
fn walk(
    root: &Path,
    dir: &Path,
    chosen: &mut BTreeMap<String, UploadFile>,
) -> Result<(), SelectError> {
    let read = |path: &Path| {
        let path = path.to_path_buf();
        move |source| SelectError::Read { path, source }
    };
    let entries = fs::read_dir(dir).map_err(read(dir))?;
    for entry in entries {
        let entry = entry.map_err(read(dir))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let path = entry.path();
        let at_root = dir == root;
        if name.starts_with('.') {
            continue;
        }
        let meta = fs::symlink_metadata(&path).map_err(read(&path))?;
        if meta.is_dir() {
            if at_root && name.starts_with("checkpoint-") {
                continue;
            }
            walk(root, &path, chosen)?;
        } else if meta.is_file() {
            if at_root && name == "debug.log" {
                continue;
            }
            let path_in_repo = repo_path(root, &path)?;
            if path_in_repo == CARD {
                continue;
            }
            let file = UploadFile {
                local: path.clone(),
                path_in_repo: path_in_repo.clone(),
                size: meta.len(),
            };
            if let Some(other) = chosen.insert(path_in_repo.clone(), file) {
                return Err(SelectError::Clash {
                    first: other.local,
                    second: path,
                    path_in_repo,
                });
            }
        }
    }
    Ok(())
}

/// The path in the repo, `/`-separated, with the `gguf/` prefix dropped.
fn repo_path(root: &Path, path: &Path) -> Result<String, SelectError> {
    let relative = path.strip_prefix(root).map_err(|_| SelectError::Outside {
        path: path.to_path_buf(),
        root: root.to_path_buf(),
    })?;
    let mut parts = Vec::new();
    for component in relative.components() {
        let part = component
            .as_os_str()
            .to_str()
            .ok_or_else(|| SelectError::NotUtf8 {
                path: path.to_path_buf(),
            })?;
        parts.push(part);
    }
    if parts.len() > 1 && parts.first() == Some(&"gguf") {
        parts.remove(0);
    }
    Ok(parts.join("/"))
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    /// A temporary tree holding the given files.
    fn tree(files: &[&str]) -> Result<tempfile::TempDir, Box<dyn Error>> {
        let dir = tempfile::tempdir()?;
        for f in files {
            let path = dir.path().join(f);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, "data")?;
        }
        Ok(dir)
    }

    /// The repo paths `select` picks for `dir`.
    fn paths(dir: &Path) -> Result<Vec<String>, Box<dyn Error>> {
        Ok(select(dir)?.into_iter().map(|f| f.path_in_repo).collect())
    }

    /// The layout flattens the GGUF files and keeps the merged model.
    #[test]
    fn layout_flattens_gguf_and_keeps_merged() -> Result<(), Box<dyn Error>> {
        let dir = tree(&[
            "adapter_config.json",
            "adapter_model.safetensors",
            "README.md",
            "debug.log",
            "checkpoint-48/adapter_model.safetensors",
            ".hidden",
            "merged/config.json",
            "merged/model.safetensors",
            "gguf/run-Q4_K_M.gguf",
            "gguf/Modelfile",
        ])?;
        assert_eq!(
            paths(dir.path())?,
            vec![
                "Modelfile",
                "adapter_config.json",
                "adapter_model.safetensors",
                "merged/config.json",
                "merged/model.safetensors",
                "run-Q4_K_M.gguf"
            ]
        );
        Ok(())
    }

    /// The sizes come from the files on disk.
    #[test]
    fn sizes_come_from_the_files() -> Result<(), Box<dyn Error>> {
        let dir = tempfile::tempdir()?;
        fs::write(dir.path().join("a.bin"), b"12345")?;
        let files = select(dir.path())?;
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].size, 5);
        assert_eq!(files[0].local, dir.path().join("a.bin"));
        Ok(())
    }

    /// Two files that get the same name after flattening are an error.
    #[test]
    fn a_name_clash_after_flattening_is_an_error() -> Result<(), Box<dyn Error>> {
        let dir = tree(&["Modelfile", "gguf/Modelfile"])?;
        let err = select(dir.path()).err().ok_or("expected a clash")?;
        let text = format!("{err:#}");
        assert!(text.contains("map to Modelfile"), "names the clashing path");
        let ours = dir.path().join("Modelfile");
        let theirs = dir.path().join("gguf/Modelfile");
        assert!(
            text.contains(&ours.display().to_string()),
            "names the root Modelfile: {text}"
        );
        assert!(
            text.contains(&theirs.display().to_string()),
            "names the gguf Modelfile: {text}"
        );
        Ok(())
    }

    /// No file may land on the card's `README.md`.
    #[test]
    fn nothing_may_land_on_the_card() -> Result<(), Box<dyn Error>> {
        let dir = tree(&["adapter_config.json", "gguf/README.md", "gguf/m.gguf"])?;
        assert_eq!(paths(dir.path())?, vec!["adapter_config.json", "m.gguf"]);
        Ok(())
    }

    /// A nested GGUF keeps its relative path, and a hidden file is skipped at any depth.
    #[test]
    fn nested_gguf_keeps_its_relative_path_and_hidden_is_skipped_at_depth()
    -> Result<(), Box<dyn Error>> {
        let dir = tree(&[
            "gguf/a/b.gguf",
            "merged/README.md",
            "merged/.cache/x",
            "merged/checkpoint-1/y",
            "checkpoint-2/z",
            "sub/debug.log",
        ])?;
        assert_eq!(
            paths(dir.path())?,
            vec![
                "a/b.gguf",
                "merged/README.md",
                "merged/checkpoint-1/y",
                "sub/debug.log"
            ]
        );
        Ok(())
    }

    /// Symbolic links are skipped.
    #[cfg(unix)]
    #[test]
    fn symlinks_are_skipped() -> Result<(), Box<dyn Error>> {
        let dir = tree(&["real.txt"])?;
        std::os::unix::fs::symlink(dir.path().join("real.txt"), dir.path().join("link.txt"))?;
        assert_eq!(paths(dir.path())?, vec!["real.txt"]);
        Ok(())
    }
}
