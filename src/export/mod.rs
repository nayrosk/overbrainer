//! Export of a trained run to GGUF with llama.cpp, and its Ollama Modelfile.
//!
//! The export is a job like training: [`ExportJob`] is a [`Trainer`] whose one
//! command runs `export.sh` with the target's runtime, so torch and
//! transformers come from the Axolotl image. It runs at the end of a training
//! job (`[export] after_training`), or alone with `overbrainer export`, in a
//! job directory of its own, `runs/<run-id>/exports/<export-id>/`.
//!
//! Once the GGUF is back, [`deliver`] puts it in `runs/<run-id>/output/gguf/`,
//! records it in `export.json`, and writes the Modelfile beside it.

mod llama_cpp;

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::Serialize;

pub use llama_cpp::{
    CUDA as LLAMA_CPP_CUDA, CUDART_ARM64_SHA256, CUDART_X64_SHA256, MACOS_ARM64_SHA256, REPO_URL,
    SOURCE_SHA256, TAG as LLAMA_CPP_TAG, UBUNTU_ARM64_SHA256, UBUNTU_CUDA_ARM64_SHA256,
    UBUNTU_CUDA_X64_SHA256, UBUNTU_X64_SHA256, env as llama_cpp_env,
};

use crate::train::{
    Artifacts, CONFIG_FILE, JobStage, METRICS_ENV, METRICS_FILE, OUTPUT_DIR, TrainError, Trainer,
    top_level_scalar,
};

/// The export script, written into the job directory.
pub const EXPORT_SCRIPT: &str = include_str!("export.sh");

/// The export script's file name, in the job directory.
pub const SCRIPT_FILE: &str = "export.sh";

/// Directory of a run holding its export jobs, one directory each.
pub const EXPORTS_DIR: &str = "exports";

/// Where an export writes its GGUF, relative to its job directory; the same
/// place in the run directory once delivered.
pub const GGUF_DIR: &str = "output/gguf";

/// What an export records of its GGUF, in its job directory.
pub const EXPORT_FILE: &str = "export.json";

/// The Ollama Modelfile, beside the GGUF.
pub const MODELFILE: &str = "Modelfile";

/// Starts the export script with the Python the runtime resolved (the virtual
/// environment's, or the image's), its directory first on `PATH` so the
/// script's `python3` and `axolotl` are that environment's too.
pub(crate) const TRAMPOLINE: &str = "import os, sys; bin = os.path.dirname(sys.executable); \
     os.environ['PATH'] = bin + os.pathsep + os.environ.get('PATH', ''); \
     os.execvp('sh', ['sh'] + sys.argv[1:])";

/// The Python program the runtime resolves to start the export.
pub(crate) const PYTHON: &str = "python3";

/// Files and directories of a run's `output/` never staged for an export: the
/// checkpoints, and the GGUF files of earlier exports.
const NOT_STAGED: [&str; 2] = ["checkpoint-", "gguf"];

/// Errors while delivering an export.
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// A file or directory could not be read or written.
    #[error("cannot access {}", path.display())]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// The job succeeded, but left no GGUF file.
    #[error("the export left no GGUF file in {}", .0.display())]
    NoGguf(PathBuf),
}

fn io_error(path: &Path) -> impl FnOnce(io::Error) -> ExportError + '_ {
    move |source| ExportError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// An export job: the run's model to GGUF, quantized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportJob {
    /// Name of the GGUF file, `<name>-<quantize>.gguf`: the run ID.
    name: String,
    /// The llama-quantize type.
    quantize: String,
    /// The adapter, or the model of a full fine-tune, relative to the job
    /// directory.
    model: String,
    /// The run's `axolotl.yaml`, relative to the job directory.
    config: String,
    /// The script as the job's program gets it.
    script: String,
    /// The local run directory and its model, relative to it, to hard-link
    /// into the job directory before it is uploaded, when the job runs where
    /// the run's files are not.
    stage: Option<(PathBuf, String)>,
}

impl ExportJob {
    /// The export at the end of a training job: its job directory is the run
    /// directory, and its model the run's `output/`.
    #[must_use]
    pub fn in_job(name: &str, quantize: &str) -> Self {
        Self {
            name: name.to_string(),
            quantize: quantize.to_string(),
            model: OUTPUT_DIR.to_string(),
            config: CONFIG_FILE.to_string(),
            script: SCRIPT_FILE.to_string(),
            stage: None,
        }
    }

    /// An export in `runs/<run-id>/exports/<export-id>/` on the run's own
    /// target, reading the run's `model` (relative to the run directory) and
    /// `axolotl.yaml` where they are, two directories up. `script` is the
    /// script's path as the job sees it.
    #[must_use]
    pub fn in_place(name: &str, quantize: &str, model: &str, script: String) -> Self {
        Self {
            name: name.to_string(),
            quantize: quantize.to_string(),
            model: format!("../../{model}"),
            config: format!("../../{CONFIG_FILE}"),
            script,
            stage: None,
        }
    }

    /// An export on another machine than the run's files: `model` (relative
    /// to the local run directory `run_dir`) and `axolotl.yaml` are
    /// hard-linked into the job directory when it is prepared, and uploaded
    /// with it.
    #[must_use]
    pub fn staged(name: &str, quantize: &str, run_dir: &Path, model: &str) -> Self {
        Self {
            name: name.to_string(),
            quantize: quantize.to_string(),
            model: model.to_string(),
            config: CONFIG_FILE.to_string(),
            script: SCRIPT_FILE.to_string(),
            stage: Some((run_dir.to_path_buf(), model.to_string())),
        }
    }

    /// The GGUF file this export writes, `<name>-<quantize>.gguf`.
    #[must_use]
    pub fn file_name(&self) -> String {
        gguf_name(&self.name, &self.quantize)
    }

    /// The llama-quantize type.
    #[must_use]
    pub fn quantize(&self) -> &str {
        &self.quantize
    }

    /// The command starting the export.
    #[must_use]
    pub fn command(&self) -> Vec<String> {
        vec![
            PYTHON.to_string(),
            "-c".to_string(),
            TRAMPOLINE.to_string(),
            self.script.clone(),
        ]
    }

    /// The environment of the script, but the metrics file: what it exports,
    /// how, and the llama.cpp release it uses.
    #[must_use]
    pub fn script_env(&self) -> Vec<(String, String)> {
        let mut env: Vec<(String, String)> = [
            ("OVERBRAINER_EXPORT_NAME", &self.name),
            ("OVERBRAINER_EXPORT_QUANTIZE", &self.quantize),
            ("OVERBRAINER_EXPORT_MODEL", &self.model),
            ("OVERBRAINER_EXPORT_CONFIG", &self.config),
        ]
        .into_iter()
        .map(|(name, value)| (name.to_string(), value.clone()))
        .collect();
        env.extend(llama_cpp::env());
        env
    }

    /// Writes the script into the local directory `dir`.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::Io`] when it cannot be written.
    pub fn write_script(dir: &Path) -> Result<(), TrainError> {
        fs::create_dir_all(dir).map_err(train_io(dir))?;
        let path = dir.join(SCRIPT_FILE);
        fs::write(&path, EXPORT_SCRIPT).map_err(train_io(&path))
    }
}

/// `<name>-<quantize>.gguf`.
#[must_use]
pub fn gguf_name(name: &str, quantize: &str) -> String {
    format!("{name}-{quantize}.gguf")
}

fn train_io(path: &Path) -> impl FnOnce(io::Error) -> TrainError + '_ {
    move |source| TrainError::Io {
        path: path.to_path_buf(),
        source,
    }
}

impl Trainer for ExportJob {
    fn prepare(&self, run_dir: &Path, _root: &str) -> Result<(), TrainError> {
        Self::write_script(run_dir)?;
        let Some((from, model)) = &self.stage else {
            return Ok(());
        };
        let config = from.join(CONFIG_FILE);
        link_file(&config, &run_dir.join(CONFIG_FILE))?;
        // The model only: a checkpoint is left out unless it is the model.
        let output = from.join(OUTPUT_DIR);
        let into = run_dir.join(OUTPUT_DIR);
        if model == OUTPUT_DIR {
            stage_output(&output, &into)
        } else {
            link_tree(&from.join(model), &run_dir.join(model))
        }
    }

    fn commands(&self) -> Vec<Vec<String>> {
        vec![self.command()]
    }

    fn env(&self, root: &str) -> Vec<(String, String)> {
        let mut env = vec![(METRICS_ENV.to_string(), format!("{root}/{METRICS_FILE}"))];
        env.extend(self.script_env());
        env
    }

    fn metrics_file(&self) -> &'static str {
        METRICS_FILE
    }

    fn artifacts(&self) -> Artifacts {
        Artifacts {
            entries: vec![GGUF_DIR.to_string(), METRICS_FILE.to_string()],
            exclude: vec!["*.part".to_string()],
            required: Some(GGUF_DIR.to_string()),
        }
    }

    fn stages(&self) -> Vec<JobStage> {
        vec![JobStage::Export]
    }

    fn metrics_required(&self) -> bool {
        false
    }

    fn caches_tools(&self) -> bool {
        true
    }
}

/// Links the run's `output/` into `into`, but its checkpoints and earlier
/// GGUF files.
fn stage_output(output: &Path, into: &Path) -> Result<(), TrainError> {
    fs::create_dir_all(into).map_err(train_io(into))?;
    for entry in fs::read_dir(output).map_err(train_io(output))? {
        let entry = entry.map_err(train_io(output))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if NOT_STAGED.iter().any(|skip| name.starts_with(skip)) {
            continue;
        }
        let kind = entry.file_type().map_err(train_io(&entry.path()))?;
        let target = into.join(&name);
        if kind.is_dir() {
            link_tree(&entry.path(), &target)?;
        } else if kind.is_file() {
            link_file(&entry.path(), &target)?;
        }
    }
    Ok(())
}

/// Copies the directory `from` into `to`, each file hard-linked when both are
/// on the same file system, copied otherwise.
fn link_tree(from: &Path, to: &Path) -> Result<(), TrainError> {
    fs::create_dir_all(to).map_err(train_io(to))?;
    for entry in fs::read_dir(from).map_err(train_io(from))? {
        let entry = entry.map_err(train_io(from))?;
        let kind = entry.file_type().map_err(train_io(&entry.path()))?;
        let target = to.join(entry.file_name());
        if kind.is_dir() {
            link_tree(&entry.path(), &target)?;
        } else if kind.is_file() {
            link_file(&entry.path(), &target)?;
        }
    }
    Ok(())
}

/// `from` hard-linked at `to`, or copied there; whatever was at `to` is
/// replaced.
pub(crate) fn link_file(from: &Path, to: &Path) -> Result<(), TrainError> {
    if let Some(dir) = to.parent() {
        fs::create_dir_all(dir).map_err(train_io(dir))?;
    }
    match fs::remove_file(to) {
        Ok(()) => {},
        Err(error) if error.kind() == io::ErrorKind::NotFound => {},
        Err(error) => return Err(train_io(to)(error)),
    }
    if fs::hard_link(from, to).is_ok() {
        return Ok(());
    }
    fs::copy(from, to)
        .map(drop)
        .map_err(|source| TrainError::Copy {
            from: from.to_path_buf(),
            to: to.to_path_buf(),
            source,
        })
}

/// What `export.json` records of an export.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct ExportRecord {
    /// The llama-quantize type.
    pub quantize: String,
    /// The llama.cpp release tag.
    pub llama_cpp: String,
    /// The GGUF file, relative to the run directory.
    pub file: String,
    /// SHA-256 of the GGUF file.
    pub sha256: String,
    /// Size of the GGUF file, in bytes.
    pub size: u64,
    /// When it was delivered, RFC 3339 UTC.
    pub created: String,
}

/// An export delivered into its run's `output/gguf/`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Delivered {
    /// The GGUF file.
    pub gguf: PathBuf,
    /// The Modelfile beside it.
    pub modelfile: PathBuf,
    /// What `export.json` records.
    pub record: ExportRecord,
}

/// Delivers the export whose job directory is `job_dir` (the run directory
/// for an export in a training job) into the run directory `run_dir`: its
/// GGUF file `file` goes to `runs/<id>/output/gguf/`, replacing one of the
/// same name, `export.json` is written in `job_dir`, and the Modelfile in
/// `output/gguf/`, with `num_ctx` from the run's own `axolotl.yaml`. What was
/// staged in `job_dir` for the job is removed.
///
/// # Errors
///
/// Returns [`ExportError::NoGguf`] when the job left no `file`, and
/// [`ExportError::Io`] when a file cannot be moved, read or written.
pub fn deliver(run_dir: &Path, job_dir: &Path, file: &str) -> Result<Delivered, ExportError> {
    let gguf_dir = run_dir.join(GGUF_DIR);
    let gguf = gguf_dir.join(file);
    if job_dir != run_dir {
        let made = job_dir.join(GGUF_DIR).join(file);
        if !made.is_file() {
            return Err(ExportError::NoGguf(job_dir.join(GGUF_DIR)));
        }
        fs::create_dir_all(&gguf_dir).map_err(io_error(&gguf_dir))?;
        fs::rename(&made, &gguf).map_err(io_error(&made))?;
        discard_staged(run_dir, job_dir)?;
    } else if !gguf.is_file() {
        return Err(ExportError::NoGguf(gguf_dir));
    }
    let size = fs::metadata(&gguf).map_err(io_error(&gguf))?.len();
    let sha256 = crate::exec::sha256_file(&gguf).map_err(io_error(&gguf))?;
    let quantize = quantize_of(file).unwrap_or_default();
    let record = ExportRecord {
        quantize,
        llama_cpp: LLAMA_CPP_TAG.to_string(),
        file: format!("{GGUF_DIR}/{file}"),
        sha256,
        size,
        created: crate::runs::rfc3339(SystemTime::now()),
    };
    let json = job_dir.join(EXPORT_FILE);
    let mut text = serde_json::to_string_pretty(&record).map_err(|error| ExportError::Io {
        path: json.clone(),
        source: io::Error::other(error),
    })?;
    text.push('\n');
    fs::write(&json, text).map_err(io_error(&json))?;
    let modelfile = gguf_dir.join(MODELFILE);
    let context = sequence_len(&run_dir.join(CONFIG_FILE));
    fs::write(&modelfile, modelfile_text(file, context)).map_err(io_error(&modelfile))?;
    Ok(Delivered {
        gguf,
        modelfile,
        record,
    })
}

/// The newest `<name>-<type>.gguf` in the run directory's `output/gguf/`,
/// by modification time: what an export in the training job wrote.
#[must_use]
pub fn latest_gguf(run_dir: &Path, name: &str) -> Option<String> {
    let entries = fs::read_dir(run_dir.join(GGUF_DIR)).ok()?;
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let file = entry.file_name().to_string_lossy().into_owned();
            let quantize = file
                .strip_prefix(&format!("{name}-"))
                .and_then(|rest| rest.strip_suffix(".gguf"))?;
            if !crate::config::QUANTIZE_TYPES.contains(&quantize) {
                return None;
            }
            let modified = entry.metadata().ok()?.modified().ok()?;
            Some((modified, file))
        })
        .max()
        .map(|(_, file)| file)
}

/// The quantize type of a `<name>-<type>.gguf` file name.
fn quantize_of(file: &str) -> Option<String> {
    let stem = file.strip_suffix(".gguf")?;
    crate::config::QUANTIZE_TYPES
        .iter()
        .find(|quantize| stem.ends_with(&format!("-{quantize}")))
        .map(|quantize| (*quantize).to_string())
}

/// Removes what an export job directory `job_dir` holds of the run in
/// `run_dir` for its job: the model and `axolotl.yaml` staged there, and the
/// GGUF the job wrote, once delivered or once the job failed. Nothing for an
/// export in a training job, whose job directory is the run's.
///
/// # Errors
///
/// Returns [`ExportError::Io`] when a file cannot be removed.
pub fn discard_staged(run_dir: &Path, job_dir: &Path) -> Result<(), ExportError> {
    if job_dir == run_dir {
        return Ok(());
    }
    for staged in [job_dir.join(OUTPUT_DIR), job_dir.join(CONFIG_FILE)] {
        remove_any(&staged)?;
    }
    Ok(())
}

fn remove_any(path: &Path) -> Result<(), ExportError> {
    let removed = if path.is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    };
    match removed {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(io_error(path)(error)),
    }
}

/// `sequence_len` of the Axolotl config at `path`, as `to_yaml` writes it.
pub(crate) fn sequence_len(path: &Path) -> Option<u32> {
    let text = fs::read_to_string(path).ok()?;
    top_level_scalar(&text, "sequence_len")?.parse().ok()
}

/// The Modelfile of the GGUF `file` beside it: Ollama takes the chat template
/// from the GGUF itself, and its context from `num_ctx`, the training
/// sequence length, when known.
#[must_use]
pub fn modelfile_text(file: &str, num_ctx: Option<u32>) -> String {
    let mut text = format!("FROM ./{file}\n");
    if let Some(num_ctx) = num_ctx {
        // Writing to a `String` cannot fail.
        let _ = writeln!(text, "PARAMETER num_ctx {num_ctx}");
    }
    text
}

/// What became of `ollama create`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ollama {
    /// The model was created.
    Created,
    /// `ollama` is not on `PATH`: the command to run by hand.
    NotFound(String),
    /// `ollama create` failed: its error output's last line.
    Failed(String),
}

/// The command creating the Ollama model `name` from the Modelfile in the
/// directory `gguf_dir`.
#[must_use]
pub fn ollama_command(gguf_dir: &Path, name: &str) -> String {
    format!(
        "cd {} && ollama create {name} -f {MODELFILE}",
        crate::exec::quote(&gguf_dir.to_string_lossy())
    )
}

/// Runs `ollama create name -f Modelfile` in `gguf_dir`, with `program` as
/// `ollama`. `name` must pass [`is_ollama_name`](crate::config::is_ollama_name).
#[must_use]
pub fn ollama_create(program: &str, gguf_dir: &Path, name: &str) -> Ollama {
    let output = std::process::Command::new(program)
        .args(["create", name, "-f", MODELFILE])
        .current_dir(gguf_dir)
        .stdin(std::process::Stdio::null())
        .output();
    match output {
        Ok(output) if output.status.success() => Ollama::Created,
        Ok(output) => {
            let error = String::from_utf8_lossy(&output.stderr);
            let last = error
                .lines()
                .rev()
                .find(|line| !line.trim().is_empty())
                .map_or_else(|| output.status.to_string(), str::to_string);
            Ollama::Failed(last)
        },
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            Ollama::NotFound(ollama_command(gguf_dir, name))
        },
        Err(error) => Ollama::Failed(error.to_string()),
    }
}

#[cfg(test)]
mod tests;
