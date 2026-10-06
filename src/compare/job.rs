//! The compare job: the run's GGUF served by `llama-server` on the target,
//! every question asked, the answers and timings brought back.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::error::io_error;
use super::{
    ANSWERS_FILE, CompareError, EvalQuestion, HARDWARE_FILE, MODEL_FILE, SERVER_LOG,
    write_child_questions,
};
use crate::export::{PYTHON, TRAMPOLINE, link_file, llama_cpp_env};
use crate::train::{Artifacts, JobStage, METRICS_ENV, METRICS_FILE, TrainError, Trainer};

/// The job script, written into the job directory.
pub const SCRIPT: &str = include_str!("compare.sh");

/// The script's file name, in the job directory.
pub const SCRIPT_FILE: &str = "compare.sh";

/// The client asking the server, written into the job directory.
pub const CLIENT: &str = include_str!("compare_client.py");

/// The client's file name, in the job directory.
pub const CLIENT_FILE: &str = "compare_client.py";

/// How the child generates, and how long its server may take to start.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChildSettings {
    /// Limit of each answer.
    pub max_tokens: u32,
    /// Sampling temperature.
    pub temperature: f64,
    /// Seconds the server may take to be ready.
    pub server_start_secs: u64,
    /// Context size of the server, 0 for the model's own.
    pub context: u32,
}

impl ChildSettings {
    /// The settings of `[compare]`, with the run's context size `context`.
    #[must_use]
    pub fn from_config(compare: &crate::config::Compare, context: u32) -> Self {
        Self {
            max_tokens: compare.max_tokens,
            temperature: compare.temperature,
            server_start_secs: compare.server_start_secs,
            context,
        }
    }
}

/// Where the job finds the GGUF it serves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// The local GGUF at this path, hard-linked into the job directory as
    /// `model.gguf` and uploaded with it; the script removes it on the
    /// target once it ends, and [`discard_model`] the local link.
    Upload(PathBuf),
    /// The absolute path of a GGUF already on the target: nothing is
    /// uploaded, and the script leaves it there.
    OnTarget(String),
}

/// A compare job: everything it needs is put in its own directory, which is
/// uploaded to the target.
#[derive(Debug, Clone, PartialEq)]
pub struct CompareJob {
    /// Where the GGUF comes from.
    model: ModelSource,
    /// The questions, written as `questions.jsonl`.
    questions: Vec<EvalQuestion>,
    /// How the child generates.
    settings: ChildSettings,
}

impl CompareJob {
    /// The compare of the GGUF `model` on `questions`.
    #[must_use]
    pub fn new(model: ModelSource, questions: Vec<EvalQuestion>, settings: ChildSettings) -> Self {
        Self {
            model,
            questions,
            settings,
        }
    }

    /// The GGUF as the script gets it: `model.gguf` in the job directory, or
    /// its path on the target.
    fn model_path(&self) -> &str {
        match &self.model {
            ModelSource::Upload(_) => MODEL_FILE,
            ModelSource::OnTarget(path) => path,
        }
    }
}

/// Maps an I/O error on `path` to [`TrainError::Io`].
fn train_io(path: &Path) -> impl FnOnce(io::Error) -> TrainError + '_ {
    move |source| TrainError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// A [`CompareError`] writing `questions.jsonl` at `path`, as a
/// [`TrainError::Io`] keeping its source.
fn questions_error(path: &Path, error: CompareError) -> TrainError {
    match error {
        CompareError::Io { path, source } => TrainError::Io { path, source },
        other => TrainError::Io {
            path: path.to_path_buf(),
            source: io::Error::other(other),
        },
    }
}

impl Trainer for CompareJob {
    fn prepare(&self, run_dir: &Path, _root: &str) -> Result<(), TrainError> {
        fs::create_dir_all(run_dir).map_err(train_io(run_dir))?;
        for (name, content) in [(SCRIPT_FILE, SCRIPT), (CLIENT_FILE, CLIENT)] {
            let path = run_dir.join(name);
            fs::write(&path, content).map_err(train_io(&path))?;
        }
        write_child_questions(run_dir, &self.questions)
            .map_err(|error| questions_error(&run_dir.join(super::QUESTIONS_FILE), error))?;
        match &self.model {
            ModelSource::Upload(gguf) => link_file(gguf, &run_dir.join(MODEL_FILE)),
            ModelSource::OnTarget(_) => Ok(()),
        }
    }

    fn commands(&self) -> Vec<Vec<String>> {
        vec![vec![
            PYTHON.to_string(),
            "-c".to_string(),
            TRAMPOLINE.to_string(),
            SCRIPT_FILE.to_string(),
        ]]
    }

    fn env(&self, root: &str) -> Vec<(String, String)> {
        let mut env = vec![
            (METRICS_ENV.to_string(), format!("{root}/{METRICS_FILE}")),
            (
                "OVERBRAINER_COMPARE_MODEL".to_string(),
                self.model_path().to_string(),
            ),
            (
                "OVERBRAINER_COMPARE_MAX_TOKENS".to_string(),
                self.settings.max_tokens.to_string(),
            ),
            (
                "OVERBRAINER_COMPARE_TEMPERATURE".to_string(),
                self.settings.temperature.to_string(),
            ),
            (
                "OVERBRAINER_COMPARE_START_SECS".to_string(),
                self.settings.server_start_secs.to_string(),
            ),
            (
                "OVERBRAINER_COMPARE_CTX".to_string(),
                self.settings.context.to_string(),
            ),
        ];
        env.extend(llama_cpp_env());
        env
    }

    fn metrics_file(&self) -> &'static str {
        METRICS_FILE
    }

    fn artifacts(&self) -> Artifacts {
        Artifacts {
            entries: vec![
                ANSWERS_FILE.to_string(),
                SERVER_LOG.to_string(),
                HARDWARE_FILE.to_string(),
                METRICS_FILE.to_string(),
            ],
            exclude: Vec::new(),
            required: Some(ANSWERS_FILE.to_string()),
        }
    }

    fn stages(&self) -> Vec<JobStage> {
        vec![JobStage::Compare]
    }

    fn metrics_required(&self) -> bool {
        false
    }

    fn caches_tools(&self) -> bool {
        true
    }
}

/// Removes `model.gguf` from the local job directory `job_dir`: the link to
/// the run's GGUF made for the upload. Nothing when it is gone already.
///
/// # Errors
///
/// Returns [`CompareError::Io`] when it cannot be removed.
pub fn discard_model(job_dir: &Path) -> Result<(), CompareError> {
    let path = job_dir.join(MODEL_FILE);
    match fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(io_error(&path)(error)),
    }
}
