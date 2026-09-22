//! Fine-tuning: the `Trainer` trait and its Axolotl implementation.

mod axolotl;
mod metrics;
mod yaml;

use std::path::{Path, PathBuf};

pub use axolotl::{Axolotl, CONFIG_FILE, METRICS_FILE, OUTPUT_DIR, reasoning_template_warning};
pub use metrics::{
    METRICS_ENV, METRICS_PLUGIN, MetricLine, PLUGIN_CLASS, PLUGIN_FILE, TrainMetric, parse_line,
};
pub use yaml::to_yaml;

/// Errors while preparing the files of a run.
#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    /// A file of the run could not be written or copied.
    #[error("cannot write {}", path.display())]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The training file is missing or empty.
    #[error("{} has no example: run `overbrainer split` first", path.display())]
    NoTrainingData {
        /// The training file.
        path: PathBuf,
    },
}

/// What to bring back from the target once the job has ended, relative to the run
/// directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Artifacts {
    /// Files and directories to copy. Missing ones are skipped.
    pub entries: Vec<String>,
    /// Name patterns left out, at any depth (`tar --exclude`).
    pub exclude: Vec<String>,
}

/// A fine-tuning framework: what a run needs on the target, and how to start it.
pub trait Trainer {
    /// Writes the job's files into the local run directory `run_dir`. `root` is the
    /// absolute path of the run directory as the job will see it, which the files
    /// may reference.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::NoTrainingData`] when there is nothing to train on, and
    /// [`TrainError::Io`] when a file cannot be written.
    fn prepare(&self, run_dir: &Path, root: &str) -> Result<(), TrainError>;

    /// Commands run one after the other in the run directory, each as a program and
    /// its arguments. The program is the framework's command line tool, which the
    /// runtime resolves (container `PATH` or virtual environment).
    fn commands(&self) -> Vec<Vec<String>>;

    /// Environment of the job, without secrets, for a run directory seen at `root`.
    fn env(&self, root: &str) -> Vec<(String, String)>;

    /// The file the job appends metric lines to, relative to the run directory.
    fn metrics_file(&self) -> &'static str;

    /// What to retrieve after the job.
    fn artifacts(&self) -> Artifacts;
}
