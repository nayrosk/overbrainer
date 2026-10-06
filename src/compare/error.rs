use std::io;
use std::path::PathBuf;

/// Errors of a compare: its files, its eval set, its judge.
#[derive(Debug, thiserror::Error)]
pub enum CompareError {
    /// A file or directory could not be read or written.
    #[error("cannot access {}", path.display())]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// A JSON file of the compare does not parse.
    #[error("{} is not valid", path.display())]
    Json {
        /// The file.
        path: PathBuf,
        /// Underlying parse error.
        #[source]
        source: serde_json::Error,
    },
    /// `data/eval.jsonl` cannot be read.
    #[error(transparent)]
    Dataset(#[from] crate::dataset::DatasetError),
    /// The judge prompt cannot be rendered.
    #[error(transparent)]
    Prompt(#[from] crate::prompts::PromptError),
    /// The judge failed on a question, retries spent.
    #[error("the judge failed on question {id}")]
    Judge {
        /// The question.
        id: String,
        /// Underlying error.
        #[source]
        source: crate::llm::LlmError,
    },
    /// The eval set has no question to ask.
    #[error("{} has no question with an answer: run `overbrainer split` first", .0.display())]
    NoQuestions(PathBuf),
    /// No compare has this ID.
    #[error("no compare {0} in runs/")]
    NotFound(String),
}

/// Maps an I/O error on `path` to [`CompareError::Io`].
pub(crate) fn io_error(path: &std::path::Path) -> impl FnOnce(io::Error) -> CompareError + '_ {
    move |source| CompareError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Maps a JSON error on `path` to [`CompareError::Json`].
pub(crate) fn json_error(
    path: &std::path::Path,
) -> impl FnOnce(serde_json::Error) -> CompareError + '_ {
    move |source| CompareError::Json {
        path: path.to_path_buf(),
        source,
    }
}
