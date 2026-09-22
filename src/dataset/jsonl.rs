use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde::de::DeserializeOwned;

/// Paths of the pipeline's data files inside a project directory.
#[derive(Debug, Clone)]
pub struct DataFiles {
    /// `data/subtopics.jsonl`
    pub subtopics: PathBuf,
    /// `data/questions.jsonl`
    pub questions: PathBuf,
    /// `data/answers.jsonl`
    pub answers: PathBuf,
    /// `data/train.jsonl`
    pub train: PathBuf,
    /// `data/eval.jsonl`
    pub eval: PathBuf,
}

impl DataFiles {
    /// Paths under `<project_dir>/data/`.
    #[must_use]
    pub fn new(project_dir: &Path) -> Self {
        let data = project_dir.join("data");
        Self {
            subtopics: data.join("subtopics.jsonl"),
            questions: data.join("questions.jsonl"),
            answers: data.join("answers.jsonl"),
            train: data.join("train.jsonl"),
            eval: data.join("eval.jsonl"),
        }
    }
}

/// Errors from reading or writing JSONL files.
#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    /// A file could not be read, written or renamed.
    #[error("cannot access {}", path.display())]
    Io {
        /// The file.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// A line is not a valid record.
    #[error("{}:{line}: invalid record", path.display())]
    Parse {
        /// The file.
        path: PathBuf,
        /// 1-based line number.
        line: usize,
        /// Underlying JSON error.
        #[source]
        source: serde_json::Error,
    },
}

fn io_error(path: &Path) -> impl FnOnce(io::Error) -> DatasetError + '_ {
    move |source| DatasetError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Reads every record of `path`. A missing file reads as empty.
///
/// A last line without its trailing newline is read like any other line when it is
/// valid JSON: the crash that interrupted the write only missed the newline, and
/// [`Appender::open`] adds it back. When it is not valid JSON it is the incomplete
/// start of a record, left by a crash during a write: it is skipped with a warning,
/// and [`Appender::open`] removes it before appending.
///
/// # Errors
///
/// Returns [`DatasetError::Io`] if the file cannot be read and [`DatasetError::Parse`]
/// if a line is valid JSON but not a valid record, or if a complete line is not valid
/// JSON.
pub fn read<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, DatasetError> {
    let content = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(io_error(path)(e)),
    };
    let complete = content.is_empty() || content.ends_with('\n');
    let lines: Vec<&str> = content.lines().collect();
    let mut items = Vec::with_capacity(lines.len());
    for (index, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str(line) {
            Ok(item) => items.push(item),
            Err(_) if !complete && index + 1 == lines.len() && !is_json(line.as_bytes()) => {
                tracing::warn!("{}: ignoring an incomplete last line", path.display());
            },
            Err(source) => {
                return Err(DatasetError::Parse {
                    path: path.to_path_buf(),
                    line: index + 1,
                    source,
                });
            },
        }
    }
    Ok(items)
}

/// Appends records to a JSONL file, one line at a time, flushed after each line so a
/// crash loses at most the line being written.
#[derive(Debug)]
pub struct Appender {
    path: PathBuf,
    file: File,
}

impl Appender {
    /// Opens `path` for appending, creating it and its directory if needed. A last
    /// line without its trailing newline gets it back when it is valid JSON, and is
    /// removed when it is not (an incomplete record left by a crash).
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError::Io`] if the directory or file cannot be created, read
    /// or truncated.
    pub fn open(path: &Path) -> Result<Self, DatasetError> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).map_err(io_error(dir))?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(path)
            .map_err(io_error(path))?;
        end_last_line(&mut file).map_err(io_error(path))?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
        })
    }

    /// Writes `item` as one line and flushes it.
    ///
    /// # Errors
    ///
    /// Returns [`DatasetError::Io`] if the line cannot be written.
    pub fn append<T: Serialize>(&mut self, item: &T) -> Result<(), DatasetError> {
        let mut line = serde_json::to_vec(item).map_err(|e| io_error(&self.path)(e.into()))?;
        line.push(b'\n');
        self.file
            .write_all(&line)
            .and_then(|()| self.file.flush())
            .map_err(io_error(&self.path))
    }
}

/// Makes `file` end with a newline: an unterminated last line that is valid JSON is
/// kept and terminated, anything else after the last newline is truncated.
fn end_last_line(file: &mut File) -> io::Result<()> {
    let len = file.metadata()?.len();
    if len == 0 {
        return Ok(());
    }
    let mut last = [0_u8; 1];
    file.seek(SeekFrom::End(-1))?;
    file.read_exact(&mut last)?;
    if last[0] == b'\n' {
        return Ok(());
    }
    let mut content = Vec::new();
    file.seek(SeekFrom::Start(0))?;
    file.read_to_end(&mut content)?;
    let keep = content
        .iter()
        .rposition(|byte| *byte == b'\n')
        .map_or(0, |index| index + 1);
    if is_json(&content[keep..]) {
        return file.write_all(b"\n");
    }
    file.set_len(u64::try_from(keep).map_err(io::Error::other)?)
}

/// Whether `line` is one complete JSON value. A record cut short by a crash never is,
/// since every record is a JSON object.
fn is_json(line: &[u8]) -> bool {
    serde_json::from_slice::<serde::de::IgnoredAny>(line).is_ok()
}

/// Replaces `path` with `items`: writes a temp file in the same directory, syncs it,
/// then renames it over `path`, so readers see either the old or the new content.
///
/// # Errors
///
/// Returns [`DatasetError::Io`] if the temp file cannot be written or renamed.
pub fn rewrite<T: Serialize>(path: &Path, items: &[T]) -> Result<(), DatasetError> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(dir).map_err(io_error(dir))?;
    let name = path
        .file_name()
        .map_or_else(|| "data".into(), |name| name.to_string_lossy());
    let tmp = dir.join(format!(".{name}.tmp"));
    let mut buffer = Vec::new();
    for item in items {
        serde_json::to_writer(&mut buffer, item).map_err(|e| io_error(&tmp)(e.into()))?;
        buffer.push(b'\n');
    }
    let mut file = File::create(&tmp).map_err(io_error(&tmp))?;
    file.write_all(&buffer)
        .and_then(|()| file.sync_all())
        .map_err(io_error(&tmp))?;
    fs::rename(&tmp, path).map_err(io_error(path))
}
