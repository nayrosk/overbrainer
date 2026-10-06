//! The files of a compare: its setup, the child's answers, the machine that
//! served it and the judge's verdicts.

use std::fs;
use std::io::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::error::{io_error, json_error};
use super::{ANSWERS_FILE, CompareError, HARDWARE_FILE, QUESTIONS_FILE, SETUP_FILE};
use crate::dataset::{Example, Role};

/// A chat message sent to the child.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// `system` or `user`.
    pub role: String,
    /// The text.
    pub content: String,
}

/// One question of the eval set as the child gets it, with the parent's answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalQuestion {
    /// ID of the question.
    pub id: String,
    /// Topic name.
    pub topic: String,
    /// What the child is sent: the system message, if any, then the question.
    pub messages: Vec<ChatMessage>,
    /// The parent's final answer, reasoning removed.
    pub parent: String,
    /// Prompt tokens the parent was billed for.
    pub parent_input_tokens: u64,
    /// Completion tokens the parent was billed for, reasoning included.
    pub parent_output_tokens: u64,
}

impl EvalQuestion {
    /// The question of an eval example: its messages before the parent's
    /// answer. `None` when the example has no user message or no answer.
    #[must_use]
    pub fn from_example(example: &Example) -> Option<Self> {
        let answer = example
            .messages
            .iter()
            .rfind(|message| message.role == Role::Assistant)?;
        let messages: Vec<ChatMessage> = example
            .messages
            .iter()
            .take_while(|message| message.role != Role::Assistant)
            .map(|message| ChatMessage {
                role: match message.role {
                    Role::System => "system",
                    Role::User | Role::Assistant => "user",
                }
                .to_string(),
                content: message.content.clone(),
            })
            .collect();
        if !messages.iter().any(|message| message.role == "user") {
            return None;
        }
        Some(Self {
            id: example.id.to_string(),
            topic: example.topic.clone(),
            messages,
            parent: super::strip_reasoning(&answer.content),
            parent_input_tokens: example.meta.input_tokens,
            parent_output_tokens: example.meta.output_tokens,
        })
    }

    /// The question: its last user message.
    #[must_use]
    pub fn text(&self) -> &str {
        self.messages
            .iter()
            .rfind(|message| message.role == "user")
            .map_or("", |message| message.content.as_str())
    }
}

/// Reads the questions of the eval set at `path`, in file order, the first
/// `limit` of them when given.
///
/// # Errors
///
/// Returns [`CompareError::Dataset`] when the file cannot be read, and
/// [`CompareError::NoQuestions`] when it has no question with an answer.
pub fn read_eval(path: &Path, limit: Option<usize>) -> Result<Vec<EvalQuestion>, CompareError> {
    let examples: Vec<Example> = crate::dataset::read(path)?;
    let questions: Vec<EvalQuestion> = examples
        .iter()
        .filter_map(EvalQuestion::from_example)
        .take(limit.unwrap_or(usize::MAX))
        .collect();
    if questions.is_empty() {
        return Err(CompareError::NoQuestions(path.to_path_buf()));
    }
    Ok(questions)
}

/// What a compare compares, written to `setup.json` before its job starts: a
/// rejudge reads it, never `data/eval.jsonl` again.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompareSetup {
    /// The run compared.
    pub run: String,
    /// The compare's ID.
    pub compare: String,
    /// The GGUF file, relative to the run directory.
    pub gguf: String,
    /// SHA-256 of the GGUF file.
    pub gguf_sha256: String,
    /// Its llama-quantize type.
    pub quantize: String,
    /// The llama.cpp release serving it.
    pub llama_cpp: String,
    /// The run's base model, from its `axolotl.yaml`, when known.
    pub base_model: Option<String>,
    /// Seed of the order the judge sees each pair in.
    pub seed: u64,
    /// When the compare was created, RFC 3339 UTC.
    pub created: String,
    /// The questions, with the parent's answers.
    pub questions: Vec<EvalQuestion>,
}

impl CompareSetup {
    /// Writes `setup.json` into the compare directory `dir`.
    ///
    /// # Errors
    ///
    /// Returns [`CompareError::Io`] when it cannot be written.
    pub fn save(&self, dir: &Path) -> Result<(), CompareError> {
        write_json(&dir.join(SETUP_FILE), self)
    }

    /// Reads `setup.json` of the compare directory `dir`.
    ///
    /// # Errors
    ///
    /// Returns [`CompareError::Io`] when it cannot be read, and
    /// [`CompareError::Json`] when it does not parse.
    pub fn load(dir: &Path) -> Result<Self, CompareError> {
        read_json(&dir.join(SETUP_FILE))
    }
}

/// A line of `questions.jsonl`: what the client sends for one question.
#[derive(Debug, Serialize)]
struct ChildQuestion<'a> {
    /// ID of the question.
    id: &'a str,
    /// The messages.
    messages: &'a [ChatMessage],
}

/// Writes `questions.jsonl` into the job directory `dir`.
///
/// # Errors
///
/// Returns [`CompareError::Io`] when it cannot be written.
pub fn write_child_questions(dir: &Path, questions: &[EvalQuestion]) -> Result<(), CompareError> {
    let path = dir.join(QUESTIONS_FILE);
    let mut text = String::new();
    for question in questions {
        let line = ChildQuestion {
            id: &question.id,
            messages: &question.messages,
        };
        text.push_str(&serde_json::to_string(&line).map_err(json_error(&path))?);
        text.push('\n');
    }
    fs::create_dir_all(dir).map_err(io_error(dir))?;
    fs::write(&path, text).map_err(io_error(&path))
}

/// A line of `child_answers.jsonl`, as `compare_client.py` writes it.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ChildAnswer {
    /// ID of the question.
    pub id: String,
    /// The answer, as streamed.
    #[serde(default)]
    pub answer: Option<String>,
    /// Why the answer stopped, as the server said it.
    #[serde(default)]
    pub finish: Option<String>,
    /// Prompt tokens.
    #[serde(default)]
    pub input_tokens: Option<u64>,
    /// Completion tokens.
    #[serde(default)]
    pub output_tokens: Option<u64>,
    /// Seconds from the request to the end of the answer.
    #[serde(default)]
    pub seconds: Option<f64>,
    /// Seconds from the request to the first token.
    #[serde(default)]
    pub first_token_seconds: Option<f64>,
    /// Completion tokens per second after the first token.
    #[serde(default)]
    pub tokens_per_second: Option<f64>,
    /// Why the request failed, when it did.
    #[serde(default)]
    pub error: Option<String>,
}

impl ChildAnswer {
    /// Whether the request failed or gave no answer.
    #[must_use]
    pub fn is_error(&self) -> bool {
        self.error.is_some() || self.answer.is_none()
    }
}

/// Reads `child_answers.jsonl` of the compare directory `dir`.
///
/// # Errors
///
/// Returns [`CompareError::Dataset`] when it cannot be read or a line does not parse.
pub fn read_answers(dir: &Path) -> Result<Vec<ChildAnswer>, CompareError> {
    Ok(crate::dataset::read(&dir.join(ANSWERS_FILE))?)
}

/// The machine that served the child, from `hardware.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hardware {
    /// The llama.cpp build: `ubuntu-cuda-13.4-x64`, `ubuntu-x64`, `macos-arm64`...
    pub build: String,
    /// The GPUs the build served on: the NVIDIA ones for a CUDA build, the
    /// AMD ones for a `ROCm` build (an integrated "AMD Radeon Graphics" left
    /// out beside a discrete GPU), the Apple chip for the Metal build; empty
    /// for a CPU build.
    #[serde(default)]
    pub gpus: Vec<String>,
    /// The CPU model, when known.
    #[serde(default)]
    pub cpu: Option<String>,
}

impl Hardware {
    /// One line: `2 x NVIDIA A40 (ubuntu-cuda-13.4-x64)`, each different GPU
    /// counted apart (`1 x A + 1 x B (...)`), or `AMD EPYC 7413, CPU (ubuntu-x64)`.
    #[must_use]
    pub fn describe(&self) -> String {
        if self.gpus.is_empty() {
            return format!(
                "{}, CPU ({})",
                self.cpu.as_deref().unwrap_or("unknown CPU"),
                self.build
            );
        }
        let mut counted: Vec<(&str, usize)> = Vec::new();
        for gpu in &self.gpus {
            match counted.iter_mut().find(|(name, _)| name == gpu) {
                Some((_, count)) => *count += 1,
                None => counted.push((gpu, 1)),
            }
        }
        let gpus: Vec<String> = counted
            .into_iter()
            .map(|(name, count)| format!("{count} x {name}"))
            .collect();
        format!("{} ({})", gpus.join(" + "), self.build)
    }

    /// Whether the child ran on a GPU: always for the CUDA, `ROCm` and macOS
    /// (Metal) builds, never for the Linux CPU builds; for another build,
    /// whether a GPU is listed.
    #[must_use]
    pub fn has_gpu(&self) -> bool {
        let build = self.build.as_str();
        if ["ubuntu-cuda-", "ubuntu-rocm-", "macos-"]
            .iter()
            .any(|prefix| build.starts_with(prefix))
        {
            return true;
        }
        if matches!(build, "ubuntu-x64" | "ubuntu-arm64") {
            return false;
        }
        !self.gpus.is_empty()
    }
}

/// `hardware.json` of the compare directory `dir`, when it is there and parses.
#[must_use]
pub fn read_hardware(dir: &Path) -> Option<Hardware> {
    read_json(&dir.join(HARDWARE_FILE)).ok()
}

/// How the child's answer fared against the parent's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// The judge preferred the child's answer.
    Win,
    /// The judge found them as good.
    Tie,
    /// The judge preferred the parent's answer.
    Loss,
    /// The judge's reply did not parse, even asked twice.
    Unparsed,
    /// The child gave no answer: counted as a loss.
    Error,
}

impl Verdict {
    /// One word: `win`, `tie`, `loss`, `unparsed`, `error`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Win => "win",
            Self::Tie => "tie",
            Self::Loss => "loss",
            Self::Unparsed => "unparsed",
            Self::Error => "error",
        }
    }
}

/// A line of a verdicts file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerdictLine {
    /// ID of the question.
    pub id: String,
    /// The verdict, for the child.
    pub verdict: Verdict,
    /// The judge's reason, when it gave one.
    #[serde(default)]
    pub reason: Option<String>,
    /// Whether the child's answer was shown first, as answer A.
    pub child_first: bool,
}

/// The verdicts of the file at `path`; none when it does not exist.
///
/// # Errors
///
/// Returns [`CompareError::Dataset`] when it cannot be read or a line does not parse.
pub fn read_verdicts(path: &Path) -> Result<Vec<VerdictLine>, CompareError> {
    Ok(crate::dataset::read(path)?)
}

/// Appends `line` to the verdicts file at `path`, creating it.
///
/// # Errors
///
/// Returns [`CompareError::Io`] when it cannot be written.
pub fn append_verdict(path: &Path, line: &VerdictLine) -> Result<(), CompareError> {
    let mut text = serde_json::to_string(line).map_err(json_error(path))?;
    text.push('\n');
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(io_error(path))?;
    file.write_all(text.as_bytes()).map_err(io_error(path))
}

/// Writes `value` as pretty JSON to `path`, with a final line break.
pub(crate) fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), CompareError> {
    let mut text = serde_json::to_string_pretty(value).map_err(json_error(path))?;
    text.push('\n');
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(io_error(dir))?;
    }
    fs::write(path, text).map_err(io_error(path))
}

/// Reads the JSON file at `path`.
pub(crate) fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, CompareError> {
    let text = fs::read_to_string(path).map_err(io_error(path))?;
    serde_json::from_str(&text).map_err(json_error(path))
}

#[cfg(test)]
/// Tests of the compare files.
mod tests {
    use super::*;
    use crate::dataset::{
        Example, FinishReason, Id, Message, Meta, ReasoningKind, Role as DataRole,
    };

    /// Result type of the tests.
    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// An eval example with a system message, its question and the parent's answer.
    fn example(id: &str, answer: &str) -> Example {
        let message = |role, content: &str| Message {
            role,
            content: content.to_string(),
            reasoning_content: None,
        };
        Example {
            id: Id::of(&[id]),
            topic: "ownership".into(),
            subtopic: "borrowing".into(),
            messages: vec![
                message(DataRole::System, "You are an expert."),
                message(DataRole::User, &format!("Question {id}?")),
                Message {
                    reasoning_content: Some("hidden".into()),
                    ..message(DataRole::Assistant, answer)
                },
            ],
            meta: Meta {
                model: "parent".into(),
                input_tokens: 40,
                output_tokens: 200,
                finish_reason: FinishReason::Stop,
                reasoning_kind: ReasoningKind::Raw,
                excluded: None,
            },
        }
    }

    /// The child gets the messages before the answer; the parent's answer is
    /// kept without its reasoning, with its billed tokens.
    #[test]
    fn an_eval_example_becomes_a_question() -> TestResult {
        let question = EvalQuestion::from_example(&example("q1", "<think>x</think>Borrow it."))
            .ok_or("no question")?;
        assert_eq!(question.messages.len(), 2);
        assert_eq!(question.messages[0].role, "system");
        assert_eq!(question.text(), "Question q1?");
        assert_eq!(question.parent, "Borrow it.");
        assert_eq!(
            (question.parent_input_tokens, question.parent_output_tokens),
            (40, 200)
        );
        let mut no_answer = example("q2", "x");
        no_answer.messages.pop();
        assert!(EvalQuestion::from_example(&no_answer).is_none());
        Ok(())
    }

    /// `--limit` keeps the first questions in file order; an empty eval set is an error.
    #[test]
    fn the_eval_set_is_read_in_order_up_to_the_limit() -> TestResult {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("eval.jsonl");
        let mut text = String::new();
        for id in ["q1", "q2", "q3"] {
            text.push_str(&serde_json::to_string(&example(id, "A."))?);
            text.push('\n');
        }
        std::fs::write(&path, text)?;
        let two = read_eval(&path, Some(2))?;
        assert_eq!(
            two.iter().map(EvalQuestion::text).collect::<Vec<_>>(),
            ["Question q1?", "Question q2?"]
        );
        assert_eq!(read_eval(&path, None)?.len(), 3);
        std::fs::write(&path, "")?;
        assert!(matches!(
            read_eval(&path, None),
            Err(CompareError::NoQuestions(_))
        ));
        Ok(())
    }

    /// Verdicts are appended one line each and read back in order.
    #[test]
    fn verdicts_append_and_read_back() -> TestResult {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("verdicts-0.jsonl");
        assert!(
            read_verdicts(&path)?.is_empty(),
            "a missing file is no verdict"
        );
        for (id, verdict) in [("q1", Verdict::Win), ("q2", Verdict::Unparsed)] {
            append_verdict(
                &path,
                &VerdictLine {
                    id: id.into(),
                    verdict,
                    reason: None,
                    child_first: true,
                },
            )?;
        }
        let read = read_verdicts(&path)?;
        assert_eq!(read.len(), 2);
        assert_eq!(read[1].verdict, Verdict::Unparsed);
        assert!(std::fs::read_to_string(&path)?.contains("\"verdict\":\"unparsed\""));
        Ok(())
    }

    /// A child answer line from the client parses with any field missing.
    #[test]
    fn child_answers_parse_with_missing_fields() -> TestResult {
        let ok: ChildAnswer = serde_json::from_str(
            r#"{"id":"q1","answer":"A.","finish":"stop","input_tokens":7,"output_tokens":3,
               "seconds":0.5,"first_token_seconds":0.1,"tokens_per_second":7.5}"#,
        )?;
        assert!(!ok.is_error());
        let failed: ChildAnswer = serde_json::from_str(r#"{"id":"q2","error":"timed out"}"#)?;
        assert!(failed.is_error());
        Ok(())
    }
    /// Hardware served by `build` with `gpus`, on an EPYC CPU.
    fn hardware(build: &str, gpus: &[&str]) -> Hardware {
        Hardware {
            build: build.into(),
            gpus: gpus.iter().map(|gpu| (*gpu).to_string()).collect(),
            cpu: Some("AMD EPYC 7413".into()),
        }
    }

    /// Identical GPUs are counted, different ones each named: a discrete
    /// Radeon beside an integrated one is never "2 x" the Radeon.
    #[test]
    fn the_hardware_line_groups_identical_gpus() {
        let rocm = "ubuntu-rocm-7.2-x64";
        assert_eq!(
            hardware(rocm, &["AMD Radeon RX 7800 XT", "AMD Radeon Graphics"]).describe(),
            "1 x AMD Radeon RX 7800 XT + 1 x AMD Radeon Graphics (ubuntu-rocm-7.2-x64)"
        );
        assert_eq!(
            hardware("ubuntu-cuda-13.4-x64", &["NVIDIA A40", "NVIDIA A40"]).describe(),
            "2 x NVIDIA A40 (ubuntu-cuda-13.4-x64)"
        );
        assert_eq!(
            hardware(
                "ubuntu-cuda-13.4-x64",
                &["NVIDIA A40", "NVIDIA T4", "NVIDIA A40"]
            )
            .describe(),
            "2 x NVIDIA A40 + 1 x NVIDIA T4 (ubuntu-cuda-13.4-x64)"
        );
        assert_eq!(
            hardware("ubuntu-x64", &[]).describe(),
            "AMD EPYC 7413, CPU (ubuntu-x64)"
        );
    }

    /// Whether the child ran on a GPU follows the build: the CPU builds
    /// never did, the CUDA, `ROCm` and Metal builds always did.
    #[test]
    fn the_build_says_whether_a_gpu_served() {
        assert!(hardware("ubuntu-cuda-13.4-x64", &["NVIDIA A40"]).has_gpu());
        assert!(hardware("ubuntu-cuda-13.4-x64", &[]).has_gpu());
        assert!(hardware("ubuntu-rocm-7.2-x64", &["AMD GPU"]).has_gpu());
        assert!(hardware("macos-arm64", &["Apple M2 Pro (Metal)"]).has_gpu());
        assert!(hardware("macos-arm64", &[]).has_gpu());
        assert!(!hardware("ubuntu-x64", &[]).has_gpu());
        assert!(!hardware("ubuntu-x64", &["NVIDIA RTX 3090"]).has_gpu());
        assert!(!hardware("ubuntu-arm64", &[]).has_gpu());
    }
}
