//! `overbrainer compare`: the child against the parent on the eval set.
//!
//! A compare is a job of a run, like an export, kept in
//! `runs/<run-id>/compares/<compare-id>/`. On the run's target it serves the
//! run's GGUF with `llama-server` and asks it every question; back here a
//! judge model compares each child answer with the parent's, and the report
//! is written beside them.
//!
//! Files of a compare's directory:
//! - `setup.json`: what is compared, written before the job ([`CompareSetup`]);
//! - `questions.jsonl` and `model.gguf`: what the job gets (the GGUF is
//!   removed once the job ended);
//! - `child_answers.jsonl`, `server.log`, `hardware.json`: what the job gives back;
//! - `verdicts-<key>.jsonl`: the verdicts of one judge, appended as they come;
//! - `compare.json` and `compare.md`: the report.

mod cost;
mod error;
mod judge;
mod stats;
mod text;
mod types;

pub use cost::{Costs, Prices, costs, prices};
pub use error::CompareError;
pub use judge::{
    Pick, child_first, judge_key, parse_reply, render_judge, verdict_of, verdicts_file,
};
pub use stats::{Summary, count_f64, percentile, summarize};
pub use text::strip_reasoning;
pub use types::{
    ChatMessage, ChildAnswer, CompareSetup, EvalQuestion, Hardware, Verdict, VerdictLine,
    append_verdict, read_answers, read_eval, read_hardware, read_verdicts, write_child_questions,
};

/// Directory of a run holding its compare jobs, one directory each.
pub const COMPARES_DIR: &str = "compares";

/// The project part of a compare's ID: `compare_YYYYMMDD-HHMMSS`.
pub const COMPARE_PREFIX: &str = "compare";

/// What is compared, written before the job starts.
pub const SETUP_FILE: &str = "setup.json";

/// The questions the child is asked, one JSON object a line.
pub const QUESTIONS_FILE: &str = "questions.jsonl";

/// The run's GGUF in the job directory, while the job runs.
pub const MODEL_FILE: &str = "model.gguf";

/// The child's answers and timings, one JSON object a line.
pub const ANSWERS_FILE: &str = "child_answers.jsonl";

/// What `llama-server` printed.
pub const SERVER_LOG: &str = "server.log";

/// The machine that served the child.
pub const HARDWARE_FILE: &str = "hardware.json";

/// The report, as JSON.
pub const REPORT_JSON: &str = "compare.json";

/// The report, as Markdown.
pub const REPORT_MD: &str = "compare.md";
