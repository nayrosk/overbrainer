//! Dataset records, stable IDs and JSONL files.

mod id;
mod jsonl;
mod types;

pub use id::{Id, normalize};
pub use jsonl::{Appender, DataFiles, DatasetError, read, rewrite};
pub use types::{
    Example, Exclusion, FinishReason, Message, Meta, Question, ReasoningKind, Role, Subtopic,
};
