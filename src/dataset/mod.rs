//! Dataset records, stable IDs and JSONL files.

mod edit;
mod id;
mod jsonl;
mod rejected;
mod types;

pub use edit::{AnswerText, Change, Dataset, EditError, Touched};
pub use id::{Id, normalize};
pub use jsonl::{Appender, DataFiles, DatasetError, Rewrite, read, rewrite};
pub use rejected::Rejected;
pub use types::{
    Example, Exclusion, FinishReason, Message, Meta, Question, ReasoningKind, Role, Subtopic,
};
