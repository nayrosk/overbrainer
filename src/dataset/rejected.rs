//! Records of `data/rejected.jsonl`: subtopics and questions deleted from the
//! dataset, which the stages must not generate again.

use serde::{Deserialize, Serialize};

use super::Id;

/// One line of `data/rejected.jsonl`, tagged by `kind`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum Rejected {
    /// A deleted subtopic: the subtopics stage drops a generated name with its ID.
    Subtopic {
        /// Its ID, [`Id::subtopic`] of its topic and name.
        id: Id,
        /// Its topic.
        topic: String,
        /// Its name.
        name: String,
    },
    /// A deleted question: the questions stage treats its text like a question
    /// already accepted in its topic.
    Question {
        /// Its ID, [`Id::question`] of its subtopic and text.
        id: Id,
        /// Its topic.
        topic: String,
        /// The ID of its subtopic.
        subtopic_id: Id,
        /// Its text.
        text: String,
    },
}

impl Rejected {
    /// The topic of the rejected item.
    #[must_use]
    pub fn topic(&self) -> &str {
        match self {
            Self::Subtopic { topic, .. } | Self::Question { topic, .. } => topic,
        }
    }
}
