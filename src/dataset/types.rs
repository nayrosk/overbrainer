use serde::{Deserialize, Serialize};

use super::Id;

/// One line of `data/subtopics.jsonl`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Subtopic {
    /// Stable ID, see [`Id::subtopic`].
    pub id: Id,
    /// Name of the topic this subtopic belongs to.
    pub topic: String,
    /// Subtopic name as generated.
    pub name: String,
}

/// One line of `data/questions.jsonl`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Question {
    /// Stable ID, see [`Id::question`].
    pub id: Id,
    /// Topic name.
    pub topic: String,
    /// ID of the subtopic.
    pub subtopic_id: Id,
    /// Subtopic name.
    pub subtopic: String,
    /// The question text.
    pub text: String,
}

/// Author of a chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System prompt.
    System,
    /// The question.
    User,
    /// The parent model's answer.
    Assistant,
}

/// A chat message in Axolotl `chat_template` format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    /// Author of the message.
    pub role: Role,
    /// Message text.
    pub content: String,
    /// The parent's reasoning, on assistant messages only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// How much of the model's reasoning a response exposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningKind {
    /// The full reasoning trace. The only kind used for training.
    Raw,
    /// A summary written by the provider.
    Summary,
    /// Reasoning happened but its content is hidden or encrypted.
    Redacted,
    /// No reasoning in the response.
    None,
}

/// Why a completion stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Natural end or stop sequence.
    Stop,
    /// The token limit was reached.
    Length,
    /// The provider filtered the content.
    ContentFilter,
    /// The model declined to answer.
    Refusal,
    /// Any other reason.
    Other,
}

/// Why an example is not used for training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Exclusion {
    /// The answer hit the token limit.
    Truncated,
    /// The answer is empty.
    Empty,
    /// The model refused or the provider filtered the answer.
    Refused,
    /// Reasoning was requested but the response has no raw reasoning.
    NoRawReasoning,
}

/// Generation metadata of an [`Example`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Meta {
    /// Parent model that produced the answer.
    pub model: String,
    /// Prompt tokens billed.
    pub input_tokens: u64,
    /// Completion tokens billed, reasoning included.
    pub output_tokens: u64,
    /// Why the completion stopped.
    pub finish_reason: FinishReason,
    /// What kind of reasoning the response exposed.
    pub reasoning_kind: ReasoningKind,
    /// Why the example is not used for training, if it is not.
    pub excluded: Option<Exclusion>,
}

/// One line of `data/answers.jsonl`, `data/train.jsonl` and `data/eval.jsonl`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Example {
    /// ID of the question this example answers.
    pub id: Id,
    /// Topic name.
    pub topic: String,
    /// Subtopic name.
    pub subtopic: String,
    /// Optional system message, the question, then the answer.
    pub messages: Vec<Message>,
    /// Generation metadata.
    pub meta: Meta,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_serializes_in_the_documented_shape() -> Result<(), serde_json::Error> {
        let example = Example {
            id: Id::of(&["x"]),
            topic: "ownership".into(),
            subtopic: "borrowing".into(),
            messages: vec![
                Message {
                    role: Role::User,
                    content: "Why?".into(),
                    reasoning_content: None,
                },
                Message {
                    role: Role::Assistant,
                    content: "Because.".into(),
                    reasoning_content: Some("Let me think.".into()),
                },
            ],
            meta: Meta {
                model: "m".into(),
                input_tokens: 3,
                output_tokens: 5,
                finish_reason: FinishReason::Stop,
                reasoning_kind: ReasoningKind::Raw,
                excluded: None,
            },
        };
        let value = serde_json::to_value(&example)?;
        assert_eq!(
            value["messages"][0],
            serde_json::json!({"role": "user", "content": "Why?"})
        );
        assert_eq!(
            value["messages"][1]["reasoning_content"],
            serde_json::json!("Let me think.")
        );
        assert_eq!(
            value["meta"],
            serde_json::json!({
                "model": "m", "input_tokens": 3, "output_tokens": 5,
                "finish_reason": "stop", "reasoning_kind": "raw", "excluded": null
            })
        );
        let back: Example = serde_json::from_value(value)?;
        assert_eq!(back, example);
        Ok(())
    }

    #[test]
    fn exclusions_use_snake_case() -> Result<(), serde_json::Error> {
        assert_eq!(
            serde_json::to_string(&Exclusion::NoRawReasoning)?,
            "\"no_raw_reasoning\""
        );
        Ok(())
    }
}
