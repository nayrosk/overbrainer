use std::ops::AddAssign;

use crate::config::{Effort, RoleModel};
use crate::dataset::{FinishReason, ReasoningKind};

/// A single-turn completion request. The model is bound to the client.
#[derive(Debug, Clone, PartialEq)]
pub struct CompletionRequest {
    /// Optional system prompt.
    pub system: Option<String>,
    /// The user message.
    pub prompt: String,
    /// Upper bound on generated tokens, reasoning included.
    pub max_tokens: u32,
    /// Sampling temperature; the provider default applies when `None`.
    pub temperature: Option<f64>,
    /// Whether to ask the model for reasoning.
    pub reasoning: bool,
    /// Reasoning effort, used only when `reasoning` is true.
    pub effort: Option<Effort>,
}

impl CompletionRequest {
    /// Builds a request with the parameters configured for `role`.
    #[must_use]
    pub fn for_role(role: &RoleModel, system: Option<String>, prompt: String) -> Self {
        Self {
            system,
            prompt,
            max_tokens: role.max_tokens,
            temperature: role.temperature,
            reasoning: role.reasoning,
            effort: role.reasoning_effort,
        }
    }
}

/// Token counts reported by the provider.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    /// Prompt tokens.
    pub input_tokens: u64,
    /// Completion tokens, reasoning included.
    pub output_tokens: u64,
}

impl AddAssign for Usage {
    fn add_assign(&mut self, other: Self) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
    }
}

/// Reasoning extracted from a response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reasoning {
    /// Reasoning text, when the provider exposed any.
    pub text: Option<String>,
    /// What the text is.
    pub kind: ReasoningKind,
}

impl Reasoning {
    /// No reasoning in the response.
    #[must_use]
    pub fn none() -> Self {
        Self {
            text: None,
            kind: ReasoningKind::None,
        }
    }
}

/// A parsed completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion {
    /// The answer text, reasoning removed.
    pub content: String,
    /// The reasoning, if any.
    pub reasoning: Reasoning,
    /// Token counts.
    pub usage: Usage,
    /// Why the completion stopped.
    pub finish: FinishReason,
}
