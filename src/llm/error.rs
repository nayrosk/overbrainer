use std::time::Duration;

use crate::retry::Retryable;

/// Errors from an LLM provider.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// The provider answered with a non-success status.
    #[error("HTTP {status}: {message}")]
    Status {
        /// HTTP status code.
        status: u16,
        /// Provider error message. Dropped for 401 and 403, which can quote the key.
        message: String,
        /// Parsed `Retry-After` header, when present.
        retry_after: Option<Duration>,
    },
    /// The request could not be sent or the response could not be read.
    #[error("request failed")]
    Transport(#[source] reqwest::Error),
    /// The response body does not have the expected shape.
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    /// The protocol does not offer this operation.
    #[error("{0} are not supported by this protocol")]
    Unsupported(&'static str),
    /// The API key cannot be sent as an HTTP header value.
    #[error("the API key is not a valid HTTP header value")]
    InvalidApiKey,
    /// The HTTP client could not be built.
    #[error("cannot build the HTTP client")]
    Client(#[source] reqwest::Error),
}

impl LlmError {
    /// Errors that will fail every request of a stage: bad credentials, missing model,
    /// unsupported operation. A stage stops at the first one instead of failing each item.
    #[must_use]
    pub fn is_fatal_for_stage(&self) -> bool {
        match self {
            Self::Status { status, .. } => (401..=404).contains(status),
            Self::Unsupported(_) | Self::InvalidApiKey | Self::Client(_) => true,
            Self::Transport(_) | Self::InvalidResponse(_) => false,
        }
    }
}

impl Retryable for LlmError {
    /// Rate limits, server errors, timeouts and connection failures are worth retrying.
    fn is_retryable(&self) -> bool {
        match self {
            Self::Status { status, .. } => *status == 408 || *status == 429 || *status >= 500,
            Self::Transport(_) => true,
            _ => false,
        }
    }

    /// The provider's `Retry-After`, when it sent one.
    fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::Status { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status(status: u16) -> LlmError {
        LlmError::Status {
            status,
            message: String::new(),
            retry_after: None,
        }
    }

    #[test]
    fn classification_of_errors() {
        for code in [408, 429, 500, 503, 529] {
            assert!(status(code).is_retryable(), "{code}");
        }
        for code in [400, 401, 403, 404] {
            assert!(!status(code).is_retryable(), "{code}");
        }
        assert!(status(401).is_fatal_for_stage());
        assert!(!status(400).is_fatal_for_stage());
        assert!(!status(429).is_fatal_for_stage());
    }
}
