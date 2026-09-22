use std::time::Duration;

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
    /// Rate limits, server errors, timeouts and connection failures are worth retrying.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Status { status, .. } => *status == 408 || *status == 429 || *status >= 500,
            Self::Transport(_) => true,
            _ => false,
        }
    }

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

    /// The provider's `Retry-After`, when it sent one.
    #[must_use]
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::Status { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}
