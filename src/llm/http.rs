use std::time::Duration;

use reqwest::StatusCode;
use reqwest::header::{HeaderMap, HeaderValue, RETRY_AFTER};
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::LlmError;

/// Longest provider error message kept in an [`LlmError::Status`].
const MAX_MESSAGE_CHARS: usize = 300;

/// A base URL plus an HTTP client carrying the protocol's default headers.
#[derive(Debug, Clone)]
pub(crate) struct Endpoint {
    http: reqwest::Client,
    base_url: String,
}

impl Endpoint {
    /// Builds a client that sends `headers` on every request and gives up after `timeout`.
    pub(crate) fn new(
        base_url: &str,
        headers: HeaderMap,
        timeout: Duration,
    ) -> Result<Self, LlmError> {
        let http = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(timeout)
            .connect_timeout(Duration::from_secs(30))
            .build()
            .map_err(LlmError::Client)?;
        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{path}", self.base_url)
    }

    /// POSTs `body` as JSON to `{base_url}/{path}` and decodes the JSON answer.
    pub(crate) async fn post<B: Serialize + Sync, T: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, LlmError> {
        let response = self
            .http
            .post(self.url(path))
            .json(body)
            .send()
            .await
            .map_err(LlmError::Transport)?;
        decode(response).await
    }

    /// GETs `{base_url}/{path}` and decodes the JSON answer.
    pub(crate) async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T, LlmError> {
        let response = self
            .http
            .get(self.url(path))
            .send()
            .await
            .map_err(LlmError::Transport)?;
        decode(response).await
    }
}

/// Marks `value` sensitive so `Debug` output of headers and requests masks it.
pub(crate) fn sensitive(value: &str) -> Result<HeaderValue, LlmError> {
    let mut header = HeaderValue::from_str(value).map_err(|_| LlmError::InvalidApiKey)?;
    header.set_sensitive(true);
    Ok(header)
}

async fn decode<T: DeserializeOwned>(response: reqwest::Response) -> Result<T, LlmError> {
    let status = response.status();
    let retry_after = retry_after(response.headers());
    let body = response.text().await.map_err(LlmError::Transport)?;
    if !status.is_success() {
        return Err(LlmError::Status {
            status: status.as_u16(),
            message: error_message(status, &body),
            retry_after,
        });
    }
    serde_json::from_str(&body).map_err(|e| LlmError::InvalidResponse(e.to_string()))
}

/// `Retry-After` in seconds. The HTTP-date form is ignored; backoff applies instead.
fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let text = headers.get(RETRY_AFTER)?.to_str().ok()?;
    let seconds: f64 = text.trim().parse().ok()?;
    Duration::try_from_secs_f64(seconds).ok()
}

/// The provider's `error.message`, or the start of the body. Authentication failures
/// never carry the provider text: some providers quote part of the key in it.
fn error_message(status: StatusCode, body: &str) -> String {
    if status == StatusCode::UNAUTHORIZED {
        return "authentication failed, check the provider API key".to_string();
    }
    if status == StatusCode::FORBIDDEN {
        return "permission denied by the provider".to_string();
    }
    let parsed = serde_json::from_str::<serde_json::Value>(body).ok();
    let message = parsed
        .as_ref()
        .and_then(|value| value.pointer("/error/message"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or(body);
    message.chars().take(MAX_MESSAGE_CHARS).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_after_accepts_seconds_only() -> Result<(), reqwest::header::InvalidHeaderValue> {
        let mut headers = HeaderMap::new();
        assert_eq!(retry_after(&headers), None);
        headers.insert(RETRY_AFTER, HeaderValue::from_str("7")?);
        assert_eq!(retry_after(&headers), Some(Duration::from_secs(7)));
        headers.insert(RETRY_AFTER, HeaderValue::from_str("1.5")?);
        assert_eq!(retry_after(&headers), Some(Duration::from_millis(1500)));
        headers.insert(
            RETRY_AFTER,
            HeaderValue::from_str("Wed, 21 Oct 2026 07:28:00 GMT")?,
        );
        assert_eq!(retry_after(&headers), None);
        headers.insert(RETRY_AFTER, HeaderValue::from_str("-3")?);
        assert_eq!(retry_after(&headers), None);
        Ok(())
    }

    #[test]
    fn error_message_prefers_the_provider_message() {
        let body = r#"{"error":{"message":"Rate limit exceeded","type":"rate_limit_error"}}"#;
        assert_eq!(
            error_message(StatusCode::TOO_MANY_REQUESTS, body),
            "Rate limit exceeded"
        );
        assert_eq!(
            error_message(StatusCode::BAD_GATEWAY, "upstream down"),
            "upstream down"
        );
    }

    #[test]
    fn authentication_errors_drop_the_provider_text() {
        let body = r#"{"error":{"message":"Incorrect API key provided: sk-abc123"}}"#;
        let message = error_message(StatusCode::UNAUTHORIZED, body);
        assert!(!message.contains("sk-abc123"), "{message}");
    }

    #[test]
    fn sensitive_headers_are_masked_in_debug() -> Result<(), LlmError> {
        let header = sensitive("Bearer sk-secret-1")?;
        assert!(!format!("{header:?}").contains("sk-secret-1"));
        Ok(())
    }
}
