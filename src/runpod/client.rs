//! `RunpodClient`: the REST v2 calls overbrainer makes, with authentication,
//! retries and error messages that never carry a secret.

use std::time::Duration;

use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue, RETRY_AFTER};
use reqwest::{Method, StatusCode};
use secrecy::{ExposeSecret, SecretString};
use serde::de::DeserializeOwned;

use super::types::{CreatePod, Pod, PodId, PodPage};
use crate::retry::{RetryPolicy, Retryable, with_retry};

/// `User-Agent` of every request. Runpod sits behind Cloudflare, which answers a
/// generic client's user agent with a 403 ("error code: 1010").
pub const USER_AGENT: &str = concat!("overbrainer/", env!("CARGO_PKG_VERSION"));

/// Longest error text kept from a Runpod answer.
const MAX_MESSAGE_CHARS: usize = 300;
/// Connection timeout of every request.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Timeout of a request, answer included.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Timeout of `POST /pods`, which answers once the pod is placed.
const CREATE_TIMEOUT: Duration = Duration::from_secs(60);
/// Pods asked for per page of `GET /pods`.
const PAGE_SIZE: &str = "1000";

/// Errors of the Runpod API. No variant ever holds the API key or a pod's host
/// key: messages built from an answer are cleaned of both.
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    /// The HTTP client could not be built.
    #[error("cannot build the HTTP client")]
    Client(#[source] reqwest::Error),
    /// The API key cannot be sent as an HTTP header value.
    #[error("the Runpod API key is not a valid HTTP header value")]
    InvalidApiKey,
    /// The request could not be sent or its answer could not be read: connection,
    /// TLS, timeout. For a `POST` the pod may or may not exist.
    #[error("cannot reach the Runpod API")]
    Transport(#[source] reqwest::Error),
    /// Runpod answered with a non-success status.
    #[error("Runpod answered {status}: {message}")]
    Status {
        /// HTTP status code.
        status: u16,
        /// What Runpod said, cleaned of secrets and capped.
        message: String,
        /// Parsed `Retry-After`, when present.
        retry_after: Option<Duration>,
    },
    /// A success answer whose body is not what the API documents.
    #[error("invalid answer from Runpod: {0}")]
    InvalidResponse(String),
}

impl ApiError {
    /// The HTTP status, for a [`ApiError::Status`].
    #[must_use]
    pub fn status(&self) -> Option<u16> {
        match self {
            Self::Status { status, .. } => Some(*status),
            _ => None,
        }
    }

    /// Whether a failed `POST` may still have created its pod: the request may
    /// have been processed although no usable answer came back.
    #[must_use]
    pub fn is_ambiguous(&self) -> bool {
        match self {
            Self::Transport(_) | Self::InvalidResponse(_) => true,
            Self::Status { status, .. } => *status == 408 || *status >= 500,
            Self::Client(_) | Self::InvalidApiKey => false,
        }
    }
}

impl Retryable for ApiError {
    fn is_retryable(&self) -> bool {
        match self {
            Self::Transport(_) => true,
            Self::Status { status, .. } => *status == 408 || *status == 429 || *status >= 500,
            Self::Client(_) | Self::InvalidApiKey | Self::InvalidResponse(_) => false,
        }
    }

    fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::Status { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}

/// A client of the Runpod REST API (v2) with the account API key.
#[derive(Debug, Clone)]
pub struct RunpodClient {
    http: reqwest::Client,
    base_url: String,
    api_key: SecretString,
    policy: RetryPolicy,
}

impl RunpodClient {
    /// A client of the API at `base_url` (for example `https://api.runpod.io/v2`)
    /// authenticated with `api_key`. `GET` and `DELETE` are retried 5 times.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::InvalidApiKey`] when the key cannot be a header value,
    /// and [`ApiError::Client`] when the HTTP client cannot be built.
    pub fn new(base_url: &str, api_key: &SecretString) -> Result<Self, ApiError> {
        let mut bearer = HeaderValue::from_str(&format!("Bearer {}", api_key.expose_secret()))
            .map_err(|_| ApiError::InvalidApiKey)?;
        bearer.set_sensitive(true);
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, bearer);
        let http = reqwest::Client::builder()
            .default_headers(headers)
            .user_agent(USER_AGENT)
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(ApiError::Client)?;
        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.clone(),
            policy: RetryPolicy::new(5),
        })
    }

    /// This client with another retry policy (tests use millisecond delays).
    #[must_use]
    pub fn with_policy(mut self, policy: RetryPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// The base URL, without a trailing `/`.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Creates a pod. Only a 429 is retried, after its `Retry-After`: Runpod turns
    /// a request away with it before processing it, so nothing was created. Any
    /// other failure is returned at once; [`ApiError::is_ambiguous`] tells whether
    /// the pod may exist anyway.
    ///
    /// # Errors
    ///
    /// Returns an [`ApiError`] when the pod was not created or the answer cannot be
    /// read. Its message never holds the request's host key nor the API key.
    pub async fn create_pod(&self, request: &CreatePod) -> Result<Pod, ApiError> {
        let secrets = [request.env.host_key.expose_secret()];
        let mut attempt = 0;
        loop {
            let builder = self
                .http
                .post(self.url("pods"))
                .json(request)
                .timeout(CREATE_TIMEOUT);
            match self.send(builder, &secrets).await {
                Err(ApiError::Status {
                    status: 429,
                    retry_after,
                    ..
                }) if attempt < self.policy.max_retries => {
                    let wait = self.policy.delay(attempt, retry_after, fastrand::f64());
                    tracing::warn!(
                        "Runpod is rate limiting pod creation; retrying in {}s",
                        wait.as_secs()
                    );
                    tokio::time::sleep(wait).await;
                    attempt += 1;
                },
                other => return decode(&other?),
            }
        }
    }

    /// The pod `id`, or `None` when Runpod does not know it (any more).
    ///
    /// # Errors
    ///
    /// Returns an [`ApiError`] once retries are exhausted or on a fatal answer.
    pub async fn get_pod(&self, id: &PodId) -> Result<Option<Pod>, ApiError> {
        let url = self.url(&format!("pods/{id}"));
        with_retry(
            &self.policy,
            || async {
                match self.send(self.http.request(Method::GET, &url), &[]).await {
                    Ok(body) => decode(&body).map(Some),
                    Err(error) if error.status() == Some(404) => Ok(None),
                    Err(error) => Err(error),
                }
            },
            log_retry,
        )
        .await
    }

    /// Every pod of the account, following the pagination.
    ///
    /// # Errors
    ///
    /// Returns an [`ApiError`] once retries are exhausted or on a fatal answer.
    pub async fn list_pods(&self) -> Result<Vec<Pod>, ApiError> {
        let mut pods = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let mut url = reqwest::Url::parse(&self.url("pods"))
                .map_err(|error| ApiError::InvalidResponse(error.to_string()))?;
            url.query_pairs_mut().append_pair("limit", PAGE_SIZE);
            if let Some(cursor) = &cursor {
                url.query_pairs_mut().append_pair("cursor", cursor);
            }
            let page: PodPage = with_retry(
                &self.policy,
                || async {
                    let body = self
                        .send(self.http.request(Method::GET, url.clone()), &[])
                        .await?;
                    decode(&body)
                },
                log_retry,
            )
            .await?;
            pods.extend(page.pods);
            match page.pagination {
                Some(next) if next.has_next_page && next.next_cursor.is_some() => {
                    cursor = next.next_cursor;
                },
                _ => return Ok(pods),
            }
        }
    }

    /// Deletes the pod `id`. A pod Runpod no longer knows counts as deleted.
    ///
    /// # Errors
    ///
    /// Returns an [`ApiError`] once retries are exhausted or on a fatal answer.
    pub async fn delete_pod(&self, id: &PodId) -> Result<(), ApiError> {
        let url = self.url(&format!("pods/{id}"));
        with_retry(
            &self.policy,
            || async {
                match self
                    .send(self.http.request(Method::DELETE, &url), &[])
                    .await
                {
                    Ok(_) => Ok(()),
                    Err(error) if error.status() == Some(404) => Ok(()),
                    Err(error) => Err(error),
                }
            },
            log_retry,
        )
        .await
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{path}", self.base_url)
    }

    /// Sends `builder` and returns the body of a success answer. A failure's
    /// message is cleaned of the API key and of every one of `secrets`.
    async fn send(
        &self,
        builder: reqwest::RequestBuilder,
        secrets: &[&str],
    ) -> Result<String, ApiError> {
        let response = builder.send().await.map_err(ApiError::Transport)?;
        let status = response.status();
        let retry_after = retry_after(response.headers());
        let body = response.text().await.map_err(ApiError::Transport)?;
        if status.is_success() {
            return Ok(body);
        }
        let mut hidden = vec![self.api_key.expose_secret()];
        hidden.extend_from_slice(secrets);
        Err(ApiError::Status {
            status: status.as_u16(),
            message: redact(&error_message(status, &body), &hidden),
            retry_after,
        })
    }
}

fn log_retry(error: &ApiError, wait: Duration) {
    tracing::warn!("{error}; retrying in {}s", wait.as_secs());
}

/// Parses a success body. The serde error names what was wrong, never the body.
fn decode<T: DeserializeOwned>(body: &str) -> Result<T, ApiError> {
    serde_json::from_str(body).map_err(|error| ApiError::InvalidResponse(error.to_string()))
}

/// `Retry-After` in seconds. The HTTP-date form is ignored; backoff applies instead.
fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let text = headers.get(RETRY_AFTER)?.to_str().ok()?;
    let seconds: f64 = text.trim().parse().ok()?;
    Duration::try_from_secs_f64(seconds).ok()
}

/// What to say about a failed answer. A 401 never quotes Runpod; a 403 keeps a
/// non-JSON body (Cloudflare's `error code: 1010` says what is wrong); anything
/// else keeps the RFC 9457 `title`, `detail` and `errors[]`, or the body start.
fn error_message(status: StatusCode, body: &str) -> String {
    if status == StatusCode::UNAUTHORIZED {
        return "Runpod rejected the API key, check OVERBRAINER_RUNPOD__API_KEY".to_string();
    }
    let parsed = serde_json::from_str::<serde_json::Value>(body).ok();
    if status == StatusCode::FORBIDDEN {
        return match parsed {
            Some(_) => "permission denied by Runpod".to_string(),
            None => cap(&format!("permission denied by Runpod: {}", body.trim())),
        };
    }
    let Some(value) = parsed else {
        return cap(body.trim());
    };
    let mut parts: Vec<String> = ["title", "detail"]
        .into_iter()
        .filter_map(|key| value.get(key)?.as_str().map(str::to_string))
        .collect();
    if let Some(errors) = value.get("errors").and_then(serde_json::Value::as_array) {
        parts.extend(errors.iter().map(error_item));
    }
    if parts.is_empty() {
        cap(body.trim())
    } else {
        cap(&parts.join("; "))
    }
}

/// One entry of an RFC 9457 `errors` array: a string, or an object's `message`
/// or `detail`, or the object itself.
fn error_item(item: &serde_json::Value) -> String {
    if let Some(text) = item.as_str() {
        return text.to_string();
    }
    ["message", "detail"]
        .into_iter()
        .find_map(|key| item.get(key)?.as_str().map(str::to_string))
        .unwrap_or_else(|| item.to_string())
}

fn cap(text: &str) -> String {
    text.chars().take(MAX_MESSAGE_CHARS).collect()
}

/// `text` with every non-empty one of `secrets` replaced by `***`.
fn redact(text: &str, secrets: &[&str]) -> String {
    secrets
        .iter()
        .filter(|secret| !secret.is_empty())
        .fold(text.to_string(), |text, secret| text.replace(secret, "***"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_follow_the_status() {
        assert_eq!(
            error_message(
                StatusCode::UNAUTHORIZED,
                r#"{"detail": "key rp_abc invalid"}"#
            ),
            "Runpod rejected the API key, check OVERBRAINER_RUNPOD__API_KEY"
        );
        assert_eq!(
            error_message(StatusCode::FORBIDDEN, "error code: 1010"),
            "permission denied by Runpod: error code: 1010"
        );
        assert_eq!(
            error_message(StatusCode::FORBIDDEN, r#"{"detail": "no"}"#),
            "permission denied by Runpod"
        );
        assert_eq!(
            error_message(
                StatusCode::UNPROCESSABLE_ENTITY,
                r#"{"title": "Invalid request", "detail": "bad body", "errors": ["env too big", {"message": "gpu.id unknown"}, {"path": "/x"}]}"#
            ),
            r#"Invalid request; bad body; env too big; gpu.id unknown; {"path":"/x"}"#
        );
        assert_eq!(
            error_message(StatusCode::BAD_GATEWAY, "upstream down"),
            "upstream down"
        );
        assert_eq!(
            error_message(StatusCode::BAD_REQUEST, &"x".repeat(1000)).len(),
            MAX_MESSAGE_CHARS
        );
    }

    #[test]
    fn secrets_are_redacted() {
        assert_eq!(
            redact("key k1 and host h1, k1 again", &["k1", "", "h1"]),
            "key *** and host ***, *** again"
        );
    }

    #[test]
    fn classification_of_errors() {
        let status = |status| ApiError::Status {
            status,
            message: String::new(),
            retry_after: None,
        };
        for code in [408, 429, 500, 503] {
            assert!(status(code).is_retryable(), "{code}");
        }
        for code in [400, 401, 402, 403, 404, 422] {
            assert!(!status(code).is_retryable(), "{code}");
        }
        for code in [408, 500, 502] {
            assert!(status(code).is_ambiguous(), "{code}");
        }
        for code in [400, 402, 403, 422, 429] {
            assert!(!status(code).is_ambiguous(), "{code}");
        }
        assert!(ApiError::InvalidResponse(String::new()).is_ambiguous());
    }
}
