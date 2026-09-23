//! `RunpodClient`: the REST v2 calls overbrainer makes, with authentication,
//! retries and error messages that never carry a secret.

use std::collections::HashSet;
use std::time::Duration;

use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue, RETRY_AFTER};
use reqwest::{Method, StatusCode};
use secrecy::{ExposeSecret, SecretString};
use serde::de::DeserializeOwned;

use super::types::{CreatePod, Pagination, Pod, PodId, PodPage};
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
/// Length of the substring windows checked against each secret, so a value
/// Runpod echoes only in part still disappears.
const REDACT_WINDOW: usize = 16;
/// Shortest run of base64-alphabet characters treated as a secret in a create
/// call's error body, regardless of whether it matches a known secret exactly.
const MIN_BASE64_RUN: usize = 40;
/// Pages of `GET /pods` followed before giving up: protects against a server
/// whose pagination never reports `hasNextPage: false`.
const MAX_PAGES: usize = 100;

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

/// What to do after fetching one page of `GET /pods`.
#[derive(Debug)]
enum NextPage {
    /// No more pages follow.
    Done,
    /// The cursor of the next page, not seen before.
    Cursor(String),
}

/// Decides what follows `pagination`, guarding against a cursor Runpod already
/// sent: a page whose `nextCursor` repeats one already used would loop forever.
fn next_page(
    pagination: Option<Pagination>,
    seen: &mut HashSet<String>,
) -> Result<NextPage, ApiError> {
    let Some(next) = pagination else {
        return Ok(NextPage::Done);
    };
    if !next.has_next_page {
        return Ok(NextPage::Done);
    }
    let Some(cursor) = next.next_cursor else {
        return Ok(NextPage::Done);
    };
    if seen.insert(cursor.clone()) {
        Ok(NextPage::Cursor(cursor))
    } else {
        Err(ApiError::InvalidResponse(format!(
            "Runpod repeated the pagination cursor `{cursor}`"
        )))
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
    /// Redirects are never followed: a redirected error surfaces as-is instead of
    /// silently sending the API key to wherever Runpod points.
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
            .redirect(reqwest::redirect::Policy::none())
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
    /// other failure, including a transport error or a timeout, is returned at
    /// once; [`ApiError::is_ambiguous`] tells whether the pod may exist anyway.
    ///
    /// # Errors
    ///
    /// Returns an [`ApiError`] when the pod was not created or the answer cannot be
    /// read. Its message never holds the request's host key nor the API key, even
    /// truncated.
    pub async fn create_pod(&self, request: &CreatePod) -> Result<Pod, ApiError> {
        let secrets = [request.env.host_key.expose_secret()];
        let mut attempt = 0;
        loop {
            let builder = self
                .http
                .post(self.url("pods"))
                .json(request)
                .timeout(CREATE_TIMEOUT);
            let (status, body, retry_after) = self.fetch(builder).await?;
            if status.is_success() {
                return decode(&body, &self.hidden(&secrets), true);
            }
            if status == StatusCode::TOO_MANY_REQUESTS && attempt < self.policy.max_retries {
                let wait = self.policy.delay(attempt, retry_after, fastrand::f64());
                tracing::warn!(
                    "Runpod is rate limiting pod creation; retrying in {}s",
                    wait.as_secs()
                );
                tokio::time::sleep(wait).await;
                attempt += 1;
                continue;
            }
            return Err(self.status_error(status, &body, retry_after, &secrets));
        }
    }

    /// The pod `id`, or `None` when Runpod says so in its own error shape (a
    /// `title`, `detail` or matching `status` field): a 404 with an unrelated
    /// body, for example from a misconfigured base URL, is a real error instead.
    ///
    /// # Errors
    ///
    /// Returns an [`ApiError`] once retries are exhausted or on a fatal answer.
    pub async fn get_pod(&self, id: &PodId) -> Result<Option<Pod>, ApiError> {
        let url = self.url(&format!("pods/{id}"));
        with_retry(
            &self.policy,
            || async {
                let (status, body, retry_after) =
                    self.fetch(self.http.request(Method::GET, &url)).await?;
                if status.is_success() {
                    return decode(&body, &self.hidden(&[]), false).map(Some);
                }
                if status == StatusCode::NOT_FOUND && is_runpod_error_shape(status, &body) {
                    return Ok(None);
                }
                Err(self.status_error(status, &body, retry_after, &[]))
            },
            log_retry,
        )
        .await
    }

    /// Every pod of the account, following the pagination up to [`MAX_PAGES`].
    ///
    /// # Errors
    ///
    /// Returns an [`ApiError`] once retries are exhausted, on a fatal answer, when
    /// a pagination cursor repeats, or when more than [`MAX_PAGES`] are followed.
    pub async fn list_pods(&self) -> Result<Vec<Pod>, ApiError> {
        let mut pods = Vec::new();
        let mut cursor: Option<String> = None;
        let mut seen = HashSet::new();
        for _ in 0..MAX_PAGES {
            let page = self.list_page(cursor.as_deref()).await?;
            pods.extend(page.pods);
            match next_page(page.pagination, &mut seen)? {
                NextPage::Done => return Ok(pods),
                NextPage::Cursor(next) => cursor = Some(next),
            }
        }
        Err(ApiError::InvalidResponse(format!(
            "Runpod's pod list did not end after {MAX_PAGES} pages"
        )))
    }

    /// One page of `GET /pods`, at `cursor` when given.
    async fn list_page(&self, cursor: Option<&str>) -> Result<PodPage, ApiError> {
        let mut url = reqwest::Url::parse(&self.url("pods"))
            .map_err(|error| ApiError::InvalidResponse(error.to_string()))?;
        url.query_pairs_mut().append_pair("limit", PAGE_SIZE);
        if let Some(cursor) = cursor {
            url.query_pairs_mut().append_pair("cursor", cursor);
        }
        with_retry(
            &self.policy,
            || async {
                let body = self
                    .send(self.http.request(Method::GET, url.clone()), &[])
                    .await?;
                decode(&body, &self.hidden(&[]), false)
            },
            log_retry,
        )
        .await
    }

    /// Deletes the pod `id`. A pod Runpod no longer knows, in its own error
    /// shape, counts as deleted; a 404 with an unrelated body is a real error.
    ///
    /// # Errors
    ///
    /// Returns an [`ApiError`] once retries are exhausted or on a fatal answer.
    pub async fn delete_pod(&self, id: &PodId) -> Result<(), ApiError> {
        let url = self.url(&format!("pods/{id}"));
        with_retry(
            &self.policy,
            || async {
                let (status, body, retry_after) =
                    self.fetch(self.http.request(Method::DELETE, &url)).await?;
                if status.is_success() {
                    return Ok(());
                }
                if status == StatusCode::NOT_FOUND && is_runpod_error_shape(status, &body) {
                    return Ok(());
                }
                Err(self.status_error(status, &body, retry_after, &[]))
            },
            log_retry,
        )
        .await
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{path}", self.base_url)
    }

    /// `secrets` (the request's own, when it has any) together with the API key,
    /// all of which must never appear in a message this client raises.
    fn hidden<'a>(&'a self, secrets: &[&'a str]) -> Vec<&'a str> {
        let mut hidden = vec![self.api_key.expose_secret()];
        hidden.extend_from_slice(secrets);
        hidden
    }

    /// Sends `builder` and returns its raw status, body and `Retry-After`. Only a
    /// transport failure (connection, TLS, timeout) becomes an [`ApiError`] here:
    /// a non-success HTTP status is returned as data, unredacted, so `get_pod` and
    /// `delete_pod` can recognize Runpod's own "not found" shape before anyone
    /// builds the client-facing, redacted error from it.
    async fn fetch(
        &self,
        builder: reqwest::RequestBuilder,
    ) -> Result<(StatusCode, String, Option<Duration>), ApiError> {
        let response = builder.send().await.map_err(ApiError::Transport)?;
        let status = response.status();
        let retry_after = retry_after(response.headers());
        let body = response.text().await.map_err(ApiError::Transport)?;
        Ok((status, body, retry_after))
    }

    /// Sends `builder` and returns the body of a success answer, or the client's
    /// redacted [`ApiError::Status`] for a failure.
    async fn send(
        &self,
        builder: reqwest::RequestBuilder,
        secrets: &[&str],
    ) -> Result<String, ApiError> {
        let (status, body, retry_after) = self.fetch(builder).await?;
        if status.is_success() {
            return Ok(body);
        }
        Err(self.status_error(status, &body, retry_after, secrets))
    }

    /// Builds the [`ApiError::Status`] a caller sees from a failed answer: `body`
    /// is cleaned of every secret in `secrets` and the API key before any text is
    /// extracted from it or truncated, so no truncation can ever leave a partial
    /// secret in the message.
    fn status_error(
        &self,
        status: StatusCode,
        body: &str,
        retry_after: Option<Duration>,
        secrets: &[&str],
    ) -> ApiError {
        let hidden = self.hidden(secrets);
        let cleaned = sanitize(body, &hidden, !secrets.is_empty());
        ApiError::Status {
            status: status.as_u16(),
            message: error_message(status, &cleaned),
            retry_after,
        }
    }
}

fn log_retry(error: &ApiError, wait: Duration) {
    tracing::warn!("{error}; retrying in {}s", wait.as_secs());
}

/// Parses a success body. A parse failure's message can quote the offending
/// value (a custom `Deserialize` does, for [`PodId`]), which may be a secret this
/// call's body echoed back, so it is cleaned exactly like a failed answer's.
fn decode<T: DeserializeOwned>(
    body: &str,
    hidden: &[&str],
    strip_base64: bool,
) -> Result<T, ApiError> {
    serde_json::from_str(body).map_err(|error| {
        ApiError::InvalidResponse(sanitize(&error.to_string(), hidden, strip_base64))
    })
}

/// `Retry-After` in seconds. The HTTP-date form is ignored; backoff applies instead.
fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let text = headers.get(RETRY_AFTER)?.to_str().ok()?;
    let seconds: f64 = text.trim().parse().ok()?;
    Duration::try_from_secs_f64(seconds).ok()
}

/// Whether `body` looks like Runpod's own JSON error shape for `status` (a
/// numeric `status` field matching the HTTP status, or a `title` or `detail`
/// field), as opposed to an HTML page or an empty body a wrong base URL or an
/// intermediary would produce for the same HTTP status.
fn is_runpod_error_shape(status: StatusCode, body: &str) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
        return false;
    };
    let Some(object) = value.as_object() else {
        return false;
    };
    let status_matches = object.get("status").and_then(serde_json::Value::as_u64)
        == Some(u64::from(status.as_u16()));
    status_matches || object.contains_key("title") || object.contains_key("detail")
}

/// What to say about a failed answer. A 401 never quotes Runpod; a 403 keeps a
/// non-JSON body (Cloudflare's `error code: 1010` says what is wrong); anything
/// else keeps the RFC 9457 `title`, `detail` and `errors[]`, or the body start.
/// `body` must already be cleaned of secrets: this only extracts and truncates.
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

/// Cleans `text` of every secret in `hidden`; when `strip_base64` is set (the
/// create call, whose failing body may echo the pod's host key, about 550 base64
/// characters, in a form the exact and windowed passes of [`redact`] miss), also
/// of any run of [`MIN_BASE64_RUN`] or more base64 characters. Order matters:
/// this must run on the raw body before any text is extracted from it or capped,
/// so a truncation can never leave a partial secret behind.
fn sanitize(text: &str, hidden: &[&str], strip_base64: bool) -> String {
    let cleaned = redact(text, hidden);
    if strip_base64 {
        redact_base64_runs(&cleaned)
    } else {
        cleaned
    }
}

/// `text` with every non-empty one of `secrets` replaced by `***`: first every
/// exact occurrence, then every [`REDACT_WINDOW`]-character piece of the secret
/// at any offset, so a value Runpod echoes only in part still disappears.
fn redact(text: &str, secrets: &[&str]) -> String {
    secrets
        .iter()
        .filter(|secret| !secret.is_empty())
        .fold(text.to_string(), |text, secret| redact_one(&text, secret))
}

/// `redact`'s two passes for one `secret`.
fn redact_one(text: &str, secret: &str) -> String {
    let text = text.replace(secret, "***");
    let chars: Vec<char> = secret.chars().collect();
    if chars.len() < REDACT_WINDOW {
        return text;
    }
    (0..=chars.len() - REDACT_WINDOW).fold(text, |text, start| {
        let window: String = chars[start..start + REDACT_WINDOW].iter().collect();
        text.replace(&window, "***")
    })
}

/// Replaces every run of [`MIN_BASE64_RUN`] or more base64-alphabet characters in
/// `text` with `<redacted>`, regardless of whether it matches a known secret.
fn redact_base64_runs(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut run = String::new();
    for c in text.chars() {
        if is_base64_char(c) {
            run.push(c);
        } else {
            flush_base64_run(&mut result, &mut run);
            result.push(c);
        }
    }
    flush_base64_run(&mut result, &mut run);
    result
}

/// Appends `run` to `result`, replaced by `<redacted>` when it reached
/// [`MIN_BASE64_RUN`], then clears it.
fn flush_base64_run(result: &mut String, run: &mut String) {
    if run.chars().count() >= MIN_BASE64_RUN {
        result.push_str("<redacted>");
    } else {
        result.push_str(run);
    }
    run.clear();
}

fn is_base64_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '='
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
    fn windows_catch_a_partially_echoed_secret() {
        let secret = "0123456789abcdefghijXYZ";
        let piece = &secret[3..19];
        assert_eq!(piece.len(), REDACT_WINDOW);
        let text = format!("prefix {piece} suffix");
        let cleaned = redact(&text, &[secret]);
        assert!(!cleaned.contains(piece), "{cleaned}");
        assert_eq!(cleaned, "prefix *** suffix");
    }

    #[test]
    fn secrets_shorter_than_a_window_only_match_exactly() {
        assert_eq!(redact("short s1 stays", &["s1"]), "short *** stays");
    }

    #[test]
    fn long_base64_runs_are_redacted_regardless_of_a_known_secret() {
        let run = "A".repeat(MIN_BASE64_RUN + 5);
        let text = format!("prefix {run} suffix");
        assert_eq!(redact_base64_runs(&text), "prefix <redacted> suffix");
        let short = "B".repeat(MIN_BASE64_RUN - 1);
        let text = format!("prefix {short} suffix");
        assert_eq!(redact_base64_runs(&text), text);
    }

    #[test]
    fn sanitize_orders_redaction_before_any_truncation_would_happen() {
        let secret = "s".repeat(500);
        let body = format!(r#"{{"detail": "bad key {secret}"}}"#);
        let cleaned = sanitize(&body, &[&secret], true);
        assert!(!cleaned.contains(&secret));
        let message = error_message(StatusCode::BAD_REQUEST, &cleaned);
        assert!(!message.contains(&secret[..REDACT_WINDOW]), "{message}");
    }

    #[test]
    fn decode_errors_are_redacted() {
        let secret = "topsecrethostkeyvalue1234567890";
        let body = format!(r#"{{"id": "{secret}"}}"#);
        let result: Result<PodId, ApiError> = decode(&body, &[secret], false);
        let message = match result {
            Err(ApiError::InvalidResponse(message)) => message,
            Err(other) => other.to_string(),
            Ok(_) => String::new(),
        };
        assert!(!message.is_empty(), "expected a decode error");
        assert!(!message.contains(secret), "{message}");
    }

    #[test]
    fn runpod_error_shape_recognition() {
        assert!(is_runpod_error_shape(
            StatusCode::NOT_FOUND,
            r#"{"status": 404}"#
        ));
        assert!(is_runpod_error_shape(
            StatusCode::NOT_FOUND,
            r#"{"title": "Not Found"}"#
        ));
        assert!(is_runpod_error_shape(
            StatusCode::NOT_FOUND,
            r#"{"detail": "gone"}"#
        ));
        assert!(!is_runpod_error_shape(
            StatusCode::NOT_FOUND,
            "<html></html>"
        ));
        assert!(!is_runpod_error_shape(StatusCode::NOT_FOUND, ""));
        assert!(!is_runpod_error_shape(
            StatusCode::NOT_FOUND,
            r#"{"status": 500}"#
        ));
        assert!(!is_runpod_error_shape(StatusCode::NOT_FOUND, "[1,2,3]"));
    }

    #[test]
    fn next_page_detects_a_repeated_cursor() {
        let mut seen = HashSet::new();
        let page = Pagination {
            has_next_page: true,
            next_cursor: Some("c2".to_string()),
        };
        let first = next_page(Some(page.clone()), &mut seen);
        assert!(matches!(first, Ok(NextPage::Cursor(ref c)) if c == "c2"));
        let second = next_page(Some(page), &mut seen);
        assert!(matches!(second, Err(ApiError::InvalidResponse(_))));
    }

    #[test]
    fn next_page_stops_without_a_cursor_or_another_page() {
        let mut seen = HashSet::new();
        assert!(matches!(next_page(None, &mut seen), Ok(NextPage::Done)));
        let no_more = Pagination {
            has_next_page: false,
            next_cursor: Some("c2".to_string()),
        };
        assert!(matches!(
            next_page(Some(no_more), &mut seen),
            Ok(NextPage::Done)
        ));
        let no_cursor = Pagination {
            has_next_page: true,
            next_cursor: None,
        };
        assert!(matches!(
            next_page(Some(no_cursor), &mut seen),
            Ok(NextPage::Done)
        ));
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
