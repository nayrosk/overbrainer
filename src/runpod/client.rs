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

/// Longest error text kept from a non-create answer.
const MAX_MESSAGE_CHARS: usize = 300;
/// Connection timeout of every request.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Timeout of a request, answer included.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Timeout of `POST /pods`, which answers once the pod is placed.
const CREATE_TIMEOUT: Duration = Duration::from_secs(60);
/// Pods asked for per page of `GET /pods`.
const PAGE_SIZE: &str = "1000";
/// Length of the substring windows checked against the account key, so a value
/// Runpod echoes only in part still disappears.
const REDACT_WINDOW: usize = 16;
/// Pages of `GET /pods` followed before giving up: protects against a server
/// whose pagination never reports `hasNextPage: false`.
const MAX_PAGES: usize = 100;
/// Substring of Runpod's capacity failure (`POST /pods`, 400) that identifies
/// it. Only used to classify a create failure; the `detail` it comes from is
/// never shown.
const CAPACITY_SIGNATURE: &str = "no longer any instances available";

/// A create call's answer had no GPU capacity for the requested type.
const CAPACITY_MESSAGE: &str = "no capacity for this GPU type";
/// A create call's answer was a 401.
const AUTH_REJECTED_MESSAGE: &str = "Runpod refused the API key";
/// A create call's answer was a 403. The provisioning walk reads a 403 as "skip
/// this GPU type", so the message names both possible causes.
const FORBIDDEN_MESSAGE: &str =
    "Runpod refused the request (403): check the API key and that the GPU type is allowed";
/// A create call's answer was still a 429 once its retries ran out.
const RATE_LIMITED_MESSAGE: &str = "Runpod is rate limiting requests; try again later";
/// A create call's answer was a 408: ambiguous, like [`ApiError::is_ambiguous`]
/// says, since Runpod may have processed the request before timing out.
const TIMEOUT_MESSAGE: &str =
    "Runpod timed out on the create request (the pod may have been created anyway)";
/// A create call's answer was a 402.
const INSUFFICIENT_BALANCE_MESSAGE: &str = "the Runpod account balance is insufficient";
/// A create call's answer was a 4xx that is none of the above.
const INVALID_REQUEST_MESSAGE: &str =
    "Runpod rejected the create request (please report it: overbrainer built an invalid request)";
/// A create call's answer was a 5xx or another status none of the above covers.
const SERVER_ERROR_MESSAGE: &str = "Runpod failed to process the create request";
/// A create call succeeded (2xx) but its body was not the shape overbrainer
/// expected.
const INVALID_CREATE_ANSWER_MESSAGE: &str = "Runpod's answer to the create call could not be read (please report it: overbrainer built an invalid request)";
/// A non-create call succeeded (2xx) but its body was not the shape overbrainer
/// expected. Followed only by serde's error category and position.
const INVALID_ANSWER_MESSAGE: &str = "Runpod's answer does not have the expected shape";

/// Errors of the Runpod API. No variant ever holds the API key or a pod's host
/// key. A create call's error never holds any text from Runpod's answer either,
/// however that text was shaped: its message is chosen only from the HTTP
/// status (see [`RunpodClient::status_error`]), because no amount of pattern
/// matching against ways a server might chunk, escape or encode an echoed value
/// can be shown complete; not showing any of it is the only claim this client
/// can make and keep.
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
        /// For a non-create call, Runpod's own text, cleaned of the account
        /// key and capped. For a create call, one of a fixed set of messages
        /// chosen only by status: never anything Runpod said.
        message: String,
        /// Parsed `Retry-After`, when present.
        retry_after: Option<Duration>,
    },
    /// A success answer whose body is not what the API documents. A decode
    /// failure's text is a fixed description plus serde's category and
    /// position: never serde's own message, which quotes the offending value.
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
        Err(ApiError::InvalidResponse(
            "Runpod repeated a pagination cursor".to_string(),
        ))
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
    /// Returns an [`ApiError`] when the pod was not created or the answer cannot
    /// be read. Its message is never built from anything Runpod said: see
    /// [`ApiError`].
    pub async fn create_pod(&self, request: &CreatePod) -> Result<Pod, ApiError> {
        let mut attempt = 0;
        loop {
            let builder = self
                .http
                .post(self.url("pods"))
                .json(request)
                .timeout(CREATE_TIMEOUT);
            let (status, body, retry_after) = self.fetch(builder).await?;
            if status.is_success() {
                return decode(&body, true);
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
            return Err(self.status_error(status, &body, retry_after, true));
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
                    return decode(&body, false).map(Some);
                }
                if status == StatusCode::NOT_FOUND && is_runpod_error_shape(status, &body) {
                    return Ok(None);
                }
                Err(self.status_error(status, &body, retry_after, false))
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
                    .send(self.http.request(Method::GET, url.clone()))
                    .await?;
                decode(&body, false)
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
                Err(self.status_error(status, &body, retry_after, false))
            },
            log_retry,
        )
        .await
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{path}", self.base_url)
    }

    /// Sends `builder` and returns its raw status, body and `Retry-After`. Only a
    /// transport failure (connection, TLS, timeout) becomes an [`ApiError`] here:
    /// a non-success HTTP status is returned as data, unredacted, so `get_pod`
    /// and `delete_pod` can recognize Runpod's own "not found" shape, and
    /// `status_error` can classify a create failure, before anyone builds the
    /// client-facing error from it.
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
    /// redacted [`ApiError::Status`] for a failure. Only used for non-create
    /// calls.
    async fn send(&self, builder: reqwest::RequestBuilder) -> Result<String, ApiError> {
        let (status, body, retry_after) = self.fetch(builder).await?;
        if status.is_success() {
            return Ok(body);
        }
        Err(self.status_error(status, &body, retry_after, false))
    }

    /// Builds the [`ApiError::Status`] a caller sees from a failed answer.
    ///
    /// A create call (`is_create`) never shows any text from Runpod: its
    /// message is a fixed string chosen only from `status` (and, for a 400,
    /// whether `body` is Runpod's capacity failure), by
    /// [`create_failure_message`]. A non-create call's message is Runpod's own
    /// text, with the account API key redacted and the result capped.
    fn status_error(
        &self,
        status: StatusCode,
        body: &str,
        retry_after: Option<Duration>,
        is_create: bool,
    ) -> ApiError {
        let message = if is_create {
            create_failure_message(status, body).to_string()
        } else {
            build_message(status, body, self.api_key.expose_secret())
        };
        ApiError::Status {
            status: status.as_u16(),
            message,
            retry_after,
        }
    }
}

fn log_retry(error: &ApiError, wait: Duration) {
    tracing::warn!("{error}; retrying in {}s", wait.as_secs());
}

/// Parses a success body into `T`.
///
/// A parse failure never shows any text from Runpod, on any call: serde's own
/// message quotes the offending value (an `env` that came back as a string
/// would carry the whole host key, and `PodId`'s custom `Deserialize` embeds
/// the raw ID), and no redaction can know every secret a body might hold. For
/// a create call (`is_create`) the message is the fixed
/// [`INVALID_CREATE_ANSWER_MESSAGE`], for the same reason
/// [`RunpodClient::status_error`] never shows any answer text either; for any
/// other call it is [`decode_failure_message`].
fn decode<T: DeserializeOwned>(body: &str, is_create: bool) -> Result<T, ApiError> {
    serde_json::from_str(body).map_err(|error| {
        ApiError::InvalidResponse(if is_create {
            INVALID_CREATE_ANSWER_MESSAGE.to_string()
        } else {
            decode_failure_message(&error)
        })
    })
}

/// [`INVALID_ANSWER_MESSAGE`] with `error`'s category, line and column: the
/// only parts of a serde error that can never hold a byte of the body.
fn decode_failure_message(error: &serde_json::Error) -> String {
    let category = match error.classify() {
        serde_json::error::Category::Io => "I/O",
        serde_json::error::Category::Syntax => "syntax",
        serde_json::error::Category::Data => "data",
        serde_json::error::Category::Eof => "end of input",
    };
    format!(
        "{INVALID_ANSWER_MESSAGE} ({category} error at line {}, column {})",
        error.line(),
        error.column()
    )
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

/// The fixed message for a create call's failed answer, chosen only from
/// `status` and, for a 400, whether `body` is Runpod's capacity failure. A 408
/// names a timeout whose outcome is unknown, matching
/// [`ApiError::is_ambiguous`]; a 429 is one still there once retries ran out. Never
/// reads anything else from `body`, and never shows any of it: this is the only
/// place a create error's message is decided, so no echo of the request, in any
/// shape a server could produce, can ever reach a caller.
fn create_failure_message(status: StatusCode, body: &str) -> &'static str {
    match status {
        StatusCode::BAD_REQUEST if is_capacity_failure(body) => CAPACITY_MESSAGE,
        StatusCode::UNAUTHORIZED => AUTH_REJECTED_MESSAGE,
        StatusCode::FORBIDDEN => FORBIDDEN_MESSAGE,
        StatusCode::PAYMENT_REQUIRED => INSUFFICIENT_BALANCE_MESSAGE,
        StatusCode::REQUEST_TIMEOUT => TIMEOUT_MESSAGE,
        StatusCode::TOO_MANY_REQUESTS => RATE_LIMITED_MESSAGE,
        status if status.is_client_error() => INVALID_REQUEST_MESSAGE,
        _ => SERVER_ERROR_MESSAGE,
    }
}

/// Whether `body`'s `detail` is Runpod's capacity failure (identified by
/// [`CAPACITY_SIGNATURE`]). Only a boolean ever leaves this function: `detail`
/// itself is never shown, even for a capacity failure, since it is still text
/// Runpod chose and could, in principle, be made to include anything.
fn is_capacity_failure(body: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|value| value.get("detail")?.as_str().map(str::to_string))
        .is_some_and(|detail| detail.contains(CAPACITY_SIGNATURE))
}

/// The extracted, doubly-redacted, capped message for a non-create call's
/// failed answer. `account_key` is redacted in two passes: JSON-decoding while
/// extracting the message (`error_text`) can rejoin a value an escape sequence
/// split (for example `\/` becomes `/`), so redacting only the raw body is not
/// enough; capping happens only after both passes, so a truncation can never
/// leave a partial key behind.
fn build_message(status: StatusCode, body: &str, account_key: &str) -> String {
    let first_pass = redact(body, account_key);
    let extracted = error_text(status, &first_pass);
    let second_pass = redact(&extracted, account_key);
    cap(&second_pass)
}

/// What to say about a failed answer, extracted but not yet capped. A 401 never
/// quotes Runpod; a 403 keeps a non-JSON body (Cloudflare's `error code: 1010`
/// says what is wrong); anything else keeps the RFC 9457 `title`, `detail` and
/// `errors[]`, or the body start. `body` must already be cleaned of the account
/// key by a first redaction pass; the result still needs a second pass before
/// it is safe to cap, because JSON-decoding here can rejoin a value an escape
/// sequence split.
fn error_text(status: StatusCode, body: &str) -> String {
    if status == StatusCode::UNAUTHORIZED {
        return "Runpod rejected the API key, check OVERBRAINER_RUNPOD__API_KEY".to_string();
    }
    let parsed = serde_json::from_str::<serde_json::Value>(body).ok();
    if status == StatusCode::FORBIDDEN {
        return match parsed {
            Some(_) => "permission denied by Runpod".to_string(),
            None => format!("permission denied by Runpod: {}", body.trim()),
        };
    }
    let Some(value) = parsed else {
        return body.trim().to_string();
    };
    let mut parts: Vec<String> = ["title", "detail"]
        .into_iter()
        .filter_map(|key| value.get(key)?.as_str().map(str::to_string))
        .collect();
    if let Some(errors) = value.get("errors").and_then(serde_json::Value::as_array) {
        parts.extend(errors.iter().map(error_item));
    }
    if parts.is_empty() {
        body.trim().to_string()
    } else {
        parts.join("; ")
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

/// `text` with every occurrence of `secret` replaced by `***`: first every
/// exact occurrence, then every [`REDACT_WINDOW`]-character piece of it at any
/// offset, so a value Runpod echoes only in part still disappears. Used only
/// for the account API key, the only secret a non-create call's answer could
/// ever echo (a create call never shows any of its answer at all).
fn redact(text: &str, secret: &str) -> String {
    if secret.is_empty() {
        return text.to_string();
    }
    let replaced = text.replace(secret, "***");
    let chars: Vec<char> = secret.chars().collect();
    if chars.len() < REDACT_WINDOW {
        return replaced;
    }
    (0..=chars.len() - REDACT_WINDOW).fold(replaced, |text, start| {
        let window: String = chars[start..start + REDACT_WINDOW].iter().collect();
        text.replace(&window, "***")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_are_extracted_by_status() {
        assert_eq!(
            error_text(
                StatusCode::UNAUTHORIZED,
                r#"{"detail": "key rp_abc invalid"}"#
            ),
            "Runpod rejected the API key, check OVERBRAINER_RUNPOD__API_KEY"
        );
        assert_eq!(
            error_text(StatusCode::FORBIDDEN, "error code: 1010"),
            "permission denied by Runpod: error code: 1010"
        );
        assert_eq!(
            error_text(StatusCode::FORBIDDEN, r#"{"detail": "no"}"#),
            "permission denied by Runpod"
        );
        assert_eq!(
            error_text(
                StatusCode::UNPROCESSABLE_ENTITY,
                r#"{"title": "Invalid request", "detail": "bad body", "errors": ["env too big", {"message": "gpu.id unknown"}, {"path": "/x"}]}"#
            ),
            r#"Invalid request; bad body; env too big; gpu.id unknown; {"path":"/x"}"#
        );
        assert_eq!(
            error_text(StatusCode::BAD_GATEWAY, "upstream down"),
            "upstream down"
        );
    }

    #[test]
    fn long_messages_are_capped() {
        assert_eq!(cap(&"x".repeat(1000)).len(), MAX_MESSAGE_CHARS);
    }

    #[test]
    fn secrets_are_redacted() {
        assert_eq!(redact("key k1, k1 again", "k1"), "key ***, *** again");
        assert_eq!(redact("nothing hidden", ""), "nothing hidden");
    }

    #[test]
    fn windows_catch_a_partially_echoed_secret() {
        let secret = "0123456789abcdefghijXYZ";
        let piece = &secret[3..19];
        assert_eq!(piece.len(), REDACT_WINDOW);
        let text = format!("prefix {piece} suffix");
        let cleaned = redact(&text, secret);
        assert!(!cleaned.contains(piece), "{cleaned}");
        assert_eq!(cleaned, "prefix *** suffix");
    }

    #[test]
    fn secrets_shorter_than_a_window_only_match_exactly() {
        assert_eq!(redact("short s1 stays", "s1"), "short *** stays");
    }

    /// Pins the order `build_message` must follow: redact, extract, redact
    /// again, only then cap. Every 16-character window of this key contains a
    /// `/`, so a redaction pass over the still-escaped raw body (which spells
    /// it `\/`) can never match any window of it directly: only after
    /// `error_text` JSON-decodes the body and restores the real slashes can a
    /// second pass catch it. Reverting to a single pre-decode pass, or capping
    /// before that second pass, lets the decoded key straight through.
    #[test]
    fn json_unescaping_is_redacted_again_before_capping() {
        let key: String = (0..64)
            .map(|i| {
                if i % 4 == 3 {
                    '/'
                } else {
                    char::from(b'a' + u8::try_from(i % 26).unwrap_or(0))
                }
            })
            .collect();
        let escaped = key.replace('/', "\\/");
        let body = format!(r#"{{"detail": "bad key {escaped}"}}"#);
        let message = build_message(StatusCode::BAD_REQUEST, &body, &key);
        assert!(!message.contains(&key), "{message}");
    }

    #[test]
    fn is_capacity_failure_recognizes_only_the_signature() {
        assert!(is_capacity_failure(
            r#"{"detail": "There are no longer any instances available with the requested specifications. Please refresh and try again."}"#
        ));
        assert!(!is_capacity_failure(r#"{"detail": "bad body"}"#));
        assert!(!is_capacity_failure("not json"));
        assert!(!is_capacity_failure(r#"{"title": "no detail field"}"#));
    }

    #[test]
    fn create_failures_are_classified_by_status_alone() {
        let capacity_body = r#"{"detail": "There are no longer any instances available with the requested specifications. Please refresh and try again."}"#;
        assert_eq!(
            create_failure_message(StatusCode::BAD_REQUEST, capacity_body),
            CAPACITY_MESSAGE
        );
        assert_eq!(
            create_failure_message(StatusCode::BAD_REQUEST, r#"{"detail": "bad body"}"#),
            INVALID_REQUEST_MESSAGE
        );
        assert_eq!(
            create_failure_message(StatusCode::UNAUTHORIZED, ""),
            AUTH_REJECTED_MESSAGE
        );
        assert_eq!(
            create_failure_message(StatusCode::FORBIDDEN, ""),
            FORBIDDEN_MESSAGE
        );
        assert_eq!(
            create_failure_message(StatusCode::REQUEST_TIMEOUT, ""),
            TIMEOUT_MESSAGE
        );
        assert_eq!(
            create_failure_message(StatusCode::PAYMENT_REQUIRED, ""),
            INSUFFICIENT_BALANCE_MESSAGE
        );
        assert_eq!(
            create_failure_message(StatusCode::UNPROCESSABLE_ENTITY, ""),
            INVALID_REQUEST_MESSAGE
        );
        assert_eq!(
            create_failure_message(StatusCode::TOO_MANY_REQUESTS, ""),
            RATE_LIMITED_MESSAGE
        );
        assert_eq!(
            create_failure_message(StatusCode::INTERNAL_SERVER_ERROR, ""),
            SERVER_ERROR_MESSAGE
        );
        assert_eq!(
            create_failure_message(StatusCode::BAD_GATEWAY, ""),
            SERVER_ERROR_MESSAGE
        );
    }

    /// A non-alphanumeric key, so `PodId::new` genuinely rejects it (its
    /// custom `Deserialize` embeds the raw value in its error, which `decode`
    /// must clean) rather than failing on an unrelated type mismatch.
    const INVALID_POD_ID_KEY: &str = "topsecret+host/key=value1234567890";

    #[test]
    fn decode_errors_are_redacted() {
        let body = format!(r#""{INVALID_POD_ID_KEY}""#);
        let result: Result<PodId, ApiError> = decode(&body, false);
        let message = match result {
            Err(ApiError::InvalidResponse(message)) => message,
            Err(other) => other.to_string(),
            Ok(_) => String::new(),
        };
        assert!(!message.is_empty(), "expected a decode error");
        assert!(!message.contains(INVALID_POD_ID_KEY), "{message}");
    }

    #[test]
    fn a_decode_error_is_only_a_fixed_text_a_category_and_a_position() {
        let body = r#"{"id": "p1", "env": "OVERBRAINER_HOST_KEY=c2VjcmV0"}"#;
        let result: Result<Pod, ApiError> = decode(body, false);
        assert!(matches!(
            result,
            Err(ApiError::InvalidResponse(ref message))
                if message == &format!("{INVALID_ANSWER_MESSAGE} (data error at line 1, column 51)")
        ));
    }

    #[test]
    fn a_create_decode_error_never_shows_the_answer() {
        let body = format!(r#""{INVALID_POD_ID_KEY}""#);
        let result: Result<PodId, ApiError> = decode(&body, true);
        assert!(matches!(
            result,
            Err(ApiError::InvalidResponse(ref message)) if message == INVALID_CREATE_ANSWER_MESSAGE
        ));
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
