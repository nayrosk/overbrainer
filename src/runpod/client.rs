//! `RunpodClient`: the REST v2 calls overbrainer makes, with authentication,
//! retries and error messages that never carry a secret.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue, RETRY_AFTER};
use reqwest::{Method, StatusCode};
use secrecy::{ExposeSecret, SecretString};
use serde::de::DeserializeOwned;

use super::logs::LogQuery;
use super::target::MIN_CUDA_VERSION;
use super::types::{
    CreatePod, DataCenter, DataCenterList, GpuType, GpuTypeList, NetworkVolume, NetworkVolumeList,
    NewSecret, Pagination, Pod, PodId, PodPage, Secret, SecretList, SecretValue, Template,
    TemplatePage,
};
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
/// Largest answer read, in bytes: a listing is far below it, so anything past
/// it is refused instead of filling memory.
const MAX_BODY: usize = 8 * 1024 * 1024;
/// Largest error answer read from the pod log stream, in bytes.
const MAX_STREAM_ERROR: usize = 64 * 1024;
/// How often an idle log stream's connection is probed by TCP keepalives, so a
/// dead connection ends instead of waiting forever.
const STREAM_KEEPALIVE: Duration = Duration::from_secs(30);
/// The media type of the pod log stream.
const EVENT_STREAM: &str = "text/event-stream";
/// Timeout of `POST /pods`, which answers once the pod is placed.
const CREATE_TIMEOUT: Duration = Duration::from_secs(60);
/// Items asked for per page of `GET /pods` (the v2 maximum).
const PODS_PAGE_SIZE: &str = "1000";
/// Retries of a network volume resize after its first attempt. Few, so a
/// failing grow soon hands over to the snapshot instead of holding the disk
/// policy for minutes while the disk fills.
const RESIZE_RETRIES: u32 = 1;
/// Items asked for per page of `GET /templates` (the v2 maximum).
const TEMPLATES_PAGE_SIZE: &str = "100";
/// The GPU types of the catalog (v2 reference: `GET /v2/catalog/gpus`).
const GPUS_PATH: &str = "catalog/gpus";
/// The data centers of the catalog (v2 reference: `GET /v2/catalog/datacenters`).
const DATA_CENTERS_PATH: &str = "catalog/datacenters";
/// The account's network volumes (v2 reference: `GET /v2/network-volumes`).
const NETWORK_VOLUMES_PATH: &str = "network-volumes";
/// The account's templates, paginated (v2 reference: `GET /v2/templates`).
const TEMPLATES_PATH: &str = "templates";
/// The account's secrets (v2 reference: `GET/POST /v2/account/secrets`,
/// `PATCH/DELETE /v2/account/secrets/{id}`).
const SECRETS_PATH: &str = "account/secrets";
/// Length of the substring windows checked against the account key, so a value
/// Runpod echoes only in part still disappears.
const REDACT_WINDOW: usize = 16;
/// Pages of a paginated list followed before giving up: protects against a server
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
/// A secret write was refused with a 403: the API key lacks the permission.
pub const SECRETS_FORBIDDEN_MESSAGE: &str = "the Runpod API key may not manage secrets (403): give it read and write access to the account's secrets";
/// A secret write was answered 404.
const SECRET_UNKNOWN_MESSAGE: &str = "Runpod does not know the secret";
/// A secret create was answered 409.
const SECRET_EXISTS_MESSAGE: &str = "a Runpod secret with this name already exists";
/// A secret write was answered another 4xx.
const SECRET_INVALID_MESSAGE: &str =
    "Runpod rejected the secret (please report it: overbrainer built an invalid request)";
/// A secret write was answered a 5xx or another status.
const SECRET_SERVER_ERROR_MESSAGE: &str = "Runpod failed to process the secret request";
/// Errors of the Runpod API. No variant ever holds the API key or a pod's host
/// key. A create call's error never holds any text from Runpod's answer either,
/// however that text was shaped: its message is chosen only from the HTTP
/// status (see `RunpodClient::status_error`), because no amount of pattern
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
        /// Whether this is a create call's 400 recognized as Runpod's capacity
        /// failure: see [`ApiError::is_capacity`].
        capacity: bool,
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

    /// Whether a create call failed because Runpod has no capacity left for the
    /// GPU type (a 400 whose body is Runpod's capacity failure), so another GPU
    /// type may still be placed. False for any other error, including any other
    /// 400, which would fail the same way for every GPU type.
    #[must_use]
    pub fn is_capacity(&self) -> bool {
        matches!(self, Self::Status { capacity: true, .. })
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

/// What to do after fetching one page of a paginated list.
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
    /// The client of the log streams, which never end by themselves: a
    /// connection timeout only, no overall one.
    stream: reqwest::Client,
    base_url: String,
    api_key: SecretString,
    /// Other secrets the redaction looks for (the job's, and the pod's host
    /// key once stored), shared by every clone of this client.
    secrets: Arc<Mutex<Vec<SecretString>>>,
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
            .default_headers(headers.clone())
            .user_agent(USER_AGENT)
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(ApiError::Client)?;
        let stream = reqwest::Client::builder()
            .default_headers(headers)
            .user_agent(USER_AGENT)
            .connect_timeout(CONNECT_TIMEOUT)
            .tcp_keepalive(STREAM_KEEPALIVE)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(ApiError::Client)?;
        Ok(Self {
            http,
            stream,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.clone(),
            secrets: Arc::new(Mutex::new(Vec::new())),
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

    /// Every pod of the account, following the pagination up to a fixed page limit.
    ///
    /// # Errors
    ///
    /// Returns an [`ApiError`] once retries are exhausted, on a fatal answer, when
    /// a pagination cursor repeats, or when the list does not end within that limit.
    pub async fn list_pods(&self) -> Result<Vec<Pod>, ApiError> {
        self.paginate("pods", PODS_PAGE_SIZE, |page: PodPage| {
            (page.pods, page.pagination)
        })
        .await
    }

    /// The Secure Cloud GPU types of the catalog with their pod stock for
    /// `gpu_count` GPUs (at least 1), overall and per data center, counting
    /// only machines whose driver has [`MIN_CUDA_VERSION`], the version every
    /// create asks for (`GET /catalog/gpus?include=AVAILABILITY&product=POD`
    /// `&cloud=SECURE&minCudaVersion=...`).
    ///
    /// # Errors
    ///
    /// Returns an [`ApiError`] once retries are exhausted or on a fatal answer.
    pub async fn list_gpu_types(&self, gpu_count: u32) -> Result<Vec<GpuType>, ApiError> {
        let count = gpu_count.max(1).to_string();
        let query = [
            ("include", "AVAILABILITY"),
            ("product", "POD"),
            ("cloud", "SECURE"),
            ("count", count.as_str()),
            ("minCudaVersion", MIN_CUDA_VERSION),
        ];
        let list: GpuTypeList = self.get_json(&self.list_url(GPUS_PATH, &query)?).await?;
        Ok(list.gpus)
    }

    /// The data centers of the catalog with the stock of each GPU type they
    /// offer (`GET /catalog/datacenters?include=GPU_AVAILABILITY`).
    ///
    /// # Errors
    ///
    /// Returns an [`ApiError`] once retries are exhausted or on a fatal answer.
    pub async fn list_data_centers(&self) -> Result<Vec<DataCenter>, ApiError> {
        let query = [("include", "GPU_AVAILABILITY")];
        let url = self.list_url(DATA_CENTERS_PATH, &query)?;
        let list: DataCenterList = self.get_json(&url).await?;
        Ok(list.data_centers)
    }

    /// The account's network volumes (`GET /network-volumes`).
    ///
    /// # Errors
    ///
    /// Returns an [`ApiError`] once retries are exhausted or on a fatal answer.
    pub async fn list_network_volumes(&self) -> Result<Vec<NetworkVolume>, ApiError> {
        let url = self.list_url(NETWORK_VOLUMES_PATH, &[])?;
        let list: NetworkVolumeList = self.get_json(&url).await?;
        Ok(list.network_volumes)
    }

    /// The network volume `id` (`GET /network-volumes/{id}`), or `None` when
    /// Runpod says in its own error shape that it does not know it.
    ///
    /// # Errors
    ///
    /// Returns an [`ApiError`] once retries are exhausted or on a fatal answer.
    pub async fn get_network_volume(&self, id: &str) -> Result<Option<NetworkVolume>, ApiError> {
        let url = self.volume_url(id)?;
        with_retry(
            &self.policy,
            || async {
                let (status, body, retry_after) = self
                    .fetch(self.http.request(Method::GET, url.clone()))
                    .await?;
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

    /// Sets the size of the network volume `id` to `size_gb` (`PATCH
    /// /network-volumes/{id}`), which Runpod only allows to grow, and returns
    /// the volume as Runpod answers. Asking twice for the same size changes
    /// nothing more, but it is retried at most `RESIZE_RETRIES` times: the
    /// disk policy waits on it while the disk fills, and it stops the job with
    /// a snapshot when the grow fails.
    ///
    /// # Errors
    ///
    /// Returns an [`ApiError`] once retries are exhausted or on a fatal answer.
    pub async fn resize_network_volume(
        &self,
        id: &str,
        size_gb: u32,
    ) -> Result<NetworkVolume, ApiError> {
        let url = self.volume_url(id)?;
        let body = serde_json::json!({ "size": size_gb });
        let policy = RetryPolicy {
            max_retries: self.policy.max_retries.min(RESIZE_RETRIES),
            ..self.policy
        };
        with_retry(
            &policy,
            || async {
                let answer = self
                    .send(self.http.request(Method::PATCH, url.clone()).json(&body))
                    .await?;
                decode(&answer, false)
            },
            log_retry,
        )
        .await
    }

    /// The URL of the network volume `id`, the ID escaped as one path segment.
    fn volume_url(&self, id: &str) -> Result<reqwest::Url, ApiError> {
        self.segment_url(NETWORK_VOLUMES_PATH, id)
    }

    /// The URL of `path` followed by `id`, escaped as one path segment.
    fn segment_url(&self, path: &str, id: &str) -> Result<reqwest::Url, ApiError> {
        let mut url = self.list_url(path, &[])?;
        url.path_segments_mut()
            .map_err(|()| ApiError::InvalidResponse("the Runpod base URL has no path".into()))?
            .push(id);
        Ok(url)
    }

    /// The account's pod templates (serverless ones are left out), following
    /// the pagination of `GET /templates` up to a fixed page limit; an ID
    /// listed twice keeps its first occurrence.
    ///
    /// # Errors
    ///
    /// Returns an [`ApiError`] once retries are exhausted, on a fatal answer, when
    /// a pagination cursor repeats, or when the list does not end within that limit.
    pub async fn list_templates(&self) -> Result<Vec<Template>, ApiError> {
        let templates = self
            .paginate(TEMPLATES_PATH, TEMPLATES_PAGE_SIZE, |page: TemplatePage| {
                (page.templates, page.pagination)
            })
            .await?;
        let mut ids = HashSet::new();
        Ok(templates
            .into_iter()
            .filter(|template| !template.serverless && ids.insert(template.id.clone()))
            .collect())
    }

    /// Every item of the cursor-paginated list at `path`, `page_size` items
    /// asked for per page, `split` taking a page apart into its items and its
    /// pagination.
    async fn paginate<P, T>(
        &self,
        path: &str,
        page_size: &str,
        split: impl Fn(P) -> (Vec<T>, Option<Pagination>),
    ) -> Result<Vec<T>, ApiError>
    where
        P: DeserializeOwned,
    {
        let mut items = Vec::new();
        let mut cursor: Option<String> = None;
        let mut seen = HashSet::new();
        for _ in 0..MAX_PAGES {
            let mut query = vec![("limit", page_size)];
            if let Some(cursor) = cursor.as_deref() {
                query.push(("cursor", cursor));
            }
            let page: P = self.get_json(&self.list_url(path, &query)?).await?;
            let (page_items, pagination) = split(page);
            items.extend(page_items);
            match next_page(pagination, &mut seen)? {
                NextPage::Done => return Ok(items),
                NextPage::Cursor(next) => cursor = Some(next),
            }
        }
        Err(ApiError::InvalidResponse(format!(
            "Runpod's {path} list did not end after {MAX_PAGES} pages"
        )))
    }

    /// The URL of `path` with `query`.
    fn list_url(&self, path: &str, query: &[(&str, &str)]) -> Result<reqwest::Url, ApiError> {
        let mut url = reqwest::Url::parse(&self.url(path))
            .map_err(|error| ApiError::InvalidResponse(error.to_string()))?;
        if !query.is_empty() {
            url.query_pairs_mut().extend_pairs(query);
        }
        Ok(url)
    }

    /// `GET url`, retried, with a redacted error for a failed answer.
    async fn get_json<T: DeserializeOwned>(&self, url: &reqwest::Url) -> Result<T, ApiError> {
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

    /// Deletes the pod `id`: `true` when Runpod deleted it, `false` when it
    /// already did not know it (a 404 in its own error shape, which counts as
    /// deleted); a 404 with an unrelated body is a real error.
    ///
    /// # Errors
    ///
    /// Returns an [`ApiError`] once retries are exhausted or on a fatal answer.
    pub async fn delete_pod(&self, id: &PodId) -> Result<bool, ApiError> {
        let url = self.url(&format!("pods/{id}"));
        with_retry(
            &self.policy,
            || async {
                let (status, body, retry_after) =
                    self.fetch(self.http.request(Method::DELETE, &url)).await?;
                if status.is_success() {
                    return Ok(true);
                }
                if status == StatusCode::NOT_FOUND && is_runpod_error_shape(status, &body) {
                    return Ok(false);
                }
                Err(self.status_error(status, &body, retry_after, false))
            },
            log_retry,
        )
        .await
    }

    /// This client with `secrets` (the values the run's job gets, such as
    /// its Hugging Face token), which the pod's logs never show.
    #[must_use]
    pub fn with_secrets(self, secrets: Vec<SecretString>) -> Self {
        *self.extra_secrets() = secrets;
        self
    }

    /// Adds `secret` to what this client and its clones redact: the pod logs
    /// and the text of Runpod's error answers never show it.
    pub fn redact_also(&self, secret: SecretString) {
        self.extra_secrets().push(secret);
    }

    /// The secrets of [`RunpodClient::with_secrets`] and
    /// [`RunpodClient::redact_also`]. A poisoned lock still holds a usable list.
    fn extra_secrets(&self) -> std::sync::MutexGuard<'_, Vec<SecretString>> {
        self.secrets.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The secrets the pod log redaction looks for: the account API key,
    /// then those of [`RunpodClient::with_secrets`] and
    /// [`RunpodClient::redact_also`].
    pub(super) fn log_secrets(&self) -> Vec<SecretString> {
        std::iter::once(self.api_key.clone())
            .chain(self.extra_secrets().iter().cloned())
            .collect()
    }

    /// The account's secrets, or only the one named `name` (Runpod matches the
    /// name without regard to case): `GET /account/secrets[?name=]`. Their
    /// values are never returned.
    ///
    /// # Errors
    ///
    /// Returns an [`ApiError`] once retries are exhausted or on a fatal answer.
    pub async fn list_secrets(&self, name: Option<&str>) -> Result<Vec<Secret>, ApiError> {
        let query: Vec<(&str, &str)> = name.map(|name| ("name", name)).into_iter().collect();
        let list: SecretList = self.get_json(&self.list_url(SECRETS_PATH, &query)?).await?;
        Ok(list.secrets)
    }

    /// Creates the secret `secret` (`POST /account/secrets`). Retried like a
    /// read: a retry after a create that went through answers 409, which the
    /// caller handles like any other 409 (the name is taken).
    ///
    /// # Errors
    ///
    /// Returns an [`ApiError`] whose message is chosen from the status alone
    /// (see `secret_write_message`): never anything Runpod said, which could
    /// echo the value.
    pub async fn create_secret(&self, secret: &NewSecret) -> Result<Secret, ApiError> {
        let url = self.list_url(SECRETS_PATH, &[])?;
        with_retry(
            &self.policy,
            || async {
                let builder = self.http.request(Method::POST, url.clone()).json(secret);
                self.write_secret(builder).await
            },
            log_retry,
        )
        .await
    }

    /// Sets the value of the secret `id` (`PATCH /account/secrets/{id}`).
    ///
    /// # Errors
    ///
    /// As [`RunpodClient::create_secret`].
    pub async fn update_secret_value(
        &self,
        id: &str,
        value: &SecretString,
    ) -> Result<Secret, ApiError> {
        let url = self.segment_url(SECRETS_PATH, id)?;
        let body = SecretValue(value);
        with_retry(
            &self.policy,
            || async {
                let builder = self.http.request(Method::PATCH, url.clone()).json(&body);
                self.write_secret(builder).await
            },
            log_retry,
        )
        .await
    }

    /// Deletes the secret `id` (`DELETE /account/secrets/{id}`): `true` when
    /// Runpod deleted it, `false` when it already did not know it (a 404 in its
    /// own error shape).
    ///
    /// # Errors
    ///
    /// Returns an [`ApiError`] once retries are exhausted or on a fatal answer.
    pub async fn delete_secret(&self, id: &str) -> Result<bool, ApiError> {
        let url = self.segment_url(SECRETS_PATH, id)?;
        with_retry(
            &self.policy,
            || async {
                let (status, body, retry_after) = self
                    .fetch(self.http.request(Method::DELETE, url.clone()))
                    .await?;
                if status.is_success() {
                    return Ok(true);
                }
                if status == StatusCode::NOT_FOUND && is_runpod_error_shape(status, &body) {
                    return Ok(false);
                }
                Err(self.status_error(status, &body, retry_after, false))
            },
            log_retry,
        )
        .await
    }

    /// Sends a secret write and reads the secret Runpod answers with. A failed
    /// answer's message is fixed by its status: see `secret_write_message`.
    async fn write_secret(&self, builder: reqwest::RequestBuilder) -> Result<Secret, ApiError> {
        let (status, body, retry_after) = self.fetch(builder).await?;
        if status.is_success() {
            return decode(&body, false);
        }
        Err(ApiError::Status {
            status: status.as_u16(),
            message: secret_write_message(status).to_string(),
            retry_after,
            capacity: false,
        })
    }

    /// Opens the log stream of the pod `id` (`GET /pods/{id}/logs`, an event
    /// stream that never ends by itself): `query`'s cursor goes in
    /// `Last-Event-ID`, which Runpod reads before `tail`. Never retried: the
    /// callers reconnect themselves.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::Status`] for a failed answer (its text cleaned of the
    /// account key), [`ApiError::Transport`] when it cannot be sent, and
    /// [`ApiError::InvalidResponse`] when a success answer is not an event
    /// stream (a proxy answering for the API).
    pub(super) async fn open_pod_logs(
        &self,
        id: &PodId,
        query: &LogQuery,
    ) -> Result<reqwest::Response, ApiError> {
        let mut url = reqwest::Url::parse(&self.url(&format!("pods/{id}/logs")))
            .map_err(|error| ApiError::InvalidResponse(error.to_string()))?;
        let tail = query.tail.map(|tail| tail.to_string());
        {
            let mut pairs = url.query_pairs_mut();
            if let Some(source) = query.source {
                pairs.append_pair("source", source.name());
            }
            if let Some(tail) = &tail {
                pairs.append_pair("tail", tail);
            }
            if let Some(since) = &query.since {
                pairs.append_pair("since", since);
            }
        }
        let mut request = self.stream.get(url).header(ACCEPT, EVENT_STREAM);
        if let Some(cursor) = &query.cursor {
            request = request.header("Last-Event-ID", cursor.as_str());
        }
        let mut response = request.send().await.map_err(ApiError::Transport)?;
        let status = response.status();
        if !status.is_success() {
            let retry_after = retry_after(response.headers());
            let mut body = Vec::new();
            while body.len() < MAX_STREAM_ERROR {
                match response.chunk().await {
                    Ok(Some(chunk)) => body.extend_from_slice(&chunk),
                    Ok(None) | Err(_) => break,
                }
            }
            body.truncate(MAX_STREAM_ERROR);
            let body = String::from_utf8_lossy(&body);
            return Err(self.status_error(status, &body, retry_after, false));
        }
        if let Some(kind) = response.headers().get(CONTENT_TYPE) {
            let kind = kind.to_str().unwrap_or_default();
            let media = kind.split(';').next().unwrap_or_default().trim();
            if !media.eq_ignore_ascii_case(EVENT_STREAM) {
                return Err(ApiError::InvalidResponse(format!(
                    "the pod log stream answered {} instead of {EVENT_STREAM}",
                    cap(media)
                )));
            }
        }
        Ok(response)
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{path}", self.base_url)
    }

    /// Sends `builder` and returns its raw status, body and `Retry-After`. Only a
    /// transport failure (connection, TLS, timeout) becomes an [`ApiError`] here:
    /// a non-success HTTP status is returned as data, unredacted, so `get_pod`
    /// and `delete_pod` can recognize Runpod's own "not found" shape, and
    /// `status_error` can classify a create failure, before anyone builds the
    /// client-facing error from it. A body past [`MAX_BODY`] is an
    /// [`ApiError::InvalidResponse`].
    async fn fetch(
        &self,
        builder: reqwest::RequestBuilder,
    ) -> Result<(StatusCode, String, Option<Duration>), ApiError> {
        let mut response = builder.send().await.map_err(ApiError::Transport)?;
        let status = response.status();
        let retry_after = retry_after(response.headers());
        let too_large = || {
            ApiError::InvalidResponse(format!(
                "the answer is larger than {} MiB",
                MAX_BODY / (1024 * 1024)
            ))
        };
        if response
            .content_length()
            .is_some_and(|length| length > MAX_BODY as u64)
        {
            return Err(too_large());
        }
        let mut body = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(ApiError::Transport)? {
            if body.len() + chunk.len() > MAX_BODY {
                return Err(too_large());
            }
            body.extend_from_slice(&chunk);
        }
        Ok((
            status,
            String::from_utf8_lossy(&body).into_owned(),
            retry_after,
        ))
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
            let secrets = self.log_secrets();
            let secrets: Vec<&str> = secrets.iter().map(ExposeSecret::expose_secret).collect();
            build_message(status, body, &secrets)
        };
        ApiError::Status {
            status: status.as_u16(),
            message,
            retry_after,
            capacity: is_create && status == StatusCode::BAD_REQUEST && is_capacity_failure(body),
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

/// The fixed message for a secret write's failed answer, chosen only from
/// `status`: a secret write's answer is never shown, since Runpod could echo
/// the value in it. A 403 names the permission the API key lacks.
fn secret_write_message(status: StatusCode) -> &'static str {
    match status {
        StatusCode::UNAUTHORIZED => AUTH_REJECTED_MESSAGE,
        StatusCode::FORBIDDEN => SECRETS_FORBIDDEN_MESSAGE,
        StatusCode::NOT_FOUND => SECRET_UNKNOWN_MESSAGE,
        StatusCode::CONFLICT => SECRET_EXISTS_MESSAGE,
        StatusCode::TOO_MANY_REQUESTS => RATE_LIMITED_MESSAGE,
        status if status.is_client_error() => SECRET_INVALID_MESSAGE,
        _ => SECRET_SERVER_ERROR_MESSAGE,
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
/// failed answer. Each of `secrets` (the account key first) is redacted in
/// two passes: JSON-decoding while
/// extracting the message (`error_text`) can rejoin a value an escape sequence
/// split (for example `\/` becomes `/`), so redacting only the raw body is not
/// enough; capping happens only after both passes, so a truncation can never
/// leave a partial key behind.
fn build_message(status: StatusCode, body: &str, secrets: &[&str]) -> String {
    let redact_all = |text: &str| {
        secrets
            .iter()
            .fold(text.to_string(), |text, secret| redact(&text, secret))
    };
    let first_pass = redact_all(body);
    let extracted = error_text(status, &first_pass);
    let second_pass = redact_all(&extracted);
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
/// offset, so a value Runpod echoes only in part still disappears. Used for
/// the account API key and the client's other secrets in a non-create call's
/// answer (a create call never shows any of its answer at all).
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
        // The messages never repeat the text: it could hold what got through.
        assert!(
            !cleaned.contains(piece),
            "a piece of the secret survived redaction"
        );
        assert!(
            cleaned == "prefix *** suffix",
            "the redacted text is not the expected one"
        );
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
        let message = build_message(StatusCode::BAD_REQUEST, &body, &[&key]);
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
            capacity: false,
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
        for code in [400, 403, 500] {
            assert!(!status(code).is_capacity(), "{code}");
        }
    }

    #[test]
    fn only_a_create_400_about_capacity_is_a_capacity_error() -> Result<(), ApiError> {
        let client = RunpodClient::new("http://127.0.0.1:1/v2", &SecretString::from("k"))?;
        let capacity_body = r#"{"detail": "There are no longer any instances available with the requested specifications. Please refresh and try again.", "status": 400, "title": "Bad Request"}"#;
        let error = |status, body, is_create| client.status_error(status, body, None, is_create);
        let capacity = error(StatusCode::BAD_REQUEST, capacity_body, true);
        assert!(capacity.is_capacity());
        assert_eq!(
            capacity.to_string(),
            format!("Runpod answered 400: {CAPACITY_MESSAGE}")
        );
        assert!(!error(StatusCode::BAD_REQUEST, r#"{"detail": "bad body"}"#, true).is_capacity());
        assert!(!error(StatusCode::BAD_REQUEST, capacity_body, false).is_capacity());
        assert!(!error(StatusCode::FORBIDDEN, capacity_body, true).is_capacity());
        Ok(())
    }
}
