//! [`Hub`] on the `hf-hub` crate.

use std::time::Duration;

use hf_hub::{
    HFClient, HFError, RepoTypeModel,
    progress::{ProgressEvent, ProgressHandler, UploadEvent},
    repository::CommitOperation,
};
use secrecy::{ExposeSecret, SecretString};

use super::{Commit, Hub, HubError, Progress, ProgressSink, RepoId, RepoState, UploadFile};

const DEFAULT_BASE_URL: &str = "https://huggingface.co";
const USER_AGENT: &str = concat!("overbrainer/", env!("CARGO_PKG_VERSION"));
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// The Hugging Face Hub, reached through `hf-hub`.
pub struct HfHub {
    client: HFClient,
    /// Only to keep it out of error messages.
    token: SecretString,
}

impl HfHub {
    /// A client for `base_url`, from the `OVERBRAINER_HUB__BASE_URL` environment
    /// variable (no `overbrainer.toml` key), default `https://huggingface.co`.
    ///
    /// The token is always passed explicitly, so `hf-hub` never reads `HF_TOKEN`
    /// or the token file. There is no total timeout: uploads are long.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP client cannot be built.
    pub fn new(base_url: Option<&str>, token: &SecretString) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .connect_timeout(CONNECT_TIMEOUT)
            .build()?;
        let client = HFClient::builder()
            .endpoint(base_url.unwrap_or(DEFAULT_BASE_URL))
            .token(token.expose_secret())
            .cache_enabled(false)
            .client(http)
            .build()?;
        Ok(Self {
            client,
            token: token.clone(),
        })
    }
}

impl HfHub {
    /// Maps a crate error. `namespace` names the owner for [`HubError::Forbidden`].
    /// The crate's `Display` carries the server's `error` field, which a server
    /// could fill with the token: such a message is withheld. `Debug` carries
    /// response bodies, so it is never used.
    fn map(&self, error: &HFError, namespace: &str) -> HubError {
        match error {
            HFError::AuthRequired { .. } => HubError::Auth,
            HFError::Forbidden { .. } => HubError::Forbidden {
                namespace: namespace.to_string(),
            },
            HFError::RateLimited { .. } => HubError::RateLimited,
            other => {
                let text = other.to_string();
                let token = self.token.expose_secret();
                if !token.is_empty() && text.contains(token) {
                    HubError::Other(
                        "the Hub's error message held the token, so it is not shown".to_string(),
                    )
                } else {
                    HubError::Other(text)
                }
            },
        }
    }
}

/// Forwards the crate's byte progress to a channel.
struct Handler(ProgressSink);

impl ProgressHandler for Handler {
    fn on_progress(&self, event: &ProgressEvent) {
        if let ProgressEvent::Upload(UploadEvent::Progress {
            bytes_completed,
            total_bytes,
            ..
        }) = event
        {
            // The receiver is gone after a cancel while the crate's poller lives on.
            let _ = self.0.send(Progress {
                done: *bytes_completed,
                total: *total_bytes,
            });
        }
    }
}

impl Hub for HfHub {
    async fn whoami(&self) -> Result<String, HubError> {
        match self.client.whoami().send().await {
            Ok(user) => Ok(user.username),
            Err(HFError::Forbidden { .. }) => Err(HubError::Auth),
            Err(error) => Err(self.map(&error, "")),
        }
    }

    async fn ensure_repo(&self, repo: &RepoId, private: bool) -> Result<RepoState, HubError> {
        let namespace = repo.namespace.as_str();
        match self
            .client
            .model(&repo.namespace, &repo.name)
            .info()
            .send()
            .await
        {
            Ok(info) => {
                return Ok(RepoState::Existing {
                    private: info.private.unwrap_or(true),
                });
            },
            Err(HFError::RepoNotFound { .. }) => {},
            Err(error) => return Err(self.map(&error, namespace)),
        }
        let id = repo.to_string();
        self.client
            .create_repository()
            .repo_id(id.as_str())
            .repo_type(RepoTypeModel)
            .private(private)
            .exist_ok(true)
            .send()
            .await
            .map_err(|error| self.map(&error, namespace))?;
        Ok(RepoState::Created { private })
    }

    async fn remote_card(&self, repo: &RepoId) -> Result<Option<String>, HubError> {
        match self
            .client
            .model(&repo.namespace, &repo.name)
            .download_file_to_bytes()
            .filename("README.md")
            .send()
            .await
        {
            Ok(bytes) => Ok(Some(String::from_utf8_lossy(&bytes).into_owned())),
            Err(HFError::EntryNotFound { .. }) => Ok(None),
            Err(error) => Err(self.map(&error, &repo.namespace)),
        }
    }

    async fn license_of(&self, model: &str) -> Result<Option<String>, HubError> {
        // A local path or a bare name is not a Hub repo: no request.
        let Some(RepoId {
            namespace: owner,
            name,
        }) = RepoId::parse(model)
        else {
            return Ok(None);
        };
        let owner = owner.as_str();
        match self
            .client
            .model(owner, &name)
            .info()
            .expand(vec!["cardData".to_string()])
            .send()
            .await
        {
            Ok(info) => Ok(info
                .card_data
                .as_ref()
                .and_then(|card| card.get("license"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)),
            Err(HFError::RepoNotFound { .. }) => Ok(None),
            Err(error) => Err(self.map(&error, owner)),
        }
    }

    async fn upload(
        &self,
        repo: &RepoId,
        files: Vec<UploadFile>,
        card: Option<String>,
        message: String,
        progress: ProgressSink,
    ) -> Result<Commit, HubError> {
        let mut operations: Vec<CommitOperation> = files
            .into_iter()
            .map(|file| CommitOperation::add_file(file.path_in_repo, file.local))
            .collect();
        if let Some(card) = card {
            operations.push(CommitOperation::add_bytes("README.md", card));
        }
        let info = self
            .client
            .model(&repo.namespace, &repo.name)
            .create_commit()
            .operations(operations)
            .commit_message(message)
            .progress(Handler(progress))
            .send()
            .await
            .map_err(|error| self.map(&error, &repo.namespace))?;
        match (info.commit_url, info.commit_oid) {
            (Some(url), Some(oid)) => Ok(Commit { url, oid }),
            _ => Err(HubError::Other(
                "the commit response carried no URL or id".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use serde_json::json;
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{method, path},
    };

    use super::*;

    type TestResult = Result<(), Box<dyn Error>>;

    const TOKEN: &str = "hf_test";

    fn hub(server: &MockServer) -> anyhow::Result<HfHub> {
        HfHub::new(Some(&server.uri()), &SecretString::from(TOKEN))
    }

    fn repo(text: &str) -> Result<RepoId, Box<dyn Error>> {
        RepoId::parse(text).ok_or_else(|| "bad repo id".into())
    }

    fn not_found() -> ResponseTemplate {
        ResponseTemplate::new(404).set_body_json(json!({"error": "Repository not found"}))
    }

    #[tokio::test]
    async fn whoami_reads_the_user_name() -> TestResult {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/whoami-v2"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({"name": "nayrosk", "type": "user"})),
            )
            .mount(&server)
            .await;
        assert_eq!(hub(&server)?.whoami().await?, "nayrosk");
        Ok(())
    }

    #[tokio::test]
    async fn a_rejected_token_is_auth_and_never_echoed() -> TestResult {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/whoami-v2"))
            .respond_with(ResponseTemplate::new(401).set_body_string(TOKEN))
            .mount(&server)
            .await;
        let error = hub(&server)?
            .whoami()
            .await
            .err()
            .ok_or("expected an error")?;
        assert!(matches!(error, HubError::Auth), "401 must map to Auth");
        assert!(!error.to_string().contains(TOKEN), "token must not appear");
        Ok(())
    }

    #[tokio::test]
    async fn whoami_maps_a_403_to_auth() -> TestResult {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/whoami-v2"))
            .respond_with(ResponseTemplate::new(403))
            .mount(&server)
            .await;
        let error = hub(&server)?
            .whoami()
            .await
            .err()
            .ok_or("expected an error")?;
        assert!(matches!(error, HubError::Auth), "403 must map to Auth");
        Ok(())
    }

    #[tokio::test]
    async fn forbidden_names_the_namespace() -> TestResult {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/org/x"))
            .respond_with(not_found())
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/repos/create"))
            .respond_with(ResponseTemplate::new(403).set_body_json(json!({"error": "no"})))
            .mount(&server)
            .await;
        let error = hub(&server)?
            .ensure_repo(&repo("org/x")?, true)
            .await
            .err()
            .ok_or("expected an error")?;
        assert!(
            matches!(&error, HubError::Forbidden { namespace } if namespace == "org"),
            "403 must name the repo namespace"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_429_is_rate_limited() -> TestResult {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/me/x"))
            .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "0"))
            .mount(&server)
            .await;
        let error = hub(&server)?
            .ensure_repo(&repo("me/x")?, true)
            .await
            .err()
            .ok_or("expected an error")?;
        assert!(
            matches!(error, HubError::RateLimited),
            "429 must map to RateLimited"
        );
        Ok(())
    }

    #[tokio::test]
    async fn ensure_repo_creates_a_missing_repo_private() -> TestResult {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/me/x"))
            .respond_with(not_found())
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/repos/create"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"url": "https://hf.co/me/x"})),
            )
            .mount(&server)
            .await;
        let state = hub(&server)?.ensure_repo(&repo("me/x")?, true).await?;
        assert_eq!(state, RepoState::Created { private: true });
        let requests = server.received_requests().await.ok_or("no request log")?;
        let create = requests
            .iter()
            .find(|request| request.url.path() == "/api/repos/create")
            .ok_or("no create request")?;
        let body: serde_json::Value = serde_json::from_slice(&create.body)?;
        assert_eq!(body["private"], json!(true));
        assert_eq!(body["name"], json!("x"));
        assert_eq!(body["organization"], json!("me"));
        Ok(())
    }

    #[tokio::test]
    async fn ensure_repo_creates_a_missing_repo_public() -> TestResult {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/me/x"))
            .respond_with(not_found())
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/repos/create"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"url": "https://hf.co/me/x"})),
            )
            .mount(&server)
            .await;
        let state = hub(&server)?.ensure_repo(&repo("me/x")?, false).await?;
        assert_eq!(state, RepoState::Created { private: false });
        let requests = server.received_requests().await.ok_or("no request log")?;
        let create = requests
            .iter()
            .find(|request| request.url.path() == "/api/repos/create")
            .ok_or("no create request")?;
        let body: serde_json::Value = serde_json::from_slice(&create.body)?;
        assert_eq!(body["private"], json!(false));
        Ok(())
    }

    #[tokio::test]
    async fn a_server_error_is_other_and_never_echoes_the_token() -> TestResult {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/me/x"))
            .respond_with(
                ResponseTemplate::new(500)
                    .set_body_json(json!({"error": format!("bad token {TOKEN}")})),
            )
            .mount(&server)
            .await;
        let error = hub(&server)?
            .ensure_repo(&repo("me/x")?, true)
            .await
            .err()
            .ok_or("expected an error")?;
        assert!(matches!(error, HubError::Other(_)), "500 must map to Other");
        assert!(!error.to_string().contains(TOKEN), "token must not appear");
        Ok(())
    }

    #[tokio::test]
    async fn ensure_repo_reports_an_existing_repo_and_its_visibility() -> TestResult {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/me/x"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"id": "me/x", "private": false})),
            )
            .mount(&server)
            .await;
        let state = hub(&server)?.ensure_repo(&repo("me/x")?, true).await?;
        assert_eq!(state, RepoState::Existing { private: false });
        Ok(())
    }

    #[tokio::test]
    async fn remote_card_is_none_without_readme() -> TestResult {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/me/none/resolve/main/README.md"))
            .respond_with(ResponseTemplate::new(404).insert_header("x-error-code", "EntryNotFound"))
            .mount(&server)
            .await;
        // The crate sends a HEAD for the metadata before the GET.
        Mock::given(method("HEAD"))
            .and(path("/me/some/resolve/main/README.md"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("etag", "\"abc\"")
                    .insert_header("x-repo-commit", "0123456789abcdef0123456789abcdef01234567")
                    .insert_header("content-length", "6"),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/me/some/resolve/main/README.md"))
            .respond_with(ResponseTemplate::new(200).set_body_string("# card"))
            .mount(&server)
            .await;
        let hub = hub(&server)?;
        assert_eq!(hub.remote_card(&repo("me/none")?).await?, None);
        assert_eq!(
            hub.remote_card(&repo("me/some")?).await?.as_deref(),
            Some("# card")
        );
        Ok(())
    }

    #[tokio::test]
    async fn license_of_reads_card_data() -> TestResult {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/Qwen/Qwen3-0.6B"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                json!({"id": "Qwen/Qwen3-0.6B", "cardData": {"license": "apache-2.0"}}),
            ))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/api/models/me/gone"))
            .respond_with(not_found())
            .mount(&server)
            .await;
        let hub = hub(&server)?;
        assert_eq!(
            hub.license_of("Qwen/Qwen3-0.6B").await?.as_deref(),
            Some("apache-2.0")
        );
        assert_eq!(hub.license_of("me/gone").await?, None);
        let before = server
            .received_requests()
            .await
            .ok_or("no request log")?
            .len();
        assert_eq!(hub.license_of("/models/x").await?, None);
        let after = server
            .received_requests()
            .await
            .ok_or("no request log")?
            .len();
        assert_eq!(before, after, "a local path must not reach the Hub");
        Ok(())
    }

    /// A commit of `files` and `card` to `me/x` on a stub; the commit request's body.
    async fn committed(card: Option<String>) -> Result<String, Box<dyn Error>> {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/models/me/x/preupload/main"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"files": [
                {"path": "adapter_config.json", "uploadMode": "regular"},
                {"path": "README.md", "uploadMode": "regular"}
            ]})))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/models/me/x/commit/main"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                json!({"commitUrl": "https://hf.co/me/x/commit/abc", "commitOid": "abc"}),
            ))
            .mount(&server)
            .await;
        let dir = tempfile::tempdir()?;
        let local = dir.path().join("adapter_config.json");
        std::fs::write(&local, "{}")?;
        let files = vec![UploadFile {
            local,
            path_in_repo: "adapter_config.json".into(),
            size: 2,
        }];
        let (sink, _receiver) = tokio::sync::mpsc::unbounded_channel();
        hub(&server)?
            .upload(&repo("me/x")?, files, card, "push run".into(), sink)
            .await?;
        let requests = server.received_requests().await.ok_or("no request log")?;
        let commit_request = requests
            .iter()
            .find(|request| request.url.path() == "/api/models/me/x/commit/main")
            .ok_or("no commit request")?;
        Ok(String::from_utf8(commit_request.body.clone())?)
    }

    #[tokio::test]
    async fn a_kept_card_is_not_in_the_commit() -> TestResult {
        let body = committed(None).await?;
        assert!(!body.contains("README.md"), "the card stays as it is");
        assert!(body.contains("adapter_config.json"));
        Ok(())
    }

    #[tokio::test]
    async fn upload_commits_small_files_inline_with_the_card() -> TestResult {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/models/me/x/preupload/main"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"files": [
                {"path": "adapter_config.json", "uploadMode": "regular"},
                {"path": "README.md", "uploadMode": "regular"}
            ]})))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/api/models/me/x/commit/main"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                json!({"commitUrl": "https://hf.co/me/x/commit/abc", "commitOid": "abc"}),
            ))
            .mount(&server)
            .await;
        let dir = tempfile::tempdir()?;
        let local = dir.path().join("adapter_config.json");
        std::fs::write(&local, "{}")?;
        let files = vec![UploadFile {
            local,
            path_in_repo: "adapter_config.json".into(),
            size: 2,
        }];
        let (sink, _receiver) = tokio::sync::mpsc::unbounded_channel();
        let commit = hub(&server)?
            .upload(
                &repo("me/x")?,
                files,
                Some("# card".into()),
                "push run".into(),
                sink,
            )
            .await?;
        assert_eq!(
            commit,
            Commit {
                url: "https://hf.co/me/x/commit/abc".into(),
                oid: "abc".into()
            }
        );
        let requests = server.received_requests().await.ok_or("no request log")?;
        let commit_request = requests
            .iter()
            .find(|request| request.url.path() == "/api/models/me/x/commit/main")
            .ok_or("no commit request")?;
        let body = String::from_utf8(commit_request.body.clone())?;
        assert!(body.contains("README.md"), "the card goes in the commit");
        assert!(body.contains("push run"), "the summary is the message");
        assert!(body.contains("adapter_config.json"));
        Ok(())
    }
}
