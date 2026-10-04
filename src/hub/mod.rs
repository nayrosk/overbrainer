//! Publishing to the Hugging Face Hub.

pub mod card;
pub mod files;
mod hf;

pub use hf::HfHub;

use std::{fmt, future::Future, path::PathBuf};

use crate::train::sizing::is_repo_id;

/// `NAMESPACE/NAME`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoId {
    pub namespace: String,
    pub name: String,
}

impl RepoId {
    /// Parses `NAMESPACE/NAME`: exactly one `/`, both parts non-empty and valid
    /// Hub name characters.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        let (namespace, name) = text.split_once('/')?;
        if name.contains('/') || !is_repo_id(text) {
            return None;
        }
        Some(Self {
            namespace: namespace.to_string(),
            name: name.to_string(),
        })
    }
}

impl fmt::Display for RepoId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.namespace, self.name)
    }
}

/// One file to upload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UploadFile {
    pub local: PathBuf,
    pub path_in_repo: String,
    pub size: u64,
}

/// What `ensure_repo` found or made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepoState {
    Created { private: bool },
    Existing { private: bool },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commit {
    pub url: String,
    pub oid: String,
}

/// Upload progress, bytes over the files that go through Xet/LFS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    pub done: u64,
    pub total: u64,
}

pub type ProgressSink = tokio::sync::mpsc::UnboundedSender<Progress>;

#[derive(Debug, thiserror::Error)]
pub enum HubError {
    #[error("the Hugging Face token is missing or invalid")]
    Auth,
    #[error("the Hugging Face token needs write access to {namespace}")]
    Forbidden { namespace: String },
    #[error("Hugging Face rate limit, retry later")]
    RateLimited,
    #[error("Hugging Face request failed: {0}")]
    Other(String),
}

/// A Hugging Face Hub the pipeline can push a run to.
///
/// Futures are `Send` so calls can run in spawned tasks.
pub trait Hub: Send + Sync {
    /// The user name the token belongs to.
    ///
    /// # Errors
    ///
    /// Returns [`HubError::Auth`] when the token is missing or invalid.
    fn whoami(&self) -> impl Future<Output = Result<String, HubError>> + Send;

    /// Makes sure the model repo exists, creating it with the given visibility
    /// when it does not.
    ///
    /// # Errors
    ///
    /// Returns a [`HubError`] when the lookup or the creation fails.
    fn ensure_repo(
        &self,
        repo: &RepoId,
        private: bool,
    ) -> impl Future<Output = Result<RepoState, HubError>> + Send;

    /// The repo's README.md, `None` when it has none.
    ///
    /// # Errors
    ///
    /// Returns a [`HubError`] when the request fails.
    fn remote_card(
        &self,
        repo: &RepoId,
    ) -> impl Future<Output = Result<Option<String>, HubError>> + Send;

    /// `license` of a Hub model's card, `None` when unknown or not found.
    ///
    /// # Errors
    ///
    /// Returns a [`HubError`] when the request fails.
    fn license_of(
        &self,
        model: &str,
    ) -> impl Future<Output = Result<Option<String>, HubError>> + Send;

    /// One commit with every file plus README.md = `card`; nothing is committed
    /// unless every upload finished.
    ///
    /// # Errors
    ///
    /// Returns a [`HubError`] when an upload or the commit fails.
    #[allow(clippy::too_many_arguments)]
    fn upload(
        &self,
        repo: &RepoId,
        files: Vec<UploadFile>,
        card: String,
        message: String,
        progress: ProgressSink,
    ) -> impl Future<Output = Result<Commit, HubError>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_one_slash_and_displays_it_back() {
        let repo = RepoId::parse("me/model-1");
        assert_eq!(
            repo,
            Some(RepoId {
                namespace: "me".into(),
                name: "model-1".into()
            })
        );
        assert_eq!(repo.map(|r| r.to_string()).as_deref(), Some("me/model-1"));
    }

    #[test]
    fn parse_rejects_anything_but_namespace_slash_name() {
        for text in ["a", "a/b/c", "/b", "a/", "", "a/b c", ".a/b"] {
            assert_eq!(RepoId::parse(text), None, "{text:?} must be rejected");
        }
    }
}
