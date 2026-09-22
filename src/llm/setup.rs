use std::time::Duration;

use super::{LlmError, ProtocolClient};
use crate::config::{RoleModel, Settings};
use crate::secrets::{Resolver, SecretError, SecretSource};

/// Errors while building a client from the configuration.
#[derive(Debug, thiserror::Error)]
pub enum SetupError {
    /// The role names a provider that is not declared.
    #[error("unknown provider `{0}`")]
    UnknownProvider(String),
    /// The provider has no base URL.
    #[error("providers.{name}.base_url is not set (set {variable})")]
    MissingBaseUrl {
        /// Provider name.
        name: String,
        /// Env variable to set.
        variable: String,
    },
    /// The API key reference could not be resolved.
    #[error("cannot resolve providers.{name}.api_key")]
    Secret {
        /// Provider name.
        name: String,
        /// Underlying secret error.
        #[source]
        source: SecretError,
    },
    /// The HTTP client could not be built.
    #[error("cannot build the client for provider `{name}`")]
    Client {
        /// Provider name.
        name: String,
        /// Underlying client error.
        #[source]
        source: LlmError,
    },
}

/// Builds the client for `role`: resolves the provider's API key (literal or Vault
/// reference) and uses `pipeline.request_timeout_secs` as request timeout. A provider
/// without `api_key` gets no auth header, which suits local servers.
///
/// # Errors
///
/// Returns a [`SetupError`] when the provider is unknown, has no base URL, its key
/// cannot be resolved, or the client cannot be built.
pub async fn connect<S: SecretSource>(
    settings: &Settings,
    role: &RoleModel,
    resolver: &Resolver<S>,
) -> Result<ProtocolClient, SetupError> {
    let name = &role.provider;
    let provider = settings
        .providers
        .get(name)
        .ok_or_else(|| SetupError::UnknownProvider(name.clone()))?;
    let base_url = provider
        .base_url
        .as_deref()
        .ok_or_else(|| SetupError::MissingBaseUrl {
            name: name.clone(),
            variable: format!("OVERBRAINER_PROVIDERS__{}__BASE_URL", name.to_uppercase()),
        })?;
    let api_key = match &provider.api_key {
        Some(value) => {
            Some(
                resolver
                    .resolve(value)
                    .await
                    .map_err(|source| SetupError::Secret {
                        name: name.clone(),
                        source,
                    })?,
            )
        },
        None => None,
    };
    let timeout = Duration::from_secs(settings.pipeline.request_timeout_secs);
    ProtocolClient::new(
        provider.protocol,
        base_url,
        api_key.as_ref(),
        &role.model,
        timeout,
    )
    .map_err(|source| SetupError::Client {
        name: name.clone(),
        source,
    })
}
