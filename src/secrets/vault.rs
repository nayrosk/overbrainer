use std::collections::HashMap;
use std::future::Future;
use std::path::Path;

use secrecy::{ExposeSecret, SecretString};
use vaultrs::client::{VaultClient, VaultClientSettingsBuilder};

use super::{SecretError, VaultRef};

/// Something able to fetch a secret referenced by a [`VaultRef`].
pub trait SecretSource {
    /// Fetches the secret at `reference`.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails or the referenced field is missing.
    fn fetch(
        &self,
        reference: &VaultRef,
    ) -> impl Future<Output = Result<SecretString, SecretError>> + Send;
}

/// Connection settings, read from `VAULT_ADDR` and `VAULT_TOKEN` (or `~/.vault-token`).
pub struct VaultSettings {
    /// Base address of the Vault or `OpenBao` server.
    pub address: url::Url,
    /// Token used to authenticate requests.
    pub token: SecretString,
}

impl VaultSettings {
    /// Returns `Ok(None)` when `VAULT_ADDR` is not set.
    ///
    /// `get` reads an environment variable; `home` is the user's home directory.
    ///
    /// # Errors
    ///
    /// Returns an error if `VAULT_ADDR` cannot be parsed as a URL, or if no token can be
    /// found in `VAULT_TOKEN` or `~/.vault-token`.
    pub fn from_env(
        get: impl Fn(&str) -> Option<String>,
        home: Option<&Path>,
    ) -> Result<Option<Self>, SecretError> {
        let Some(raw_address) = get("VAULT_ADDR") else {
            return Ok(None);
        };
        let address = url::Url::parse(&raw_address)
            .map_err(|e| SecretError::InvalidVaultAddress(e.to_string()))?;
        let token = match get("VAULT_TOKEN") {
            Some(token) => token,
            None => home
                .map(|home| home.join(".vault-token"))
                .and_then(|path| std::fs::read_to_string(path).ok())
                .map(|token| token.trim().to_string())
                .filter(|token| !token.is_empty())
                .ok_or(SecretError::MissingVaultToken)?,
        };
        Ok(Some(Self {
            address,
            token: SecretString::from(token),
        }))
    }
}

/// Reads KV v2 secrets from `HashiCorp` Vault or `OpenBao`.
pub struct VaultSource {
    client: VaultClient,
}

impl VaultSource {
    /// Builds a [`VaultSource`] from `settings`.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying Vault client cannot be built.
    pub fn new(settings: &VaultSettings) -> Result<Self, SecretError> {
        // The address is already a parsed `Url`, so the vaultrs setter cannot panic on it.
        let client_settings = VaultClientSettingsBuilder::default()
            .address(settings.address.as_str())
            .token(settings.token.expose_secret())
            .build()
            .map_err(|e| SecretError::Vault(e.to_string()))?;
        let client =
            VaultClient::new(client_settings).map_err(|e| SecretError::Vault(e.to_string()))?;
        Ok(Self { client })
    }
}

impl SecretSource for VaultSource {
    async fn fetch(&self, reference: &VaultRef) -> Result<SecretString, SecretError> {
        let data: HashMap<String, serde_json::Value> =
            vaultrs::kv2::read(&self.client, &reference.mount, &reference.path)
                .await
                .map_err(|e| SecretError::Vault(e.to_string()))?;
        match data.get(&reference.field) {
            Some(serde_json::Value::String(value)) => Ok(SecretString::from(value.clone())),
            _ => Err(SecretError::MissingField {
                reference: reference.to_string(),
            }),
        }
    }
}
