//! Secret resolution: literal values or `vault:` references.

mod reference;
mod vault;

use secrecy::{ExposeSecret, SecretString};

pub use reference::{VaultRef, parse_reference};
pub use vault::{SecretSource, VaultSettings, VaultSource};

/// Errors from parsing secret references or fetching secrets from Vault.
#[derive(Debug, thiserror::Error)]
pub enum SecretError {
    /// The value looked like a `vault:` reference but did not match the expected shape.
    #[error("invalid secret reference: {0}")]
    InvalidReference(String),
    /// `VAULT_ADDR` is set but could not be parsed as a URL.
    #[error("invalid VAULT_ADDR: {0}")]
    InvalidVaultAddress(String),
    /// `VAULT_ADDR` is set but no token was found in `VAULT_TOKEN` or `~/.vault-token`.
    #[error("VAULT_ADDR is set but no token was found in VAULT_TOKEN or ~/.vault-token")]
    MissingVaultToken,
    /// A `vault:` reference was found but `VAULT_ADDR` is not set.
    #[error("a vault: reference was found but VAULT_ADDR is not set")]
    VaultNotConfigured,
    /// The Vault request failed.
    #[error("vault request failed: {0}")]
    Vault(String),
    /// The referenced field is missing or not a string.
    #[error("field missing or not a string at {reference}")]
    MissingField {
        /// The reference that was looked up.
        reference: String,
    },
}

/// Resolves configuration values that may be literals or Vault references.
pub struct Resolver<S> {
    source: Option<S>,
}

impl<S: SecretSource> Resolver<S> {
    /// `source: None` means Vault is not configured; references then fail with
    /// [`SecretError::VaultNotConfigured`].
    #[must_use]
    pub fn new(source: Option<S>) -> Self {
        Self { source }
    }

    /// Resolves `value`: literals are returned unchanged, `vault:` references are fetched
    /// from the configured [`SecretSource`].
    ///
    /// # Errors
    ///
    /// Returns an error if `value` is a malformed reference, if it is a reference but no
    /// source was configured, or if fetching from the source fails.
    pub async fn resolve(&self, value: &SecretString) -> Result<SecretString, SecretError> {
        match parse_reference(value.expose_secret())? {
            None => Ok(value.clone()),
            Some(reference) => match &self.source {
                Some(source) => source.fetch(&reference).await,
                None => Err(SecretError::VaultNotConfigured),
            },
        }
    }
}
