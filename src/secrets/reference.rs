use super::SecretError;

const PREFIX: &str = "vault:";

/// Location of a secret in a Vault or `OpenBao` KV v2 engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VaultRef {
    /// KV v2 mount point, e.g. `secret`.
    pub mount: String,
    /// Path within the mount, e.g. `overbrainer/nanogpt`.
    pub path: String,
    /// Field name within the secret's data.
    pub field: String,
}

impl std::fmt::Display for VaultRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{PREFIX}{}/{}#{}", self.mount, self.path, self.field)
    }
}

/// Parses `vault:<mount>/<path>#<field>`. Returns `Ok(None)` for literal values.
///
/// Error messages never include the raw value, since a malformed literal may be a secret.
///
/// # Errors
///
/// Returns [`SecretError::InvalidReference`] when `raw` starts with the `vault:` prefix
/// but does not match the expected shape.
pub fn parse_reference(raw: &str) -> Result<Option<VaultRef>, SecretError> {
    let Some(rest) = raw.strip_prefix(PREFIX) else {
        return Ok(None);
    };
    let invalid =
        || SecretError::InvalidReference("expected vault:<mount>/<path>#<field>".to_string());
    let (location, field) = rest.split_once('#').ok_or_else(invalid)?;
    let (mount, path) = location.split_once('/').ok_or_else(invalid)?;
    if mount.is_empty() || path.is_empty() || field.is_empty() {
        return Err(invalid());
    }
    Ok(Some(VaultRef {
        mount: mount.to_string(),
        path: path.to_string(),
        field: field.to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_values_are_not_references() -> Result<(), SecretError> {
        assert_eq!(parse_reference("sk-abc")?, None);
        Ok(())
    }

    #[test]
    fn full_reference_is_parsed() -> Result<(), SecretError> {
        let parsed = parse_reference("vault:secret/overbrainer/nanogpt#api_key")?;
        assert_eq!(
            parsed,
            Some(VaultRef {
                mount: "secret".into(),
                path: "overbrainer/nanogpt".into(),
                field: "api_key".into(),
            })
        );
        Ok(())
    }

    #[test]
    fn malformed_references_are_rejected() {
        for raw in [
            "vault:",
            "vault:secret#k",
            "vault:secret/path",
            "vault:/path#k",
            "vault:secret/#k",
            "vault:secret/p#",
        ] {
            assert!(
                matches!(parse_reference(raw), Err(SecretError::InvalidReference(_))),
                "{raw}"
            );
        }
    }
}
