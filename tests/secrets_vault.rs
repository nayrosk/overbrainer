use std::path::Path;

use overbrainer::secrets::{
    Resolver, SecretError, SecretSource, VaultRef, VaultSettings, VaultSource,
};
use secrecy::{ExposeSecret, SecretString};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn kv2_body(data: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "request_id": "req-1",
        "lease_id": "",
        "lease_duration": 0,
        "renewable": false,
        "data": {
            "data": data,
            "metadata": {
                "created_time": "2026-09-22T00:00:00Z",
                "deletion_time": "",
                "custom_metadata": null,
                "destroyed": false,
                "version": 1
            }
        },
        "warnings": null,
        "wrap_info": null,
        "auth": null
    })
}

fn settings(server: &MockServer) -> Result<VaultSettings, Box<dyn std::error::Error>> {
    Ok(VaultSettings {
        address: url::Url::parse(&server.uri())?,
        token: SecretString::from("test-token"),
    })
}

fn nanogpt_ref() -> VaultRef {
    VaultRef {
        mount: "secret".into(),
        path: "overbrainer/nanogpt".into(),
        field: "api_key".into(),
    }
}

#[tokio::test]
async fn reads_a_kv2_field() -> Result<(), Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/overbrainer/nanogpt"))
        .and(header("X-Vault-Token", "test-token"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(kv2_body(&serde_json::json!({"api_key": "sk-123"}))),
        )
        .mount(&server)
        .await;

    let source = VaultSource::new(&settings(&server)?)?;
    let secret = source.fetch(&nanogpt_ref()).await?;
    assert_eq!(secret.expose_secret(), "sk-123");
    Ok(())
}

#[tokio::test]
async fn missing_field_is_reported_without_value() -> Result<(), Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/overbrainer/nanogpt"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(kv2_body(&serde_json::json!({"other": "sk-123"}))),
        )
        .mount(&server)
        .await;

    let source = VaultSource::new(&settings(&server)?)?;
    match source.fetch(&nanogpt_ref()).await {
        Err(SecretError::MissingField { reference }) => {
            assert_eq!(reference, "vault:secret/overbrainer/nanogpt#api_key");
            Ok(())
        },
        other => Err(format!("expected MissingField, got {:?}", other.map(|_| "***")).into()),
    }
}

#[tokio::test]
async fn resolver_passes_literals_through_and_fetches_references()
-> Result<(), Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/overbrainer/nanogpt"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(kv2_body(&serde_json::json!({"api_key": "sk-123"}))),
        )
        .mount(&server)
        .await;

    let resolver = Resolver::new(Some(VaultSource::new(&settings(&server)?)?));
    let literal = resolver.resolve(&SecretString::from("plain")).await?;
    assert_eq!(literal.expose_secret(), "plain");
    let fetched = resolver
        .resolve(&SecretString::from(
            "vault:secret/overbrainer/nanogpt#api_key",
        ))
        .await?;
    assert_eq!(fetched.expose_secret(), "sk-123");
    Ok(())
}

#[tokio::test]
async fn reference_without_vault_fails_clearly() {
    let resolver: Resolver<VaultSource> = Resolver::new(None);
    let result = resolver
        .resolve(&SecretString::from("vault:secret/a#b"))
        .await;
    assert!(matches!(result, Err(SecretError::VaultNotConfigured)));
}

#[test]
fn settings_come_from_env_and_token_file() -> Result<(), Box<dyn std::error::Error>> {
    let home = tempfile::tempdir()?;
    std::fs::write(home.path().join(".vault-token"), "file-token\n")?;

    let none = VaultSettings::from_env(|_| None, Some(home.path()))?;
    assert!(none.is_none());

    let from_file = VaultSettings::from_env(
        |key| (key == "VAULT_ADDR").then(|| "http://127.0.0.1:8200".to_string()),
        Some(home.path()),
    )?
    .ok_or("settings expected")?;
    assert_eq!(from_file.token.expose_secret(), "file-token");

    let from_env = VaultSettings::from_env(
        |key| match key {
            "VAULT_ADDR" => Some("http://127.0.0.1:8200".to_string()),
            "VAULT_TOKEN" => Some("env-token".to_string()),
            _ => None,
        },
        Some(home.path()),
    )?
    .ok_or("settings expected")?;
    assert_eq!(from_env.token.expose_secret(), "env-token");

    let no_token = VaultSettings::from_env(
        |key| (key == "VAULT_ADDR").then(|| "http://127.0.0.1:8200".to_string()),
        Some(Path::new("/nonexistent")),
    );
    assert!(matches!(no_token, Err(SecretError::MissingVaultToken)));

    let bad_addr = VaultSettings::from_env(
        |key| (key == "VAULT_ADDR").then(|| "not a url".to_string()),
        Some(home.path()),
    );
    assert!(matches!(bad_addr, Err(SecretError::InvalidVaultAddress(_))));
    Ok(())
}

/// Renders `error` and every source below it, the way `{:#}` does in main.
fn full_chain(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut current = error.source();
    while let Some(source) = current {
        text.push_str(": ");
        text.push_str(&source.to_string());
        current = source.source();
    }
    text
}

#[tokio::test]
async fn connection_failure_keeps_its_source_chain() -> Result<(), Box<dyn std::error::Error>> {
    // Reserve a free port, then close it so the connection is refused.
    let port = std::net::TcpListener::bind("127.0.0.1:0")?
        .local_addr()?
        .port();
    let source = VaultSource::new(&VaultSettings {
        address: url::Url::parse(&format!("http://127.0.0.1:{port}"))?,
        token: SecretString::from("test-token"),
    })?;
    match source.fetch(&nanogpt_ref()).await {
        Err(error @ SecretError::Vault(_)) => {
            let chain = full_chain(&error);
            assert!(
                std::error::Error::source(&error).is_some(),
                "source missing: {chain}"
            );
            assert!(
                chain.to_lowercase().contains("connection refused"),
                "cause missing: {chain}"
            );
            assert!(!chain.contains("test-token"), "token leaked: {chain}");
            assert!(!format!("{error:?}").contains("test-token"), "token leaked");
            Ok(())
        },
        other => Err(format!("expected Vault, got {:?}", other.map(|_| "***")).into()),
    }
}

#[tokio::test]
async fn unparsable_response_does_not_echo_its_content() -> Result<(), Box<dyn std::error::Error>> {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/secret/data/overbrainer/nanogpt"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(kv2_body(&serde_json::json!("sk-leak-42"))),
        )
        .mount(&server)
        .await;

    let source = VaultSource::new(&settings(&server)?)?;
    match source.fetch(&nanogpt_ref()).await {
        Err(error) => {
            let chain = full_chain(&error);
            assert!(!chain.contains("sk-leak-42"), "secret leaked: {chain}");
            assert!(
                !format!("{error:?}").contains("sk-leak-42"),
                "secret leaked in Debug"
            );
            Ok(())
        },
        Ok(_) => Err("expected an error".into()),
    }
}

#[test]
fn unreadable_token_file_is_not_reported_as_missing() -> Result<(), Box<dyn std::error::Error>> {
    let home = tempfile::tempdir()?;
    // A directory where the token file should be: it exists but cannot be read.
    std::fs::create_dir(home.path().join(".vault-token"))?;
    let result = VaultSettings::from_env(
        |key| (key == "VAULT_ADDR").then(|| "http://127.0.0.1:8200".to_string()),
        Some(home.path()),
    );
    match result {
        Err(error @ SecretError::VaultTokenFile { .. }) => {
            assert!(std::error::Error::source(&error).is_some());
            Ok(())
        },
        other => Err(format!("expected VaultTokenFile, got {:?}", other.map(|_| "***")).into()),
    }
}

#[test]
fn empty_vault_token_falls_back_to_the_token_file() -> Result<(), Box<dyn std::error::Error>> {
    let home = tempfile::tempdir()?;
    std::fs::write(home.path().join(".vault-token"), "file-token\n")?;
    for empty in ["", "  \n"] {
        let settings = VaultSettings::from_env(
            |key| match key {
                "VAULT_ADDR" => Some("http://127.0.0.1:8200".to_string()),
                "VAULT_TOKEN" => Some(empty.to_string()),
                _ => None,
            },
            Some(home.path()),
        )?
        .ok_or("settings expected")?;
        assert_eq!(settings.token.expose_secret(), "file-token");
    }

    let no_file = VaultSettings::from_env(
        |key| match key {
            "VAULT_ADDR" => Some("http://127.0.0.1:8200".to_string()),
            "VAULT_TOKEN" => Some(String::new()),
            _ => None,
        },
        Some(Path::new("/nonexistent")),
    );
    assert!(matches!(no_file, Err(SecretError::MissingVaultToken)));
    Ok(())
}
