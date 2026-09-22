use std::fs;

use overbrainer::config::{ConfigError, EnvSource, load};
use secrecy::ExposeSecret;

const BASE: &str = r#"
[project]
name = "demo"

[providers.nanogpt]
protocol = "openai"

[roles]
generator = { provider = "nanogpt", model = "m1" }
parent = { provider = "nanogpt", model = "m2" }
"#;

fn project(toml: &str) -> Result<tempfile::TempDir, std::io::Error> {
    let dir = tempfile::tempdir()?;
    fs::write(dir.path().join("overbrainer.toml"), toml)?;
    Ok(dir)
}

fn env(pairs: &[(&str, &str)]) -> EnvSource {
    EnvSource::Vars(
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect(),
    )
}

#[test]
fn env_overrides_file_values() -> Result<(), Box<dyn std::error::Error>> {
    let dir = project(BASE)?;
    let settings = load(
        dir.path(),
        env(&[("OVERBRAINER_PIPELINE__CONCURRENCY", "16")]),
    )?;
    assert_eq!(settings.pipeline.concurrency, 16);
    Ok(())
}

#[test]
fn secrets_from_env_keep_their_exact_text() -> Result<(), Box<dyn std::error::Error>> {
    let dir = project(BASE)?;
    let settings = load(
        dir.path(),
        env(&[
            ("OVERBRAINER_PROVIDERS__NANOGPT__API_KEY", "0123"),
            (
                "OVERBRAINER_PROVIDERS__NANOGPT__BASE_URL",
                "https://nano-gpt.com/api/v1",
            ),
        ]),
    )?;
    let provider = settings
        .providers
        .get("nanogpt")
        .ok_or("provider missing")?;
    let key = provider.api_key.as_ref().ok_or("api_key missing")?;
    assert_eq!(key.expose_secret(), "0123");
    assert_eq!(
        provider.base_url.as_deref(),
        Some("https://nano-gpt.com/api/v1")
    );
    Ok(())
}

#[test]
fn env_only_keys_in_file_are_rejected() -> Result<(), Box<dyn std::error::Error>> {
    // `hf_token` must land in the document's root table, not inside the last
    // `[roles]` table, so it is prepended rather than appended.
    let toml = "hf_token = \"hf_leak\"\n".to_string()
        + &BASE.replace(
            "protocol = \"openai\"",
            "protocol = \"openai\"\napi_key = \"sk-leak\"\nbase_url = \"https://x\"",
        );
    let dir = project(&toml)?;
    match load(dir.path(), env(&[])) {
        Err(ConfigError::Invalid(problems)) => {
            assert!(
                problems.contains(
                    &"providers.nanogpt.api_key: must be set through env, not in overbrainer.toml"
                        .to_string()
                )
            );
            assert!(
                problems.contains(
                    &"providers.nanogpt.base_url: must be set through env, not in overbrainer.toml"
                        .to_string()
                )
            );
            assert!(problems.contains(
                &"hf_token: must be set through env, not in overbrainer.toml".to_string()
            ));
            let joined = problems.join("\n");
            assert!(
                !joined.contains("sk-leak") && !joined.contains("hf_leak"),
                "values must not leak"
            );
            Ok(())
        }
        other => Err(format!("expected Invalid, got {other:?}").into()),
    }
}

#[test]
fn unknown_env_variable_with_prefix_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
    let dir = project(BASE)?;
    assert!(matches!(
        load(dir.path(), env(&[("OVERBRAINER_TYPO", "1")])),
        Err(ConfigError::Parse(_))
    ));
    Ok(())
}

#[test]
fn missing_file_reports_its_path() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    match load(dir.path(), env(&[])) {
        Err(ConfigError::Read { path, .. }) => {
            assert!(path.ends_with("overbrainer.toml"));
            Ok(())
        }
        other => Err(format!("expected Read, got {other:?}").into()),
    }
}

#[test]
fn process_env_source_is_accepted() -> Result<(), Box<dyn std::error::Error>> {
    // Exercises the `EnvSource::Process` code path (what a real binary uses) without
    // depending on, or mutating, the actual process environment: `BASE` alone is a
    // complete, valid configuration, so this only has to type-check and run, not
    // assert on any particular outcome.
    let dir = project(BASE)?;
    let _ = load(dir.path(), EnvSource::Process);
    Ok(())
}
