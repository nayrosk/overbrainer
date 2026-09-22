use assert_cmd::Command;
use predicates::prelude::*;

fn overbrainer() -> Result<Command, Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("overbrainer")?;
    // Isolate from the developer's environment.
    cmd.env_clear()
        .env("NO_COLOR", "1")
        .env("HOME", "/nonexistent");
    Ok(cmd)
}

#[test]
fn init_creates_project_files_and_refuses_to_overwrite() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    overbrainer()?
        .arg("init")
        .arg(dir.path())
        .assert()
        .success();
    for file in ["overbrainer.toml", ".env.example", ".gitignore"] {
        assert!(dir.path().join(file).is_file(), "{file} missing");
    }
    overbrainer()?
        .arg("init")
        .arg(dir.path())
        .assert()
        .failure()
        .stderr(predicate::str::contains("already exists"));
    Ok(())
}

#[test]
fn config_check_masks_secrets() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    overbrainer()?
        .arg("init")
        .arg(dir.path())
        .assert()
        .success();
    overbrainer()?
        .arg("-C")
        .arg(dir.path())
        .args(["config", "check"])
        .env(
            "OVERBRAINER_PROVIDERS__OPENROUTER__API_KEY",
            "sk-very-secret",
        )
        .env(
            "OVERBRAINER_PROVIDERS__OPENROUTER__BASE_URL",
            "https://openrouter.ai/api/v1",
        )
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "providers.openrouter.api_key = ***",
        ))
        .stdout(predicate::str::contains(
            "providers.openrouter.base_url = https://openrouter.ai/api/v1",
        ))
        .stdout(predicate::str::contains("sk-very-secret").not());
    Ok(())
}

#[test]
fn config_check_reports_invalid_config() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    overbrainer()?
        .arg("init")
        .arg(dir.path())
        .assert()
        .success();
    let path = dir.path().join("overbrainer.toml");
    let content = std::fs::read_to_string(&path)?.replace(
        "[providers.openrouter]",
        "[providers.openrouter]\napi_key = \"sk-leak\"",
    );
    std::fs::write(&path, content)?;
    overbrainer()?
        .arg("-C")
        .arg(dir.path())
        .args(["config", "check"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "providers.openrouter.api_key: must be set through env",
        ))
        .stderr(predicate::str::contains("sk-leak").not());
    Ok(())
}

#[test]
fn resolve_without_vault_fails_on_reference() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    overbrainer()?
        .arg("init")
        .arg(dir.path())
        .assert()
        .success();
    overbrainer()?
        .arg("-C")
        .arg(dir.path())
        .args(["config", "check", "--resolve"])
        .env(
            "OVERBRAINER_PROVIDERS__OPENROUTER__API_KEY",
            "vault:secret/overbrainer/openrouter#api_key",
        )
        .assert()
        .failure()
        .stderr(predicate::str::contains("VAULT_ADDR is not set"));
    Ok(())
}

#[test]
fn malformed_dotenv_does_not_leak_its_content() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    overbrainer()?
        .arg("init")
        .arg(dir.path())
        .assert()
        .success();
    std::fs::write(
        dir.path().join(".env"),
        "OVERBRAINER_HF_TOKEN=hf_marker_abc def\"x\n",
    )?;
    overbrainer()?
        .arg("-C")
        .arg(dir.path())
        .args(["config", "check"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cannot parse .env"))
        .stderr(predicate::str::contains("hf_marker").not())
        .stdout(predicate::str::contains("hf_marker").not());
    Ok(())
}
