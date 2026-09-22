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

#[test]
fn dependency_logs_stay_quiet_by_default() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    overbrainer()?
        .arg("init")
        .arg(dir.path())
        .assert()
        .success();
    // Reserve a free port, then close it so the Vault request is refused.
    let port = std::net::TcpListener::bind("127.0.0.1:0")?
        .local_addr()?
        .port();
    overbrainer()?
        .arg("-C")
        .arg(dir.path())
        .args(["config", "check", "--resolve"])
        .env("VAULT_ADDR", format!("http://127.0.0.1:{port}"))
        .env("VAULT_TOKEN", "test-token")
        .env(
            "OVERBRAINER_PROVIDERS__OPENROUTER__API_KEY",
            "vault:secret/overbrainer/openrouter#api_key",
        )
        .assert()
        .failure()
        .stderr(predicate::str::contains("error: cannot resolve"))
        .stderr(predicate::str::contains("ERROR").not());
    Ok(())
}

#[test]
fn init_without_dir_uses_the_project_dir() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let project = dir.path().join("demo");
    overbrainer()?
        .current_dir(dir.path())
        .arg("-C")
        .arg(&project)
        .arg("init")
        .assert()
        .success();
    for file in ["overbrainer.toml", ".env.example", ".gitignore"] {
        assert!(project.join(file).is_file(), "{file} missing");
        assert!(
            !dir.path().join(file).exists(),
            "{file} written to the current dir"
        );
    }
    Ok(())
}

#[test]
fn init_appends_missing_gitignore_entries_once() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let gitignore = dir.path().join(".gitignore");
    // No trailing newline, and one entry already present.
    std::fs::write(&gitignore, "target/\n.env")?;
    overbrainer()?
        .arg("init")
        .arg(dir.path())
        .assert()
        .success();
    let content = std::fs::read_to_string(&gitignore)?;
    assert_eq!(content, "target/\n.env\n/data/\n/runs/\n");

    // A second init (after removing the non-appendable files) adds nothing.
    std::fs::remove_file(dir.path().join("overbrainer.toml"))?;
    std::fs::remove_file(dir.path().join(".env.example"))?;
    overbrainer()?
        .arg("init")
        .arg(dir.path())
        .assert()
        .success();
    assert_eq!(std::fs::read_to_string(&gitignore)?, content);
    Ok(())
}

#[test]
fn init_refusal_writes_nothing() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    std::fs::write(dir.path().join(".env.example"), "keep me\n")?;
    std::fs::write(dir.path().join(".gitignore"), "target/\n")?;
    overbrainer()?
        .arg("init")
        .arg(dir.path())
        .assert()
        .failure()
        .stderr(predicate::str::contains(".env.example already exists"));
    assert!(!dir.path().join("overbrainer.toml").exists());
    assert_eq!(
        std::fs::read_to_string(dir.path().join(".env.example"))?,
        "keep me\n"
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join(".gitignore"))?,
        "target/\n"
    );
    Ok(())
}
