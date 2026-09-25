//! `overbrainer skill install`.

use std::path::Path;

use assert_cmd::Command;
use predicates::prelude::*;

fn overbrainer() -> Result<Command, Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("overbrainer")?;
    cmd.env_clear()
        .env("NO_COLOR", "1")
        .env("HOME", "/nonexistent");
    Ok(cmd)
}

fn skill_file(base: &Path) -> std::path::PathBuf {
    base.join("overbrainer").join("SKILL.md")
}

#[test]
fn installs_into_the_project_then_reports_it_up_to_date() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = tempfile::tempdir()?;
    let path = skill_file(&dir.path().join(".claude/skills"));
    overbrainer()?
        .arg("-C")
        .arg(dir.path())
        .args(["skill", "install"])
        .assert()
        .success()
        .stdout(predicate::str::contains("installed"));
    let content = std::fs::read_to_string(&path)?;
    assert!(content.starts_with("---\nname: overbrainer\n"));
    overbrainer()?
        .arg("-C")
        .arg(dir.path())
        .args(["skill", "install"])
        .assert()
        .success()
        .stdout(predicate::str::contains("is up to date"));
    Ok(())
}

#[test]
fn refuses_to_replace_a_modified_skill_without_force() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let base = dir.path().join("skills");
    let path = skill_file(&base);
    std::fs::create_dir_all(path.parent().ok_or("no parent")?)?;
    std::fs::write(&path, "my own notes\n")?;
    overbrainer()?
        .args(["skill", "install", "--dir"])
        .arg(&base)
        .assert()
        .failure()
        .stderr(predicate::str::contains("use --force to replace it"));
    assert_eq!(std::fs::read_to_string(&path)?, "my own notes\n");
    overbrainer()?
        .args(["skill", "install", "--force", "--dir"])
        .arg(&base)
        .assert()
        .success()
        .stdout(predicate::str::contains("replaced"));
    assert!(std::fs::read_to_string(&path)?.starts_with("---\nname: overbrainer\n"));
    Ok(())
}

#[test]
fn installs_globally_under_home() -> Result<(), Box<dyn std::error::Error>> {
    let home = tempfile::tempdir()?;
    overbrainer()?
        .env("HOME", home.path())
        .args(["skill", "install", "--global"])
        .assert()
        .success();
    assert!(skill_file(&home.path().join(".claude/skills")).is_file());
    Ok(())
}

#[test]
fn global_needs_home() -> Result<(), Box<dyn std::error::Error>> {
    // Run from a temp dir: a regression would write .claude/skills there.
    let cwd = tempfile::tempdir()?;
    overbrainer()?
        .current_dir(cwd.path())
        .env_remove("HOME")
        .args(["skill", "install", "--global"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("HOME"));
    overbrainer()?
        .current_dir(cwd.path())
        .env("HOME", "")
        .args(["skill", "install", "--global"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("HOME"));
    assert!(!cwd.path().join(".claude").exists());
    Ok(())
}

#[test]
fn global_and_dir_conflict() -> Result<(), Box<dyn std::error::Error>> {
    overbrainer()?
        .args(["skill", "install", "--global", "--dir", "x"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cannot be used with"));
    Ok(())
}

#[test]
fn ignores_a_malformed_env_file() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    std::fs::write(dir.path().join(".env"), "OVERBRAINER_HF_TOKEN=a b\"c\n")?;
    overbrainer()?
        .arg("-C")
        .arg(dir.path())
        .args(["skill", "install"])
        .assert()
        .success();
    Ok(())
}
