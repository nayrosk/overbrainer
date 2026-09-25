//! Dynamic shell completion (`COMPLETE=<shell> overbrainer -- <words>`).
//!
//! Tests use fish: its protocol needs no index variable and prints `value\thelp`.

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

/// Runs a fish completion request for `words` (the command line after
/// `overbrainer`, the last word being the one completed) from `cwd`, and returns
/// stdout.
fn complete(cwd: &Path, words: &[&str]) -> Result<String, Box<dyn std::error::Error>> {
    let output = overbrainer()?
        .current_dir(cwd)
        .env("COMPLETE", "fish")
        .arg("--")
        .arg("overbrainer")
        .args(words)
        .output()?;
    assert!(output.status.success(), "completion failed: {output:?}");
    Ok(String::from_utf8(output.stdout)?)
}

#[test]
fn prints_a_registration_script_for_each_supported_shell() -> Result<(), Box<dyn std::error::Error>>
{
    for shell in ["bash", "zsh", "fish"] {
        overbrainer()?
            .env("COMPLETE", shell)
            .assert()
            .success()
            .stdout(predicate::str::contains("overbrainer"));
    }
    Ok(())
}

#[test]
fn refuses_an_unsupported_shell() -> Result<(), Box<dyn std::error::Error>> {
    overbrainer()?
        .env("COMPLETE", "elvish")
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown shell `elvish`"));
    Ok(())
}

#[test]
fn completes_subcommands() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let out = complete(dir.path(), &["tr"])?;
    assert!(out.lines().any(|l| l.starts_with("train\t")), "{out}");
    Ok(())
}

#[test]
fn completes_despite_a_malformed_env_file() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    std::fs::write(dir.path().join(".env"), "OVERBRAINER_HF_TOKEN=a b\"c\n")?;
    let out = complete(dir.path(), &["tr"])?;
    assert!(out.lines().any(|l| l.starts_with("train\t")), "{out}");
    Ok(())
}
