//! Dynamic shell completion (`COMPLETE=<shell> overbrainer -- <words>`).
//!
//! Tests use fish: its protocol needs no index variable and prints `value\thelp`.

use std::path::Path;

use assert_cmd::Command;
use overbrainer::runs::{RunRecord, RunState, Runs};
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

fn save_run(
    project: &Path,
    id: &str,
    state: RunState,
    target: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    Runs::new(project).save(&RunRecord {
        id: id.to_string(),
        target: target.to_string(),
        created: "2026-09-22T14:30:05Z".to_string(),
        remote_dir: "/tmp/run".to_string(),
        job: None,
        state,
        message: None,
    })?;
    Ok(())
}

#[test]
fn completes_run_ids_with_their_state_and_target() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    save_run(
        dir.path(),
        "20260922-143005-a1b2",
        RunState::Succeeded,
        "local",
    )?;
    save_run(
        dir.path(),
        "20260923-090000-ffff",
        RunState::Running,
        "gpu_cloud",
    )?;
    for words in [
        ["train", "attach", ""].as_slice(),
        &["train", "cancel", ""],
        &["pod", "rm", ""],
    ] {
        let out = complete(dir.path(), words)?;
        assert!(
            out.contains("20260922-143005-a1b2\tsucceeded, local"),
            "{words:?}: {out}"
        );
        assert!(
            out.contains("20260923-090000-ffff\trunning, gpu_cloud"),
            "{words:?}: {out}"
        );
    }
    let out = complete(dir.path(), &["train", "attach", "20260923"])?;
    assert!(!out.contains("20260922-143005-a1b2"), "{out}");
    assert!(out.contains("20260923-090000-ffff"), "{out}");
    Ok(())
}

#[test]
fn completes_run_ids_of_the_project_named_by_dash_c() -> Result<(), Box<dyn std::error::Error>> {
    let cwd = tempfile::tempdir()?;
    let project = tempfile::tempdir()?;
    save_run(
        project.path(),
        "20260922-143005-a1b2",
        RunState::Failed,
        "local",
    )?;
    let dir = project.path().to_str().ok_or("temp dir is not UTF-8")?;
    let out = complete(cwd.path(), &["-C", dir, "train", "attach", ""])?;
    assert!(out.contains("20260922-143005-a1b2\tfailed, local"), "{out}");
    let out = complete(cwd.path(), &["train", "attach", ""])?;
    assert!(!out.contains("20260922-143005-a1b2"), "{out}");
    Ok(())
}

#[test]
fn completes_no_run_id_outside_a_project() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let out = complete(dir.path(), &["train", "attach", ""])?;
    assert!(!out.lines().any(|l| l.starts_with("2026")), "{out}");
    Ok(())
}

#[test]
fn completes_topics_of_every_stage() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    overbrainer()?
        .arg("init")
        .arg(dir.path())
        .assert()
        .success();
    for stage in ["subtopics", "questions", "answers", "split"] {
        let out = complete(dir.path(), &[stage, "--topic", ""])?;
        assert!(out.lines().any(|l| l == "ownership"), "{stage}: {out}");
    }
    Ok(())
}

#[test]
fn completes_training_targets() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    overbrainer()?
        .arg("init")
        .arg(dir.path())
        .assert()
        .success();
    let out = complete(dir.path(), &["train", "--target", ""])?;
    assert!(out.lines().any(|l| l == "local"), "{out}");
    Ok(())
}

#[test]
fn completes_directories_for_the_project_flag_and_init() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    std::fs::create_dir(dir.path().join("proj"))?;
    std::fs::write(dir.path().join("prose.txt"), "")?;
    for words in [["-C", "pr"].as_slice(), &["init", "pr"]] {
        let out = complete(dir.path(), words)?;
        assert!(out.contains("proj"), "{words:?}: {out}");
        assert!(!out.contains("prose.txt"), "{words:?}: {out}");
    }
    Ok(())
}
