//! `overbrainer train`, `train attach`, `train cancel` and `runs ls` on a local
//! target, with a fake `axolotl` in a virtual environment.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Output, Stdio};
use std::time::Duration;

use assert_cmd::Command;
use predicates::prelude::*;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const TOKEN: &str = "hf_cli_secret_17";

/// Fixed `PATH` for every spawned `overbrainer`, so the fake `axolotl` script's own
/// use of `mkdir` and `cat` resolves deterministically. Never the developer's PATH:
/// tests must not depend on the developer environment.
const PATH: &str = "/usr/bin:/bin:/usr/sbin:/sbin";

/// `mode` next to `bin/` picks the behavior of `axolotl train`: `ok` trains at
/// once, `fail` exits 1, and `slow` waits (at most a minute) for a `release` file
/// next to `bin/`, then trains.
const FAKE_AXOLOTL: &str = r#"#!/bin/sh
here="$(dirname "$0")/.."
mode=$(cat "$here/mode")
[ "$1" = train ] || exit 0
case "$mode" in
  fail) echo "boom" >&2; exit 1 ;;
  slow)
    i=0
    until [ -f "$here/release" ]; do
      i=$((i + 1))
      [ "$i" -gt 600 ] && exit 3
      sleep 0.1
    done
    ;;
esac
printf '{"event": "begin", "time": 1, "max_steps": 2}\n' >> "$OVERBRAINER_METRICS"
printf '{"event": "log", "time": 2, "step": 1, "epoch": 0.5, "max_steps": 2, "loss": 1.5}\n' >> "$OVERBRAINER_METRICS"
printf '{"event": "log", "time": 3, "step": 2, "epoch": 1.0, "max_steps": 2, "eval_loss": 1.25}\n' >> "$OVERBRAINER_METRICS"
mkdir -p output && echo adapter > output/adapter_model.safetensors
[ "$HF_TOKEN" = hf_cli_secret_17 ] && echo "token present"
exit 0
"#;

fn project(mode: &str) -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let venv = dir.path().join("venv");
    fs::create_dir_all(venv.join("bin"))?;
    fs::write(venv.join("bin/axolotl"), FAKE_AXOLOTL)?;
    fs::set_permissions(venv.join("bin/axolotl"), fs::Permissions::from_mode(0o755))?;
    fs::write(venv.join("mode"), mode)?;
    fs::create_dir_all(dir.path().join("data"))?;
    fs::write(dir.path().join("data/train.jsonl"), "{\"id\": 1}\n")?;
    fs::write(dir.path().join("data/eval.jsonl"), "{\"id\": 2}\n")?;
    fs::write(
        dir.path().join("overbrainer.toml"),
        format!(
            r#"[project]
name = "demo"

[providers.mock]
protocol = "openai"

[roles]
generator = {{ provider = "mock", model = "gen" }}
parent = {{ provider = "mock", model = "parent" }}

[training]
target = "here"
base_model = "Qwen/Qwen3-4B"
adapter = "qlora"

[targets.here]
kind = "local"
runtime = "native"
venv = "{}"

[targets.gpu]
kind = "runpod"
gpu_type = "NVIDIA A40"
image = "i"
max_hours = 1.0
"#,
            venv.display()
        ),
    )?;
    Ok(dir)
}

fn overbrainer(dir: &Path) -> Result<Command, Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("overbrainer")?;
    cmd.env_clear()
        .env("NO_COLOR", "1")
        .env("HOME", "/nonexistent")
        .env("PATH", PATH)
        .env("OVERBRAINER_HF_TOKEN", TOKEN)
        .arg("-C")
        .arg(dir);
    Ok(cmd)
}

fn only_run(dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let runs: Vec<PathBuf> = fs::read_dir(dir.join("runs"))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.join("run.json").is_file())
        .collect();
    match runs.as_slice() {
        [run] => Ok(run.clone()),
        other => Err(format!("expected one run, found {other:?}").into()),
    }
}

fn run_id(run: &Path) -> Result<String, Box<dyn std::error::Error>> {
    Ok(run
        .file_name()
        .ok_or("no run directory name")?
        .to_string_lossy()
        .into_owned())
}

/// Every file under `dir` whose content holds `needle`.
fn files_containing(dir: &Path, needle: &str) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut found = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            found.extend(files_containing(&path, needle)?);
        } else if String::from_utf8_lossy(&fs::read(&path)?).contains(needle) {
            found.push(path);
        }
    }
    Ok(found)
}

/// Starts `overbrainer train` in `dir` and sends it `SIGINT` once its run record
/// holds one of `states`, which it checks every `every`. Returns the run directory and the
/// output of the interrupted command.
fn interrupt_train(
    dir: &Path,
    states: &[&str],
    every: Duration,
) -> Result<(PathBuf, Output), Box<dyn std::error::Error>> {
    let binary = assert_cmd::cargo::cargo_bin("overbrainer");
    let mut train = std::process::Command::new(binary)
        .env_clear()
        .env("NO_COLOR", "1")
        .env("HOME", "/nonexistent")
        .env("PATH", PATH)
        .env("OVERBRAINER_HF_TOKEN", TOKEN)
        .arg("-C")
        .arg(dir)
        .arg("train")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let needles: Vec<String> = states
        .iter()
        .map(|state| format!("\"state\": \"{state}\""))
        .collect();
    let mut run = None;
    for _ in 0..10_000 {
        std::thread::sleep(every);
        if train.try_wait()?.is_some() {
            break;
        }
        if let Ok(found) = only_run(dir)
            && fs::read_to_string(found.join("run.json"))
                .is_ok_and(|record| needles.iter().any(|needle| record.contains(needle)))
        {
            run = Some(found);
            break;
        }
    }
    let killed = std::process::Command::new("kill")
        .args(["-INT", &train.id().to_string()])
        .status()?;
    let output = train.wait_with_output()?;
    let run = run.ok_or_else(|| {
        format!(
            "the run did not reach the awaited state: {}",
            String::from_utf8_lossy(&output.stderr)
        )
    })?;
    assert!(killed.success(), "kill failed");
    Ok((run, output))
}

/// Checks that an interrupted `train` failed with the attach command and left its
/// job running and recorded.
fn assert_detached(run: &Path, output: &Output) -> TestResult {
    let id = run_id(run)?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{stdout}\n{stderr}");
    assert!(
        stderr.contains(&format!(
            "follow it again with `overbrainer train attach {id}`"
        )),
        "{stderr}"
    );
    assert!(!stdout.contains(TOKEN) && !stderr.contains(TOKEN));
    assert!(
        !run.join("exit_code").exists(),
        "the job stopped with overbrainer"
    );
    let record = fs::read_to_string(run.join("run.json"))?;
    assert!(record.contains("\"state\": \"running\""), "{record}");
    assert!(record.contains("\"pid\""), "{record}");
    Ok(())
}

#[test]
fn train_runs_the_job_and_prints_a_summary() -> TestResult {
    let dir = project("ok")?;
    overbrainer(dir.path())?
        .arg("train")
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "succeeded; step 2/2, epoch 0.50, loss 1.5000, eval_loss 1.2500; output in runs/",
        ))
        .stdout(predicate::str::contains(TOKEN).not())
        .stderr(predicate::str::contains("train: step 1/2 (50%)"))
        .stderr(predicate::str::contains(TOKEN).not());
    let run = only_run(dir.path())?;
    let id = run_id(&run)?;
    assert_eq!(fs::read_to_string(run.join("job.log"))?, "token present\n");
    assert!(run.join("output/adapter_model.safetensors").is_file());
    assert!(
        fs::read_to_string(run.join("axolotl.yaml"))?
            .contains("overbrainer_metrics.OverbrainerMetricsPlugin")
    );
    assert_eq!(files_containing(&run, TOKEN)?, Vec::<PathBuf>::new());

    overbrainer(dir.path())?
        .args(["runs", "ls"])
        .assert()
        .success()
        .stdout(predicate::str::starts_with(format!(
            "{id}  succeeded  here  "
        )));
    overbrainer(dir.path())?
        .args(["train", "attach", &id])
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "train: run {id} succeeded; step 2/2"
        )));
    // A finished run still accepts a cancel, which stops anything left on the
    // target; its job wrote an exit code, so the cancel signals nothing.
    overbrainer(dir.path())?
        .args(["train", "cancel", &id])
        .assert()
        .success()
        .stderr(predicate::str::contains(format!(
            "run {id} already ended: succeeded; stopping any job left on the target"
        )))
        .stdout(predicate::str::contains(format!(
            "train: the job of run {id} had already ended (exited with code 0)"
        )));
    Ok(())
}

#[test]
fn cancel_stops_the_job_of_a_run_already_recorded_as_failed() -> TestResult {
    let dir = project("slow")?;
    let (run, output) = interrupt_train(dir.path(), &["running"], Duration::from_millis(100))?;
    assert_detached(&run, &output)?;
    let id = run_id(&run)?;
    // As if the job had been reported lost while it, or its container, kept
    // running: the record says failed, the job is still on the target.
    let record = fs::read_to_string(run.join("run.json"))?;
    fs::write(
        run.join("run.json"),
        record.replace("\"running\"", "\"failed\""),
    )?;

    overbrainer(dir.path())?
        .args(["train", "cancel", &id])
        .assert()
        .success()
        .stderr(predicate::str::contains(format!(
            "run {id} already ended: failed; stopping any job left on the target"
        )))
        .stdout(format!("train: run {id} cancelled\n"));
    // Only a record already cancelled is refused.
    overbrainer(dir.path())?
        .args(["train", "cancel", &id])
        .assert()
        .failure()
        .stderr(predicate::str::contains(format!(
            "run {id} already ended: cancelled"
        )));
    Ok(())
}

#[test]
fn a_failed_job_fails_the_command() -> TestResult {
    let dir = project("fail")?;
    overbrainer(dir.path())?
        .arg("train")
        .assert()
        .failure()
        .stdout(predicate::str::contains("failed; step 0"))
        .stderr(predicate::str::contains(
            "the job exited with code 1 (see runs/",
        ));
    Ok(())
}

#[test]
fn unusable_targets_are_refused() -> TestResult {
    let dir = project("ok")?;
    overbrainer(dir.path())?
        .args(["train", "--target", "nowhere"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown target `nowhere`"));
    overbrainer(dir.path())?
        .args(["train", "--target", "gpu"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cannot train on yet"))
        .stderr(predicate::str::contains("M4"));
    assert!(!dir.path().join("runs").exists());
    overbrainer(dir.path())?
        .args(["train", "attach", "20260101-000000-abcd"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("no run `20260101-000000-abcd`"));
    Ok(())
}

#[test]
fn ctrl_c_detaches_and_cancel_stops_the_job() -> TestResult {
    let dir = project("slow")?;
    let (run, output) = interrupt_train(dir.path(), &["running"], Duration::from_millis(100))?;
    assert_detached(&run, &output)?;
    let id = run_id(&run)?;

    overbrainer(dir.path())?
        .args(["train", "cancel", &id])
        .assert()
        .success()
        .stdout(format!("train: run {id} cancelled\n"));
    overbrainer(dir.path())?
        .args(["runs", "ls"])
        .assert()
        .success()
        .stdout(predicate::str::contains(format!("{id}  cancelled")));
    overbrainer(dir.path())?
        .args(["train", "cancel", &id])
        .assert()
        .failure()
        .stderr(predicate::str::contains(format!(
            "run {id} already ended: cancelled"
        )));
    Ok(())
}

#[test]
fn ctrl_c_while_the_job_starts_still_leaves_it_attachable() -> TestResult {
    let dir = project("slow")?;
    // SIGINT as soon as the run exists, most often while its job is starting.
    let (run, output) = interrupt_train(
        dir.path(),
        &["preparing", "running"],
        Duration::from_millis(1),
    )?;
    assert_detached(&run, &output)?;
    let id = run_id(&run)?;

    fs::write(dir.path().join("venv/release"), "")?;
    overbrainer(dir.path())?
        .args(["train", "attach", &id])
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "train: run {id} succeeded; step 2/2"
        )))
        .stderr(predicate::str::contains(TOKEN).not());
    assert!(run.join("output/adapter_model.safetensors").is_file());
    assert_eq!(fs::read_to_string(run.join("job.log"))?, "token present\n");
    Ok(())
}

#[test]
fn cancel_refuses_a_run_that_has_not_started() -> TestResult {
    let dir = project("ok")?;
    let id = "20260101-000000-abcd";
    let run = dir.path().join("runs").join(id);
    fs::create_dir_all(&run)?;
    fs::write(
        run.join("run.json"),
        format!(
            r#"{{"id": "{id}", "target": "here", "created": "2026-01-01T00:00:00Z",
"remote_dir": "/w/{id}", "job": null, "state": "preparing", "message": null}}"#
        ),
    )?;
    overbrainer(dir.path())?
        .args(["train", "cancel", id])
        .assert()
        .failure()
        .stderr(predicate::str::contains(format!(
            "run {id} has not started"
        )));
    Ok(())
}

#[test]
fn cancelling_a_job_that_already_ended_asks_to_attach() -> TestResult {
    let dir = project("ok")?;
    overbrainer(dir.path())?.arg("train").assert().success();
    let run = only_run(dir.path())?;
    let id = run_id(&run)?;
    // As if overbrainer had stopped following the run before its job ended.
    let record = fs::read_to_string(run.join("run.json"))?;
    fs::write(
        run.join("run.json"),
        record.replace("\"succeeded\"", "\"running\""),
    )?;

    overbrainer(dir.path())?
        .args(["train", "cancel", &id])
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "train: the job of run {id} had already ended (exited with code 0)"
        )))
        .stdout(predicate::str::contains(format!(
            "overbrainer train attach {id}"
        )));
    assert!(fs::read_to_string(run.join("run.json"))?.contains("\"running\""));
    overbrainer(dir.path())?
        .args(["train", "attach", &id])
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "train: run {id} succeeded"
        )));
    Ok(())
}

/// The `[training]` section written by [`project`].
const TRAINING: &str =
    "[training]\ntarget = \"here\"\nbase_model = \"Qwen/Qwen3-4B\"\nadapter = \"qlora\"\n";

/// Rewrites the `overbrainer.toml` of `dir` with `edit`.
fn edit_config(dir: &Path, edit: impl FnOnce(String) -> String) -> TestResult {
    let path = dir.join("overbrainer.toml");
    let config = fs::read_to_string(&path)?;
    fs::write(&path, edit(config))?;
    Ok(())
}

#[test]
fn a_hub_model_id_without_a_token_warns_and_still_trains() -> TestResult {
    let dir = project("ok")?;
    edit_config(dir.path(), |config| {
        config.replace(
            "adapter = \"qlora\"\n",
            "adapter = \"qlora\"\nhub_model_id = \"me/demo\"\n",
        )
    })?;
    overbrainer(dir.path())?
        .env_remove("OVERBRAINER_HF_TOKEN")
        .arg("train")
        .assert()
        .success()
        .stderr(predicate::str::contains(
            "training.hub_model_id is set but OVERBRAINER_HF_TOKEN is not: the push will fail",
        ))
        .stdout(predicate::str::contains("succeeded; step 2/2"));
    let run = only_run(dir.path())?;
    assert_eq!(fs::read_to_string(run.join("job.log"))?, "");
    Ok(())
}

#[test]
fn cancel_explains_that_it_needs_the_training_section() -> TestResult {
    let dir = project("slow")?;
    let (run, output) = interrupt_train(dir.path(), &["running"], Duration::from_millis(100))?;
    assert_detached(&run, &output)?;
    let id = run_id(&run)?;
    let config = fs::read_to_string(dir.path().join("overbrainer.toml"))?;
    edit_config(dir.path(), |config| {
        assert!(config.contains(TRAINING));
        config.replace(TRAINING, "")
    })?;
    overbrainer(dir.path())?
        .args(["train", "cancel", &id])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "cancel needs [training] to retrieve the run's artifacts",
        ));
    fs::write(dir.path().join("overbrainer.toml"), &config)?;
    overbrainer(dir.path())?
        .args(["train", "cancel", &id])
        .assert()
        .success()
        .stdout(format!("train: run {id} cancelled\n"));

    // The record is checked first: a run that ended needs no [training] to say so.
    edit_config(dir.path(), |_| config.replace(TRAINING, ""))?;
    overbrainer(dir.path())?
        .args(["train", "cancel", &id])
        .assert()
        .failure()
        .stderr(predicate::str::contains(format!(
            "run {id} already ended: cancelled"
        )));
    Ok(())
}
