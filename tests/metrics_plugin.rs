//! Runs the embedded Axolotl metrics plugin under `python3` with stub `axolotl` and
//! `transformers` modules, and parses what it writes. Skipped when `python3` is not
//! installed.

use std::fs;
use std::path::Path;
use std::process::Command;

use overbrainer::train::{METRICS_PLUGIN, Mark, MetricLine, PLUGIN_FILE, TrainMetric, parse_line};

type TestResult = Result<(), Box<dyn std::error::Error>>;

const DRIVER: &str = r#"
from types import SimpleNamespace
from overbrainer_metrics import OverbrainerMetricsPlugin

callbacks = OverbrainerMetricsPlugin().add_callbacks_pre_trainer(cfg=None, model=None)
assert len(callbacks) == 1
callback = callbacks[0]
main = SimpleNamespace(is_world_process_zero=True, max_steps=12, global_step=0, epoch=None)
other = SimpleNamespace(is_world_process_zero=False, max_steps=12, global_step=3, epoch=0.25)
callback.on_train_begin(None, main, None)
main.global_step = 3
main.epoch = 0.25
callback.on_log(None, main, None, logs={"loss": 1.5, "learning_rate": 2e-4, "grad_norm": float("nan"), "epoch": 0.25})
callback.on_log(None, other, None, logs={"loss": 9.0})
callback.on_log(None, main, None, logs={"eval_loss": 1.75, "eval_runtime": 3.0})
"#;

const BEGIN_DRIVER: &str = r"
from types import SimpleNamespace
from overbrainer_metrics import OverbrainerMetricsPlugin

callbacks = OverbrainerMetricsPlugin().add_callbacks_pre_trainer(cfg=None, model=None)
assert len(callbacks) == 1
callback = callbacks[0]
main = SimpleNamespace(is_world_process_zero=True, max_steps=1, global_step=0, epoch=None)
callback.on_train_begin(None, main, None)
";

/// Drives the snapshot callback alone (no `OVERBRAINER_METRICS`): nothing at a
/// step without a request, nothing for an ordinary save, then the request makes
/// the step save and stop, and the save writes the proof.
const SNAPSHOT_DRIVER: &str = r#"
import os
from types import SimpleNamespace
from overbrainer_metrics import OverbrainerMetricsPlugin

callbacks = OverbrainerMetricsPlugin().add_callbacks_pre_trainer(cfg=None, model=None)
assert len(callbacks) == 1
callback = callbacks[0]
request = os.environ["OVERBRAINER_SNAPSHOT"]
root = os.path.dirname(request)
proof = os.path.join(root, "snapshot.json")
rank0 = os.environ.get("RANK0", "1") == "1"
args = SimpleNamespace(output_dir=os.path.join(root, "output"), device="cpu")
state = SimpleNamespace(is_world_process_zero=rank0, global_step=2)
control = SimpleNamespace(should_save=False, should_training_stop=False)
callback.on_step_end(args, state, control)
assert not control.should_save and not control.should_training_stop
callback.on_save(args, state, control)
assert not os.path.exists(proof)
with open(request, "w") as file:
    file.write(os.environ.get("REASON", ""))
state.global_step = 3
callback.on_step_end(args, state, control)
print("stopped" if control.should_save and control.should_training_stop else "running")
callback.on_save(args, state, control)
"#;

/// A trainer loop in the order of transformers' `Trainer`: after each step,
/// `on_step_end`, then a save when `should_save` (its checkpoint first, then
/// `on_save`), then a stop when `should_training_stop`. The request appears
/// before the step `REQUEST_AT`; prints the step training stopped at and the
/// checkpoints saved.
const LOOP_DRIVER: &str = r#"
import os
from types import SimpleNamespace
from overbrainer_metrics import OverbrainerMetricsPlugin

callback = OverbrainerMetricsPlugin().add_callbacks_pre_trainer(cfg=None, model=None)[0]
request = os.environ["OVERBRAINER_SNAPSHOT"]
root = os.path.dirname(request)
args = SimpleNamespace(output_dir=os.path.join(root, "output"), device="cpu")
max_steps = 5
state = SimpleNamespace(is_world_process_zero=True, global_step=0, max_steps=max_steps)
saved = []
for step in range(1, max_steps + 1):
    if step == int(os.environ["REQUEST_AT"]):
        open(request, "w").close()
    state.global_step = step
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    callback.on_step_end(args, state, control)
    if control.should_save:
        os.makedirs(os.path.join(args.output_dir, f"checkpoint-{step}"))
        saved.append(step)
        callback.on_save(args, state, control)
    if control.should_training_stop:
        break
print(state.global_step, saved)
"#;

fn write(path: &Path, content: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(path, content)
}

/// Stub `axolotl` and `transformers` modules, and the real plugin, under `dir`.
/// Returns the `PYTHONPATH` entries to run it with.
fn stage_plugin(dir: &std::path::Path) -> std::io::Result<String> {
    let stubs = dir.join("stubs");
    write(
        &stubs.join("transformers/__init__.py"),
        "class TrainerCallback:\n    pass\n",
    )?;
    write(&stubs.join("axolotl/__init__.py"), "")?;
    write(&stubs.join("axolotl/integrations/__init__.py"), "")?;
    write(
        &stubs.join("axolotl/integrations/base.py"),
        "class BasePlugin:\n    pass\n",
    )?;
    // A stand-in for `torch.distributed`: initialized with `STUB_DIST=1`, when
    // its broadcast hands every rank what rank 0 finds, the request once it
    // exists with `STUB_FOUND=1`, never without.
    write(
        &stubs.join("torch/__init__.py"),
        "class _Tensor:\n    def __init__(self, values):\n        self.values = values\n\n    \
         def item(self):\n        return self.values[0]\n\n\n\
         def tensor(values, device=None):\n    return _Tensor(list(values))\n",
    )?;
    write(
        &stubs.join("torch/distributed.py"),
        "import os\n\n\ndef is_available():\n    return True\n\n\n\
         def is_initialized():\n    return os.environ.get(\"STUB_DIST\") == \"1\"\n\n\n\
         def broadcast(flag, src):\n    found = os.environ.get(\"STUB_FOUND\") == \"1\"\n    \
         flag.values[0] = int(found and os.path.exists(os.environ[\"OVERBRAINER_SNAPSHOT\"]))\n",
    )?;
    let plugin = dir.join("plugin");
    write(&plugin.join(PLUGIN_FILE), METRICS_PLUGIN)?;
    Ok(format!("{}:{}", plugin.display(), stubs.display()))
}

#[test]
fn the_plugin_writes_parseable_lines_from_the_main_process_only() -> TestResult {
    if Command::new("python3").arg("--version").output().is_err() {
        eprintln!("skipped: python3 is not installed");
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let python_path = stage_plugin(dir.path())?;
    let metrics = dir.path().join("metrics.jsonl");

    let output = Command::new("python3")
        .arg("-c")
        .arg(DRIVER)
        .env("PYTHONPATH", python_path)
        .env("OVERBRAINER_METRICS", &metrics)
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let lines: Vec<MetricLine> = fs::read_to_string(&metrics)?
        .lines()
        .map(parse_line)
        .collect::<Result<_, _>>()?;
    assert_eq!(lines.len(), 3);
    assert!(matches!(
        lines[0],
        MetricLine::Begin {
            max_steps: Some(12),
            ..
        }
    ));
    let MetricLine::Log(train) = &lines[1] else {
        return Err("expected a log line".into());
    };
    assert_eq!(
        (train.step, train.epoch, train.loss, train.grad_norm),
        (3, Some(0.25), Some(1.5), None)
    );
    assert_eq!(train.learning_rate, Some(2e-4));
    let MetricLine::Log(TrainMetric {
        eval_loss, loss, ..
    }) = &lines[2]
    else {
        return Err("expected a log line".into());
    };
    assert_eq!((*eval_loss, *loss), (Some(1.75), None));
    Ok(())
}

#[test]
fn the_plugin_creates_a_missing_metrics_directory() -> TestResult {
    if Command::new("python3").arg("--version").output().is_err() {
        eprintln!("skipped: python3 is not installed");
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let python_path = stage_plugin(dir.path())?;
    // Neither `runs` nor `r1` exists yet: the plugin must create them itself.
    let metrics = dir.path().join("runs/r1/metrics.jsonl");

    let output = Command::new("python3")
        .arg("-c")
        .arg(BEGIN_DRIVER)
        .env("PYTHONPATH", python_path)
        .env("OVERBRAINER_METRICS", &metrics)
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(metrics.is_file());
    Ok(())
}

/// Runs [`SNAPSHOT_DRIVER`] with `env`; returns what it printed and the proof
/// it left, if any.
fn drive_snapshot(
    env: &[(&str, &str)],
) -> Result<Option<(String, serde_json::Value)>, Box<dyn std::error::Error>> {
    if Command::new("python3").arg("--version").output().is_err() {
        eprintln!("skipped: python3 is not installed");
        return Ok(None);
    }
    let dir = tempfile::tempdir()?;
    let python_path = stage_plugin(dir.path())?;
    let root = dir.path().join("run");
    fs::create_dir_all(&root)?;
    let output = Command::new("python3")
        .arg("-c")
        .arg(SNAPSHOT_DRIVER)
        .env("PYTHONPATH", python_path)
        .env_remove("OVERBRAINER_METRICS")
        .env("OVERBRAINER_SNAPSHOT", root.join("snapshot.request"))
        .envs(env.iter().copied())
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let printed = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let proof = match fs::read_to_string(root.join("snapshot.json")) {
        Ok(text) => serde_json::from_str(&text)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => serde_json::Value::Null,
        Err(error) => return Err(error.into()),
    };
    assert!(!root.join("snapshot.json.tmp").exists());
    Ok(Some((printed, proof)))
}

#[test]
fn a_request_saves_a_checkpoint_stops_and_leaves_its_proof() -> TestResult {
    let Some((printed, proof)) = drive_snapshot(&[("REASON", "deadline\n")])? else {
        return Ok(());
    };
    assert_eq!(printed, "stopped");
    assert_eq!(proof["checkpoint"], "output/checkpoint-3");
    assert_eq!(proof["step"], 3);
    assert_eq!(proof["reason"], "deadline");
    assert!(proof["time"].as_f64().is_some_and(|time| time > 0.0));
    Ok(())
}

#[test]
fn a_request_without_a_known_reason_reads_as_requested() -> TestResult {
    for reason in ["", "shutdown"] {
        let Some((_, proof)) = drive_snapshot(&[("REASON", reason)])? else {
            return Ok(());
        };
        assert_eq!(proof["reason"], "requested", "{reason:?}");
    }
    Ok(())
}

#[test]
fn every_rank_stops_with_rank_0_and_only_rank_0_writes_the_proof() -> TestResult {
    // Another rank never looks at the file: the broadcast tells it.
    let Some((printed, proof)) =
        drive_snapshot(&[("RANK0", "0"), ("STUB_DIST", "1"), ("STUB_FOUND", "1")])?
    else {
        return Ok(());
    };
    assert_eq!(printed, "stopped");
    assert!(proof.is_null(), "{proof}");
    let Some((printed, _)) =
        drive_snapshot(&[("RANK0", "0"), ("STUB_DIST", "1"), ("STUB_FOUND", "0")])?
    else {
        return Ok(());
    };
    assert_eq!(printed, "running");
    Ok(())
}

/// Runs [`LOOP_DRIVER`] with the request before step `at`; what it printed and
/// the proof it left (`Null` when none).
fn drive_loop(at: u32) -> Result<Option<(String, serde_json::Value)>, Box<dyn std::error::Error>> {
    if Command::new("python3").arg("--version").output().is_err() {
        eprintln!("skipped: python3 is not installed");
        return Ok(None);
    }
    let dir = tempfile::tempdir()?;
    let python_path = stage_plugin(dir.path())?;
    let root = dir.path().join("run");
    fs::create_dir_all(&root)?;
    let output = Command::new("python3")
        .arg("-c")
        .arg(LOOP_DRIVER)
        .env("PYTHONPATH", python_path)
        .env_remove("OVERBRAINER_METRICS")
        .env("OVERBRAINER_SNAPSHOT", root.join("snapshot.request"))
        .env("REQUEST_AT", at.to_string())
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let printed = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let proof = match fs::read_to_string(root.join("snapshot.json")) {
        Ok(text) => serde_json::from_str(&text)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => serde_json::Value::Null,
        Err(error) => return Err(error.into()),
    };
    Ok(Some((printed, proof)))
}

#[test]
fn in_the_trainer_s_order_the_checkpoint_lands_before_the_stop() -> TestResult {
    let Some((printed, proof)) = drive_loop(3)? else {
        return Ok(());
    };
    assert_eq!(printed, "3 [3]");
    assert_eq!(proof["checkpoint"], "output/checkpoint-3");
    assert_eq!(proof["step"], 3);
    Ok(())
}

#[test]
fn a_request_at_the_last_step_lets_the_run_end_as_usual() -> TestResult {
    let Some((printed, proof)) = drive_loop(5)? else {
        return Ok(());
    };
    assert_eq!(printed, "5 []");
    assert!(proof.is_null(), "{proof}");
    Ok(())
}

/// Drives the metrics callback through two evaluations on a fake clock: the
/// first prediction step of each is written at once, the next ones at most
/// every 2 seconds, with the loader's length as total when it has one; a
/// loader whose length fails, another rank, or a broken file never raise.
const EVAL_DRIVER: &str = r#"
import overbrainer_metrics
from types import SimpleNamespace

clock = [100.0]
overbrainer_metrics.time = SimpleNamespace(time=lambda: clock[0])

class Iterable:
    def __iter__(self):
        return iter(())

class Broken:
    def __len__(self):
        raise TypeError("no length")

callback = overbrainer_metrics.OverbrainerMetricsPlugin().add_callbacks_pre_trainer(cfg=None, model=None)[0]
main = SimpleNamespace(is_world_process_zero=True, max_steps=10, global_step=10, epoch=1.0)
other = SimpleNamespace(is_world_process_zero=False, max_steps=10, global_step=10, epoch=1.0)
loader = [0] * 1200
for tick in (0.0, 1.0, 1.5, 2.5, 3.0):
    clock[0] = 100.0 + tick
    callback.on_prediction_step(None, main, None, eval_dataloader=loader)
    callback.on_prediction_step(None, other, None, eval_dataloader=loader)
callback.on_evaluate(None, main, None)
clock[0] = 104.0
callback.on_prediction_step(None, main, None, eval_dataloader=Iterable())
callback.on_evaluate(None, main, None)
callback.on_prediction_step(None, main, None, eval_dataloader=Broken())
callback.on_prediction_step(None, main, None)
callback.on_train_end(None, other, None)
callback.on_train_end(None, main, None)
callback.path = "/nonexistent/dir/metrics.jsonl"
callback.on_evaluate(None, main, None)
callback.on_prediction_step(None, main, None, eval_dataloader=loader)
callback.on_train_end(None, main, None)
"#;

#[test]
fn evaluations_report_their_progress_and_the_end_of_training_is_said() -> TestResult {
    if Command::new("python3").arg("--version").output().is_err() {
        eprintln!("skipped: python3 is not installed");
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let python_path = stage_plugin(dir.path())?;
    let metrics = dir.path().join("metrics.jsonl");
    let output = Command::new("python3")
        .arg("-c")
        .arg(EVAL_DRIVER)
        .env("PYTHONPATH", python_path)
        .env("OVERBRAINER_METRICS", &metrics)
        .env_remove("OVERBRAINER_SNAPSHOT")
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let marks: Vec<Option<Mark>> = fs::read_to_string(&metrics)?
        .lines()
        .map(|line| parse_line(line).map(|line| line.mark()))
        .collect::<Result<_, _>>()?;
    let eval = |step, total| Some(Mark::Eval { step, total });
    assert_eq!(
        marks,
        [
            eval(1, Some(1200)),
            eval(4, Some(1200)),
            eval(1, None),
            eval(1, None),
            Some(Mark::End { step: 10 }),
        ]
    );
    Ok(())
}
