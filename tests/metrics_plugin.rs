//! Runs the embedded Axolotl metrics plugin under `python3` with stub `axolotl` and
//! `transformers` modules, and parses what it writes. Skipped when `python3` is not
//! installed.

use std::fs;
use std::path::Path;
use std::process::Command;

use overbrainer::train::{METRICS_PLUGIN, MetricLine, PLUGIN_FILE, TrainMetric, parse_line};

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

fn write(path: &Path, content: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(path, content)
}

#[test]
fn the_plugin_writes_parseable_lines_from_the_main_process_only() -> TestResult {
    if Command::new("python3").arg("--version").output().is_err() {
        eprintln!("skipped: python3 is not installed");
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let stubs = dir.path().join("stubs");
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
    let plugin = dir.path().join("plugin");
    write(&plugin.join(PLUGIN_FILE), METRICS_PLUGIN)?;
    let metrics = dir.path().join("metrics.jsonl");

    let output = Command::new("python3")
        .arg("-c")
        .arg(DRIVER)
        .env(
            "PYTHONPATH",
            format!("{}:{}", plugin.display(), stubs.display()),
        )
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
