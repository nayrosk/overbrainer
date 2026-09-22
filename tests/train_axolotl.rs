use std::fs;
use std::process::Command;

use overbrainer::config::{EnvSource, Settings, Training, load};
use overbrainer::dataset::DataFiles;
use overbrainer::train::{
    Artifacts, Axolotl, TrainError, Trainer, reasoning_template_warning, to_yaml,
};
use serde_json::Value;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const PROJECT: &str = r#"
[project]
name = "demo"

[providers.mock]
protocol = "openai"

[roles]
generator = { provider = "mock", model = "gen" }
parent = { provider = "mock", model = "parent" }

[targets.box]
kind = "local"
runtime = "native"
"#;

const QLORA: &str = r#"
[training]
target = "box"
base_model = "Qwen/Qwen3-4B"
adapter = "qlora"
hub_model_id = "me/qwen3-rust"
merge = true

[training.axolotl_extra]
chat_template = "qwen3"
special_tokens = { pad_token = "<|endoftext|>" }
"#;

fn settings(training: &str) -> Result<(tempfile::TempDir, Settings), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    fs::write(
        dir.path().join("overbrainer.toml"),
        format!("{PROJECT}{training}"),
    )?;
    let settings = load(dir.path(), EnvSource::Vars(Vec::new()))?;
    Ok((dir, settings))
}

fn training(settings: &Settings) -> Result<&Training, Box<dyn std::error::Error>> {
    Ok(settings.training.as_ref().ok_or("training missing")?)
}

const QLORA_YAML: &str = r#"adapter: "qlora"
attn_implementation: "sdpa"
base_model: "Qwen/Qwen3-4B"
chat_template: "qwen3"
dataset_prepared_path: "/workspace/run/prepared"
datasets:
  - ds_type: "json"
    field_messages: "messages"
    field_thinking: "reasoning_content"
    path: "/workspace/run/data/train.jsonl"
    roles_to_train:
      - "assistant"
    train_on_eos: "turn"
    type: "chat_template"
eval_sample_packing: false
evals_per_epoch: 4
gradient_accumulation_steps: 4
gradient_checkpointing: true
hub_model_id: "me/qwen3-rust"
learning_rate: 0.0002
load_in_4bit: true
load_in_8bit: false
logging_steps: 1
lora_alpha: 32
lora_dropout: 0.05
lora_r: 16
lora_target_linear: true
lr_scheduler: "cosine"
micro_batch_size: 2
num_epochs: 3
optimizer: "adamw_torch_fused"
output_dir: "/workspace/run/output"
plugins:
  - "overbrainer_metrics.OverbrainerMetricsPlugin"
sample_packing: true
saves_per_epoch: 1
sequence_len: 4096
special_tokens:
  pad_token: "<|endoftext|>"
test_datasets:
  - ds_type: "json"
    field_messages: "messages"
    field_thinking: "reasoning_content"
    path: "/workspace/run/data/eval.jsonl"
    roles_to_train:
      - "assistant"
    split: "train"
    train_on_eos: "turn"
    type: "chat_template"
val_set_size: 0
warmup_ratio: 0.1
"#;

#[test]
fn qlora_config_renders_the_documented_yaml() -> TestResult {
    let (dir, settings) = settings(QLORA)?;
    let trainer = Axolotl::new(training(&settings)?, &DataFiles::new(dir.path()));
    assert_eq!(to_yaml(&trainer.config("/workspace/run", true)), QLORA_YAML);
    Ok(())
}

#[test]
fn full_fine_tuning_has_no_adapter_keys_and_no_eval_without_eval_data() -> TestResult {
    let (dir, settings) =
        settings("[training]\ntarget = \"box\"\nbase_model = \"m\"\nadapter = \"full\"\n")?;
    let trainer = Axolotl::new(training(&settings)?, &DataFiles::new(dir.path()));
    let config = trainer.config("/r", false);
    for key in [
        "adapter",
        "load_in_4bit",
        "lora_r",
        "test_datasets",
        "evals_per_epoch",
    ] {
        assert!(config.get(key).is_none(), "{key} present");
    }
    assert_eq!(config["saves_per_epoch"], 1);
    assert_eq!(
        trainer.commands(),
        vec![vec![
            "axolotl".to_string(),
            "train".into(),
            "axolotl.yaml".into()
        ]]
    );
    Ok(())
}

#[test]
fn step_cadence_in_axolotl_extra_replaces_the_per_epoch_one() -> TestResult {
    let (dir, settings) = settings(
        "[training]\ntarget = \"box\"\nbase_model = \"m\"\nadapter = \"lora\"\n[training.axolotl_extra]\neval_steps = 50\nsave_steps = 100\n",
    )?;
    let trainer = Axolotl::new(training(&settings)?, &DataFiles::new(dir.path()));
    let config = trainer.config("/r", true);
    assert!(config.get("evals_per_epoch").is_none());
    assert!(config.get("saves_per_epoch").is_none());
    assert_eq!(
        (&config["eval_steps"], &config["save_steps"]),
        (&Value::from(50), &Value::from(100))
    );
    assert_eq!(config["load_in_4bit"], false);
    Ok(())
}

#[test]
fn commands_env_and_artifacts() -> TestResult {
    let (dir, settings) = settings(QLORA)?;
    let trainer = Axolotl::new(training(&settings)?, &DataFiles::new(dir.path()));
    assert_eq!(
        trainer.commands(),
        vec![
            vec!["axolotl".to_string(), "train".into(), "axolotl.yaml".into()],
            vec![
                "axolotl".to_string(),
                "merge-lora".into(),
                "axolotl.yaml".into()
            ],
        ]
    );
    assert_eq!(
        trainer.env("/r"),
        vec![
            ("AXOLOTL_DO_NOT_TRACK".to_string(), "1".to_string()),
            ("PYTHONPATH".into(), "/r/plugin".into()),
            ("OVERBRAINER_METRICS".into(), "/r/metrics.jsonl".into()),
        ]
    );
    assert_eq!(trainer.metrics_file(), "metrics.jsonl");
    assert_eq!(
        trainer.artifacts(),
        Artifacts {
            entries: vec!["output".into(), "metrics.jsonl".into()],
            exclude: vec!["checkpoint-*".into()],
        }
    );
    Ok(())
}

#[test]
fn prepare_writes_the_run_files() -> TestResult {
    let (dir, settings) = settings(QLORA)?;
    let files = DataFiles::new(dir.path());
    let trainer = Axolotl::new(training(&settings)?, &files);
    let run = dir.path().join("runs/r1");

    assert!(matches!(
        trainer.prepare(&run, "/workspace/run"),
        Err(TrainError::NoTrainingData { .. })
    ));

    fs::create_dir_all(dir.path().join("data"))?;
    fs::write(&files.train, "{\"a\":1}\n")?;
    trainer.prepare(&run, "/workspace/run")?;
    assert_eq!(
        fs::read_to_string(run.join("data/train.jsonl"))?,
        "{\"a\":1}\n"
    );
    assert!(!run.join("data/eval.jsonl").exists());
    assert!(run.join("plugin/overbrainer_metrics.py").is_file());
    let yaml = fs::read_to_string(run.join("axolotl.yaml"))?;
    assert!(yaml.contains("base_model: \"Qwen/Qwen3-4B\""));
    assert!(!yaml.contains("test_datasets"));

    fs::write(&files.eval, "{\"b\":2}\n")?;
    trainer.prepare(&run, "/workspace/run")?;
    assert!(run.join("data/eval.jsonl").is_file());
    assert!(fs::read_to_string(run.join("axolotl.yaml"))?.contains("test_datasets"));
    Ok(())
}

#[test]
fn pyyaml_reads_the_config_back_unchanged() -> TestResult {
    let probe = Command::new("python3").args(["-c", "import yaml"]).output();
    if !probe.is_ok_and(|output| output.status.success()) {
        eprintln!("skipped: python3 with PyYAML is not installed");
        return Ok(());
    }
    let (dir, settings) = settings(&QLORA.replace(
        "chat_template = \"qwen3\"",
        "chat_template = \"qwen3\"\nweight_decay = 1e-7\nyes = \"on\"\nnote = \"tab\\t\\u007f é\"",
    ))?;
    let trainer = Axolotl::new(training(&settings)?, &DataFiles::new(dir.path()));
    let config = trainer.config("/workspace/run", true);
    let path = dir.path().join("axolotl.yaml");
    fs::write(&path, to_yaml(&config))?;
    let output = Command::new("python3")
        .args([
            "-c",
            "import json, sys, yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))))",
        ])
        .arg(&path)
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let read_back: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(read_back, config);
    Ok(())
}

#[test]
fn reasoning_template_warning_follows_the_template_in_use() -> TestResult {
    let (_dir, qwen) = settings(QLORA)?;
    assert_eq!(reasoning_template_warning(training(&qwen)?), None);

    let (_dir, llama) = settings(
        "[training]\ntarget = \"box\"\nbase_model = \"meta-llama/Llama-3.2-1B\"\nadapter = \"lora\"\n",
    )?;
    let warning = reasoning_template_warning(training(&llama)?).ok_or("no warning")?;
    assert!(warning.contains("meta-llama/Llama-3.2-1B"), "{warning}");

    let (_dir, qwen_default) = settings(
        "[training]\ntarget = \"box\"\nbase_model = \"Qwen/Qwen3-8B\"\nadapter = \"lora\"\n",
    )?;
    assert_eq!(reasoning_template_warning(training(&qwen_default)?), None);

    let (_dir, chatml) = settings(
        "[training]\ntarget = \"box\"\nbase_model = \"Qwen/Qwen3-8B\"\nadapter = \"lora\"\n[training.axolotl_extra]\nchat_template = \"chatml\"\n",
    )?;
    let warning = reasoning_template_warning(training(&chatml)?).ok_or("no warning")?;
    assert!(warning.contains("chat_template `chatml`"), "{warning}");
    Ok(())
}
