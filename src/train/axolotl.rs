//! The Axolotl trainer: config, commands, environment and artifacts of a run.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use super::metrics::{METRICS_ENV, METRICS_PLUGIN, PLUGIN_CLASS, PLUGIN_FILE};
use super::{Artifacts, TrainError, Trainer, to_yaml};
use crate::config::{Adapter, Training};
use crate::dataset::DataFiles;

/// The Axolotl config, relative to the run directory.
pub const CONFIG_FILE: &str = "axolotl.yaml";
/// Metrics written by the plugin, relative to the run directory.
pub const METRICS_FILE: &str = "metrics.jsonl";
/// Axolotl's `output_dir`, relative to the run directory. The merged model, when
/// `merge = true`, goes to its `merged/` subdirectory.
pub const OUTPUT_DIR: &str = "output";
/// Copies of `data/train.jsonl` and `data/eval.jsonl`, relative to the run directory.
const TRAIN_FILE: &str = "data/train.jsonl";
const EVAL_FILE: &str = "data/eval.jsonl";
/// Directory put on `PYTHONPATH`, holding the metrics plugin.
const PLUGIN_DIR: &str = "plugin";
/// Axolotl's `dataset_prepared_path`: one per run, since Axolotl's cache key ignores
/// file contents.
const PREPARED_DIR: &str = "prepared";

/// Chat templates bundled with Axolotl 0.19.0 that render `reasoning_content`.
const REASONING_TEMPLATES: [&str; 5] = ["qwen3", "qwen3_5", "exaone4", "gemma4", "gemma4_unified"];
/// Base model family whose own chat template is known to render `reasoning_content`.
const REASONING_MODEL: &str = "qwen3";

/// Fine-tunes with Axolotl from `data/train.jsonl`, evaluating on `data/eval.jsonl`.
#[derive(Debug)]
pub struct Axolotl<'a> {
    training: &'a Training,
    train: PathBuf,
    eval: PathBuf,
}

impl<'a> Axolotl<'a> {
    /// A trainer for `training`, reading the split files of `files`.
    #[must_use]
    pub fn new(training: &'a Training, files: &DataFiles) -> Self {
        Self {
            training,
            train: files.train.clone(),
            eval: files.eval.clone(),
        }
    }

    /// The Axolotl config of a run whose directory the job sees at `root`, with
    /// `training.axolotl_extra` merged in. Without `has_eval`, no evaluation is set.
    #[must_use]
    pub fn config(&self, root: &str, has_eval: bool) -> Value {
        let training = self.training;
        let mut config = json!({
            "base_model": training.base_model,
            "datasets": [dataset(&format!("{root}/{TRAIN_FILE}"))],
            "val_set_size": 0,
            "sequence_len": training.sequence_len,
            "sample_packing": training.sample_packing,
            "eval_sample_packing": false,
            "attn_implementation": "sdpa",
            "num_epochs": training.epochs,
            "micro_batch_size": training.micro_batch_size,
            "gradient_accumulation_steps": training.gradient_accumulation_steps,
            "learning_rate": training.learning_rate,
            "optimizer": training.optimizer,
            "lr_scheduler": training.lr_scheduler,
            "warmup_ratio": 0.1,
            "gradient_checkpointing": true,
            "logging_steps": 1,
            "output_dir": format!("{root}/{OUTPUT_DIR}"),
            "dataset_prepared_path": format!("{root}/{PREPARED_DIR}"),
            "plugins": [PLUGIN_CLASS],
        });
        if let Value::Object(map) = &mut config {
            self.adapter_keys(map);
            self.cadence_keys(map, root, has_eval);
            if let Some(hub_model_id) = &training.hub_model_id {
                map.insert("hub_model_id".into(), json!(hub_model_id));
            }
        }
        for (key, value) in &training.axolotl_extra {
            merge(&mut config, key, value);
        }
        // Without eval data, Axolotl (via the HF `Trainer`) rejects a step-based or
        // strategy-based eval cadence set through `axolotl_extra`: there is no eval
        // dataset for it to apply to.
        if !has_eval && let Value::Object(map) = &mut config {
            map.remove("eval_steps");
            map.remove("eval_strategy");
        }
        config
    }

    fn adapter_keys(&self, map: &mut Map<String, Value>) {
        let training = self.training;
        let (adapter, load_in_4bit) = match training.adapter {
            Adapter::Lora => ("lora", false),
            Adapter::Qlora => ("qlora", true),
            Adapter::Full => return,
        };
        map.extend([
            ("adapter".into(), json!(adapter)),
            ("load_in_4bit".into(), json!(load_in_4bit)),
            ("load_in_8bit".into(), json!(false)),
            ("lora_r".into(), json!(training.lora_r)),
            ("lora_alpha".into(), json!(training.lora_alpha)),
            ("lora_dropout".into(), json!(training.lora_dropout)),
            ("lora_target_linear".into(), json!(true)),
        ]);
    }

    /// Evaluation and save cadence. A step-based setting in `axolotl_extra`
    /// (`eval_steps`, `save_steps`) replaces the per-epoch one, which Axolotl
    /// rejects alongside it.
    fn cadence_keys(&self, map: &mut Map<String, Value>, root: &str, has_eval: bool) {
        let extra = &self.training.axolotl_extra;
        if has_eval {
            map.insert(
                "test_datasets".into(),
                json!([test_dataset(&format!("{root}/{EVAL_FILE}"))]),
            );
            if !extra.contains_key("eval_steps") {
                map.insert(
                    "evals_per_epoch".into(),
                    json!(self.training.evals_per_epoch),
                );
            }
        }
        if !extra.contains_key("save_steps") {
            map.insert(
                "saves_per_epoch".into(),
                json!(self.training.saves_per_epoch),
            );
        }
    }
}

fn dataset(path: &str) -> Value {
    json!({
        "path": path,
        "ds_type": "json",
        "type": "chat_template",
        "field_messages": "messages",
        "field_thinking": "reasoning_content",
        "roles_to_train": ["assistant"],
        "train_on_eos": "turn",
    })
}

/// Single local files only expose a `train` split, also for evaluation.
fn test_dataset(path: &str) -> Value {
    let mut value = dataset(path);
    if let Value::Object(map) = &mut value {
        map.insert("split".into(), json!("train"));
    }
    value
}

/// Merges `value` into `config` at `key`: mappings merge key by key, anything else
/// replaces what is there.
fn merge(config: &mut Value, key: &str, value: &Value) {
    let Value::Object(map) = config else {
        return;
    };
    match (map.get_mut(key), value) {
        (Some(existing @ Value::Object(_)), Value::Object(inner)) => {
            for (inner_key, inner_value) in inner {
                merge(existing, inner_key, inner_value);
            }
        },
        _ => {
            map.insert(key.to_string(), value.clone());
        },
    }
}

impl Trainer for Axolotl<'_> {
    fn prepare(&self, run_dir: &Path, root: &str) -> Result<(), TrainError> {
        if file_size(&self.train)?.unwrap_or(0) == 0 {
            return Err(TrainError::NoTrainingData {
                path: self.train.clone(),
            });
        }
        copy(&self.train, &run_dir.join(TRAIN_FILE))?;
        let has_eval = file_size(&self.eval)?.is_some_and(|len| len > 0);
        if has_eval {
            copy(&self.eval, &run_dir.join(EVAL_FILE))?;
        }
        write(&run_dir.join(PLUGIN_DIR).join(PLUGIN_FILE), METRICS_PLUGIN)?;
        write(
            &run_dir.join(CONFIG_FILE),
            &to_yaml(&self.config(root, has_eval)),
        )
    }

    fn commands(&self) -> Vec<Vec<String>> {
        let mut commands = vec![command("train")];
        if self.training.merge {
            commands.push(command("merge-lora"));
        }
        commands
    }

    fn env(&self, root: &str) -> Vec<(String, String)> {
        vec![
            ("AXOLOTL_DO_NOT_TRACK".into(), "1".into()),
            ("PYTHONPATH".into(), format!("{root}/{PLUGIN_DIR}")),
            (METRICS_ENV.into(), format!("{root}/{METRICS_FILE}")),
        ]
    }

    fn metrics_file(&self) -> &'static str {
        METRICS_FILE
    }

    fn artifacts(&self) -> Artifacts {
        Artifacts {
            entries: vec![OUTPUT_DIR.into(), METRICS_FILE.into()],
            exclude: vec!["checkpoint-*".into()],
        }
    }
}

fn command(subcommand: &str) -> Vec<String> {
    vec!["axolotl".into(), subcommand.into(), CONFIG_FILE.into()]
}

/// The size of `path`, or `None` when it does not exist.
///
/// A metadata error other than "not found" (for example a permission error, or a
/// path component that is not a directory) is not a missing file and is reported as
/// [`TrainError::Io`] rather than treated as one.
fn file_size(path: &Path) -> Result<Option<u64>, TrainError> {
    match fs::metadata(path) {
        Ok(meta) => Ok(Some(meta.len())),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(io_error(path)(source)),
    }
}

fn copy(from: &Path, to: &Path) -> Result<(), TrainError> {
    if let Some(dir) = to.parent() {
        fs::create_dir_all(dir).map_err(io_error(dir))?;
    }
    fs::copy(from, to)
        .map(drop)
        .map_err(|source| TrainError::Copy {
            from: from.to_path_buf(),
            to: to.to_path_buf(),
            source,
        })
}

fn write(path: &Path, content: &str) -> Result<(), TrainError> {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(io_error(dir))?;
    }
    fs::write(path, content).map_err(io_error(path))
}

fn io_error(path: &Path) -> impl FnOnce(std::io::Error) -> TrainError + '_ {
    move |source| TrainError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// A warning when the chat template in use may drop `reasoning_content`, which
/// leaves the parent's reasoning out of training without any error.
///
/// With `axolotl_extra.chat_template_jinja` set, that custom template decides the
/// answer on its own (it is checked for a `reasoning_content` reference), regardless
/// of `chat_template`. Otherwise, with `axolotl_extra.chat_template` set to anything
/// but `tokenizer_default` (Axolotl's own name for "use the base model's template"),
/// only Axolotl's reasoning templates pass. Otherwise, or with `tokenizer_default`,
/// the base model's own template is used, and only Qwen3 model names pass (their
/// official template renders reasoning).
///
/// This is a name heuristic: a local path or a renamed model may be warned about
/// wrongly, and it can pass wrongly too. A non-thinking Qwen3 variant, for example
/// `Qwen3-8B-Instruct-2507` or `Qwen3-Coder-30B-A3B-Instruct`, matches the Qwen3 name
/// check but its own template does not render reasoning.
#[must_use]
pub fn reasoning_template_warning(training: &Training) -> Option<String> {
    let extra = &training.axolotl_extra;
    if let Some(jinja) = extra.get("chat_template_jinja").and_then(Value::as_str) {
        return if jinja.contains("reasoning_content") {
            None
        } else {
            Some(
                "chat_template_jinja does not reference reasoning_content, leaving \
                 the parent's reasoning out of training"
                    .to_string(),
            )
        };
    }
    let known = format!(
        "templates known to render it: {}",
        REASONING_TEMPLATES.join(", ")
    );
    let chat_template = extra
        .get("chat_template")
        .and_then(Value::as_str)
        .filter(|template| *template != "tokenizer_default");
    match chat_template {
        Some(template) if REASONING_TEMPLATES.contains(&template) => None,
        Some(template) => Some(format!(
            "chat_template `{template}` may drop reasoning_content, leaving the parent's reasoning out of training ({known})"
        )),
        None => {
            let name = training.base_model.to_ascii_lowercase();
            if name.contains(REASONING_MODEL) {
                None
            } else {
                Some(format!(
                    "the chat template of {} may drop reasoning_content, leaving the parent's reasoning out of training; set training.axolotl_extra.chat_template if the model uses one of the {known}",
                    training.base_model
                ))
            }
        },
    }
}
