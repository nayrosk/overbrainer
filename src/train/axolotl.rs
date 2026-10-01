//! The Axolotl trainer: config, commands, environment and artifacts of a run.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use super::metrics::{
    JobStage, METRICS_ENV, METRICS_PLUGIN, PLUGIN_CLASS, PLUGIN_FILE, SNAPSHOT_ENV, SNAPSHOT_FILE,
    SNAPSHOT_REQUEST,
};
use super::{Artifacts, TrainError, Trainer, to_yaml};
use crate::config::{Adapter, Training};
use crate::dataset::DataFiles;
use crate::export::ExportJob;

/// The Axolotl config, relative to the run directory.
pub const CONFIG_FILE: &str = "axolotl.yaml";
/// Metrics written by the plugin, relative to the run directory.
pub const METRICS_FILE: &str = "metrics.jsonl";
/// Axolotl's `output_dir`, relative to the run directory. The merged model, when
/// `merge = true`, goes to its `merged/` subdirectory.
pub const OUTPUT_DIR: &str = "output";
/// Where Axolotl's `merge-lora` writes the merged model, relative to [`OUTPUT_DIR`].
pub const MERGED_DIR: &str = "merged";
/// Copies of `data/train.jsonl` and `data/eval.jsonl`, relative to the run directory.
const TRAIN_FILE: &str = "data/train.jsonl";
const EVAL_FILE: &str = "data/eval.jsonl";
/// Directory put on `PYTHONPATH`, holding the metrics plugin.
const PLUGIN_DIR: &str = "plugin";
/// Axolotl's `dataset_prepared_path`: one per run, since Axolotl's cache key ignores
/// file contents.
const PREPARED_DIR: &str = "prepared";
/// Where a resumed run holds the checkpoint it resumes from, relative to the run
/// directory.
pub const RESUME_DIR: &str = "resume";
/// Checkpoints Axolotl keeps in [`OUTPUT_DIR`] when `axolotl_extra` sets no
/// `save_total_limit`.
pub const SAVE_TOTAL_LIMIT: u32 = 2;

/// Top-level Axolotl keys a resumed run may set differently from the run it
/// resumes: none changes what the checkpoint was trained with. The cadence of
/// evaluations, saves and logs, the Hub push, and the logging integrations.
const MAY_DIFFER: [&str; 18] = [
    "resume_from_checkpoint",
    "hub_model_id",
    "hub_strategy",
    "evals_per_epoch",
    "eval_steps",
    "eval_strategy",
    "saves_per_epoch",
    "save_steps",
    "save_strategy",
    "save_total_limit",
    "logging_steps",
    "use_tensorboard",
    "use_wandb",
    "use_mlflow",
    "use_comet",
    "wandb_*",
    "mlflow_*",
    "comet_*",
];

/// Whether a resumed run may set `key` differently: [`MAY_DIFFER`], where a
/// trailing `*` stands for any rest of the name.
fn may_differ(key: &str) -> bool {
    MAY_DIFFER
        .iter()
        .any(|allowed| match allowed.strip_suffix('*') {
            Some(prefix) => key.starts_with(prefix),
            None => key == *allowed,
        })
}

/// Chat templates bundled with Axolotl 0.19.0 that render `reasoning_content`.
const REASONING_TEMPLATES: [&str; 5] = ["qwen3", "qwen3_5", "exaone4", "gemma4", "gemma4_unified"];
/// Base model family whose own chat template is known to render `reasoning_content`.
const REASONING_MODEL: &str = "qwen3";

/// What a succeeded run leaves in its directory, from the `[training]` it ran
/// with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Outputs {
    /// `training.adapter`: `full` leaves a whole model, not an adapter.
    pub adapter: Adapter,
    /// `training.merge`: a merged model too.
    pub merge: bool,
}

impl Outputs {
    /// The outputs of a run of `training`.
    #[must_use]
    pub fn of(training: &Training) -> Self {
        Self {
            adapter: training.adapter,
            merge: training.merge,
        }
    }

    /// What the succeeded run in the local directory `run_dir` left, read
    /// from its own files: its `axolotl.yaml` names the adapter (none for
    /// `full`), and `output/merged/` is there when it was merged. `None` when
    /// its `axolotl.yaml` cannot be read or names an unknown adapter.
    #[must_use]
    pub fn recorded(run_dir: &Path) -> Option<Self> {
        let config = fs::read_to_string(run_dir.join(CONFIG_FILE)).ok()?;
        let adapter = config
            .lines()
            .find_map(|line| line.strip_prefix("adapter:"))
            .map(|value| value.trim().trim_matches(['"', '\'']));
        let adapter = match adapter {
            None => Adapter::Full,
            Some("lora") => Adapter::Lora,
            Some("qlora") => Adapter::Qlora,
            Some(_) => return None,
        };
        Some(Self {
            adapter,
            merge: run_dir.join(OUTPUT_DIR).join(MERGED_DIR).is_dir(),
        })
    }

    /// Where the model of succeeded run `run_id` is, relative to the project
    /// directory, each with what it is: the adapter (the model for `full`),
    /// then the merged model with `merge = true`.
    #[must_use]
    pub fn paths(self, run_id: &str) -> Vec<(&'static str, String)> {
        let output = format!("{}/{run_id}/{OUTPUT_DIR}", crate::runs::RUNS_DIR);
        let what = if self.adapter == Adapter::Full {
            "model"
        } else {
            "adapter"
        };
        let mut paths = vec![(what, output.clone())];
        if self.merge && self.adapter != Adapter::Full {
            paths.push(("merged model", format!("{output}/{MERGED_DIR}")));
        }
        paths
    }
}

/// The snapshot a new run resumes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resume {
    /// The stopped run.
    pub run_id: String,
    /// Its local directory, holding its data, its `axolotl.yaml` and the
    /// checkpoint.
    pub dir: PathBuf,
    /// The checkpoint, relative to `dir`, for example `output/checkpoint-120`.
    pub checkpoint: String,
}

impl Resume {
    /// The checkpoint's directory name, `checkpoint-120`.
    fn name(&self) -> &str {
        self.checkpoint
            .rsplit('/')
            .next()
            .unwrap_or(self.checkpoint.as_str())
    }
}

/// Fine-tunes with Axolotl from `data/train.jsonl`, evaluating on `data/eval.jsonl`.
#[derive(Debug, Clone)]
pub struct Axolotl<'a> {
    training: &'a Training,
    train: PathBuf,
    eval: PathBuf,
    resume: Option<Resume>,
    export: Option<ExportJob>,
}

impl<'a> Axolotl<'a> {
    /// A trainer for `training`, reading the split files of `files`.
    #[must_use]
    pub fn new(training: &'a Training, files: &DataFiles) -> Self {
        Self {
            training,
            train: files.train.clone(),
            eval: files.eval.clone(),
            resume: None,
            export: None,
        }
    }

    /// The same trainer exporting the model of run `run_id` to GGUF at the
    /// end of its job, quantized to `quantize`, after the merge (and, like
    /// it, not once the job stopped with a snapshot).
    #[must_use]
    pub fn exporting(self, run_id: &str, quantize: &str) -> Self {
        Self {
            export: Some(ExportJob::in_job(run_id, quantize)),
            ..self
        }
    }

    /// The export at the end of the job, if any.
    #[must_use]
    pub fn export(&self) -> Option<&ExportJob> {
        self.export.as_ref()
    }

    /// The same trainer resuming from `resume`: it trains on the data of the
    /// stopped run, and its runs start from a copy of the checkpoint.
    #[must_use]
    pub fn resuming(self, resume: Resume) -> Self {
        Self {
            train: resume.dir.join(TRAIN_FILE),
            eval: resume.dir.join(EVAL_FILE),
            resume: Some(resume),
            ..self
        }
    }

    /// The snapshot this trainer resumes from, if any.
    #[must_use]
    pub fn resume(&self) -> Option<&Resume> {
        self.resume.as_ref()
    }

    /// The top-level Axolotl keys whose value differs between the stopped run
    /// this trainer resumes from (its `axolotl.yaml`) and what the settings give
    /// now, for the same run directory: the base model, the adapter, the
    /// learning rate, the batch size, and any other that the checkpoint depends
    /// on. Keys that only change the cadence of evaluations and saves, or the
    /// Hub push, may differ. Empty when nothing differs or nothing is resumed.
    ///
    /// # Errors
    ///
    /// Returns [`TrainError::Io`] when the stopped run's `axolotl.yaml` cannot
    /// be read, or names no `output_dir`.
    pub fn resume_mismatch(&self) -> Result<Vec<String>, TrainError> {
        let Some(resume) = &self.resume else {
            return Ok(Vec::new());
        };
        let path = resume.dir.join(CONFIG_FILE);
        let text = fs::read_to_string(&path).map_err(io_error(&path))?;
        let theirs = top_level(&text);
        let root = theirs
            .get("output_dir")
            .and_then(|block| block.strip_prefix("output_dir: \""))
            .and_then(|value| value.trim_end().strip_suffix(&format!("/{OUTPUT_DIR}\"")))
            .ok_or_else(|| TrainError::Io {
                path: path.clone(),
                source: std::io::Error::other("no output_dir"),
            })?;
        let ours = to_yaml(&self.config(root, theirs.contains_key("test_datasets")));
        let ours = top_level(&ours);
        let mut keys: Vec<&str> = theirs.keys().chain(ours.keys()).copied().collect();
        keys.sort_unstable();
        keys.dedup();
        Ok(keys
            .into_iter()
            .filter(|key| !may_differ(key) && theirs.get(key) != ours.get(key))
            .map(str::to_string)
            .collect())
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
            if let Some(resume) = &self.resume {
                map.insert(
                    "resume_from_checkpoint".into(),
                    json!(format!("{root}/{RESUME_DIR}/{}", resume.name())),
                );
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
        // Older checkpoints are removed so they never fill the disk; a
        // snapshot is always the newest one, so it stays.
        if !extra.contains_key("save_total_limit") {
            map.insert("save_total_limit".into(), json!(SAVE_TOTAL_LIMIT));
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

/// Each top-level key of the YAML `text` with its whole block (its line and the
/// indented lines under it), as [`to_yaml`] writes them.
fn top_level(text: &str) -> std::collections::BTreeMap<&str, String> {
    let mut blocks = std::collections::BTreeMap::new();
    let mut current: Option<(&str, String)> = None;
    for line in text.lines() {
        let nested = line.starts_with(' ') || line.starts_with('-');
        match &mut current {
            Some((_, block)) if nested => {
                block.push('\n');
                block.push_str(line);
            },
            _ => {
                if let Some((key, block)) = current.take() {
                    blocks.insert(key, block);
                }
                let key = line.split_once(':').map_or(line, |(key, _)| key);
                current = Some((key, line.to_string()));
            },
        }
    }
    if let Some((key, block)) = current {
        blocks.insert(key, block);
    }
    blocks
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
        if self.export.is_some() {
            ExportJob::write_script(run_dir)?;
        }
        if let Some(resume) = &self.resume {
            let into = run_dir.join(RESUME_DIR).join(resume.name());
            link_tree(&resume.dir.join(&resume.checkpoint), &into)?;
        }
        write(
            &run_dir.join(CONFIG_FILE),
            &to_yaml(&self.config(root, has_eval)),
        )
    }

    fn commands(&self) -> Vec<Vec<String>> {
        let mut commands: Vec<Vec<String>> = self
            .steps()
            .into_iter()
            .map(|(_, subcommand)| command(subcommand))
            .collect();
        if let Some(export) = &self.export {
            commands.push(export.command());
        }
        commands
    }

    fn stages(&self) -> Vec<JobStage> {
        let mut stages: Vec<JobStage> = self.steps().into_iter().map(|(stage, _)| stage).collect();
        if self.export.is_some() {
            stages.push(JobStage::Export);
        }
        stages
    }

    fn env(&self, root: &str) -> Vec<(String, String)> {
        let mut env = vec![
            ("AXOLOTL_DO_NOT_TRACK".into(), "1".into()),
            ("PYTHONPATH".into(), format!("{root}/{PLUGIN_DIR}")),
            (METRICS_ENV.into(), format!("{root}/{METRICS_FILE}")),
            (SNAPSHOT_ENV.into(), format!("{root}/{SNAPSHOT_REQUEST}")),
        ];
        if let Some(export) = &self.export {
            env.extend(export.script_env());
        }
        env
    }

    fn caches_tools(&self) -> bool {
        self.export.is_some()
    }

    fn stop_marker(&self) -> Option<&'static str> {
        Some(SNAPSHOT_FILE)
    }

    fn metrics_file(&self) -> &'static str {
        METRICS_FILE
    }

    fn artifacts(&self) -> Artifacts {
        // `output/` holds the GGUF of an export too, in `output/gguf/`.
        Artifacts {
            entries: vec![OUTPUT_DIR.into(), METRICS_FILE.into()],
            exclude: vec!["checkpoint-*".into()],
            required: Some(OUTPUT_DIR.into()),
        }
    }
}

impl Axolotl<'_> {
    /// The job's commands, as stage and `axolotl` subcommand.
    fn steps(&self) -> Vec<(JobStage, &'static str)> {
        let mut steps = vec![(JobStage::Train, "train")];
        if self.training.merge {
            steps.push((JobStage::Merge, "merge-lora"));
        }
        steps
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

/// Copies the directory `from` into `to`, file by file: each is hard-linked when
/// both are on the same file system (a checkpoint can weigh gigabytes), copied
/// otherwise. Anything that is neither a file nor a directory is left out.
fn link_tree(from: &Path, to: &Path) -> Result<(), TrainError> {
    fs::create_dir_all(to).map_err(io_error(to))?;
    for entry in fs::read_dir(from).map_err(io_error(from))? {
        let entry = entry.map_err(io_error(from))?;
        let kind = entry.file_type().map_err(io_error(&entry.path()))?;
        let target = to.join(entry.file_name());
        if kind.is_dir() {
            link_tree(&entry.path(), &target)?;
        } else if kind.is_file() && fs::hard_link(entry.path(), &target).is_err() {
            copy(&entry.path(), &target)?;
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorded_outputs_come_from_the_run_s_own_files() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        assert_eq!(Outputs::recorded(dir.path()), None, "no axolotl.yaml");
        fs::write(
            dir.path().join(CONFIG_FILE),
            "base_model: m\nadapter: qlora\n",
        )?;
        let qlora = Outputs {
            adapter: Adapter::Qlora,
            merge: false,
        };
        assert_eq!(Outputs::recorded(dir.path()), Some(qlora));
        fs::create_dir_all(dir.path().join(OUTPUT_DIR).join(MERGED_DIR))?;
        let merged = Outputs {
            merge: true,
            ..qlora
        };
        assert_eq!(Outputs::recorded(dir.path()), Some(merged));
        fs::write(dir.path().join(CONFIG_FILE), "base_model: m\n")?;
        let full = Outputs::recorded(dir.path()).map(|outputs| outputs.adapter);
        assert_eq!(full, Some(Adapter::Full));
        fs::write(dir.path().join(CONFIG_FILE), "adapter: dora\n")?;
        assert_eq!(Outputs::recorded(dir.path()), None);
        Ok(())
    }

    #[test]
    fn an_exporting_trainer_exports_after_the_merge() -> Result<(), Box<dyn std::error::Error>> {
        let training: Training = serde_json::from_value(json!({
            "target": "local", "base_model": "m", "adapter": "lora", "merge": true
        }))?;
        let project = tempfile::tempdir()?;
        let files = DataFiles::new(project.path());
        std::fs::create_dir_all(project.path().join("data"))?;
        std::fs::write(&files.train, "{}\n")?;
        let plain = Axolotl::new(&training, &files);
        assert_eq!(plain.commands().len(), 2);
        assert!(!plain.caches_tools());
        let trainer = plain.exporting("r1", "Q8_0");
        let commands = trainer.commands();
        assert_eq!(commands.len(), 3);
        assert_eq!(commands[1][1], "merge-lora");
        assert_eq!(commands[2], ExportJob::in_job("r1", "Q8_0").command());
        assert!(trainer.caches_tools());
        assert_eq!(trainer.stop_marker(), Some(SNAPSHOT_FILE));
        let env = trainer.env("/w/r1");
        assert!(env.contains(&("OVERBRAINER_EXPORT_QUANTIZE".into(), "Q8_0".into())));
        assert!(env.contains(&(METRICS_ENV.into(), "/w/r1/metrics.jsonl".into())));
        let run = tempfile::tempdir()?;
        trainer.prepare(run.path(), "/w/r1")?;
        assert!(run.path().join(crate::export::SCRIPT_FILE).is_file());
        Ok(())
    }

    #[test]
    fn outputs_name_the_adapter_then_the_merged_model() {
        let paths = |adapter, merge| Outputs { adapter, merge }.paths("r1");
        assert_eq!(
            paths(Adapter::Qlora, true),
            [
                ("adapter", "runs/r1/output".to_string()),
                ("merged model", "runs/r1/output/merged".to_string()),
            ]
        );
        assert_eq!(
            paths(Adapter::Lora, false),
            [("adapter", "runs/r1/output".to_string())]
        );
        assert_eq!(
            paths(Adapter::Full, true),
            [("model", "runs/r1/output".to_string())]
        );
    }
}
