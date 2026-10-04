//! The model card (`README.md`) of a pushed run.

use std::fmt::{self, Write as _};
use std::path::{Path, PathBuf};
use std::time::Duration;
use std::{fs, io};

use serde::de::IgnoredAny;
use toml_edit::{ArrayOfTables, DocumentMut, Item, Table, value};

use super::{RepoId, UploadFile};
use crate::config::{Adapter, Pipeline, RoleModel, Settings, Training};
use crate::runpod::PodRecord;
use crate::runs::{MetricsSummary, RunRecord, Runs, RunsError};
use crate::train::sizing::is_repo_id;
use crate::train::{CONFIG_FILE, METRICS_FILE, MetricLine, Outputs, parse_line, top_level_scalar};

/// Last line of every card overbrainer writes: a remote card carrying it may be
/// replaced.
pub const MARKER: &str = "<!-- overbrainer:card -->";
/// The banner at the top of the card.
pub const BANNER_URL: &str =
    "https://raw.githubusercontent.com/nayrosk/overbrainer/main/docs/assets/hf-banner.webp";
/// The project.
pub const REPO_URL: &str = "https://github.com/nayrosk/overbrainer";
/// Runpod, with the project's referral code.
pub const RUNPOD_URL: &str = "https://runpod.io?ref=ym24z23f";

/// Errors reading what the card says of a run.
#[derive(Debug, thiserror::Error)]
pub enum CardError {
    /// The run's directory cannot be named.
    #[error(transparent)]
    Run(#[from] RunsError),
    /// The run's `axolotl.yaml` cannot be read or names an unknown adapter.
    #[error("cannot read the run config {}", path.display())]
    Config {
        /// The run's `axolotl.yaml`.
        path: PathBuf,
    },
    /// A file of the run cannot be read.
    #[error("cannot read {}", path.display())]
    Read {
        /// The file.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// The run's `axolotl.yaml` names no base model.
    #[error("{} names no base_model", path.display())]
    NoBaseModel {
        /// The run's `axolotl.yaml`.
        path: PathBuf,
    },
}

/// What the card says of a run.
#[derive(Debug, Clone, PartialEq)]
pub struct CardInput {
    pub repo: RepoId,
    pub base_model: String,
    pub adapter: Adapter,
    pub license: Option<String>,
    pub parent: String,
    pub generator: String,
    /// Name and description, empty when the topic has none.
    pub topics: Vec<(String, String)>,
    pub train_examples: Option<usize>,
    pub eval_examples: Option<usize>,
    pub epochs: Option<f64>,
    /// As written in the run's `axolotl.yaml`.
    pub learning_rate: Option<String>,
    pub sequence_len: Option<u32>,
    pub train_loss: Option<f64>,
    pub eval_loss: Option<f64>,
    pub duration: Option<Duration>,
    /// Quantization types of the GGUF files pushed, sorted.
    pub gguf: Vec<String>,
    pub runpod: Option<RunpodLine>,
    /// The non-secret part of `overbrainer.toml`.
    pub reproduce_toml: String,
}

/// Where a Runpod run trained, and what it cost.
#[derive(Debug, Clone, PartialEq)]
pub struct RunpodLine {
    pub gpu: String,
    pub spend_usd: f64,
}

/// The card's Markdown, front matter included, marker last.
#[must_use]
pub fn render(input: &CardInput) -> String {
    let mut out = String::new();
    // Writing to a String cannot fail.
    let _ = write_card(&mut out, input);
    out
}

fn write_card(out: &mut String, input: &CardInput) -> fmt::Result {
    front_matter(out, input)?;
    writeln!(out, "[![overbrainer]({BANNER_URL})]({REPO_URL})\n")?;
    writeln!(
        out,
        "Distilled with [overbrainer]({REPO_URL}) [![overbrainer](https://img.shields.io/badge/distilled%20with-overbrainer-E93D82)]({REPO_URL})\n"
    )?;
    writeln!(out, "# {}\n", text(&input.repo.to_string()))?;
    let tuned = match input.adapter {
        Adapter::Full => "fully fine-tuned".to_string(),
        adapter => format!("fine-tuned ({})", adapter_name(adapter)),
    };
    let base = text(&base_label(&input.base_model));
    let parent = text(&input.parent);
    match input.train_examples {
        Some(count) => writeln!(
            out,
            "{base} {tuned} on {count} questions answered by {parent}.\n"
        )?,
        None => writeln!(out, "{base} {tuned} on answers from {parent}.\n")?,
    }
    use_it(out, input)?;
    how_it_was_made(out, input)?;
    reproduce(out, input)?;
    writeln!(out, "---\n")?;
    writeln!(
        out,
        "[overbrainer]({REPO_URL}) distills a big LLM into a small one from your terminal. If this model is useful, a star on [GitHub]({REPO_URL}) helps.\n"
    )?;
    writeln!(out, "{MARKER}")
}

fn front_matter(out: &mut String, input: &CardInput) -> fmt::Result {
    let full = input.adapter == Adapter::Full;
    writeln!(out, "---")?;
    if is_repo_id(&input.base_model) {
        writeln!(out, "base_model: {}", yaml_scalar(&input.base_model))?;
        let relation = if full { "finetune" } else { "adapter" };
        writeln!(out, "base_model_relation: {relation}")?;
    }
    let library = if full { "transformers" } else { "peft" };
    writeln!(out, "library_name: {library}")?;
    if let Some(license) = &input.license {
        writeln!(out, "license: {}", yaml_scalar(license))?;
    }
    writeln!(out, "pipeline_tag: text-generation")?;
    writeln!(out, "tags:\n- overbrainer\n- distillation")?;
    if !input.gguf.is_empty() {
        writeln!(out, "- gguf")?;
    }
    writeln!(out, "---\n")
}

fn use_it(out: &mut String, input: &CardInput) -> fmt::Result {
    let repo = input.repo.to_string();
    writeln!(out, "## Use it\n")?;
    if let Some(tag) = ollama_tag(&input.gguf) {
        writeln!(out, "```sh")?;
        writeln!(out, "ollama run hf.co/{repo}:{tag}")?;
        writeln!(out, "llama-cli -hf {repo}:{tag}")?;
    } else if input.adapter == Adapter::Full {
        writeln!(out, "```python")?;
        writeln!(out, "from transformers import AutoModelForCausalLM\n")?;
        writeln!(
            out,
            "model = AutoModelForCausalLM.from_pretrained({})",
            py_string(&repo)
        )?;
    } else {
        writeln!(out, "```python")?;
        writeln!(out, "from peft import PeftModel")?;
        writeln!(out, "from transformers import AutoModelForCausalLM\n")?;
        writeln!(
            out,
            "model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained({}), {})",
            py_string(&shown_base(&input.base_model)),
            py_string(&repo)
        )?;
    }
    writeln!(out, "```\n")
}

fn how_it_was_made(out: &mut String, input: &CardInput) -> fmt::Result {
    writeln!(out, "## How it was made\n")?;
    writeln!(
        out,
        "overbrainer wrote questions on the topics below with the generator model, kept the parent model's answers and reasoning, then fine-tuned the base model on them with Axolotl.\n"
    )?;
    writeln!(out, "| | |\n|---|---|")?;
    for (name, value) in rows(input) {
        writeln!(out, "| {name} | {} |", cell(&value))?;
    }
    writeln!(out)?;

    if !input.topics.is_empty() {
        writeln!(out, "Topics:\n")?;
        for (name, description) in &input.topics {
            let description = text(description);
            if description.is_empty() {
                writeln!(out, "- **{}**", text(name))?;
            } else {
                writeln!(out, "- **{}**: {description}", text(name))?;
            }
        }
        writeln!(out)?;
    }

    if let Some(runpod) = &input.runpod {
        let spend = if runpod.spend_usd < 0.01 {
            "less than $0.01".to_string()
        } else {
            format!("about ${:.2}", runpod.spend_usd)
        };
        writeln!(
            out,
            "Trained on [Runpod]({RUNPOD_URL}) ({}) for {spend}.\n",
            text(&runpod.gpu)
        )?;
    }
    Ok(())
}

/// The rows of the "How it was made" table; a value that is not known has no row.
fn rows(input: &CardInput) -> Vec<(&'static str, String)> {
    let mut rows = vec![
        ("Parent model", input.parent.clone()),
        ("Generator model", input.generator.clone()),
        ("Base model", base_label(&input.base_model)),
        ("Adapter", adapter_name(input.adapter).to_string()),
    ];
    let examples: Vec<String> = [
        (input.train_examples, "train"),
        (input.eval_examples, "eval"),
    ]
    .into_iter()
    .filter_map(|(count, what)| count.map(|count| format!("{count} {what}")))
    .collect();
    if !examples.is_empty() {
        rows.push(("Examples", examples.join(", ")));
    }
    if let Some(epochs) = input.epochs {
        rows.push(("Epochs", epochs.to_string()));
    }
    if let Some(rate) = &input.learning_rate {
        rows.push(("Learning rate", rate.clone()));
    }
    if let Some(len) = input.sequence_len {
        rows.push(("Sequence length", len.to_string()));
    }
    let losses: Vec<String> = [(input.train_loss, "train"), (input.eval_loss, "eval")]
        .into_iter()
        .filter_map(|(loss, what)| loss.map(|loss| format!("{loss:.2} {what}")))
        .collect();
    if !losses.is_empty() {
        rows.push(("Final loss", losses.join(", ")));
    }
    if let Some(duration) = input.duration {
        rows.push(("Training time", duration_text(duration)));
    }
    rows
}

fn reproduce(out: &mut String, input: &CardInput) -> fmt::Result {
    writeln!(out, "## Reproduce\n")?;
    writeln!(out, "```sh\ncargo install --locked overbrainer\n```\n")?;
    writeln!(
        out,
        "With these sections of overbrainer.toml (project, providers, targets and keys left out), then `overbrainer run`:\n"
    )?;
    let fence = fence_for(&input.reproduce_toml);
    writeln!(out, "{fence}toml")?;
    write!(out, "{}", input.reproduce_toml)?;
    if !input.reproduce_toml.ends_with('\n') {
        writeln!(out)?;
    }
    writeln!(out, "{fence}\n")?;
    writeln!(out, "See the [overbrainer docs]({REPO_URL}#readme).\n")
}

/// The final path component of a base model that is not a Hub repo ID, so no
/// local path reaches the card; `None` for a repo ID.
fn local_name(model: &str) -> Option<String> {
    if is_repo_id(model) {
        return None;
    }
    Some(
        Path::new(model)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default(),
    )
}

/// The base model as code and the reproduce excerpt write it: a repo ID as
/// is, a local model by its final path component.
fn shown_base(model: &str) -> String {
    match local_name(model) {
        None => model.to_string(),
        Some(name) if name.is_empty() => "local-model".to_string(),
        Some(name) => name,
    }
}

/// The base model as prose names it: a repo ID as is, else "a local model".
fn base_label(model: &str) -> String {
    match local_name(model) {
        None => model.to_string(),
        Some(name) if name.is_empty() => "a local model".to_string(),
        Some(name) => format!("a local model (`{name}`)"),
    }
}

/// How the card names an adapter.
fn adapter_name(adapter: Adapter) -> &'static str {
    match adapter {
        Adapter::Lora => "LoRA",
        Adapter::Qlora => "QLoRA",
        Adapter::Full => "none (full fine-tune)",
    }
}

/// The tag of the GGUF file the commands pull: `Q4_K_M` when pushed, else the
/// first one.
fn ollama_tag(gguf: &[String]) -> Option<&str> {
    gguf.iter()
        .find(|tag| *tag == "Q4_K_M")
        .or_else(|| gguf.first())
        .map(String::as_str)
}

/// `value` as prose: on one line, line breaks becoming spaces, with `<` and
/// `>` escaped so it cannot open an HTML tag or comment on the Hub.
fn text(value: &str) -> String {
    value
        .split(['\r', '\n'])
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// `value` as a table cell: on one line, with `|` escaped.
fn cell(value: &str) -> String {
    text(value).replace('|', "\\|")
}

/// Plain scalars YAML 1.1 reads as booleans or null.
const YAML_KEYWORDS: [&str; 12] = [
    "true", "false", "yes", "no", "on", "off", "y", "n", "null", "nan", "inf", "infinity",
];

/// A YAML scalar: plain when it starts with a letter, holds only safe
/// characters and is no YAML keyword, else double-quoted, so it always reads
/// back as a string.
fn yaml_scalar(value: &str) -> String {
    let plain = value.starts_with(|c: char| c.is_ascii_alphabetic())
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/'))
        && !YAML_KEYWORDS.contains(&value.to_ascii_lowercase().as_str());
    if plain {
        value.to_string()
    } else {
        serde_json::Value::from(value).to_string()
    }
}

/// A Python string literal.
fn py_string(value: &str) -> String {
    serde_json::Value::from(value).to_string()
}

/// A code fence longer than any run of backticks in `body`.
fn fence_for(body: &str) -> String {
    let longest = body.split(|c| c != '`').map(str::len).max().unwrap_or(0);
    "`".repeat(longest.max(2) + 1)
}

/// `duration` in its largest sensible unit: `45 s`, `6 min`, `1 h 20 min`.
fn duration_text(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs < 60 {
        return format!("{secs} s");
    }
    let minutes = (secs + 30) / 60;
    match (minutes / 60, minutes % 60) {
        (0, minutes) => format!("{minutes} min"),
        (hours, 0) => format!("{hours} h"),
        (hours, minutes) => format!("{hours} h {minutes} min"),
    }
}

/// Whether a remote card may be replaced: there is none, or overbrainer wrote it.
#[must_use]
pub fn replaceable(remote: Option<&str>) -> bool {
    remote.is_none_or(|card| card.contains(MARKER))
}

/// The card input of run `record`, from its own files and `settings`: the
/// run's `axolotl.yaml` for what it trained, its data files for the example
/// counts, its `metrics.jsonl` for the losses and the training time, its pod
/// record for the GPU and the spend, and `files` for the GGUF types. A value
/// that cannot be read is left out. The license is left `None`: the caller
/// looks it up on the Hub.
///
/// # Errors
///
/// Returns a [`CardError`] when the run's `axolotl.yaml` cannot be read or
/// names no base model. Data, metrics or a pod record that cannot be read
/// only leave their values out, with a warning.
pub fn gather(
    runs: &Runs,
    record: &RunRecord,
    settings: &Settings,
    repo: &RepoId,
    files: &[UploadFile],
) -> Result<CardInput, CardError> {
    let dir = runs.run_dir(&record.id)?;
    let config_path = dir.join(CONFIG_FILE);
    let outputs = Outputs::recorded(&dir).ok_or_else(|| CardError::Config {
        path: config_path.clone(),
    })?;
    let config = fs::read_to_string(&config_path).map_err(|source| CardError::Read {
        path: config_path.clone(),
        source,
    })?;
    let scalar = |key: &str| top_level_scalar(&config, key);
    let base_model = scalar("base_model").ok_or_else(|| CardError::NoBaseModel {
        path: config_path.clone(),
    })?;

    let count = |file: &str| {
        let path = dir.join("data").join(file);
        crate::dataset::read::<IgnoredAny>(&path)
            .map(|items| items.len())
            .map_err(|error| tracing::warn!("leaving the example count out of the card: {error:#}"))
            .ok()
    };
    let metrics = Metrics::read(&dir.join(METRICS_FILE)).unwrap_or_else(|error| {
        tracing::warn!("leaving the losses and training time out of the card: {error:#}");
        Metrics::default()
    });
    let pod = PodRecord::load(runs, &record.id).unwrap_or_else(|error| {
        tracing::warn!("leaving the Runpod line out of the card: {error:#}");
        None
    });
    let runpod = pod.and_then(|pod| {
        Some(RunpodLine {
            gpu: pod.gpu_type?,
            spend_usd: pod.estimated_spend?,
        })
    });

    Ok(CardInput {
        repo: repo.clone(),
        base_model,
        adapter: outputs.adapter,
        license: None,
        parent: settings.roles.parent.model.clone(),
        generator: settings.roles.generator.model.clone(),
        topics: settings
            .topics
            .iter()
            .map(|t| (t.name.clone(), t.description.clone().unwrap_or_default()))
            .collect(),
        train_examples: count("train.jsonl"),
        eval_examples: count("eval.jsonl"),
        epochs: scalar("num_epochs").and_then(|v| v.parse::<f64>().ok()),
        learning_rate: scalar("learning_rate").filter(|v| v.parse::<f64>().is_ok()),
        sequence_len: scalar("sequence_len").and_then(|v| v.parse().ok()),
        train_loss: metrics.summary.last_train.and_then(|m| m.loss),
        eval_loss: metrics.summary.eval_loss,
        duration: metrics.duration,
        gguf: gguf_types(&record.id, files),
        runpod,
        reproduce_toml: reproduce_toml(settings),
    })
}

/// What a run's `metrics.jsonl` says.
#[derive(Default)]
struct Metrics {
    summary: MetricsSummary,
    /// From the first training log to the last.
    duration: Option<Duration>,
}

impl Metrics {
    /// Reads `path`; a missing file gives no values.
    fn read(path: &Path) -> Result<Self, CardError> {
        let content = match fs::read_to_string(path) {
            Ok(content) => content,
            Err(e) if e.kind() == io::ErrorKind::NotFound => String::new(),
            Err(source) => {
                return Err(CardError::Read {
                    path: path.to_path_buf(),
                    source,
                });
            },
        };
        let mut summary = MetricsSummary::default();
        let mut times = Vec::new();
        for line in content.lines().filter(|l| !l.trim().is_empty()) {
            if let Ok(MetricLine::Log(metric)) = parse_line(line) {
                times.push(metric.time);
            }
            summary.add(line);
        }
        let duration = match times.as_slice() {
            [first, .., last] => Duration::try_from_secs_f64(last - first).ok(),
            _ => None,
        };
        Ok(Self { summary, duration })
    }
}

/// The quantization types of the GGUF files among `files`, sorted: `TYPE`
/// of `<run-id>-<TYPE>.gguf`.
fn gguf_types(run_id: &str, files: &[UploadFile]) -> Vec<String> {
    let prefix = format!("{run_id}-");
    let mut types: Vec<String> = files
        .iter()
        .filter_map(|f| {
            let stem = f.path_in_repo.strip_suffix(".gguf")?;
            let kind = stem.strip_prefix(&prefix)?;
            (!kind.is_empty() && !kind.contains('/')).then(|| kind.to_string())
        })
        .collect();
    types.sort();
    types.dedup();
    types
}

/// The non-secret part of `settings`, as TOML: the topics, the roles (provider
/// names, never provider sections), `[training]` without `target`,
/// `hub_model_id` and `axolotl_extra`, and `[pipeline]`. Built key by key from
/// an allowlist, so nothing else of the configuration can reach it.
#[must_use]
pub fn reproduce_toml(settings: &Settings) -> String {
    let mut doc = DocumentMut::new();

    let mut topics = ArrayOfTables::new();
    for topic in &settings.topics {
        let mut table = Table::new();
        table["name"] = value(topic.name.as_str());
        if let Some(description) = &topic.description {
            table["description"] = value(description.as_str());
        }
        table["subtopics"] = value(i64::from(topic.subtopics));
        table["questions_per_subtopic"] = value(i64::from(topic.questions_per_subtopic));
        topics.push(table);
    }
    if !topics.is_empty() {
        doc["topics"] = Item::ArrayOfTables(topics);
    }

    let mut roles = Table::new();
    roles.set_implicit(true);
    for (name, role) in settings.roles.all() {
        roles[name] = Item::Table(role_table(role));
    }
    doc["roles"] = Item::Table(roles);

    if let Some(training) = &settings.training {
        doc["training"] = Item::Table(training_table(training));
    }
    doc["pipeline"] = Item::Table(pipeline_table(&settings.pipeline));
    doc.to_string()
}

fn role_table(role: &RoleModel) -> Table {
    let mut table = Table::new();
    table["provider"] = value(role.provider.as_str());
    table["model"] = value(role.model.as_str());
    table["reasoning"] = value(role.reasoning);
    table["max_tokens"] = value(i64::from(role.max_tokens));
    if let Some(temperature) = role.temperature {
        table["temperature"] = value(temperature);
    }
    if let Some(effort) = role.reasoning_effort {
        table["reasoning_effort"] = value(effort.as_str());
    }
    if let Some(budget) = role.thinking_budget {
        table["thinking_budget"] = value(i64::from(budget));
    }
    table
}

fn training_table(training: &Training) -> Table {
    let mut table = Table::new();
    table["base_model"] = value(shown_base(&training.base_model));
    table["adapter"] = value(match training.adapter {
        Adapter::Lora => "lora",
        Adapter::Qlora => "qlora",
        Adapter::Full => "full",
    });
    table["epochs"] = value(i64::from(training.epochs));
    table["learning_rate"] = value(training.learning_rate);
    table["lora_r"] = value(i64::from(training.lora_r));
    table["lora_alpha"] = value(i64::from(training.lora_alpha));
    table["lora_dropout"] = value(training.lora_dropout);
    table["sequence_len"] = value(i64::from(training.sequence_len));
    table["micro_batch_size"] = value(i64::from(training.micro_batch_size));
    table["gradient_accumulation_steps"] = value(i64::from(training.gradient_accumulation_steps));
    table["optimizer"] = value(training.optimizer.as_str());
    table["lr_scheduler"] = value(training.lr_scheduler.as_str());
    table["sample_packing"] = value(training.sample_packing);
    table["evals_per_epoch"] = value(i64::from(training.evals_per_epoch));
    table["saves_per_epoch"] = value(i64::from(training.saves_per_epoch));
    table["merge"] = value(training.merge);
    table
}

fn pipeline_table(pipeline: &Pipeline) -> Table {
    let mut table = Table::new();
    if let Ok(concurrency) = i64::try_from(pipeline.concurrency) {
        table["concurrency"] = value(concurrency);
    }
    table["max_retries"] = value(i64::from(pipeline.max_retries));
    table["dedup_threshold"] = value(pipeline.dedup_threshold);
    table["eval_ratio"] = value(pipeline.eval_ratio);
    if let Ok(seed) = i64::try_from(pipeline.seed) {
        table["seed"] = value(seed);
    }
    table["include_system_prompt"] = value(pipeline.include_system_prompt);
    table["embedding_threshold"] = value(pipeline.embedding_threshold);
    table["question_batch_size"] = value(i64::from(pipeline.question_batch_size));
    if let Ok(timeout) = i64::try_from(pipeline.request_timeout_secs) {
        table["request_timeout_secs"] = value(timeout);
    }
    table
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::fs;
    use std::path::PathBuf;

    use super::*;
    use crate::config::{EnvSource, load_str};
    use crate::runpod::PodRecord;
    use crate::runs::RunState;

    const SNAPSHOTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/snapshots/hub");

    fn snapshot(name: &str, card: &str) {
        let mut settings = insta::Settings::clone_current();
        settings.set_snapshot_path(SNAPSHOTS);
        settings.set_prepend_module_to_snapshot(false);
        settings.set_omit_expression(true);
        settings.bind(|| insta::assert_snapshot!(name.to_string(), card));
    }

    fn repo() -> RepoId {
        RepoId {
            namespace: "nayrosk".into(),
            name: "rust-mentor".into(),
        }
    }

    const REPRODUCE: &str = "[[topics]]\nname = \"ownership\"\n";

    /// The configuration the sample card's run was made with.
    const CARD_CONFIG: &str = r#"
[project]
name = "rust-mentor"

[[topics]]
name = "ownership"
description = "Rust ownership, borrowing and lifetimes"
subtopics = 8
questions_per_subtopic = 30

[[topics]]
name = "async"
description = "Async Rust with tokio: futures, tasks and cancellation"
subtopics = 8
questions_per_subtopic = 30

[providers.openrouter]
protocol = "openai"

[roles.generator]
provider = "openrouter"
model = "deepseek/deepseek-v4-flash"

[roles.parent]
provider = "openrouter"
model = "deepseek/deepseek-v4-flash:thinking"
reasoning = true

[training]
target = "cloud"
base_model = "Qwen/Qwen3-0.6B"
adapter = "qlora"
epochs = 3
learning_rate = 2e-4
sequence_len = 4096

[targets.cloud]
kind = "runpod"
gpu_types = ["NVIDIA GeForce RTX 4090"]
max_hours = 6
"#;

    fn qlora_on_runpod() -> CardInput {
        CardInput {
            repo: repo(),
            base_model: "Qwen/Qwen3-0.6B".into(),
            adapter: Adapter::Qlora,
            license: Some("apache-2.0".into()),
            parent: "deepseek/deepseek-v4-flash:thinking".into(),
            generator: "deepseek/deepseek-v4-flash".into(),
            topics: vec![
                (
                    "ownership".into(),
                    "Rust ownership, borrowing and lifetimes".into(),
                ),
                (
                    "async".into(),
                    "Async Rust with tokio: futures, tasks and cancellation".into(),
                ),
            ],
            train_examples: Some(412),
            eval_examples: Some(46),
            epochs: Some(3.0),
            learning_rate: Some("0.0002".into()),
            sequence_len: Some(4096),
            train_loss: Some(0.8412),
            eval_loss: Some(0.9071),
            duration: Some(Duration::from_secs(371)),
            gguf: vec!["Q4_K_M".into(), "Q8_0".into()],
            runpod: Some(RunpodLine {
                gpu: "NVIDIA GeForce RTX 4090".into(),
                spend_usd: 0.0123,
            }),
            reproduce_toml: REPRODUCE.into(),
        }
    }

    #[test]
    fn qlora_with_gguf_on_runpod() -> Result<(), Box<dyn Error>> {
        let settings = load_str(CARD_CONFIG, EnvSource::Vars(Vec::new()))?;
        let input = CardInput {
            reproduce_toml: reproduce_toml(&settings),
            ..qlora_on_runpod()
        };
        snapshot("qlora_with_gguf_on_runpod", &render(&input));
        Ok(())
    }

    #[test]
    fn the_reproduce_intro_names_what_is_left_out() {
        assert!(render(&qlora_on_runpod()).contains(
            "With these sections of overbrainer.toml (project, providers, targets and keys left out), then `overbrainer run`:\n"
        ));
    }

    #[test]
    fn a_full_fine_tune_is_named_as_such() {
        let input = CardInput {
            adapter: Adapter::Full,
            ..qlora_on_runpod()
        };
        let card = render(&input);
        assert!(card.contains(
            "Qwen/Qwen3-0.6B fully fine-tuned on 412 questions answered by deepseek/deepseek-v4-flash:thinking.\n"
        ));
        assert!(card.contains("| Adapter | none (full fine-tune) |"));
    }

    #[test]
    fn prose_cannot_open_html() {
        let input = CardInput {
            parent: "p<script>alert(1)</script>".into(),
            topics: vec![("t<b>".into(), "close <!-- the card".into())],
            ..qlora_on_runpod()
        };
        let card = render(&input);
        assert!(
            card.contains("- **t&lt;b&gt;**: close &lt;!-- the card\n"),
            "{card}"
        );
        assert!(
            card.contains("| Parent model | p&lt;script&gt;alert(1)&lt;/script&gt; |"),
            "{card}"
        );
        assert!(card.contains("answered by p&lt;script&gt;"), "{card}");
        assert!(
            !card.contains("<script>") && !card.contains("<!-- the"),
            "{card}"
        );
    }

    #[test]
    fn licenses_yaml_would_not_read_as_strings_are_quoted() {
        for (license, line) in [
            ("no", "license: \"no\"\n"),
            ("null", "license: \"null\"\n"),
            ("1.0", "license: \"1.0\"\n"),
            ("Off", "license: \"Off\"\n"),
            ("-x", "license: \"-x\"\n"),
            ("apache-2.0", "license: apache-2.0\n"),
        ] {
            let input = CardInput {
                license: Some(license.into()),
                ..qlora_on_runpod()
            };
            assert!(render(&input).contains(line), "{license}");
        }
    }

    #[test]
    fn full_without_gguf_local() {
        let input = CardInput {
            adapter: Adapter::Full,
            license: Some("mit".into()),
            duration: Some(Duration::from_mins(80)),
            gguf: Vec::new(),
            runpod: None,
            ..qlora_on_runpod()
        };
        snapshot("full_without_gguf_local", &render(&input));
    }

    #[test]
    fn unknown_license_and_no_metrics() {
        let input = CardInput {
            adapter: Adapter::Lora,
            license: None,
            topics: vec![("ownership".into(), String::new())],
            epochs: Some(2.5),
            learning_rate: Some("1.0e-5".into()),
            sequence_len: None,
            train_loss: None,
            eval_loss: None,
            duration: None,
            gguf: vec!["Q8_0".into(), "f16".into()],
            ..qlora_on_runpod()
        };
        snapshot("unknown_license_and_no_metrics", &render(&input));
    }

    #[test]
    fn without_gguf_an_adapter_is_loaded_with_peft() {
        let input = CardInput {
            gguf: Vec::new(),
            ..qlora_on_runpod()
        };
        let card = render(&input);
        assert!(card.contains(
            "PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(\"Qwen/Qwen3-0.6B\"), \"nayrosk/rust-mentor\")"
        ));
        assert!(!card.contains("ollama run"));
        assert!(!card.contains("- gguf\n"));
    }

    #[test]
    fn a_local_base_model_shows_only_its_name() -> Result<(), Box<dyn Error>> {
        let input = CardInput {
            base_model: "/home/me/models/qwen3-0.6b/".into(),
            gguf: Vec::new(),
            ..qlora_on_runpod()
        };
        let card = render(&input);
        assert!(!card.contains("/home/"), "{card}");
        assert!(
            card.contains(
                "a local model (`qwen3-0.6b`) fine-tuned (QLoRA) on 412 questions answered by"
            ),
            "{card}"
        );
        assert!(
            card.contains("| Base model | a local model (`qwen3-0.6b`) |"),
            "{card}"
        );
        assert!(
            card.contains(
                "model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(\"qwen3-0.6b\"), \"nayrosk/rust-mentor\")"
            ),
            "{card}"
        );
        assert!(!card.contains("base_model:"), "{card}");
        let settings = load_str(
            &CARD_CONFIG.replace("\"Qwen/Qwen3-0.6B\"", "\"/home/me/models/qwen3-0.6b\""),
            EnvSource::Vars(Vec::new()),
        )?;
        let toml = reproduce_toml(&settings);
        assert!(!toml.contains("/home/"), "{toml}");
        assert!(toml.contains("base_model = \"qwen3-0.6b\""), "{toml}");
        Ok(())
    }

    #[test]
    fn a_summary_without_a_count_names_the_answers() {
        for (adapter, line) in [
            (
                Adapter::Qlora,
                "Qwen/Qwen3-0.6B fine-tuned (QLoRA) on answers from p.\n",
            ),
            (
                Adapter::Full,
                "Qwen/Qwen3-0.6B fully fine-tuned on answers from p.\n",
            ),
        ] {
            let input = CardInput {
                adapter,
                parent: "p".into(),
                train_examples: None,
                ..qlora_on_runpod()
            };
            let card = render(&input);
            assert!(card.contains(line), "{card}");
        }
    }

    #[test]
    fn replaceable_needs_the_marker() {
        assert!(replaceable(None));
        assert!(replaceable(Some("# old\n\n<!-- overbrainer:card -->\n")));
        assert!(!replaceable(Some("# Hand written\n")));
        assert!(!replaceable(Some("")));
    }

    #[test]
    fn table_cells_escape_pipes() {
        let input = CardInput {
            parent: "a|b\nc".into(),
            topics: vec![("t".into(), "one\r\ntwo".into())],
            ..qlora_on_runpod()
        };
        let card = render(&input);
        assert!(card.contains("| Parent model | a\\|b c |"), "{card}");
        assert!(card.contains("- **t**: one two\n"), "{card}");
        assert!(card.contains("answered by a|b c."), "{card}");
    }

    #[test]
    fn tiny_spend_reads_less_than_a_cent() {
        let mut input = qlora_on_runpod();
        input.runpod = Some(RunpodLine {
            gpu: "NVIDIA A40".into(),
            spend_usd: 0.004,
        });
        assert!(render(&input).contains(
            "Trained on [Runpod](https://runpod.io?ref=ym24z23f) (NVIDIA A40) for less than $0.01."
        ));
        input.runpod = Some(RunpodLine {
            gpu: "NVIDIA A40".into(),
            spend_usd: 1.234,
        });
        assert!(render(&input).contains("(NVIDIA A40) for about $1.23."));
    }

    #[test]
    fn durations_use_the_largest_unit() {
        let cases = [
            (45, "45 s"),
            (371, "6 min"),
            (3600, "1 h"),
            (4800, "1 h 20 min"),
        ];
        for (secs, text) in cases {
            assert_eq!(duration_text(Duration::from_secs(secs)), text);
        }
    }

    const SECRET_CONFIG: &str = r#"
[project]
name = "demo"

[[topics]]
name = "ownership"
description = "Rust ownership"
subtopics = 3
questions_per_subtopic = 5

[providers.nanogpt]
protocol = "openai"

[roles.generator]
provider = "nanogpt"
model = "m1"

[roles.parent]
provider = "nanogpt"
model = "m2"
reasoning = true

[pipeline]
seed = 7

[training]
target = "box"
base_model = "Qwen/Qwen3-0.6B"
adapter = "qlora"
hub_model_id = "me/old-model"

[targets.box]
kind = "ssh"
runtime = "docker"
"#;

    fn secret_settings() -> Result<Settings, Box<dyn Error>> {
        let env = EnvSource::Vars(vec![
            (
                "OVERBRAINER_PROVIDERS__NANOGPT__API_KEY".into(),
                "sk-very-secret-key".into(),
            ),
            (
                "OVERBRAINER_PROVIDERS__NANOGPT__BASE_URL".into(),
                "https://private.example/v1".into(),
            ),
            (
                "OVERBRAINER_TARGETS__BOX__HOST".into(),
                "gpu.box.lan".into(),
            ),
        ]);
        Ok(load_str(SECRET_CONFIG, env)?)
    }

    #[test]
    fn reproduce_toml_has_no_secrets() -> Result<(), Box<dyn Error>> {
        let settings = secret_settings()?;
        let text = reproduce_toml(&settings);
        for secret in [
            "sk-very-secret-key",
            "https://private.example/v1",
            "private.example",
            "gpu.box.lan",
            "me/old-model",
            "[providers",
            "[targets",
            "[project",
            "target =",
        ] {
            assert!(!text.contains(secret), "the excerpt leaks a left-out value");
        }
        let parsed: toml_edit::DocumentMut = text.parse()?;
        assert_eq!(
            parsed["topics"][0]["description"].as_str(),
            Some("Rust ownership")
        );
        assert_eq!(
            parsed["roles"]["parent"]["provider"].as_str(),
            Some("nanogpt")
        );
        assert_eq!(parsed["roles"]["parent"]["reasoning"].as_bool(), Some(true));
        assert_eq!(parsed["training"]["adapter"].as_str(), Some("qlora"));
        assert_eq!(parsed["training"]["learning_rate"].as_float(), Some(2e-4));
        assert_eq!(parsed["pipeline"]["seed"].as_integer(), Some(7));
        Ok(())
    }

    fn write(path: PathBuf, text: &str) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, text)?;
        Ok(())
    }

    fn run_record(id: &str) -> RunRecord {
        RunRecord {
            id: id.to_string(),
            target: "cloud".to_string(),
            created: "2026-10-04T10:00:00Z".to_string(),
            remote_dir: format!("/workspace/{id}"),
            job: None,
            state: RunState::Succeeded,
            message: None,
            snapshot: None,
            resumed_from: None,
            snapshots: true,
        }
    }

    fn upload(path_in_repo: &str) -> UploadFile {
        UploadFile {
            local: PathBuf::from(path_in_repo),
            path_in_repo: path_in_repo.into(),
            size: 1,
        }
    }

    #[test]
    fn gather_reads_the_run_as_it_ran() -> Result<(), Box<dyn Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let id = "demo_20261004-100000";
        let record = run_record(id);
        let dir = runs.run_dir(id)?;
        // The run trained another base model, with other values, than the
        // current config says.
        write(
            dir.join("axolotl.yaml"),
            "base_model: \"Qwen/Qwen3-1.7B\"\nsequence_len: 2048\nadapter: \"lora\"\nnum_epochs: 2\nlearning_rate: 1.0e-5\n",
        )?;
        write(
            dir.join("metrics.jsonl"),
            concat!(
                "{\"event\": \"log\", \"time\": 1000.0, \"step\": 1, \"loss\": 2.5}\n",
                "{\"event\": \"log\", \"time\": 1390.0, \"step\": 40, \"loss\": 0.75, \"eval_loss\": null}\n",
                "{\"event\": \"log\", \"time\": 1395.0, \"step\": 40, \"eval_loss\": 0.875}\n",
            ),
        )?;
        write(dir.join("data/train.jsonl"), "{}\n{}\n{}\n")?;
        write(dir.join("data/eval.jsonl"), "{}\n")?;
        let mut pod = PodRecord::new(id, false, 1, "ssh-ed25519 AAAA");
        pod.gpu_type = Some("NVIDIA A40".into());
        pod.estimated_spend = Some(0.42);
        pod.save(&runs)?;
        let settings = secret_settings()?;
        let files = [
            upload("adapter_config.json"),
            upload(&format!("{id}-Q8_0.gguf")),
            upload(&format!("{id}-Q4_K_M.gguf")),
            upload("Modelfile"),
        ];

        let input = gather(&runs, &record, &settings, &repo(), &files)?;

        assert_eq!(input.base_model, "Qwen/Qwen3-1.7B");
        assert_eq!(input.adapter, Adapter::Lora);
        assert_eq!(input.license, None, "the caller looks the license up");
        assert_eq!(
            (input.parent.as_str(), input.generator.as_str()),
            ("m2", "m1")
        );
        assert_eq!(
            input.topics,
            vec![("ownership".to_string(), "Rust ownership".to_string())]
        );
        assert_eq!(
            (input.train_examples, input.eval_examples),
            (Some(3), Some(1))
        );
        assert_eq!(input.epochs, Some(2.0));
        assert_eq!(input.learning_rate.as_deref(), Some("1.0e-5"));
        assert_eq!(input.sequence_len, Some(2048));
        assert_eq!(
            (input.train_loss, input.eval_loss),
            (Some(0.75), Some(0.875))
        );
        assert_eq!(input.duration, Some(Duration::from_secs(395)));
        assert_eq!(input.gguf, vec!["Q4_K_M", "Q8_0"]);
        assert_eq!(
            input.runpod,
            Some(RunpodLine {
                gpu: "NVIDIA A40".into(),
                spend_usd: 0.42
            })
        );
        assert_eq!(input.reproduce_toml, reproduce_toml(&settings));
        Ok(())
    }

    #[test]
    fn gather_leaves_out_what_a_bare_run_lacks() -> Result<(), Box<dyn Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let id = "demo_20261004-110000";
        let dir = runs.run_dir(id)?;
        write(dir.join("axolotl.yaml"), "base_model: \"m\"\n")?;
        let input = gather(&runs, &run_record(id), &secret_settings()?, &repo(), &[])?;
        assert_eq!(input.adapter, Adapter::Full);
        assert_eq!(
            (input.train_examples, input.eval_examples),
            (Some(0), Some(0))
        );
        assert_eq!(
            (input.epochs, input.learning_rate, input.sequence_len),
            (None, None, None)
        );
        assert_eq!(
            (input.train_loss, input.eval_loss, input.duration),
            (None, None, None)
        );
        assert_eq!(input.gguf, Vec::<String>::new());
        assert_eq!(input.runpod, None);
        Ok(())
    }

    #[test]
    fn unreadable_run_files_drop_their_rows() -> Result<(), Box<dyn Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let id = "demo_20261004-130000";
        let dir = runs.run_dir(id)?;
        write(
            dir.join("axolotl.yaml"),
            "base_model: \"m\"\nadapter: \"lora\"\n",
        )?;
        write(dir.join("pod.json"), "{not json")?;
        write(dir.join("data/train.jsonl"), "garbage\n{}\n")?;
        // A directory where the metrics file should be cannot be read.
        fs::create_dir_all(dir.join("metrics.jsonl"))?;
        let input = gather(&runs, &run_record(id), &secret_settings()?, &repo(), &[])?;
        assert_eq!(input.runpod, None);
        assert_eq!((input.train_examples, input.eval_examples), (None, Some(0)));
        assert_eq!(
            (input.train_loss, input.eval_loss, input.duration),
            (None, None, None)
        );
        let card = render(&input);
        assert!(card.contains("| Examples | 0 eval |"), "{card}");
        assert!(
            card.contains("fine-tuned (LoRA) on answers from m2."),
            "{card}"
        );
        assert!(
            !card.contains("Final loss") && !card.contains("Trained on"),
            "{card}"
        );
        Ok(())
    }

    #[test]
    fn a_metrics_file_of_bad_lines_gives_no_rows() -> Result<(), Box<dyn Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let id = "demo_20261004-140000";
        let dir = runs.run_dir(id)?;
        write(dir.join("axolotl.yaml"), "base_model: \"m\"\n")?;
        write(
            dir.join("metrics.jsonl"),
            "not a record\n{\"event\": \"log\"}\n\u{0}\n",
        )?;
        let input = gather(&runs, &run_record(id), &secret_settings()?, &repo(), &[])?;
        assert_eq!(
            (input.train_loss, input.eval_loss, input.duration),
            (None, None, None)
        );
        Ok(())
    }

    #[test]
    fn one_training_log_gives_no_duration() -> Result<(), Box<dyn Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("metrics.jsonl");
        fs::write(
            &path,
            "{\"event\": \"log\", \"time\": 5.0, \"step\": 1, \"loss\": 1.5}\n",
        )?;
        let metrics = Metrics::read(&path)?;
        assert_eq!(metrics.duration, None);
        assert_eq!(metrics.summary.last_train.and_then(|m| m.loss), Some(1.5));
        Ok(())
    }

    #[test]
    fn gather_fails_without_the_run_config() -> Result<(), Box<dyn Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let id = "demo_20261004-120000";
        fs::create_dir_all(runs.run_dir(id)?)?;
        let result = gather(&runs, &run_record(id), &secret_settings()?, &repo(), &[]);
        assert!(result.is_err());
        Ok(())
    }
}
