use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;
use std::net::SocketAddr;
use std::str::FromStr;

use secrecy::SecretString;
use serde::de::{self, Unexpected, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

/// Fully resolved configuration: `overbrainer.toml` layered with `OVERBRAINER_*` env vars.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    /// Top-level project identity.
    pub project: Project,
    /// Subject areas to generate training questions about.
    #[serde(default)]
    pub topics: Vec<Topic>,
    /// Model providers, keyed by name.
    #[serde(default)]
    pub providers: BTreeMap<String, Provider>,
    /// The models used for each pipeline role.
    pub roles: Roles,
    /// Tunables that control question generation and answer collection.
    #[serde(default)]
    pub pipeline: Pipeline,
    /// Fine-tuning configuration for the child model. Absent when training is not configured.
    pub training: Option<Training>,
    /// Training targets, keyed by name.
    #[serde(default)]
    pub targets: BTreeMap<String, Target>,
    /// Credentials for the Runpod API.
    #[serde(default)]
    pub runpod: Runpod,
    /// Export of a trained model to GGUF, and its Ollama Modelfile.
    #[serde(default)]
    pub export: Export,
    /// Push of a run to the Hugging Face Hub.
    #[serde(default)]
    pub hub: Hub,
    /// The Prometheus endpoint, off unless `metrics.listen` is set.
    #[serde(default)]
    pub metrics: Metrics,
    /// Hugging Face access token. Env only.
    pub hf_token: Option<SecretString>,
    /// Log filter. Env only. Read directly from `OVERBRAINER_LOG` at startup; declared
    /// here so strict parsing accepts the variable.
    pub log: Option<String>,
}

/// Top-level project identity.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Project {
    /// Name of the project.
    pub name: String,
}

/// A subject area to generate training questions about.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Topic {
    /// Name of the topic. Must be unique across all topics.
    pub name: String,
    /// Optional human-readable description of the topic.
    pub description: Option<String>,
    /// Number of subtopics to generate. Must be at least 1.
    pub subtopics: u32,
    /// Number of questions to generate per subtopic. Must be at least 1.
    pub questions_per_subtopic: u32,
}

/// Wire protocol used to talk to a model provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Protocol {
    /// OpenAI-compatible chat completions API.
    Openai,
    /// Anthropic Messages API.
    Anthropic,
}

/// An API endpoint that serves one or more models.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provider {
    /// Wire protocol used to talk to this provider.
    pub protocol: Protocol,
    /// Env only.
    pub base_url: Option<String>,
    /// Env only. Literal value or `vault:<mount>/<path>#<field>`.
    pub api_key: Option<SecretString>,
}

/// The models used for each role in the pipeline.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Roles {
    /// Model that generates training questions.
    pub generator: RoleModel,
    /// Model that answers generated questions to produce training data.
    pub parent: RoleModel,
    /// Model used for deduplication embeddings, if enabled.
    pub embedder: Option<RoleModel>,
}

impl Roles {
    /// Every configured role with its name: `generator`, `parent`, then `embedder` when set.
    #[must_use]
    pub fn all(&self) -> Vec<(&'static str, &RoleModel)> {
        let mut roles = vec![("generator", &self.generator), ("parent", &self.parent)];
        if let Some(embedder) = &self.embedder {
            roles.push(("embedder", embedder));
        }
        roles
    }
}

/// A provider and model pair assigned to a pipeline role.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoleModel {
    /// Name of the provider that serves this model.
    pub provider: String,
    /// Model identifier as understood by the provider.
    pub model: String,
    /// Whether to request reasoning output from the model.
    #[serde(default)]
    pub reasoning: bool,
    /// Upper bound on generated tokens per request, reasoning included. Must be at least 1.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Sampling temperature in [0, 2]. The provider default applies when unset.
    pub temperature: Option<f64>,
    /// Reasoning effort. Only valid with `reasoning = true`.
    pub reasoning_effort: Option<Effort>,
    /// Fixed extended-thinking token budget, for the `anthropic` protocol only. Only
    /// valid with `reasoning = true`, must be at least 1024 and less than `max_tokens`,
    /// and cannot be combined with `reasoning_effort`.
    ///
    /// Absent (the default), overbrainer asks for adaptive thinking
    /// (`thinking: {"type": "adaptive"}`), which Claude Sonnet 5, Opus 5, Opus 4.8,
    /// Opus 4.7 and Fable 5.x require. Claude Opus 4.5, Sonnet 4.5 and Haiku 4.5 reject
    /// adaptive thinking instead and need this key set, which sends
    /// `thinking: {"type": "enabled", "budget_tokens": ...}` and omits
    /// `output_config.effort`.
    pub thinking_budget: Option<u32>,
}

fn default_max_tokens() -> u32 {
    16_384
}

/// Reasoning effort requested from a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    /// Short reasoning.
    Low,
    /// Balanced reasoning. Sent to `openai` providers when no effort is configured.
    Medium,
    /// Long reasoning.
    High,
}

impl Effort {
    /// The wire value: `low`, `medium` or `high`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// Tunables that control question generation and answer collection.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Pipeline {
    /// Number of requests to run concurrently in the questions and answers stages. Must be between 1 and 1024.
    pub concurrency: usize,
    /// Maximum number of retries for a failed request.
    pub max_retries: u32,
    /// Similarity threshold above which two questions are considered duplicates. Must be in (0, 1].
    pub dedup_threshold: f64,
    /// Fraction of collected data reserved for evaluation. Must be in (0, 1).
    pub eval_ratio: f64,
    /// Random seed used for deterministic sampling and splitting.
    pub seed: u64,
    /// Whether to include the system prompt when collecting answers.
    pub include_system_prompt: bool,
    /// Cosine similarity above which two questions are duplicates when `roles.embedder`
    /// is set. Must be in (0, 1].
    pub embedding_threshold: f64,
    /// Number of questions requested per generation call. Must be at least 1.
    pub question_batch_size: u32,
    /// Timeout of one LLM request, in seconds. Must be at least 1.
    pub request_timeout_secs: u64,
}

impl Default for Pipeline {
    fn default() -> Self {
        Self {
            concurrency: 8,
            max_retries: 5,
            dedup_threshold: 0.8,
            eval_ratio: 0.1,
            seed: 42,
            include_system_prompt: false,
            embedding_threshold: 0.9,
            question_batch_size: 10,
            request_timeout_secs: 600,
        }
    }
}

/// Fine-tuning strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Adapter {
    /// Low-Rank Adaptation.
    Lora,
    /// Quantized Low-Rank Adaptation.
    Qlora,
    /// Full fine-tuning of all model weights.
    Full,
}

/// Fine-tuning configuration for the child model.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Training {
    /// Name of the target the training job runs on.
    pub target: String,
    /// Hugging Face repo ID or a path on the training target.
    pub base_model: String,
    /// Fine-tuning strategy.
    pub adapter: Adapter,
    /// Number of training epochs. Must be at least 1.
    #[serde(default = "default_epochs")]
    pub epochs: u32,
    /// Optimizer learning rate. Must be greater than 0.
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    /// `LoRA` rank. Must be at least 1.
    #[serde(default = "default_lora_r")]
    pub lora_r: u32,
    /// `LoRA` scaling factor. Must be at least 1.
    #[serde(default = "default_lora_alpha")]
    pub lora_alpha: u32,
    /// `LoRA` dropout probability, in [0, 1).
    #[serde(default = "default_lora_dropout")]
    pub lora_dropout: f64,
    /// Maximum training sequence length, in tokens. Must be at least 1.
    #[serde(default = "default_sequence_len")]
    pub sequence_len: u32,
    /// Examples per GPU per step. Must be at least 1.
    #[serde(default = "default_micro_batch_size")]
    pub micro_batch_size: u32,
    /// Steps whose gradients are summed before each optimizer update. Must be at least 1.
    #[serde(default = "default_gradient_accumulation_steps")]
    pub gradient_accumulation_steps: u32,
    /// Axolotl optimizer name, for example `adamw_torch_fused` or `paged_adamw_8bit`.
    #[serde(default = "default_optimizer")]
    pub optimizer: String,
    /// Axolotl learning rate scheduler name, for example `cosine` or `linear`.
    #[serde(default = "default_lr_scheduler")]
    pub lr_scheduler: String,
    /// Packs several short examples into one sequence.
    #[serde(default = "default_sample_packing")]
    pub sample_packing: bool,
    /// Evaluations on `data/eval.jsonl` per epoch. Must be at least 1.
    #[serde(default = "default_evals_per_epoch")]
    pub evals_per_epoch: u32,
    /// Checkpoints saved per epoch. Must be at least 1.
    #[serde(default = "default_saves_per_epoch")]
    pub saves_per_epoch: u32,
    /// Whether to merge the adapter into the base model after training. Only with
    /// `adapter = "lora"` or `"qlora"`.
    #[serde(default)]
    pub merge: bool,
    /// Hugging Face Hub repo ID to push the trained model to, if any.
    pub hub_model_id: Option<String>,
    /// Merged into the Axolotl YAML: tables are merged key by key, any other value
    /// replaces the generated one.
    #[serde(default)]
    pub axolotl_extra: BTreeMap<String, serde_json::Value>,
}

fn default_epochs() -> u32 {
    3
}
fn default_learning_rate() -> f64 {
    2e-4
}
fn default_lora_r() -> u32 {
    16
}
fn default_lora_alpha() -> u32 {
    32
}
fn default_lora_dropout() -> f64 {
    0.05
}
fn default_sequence_len() -> u32 {
    4096
}
fn default_micro_batch_size() -> u32 {
    2
}
fn default_gradient_accumulation_steps() -> u32 {
    4
}
fn default_optimizer() -> String {
    "adamw_torch_fused".to_string()
}
fn default_lr_scheduler() -> String {
    "cosine".to_string()
}
fn default_sample_packing() -> bool {
    true
}
fn default_evals_per_epoch() -> u32 {
    4
}
fn default_saves_per_epoch() -> u32 {
    1
}

/// How a training target runs the fine-tuning job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Runtime {
    /// Runs inside a container image.
    Docker,
    /// Runs directly on the target's host, optionally inside a virtual environment.
    Native,
}

/// Container engine used by the `docker` runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    /// Docker with the NVIDIA Container Toolkit (`--gpus all`).
    Docker,
    /// Podman with NVIDIA CDI devices (`--device nvidia.com/gpu=all`).
    Podman,
}

impl Engine {
    /// The command name: `docker` or `podman`.
    #[must_use]
    pub fn command(self) -> &'static str {
        match self {
            Self::Docker => "docker",
            Self::Podman => "podman",
        }
    }
}

/// Image used by the `docker` runtime when a target sets none: Axolotl 0.19.0 for
/// CUDA 13 (NVIDIA driver 580 or newer), pinned by digest.
pub const DEFAULT_IMAGE: &str = "axolotlai/axolotl:0.19.0-py3.12-cu130-2.12.1@sha256:9de7c7a5b8830480a7d2eb3b6d49759586615f5f8eb1126d5df29f8bd9fa324b";

/// Directory of an `ssh` target that holds the runs when `workdir` is unset,
/// relative to the remote user's home directory.
pub const DEFAULT_WORKDIR: &str = "overbrainer";

/// Where a training job runs.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum Target {
    /// Runs on the machine executing overbrainer, in the project's `runs/` directory.
    Local {
        /// How the training job runs.
        runtime: Runtime,
        /// Container engine, only with `runtime = "docker"`. Defaults to `docker`.
        engine: Option<Engine>,
        /// Container image, only with `runtime = "docker"`. Defaults to [`DEFAULT_IMAGE`].
        image: Option<String>,
        /// Virtual environment holding `bin/axolotl`, only with `runtime = "native"`.
        /// Without it, `axolotl` must be on `PATH`.
        venv: Option<String>,
    },
    /// Runs on a remote machine reached over SSH.
    Ssh {
        /// How the training job runs.
        runtime: Runtime,
        /// Env only. `user@host` or an alias from `~/.ssh/config`.
        host: Option<String>,
        /// Directory holding the runs on the remote machine, absolute or relative to
        /// the remote home directory. Defaults to [`DEFAULT_WORKDIR`].
        workdir: Option<String>,
        /// Container engine, only with `runtime = "docker"`. Defaults to `docker`.
        engine: Option<Engine>,
        /// Container image, only with `runtime = "docker"`. Defaults to [`DEFAULT_IMAGE`].
        image: Option<String>,
        /// Virtual environment holding `bin/axolotl`, only with `runtime = "native"`.
        /// Without it, `axolotl` must be on the remote `PATH`.
        venv: Option<String>,
    },
    /// Runs on a Runpod GPU pod, created for the run and deleted after it.
    Runpod {
        /// Runpod GPU type IDs, tried in order until one can be placed, or `"auto"`
        /// to try every GPU type in stock, cheapest first, when the run starts. A
        /// TOML array, or a comma-separated string from env.
        gpu_types: ListOrAuto,
        /// Least VRAM per GPU, in GB, for `gpu_types = "auto"` only. At least 1.
        #[serde(default, deserialize_with = "optional_number")]
        min_vram_gb: Option<u32>,
        /// Highest list price of one GPU, in USD per hour, for `gpu_types = "auto"`
        /// only. Greater than 0.
        #[serde(default, deserialize_with = "optional_number")]
        max_price_per_hour: Option<f64>,
        /// Number of GPUs to provision. Must be at least 1.
        #[serde(default = "default_gpu_count", deserialize_with = "number")]
        gpu_count: u32,
        /// Container image of the pod. Defaults to [`DEFAULT_RUNPOD_IMAGE`].
        image: Option<String>,
        /// Virtual environment on the pod holding `bin/axolotl`, absolute. Defaults
        /// to [`DEFAULT_RUNPOD_VENV`].
        venv: Option<String>,
        /// Container disk size, in gigabytes. Must be at least 20.
        #[serde(default = "default_container_disk_gb", deserialize_with = "number")]
        container_disk_gb: u32,
        /// Hours after which the pod watchdog deletes the pod, whatever it is doing.
        /// Must be greater than 0 and at most 720. Not applied to a pod kept with
        /// `--keep-pod`.
        #[serde(deserialize_with = "number")]
        max_hours: f64,
        /// Most a run may spend on its pod, in USD, at the pod's hourly rate: at
        /// 95% the job is stopped with a snapshot, at 100% the watchdog deletes the
        /// pod. Greater than 0. None by default. Not applied to a pod kept with
        /// `--keep-pod`.
        #[serde(default, deserialize_with = "optional_number")]
        max_cost_usd: Option<f64>,
        /// Minutes the watchdog waits for a job to start before deleting the pod.
        /// Must be at least 5.
        #[serde(default = "default_boot_grace_minutes", deserialize_with = "number")]
        boot_grace_minutes: u32,
        /// Minutes the watchdog keeps a pod whose ended job was not retrieved. Must
        /// be at least 1.
        #[serde(
            default = "default_retrieve_grace_minutes",
            deserialize_with = "number"
        )]
        retrieve_grace_minutes: u32,
        /// Runpod data centers the pod may be placed in, for example `EU-RO-1`, any
        /// when empty; or `"auto"` for those with a chosen GPU type in stock when
        /// the run starts, cheapest first.
        #[serde(default)]
        data_center_ids: ListOrAuto,
        /// Network volume mounted at `/workspace/data`. Requires exactly one entry
        /// in `data_center_ids`, the volume's data center (never `"auto"`).
        network_volume_id: Option<String>,
        /// Largest size, in GB, overbrainer may grow the network volume to when
        /// the run fills it; without it, a full disk stops the job with a
        /// snapshot. A grow is permanent: the volume never shrinks and stays
        /// billed at its new size. At least 1, and only with
        /// `network_volume_id`. None by default.
        #[serde(default, deserialize_with = "optional_number")]
        max_volume_gb: Option<u32>,
    },
}

/// Image of a Runpod pod when a target sets none: Axolotl 0.19.0's cloud image for
/// CUDA 13 (NVIDIA driver 580 or newer), pinned by its index digest.
pub const DEFAULT_RUNPOD_IMAGE: &str = "axolotlai/axolotl-cloud-term:0.19.0-py3.12-cu130-2.12.1@sha256:f7b94da82913920a003e28e091d8528f57f76da7360fca87e95faf62fa32a680";

/// Virtual environment holding Axolotl in [`DEFAULT_RUNPOD_IMAGE`].
pub const DEFAULT_RUNPOD_VENV: &str = "/workspace/axolotl-venv";

/// Base URL of the Runpod REST API (v2) when `runpod.base_url` is unset.
pub const DEFAULT_RUNPOD_BASE_URL: &str = "https://api.runpod.io/v2";

/// A list of strings, or `"auto"` for a choice made when a run starts.
///
/// Read from a TOML array or, since env values always arrive as strings, from
/// one comma-separated string; every item is trimmed. Only the exact lower-case
/// string `auto` (surrounding spaces aside) means [`ListOrAuto::Auto`]: `["auto"]`
/// is a list, which validation refuses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ListOrAuto {
    /// Chosen from the Runpod catalog when the run starts.
    Auto,
    /// These items, in order.
    List(Vec<String>),
}

impl ListOrAuto {
    /// The value that selects [`ListOrAuto::Auto`].
    pub const AUTO: &'static str = "auto";

    /// Whether the choice is left to the run's start.
    #[must_use]
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }

    /// The listed items; none for [`ListOrAuto::Auto`].
    #[must_use]
    pub fn list(&self) -> &[String] {
        match self {
            Self::Auto => &[],
            Self::List(items) => items,
        }
    }

    /// Whether this is the default, empty list: no items given, and not
    /// `auto`, so any value is accepted.
    #[must_use]
    pub fn is_any(&self) -> bool {
        *self == Self::default()
    }

    /// `text` as a form holds it: `auto`, or items comma-separated. Unlike
    /// deserialization, empty items are dropped.
    #[must_use]
    pub fn from_form_text(text: &str) -> Self {
        if text.trim() == Self::AUTO {
            return Self::Auto;
        }
        Self::List(
            text.split(',')
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(str::to_string)
                .collect(),
        )
    }
}

impl Default for ListOrAuto {
    fn default() -> Self {
        Self::List(Vec::new())
    }
}

impl fmt::Display for ListOrAuto {
    /// `auto`, or the items joined by `, `.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => formatter.write_str(Self::AUTO),
            Self::List(items) => formatter.write_str(&items.join(", ")),
        }
    }
}

impl<'de> Deserialize<'de> for ListOrAuto {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ListVisitor;

        impl<'de> Visitor<'de> for ListVisitor {
            type Value = ListOrAuto;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("\"auto\", a list of strings or a comma-separated string")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<ListOrAuto, E> {
                let value = value.trim();
                if value == ListOrAuto::AUTO {
                    return Ok(ListOrAuto::Auto);
                }
                if value.is_empty() {
                    return Ok(ListOrAuto::default());
                }
                Ok(ListOrAuto::List(
                    value
                        .split(',')
                        .map(|item| item.trim().to_string())
                        .collect(),
                ))
            }

            fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<ListOrAuto, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element::<String>()? {
                    items.push(item.trim().to_string());
                }
                Ok(ListOrAuto::List(items))
            }
        }

        deserializer.deserialize_any(ListVisitor)
    }
}

/// [`number`] for an optional field, used with `#[serde(default)]`.
fn optional_number<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: FromStr,
{
    number(deserializer).map(Some)
}

/// Deserializes a number that may arrive as a string.
///
/// `Target` is an internally tagged enum, so serde buffers its fields before
/// deserializing them, and the environment source (with `try_parsing(false)`, which
/// keeps secrets such as `0123` intact) delivers every value as a string. The config
/// crate's own string-to-number coercion is lost in that buffering, so numeric fields
/// of a target accept both forms here.
fn number<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: FromStr,
{
    struct NumberVisitor<T>(PhantomData<T>);

    impl<T: FromStr> NumberVisitor<T> {
        fn parse<E: de::Error>(&self, text: &str, unexpected: Unexpected<'_>) -> Result<T, E> {
            text.trim()
                .parse()
                .map_err(|_| E::invalid_value(unexpected, self))
        }
    }

    impl<T: FromStr> Visitor<'_> for NumberVisitor<T> {
        type Value = T;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a number")
        }

        fn visit_u64<E: de::Error>(self, value: u64) -> Result<T, E> {
            self.parse(&value.to_string(), Unexpected::Unsigned(value))
        }

        fn visit_i64<E: de::Error>(self, value: i64) -> Result<T, E> {
            self.parse(&value.to_string(), Unexpected::Signed(value))
        }

        fn visit_f64<E: de::Error>(self, value: f64) -> Result<T, E> {
            self.parse(&value.to_string(), Unexpected::Float(value))
        }

        fn visit_str<E: de::Error>(self, value: &str) -> Result<T, E> {
            self.parse(value, Unexpected::Str(value))
        }
    }

    deserializer.deserialize_any(NumberVisitor(PhantomData))
}

fn default_gpu_count() -> u32 {
    1
}
fn default_container_disk_gb() -> u32 {
    50
}
fn default_boot_grace_minutes() -> u32 {
    30
}
fn default_retrieve_grace_minutes() -> u32 {
    60
}

/// Access to the Runpod API.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Runpod {
    /// Env only. Literal value or `vault:<mount>/<path>#<field>`.
    pub api_key: Option<SecretString>,
    /// Env only. Base URL of the REST API, [`DEFAULT_RUNPOD_BASE_URL`] when unset.
    /// Must be `https`, or `http` on a loopback host (a test stub).
    pub base_url: Option<String>,
}

/// The llama-quantize types `export.quantize` accepts. `F16` and `BF16` skip
/// the quantization: the GGUF keeps 16-bit weights.
pub const QUANTIZE_TYPES: [&str; 10] = [
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "Q3_K_M", "Q2_K", "F16", "BF16",
];

/// `export.quantize` when unset.
pub const DEFAULT_QUANTIZE: &str = "Q4_K_M";

/// Export of a trained model to GGUF with llama.cpp, and its Ollama Modelfile.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Export {
    /// Whether a training job exports its model at its end, after the merge.
    #[serde(default)]
    pub after_training: bool,
    /// The llama-quantize type of the GGUF, one of [`QUANTIZE_TYPES`].
    #[serde(default = "default_quantize")]
    pub quantize: String,
    /// The Ollama model `ollama create` makes from the Modelfile once an
    /// export in a training job is retrieved, when `ollama` is on `PATH`.
    pub ollama_name: Option<String>,
}

impl Default for Export {
    fn default() -> Self {
        Self {
            after_training: false,
            quantize: default_quantize(),
            ollama_name: None,
        }
    }
}

/// Push of a run to the Hugging Face Hub: `overbrainer push`.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Hub {
    /// Repo to push to, `NAMESPACE/NAME`; default `<whoami>/<project name>`.
    #[serde(default)]
    pub repo: Option<String>,
    /// Create the repo private (default true).
    #[serde(default = "default_true")]
    pub private: bool,
    /// Push each run once it succeeded and came back (and was exported).
    #[serde(default)]
    pub after_training: bool,
    /// Env only. Base URL of the Hub, `https://huggingface.co` when unset.
    /// Must be `https`, or `http` on a loopback host (a test stub).
    #[serde(default)]
    pub base_url: Option<String>,
}

impl Default for Hub {
    fn default() -> Self {
        Self {
            repo: None,
            private: true,
            after_training: false,
            base_url: None,
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_quantize() -> String {
    DEFAULT_QUANTIZE.to_string()
}

/// A part of an Ollama model name, as Ollama's `types/model/name.go` tells
/// them apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OllamaPart {
    Host,
    Namespace,
    Model,
    Tag,
}

/// Whether `text` is a valid `kind` part of an Ollama model name, by
/// `isValidPart` of Ollama's `types/model/name.go`: 1 to 80 characters (350
/// for a host), the first an ASCII letter, digit or `_`, the others also `-`,
/// `.` (not in a namespace) or `:` (in a host only).
fn ollama_part(kind: OllamaPart, text: &str) -> bool {
    let max = if kind == OllamaPart::Host { 350 } else { 80 };
    let word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    (1..=max).contains(&text.len())
        && text.starts_with(word)
        && text.chars().all(|c| {
            word(c)
                || c == '-'
                || (c == '.' && kind != OllamaPart::Namespace)
                || (c == ':' && kind == OllamaPart::Host)
        })
}

/// Whether `name` is a valid Ollama model name, `[host/][namespace/]model[:tag]`,
/// split and checked as Ollama's `types/model/name.go` does: the tag after
/// the last `:` when it comes after the last `/`, the model after the last
/// `/`, then the namespace, then the host (an optional `scheme://` before it
/// is dropped). Each part present must be valid (see `ollama_part`): an
/// empty one, such as in `a//b` or `a/`, is refused. Such a name is also
/// safe on a command line.
#[must_use]
pub fn is_ollama_name(name: &str) -> bool {
    let (rest, tag) = match (name.rfind(':'), name.rfind('/')) {
        (Some(colon), slash) if slash.is_none_or(|slash| colon > slash) => {
            (&name[..colon], Some(&name[colon + 1..]))
        },
        _ => (name, None),
    };
    let (rest, model) = match rest.rsplit_once('/') {
        Some((rest, model)) => (Some(rest), model),
        None => (None, rest),
    };
    let (host, namespace) = match rest.map(|rest| rest.rsplit_once('/')) {
        None => (None, None),
        Some(None) => (None, rest),
        Some(Some((host, namespace))) => (Some(host), Some(namespace)),
    };
    let host = host.map(|host| host.split_once("://").map_or(host, |(_, host)| host));
    ollama_part(OllamaPart::Model, model)
        && tag.is_none_or(|tag| ollama_part(OllamaPart::Tag, tag))
        && namespace.is_none_or(|namespace| ollama_part(OllamaPart::Namespace, namespace))
        && host.is_none_or(|host| ollama_part(OllamaPart::Host, host))
}

/// The Prometheus endpoint.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metrics {
    /// Address `GET /metrics` listens on while a command holds the project, such
    /// as `127.0.0.1:9464`. No endpoint when unset.
    pub listen: Option<SocketAddr>,
}
