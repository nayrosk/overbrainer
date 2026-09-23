use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;
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
    /// Number of requests to run concurrently. Must be between 1 and 1024.
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
        /// Runpod GPU type IDs, tried in order until one can be placed. A TOML array,
        /// or a comma-separated string from env.
        #[serde(deserialize_with = "string_list")]
        gpu_types: Vec<String>,
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
        /// Runpod data centers the pod may be placed in, for example `EU-RO-1`.
        /// Any when empty.
        #[serde(default, deserialize_with = "string_list")]
        data_center_ids: Vec<String>,
        /// Network volume mounted at `/workspace/data`. Requires exactly one entry
        /// in `data_center_ids`, the volume's data center.
        network_volume_id: Option<String>,
    },
}

/// Image of a Runpod pod when a target sets none: Axolotl 0.19.0's cloud image for
/// CUDA 13 (NVIDIA driver 580 or newer), pinned by its index digest.
pub const DEFAULT_RUNPOD_IMAGE: &str = "axolotlai/axolotl-cloud-term:0.19.0-py3.12-cu130-2.12.1@sha256:f7b94da82913920a003e28e091d8528f57f76da7360fca87e95faf62fa32a680";

/// Virtual environment holding Axolotl in [`DEFAULT_RUNPOD_IMAGE`].
pub const DEFAULT_RUNPOD_VENV: &str = "/workspace/axolotl-venv";

/// Base URL of the Runpod REST API (v2) when `runpod.base_url` is unset.
pub const DEFAULT_RUNPOD_BASE_URL: &str = "https://api.runpod.io/v2";

/// Deserializes a list of strings given as a TOML array or, since env values
/// always arrive as strings, as one comma-separated string. Every item is trimmed.
fn string_list<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct ListVisitor;

    impl<'de> Visitor<'de> for ListVisitor {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a list of strings or a comma-separated string")
        }

        fn visit_str<E: de::Error>(self, value: &str) -> Result<Vec<String>, E> {
            if value.trim().is_empty() {
                return Ok(Vec::new());
            }
            Ok(value
                .split(',')
                .map(|item| item.trim().to_string())
                .collect())
        }

        fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Vec<String>, A::Error> {
            let mut items = Vec::new();
            while let Some(item) = seq.next_element::<String>()? {
                items.push(item.trim().to_string());
            }
            Ok(items)
        }
    }

    deserializer.deserialize_any(ListVisitor)
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
