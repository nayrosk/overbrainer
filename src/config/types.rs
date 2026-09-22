use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;
use std::str::FromStr;

use secrecy::SecretString;
use serde::de::{self, Unexpected, Visitor};
use serde::{Deserialize, Deserializer};

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

/// A provider and model pair assigned to a pipeline role.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoleModel {
    /// Name of the provider that serves this model.
    pub provider: String,
    /// Model identifier as understood by the provider.
    pub model: String,
    /// Whether to request reasoning output from the model.
    #[serde(default)]
    pub reasoning: bool,
}

/// Tunables that control question generation and answer collection.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Pipeline {
    /// Number of requests to run concurrently. Must be at least 1.
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
    /// Number of training epochs.
    #[serde(default = "default_epochs")]
    pub epochs: u32,
    /// Optimizer learning rate.
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    /// `LoRA` rank.
    #[serde(default = "default_lora_r")]
    pub lora_r: u32,
    /// Maximum training sequence length, in tokens.
    #[serde(default = "default_sequence_len")]
    pub sequence_len: u32,
    /// Whether to merge the adapter into the base model after training.
    #[serde(default)]
    pub merge: bool,
    /// Hugging Face Hub repo ID to push the trained model to, if any.
    pub hub_model_id: Option<String>,
    /// Passed through verbatim into the Axolotl YAML.
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
fn default_sequence_len() -> u32 {
    4096
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

/// Where a training job runs.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum Target {
    /// Runs on the machine executing overbrainer.
    Local {
        /// How the training job runs.
        runtime: Runtime,
        /// Required with `runtime = "native"` unless `axolotl` is on PATH.
        venv: Option<String>,
        /// Required with `runtime = "docker"`.
        image: Option<String>,
    },
    /// Runs on a remote machine reached over SSH.
    Ssh {
        /// How the training job runs.
        runtime: Runtime,
        /// Env only. `user@host` or an alias from `~/.ssh/config`.
        host: Option<String>,
        /// Required with `runtime = "native"` unless `axolotl` is on PATH.
        venv: Option<String>,
        /// Required with `runtime = "docker"`.
        image: Option<String>,
    },
    /// Runs on a Runpod-provisioned GPU pod.
    Runpod {
        /// Runpod GPU type identifier.
        gpu_type: String,
        /// Number of GPUs to provision. Must be at least 1.
        #[serde(default = "default_gpu_count", deserialize_with = "number")]
        gpu_count: u32,
        /// Container image to run.
        image: String,
        /// Container disk size, in gigabytes.
        #[serde(default = "default_container_disk_gb", deserialize_with = "number")]
        container_disk_gb: u32,
        /// Maximum pod lifetime, in hours. Must be greater than 0.
        #[serde(deserialize_with = "number")]
        max_hours: f64,
        /// Network volume to attach, if any.
        network_volume_id: Option<String>,
    },
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

/// Credentials for the Runpod API.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Runpod {
    /// Env only.
    pub api_key: Option<SecretString>,
}
