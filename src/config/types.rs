use std::collections::BTreeMap;

use secrecy::SecretString;
use serde::Deserialize;

/// Fully resolved configuration: `overbrainer.toml` layered with `OVERBRAINER_*` env vars.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub project: Project,
    #[serde(default)]
    pub topics: Vec<Topic>,
    #[serde(default)]
    pub providers: BTreeMap<String, Provider>,
    pub roles: Roles,
    #[serde(default)]
    pub pipeline: Pipeline,
    pub training: Option<Training>,
    #[serde(default)]
    pub targets: BTreeMap<String, Target>,
    #[serde(default)]
    pub runpod: Runpod,
    pub hf_token: Option<SecretString>,
    /// Log filter. Read directly from `OVERBRAINER_LOG` at startup; declared here so
    /// strict parsing accepts the variable.
    pub log: Option<String>,
}

/// Top-level project identity.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Project {
    pub name: String,
}

/// A subject area to generate training questions about.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Topic {
    pub name: String,
    pub description: Option<String>,
    pub subtopics: u32,
    pub questions_per_subtopic: u32,
}

/// Wire protocol used to talk to a model provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Protocol {
    Openai,
    Anthropic,
}

/// An API endpoint that serves one or more models.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provider {
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
    pub generator: RoleModel,
    pub parent: RoleModel,
    pub embedder: Option<RoleModel>,
}

/// A provider and model pair assigned to a pipeline role.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoleModel {
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub reasoning: bool,
}

/// Tunables that control question generation and answer collection.
#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Pipeline {
    pub concurrency: usize,
    pub max_retries: u32,
    pub dedup_threshold: f64,
    pub eval_ratio: f64,
    pub seed: u64,
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
    Lora,
    Qlora,
    Full,
}

/// Fine-tuning configuration for the child model.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Training {
    pub target: String,
    /// Hugging Face repo ID or a path on the training target.
    pub base_model: String,
    pub adapter: Adapter,
    #[serde(default = "default_epochs")]
    pub epochs: u32,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_lora_r")]
    pub lora_r: u32,
    #[serde(default = "default_sequence_len")]
    pub sequence_len: u32,
    #[serde(default)]
    pub merge: bool,
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
    Docker,
    Native,
}

/// Where a training job runs.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum Target {
    Local {
        runtime: Runtime,
        /// Required with `runtime = "native"` unless `axolotl` is on PATH.
        venv: Option<String>,
        /// Required with `runtime = "docker"`.
        image: Option<String>,
    },
    Ssh {
        runtime: Runtime,
        /// Env only. `user@host` or an alias from `~/.ssh/config`.
        host: Option<String>,
        venv: Option<String>,
        image: Option<String>,
    },
    Runpod {
        gpu_type: String,
        #[serde(default = "default_gpu_count")]
        gpu_count: u32,
        image: String,
        #[serde(default = "default_container_disk_gb")]
        container_disk_gb: u32,
        max_hours: f64,
        network_volume_id: Option<String>,
    },
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
