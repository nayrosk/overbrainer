//! Schema of the fields a form may edit in `overbrainer.toml`: kind, bounds, choices.
//!
//! Env-only keys (provider `base_url` and `api_key`, target `host`, `runpod.*`,
//! `hf_token`, `log`), the target `kind` tag and the free-form
//! `training.axolotl_extra` are not listed: a form never writes them. Bounds mirror
//! `validate.rs`, open or closed ends included; [`FieldKind::describe`] words them.

use crate::config::{ListOrAuto, QUANTIZE_TYPES};

/// A part of the configuration whose fields share one schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    /// `[project]`.
    Project,
    /// One `[[topics]]` entry.
    Topic,
    /// One `[providers.<name>]` table.
    Provider,
    /// One of `roles.generator`, `roles.parent`, `roles.embedder`.
    Role,
    /// `[pipeline]`.
    Pipeline,
    /// `[training]`.
    Training,
    /// One `[targets.<name>]` table of the given kind.
    Target(TargetKind),
    /// `[export]`.
    Export,
    /// `[hub]`.
    Hub,
    /// `[metrics]`.
    Metrics,
}

/// The `kind` of a training target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetKind {
    /// Runs on this machine.
    Local,
    /// Runs on a machine reached over SSH.
    Ssh,
    /// Runs on a Runpod pod.
    Runpod,
}

impl TargetKind {
    /// Every kind, in the order a form offers them.
    pub const ALL: [Self; 3] = [Self::Local, Self::Ssh, Self::Runpod];

    /// The `kind` value in `overbrainer.toml`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Ssh => "ssh",
            Self::Runpod => "runpod",
        }
    }

    /// The kind whose `kind` value is `text`.
    #[must_use]
    pub fn from_name(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == text)
    }
}

/// What a field holds and which values it accepts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FieldKind {
    /// Any string.
    Text,
    /// An integer in `[min, max]`.
    Int {
        /// Smallest accepted value.
        min: i64,
        /// Largest accepted value.
        max: i64,
    },
    /// A finite float between `min` and `max`.
    Float {
        /// Lower end.
        min: Bound,
        /// Upper end.
        max: Bound,
    },
    /// `true` or `false`.
    Bool,
    /// One of these strings.
    Choice(&'static [&'static str]),
    /// `auto` (a [`FieldValue::Text`]) or a list of strings typed as
    /// comma-separated text; see [`ListOrAuto`].
    ListOrAuto,
}

/// One end of a float range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bound {
    /// The end is accepted.
    Incl(f64),
    /// The end is refused.
    Excl(f64),
    /// No end.
    Unbounded,
}

impl Bound {
    fn below(self, number: f64) -> bool {
        match self {
            Self::Incl(end) => end <= number,
            Self::Excl(end) => end < number,
            Self::Unbounded => true,
        }
    }

    fn above(self, number: f64) -> bool {
        match self {
            Self::Incl(end) => number <= end,
            Self::Excl(end) => number < end,
            Self::Unbounded => true,
        }
    }
}

/// A typed value for a field. Text is always a string, never parsed as TOML.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    /// A string, for `Text` and `Choice` fields.
    Text(String),
    /// An integer.
    Int(i64),
    /// A float.
    Float(f64),
    /// A boolean.
    Bool(bool),
    /// A list of strings.
    List(Vec<String>),
}

/// Why a value does not fit a field. No message quotes the value.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FieldError {
    /// Text typed for an integer field is not a whole number.
    #[error("must be a whole number")]
    NotInt,
    /// Text typed for a float field is not a number.
    #[error("must be a number")]
    NotNumber,
    /// Text typed for a boolean field is neither `true` nor `false`.
    #[error("must be true or false")]
    NotBool,
    /// A number outside the field's bounds, with the bounds in words.
    #[error("must be {0}")]
    OutOfRange(String),
    /// Text outside the field's choices, with the choices in words.
    #[error("must be {0}")]
    NotAChoice(String),
    /// A value of another kind than the field's.
    #[error("has the wrong type")]
    WrongType,
    /// A required field cannot be removed.
    #[error("is required and cannot be removed")]
    Required,
}

impl FieldKind {
    /// Reads `text`, as typed in a form, into a value of this kind.
    ///
    /// # Errors
    ///
    /// Returns why `text` is not accepted. The message never quotes `text`.
    pub fn parse(self, text: &str) -> Result<FieldValue, FieldError> {
        let value = match self {
            Self::Text => FieldValue::Text(text.to_string()),
            Self::Int { .. } => text
                .trim()
                .parse()
                .map(FieldValue::Int)
                .map_err(|_| FieldError::NotInt)?,
            Self::Float { .. } => text
                .trim()
                .parse()
                .map(FieldValue::Float)
                .map_err(|_| FieldError::NotNumber)?,
            Self::Bool => match text.trim() {
                "true" => FieldValue::Bool(true),
                "false" => FieldValue::Bool(false),
                _ => return Err(FieldError::NotBool),
            },
            Self::Choice(_) => FieldValue::Text(text.trim().to_string()),
            Self::ListOrAuto => match ListOrAuto::from_form_text(text) {
                ListOrAuto::Auto => FieldValue::Text(ListOrAuto::AUTO.to_string()),
                ListOrAuto::List(items) => FieldValue::List(items),
            },
        };
        self.check(&value)?;
        Ok(value)
    }

    /// Checks that `value` has this kind and is within its bounds or choices.
    ///
    /// # Errors
    ///
    /// Returns why `value` is not accepted. The message never quotes the value.
    pub fn check(self, value: &FieldValue) -> Result<(), FieldError> {
        match (self, value) {
            (Self::Text, FieldValue::Text(_))
            | (Self::Bool, FieldValue::Bool(_))
            | (Self::ListOrAuto, FieldValue::List(_)) => Ok(()),
            (Self::ListOrAuto, FieldValue::Text(text)) if text == ListOrAuto::AUTO => Ok(()),
            (Self::Int { min, max }, FieldValue::Int(number)) if (min..=max).contains(number) => {
                Ok(())
            },
            (Self::Float { min, max }, FieldValue::Float(number))
                if number.is_finite() && min.below(*number) && max.above(*number) =>
            {
                Ok(())
            },
            (Self::Choice(choices), FieldValue::Text(text)) if choices.contains(&text.as_str()) => {
                Ok(())
            },
            (Self::Int { .. }, FieldValue::Int(_)) | (Self::Float { .. }, FieldValue::Float(_)) => {
                Err(FieldError::OutOfRange(self.describe()))
            },
            (Self::Choice(_) | Self::ListOrAuto, FieldValue::Text(_)) => {
                Err(FieldError::NotAChoice(self.describe()))
            },
            _ => Err(FieldError::WrongType),
        }
    }

    /// The accepted values in words, for a form's hint and error messages:
    /// `at least 1`, `in (0, 1]`, `one of lora, qlora, full`.
    #[must_use]
    pub fn describe(self) -> String {
        match self {
            Self::Text => "any text".to_string(),
            Self::Int {
                min,
                max: i64::MAX | U32_MAX,
            } => format!("at least {min}"),
            Self::Int { min, max } => format!("between {min} and {max}"),
            Self::Float { min, max } => describe_floats(min, max),
            Self::Bool => "true or false".to_string(),
            Self::Choice(choices) => format!("one of {}", choices.join(", ")),
            Self::ListOrAuto => "auto or a comma-separated list".to_string(),
        }
    }
}

/// A float range in words: `in [0, 2]`, `in (0, 1)`, `greater than 0`.
fn describe_floats(min: Bound, max: Bound) -> String {
    let lower = match min {
        Bound::Incl(end) => Some(("[", end)),
        Bound::Excl(end) => Some(("(", end)),
        Bound::Unbounded => None,
    };
    let upper = match max {
        Bound::Incl(end) => Some((end, "]")),
        Bound::Excl(end) => Some((end, ")")),
        Bound::Unbounded => None,
    };
    match (lower, upper) {
        (Some((open, low)), Some((high, close))) => format!("in {open}{low}, {high}{close}"),
        (Some(("(", low)), None) => format!("greater than {low}"),
        (Some((_, low)), None) => format!("at least {low}"),
        (None, Some((high, "]"))) => format!("at most {high}"),
        (None, Some((high, _))) => format!("less than {high}"),
        (None, None) => "a number".to_string(),
    }
}

/// One editable field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldSpec {
    /// The key in its table.
    pub name: &'static str,
    /// What the field holds.
    pub kind: FieldKind,
    /// Whether the key may be left out of the file (an `Option` or a default applies).
    pub optional: bool,
    /// One line for the form.
    pub help: &'static str,
}

/// Largest `u32`, as the `i64` bound of a `u32` field.
const U32_MAX: i64 = u32::MAX as i64;

/// Highest `pipeline.concurrency`, as in `validate.rs`.
const MAX_CONCURRENCY: i64 = 1024;
/// Lowest `thinking_budget`, as in `validate.rs`.
const MIN_THINKING_BUDGET: i64 = 1024;
/// Longest runpod `max_hours`, as in `validate.rs`.
const MAX_RUNPOD_HOURS: f64 = 720.0;
/// Smallest runpod `container_disk_gb`, as in `validate.rs`.
const MIN_CONTAINER_DISK_GB: i64 = 20;
/// Smallest runpod `boot_grace_minutes`, as in `validate.rs`.
const MIN_BOOT_GRACE_MINUTES: i64 = 5;

const fn spec(
    name: &'static str,
    kind: FieldKind,
    optional: bool,
    help: &'static str,
) -> FieldSpec {
    FieldSpec {
        name,
        kind,
        optional,
        help,
    }
}

const TEXT: FieldKind = FieldKind::Text;
const BOOL: FieldKind = FieldKind::Bool;
const LIST_OR_AUTO: FieldKind = FieldKind::ListOrAuto;
const fn at_least(min: i64) -> FieldKind {
    FieldKind::Int { min, max: U32_MAX }
}
const COUNT: FieldKind = at_least(1);
const fn floats(min: Bound, max: Bound) -> FieldKind {
    FieldKind::Float { min, max }
}
/// `(0, 1]`: a similarity threshold.
const THRESHOLD: FieldKind = floats(Bound::Excl(0.0), Bound::Incl(1.0));
const RUNTIMES: FieldKind = FieldKind::Choice(&["docker", "native"]);
const ENGINES: FieldKind = FieldKind::Choice(&["docker", "podman"]);

const PROJECT: &[FieldSpec] = &[spec("name", TEXT, false, "Name of the project")];

const TOPIC: &[FieldSpec] = &[
    spec("name", TEXT, false, "Unique name of the topic"),
    spec("description", TEXT, true, "What the topic covers"),
    spec("subtopics", COUNT, false, "Subtopics to generate"),
    spec(
        "questions_per_subtopic",
        COUNT,
        false,
        "Questions per subtopic",
    ),
];

const PROVIDER: &[FieldSpec] = &[spec(
    "protocol",
    FieldKind::Choice(&["openai", "anthropic"]),
    false,
    "Wire protocol; base_url and api_key come from env",
)];

const ROLE: &[FieldSpec] = &[
    spec("provider", TEXT, false, "Name of a declared provider"),
    spec("model", TEXT, false, "Model ID as the provider knows it"),
    spec(
        "reasoning",
        BOOL,
        true,
        "Ask for reasoning output (default false)",
    ),
    spec(
        "max_tokens",
        COUNT,
        true,
        "Token cap per request, reasoning included (default 16384)",
    ),
    spec(
        "temperature",
        floats(Bound::Incl(0.0), Bound::Incl(2.0)),
        true,
        "Sampling temperature (provider default)",
    ),
    spec(
        "reasoning_effort",
        FieldKind::Choice(&["low", "medium", "high"]),
        true,
        "Reasoning effort, only with reasoning = true",
    ),
    spec(
        "thinking_budget",
        at_least(MIN_THINKING_BUDGET),
        true,
        "Anthropic thinking budget, below max_tokens, with reasoning = true",
    ),
];

const PIPELINE: &[FieldSpec] = &[
    spec(
        "concurrency",
        FieldKind::Int {
            min: 1,
            max: MAX_CONCURRENCY,
        },
        true,
        "Parallel requests (default 8)",
    ),
    spec(
        "max_retries",
        at_least(0),
        true,
        "Retries per failed request (default 5)",
    ),
    spec(
        "dedup_threshold",
        THRESHOLD,
        true,
        "Word-overlap duplicate threshold (default 0.8)",
    ),
    spec(
        "eval_ratio",
        floats(Bound::Excl(0.0), Bound::Excl(1.0)),
        true,
        "Share kept for evaluation (default 0.1)",
    ),
    spec(
        "seed",
        FieldKind::Int {
            min: 0,
            max: i64::MAX,
        },
        true,
        "Seed of the train/eval split (default 42)",
    ),
    spec(
        "include_system_prompt",
        BOOL,
        true,
        "Store the system prompt in the dataset (default false)",
    ),
    spec(
        "embedding_threshold",
        THRESHOLD,
        true,
        "Embedding duplicate threshold (default 0.9)",
    ),
    spec(
        "question_batch_size",
        COUNT,
        true,
        "Questions per generation call (default 10)",
    ),
    spec(
        "request_timeout_secs",
        FieldKind::Int {
            min: 1,
            max: i64::MAX,
        },
        true,
        "Timeout of one request, in seconds (default 600)",
    ),
];

/// The fields of the `[training]` section.
const TRAINING: &[FieldSpec] = &[
    spec("target", TEXT, false, "Name of the target the job runs on"),
    spec(
        "base_model",
        TEXT,
        false,
        "Hugging Face repo ID or a path on the target",
    ),
    spec(
        "adapter",
        FieldKind::Choice(&["lora", "qlora", "full"]),
        false,
        "Fine-tuning strategy",
    ),
    spec("epochs", COUNT, true, "Training epochs (default 3)"),
    spec(
        "learning_rate",
        floats(Bound::Excl(0.0), Bound::Unbounded),
        true,
        "Learning rate (default 0.0002)",
    ),
    spec("lora_r", COUNT, true, "LoRA rank (default 16)"),
    spec(
        "lora_alpha",
        COUNT,
        true,
        "LoRA scaling factor (default 32)",
    ),
    spec(
        "lora_dropout",
        floats(Bound::Incl(0.0), Bound::Excl(1.0)),
        true,
        "LoRA dropout (default 0.05)",
    ),
    spec(
        "sequence_len",
        COUNT,
        true,
        "Maximum sequence length, in tokens (default 4096)",
    ),
    spec(
        "micro_batch_size",
        COUNT,
        true,
        "Examples per GPU per step (default 2)",
    ),
    spec(
        "gradient_accumulation_steps",
        COUNT,
        true,
        "Steps summed per optimizer update (default 4)",
    ),
    spec(
        "optimizer",
        TEXT,
        true,
        "Axolotl optimizer (default adamw_torch_fused)",
    ),
    spec(
        "lr_scheduler",
        TEXT,
        true,
        "Axolotl learning rate scheduler (default cosine)",
    ),
    spec(
        "sample_packing",
        BOOL,
        true,
        "Pack short examples into one sequence (default true)",
    ),
    spec(
        "evals_per_epoch",
        COUNT,
        true,
        "Evaluations per epoch (default 4)",
    ),
    spec(
        "saves_per_epoch",
        COUNT,
        true,
        "Checkpoints per epoch (default 1)",
    ),
    spec(
        "merge",
        BOOL,
        true,
        "Merge the adapter after training, lora or qlora only (default false)",
    ),
    spec(
        "hub_model_id",
        TEXT,
        true,
        "Deprecated: use [hub] repo (overbrainer migrate moves it)",
    ),
];

const LOCAL: &[FieldSpec] = &[
    spec("runtime", RUNTIMES, false, "Run in a container or natively"),
    spec(
        "engine",
        ENGINES,
        true,
        "Container engine, docker runtime only (default docker)",
    ),
    spec(
        "image",
        TEXT,
        true,
        "Container image, docker runtime only (default Axolotl)",
    ),
    spec(
        "venv",
        TEXT,
        true,
        "Virtual environment with bin/axolotl, native runtime only",
    ),
];

const SSH: &[FieldSpec] = &[
    spec("runtime", RUNTIMES, false, "Run in a container or natively"),
    spec(
        "workdir",
        TEXT,
        true,
        "Remote directory of the runs (default overbrainer); host comes from env",
    ),
    spec(
        "engine",
        ENGINES,
        true,
        "Container engine, docker runtime only (default docker)",
    ),
    spec(
        "image",
        TEXT,
        true,
        "Container image, docker runtime only (default Axolotl)",
    ),
    spec(
        "venv",
        TEXT,
        true,
        "Virtual environment with bin/axolotl, native runtime only",
    ),
];

const RUNPOD: &[FieldSpec] = &[
    spec(
        "gpu_types",
        LIST_OR_AUTO,
        false,
        "Runpod GPU type IDs, tried in order, comma-separated; auto: cheapest in stock",
    ),
    spec(
        "min_vram_gb",
        COUNT,
        true,
        "Least VRAM per GPU in GB, with gpu_types = auto only",
    ),
    spec(
        "max_price_per_hour",
        floats(Bound::Excl(0.0), Bound::Unbounded),
        true,
        "Highest list price of one GPU in USD/h, with gpu_types = auto only",
    ),
    spec("gpu_count", COUNT, true, "GPUs per pod (default 1)"),
    spec(
        "image",
        TEXT,
        true,
        "Container image of the pod (default Axolotl cloud)",
    ),
    spec(
        "venv",
        TEXT,
        true,
        "Absolute virtual environment with bin/axolotl",
    ),
    spec(
        "container_disk_gb",
        at_least(MIN_CONTAINER_DISK_GB),
        true,
        "Container disk, in GB (default 50)",
    ),
    spec(
        "max_hours",
        floats(Bound::Excl(0.0), Bound::Incl(MAX_RUNPOD_HOURS)),
        false,
        "Hours before the watchdog deletes a pod nothing follows",
    ),
    spec(
        "max_cost_usd",
        floats(Bound::Excl(0.0), Bound::Unbounded),
        true,
        "Most a run spends on its pod in USD: snapshot at 95%, pod deleted at 100%",
    ),
    spec(
        "boot_grace_minutes",
        at_least(MIN_BOOT_GRACE_MINUTES),
        true,
        "Minutes to wait for the job to start (default 30)",
    ),
    spec(
        "retrieve_grace_minutes",
        COUNT,
        true,
        "Minutes to keep an unretrieved ended job (default 60)",
    ),
    spec(
        "data_center_ids",
        LIST_OR_AUTO,
        true,
        "Allowed data centers, comma-separated (any when empty); auto: those in stock",
    ),
    spec(
        "network_volume_id",
        TEXT,
        true,
        "Network volume, needs exactly one data center",
    ),
    spec(
        "max_volume_gb",
        COUNT,
        true,
        "Largest size in GB the network volume may grow to when it fills; a grow is permanent and billed monthly",
    ),
];

const EXPORT: &[FieldSpec] = &[
    spec(
        "after_training",
        BOOL,
        true,
        "Export the model to GGUF at the end of each training job (default false)",
    ),
    spec(
        "quantize",
        FieldKind::Choice(&QUANTIZE_TYPES),
        true,
        "llama-quantize type of the GGUF, F16 and BF16 unquantized (default Q4_K_M)",
    ),
    spec(
        "ollama_name",
        TEXT,
        true,
        "Ollama model created from an export in a training job, when ollama is on PATH",
    ),
];

/// `[hub]`: the push of a run to the Hugging Face Hub.
const HUB: &[FieldSpec] = &[
    spec(
        "repo",
        TEXT,
        true,
        "Hugging Face repo to push runs to, NAMESPACE/NAME (default <you>/<project>)",
    ),
    spec(
        "private",
        BOOL,
        true,
        "Create the repo private (default true)",
    ),
    spec(
        "after_training",
        BOOL,
        true,
        "Push each run once it succeeded (default false)",
    ),
];

const METRICS: &[FieldSpec] = &[spec(
    "listen",
    TEXT,
    true,
    "Address of the Prometheus endpoint, such as 127.0.0.1:9464; unset it to turn it off",
)];

/// The editable fields of `section`, in file order.
#[must_use]
pub fn for_section(section: Section) -> &'static [FieldSpec] {
    match section {
        Section::Project => PROJECT,
        Section::Topic => TOPIC,
        Section::Provider => PROVIDER,
        Section::Role => ROLE,
        Section::Pipeline => PIPELINE,
        Section::Training => TRAINING,
        Section::Target(TargetKind::Local) => LOCAL,
        Section::Target(TargetKind::Ssh) => SSH,
        Section::Target(TargetKind::Runpod) => RUNPOD,
        Section::Export => EXPORT,
        Section::Hub => HUB,
        Section::Metrics => METRICS,
    }
}

/// The spec of `field` in `section`, when it is editable.
#[must_use]
pub fn find(section: Section, field: &str) -> Option<&'static FieldSpec> {
    for_section(section).iter().find(|spec| spec.name == field)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::config::{ConfigError, EnvSource, load_str};

    /// A valid configuration with a `#ZZ <section>` marker in each table.
    const FIXTURE: &str = r#"#ZZ settings
[project]
name = "demo"
#ZZ project
[[topics]]
name = "ownership"
subtopics = 3
questions_per_subtopic = 5
#ZZ topic
[providers.nanogpt]
protocol = "openai"
#ZZ provider
[roles]
#ZZ roles
[roles.generator]
provider = "nanogpt"
model = "m1"
#ZZ role
[roles.parent]
provider = "nanogpt"
model = "m2"
[pipeline]
#ZZ pipeline
[training]
target = "local"
base_model = "Qwen/Qwen3-4B"
adapter = "qlora"
#ZZ training
[targets.local]
kind = "local"
runtime = "native"
#ZZ local
[targets.box]
kind = "ssh"
runtime = "docker"
#ZZ ssh
[targets.cloud]
kind = "runpod"
gpu_types = ["NVIDIA A40"]
max_hours = 6
#ZZ runpod
[export]
#ZZ export
[hub]
#ZZ hub
[metrics]
#ZZ metrics
"#;

    /// Keys a form never writes: env-only secrets and hosts, and free-form tables.
    const NOT_IN_FORM: [(&str, &str); 8] = [
        ("settings", "runpod"),
        ("settings", "hf_token"),
        ("settings", "log"),
        ("provider", "base_url"),
        ("provider", "api_key"),
        ("ssh", "host"),
        ("training", "axolotl_extra"),
        ("hub", "base_url"),
    ];

    /// Every section name with its `Section`.
    const SECTIONS: [(&str, Section); 12] = [
        ("project", Section::Project),
        ("topic", Section::Topic),
        ("provider", Section::Provider),
        ("role", Section::Role),
        ("pipeline", Section::Pipeline),
        ("training", Section::Training),
        ("local", Section::Target(TargetKind::Local)),
        ("ssh", Section::Target(TargetKind::Ssh)),
        ("runpod", Section::Target(TargetKind::Runpod)),
        ("export", Section::Export),
        ("hub", Section::Hub),
        ("metrics", Section::Metrics),
    ];

    /// The fields serde expects in `marker`'s table, read from its unknown-field error.
    fn struct_fields(marker: &str) -> Result<BTreeSet<String>, String> {
        let text = FIXTURE.replace(&format!("#ZZ {marker}\n"), "zz_drift = 1\n");
        let message = match load_str(&text, EnvSource::Vars(Vec::new())) {
            Err(ConfigError::Parse(message)) => message,
            other => {
                return Err(format!(
                    "{marker}: expected an unknown field, got {other:?}"
                ));
            },
        };
        let (_, expected) = message
            .split_once("expected")
            .ok_or_else(|| format!("{marker}: {message}"))?;
        Ok(expected
            .split('`')
            .skip(1)
            .step_by(2)
            .map(str::to_string)
            .collect())
    }

    #[test]
    fn the_fixture_is_valid() -> Result<(), ConfigError> {
        load_str(FIXTURE, EnvSource::Vars(Vec::new())).map(|_| ())
    }

    #[test]
    fn every_struct_field_is_editable_or_listed_as_not_in_the_form() -> Result<(), String> {
        for (marker, section) in SECTIONS {
            let expected = struct_fields(marker)?;
            let mut covered: BTreeSet<String> = for_section(section)
                .iter()
                .map(|spec| spec.name.to_string())
                .collect();
            covered.extend(
                NOT_IN_FORM
                    .iter()
                    .filter(|(owner, _)| *owner == marker)
                    .map(|(_, field)| (*field).to_string()),
            );
            assert_eq!(covered, expected, "{marker}");
        }
        Ok(())
    }

    /// The fields listed outside `for_section`: the form's sections and the roles.
    const LISTED_ELSEWHERE: [(&str, &[&str]); 2] = [
        (
            "settings",
            &[
                "project",
                "topics",
                "providers",
                "roles",
                "pipeline",
                "training",
                "targets",
                "export",
                "hub",
                "metrics",
            ],
        ),
        ("roles", &["generator", "parent", "embedder"]),
    ];

    #[test]
    fn every_settings_and_roles_key_is_covered() -> Result<(), String> {
        for (marker, listed) in LISTED_ELSEWHERE {
            let mut covered: BTreeSet<String> =
                listed.iter().map(|key| (*key).to_string()).collect();
            covered.extend(
                NOT_IN_FORM
                    .iter()
                    .filter(|(owner, _)| *owner == marker)
                    .map(|(_, field)| (*field).to_string()),
            );
            assert_eq!(covered, struct_fields(marker)?, "{marker}");
        }
        let roles: Vec<&str> = crate::config::edit::Role::ALL
            .iter()
            .map(|role| role.as_str())
            .collect();
        assert_eq!(roles, LISTED_ELSEWHERE[1].1);
        Ok(())
    }

    /// `FIXTURE` without the `field` line of the table marked `#ZZ <marker>`.
    fn without(marker: &str, field: &str) -> String {
        let prefix = format!("{field} =");
        let mut result = Vec::new();
        let mut table = Vec::new();
        for line in FIXTURE.lines() {
            if line.starts_with('[') {
                result.append(&mut strip(&mut table, marker, &prefix));
            }
            table.push(line);
        }
        result.append(&mut strip(&mut table, marker, &prefix));
        result.join("\n")
    }

    /// Drains `table`, leaving out the `prefix` line when it is the marked table.
    fn strip<'a>(table: &mut Vec<&'a str>, marker: &str, prefix: &str) -> Vec<&'a str> {
        let in_marked_table = table.contains(&format!("#ZZ {marker}").as_str());
        table
            .drain(..)
            .filter(|line| !(in_marked_table && line.starts_with(prefix)))
            .collect()
    }

    #[test]
    fn each_required_field_alone_is_required_and_optional_ones_are_not() {
        for (marker, section) in SECTIONS {
            for spec in for_section(section) {
                let text = without(marker, spec.name);
                let result = load_str(&text, EnvSource::Vars(Vec::new()));
                if spec.optional {
                    assert!(
                        result.is_ok(),
                        "{marker}.{} is optional: {result:?}",
                        spec.name
                    );
                } else {
                    assert!(
                        text.len() < FIXTURE.len(),
                        "{marker}.{} not in the fixture",
                        spec.name
                    );
                    assert!(result.is_err(), "{marker}.{} is required", spec.name);
                }
            }
        }
    }

    #[test]
    fn numbers_are_parsed_with_their_bounds() {
        let int = FieldKind::Int { min: 1, max: 10 };
        assert_eq!(int.parse(" 7 "), Ok(FieldValue::Int(7)));
        assert_eq!(
            int.parse("0").map_err(|error| error.to_string()),
            Err("must be between 1 and 10".to_string())
        );
        assert_eq!(
            COUNT.parse("0").map_err(|error| error.to_string()),
            Err("must be at least 1".to_string())
        );
        assert!(int.parse("7.5").is_err());
        let float = floats(Bound::Excl(0.0), Bound::Incl(1.0));
        assert_eq!(float.parse("0.5"), Ok(FieldValue::Float(0.5)));
        assert_eq!(float.parse("1"), Ok(FieldValue::Float(1.0)));
        assert_eq!(
            float.parse("0").map_err(|error| error.to_string()),
            Err("must be in (0, 1]".to_string())
        );
        for text in ["NaN", "inf", "1.5", "x"] {
            assert!(float.parse(text).is_err(), "{text}");
        }
        let rate = floats(Bound::Excl(0.0), Bound::Unbounded);
        assert_eq!(rate.parse("1e9"), Ok(FieldValue::Float(1e9)));
        assert_eq!(
            rate.parse("0").map_err(|error| error.to_string()),
            Err("must be greater than 0".to_string())
        );
        let dropout = floats(Bound::Incl(0.0), Bound::Excl(1.0));
        assert_eq!(dropout.parse("0"), Ok(FieldValue::Float(0.0)));
        assert_eq!(
            dropout.parse("1").map_err(|error| error.to_string()),
            Err("must be in [0, 1)".to_string())
        );
    }

    #[test]
    fn other_text_is_parsed_by_kind() {
        assert_eq!(BOOL.parse("true"), Ok(FieldValue::Bool(true)));
        assert!(BOOL.parse("yes").is_err());
        assert_eq!(
            RUNTIMES.parse("native"),
            Ok(FieldValue::Text("native".to_string()))
        );
        assert_eq!(
            RUNTIMES.parse("vm").map_err(|error| error.to_string()),
            Err("must be one of docker, native".to_string())
        );
        assert_eq!(
            TEXT.parse(" a = \"b\" "),
            Ok(FieldValue::Text(" a = \"b\" ".to_string()))
        );
    }

    #[test]
    fn a_list_or_auto_field_takes_auto_or_a_list() {
        let kind = FieldKind::ListOrAuto;
        assert_eq!(
            kind.parse(" auto "),
            Ok(FieldValue::Text("auto".to_string()))
        );
        assert_eq!(
            kind.parse("A40, auto"),
            Ok(FieldValue::List(vec![
                "A40".to_string(),
                "auto".to_string()
            ]))
        );
        assert_eq!(kind.parse(""), Ok(FieldValue::List(Vec::new())));
        // The TUI hints read the same text through the same parser.
        assert_eq!(
            ListOrAuto::from_form_text("A40,, H100 ,"),
            ListOrAuto::List(vec!["A40".to_string(), "H100".to_string()])
        );
        assert_eq!(ListOrAuto::from_form_text(" auto "), ListOrAuto::Auto);
        assert_eq!(
            kind.check(&FieldValue::Text("A40".to_string()))
                .map_err(|error| error.to_string()),
            Err("must be auto or a comma-separated list".to_string())
        );
        assert_eq!(kind.check(&FieldValue::Int(1)), Err(FieldError::WrongType));
        for field in ["gpu_types", "data_center_ids"] {
            let spec = find(Section::Target(TargetKind::Runpod), field);
            assert_eq!(spec.map(|spec| spec.kind), Some(kind), "{field}");
        }
    }

    #[test]
    fn errors_never_quote_the_value() {
        let secret = "sk-secret-value";
        for (index, kind) in [COUNT, BOOL, RUNTIMES, THRESHOLD].into_iter().enumerate() {
            let message = kind
                .parse(secret)
                .err()
                .map(|error| error.to_string())
                .unwrap_or_default();
            assert!(
                !message.is_empty() && !message.contains(secret),
                "kind #{index} quotes the value or says nothing"
            );
        }
    }
}
