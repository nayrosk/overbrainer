//! Schema of the fields a form may edit in `overbrainer.toml`: kind, bounds, choices.
//!
//! Env-only keys (provider `base_url` and `api_key`, target `host`, `runpod.*`,
//! `hf_token`, `log`), the target `kind` tag and the free-form
//! `training.axolotl_extra` are not listed: a form never writes them. Bounds mirror
//! `validate.rs`; `Float` bounds are inclusive, and the help text states when
//! `validate` excludes an end.

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
    /// A finite float in `[min, max]`.
    Float {
        /// Smallest accepted value.
        min: f64,
        /// Largest accepted value.
        max: f64,
    },
    /// `true` or `false`.
    Bool,
    /// One of these strings.
    Choice(&'static [&'static str]),
    /// A list of strings, typed as comma-separated text.
    List,
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

impl FieldKind {
    /// Reads `text`, as typed in a form, into a value of this kind.
    ///
    /// # Errors
    ///
    /// Returns why `text` is not accepted. The message never quotes `text`.
    pub fn parse(self, text: &str) -> Result<FieldValue, String> {
        let value = match self {
            Self::Text => FieldValue::Text(text.to_string()),
            Self::Int { .. } => text
                .trim()
                .parse()
                .map(FieldValue::Int)
                .map_err(|_| "must be a whole number".to_string())?,
            Self::Float { .. } => text
                .trim()
                .parse()
                .map(FieldValue::Float)
                .map_err(|_| "must be a number".to_string())?,
            Self::Bool => match text.trim() {
                "true" => FieldValue::Bool(true),
                "false" => FieldValue::Bool(false),
                _ => return Err("must be true or false".to_string()),
            },
            Self::Choice(_) => FieldValue::Text(text.trim().to_string()),
            Self::List => FieldValue::List(
                text.split(',')
                    .map(str::trim)
                    .filter(|item| !item.is_empty())
                    .map(str::to_string)
                    .collect(),
            ),
        };
        self.check(&value)?;
        Ok(value)
    }

    /// Checks that `value` has this kind and is within its bounds or choices.
    ///
    /// # Errors
    ///
    /// Returns why `value` is not accepted. The message never quotes the value.
    pub fn check(self, value: &FieldValue) -> Result<(), String> {
        match (self, value) {
            (Self::Text, FieldValue::Text(_))
            | (Self::Bool, FieldValue::Bool(_))
            | (Self::List, FieldValue::List(_)) => Ok(()),
            (Self::Int { min, max }, FieldValue::Int(number)) => {
                if (min..=max).contains(number) {
                    Ok(())
                } else {
                    Err(int_range(min, max))
                }
            },
            (Self::Float { min, max }, FieldValue::Float(number)) => {
                if number.is_finite() && (min..=max).contains(number) {
                    Ok(())
                } else {
                    Err(format!("must be a number in [{min}, {max}]"))
                }
            },
            (Self::Choice(choices), FieldValue::Text(text)) => {
                if choices.contains(&text.as_str()) {
                    Ok(())
                } else {
                    Err(format!("must be one of {}", choices.join(", ")))
                }
            },
            _ => Err("has the wrong type".to_string()),
        }
    }
}

/// The range message of an integer field, without an unbounded end.
fn int_range(min: i64, max: i64) -> String {
    match (min, max) {
        (min, i64::MAX | U32_MAX) => format!("must be at least {min}"),
        (min, max) => format!("must be between {min} and {max}"),
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
const LIST: FieldKind = FieldKind::List;
const fn at_least(min: i64) -> FieldKind {
    FieldKind::Int { min, max: U32_MAX }
}
const COUNT: FieldKind = at_least(1);
const RUNTIMES: FieldKind = FieldKind::Choice(&["docker", "native"]);
const ENGINES: FieldKind = FieldKind::Choice(&["docker", "podman"]);

const PROJECT: &[FieldSpec] = &[spec("name", TEXT, false, "Name of the project")];

const TOPIC: &[FieldSpec] = &[
    spec("name", TEXT, false, "Unique name of the topic"),
    spec("description", TEXT, true, "What the topic covers"),
    spec(
        "subtopics",
        COUNT,
        false,
        "Subtopics to generate, at least 1",
    ),
    spec(
        "questions_per_subtopic",
        COUNT,
        false,
        "Questions per subtopic, at least 1",
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
        FieldKind::Float { min: 0.0, max: 2.0 },
        true,
        "Sampling temperature in [0, 2] (provider default)",
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
        "Anthropic thinking budget in [1024, max_tokens), with reasoning = true",
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
        "Parallel requests, 1 to 1024 (default 8)",
    ),
    spec(
        "max_retries",
        at_least(0),
        true,
        "Retries per failed request (default 5)",
    ),
    spec(
        "dedup_threshold",
        FieldKind::Float { min: 0.0, max: 1.0 },
        true,
        "Word-overlap duplicate threshold in (0, 1] (default 0.8)",
    ),
    spec(
        "eval_ratio",
        FieldKind::Float { min: 0.0, max: 1.0 },
        true,
        "Share kept for evaluation, in (0, 1) (default 0.1)",
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
        FieldKind::Float { min: 0.0, max: 1.0 },
        true,
        "Embedding duplicate threshold in (0, 1] (default 0.9)",
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
        FieldKind::Float {
            min: 0.0,
            max: f64::MAX,
        },
        true,
        "Learning rate, greater than 0 (default 0.0002)",
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
        FieldKind::Float { min: 0.0, max: 1.0 },
        true,
        "LoRA dropout in [0, 1) (default 0.05)",
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
        "Hugging Face repo to push the model to",
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
        LIST,
        false,
        "Runpod GPU type IDs, tried in order, comma-separated",
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
        "Container disk, in GB, at least 20 (default 50)",
    ),
    spec(
        "max_hours",
        FieldKind::Float {
            min: 0.0,
            max: MAX_RUNPOD_HOURS,
        },
        false,
        "Hours before the watchdog deletes the pod, in (0, 720]",
    ),
    spec(
        "boot_grace_minutes",
        at_least(MIN_BOOT_GRACE_MINUTES),
        true,
        "Minutes to wait for the job to start, at least 5 (default 30)",
    ),
    spec(
        "retrieve_grace_minutes",
        COUNT,
        true,
        "Minutes to keep an unretrieved ended job (default 60)",
    ),
    spec(
        "data_center_ids",
        LIST,
        true,
        "Allowed data centers, comma-separated (any when empty)",
    ),
    spec(
        "network_volume_id",
        TEXT,
        true,
        "Network volume, needs exactly one data center",
    ),
];

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
    const FIXTURE: &str = r#"
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
"#;

    /// Keys a form never writes: env-only secrets and hosts, and free-form tables.
    const NOT_IN_FORM: [(&str, &str); 4] = [
        ("provider", "base_url"),
        ("provider", "api_key"),
        ("ssh", "host"),
        ("training", "axolotl_extra"),
    ];

    const SECTIONS: [(&str, Section); 9] = [
        ("project", Section::Project),
        ("topic", Section::Topic),
        ("provider", Section::Provider),
        ("role", Section::Role),
        ("pipeline", Section::Pipeline),
        ("training", Section::Training),
        ("local", Section::Target(TargetKind::Local)),
        ("ssh", Section::Target(TargetKind::Ssh)),
        ("runpod", Section::Target(TargetKind::Runpod)),
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

    #[test]
    fn required_fields_match_the_fixture() {
        for (marker, section) in SECTIONS {
            for spec in for_section(section).iter().filter(|spec| !spec.optional) {
                let removed = FIXTURE
                    .lines()
                    .filter(|line| !line.starts_with(&format!("{} =", spec.name)))
                    .collect::<Vec<_>>()
                    .join("\n");
                assert!(
                    load_str(&removed, EnvSource::Vars(Vec::new())).is_err(),
                    "{marker}.{} is required",
                    spec.name
                );
            }
        }
    }

    #[test]
    fn text_is_parsed_by_kind() {
        let int = FieldKind::Int { min: 1, max: 10 };
        assert_eq!(int.parse(" 7 "), Ok(FieldValue::Int(7)));
        assert_eq!(int.parse("0"), Err("must be between 1 and 10".to_string()));
        assert_eq!(COUNT.parse("0"), Err("must be at least 1".to_string()));
        assert!(int.parse("7.5").is_err());
        let float = FieldKind::Float { min: 0.0, max: 1.0 };
        assert_eq!(float.parse("0.5"), Ok(FieldValue::Float(0.5)));
        for text in ["NaN", "inf", "1.5", "x"] {
            assert!(float.parse(text).is_err(), "{text}");
        }
        assert_eq!(BOOL.parse("true"), Ok(FieldValue::Bool(true)));
        assert!(BOOL.parse("yes").is_err());
        assert_eq!(
            RUNTIMES.parse("native"),
            Ok(FieldValue::Text("native".to_string()))
        );
        assert_eq!(
            RUNTIMES.parse("vm"),
            Err("must be one of docker, native".to_string())
        );
        assert_eq!(
            LIST.parse(" A40 , ,L40S"),
            Ok(FieldValue::List(vec![
                "A40".to_string(),
                "L40S".to_string()
            ]))
        );
        assert_eq!(LIST.parse(""), Ok(FieldValue::List(Vec::new())));
        assert_eq!(
            TEXT.parse(" a = \"b\" "),
            Ok(FieldValue::Text(" a = \"b\" ".to_string()))
        );
    }

    #[test]
    fn errors_never_quote_the_value() {
        let secret = "sk-secret-value";
        for kind in [
            COUNT,
            BOOL,
            RUNTIMES,
            FieldKind::Float { min: 0.0, max: 1.0 },
        ] {
            let message = kind.parse(secret).err().unwrap_or_default();
            assert!(
                !message.is_empty() && !message.contains(secret),
                "{message}"
            );
        }
    }
}
