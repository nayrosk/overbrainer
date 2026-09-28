//! The files the init wizard writes: `overbrainer.toml`, edited from the
//! template `overbrainer init` writes (its comments kept) and validated with
//! the `.env` values about to be written, then `.env` (mode 600),
//! `.env.example`, the prompt templates and `.gitignore`. Nothing is ever
//! overwritten, and nothing is written when a file exists.

use std::path::{Path, PathBuf};

use super::{Answers, TrainingKind};
use crate::cli::init::{
    CONFIG_TEMPLATE, ENV_EXAMPLE_FILE, ENV_FILE, GITIGNORE, create_new, create_private,
    prompt_files, update_gitignore,
};
use crate::config::edit::{Collection, ConfigDoc, FieldPath, Role};
use crate::config::fields::FieldValue;
use crate::config::{CONFIG_FILE, ENV_PREFIX, EnvSource, ListOrAuto, load_str};
use crate::prompts;

/// The provider of the template, replaced by the one chosen.
const TEMPLATE_PROVIDER: &str = "openrouter";
/// The target of the template, kept for a local target.
const LOCAL_TARGET: &str = "local";
/// The name of an SSH target.
const SSH_TARGET: &str = "homelab";
/// The name of a Runpod target.
const RUNPOD_TARGET: &str = "gpu_cloud";

/// The first line of the template's `[training]`.
const TRAINING_START: &str = "[training]";
/// The first lines of the template's local target.
const LOCAL_START: &str = "[targets.local]\nkind = \"local\"";
/// The first line after the template's local target: the commented targets.
const TARGETS_EXAMPLES: &str = "# [targets.homelab]";
/// What replaces `[training]` and the local target when training is skipped.
const NO_TRAINING: &str = "\
# No [training]: auto mode stops after split. To train, add [training] and a
# target: docs/training.md, or the overbrainer.toml `overbrainer init` writes.
";
/// What the local target's first lines become for an SSH target: its
/// comments apply as they are.
const SSH_START: &str = "\
[targets.homelab]
kind = \"ssh\"                   # host comes from OVERBRAINER_TARGETS__HOMELAB__HOST";
/// What replaces the local target, its comments included, for a Runpod one.
const RUNPOD_TARGET_TEXT: &str = "\
[targets.gpu_cloud]
kind = \"runpod\"              # needs OVERBRAINER_RUNPOD__API_KEY; Secure Cloud only
gpu_types = \"auto\"           # \"auto\": the cheapest in stock at start; or a list, tried in order
max_hours = 6                # the pod watchdog deletes the pod after this, whatever it is doing
";

/// What `.env` and `.env.example` end with: the optional variables.
const OPTIONAL_ENV: &str = "\
# OVERBRAINER_HF_TOKEN=          # gated or private base models, and training.hub_model_id
# OVERBRAINER_LOG=info
# VAULT_ADDR=https://vault.example.com
# VAULT_TOKEN=
";

/// The texts of the files the answers give.
#[derive(Clone, PartialEq, Eq)]
pub(super) struct Files {
    /// `overbrainer.toml`.
    pub(super) config: String,
    /// `.env`, secrets included.
    pub(super) env: String,
    /// `.env.example`, without them.
    pub(super) example: String,
}

/// One `OVERBRAINER_*` variable the choices need.
struct Variable {
    name: String,
    value: String,
    /// Whether `.env.example` keeps its value: not for a key or a host.
    public: bool,
}

/// Builds the files of `answers`, and validates `overbrainer.toml` with the
/// variables of `.env`, as the next load will read them.
///
/// # Errors
///
/// Returns why the template cannot be edited or the configuration is
/// invalid. No message quotes a value.
pub(super) fn build(answers: &Answers) -> Result<Files, String> {
    let config =
        config_text(answers).map_err(|error| format!("cannot edit the template: {error}"))?;
    let variables = variables(answers);
    let pairs = variables
        .iter()
        .map(|variable| (variable.name.clone(), variable.value.clone()))
        .collect();
    load_str(&config, EnvSource::Vars(pairs)).map_err(|error| format!("{error:#}"))?;
    Ok(Files {
        config,
        env: env_text(
            "# Written by the overbrainer init wizard: never commit it.\n",
            &variables,
            false,
        ),
        example: env_text(
            "# Copy to .env (gitignored) or export in your shell.\n",
            &variables,
            true,
        ),
    })
}

/// Writes the files of `answers` into `dir`: `.env` first, `overbrainer.toml`
/// last, so a failure never leaves a configuration without its `.env`.
///
/// # Errors
///
/// Returns why nothing was written (a file exists, the configuration is
/// invalid), or why writing stopped. No message quotes a value.
pub(super) fn write(dir: &Path, answers: &Answers) -> Result<(), String> {
    let files = build(answers)?;
    let mut paths = vec![
        PathBuf::from(ENV_FILE),
        PathBuf::from(ENV_EXAMPLE_FILE),
        PathBuf::from(CONFIG_FILE),
    ];
    paths.extend(prompt_files().into_iter().map(|(path, _)| path));
    // A dangling link counts as a file: creating it would write through it.
    if let Some(path) = paths
        .iter()
        .find(|path| dir.join(path).symlink_metadata().is_ok())
    {
        return Err(format!(
            "{} already exists: nothing was written",
            path.display()
        ));
    }
    let written = || -> anyhow::Result<()> {
        std::fs::create_dir_all(dir.join(prompts::DIR))?;
        create_private(&dir.join(ENV_FILE), &files.env)?;
        create_new(&dir.join(ENV_EXAMPLE_FILE), &files.example)?;
        for (path, content) in prompt_files() {
            create_new(&dir.join(path), content)?;
        }
        update_gitignore(&dir.join(GITIGNORE))?;
        create_new(&dir.join(CONFIG_FILE), &files.config)
    };
    written().map_err(|error| format!("{error:#}"))
}

/// `overbrainer.toml`: the template with the answers set.
fn config_text(answers: &Answers) -> Result<String, crate::config::edit::EditError> {
    let mut doc = ConfigDoc::parse(&template(answers.training))?;
    doc.set(&FieldPath::Project("name"), text(answers.name))?;
    let provider = answers.provider.as_str();
    if provider == TEMPLATE_PROVIDER {
        let path = FieldPath::Provider {
            name: provider.to_string(),
            field: "protocol",
        };
        doc.set(&path, text(super::protocol_name(answers.protocol)))?;
    } else {
        doc.add_provider(provider, answers.protocol)?;
        doc.remove_table(Collection::Providers, TEMPLATE_PROVIDER);
    }
    let mut models = vec![
        (Role::Generator, answers.generator),
        (Role::Parent, answers.parent),
    ];
    if let Some(embedder) = answers.embedder {
        models.push((Role::Embedder, embedder));
    }
    for (role, model) in models {
        doc.set(
            &FieldPath::Role {
                role,
                field: "provider",
            },
            text(provider),
        )?;
        doc.set(
            &FieldPath::Role {
                role,
                field: "model",
            },
            text(model),
        )?;
    }
    let reasoning = FieldPath::Role {
        role: Role::Parent,
        field: "reasoning",
    };
    doc.set(&reasoning, FieldValue::Bool(answers.reasoning))?;
    set_topics(&mut doc, answers)?;
    set_training(&mut doc, answers)?;
    Ok(doc.text())
}

/// The topics: the template's first one renamed to the first answer, so its
/// place and comments stay, then the others after it.
fn set_topics(
    doc: &mut ConfigDoc,
    answers: &Answers,
) -> Result<(), crate::config::edit::EditError> {
    let template = doc.topic_names().first().cloned().unwrap_or_default();
    for (index, topic) in answers.topics.iter().enumerate() {
        if index == 0 {
            let path = FieldPath::Topic {
                index,
                name: template.clone(),
                field: "name",
            };
            doc.set(&path, text(&topic.name))?;
        } else {
            doc.add_topic(&topic.name)?;
        }
        let path = |field| FieldPath::Topic {
            index,
            name: topic.name.clone(),
            field,
        };
        if topic.description.is_empty() {
            doc.unset(&path("description"))?;
        } else {
            doc.set(&path("description"), text(&topic.description))?;
        }
        doc.set(&path("subtopics"), FieldValue::Int(topic.subtopics.into()))?;
        let questions = FieldValue::Int(topic.questions_per_subtopic.into());
        doc.set(&path("questions_per_subtopic"), questions)?;
    }
    Ok(())
}

/// `[training]` and its target, or neither when training is skipped.
fn set_training(
    doc: &mut ConfigDoc,
    answers: &Answers,
) -> Result<(), crate::config::edit::EditError> {
    let target = match answers.training {
        TrainingKind::Skip => return Ok(()),
        TrainingKind::Local => LOCAL_TARGET,
        TrainingKind::Ssh => SSH_TARGET,
        TrainingKind::Runpod => RUNPOD_TARGET,
    };
    let field = |field| FieldPath::Target {
        name: target.to_string(),
        field,
    };
    if answers.training == TrainingKind::Runpod {
        let gpu_types = match &answers.gpu_types {
            ListOrAuto::Auto => text(ListOrAuto::AUTO),
            ListOrAuto::List(ids) => FieldValue::List(ids.clone()),
        };
        doc.set(&field("gpu_types"), gpu_types)?;
    } else {
        doc.set(
            &field("runtime"),
            text(super::runtime_name(answers.runtime)),
        )?;
    }
    doc.set(&FieldPath::Training("target"), text(target))?;
    doc.set(&FieldPath::Training("base_model"), text(answers.base_model))?;
    let adapter = text(super::adapter_name(answers.adapter));
    doc.set(&FieldPath::Training("adapter"), adapter)
}

/// The template for `training`, its tables edited as text where a table
/// removal would move their comments to the end of the file: without
/// `[training]` and the local target when skipped, the local target renamed
/// for SSH, replaced for Runpod. A template without the lines this looks for
/// is kept whole; a test checks it has them.
fn template(training: TrainingKind) -> String {
    let template = CONFIG_TEMPLATE;
    let at = |line: &str| template.find(&format!("\n{line}\n")).map(|at| at + 1);
    let replaced = |from: Option<usize>, to: Option<usize>, with: &str| match (from, to) {
        (Some(from), Some(to)) if from < to => {
            format!("{}{with}\n{}", &template[..from], &template[to..])
        },
        _ => template.to_string(),
    };
    match training {
        TrainingKind::Skip => replaced(at(TRAINING_START), at(TARGETS_EXAMPLES), NO_TRAINING),
        TrainingKind::Local => template.to_string(),
        TrainingKind::Ssh => template.replacen(LOCAL_START, SSH_START, 1),
        TrainingKind::Runpod => replaced(at(LOCAL_START), at(TARGETS_EXAMPLES), RUNPOD_TARGET_TEXT),
    }
}

fn text(value: &str) -> FieldValue {
    FieldValue::Text(value.to_string())
}

/// The variables the choices need: the provider's base URL and key, the SSH
/// host, the Runpod key. Keys and a host are written even when empty, so
/// `.env` shows what to fill.
fn variables(answers: &Answers) -> Vec<Variable> {
    let provider = answers.provider.to_uppercase();
    let mut variables = vec![
        Variable {
            name: format!("{ENV_PREFIX}_PROVIDERS__{provider}__BASE_URL"),
            value: answers.base_url.clone(),
            public: true,
        },
        Variable {
            name: format!("{ENV_PREFIX}_PROVIDERS__{provider}__API_KEY"),
            value: answers.api_key.to_string(),
            public: false,
        },
    ];
    match answers.training {
        TrainingKind::Ssh => variables.push(Variable {
            name: format!("{ENV_PREFIX}_TARGETS__{}__HOST", SSH_TARGET.to_uppercase()),
            value: answers.host.to_string(),
            public: false,
        }),
        TrainingKind::Runpod => variables.push(Variable {
            name: format!("{ENV_PREFIX}_RUNPOD__API_KEY"),
            value: answers.runpod_key.to_string(),
            public: false,
        }),
        TrainingKind::Skip | TrainingKind::Local => {},
    }
    variables
}

/// An env file: `header`, the variables (without their private values for
/// the `example`), then the optional ones, commented out.
fn env_text(header: &str, variables: &[Variable], example: bool) -> String {
    let lines = variables
        .iter()
        .map(|variable| {
            let value = if example && !variable.public {
                String::new()
            } else {
                quoted(&variable.value)
            };
            format!("{}={value}\n", variable.name)
        })
        .collect::<Vec<_>>()
        .concat();
    format!(
        "{header}# Values may be literals or Vault references: \
         vault:<mount>/<path>#<field>\n{lines}{OPTIONAL_ENV}"
    )
}

/// `value` as `dotenvy` reads it back unchanged: bare when plain, else in
/// single quotes (no escape, no `$` expansion), else in double quotes with
/// `\`, `"` and `$` escaped.
fn quoted(value: &str) -> String {
    let plain = value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "-_./:@+,=".contains(c));
    if plain {
        value.to_string()
    } else if !value.contains('\'') {
        format!("'{value}'")
    } else {
        let escaped = value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('$', "\\$");
        format!("\"{escaped}\"")
    }
}

#[cfg(test)]
mod tests {
    use super::super::{TopicDraft, Wizard};
    use super::*;
    use crate::config::{Adapter, Protocol, Runtime};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    const KEY: &str = "sk-placeholder-provider-key";
    const RUNPOD_KEY: &str = "rp-placeholder-runpod-key";

    /// Where the snapshots of the written files are kept.
    const SNAPSHOTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/snapshots/wizard");

    fn topics() -> Vec<TopicDraft> {
        vec![
            TopicDraft {
                name: "ownership".into(),
                description: "Moves, borrows and lifetimes".into(),
                subtopics: 4,
                questions_per_subtopic: 12,
            },
            TopicDraft {
                name: "traits".into(),
                description: String::new(),
                subtopics: 3,
                questions_per_subtopic: 8,
            },
        ]
    }

    fn answers(training: TrainingKind, topics: &[TopicDraft]) -> Answers<'_> {
        Answers {
            name: "rust_expert",
            provider: "nanogpt".into(),
            protocol: Protocol::Openai,
            base_url: "https://nano-gpt.com/api/v1".into(),
            api_key: KEY,
            generator: "qwen/qwen3-235b-a22b",
            parent: "deepseek/deepseek-r1",
            reasoning: true,
            embedder: None,
            topics,
            training,
            runtime: Runtime::Docker,
            host: "user@gpu-box",
            runpod_key: RUNPOD_KEY,
            gpu_types: ListOrAuto::Auto,
            base_model: "Qwen/Qwen3-8B",
            adapter: Adapter::Lora,
        }
    }

    fn snapshot(name: &str, files: &Files) {
        let mut settings = insta::Settings::clone_current();
        settings.set_snapshot_path(SNAPSHOTS);
        settings.set_prepend_module_to_snapshot(false);
        settings.set_omit_expression(true);
        let shown = format!(
            "--- overbrainer.toml\n{}--- .env\n{}--- .env.example\n{}",
            files.config, files.env, files.example
        );
        settings.bind(|| insta::assert_snapshot!(name.to_string(), shown));
    }

    #[test]
    fn each_training_kind_gives_valid_files() -> TestResult {
        let topics = topics();
        for (name, kind) in [
            ("skip", TrainingKind::Skip),
            ("local", TrainingKind::Local),
            ("ssh", TrainingKind::Ssh),
            ("runpod", TrainingKind::Runpod),
        ] {
            let answers = answers(kind, &topics);
            let files = build(&answers)?;
            assert!(!files.example.contains(KEY) && !files.example.contains(RUNPOD_KEY));
            let pairs: Vec<(String, String)> =
                dotenvy::from_read_iter(files.env.as_bytes()).collect::<Result<_, _>>()?;
            let settings = load_str(&files.config, EnvSource::Vars(pairs))?;
            assert_eq!(settings.topics.len(), 2);
            assert_eq!(settings.training.is_some(), kind != TrainingKind::Skip);
            snapshot(&format!("wizard_{name}"), &files);
        }
        Ok(())
    }

    #[test]
    fn the_template_provider_and_an_embedder_are_kept_in_place() -> TestResult {
        let topics = topics();
        let mut answers = answers(TrainingKind::Local, &topics);
        answers.provider = "openrouter".into();
        answers.embedder = Some("openai/text-embedding-3-small");
        answers.reasoning = false;
        let files = build(&answers)?;
        let settings = load_str(&files.config, EnvSource::Vars(Vec::new()))?;
        assert_eq!(settings.providers.len(), 1);
        let embedder = settings.roles.embedder.ok_or("no embedder")?;
        assert_eq!(embedder.provider, "openrouter");
        assert!(!settings.roles.parent.reasoning);
        Ok(())
    }

    #[test]
    fn the_template_has_the_lines_its_edits_look_for() {
        let skipped = template(TrainingKind::Skip);
        assert!(!skipped.contains("\n[training]\n") && !skipped.contains("[targets.local]"));
        assert!(skipped.contains(TARGETS_EXAMPLES) && skipped.contains("# request_timeout_secs"));
        for kind in [TrainingKind::Ssh, TrainingKind::Runpod] {
            let edited = template(kind);
            assert!(!edited.contains("[targets.local]"), "{kind:?}");
            assert!(edited.contains("\n[training]\n"), "{kind:?}");
        }
    }

    #[test]
    fn listed_gpu_types_are_written_as_a_list() -> TestResult {
        let topics = topics();
        let mut answers = answers(TrainingKind::Runpod, &topics);
        answers.gpu_types = ListOrAuto::List(vec!["NVIDIA A40".into()]);
        let files = build(&answers)?;
        assert!(files.config.contains("gpu_types = [\"NVIDIA A40\"]"));
        Ok(())
    }

    #[test]
    fn values_read_back_unchanged_whatever_they_hold() -> TestResult {
        for value in [
            "",
            "sk-abc_123",
            "vault:secret/overbrainer/nanogpt#api_key",
            "a b $HOME #x",
            "it's \"quoted\" \\ $HOME",
        ] {
            let line = format!("KEY={}\n", quoted(value));
            let pairs: Vec<(String, String)> =
                dotenvy::from_read_iter(line.as_bytes()).collect::<Result<_, _>>()?;
            assert_eq!(pairs, [("KEY".to_string(), value.to_string())]);
        }
        Ok(())
    }

    #[test]
    fn a_refused_configuration_names_no_secret() {
        let topics = topics();
        let mut answers = answers(TrainingKind::Local, &topics);
        answers.protocol = Protocol::Anthropic;
        answers.embedder = Some("text-embedding-3-small");
        let error = build(&answers).err().unwrap_or_default();
        assert!(error.contains("roles.embedder"), "{error}");
        assert!(!error.contains(KEY));
    }

    #[test]
    fn the_files_are_written_with_env_private_and_gitignore_appended() -> TestResult {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join(GITIGNORE), "target/\n")?;
        let topics = topics();
        write(dir.path(), &answers(TrainingKind::Local, &topics))?;
        let mode = std::fs::metadata(dir.path().join(ENV_FILE))?
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
        let gitignore = std::fs::read_to_string(dir.path().join(GITIGNORE))?;
        assert!(gitignore.starts_with("target/\n") && gitignore.contains("\n.env\n"));
        crate::config::load(dir.path(), EnvSource::Vars(Vec::new()))?;
        for (path, _) in prompt_files() {
            assert!(dir.path().join(path).is_file());
        }
        Ok(())
    }

    #[test]
    fn an_existing_file_refuses_and_writes_nothing() -> TestResult {
        let topics = topics();
        for existing in [ENV_FILE, CONFIG_FILE, ENV_EXAMPLE_FILE] {
            let dir = tempfile::tempdir()?;
            std::fs::write(dir.path().join(existing), "kept\n")?;
            let refused = write(dir.path(), &answers(TrainingKind::Local, &topics));
            assert_eq!(
                refused,
                Err(format!("{existing} already exists: nothing was written"))
            );
            let mut left: Vec<String> = std::fs::read_dir(dir.path())?
                .map(|entry| entry.map(|e| e.file_name().to_string_lossy().into_owned()))
                .collect::<Result<_, _>>()?;
            left.sort();
            assert_eq!(left, [existing]);
            assert_eq!(
                std::fs::read_to_string(dir.path().join(existing))?,
                "kept\n"
            );
        }
        Ok(())
    }

    #[test]
    fn the_wizard_answers_build_valid_files() -> TestResult {
        let wizard = Wizard::new("demo");
        let mut answers = wizard.answers();
        let topics = topics();
        answers.generator = "g";
        answers.parent = "p";
        answers.topics = &topics;
        build(&answers)?;
        Ok(())
    }
}
