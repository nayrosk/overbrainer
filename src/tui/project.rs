//! The Project view's model: one row per configuration field, built from the
//! effective settings, the text of `overbrainer.toml`, the keys the environment
//! sets, the pending changes and what a running stage or training uses; and the
//! project's stats.
//! Secrets are never read: a row says `set`, `unset` or `vault ref`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use super::app::App;
use super::cost::pods_cost;
use super::dataset::sizes;
use super::pipeline::command_name;
use super::tasks::TaskId;
use super::widgets::form::Input;
use crate::cli::data::Command;
use crate::config::edit::{Collection, ConfigDoc, EditError, FieldPath, Role};
use crate::config::fields::{self, FieldKind, FieldSpec, Section, TargetKind};
use crate::config::{
    Adapter, ConfigError, ENV_PREFIX, Engine, EnvSource, Pipeline, Protocol, RoleModel, Runtime,
    Settings, Target, Topic, Training, env_keys, load_str,
};
use crate::history::{Cost, Total};
use crate::runs::RunState;
use crate::secrets::is_reference;

/// The configuration the Project view shows: the effective settings, the
/// document they were read from, and which keys the environment sets.
#[derive(Debug)]
pub(super) struct ProjectConfig {
    /// `overbrainer.toml` layered with the environment.
    pub(super) settings: Settings,
    /// `overbrainer.toml` alone, comments kept.
    pub(super) doc: ConfigDoc,
    /// The dotted keys set by `OVERBRAINER_*` variables.
    pub(super) env: BTreeSet<String>,
    /// Those whose value is a `vault:` reference.
    pub(super) vault: BTreeSet<String>,
    /// The text it was read from: a save refuses to overwrite a file that no
    /// longer holds it.
    pub(super) text: String,
}

impl ProjectConfig {
    /// The configuration of the text `content` with the variables of `env`.
    ///
    /// # Errors
    ///
    /// Returns what [`load_str`] returns, or a parse error of the document.
    pub(super) fn new(content: &str, env: &EnvSource) -> Result<Self, ConfigError> {
        let settings = load_str(content, env.clone())?;
        let doc =
            ConfigDoc::parse(content).map_err(|error| ConfigError::Parse(error.to_string()))?;
        Ok(Self {
            settings,
            doc,
            env: env_keys(env),
            vault: vault_keys(env),
            text: content.to_string(),
        })
    }
}

/// The dotted keys of `env` whose value is a `vault:` reference. Only the
/// `OVERBRAINER_*` variables are looked at, and only the prefix of their value;
/// nothing of it is kept.
pub(super) fn vault_keys(env: &EnvSource) -> BTreeSet<String> {
    let prefix = format!("{ENV_PREFIX}_").to_lowercase();
    let ours = |name: &str| name.to_lowercase().starts_with(&prefix);
    let references: Vec<String> = match env {
        EnvSource::Process => std::env::vars_os()
            .filter_map(|(name, value)| Some((name.into_string().ok()?, value)))
            .filter(|(name, _)| ours(name))
            .filter(|(_, value)| value.to_str().is_some_and(is_reference))
            .map(|(name, _)| name)
            .collect(),
        EnvSource::Vars(pairs) => pairs
            .iter()
            .filter(|(name, _)| ours(name))
            .filter(|(_, value)| is_reference(value))
            .map(|(name, _)| name.clone())
            .collect(),
    };
    // Names only: `env_keys` maps them to dotted keys as `load` reads them.
    env_keys(&EnvSource::Vars(
        references
            .into_iter()
            .map(|name| (name, String::new()))
            .collect(),
    ))
}

/// The variable that sets the dotted `key`.
pub(super) fn env_var(key: &str) -> String {
    format!("{ENV_PREFIX}_{}", key.to_uppercase().replace('.', "__"))
}

/// What a field shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Shown {
    /// A value, from the file or the environment.
    Value(String),
    /// Not in the file: the value applied.
    Default(String),
    /// Not set, and no default.
    Unset,
    /// A secret with a value.
    Set,
    /// A secret naming a Vault field.
    VaultRef,
}

impl Shown {
    /// The text of the value column.
    pub(super) fn text(&self) -> &str {
        match self {
            Self::Value(value) | Self::Default(value) => value,
            Self::Unset => "unset",
            Self::Set => "set",
            Self::VaultRef => "vault ref",
        }
    }
}

/// One field of the configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Field {
    /// The key in its table.
    pub(super) name: &'static str,
    /// The dotted key, as `validate` and the environment name it.
    pub(super) key: String,
    /// Where a form edits it; `None` for an env-only field.
    pub(super) path: Option<FieldPath>,
    /// What it shows.
    pub(super) shown: Shown,
    /// Whether the environment sets it.
    pub(super) env: bool,
    /// What uses it now, making it read-only: `answers`, `run 20260921-a1`.
    pub(super) lock: Option<String>,
    /// What it is.
    pub(super) help: &'static str,
    /// Whether a pending change sets it, or adds its table.
    pub(super) changed: bool,
    /// Why the last save refused it.
    pub(super) error: Option<String>,
}

impl Field {
    /// The line under the list for the selected field: why it cannot be
    /// changed here, or what it is.
    pub(super) fn detail(&self) -> String {
        let key = &self.key;
        if let Some(error) = &self.error {
            return error.clone();
        }
        if let Some(user) = &self.lock {
            return format!("{key}: used by {user}, read-only until it ends");
        }
        self.env_note()
            .unwrap_or_else(|| format!("{key}: {}", self.help))
    }

    /// Where to change the field when only the environment may: `.env`.
    pub(super) fn env_note(&self) -> Option<String> {
        let key = &self.key;
        if self.path.is_none() {
            return Some(format!("{key}: env only, set {} in .env", env_var(key)));
        }
        self.env
            .then(|| format!("{key}: set by {}, change it in .env", env_var(key)))
    }
}

/// One row of the configuration list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Row {
    /// A table: `pipeline`, `providers.nanogpt`.
    Heading(String),
    /// A field of the table above.
    Field(Field),
}

/// What a running stage or training uses, read-only meanwhile.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct Locks {
    /// The stage running and the roles it uses.
    pub(super) stage: Option<(&'static str, Vec<Role>)>,
    /// Each training run started or followed (`run <id>`, `a new run`) and
    /// its target.
    pub(super) runs: Vec<(String, String)>,
}

impl Locks {
    /// What `app` runs now.
    pub(super) fn of(app: &App) -> Self {
        let stage = app
            .pipeline_task
            .and(app.pipeline.command)
            .map(|command| (command_name(command), roles_of(command)));
        let runs = app
            .training
            .tasks
            .values()
            .map(|follow| {
                let target = app
                    .training
                    .runs
                    .iter()
                    .find(|row| row.record.id == follow.run_id)
                    .map(|row| row.record.target.clone())
                    .or_else(|| app.project.target.clone())
                    .unwrap_or_default();
                (follow.run(), target)
            })
            .collect();
        Self { stage, runs }
    }
}

impl Locks {
    /// What uses `key` now, a field or a table, if anything: the roles the
    /// stage running uses and their providers, the training table and its
    /// run's target.
    pub(super) fn user_of(&self, settings: &Settings, key: &str) -> Option<String> {
        let mut tables: Vec<(String, &str)> = Vec::new();
        if let Some((stage, roles)) = &self.stage {
            for role in roles {
                tables.push((format!("roles.{}", role.as_str()), stage));
                let model = match role {
                    Role::Generator => Some(&settings.roles.generator),
                    Role::Parent => Some(&settings.roles.parent),
                    Role::Embedder => settings.roles.embedder.as_ref(),
                };
                if let Some(model) = model {
                    tables.push((format!("providers.{}", model.provider), stage));
                }
            }
        }
        for (run, target) in &self.runs {
            tables.push(("training".to_string(), run));
            tables.push((format!("targets.{target}"), run));
        }
        tables
            .into_iter()
            .find(|(table, _)| {
                key == table
                    || key
                        .strip_prefix(table.as_str())
                        .is_some_and(|rest| rest.starts_with('.'))
            })
            .map(|(_, user)| user.to_string())
    }
}

/// The roles `command` sends requests to.
fn roles_of(command: Command) -> Vec<Role> {
    match command {
        Command::Subtopics | Command::Questions => vec![Role::Generator, Role::Embedder],
        Command::Answers => vec![Role::Parent],
        Command::Run => Role::ALL.to_vec(),
        Command::Split => Vec::new(),
    }
}

/// The changes not saved yet: the edited document, and the dotted keys of the
/// fields they set to another value and of the tables they add or remove.
#[derive(Debug, Clone)]
pub(super) struct Pending {
    /// `overbrainer.toml` as edited.
    pub(super) doc: ConfigDoc,
    /// What changed: `pipeline.seed`, `providers.local`.
    pub(super) changed: BTreeSet<String>,
    /// For each topic of `doc`, its position in the file's document; `None`
    /// for a topic added.
    origins: Vec<Option<usize>>,
}

impl Pending {
    /// No change yet to `doc`.
    pub(super) fn new(doc: &ConfigDoc) -> Self {
        Self {
            doc: doc.clone(),
            changed: BTreeSet::new(),
            origins: (0..doc.topic_names().len()).map(Some).collect(),
        }
    }

    /// Where the field `path` of the edited document was in `original`, the
    /// file's document: `None` for a field of a topic added.
    fn original_path(&self, path: &FieldPath, original: &ConfigDoc) -> Option<FieldPath> {
        let FieldPath::Topic { index, field, .. } = path else {
            return Some(path.clone());
        };
        let at = self.origins.get(*index).copied().flatten()?;
        let name = original.topic_names().get(at)?.clone();
        Some(FieldPath::Topic {
            index: at,
            name,
            field,
        })
    }

    /// Marks the field `path` changed when its value differs from the one it
    /// has in `original`, the file's document, and unmarks it otherwise.
    pub(super) fn mark(&mut self, path: &FieldPath, original: &ConfigDoc) {
        let before = self
            .original_path(path, original)
            .map(|at| original.get(&at));
        let key = path.to_string();
        if before == Some(self.doc.get(path)) {
            self.changed.remove(&key);
        } else {
            self.changed.insert(key);
        }
    }

    /// The topic `old` is now named `new`: what was noted of it follows.
    pub(super) fn rename(&mut self, old: &str, new: &str) {
        let (old, new) = (format!("topics.{old}"), format!("topics.{new}"));
        self.changed = std::mem::take(&mut self.changed)
            .into_iter()
            .map(|key| match key.strip_prefix(old.as_str()) {
                Some(rest) if rest.is_empty() || rest.starts_with('.') => format!("{new}{rest}"),
                _ => key,
            })
            .collect();
    }

    /// Appends a topic named `name`.
    ///
    /// # Errors
    ///
    /// Returns what [`ConfigDoc::add_topic`] returns.
    pub(super) fn add_topic(&mut self, name: &str) -> Result<(), EditError> {
        let index = self.doc.add_topic(name)?;
        self.origins.truncate(index);
        self.origins.push(None);
        Ok(())
    }

    /// Removes the `index`-th topic; returns whether it existed.
    pub(super) fn remove_topic(&mut self, index: usize) -> bool {
        let removed = self.doc.remove_topic(index);
        if removed && index < self.origins.len() {
            self.origins.remove(index);
        }
        removed
    }

    /// Notes that the table `key` was added or removed: a table added then
    /// removed leaves no mark, and nothing noted inside it stays.
    pub(super) fn note_table(&mut self, key: &str, added: bool) {
        let inside = format!("{key}.");
        self.changed.retain(|noted| !noted.starts_with(&inside));
        if !added && self.changed.remove(key) {
            return;
        }
        self.changed.insert(key.to_string());
    }

    /// Whether a change sets `key` or adds its table.
    fn covers(&self, key: &str) -> bool {
        self.changed.contains(key)
            || key
                .rsplit_once('.')
                .is_some_and(|(table, _)| self.changed.contains(table))
    }
}

/// The rows of `config` with the `pending` changes: project, topics,
/// providers, roles, pipeline, training, targets, then the env-only `runpod`
/// and the rest. Topics, providers and targets are those of the document,
/// with the tables only the environment sets.
pub(super) fn rows(config: &ProjectConfig, pending: Option<&Pending>, locks: &Locks) -> Vec<Row> {
    let settings = &config.settings;
    let mut rows = Builder {
        config,
        doc: pending.map_or(&config.doc, |pending| &pending.doc),
        pending,
        rows: Vec::new(),
    };
    rows.heading("project");
    rows.fields(&Table {
        section: Section::Project,
        path: &|field| FieldPath::Project(field),
        value: &|field| (field == "name").then(|| settings.project.name.clone()),
        lock: None,
    });
    for (index, name) in rows.doc.topic_names().into_iter().enumerate() {
        let topic = settings.topics.iter().find(|topic| topic.name == name);
        rows.heading(&format!("topics.{name}"));
        rows.fields(&Table {
            section: Section::Topic,
            path: &|field| FieldPath::Topic {
                index,
                name: name.clone(),
                field,
            },
            value: &|field| topic.and_then(|topic| topic_value(topic, field)),
            lock: None,
        });
    }
    providers_and_roles(&mut rows, settings, locks);
    rows.heading("pipeline");
    rows.fields(&Table {
        section: Section::Pipeline,
        path: &|field| FieldPath::Pipeline(field),
        value: &|field| pipeline_value(&settings.pipeline, field),
        lock: None,
    });
    training_and_targets(&mut rows, settings, locks);
    rows.heading("runpod");
    rows.secret(
        ("runpod", "api_key", "Runpod API key"),
        settings.runpod.api_key.is_some(),
        None,
    );
    rows.env_only(
        ("runpod", "base_url", "Base URL of the Runpod API"),
        settings
            .runpod
            .base_url
            .clone()
            .map_or(Shown::Unset, Shown::Value),
        None,
    );
    rows.heading("other");
    rows.secret(
        ("", "hf_token", "Hugging Face token"),
        settings.hf_token.is_some(),
        None,
    );
    rows.env_only(
        ("", "log", "Log filter"),
        settings.log.clone().map_or(Shown::Unset, Shown::Value),
        None,
    );
    rows.rows
}

/// The names of `collection` in the document, with those only the
/// environment sets among `effective`, sorted.
fn names<'a>(
    rows: &Builder<'_>,
    collection: Collection,
    effective: impl Iterator<Item = &'a String>,
) -> BTreeSet<String> {
    let mut names: BTreeSet<String> = rows.doc.names(collection).into_iter().collect();
    for name in effective {
        let table = format!("{}.{name}.", collection.key());
        if rows.config.env.iter().any(|key| key.starts_with(&table)) {
            names.insert(name.clone());
        }
    }
    names
}

/// The providers, then the roles; those the stage running uses are locked.
fn providers_and_roles(rows: &mut Builder<'_>, settings: &Settings, locks: &Locks) {
    let stage = locks.stage.as_ref();
    let locked_roles: Vec<Role> = stage.map(|(_, roles)| roles.clone()).unwrap_or_default();
    let stage_user = stage.map(|(name, _)| (*name).to_string());
    let role_model = |role: Role| match role {
        Role::Generator => Some(&settings.roles.generator),
        Role::Parent => Some(&settings.roles.parent),
        Role::Embedder => settings.roles.embedder.as_ref(),
    };
    let locked_providers: BTreeSet<&str> = locked_roles
        .iter()
        .filter_map(|role| role_model(*role))
        .map(|model| model.provider.as_str())
        .collect();
    for name in names(rows, Collection::Providers, settings.providers.keys()) {
        let provider = settings.providers.get(&name);
        let lock = locked_providers
            .contains(name.as_str())
            .then(|| stage_user.clone())
            .flatten();
        rows.heading(&format!("providers.{name}"));
        rows.fields(&Table {
            section: Section::Provider,
            path: &|field| FieldPath::Provider {
                name: name.clone(),
                field,
            },
            value: &|field| {
                provider
                    .filter(|_| field == "protocol")
                    .map(|provider| protocol(provider.protocol).to_string())
            },
            lock: lock.clone(),
        });
        let key = format!("providers.{name}");
        rows.env_only(
            (&key, "base_url", "Base URL of the API"),
            provider
                .and_then(|provider| provider.base_url.clone())
                .map_or(Shown::Unset, Shown::Value),
            lock.clone(),
        );
        rows.secret(
            (&key, "api_key", "API key"),
            provider.is_some_and(|provider| provider.api_key.is_some()),
            lock,
        );
    }
    for role in Role::ALL {
        let lock = locked_roles
            .contains(&role)
            .then(|| stage_user.clone())
            .flatten();
        rows.heading(&format!("roles.{}", role.as_str()));
        let model = role_model(role);
        rows.fields(&Table {
            section: Section::Role,
            path: &|field| FieldPath::Role { role, field },
            value: &|field| model.and_then(|model| role_value(model, field)),
            lock,
        });
    }
}

/// The training table, then the targets; the training run and its target
/// are locked.
fn training_and_targets(rows: &mut Builder<'_>, settings: &Settings, locks: &Locks) {
    let run_user = locks.runs.first().map(|(run, _)| run.clone());
    rows.heading("training");
    rows.fields(&Table {
        section: Section::Training,
        path: &|field| FieldPath::Training(field),
        value: &|field| {
            settings
                .training
                .as_ref()
                .and_then(|training| training_value(training, field))
        },
        lock: run_user.clone(),
    });
    for name in names(rows, Collection::Targets, settings.targets.keys()) {
        let target = settings.targets.get(&name);
        let Some(kind) = rows.doc.target_kind(&name).or(target.map(target_kind)) else {
            continue;
        };
        let lock = locks
            .runs
            .iter()
            .find(|(_, used)| *used == name)
            .map(|(run, _)| run.clone());
        rows.heading(&format!("targets.{name} ({})", kind.as_str()));
        rows.fields(&Table {
            section: Section::Target(kind),
            path: &|field| FieldPath::Target {
                name: name.clone(),
                field,
            },
            value: &|field| target.and_then(|target| target_value(target, field)),
            lock: lock.clone(),
        });
        if kind == TargetKind::Ssh {
            let host = match target {
                Some(Target::Ssh { host, .. }) => host.clone(),
                _ => None,
            };
            rows.env_only(
                (
                    &format!("targets.{name}"),
                    "host",
                    "user@host or an ssh alias",
                ),
                host.map_or(Shown::Unset, Shown::Value),
                lock,
            );
        }
    }
}

/// The fields of one table.
struct Table<'a> {
    section: Section,
    /// Where each field is.
    path: &'a dyn Fn(&'static str) -> FieldPath,
    /// Its effective value, `None` when unset.
    value: &'a dyn Fn(&str) -> Option<String>,
    /// What uses the table now.
    lock: Option<String>,
}

/// The rows being built.
struct Builder<'a> {
    config: &'a ProjectConfig,
    /// The document shown: the pending one, else the file's.
    doc: &'a ConfigDoc,
    pending: Option<&'a Pending>,
    rows: Vec<Row>,
}

impl Builder<'_> {
    fn heading(&mut self, text: &str) {
        self.rows.push(Row::Heading(text.to_string()));
    }

    /// Every editable field of `table`: its value in the document, else the
    /// one the environment sets, else its default. A field a pending change
    /// took out of the document is unset: the value read before no longer
    /// applies.
    fn fields(&mut self, table: &Table<'_>) {
        for FieldSpec { name, help, .. } in fields::for_section(table.section) {
            let path = (table.path)(name);
            let key = path.to_string();
            let env = self.config.env.contains(&key);
            let changed = self.pending.is_some_and(|pending| pending.covers(&key));
            let effective = (table.value)(name);
            let shown = match (env, self.doc.get(&path), effective) {
                (true, _, Some(value)) | (false, Some(value), _) => Shown::Value(value),
                (false, None, Some(value)) if !changed => Shown::Default(value),
                _ => Shown::Unset,
            };
            self.rows.push(Row::Field(Field {
                name,
                key,
                path: Some(path),
                shown,
                env,
                lock: table.lock.clone(),
                help,
                changed,
                error: None,
            }));
        }
    }

    /// An env-only field `name` of the table `table` (`""` at the top).
    fn env_only(
        &mut self,
        (table, name, help): (&str, &'static str, &'static str),
        shown: Shown,
        lock: Option<String>,
    ) {
        let key = if table.is_empty() {
            name.to_string()
        } else {
            format!("{table}.{name}")
        };
        self.rows.push(Row::Field(Field {
            name,
            env: self.config.env.contains(&key),
            key,
            path: None,
            shown,
            lock,
            help,
            changed: false,
            error: None,
        }));
    }

    /// An env-only secret: whether it is `set` decides the row, never its value.
    fn secret(&mut self, at: (&str, &'static str, &'static str), set: bool, lock: Option<String>) {
        let key = if at.0.is_empty() {
            at.1.to_string()
        } else {
            format!("{}.{}", at.0, at.1)
        };
        let shown = match (set, self.config.vault.contains(&key)) {
            (false, _) => Shown::Unset,
            (true, true) => Shown::VaultRef,
            (true, false) => Shown::Set,
        };
        self.env_only(at, shown, lock);
    }
}

fn protocol(protocol: Protocol) -> &'static str {
    match protocol {
        Protocol::Openai => "openai",
        Protocol::Anthropic => "anthropic",
    }
}

fn adapter(adapter: Adapter) -> &'static str {
    match adapter {
        Adapter::Lora => "lora",
        Adapter::Qlora => "qlora",
        Adapter::Full => "full",
    }
}

fn runtime(runtime: Runtime) -> &'static str {
    match runtime {
        Runtime::Docker => "docker",
        Runtime::Native => "native",
    }
}

/// A float as `ConfigDoc` shows it.
fn float(value: f64) -> String {
    format!("{value:?}")
}

fn target_kind(target: &Target) -> TargetKind {
    match target {
        Target::Local { .. } => TargetKind::Local,
        Target::Ssh { .. } => TargetKind::Ssh,
        Target::Runpod { .. } => TargetKind::Runpod,
    }
}

fn topic_value(topic: &Topic, field: &str) -> Option<String> {
    match field {
        "name" => Some(topic.name.clone()),
        "description" => topic.description.clone(),
        "subtopics" => Some(topic.subtopics.to_string()),
        "questions_per_subtopic" => Some(topic.questions_per_subtopic.to_string()),
        _ => None,
    }
}

fn role_value(role: &RoleModel, field: &str) -> Option<String> {
    match field {
        "provider" => Some(role.provider.clone()),
        "model" => Some(role.model.clone()),
        "reasoning" => Some(role.reasoning.to_string()),
        "max_tokens" => Some(role.max_tokens.to_string()),
        "temperature" => role.temperature.map(float),
        "reasoning_effort" => role
            .reasoning_effort
            .map(|effort| effort.as_str().to_string()),
        "thinking_budget" => role.thinking_budget.map(|budget| budget.to_string()),
        _ => None,
    }
}

fn pipeline_value(pipeline: &Pipeline, field: &str) -> Option<String> {
    Some(match field {
        "concurrency" => pipeline.concurrency.to_string(),
        "max_retries" => pipeline.max_retries.to_string(),
        "dedup_threshold" => float(pipeline.dedup_threshold),
        "eval_ratio" => float(pipeline.eval_ratio),
        "seed" => pipeline.seed.to_string(),
        "include_system_prompt" => pipeline.include_system_prompt.to_string(),
        "embedding_threshold" => float(pipeline.embedding_threshold),
        "question_batch_size" => pipeline.question_batch_size.to_string(),
        "request_timeout_secs" => pipeline.request_timeout_secs.to_string(),
        _ => return None,
    })
}

fn training_value(training: &Training, field: &str) -> Option<String> {
    Some(match field {
        "target" => training.target.clone(),
        "base_model" => training.base_model.clone(),
        "adapter" => adapter(training.adapter).to_string(),
        "epochs" => training.epochs.to_string(),
        "learning_rate" => float(training.learning_rate),
        "lora_r" => training.lora_r.to_string(),
        "lora_alpha" => training.lora_alpha.to_string(),
        "lora_dropout" => float(training.lora_dropout),
        "sequence_len" => training.sequence_len.to_string(),
        "micro_batch_size" => training.micro_batch_size.to_string(),
        "gradient_accumulation_steps" => training.gradient_accumulation_steps.to_string(),
        "optimizer" => training.optimizer.clone(),
        "lr_scheduler" => training.lr_scheduler.clone(),
        "sample_packing" => training.sample_packing.to_string(),
        "evals_per_epoch" => training.evals_per_epoch.to_string(),
        "saves_per_epoch" => training.saves_per_epoch.to_string(),
        "merge" => training.merge.to_string(),
        "hub_model_id" => return training.hub_model_id.clone(),
        _ => return None,
    })
}

fn target_value(target: &Target, field: &str) -> Option<String> {
    let engine = |engine: &Option<Engine>| engine.map(|engine| engine.command().to_string());
    match target {
        Target::Local {
            runtime: run,
            engine: used,
            image,
            venv,
        } => match field {
            "runtime" => Some(runtime(*run).to_string()),
            "engine" => engine(used),
            "image" => image.clone(),
            "venv" => venv.clone(),
            _ => None,
        },
        Target::Ssh {
            runtime: run,
            workdir,
            engine: used,
            image,
            venv,
            ..
        } => match field {
            "runtime" => Some(runtime(*run).to_string()),
            "workdir" => workdir.clone(),
            "engine" => engine(used),
            "image" => image.clone(),
            "venv" => venv.clone(),
            _ => None,
        },
        Target::Runpod {
            gpu_types,
            gpu_count,
            image,
            venv,
            container_disk_gb,
            max_hours,
            boot_grace_minutes,
            retrieve_grace_minutes,
            data_center_ids,
            network_volume_id,
        } => match field {
            "gpu_types" => Some(gpu_types.join(", ")),
            "gpu_count" => Some(gpu_count.to_string()),
            "image" => image.clone(),
            "venv" => venv.clone(),
            "container_disk_gb" => Some(container_disk_gb.to_string()),
            "max_hours" => Some(float(*max_hours)),
            "boot_grace_minutes" => Some(boot_grace_minutes.to_string()),
            "retrieve_grace_minutes" => Some(retrieve_grace_minutes.to_string()),
            "data_center_ids" => (!data_center_ids.is_empty()).then(|| data_center_ids.join(", ")),
            "network_volume_id" => network_volume_id.clone(),
            _ => None,
        },
    }
}

/// The rows of the Project view, with where its fields are.
#[derive(Debug, Default)]
pub(super) struct Listing {
    /// Headings and fields, in order.
    pub(super) rows: Vec<Row>,
    /// The position in `rows` of each field.
    pub(super) fields: Vec<usize>,
}

impl Listing {
    /// `rows`, each field marked with its error in `errors`.
    pub(super) fn new(mut rows: Vec<Row>, errors: &BTreeMap<String, String>) -> Self {
        let mut fields = Vec::new();
        for (at, row) in rows.iter_mut().enumerate() {
            if let Row::Field(field) = row {
                field.error = errors.get(&field.key).cloned();
                fields.push(at);
            }
        }
        Self { rows, fields }
    }

    /// The `index`-th field.
    pub(super) fn field(&self, index: usize) -> Option<&Field> {
        match self.fields.get(index).and_then(|at| self.rows.get(*at)) {
            Some(Row::Field(field)) => Some(field),
            _ => None,
        }
    }

    /// The position among the fields of the first one whose key is `key` or
    /// lies in the table `key`.
    pub(super) fn find(&self, key: &str) -> Option<usize> {
        let table = format!("{key}.");
        (0..self.fields.len()).find(|index| {
            self.field(*index)
                .is_some_and(|field| field.key == key || field.key.starts_with(&table))
        })
    }
}

/// What `a` adds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Addable {
    /// A `[[topics]]` entry.
    Topic,
    /// A `[providers.<name>]` table.
    Provider,
    /// A `[targets.<name>]` table.
    Target,
}

impl Addable {
    /// Every kind, in the order the form offers them.
    pub(super) const ALL: [Self; 3] = [Self::Topic, Self::Provider, Self::Target];

    /// Its name in the form.
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Topic => "topic",
            Self::Provider => "provider",
            Self::Target => "target",
        }
    }

    /// The choices asked after the name: a provider's protocol, a target's kind.
    pub(super) fn kinds(self) -> &'static [&'static str] {
        match self {
            Self::Topic => &[],
            Self::Provider => match fields::find(Section::Provider, "protocol") {
                Some(FieldSpec {
                    kind: FieldKind::Choice(protocols),
                    ..
                }) => protocols,
                _ => &[],
            },
            // `TargetKind::ALL` by name; a test keeps them in step.
            Self::Target => &["local", "ssh", "runpod"],
        }
    }
}

/// What the Project view's form asks, in the detail rows under the list.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Form {
    /// The value of the field `path`.
    Value {
        /// The field.
        path: FieldPath,
        /// What it accepts.
        kind: FieldKind,
        /// Whether an empty value unsets it.
        optional: bool,
        /// The text typed.
        input: Input,
        /// Why the last Enter was refused.
        error: Option<String>,
    },
    /// What to add, the `0`-th of [`Addable::ALL`] selected.
    Adding(usize),
    /// The name of the new table.
    Name {
        /// What is added.
        what: Addable,
        /// The name typed.
        input: Input,
        /// Why the last Enter was refused.
        error: Option<String>,
    },
    /// A provider's protocol or a target's kind.
    Kind {
        /// What is added.
        what: Addable,
        /// Its name.
        name: String,
        /// The choice selected, among [`Addable::kinds`].
        choice: usize,
    },
}

/// The Project view's state: the selection, the changes not saved yet, the
/// form open, and the rows as last built.
#[derive(Debug, Default)]
pub(super) struct ProjectView {
    /// Position of the selected field among the fields.
    pub(super) selected: usize,
    /// The first row shown, set at each draw so the selection stays in view.
    pub(super) offset: usize,
    /// The changes not saved yet.
    pub(super) pending: Option<Pending>,
    /// Why the last save was refused, by field key.
    pub(super) errors: BTreeMap<String, String>,
    /// The form open, if any.
    pub(super) form: Option<Form>,
    /// The save running, if any.
    pub(super) save: Option<TaskId>,
    /// Whether the editor open is on `overbrainer.toml`.
    pub(super) editing: bool,
    /// Bumped by every change of what the rows show but the locks.
    generation: u64,
    /// The rows built last, for this generation and these locks.
    cache: Option<(u64, Locks, Arc<Listing>)>,
}

impl ProjectView {
    /// Moves the selection `by` fields, down when positive, kept within the
    /// `count` fields.
    pub(super) fn step(&mut self, by: isize, count: usize) {
        let last = count.saturating_sub(1);
        self.selected = self.selected.saturating_add_signed(by).min(last);
    }

    /// Notes that the configuration, the pending changes or the errors changed:
    /// the rows are built again.
    pub(super) fn touch(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    /// The rows of `config` with the pending changes and `locks`: those built
    /// last while nothing changed.
    pub(super) fn listing(&mut self, config: Option<&ProjectConfig>, locks: Locks) -> Arc<Listing> {
        if let Some((generation, cached, listing)) = &self.cache
            && *generation == self.generation
            && *cached == locks
        {
            return Arc::clone(listing);
        }
        let listing = Arc::new(config.map_or_else(Listing::default, |config| {
            Listing::new(rows(config, self.pending.as_ref(), &locks), &self.errors)
        }));
        self.cache = Some((self.generation, locks, Arc::clone(&listing)));
        listing
    }

    /// How many changes are pending.
    pub(super) fn changes(&self) -> usize {
        self.pending
            .as_ref()
            .map_or(0, |pending| pending.changed.len())
    }
}

/// A line of the stats pane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Stat {
    /// A group: `cost and tokens`, `dataset`.
    Heading(&'static str),
    /// A label and its value.
    Pair(String, String),
    /// A line alone.
    Note(String),
}

/// Tokens in a few columns: `980`, `3.0k`, `53k`, `1.2M`.
fn tokens(count: u64) -> String {
    let tenths = |unit: u64| {
        let tenths = (count + unit / 20) / (unit / 10);
        format!("{}.{}", tenths / 10, tenths % 10)
    };
    match count {
        0..1000 => count.to_string(),
        1000..9950 => format!("{}k", tenths(1000)),
        9950..999_500 => format!("{}k", (count + 500) / 1000),
        _ => format!("{}M", tenths(1_000_000)),
    }
}

/// Tokens in and out, then the cost, of `total`.
fn spent(total: &Total) -> String {
    let cost = match total.cost {
        Cost::Unknown => "cost ?".to_string(),
        cost => cost.to_string(),
    };
    format!(
        "{}/{} {cost}",
        tokens(total.input_tokens),
        tokens(total.output_tokens)
    )
}

/// The stats of `app`: cost and tokens per stage and per model, the dataset,
/// the training runs and what their Runpod pods spent.
pub(super) fn stats(app: &App) -> Vec<Stat> {
    let mut lines = vec![Stat::Heading("cost and tokens (in/out)")];
    let history = &app.history;
    if history.stages.values().any(Total::spent) {
        for (stage, total) in history.stages.iter().filter(|(_, total)| total.spent()) {
            lines.push(Stat::Pair(stage.name().to_string(), spent(total)));
        }
        lines.push(Stat::Pair("all".to_string(), spent(&history.all)));
    } else {
        lines.push(Stat::Note("nothing spent yet".to_string()));
    }
    lines.push(Stat::Heading("dataset"));
    lines.push(Stat::Pair(
        "topics".to_string(),
        app.project.topics.len().to_string(),
    ));
    match app
        .dataset
        .model
        .as_ref()
        .and_then(|model| model.stats.get(&None))
    {
        Some(all) => {
            let excluded: usize = all.excluded.values().sum();
            let (train, eval, label) = sizes(
                all.usable,
                app.project.eval_ratio,
                app.dataset.split.as_ref(),
            );
            lines.extend([
                Stat::Pair("subtopics".to_string(), all.subtopics.to_string()),
                Stat::Pair("questions".to_string(), all.questions.to_string()),
                Stat::Pair(
                    "answers".to_string(),
                    format!("{} usable, {excluded} excluded", all.usable),
                ),
                Stat::Pair(
                    "train/eval".to_string(),
                    format!("{train} / {eval} ({label})"),
                ),
            ]);
        },
        None => lines.push(Stat::Note("not read yet".to_string())),
    }
    lines.push(Stat::Heading("training"));
    // An exhaustive match orders the states: a new one cannot be left out.
    let rank = |state: RunState| match state {
        RunState::Preparing => 0,
        RunState::Running => 1,
        RunState::Succeeded => 2,
        RunState::Failed => 3,
        RunState::Cancelled => 4,
    };
    let mut states: BTreeMap<u8, (&str, usize)> = BTreeMap::new();
    for run in &app.training.runs {
        let state = run.record.state;
        states.entry(rank(state)).or_insert((state.name(), 0)).1 += 1;
    }
    if states.is_empty() {
        lines.push(Stat::Note("no runs yet".to_string()));
    }
    for (name, count) in states.into_values() {
        lines.push(Stat::Pair(name.to_string(), count.to_string()));
    }
    if let Some(pods) = pods_cost(app) {
        let spend = match pods {
            Cost::Unknown => "unknown".to_string(),
            Cost::Partial(cost) => format!("about ${cost:.2}+"),
            Cost::Known(cost) => format!("about ${cost:.2}"),
        };
        lines.push(Stat::Pair("pod spend".to_string(), spend));
    }
    if !history.models.is_empty() {
        lines.push(Stat::Heading("models"));
        for (model, total) in &history.models {
            lines.push(Stat::Pair(model.clone(), spent(total)));
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::snapshots::{
        FINISHED, SECRET, app, dataset_app, pipeline_running, project_config as config,
        project_env as env, training_app,
    };
    use crate::tui::tasks::TaskId;
    use crate::tui::training::{Follow, Job};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn field<'a>(rows: &'a [Row], key: &str) -> Result<&'a Field, String> {
        rows.iter()
            .find_map(|row| match row {
                Row::Field(field) if field.key == key => Some(field),
                _ => None,
            })
            .ok_or_else(|| format!("no row {key}"))
    }

    #[test]
    fn vault_keys_are_the_references_of_the_environment() {
        let keys = vault_keys(&env());
        assert_eq!(
            keys.into_iter().collect::<Vec<_>>(),
            vec!["runpod.api_key".to_string()]
        );
        let others = EnvSource::Vars(vec![
            ("VAULT_REF".into(), "vault:secret/x#y".into()),
            (
                "overbrainer_hf_token".into(),
                "vault:secret/hf#token".into(),
            ),
        ]);
        assert_eq!(
            vault_keys(&others).into_iter().collect::<Vec<_>>(),
            vec!["hf_token".to_string()],
            "only OVERBRAINER_* variables, in any case"
        );
    }

    #[test]
    fn the_rows_follow_the_file_order_with_the_env_only_tables_last() -> TestResult {
        let rows = rows(&config()?, None, &Locks::default());
        let headings: Vec<&str> = rows
            .iter()
            .filter_map(|row| match row {
                Row::Heading(text) => Some(text.as_str()),
                Row::Field(_) => None,
            })
            .collect();
        assert_eq!(
            headings,
            [
                "project",
                "topics.ownership",
                "providers.claude",
                "providers.nanogpt",
                "roles.generator",
                "roles.parent",
                "roles.embedder",
                "pipeline",
                "training",
                "targets.gpu_cloud (runpod)",
                "runpod",
                "other",
            ]
        );
        Ok(())
    }

    #[test]
    fn values_come_from_the_file_the_environment_or_the_defaults() -> TestResult {
        let rows = rows(&config()?, None, &Locks::default());
        let concurrency = field(&rows, "pipeline.concurrency")?;
        assert_eq!(concurrency.shown, Shown::Value("4".into()));
        assert!(concurrency.env);
        let seed = field(&rows, "pipeline.seed")?;
        assert_eq!(seed.shown, Shown::Default("42".into()));
        assert!(!seed.env);
        let name = field(&rows, "project.name")?;
        assert_eq!(name.shown, Shown::Value("rust_expert".into()));
        assert_eq!(
            field(&rows, "roles.parent.reasoning")?.shown,
            Shown::Value("true".into())
        );
        assert_eq!(
            field(&rows, "roles.parent.temperature")?.shown,
            Shown::Unset
        );
        assert_eq!(
            field(&rows, "targets.gpu_cloud.max_hours")?.shown,
            Shown::Value("6".into()),
            "as written in the file"
        );
        assert_eq!(
            field(&rows, "targets.gpu_cloud.container_disk_gb")?.shown,
            Shown::Default("50".into())
        );
        assert_eq!(field(&rows, "roles.embedder.model")?.shown, Shown::Unset);
        Ok(())
    }

    #[test]
    fn secrets_show_set_unset_or_vault_ref_and_never_their_value() -> TestResult {
        let rows = rows(&config()?, None, &Locks::default());
        let key = field(&rows, "providers.nanogpt.api_key")?;
        assert_eq!(key.shown, Shown::Set);
        assert!(key.env);
        assert!(key.path.is_none(), "env only");
        assert_eq!(
            field(&rows, "providers.claude.api_key")?.shown,
            Shown::Unset
        );
        assert_eq!(field(&rows, "runpod.api_key")?.shown, Shown::VaultRef);
        assert_eq!(field(&rows, "hf_token")?.shown, Shown::Unset);
        assert_eq!(
            field(&rows, "providers.nanogpt.base_url")?.shown,
            Shown::Value("https://nano-gpt.com/api/v1".into())
        );
        let all = format!("{rows:?}");
        assert!(!all.contains(SECRET), "the secret value is in a row");
        assert!(
            !all.contains("secret/overbrainer"),
            "the vault path is in a row"
        );
        Ok(())
    }

    #[test]
    fn the_detail_says_where_a_value_comes_from() -> TestResult {
        let rows = rows(&config()?, None, &Locks::default());
        assert_eq!(
            field(&rows, "providers.nanogpt.api_key")?.detail(),
            "providers.nanogpt.api_key: env only, set \
             OVERBRAINER_PROVIDERS__NANOGPT__API_KEY in .env"
        );
        assert_eq!(
            field(&rows, "pipeline.concurrency")?.detail(),
            "pipeline.concurrency: set by OVERBRAINER_PIPELINE__CONCURRENCY, change it in .env"
        );
        assert_eq!(
            field(&rows, "pipeline.seed")?.detail(),
            "pipeline.seed: Seed of the train/eval split (default 42)"
        );
        Ok(())
    }

    #[test]
    fn a_stage_locks_its_roles_and_their_providers() -> TestResult {
        let locks = Locks {
            stage: Some(("answers", roles_of(Command::Answers))),
            runs: Vec::new(),
        };
        let rows = rows(&config()?, None, &locks);
        let locked: Vec<&str> = rows
            .iter()
            .filter_map(|row| match row {
                Row::Field(field) if field.lock.is_some() => Some(field.key.as_str()),
                _ => None,
            })
            .collect();
        assert!(locked.contains(&"roles.parent.model"), "{locked:?}");
        assert!(locked.contains(&"providers.claude.protocol"), "{locked:?}");
        assert!(locked.contains(&"providers.claude.api_key"), "{locked:?}");
        assert!(!locked.iter().any(|key| key.starts_with("roles.generator")));
        assert!(
            !locked
                .iter()
                .any(|key| key.starts_with("providers.nanogpt"))
        );
        assert!(!locked.iter().any(|key| key.starts_with("training")));
        assert_eq!(
            field(&rows, "roles.parent.model")?.detail(),
            "roles.parent.model: used by answers, read-only until it ends"
        );
        Ok(())
    }

    #[test]
    fn a_training_locks_the_training_table_and_its_target() -> TestResult {
        let locks = Locks {
            stage: None,
            runs: vec![("run 20260921-a1".into(), "gpu_cloud".into())],
        };
        let rows = rows(&config()?, None, &locks);
        for key in ["training.epochs", "targets.gpu_cloud.max_hours"] {
            assert_eq!(
                field(&rows, key)?.lock.as_deref(),
                Some("run 20260921-a1"),
                "{key}"
            );
        }
        assert_eq!(field(&rows, "roles.parent.model")?.lock, None);
        Ok(())
    }

    #[test]
    fn the_locks_of_an_app_are_what_it_runs() {
        let mut app = app();
        assert_eq!(Locks::of(&app), Locks::default());
        pipeline_running(&mut app);
        app.training
            .tasks
            .insert(TaskId(9), Follow::new(Job::Attach, "20260921-a1"));
        let locks = Locks::of(&app);
        assert_eq!(locks.stage, Some(("run", Role::ALL.to_vec())));
        assert_eq!(
            locks.runs,
            [("run 20260921-a1".to_string(), "gpu_cloud".to_string())]
        );
    }

    #[test]
    fn a_start_locks_for_a_new_run_on_the_configured_target() {
        let mut app = app();
        app.training
            .tasks
            .insert(TaskId(4), Follow::new(Job::Start { runpod: true }, ""));
        assert_eq!(
            Locks::of(&app).runs,
            [("a new run".to_string(), "gpu_cloud".to_string())]
        );
    }

    #[test]
    fn a_followed_run_locks_the_target_of_its_record() -> TestResult {
        let mut app = training_app()?;
        app.training.tasks.clear();
        app.training
            .tasks
            .insert(TaskId(4), Follow::new(Job::Attach, FINISHED));
        assert_eq!(
            Locks::of(&app).runs,
            [(format!("run {FINISHED}"), "homelab".to_string())],
            "the run's own target, not training.target"
        );
        Ok(())
    }

    #[test]
    fn two_followed_runs_lock_both_targets() -> TestResult {
        let mut app = training_app()?;
        app.training
            .tasks
            .insert(TaskId(4), Follow::new(Job::Attach, FINISHED));
        let locks = Locks::of(&app);
        assert_eq!(locks.runs.len(), 2, "{:?}", locks.runs);
        let config = config()?;
        let settings = &config.settings;
        let followed = locks.user_of(settings, "targets.gpu_cloud.max_hours");
        assert_eq!(followed.as_deref(), Some("run 20260921-133200-a1b2"));
        let finished = locks.user_of(settings, "targets.homelab.host");
        assert_eq!(finished, Some(format!("run {FINISHED}")));
        assert!(locks.user_of(settings, "training.epochs").is_some());
        let rows = rows(&config, None, &locks);
        assert_eq!(
            field(&rows, "targets.gpu_cloud.max_hours")?.lock.as_deref(),
            Some("run 20260921-133200-a1b2")
        );
        Ok(())
    }

    #[test]
    fn questions_lock_the_embedder_and_the_generator() -> TestResult {
        let locks = Locks {
            stage: Some(("questions", roles_of(Command::Questions))),
            runs: Vec::new(),
        };
        let rows = rows(&config()?, None, &locks);
        for key in [
            "roles.embedder.model",
            "roles.generator.model",
            "providers.nanogpt.protocol",
        ] {
            assert_eq!(
                field(&rows, key)?.lock.as_deref(),
                Some("questions"),
                "{key}"
            );
        }
        assert_eq!(field(&rows, "roles.parent.model")?.lock, None);
        Ok(())
    }

    #[test]
    fn the_stats_count_the_runs_per_state_in_order() -> TestResult {
        let app = training_app()?;
        let lines = stats(&app);
        let runs: Vec<&Stat> = lines
            .iter()
            .skip_while(|line| **line != Stat::Heading("training"))
            .skip(1)
            .take(2)
            .collect();
        assert_eq!(
            runs,
            [
                &Stat::Pair("running".into(), "2".into()),
                &Stat::Pair("succeeded".into(), "1".into()),
            ]
        );
        Ok(())
    }

    #[test]
    fn step_stays_within_the_fields() {
        let mut view = ProjectView::default();
        view.step(-1, 5);
        assert_eq!(view.selected, 0);
        view.step(10, 5);
        assert_eq!(view.selected, 4);
        view.step(-2, 5);
        assert_eq!(view.selected, 2);
    }

    #[test]
    fn tokens_are_compact() {
        assert_eq!(tokens(980), "980");
        assert_eq!(tokens(3000), "3.0k");
        assert_eq!(tokens(9949), "9.9k");
        assert_eq!(tokens(9950), "10k");
        assert_eq!(tokens(53_290), "53k");
        assert_eq!(tokens(999_499), "999k");
        assert_eq!(tokens(999_500), "1.0M");
        assert_eq!(tokens(1_234_567), "1.2M");
    }

    #[test]
    fn the_stats_say_what_is_not_known_yet() {
        let app = app();
        let lines = stats(&app);
        assert!(lines.contains(&Stat::Note("nothing spent yet".into())));
        assert!(lines.contains(&Stat::Note("not read yet".into())));
        assert!(lines.contains(&Stat::Note("no runs yet".into())));
    }

    #[test]
    fn the_stats_count_the_dataset() {
        let app = dataset_app();
        let lines = stats(&app);
        assert!(
            lines.contains(&Stat::Pair("answers".into(), "3 usable, 1 excluded".into())),
            "{lines:?}"
        );
    }
}
