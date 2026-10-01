//! Editing the configuration from the Project view: the keys, the form under
//! the list, the confirmations, the writes and the read after `$EDITOR`. Each
//! change is validated, then written to `overbrainer.toml` at once; `u` undoes
//! the last write.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;
use std::process::ExitStatus;
use std::sync::Arc;

use crossterm::event::KeyCode;

use super::app::{Action, App, Confirm, Effect, Origin, Overlay, PAGE, Picked, Project, Severity};
use super::catalog::{
    CatalogKind, DEFAULT_IMAGE, FitBy, NO_VOLUME, Query, Sizing, cost_hint, gpu_count_hint,
    volume_data_center,
};
use super::follow::NOT_STARTED;
use super::project::{Addable, Form, Listing, Locks, ProjectConfig, Shown, Undo, Writing, rows};
use super::start::{AUTO_LIMITS, GPU_TYPES};
use super::tasks::Task;
use super::widgets::form::{Input, InputOutcome};
use super::widgets::picker::Choice;
use crate::config::edit::{Collection, ConfigDoc, FieldPath};
use crate::config::fields::{FieldKind, FieldValue, TargetKind};
use crate::config::{
    CONFIG_FILE, ConfigError, EnvSource, ListOrAuto, Protocol, is_valid_name, stamp,
};

/// What a confirmed `d` takes out of `overbrainer.toml`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Removal {
    /// The topic at this index, of this name.
    Topic(usize, String),
    /// A provider or a target.
    Table(Collection, String),
}

impl Removal {
    /// The dotted key of what goes: `topics.traits`, `providers.claude`.
    pub(super) fn key(&self) -> String {
        match self {
            Self::Topic(_, name) => format!("topics.{name}"),
            Self::Table(collection, name) => format!("{}.{name}", collection.key()),
        }
    }
}

/// Why a save wrote nothing.
#[derive(Debug)]
pub(super) enum SaveRefusal {
    /// The text does not validate: one message per problem, each naming the
    /// key it is about.
    Invalid(Vec<String>),
    /// The file changed since it was read, or it cannot be written.
    Failed(String),
}

/// Validates `text` with the variables of `env`, then writes it to
/// `overbrainer.toml` in `dir` atomically, unless the file no longer holds
/// `base`, the text it was read from, up to the rename
/// ([`super::project_save`]). Returns the configuration written.
///
/// # Errors
///
/// Returns [`SaveRefusal::Invalid`] when `text` does not validate and
/// [`SaveRefusal::Failed`] when the file changed or cannot be read or written;
/// nothing is written then.
pub(super) fn save_config(
    dir: &Path,
    text: &str,
    base: &str,
    env: &EnvSource,
) -> Result<ProjectConfig, SaveRefusal> {
    let config = ProjectConfig::new(text, env).map_err(|error| match error {
        ConfigError::Invalid(problems) => SaveRefusal::Invalid(problems),
        ConfigError::Parse(problem) => SaveRefusal::Invalid(vec![problem]),
        ConfigError::Read { .. } => SaveRefusal::Failed(error.to_string()),
    })?;
    super::project_save::stage(dir, text, base)?.commit()?;
    let mut config = config;
    config.stamp = Some(stamp(dir));
    Ok(config)
}

/// The dotted path a validation message names first: `roles.parent`,
/// `topics.ownership.subtopics` (from `topics[0]subtopics` with the topic
/// names `topics`), `targets.cloud`.
fn error_path(problem: &str, topics: &[String]) -> Option<String> {
    let raw = match problem.strip_prefix('`') {
        Some(rest) => rest.split('`').next()?,
        None => problem.split_once(':')?.0,
    };
    if raw.is_empty() || raw.contains(' ') {
        return None;
    }
    let Some(rest) = raw.strip_prefix("topics[") else {
        return Some(raw.to_string());
    };
    let (index, field) = rest.split_once(']')?;
    let name = topics.get(index.parse::<usize>().ok()?)?;
    let field = field.trim_start_matches('.');
    Some(if field.is_empty() {
        format!("topics.{name}")
    } else {
        format!("topics.{name}.{field}")
    })
}

/// The key of the field of `listing` that `path` names, or the first field
/// of the table it names, or of the nearest table above it.
fn error_key(mut path: &str, listing: &Listing) -> Option<String> {
    loop {
        if let Some(field) = listing.find(path).and_then(|index| listing.field(index)) {
            return Some(field.key.clone());
        }
        path = path.rsplit_once('.')?.0;
    }
}

/// What Enter on the field `field` of a Runpod target picks from the catalog;
/// its other fields keep the form.
fn picked_kind(field: &str) -> Option<CatalogKind> {
    match field {
        "gpu_types" => Some(CatalogKind::Gpus),
        "data_center_ids" => Some(CatalogKind::DataCenters),
        "network_volume_id" => Some(CatalogKind::Volumes),
        "image" => Some(CatalogKind::Templates),
        _ => None,
    }
}

/// The value the field `key` of `listing` shows, from the file, the
/// environment or its default; none when unset.
fn shown_value(listing: &Listing, key: &str) -> Option<String> {
    let field = listing.field(listing.find(key)?)?;
    match &field.shown {
        Shown::Value(text) | Shown::Default(text) => Some(text.clone()),
        Shown::Unset | Shown::Set | Shown::VaultRef => None,
    }
}

/// The fields of the Runpod target `name` that the hints and the pickers
/// need, as `listing` shows them.
fn sizing(listing: &Listing, name: &str) -> Sizing {
    let value = |field: &str| shown_value(listing, &format!("targets.{name}.{field}"));
    Sizing {
        gpu_types: value("gpu_types").unwrap_or_default(),
        gpu_count: value("gpu_count")
            .and_then(|count| count.parse().ok())
            .unwrap_or(1),
        max_hours: value("max_hours").and_then(|hours| hours.parse().ok()),
        max_price: value("max_price_per_hour").and_then(|price| price.parse().ok()),
    }
}

/// What a volume listing's reconciliation says: nothing when nothing
/// changed, a warning listing `failed` first when some change could not be
/// made, otherwise `done`.
fn reconciled(failed: &[String], done: &[String]) -> Option<(Severity, String)> {
    let done = match done.len() {
        0 => None,
        1 => Some(format!("{}, {VOLUME_CENTER}", done.join(""))),
        _ => Some(format!(
            "{}, the network volumes' data centers",
            done.join(", ")
        )),
    };
    let severity = if failed.is_empty() {
        Severity::Info
    } else {
        Severity::Warn
    };
    let parts: Vec<String> = failed.iter().cloned().chain(done).collect();
    (!parts.is_empty()).then(|| (severity, parts.join("; ")))
}

/// What a reconciled `data_center_ids` is.
const VOLUME_CENTER: &str = "the network volume's data center";

/// The first of `problems`, with how many more there are.
pub(super) fn first_of(problems: &[String]) -> String {
    let first = problems.first().cloned().unwrap_or_default();
    match problems.len() {
        0 | 1 => first,
        count => format!("{first} (+{} more)", count - 1),
    }
}

/// Sets the field `path` of `doc` to `value`, or unsets it for `None`. GPU
/// types listed for a target also unset its `min_vram_gb` and
/// `max_price_per_hour`, which go with `auto` only: returns what that removed,
/// in words.
///
/// # Errors
///
/// Returns why the document cannot be edited so.
fn set_field(
    doc: &mut ConfigDoc,
    path: &FieldPath,
    value: Option<&FieldValue>,
) -> Result<Option<String>, String> {
    match value {
        Some(value) => doc.set(path, value.clone()),
        None => doc.unset(path).map(drop),
    }
    .map_err(|error| error.to_string())?;
    let (FieldPath::Target { name, field }, Some(FieldValue::List(_))) = (path, value) else {
        return Ok(None);
    };
    if *field != GPU_TYPES {
        return Ok(None);
    }
    let mut removed = Vec::new();
    for field in AUTO_LIMITS {
        let limit = FieldPath::Target {
            name: name.clone(),
            field,
        };
        if doc.get(&limit).is_some() {
            doc.unset(&limit).map_err(|error| error.to_string())?;
            removed.push(limit.to_string());
        }
    }
    Ok((!removed.is_empty()).then(|| {
        format!(
            "{} removed: only with gpu_types = \"auto\"",
            removed.join(" and ")
        )
    }))
}

impl App {
    /// Shows `config`, read from `overbrainer.toml`: the rows are built again.
    pub(super) fn set_config(&mut self, config: ProjectConfig) {
        self.config = Some(config);
        self.project_view.touch();
    }

    /// The rows of the Project view, built again only when the configuration,
    /// the errors or the locks changed.
    pub(super) fn project_listing(&mut self) -> Arc<Listing> {
        let locks = Locks::of(self);
        self.project_view.listing(self.config.as_ref(), locks)
    }

    /// A key in the Project view with no form open.
    pub(super) fn on_project_key(&mut self, code: KeyCode) -> Vec<Effect> {
        match code {
            KeyCode::Enter => return self.edit_field(),
            KeyCode::Char('a') => self.start_adding(),
            KeyCode::Char('d') => self.ask_removal(),
            KeyCode::Char('u') => return self.undo(),
            KeyCode::Char('E') => return self.open_config(),
            code => self.move_field(code),
        }
        Vec::new()
    }

    /// Moves in the fields.
    fn move_field(&mut self, code: KeyCode) {
        let page = isize::try_from(PAGE).unwrap_or(isize::MAX);
        let by = match code {
            KeyCode::Up | KeyCode::Char('k') => -1,
            KeyCode::Down | KeyCode::Char('j') => 1,
            KeyCode::PageUp => -page,
            KeyCode::PageDown => page,
            KeyCode::Home => isize::MIN,
            KeyCode::End => isize::MAX,
            _ => return,
        };
        let count = self.project_listing().fields.len();
        self.project_view.step(by, count);
    }

    /// Refuses a change while a save runs, or with no configuration read.
    pub(super) fn refuse_change(&mut self) -> bool {
        if self.config.is_none() {
            return true;
        }
        if self.project_view.save.is_some() {
            self.say(
                Severity::Warn,
                format!("refused: {CONFIG_FILE} is being saved"),
            );
            return true;
        }
        false
    }

    /// The document of the file read, to edit, then hand to [`Self::write`].
    fn draft(&self) -> Result<ConfigDoc, String> {
        self.config
            .as_ref()
            .map(|config| config.doc.clone())
            .ok_or_else(|| "no configuration read".to_string())
    }

    /// Validates `doc`, an edited [`Self::draft`], then writes it to
    /// `overbrainer.toml` off the UI thread; `writing` says what the write
    /// leaves once it succeeds, and gets the text it replaces for `u`. Nothing
    /// happens when `doc` leaves the file as it is.
    ///
    /// # Errors
    ///
    /// Returns the first validation problem, or why no write starts (quitting,
    /// or another write running); nothing is written then.
    fn write(&mut self, doc: &ConfigDoc, mut writing: Writing) -> Result<Vec<Effect>, String> {
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| "no configuration read".to_string())?;
        let text = doc.text();
        if text == config.doc.text() {
            return Ok(Vec::new());
        }
        if self.leaving.is_some() {
            return Err("quitting; no save starts".to_string());
        }
        ProjectConfig::new(&text, &self.env).map_err(|error| match error {
            ConfigError::Invalid(problems) => first_of(&problems),
            error => error.to_string(),
        })?;
        let base = config.text.clone();
        writing.before = Some(base.clone());
        self.spawn_write(text, base, writing)
    }

    /// Writes `text` to `overbrainer.toml` off the UI thread, unless the file
    /// no longer holds `base`. Every write of the TUI starts here.
    ///
    /// # Errors
    ///
    /// Refuses while another write runs: its end would be ignored, and this
    /// one would find the file it changed. Refuses a change of a field a
    /// stage or a training run uses, whichever key asked for it.
    pub(super) fn spawn_write(
        &mut self,
        text: String,
        base: String,
        writing: Writing,
    ) -> Result<Vec<Effect>, String> {
        if self.project_view.save.is_some() {
            return Err(format!("{CONFIG_FILE} is being saved; try again"));
        }
        if let Some((key, user)) = self.locked_change(&text) {
            return Err(format!("{key} is used by {user}; read-only until it ends"));
        }
        let id = self.task_id();
        self.project_view.save = Some(id);
        self.project_view.writing = writing;
        let task = Task::SaveConfig {
            text,
            base,
            env: self.env.clone(),
        };
        Ok(vec![Effect::Spawn(id, task)])
    }

    /// The first field something uses now that `text`, written over the
    /// file read, would change, and what uses it. A text that does not load
    /// is left to the save to refuse.
    fn locked_change(&mut self, text: &str) -> Option<(String, String)> {
        let then = ProjectConfig::new(text, &self.env).ok()?;
        let locks = Locks::of(self);
        let then = Listing::new(rows(&then, &locks), &BTreeMap::new());
        let now = self.project_listing();
        let fields = |listing: &Listing| -> BTreeMap<String, (Shown, Option<String>)> {
            (0..listing.fields.len())
                .filter_map(|index| listing.field(index))
                .map(|field| (field.key.clone(), (field.shown.clone(), field.lock.clone())))
                .collect()
        };
        let (now, then) = (fields(&now), fields(&then));
        now.keys().chain(then.keys()).find_map(|key| {
            let (was, is) = (now.get(key), then.get(key));
            let user = was
                .and_then(|(_, lock)| lock.clone())
                .or_else(|| is.and_then(|(_, lock)| lock.clone()))?;
            let same = matches!((was, is), (Some((a, _)), Some((b, _))) if a == b);
            (!same).then(|| (key.clone(), user))
        })
    }

    /// Sets the field `path` to `value`, or unsets it for `None`, and writes
    /// the file.
    ///
    /// # Errors
    ///
    /// Returns why nothing is written.
    fn edit(
        &mut self,
        path: &FieldPath,
        value: Option<&FieldValue>,
    ) -> Result<Vec<Effect>, String> {
        let mut doc = self.draft()?;
        let note = set_field(&mut doc, path, value)?;
        self.write(
            &doc,
            Writing {
                note,
                ..Writing::default()
            },
        )
    }

    /// The effects of an edit, or none with the status line saying why
    /// nothing was written.
    fn written(&mut self, edited: Result<Vec<Effect>, String>) -> Vec<Effect> {
        edited.unwrap_or_else(|error| {
            self.say(Severity::Warn, format!("not saved: {error}"));
            Vec::new()
        })
    }

    /// Enter: toggles a bool, cycles a choice (to unset after the last, when
    /// the field may be left out), opens the catalog picker on the GPU types,
    /// data centers, volume or image of a Runpod target, or opens the form on
    /// the value (reading the GPU types for the hints of a Runpod target's
    /// `gpu_count` and `max_hours`); refused on a locked, env-only or env-set
    /// field.
    fn edit_field(&mut self) -> Vec<Effect> {
        if self.refuse_change() {
            return Vec::new();
        }
        let listing = self.project_listing();
        let Some(field) = listing.field(self.project_view.selected) else {
            return Vec::new();
        };
        let key = &field.key;
        if self.refuse_lock(key, field.lock.as_deref()) {
            return Vec::new();
        }
        if let Some(note) = field.env_note() {
            self.say(Severity::Warn, note);
            return Vec::new();
        }
        let Some(path) = field.path.clone() else {
            return Vec::new();
        };
        let Some(spec) = self
            .config
            .as_ref()
            .and_then(|config| config.doc.spec(&path))
        else {
            let said = format!("{key}: not edited here; E opens {CONFIG_FILE}");
            self.say(Severity::Warn, said);
            return Vec::new();
        };
        let mut effects = Vec::new();
        if let FieldPath::Target {
            name,
            field: name_of,
        } = &path
            && self.target_kind(name) == Some(TargetKind::Runpod)
        {
            let sizing = sizing(&listing, name);
            if let Some(kind) = picked_kind(name_of) {
                let shown = shown_value(&listing, key).unwrap_or_default();
                return self.open_field_picker(path, kind, &shown, &sizing);
            }
            if matches!(*name_of, "gpu_count" | "max_hours") {
                effects = self.read_gpu_catalog(sizing.gpu_count);
            }
        }
        let now = field.shown.text();
        let value = match spec.kind {
            FieldKind::Bool => Some(FieldValue::Bool(now != "true")),
            FieldKind::Choice(choices) => {
                let next = choices
                    .iter()
                    .position(|choice| *choice == now)
                    .map_or(0, |at| at + 1);
                match choices.get(next) {
                    Some(choice) => Some(FieldValue::Text((*choice).to_string())),
                    None if spec.optional => None,
                    None => choices
                        .first()
                        .map(|choice| FieldValue::Text((*choice).to_string())),
                }
            },
            kind => {
                let text = match &field.shown {
                    Shown::Value(text) | Shown::Default(text) => text.clone(),
                    _ => String::new(),
                };
                self.project_view.form = Some(Form::Value {
                    path,
                    kind,
                    optional: spec.optional,
                    input: Input::new(text),
                    error: None,
                });
                return effects;
            },
        };
        let edited = self.edit(&path, value.as_ref());
        effects.extend(self.written(edited));
        effects
    }

    /// The kind of the target `name` in the file read.
    fn target_kind(&self, name: &str) -> Option<TargetKind> {
        self.config.as_ref()?.doc.target_kind(name)
    }

    /// Opens the picker of `kind` on the Runpod target field `path`, which
    /// shows `shown`: what it shows is chosen, its target's GPU count and GPU
    /// types scope the stock.
    fn open_field_picker(
        &mut self,
        path: FieldPath,
        kind: CatalogKind,
        shown: &str,
        sizing: &Sizing,
    ) -> Vec<Effect> {
        let preselected = match kind {
            CatalogKind::Gpus | CatalogKind::DataCenters => {
                Choice::from(&ListOrAuto::from_form_text(shown))
            },
            // Unset: the `none` or `default` entry is the one chosen.
            CatalogKind::Volumes | CatalogKind::Templates => Choice::List(vec![shown.to_string()]),
        };
        let fit_by = match (&path, kind) {
            (FieldPath::Target { name, .. }, CatalogKind::Gpus) => FitBy::Target(name.clone()),
            _ => FitBy::Nothing,
        };
        let query = Query {
            kind,
            gpu_count: sizing.gpu_count,
            gpu_types: ListOrAuto::from_form_text(&sizing.gpu_types)
                .list()
                .to_vec(),
            fit_by,
        };
        self.open_picker(query, preselected, Origin::Field(path))
    }

    /// A picker opened on the Runpod target field `path` kept `picked`: it is
    /// written to that field. `auto` is written as such, an empty data center
    /// list unsets it (any), a volume sets the target's data centers to its
    /// own, `none` unsets it and leaves them.
    pub(super) fn picked_field(&mut self, path: &FieldPath, picked: Picked) -> Vec<Effect> {
        if self.refuse_change() || self.refuse_locked(path) {
            return Vec::new();
        }
        if picked.kind == CatalogKind::DataCenters {
            let value = match &picked.choice {
                Choice::Auto => Some(FieldValue::Text(ListOrAuto::AUTO.to_string())),
                Choice::List(ids) if ids.is_empty() => None,
                Choice::List(ids) => Some(FieldValue::List(ids.clone())),
            };
            if let Some(refusal) = self.volume_centers_refusal(path, value.as_ref()) {
                self.say(Severity::Warn, refusal);
                return Vec::new();
            }
        }
        let ids = match picked.choice {
            Choice::Auto => {
                let auto = FieldValue::Text(ListOrAuto::AUTO.to_string());
                let edited = self.edit(path, Some(&auto));
                return self.written(edited);
            },
            Choice::List(ids) => ids,
        };
        let edited = match (picked.kind, ids.first()) {
            (CatalogKind::Gpus, None) => {
                self.say(
                    Severity::Warn,
                    format!("{path}: choose a GPU type or auto; nothing changed"),
                );
                return Vec::new();
            },
            (CatalogKind::DataCenters, None) => self.edit(path, None),
            (CatalogKind::Gpus | CatalogKind::DataCenters, Some(_)) => {
                self.edit(path, Some(&FieldValue::List(ids)))
            },
            (CatalogKind::Volumes | CatalogKind::Templates, None) => Ok(Vec::new()),
            (CatalogKind::Volumes, Some(id)) if id == NO_VOLUME => self.edit(path, None),
            (CatalogKind::Volumes, Some(id)) => self.picked_volume(path, id, &picked.entries),
            (CatalogKind::Templates, Some(image)) if image == DEFAULT_IMAGE => {
                self.edit(path, None)
            },
            (CatalogKind::Templates, Some(image)) => {
                self.edit(path, Some(&FieldValue::Text(image.clone())))
            },
        };
        self.written(edited)
    }

    /// Why `value` cannot be set on the field `path`, when it is the
    /// `data_center_ids` of a Runpod target whose network volume the volume
    /// listing read last has: its one data center is the volume's, so it only
    /// changes with the volume. A volume not listed, or no listing, holds
    /// nothing.
    fn volume_centers_refusal(
        &mut self,
        path: &FieldPath,
        value: Option<&FieldValue>,
    ) -> Option<String> {
        let FieldPath::Target { name, field } = path else {
            return None;
        };
        if *field != "data_center_ids" {
            return None;
        }
        let listing = self.project_listing();
        let volume = shown_value(&listing, &format!("targets.{name}.network_volume_id"))
            .filter(|volume| !volume.is_empty())?;
        let center = self.volume_center(&volume)?;
        let new = match value {
            Some(FieldValue::List(ids)) => Some(ListOrAuto::List(ids.clone())),
            Some(FieldValue::Text(text)) => Some(ListOrAuto::from_form_text(text)),
            _ => None,
        };
        (new != Some(ListOrAuto::List(vec![center]))).then(|| {
            format!(
                "refused: {path} is the network volume's data center; pick another \
                 network_volume_id to change it"
            )
        })
    }

    /// The data center of the network volume `volume`, when the volume
    /// listing read last has it.
    fn volume_center(&self, volume: &str) -> Option<String> {
        self.volume_catalog
            .as_ref()?
            .iter()
            .find(|entry| entry.id == volume)
            .and_then(volume_data_center)
            .map(str::to_string)
    }

    /// A volume listing was read: each Runpod target whose network volume it
    /// lists in another data center than `data_center_ids` gets that data
    /// center, all in one write. One status line, once written, says what
    /// changed, a warning listing first what could not (a field the
    /// environment sets or a task locks, or a refused edit). Nothing while
    /// `overbrainer.toml` is open in the editor or being saved, or while
    /// quitting: it runs once the file is read again.
    pub(super) fn reconcile_volume_centers(&mut self) -> Vec<Effect> {
        let (effects, said) = self.volume_centers_reconciled();
        if let Some((severity, text)) = said {
            self.say(severity, text);
        }
        effects
    }

    /// [`Self::reconcile_volume_centers`], what it would say now returned.
    fn volume_centers_reconciled(&mut self) -> (Vec<Effect>, Option<(Severity, String)>) {
        if self.leaving.is_some() || self.project_view.save.is_some() || self.project_view.editing {
            return (Vec::new(), None);
        }
        let Ok(mut doc) = self.draft() else {
            return (Vec::new(), None);
        };
        let names: Vec<String> = doc
            .names(Collection::Targets)
            .into_iter()
            .filter(|name| doc.target_kind(name) == Some(TargetKind::Runpod))
            .collect();
        let listing = self.project_listing();
        let mut changes = Vec::new();
        let mut failed = Vec::new();
        for name in names {
            let Some(center) = shown_value(&listing, &format!("targets.{name}.network_volume_id"))
                .filter(|volume| !volume.is_empty())
                .and_then(|volume| self.volume_center(&volume))
            else {
                continue;
            };
            let key = format!("targets.{name}.data_center_ids");
            let now = shown_value(&listing, &key).map(|text| ListOrAuto::from_form_text(&text));
            if now == Some(ListOrAuto::List(vec![center.clone()])) {
                continue;
            }
            let field = listing.find(&key).and_then(|index| listing.field(index));
            let held = match field {
                Some(field) if field.env => Some("set by the environment".to_string()),
                Some(field) => field.lock.as_ref().map(|user| format!("used by {user}")),
                None => Some("not found".to_string()),
            };
            match held {
                Some(why) => {
                    failed.push(format!("{key} not set to {center}, {VOLUME_CENTER}: {why}"));
                },
                None => changes.push((name, center)),
            }
        }
        let mut done = Vec::new();
        for (name, center) in changes {
            let path = FieldPath::Target {
                name,
                field: "data_center_ids",
            };
            match set_field(
                &mut doc,
                &path,
                Some(&FieldValue::List(vec![center.clone()])),
            ) {
                Ok(_) => done.push(format!("{path} = {center}")),
                Err(error) => failed.push(format!(
                    "{path} not set to {center}, {VOLUME_CENTER}: {error}"
                )),
            }
        }
        let said = reconciled(&failed, &done);
        if done.is_empty() {
            return (Vec::new(), said);
        }
        // Said once written, the warnings first: a later line would hide them.
        let writing = Writing {
            warn: !failed.is_empty(),
            note: said.as_ref().map(|(_, text)| text.clone()),
            ..Writing::default()
        };
        match self.write(&doc, writing) {
            Ok(effects) if !effects.is_empty() => (effects, None),
            Ok(_) => (Vec::new(), said),
            Err(error) => {
                failed.push(format!("{} not saved: {error}", done.join(", ")));
                (Vec::new(), reconciled(&failed, &[]))
            },
        }
    }

    /// Refuses a change of the field `path` while something uses it, as the
    /// form does.
    fn refuse_locked(&mut self, path: &FieldPath) -> bool {
        let key = path.to_string();
        let listing = self.project_listing();
        let lock = listing
            .find(&key)
            .and_then(|index| listing.field(index))
            .and_then(|field| field.lock.clone());
        self.refuse_lock(&key, lock.as_deref())
    }

    /// Refuses a change of the field `key` when `lock` names what uses it.
    fn refuse_lock(&mut self, key: &str, lock: Option<&str>) -> bool {
        let Some(user) = lock else {
            return false;
        };
        let said = format!("refused: {key} is used by {user}; read-only until it ends");
        self.say(Severity::Warn, said);
        true
    }

    /// `t` in a picker opened on the field `path`: the form opens on its value
    /// instead.
    pub(super) fn type_field(&mut self, path: &FieldPath) {
        if self.refuse_change() || self.refuse_locked(path) {
            return;
        }
        let Some(spec) = self
            .config
            .as_ref()
            .and_then(|config| config.doc.spec(path))
        else {
            return;
        };
        let listing = self.project_listing();
        let text = shown_value(&listing, &path.to_string()).unwrap_or_default();
        self.project_view.form = Some(Form::Value {
            path: path.clone(),
            kind: spec.kind,
            optional: spec.optional,
            input: Input::new(text),
            error: None,
        });
    }

    /// The volume `volume` typed for the field `path`: as if picked when the
    /// volume listing read last has it, so its data center is set too;
    /// otherwise set alone, `data_center_ids` left as is.
    fn typed_volume(&mut self, path: &FieldPath, volume: &str) -> Result<Vec<Effect>, String> {
        let known = self.volume_catalog.as_ref().and_then(|entries| {
            entries
                .iter()
                .find(|entry| entry.id == volume && volume_data_center(entry).is_some())
                .cloned()
        });
        if let Some(entry) = known {
            return self.picked_volume(path, volume, &[entry]);
        }
        self.edit(path, Some(&FieldValue::Text(volume.to_string())))
    }

    /// The volume `id` picked for the field `path`: it is set, and its
    /// target's data centers become the volume's, when `entries` says which;
    /// both in one write, or neither.
    fn picked_volume(
        &mut self,
        path: &FieldPath,
        id: &str,
        entries: &[super::widgets::picker::Entry],
    ) -> Result<Vec<Effect>, String> {
        let center = entries
            .iter()
            .find(|entry| entry.id == id)
            .and_then(volume_data_center)
            .map(str::to_string);
        let mut doc = self.draft()?;
        set_field(&mut doc, path, Some(&FieldValue::Text(id.to_string())))?;
        let note = match (path, center) {
            (FieldPath::Target { name, .. }, Some(center)) => {
                let centers = FieldPath::Target {
                    name: name.clone(),
                    field: "data_center_ids",
                };
                set_field(
                    &mut doc,
                    &centers,
                    Some(&FieldValue::List(vec![center.clone()])),
                )?;
                Some(format!("{centers} = {center}, the volume's data center"))
            },
            _ => None,
        };
        self.write(
            &doc,
            Writing {
                note,
                ..Writing::default()
            },
        )
    }

    /// What the selected field's hint adds, from the GPU types read last: the
    /// most GPUs per pod for a Runpod target's `gpu_count`, the most the run
    /// can cost for its `max_hours`.
    pub(super) fn field_hint(&mut self) -> Option<String> {
        self.gpu_catalog.as_ref()?;
        let listing = self.project_listing();
        let field = listing.field(self.project_view.selected)?;
        let Some(FieldPath::Target { name, field: key }) = &field.path else {
            return None;
        };
        if self.target_kind(name) != Some(TargetKind::Runpod) {
            return None;
        }
        let gpus = self.gpu_catalog.as_deref()?;
        let sizing = sizing(&listing, name);
        match *key {
            "gpu_count" => gpu_count_hint(gpus, &sizing),
            "max_hours" => cost_hint(gpus, &sizing),
            _ => None,
        }
    }

    /// A key while the form is open: Enter on the last step writes the file,
    /// or keeps the form open with why it does not.
    pub(super) fn on_form_key(&mut self, code: KeyCode) -> Vec<Effect> {
        let Some(form) = self.project_view.form.take() else {
            return Vec::new();
        };
        let choices = Addable::ALL.len();
        let cycle = |at: usize, count: usize| match code {
            KeyCode::Left | KeyCode::Up | KeyCode::Char('h' | 'k') | KeyCode::BackTab => {
                Some((at + count - 1) % count)
            },
            KeyCode::Right | KeyCode::Down | KeyCode::Char('l' | 'j') | KeyCode::Tab => {
                Some((at + 1) % count)
            },
            _ => None,
        };
        let mut effects = Vec::new();
        self.project_view.form = match form {
            Form::Value {
                path,
                kind,
                optional,
                mut input,
                error,
            } => match input.on_key(code) {
                InputOutcome::Editing => Some(Form::Value {
                    path,
                    kind,
                    optional,
                    input,
                    error,
                }),
                InputOutcome::Cancelled => None,
                InputOutcome::Done(text) => match self.typed(&path, kind, optional, &text) {
                    Ok(written) => {
                        effects = written;
                        None
                    },
                    Err(error) => Some(Form::Value {
                        path,
                        kind,
                        optional,
                        input,
                        error: Some(error),
                    }),
                },
            },
            Form::Adding(at) => match code {
                KeyCode::Enter => Addable::ALL.get(at).map(|what| Form::Name {
                    what: *what,
                    input: Input::new(""),
                    error: None,
                }),
                KeyCode::Esc => None,
                _ => Some(Form::Adding(cycle(at, choices).unwrap_or(at))),
            },
            Form::Name {
                what,
                mut input,
                error,
            } => match input.on_key(code) {
                InputOutcome::Editing => Some(Form::Name { what, input, error }),
                InputOutcome::Cancelled => None,
                InputOutcome::Done(name) => match self.named(what, name.trim()) {
                    Ok((next, written)) => {
                        effects = written;
                        next
                    },
                    Err(error) => Some(Form::Name {
                        what,
                        input,
                        error: Some(error),
                    }),
                },
            },
            Form::Kind {
                what,
                name,
                choice,
                error,
            } => match code {
                KeyCode::Enter => match self.add(what, &name, choice) {
                    Ok(written) => {
                        effects = written;
                        None
                    },
                    Err(error) => Some(Form::Kind {
                        what,
                        name,
                        choice,
                        error: Some(error),
                    }),
                },
                KeyCode::Esc => None,
                _ => Some(Form::Kind {
                    what,
                    choice: cycle(choice, what.kinds().len().max(1)).unwrap_or(choice),
                    name,
                    error,
                }),
            },
        };
        effects
    }

    /// Enter in the value form: checks `text` against `kind` and its bounds,
    /// then writes it; an empty text unsets an `optional` field.
    ///
    /// # Errors
    ///
    /// Returns why nothing is written, shown in the form.
    fn typed(
        &mut self,
        path: &FieldPath,
        kind: FieldKind,
        optional: bool,
        text: &str,
    ) -> Result<Vec<Effect>, String> {
        let value = match (text.trim().is_empty(), optional) {
            (true, true) => None,
            (true, false) => return Err("is required".to_string()),
            (false, _) => Some(kind.parse(text).map_err(|error| error.to_string())?),
        };
        if let Some(refusal) = self.volume_centers_refusal(path, value.as_ref()) {
            return Err(refusal);
        }
        if let (FieldPath::Target { field, .. }, Some(FieldValue::Text(volume))) = (path, &value)
            && *field == "network_volume_id"
        {
            return self.typed_volume(path, volume);
        }
        if let (FieldPath::Topic { index, field, .. }, Some(FieldValue::Text(new))) = (path, &value)
            && *field == "name"
        {
            let names = self
                .config
                .as_ref()
                .map(|config| config.doc.topic_names())
                .unwrap_or_default();
            let clash = names
                .iter()
                .enumerate()
                .any(|(at, topic)| at != *index && topic == new);
            if clash {
                return Err(format!("topics.{new} already exists"));
            }
        }
        self.edit(path, value.as_ref())
    }

    /// A key while a bracketed paste arrives: it goes to the form's input.
    pub(super) fn on_project_paste(&mut self, text: &str) {
        if let Some(Form::Value { input, .. } | Form::Name { input, .. }) =
            &mut self.project_view.form
        {
            input.paste(text);
            self.dirty = true;
        }
    }

    /// `a`: asks what to add, the kind of the selected table first.
    fn start_adding(&mut self) {
        if self.refuse_change() {
            return;
        }
        let listing = self.project_listing();
        let key = listing
            .field(self.project_view.selected)
            .map(|field| field.key.clone())
            .unwrap_or_default();
        let at = match key.split('.').next() {
            Some("providers") => 1,
            Some("targets") => 2,
            _ => 0,
        };
        self.project_view.form = Some(Form::Adding(at));
    }

    /// Enter on the name of what is added, which must match `^[a-z0-9_]+$`
    /// and be new: a topic is added and written at once, a provider or a
    /// target asks its protocol or kind next.
    ///
    /// # Errors
    ///
    /// Returns why the name is refused, or why nothing is written.
    fn named(&mut self, what: Addable, name: &str) -> Result<(Option<Form>, Vec<Effect>), String> {
        let Some(config) = &self.config else {
            return Ok((None, Vec::new()));
        };
        let doc = &config.doc;
        if !is_valid_name(name) {
            return Err("the name must match ^[a-z0-9_]+$".to_string());
        }
        let exists = match collection(what) {
            None => doc.topic_names().iter().any(|topic| topic == name),
            Some(collection) => doc.names(collection).iter().any(|table| table == name),
        };
        if exists {
            return Err(format!("{} already exists", table_key(what, name)));
        }
        if what == Addable::Topic {
            return Ok((None, self.add(what, name, 0)?));
        }
        let form = Form::Kind {
            what,
            name: name.to_string(),
            choice: 0,
            error: None,
        };
        Ok((Some(form), Vec::new()))
    }

    /// Adds `what` named `name`, with the `choice`-th of its kinds, and writes
    /// the file; its first field is selected once written.
    ///
    /// # Errors
    ///
    /// Returns why nothing is written.
    fn add(&mut self, what: Addable, name: &str, choice: usize) -> Result<Vec<Effect>, String> {
        let kind = what.kinds().get(choice).copied().unwrap_or_default();
        let key = table_key(what, name);
        let mut doc = self.draft()?;
        let added = match what {
            Addable::Topic => doc.add_topic(name).map(drop),
            Addable::Provider => {
                let protocol = if kind == "anthropic" {
                    Protocol::Anthropic
                } else {
                    Protocol::Openai
                };
                doc.add_provider(name, protocol)
            },
            Addable::Target => doc.add_target(
                name,
                TargetKind::from_name(kind).unwrap_or(TargetKind::Local),
            ),
        };
        added.map_err(|error| error.to_string())?;
        let writing = Writing {
            note: Some(format!("{key} added")),
            select: Some(key),
            ..Writing::default()
        };
        self.write(&doc, writing)
    }

    /// `d`: asks to delete the topic, provider or target of the selected
    /// field; refused when it is not one, or while something uses it.
    fn ask_removal(&mut self) {
        if self.refuse_change() {
            return;
        }
        let listing = self.project_listing();
        let Some(field) = listing.field(self.project_view.selected) else {
            return;
        };
        let removal = match &field.path {
            Some(FieldPath::Topic { index, name, .. }) => {
                Some(Removal::Topic(*index, name.clone()))
            },
            Some(FieldPath::Provider { name, .. }) => {
                Some(Removal::Table(Collection::Providers, name.clone()))
            },
            Some(FieldPath::Target { name, .. }) => {
                Some(Removal::Table(Collection::Targets, name.clone()))
            },
            Some(_) => None,
            // An env-only field of a provider or a target: its names are plain.
            None => {
                let mut parts = field.key.split('.');
                match (parts.next(), parts.next()) {
                    (Some("providers"), Some(name)) => {
                        Some(Removal::Table(Collection::Providers, name.to_string()))
                    },
                    (Some("targets"), Some(name)) => {
                        Some(Removal::Table(Collection::Targets, name.to_string()))
                    },
                    _ => None,
                }
            },
        };
        let Some(removal) = removal else {
            self.say(Severity::Warn, "d deletes a topic, a provider or a target");
            return;
        };
        let key = removal.key();
        let user = self
            .config
            .as_ref()
            .and_then(|config| Locks::of(self).user_of(&config.settings, &key));
        if let Some(user) = user {
            self.say(
                Severity::Warn,
                format!("refused: {key} is used by {user}; delete it once it ends"),
            );
            return;
        }
        self.overlay = Some(Overlay::Confirm(Confirm {
            title: format!(" Delete {key}? "),
            text: vec![format!(
                "{key} is deleted from {CONFIG_FILE} at once; u undoes it."
            )],
            yes: "delete",
            no: "keep",
            action: Action::Remove(removal),
        }));
    }

    /// A confirmed `d`: takes `removal` out of `overbrainer.toml`.
    pub(super) fn remove(&mut self, removal: &Removal) -> Vec<Effect> {
        if self.refuse_change() {
            return Vec::new();
        }
        let key = removal.key();
        let Ok(mut doc) = self.draft() else {
            return Vec::new();
        };
        let removed = match removal {
            Removal::Topic(index, name) => {
                doc.topic_names().get(*index) == Some(name) && doc.remove_topic(*index)
            },
            Removal::Table(collection, name) => doc.remove_table(*collection, name),
        };
        if !removed {
            self.say(
                Severity::Warn,
                format!("{key} is not in {CONFIG_FILE}: the environment sets it"),
            );
            return Vec::new();
        }
        let writing = Writing {
            note: Some(format!("{key} deleted")),
            ..Writing::default()
        };
        let edited = self.write(&doc, writing);
        self.written(edited)
    }

    /// The save ended: the app reads the configuration written, or each
    /// problem is shown on the field it names and nothing changed. A save of
    /// the choices made at start then starts the run, or says why it does not.
    pub(super) fn config_saved(
        &mut self,
        saved: Result<Box<ProjectConfig>, SaveRefusal>,
    ) -> Vec<Effect> {
        self.project_view.save = None;
        let writing = std::mem::take(&mut self.project_view.writing);
        let start = self.start_after_save.take();
        let mut effects = Vec::new();
        match saved {
            Ok(config) => {
                let after = config.text.clone();
                let used = self.adopt(*config);
                self.project_view.undo = writing.before.map(|before| Undo { before, after });
                let listing = self.project_listing();
                match writing.select.and_then(|key| listing.find(&key)) {
                    Some(index) => self.project_view.selected = index,
                    None => self.project_view.step(0, listing.fields.len()),
                }
                if self.leaving.is_some() {
                    self.exit_notes.push(format!("{CONFIG_FILE} was saved"));
                }
                let said = match writing.note {
                    Some(note) => format!("✓ saved {CONFIG_FILE}; {note}"),
                    None => format!("✓ saved {CONFIG_FILE}"),
                };
                let severity = if writing.warn {
                    Severity::Warn
                } else {
                    Severity::Info
                };
                self.say(severity, said);
                effects = vec![used];
                effects.extend(self.reload());
                match start {
                    Some(_) if self.leaving.is_some() => {
                        self.exit_notes.push(NOT_STARTED.to_string());
                    },
                    Some(plan) => effects.extend(self.started_after_save(&plan)),
                    None => {},
                }
            },
            Err(refusal) if start.is_some() => {
                // The file keeps its values: the Project view marks nothing.
                let why = match refusal {
                    SaveRefusal::Invalid(problems) => first_of(&problems),
                    SaveRefusal::Failed(error) => error,
                };
                if self.leaving.is_some() {
                    self.exit_notes.push(format!(
                        "a new training run was not started: {CONFIG_FILE} not saved: {why}"
                    ));
                }
                self.say(
                    Severity::Error,
                    format!("run not started: {CONFIG_FILE} not saved: {why}"),
                );
            },
            Err(SaveRefusal::Invalid(problems)) => {
                self.mark_errors(&problems);
                self.say(
                    Severity::Error,
                    format!("not saved: {}", first_of(&problems)),
                );
            },
            Err(SaveRefusal::Failed(error)) => {
                self.say(Severity::Error, format!("not saved: {error}"));
            },
        }
        self.leave_when_idle();
        if self.exit.is_some() {
            return Vec::new();
        }
        effects
    }

    /// Makes `config`, just read from `overbrainer.toml`, the configuration:
    /// the project follows it, the errors shown go, and the look at the files
    /// takes its stamp, so it reads them again only once they change; the
    /// returned effect has the next tasks read their settings from it, with
    /// the environment of the app. A save, a reload and `E` all end here.
    pub(super) fn adopt(&mut self, config: ProjectConfig) -> Effect {
        let dir = self.project.dir.clone();
        self.project = Project::new(&dir, &config.settings);
        self.project_view.errors.clear();
        if let (Some(watch), Some(stamp)) = (self.watch.as_mut(), config.stamp) {
            watch.seen(stamp);
        }
        let used = self.use_config(&config);
        self.set_config(config);
        used
    }

    /// Shows each of `problems` on the field it names, and selects the first.
    fn mark_errors(&mut self, problems: &[String]) {
        self.note_errors(problems);
        let listing = self.project_listing();
        if let Some(index) = (0..listing.fields.len()).find(|index| {
            listing
                .field(*index)
                .is_some_and(|field| field.error.is_some())
        }) {
            self.project_view.selected = index;
        }
    }

    /// Shows each of `problems` on the field it names.
    pub(super) fn note_errors(&mut self, problems: &[String]) {
        let topics = self
            .config
            .as_ref()
            .map(|config| config.doc.topic_names())
            .unwrap_or_default();
        self.project_view.errors.clear();
        self.project_view.touch();
        let listing = self.project_listing();
        let mut errors: BTreeMap<String, String> = BTreeMap::new();
        for problem in problems {
            let Some(key) =
                error_path(problem, &topics).and_then(|path| error_key(&path, &listing))
            else {
                continue;
            };
            errors
                .entry(key)
                .and_modify(|said| {
                    said.push_str("; ");
                    said.push_str(problem);
                })
                .or_insert_with(|| problem.clone());
        }
        self.project_view.errors = errors;
        self.project_view.touch();
    }

    /// `u`: writes back the text before the last write of the TUI, unless the
    /// file changed on disk since, or a stage or a training run uses a field
    /// it would change (the undo then stays for later); there is no undo of
    /// the undo.
    fn undo(&mut self) -> Vec<Effect> {
        if self.refuse_change() {
            return Vec::new();
        }
        // Taken once its write starts: a refused undo stays for later.
        let Some(Undo { before, after }) = self.project_view.undo.clone() else {
            self.say(Severity::Info, "nothing to undo");
            return Vec::new();
        };
        if self.leaving.is_some() {
            self.say(Severity::Warn, "not undone: quitting; no save starts");
            return Vec::new();
        }
        if let Err(error) = ProjectConfig::new(&before, &self.env) {
            let why = match error {
                ConfigError::Invalid(problems) => first_of(&problems),
                error => error.to_string(),
            };
            self.say(Severity::Warn, format!("not undone: {why}"));
            return Vec::new();
        }
        let writing = Writing {
            note: Some("the last write undone".to_string()),
            ..Writing::default()
        };
        match self.spawn_write(before, after, writing) {
            Ok(effects) => {
                self.project_view.undo = None;
                effects
            },
            Err(error) => {
                self.say(Severity::Warn, format!("not undone: {error}"));
                Vec::new()
            },
        }
    }

    /// `E`: opens `overbrainer.toml` in the editor, after which `u` has
    /// nothing to undo; refused while a save runs, or while a stage, an edit
    /// or a training uses the configuration.
    fn open_config(&mut self) -> Vec<Effect> {
        if self.refuse_change() {
            return Vec::new();
        }
        if self.refuse_new("E edits the whole file", "edit") {
            return Vec::new();
        }
        if let Some((run, _)) = Locks::of(self).runs.first().cloned() {
            self.say(
                Severity::Warn,
                format!("refused: {run} uses the training table; E edits the whole file"),
            );
            return Vec::new();
        }
        self.project_view.editing = true;
        self.project_view.undo = None;
        vec![Effect::OpenEditor {
            command: self.editor.clone(),
            path: self.project.dir.join(CONFIG_FILE),
        }]
    }

    /// The editor on `overbrainer.toml` ended with `status`: the file is read
    /// again, and the volume listing read last reconciled with it; one that
    /// does not load leaves the view as it was.
    pub(super) fn config_edited(&mut self, status: io::Result<ExitStatus>) -> Vec<Effect> {
        let failed = editor_failure(status);
        let path = self.project.dir.join(CONFIG_FILE);
        let read_at = stamp(&self.project.dir);
        let read = std::fs::read_to_string(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))
            .and_then(|text| {
                ProjectConfig::new(&text, &self.env).map_err(|error| {
                    if let ConfigError::Invalid(problems) = &error {
                        self.note_errors(problems);
                    }
                    let text = error.to_string();
                    text.lines().map(str::trim).collect::<Vec<_>>().join(" ")
                })
            });
        // The file just read is not reported again by the look at the files.
        if let Some(watch) = self.watch.as_mut() {
            watch.seen(read_at);
        }
        let mut config = match read {
            Ok(config) => config,
            Err(error) => {
                let said = format!("{error}; the view shows {CONFIG_FILE} as it was");
                self.say(Severity::Error, said);
                return Vec::new();
            },
        };
        let unchanged = self
            .config
            .as_ref()
            .is_some_and(|shown| shown.text == config.text);
        config.stamp = Some(read_at);
        let used = self.adopt(config);
        let what = if unchanged {
            format!("{CONFIG_FILE} unchanged")
        } else {
            format!("{CONFIG_FILE} read again")
        };
        let (mut severity, mut said) = match failed {
            Some(failed) => (Severity::Warn, format!("{failed}; {what}")),
            None => (Severity::Info, what),
        };
        // A volume listing read while the file was edited applies to it now.
        let (written, reconciled) = self.volume_centers_reconciled();
        if let Some((reconciled, text)) = reconciled {
            if reconciled != Severity::Info {
                severity = reconciled;
            }
            said = format!("{said}; {text}");
        }
        self.say(severity, said);
        let mut effects = vec![used];
        effects.extend(self.reload());
        effects.extend(written);
        effects
    }
}

/// How the editor failed, when it did: killed, a non-zero status, or not run.
pub(super) fn editor_failure(status: io::Result<ExitStatus>) -> Option<String> {
    match status {
        Ok(status) if status.success() => None,
        Ok(status) => Some(status.code().map_or_else(
            || "the editor was killed".to_string(),
            |code| format!("the editor exited with status {code}"),
        )),
        Err(error) => Some(format!("cannot run the editor: {error}")),
    }
}

/// The collection a provider or a target is added to; `None` for a topic.
fn collection(what: Addable) -> Option<Collection> {
    match what {
        Addable::Topic => None,
        Addable::Provider => Some(Collection::Providers),
        Addable::Target => Some(Collection::Targets),
    }
}

/// The dotted key of the table `what` named `name`: `topics.traits`.
fn table_key(what: Addable, name: &str) -> String {
    let parent = collection(what).map_or("topics", Collection::key);
    format!("{parent}.{name}")
}

#[cfg(test)]
mod tests {
    use std::os::unix::process::ExitStatusExt;

    use super::*;
    use crate::cli::data::Command;
    use crate::runpod::{GpuType, NetworkVolume, Template};
    use crate::tui::catalog::{Listed, template_entries, volume_entries};
    use crate::tui::snapshots::{
        PROJECT_CONFIG, draw, key, project_app, project_env, text as screen,
    };
    use crate::tui::snapshots::{gpu_catalog, gpu_types};
    use crate::tui::tasks::{Done, TaskId};
    use crate::tui::widgets::picker::Entry;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// [`project_app`] on a project directory holding [`PROJECT_CONFIG`].
    fn editing_app() -> Result<(tempfile::TempDir, App), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join(CONFIG_FILE), PROJECT_CONFIG)?;
        let mut app = project_app()?;
        app.project.dir = dir.path().to_path_buf();
        app.env = project_env();
        Ok((dir, app))
    }

    /// Runs each write `effects` spawn in the project directory, as its task
    /// would, and hands its end back to `app`; returns `effects` with what
    /// those ends returned.
    fn run_saves(app: &mut App, effects: Vec<Effect>) -> Vec<Effect> {
        let mut all = Vec::new();
        let mut queue = effects;
        while !queue.is_empty() {
            let mut next = Vec::new();
            for effect in queue {
                if let Effect::Spawn(id, Task::SaveConfig { text, base, env }) = &effect {
                    assert_eq!(app.project_view.save, Some(*id));
                    assert_eq!(env, &app.env, "the app's environment");
                    let saved = save_config(&app.project.dir, text, base, env).map(Box::new);
                    next.extend(app.on_done(*id, Ok(Done::ConfigSaved(saved))));
                }
                all.push(effect);
            }
            queue = next;
        }
        all
    }

    /// Each key of `codes`, the write it starts run to its end.
    fn press(app: &mut App, codes: &[KeyCode]) -> Vec<Effect> {
        let mut effects = Vec::new();
        for code in codes {
            let pressed = app.on_input(&key(*code));
            effects.extend(run_saves(app, pressed));
        }
        effects
    }

    fn chars(app: &mut App, text: &str) {
        for c in text.chars() {
            app.on_input(&key(KeyCode::Char(c)));
        }
    }

    /// Selects the field `key`.
    fn select(app: &mut App, key: &str) -> Result<(), String> {
        let listing = app.project_listing();
        let index = (0..listing.fields.len())
            .find(|index| listing.field(*index).is_some_and(|field| field.key == key))
            .ok_or_else(|| format!("no field {key}"))?;
        app.project_view.selected = index;
        Ok(())
    }

    /// Enter on the field `key`, its text replaced by `value`, not Enter yet.
    fn type_value(app: &mut App, key: &str, value: &str) -> Result<(), String> {
        select(app, key)?;
        press(app, &[KeyCode::Enter, KeyCode::End]);
        press(app, &[KeyCode::Backspace; 80]);
        chars(app, value);
        Ok(())
    }

    /// [`type_value`], then Enter, its write run: what Enter returned.
    fn set(app: &mut App, key: &str, value: &str) -> Result<Vec<Effect>, String> {
        type_value(app, key, value)?;
        Ok(press(app, &[KeyCode::Enter]))
    }

    /// The field `path` set to `value`, as Enter sets it, and written.
    fn edit_now(app: &mut App, path: &FieldPath, value: Option<&FieldValue>) -> TestResult {
        let effects = app.edit(path, value)?;
        run_saves(app, effects);
        Ok(())
    }

    fn status(app: &App) -> &str {
        app.status
            .as_ref()
            .map_or("", |status| status.text.as_str())
    }

    fn field(app: &mut App, key: &str) -> Result<crate::tui::project::Field, String> {
        let listing = app.project_listing();
        (0..listing.fields.len())
            .find_map(|index| listing.field(index).filter(|field| field.key == key))
            .cloned()
            .ok_or_else(|| format!("no field {key}"))
    }

    fn written(dir: &Path) -> std::io::Result<String> {
        std::fs::read_to_string(dir.join(CONFIG_FILE))
    }

    /// The text before the last write, which `u` writes back.
    fn before_last_write(app: &App) -> Result<&str, String> {
        app.project_view
            .undo
            .as_ref()
            .map(|undo| undo.before.as_str())
            .ok_or_else(|| "nothing to undo".to_string())
    }

    #[test]
    fn a_number_out_of_bounds_is_refused_on_enter() -> TestResult {
        let (dir, mut app) = editing_app()?;
        set(&mut app, "topics.ownership.subtopics", "0")?;
        let Some(Form::Value { error, .. }) = &app.project_view.form else {
            return Err("the form closed".into());
        };
        assert_eq!(error.as_deref(), Some("must be at least 1"));
        press(&mut app, &[KeyCode::Backspace]);
        chars(&mut app, "x");
        press(&mut app, &[KeyCode::Enter]);
        let Some(Form::Value { error, .. }) = &app.project_view.form else {
            return Err("the form closed".into());
        };
        assert_eq!(error.as_deref(), Some("must be a whole number"));
        press(&mut app, &[KeyCode::Esc]);
        assert_eq!(app.project_view.form, None);
        assert_eq!(written(dir.path())?, PROJECT_CONFIG, "nothing written");
        assert_eq!(app.project_view.undo, None);
        Ok(())
    }

    #[test]
    fn enter_writes_the_value_at_once_with_the_comments_kept() -> TestResult {
        let (dir, mut app) = editing_app()?;
        let effects = set(&mut app, "project.name", "rust_pro")?;
        assert_eq!(app.project_view.form, None);
        let text = written(dir.path())?;
        assert!(text.contains("name = \"rust_pro\""), "{text}");
        assert!(text.starts_with("# the project\n"), "{text}");
        assert!(
            text.contains("concurrency = 16 # overridden by env"),
            "{text}"
        );
        assert_eq!(status(&app), "✓ saved overbrainer.toml");
        assert_eq!(app.project.name, "rust_pro", "the settings are read again");
        assert_eq!(field(&mut app, "project.name")?.shown.text(), "rust_pro");
        assert_eq!(app.project_view.save, None);
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, Effect::Spawn(_, Task::Load))),
            "the data is read again with the new topics"
        );
        assert_eq!(before_last_write(&app)?, PROJECT_CONFIG);
        Ok(())
    }

    #[test]
    fn a_value_that_does_not_validate_keeps_the_form_open_and_writes_nothing() -> TestResult {
        let (dir, mut app) = editing_app()?;
        set(&mut app, "roles.parent.provider", "nope")?;
        let Some(Form::Value { error, .. }) = &app.project_view.form else {
            return Err("the form closed".into());
        };
        assert_eq!(
            error.as_deref(),
            Some("roles.parent: unknown provider `nope`")
        );
        assert_eq!(app.project_view.save, None, "no write started");
        assert_eq!(written(dir.path())?, PROJECT_CONFIG);
        let rows = screen(&draw(&mut app, 80, 24)?).join("\n");
        assert!(rows.contains("unknown provider `nope`"), "{rows}");
        press(&mut app, &[KeyCode::Backspace; 4]);
        chars(&mut app, "nanogpt");
        press(&mut app, &[KeyCode::Enter]);
        assert_eq!(app.project_view.form, None);
        assert!(written(dir.path())?.contains("parent = { provider = \"nanogpt\""));
        Ok(())
    }

    #[test]
    fn a_choice_that_does_not_validate_writes_nothing_and_says_why() -> TestResult {
        let (dir, mut app) = editing_app()?;
        select(&mut app, "roles.generator.reasoning_effort")?;
        let effects = press(&mut app, &[KeyCode::Enter]);
        assert!(effects.is_empty(), "{effects:?}");
        assert_eq!(written(dir.path())?, PROJECT_CONFIG);
        assert_eq!(
            status(&app),
            "not saved: roles.generator.reasoning_effort: requires reasoning = true"
        );
        assert_eq!(
            field(&mut app, "roles.generator.reasoning_effort")?.shown,
            Shown::Unset
        );
        Ok(())
    }

    #[test]
    fn a_adds_a_topic_written_at_once() -> TestResult {
        let (dir, mut app) = editing_app()?;
        press(&mut app, &[KeyCode::Char('a')]);
        assert_eq!(app.project_view.form, Some(Form::Adding(0)));
        press(&mut app, &[KeyCode::Enter]);
        chars(&mut app, "rust.traits");
        press(&mut app, &[KeyCode::Enter]);
        let Some(Form::Name { error, .. }) = &app.project_view.form else {
            return Err("the form closed".into());
        };
        assert_eq!(error.as_deref(), Some("the name must match ^[a-z0-9_]+$"));
        press(&mut app, &[KeyCode::Backspace; 11]);
        chars(&mut app, "ownership");
        press(&mut app, &[KeyCode::Enter]);
        let Some(Form::Name { error, .. }) = &app.project_view.form else {
            return Err("the form closed".into());
        };
        assert_eq!(error.as_deref(), Some("topics.ownership already exists"));
        press(&mut app, &[KeyCode::Backspace; 9]);
        chars(&mut app, "traits");
        press(&mut app, &[KeyCode::Enter]);
        assert_eq!(app.project_view.form, None);
        let text = written(dir.path())?;
        assert!(
            text.contains("[[topics]]\nname = \"traits\"\nsubtopics = 10"),
            "{text}"
        );
        assert_eq!(app.project.topics.len(), 2);
        let selected = app
            .project_listing()
            .field(app.project_view.selected)
            .map(|field| field.key.clone());
        assert_eq!(selected.as_deref(), Some("topics.traits.name"));
        assert_eq!(
            status(&app),
            "✓ saved overbrainer.toml; topics.traits added"
        );
        Ok(())
    }

    /// [`editing_app`] with a second topic, `traits`, written.
    fn two_topics() -> Result<(tempfile::TempDir, App), Box<dyn std::error::Error>> {
        let (dir, mut app) = editing_app()?;
        press(&mut app, &[KeyCode::Char('a'), KeyCode::Enter]);
        chars(&mut app, "traits");
        press(&mut app, &[KeyCode::Enter]);
        Ok((dir, app))
    }

    fn topic_names(app: &App) -> Vec<String> {
        app.config
            .as_ref()
            .map(|config| config.doc.topic_names())
            .unwrap_or_default()
    }

    #[test]
    fn a_rename_to_the_name_of_another_topic_is_refused_on_enter() -> TestResult {
        let (_dir, mut app) = two_topics()?;
        set(&mut app, "topics.traits.name", "ownership")?;
        let Some(Form::Value { error, .. }) = &app.project_view.form else {
            return Err("the form closed".into());
        };
        assert_eq!(error.as_deref(), Some("topics.ownership already exists"));
        assert_eq!(topic_names(&app), ["ownership", "traits"]);
        press(&mut app, &[KeyCode::Esc]);
        let effects = set(&mut app, "topics.traits.name", "traits")?;
        assert_eq!(app.project_view.form, None, "its own name is no clash");
        assert!(effects.is_empty(), "the same value writes nothing");
        Ok(())
    }

    #[test]
    fn d_deletes_the_topic_selected_once_confirmed() -> TestResult {
        let (dir, mut app) = two_topics()?;
        select(&mut app, "topics.traits.subtopics")?;
        press(&mut app, &[KeyCode::Char('d')]);
        let Some(Overlay::Confirm(confirm)) = &app.overlay else {
            return Err("no dialog".into());
        };
        assert_eq!(
            confirm.text,
            ["topics.traits is deleted from overbrainer.toml at once; u undoes it."]
        );
        press(&mut app, &[KeyCode::Char('y')]);
        assert_eq!(topic_names(&app), ["ownership"]);
        assert!(!written(dir.path())?.contains("traits"));
        assert_eq!(
            status(&app),
            "✓ saved overbrainer.toml; topics.traits deleted"
        );
        let selected = app.project_view.selected;
        assert!(app.project_listing().field(selected).is_some(), "in range");
        Ok(())
    }

    #[test]
    fn a_provider_and_targets_are_added_with_their_kind() -> TestResult {
        let (dir, mut app) = editing_app()?;
        select(&mut app, "providers.claude.protocol")?;
        press(&mut app, &[KeyCode::Char('a')]);
        assert_eq!(app.project_view.form, Some(Form::Adding(1)), "a provider");
        press(&mut app, &[KeyCode::Enter]);
        chars(&mut app, "Local AI");
        press(&mut app, &[KeyCode::Enter]);
        let Some(Form::Name { error, .. }) = &app.project_view.form else {
            return Err("the form closed".into());
        };
        assert_eq!(error.as_deref(), Some("the name must match ^[a-z0-9_]+$"));
        press(&mut app, &[KeyCode::Backspace; 8]);
        chars(&mut app, "local");
        press(&mut app, &[KeyCode::Enter, KeyCode::Right, KeyCode::Enter]);
        assert_eq!(
            field(&mut app, "providers.local.protocol")?.shown.text(),
            "anthropic"
        );
        press(
            &mut app,
            &[KeyCode::Char('a'), KeyCode::Right, KeyCode::Enter],
        );
        chars(&mut app, "box");
        press(
            &mut app,
            &[KeyCode::Enter, KeyCode::Char('j'), KeyCode::Enter],
        );
        assert_eq!(
            field(&mut app, "targets.box.runtime")?.shown.text(),
            "docker"
        );
        assert!(field(&mut app, "targets.box.host").is_ok(), "an ssh target");
        // A Runpod target validates as added: its GPU types are `auto`.
        press(&mut app, &[KeyCode::Char('a')]);
        assert_eq!(app.project_view.form, Some(Form::Adding(2)), "a target");
        press(&mut app, &[KeyCode::Enter]);
        chars(&mut app, "pod");
        press(&mut app, &[KeyCode::Enter, KeyCode::Left, KeyCode::Enter]);
        assert_eq!(
            field(&mut app, "targets.pod.gpu_types")?.shown.text(),
            "auto"
        );
        let text = written(dir.path())?;
        assert!(
            text.contains("[providers.local]\nprotocol = \"anthropic\"\n"),
            "{text}"
        );
        assert!(
            text.contains("[targets.box]\nkind = \"ssh\"\nruntime = \"docker\"\n"),
            "{text}"
        );
        assert!(
            text.contains("[targets.pod]\nkind = \"runpod\"\ngpu_types = \"auto\"\n"),
            "{text}"
        );
        Ok(())
    }

    #[test]
    fn deleting_a_provider_a_role_uses_is_refused_naming_the_role() -> TestResult {
        let (dir, mut app) = editing_app()?;
        select(&mut app, "providers.claude.api_key")?;
        press(&mut app, &[KeyCode::Char('d')]);
        let Some(Overlay::Confirm(confirm)) = &app.overlay else {
            return Err("no dialog".into());
        };
        assert_eq!(confirm.title, " Delete providers.claude? ");
        let effects = press(&mut app, &[KeyCode::Char('y')]);
        assert!(effects.is_empty(), "{effects:?}");
        assert_eq!(written(dir.path())?, PROJECT_CONFIG);
        assert_eq!(
            status(&app),
            "not saved: roles.parent: unknown provider `claude`"
        );
        assert!(field(&mut app, "providers.claude.protocol").is_ok(), "kept");
        Ok(())
    }

    #[test]
    fn a_locked_field_refuses_enter_and_a_locked_table_refuses_d() -> TestResult {
        let (dir, mut app) = editing_app()?;
        set(&mut app, "roles.parent.model", "claude-opus-6")?;
        app.pipeline_task = Some(TaskId(7));
        app.pipeline.started(Command::Answers, 8);
        let before = written(dir.path())?;
        select(&mut app, "roles.parent.model")?;
        press(&mut app, &[KeyCode::Enter]);
        assert_eq!(app.project_view.form, None);
        assert_eq!(
            status(&app),
            "refused: roles.parent.model is used by answers; read-only until it ends"
        );
        select(&mut app, "providers.claude.protocol")?;
        press(&mut app, &[KeyCode::Char('d')]);
        assert_eq!(app.overlay, None);
        assert!(status(&app).starts_with("refused: providers.claude is used by answers"));
        assert_eq!(written(dir.path())?, before);
        Ok(())
    }

    #[test]
    fn env_fields_refuse_enter_and_say_to_use_dot_env() -> TestResult {
        let (dir, mut app) = editing_app()?;
        select(&mut app, "providers.nanogpt.api_key")?;
        press(&mut app, &[KeyCode::Enter]);
        assert_eq!(
            status(&app),
            "providers.nanogpt.api_key: env only, set \
             OVERBRAINER_PROVIDERS__NANOGPT__API_KEY in .env"
        );
        select(&mut app, "pipeline.concurrency")?;
        press(&mut app, &[KeyCode::Enter]);
        assert_eq!(
            status(&app),
            "pipeline.concurrency: set by OVERBRAINER_PIPELINE__CONCURRENCY, \
             change it in .env"
        );
        assert_eq!(app.project_view.form, None);
        assert_eq!(written(dir.path())?, PROJECT_CONFIG);
        Ok(())
    }

    #[test]
    fn a_bool_toggles_a_choice_cycles_and_a_paste_goes_to_the_form() -> TestResult {
        let (dir, mut app) = editing_app()?;
        select(&mut app, "pipeline.include_system_prompt")?;
        let before = field(&mut app, "pipeline.include_system_prompt")?;
        press(&mut app, &[KeyCode::Enter]);
        let after = field(&mut app, "pipeline.include_system_prompt")?;
        assert_ne!(before.shown.text(), after.shown.text());
        assert!(
            written(dir.path())?
                .contains(&format!("include_system_prompt = {}", after.shown.text()))
        );
        select(&mut app, "training.adapter")?;
        press(&mut app, &[KeyCode::Enter]);
        assert_eq!(field(&mut app, "training.adapter")?.shown.text(), "qlora");
        select(&mut app, "topics.ownership.description")?;
        press(&mut app, &[KeyCode::Enter, KeyCode::End]);
        press(&mut app, &[KeyCode::Backspace; 60]);
        app.on_input(&crossterm::event::Event::Paste(
            "Owners\nand borrows\n".into(),
        ));
        press(&mut app, &[KeyCode::Enter]);
        assert_eq!(
            field(&mut app, "topics.ownership.description")?
                .shown
                .text(),
            "Owners and borrows"
        );
        set(&mut app, "topics.ownership.description", "")?;
        assert_eq!(
            field(&mut app, "topics.ownership.description")?.shown,
            Shown::Unset,
            "an empty value unsets an optional field"
        );
        assert!(!written(dir.path())?.contains("description"));
        Ok(())
    }

    #[test]
    fn u_undoes_the_last_write_only_once() -> TestResult {
        let (dir, mut app) = editing_app()?;
        press(&mut app, &[KeyCode::Char('u')]);
        assert_eq!(status(&app), "nothing to undo");
        set(&mut app, "project.name", "rust_pro")?;
        let renamed = written(dir.path())?;
        set(&mut app, "topics.ownership.subtopics", "5")?;
        assert!(written(dir.path())?.contains("subtopics = 5"));
        press(&mut app, &[KeyCode::Char('u')]);
        assert_eq!(written(dir.path())?, renamed, "the last write alone");
        assert_eq!(
            status(&app),
            "✓ saved overbrainer.toml; the last write undone"
        );
        assert_eq!(
            field(&mut app, "topics.ownership.subtopics")?.shown.text(),
            "2"
        );
        assert_eq!(app.project.name, "rust_pro");
        press(&mut app, &[KeyCode::Char('u')]);
        assert_eq!(status(&app), "nothing to undo", "no undo of the undo");
        assert_eq!(written(dir.path())?, renamed);
        Ok(())
    }

    #[test]
    fn u_is_refused_while_a_run_or_a_stage_uses_what_it_changes() -> TestResult {
        let (dir, mut app) = editing_app()?;
        set(&mut app, "targets.gpu_cloud.gpu_count", "2")?;
        let counted = written(dir.path())?;
        let mut follow =
            crate::tui::training::Follow::new(crate::tui::training::Job::Attach, "20260921-a1");
        follow.watching = true;
        app.training.tasks.insert(TaskId(9), follow);
        assert_eq!(press(&mut app, &[KeyCode::Char('u')]), []);
        assert_eq!(
            status(&app),
            "not undone: targets.gpu_cloud.gpu_count is used by run 20260921-a1; read-only \
             until it ends"
        );
        assert_eq!(written(dir.path())?, counted);
        assert!(app.project_view.undo.is_some(), "kept for later");
        app.training.tasks.clear();
        press(&mut app, &[KeyCode::Char('u')]);
        assert_eq!(written(dir.path())?, PROJECT_CONFIG, "undone once it ended");
        // A stage uses the parent's role.
        set(&mut app, "roles.parent.model", "claude-opus-6")?;
        app.pipeline_task = Some(TaskId(7));
        app.pipeline.started(Command::Answers, 8);
        assert_eq!(press(&mut app, &[KeyCode::Char('u')]), []);
        assert_eq!(
            status(&app),
            "not undone: roles.parent.model is used by answers; read-only until it ends"
        );
        assert!(written(dir.path())?.contains("claude-opus-6"));
        // A field no one uses is undone meanwhile.
        set(&mut app, "project.name", "rust_pro")?;
        press(&mut app, &[KeyCode::Char('u')]);
        assert!(!written(dir.path())?.contains("rust_pro"));
        Ok(())
    }

    #[test]
    fn u_is_refused_when_the_file_changed_on_disk() -> TestResult {
        let (dir, mut app) = editing_app()?;
        set(&mut app, "project.name", "rust_pro")?;
        std::fs::write(dir.path().join(CONFIG_FILE), "# changed elsewhere\n")?;
        press(&mut app, &[KeyCode::Char('u')]);
        assert_eq!(written(dir.path())?, "# changed elsewhere\n");
        assert_eq!(
            status(&app),
            "not saved: overbrainer.toml changed on disk since it was read; nothing written"
        );
        assert_eq!(app.project_view.undo, None);
        Ok(())
    }

    #[test]
    fn e_leaves_nothing_to_undo() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        set(&mut app, "project.name", "rust_pro")?;
        assert!(app.project_view.undo.is_some());
        press(&mut app, &[KeyCode::Char('E')]);
        assert_eq!(app.project_view.undo, None);
        app.on_editor_exit(Ok(ExitStatus::from_raw(0)));
        press(&mut app, &[KeyCode::Char('u')]);
        assert_eq!(status(&app), "nothing to undo");
        Ok(())
    }

    #[test]
    fn nothing_changes_while_a_write_runs() -> TestResult {
        let (dir, mut app) = editing_app()?;
        select(&mut app, "pipeline.include_system_prompt")?;
        let effects = app.on_input(&key(KeyCode::Enter));
        let [Effect::Spawn(id, Task::SaveConfig { .. })] = effects.as_slice() else {
            return Err(format!("no write: {effects:?}").into());
        };
        assert_eq!(app.project_view.save, Some(*id));
        for code in ['u', 'a', 'd', 'E'] {
            assert!(app.on_input(&key(KeyCode::Char(code))).is_empty(), "{code}");
            assert_eq!(app.overlay, None, "{code}");
            assert_eq!(app.project_view.form, None, "{code}");
            assert_eq!(status(&app), "refused: overbrainer.toml is being saved");
        }
        assert_eq!(
            app.on_input(&key(KeyCode::Enter)),
            [] as [crate::tui::app::Effect; 0]
        );
        assert_eq!(app.project_view.form, None);
        assert_eq!(written(dir.path())?, PROJECT_CONFIG, "the task writes");
        Ok(())
    }

    #[test]
    fn a_rename_is_one_write_and_the_same_value_writes_nothing() -> TestResult {
        let (dir, mut app) = editing_app()?;
        set(&mut app, "topics.ownership.name", "owning")?;
        assert_eq!(before_last_write(&app)?, PROJECT_CONFIG);
        assert_eq!(topic_names(&app), ["owning"]);
        assert!(written(dir.path())?.contains("name = \"owning\""));
        let effects = set(&mut app, "topics.owning.subtopics", "2")?;
        assert!(effects.is_empty(), "{effects:?}");
        assert_eq!(app.project_view.form, None);
        assert_eq!(before_last_write(&app)?, PROJECT_CONFIG, "no new write");
        Ok(())
    }

    #[test]
    fn quitting_after_a_write_asks_nothing_about_it() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        set(&mut app, "project.name", "rust_pro")?;
        press(&mut app, &[KeyCode::Char('q')]);
        if let Some(Overlay::Confirm(confirm)) = &app.overlay {
            assert!(
                !confirm
                    .text
                    .iter()
                    .any(|line| line.contains("overbrainer.toml")),
                "{confirm:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn e_opens_the_file_in_the_editor_and_the_view_reads_it_again() -> TestResult {
        let (dir, mut app) = editing_app()?;
        let effects = press(&mut app, &[KeyCode::Char('E')]);
        let [Effect::OpenEditor { path, .. }] = effects.as_slice() else {
            return Err(format!("{effects:?}").into());
        };
        assert_eq!(path, &dir.path().join(CONFIG_FILE));
        std::fs::write(
            path,
            PROJECT_CONFIG.replace("name = \"rust_expert\"", "name = \"renamed\""),
        )?;
        let effects = app.on_editor_exit(Ok(ExitStatus::from_raw(0)));
        assert!(!app.project_view.editing);
        assert_eq!(status(&app), "overbrainer.toml read again");
        assert_eq!(field(&mut app, "project.name")?.shown.text(), "renamed");
        assert_eq!(app.project.name, "renamed");
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, Effect::Spawn(_, Task::Load)))
        );
        std::fs::write(path, "[project\n")?;
        press(&mut app, &[KeyCode::Char('E')]);
        app.on_editor_exit(Ok(ExitStatus::from_raw(0)));
        assert!(
            status(&app).ends_with("; the view shows overbrainer.toml as it was"),
            "{}",
            status(&app)
        );
        assert_eq!(field(&mut app, "project.name")?.shown.text(), "renamed");
        Ok(())
    }

    #[test]
    fn a_volume_listing_read_while_the_file_is_edited_reconciles_after() -> TestResult {
        let (dir, mut app) = editing_app()?;
        let with_volume = |volume: &str, center: &str| {
            PROJECT_CONFIG.replace(
                "max_hours = 6\n",
                &format!(
                    "max_hours = 6\nnetwork_volume_id = \"{volume}\"\n\
                     data_center_ids = [\"{center}\"]\n"
                ),
            )
        };
        let before = with_volume("voleu", "US-KS-2");
        std::fs::write(dir.path().join(CONFIG_FILE), &before)?;
        app.set_config(crate::tui::project::ProjectConfig::new(&before, &app.env)?);
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.network_volume_id")?;
        press(&mut app, &[KeyCode::Esc]);
        press(&mut app, &[KeyCode::Char('E')]);
        assert!(app.project_view.editing);
        let volumes = volume_entries(&[
            NetworkVolume {
                id: "voleu".into(),
                name: "alpha".into(),
                size: 100,
                data_center: "EU-RO-1".into(),
            },
            NetworkVolume {
                id: "volus".into(),
                name: "zeta".into(),
                size: 50,
                data_center: "US-KS-2".into(),
            },
        ]);
        listed(&mut app, id, volumes, Vec::new());
        assert_eq!(written(dir.path())?, before, "nothing while edited");
        let edited = with_volume("volus", "EU-RO-1");
        std::fs::write(dir.path().join(CONFIG_FILE), &edited)?;
        let effects = app.on_editor_exit(Ok(ExitStatus::from_raw(0)));
        assert_eq!(status(&app), "overbrainer.toml read again");
        run_saves(&mut app, effects);
        let key = "targets.gpu_cloud.data_center_ids";
        assert_eq!(
            shown(&mut app, key)?,
            "US-KS-2",
            "reconciled on the new file"
        );
        assert_eq!(written(dir.path())?, with_volume("volus", "US-KS-2"));
        assert_eq!(before_last_write(&app)?, edited, "u undoes it");
        assert_eq!(
            status(&app),
            "✓ saved overbrainer.toml; targets.gpu_cloud.data_center_ids = US-KS-2, the \
             network volume's data center"
        );
        Ok(())
    }

    #[test]
    fn a_form_kept_open_across_a_background_write_waits_for_it() -> TestResult {
        let (dir, mut app) = editing_app()?;
        let before = PROJECT_CONFIG.replace(
            "max_hours = 6\n",
            "max_hours = 6\nnetwork_volume_id = \"voleu\"\ndata_center_ids = [\"US-KS-2\"]\n",
        );
        std::fs::write(dir.path().join(CONFIG_FILE), &before)?;
        app.set_config(ProjectConfig::new(&before, &app.env)?);
        // The form opens on the volume before its listing ends.
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.network_volume_id")?;
        press(&mut app, &[KeyCode::Char('t'), KeyCode::End]);
        press(&mut app, &[KeyCode::Backspace; 20]);
        chars(&mut app, "volus");
        // The listing ends: its reconciliation starts a write.
        let reconciled = app.on_done(
            id,
            Ok(Done::Catalog(Ok(Listed {
                entries: volumes(),
                gpus: Vec::new(),
                note: None,
            }))),
        );
        let [Effect::Spawn(running, Task::SaveConfig { .. })] = reconciled.as_slice() else {
            return Err(format!("no reconciliation: {reconciled:?}").into());
        };
        let running = *running;
        // Enter meanwhile starts nothing and keeps the value typed.
        assert_eq!(app.on_input(&key(KeyCode::Enter)), []);
        let Some(Form::Value { error, input, .. }) = &app.project_view.form else {
            return Err("the form closed".into());
        };
        assert_eq!(
            error.as_deref(),
            Some("overbrainer.toml is being saved; try again")
        );
        assert_eq!(input.text(), "volus");
        assert_eq!(app.project_view.save, Some(running), "the first write kept");
        // A reconciliation meanwhile starts nothing either.
        assert_eq!(app.reconcile_volume_centers(), []);
        // The first write ends and is adopted; Enter then writes the value.
        run_saves(&mut app, reconciled);
        assert_eq!(
            shown(&mut app, "targets.gpu_cloud.data_center_ids")?,
            "EU-RO-1"
        );
        press(&mut app, &[KeyCode::Enter]);
        assert_eq!(app.project_view.form, None, "{}", status(&app));
        let text = written(dir.path())?;
        assert!(text.contains("network_volume_id = \"volus\""), "{text}");
        assert!(text.contains("data_center_ids = [\"US-KS-2\"]"), "{text}");
        Ok(())
    }

    #[test]
    fn no_write_starts_while_another_runs() -> TestResult {
        let (dir, mut app) = editing_app()?;
        select(&mut app, "pipeline.include_system_prompt")?;
        let effects = app.on_input(&key(KeyCode::Enter));
        assert_eq!(effects.len(), 1, "{effects:?}");
        let again = app.spawn_write(
            PROJECT_CONFIG.to_string(),
            PROJECT_CONFIG.to_string(),
            Writing::default(),
        );
        assert_eq!(
            again,
            Err("overbrainer.toml is being saved; try again".into())
        );
        assert_eq!(written(dir.path())?, PROJECT_CONFIG);
        Ok(())
    }

    #[test]
    fn e_is_refused_while_a_stage_runs() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        app.pipeline_task = Some(TaskId(7));
        app.pipeline.started(Command::Split, 8);
        assert_eq!(
            press(&mut app, &[KeyCode::Char('E')]),
            [] as [crate::tui::app::Effect; 0]
        );
        assert!(
            status(&app).ends_with("; E edits the whole file"),
            "{}",
            status(&app)
        );
        Ok(())
    }

    #[test]
    fn a_write_is_refused_when_the_file_changed_on_disk() -> TestResult {
        let (dir, mut app) = editing_app()?;
        type_value(&mut app, "project.name", "rust_pro")?;
        std::fs::write(dir.path().join(CONFIG_FILE), "# changed elsewhere\n")?;
        press(&mut app, &[KeyCode::Enter]);
        assert_eq!(written(dir.path())?, "# changed elsewhere\n");
        assert_eq!(
            status(&app),
            "not saved: overbrainer.toml changed on disk since it was read; nothing written"
        );
        assert_eq!(app.project_view.undo, None);
        Ok(())
    }

    #[test]
    fn the_rows_are_built_again_only_when_something_changed() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        let first = app.project_listing();
        assert!(Arc::ptr_eq(&first, &app.project_listing()));
        app.pipeline_task = Some(TaskId(7));
        app.pipeline.started(Command::Answers, 8);
        let locked = app.project_listing();
        assert!(!Arc::ptr_eq(&first, &locked), "the locks changed");
        set(&mut app, "project.name", "rust_pro")?;
        assert!(!Arc::ptr_eq(&locked, &app.project_listing()), "an edit");
        Ok(())
    }

    #[test]
    fn the_form_is_drawn() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        set(&mut app, "project.name", "rust_pro")?;
        select(&mut app, "topics.ownership.subtopics")?;
        press(&mut app, &[KeyCode::Enter]);
        chars(&mut app, "x");
        app.status = None;
        let rows = screen(&draw(&mut app, 80, 24)?).join("\n");
        assert!(rows.contains("╭ configuration ─"), "{rows}");
        assert!(rows.contains("rust_pro"), "{rows}");
        assert!(rows.contains("subtopics = 2x"), "{rows}");
        assert!(rows.contains("at least 1"), "{rows}");
        assert!(rows.contains("Enter save"), "{rows}");
        press(
            &mut app,
            &[KeyCode::Esc, KeyCode::Char('a'), KeyCode::Enter],
        );
        let rows = screen(&draw(&mut app, 80, 24)?).join("\n");
        assert!(rows.contains("a-z, 0-9 and _, not already used"), "{rows}");
        Ok(())
    }

    #[test]
    fn the_kinds_offered_are_those_of_the_schema() {
        let targets: Vec<&str> = TargetKind::ALL.iter().map(|kind| kind.as_str()).collect();
        assert_eq!(Addable::Target.kinds(), targets.as_slice());
        assert_eq!(Addable::Provider.kinds(), ["openai", "anthropic"]);
        assert_eq!(Addable::Topic.kinds(), [] as [&str; 0]);
    }

    #[test]
    fn a_write_reloads_the_settings_the_app_uses() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        assert_eq!(app.project.concurrency, 8, "the fixture's");
        set(&mut app, "pipeline.eval_ratio", "0.2")?;
        press(
            &mut app,
            &[KeyCode::Char('a'), KeyCode::Right, KeyCode::Right],
        );
        press(&mut app, &[KeyCode::Enter]);
        chars(&mut app, "box");
        press(&mut app, &[KeyCode::Enter, KeyCode::Enter]);
        set(&mut app, "training.target", "box")?;
        assert_eq!(status(&app), "✓ saved overbrainer.toml");
        assert!((app.project.eval_ratio - 0.2).abs() < f64::EPSILON);
        assert_eq!(app.project.concurrency, 4, "from the environment");
        assert_eq!(app.project.target.as_deref(), Some("box"));
        Ok(())
    }

    #[test]
    fn a_write_keeps_the_file_mode_and_refuses_a_symlink() -> TestResult {
        use std::os::unix::fs::PermissionsExt;
        let (dir, mut app) = editing_app()?;
        let path = dir.path().join(CONFIG_FILE);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640))?;
        set(&mut app, "project.name", "rust_pro")?;
        assert_eq!(status(&app), "✓ saved overbrainer.toml");
        let mode = |path: &Path| -> std::io::Result<u32> {
            Ok(std::fs::metadata(path)?.permissions().mode() & 0o7777)
        };
        assert_eq!(mode(&path)?, 0o640);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o4755))?;
        set(&mut app, "project.name", "rust_suid")?;
        assert_eq!(status(&app), "✓ saved overbrainer.toml");
        assert_eq!(mode(&path)?, 0o755, "never setuid, setgid or sticky");
        set(&mut app, "project.name", "rust_pro")?;
        let real = dir.path().join("real.toml");
        std::fs::rename(&path, &real)?;
        std::os::unix::fs::symlink(&real, &path)?;
        set(&mut app, "project.name", "rust_max")?;
        assert_eq!(
            status(&app),
            "not saved: overbrainer.toml is a symlink; nothing written, edit it with E"
        );
        assert!(std::fs::symlink_metadata(&path)?.file_type().is_symlink());
        assert!(std::fs::read_to_string(&real)?.contains("rust_pro"));
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_write_never_blocks_on_a_fifo() -> TestResult {
        let dir = tempfile::tempdir()?;
        rustix::fs::mkfifoat(
            rustix::fs::CWD,
            dir.path().join(CONFIG_FILE),
            rustix::fs::Mode::from_raw_mode(0o600),
        )?;
        let (sender, receiver) = std::sync::mpsc::channel();
        let path = dir.path().to_path_buf();
        std::thread::spawn(move || {
            let saved = save_config(&path, PROJECT_CONFIG, PROJECT_CONFIG, &project_env());
            sender.send(saved.map(drop)).ok();
        });
        let saved = receiver.recv_timeout(std::time::Duration::from_secs(5))?;
        let Err(SaveRefusal::Failed(error)) = saved else {
            return Err(format!("{saved:?}").into());
        };
        assert!(error.ends_with("not a regular file"), "{error}");
        let kind = std::fs::symlink_metadata(dir.path().join(CONFIG_FILE))?.file_type();
        assert!(std::os::unix::fs::FileTypeExt::is_fifo(&kind), "untouched");
        Ok(())
    }

    #[test]
    fn a_type_error_in_a_topic_is_shown_on_its_field() -> TestResult {
        let (dir, mut app) = editing_app()?;
        let text = PROJECT_CONFIG.replace("subtopics = 2", "subtopics = \"many\"");
        let refusal = save_config(dir.path(), &text, PROJECT_CONFIG, &project_env());
        let Err(SaveRefusal::Invalid(problems)) = refusal else {
            return Err(format!("{refusal:?}").into());
        };
        assert!(
            problems.iter().any(|problem| problem.contains("topics[0]")),
            "{problems:?}"
        );
        assert_eq!(written(dir.path())?, PROJECT_CONFIG);
        app.project_view.save = Some(TaskId(40));
        app.on_done(
            TaskId(40),
            Ok(Done::ConfigSaved(Err(SaveRefusal::Invalid(problems)))),
        );
        assert!(
            field(&mut app, "topics.ownership.subtopics")?
                .error
                .is_some()
        );
        Ok(())
    }

    #[test]
    fn quitting_or_a_signal_waits_for_a_write_then_notes_it() -> TestResult {
        for signal in [false, true] {
            let (dir, mut app) = editing_app()?;
            type_value(&mut app, "project.name", "rust_pro")?;
            let effects = app.on_input(&key(KeyCode::Enter));
            let [Effect::Spawn(id, Task::SaveConfig { text, base, env })] = effects.as_slice()
            else {
                return Err(format!("{effects:?}").into());
            };
            if signal {
                app.on_signal();
            } else {
                press(&mut app, &[KeyCode::Char('q')]);
                let Some(Overlay::Confirm(confirm)) = &app.overlay else {
                    return Err("no dialog".into());
                };
                assert!(
                    confirm.text.iter().any(|line| line.contains("being saved")),
                    "{confirm:?}"
                );
                press(&mut app, &[KeyCode::Char('y')]);
            }
            assert_eq!(app.exit, None, "waits for the write");
            let saved = save_config(dir.path(), text, base, env).map(Box::new);
            app.on_done(*id, Ok(Done::ConfigSaved(saved)));
            assert!(app.exit.is_some());
            assert!(written(dir.path())?.contains("rust_pro"));
            assert!(
                app.exit_notes
                    .contains(&"overbrainer.toml was saved".to_string()),
                "{:?}",
                app.exit_notes
            );
        }
        Ok(())
    }

    #[test]
    fn no_write_starts_while_quitting() -> TestResult {
        let (dir, mut app) = editing_app()?;
        app.leaving = Some(crate::tui::app::Exit::Quit);
        set(&mut app, "project.name", "rust_pro")?;
        let Some(Form::Value { error, .. }) = &app.project_view.form else {
            return Err("the form closed".into());
        };
        assert_eq!(error.as_deref(), Some("quitting; no save starts"));
        assert_eq!(written(dir.path())?, PROJECT_CONFIG);
        Ok(())
    }

    #[test]
    fn e_is_refused_while_a_training_run_is_followed() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        let mut follow =
            crate::tui::training::Follow::new(crate::tui::training::Job::Attach, "20260921-a1");
        follow.watching = true;
        app.training.tasks.insert(TaskId(9), follow);
        assert_eq!(
            press(&mut app, &[KeyCode::Char('E')]),
            [] as [crate::tui::app::Effect; 0]
        );
        assert_eq!(
            status(&app),
            "refused: run 20260921-a1 uses the training table; E edits the whole file"
        );
        Ok(())
    }

    /// Enter on the field `key`: the catalog listing it spawns, and the query.
    fn open_picker_on(app: &mut App, key: &str) -> Result<(TaskId, Query), String> {
        select(app, key)?;
        let effects = press(app, &[KeyCode::Enter]);
        match effects.as_slice() {
            [Effect::Spawn(id, Task::Catalog(query))] => Ok((*id, query.clone())),
            _ => Err(format!("no listing: {effects:?} ({})", status(app))),
        }
    }

    /// The picker open lists `entries` (and `gpus`), read by `id`; the write
    /// it starts is run.
    fn listed(app: &mut App, id: TaskId, entries: Vec<Entry>, gpus: Vec<GpuType>) {
        let effects = app.on_done(
            id,
            Ok(Done::Catalog(Ok(Listed {
                entries,
                gpus,
                note: None,
            }))),
        );
        run_saves(app, effects);
    }

    fn shown(app: &mut App, key: &str) -> Result<String, String> {
        Ok(field(app, key)?.shown.text().to_string())
    }

    fn picker_open(app: &App) -> bool {
        matches!(app.overlay, Some(Overlay::Picker(_)))
    }

    #[test]
    fn picking_gpus_writes_an_ordered_list() -> TestResult {
        let (dir, mut app) = editing_app()?;
        let (id, query) = open_picker_on(&mut app, "targets.gpu_cloud.gpu_types")?;
        assert_eq!(
            query,
            Query {
                kind: CatalogKind::Gpus,
                gpu_count: 1,
                gpu_types: vec!["NVIDIA A40".into()],
                fit_by: FitBy::Target("gpu_cloud".into()),
            }
        );
        assert!(picker_open(&app));
        listed(&mut app, id, gpu_catalog(1)?, gpu_types()?);
        // The cursor is on A40, chosen; the RTX 2000 under it is chosen, then
        // moved before A40.
        press(
            &mut app,
            &[
                KeyCode::Down,
                KeyCode::Char(' '),
                KeyCode::Char('K'),
                KeyCode::Enter,
            ],
        );
        assert!(!picker_open(&app));
        assert_eq!(
            shown(&mut app, "targets.gpu_cloud.gpu_types")?,
            "NVIDIA RTX 2000 Ada Generation, NVIDIA A40"
        );
        let text = written(dir.path())?;
        assert!(
            text.contains(r#"gpu_types = ["NVIDIA RTX 2000 Ada Generation", "NVIDIA A40"]"#),
            "{text}"
        );
        Ok(())
    }

    #[test]
    fn auto_is_offered_at_the_top_of_the_gpu_and_data_center_pickers() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.gpu_types")?;
        listed(&mut app, id, gpu_catalog(1)?, gpu_types()?);
        press(
            &mut app,
            &[KeyCode::Home, KeyCode::Char(' '), KeyCode::Enter],
        );
        assert_eq!(shown(&mut app, "targets.gpu_cloud.gpu_types")?, "auto");
        let (id, query) = open_picker_on(&mut app, "targets.gpu_cloud.data_center_ids")?;
        assert_eq!(query.kind, CatalogKind::DataCenters);
        assert!(query.gpu_types.is_empty(), "auto GPUs: any GPU's stock");
        let center = |id: &str| Entry {
            id: id.into(),
            columns: vec![id.into(), String::new(), String::new(), "1 GPU type".into()],
            selectable: true,
            ranks: Vec::new(),
        };
        listed(
            &mut app,
            id,
            vec![center("EU-RO-1"), center("US-KS-2")],
            Vec::new(),
        );
        press(
            &mut app,
            &[KeyCode::Home, KeyCode::Char(' '), KeyCode::Enter],
        );
        assert_eq!(
            shown(&mut app, "targets.gpu_cloud.data_center_ids")?,
            "auto"
        );
        // Opened again: auto is chosen; choosing a data center takes it out.
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.data_center_ids")?;
        listed(
            &mut app,
            id,
            vec![center("EU-RO-1"), center("US-KS-2")],
            Vec::new(),
        );
        press(
            &mut app,
            &[KeyCode::End, KeyCode::Char(' '), KeyCode::Enter],
        );
        assert_eq!(
            shown(&mut app, "targets.gpu_cloud.data_center_ids")?,
            "US-KS-2"
        );
        // Nothing chosen: any data center, the field unset.
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.data_center_ids")?;
        listed(
            &mut app,
            id,
            vec![center("EU-RO-1"), center("US-KS-2")],
            Vec::new(),
        );
        press(&mut app, &[KeyCode::Char(' '), KeyCode::Enter]);
        assert_eq!(
            field(&mut app, "targets.gpu_cloud.data_center_ids")?.shown,
            Shown::Unset
        );
        Ok(())
    }

    #[test]
    fn the_data_center_picker_asks_the_stock_of_the_chosen_gpus() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        set(&mut app, "targets.gpu_cloud.gpu_count", "2")?;
        let (_, query) = open_picker_on(&mut app, "targets.gpu_cloud.data_center_ids")?;
        assert_eq!(
            query,
            Query {
                kind: CatalogKind::DataCenters,
                gpu_count: 2,
                gpu_types: vec!["NVIDIA A40".into()],
                fit_by: FitBy::Nothing,
            }
        );
        Ok(())
    }

    #[test]
    fn no_gpu_type_chosen_changes_nothing() -> TestResult {
        let (dir, mut app) = editing_app()?;
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.gpu_types")?;
        listed(&mut app, id, gpu_catalog(1)?, gpu_types()?);
        press(&mut app, &[KeyCode::Char(' '), KeyCode::Enter]);
        assert_eq!(written(dir.path())?, PROJECT_CONFIG);
        assert_eq!(
            status(&app),
            "targets.gpu_cloud.gpu_types: choose a GPU type or auto; nothing changed"
        );
        Ok(())
    }

    fn volumes() -> Vec<Entry> {
        volume_entries(&[
            NetworkVolume {
                id: "voleu".into(),
                name: "alpha".into(),
                size: 100,
                data_center: "EU-RO-1".into(),
            },
            NetworkVolume {
                id: "volus".into(),
                name: "zeta".into(),
                size: 50,
                data_center: "US-KS-2".into(),
            },
        ])
    }

    #[test]
    fn picking_a_volume_writes_it_and_its_single_data_center_at_once() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        let centers = FieldPath::Target {
            name: "gpu_cloud".into(),
            field: "data_center_ids",
        };
        let two = FieldValue::List(vec!["US-KS-2".into(), "CA-MTL-1".into()]);
        edit_now(&mut app, &centers, Some(&two))?;
        let two_centers = app.config.as_ref().map(|config| config.text.clone());
        let (id, query) = open_picker_on(&mut app, "targets.gpu_cloud.network_volume_id")?;
        assert_eq!(query.kind, CatalogKind::Volumes);
        listed(&mut app, id, volumes(), Vec::new());
        // none, alpha, zeta: alpha is picked.
        press(&mut app, &[KeyCode::Down, KeyCode::Enter]);
        assert_eq!(
            shown(&mut app, "targets.gpu_cloud.network_volume_id")?,
            "voleu"
        );
        assert_eq!(
            shown(&mut app, "targets.gpu_cloud.data_center_ids")?,
            "EU-RO-1"
        );
        assert_eq!(
            status(&app),
            "✓ saved overbrainer.toml; targets.gpu_cloud.data_center_ids = EU-RO-1, the \
             volume's data center"
        );
        assert_eq!(
            Some(before_last_write(&app)?.to_string()),
            two_centers,
            "both in one write"
        );
        // none unsets the volume and leaves the data centers.
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.network_volume_id")?;
        listed(&mut app, id, volumes(), Vec::new());
        press(&mut app, &[KeyCode::Home, KeyCode::Enter]);
        assert_eq!(
            field(&mut app, "targets.gpu_cloud.network_volume_id")?.shown,
            Shown::Unset
        );
        assert_eq!(
            shown(&mut app, "targets.gpu_cloud.data_center_ids")?,
            "EU-RO-1"
        );
        Ok(())
    }

    #[test]
    fn with_a_volume_the_data_centers_change_only_with_the_volume() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.network_volume_id")?;
        listed(&mut app, id, volumes(), Vec::new());
        press(&mut app, &[KeyCode::Down, KeyCode::Enter]);
        let key = "targets.gpu_cloud.data_center_ids";
        assert_eq!(shown(&mut app, key)?, "EU-RO-1");
        let centers = || {
            ["EU-RO-1", "US-KS-2"]
                .map(|id| Entry {
                    id: id.into(),
                    columns: vec![id.into(), String::new(), String::new(), "HIGH".into()],
                    selectable: true,
                    ranks: Vec::new(),
                })
                .to_vec()
        };
        let refused = "refused: targets.gpu_cloud.data_center_ids is the network volume's data \
                       center; pick another network_volume_id to change it";
        // Another data center, none, then auto: each refused.
        for codes in [
            &[KeyCode::End, KeyCode::Char(' '), KeyCode::Enter][..],
            &[KeyCode::Char(' '), KeyCode::Enter],
            &[KeyCode::Home, KeyCode::Char(' '), KeyCode::Enter],
        ] {
            app.status = None;
            let (id, _) = open_picker_on(&mut app, key)?;
            listed(&mut app, id, centers(), Vec::new());
            press(&mut app, codes);
            assert!(!picker_open(&app), "{codes:?}");
            assert_eq!(shown(&mut app, key)?, "EU-RO-1", "{codes:?}");
            assert_eq!(status(&app), refused, "{codes:?}");
        }
        // Keeping the volume's own data center is no change.
        app.status = None;
        let (id, _) = open_picker_on(&mut app, key)?;
        listed(&mut app, id, centers(), Vec::new());
        assert!(press(&mut app, &[KeyCode::Enter]).is_empty(), "no write");
        assert_eq!(status(&app), "");
        // Typed instead: refused in the form.
        open_picker_on(&mut app, key)?;
        press(&mut app, &[KeyCode::Char('t'), KeyCode::End]);
        press(&mut app, &[KeyCode::Backspace; 20]);
        chars(&mut app, "US-KS-2");
        press(&mut app, &[KeyCode::Enter]);
        let Some(Form::Value { error, .. }) = &app.project_view.form else {
            return Err("the form closed".into());
        };
        assert_eq!(error.as_deref(), Some(refused));
        press(&mut app, &[KeyCode::Esc]);
        assert_eq!(shown(&mut app, key)?, "EU-RO-1");
        // Without a volume, the data centers change again.
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.network_volume_id")?;
        listed(&mut app, id, volumes(), Vec::new());
        press(&mut app, &[KeyCode::Home, KeyCode::Enter]);
        let (id, _) = open_picker_on(&mut app, key)?;
        listed(&mut app, id, centers(), Vec::new());
        press(
            &mut app,
            &[KeyCode::End, KeyCode::Char(' '), KeyCode::Enter],
        );
        assert_eq!(shown(&mut app, key)?, "EU-RO-1, US-KS-2");
        Ok(())
    }

    /// `t` in the volume picker, `volume` typed, then Enter.
    fn type_volume(app: &mut App, volume: &str, catalog: Option<Vec<Entry>>) -> TestResult {
        let (id, _) = open_picker_on(app, "targets.gpu_cloud.network_volume_id")?;
        if let Some(entries) = catalog {
            listed(app, id, entries, Vec::new());
        }
        press(app, &[KeyCode::Char('t'), KeyCode::End]);
        press(app, &[KeyCode::Backspace; 20]);
        chars(app, volume);
        press(app, &[KeyCode::Enter]);
        assert_eq!(app.project_view.form, None, "{}", status(app));
        Ok(())
    }

    #[test]
    fn a_typed_volume_the_catalog_lists_sets_its_data_center() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        type_volume(&mut app, "voleu", Some(volumes()))?;
        let key = "targets.gpu_cloud.data_center_ids";
        assert_eq!(shown(&mut app, key)?, "EU-RO-1");
        // Typed again, while the listing is still read: the one read is used.
        type_volume(&mut app, "volus", None)?;
        assert_eq!(
            shown(&mut app, "targets.gpu_cloud.network_volume_id")?,
            "volus"
        );
        assert_eq!(shown(&mut app, key)?, "US-KS-2");
        assert_eq!(
            status(&app),
            "✓ saved overbrainer.toml; targets.gpu_cloud.data_center_ids = US-KS-2, the \
             volume's data center"
        );
        Ok(())
    }

    #[test]
    fn a_typed_volume_the_catalog_lacks_leaves_the_data_centers_free() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        type_volume(&mut app, "voleu", Some(volumes()))?;
        type_volume(&mut app, "volnew", Some(volumes()))?;
        let key = "targets.gpu_cloud.data_center_ids";
        assert_eq!(shown(&mut app, key)?, "EU-RO-1", "unchanged");
        app.status = None;
        open_picker_on(&mut app, key)?;
        press(&mut app, &[KeyCode::Char('t'), KeyCode::End]);
        press(&mut app, &[KeyCode::Backspace; 20]);
        chars(&mut app, "US-KS-2");
        press(&mut app, &[KeyCode::Enter]);
        assert_eq!(app.project_view.form, None, "not refused: {}", status(&app));
        assert_eq!(shown(&mut app, key)?, "US-KS-2");
        // Picking a listed volume makes its data center known again.
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.network_volume_id")?;
        listed(&mut app, id, volumes(), Vec::new());
        press(&mut app, &[KeyCode::Home, KeyCode::Down, KeyCode::Enter]);
        assert_eq!(shown(&mut app, key)?, "EU-RO-1");
        let (id, _) = open_picker_on(&mut app, key)?;
        listed(&mut app, id, Vec::new(), Vec::new());
        press(&mut app, &[KeyCode::Char('t'), KeyCode::End]);
        press(&mut app, &[KeyCode::Backspace; 20]);
        chars(&mut app, "US-KS-2");
        press(&mut app, &[KeyCode::Enter]);
        assert!(app.project_view.form.is_some(), "refused again");
        Ok(())
    }

    /// `t` on the field `key` (a picker), `value` typed, then Enter; whether
    /// the form closed, that is the value was taken.
    fn type_in(app: &mut App, key: &str, value: &str) -> Result<bool, String> {
        open_picker_on(app, key)?;
        press(app, &[KeyCode::Char('t'), KeyCode::End]);
        press(app, &[KeyCode::Backspace; 20]);
        chars(app, value);
        press(app, &[KeyCode::Enter]);
        let taken = app.project_view.form.is_none();
        press(app, &[KeyCode::Esc]);
        Ok(taken)
    }

    #[test]
    fn an_unlisted_volume_kept_in_its_picker_holds_no_data_center() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        let target = |field| FieldPath::Target {
            name: "gpu_cloud".into(),
            field,
        };
        edit_now(
            &mut app,
            &target("data_center_ids"),
            Some(&FieldValue::List(vec!["EU-RO-1".into()])),
        )?;
        edit_now(
            &mut app,
            &target("network_volume_id"),
            Some(&FieldValue::Text("volnew".into())),
        )?;
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.network_volume_id")?;
        listed(&mut app, id, volumes(), Vec::new());
        press(&mut app, &[KeyCode::Enter]);
        assert_eq!(
            shown(&mut app, "targets.gpu_cloud.network_volume_id")?,
            "volnew"
        );
        let key = "targets.gpu_cloud.data_center_ids";
        assert!(type_in(&mut app, key, "US-KS-2")?, "{}", status(&app));
        assert_eq!(shown(&mut app, key)?, "US-KS-2");
        Ok(())
    }

    #[test]
    fn a_typed_volume_listed_later_gets_its_data_center_in_its_own_write() -> TestResult {
        let (dir, mut app) = editing_app()?;
        let key = "targets.gpu_cloud.data_center_ids";
        // A volume needs one data center: without it, the file does not
        // validate and the form keeps the value.
        assert!(!type_in(
            &mut app,
            "targets.gpu_cloud.network_volume_id",
            "volus"
        )?);
        assert_eq!(written(dir.path())?, PROJECT_CONFIG);
        assert!(type_in(&mut app, key, "EU-RO-1")?);
        assert!(
            type_in(&mut app, "targets.gpu_cloud.network_volume_id", "volus")?,
            "not listed yet: allowed"
        );
        let before = written(dir.path())?;
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.network_volume_id")?;
        listed(&mut app, id, volumes(), Vec::new());
        assert_eq!(shown(&mut app, key)?, "US-KS-2", "reconciled");
        assert_eq!(
            status(&app),
            "✓ saved overbrainer.toml; targets.gpu_cloud.data_center_ids = US-KS-2, the \
             network volume's data center"
        );
        assert!(written(dir.path())?.contains("data_center_ids = [\"US-KS-2\"]"));
        assert_eq!(before_last_write(&app)?, before, "a write of its own");
        press(&mut app, &[KeyCode::Esc]);
        assert!(!type_in(&mut app, key, "EU-RO-1")?, "refused now");
        assert_eq!(shown(&mut app, key)?, "US-KS-2");
        Ok(())
    }

    #[test]
    fn a_listing_reconciles_every_target_in_one_write_and_says_it_once() -> TestResult {
        let (dir, mut app) = editing_app()?;
        let config = PROJECT_CONFIG.replace(
            "max_hours = 6\n",
            "max_hours = 6\nnetwork_volume_id = \"voleu\"\ndata_center_ids = [\"US-KS-2\"]\n\n\
             [targets.gpu_two]\nkind = \"runpod\"\ngpu_types = [\"NVIDIA A40\"]\n\
             max_hours = 6\nnetwork_volume_id = \"volus\"\ndata_center_ids = [\"EU-RO-1\"]\n\n\
             [targets.gpu_three]\nkind = \"runpod\"\ngpu_types = [\"NVIDIA A40\"]\n\
             max_hours = 6\nnetwork_volume_id = \"volus\"\ndata_center_ids = [\"CA-MTL-1\"]\n",
        );
        std::fs::write(dir.path().join(CONFIG_FILE), &config)?;
        let read = crate::tui::project::ProjectConfig::new(&config, &app.env)?;
        app.set_config(read);
        let mut follow =
            crate::tui::training::Follow::new(crate::tui::training::Job::Attach, "20260921-a1");
        follow.watching = true;
        app.training.tasks.insert(TaskId(90), follow);
        let volumes = volume_entries(&[
            NetworkVolume {
                id: "voleu".into(),
                name: "alpha".into(),
                size: 100,
                data_center: "EU-RO-1".into(),
            },
            NetworkVolume {
                id: "volus".into(),
                name: "zeta".into(),
                size: 50,
                data_center: "US-KS-2".into(),
            },
        ]);
        let (id, _) = open_picker_on(&mut app, "targets.gpu_two.network_volume_id")?;
        listed(&mut app, id, volumes, Vec::new());
        for name in ["gpu_two", "gpu_three"] {
            assert_eq!(
                shown(&mut app, &format!("targets.{name}.data_center_ids"))?,
                "US-KS-2"
            );
        }
        assert_eq!(
            shown(&mut app, "targets.gpu_cloud.data_center_ids")?,
            "US-KS-2",
            "locked by the run"
        );
        assert_eq!(before_last_write(&app)?, config, "one write");
        let said = app.status.as_ref().ok_or("nothing said")?;
        assert_eq!(said.severity, Severity::Warn);
        assert_eq!(
            said.text,
            "✓ saved overbrainer.toml; targets.gpu_cloud.data_center_ids not set to EU-RO-1, \
             the network volume's data center: used by run 20260921-a1; \
             targets.gpu_two.data_center_ids = US-KS-2, targets.gpu_three.data_center_ids = \
             US-KS-2, the network volumes' data centers"
        );
        Ok(())
    }

    #[test]
    fn picking_a_template_sets_its_image_and_esc_keeps_the_value() -> TestResult {
        let (dir, mut app) = editing_app()?;
        let templates = template_entries(&[Template {
            id: "t1".into(),
            name: "trainer".into(),
            image: "img/trainer:2".into(),
            serverless: false,
        }]);
        let (id, query) = open_picker_on(&mut app, "targets.gpu_cloud.image")?;
        assert_eq!(query.kind, CatalogKind::Templates);
        listed(&mut app, id, templates.clone(), Vec::new());
        press(&mut app, &[KeyCode::Esc]);
        assert_eq!(written(dir.path())?, PROJECT_CONFIG, "Esc keeps the image");
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.image")?;
        listed(&mut app, id, templates, Vec::new());
        press(&mut app, &[KeyCode::End, KeyCode::Enter]);
        assert_eq!(shown(&mut app, "targets.gpu_cloud.image")?, "img/trainer:2");
        assert!(written(dir.path())?.contains("image = \"img/trainer:2\""));
        Ok(())
    }

    #[test]
    fn a_locked_field_refuses_the_picker_like_the_form() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        let mut follow =
            crate::tui::training::Follow::new(crate::tui::training::Job::Attach, "20260921-a1");
        follow.watching = true;
        app.training.tasks.insert(TaskId(9), follow);
        select(&mut app, "targets.gpu_cloud.gpu_types")?;
        assert_eq!(
            press(&mut app, &[KeyCode::Enter]),
            [] as [crate::tui::app::Effect; 0]
        );
        assert!(!picker_open(&app));
        assert_eq!(
            status(&app),
            "refused: targets.gpu_cloud.gpu_types is used by run 20260921-a1; \
             read-only until it ends"
        );
        Ok(())
    }

    #[test]
    fn other_runpod_fields_keep_the_form_with_catalog_hints() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        select(&mut app, "targets.gpu_cloud.gpu_count")?;
        let effects = press(&mut app, &[KeyCode::Enter]);
        let [Effect::Spawn(id, Task::Catalog(query))] = effects.as_slice() else {
            return Err(format!("no GPU listing: {effects:?}").into());
        };
        assert_eq!(query.kind, CatalogKind::Gpus);
        assert!(matches!(app.project_view.form, Some(Form::Value { .. })));
        assert_eq!(app.field_hint(), None, "not read yet");
        let id = *id;
        listed(&mut app, id, gpu_catalog(1)?, gpu_types()?);
        assert!(!picker_open(&app), "a hint's listing opens nothing");
        assert_eq!(
            app.field_hint().as_deref(),
            Some("at most 10 with the chosen types")
        );
        let rows = screen(&draw(&mut app, 120, 40)?).join("\n");
        assert!(rows.contains("at most 10 with the chosen types"), "{rows}");
        press(&mut app, &[KeyCode::Esc]);
        select(&mut app, "targets.gpu_cloud.max_hours")?;
        assert!(press(&mut app, &[KeyCode::Enter]).is_empty(), "read once");
        press(&mut app, &[KeyCode::Esc]);
        assert_eq!(
            app.field_hint().as_deref(),
            Some("at most $2.40 at the chosen prices (1 × $0.40/h × 6 h)")
        );
        let rows = screen(&draw(&mut app, 120, 40)?).join("\n");
        assert!(
            rows.contains("at most $2.40 at the chosen prices"),
            "{rows}"
        );
        select(&mut app, "targets.gpu_cloud.container_disk_gb")?;
        assert_eq!(app.field_hint(), None);
        Ok(())
    }

    #[test]
    fn the_fields_of_other_targets_keep_the_form() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        press(
            &mut app,
            &[KeyCode::Char('a'), KeyCode::Right, KeyCode::Right],
        );
        press(&mut app, &[KeyCode::Enter]);
        chars(&mut app, "box");
        press(&mut app, &[KeyCode::Enter, KeyCode::Enter]);
        select(&mut app, "targets.box.image")?;
        assert_eq!(
            press(&mut app, &[KeyCode::Enter]),
            [] as [crate::tui::app::Effect; 0]
        );
        assert!(!picker_open(&app));
        assert!(matches!(app.project_view.form, Some(Form::Value { .. })));
        Ok(())
    }

    #[test]
    fn t_in_a_picker_types_the_value_in_the_form_instead() -> TestResult {
        let (dir, mut app) = editing_app()?;
        // While the catalog is read, and for each of the four fields.
        for (key, now) in [
            ("targets.gpu_cloud.gpu_types", "NVIDIA A40"),
            ("targets.gpu_cloud.data_center_ids", ""),
            ("targets.gpu_cloud.network_volume_id", ""),
            ("targets.gpu_cloud.image", ""),
        ] {
            open_picker_on(&mut app, key)?;
            press(&mut app, &[KeyCode::Char('t')]);
            assert!(!picker_open(&app), "{key}");
            let Some(Form::Value { path, input, .. }) = &app.project_view.form else {
                return Err(format!("no form on {key}").into());
            };
            assert_eq!(path.to_string(), key);
            assert_eq!(input.text(), now, "{key}");
            press(&mut app, &[KeyCode::Esc]);
        }
        // A failed listing still lets the value be typed.
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.gpu_types")?;
        app.on_done(id, Ok(Done::Catalog(Err("no Runpod API key".into()))));
        press(&mut app, &[KeyCode::Char('t'), KeyCode::End]);
        press(&mut app, &[KeyCode::Backspace; 20]);
        chars(&mut app, "auto");
        press(&mut app, &[KeyCode::Enter]);
        assert_eq!(app.project_view.form, None);
        assert_eq!(shown(&mut app, "targets.gpu_cloud.gpu_types")?, "auto");
        open_picker_on(&mut app, "targets.gpu_cloud.data_center_ids")?;
        press(&mut app, &[KeyCode::Char('t')]);
        chars(&mut app, "EU-RO-1, US-KS-2");
        press(&mut app, &[KeyCode::Enter]);
        assert_eq!(
            shown(&mut app, "targets.gpu_cloud.data_center_ids")?,
            "EU-RO-1, US-KS-2"
        );
        let text = written(dir.path())?;
        assert!(text.contains(r#"gpu_types = "auto""#), "{text}");
        assert!(
            text.contains(r#"data_center_ids = ["EU-RO-1", "US-KS-2"]"#),
            "{text}"
        );
        Ok(())
    }

    #[test]
    fn auto_picked_is_written_as_a_string() -> TestResult {
        let (dir, mut app) = editing_app()?;
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.gpu_types")?;
        listed(&mut app, id, gpu_catalog(1)?, gpu_types()?);
        press(
            &mut app,
            &[KeyCode::Home, KeyCode::Char(' '), KeyCode::Enter],
        );
        assert_eq!(status(&app), "✓ saved overbrainer.toml");
        let text = written(dir.path())?;
        assert!(text.contains(r#"gpu_types = "auto""#), "{text}");
        Ok(())
    }

    #[test]
    fn the_default_row_of_the_template_picker_unsets_the_image() -> TestResult {
        let (dir, mut app) = editing_app()?;
        let templates = template_entries(&[Template {
            id: "t1".into(),
            name: "trainer".into(),
            image: "img/trainer:2".into(),
            serverless: false,
        }]);
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.image")?;
        listed(&mut app, id, templates.clone(), Vec::new());
        press(&mut app, &[KeyCode::End, KeyCode::Enter]);
        assert_eq!(shown(&mut app, "targets.gpu_cloud.image")?, "img/trainer:2");
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.image")?;
        listed(&mut app, id, templates, Vec::new());
        press(&mut app, &[KeyCode::Home, KeyCode::Enter]);
        assert_eq!(
            field(&mut app, "targets.gpu_cloud.image")?.shown,
            Shown::Unset,
            "the pinned default applies"
        );
        assert_eq!(written(dir.path())?, PROJECT_CONFIG, "back to the file");
        Ok(())
    }

    #[test]
    fn a_lock_taken_while_the_picker_is_open_refuses_the_choice() -> TestResult {
        let (dir, mut app) = editing_app()?;
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.gpu_types")?;
        listed(&mut app, id, gpu_catalog(1)?, gpu_types()?);
        let mut follow =
            crate::tui::training::Follow::new(crate::tui::training::Job::Attach, "20260921-a1");
        follow.watching = true;
        app.training.tasks.insert(TaskId(9), follow);
        press(
            &mut app,
            &[KeyCode::Home, KeyCode::Char(' '), KeyCode::Enter],
        );
        assert_eq!(written(dir.path())?, PROJECT_CONFIG);
        assert_eq!(
            status(&app),
            "refused: targets.gpu_cloud.gpu_types is used by run 20260921-a1; \
             read-only until it ends"
        );
        Ok(())
    }

    #[test]
    fn an_env_set_picker_field_refuses_enter() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        let EnvSource::Vars(mut vars) = project_env() else {
            return Err("vars expected".into());
        };
        vars.push((
            "OVERBRAINER_TARGETS__GPU_CLOUD__GPU_TYPES".into(),
            "NVIDIA L4".into(),
        ));
        app.env = EnvSource::Vars(vars);
        let config = ProjectConfig::new(PROJECT_CONFIG, &app.env)?;
        app.set_config(config);
        select(&mut app, "targets.gpu_cloud.gpu_types")?;
        assert_eq!(
            press(&mut app, &[KeyCode::Enter]),
            [] as [crate::tui::app::Effect; 0]
        );
        assert!(!picker_open(&app));
        assert_eq!(
            status(&app),
            "targets.gpu_cloud.gpu_types: set by OVERBRAINER_TARGETS__GPU_CLOUD__GPU_TYPES, \
             change it in .env"
        );
        Ok(())
    }

    #[test]
    fn a_volume_the_catalog_does_not_list_leaves_the_data_centers() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        let target = |field| FieldPath::Target {
            name: "gpu_cloud".into(),
            field,
        };
        let centers = FieldValue::List(vec!["CA-MTL-1".into()]);
        edit_now(&mut app, &target("data_center_ids"), Some(&centers))?;
        let volume = FieldValue::Text("volgone".into());
        edit_now(&mut app, &target("network_volume_id"), Some(&volume))?;
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.network_volume_id")?;
        listed(&mut app, id, volumes(), Vec::new());
        // none, alpha, zeta, then volgone, kept in view: picked again.
        press(&mut app, &[KeyCode::End, KeyCode::Enter]);
        assert_eq!(
            shown(&mut app, "targets.gpu_cloud.network_volume_id")?,
            "volgone"
        );
        assert_eq!(
            shown(&mut app, "targets.gpu_cloud.data_center_ids")?,
            "CA-MTL-1"
        );
        Ok(())
    }

    /// [`editing_app`] on `gpu_types = "auto"` with both limits.
    fn auto_limits_app() -> Result<(tempfile::TempDir, App), Box<dyn std::error::Error>> {
        let (dir, mut app) = editing_app()?;
        let config = PROJECT_CONFIG.replace(
            "gpu_types = [\"NVIDIA A40\"]",
            "gpu_types = \"auto\"\nmin_vram_gb = 24\nmax_price_per_hour = 1.5",
        );
        std::fs::write(dir.path().join(CONFIG_FILE), &config)?;
        app.set_config(ProjectConfig::new(&config, &app.env)?);
        Ok((dir, app))
    }

    /// Both `auto` limits are unset in the same write as the GPU types, and
    /// the status says so.
    fn limits_dropped(app: &mut App, dir: &Path) -> TestResult {
        for key in [
            "targets.gpu_cloud.min_vram_gb",
            "targets.gpu_cloud.max_price_per_hour",
        ] {
            assert_eq!(field(app, key)?.shown, Shown::Unset, "{key}");
        }
        assert_eq!(
            status(app),
            "✓ saved overbrainer.toml; targets.gpu_cloud.min_vram_gb and \
             targets.gpu_cloud.max_price_per_hour removed: only with gpu_types = \"auto\""
        );
        let text = written(dir)?;
        assert!(text.contains(r#"gpu_types = ["NVIDIA A40"]"#), "{text}");
        assert!(!text.contains("min_vram_gb") && !text.contains("max_price_per_hour"));
        let before = before_last_write(app)?;
        assert!(
            before.contains("gpu_types = \"auto\"\nmin_vram_gb = 24"),
            "one write: {before}"
        );
        Ok(())
    }

    #[test]
    fn gpu_types_picked_as_a_list_drop_the_auto_limits() -> TestResult {
        let (dir, mut app) = auto_limits_app()?;
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.gpu_types")?;
        listed(&mut app, id, gpu_catalog(1)?, gpu_types()?);
        // The A40, second cheapest, alone.
        press(
            &mut app,
            &[
                KeyCode::Down,
                KeyCode::Down,
                KeyCode::Char(' '),
                KeyCode::Enter,
            ],
        );
        assert_eq!(
            shown(&mut app, "targets.gpu_cloud.gpu_types")?,
            "NVIDIA A40"
        );
        limits_dropped(&mut app, dir.path())
    }

    #[test]
    fn gpu_types_typed_as_a_list_drop_the_auto_limits() -> TestResult {
        let (dir, mut app) = auto_limits_app()?;
        select(&mut app, "targets.gpu_cloud.gpu_types")?;
        press(&mut app, &[KeyCode::Enter, KeyCode::Char('t')]);
        press(&mut app, &[KeyCode::End]);
        press(&mut app, &[KeyCode::Backspace; 10]);
        chars(&mut app, "NVIDIA A40");
        press(&mut app, &[KeyCode::Enter]);
        limits_dropped(&mut app, dir.path())
    }

    #[test]
    fn validation_messages_name_their_field() {
        let topics = ["ownership".to_string(), "traits".to_string()];
        let path = |problem: &str| error_path(problem, &topics);
        assert_eq!(
            path("`topics[1]subtopics` has the wrong type, expected an integer").as_deref(),
            Some("topics.traits.subtopics")
        );
        assert_eq!(
            path("`topics[0].name` has the wrong type").as_deref(),
            Some("topics.ownership.name")
        );
        assert_eq!(
            path("`targets.cloud`: invalid value, expected one of").as_deref(),
            Some("targets.cloud")
        );
        assert_eq!(
            path("roles.parent: unknown provider `claude`").as_deref(),
            Some("roles.parent")
        );
        assert_eq!(path("unknown field `x`, expected one of"), None);
        assert_eq!(path("`topics[7]subtopics` is wrong"), None);
    }
}
