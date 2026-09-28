//! Editing the configuration from the Project view: the keys, the form under
//! the list, the confirmations, the save and the read after `$EDITOR`. Changes
//! stay in a pending document until `s` validates and writes it.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;
use std::process::ExitStatus;
use std::sync::Arc;

use crossterm::event::KeyCode;

use super::app::{Action, App, Confirm, Effect, Origin, Overlay, PAGE, Picked, Project, Severity};
use super::catalog::{
    CatalogKind, DEFAULT_IMAGE, NO_VOLUME, Query, Sizing, cost_hint, gpu_count_hint, parse_list,
    volume_data_center,
};
use super::project::{Addable, Form, Listing, Locks, Pending, ProjectConfig, Shown};
use super::tasks::Task;
use super::widgets::form::{Input, InputOutcome};
use super::widgets::picker::Choice;
use crate::config::edit::{Collection, ConfigDoc, FieldPath};
use crate::config::fields::{FieldKind, FieldValue, TargetKind};
use crate::config::{CONFIG_FILE, ConfigError, EnvSource, ListOrAuto, Protocol, is_valid_name};

/// What a confirmed `d` takes out of the pending document.
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

/// The first of `problems`, with how many more there are.
fn first_of(problems: &[String]) -> String {
    let first = problems.first().cloned().unwrap_or_default();
    match problems.len() {
        0 | 1 => first,
        count => format!("{first} (+{} more)", count - 1),
    }
}

/// `count` changes, in words.
fn changes(count: usize) -> String {
    if count == 1 {
        "1 change".to_string()
    } else {
        format!("{count} changes")
    }
}

impl App {
    /// Shows `config`, read from `overbrainer.toml`: the rows are built again.
    pub(super) fn set_config(&mut self, config: ProjectConfig) {
        self.config = Some(config);
        self.project_view.touch();
    }

    /// The rows of the Project view, built again only when the configuration,
    /// the pending changes, the errors or the locks changed.
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
            KeyCode::Char('s') => return self.save_changes(),
            KeyCode::Char('u') => self.ask_drop(),
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

    /// The pending changes, started from the file's document when there are
    /// none yet.
    fn pending_mut(&mut self) -> Option<&mut Pending> {
        let config = self.config.as_ref()?;
        Some(
            self.project_view
                .pending
                .get_or_insert_with(|| Pending::new(&config.doc)),
        )
    }

    /// The document shown: the pending one, else the file's.
    fn shown_doc(&self) -> Option<&ConfigDoc> {
        match &self.project_view.pending {
            Some(pending) => Some(&pending.doc),
            None => self.config.as_ref().map(|config| &config.doc),
        }
    }

    /// Drops the pending changes when they leave the document as it was read,
    /// and has the rows built again.
    fn settle_pending(&mut self) {
        let same = match (&self.config, &self.project_view.pending) {
            (Some(config), Some(pending)) => pending.doc.text() == config.doc.text(),
            _ => false,
        };
        if same {
            self.project_view.pending = None;
        }
        self.project_view.touch();
    }

    /// Sets the field `path` to `value` in the pending document, or unsets it
    /// for `None`, and marks it changed unless it is back to the file's value.
    fn apply(&mut self, path: &FieldPath, value: Option<&FieldValue>) -> Result<(), String> {
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| "no configuration read".to_string())?;
        let pending = self
            .project_view
            .pending
            .get_or_insert_with(|| Pending::new(&config.doc));
        let applied = match value {
            Some(value) => pending.doc.set(path, value.clone()),
            None => pending.doc.unset(path).map(drop),
        };
        if let Err(error) = applied {
            self.settle_pending();
            return Err(error.to_string());
        }
        self.project_view.errors.remove(&path.to_string());
        // A renamed topic keeps its marks, under its new name.
        let path = match (path, value) {
            (FieldPath::Topic { index, name, field }, Some(FieldValue::Text(new)))
                if *field == "name" =>
            {
                pending.rename(name, new);
                FieldPath::Topic {
                    index: *index,
                    name: new.clone(),
                    field,
                }
            },
            _ => path.clone(),
        };
        pending.mark(&path, &config.doc);
        self.settle_pending();
        Ok(())
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
        if let Some(user) = &field.lock {
            let said = format!("refused: {key} is used by {user}; read-only until it ends");
            self.say(Severity::Warn, said);
            return Vec::new();
        }
        if let Some(note) = field.env_note() {
            self.say(Severity::Warn, note);
            return Vec::new();
        }
        let Some(path) = field.path.clone() else {
            return Vec::new();
        };
        let Some(spec) = self.shown_doc().and_then(|doc| doc.spec(&path)) else {
            let said = format!("{key}: not edited here; E opens {CONFIG_FILE}");
            self.say(Severity::Warn, said);
            return Vec::new();
        };
        let mut effects = Vec::new();
        if let FieldPath::Target {
            name,
            field: name_of,
        } = &path
            && self.shown_doc().and_then(|doc| doc.target_kind(name)) == Some(TargetKind::Runpod)
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
        if let Err(error) = self.apply(&path, value.as_ref()) {
            self.say(Severity::Warn, error);
        }
        effects
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
            CatalogKind::Gpus | CatalogKind::DataCenters => Choice::from(&parse_list(shown)),
            // Unset: the `none` or `default` entry is the one chosen.
            CatalogKind::Volumes | CatalogKind::Templates => Choice::List(vec![shown.to_string()]),
        };
        let query = Query {
            kind,
            gpu_count: sizing.gpu_count,
            gpu_types: parse_list(&sizing.gpu_types).list().to_vec(),
        };
        self.open_picker(query, preselected, Origin::Field(path))
    }

    /// A picker opened on the Runpod target field `path` kept `picked`: it
    /// becomes a pending edit of that field. `auto` is written as such, an
    /// empty data center list unsets it (any), a volume sets the target's
    /// data centers to its own, `none` unsets it and leaves them.
    pub(super) fn picked_field(&mut self, path: &FieldPath, picked: Picked) {
        if self.refuse_change() || self.refuse_locked(path) {
            return;
        }
        let ids = match picked.choice {
            Choice::Auto => {
                let auto = FieldValue::Text(ListOrAuto::AUTO.to_string());
                if let Err(error) = self.apply(path, Some(&auto)) {
                    self.say(Severity::Warn, error);
                }
                return;
            },
            Choice::List(ids) => ids,
        };
        let applied = match (picked.kind, ids.first()) {
            (CatalogKind::Gpus, None) => Err(format!(
                "{path}: choose a GPU type or auto; nothing changed"
            )),
            (CatalogKind::DataCenters, None) => self.apply(path, None),
            (CatalogKind::Gpus | CatalogKind::DataCenters, Some(_)) => {
                self.apply(path, Some(&FieldValue::List(ids)))
            },
            (CatalogKind::Volumes | CatalogKind::Templates, None) => Ok(()),
            (CatalogKind::Volumes, Some(id)) if id == NO_VOLUME => self.apply(path, None),
            (CatalogKind::Volumes, Some(id)) => self.picked_volume(path, id, &picked.entries),
            (CatalogKind::Templates, Some(image)) if image == DEFAULT_IMAGE => {
                self.apply(path, None)
            },
            (CatalogKind::Templates, Some(image)) => {
                self.apply(path, Some(&FieldValue::Text(image.clone())))
            },
        };
        if let Err(error) = applied {
            self.say(Severity::Warn, error);
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
        let Some(spec) = self.shown_doc().and_then(|doc| doc.spec(path)) else {
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

    /// The volume `id` picked for the field `path`: it is set, and its
    /// target's data centers become the volume's, when `entries` says which;
    /// both or neither.
    fn picked_volume(
        &mut self,
        path: &FieldPath,
        id: &str,
        entries: &[super::widgets::picker::Entry],
    ) -> Result<(), String> {
        let center = entries
            .iter()
            .find(|entry| entry.id == id)
            .and_then(volume_data_center)
            .map(str::to_string);
        let before = self.project_view.pending.clone();
        let applied = self.apply(path, Some(&FieldValue::Text(id.to_string())));
        let (FieldPath::Target { name, .. }, Some(center), Ok(())) = (path, center, &applied)
        else {
            return applied;
        };
        let centers = FieldPath::Target {
            name: name.clone(),
            field: "data_center_ids",
        };
        if let Err(error) = self.apply(&centers, Some(&FieldValue::List(vec![center.clone()]))) {
            self.project_view.pending = before;
            self.settle_pending();
            return Err(error);
        }
        self.say(
            Severity::Info,
            format!("{centers} = {center}, the volume's data center"),
        );
        Ok(())
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
        if self.shown_doc()?.target_kind(name) != Some(TargetKind::Runpod) {
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

    /// A key while the form is open.
    pub(super) fn on_form_key(&mut self, code: KeyCode) {
        let Some(form) = self.project_view.form.take() else {
            return;
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
                    Ok(()) => None,
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
                    Ok(next) => next,
                    Err(error) => Some(Form::Name {
                        what,
                        input,
                        error: Some(error),
                    }),
                },
            },
            Form::Kind { what, name, choice } => match code {
                KeyCode::Enter => {
                    self.add(what, &name, choice);
                    None
                },
                KeyCode::Esc => None,
                _ => Some(Form::Kind {
                    what,
                    choice: cycle(choice, what.kinds().len().max(1)).unwrap_or(choice),
                    name,
                }),
            },
        };
    }

    /// Enter in the value form: checks `text` against `kind` and its bounds,
    /// then keeps it in the pending document; an empty text unsets an
    /// `optional` field.
    fn typed(
        &mut self,
        path: &FieldPath,
        kind: FieldKind,
        optional: bool,
        text: &str,
    ) -> Result<(), String> {
        let value = match (text.trim().is_empty(), optional) {
            (true, true) => None,
            (true, false) => return Err("is required".to_string()),
            (false, _) => Some(kind.parse(text).map_err(|error| error.to_string())?),
        };
        if let (FieldPath::Topic { index, field, .. }, Some(FieldValue::Text(new))) = (path, &value)
            && *field == "name"
        {
            let names = self
                .shown_doc()
                .map(ConfigDoc::topic_names)
                .unwrap_or_default();
            let clash = names
                .iter()
                .enumerate()
                .any(|(at, topic)| at != *index && topic == new);
            if clash {
                return Err(format!("topics.{new} already exists"));
            }
        }
        self.apply(path, value.as_ref())
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
    /// and be new: a topic is added at once, a provider or a target asks its
    /// protocol or kind next.
    fn named(&mut self, what: Addable, name: &str) -> Result<Option<Form>, String> {
        let Some(doc) = self.shown_doc() else {
            return Ok(None);
        };
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
            self.add(what, name, 0);
            return Ok(None);
        }
        Ok(Some(Form::Kind {
            what,
            name: name.to_string(),
            choice: 0,
        }))
    }

    /// Adds `what` named `name` to the pending document, with the `choice`-th
    /// of its kinds, and selects its first field.
    fn add(&mut self, what: Addable, name: &str, choice: usize) {
        let kind = what.kinds().get(choice).copied().unwrap_or_default();
        let key = table_key(what, name);
        let Some(pending) = self.pending_mut() else {
            return;
        };
        let added = match what {
            Addable::Topic => pending.add_topic(name),
            Addable::Provider => {
                let protocol = if kind == "anthropic" {
                    Protocol::Anthropic
                } else {
                    Protocol::Openai
                };
                pending.doc.add_provider(name, protocol)
            },
            Addable::Target => pending.doc.add_target(
                name,
                TargetKind::from_name(kind).unwrap_or(TargetKind::Local),
            ),
        };
        match added {
            Ok(()) => {
                pending.note_table(&key, true);
                self.settle_pending();
                if let Some(index) = self.project_listing().find(&key) {
                    self.project_view.selected = index;
                }
                self.say(Severity::Info, format!("{key} added; s saves it"));
            },
            Err(error) => {
                self.settle_pending();
                self.say(Severity::Warn, error.to_string());
            },
        }
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
                "{key} goes from the pending changes: s saves them, u drops them."
            )],
            yes: "delete",
            no: "keep",
            action: Action::Remove(removal),
        }));
    }

    /// A confirmed `d`: takes `removal` out of the pending document.
    pub(super) fn remove(&mut self, removal: &Removal) {
        if self.refuse_change() {
            return;
        }
        let key = removal.key();
        let Some(pending) = self.pending_mut() else {
            return;
        };
        let removed = match removal {
            Removal::Topic(index, name) => {
                pending.doc.topic_names().get(*index) == Some(name) && pending.remove_topic(*index)
            },
            Removal::Table(collection, name) => pending.doc.remove_table(*collection, name),
        };
        if removed {
            pending.note_table(&key, false);
        }
        self.settle_pending();
        let count = self.project_listing().fields.len();
        self.project_view.step(0, count);
        if removed {
            self.say(Severity::Info, format!("{key} deleted; s saves it"));
        } else {
            self.say(
                Severity::Warn,
                format!("{key} is not in {CONFIG_FILE}: the environment sets it"),
            );
        }
    }

    /// `s`: validates and writes the pending document, off the UI thread;
    /// refused while something uses a changed field or table.
    fn save_changes(&mut self) -> Vec<Effect> {
        if self.refuse_change() {
            return Vec::new();
        }
        let (Some(config), Some(pending)) = (&self.config, &self.project_view.pending) else {
            self.say(Severity::Info, "nothing to save");
            return Vec::new();
        };
        let locks = Locks::of(self);
        let conflict = pending.changed.iter().find_map(|key| {
            locks
                .user_of(&config.settings, key)
                .map(|user| (key.clone(), user))
        });
        if let Some((key, user)) = conflict {
            self.say(
                Severity::Warn,
                format!("refused: {key} is used by {user}; save once it ends"),
            );
            return Vec::new();
        }
        if self.leaving.is_some() {
            self.say(Severity::Warn, "refused: quitting; no save starts");
            return Vec::new();
        }
        let (Some(config), Some(pending)) = (&self.config, &self.project_view.pending) else {
            return Vec::new();
        };
        let task = Task::SaveConfig {
            text: pending.doc.text(),
            base: config.text.clone(),
            env: self.env.clone(),
        };
        let id = self.task_id();
        self.project_view.save = Some(id);
        vec![Effect::Spawn(id, task)]
    }

    /// The save ended: the app reads the configuration written, or each
    /// problem is shown on the field it names and nothing changed. A save of
    /// the choices made at start then starts the run, or says why it does not.
    pub(super) fn config_saved(
        &mut self,
        saved: Result<Box<ProjectConfig>, SaveRefusal>,
    ) -> Vec<Effect> {
        self.project_view.save = None;
        let start = self.start_after_save.take();
        let mut effects = Vec::new();
        match saved {
            Ok(config) => {
                let dir = self.project.dir.clone();
                self.project = Project::new(&dir, &config.settings);
                self.project_view.pending = None;
                self.project_view.errors.clear();
                self.set_config(*config);
                if self.leaving.is_some() {
                    self.exit_notes.push(format!("{CONFIG_FILE} was saved"));
                }
                self.say(Severity::Info, format!("✓ saved {CONFIG_FILE}"));
                effects = self.reload();
                if let Some(plan) = start {
                    effects.extend(self.started_after_save(&plan));
                }
            },
            Err(refusal) if start.is_some() => {
                // The file keeps its values: the Project view marks nothing.
                let why = match refusal {
                    SaveRefusal::Invalid(problems) => first_of(&problems),
                    SaveRefusal::Failed(error) => error,
                };
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

    /// Shows each of `problems` on the field it names, and selects the first.
    fn mark_errors(&mut self, problems: &[String]) {
        let topics = self
            .shown_doc()
            .map(ConfigDoc::topic_names)
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
        let listing = self.project_listing();
        if let Some(index) = (0..listing.fields.len()).find(|index| {
            listing
                .field(*index)
                .is_some_and(|field| field.error.is_some())
        }) {
            self.project_view.selected = index;
        }
    }

    /// `u`: asks to drop the pending changes.
    fn ask_drop(&mut self) {
        if self.refuse_change() {
            return;
        }
        let count = self.project_view.changes();
        if self.project_view.pending.is_none() {
            self.say(Severity::Info, "nothing to drop");
            return;
        }
        self.overlay = Some(Overlay::Confirm(Confirm {
            title: " Drop the pending changes? ".to_string(),
            text: vec![format!(
                "{} to {CONFIG_FILE} {} dropped; the file stays as it is.",
                changes(count),
                if count == 1 { "is" } else { "are" },
            )],
            yes: "drop",
            no: "keep",
            action: Action::DropChanges,
        }));
    }

    /// Drops the pending changes and the errors of the last save.
    pub(super) fn drop_changes(&mut self) {
        self.project_view.pending = None;
        self.project_view.errors.clear();
        self.project_view.form = None;
        self.project_view.touch();
    }

    /// `E`: opens `overbrainer.toml` in the editor; refused with pending
    /// changes, while a save runs, or while a stage, an edit or a training
    /// uses the configuration.
    fn open_config(&mut self) -> Vec<Effect> {
        if self.refuse_change() {
            return Vec::new();
        }
        if self.project_view.pending.is_some() {
            self.say(
                Severity::Warn,
                "refused: save (s) or drop (u) the pending changes first",
            );
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
        vec![Effect::OpenEditor {
            command: self.editor.clone(),
            path: self.project.dir.join(CONFIG_FILE),
        }]
    }

    /// The editor on `overbrainer.toml` ended with `status`: the file is read
    /// again; one that does not load leaves the view as it was.
    pub(super) fn config_edited(&mut self, status: io::Result<ExitStatus>) -> Vec<Effect> {
        let failed = editor_failure(status);
        let path = self.project.dir.join(CONFIG_FILE);
        let read = std::fs::read_to_string(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))
            .and_then(|text| {
                ProjectConfig::new(&text, &self.env).map_err(|error| {
                    let text = error.to_string();
                    text.lines().map(str::trim).collect::<Vec<_>>().join(" ")
                })
            });
        let config = match read {
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
        let dir = self.project.dir.clone();
        self.project = Project::new(&dir, &config.settings);
        self.project_view.errors.clear();
        self.set_config(config);
        let what = if unchanged {
            format!("{CONFIG_FILE} unchanged")
        } else {
            format!("{CONFIG_FILE} read again")
        };
        match failed {
            Some(failed) => self.say(Severity::Warn, format!("{failed}; {what}")),
            None => self.say(Severity::Info, what),
        }
        self.reload()
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
    use crate::tui::app::View;
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

    fn press(app: &mut App, codes: &[KeyCode]) -> Vec<Effect> {
        codes
            .iter()
            .flat_map(|code| app.on_input(&key(*code)))
            .collect()
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

    /// Enter on the field `key`, its text replaced by `value`, then Enter.
    fn set(app: &mut App, key: &str, value: &str) -> Result<(), String> {
        select(app, key)?;
        press(app, &[KeyCode::Enter, KeyCode::End]);
        press(app, &[KeyCode::Backspace; 80]);
        chars(app, value);
        press(app, &[KeyCode::Enter]);
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

    /// `s`, then the save task run in `dir`, its end handed back to `app`.
    fn save(app: &mut App, dir: &Path) -> Result<Vec<Effect>, String> {
        let effects = press(app, &[KeyCode::Char('s')]);
        let [Effect::Spawn(id, Task::SaveConfig { text, base, env })] = effects.as_slice() else {
            return Err(format!("no save: {effects:?} ({})", status(app)));
        };
        assert_eq!(app.project_view.save, Some(*id));
        assert_eq!(env, &project_env(), "the app's environment");
        let saved = save_config(dir, text, base, env).map(Box::new);
        Ok(app.on_done(*id, Ok(Done::ConfigSaved(saved))))
    }

    fn written(dir: &Path) -> std::io::Result<String> {
        std::fs::read_to_string(dir.join(CONFIG_FILE))
    }

    #[test]
    fn a_number_out_of_bounds_is_refused_on_enter() -> TestResult {
        let (_dir, mut app) = editing_app()?;
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
        assert!(app.project_view.pending.is_none(), "Esc keeps nothing");
        Ok(())
    }

    #[test]
    fn a_text_edited_and_saved_is_written_with_its_comments() -> TestResult {
        let (dir, mut app) = editing_app()?;
        set(&mut app, "project.name", "rust_pro")?;
        assert_eq!(app.project_view.form, None);
        let name = field(&mut app, "project.name")?;
        assert!(name.changed, "marked with *");
        assert_eq!(name.shown.text(), "rust_pro");
        assert_eq!(written(dir.path())?, PROJECT_CONFIG, "nothing written yet");
        let effects = save(&mut app, dir.path())?;
        let text = written(dir.path())?;
        assert!(text.contains("name = \"rust_pro\""), "{text}");
        assert!(text.starts_with("# the project\n"), "{text}");
        assert!(
            text.contains("concurrency = 16 # overridden by env"),
            "{text}"
        );
        assert_eq!(status(&app), "✓ saved overbrainer.toml");
        assert_eq!(app.project.name, "rust_pro", "the settings are read again");
        assert!(app.project_view.pending.is_none());
        assert_eq!(app.project_view.save, None);
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, Effect::Spawn(_, Task::Load))),
            "the data is read again with the new topics"
        );
        assert!(!field(&mut app, "project.name")?.changed);
        Ok(())
    }

    #[test]
    fn a_save_that_does_not_validate_writes_nothing_and_marks_the_field() -> TestResult {
        let (dir, mut app) = editing_app()?;
        select(&mut app, "roles.generator.reasoning_effort")?;
        press(&mut app, &[KeyCode::Enter]);
        assert_eq!(
            field(&mut app, "roles.generator.reasoning_effort")?
                .shown
                .text(),
            "low",
            "a choice cycles from unset to the first"
        );
        select(&mut app, "project.name")?;
        save(&mut app, dir.path())?;
        assert_eq!(written(dir.path())?, PROJECT_CONFIG);
        assert!(
            status(&app).starts_with(
                "not saved: roles.generator.reasoning_effort: requires reasoning = true"
            ),
            "{}",
            status(&app)
        );
        let marked = field(&mut app, "roles.generator.reasoning_effort")?;
        assert_eq!(
            marked.error.as_deref(),
            Some("roles.generator.reasoning_effort: requires reasoning = true")
        );
        assert_eq!(
            marked.detail(None),
            marked.error.clone().unwrap_or_default()
        );
        assert_eq!(
            app.project_listing()
                .field(app.project_view.selected)
                .map(|field| field.key.clone()),
            Some(marked.key),
            "the first marked field is selected"
        );
        assert!(app.project_view.pending.is_some(), "the changes are kept");
        Ok(())
    }

    #[test]
    fn a_adds_a_topic_and_s_saves_it() -> TestResult {
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
        let selected = app
            .project_listing()
            .field(app.project_view.selected)
            .map(|field| (field.key.clone(), field.changed));
        assert_eq!(selected, Some(("topics.traits.name".to_string(), true)));
        save(&mut app, dir.path())?;
        let text = written(dir.path())?;
        assert!(
            text.contains("[[topics]]\nname = \"traits\"\nsubtopics = 10"),
            "{text}"
        );
        assert_eq!(app.project.topics.len(), 2);
        Ok(())
    }

    /// [`editing_app`] with a second topic, `traits`, pending.
    fn two_topics() -> Result<(tempfile::TempDir, App), Box<dyn std::error::Error>> {
        let (dir, mut app) = editing_app()?;
        press(&mut app, &[KeyCode::Char('a'), KeyCode::Enter]);
        chars(&mut app, "traits");
        press(&mut app, &[KeyCode::Enter]);
        Ok((dir, app))
    }

    fn topic_names(app: &App) -> Vec<String> {
        app.shown_doc()
            .map(ConfigDoc::topic_names)
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
        set(&mut app, "topics.traits.name", "traits")?;
        assert_eq!(app.project_view.form, None, "its own name is no clash");
        Ok(())
    }

    #[test]
    fn d_removes_the_topic_selected() -> TestResult {
        let (_dir, mut app) = two_topics()?;
        select(&mut app, "topics.traits.subtopics")?;
        press(&mut app, &[KeyCode::Char('d'), KeyCode::Char('y')]);
        assert_eq!(topic_names(&app), ["ownership"]);
        assert_eq!(status(&app), "topics.traits deleted; s saves it");
        Ok(())
    }

    #[test]
    fn a_provider_and_a_target_are_added_with_their_kind() -> TestResult {
        let (_dir, mut app) = editing_app()?;
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
        assert_eq!(app.project_view.changes(), 2);
        Ok(())
    }

    #[test]
    fn a_provider_a_role_uses_is_refused_at_save_naming_the_role() -> TestResult {
        let (dir, mut app) = editing_app()?;
        select(&mut app, "providers.claude.api_key")?;
        press(&mut app, &[KeyCode::Char('d')]);
        let Some(Overlay::Confirm(confirm)) = &app.overlay else {
            return Err("no dialog".into());
        };
        assert_eq!(confirm.title, " Delete providers.claude? ");
        press(&mut app, &[KeyCode::Char('y')]);
        assert!(
            field(&mut app, "providers.claude.protocol").is_err(),
            "gone"
        );
        assert_eq!(status(&app), "providers.claude deleted; s saves it");
        save(&mut app, dir.path())?;
        assert_eq!(written(dir.path())?, PROJECT_CONFIG);
        assert_eq!(
            status(&app),
            "not saved: roles.parent: unknown provider `claude`"
        );
        assert!(field(&mut app, "roles.parent.provider")?.error.is_some());
        Ok(())
    }

    #[test]
    fn a_locked_field_refuses_enter_and_a_locked_change_is_not_saved() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        set(&mut app, "roles.parent.model", "claude-opus-6")?;
        app.pipeline_task = Some(TaskId(7));
        app.pipeline.started(Command::Answers, 8);
        select(&mut app, "roles.parent.model")?;
        press(&mut app, &[KeyCode::Enter]);
        assert_eq!(app.project_view.form, None);
        assert_eq!(
            status(&app),
            "refused: roles.parent.model is used by answers; read-only until it ends"
        );
        let effects = press(&mut app, &[KeyCode::Char('s')]);
        assert!(effects.is_empty());
        assert_eq!(
            status(&app),
            "refused: roles.parent.model is used by answers; save once it ends"
        );
        select(&mut app, "providers.claude.protocol")?;
        press(&mut app, &[KeyCode::Char('d')]);
        assert_eq!(app.overlay, None);
        assert!(status(&app).starts_with("refused: providers.claude is used by answers"));
        Ok(())
    }

    #[test]
    fn env_fields_refuse_enter_and_say_to_use_dot_env() -> TestResult {
        let (_dir, mut app) = editing_app()?;
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
        assert!(app.project_view.pending.is_none());
        Ok(())
    }

    #[test]
    fn a_bool_toggles_a_choice_cycles_and_a_paste_goes_to_the_form() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        select(&mut app, "pipeline.include_system_prompt")?;
        let before = field(&mut app, "pipeline.include_system_prompt")?;
        press(&mut app, &[KeyCode::Enter]);
        let after = field(&mut app, "pipeline.include_system_prompt")?;
        assert_ne!(before.shown.text(), after.shown.text());
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
        Ok(())
    }

    #[test]
    fn switching_views_keeps_the_pending_changes_and_u_drops_them() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        set(&mut app, "project.name", "rust_pro")?;
        press(&mut app, &[KeyCode::Char('2')]);
        assert_eq!((app.view, &app.overlay), (View::Dataset, &None));
        press(
            &mut app,
            &[KeyCode::Tab, KeyCode::BackTab, KeyCode::Char('1')],
        );
        assert_eq!(field(&mut app, "project.name")?.shown.text(), "rust_pro");
        let effects = press(&mut app, &[KeyCode::Char('r'), KeyCode::Enter]);
        assert!(!effects.is_empty(), "a stage starts");
        assert_eq!(app.view, View::Pipeline);
        assert!(app.project_view.pending.is_some(), "kept");
        app.view = View::Project;
        app.pipeline_task = None;
        press(&mut app, &[KeyCode::Char('u')]);
        assert!(matches!(
            &app.overlay,
            Some(Overlay::Confirm(Confirm {
                action: Action::DropChanges,
                ..
            }))
        ));
        press(&mut app, &[KeyCode::Char('y')]);
        assert!(app.project_view.pending.is_none());
        assert_eq!(status(&app), "pending changes dropped");
        assert_eq!(field(&mut app, "project.name")?.shown.text(), "rust_expert");
        Ok(())
    }

    #[test]
    fn nothing_changes_while_a_save_runs() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        set(&mut app, "project.name", "rust_pro")?;
        let effects = press(&mut app, &[KeyCode::Char('s')]);
        assert_eq!(effects.len(), 1);
        for code in ['u', 'a', 'd', 's', 'E'] {
            assert!(press(&mut app, &[KeyCode::Char(code)]).is_empty(), "{code}");
            assert_eq!(app.overlay, None, "{code}");
            assert_eq!(app.project_view.form, None, "{code}");
            assert_eq!(status(&app), "refused: overbrainer.toml is being saved");
        }
        press(&mut app, &[KeyCode::Enter]);
        assert_eq!(app.project_view.form, None);
        assert_eq!(field(&mut app, "project.name")?.shown.text(), "rust_pro");
        Ok(())
    }

    #[test]
    fn a_field_set_back_loses_its_mark_and_a_rename_is_one_change() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        set(&mut app, "topics.ownership.subtopics", "5")?;
        set(&mut app, "topics.ownership.name", "owning")?;
        assert_eq!(app.project_view.changes(), 2);
        assert!(field(&mut app, "topics.owning.subtopics")?.changed);
        assert!(field(&mut app, "topics.owning.name")?.changed);
        set(&mut app, "topics.owning.subtopics", "2")?;
        assert_eq!(app.project_view.changes(), 1, "the rename alone");
        assert!(!field(&mut app, "topics.owning.subtopics")?.changed);
        set(&mut app, "project.name", "rust_pro")?;
        set(&mut app, "project.name", "rust_expert")?;
        assert!(!field(&mut app, "project.name")?.changed);
        assert_eq!(app.project_view.changes(), 1);
        set(&mut app, "topics.owning.name", "ownership")?;
        assert!(app.project_view.pending.is_none(), "back to the file");
        Ok(())
    }

    #[test]
    fn quitting_with_pending_changes_asks() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        set(&mut app, "project.name", "rust_pro")?;
        press(&mut app, &[KeyCode::Char('q')]);
        let Some(Overlay::Confirm(confirm)) = &app.overlay else {
            return Err("no dialog".into());
        };
        assert!(
            confirm
                .text
                .iter()
                .any(|line| line.contains("pending changes to overbrainer.toml")),
            "{confirm:?}"
        );
        assert_eq!(app.exit, None);
        Ok(())
    }

    #[test]
    fn a_change_that_undoes_itself_leaves_nothing_pending() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        select(&mut app, "roles.parent.reasoning")?;
        press(&mut app, &[KeyCode::Enter]);
        assert!(app.project_view.pending.is_some());
        press(&mut app, &[KeyCode::Enter]);
        assert!(app.project_view.pending.is_none());
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
    fn e_is_refused_with_pending_changes_or_while_a_stage_runs() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        set(&mut app, "project.name", "rust_pro")?;
        assert!(press(&mut app, &[KeyCode::Char('E')]).is_empty());
        assert_eq!(
            status(&app),
            "refused: save (s) or drop (u) the pending changes first"
        );
        app.drop_changes();
        app.pipeline_task = Some(TaskId(7));
        app.pipeline.started(Command::Split, 8);
        assert!(press(&mut app, &[KeyCode::Char('E')]).is_empty());
        assert!(
            status(&app).ends_with("; E edits the whole file"),
            "{}",
            status(&app)
        );
        Ok(())
    }

    #[test]
    fn a_save_is_refused_when_the_file_changed_on_disk() -> TestResult {
        let (dir, mut app) = editing_app()?;
        set(&mut app, "project.name", "rust_pro")?;
        std::fs::write(dir.path().join(CONFIG_FILE), "# changed elsewhere\n")?;
        save(&mut app, dir.path())?;
        assert_eq!(written(dir.path())?, "# changed elsewhere\n");
        assert!(
            status(&app).starts_with("not saved: overbrainer.toml changed on disk"),
            "{}",
            status(&app)
        );
        assert!(app.project_view.pending.is_some());
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
    fn the_form_and_the_marks_are_drawn() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        set(&mut app, "project.name", "rust_pro")?;
        select(&mut app, "topics.ownership.subtopics")?;
        press(&mut app, &[KeyCode::Enter]);
        chars(&mut app, "x");
        let rows = screen(&draw(&mut app, 80, 24)?).join("\n");
        assert!(rows.contains("configuration · 1 change"), "{rows}");
        assert!(rows.contains("rust_pro *"), "{rows}");
        assert!(rows.contains("subtopics = 2x"), "{rows}");
        assert!(rows.contains("at least 1"), "{rows}");
        assert!(rows.contains("Enter keep"), "{rows}");
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
        assert!(Addable::Topic.kinds().is_empty());
    }

    #[test]
    fn a_save_reloads_the_settings_the_app_uses() -> TestResult {
        let (dir, mut app) = editing_app()?;
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
        save(&mut app, dir.path())?;
        assert_eq!(status(&app), "✓ saved overbrainer.toml");
        assert!((app.project.eval_ratio - 0.2).abs() < f64::EPSILON);
        assert_eq!(app.project.concurrency, 4, "from the environment");
        assert_eq!(app.project.target.as_deref(), Some("box"));
        Ok(())
    }

    #[test]
    fn a_save_keeps_the_file_mode_and_refuses_a_symlink() -> TestResult {
        use std::os::unix::fs::PermissionsExt;
        let (dir, mut app) = editing_app()?;
        let path = dir.path().join(CONFIG_FILE);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640))?;
        set(&mut app, "project.name", "rust_pro")?;
        save(&mut app, dir.path())?;
        assert_eq!(status(&app), "✓ saved overbrainer.toml");
        let mode = |path: &Path| -> std::io::Result<u32> {
            Ok(std::fs::metadata(path)?.permissions().mode() & 0o7777)
        };
        assert_eq!(mode(&path)?, 0o640);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o4755))?;
        set(&mut app, "project.name", "rust_suid")?;
        save(&mut app, dir.path())?;
        assert_eq!(status(&app), "✓ saved overbrainer.toml");
        assert_eq!(mode(&path)?, 0o755, "never setuid, setgid or sticky");
        set(&mut app, "project.name", "rust_pro")?;
        save(&mut app, dir.path())?;
        let real = dir.path().join("real.toml");
        std::fs::rename(&path, &real)?;
        std::os::unix::fs::symlink(&real, &path)?;
        set(&mut app, "project.name", "rust_max")?;
        save(&mut app, dir.path())?;
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
    fn a_save_never_blocks_on_a_fifo() -> TestResult {
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
    fn quitting_or_a_signal_waits_for_a_save_then_notes_it() -> TestResult {
        for signal in [false, true] {
            let (dir, mut app) = editing_app()?;
            set(&mut app, "project.name", "rust_pro")?;
            let effects = press(&mut app, &[KeyCode::Char('s')]);
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
            assert_eq!(app.exit, None, "waits for the save");
            let saved = save_config(dir.path(), text, base, env).map(Box::new);
            app.on_done(*id, Ok(Done::ConfigSaved(saved)));
            assert!(app.exit.is_some());
            assert!(app.project_view.pending.is_none());
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
    fn e_is_refused_while_a_training_run_is_followed() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        let mut follow =
            crate::tui::training::Follow::new(crate::tui::training::Job::Attach, "20260921-a1");
        follow.watching = true;
        app.training.tasks.insert(TaskId(9), follow);
        assert!(press(&mut app, &[KeyCode::Char('E')]).is_empty());
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

    /// The picker open lists `entries` (and `gpus`), read by `id`.
    fn listed(app: &mut App, id: TaskId, entries: Vec<Entry>, gpus: Vec<GpuType>) {
        app.on_done(id, Ok(Done::Catalog(Ok(Listed { entries, gpus }))));
    }

    fn shown(app: &mut App, key: &str) -> Result<String, String> {
        Ok(field(app, key)?.shown.text().to_string())
    }

    fn picker_open(app: &App) -> bool {
        matches!(app.overlay, Some(Overlay::Picker(_)))
    }

    #[test]
    fn picking_gpus_writes_an_ordered_list_to_the_pending_document() -> TestResult {
        let (dir, mut app) = editing_app()?;
        let (id, query) = open_picker_on(&mut app, "targets.gpu_cloud.gpu_types")?;
        assert_eq!(
            query,
            Query {
                kind: CatalogKind::Gpus,
                gpu_count: 1,
                gpu_types: vec!["NVIDIA A40".into()],
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
        let gpus = field(&mut app, "targets.gpu_cloud.gpu_types")?;
        assert!(gpus.changed, "a pending change");
        assert_eq!(
            gpus.shown.text(),
            "NVIDIA RTX 2000 Ada Generation, NVIDIA A40"
        );
        assert_eq!(written(dir.path())?, PROJECT_CONFIG, "nothing written yet");
        save(&mut app, dir.path())?;
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
            }
        );
        Ok(())
    }

    #[test]
    fn no_gpu_type_chosen_changes_nothing() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.gpu_types")?;
        listed(&mut app, id, gpu_catalog(1)?, gpu_types()?);
        press(&mut app, &[KeyCode::Char(' '), KeyCode::Enter]);
        assert!(app.project_view.pending.is_none());
        assert_eq!(
            status(&app),
            "targets.gpu_cloud.gpu_types: choose a GPU type or auto; nothing changed"
        );
        Ok(())
    }

    fn volumes() -> Vec<Entry> {
        volume_entries(&[
            NetworkVolume {
                id: "vol-eu".into(),
                name: "alpha".into(),
                size: 100,
                data_center: "EU-RO-1".into(),
            },
            NetworkVolume {
                id: "vol-us".into(),
                name: "zeta".into(),
                size: 50,
                data_center: "US-KS-2".into(),
            },
        ])
    }

    #[test]
    fn picking_a_volume_sets_it_and_its_single_data_center() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        let centers = FieldPath::Target {
            name: "gpu_cloud".into(),
            field: "data_center_ids",
        };
        let two = FieldValue::List(vec!["US-KS-2".into(), "CA-MTL-1".into()]);
        app.apply(&centers, Some(&two))?;
        let (id, query) = open_picker_on(&mut app, "targets.gpu_cloud.network_volume_id")?;
        assert_eq!(query.kind, CatalogKind::Volumes);
        listed(&mut app, id, volumes(), Vec::new());
        // none, alpha, zeta: alpha is picked.
        press(&mut app, &[KeyCode::Down, KeyCode::Enter]);
        assert_eq!(
            shown(&mut app, "targets.gpu_cloud.network_volume_id")?,
            "vol-eu"
        );
        assert_eq!(
            shown(&mut app, "targets.gpu_cloud.data_center_ids")?,
            "EU-RO-1"
        );
        assert_eq!(
            status(&app),
            "targets.gpu_cloud.data_center_ids = EU-RO-1, the volume's data center"
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
    fn picking_a_template_sets_its_image_and_esc_keeps_the_value() -> TestResult {
        let (_dir, mut app) = editing_app()?;
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
        assert!(app.project_view.pending.is_none(), "Esc keeps the image");
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.image")?;
        listed(&mut app, id, templates, Vec::new());
        press(&mut app, &[KeyCode::End, KeyCode::Enter]);
        assert_eq!(shown(&mut app, "targets.gpu_cloud.image")?, "img/trainer:2");
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
        assert!(press(&mut app, &[KeyCode::Enter]).is_empty());
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
        assert!(press(&mut app, &[KeyCode::Enter]).is_empty());
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
        save(&mut app, dir.path())?;
        let text = written(dir.path())?;
        assert!(text.contains(r#"gpu_types = "auto""#), "{text}");
        Ok(())
    }

    #[test]
    fn auto_picked_is_saved_as_a_string() -> TestResult {
        let (dir, mut app) = editing_app()?;
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.gpu_types")?;
        listed(&mut app, id, gpu_catalog(1)?, gpu_types()?);
        press(
            &mut app,
            &[KeyCode::Home, KeyCode::Char(' '), KeyCode::Enter],
        );
        save(&mut app, dir.path())?;
        assert_eq!(status(&app), "✓ saved overbrainer.toml");
        let text = written(dir.path())?;
        assert!(text.contains(r#"gpu_types = "auto""#), "{text}");
        Ok(())
    }

    #[test]
    fn the_default_row_of_the_template_picker_unsets_the_image() -> TestResult {
        let (_dir, mut app) = editing_app()?;
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
        assert!(app.project_view.pending.is_none(), "back to the file");
        Ok(())
    }

    #[test]
    fn a_lock_taken_while_the_picker_is_open_refuses_the_choice() -> TestResult {
        let (_dir, mut app) = editing_app()?;
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
        assert!(app.project_view.pending.is_none());
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
        assert!(press(&mut app, &[KeyCode::Enter]).is_empty());
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
        let path = FieldPath::Target {
            name: "gpu_cloud".into(),
            field: "network_volume_id",
        };
        app.apply(&path, Some(&FieldValue::Text("vol-gone".into())))?;
        let (id, _) = open_picker_on(&mut app, "targets.gpu_cloud.network_volume_id")?;
        listed(&mut app, id, volumes(), Vec::new());
        // none, alpha, zeta, then vol-gone, kept in view: picked again.
        press(&mut app, &[KeyCode::End, KeyCode::Enter]);
        assert_eq!(
            shown(&mut app, "targets.gpu_cloud.network_volume_id")?,
            "vol-gone"
        );
        assert_eq!(
            field(&mut app, "targets.gpu_cloud.data_center_ids")?.shown,
            Shown::Unset
        );
        Ok(())
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
