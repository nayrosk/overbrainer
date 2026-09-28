//! Editing the configuration from the Project view: the keys, the form under
//! the list, the confirmations, the save and the read after `$EDITOR`. Changes
//! stay in a pending document until `s` validates and writes it.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;
use std::process::ExitStatus;
use std::sync::Arc;

use crossterm::event::KeyCode;

use super::app::{Action, App, Confirm, Effect, Overlay, PAGE, Project, Severity, View};
use super::project::{Addable, Form, Listing, Locks, Pending, ProjectConfig, Shown, env_var};
use super::widgets::form::{Input, InputOutcome};
use crate::config::edit::{Collection, ConfigDoc, FieldPath};
use crate::config::fields::{FieldKind, FieldValue, TargetKind};
use crate::config::{CONFIG_FILE, ConfigError, EnvSource, Protocol, is_valid_name};

/// What a confirmed `d` takes out of the pending document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Removal {
    /// The topic of this name.
    Topic(String),
    /// A provider or a target.
    Table(Collection, String),
}

impl Removal {
    /// The dotted key of what goes: `topics.traits`, `providers.claude`.
    pub(super) fn key(&self) -> String {
        match self {
            Self::Topic(name) => format!("topics.{name}"),
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
/// `base`, the text it was read from. Returns the configuration written.
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
    let path = dir.join(CONFIG_FILE);
    let on_disk = std::fs::read_to_string(&path)
        .map_err(|error| SaveRefusal::Failed(format!("cannot read {}: {error}", path.display())))?;
    if on_disk != base {
        return Err(SaveRefusal::Failed(format!(
            "{CONFIG_FILE} changed on disk since it was read; drop the changes (u), then E"
        )));
    }
    crate::runs::write_atomic(dir, CONFIG_FILE, text.as_bytes())
        .map_err(|error| SaveRefusal::Failed(error.to_string()))?;
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
            KeyCode::Enter => self.edit_field(),
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
    fn refuse_change(&mut self) -> bool {
        if self.config.is_none() {
            return true;
        }
        if self.project_view.saving {
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
    /// for `None`, and marks it changed.
    fn apply(&mut self, path: &FieldPath, value: Option<&FieldValue>) -> Result<(), String> {
        let key = path.to_string();
        let pending = self
            .pending_mut()
            .ok_or_else(|| "no configuration read".to_string())?;
        let applied = match &value {
            Some(value) => pending.doc.set(path, (*value).clone()),
            None => pending.doc.unset(path).map(drop),
        };
        if let Err(error) = applied {
            self.settle_pending();
            return Err(error.to_string());
        }
        pending.changed.insert(key.clone());
        // A renamed topic is a table of its own for the marks.
        if let (FieldPath::Topic { field: "name", .. }, Some(FieldValue::Text(name))) =
            (path, value)
        {
            pending.changed.insert(format!("topics.{name}"));
        }
        self.project_view.errors.remove(&key);
        self.settle_pending();
        Ok(())
    }

    /// Enter: toggles a bool, cycles a choice (to unset after the last, when
    /// the field may be left out), or opens the form on the value; refused on
    /// a locked, env-only or env-set field.
    fn edit_field(&mut self) {
        if self.refuse_change() {
            return;
        }
        let listing = self.project_listing();
        let Some(field) = listing.field(self.project_view.selected) else {
            return;
        };
        let key = &field.key;
        if let Some(user) = &field.lock {
            let said = format!("refused: {key} is used by {user}; read-only until it ends");
            self.say(Severity::Warn, said);
            return;
        }
        let Some(path) = field.path.clone() else {
            let said = format!("{key}: env only, set {} in .env", env_var(key));
            self.say(Severity::Warn, said);
            return;
        };
        if field.env {
            let said = format!("{key} is set by {}; change it in .env", env_var(key));
            self.say(Severity::Warn, said);
            return;
        }
        let Some(spec) = self.shown_doc().and_then(|doc| doc.spec(&path)) else {
            let said = format!("{key}: not edited here; E opens {CONFIG_FILE}");
            self.say(Severity::Warn, said);
            return;
        };
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
                return;
            },
        };
        if let Err(error) = self.apply(&path, value.as_ref()) {
            self.say(Severity::Warn, error);
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
            (false, _) => Some(kind.parse(text)?),
        };
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

    /// Enter on the name of what is added: a topic is added at once, a
    /// provider or a target asks its protocol or kind next.
    fn named(&mut self, what: Addable, name: &str) -> Result<Option<Form>, String> {
        let Some(doc) = self.shown_doc() else {
            return Ok(None);
        };
        if name.is_empty() {
            return Err("a name is needed".to_string());
        }
        let (exists, key) = match what {
            Addable::Topic => (
                doc.topic_names().iter().any(|topic| topic == name),
                format!("topics.{name}"),
            ),
            Addable::Provider | Addable::Target => {
                if !is_valid_name(name) {
                    return Err("the name must match ^[a-z0-9_]+$".to_string());
                }
                let collection = collection(what);
                (
                    doc.names(collection).iter().any(|table| table == name),
                    format!("{}.{name}", collection.key()),
                )
            },
        };
        if exists {
            return Err(format!("{key} already exists"));
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
        let Some(pending) = self.pending_mut() else {
            return;
        };
        let added = match what {
            Addable::Topic => pending.doc.add_topic(name).map(drop),
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
        let key = match what {
            Addable::Topic => format!("topics.{name}"),
            Addable::Provider | Addable::Target => format!("{}.{name}", collection(what).key()),
        };
        match added {
            Ok(()) => {
                pending.changed.insert(key.clone());
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
            Some(FieldPath::Topic { name, .. }) => Some(Removal::Topic(name.clone())),
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
            Removal::Topic(name) => pending
                .doc
                .topic_names()
                .iter()
                .position(|topic| topic == name)
                .is_some_and(|index| pending.doc.remove_topic(index)),
            Removal::Table(collection, name) => pending.doc.remove_table(*collection, name),
        };
        if removed {
            pending.changed.insert(key.clone());
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
        let effect = Effect::SaveConfig {
            text: pending.doc.text(),
            base: config.text.clone(),
        };
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
        self.project_view.saving = true;
        vec![effect]
    }

    /// The save ended: the app reads the configuration written, or each
    /// problem is shown on the field it names and nothing changed.
    pub(super) fn config_saved(
        &mut self,
        saved: Result<Box<ProjectConfig>, SaveRefusal>,
    ) -> Vec<Effect> {
        self.project_view.saving = false;
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
            },
            Err(SaveRefusal::Invalid(problems)) => {
                self.mark_errors(&problems);
                let first = problems.first().cloned().unwrap_or_default();
                let more = match problems.len() {
                    0 | 1 => String::new(),
                    count => format!(" (+{} more)", count - 1),
                };
                self.say(Severity::Error, format!("not saved: {first}{more}"));
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

    /// Whether leaving the Project view for `view` must ask first, because of
    /// pending changes: then it asks.
    pub(super) fn ask_leave(&mut self, view: View) -> bool {
        if self.view != View::Project || view == View::Project {
            return false;
        }
        let count = self.project_view.changes();
        if self.project_view.pending.is_none() {
            return false;
        }
        self.overlay = Some(Overlay::Confirm(Confirm {
            title: " Leave the Project view? ".to_string(),
            text: vec![format!(
                "{} to {CONFIG_FILE} not saved: leaving drops {}. s saves them first.",
                changes(count),
                if count == 1 { "it" } else { "them" },
            )],
            yes: "leave",
            no: "stay",
            action: Action::Leave(view),
        }));
        true
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
        if let Some((run, _)) = Locks::of(self).run {
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
        let failed = match status {
            Ok(status) if status.success() => None,
            Ok(status) => Some(status.code().map_or_else(
                || "the editor was killed".to_string(),
                |code| format!("the editor exited with status {code}"),
            )),
            Err(error) => Some(format!("cannot run the editor: {error}")),
        };
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
        self.set_config(config);
        self.project_view.errors.clear();
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

/// The collection a provider or a target is added to.
fn collection(what: Addable) -> Collection {
    match what {
        Addable::Target => Collection::Targets,
        Addable::Topic | Addable::Provider => Collection::Providers,
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::process::ExitStatusExt;

    use super::*;
    use crate::cli::data::Command;
    use crate::tui::snapshots::{
        PROJECT_CONFIG, draw, key, project_app, project_env, text as screen,
    };
    use crate::tui::tasks::{Msg, Task, TaskId};

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

    /// `s`, then the save the loop would run, in `dir`, handed back to `app`.
    fn save(app: &mut App, dir: &Path) -> Result<Vec<Effect>, String> {
        let effects = press(app, &[KeyCode::Char('s')]);
        let [Effect::SaveConfig { text, base }] = effects.as_slice() else {
            return Err(format!("no save: {effects:?} ({})", status(app)));
        };
        assert!(app.project_view.saving);
        let saved = save_config(dir, text, base, &project_env()).map(Box::new);
        Ok(app.on_message(Msg::ConfigSaved(saved)))
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
        assert!(!app.project_view.saving);
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
        assert_eq!(marked.detail(), marked.error.clone().unwrap_or_default());
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
            "pipeline.concurrency is set by OVERBRAINER_PIPELINE__CONCURRENCY; \
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
    fn leaving_the_view_with_pending_changes_asks_and_u_drops_them() -> TestResult {
        let (_dir, mut app) = editing_app()?;
        set(&mut app, "project.name", "rust_pro")?;
        press(&mut app, &[KeyCode::Char('2')]);
        assert_eq!(app.view, View::Project);
        assert!(matches!(
            &app.overlay,
            Some(Overlay::Confirm(Confirm {
                action: Action::Leave(View::Dataset),
                ..
            }))
        ));
        press(&mut app, &[KeyCode::Char('n')]);
        assert_eq!(app.view, View::Project);
        assert!(app.project_view.pending.is_some());
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
        set(&mut app, "project.name", "rust_pro")?;
        press(&mut app, &[KeyCode::Tab, KeyCode::Char('y')]);
        assert_eq!(app.view, View::Dataset);
        assert!(app.project_view.pending.is_none(), "leaving dropped them");
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
