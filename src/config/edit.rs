//! Edits `overbrainer.toml` field by field, keeping its comments, key order and layout.
//!
//! [`ConfigDoc`] wraps a `toml_edit` document. Values are set as typed TOML values
//! ([`FieldValue`]), never as TOML text, and only for fields listed in [`fields`], so
//! env-only keys are never written. The caller validates the whole text with
//! [`load_str`](super::load_str) before saving it.

use std::fmt;

use toml_edit::{
    Array, ArrayOfTables, Decor, DocumentMut, InlineTable, Item, RawString, Table, TableLike,
    Value, value,
};

use super::fields::{self, FieldError, FieldSpec, FieldValue, Section, TargetKind};
use super::types::Protocol;
use super::validate::is_valid_name;

/// One of the pipeline roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// `roles.generator`.
    Generator,
    /// `roles.parent`.
    Parent,
    /// `roles.embedder`.
    Embedder,
}

/// A named table collection: `[providers.<name>]` or `[targets.<name>]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Collection {
    /// `[providers.<name>]`.
    Providers,
    /// `[targets.<name>]`.
    Targets,
}

/// Where a field lives. Displays as the dotted path `validate` names it with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldPath {
    /// `project.<field>`.
    Project(&'static str),
    /// `topics.<name>.<field>`: the `index`-th `[[topics]]` entry, named `name`.
    Topic {
        /// Position in the `[[topics]]` array.
        index: usize,
        /// Name of the topic, as `validate` names it.
        name: String,
        /// The key.
        field: &'static str,
    },
    /// `providers.<name>.<field>`.
    Provider {
        /// Name of the provider.
        name: String,
        /// The key.
        field: &'static str,
    },
    /// `roles.<role>.<field>`.
    Role {
        /// The role.
        role: Role,
        /// The key.
        field: &'static str,
    },
    /// `pipeline.<field>`.
    Pipeline(&'static str),
    /// `training.<field>`.
    Training(&'static str),
    /// `targets.<name>.<field>`.
    Target {
        /// Name of the target.
        name: String,
        /// The key.
        field: &'static str,
    },
}

/// Why an edit was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EditError {
    /// The text is not valid TOML. Only the location is kept, never the text.
    #[error("TOML syntax error in overbrainer.toml at line {line}, column {column}")]
    Syntax {
        /// 1-based line.
        line: usize,
        /// 1-based column, in characters.
        column: usize,
    },
    /// The table a path points into is not in the document.
    #[error("{0}: not found in overbrainer.toml")]
    Missing(String),
    /// The path names a key the form does not edit.
    #[error("{0}: not an editable field")]
    NotEditable(String),
    /// The value does not fit the field. The reason never quotes the value.
    #[error("{path}: {reason}")]
    Invalid {
        /// Dotted path of the field.
        path: String,
        /// Why the value is refused.
        reason: FieldError,
    },
    /// A table or topic with this name already exists.
    #[error("{0}: already exists")]
    Exists(String),
    /// A provider or target name outside `^[a-z0-9_]+$`.
    #[error("{0}: name must match ^[a-z0-9_]+$")]
    InvalidName(String),
}

/// An `overbrainer.toml` document being edited.
#[derive(Debug, Clone)]
pub struct ConfigDoc {
    doc: DocumentMut,
}

impl Role {
    /// Every role, in file order.
    pub const ALL: [Self; 3] = [Self::Generator, Self::Parent, Self::Embedder];

    /// The key under `[roles]`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Generator => "generator",
            Self::Parent => "parent",
            Self::Embedder => "embedder",
        }
    }
}

impl Collection {
    /// The top-level key: `providers` or `targets`.
    #[must_use]
    pub fn key(self) -> &'static str {
        match self {
            Self::Providers => "providers",
            Self::Targets => "targets",
        }
    }
}

impl FieldPath {
    /// The key of the field in its table.
    #[must_use]
    pub fn field(&self) -> &'static str {
        match self {
            Self::Project(field)
            | Self::Pipeline(field)
            | Self::Training(field)
            | Self::Topic { field, .. }
            | Self::Provider { field, .. }
            | Self::Role { field, .. }
            | Self::Target { field, .. } => field,
        }
    }

    /// The dotted path of the table holding the field, as `validate` names it.
    #[must_use]
    pub fn table(&self) -> String {
        match self {
            Self::Project(_) => "project".to_string(),
            Self::Topic { name, .. } => format!("topics.{name}"),
            Self::Provider { name, .. } => format!("providers.{name}"),
            Self::Role { role, .. } => format!("roles.{}", role.as_str()),
            Self::Pipeline(_) => "pipeline".to_string(),
            Self::Training(_) => "training".to_string(),
            Self::Target { name, .. } => format!("targets.{name}"),
        }
    }
}

impl fmt::Display for FieldPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}", self.table(), self.field())
    }
}

/// `subtopics` of a topic added by [`ConfigDoc::add_topic`].
const NEW_TOPIC_SUBTOPICS: i64 = 10;
/// `questions_per_subtopic` of a topic added by [`ConfigDoc::add_topic`].
const NEW_TOPIC_QUESTIONS: i64 = 30;
/// `max_hours` of a runpod target added by [`ConfigDoc::add_target`]: a safe cap.
const NEW_RUNPOD_MAX_HOURS: f64 = 6.0;
impl ConfigDoc {
    /// Parses the text of `overbrainer.toml`.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::Syntax`] with the location of the first TOML error.
    pub fn parse(text: &str) -> Result<Self, EditError> {
        text.parse::<DocumentMut>()
            .map(|doc| Self { doc })
            .map_err(|error| syntax(text, error.span().map_or(0, |span| span.start)))
    }

    /// The document as text, comments and layout included.
    #[must_use]
    pub fn text(&self) -> String {
        self.doc.to_string()
    }

    /// The value of an editable field in display form (strings unquoted, lists
    /// comma-separated), or `None` when it is absent or not editable.
    #[must_use]
    pub fn get(&self, path: &FieldPath) -> Option<String> {
        self.spec(path)?;
        let value = self.entry(path)?.get(path.field())?.as_value()?;
        Some(display(value))
    }

    /// The spec of the field `path` names, when it is editable. A target's fields
    /// depend on its `kind`, read from the document.
    #[must_use]
    pub fn spec(&self, path: &FieldPath) -> Option<&'static FieldSpec> {
        self.spec_of(path).ok()
    }

    /// Sets a field to a typed value, keeping the comment that follows the old value.
    /// An absent key is added at the end of its table; an absent `[project]`,
    /// `[pipeline]`, `[training]`, `[roles]` or role is created.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::NotEditable`] for a field outside [`fields`],
    /// [`EditError::Invalid`] when the value does not fit the field's kind, and
    /// [`EditError::Missing`] when the topic, provider or target does not exist.
    pub fn set(&mut self, path: &FieldPath, value: FieldValue) -> Result<(), EditError> {
        let spec = self.spec_of(path)?;
        spec.kind
            .check(&value)
            .map_err(|reason| EditError::Invalid {
                path: path.to_string(),
                reason,
            })?;
        self.raw_set(path, value)
    }

    /// Removes an optional field. Returns whether it was present. Comment lines
    /// above the key move to the next key, or the next table, or the end.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::NotEditable`] for a field outside [`fields`] and
    /// [`EditError::Invalid`] for a required field.
    pub fn unset(&mut self, path: &FieldPath) -> Result<bool, EditError> {
        match self.spec_of(path) {
            Ok(spec) if !spec.optional => {
                return Err(EditError::Invalid {
                    path: path.to_string(),
                    reason: FieldError::Required,
                });
            },
            Ok(_) => {},
            Err(EditError::Missing(_)) => return Ok(false),
            Err(error) => return Err(error),
        }
        let field = path.field();
        let Some(entry) = self.entry_mut(path) else {
            return Ok(false);
        };
        if let Item::Value(Value::InlineTable(table)) = entry {
            let last = table.iter().last().is_some_and(|(key, _)| key == field);
            let Some(removed) = table.remove(field) else {
                return Ok(false);
            };
            // The space before `}` sat in the removed value's suffix: hand it back.
            if last
                && let Some((_, value)) = table.iter_mut().last()
                && let Some(suffix) = removed.decor().suffix()
            {
                value.decor_mut().set_suffix(suffix.clone());
            }
            return Ok(true);
        }
        let Item::Table(table) = entry else {
            return Ok(entry
                .as_table_like_mut()
                .is_some_and(|table| table.remove(field).is_some()));
        };
        let position = table.position();
        let moved = table
            .key(field)
            .map(|key| comment_lines(key.leaf_decor()))
            .unwrap_or_default();
        let next = table
            .iter()
            .skip_while(|(key, _)| *key != field)
            .skip(1)
            .find(|(_, item)| item.is_value())
            .map(|(key, _)| key.to_string());
        if table.remove(field).is_none() {
            return Ok(false);
        }
        if let Some(mut key) = next.as_deref().and_then(|next| table.key_mut(next)) {
            let prefix = raw(key.leaf_decor().prefix());
            key.leaf_decor_mut().set_prefix(format!("{moved}{prefix}"));
        } else {
            self.rehome(position, &moved);
        }
        Ok(true)
    }

    /// Names of the `[[topics]]` entries, in order (empty for an entry without one).
    #[must_use]
    pub fn topic_names(&self) -> Vec<String> {
        let Some(topics) = self.doc.get("topics") else {
            return Vec::new();
        };
        let tables: Vec<&dyn TableLike> = match topics {
            Item::ArrayOfTables(array) => array.iter().map(|t| t as &dyn TableLike).collect(),
            Item::Value(Value::Array(array)) => array
                .iter()
                .filter_map(|value| value.as_inline_table().map(|t| t as &dyn TableLike))
                .collect(),
            _ => Vec::new(),
        };
        tables
            .into_iter()
            .map(|table| {
                table
                    .get("name")
                    .and_then(Item::as_str)
                    .unwrap_or_default()
                    .to_string()
            })
            .collect()
    }

    /// Appends a `[[topics]]` entry named `name`, with starting counts. Returns its index.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::Exists`] when a topic has this name, and
    /// [`EditError::Missing`] when `topics` is not an array.
    pub fn add_topic(&mut self, name: &str) -> Result<usize, EditError> {
        if self.topic_names().iter().any(|topic| topic == name) {
            return Err(EditError::Exists(format!("topics.{name}")));
        }
        let mut table = Table::new();
        table.insert("name", value(name));
        table.insert("subtopics", value(NEW_TOPIC_SUBTOPICS));
        table.insert("questions_per_subtopic", value(NEW_TOPIC_QUESTIONS));
        let root = self.doc.as_table_mut();
        match root.get_mut("topics") {
            None => {
                let mut array = ArrayOfTables::new();
                array.push(table);
                root.insert("topics", Item::ArrayOfTables(array));
                self.renumber();
                Ok(0)
            },
            Some(Item::ArrayOfTables(array)) => {
                array.push(table);
                let index = array.len() - 1;
                self.renumber();
                Ok(index)
            },
            Some(Item::Value(Value::Array(array))) => {
                array.push(table.into_inline_table());
                Ok(array.len() - 1)
            },
            Some(_) => Err(EditError::Missing("topics".to_string())),
        }
    }

    /// Removes the `index`-th topic. Returns whether it existed.
    pub fn remove_topic(&mut self, index: usize) -> bool {
        let root = self.doc.as_table_mut();
        let (remaining, removed) = match root.get_mut("topics") {
            Some(Item::ArrayOfTables(array)) if index < array.len() => {
                let removed = array.remove(index);
                let comments = comment_lines(removed.decor());
                if let Some(next) = array.get_mut(index) {
                    prepend_comments(next, &comments);
                    return true;
                }
                (array.len(), Some(removed))
            },
            Some(Item::Value(Value::Array(array))) if index < array.len() => {
                array.remove(index);
                (array.len(), None)
            },
            _ => return false,
        };
        if remaining == 0 {
            root.remove("topics");
        }
        if let Some(removed) = removed {
            self.rehome(removed.position(), &comment_lines(removed.decor()));
        }
        true
    }

    /// Names of the providers or targets, in file order.
    #[must_use]
    pub fn names(&self, collection: Collection) -> Vec<String> {
        self.doc
            .get(collection.key())
            .and_then(Item::as_table_like)
            .map(|table| table.iter().map(|(key, _)| key.to_string()).collect())
            .unwrap_or_default()
    }

    /// The `kind` of the target `name`, when it exists and the kind is known.
    #[must_use]
    pub fn target_kind(&self, name: &str) -> Option<TargetKind> {
        let kind = self.doc.get("targets")?.get(name)?.get("kind")?.as_str()?;
        TargetKind::from_name(kind)
    }

    /// Adds `[providers.<name>]` after the other providers.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::InvalidName`] for a name outside `^[a-z0-9_]+$` and
    /// [`EditError::Exists`] when the provider exists.
    pub fn add_provider(&mut self, name: &str, protocol: Protocol) -> Result<(), EditError> {
        let mut table = Table::new();
        table.insert("protocol", value(protocol_name(protocol)));
        self.add_named(Collection::Providers, name, table)
    }

    /// Adds `[targets.<name>]` of `kind` after the other targets: `runtime = "docker"`
    /// for `local` and `ssh`; for `runpod`, an empty `gpu_types` to fill, a 6-hour
    /// `max_hours` and the defaults of the other counts.
    ///
    /// # Errors
    ///
    /// Returns [`EditError::InvalidName`] for a name outside `^[a-z0-9_]+$` and
    /// [`EditError::Exists`] when the target exists.
    pub fn add_target(&mut self, name: &str, kind: TargetKind) -> Result<(), EditError> {
        let mut table = Table::new();
        table.insert("kind", value(kind.as_str()));
        match kind {
            TargetKind::Local | TargetKind::Ssh => {
                table.insert("runtime", value("docker"));
            },
            TargetKind::Runpod => {
                // The serde defaults of `types.rs`, written so the file shows them.
                table.insert("gpu_types", value(Array::new()));
                table.insert("gpu_count", value(1));
                table.insert("container_disk_gb", value(50));
                table.insert("max_hours", value(NEW_RUNPOD_MAX_HOURS));
                table.insert("boot_grace_minutes", value(30));
                table.insert("retrieve_grace_minutes", value(60));
            },
        }
        self.add_named(Collection::Targets, name, table)
    }

    /// Removes `[providers.<name>]` or `[targets.<name>]`. Returns whether it existed.
    pub fn remove_table(&mut self, collection: Collection, name: &str) -> bool {
        let root = self.doc.as_table_mut();
        let Some(tables) = root
            .get_mut(collection.key())
            .and_then(Item::as_table_like_mut)
        else {
            return false;
        };
        let next = tables
            .iter()
            .skip_while(|(key, _)| *key != name)
            .skip(1)
            .find(|(_, item)| item.is_table())
            .map(|(key, _)| key.to_string());
        let Some(removed) = tables.remove(name) else {
            return false;
        };
        let Item::Table(removed) = removed else {
            if tables.is_empty() {
                root.remove(collection.key());
            }
            return true;
        };
        let comments = comment_lines(removed.decor());
        if let Some(next) = next
            .as_deref()
            .and_then(|next| tables.get_mut(next))
            .and_then(Item::as_table_mut)
        {
            prepend_comments(next, &comments);
            return true;
        }
        if tables.is_empty() {
            root.remove(collection.key());
        }
        self.rehome(removed.position(), &comments);
        true
    }

    /// Puts the comment lines of a removed item before the first table after
    /// document position `after`, or at the end of the document.
    fn rehome(&mut self, after: Option<isize>, comments: &str) {
        if comments.is_empty() {
            return;
        }
        let next = after.and_then(|after| {
            self.tables()
                .into_iter()
                .filter(|table| table.headed && table.position > after)
                .min_by_key(|table| table.position)
        });
        if let Some(table) = next.and_then(|table| self.table_at(&table.steps)) {
            prepend_comments(table, comments);
        } else {
            let trailing = raw(Some(self.doc.trailing()));
            self.doc.set_trailing(format!("{trailing}\n{comments}"));
        }
    }

    /// Every table below the root, in the order `toml_edit` visits them.
    fn tables(&self) -> Vec<Located> {
        let mut found = Vec::new();
        tables_under(self.doc.as_table(), &mut Vec::new(), &mut 0, &mut found);
        found
    }

    /// Gives every table its printed rank as position, so a table added here gets
    /// one too and the tables are printed in the same order.
    fn renumber(&mut self) {
        let mut tables = self.tables();
        tables.sort_by_key(|table| table.position);
        for (rank, located) in (1..).zip(tables) {
            if let Some(table) = self.table_at(&located.steps) {
                table.set_position(Some(rank));
            }
        }
    }

    /// The table at `steps` from the root.
    fn table_at(&mut self, steps: &[Step]) -> Option<&mut Table> {
        let mut item = self.doc.as_item_mut();
        for step in steps {
            item = match step {
                Step::Key(key) => item.get_mut(key.as_str())?,
                Step::Index(index) => item.get_mut(*index)?,
            };
        }
        item.as_table_mut()
    }

    fn add_named(
        &mut self,
        collection: Collection,
        name: &str,
        table: Table,
    ) -> Result<(), EditError> {
        if !is_valid_name(name) {
            return Err(EditError::InvalidName(name.to_string()));
        }
        let key = collection.key();
        let root = self.doc.as_table_mut();
        if !root.contains_key(key) {
            let mut parent = Table::new();
            parent.set_implicit(true);
            root.insert(key, Item::Table(parent));
        }
        let tables = root
            .get_mut(key)
            .and_then(Item::as_table_like_mut)
            .ok_or_else(|| EditError::Missing(key.to_string()))?;
        if tables.contains_key(name) {
            return Err(EditError::Exists(format!("{key}.{name}")));
        }
        tables.insert(name, Item::Table(table));
        self.renumber();
        Ok(())
    }

    fn spec_of(&self, path: &FieldPath) -> Result<&'static FieldSpec, EditError> {
        let section = match path {
            FieldPath::Project(_) => Section::Project,
            FieldPath::Topic { .. } => Section::Topic,
            FieldPath::Provider { .. } => Section::Provider,
            FieldPath::Role { .. } => Section::Role,
            FieldPath::Pipeline(_) => Section::Pipeline,
            FieldPath::Training(_) => Section::Training,
            FieldPath::Target { name, .. } => {
                if self.entry(path).is_none() {
                    return Err(EditError::Missing(format!("targets.{name}")));
                }
                self.target_kind(name)
                    .map(Section::Target)
                    .ok_or_else(|| EditError::NotEditable(path.to_string()))?
            },
        };
        fields::find(section, path.field()).ok_or_else(|| EditError::NotEditable(path.to_string()))
    }

    /// Sets a field without checking it against its spec.
    fn raw_set(&mut self, path: &FieldPath, value: FieldValue) -> Result<(), EditError> {
        let missing = || EditError::Missing(path.table());
        let created = self.entry(path).is_none();
        let entry = self.entry_or_create(path).ok_or_else(missing)?;
        let mut new = toml_value(value);
        if let Item::Value(Value::InlineTable(table)) = entry
            && !table.contains_key(path.field())
        {
            // The space before `}` sits in the last value's suffix: hand it over.
            if let Some((_, last)) = table.iter_mut().last() {
                let suffix = last.decor().suffix().cloned();
                last.decor_mut().set_suffix("");
                if let Some(suffix) = suffix {
                    new.decor_mut().set_suffix(suffix);
                }
            }
            table.insert(path.field(), new);
            return Ok(());
        }
        let table = entry.as_table_like_mut().ok_or_else(missing)?;
        match table.get_mut(path.field()) {
            Some(item) => {
                if let Some(old) = item.as_value() {
                    *new.decor_mut() = old.decor().clone();
                }
                *item = Item::Value(new);
            },
            None => {
                table.insert(path.field(), Item::Value(new));
            },
        }
        if created {
            self.renumber();
        }
        Ok(())
    }

    /// The item of the table holding `path`'s field, when it exists.
    fn entry(&self, path: &FieldPath) -> Option<&Item> {
        let root = self.doc.as_item();
        match path {
            FieldPath::Project(_) => root.get("project"),
            FieldPath::Topic { index, name, .. } => root
                .get("topics")?
                .get(*index)
                .filter(|topic| topic.get("name").and_then(Item::as_str) == Some(name)),
            FieldPath::Provider { name, .. } => root.get("providers")?.get(name.as_str()),
            FieldPath::Role { role, .. } => root.get("roles")?.get(role.as_str()),
            FieldPath::Pipeline(_) => root.get("pipeline"),
            FieldPath::Training(_) => root.get("training"),
            FieldPath::Target { name, .. } => root.get("targets")?.get(name.as_str()),
        }
    }

    /// The item of the table holding `path`'s field, when it exists.
    fn entry_mut(&mut self, path: &FieldPath) -> Option<&mut Item> {
        let root = self.doc.as_item_mut();
        match path {
            FieldPath::Project(_) => root.get_mut("project"),
            FieldPath::Topic { index, name, .. } => root
                .get_mut("topics")?
                .get_mut(*index)
                .filter(|topic| topic.get("name").and_then(Item::as_str) == Some(name)),
            FieldPath::Provider { name, .. } => root.get_mut("providers")?.get_mut(name.as_str()),
            FieldPath::Role { role, .. } => root.get_mut("roles")?.get_mut(role.as_str()),
            FieldPath::Pipeline(_) => root.get_mut("pipeline"),
            FieldPath::Training(_) => root.get_mut("training"),
            FieldPath::Target { name, .. } => root.get_mut("targets")?.get_mut(name.as_str()),
        }
    }

    /// The item of the table holding `path`'s field, creating a missing singleton
    /// section or role.
    fn entry_or_create(&mut self, path: &FieldPath) -> Option<&mut Item> {
        if matches!(
            path,
            FieldPath::Topic { .. } | FieldPath::Provider { .. } | FieldPath::Target { .. }
        ) {
            return self.entry_mut(path);
        }
        let root = self.doc.as_table_mut();
        match path {
            FieldPath::Project(_) => ensure_table(root, "project"),
            FieldPath::Pipeline(_) => ensure_table(root, "pipeline"),
            FieldPath::Training(_) => ensure_table(root, "training"),
            FieldPath::Role { role, .. } => {
                let roles = ensure_table(root, "roles")?.as_table_like_mut()?;
                let key = role.as_str();
                if !roles.contains_key(key) {
                    // Follow the file's style: inline roles stay inline.
                    let inline = roles.iter().any(|(_, item)| item.is_inline_table());
                    let entry = if inline {
                        Item::Value(Value::InlineTable(InlineTable::new()))
                    } else {
                        Item::Table(Table::new())
                    };
                    roles.insert(key, entry);
                }
                roles.get_mut(key)
            },
            FieldPath::Topic { .. } | FieldPath::Provider { .. } | FieldPath::Target { .. } => None,
        }
    }
}

/// The table under `key` in `root`, created when absent.
fn ensure_table<'a>(root: &'a mut Table, key: &str) -> Option<&'a mut Item> {
    if !root.contains_key(key) {
        root.insert(key, Item::Table(Table::new()));
    }
    root.get_mut(key)
}

/// A syntax error at byte `offset` of `text`, as a 1-based line and column.
fn syntax(text: &str, offset: usize) -> EditError {
    let mut end = offset.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let before = text.get(..end).unwrap_or_default();
    let line_start = before.rfind('\n').map_or(0, |index| index + 1);
    EditError::Syntax {
        line: before.matches('\n').count() + 1,
        column: before[line_start..].chars().count() + 1,
    }
}

/// One step from the root to a table: a key, or an index in an array of tables.
#[derive(Debug, Clone)]
enum Step {
    Key(String),
    Index(usize),
}

/// A table of the document: its steps from the root, its position as printed
/// (a table without one follows the table before it), and whether it has a header.
struct Located {
    steps: Vec<Step>,
    position: isize,
    headed: bool,
}

/// Collects the tables under `table` in the order `toml_edit` visits them.
fn tables_under(table: &Table, steps: &mut Vec<Step>, last: &mut isize, found: &mut Vec<Located>) {
    for (key, item) in table {
        let children: Vec<(Option<usize>, &Table)> = match item {
            Item::Table(child) => vec![(None, child)],
            Item::ArrayOfTables(array) => array
                .iter()
                .enumerate()
                .map(|(index, child)| (Some(index), child))
                .collect(),
            _ => continue,
        };
        for (index, child) in children {
            steps.push(Step::Key(key.to_string()));
            if let Some(index) = index {
                steps.push(Step::Index(index));
            }
            if !child.is_dotted() {
                *last = child.position().unwrap_or(*last);
                found.push(Located {
                    steps: steps.clone(),
                    position: *last,
                    headed: !child.is_implicit(),
                });
            }
            tables_under(child, steps, last, found);
            steps.truncate(steps.len() - 1 - usize::from(index.is_some()));
        }
    }
}

/// Puts `comments` before `table`'s header, after its leading blank line.
fn prepend_comments(table: &mut Table, comments: &str) {
    if comments.is_empty() {
        return;
    }
    // An unset prefix prints as the default blank line.
    let prefix = table
        .decor()
        .prefix()
        .map_or_else(|| "\n".to_string(), |prefix| raw(Some(prefix)));
    let prefix = match prefix.strip_prefix('\n') {
        Some(rest) => format!("\n{comments}{rest}"),
        None => format!("{comments}{prefix}"),
    };
    table.decor_mut().set_prefix(prefix);
}

/// The comment lines of a decor's prefix, each ending with a newline.
fn comment_lines(decor: &Decor) -> String {
    let mut comments = String::new();
    for line in raw(decor.prefix()).lines() {
        if line.trim_start().starts_with('#') {
            comments.push_str(line);
            comments.push('\n');
        }
    }
    comments
}

/// The text of a raw string, empty when absent or still a span.
fn raw(text: Option<&RawString>) -> String {
    text.and_then(RawString::as_str)
        .unwrap_or_default()
        .to_string()
}

fn protocol_name(protocol: Protocol) -> &'static str {
    match protocol {
        Protocol::Openai => "openai",
        Protocol::Anthropic => "anthropic",
    }
}

fn toml_value(value: FieldValue) -> Value {
    match value {
        FieldValue::Text(text) => Value::from(text),
        FieldValue::Int(number) => Value::from(number),
        FieldValue::Float(number) => Value::from(number),
        FieldValue::Bool(flag) => Value::from(flag),
        FieldValue::List(items) => Value::Array(items.into_iter().collect()),
    }
}

/// A value as a form shows it: strings unquoted, lists comma-separated.
fn display(value: &Value) -> String {
    match value {
        Value::String(text) => text.value().clone(),
        Value::Integer(number) => number.value().to_string(),
        Value::Float(number) => format!("{:?}", number.value()),
        Value::Boolean(flag) => flag.value().to_string(),
        Value::Array(items) => items.iter().map(display).collect::<Vec<_>>().join(", "),
        Value::Datetime(_) | Value::InlineTable(_) => value.to_string().trim().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::fields::{Bound, FieldKind};
    use crate::config::{ConfigError, EnvSource, load_str};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    const COMMENTED: &str = r#"# overbrainer project
[project]
name = "demo" # the project

# What to ask about
[[topics]]
name = "ownership"
subtopics = 3 # a few
questions_per_subtopic = 5

# Providers: keys come from env
[providers.nanogpt]
protocol = "openai"

[roles]
# the question writer
generator = { provider = "nanogpt", model = "m1" }
parent = { provider = "nanogpt", model = "m2", reasoning = true }

[pipeline]
concurrency = 8 # parallel requests
seed = 42

[training]
target = "local"
base_model = "Qwen/Qwen3-4B"
adapter = "qlora"
epochs = 3 # one pass is too few

# Where training runs
[targets.local]
kind = "local"
runtime = "native"
"#;

    fn validate(doc: &ConfigDoc) -> Result<(), ConfigError> {
        load_str(&doc.text(), EnvSource::Vars(Vec::new())).map(|_| ())
    }

    fn problems(doc: &ConfigDoc) -> Vec<String> {
        match load_str(&doc.text(), EnvSource::Vars(Vec::new())) {
            Err(ConfigError::Invalid(problems)) => problems,
            Err(ConfigError::Parse(message)) => vec![message],
            Err(ConfigError::Read { .. }) | Ok(_) => Vec::new(),
        }
    }

    #[test]
    fn an_untouched_document_keeps_its_text() -> TestResult {
        let doc = ConfigDoc::parse(COMMENTED)?;
        assert_eq!(doc.text(), COMMENTED);
        validate(&doc)?;
        Ok(())
    }

    #[test]
    fn a_round_trip_keeps_comments_and_key_order() -> TestResult {
        let mut doc = ConfigDoc::parse(COMMENTED)?;
        doc.set(&FieldPath::Pipeline("concurrency"), FieldValue::Int(16))?;
        let model = FieldPath::Role {
            role: Role::Generator,
            field: "model",
        };
        doc.set(&model, FieldValue::Text("m3".to_string()))?;
        assert!(doc.unset(&FieldPath::Training("epochs"))?);
        let text = doc.text();
        for line in COMMENTED
            .lines()
            .filter(|line| line.trim_start().starts_with('#'))
        {
            assert!(text.contains(line), "lost comment {line:?} in:\n{text}");
        }
        assert!(text.contains("concurrency = 16 # parallel requests\nseed = 42"));
        assert!(text.contains("generator = { provider = \"nanogpt\", model = \"m3\" }"));
        assert!(!text.contains("epochs"));
        assert!(!text.contains("one pass is too few"));
        assert_eq!(doc.get(&model), Some("m3".to_string()));
        validate(&doc)?;
        Ok(())
    }

    #[test]
    fn setting_an_absent_optional_adds_it_in_its_table() -> TestResult {
        let mut doc = ConfigDoc::parse(COMMENTED)?;
        doc.set(
            &FieldPath::Training("hub_model_id"),
            FieldValue::Text("me/child".to_string()),
        )?;
        let text = doc.text();
        let training = text.find("[training]").ok_or("no [training]")?;
        let key = text.find("hub_model_id = \"me/child\"").ok_or("no key")?;
        let next = text.find("# Where training runs").ok_or("no targets")?;
        assert!(training < key && key < next, "misplaced key in:\n{text}");

        let effort = FieldPath::Role {
            role: Role::Parent,
            field: "reasoning_effort",
        };
        doc.set(&effort, FieldValue::Text("high".to_string()))?;
        assert!(doc.text().contains(
            "parent = { provider = \"nanogpt\", model = \"m2\", reasoning = true, reasoning_effort = \"high\" }"
        ));
        validate(&doc)?;
        Ok(())
    }

    #[test]
    fn setting_in_an_absent_section_creates_it() -> TestResult {
        let base = COMMENTED.replace(
            "[pipeline]\nconcurrency = 8 # parallel requests\nseed = 42\n\n",
            "",
        );
        let mut doc = ConfigDoc::parse(&base)?;
        doc.set(&FieldPath::Pipeline("eval_ratio"), FieldValue::Float(0.2))?;
        assert!(doc.text().contains("[pipeline]\neval_ratio = 0.2\n"));
        let embedder = |field| FieldPath::Role {
            role: Role::Embedder,
            field,
        };
        doc.set(
            &embedder("provider"),
            FieldValue::Text("nanogpt".to_string()),
        )?;
        doc.set(&embedder("model"), FieldValue::Text("e1".to_string()))?;
        assert!(
            doc.text()
                .contains("embedder = { provider = \"nanogpt\", model = \"e1\" }")
        );
        validate(&doc)?;
        Ok(())
    }

    #[test]
    fn named_entries_must_exist_before_a_set() -> TestResult {
        let mut doc = ConfigDoc::parse(COMMENTED)?;
        let path = FieldPath::Provider {
            name: "missing".to_string(),
            field: "protocol",
        };
        assert_eq!(
            doc.set(&path, FieldValue::Text("openai".to_string())),
            Err(EditError::Missing("providers.missing".to_string()))
        );
        Ok(())
    }

    #[test]
    fn only_editable_fields_with_valid_values_are_set() -> TestResult {
        let mut doc = ConfigDoc::parse(COMMENTED)?;
        let key = FieldPath::Provider {
            name: "nanogpt".to_string(),
            field: "api_key",
        };
        assert_eq!(
            doc.set(&key, FieldValue::Text("sk-secret".to_string())),
            Err(EditError::NotEditable(
                "providers.nanogpt.api_key".to_string()
            ))
        );
        let other_kind = FieldPath::Target {
            name: "local".to_string(),
            field: "gpu_types",
        };
        assert!(matches!(
            doc.set(&other_kind, FieldValue::List(Vec::new())),
            Err(EditError::NotEditable(_))
        ));
        doc.add_target("box", TargetKind::Ssh)?;
        let host = FieldPath::Target {
            name: "box".to_string(),
            field: "host",
        };
        assert_eq!(
            doc.set(&host, FieldValue::Text("me@gpu".to_string())),
            Err(EditError::NotEditable("targets.box.host".to_string()))
        );
        assert_eq!(
            doc.unset(&FieldPath::Training("target")),
            Err(EditError::Invalid {
                path: "training.target".to_string(),
                reason: FieldError::Required,
            })
        );
        assert!(doc.remove_table(Collection::Targets, "box"));
        let refused = [
            (FieldPath::Pipeline("concurrency"), FieldValue::Int(0)),
            (
                FieldPath::Pipeline("concurrency"),
                FieldValue::Text("8".to_string()),
            ),
            (
                FieldPath::Training("adapter"),
                FieldValue::Text("x".to_string()),
            ),
            (
                FieldPath::Pipeline("eval_ratio"),
                FieldValue::Float(f64::NAN),
            ),
        ];
        for (path, value) in refused {
            let error = doc.set(&path, value);
            assert!(
                matches!(&error, Err(EditError::Invalid { path: p, .. }) if *p == path.to_string()),
                "{path}: {error:?}"
            );
        }
        assert_eq!(doc.text(), COMMENTED);
        Ok(())
    }

    #[test]
    fn unsetting_the_last_key_of_an_inline_table_keeps_its_spacing() -> TestResult {
        let mut doc = ConfigDoc::parse(COMMENTED)?;
        let reasoning = FieldPath::Role {
            role: Role::Parent,
            field: "reasoning",
        };
        assert!(doc.unset(&reasoning)?);
        assert!(!doc.unset(&reasoning)?);
        assert!(
            doc.text()
                .contains("parent = { provider = \"nanogpt\", model = \"m2\" }\n")
        );
        let absent = FieldPath::Target {
            name: "cloud".to_string(),
            field: "image",
        };
        assert!(!doc.unset(&absent)?);
        Ok(())
    }

    #[test]
    fn a_new_runpod_target_writes_the_serde_defaults() -> TestResult {
        let minimal = format!(
            "{COMMENTED}\n[targets.cloud]\nkind = \"runpod\"\ngpu_types = [\"A40\"]\nmax_hours = 6.0\n"
        );
        let mut doc = ConfigDoc::parse(COMMENTED)?;
        doc.add_target("cloud", TargetKind::Runpod)?;
        let gpus = FieldPath::Target {
            name: "cloud".to_string(),
            field: "gpu_types",
        };
        doc.set(&gpus, FieldValue::List(vec!["A40".to_string()]))?;
        let written = load_str(&doc.text(), EnvSource::Vars(Vec::new()))?;
        let defaults = load_str(&minimal, EnvSource::Vars(Vec::new()))?;
        assert_eq!(
            format!("{:?}", written.targets.get("cloud")),
            format!("{:?}", defaults.targets.get("cloud"))
        );
        Ok(())
    }

    #[test]
    fn a_stray_secret_in_the_file_is_never_returned() -> TestResult {
        let text = COMMENTED.replace(
            "protocol = \"openai\"\n",
            "protocol = \"openai\"\napi_key = \"sk-secret\"\n",
        );
        let doc = ConfigDoc::parse(&text)?;
        let key = FieldPath::Provider {
            name: "nanogpt".to_string(),
            field: "api_key",
        };
        assert_eq!(doc.get(&key), None);
        assert_eq!(doc.spec(&key), None);
        Ok(())
    }

    #[test]
    fn a_topic_path_must_match_the_name_at_its_index() -> TestResult {
        let mut doc = ConfigDoc::parse(COMMENTED)?;
        let stale = FieldPath::Topic {
            index: 0,
            name: "traits".to_string(),
            field: "subtopics",
        };
        assert_eq!(doc.get(&stale), None);
        assert_eq!(
            doc.set(&stale, FieldValue::Int(4)),
            Err(EditError::Missing("topics.traits".to_string()))
        );
        let description = FieldPath::Topic {
            index: 0,
            name: "traits".to_string(),
            field: "description",
        };
        assert!(!doc.unset(&description)?);
        assert_eq!(doc.text(), COMMENTED);
        Ok(())
    }

    #[test]
    fn removals_keep_the_comments_above_them() -> TestResult {
        // A key's comment moves to the next key, or to the next table when last.
        let text = COMMENTED.replace("seed = 42\n", "# the split\nseed = 42\n");
        let mut doc = ConfigDoc::parse(&text)?;
        assert!(doc.unset(&FieldPath::Pipeline("seed"))?);
        assert!(
            doc.text()
                .contains("concurrency = 8 # parallel requests\n\n# the split\n[training]")
        );
        let text = text.replace("adapter = \"qlora\"\n", "adapter = \"qlora\"\n# rounds\n");
        let mut doc = ConfigDoc::parse(&text)?;
        assert!(doc.unset(&FieldPath::Training("epochs"))?);
        assert!(doc.text().contains("[training]\ntarget"));

        // A table's comment moves to its next sibling, else to the next table.
        let mut doc = ConfigDoc::parse(COMMENTED)?;
        doc.add_topic("traits")?;
        assert!(doc.remove_topic(0));
        assert!(
            doc.text()
                .contains("# the project\n\n# What to ask about\n[[topics]]\nname = \"traits\"")
        );
        assert!(doc.remove_topic(0));
        assert!(
            doc.text().contains(
                "# What to ask about\n# Providers: keys come from env\n[providers.nanogpt]"
            )
        );
        assert!(doc.remove_table(Collection::Providers, "nanogpt"));
        assert!(
            doc.text()
                .contains("# What to ask about\n# Providers: keys come from env\n[roles]")
        );
        assert!(doc.remove_table(Collection::Targets, "local"));
        assert!(
            doc.text()
                .ends_with("epochs = 3 # one pass is too few\n\n# Where training runs\n")
        );
        validate(&ConfigDoc::parse(&doc.text())?).err();

        let mut doc = ConfigDoc::parse(COMMENTED)?;
        doc.add_provider("claude", Protocol::Anthropic)?;
        assert!(doc.remove_table(Collection::Providers, "nanogpt"));
        assert!(
            doc.text()
                .contains("# Providers: keys come from env\n[providers.claude]")
        );
        Ok(())
    }

    #[test]
    fn get_shows_values_in_display_form() -> TestResult {
        let mut doc = ConfigDoc::parse(COMMENTED)?;
        assert_eq!(
            doc.get(&FieldPath::Project("name")),
            Some("demo".to_string())
        );
        assert_eq!(
            doc.get(&FieldPath::Pipeline("concurrency")),
            Some("8".to_string())
        );
        assert_eq!(doc.get(&FieldPath::Pipeline("eval_ratio")), None);
        let reasoning = FieldPath::Role {
            role: Role::Parent,
            field: "reasoning",
        };
        assert_eq!(doc.get(&reasoning), Some("true".to_string()));
        doc.add_target("cloud", TargetKind::Runpod)?;
        let gpus = FieldPath::Target {
            name: "cloud".to_string(),
            field: "gpu_types",
        };
        doc.set(
            &gpus,
            FieldValue::List(vec!["A40".to_string(), "L40S".to_string()]),
        )?;
        assert_eq!(doc.get(&gpus), Some("A40, L40S".to_string()));
        let hours = FieldPath::Target {
            name: "cloud".to_string(),
            field: "max_hours",
        };
        assert_eq!(doc.get(&hours), Some("6.0".to_string()));
        Ok(())
    }

    #[test]
    fn topics_are_added_and_removed() -> TestResult {
        let mut doc = ConfigDoc::parse(COMMENTED)?;
        assert_eq!(doc.add_topic("traits")?, 1);
        assert_eq!(
            doc.add_topic("traits"),
            Err(EditError::Exists("topics.traits".to_string()))
        );
        assert_eq!(doc.topic_names(), vec!["ownership", "traits"]);
        let text = doc.text();
        let first = text.find("name = \"ownership\"").ok_or("no first")?;
        let second = text.find("name = \"traits\"").ok_or("no second")?;
        let providers = text.find("# Providers").ok_or("no providers")?;
        assert!(
            first < second && second < providers,
            "misplaced topic in:\n{text}"
        );
        validate(&doc)?;

        assert!(doc.remove_topic(0));
        assert!(!doc.remove_topic(5));
        assert_eq!(doc.topic_names(), vec!["traits"]);
        let subtopics = FieldPath::Topic {
            index: 0,
            name: "traits".to_string(),
            field: "subtopics",
        };
        assert_eq!(doc.get(&subtopics), Some("10".to_string()));
        assert!(doc.text().contains("# Providers: keys come from env"));
        validate(&doc)?;

        assert!(doc.remove_topic(0));
        assert!(doc.topic_names().is_empty());
        assert!(!doc.text().contains("[[topics]]"));
        Ok(())
    }

    #[test]
    fn providers_and_targets_are_added_after_their_kind() -> TestResult {
        let mut doc = ConfigDoc::parse(COMMENTED)?;
        doc.add_provider("claude", Protocol::Anthropic)?;
        assert_eq!(
            doc.add_provider("claude", Protocol::Openai),
            Err(EditError::Exists("providers.claude".to_string()))
        );
        assert_eq!(
            doc.add_provider("Bad-Name", Protocol::Openai),
            Err(EditError::InvalidName("Bad-Name".to_string()))
        );
        doc.add_target("cloud", TargetKind::Runpod)?;
        doc.add_target("box", TargetKind::Ssh)?;
        let text = doc.text();
        let nanogpt = text.find("[providers.nanogpt]").ok_or("no nanogpt")?;
        let claude = text
            .find("[providers.claude]\nprotocol = \"anthropic\"\n")
            .ok_or("no claude")?;
        let roles = text.find("[roles]").ok_or("no roles")?;
        assert!(
            nanogpt < claude && claude < roles,
            "misplaced provider in:\n{text}"
        );
        let local = text.find("[targets.local]").ok_or("no local")?;
        let cloud = text.find("[targets.cloud]").ok_or("no cloud")?;
        let then = text.find("[targets.box]").ok_or("no box")?;
        assert!(
            local < cloud && cloud < then,
            "misplaced target in:\n{text}"
        );
        assert!(text.contains(
            "[targets.cloud]\nkind = \"runpod\"\ngpu_types = []\ngpu_count = 1\ncontainer_disk_gb = 50\nmax_hours = 6.0\nboot_grace_minutes = 30\nretrieve_grace_minutes = 60\n"
        ));
        assert!(text.contains("[targets.box]\nkind = \"ssh\"\nruntime = \"docker\"\n"));
        assert_eq!(doc.target_kind("cloud"), Some(TargetKind::Runpod));
        assert_eq!(doc.names(Collection::Providers), vec!["nanogpt", "claude"]);

        assert!(
            problems(&doc)
                .iter()
                .any(|problem| problem.starts_with("targets.cloud.gpu_types:"))
        );
        let gpus = FieldPath::Target {
            name: "cloud".to_string(),
            field: "gpu_types",
        };
        doc.set(&gpus, FieldValue::List(vec!["NVIDIA A40".to_string()]))?;
        validate(&doc)?;
        Ok(())
    }

    #[test]
    fn a_first_provider_or_target_creates_its_collection() -> TestResult {
        let mut doc = ConfigDoc::parse("[project]\nname = \"demo\"\n")?;
        doc.add_provider("nanogpt", Protocol::Openai)?;
        doc.add_target("local", TargetKind::Local)?;
        assert_eq!(doc.add_topic("ownership")?, 0);
        let text = doc.text();
        assert!(!text.contains("[providers]\n"), "bare header in:\n{text}");
        assert!(text.contains("[providers.nanogpt]\nprotocol = \"openai\"\n"));
        assert!(text.contains("[targets.local]\nkind = \"local\"\nruntime = \"docker\"\n"));
        assert!(text.contains(
            "[[topics]]\nname = \"ownership\"\nsubtopics = 10\nquestions_per_subtopic = 30\n"
        ));
        Ok(())
    }

    #[test]
    fn removing_a_table_keeps_the_rest() -> TestResult {
        let mut doc = ConfigDoc::parse(COMMENTED)?;
        doc.add_provider("claude", Protocol::Anthropic)?;
        assert!(doc.remove_table(Collection::Providers, "nanogpt"));
        assert!(!doc.remove_table(Collection::Providers, "nanogpt"));
        assert!(!doc.remove_table(Collection::Targets, "cloud"));
        let text = doc.text();
        assert!(!text.contains("nanogpt]"));
        assert!(text.contains("[providers.claude]"));
        assert!(text.contains("# Where training runs\n[targets.local]"));
        assert!(text.contains("# the question writer"));
        assert!(doc.remove_table(Collection::Targets, "local"));
        assert!(!doc.text().contains("[targets.local]"));
        Ok(())
    }

    #[test]
    fn a_syntax_error_names_only_its_location() {
        let error = ConfigDoc::parse("[project]\nname = \"sk-secret\nx = 1\n");
        assert!(
            matches!(error, Err(EditError::Syntax { line: 2, .. })),
            "{error:?}"
        );
        let message = error
            .map(|_| ())
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(!message.contains("sk-secret"));
        assert_eq!(syntax("é\nx", 1), EditError::Syntax { line: 1, column: 1 });
        assert_eq!(syntax("a\né", 9), EditError::Syntax { line: 2, column: 2 });
    }

    #[test]
    fn paths_display_as_validate_names_them() -> TestResult {
        let mut doc = ConfigDoc::parse(COMMENTED)?;
        doc.add_target("cloud", TargetKind::Runpod)?;
        let cases = [
            (
                FieldPath::Topic {
                    index: 0,
                    name: "ownership".to_string(),
                    field: "subtopics",
                },
                FieldValue::Int(0),
            ),
            (
                FieldPath::Role {
                    role: Role::Generator,
                    field: "max_tokens",
                },
                FieldValue::Int(0),
            ),
            (
                FieldPath::Pipeline("question_batch_size"),
                FieldValue::Int(0),
            ),
            (
                FieldPath::Training("lr_scheduler"),
                FieldValue::Text(" ".to_string()),
            ),
            (
                FieldPath::Target {
                    name: "cloud".to_string(),
                    field: "gpu_types",
                },
                FieldValue::List(Vec::new()),
            ),
        ];
        for (path, value) in cases {
            let mut edited = doc.clone();
            edited.raw_set(&path, value)?;
            let problems = problems(&edited);
            let prefix = format!("{path}:");
            assert!(
                problems.iter().any(|problem| problem.starts_with(&prefix)),
                "{prefix} not in {problems:?}"
            );
        }
        let provider = FieldPath::Provider {
            name: "nanogpt".to_string(),
            field: "protocol",
        };
        assert_eq!(provider.to_string(), "providers.nanogpt.protocol");
        assert_eq!(FieldPath::Project("name").to_string(), "project.name");
        Ok(())
    }

    /// A path to `field` in `section` of `COMMENTED` plus a runpod target `cloud`.
    fn path_in(section: Section, field: &'static str) -> Option<FieldPath> {
        Some(match section {
            Section::Topic => FieldPath::Topic {
                index: 0,
                name: "ownership".to_string(),
                field,
            },
            Section::Role => FieldPath::Role {
                role: Role::Generator,
                field,
            },
            Section::Pipeline => FieldPath::Pipeline(field),
            Section::Training => FieldPath::Training(field),
            Section::Target(TargetKind::Runpod) => FieldPath::Target {
                name: "cloud".to_string(),
                field,
            },
            Section::Project | Section::Provider | Section::Target(_) => return None,
        })
    }

    /// Values at, just inside and just outside each end of a numeric kind.
    fn probes(kind: FieldKind) -> Vec<FieldValue> {
        match kind {
            FieldKind::Int { min, max } => [
                Some(min - 1),
                Some(min),
                Some(min + 1),
                Some(max),
                max.checked_add(1),
            ]
            .into_iter()
            .flatten()
            .map(FieldValue::Int)
            .collect(),
            FieldKind::Float { min, max } => [min, max]
                .into_iter()
                .filter_map(|bound| match bound {
                    Bound::Incl(end) | Bound::Excl(end) => Some(end),
                    Bound::Unbounded => None,
                })
                .flat_map(|end| [end - 0.5, end, end + 0.5])
                .map(FieldValue::Float)
                .collect(),
            _ => Vec::new(),
        }
    }

    #[test]
    fn numeric_bounds_match_validate() -> TestResult {
        let mut doc = ConfigDoc::parse(COMMENTED)?;
        doc.add_target("cloud", TargetKind::Runpod)?;
        let gpus = FieldPath::Target {
            name: "cloud".to_string(),
            field: "gpu_types",
        };
        doc.set(&gpus, FieldValue::List(vec!["A40".to_string()]))?;
        validate(&doc)?;
        let sections = [
            Section::Topic,
            Section::Role,
            Section::Pipeline,
            Section::Training,
            Section::Target(TargetKind::Runpod),
        ];
        let mut probed = 0;
        for section in sections {
            for spec in fields::for_section(section) {
                let Some(path) = path_in(section, spec.name) else {
                    continue;
                };
                for value in probes(spec.kind) {
                    let accepted = spec.kind.check(&value).is_ok();
                    let mut edited = doc.clone();
                    edited.raw_set(&path, value.clone())?;
                    let found = problems(&edited);
                    // The fixture is valid, so any problem comes from this value. Type
                    // errors name a topic `topics[0]subtopics` and a target number only
                    // by its table.
                    let refused = !found.is_empty();
                    let key = match &path {
                        FieldPath::Topic { index, field, .. } => format!("topics[{index}]{field}"),
                        _ => path.table(),
                    };
                    assert!(
                        found
                            .iter()
                            .all(|problem| problem.contains(&path.to_string())
                                || problem.contains(&key)),
                        "{path} = {value:?}: {found:?}"
                    );
                    // thinking_budget also needs an anthropic provider: only refusals agree.
                    if spec.name == "thinking_budget" && !accepted {
                        assert!(refused, "{path} = {value:?} accepted: {found:?}");
                    } else if spec.name != "thinking_budget" {
                        assert_eq!(accepted, !refused, "{path} = {value:?}: {found:?}");
                    }
                    probed += 1;
                }
            }
        }
        assert!(probed > 100, "{probed}");
        Ok(())
    }
}
