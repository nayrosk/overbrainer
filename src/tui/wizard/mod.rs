//! The init wizard: `overbrainer tui` in a directory without `overbrainer.toml`
//! asks, screen by screen, what a project needs, then writes it. This module is
//! the state alone: the screens, their fields, what a key does, and what a
//! screen refuses before the next one. Nothing here draws or writes a file.

mod view;
mod write;

pub(super) use write::existing;

use std::fmt;
use std::io;
use std::path::Path;

use anyhow::Context as _;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Terminal;
use ratatui::backend::Backend;
use secrecy::{ExposeSecret as _, SecretString};

use super::terminal::{self, TerminalGuard};
use super::theme::{ColorLevel, LookEnv, Theme};
use super::widgets::form::{Input, InputOutcome};
use crate::config::{Adapter, ListOrAuto, Protocol, Runtime, is_valid_name};
use crate::secrets::{is_reference, parse_reference};

/// How the wizard ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ended {
    /// Ctrl-C: before the files were written, nothing was.
    Quit {
        /// Whether the files were written first.
        written: bool,
    },
    /// The files are written: the TUI opens, in auto mode when `auto`.
    Open {
        /// Start auto mode now.
        auto: bool,
    },
}

/// Runs the wizard on the real terminal for the project in `dir`. It reads
/// the terminal on this thread alone, starting none: the caller loads `.env`
/// after it, while the process still has a single thread.
///
/// # Errors
///
/// Returns an error when the terminal cannot be set up, read or drawn on.
pub(super) fn run(dir: &Path) -> anyhow::Result<Ended> {
    let name = dir
        .canonicalize()
        .ok()
        .and_then(|dir| {
            dir.file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .unwrap_or_default();
    let theme = Theme::new(ColorLevel::detect(&LookEnv::from_process()));
    let guard = TerminalGuard::enter();
    let mut terminal = terminal::init().context("cannot set up the terminal")?;
    let result = drive(
        &mut *terminal,
        &mut Wizard::new(&name),
        dir,
        &theme,
        event::read,
    );
    drop(guard);
    result
}

/// The wizard's loop on any backend: draws, then hands `next`'s event to the
/// wizard, until it ends. Enter on the summary writes the files into `dir`.
///
/// # Errors
///
/// Returns an error when drawing fails or `next` cannot read an event.
fn drive<B>(
    terminal: &mut Terminal<B>,
    wizard: &mut Wizard,
    dir: &Path,
    theme: &Theme,
    mut next: impl FnMut() -> io::Result<Event>,
) -> anyhow::Result<Ended>
where
    B: Backend,
    B::Error: Send + Sync + 'static,
{
    loop {
        terminal.draw(|frame| view::render(frame, wizard, theme))?;
        let key = match next().context("cannot read the terminal")? {
            Event::Key(key) if key.kind != KeyEventKind::Release => key,
            Event::Paste(text) => {
                wizard.on_paste(&text);
                continue;
            },
            _ => continue,
        };
        match wizard.on_key(key) {
            Step::Stay => {},
            Step::Write => match write::write(dir, &wizard.answers()) {
                Ok(()) => wizard.written(),
                Err(error) => wizard.write_failed(error),
            },
            Step::Quit => {
                return Ok(Ended::Quit {
                    written: wizard.screen() == Screen::Start,
                });
            },
            Step::Done { auto } => return Ok(Ended::Open { auto }),
        }
    }
}

/// A provider the wizard knows: its table name, protocol and base URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Preset {
    /// What the screen shows.
    pub(super) label: &'static str,
    /// The `[providers.<name>]` table.
    pub(super) name: &'static str,
    /// The wire protocol.
    pub(super) protocol: Protocol,
    /// The base URL the client appends `/chat/completions` or `/messages` to.
    pub(super) base_url: &'static str,
}

/// The providers offered before `custom`, in screen order.
pub(super) const PRESETS: [Preset; 4] = [
    Preset {
        label: "OpenRouter",
        name: "openrouter",
        protocol: Protocol::Openai,
        base_url: "https://openrouter.ai/api/v1",
    },
    Preset {
        label: "NanoGPT",
        name: "nanogpt",
        protocol: Protocol::Openai,
        base_url: "https://nano-gpt.com/api/v1",
    },
    Preset {
        label: "OpenAI",
        name: "openai",
        protocol: Protocol::Openai,
        base_url: "https://api.openai.com/v1",
    },
    Preset {
        label: "Anthropic",
        name: "anthropic",
        protocol: Protocol::Anthropic,
        base_url: "https://api.anthropic.com/v1",
    },
];

/// `subtopics` of a new topic.
const NEW_SUBTOPICS: u32 = 10;
/// `questions_per_subtopic` of a new topic.
const NEW_QUESTIONS: u32 = 30;
/// `training.base_model` until changed.
pub(super) const DEFAULT_BASE_MODEL: &str = "Qwen/Qwen3-4B";

/// A screen of the wizard, in the order they come.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Screen {
    /// 1: the project name.
    Name,
    /// 2: the provider.
    Provider,
    /// 3: its API key.
    ApiKey,
    /// 4: the models of the roles.
    Roles,
    /// 5: the topics.
    Topics,
    /// 6: where training runs, if it does.
    Training,
    /// 6, continued: the base model and the adapter.
    Model,
    /// 7: what is written.
    Summary,
    /// Written: start auto mode now?
    Start,
}

impl Screen {
    /// The title the screen shows.
    pub(super) fn title(self) -> &'static str {
        match self {
            Self::Name => "Project",
            Self::Provider => "Provider",
            Self::ApiKey => "API key",
            Self::Roles => "Models",
            Self::Topics => "Topics",
            Self::Training => "Training",
            Self::Model => "Child model",
            Self::Summary => "Summary",
            Self::Start => "Project written",
        }
    }

    /// Its step, out of [`STEPS`]; none once the files are written.
    pub(super) fn step(self) -> Option<usize> {
        match self {
            Self::Name => Some(1),
            Self::Provider => Some(2),
            Self::ApiKey => Some(3),
            Self::Roles => Some(4),
            Self::Topics => Some(5),
            Self::Training | Self::Model => Some(6),
            Self::Summary => Some(7),
            Self::Start => None,
        }
    }

    /// The screen after this one: the child model only when training runs.
    fn after(self, training: TrainingKind) -> Option<Self> {
        match self {
            Self::Name => Some(Self::Provider),
            Self::Provider => Some(Self::ApiKey),
            Self::ApiKey => Some(Self::Roles),
            Self::Roles => Some(Self::Topics),
            Self::Topics => Some(Self::Training),
            Self::Training if training == TrainingKind::Skip => Some(Self::Summary),
            Self::Training => Some(Self::Model),
            Self::Model => Some(Self::Summary),
            Self::Summary | Self::Start => None,
        }
    }

    /// The screen before this one; none before the first and after writing.
    fn before(self, training: TrainingKind) -> Option<Self> {
        match self {
            Self::Name | Self::Start => None,
            Self::Provider => Some(Self::Name),
            Self::ApiKey => Some(Self::Provider),
            Self::Roles => Some(Self::ApiKey),
            Self::Topics => Some(Self::Roles),
            Self::Training => Some(Self::Topics),
            Self::Model => Some(Self::Training),
            Self::Summary if training == TrainingKind::Skip => Some(Self::Training),
            Self::Summary => Some(Self::Model),
        }
    }
}

/// Steps before the files are written.
pub(super) const STEPS: usize = 7;

/// Where training runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TrainingKind {
    /// No `[training]`: auto mode stops after split.
    Skip,
    /// On this machine.
    Local,
    /// On a machine reached over SSH.
    Ssh,
    /// On a Runpod pod.
    Runpod,
}

impl TrainingKind {
    /// Every kind, in screen order.
    const ALL: [Self; 4] = [Self::Skip, Self::Local, Self::Ssh, Self::Runpod];

    /// What the screen shows.
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Skip => "skip (no training)",
            Self::Local => "local",
            Self::Ssh => "ssh",
            Self::Runpod => "runpod",
        }
    }
}

/// One field of a screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Field {
    /// `project.name`.
    Name,
    /// A preset or `custom`.
    Provider,
    /// The name of a custom provider.
    CustomName,
    /// The protocol of a custom provider.
    Protocol,
    /// The base URL of a custom provider.
    BaseUrl,
    /// The provider's API key.
    ApiKey,
    /// `roles.generator.model`.
    Generator,
    /// `roles.parent.model`.
    Parent,
    /// `roles.parent.reasoning`.
    Reasoning,
    /// `roles.embedder.model`, optional.
    Embedder,
    /// Where training runs.
    Training,
    /// The target's runtime.
    Runtime,
    /// The SSH target's host.
    Host,
    /// The Runpod API key.
    RunpodKey,
    /// The Runpod target's GPU types.
    GpuTypes,
    /// `training.base_model`.
    BaseModel,
    /// `training.adapter`.
    Adapter,
}

impl Field {
    /// Its label.
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Name | Self::CustomName => "name",
            Self::Provider => "provider",
            Self::Protocol => "protocol",
            Self::BaseUrl => "base URL",
            Self::ApiKey | Self::RunpodKey => "API key",
            Self::Generator => "generator",
            Self::Parent => "parent",
            Self::Reasoning => "reasoning",
            Self::Embedder => "embedder",
            Self::Training => "target",
            Self::Runtime => "runtime",
            Self::Host => "host",
            Self::GpuTypes => "GPU types",
            Self::BaseModel => "base model",
            Self::Adapter => "adapter",
        }
    }

    /// What it takes, shown under it.
    pub(super) fn hint(self) -> &'static str {
        match self {
            Self::Name => "the name of the project",
            Self::Provider => "← → choose; a preset fills protocol and base URL",
            Self::CustomName => "the [providers.<name>] table: a-z, 0-9 and _",
            Self::Protocol => "← → choose: openai or anthropic",
            Self::BaseUrl => "the URL the protocol's paths go after",
            Self::ApiKey | Self::RunpodKey => {
                "a key, vault:<mount>/<path>#<field>, or empty for later"
            },
            Self::Generator => "the model writing subtopics and questions",
            Self::Parent => "the model whose answers the child learns",
            Self::Reasoning => "← → choose: ask the parent for its reasoning",
            Self::Embedder => "optional: an embedding model for duplicate detection",
            Self::Training => "← → choose; skip stops auto mode after split",
            Self::Runtime => "← → choose: native (a venv) or docker (a container)",
            Self::Host => "the SSH destination, for example user@gpu-box",
            Self::GpuTypes => "auto (cheapest in stock) or GPU types, comma-separated",
            Self::BaseModel => "a Hugging Face repo ID or a path on the target",
            Self::Adapter => "← → choose: lora, qlora or full",
        }
    }

    /// Whether it is chosen with ← and →, not typed.
    pub(super) fn is_choice(self) -> bool {
        matches!(
            self,
            Self::Provider
                | Self::Protocol
                | Self::Reasoning
                | Self::Training
                | Self::Runtime
                | Self::Adapter
        )
    }
}

/// A secret field, never shown and never in its `Debug`. While its screen
/// is shown, it is typed into an [`Input`], which edits a plain `String` like
/// every form input; leaving the screen seals it into a [`SecretString`] and
/// drops the input. Typing into it again opens a new input holding the
/// sealed value: the one copy editing it needs.
pub(super) struct Secret {
    typing: Option<Input>,
    sealed: SecretString,
}

impl fmt::Debug for Secret {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Secret(..)")
    }
}

/// What a secret field holds, as the summary tells it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SecretKind {
    /// Nothing: `.env` gets an empty line to fill.
    Empty,
    /// A `vault:` reference.
    Reference,
    /// A typed key.
    Typed,
}

impl Secret {
    fn new() -> Self {
        Self::sealed("")
    }

    /// A secret sealed with `value`.
    fn sealed(value: &str) -> Self {
        Self {
            typing: None,
            sealed: SecretString::from(value.to_string()),
        }
    }

    /// The input to type into, opened on the sealed value when closed.
    fn input_mut(&mut self) -> &mut Input {
        let sealed = &self.sealed;
        self.typing
            .get_or_insert_with(|| Input::new(sealed.expose_secret()))
    }

    /// Seals what was typed, trimmed, and drops the input.
    fn seal(&mut self) {
        if let Some(input) = self.typing.take() {
            self.sealed = SecretString::from(input.text().trim().to_string());
        }
    }

    /// What it holds, typed or sealed, trimmed: for the checks alone.
    fn text(&self) -> &str {
        match &self.typing {
            Some(input) => input.text().trim(),
            None => self.sealed.expose_secret(),
        }
    }

    /// The sealed value, for the file it goes to.
    pub(super) fn value(&self) -> &SecretString {
        &self.sealed
    }

    /// The input with every char masked.
    pub(super) fn masked(&self) -> Input {
        match &self.typing {
            Some(input) => input.masked(),
            None => Input::new("•".repeat(self.sealed.expose_secret().chars().count())),
        }
    }

    /// What it holds.
    pub(super) fn kind(&self) -> SecretKind {
        match self.text() {
            "" => SecretKind::Empty,
            value if is_reference(value) => SecretKind::Reference,
            _ => SecretKind::Typed,
        }
    }

    /// Why it is refused: a malformed `vault:` reference. The message never
    /// quotes it.
    fn check(&self) -> Result<(), String> {
        parse_reference(self.text())
            .map(drop)
            .map_err(|_| "expected vault:<mount>/<path>#<field>".to_string())
    }
}

/// A topic of the project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TopicDraft {
    /// Its name.
    pub(super) name: String,
    /// What it covers; empty for none.
    pub(super) description: String,
    /// `subtopics`.
    pub(super) subtopics: u32,
    /// `questions_per_subtopic`.
    pub(super) questions_per_subtopic: u32,
}

/// The labels of a topic form's inputs, in order.
pub(super) const TOPIC_FIELDS: [&str; 4] =
    ["name", "description", "subtopics", "questions per subtopic"];

/// A topic being added or edited.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TopicForm {
    /// The topic edited, or `None` for a new one.
    pub(super) editing: Option<usize>,
    /// The input focused, in [`TOPIC_FIELDS`].
    pub(super) focus: usize,
    /// Name, description, subtopics, questions per subtopic.
    pub(super) inputs: [Input; 4],
    /// Why Enter was refused.
    pub(super) error: Option<String>,
}

impl TopicForm {
    fn new(editing: Option<usize>, topic: Option<&TopicDraft>) -> Self {
        let inputs = match topic {
            Some(topic) => [
                Input::new(topic.name.clone()),
                Input::new(topic.description.clone()),
                Input::new(topic.subtopics.to_string()),
                Input::new(topic.questions_per_subtopic.to_string()),
            ],
            None => [
                Input::new(""),
                Input::new(""),
                Input::new(NEW_SUBTOPICS.to_string()),
                Input::new(NEW_QUESTIONS.to_string()),
            ],
        };
        Self {
            editing,
            focus: 0,
            inputs,
            error: None,
        }
    }
}

/// What a key asks the wizard's loop to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Step {
    /// Nothing: draw again.
    Stay,
    /// Write the files (Enter on the summary).
    Write,
    /// Leave: before writing, nothing is written.
    Quit,
    /// The files are written: open the TUI, in auto mode when `auto`.
    Done {
        /// Start auto mode now.
        auto: bool,
    },
}

/// The whole state of the wizard. Neither `Clone` nor `PartialEq`: it holds
/// secrets.
#[derive(Debug)]
pub(super) struct Wizard {
    screen: Screen,
    /// The field focused, in [`Wizard::fields`].
    focus: usize,
    name: Input,
    /// A position in [`PRESETS`], or its length for `custom`.
    provider: usize,
    custom_name: Input,
    protocol: Protocol,
    base_url: Input,
    api_key: Secret,
    generator: Input,
    parent: Input,
    reasoning: bool,
    embedder: Input,
    topics: Vec<TopicDraft>,
    /// The topic selected in the list.
    selected: usize,
    topic_form: Option<TopicForm>,
    training: TrainingKind,
    runtime: Runtime,
    host: Input,
    runpod_key: Secret,
    gpu_types: Input,
    base_model: Input,
    adapter: Adapter,
    /// Why the screen refused Next, shown under the focused field.
    error: Option<String>,
    /// Whether Ctrl-C asks to quit.
    quitting: bool,
    /// The answer selected on the last screen.
    auto: bool,
}

impl Wizard {
    /// A wizard at its first screen, the project named `name` until changed.
    pub(super) fn new(name: &str) -> Self {
        Self {
            screen: Screen::Name,
            focus: 0,
            name: Input::new(name),
            provider: 0,
            custom_name: Input::new(""),
            protocol: Protocol::Openai,
            base_url: Input::new(""),
            api_key: Secret::new(),
            generator: Input::new(""),
            parent: Input::new(""),
            reasoning: true,
            embedder: Input::new(""),
            topics: Vec::new(),
            selected: 0,
            topic_form: None,
            training: TrainingKind::Local,
            runtime: Runtime::Native,
            host: Input::new(""),
            runpod_key: Secret::new(),
            gpu_types: Input::new(ListOrAuto::AUTO),
            base_model: Input::new(DEFAULT_BASE_MODEL),
            adapter: Adapter::Qlora,
            error: None,
            quitting: false,
            auto: true,
        }
    }

    /// The screen shown.
    pub(super) fn screen(&self) -> Screen {
        self.screen
    }

    /// Why the screen refused Next, if it did.
    pub(super) fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Whether Ctrl-C is asking to quit.
    pub(super) fn quitting(&self) -> bool {
        self.quitting
    }

    /// The answer selected on the last screen: start auto mode.
    pub(super) fn auto(&self) -> bool {
        self.auto
    }

    /// The topics, in order.
    pub(super) fn topics(&self) -> &[TopicDraft] {
        &self.topics
    }

    /// The topic selected in the list.
    pub(super) fn selected(&self) -> usize {
        self.selected
    }

    /// The topic being added or edited.
    pub(super) fn topic_form(&self) -> Option<&TopicForm> {
        self.topic_form.as_ref()
    }

    /// The fields of the screen shown, in order: they depend on the choices.
    pub(super) fn fields(&self) -> Vec<Field> {
        match self.screen {
            Screen::Name => vec![Field::Name],
            Screen::Provider if self.custom() => vec![
                Field::Provider,
                Field::CustomName,
                Field::Protocol,
                Field::BaseUrl,
            ],
            Screen::Provider => vec![Field::Provider],
            Screen::ApiKey => vec![Field::ApiKey],
            Screen::Roles => vec![
                Field::Generator,
                Field::Parent,
                Field::Reasoning,
                Field::Embedder,
            ],
            Screen::Training => match self.training {
                TrainingKind::Skip => vec![Field::Training],
                TrainingKind::Local => vec![Field::Training, Field::Runtime],
                TrainingKind::Ssh => vec![Field::Training, Field::Host, Field::Runtime],
                TrainingKind::Runpod => {
                    vec![Field::Training, Field::RunpodKey, Field::GpuTypes]
                },
            },
            Screen::Model => vec![Field::BaseModel, Field::Adapter],
            Screen::Topics | Screen::Summary | Screen::Start => Vec::new(),
        }
    }

    /// The field focused, if the screen has any.
    pub(super) fn focused(&self) -> Option<Field> {
        let fields = self.fields();
        fields
            .get(self.focus.min(fields.len().saturating_sub(1)))
            .copied()
    }

    /// Whether the provider is a custom one.
    fn custom(&self) -> bool {
        self.provider >= PRESETS.len()
    }

    /// The provider's table name.
    pub(super) fn provider_name(&self) -> String {
        match PRESETS.get(self.provider) {
            Some(preset) => preset.name.to_string(),
            None => self.custom_name.text().trim().to_string(),
        }
    }

    /// The provider's protocol.
    pub(super) fn provider_protocol(&self) -> Protocol {
        PRESETS
            .get(self.provider)
            .map_or(self.protocol, |preset| preset.protocol)
    }

    /// The provider's base URL.
    pub(super) fn base_url(&self) -> String {
        match PRESETS.get(self.provider) {
            Some(preset) => preset.base_url.to_string(),
            None => self.base_url.text().trim().to_string(),
        }
    }

    /// The input of a text field.
    fn input(&self, field: Field) -> Option<&Input> {
        match field {
            Field::Name => Some(&self.name),
            Field::CustomName => Some(&self.custom_name),
            Field::BaseUrl => Some(&self.base_url),
            Field::Generator => Some(&self.generator),
            Field::Parent => Some(&self.parent),
            Field::Embedder => Some(&self.embedder),
            Field::Host => Some(&self.host),
            Field::GpuTypes => Some(&self.gpu_types),
            Field::BaseModel => Some(&self.base_model),
            Field::ApiKey
            | Field::RunpodKey
            | Field::Provider
            | Field::Protocol
            | Field::Reasoning
            | Field::Training
            | Field::Runtime
            | Field::Adapter => None,
        }
    }

    fn input_mut(&mut self, field: Field) -> Option<&mut Input> {
        match field {
            Field::Name => Some(&mut self.name),
            Field::CustomName => Some(&mut self.custom_name),
            Field::BaseUrl => Some(&mut self.base_url),
            Field::Generator => Some(&mut self.generator),
            Field::Parent => Some(&mut self.parent),
            Field::Embedder => Some(&mut self.embedder),
            Field::Host => Some(&mut self.host),
            Field::GpuTypes => Some(&mut self.gpu_types),
            Field::BaseModel => Some(&mut self.base_model),
            Field::ApiKey => Some(self.api_key.input_mut()),
            Field::RunpodKey => Some(self.runpod_key.input_mut()),
            Field::Provider
            | Field::Protocol
            | Field::Reasoning
            | Field::Training
            | Field::Runtime
            | Field::Adapter => None,
        }
    }

    /// The secret of a secret field.
    pub(super) fn secret(&self, field: Field) -> Option<&Secret> {
        match field {
            Field::ApiKey => Some(&self.api_key),
            Field::RunpodKey => Some(&self.runpod_key),
            _ => None,
        }
    }

    /// The input `field` draws: its own, or a masked copy for a secret; none
    /// for a choice.
    pub(super) fn shown_input(&self, field: Field) -> Option<Input> {
        self.secret(field)
            .map(Secret::masked)
            .or_else(|| self.input(field).cloned())
    }

    /// What `field` shows: its text, a secret masked, a choice by name.
    pub(super) fn display(&self, field: Field) -> String {
        if let Some(input) = self.shown_input(field) {
            return input.text().to_string();
        }
        match field {
            Field::Provider => PRESETS
                .get(self.provider)
                .map_or("custom", |preset| preset.label)
                .to_string(),
            Field::Protocol => protocol_name(self.protocol).to_string(),
            Field::Reasoning => on_off(self.reasoning).to_string(),
            Field::Training => self.training.label().to_string(),
            Field::Runtime => runtime_name(self.runtime).to_string(),
            Field::Adapter => adapter_name(self.adapter).to_string(),
            _ => String::new(),
        }
    }

    /// Handles a key.
    pub(super) fn on_key(&mut self, key: KeyEvent) -> Step {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            // Once written there is nothing to lose; before, it asks first.
            if self.screen == Screen::Start || self.quitting {
                return Step::Quit;
            }
            self.quitting = true;
            return Step::Stay;
        }
        if self.quitting {
            self.quitting = false;
            return if key.code == KeyCode::Char('y') {
                Step::Quit
            } else {
                Step::Stay
            };
        }
        if !key.modifiers.difference(KeyModifiers::SHIFT).is_empty() {
            return Step::Stay;
        }
        if self.topic_form.is_some() {
            self.on_form_key(key.code);
            return Step::Stay;
        }
        let screen = self.screen;
        if screen == Screen::Start {
            return self.on_start_key(key.code);
        }
        if screen == Screen::Topics && self.on_topics_key(key.code) {
            return Step::Stay;
        }
        match key.code {
            KeyCode::Enter | KeyCode::Tab => return self.next(),
            KeyCode::BackTab | KeyCode::Esc => self.previous(),
            KeyCode::Up => self.focus = self.focus.saturating_sub(1),
            KeyCode::Down => {
                self.focus = (self.focus + 1).min(self.fields().len().saturating_sub(1));
            },
            code => self.on_field_key(code),
        }
        Step::Stay
    }

    /// Pasted text goes to the input focused, if any.
    pub(super) fn on_paste(&mut self, text: &str) {
        if self.quitting {
            return;
        }
        if let Some(form) = &mut self.topic_form {
            if let Some(input) = form.inputs.get_mut(form.focus) {
                input.paste(text);
            }
            return;
        }
        if let Some(field) = self.focused()
            && let Some(input) = self.input_mut(field)
        {
            input.paste(text);
            self.error = None;
        }
    }

    /// The files were written: the last screen asks whether to start auto mode.
    pub(super) fn written(&mut self) {
        self.screen = Screen::Start;
        self.focus = 0;
        self.error = None;
    }

    /// Writing failed: the summary says why, and stays.
    pub(super) fn write_failed(&mut self, error: String) {
        self.error = Some(error);
    }

    /// Enter or Tab: the next screen, unless this one refuses; the summary
    /// asks for the files to be written.
    fn next(&mut self) -> Step {
        if let Err((field, message)) = self.check() {
            if let Some(at) = field.and_then(|field| self.fields().iter().position(|f| *f == field))
            {
                self.focus = at;
            }
            self.error = Some(message);
            return Step::Stay;
        }
        self.error = None;
        if self.screen == Screen::Summary {
            return Step::Write;
        }
        if let Some(next) = self.screen.after(self.training) {
            self.seal();
            self.screen = next;
            self.focus = 0;
        }
        Step::Stay
    }

    /// The screen is left: its secrets are sealed.
    fn seal(&mut self) {
        self.api_key.seal();
        self.runpod_key.seal();
    }

    /// Shift-Tab or Esc: the screen before, every value kept.
    fn previous(&mut self) {
        if let Some(before) = self.screen.before(self.training) {
            self.seal();
            self.screen = before;
            self.focus = 0;
            self.error = None;
        }
    }

    /// A key on the focused field: typed into an input, or ←, → and space
    /// changing a choice.
    fn on_field_key(&mut self, code: KeyCode) {
        let Some(field) = self.focused() else {
            return;
        };
        if field.is_choice() {
            let forward = match code {
                KeyCode::Right | KeyCode::Char(' ') => true,
                KeyCode::Left => false,
                _ => return,
            };
            self.choose(field, forward);
            self.error = None;
            return;
        }
        if let Some(input) = self.input_mut(field) {
            // Enter and Esc never get here: they change the screen.
            input.on_key(code);
            self.error = None;
        }
    }

    /// Moves the choice `field` to its next or previous value.
    fn choose(&mut self, field: Field, forward: bool) {
        let step = |at: usize, len: usize| {
            if forward {
                (at + 1) % len
            } else {
                (at + len - 1) % len
            }
        };
        match field {
            Field::Provider => self.provider = step(self.provider, PRESETS.len() + 1),
            Field::Protocol => {
                self.protocol = match self.protocol {
                    Protocol::Openai => Protocol::Anthropic,
                    Protocol::Anthropic => Protocol::Openai,
                };
            },
            Field::Reasoning => self.reasoning = !self.reasoning,
            Field::Training => {
                let at = TrainingKind::ALL
                    .iter()
                    .position(|kind| *kind == self.training)
                    .unwrap_or(0);
                self.training = TrainingKind::ALL[step(at, TrainingKind::ALL.len())];
            },
            Field::Runtime => {
                self.runtime = match self.runtime {
                    Runtime::Native => Runtime::Docker,
                    Runtime::Docker => Runtime::Native,
                };
            },
            Field::Adapter => {
                const ALL: [Adapter; 3] = [Adapter::Lora, Adapter::Qlora, Adapter::Full];
                let at = ALL.iter().position(|a| *a == self.adapter).unwrap_or(0);
                self.adapter = ALL[step(at, ALL.len())];
            },
            _ => {},
        }
    }

    /// A key on the topic list: `a` adds, `e` edits, `d` deletes, ↑ and ↓
    /// select. Returns whether it was one of them.
    fn on_topics_key(&mut self, code: KeyCode) -> bool {
        match code {
            KeyCode::Char('a') => self.topic_form = Some(TopicForm::new(None, None)),
            KeyCode::Char('e') => {
                if let Some(topic) = self.topics.get(self.selected) {
                    self.topic_form = Some(TopicForm::new(Some(self.selected), Some(topic)));
                }
            },
            KeyCode::Char('d') => {
                if self.selected < self.topics.len() {
                    self.topics.remove(self.selected);
                    self.selected = self.selected.min(self.topics.len().saturating_sub(1));
                }
            },
            KeyCode::Up => self.selected = self.selected.saturating_sub(1),
            KeyCode::Down => {
                self.selected = (self.selected + 1).min(self.topics.len().saturating_sub(1));
            },
            _ => return false,
        }
        self.error = None;
        true
    }

    /// A key in the topic form: ↑, ↓, Tab and Shift-Tab move, Enter keeps
    /// the topic unless refused, Esc drops it.
    fn on_form_key(&mut self, code: KeyCode) {
        let Some(form) = &mut self.topic_form else {
            return;
        };
        match code {
            KeyCode::Up | KeyCode::BackTab => form.focus = form.focus.saturating_sub(1),
            KeyCode::Down | KeyCode::Tab => {
                form.focus = (form.focus + 1).min(TOPIC_FIELDS.len() - 1);
            },
            code => {
                let Some(input) = form.inputs.get_mut(form.focus) else {
                    return;
                };
                match input.on_key(code) {
                    InputOutcome::Editing => form.error = None,
                    InputOutcome::Cancelled => self.topic_form = None,
                    InputOutcome::Done(_) => self.keep_topic(),
                }
            },
        }
    }

    /// Enter in the topic form: the topic is added or changed, unless a value
    /// is refused; the form then says why.
    fn keep_topic(&mut self) {
        let Some(form) = &mut self.topic_form else {
            return;
        };
        match topic_of(form, &self.topics) {
            Ok(topic) => {
                match form.editing {
                    Some(at) if at < self.topics.len() => {
                        self.topics[at] = topic;
                        self.selected = at;
                    },
                    _ => {
                        self.topics.push(topic);
                        self.selected = self.topics.len() - 1;
                    },
                }
                self.topic_form = None;
            },
            Err((at, message)) => {
                form.focus = at;
                form.error = Some(message);
            },
        }
    }

    /// A key on the last screen: `y` or `n`, or ← and → then Enter.
    fn on_start_key(&mut self, code: KeyCode) -> Step {
        match code {
            KeyCode::Char('y') => return Step::Done { auto: true },
            KeyCode::Char('n') => return Step::Done { auto: false },
            KeyCode::Enter => return Step::Done { auto: self.auto },
            KeyCode::Left | KeyCode::Right | KeyCode::Tab | KeyCode::BackTab => {
                self.auto = !self.auto;
            },
            _ => {},
        }
        Step::Stay
    }

    /// Why the screen shown refuses Next: the field to fix, if one, and what
    /// to do. No message quotes a value.
    fn check(&self) -> Result<(), (Option<Field>, String)> {
        let empty = |input: &Input| input.text().trim().is_empty();
        let refuse = |field: Field, message: &str| Err((Some(field), message.to_string()));
        match self.screen {
            Screen::Name if empty(&self.name) => refuse(Field::Name, "type a project name"),
            Screen::Provider if self.custom() => {
                if !is_valid_name(self.custom_name.text().trim()) {
                    return refuse(Field::CustomName, "the name must match ^[a-z0-9_]+$");
                }
                if !is_http_url(self.base_url.text().trim()) {
                    return refuse(Field::BaseUrl, "type an http or https URL");
                }
                Ok(())
            },
            Screen::ApiKey => self.api_key.check().map_err(|m| (Some(Field::ApiKey), m)),
            Screen::Roles => {
                if empty(&self.generator) {
                    return refuse(Field::Generator, "type the generator's model ID");
                }
                if empty(&self.parent) {
                    return refuse(Field::Parent, "type the parent's model ID");
                }
                if !empty(&self.embedder) && self.provider_protocol() == Protocol::Anthropic {
                    return refuse(
                        Field::Embedder,
                        "the anthropic protocol has no embeddings: leave it empty",
                    );
                }
                Ok(())
            },
            Screen::Topics if self.topics.is_empty() => {
                Err((None, "add at least one topic with a".to_string()))
            },
            Screen::Training => match self.training {
                TrainingKind::Ssh if empty(&self.host) => {
                    refuse(Field::Host, "type the host, for example user@gpu-box")
                },
                TrainingKind::Runpod => {
                    self.runpod_key
                        .check()
                        .map_err(|m| (Some(Field::RunpodKey), m))?;
                    if ListOrAuto::from_form_text(self.gpu_types.text()) == ListOrAuto::default() {
                        return refuse(
                            Field::GpuTypes,
                            "type auto or GPU types separated by commas",
                        );
                    }
                    Ok(())
                },
                _ => Ok(()),
            },
            Screen::Model if empty(&self.base_model) => {
                refuse(Field::BaseModel, "type a repo ID or a path")
            },
            _ => Ok(()),
        }
    }

    /// What the summary lists, label then value. Secrets show only what
    /// kind of value they hold.
    pub(super) fn summary(&self) -> Vec<(&'static str, String)> {
        let embedder = self.embedder.text().trim();
        let topics: Vec<String> = self
            .topics
            .iter()
            .map(|topic| {
                format!(
                    "{} ({} x {})",
                    topic.name, topic.subtopics, topic.questions_per_subtopic
                )
            })
            .collect();
        let mut lines = vec![
            ("project", self.name.text().trim().to_string()),
            (
                "provider",
                format!(
                    "{}, {}, {}",
                    self.provider_name(),
                    protocol_name(self.provider_protocol()),
                    self.base_url()
                ),
            ),
            ("API key", secret_summary(self.api_key.kind())),
            ("generator", self.generator.text().trim().to_string()),
            (
                "parent",
                format!(
                    "{}, reasoning {}",
                    self.parent.text().trim(),
                    on_off(self.reasoning)
                ),
            ),
            (
                "embedder",
                if embedder.is_empty() {
                    "none".to_string()
                } else {
                    embedder.to_string()
                },
            ),
            ("topics", topics.join(", ")),
        ];
        let runtime = runtime_name(self.runtime);
        let training = match self.training {
            TrainingKind::Skip => "none: auto mode stops after split".to_string(),
            TrainingKind::Local => format!("local, {runtime}"),
            TrainingKind::Ssh => format!("ssh, {runtime}, {}", self.host.text().trim()),
            TrainingKind::Runpod => format!(
                "runpod, GPU types {}, API key {}",
                ListOrAuto::from_form_text(self.gpu_types.text()),
                secret_summary(self.runpod_key.kind())
            ),
        };
        lines.push(("training", training));
        if self.training != TrainingKind::Skip {
            lines.push((
                "child",
                format!(
                    "{}, {}",
                    self.base_model.text().trim(),
                    adapter_name(self.adapter)
                ),
            ));
        }
        lines
    }

    /// The typed values the writer needs.
    pub(super) fn answers(&self) -> Answers<'_> {
        Answers {
            name: self.name.text().trim(),
            provider: self.provider_name(),
            protocol: self.provider_protocol(),
            base_url: self.base_url(),
            api_key: self.api_key.value(),
            generator: self.generator.text().trim(),
            parent: self.parent.text().trim(),
            reasoning: self.reasoning,
            embedder: Some(self.embedder.text().trim()).filter(|model| !model.is_empty()),
            topics: &self.topics,
            training: self.training,
            runtime: self.runtime,
            host: self.host.text().trim(),
            runpod_key: self.runpod_key.value(),
            gpu_types: ListOrAuto::from_form_text(self.gpu_types.text()),
            base_model: self.base_model.text().trim(),
            adapter: self.adapter,
        }
    }
}

/// What the wizard was told, trimmed, for the files it writes. Its `Debug`
/// never prints the secrets.
#[derive(Clone)]
pub(super) struct Answers<'a> {
    /// `project.name`.
    pub(super) name: &'a str,
    /// The provider's table name.
    pub(super) provider: String,
    /// Its protocol.
    pub(super) protocol: Protocol,
    /// Its base URL.
    pub(super) base_url: String,
    /// Its API key: a key, a reference, or empty.
    pub(super) api_key: &'a SecretString,
    /// The generator's model.
    pub(super) generator: &'a str,
    /// The parent's model.
    pub(super) parent: &'a str,
    /// Whether the parent reasons.
    pub(super) reasoning: bool,
    /// The embedder's model, if any.
    pub(super) embedder: Option<&'a str>,
    /// The topics.
    pub(super) topics: &'a [TopicDraft],
    /// Where training runs.
    pub(super) training: TrainingKind,
    /// The local or SSH target's runtime.
    pub(super) runtime: Runtime,
    /// The SSH host.
    pub(super) host: &'a str,
    /// The Runpod API key: a key, a reference, or empty.
    pub(super) runpod_key: &'a SecretString,
    /// The Runpod GPU types.
    pub(super) gpu_types: ListOrAuto,
    /// `training.base_model`.
    pub(super) base_model: &'a str,
    /// `training.adapter`.
    pub(super) adapter: Adapter,
}

impl fmt::Debug for Answers<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Answers")
            .field("name", &self.name)
            .field("provider", &self.provider)
            .field("training", &self.training)
            .finish_non_exhaustive()
    }
}

/// The topic `form` holds, or which input is refused and why.
fn topic_of(form: &TopicForm, topics: &[TopicDraft]) -> Result<TopicDraft, (usize, String)> {
    let [name, description, subtopics, questions] = &form.inputs;
    let name = name.text().trim();
    if name.is_empty() {
        return Err((0, "type a topic name".to_string()));
    }
    let taken = topics
        .iter()
        .enumerate()
        .any(|(at, topic)| topic.name == name && form.editing != Some(at));
    if taken {
        return Err((0, "another topic has this name".to_string()));
    }
    let count = |input: &Input, at: usize| match input.text().trim().parse::<u32>() {
        Ok(count) if count >= 1 => Ok(count),
        _ => Err((at, "type a whole number, at least 1".to_string())),
    };
    Ok(TopicDraft {
        name: name.to_string(),
        description: description.text().trim().to_string(),
        subtopics: count(subtopics, 2)?,
        questions_per_subtopic: count(questions, 3)?,
    })
}

/// Whether `text` is an `http` or `https` URL with a host.
fn is_http_url(text: &str) -> bool {
    url::Url::parse(text)
        .is_ok_and(|url| matches!(url.scheme(), "http" | "https") && url.host().is_some())
}

/// The `protocol` value of `protocol`.
pub(super) fn protocol_name(protocol: Protocol) -> &'static str {
    match protocol {
        Protocol::Openai => "openai",
        Protocol::Anthropic => "anthropic",
    }
}

/// The `runtime` value of `runtime`.
pub(super) fn runtime_name(runtime: Runtime) -> &'static str {
    match runtime {
        Runtime::Native => "native",
        Runtime::Docker => "docker",
    }
}

/// The `adapter` value of `adapter`.
pub(super) fn adapter_name(adapter: Adapter) -> &'static str {
    match adapter {
        Adapter::Lora => "lora",
        Adapter::Qlora => "qlora",
        Adapter::Full => "full",
    }
}

fn on_off(on: bool) -> &'static str {
    if on { "on" } else { "off" }
}

/// What the summary says of a secret of `kind`.
fn secret_summary(kind: SecretKind) -> String {
    match kind {
        SecretKind::Empty => "empty: fill it in .env",
        SecretKind::Reference => "a vault reference",
        SecretKind::Typed => "typed (hidden)",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use ratatui::backend::TestBackend;

    use super::*;

    const SECRET: &str = "sk-placeholder-secret";

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn press(wizard: &mut Wizard, codes: &[KeyCode]) -> Vec<Step> {
        codes.iter().map(|code| wizard.on_key(key(*code))).collect()
    }

    fn typed(wizard: &mut Wizard, text: &str) {
        for c in text.chars() {
            wizard.on_key(key(KeyCode::Char(c)));
        }
    }

    fn ctrl_c() -> KeyEvent {
        KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)
    }

    /// A wizard filled up to the topics screen, with one topic.
    fn to_training() -> Wizard {
        let mut wizard = Wizard::new("demo");
        press(&mut wizard, &[KeyCode::Enter, KeyCode::Enter]);
        wizard.on_paste(SECRET);
        press(&mut wizard, &[KeyCode::Enter]);
        typed(&mut wizard, "gen");
        press(&mut wizard, &[KeyCode::Down]);
        typed(&mut wizard, "par");
        press(&mut wizard, &[KeyCode::Enter, KeyCode::Char('a')]);
        typed(&mut wizard, "ownership");
        press(&mut wizard, &[KeyCode::Enter, KeyCode::Enter]);
        wizard
    }

    #[test]
    fn next_and_previous_keep_the_values() {
        let mut wizard = Wizard::new("");
        typed(&mut wizard, "demo");
        press(&mut wizard, &[KeyCode::Tab]);
        assert_eq!(wizard.screen(), Screen::Provider);
        press(&mut wizard, &[KeyCode::Right, KeyCode::Enter]);
        assert_eq!(wizard.screen(), Screen::ApiKey);
        press(&mut wizard, &[KeyCode::Esc, KeyCode::BackTab]);
        assert_eq!(wizard.screen(), Screen::Name);
        assert_eq!(wizard.display(Field::Name), "demo");
        press(&mut wizard, &[KeyCode::Enter]);
        assert_eq!(wizard.display(Field::Provider), "NanoGPT");
        assert_eq!(wizard.provider_name(), "nanogpt");
    }

    #[test]
    fn previous_on_the_first_screen_stays() {
        let mut wizard = Wizard::new("demo");
        press(&mut wizard, &[KeyCode::Esc]);
        assert_eq!(wizard.screen(), Screen::Name);
    }

    #[test]
    fn a_required_field_left_empty_blocks_next_with_a_message() {
        let mut wizard = Wizard::new("  ");
        assert_eq!(press(&mut wizard, &[KeyCode::Enter]), [Step::Stay]);
        assert_eq!(wizard.screen(), Screen::Name);
        assert_eq!(wizard.error(), Some("type a project name"));
        typed(&mut wizard, "x");
        assert_eq!(wizard.error(), None, "typing clears the message");

        let mut wizard = to_training();
        wizard.screen = Screen::Roles;
        wizard.parent = Input::new("");
        press(&mut wizard, &[KeyCode::Enter]);
        assert_eq!(wizard.screen(), Screen::Roles);
        assert_eq!(
            wizard.focused(),
            Some(Field::Parent),
            "focus goes to the field"
        );
    }

    #[test]
    fn presets_fill_the_protocol_and_the_base_url() {
        let mut wizard = Wizard::new("demo");
        press(&mut wizard, &[KeyCode::Enter]);
        let mut seen = Vec::new();
        for _ in 0..PRESETS.len() {
            seen.push((
                wizard.provider_name(),
                wizard.provider_protocol(),
                wizard.base_url(),
            ));
            press(&mut wizard, &[KeyCode::Right]);
        }
        assert_eq!(
            seen,
            [
                (
                    "openrouter".into(),
                    Protocol::Openai,
                    "https://openrouter.ai/api/v1".into()
                ),
                (
                    "nanogpt".into(),
                    Protocol::Openai,
                    "https://nano-gpt.com/api/v1".into()
                ),
                (
                    "openai".into(),
                    Protocol::Openai,
                    "https://api.openai.com/v1".into()
                ),
                (
                    "anthropic".into(),
                    Protocol::Anthropic,
                    "https://api.anthropic.com/v1".into()
                ),
            ]
        );
        assert_eq!(wizard.display(Field::Provider), "custom");
        assert_eq!(
            wizard.fields().len(),
            4,
            "custom asks for name, protocol and URL"
        );
        press(&mut wizard, &[KeyCode::Left]);
        assert_eq!(wizard.fields(), [Field::Provider]);
    }

    #[test]
    fn a_custom_provider_needs_a_valid_name_and_url() {
        let mut wizard = Wizard::new("demo");
        press(&mut wizard, &[KeyCode::Enter, KeyCode::Left, KeyCode::Down]);
        typed(&mut wizard, "My-Provider");
        press(&mut wizard, &[KeyCode::Enter]);
        assert_eq!(wizard.error(), Some("the name must match ^[a-z0-9_]+$"));
        wizard.custom_name = Input::new("local_llm");
        press(&mut wizard, &[KeyCode::Down, KeyCode::Right, KeyCode::Down]);
        typed(&mut wizard, "ftp://x");
        press(&mut wizard, &[KeyCode::Enter]);
        assert_eq!(wizard.focused(), Some(Field::BaseUrl));
        assert_eq!(wizard.error(), Some("type an http or https URL"));
        wizard.base_url = Input::new("http://localhost:8080/v1");
        press(&mut wizard, &[KeyCode::Enter]);
        assert_eq!(wizard.screen(), Screen::ApiKey);
        assert_eq!(wizard.provider_name(), "local_llm");
        assert_eq!(wizard.provider_protocol(), Protocol::Anthropic);
        assert_eq!(wizard.base_url(), "http://localhost:8080/v1");
    }

    #[test]
    fn a_secret_is_masked_in_what_the_wizard_shows() {
        let mut wizard = Wizard::new("demo");
        press(&mut wizard, &[KeyCode::Enter, KeyCode::Enter]);
        typed(&mut wizard, SECRET);
        let shown = wizard.display(Field::ApiKey);
        assert_eq!(shown, "•".repeat(SECRET.chars().count()));
        assert_eq!(wizard.api_key.text(), SECRET);
        press(&mut wizard, &[KeyCode::Enter]);
        assert!(
            wizard.api_key.typing.is_none(),
            "sealed once the screen is left"
        );
        assert_eq!(wizard.api_key.value().expose_secret(), SECRET);
        assert_eq!(
            wizard.display(Field::ApiKey),
            "•".repeat(SECRET.chars().count())
        );
        // Back on its screen, typing edits the sealed value.
        press(
            &mut wizard,
            &[KeyCode::Esc, KeyCode::Char('x'), KeyCode::Enter],
        );
        assert_eq!(wizard.api_key.value().expose_secret(), format!("{SECRET}x"));
        let debug = format!("{wizard:?} {:?}", wizard.answers());
        assert!(!debug.contains(SECRET));
        let summary = format!("{:?}", wizard.summary());
        assert!(!summary.contains(SECRET));
        assert!(summary.contains("typed (hidden)"));
    }

    #[test]
    fn a_malformed_vault_reference_is_refused_without_quoting_it() {
        let mut wizard = Wizard::new("demo");
        press(&mut wizard, &[KeyCode::Enter, KeyCode::Enter]);
        wizard.on_paste("vault:secret-only");
        press(&mut wizard, &[KeyCode::Enter]);
        assert_eq!(wizard.screen(), Screen::ApiKey);
        assert_eq!(
            wizard.error(),
            Some("expected vault:<mount>/<path>#<field>")
        );
        wizard.api_key = Secret::sealed("vault:secret/demo#api_key");
        press(&mut wizard, &[KeyCode::Enter]);
        assert_eq!(wizard.screen(), Screen::Roles);
        assert_eq!(wizard.api_key.kind(), SecretKind::Reference);
    }

    #[test]
    fn an_embedder_is_refused_on_the_anthropic_protocol() {
        let mut wizard = to_training();
        wizard.screen = Screen::Roles;
        wizard.provider = 3;
        wizard.embedder = Input::new("text-embedding-3-small");
        press(&mut wizard, &[KeyCode::Enter]);
        assert_eq!(wizard.focused(), Some(Field::Embedder));
    }

    #[test]
    fn topics_are_added_edited_and_deleted_and_one_is_needed() {
        let mut wizard = to_training();
        assert_eq!(wizard.screen(), Screen::Training);
        press(&mut wizard, &[KeyCode::Esc]);
        assert_eq!(wizard.topics().len(), 1);
        assert_eq!(wizard.topics()[0].subtopics, NEW_SUBTOPICS);

        // Edit: a count that is not a number is refused, the form stays.
        press(
            &mut wizard,
            &[KeyCode::Char('e'), KeyCode::Down, KeyCode::Down],
        );
        wizard.on_paste("x");
        press(&mut wizard, &[KeyCode::Enter]);
        let form = wizard.topic_form().cloned();
        assert_eq!(form.as_ref().map(|form| form.focus), Some(2));
        assert!(form.and_then(|form| form.error).is_some());
        press(&mut wizard, &[KeyCode::Backspace, KeyCode::Up, KeyCode::Up]);
        typed(&mut wizard, "_rust");
        press(&mut wizard, &[KeyCode::Enter]);
        assert_eq!(wizard.topics()[0].name, "ownership_rust");
        assert_eq!(wizard.topics()[0].subtopics, 10);

        // Add a second one with the same name: refused.
        press(&mut wizard, &[KeyCode::Char('a')]);
        typed(&mut wizard, "ownership_rust");
        press(&mut wizard, &[KeyCode::Enter]);
        assert!(wizard.topic_form().is_some());
        press(&mut wizard, &[KeyCode::Esc]);
        assert!(wizard.topic_form().is_none());
        assert_eq!(wizard.topics().len(), 1);

        press(&mut wizard, &[KeyCode::Char('d')]);
        assert!(wizard.topics().is_empty());
        press(&mut wizard, &[KeyCode::Enter]);
        assert_eq!(wizard.screen(), Screen::Topics);
        assert_eq!(wizard.error(), Some("add at least one topic with a"));
    }

    #[test]
    fn training_skip_goes_straight_to_the_summary_and_back() {
        let mut wizard = to_training();
        press(&mut wizard, &[KeyCode::Left]);
        assert_eq!(wizard.display(Field::Training), "skip (no training)");
        press(&mut wizard, &[KeyCode::Enter]);
        assert_eq!(wizard.screen(), Screen::Summary);
        press(&mut wizard, &[KeyCode::Esc]);
        assert_eq!(wizard.screen(), Screen::Training);
    }

    #[test]
    fn ssh_needs_a_host_and_runpod_gpu_types() {
        let mut wizard = to_training();
        press(&mut wizard, &[KeyCode::Right]);
        assert_eq!(
            wizard.fields(),
            [Field::Training, Field::Host, Field::Runtime]
        );
        press(&mut wizard, &[KeyCode::Enter]);
        assert_eq!(wizard.focused(), Some(Field::Host));
        press(&mut wizard, &[KeyCode::Up, KeyCode::Right]);
        assert_eq!(wizard.display(Field::GpuTypes), "auto");
        wizard.gpu_types = Input::new(" , ");
        press(&mut wizard, &[KeyCode::Enter]);
        assert_eq!(wizard.focused(), Some(Field::GpuTypes));
        wizard.gpu_types = Input::new("NVIDIA A40, NVIDIA L40S");
        press(&mut wizard, &[KeyCode::Up]);
        typed(&mut wizard, SECRET);
        assert!(!wizard.display(Field::RunpodKey).contains(SECRET));
        press(&mut wizard, &[KeyCode::Enter]);
        assert_eq!(wizard.screen(), Screen::Model);
        assert_eq!(
            wizard.answers().gpu_types,
            ListOrAuto::List(vec!["NVIDIA A40".into(), "NVIDIA L40S".into()])
        );
    }

    #[test]
    fn the_summary_asks_to_write_then_the_last_screen_answers() {
        let mut wizard = to_training();
        assert_eq!(
            press(
                &mut wizard,
                &[KeyCode::Enter, KeyCode::Enter, KeyCode::Enter]
            ),
            [Step::Stay, Step::Stay, Step::Write]
        );
        wizard.write_failed("overbrainer.toml already exists".into());
        assert_eq!(wizard.screen(), Screen::Summary);
        wizard.written();
        assert_eq!(press(&mut wizard, &[KeyCode::Esc]), [Step::Stay]);
        assert_eq!(wizard.screen(), Screen::Start, "nothing to go back to");
        assert_eq!(
            press(&mut wizard, &[KeyCode::Char('n')]),
            [Step::Done { auto: false }]
        );
        assert_eq!(
            press(
                &mut wizard,
                &[KeyCode::Right, KeyCode::Right, KeyCode::Enter]
            ),
            [Step::Stay, Step::Stay, Step::Done { auto: true }]
        );
    }

    #[test]
    fn ctrl_c_asks_before_quitting_until_written() {
        let mut wizard = Wizard::new("demo");
        assert_eq!(wizard.on_key(ctrl_c()), Step::Stay);
        assert!(wizard.quitting());
        assert_eq!(press(&mut wizard, &[KeyCode::Char('n')]), [Step::Stay]);
        assert!(!wizard.quitting());
        assert_eq!(wizard.display(Field::Name), "demo", "n typed nothing");
        wizard.on_key(ctrl_c());
        assert_eq!(press(&mut wizard, &[KeyCode::Char('y')]), [Step::Quit]);
        wizard.on_key(ctrl_c());
        assert_eq!(wizard.on_key(ctrl_c()), Step::Quit, "twice quits too");
        wizard.written();
        assert_eq!(
            wizard.on_key(ctrl_c()),
            Step::Quit,
            "written: nothing to lose"
        );
    }

    /// Runs the loop on `events`, in a new directory; returns how it ended
    /// and the names of the files it holds then.
    fn drove(events: Vec<Event>) -> Result<(Ended, Vec<String>), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;
        let mut events = events.into_iter();
        let theme = Theme::mono();
        let mut wizard = Wizard::new("demo");
        let ended = drive(&mut terminal, &mut wizard, dir.path(), &theme, || {
            events
                .next()
                .ok_or_else(|| io::Error::other("no more events"))
        })?;
        let mut files: Vec<String> = std::fs::read_dir(dir.path())?
            .map(|entry| entry.map(|e| e.file_name().to_string_lossy().into_owned()))
            .collect::<Result<_, _>>()?;
        files.sort();
        Ok((ended, files))
    }

    fn keys(codes: &[KeyCode]) -> Vec<Event> {
        codes.iter().map(|code| Event::Key(key(*code))).collect()
    }

    #[test]
    fn ctrl_c_then_y_quits_the_loop_writing_nothing() -> Result<(), Box<dyn std::error::Error>> {
        let mut events = keys(&[KeyCode::Enter]);
        events.push(Event::Key(ctrl_c()));
        events.extend(keys(&[KeyCode::Char('y')]));
        assert_eq!(drove(events)?, (Ended::Quit { written: false }, Vec::new()));
        Ok(())
    }

    #[test]
    fn the_loop_writes_the_files_then_opens_the_tui_as_answered()
    -> Result<(), Box<dyn std::error::Error>> {
        for (answer, auto) in [(KeyCode::Char('y'), true), (KeyCode::Char('n'), false)] {
            let mut events = keys(&[KeyCode::Enter, KeyCode::Enter]);
            events.push(Event::Paste(SECRET.to_string()));
            events.extend(keys(&[KeyCode::Enter]));
            events.extend("gen".chars().map(|c| Event::Key(key(KeyCode::Char(c)))));
            events.extend(keys(&[KeyCode::Down]));
            events.extend("par".chars().map(|c| Event::Key(key(KeyCode::Char(c)))));
            events.extend(keys(&[KeyCode::Enter, KeyCode::Char('a')]));
            events.push(Event::Paste("ownership".to_string()));
            // Topic kept, Training, Model, Summary, write, then the answer.
            events.extend(keys(&[
                KeyCode::Enter,
                KeyCode::Enter,
                KeyCode::Enter,
                KeyCode::Enter,
                KeyCode::Enter,
                answer,
            ]));
            let (ended, files) = drove(events)?;
            assert_eq!(ended, Ended::Open { auto });
            assert_eq!(
                files,
                [
                    ".env",
                    ".env.example",
                    ".gitignore",
                    "overbrainer.toml",
                    "prompts"
                ]
            );
        }
        Ok(())
    }

    #[test]
    fn a_paste_goes_to_the_focused_input_on_one_line() {
        let mut wizard = Wizard::new("");
        wizard.on_paste("my\nproject\n");
        assert_eq!(wizard.display(Field::Name), "my project");
        press(&mut wizard, &[KeyCode::Enter]);
        wizard.on_paste("ignored");
        assert_eq!(
            wizard.display(Field::Provider),
            "OpenRouter",
            "a choice takes no paste"
        );
    }
}
