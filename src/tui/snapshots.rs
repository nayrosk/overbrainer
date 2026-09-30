//! Snapshot tests of every view on `TestBackend`, at 80x24 and 120x40, and the
//! monochrome theme. Fixtures build the app from fixed records and a fixed clock,
//! so no runtime, file or network is needed.

use std::convert::Infallible;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crossterm::event::{
    Event as TermEvent, KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers,
};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::style::{Color, Modifier};
use tracing::Level;

use super::app::{App, Effect, Overlay, Project, View};
use super::dataset::{Node, TopicInfo};
use super::motion::MotionLevel;
use super::project::ProjectConfig;
use super::tasks::{Done, History, TaskId};
use super::theme::{ColorLevel, LookEnv, Theme};
use super::ui;
use super::widgets::status::VERSION;
use crate::cli::data::Command;
use crate::config::{ConfigError, EnvSource, ListOrAuto};
use crate::dataset::{
    Dataset, Example, Exclusion, FinishReason, Id, Message, Meta, Question, ReasoningKind,
    Rejected, Role, Subtopic,
};
use crate::events::{Event, Stage, StageStats};
use crate::llm::Usage;
use crate::logging::{LogBuffer, LogLine};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Where the snapshot files are kept: outside `src/`, so the published crate does
/// not ship them.
const SNAPSHOTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/snapshots/tui");

/// The fixtures' clock, 2026-09-21 14:13:20 UTC.
pub(super) const NOW: u64 = 1_790_000_000;

/// `seconds` after the Unix epoch.
pub(super) fn at(seconds: u64) -> SystemTime {
    UNIX_EPOCH + Duration::from_secs(seconds)
}

/// An app on the project `rust_expert` at [`NOW`], with the 24-bit theme.
pub(super) fn app() -> App {
    app_with(&Theme::new(ColorLevel::TrueColor))
}

/// An app on the project `rust_expert` at [`NOW`], with `theme`, on the
/// Dataset view: most tests are about the data.
pub(super) fn app_with(theme: &Theme) -> App {
    let project = Project {
        name: "rust_expert".into(),
        dir: "/nonexistent/rust_expert".into(),
        topics: Vec::new(),
        eval_ratio: 0.1,
        concurrency: 8,
        target: Some("gpu_cloud".into()),
    };
    let mut app = App::new(project, LogBuffer::new(100), theme, at(NOW));
    app.view = View::Dataset;
    app
}

/// The fixtures' topics: `ownership` and `traits`.
pub(super) fn topics() -> Vec<TopicInfo> {
    vec![
        TopicInfo {
            name: "ownership".into(),
            description: Some("Moves, borrows and lifetimes in Rust".into()),
            subtopics: 2,
            questions_per_subtopic: 3,
        },
        TopicInfo {
            name: "traits".into(),
            description: None,
            subtopics: 2,
            questions_per_subtopic: 2,
        },
    ]
}

fn subtopic(topic: &str, name: &str) -> Subtopic {
    Subtopic {
        id: Id::subtopic(topic, name),
        topic: topic.into(),
        name: name.into(),
    }
}

fn question(topic: &str, subtopic: &str, text: &str) -> Question {
    let subtopic_id = Id::subtopic(topic, subtopic);
    Question {
        id: Id::question(&subtopic_id, text),
        topic: topic.into(),
        subtopic_id,
        subtopic: subtopic.into(),
        text: text.into(),
    }
}

fn answer(
    question: &Question,
    reasoning: Option<&str>,
    content: &str,
    excluded: Option<Exclusion>,
) -> Example {
    Example {
        id: question.id.clone(),
        topic: question.topic.clone(),
        subtopic: question.subtopic.clone(),
        messages: vec![
            Message {
                role: Role::User,
                content: question.text.clone(),
                reasoning_content: None,
            },
            Message {
                role: Role::Assistant,
                content: content.into(),
                reasoning_content: reasoning.map(str::to_string),
            },
        ],
        meta: Meta {
            model: "deepseek-r1".into(),
            input_tokens: 312,
            output_tokens: 1840,
            finish_reason: if excluded == Some(Exclusion::Truncated) {
                FinishReason::Length
            } else {
                FinishReason::Stop
            },
            reasoning_kind: ReasoningKind::Raw,
            excluded,
        },
    }
}

/// The first question of the fixture: answered, usable, with a reasoning.
pub(super) const MOVED: &str = "What happens to a borrow when the owner is moved?";

/// A dataset with every kind of node: answered, excluded and open questions, a
/// question whose subtopic is gone, and a topic no longer configured.
pub(super) fn dataset() -> Dataset {
    let moved = question("ownership", "Borrowing", MOVED);
    let coexist = question(
        "ownership",
        "Borrowing",
        "Why can't a &mut and a & borrow coexist?",
    );
    let nll = question("ownership", "Borrowing", "When does NLL end a borrow?");
    let lifetime = question(
        "ownership",
        "Lifetimes",
        "What does 'a mean in fn f<'a>(x: &'a str)?",
    );
    let copy = question("ownership", "Moves", "Is a move a copy of the bytes?");
    let legacy = question("old_topic", "Legacy", "Is this question still used?");
    let reasoning = "Let me think about what a move does to a borrow.\n\
                     The borrow checker tracks every live reference to the owner.\n\
                     A move invalidates the owner's place, so no borrow may outlive it.";
    let content = "The borrow must end before the move.\n\
                   If a reference is still used after the move, the compiler reports E0505: \
                   cannot move out of the value because it is borrowed.\n\
                   Non-lexical lifetimes end a borrow at its last use, not at the end of \
                   its scope, so many such programs compile.";
    Dataset {
        subtopics: vec![
            subtopic("ownership", "Borrowing"),
            subtopic("ownership", "Lifetimes"),
            subtopic("old_topic", "Legacy"),
        ],
        answers: vec![
            answer(&moved, Some(reasoning), content, None),
            answer(
                &coexist,
                Some("Aliasing and mutation."),
                "Because",
                Some(Exclusion::Truncated),
            ),
            answer(&lifetime, None, "A lifetime parameter.", None),
            answer(&copy, None, "Yes, a bitwise copy.", None),
            answer(&legacy, None, "Maybe.", None),
        ],
        rejected: vec![Rejected::Question {
            id: Id::question(&Id::subtopic("ownership", "Borrowing"), "What is a borrow?"),
            topic: "ownership".into(),
            subtopic_id: Id::subtopic("ownership", "Borrowing"),
            text: "What is a borrow?".into(),
        }],
        questions: vec![moved, coexist, nll, lifetime, copy, legacy],
    }
}

/// [`app`] on the `ownership` and `traits` topics with [`dataset`] loaded.
pub(super) fn dataset_app() -> App {
    dataset_app_with(&Theme::new(ColorLevel::TrueColor))
}

/// [`dataset_app`] with `theme`.
pub(super) fn dataset_app_with(theme: &Theme) -> App {
    let mut app = app_with(theme);
    app.project.topics = topics();
    app.dataset.loaded(dataset(), &app.project.topics);
    app
}

/// `overbrainer.toml` of the fixtures' project.
pub(super) const CONFIG: &str = r#"
[project]
name = "rust_expert"

[[topics]]
name = "ownership"
description = "Moves, borrows and lifetimes in Rust"
subtopics = 2
questions_per_subtopic = 3

[[topics]]
name = "traits"
subtopics = 2
questions_per_subtopic = 2

[providers.fake]
protocol = "openai"

[roles]
generator = { provider = "fake", model = "gen" }
parent = { provider = "fake", model = "parent" }
"#;

/// A provider key, a runpod key by Vault reference, an env value and a
/// runpod target.
pub(super) const PROJECT_CONFIG: &str = r#"# the project
[project]
name = "rust_expert"

[[topics]]
name = "ownership"
description = "Moves, borrows and lifetimes in Rust"
subtopics = 2
questions_per_subtopic = 3

[providers.nanogpt]
protocol = "openai"

[providers.claude]
protocol = "anthropic"

[roles]
generator = { provider = "nanogpt", model = "gen" }
parent = { provider = "claude", model = "claude-opus-5", reasoning = true }

[pipeline]
concurrency = 16 # overridden by env

[training]
target = "gpu_cloud"
base_model = "Qwen/Qwen3-8B"
adapter = "lora"

[targets.gpu_cloud]
kind = "runpod"
gpu_types = ["NVIDIA A40"]
max_hours = 6
"#;

/// A secret no row may show.
pub(super) const SECRET: &str = "sk-live-0123456789abcdef";

/// The environment of [`PROJECT_CONFIG`].
pub(super) fn project_env() -> EnvSource {
    EnvSource::Vars(vec![
        (
            "OVERBRAINER_PROVIDERS__NANOGPT__API_KEY".into(),
            SECRET.into(),
        ),
        (
            "OVERBRAINER_PROVIDERS__NANOGPT__BASE_URL".into(),
            "https://nano-gpt.com/api/v1".into(),
        ),
        (
            "OVERBRAINER_RUNPOD__API_KEY".into(),
            "vault:secret/overbrainer/runpod#api_key".into(),
        ),
        ("OVERBRAINER_PIPELINE__CONCURRENCY".into(), "4".into()),
        ("OVERBRAINER_TUI_COLOR".into(), "none".into()),
        ("HOME".into(), "/home/me".into()),
    ])
}

/// [`PROJECT_CONFIG`] in [`project_env`].
pub(super) fn project_config() -> Result<ProjectConfig, ConfigError> {
    ProjectConfig::new(PROJECT_CONFIG, &project_env())
}

/// A project directory holding [`CONFIG`] and the files of [`dataset`].
pub(super) fn project() -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    std::fs::write(dir.path().join("overbrainer.toml"), CONFIG)?;
    let files = crate::dataset::DataFiles::new(dir.path());
    let data = dataset();
    crate::dataset::rewrite(&files.subtopics, &data.subtopics)?;
    crate::dataset::rewrite(&files.questions, &data.questions)?;
    crate::dataset::rewrite(&files.answers, &data.answers)?;
    crate::dataset::rewrite(&files.rejected, &data.rejected)?;
    Ok(dir)
}

/// The tree path of a question of `ownership`/`Borrowing`.
pub(super) fn path_to(text: &str) -> Vec<Node> {
    let subtopic = Id::subtopic("ownership", "Borrowing");
    vec![
        Node::Topic("ownership".into()),
        Node::Subtopic(subtopic.clone()),
        Node::Question(Id::question(&subtopic, text)),
    ]
}

/// Opens every node of `path` but the last and selects it.
pub(super) fn open_to(app: &mut App, path: &[Node]) {
    for depth in 1..path.len() {
        app.dataset.tree.open(path[..depth].to_vec());
    }
    app.dataset.tree.select(path.to_vec());
}

fn usage(input_tokens: u64, output_tokens: u64) -> Usage {
    Usage {
        input_tokens,
        output_tokens,
    }
}

/// `stage` run to its end: `total` items, `retries` retried attempts.
fn stage_events(stage: Stage, total: usize, retries: usize) -> Vec<Event> {
    let mut events = vec![Event::StageStarted { stage, total }];
    for n in 0..retries {
        events.push(Event::ItemFailed {
            stage,
            id: format!("item{n}"),
            error: "429 Too Many Requests; retrying in 4.0s".into(),
            retryable: true,
        });
    }
    for n in 0..total {
        events.push(Event::ItemDone {
            stage,
            id: format!("item{n}"),
            usage: Some(usage(1000, 250)),
            cost: None,
            excluded: false,
        });
    }
    events
}

/// A `run` task past its subtopics and questions, answering: 120 of 400 done,
/// 14 retries, 1 failure.
pub(super) fn pipeline_running(app: &mut App) {
    app.pipeline_task = Some(TaskId(7));
    app.pipeline.started(Command::Run, 8);
    let mut events = stage_events(Stage::Subtopics, 3, 0);
    events.push(Event::StageFinished {
        stage: Stage::Subtopics,
        stats: StageStats {
            done: 3,
            usage: usage(3000, 750),
            cost: Some(0.0012),
            ..StageStats::default()
        },
    });
    events.extend(stage_events(Stage::Questions, 36, 2));
    events.push(Event::StageFinished {
        stage: Stage::Questions,
        stats: StageStats {
            done: 36,
            usage: usage(40_210, 9120),
            cost: Some(0.0123),
            ..StageStats::default()
        },
    });
    events.extend(stage_events(Stage::Answers, 400, 0).into_iter().take(120));
    for n in 0..14 {
        events.push(Event::ItemFailed {
            stage: Stage::Answers,
            id: format!("3f1c000000000000000000000000{n:04x}"),
            error: "429 Too Many Requests; retrying in 4.0s".into(),
            retryable: true,
        });
    }
    events.push(Event::ItemFailed {
        stage: Stage::Answers,
        id: "77aa00000000000000000000000010bc".into(),
        error: "gave up: the answer is not a JSON array of strings".into(),
        retryable: false,
    });
    for event in &events {
        app.pipeline.event(event);
    }
    for line in [
        "subtopics: 3 done, 0 skipped, 0 failed, 0 excluded; tokens 3000 in, 750 out; cost $0.0012",
        "questions: 36 done, 0 skipped, 0 failed, 0 excluded; tokens 40210 in, 9120 out; cost $0.0123",
    ] {
        app.pipeline.results.push(line.into());
    }
}

/// A run record of the fixtures.
pub(super) fn run(id: &str, target: &str, state: crate::runs::RunState) -> crate::runs::RunRecord {
    crate::runs::RunRecord {
        id: id.into(),
        target: target.into(),
        created: "2026-09-21T13:00:00Z".into(),
        remote_dir: format!("/workspace/overbrainer/{id}"),
        job: None,
        state,
        message: None,
        snapshot: None,
        resumed_from: None,
    }
}

/// The pod of a Runpod run created 41 minutes before [`NOW`], at $0.53/h.
pub(super) fn pod(run_id: &str) -> Result<crate::runpod::PodRecord, serde_json::Error> {
    let mut record = crate::runpod::PodRecord::new(run_id, false, 1, "ssh-ed25519 AAAAhost");
    let remote: crate::runpod::Pod = serde_json::from_value(serde_json::json!({
        "id": "k3x9abc", "status": "RUNNING", "cost": 0.53,
        "gpu": {"id": "NVIDIA GeForce RTX 4090", "count": 1}
    }))?;
    let created = at(NOW - 41 * 60);
    record.begin_attempt("NVIDIA GeForce RTX 4090", created, 6.0);
    record.created(&remote, crate::runpod::AttemptResult::Created, created);
    let endpoint = crate::runpod::SshEndpoint {
        host: "203.0.113.7".into(),
        port: 40022,
        user: "root".into(),
    };
    record.ready(endpoint, at(NOW - 38 * 60));
    Ok(record)
}

/// 1200 steps of a 4000-step run at half a step per second: a training log every
/// 10 steps, an evaluation every 200.
pub(super) fn series() -> Vec<crate::train::TrainMetric> {
    (1..=120_u32)
        .map(|n| {
            let step = n * 10;
            let x = f64::from(step) / 1200.0;
            let eval = step % 200 == 0;
            crate::train::TrainMetric {
                time: 1000.0 + f64::from(step) * 2.0,
                step: u64::from(step),
                epoch: Some(f64::from(step) / 1333.0),
                max_steps: Some(4000),
                loss: (!eval).then_some(0.8 + 1.1 * (1.0 - x) * (1.0 - x)),
                eval_loss: eval.then_some(0.9 + 1.0 * (1.0 - x) * (1.0 - x)),
                learning_rate: Some(2e-4 * (0.2 + 0.8 * x.min(0.5) * 2.0).min(1.0)),
                grad_norm: Some(0.7 + 0.05 * f64::from(n % 5)),
            }
        })
        .collect()
}

/// The fixture GPU catalog as the GPU picker lists it for `gpu_count` GPUs,
/// the fit of each unknown.
pub(super) fn gpu_catalog(
    gpu_count: u32,
) -> Result<Vec<super::widgets::picker::Entry>, serde_json::Error> {
    Ok(super::catalog::gpu_entries(&gpu_types()?, gpu_count, None))
}

/// What the fixture run needs per GPU: Qwen3-4B with `LoRA`, about 19.1 GB.
pub(super) fn need() -> crate::train::sizing::Estimate {
    use crate::train::sizing::{ModelShape, Recipe, estimate};
    let shape = ModelShape {
        params: 4_022_468_096,
        hidden_size: 2560,
        layers: 36,
        vocab_size: 151_936,
    };
    let recipe = Recipe {
        adapter: crate::config::Adapter::Lora,
        sequence_len: 4096,
        micro_batch_size: 2,
        lora_r: 16,
        eight_bit_optimizer: false,
        gradient_checkpointing: true,
    };
    estimate(&shape, &recipe)
}

/// The start dialog's lookup of `gpus`, the run needing [`need`].
pub(super) fn looked_up(gpus: super::start::Gpus) -> super::start::Catalog {
    super::start::Catalog {
        gpus,
        need: Ok(need()),
    }
}

/// The fixture GPU catalog as listed.
pub(super) fn gpu_types() -> Result<Vec<crate::runpod::GpuType>, serde_json::Error> {
    serde_json::from_value(serde_json::json!([
        {"id": "NVIDIA A40", "memory": 48, "price": {"secure": 0.4},
         "maxCount": {"secure": 10}, "availability": "HIGH"},
        {"id": "NVIDIA GeForce RTX 4090", "memory": 24, "price": {"secure": 0.69},
         "maxCount": {"secure": 8}, "availability": "MEDIUM"},
        {"id": "NVIDIA L4", "memory": 24, "price": {"secure": 0.43},
         "maxCount": {"secure": 1}, "availability": "HIGH"},
        {"id": "NVIDIA RTX A6000", "memory": 48, "price": {"secure": 0.49},
         "maxCount": {"secure": 8}, "availability": "LOW"},
        {"id": "NVIDIA A100 80GB PCIe", "memory": 80, "price": {"secure": 1.64},
         "maxCount": {"secure": 8}, "availability": "NONE"},
        {"id": "NVIDIA H100 80GB HBM3", "memory": 80, "price": {"secure": 2.99},
         "maxCount": {"secure": 8}, "availability": "LOW"},
        {"id": "NVIDIA RTX 2000 Ada Generation", "memory": 16, "price": {"secure": 0.24},
         "maxCount": {"secure": 4}, "availability": "HIGH"},
        {"id": "NVIDIA L40S", "memory": 48, "price": {"secure": 0.86},
         "maxCount": {"secure": 8}, "availability": "MEDIUM"},
        {"id": "AMD Instinct MI300X OAM", "memory": 192, "price": {"secure": 2.49},
         "maxCount": {"secure": 8}, "availability": "NONE"},
        {"id": "NVIDIA H200", "memory": 141, "maxCount": {"secure": 8}}
    ]))
}

/// A listing of `entries` only, as a picker of volumes or templates reads it.
fn listed(entries: Vec<super::widgets::picker::Entry>) -> super::catalog::Listed {
    super::catalog::Listed {
        entries,
        gpus: Vec::new(),
        note: None,
    }
}

/// The fixture GPU catalog as the GPU picker reads it for `gpu_count` GPUs,
/// the run needing [`need`].
fn gpu_listing(gpu_count: u32) -> Result<super::catalog::Listed, serde_json::Error> {
    let entries = super::catalog::gpu_entries(&gpu_types()?, gpu_count, Some(&need()));
    Ok(super::catalog::Listed {
        note: Some(format!("{} per GPU needed", need())),
        ..listed(entries)
    })
}

/// A key press.
pub(super) fn key(code: KeyCode) -> TermEvent {
    TermEvent::Key(KeyEvent {
        code,
        modifiers: KeyModifiers::NONE,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    })
}

/// Ctrl-C, a key in raw mode.
pub(super) fn ctrl_c() -> TermEvent {
    TermEvent::Key(KeyEvent {
        code: KeyCode::Char('c'),
        modifiers: KeyModifiers::CONTROL,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    })
}

/// `app` drawn on a `width` by `height` test terminal.
pub(super) fn draw(
    app: &mut App,
    width: u16,
    height: u16,
) -> Result<Terminal<TestBackend>, Infallible> {
    let mut terminal = Terminal::new(TestBackend::new(width, height))?;
    terminal.draw(|frame| ui::render(frame, app))?;
    Ok(terminal)
}

/// The rows of `terminal`'s buffer as text.
pub(super) fn text(terminal: &Terminal<TestBackend>) -> Vec<String> {
    let buffer = terminal.backend().buffer();
    let width = usize::from(buffer.area.width).max(1);
    buffer
        .content()
        .chunks(width)
        .map(|row| row.iter().map(ratatui::buffer::Cell::symbol).collect())
        .collect()
}

/// Checks `app` drawn at `width` by `height` against the snapshot `name`.
fn snapshot_at(name: &str, app: &mut App, width: u16, height: u16) -> Result<(), Infallible> {
    let terminal = draw(app, width, height)?;
    let mut settings = insta::Settings::clone_current();
    settings.set_snapshot_path(SNAPSHOTS);
    settings.set_prepend_module_to_snapshot(false);
    settings.set_omit_expression(true);
    // The version changes at every release: its digits are masked, its width kept.
    // A release that adds a digit (v0.9.9 to v0.10.0) still changes the snapshots.
    let masked: String = VERSION
        .chars()
        .map(|c| if c.is_ascii_digit() { '#' } else { c })
        .collect();
    let screen = terminal.backend().to_string().replace(VERSION, &masked);
    settings.bind(|| insta::assert_snapshot!(name.to_string(), screen));
    Ok(())
}

/// Checks `app` against the snapshots `<name>_80x24` and `<name>_120x40`.
pub(super) fn snapshot(name: &str, app: &mut App) -> Result<(), Infallible> {
    snapshot_at(&format!("{name}_80x24"), app, 80, 24)?;
    snapshot_at(&format!("{name}_120x40"), app, 120, 40)
}

fn log(buffer: &LogBuffer, second: u64, level: Level, target: &str, message: &str) {
    buffer.push(LogLine {
        seq: 0,
        level,
        target: target.into(),
        time: at(NOW - 60 + second),
        message: message.into(),
    });
}

/// Log lines of every level.
fn logs(app: &App) {
    let buffer = &app.logs;
    log(
        buffer,
        1,
        Level::INFO,
        "overbrainer::cli::progress",
        "answers: 400 to process",
    );
    log(
        buffer,
        2,
        Level::DEBUG,
        "overbrainer::cli::progress",
        "answers: 3f1c done",
    );
    log(
        buffer,
        3,
        Level::WARN,
        "overbrainer::cli::progress",
        "answers: 77aa: 429 Too Many Requests; retrying in 4.0s",
    );
    log(
        buffer,
        4,
        Level::ERROR,
        "overbrainer::cli::progress",
        "answers: 77aa failed: the answer is not a JSON array of strings",
    );
    log(buffer, 5, Level::TRACE, "overbrainer::llm", "request sent");
    log(
        buffer,
        6,
        Level::INFO,
        "overbrainer::cli::progress",
        "answers: 40/400",
    );
}

#[test]
fn logs_of_mixed_levels() -> TestResult {
    let mut app = app();
    logs(&app);
    app.view = View::Logs;
    snapshot("logs", &mut app)?;
    Ok(())
}

#[test]
fn logs_filtered_at_warn() -> TestResult {
    let mut app = app();
    logs(&app);
    app.view = View::Logs;
    app.log_view.min = Level::WARN;
    snapshot("logs_warn", &mut app)?;
    Ok(())
}

#[test]
fn logs_at_trace_with_nothing_captured() -> TestResult {
    let mut app = app();
    app.view = View::Logs;
    app.log_view.min = Level::TRACE;
    snapshot("logs_empty_trace", &mut app)?;
    Ok(())
}

#[test]
fn the_help_overlay_lists_global_and_view_keys() -> TestResult {
    let mut app = app();
    app.view = View::Logs;
    app.overlay = Some(Overlay::Help);
    snapshot("help_logs", &mut app)?;
    Ok(())
}

#[test]
fn a_terminal_below_80x24_says_so() -> TestResult {
    let mut app = app();
    snapshot_at("too_small_79x24", &mut app, 79, 24)?;
    snapshot_at("too_small_80x23", &mut app, 80, 23)?;
    Ok(())
}

/// Every view drawn with the monochrome theme uses no color, and the selection
/// shows as reversed; `NO_COLOR` also turns motion off.
#[test]
fn the_monochrome_theme_uses_no_color() -> TestResult {
    let env = LookEnv {
        no_color: Some("1".into()),
        motion: Some("on".into()),
        ..LookEnv::default()
    };
    let level = ColorLevel::detect(&env);
    assert_eq!(level, ColorLevel::Mono);
    assert_eq!(MotionLevel::detect(&env, level), MotionLevel::Off);
    let mut app = dataset_app_with(&Theme::mono());
    app.config = Some(project_config()?);
    logs(&app);
    for view in View::ALL {
        app.view = view;
        for overlay in [None, Some(Overlay::Help)] {
            app.overlay = overlay.clone();
            let terminal = draw(&mut app, 120, 40)?;
            let buffer = terminal.backend().buffer();
            let colored = buffer
                .content()
                .iter()
                .filter(|cell| cell.fg != Color::Reset || cell.bg != Color::Reset)
                .count();
            assert_eq!(colored, 0, "{view:?}, {overlay:?}");
            assert!(
                buffer
                    .content()
                    .iter()
                    .any(|cell| cell.modifier.contains(Modifier::REVERSED)),
                "{view:?}: no reversed selection"
            );
        }
    }
    Ok(())
}

#[test]
fn dataset_with_every_topic_collapsed() -> TestResult {
    let mut app = dataset_app();
    snapshot("dataset", &mut app)?;
    Ok(())
}

#[test]
fn dataset_on_an_answered_question_with_its_reasoning() -> TestResult {
    let mut app = dataset_app();
    open_to(&mut app, &path_to(MOVED));
    snapshot("dataset_answered", &mut app)?;
    Ok(())
}

#[test]
fn dataset_on_an_excluded_question() -> TestResult {
    let mut app = dataset_app();
    open_to(
        &mut app,
        &path_to("Why can't a &mut and a & borrow coexist?"),
    );
    snapshot("dataset_question", &mut app)?;
    Ok(())
}

#[test]
fn dataset_stats_pane() -> TestResult {
    let mut app = dataset_app();
    app.dataset.stats = true;
    snapshot("dataset_stats", &mut app)?;
    Ok(())
}

/// A newer release: its marker follows the version in the footer.
#[test]
fn dataset_with_a_newer_release() -> TestResult {
    let mut app = dataset_app();
    app.newer = Some("0.9.0".into());
    snapshot_at("dataset_newer_120x40", &mut app, 120, 40)?;
    Ok(())
}

#[test]
fn dataset_with_a_filter() -> TestResult {
    let mut app = dataset_app();
    app.dataset.apply_filter("BORROW".into());
    app.dataset.tree.open(vec![Node::Topic("ownership".into())]);
    app.dataset.tree.open(vec![
        Node::Topic("ownership".into()),
        Node::Subtopic(Id::subtopic("ownership", "Borrowing")),
    ]);
    snapshot("dataset_filter", &mut app)?;
    Ok(())
}

/// A filter being typed: the footer says what Enter and Esc do with it.
#[test]
fn dataset_while_a_filter_is_typed() -> TestResult {
    let mut app = dataset_app();
    for code in [KeyCode::Char('/'), KeyCode::Char('b'), KeyCode::Char('o')] {
        app.on_input(&key(code));
    }
    snapshot("dataset_filter_typing", &mut app)?;
    let terminal = draw(&mut app, 80, 24)?;
    let rows = text(&terminal);
    let footer = rows.last().ok_or("no footer")?;
    assert!(footer.contains("Enter keep · Esc clear"), "{footer}");
    // The cursor, after "╰ /bo" on the pane's bottom border.
    let buffer = terminal.backend().buffer();
    assert!(buffer[(5, 22)].modifier.contains(Modifier::REVERSED));
    assert!(!buffer[(4, 22)].modifier.contains(Modifier::REVERSED));
    Ok(())
}

#[test]
fn dataset_with_missing_subtopic_and_unconfigured_topic() -> TestResult {
    let mut app = dataset_app();
    app.dataset.tree.open(vec![Node::Topic("ownership".into())]);
    app.dataset.tree.open(vec![Node::Topic("old_topic".into())]);
    app.dataset
        .tree
        .select(vec![Node::Topic("old_topic".into())]);
    snapshot("dataset_leftovers", &mut app)?;
    Ok(())
}

/// No data and no run: the empty tree says which key fills it, and the first
/// keys to know.
#[test]
fn dataset_empty_on_a_new_project() -> TestResult {
    let mut app = app();
    app.project.topics = topics();
    app.dataset.loaded(Dataset::default(), &app.project.topics);
    snapshot("dataset_empty", &mut app)?;
    let rows = text(&draw(&mut app, 120, 40)?).join("\n");
    assert!(rows.contains("No data yet: press r to generate"), "{rows}");
    assert!(
        rows.contains("r generates data · t trains · ? all keys"),
        "{rows}"
    );
    let rows = text(&draw(&mut app, 80, 24)?);
    let listed = |y: usize, label: &str| rows.get(y).is_some_and(|row| row.contains(label));
    assert!(listed(2, "ownership  0 sub, 0 q, 0 a"), "{rows:#?}");
    assert!(listed(3, "traits  0 sub, 0 q, 0 a"), "{rows:#?}");
    let keys = rows
        .iter()
        .find(|row| row.contains("r generates data"))
        .ok_or("no first keys")?;
    assert!(
        keys.contains("r generates data · t trains ") && !keys.contains('?'),
        "whole hints only: {keys}"
    );
    Ok(())
}

/// A topic's counts that do not fit whole are left out, never cut.
#[test]
fn a_topic_whose_counts_do_not_fit_shows_none() -> TestResult {
    let mut app = dataset_app();
    let rows = text(&draw(&mut app, 80, 24)?);
    let old = rows
        .iter()
        .find(|row| row.contains("old_topic"))
        .ok_or("no old_topic")?;
    assert!(
        old.starts_with("│ ▶ old_topic (not configured)     │"),
        "{old}"
    );
    let rows = text(&draw(&mut app, 120, 40)?).join("\n");
    assert!(
        rows.contains("old_topic (not configured)  1 sub, 1 q, 1 a"),
        "{rows}"
    );
    Ok(())
}

#[test]
fn dataset_load_error() -> TestResult {
    let mut app = app();
    let error = "data/answers.jsonl:3: invalid record: expected value at line 1 column 2";
    let Some(Effect::Spawn(id, _)) = app.start().first().cloned() else {
        return Err("no load started".into());
    };
    app.on_done(
        id,
        Ok(Done::Loaded(Err(error.into()), Ok(History::default()))),
    );
    app.status = None;
    snapshot("dataset_error", &mut app)?;
    Ok(())
}

#[test]
fn the_delete_dialog_says_what_goes_with_a_subtopic() -> TestResult {
    let mut app = dataset_app();
    open_to(&mut app, &path_to(MOVED)[..2]);
    app.on_input(&key(KeyCode::Char('d')));
    assert!(matches!(app.overlay, Some(Overlay::Confirm(_))));
    snapshot("dataset_delete", &mut app)?;
    Ok(())
}

#[test]
fn pipeline_before_any_run() -> TestResult {
    let mut app = app();
    app.view = View::Pipeline;
    snapshot("pipeline_idle", &mut app)?;
    Ok(())
}

#[test]
fn pipeline_running_with_retries_and_errors() -> TestResult {
    let mut app = app();
    app.view = View::Pipeline;
    pipeline_running(&mut app);
    snapshot("pipeline_running", &mut app)?;
    app.pipeline.skipped = 312;
    snapshot("pipeline_lagged", &mut app)?;
    Ok(())
}

#[test]
fn pipeline_finished_with_results() -> TestResult {
    let mut app = app();
    app.view = View::Pipeline;
    app.pipeline.started(Command::Answers, 8);
    for event in stage_events(Stage::Answers, 4, 1) {
        app.pipeline.event(&event);
    }
    app.pipeline.event(&Event::StageFinished {
        stage: Stage::Answers,
        stats: StageStats {
            done: 4,
            usage: usage(4000, 7360),
            cost: Some(0.0431),
            ..StageStats::default()
        },
    });
    app.pipeline.results.push(
        "answers: 4 done, 0 skipped, 0 failed, 0 excluded; tokens 4000 in, 7360 out; cost $0.0431"
            .into(),
    );
    app.pipeline.running = false;
    app.pipeline.outcome = Some(Ok(()));
    snapshot("pipeline_finished", &mut app)?;
    Ok(())
}

/// A stopped task: the answers row says so, nothing is in flight.
#[test]
fn pipeline_stopped_by_quitting() -> TestResult {
    let mut app = app();
    app.view = View::Pipeline;
    pipeline_running(&mut app);
    app.on_done(
        TaskId(7),
        Ok(Done::Pipeline(Err(
            "interrupted: the stage resumes on its next run".into(),
        ))),
    );
    app.status = None;
    snapshot("pipeline_stopped", &mut app)?;
    Ok(())
}

/// Progress stays readable without color: bars are blocks, stages have glyphs.
#[test]
fn progress_shows_without_color() -> TestResult {
    let mut app = app();
    app.theme = Theme::mono();
    app.view = View::Pipeline;
    pipeline_running(&mut app);
    let rows = text(&draw(&mut app, 80, 24)?);
    let answers = rows
        .iter()
        .find(|row| row.contains("answers") && row.contains("120/400"))
        .ok_or("no answers row")?;
    assert!(answers.contains("███████▌ "), "{answers}");
    assert!(
        rows.iter().any(|row| row.starts_with("  ✓ subtopics")),
        "{rows:#?}"
    );
    assert!(
        rows.iter()
            .any(|row| row.starts_with("  · split       pending")),
        "{rows:#?}"
    );
    let mut app = training_app()?;
    app.theme = Theme::mono();
    let rows = text(&draw(&mut app, 80, 24)?).join("\n");
    assert!(
        rows.contains("● 20260921-133200-a1b2  step 1200/4000 ████▊"),
        "{rows}"
    );
    Ok(())
}

/// A failed `run` at the minimum size still shows its final error and its
/// newest item failures: older failures give way first.
#[test]
fn a_failed_run_keeps_its_error_in_view_at_80x24() -> TestResult {
    let mut app = app();
    app.view = View::Pipeline;
    pipeline_running(&mut app);
    app.pipeline.skipped = 312;
    for line in [
        "answers: 400 done, 0 skipped, 1 failed, 0 excluded; tokens 400000 in, 100000 out; cost $0.4000",
        "split: 399 train, 1 eval, 0 excluded, 0 orphaned",
    ] {
        app.pipeline.results.push(line.into());
    }
    app.pipeline_task = None;
    app.pipeline.running = false;
    app.pipeline.outcome = Some(Err("1 item failed; it is retried on the next run".into()));
    let rows = text(&draw(&mut app, 80, 24)?);
    assert!(
        rows.iter()
            .any(|row| row.contains("1 item failed; it is retried on the next run")),
        "{rows:#?}"
    );
    assert!(
        rows.iter().any(|row| row.contains("77aa…10bc")),
        "{rows:#?}"
    );
    assert!(
        !rows.iter().any(|row| row.contains("3f1c…000a")),
        "{rows:#?}"
    );
    Ok(())
}

/// The app with the picker of `kind` open, `chosen` preselected; returns the
/// task reading its entries.
fn picker_app(
    kind: super::catalog::CatalogKind,
    chosen: &[&str],
) -> Result<(App, TaskId), Box<dyn std::error::Error>> {
    let mut app = app();
    let chosen = chosen.iter().map(|id| (*id).to_string()).collect();
    let query = super::catalog::Query {
        kind,
        gpu_count: 2,
        gpu_types: Vec::new(),
        fit_by: super::catalog::FitBy::Nothing,
    };
    let origin = super::app::Origin::Field(crate::config::edit::FieldPath::Target {
        name: "gpu_cloud".into(),
        field: "gpu_types",
    });
    let choice = super::widgets::picker::Choice::List(chosen);
    let effects = app.open_picker(query, choice, origin);
    match effects.as_slice() {
        [Effect::Spawn(id, _)] => Ok((app, *id)),
        _ => Err(format!("{effects:?}").into()),
    }
}

#[test]
fn the_gpu_picker_on_a_fixture_catalog() -> TestResult {
    let (mut app, id) = picker_app(
        super::catalog::CatalogKind::Gpus,
        &["NVIDIA A40", "NVIDIA GeForce RTX 4090"],
    )?;
    let rows = text(&draw(&mut app, 80, 24)?);
    assert!(
        rows.iter()
            .any(|row| row.contains("reading the Runpod catalog")),
        "{rows:#?}"
    );
    app.on_done(id, Ok(Done::Catalog(Ok(gpu_listing(2)?))));
    app.on_input(&key(KeyCode::Down));
    app.on_input(&key(KeyCode::Down));
    snapshot("picker_gpus", &mut app)?;
    Ok(())
}

/// `o` sorts the GPU picker by VRAM, most first; the title says so.
#[test]
fn the_gpu_picker_sorted_by_vram() -> TestResult {
    let (mut app, id) = picker_app(super::catalog::CatalogKind::Gpus, &["NVIDIA A40"])?;
    app.on_done(id, Ok(Done::Catalog(Ok(gpu_listing(2)?))));
    app.on_input(&key(KeyCode::Char('o')));
    snapshot_at("picker_gpus_by_vram_80x24", &mut app, 80, 24)?;
    Ok(())
}

#[test]
fn a_picker_filtered_while_typed() -> TestResult {
    let (mut app, id) = picker_app(super::catalog::CatalogKind::Gpus, &[])?;
    app.on_done(id, Ok(Done::Catalog(Ok(gpu_listing(2)?))));
    for c in "/h1".chars() {
        app.on_input(&key(KeyCode::Char(c)));
    }
    let rows = text(&draw(&mut app, 80, 24)?);
    let listed: Vec<&String> = rows.iter().filter(|row| row.contains("NVIDIA")).collect();
    assert_eq!(listed.len(), 1, "{rows:#?}");
    assert!(listed[0].contains("NVIDIA H100 80GB HBM3"), "{rows:#?}");
    assert!(rows.iter().any(|row| row.contains("/ h1")), "{rows:#?}");
    assert!(
        rows.iter()
            .any(|row| row.contains("Enter keep · Esc clear"))
    );
    Ok(())
}

#[test]
fn a_picker_whose_catalog_cannot_be_read() -> TestResult {
    let (mut app, id) = picker_app(super::catalog::CatalogKind::Volumes, &["vol-1"])?;
    let error =
        "cannot read the Runpod catalog: no Runpod API key: set OVERBRAINER_RUNPOD__API_KEY";
    app.on_done(id, Ok(Done::Catalog(Err(error.into()))));
    snapshot("picker_error", &mut app)?;
    Ok(())
}

#[test]
fn the_run_menu() -> TestResult {
    let mut app = app();
    app.on_input(&key(KeyCode::Char('r')));
    for _ in 0..3 {
        app.on_input(&key(KeyCode::Down));
    }
    snapshot("run_menu", &mut app)?;
    Ok(())
}

#[test]
fn the_quit_dialog_says_what_becomes_of_the_work() -> TestResult {
    let mut app = app();
    pipeline_running(&mut app);
    app.edit = Some(TaskId(8));
    app.on_input(&key(KeyCode::Char('q')));
    snapshot("quit_dialog", &mut app)?;
    Ok(())
}

use crate::runs::RunState;
use crate::tui::training::{Ended, Follow, Job, RunRow};

const FOLLOWED: &str = "20260921-133200-a1b2";
pub(super) const FINISHED: &str = "20260920-101500-9f00";
const LEFT: &str = "20260919-090000-c3d4";

/// Three runs: a Runpod run followed live, a finished local run, and a run left
/// running that nothing follows.
pub(super) fn training_app() -> Result<App, serde_json::Error> {
    let mut app = app();
    app.view = View::Training;
    app.training.runs = vec![
        RunRow {
            record: run(FOLLOWED, "gpu_cloud", RunState::Running),
            pod: Some(pod(FOLLOWED)?),
        },
        RunRow {
            record: run(FINISHED, "homelab", RunState::Succeeded),
            pod: None,
        },
        RunRow {
            record: run(LEFT, "homelab", RunState::Running),
            pod: None,
        },
    ];
    let mut follow = Follow::new(Job::Attach, FOLLOWED);
    follow.watching = true;
    app.training.tasks.insert(TaskId(3), follow);
    app.training.series.insert(FOLLOWED.into(), series());
    Ok(app)
}

#[test]
fn training_without_runs() -> TestResult {
    let mut app = app();
    app.view = View::Training;
    snapshot("training_empty", &mut app)?;
    Ok(())
}

#[test]
fn training_of_a_followed_runpod_run() -> TestResult {
    let mut app = training_app()?;
    snapshot("training_followed", &mut app)?;
    Ok(())
}

#[test]
fn a_step_past_max_steps_reads_100_percent() -> TestResult {
    let mut app = training_app()?;
    let mut metrics = series();
    for metric in &mut metrics {
        metric.max_steps = Some(1000);
    }
    app.training.series.insert(FOLLOWED.into(), metrics);
    let shown = text(&draw(&mut app, 120, 40)?).join("\n");
    assert!(shown.contains("step 1200/1000"), "{shown}");
    assert!(shown.contains(" 100%"), "{shown}");
    assert!(!shown.contains("120%"), "{shown}");
    Ok(())
}

#[test]
fn the_run_column_fits_the_longest_run_id() -> TestResult {
    let mut app = training_app()?;
    let long = "malware_development_20260930-120000_2";
    app.training.runs[2].record = run(long, "homelab", RunState::Running);
    let shown = text(&draw(&mut app, 120, 40)?).join("\n");
    assert!(shown.contains(&format!("{long}  running")), "{shown}");
    assert!(
        shown.contains(&format!("{FOLLOWED}                   running")),
        "{shown}"
    );
    Ok(())
}

#[test]
fn training_of_a_finished_local_run() -> TestResult {
    let mut app = training_app()?;
    app.training.selected = 1;
    let mut metrics = series();
    metrics.truncate(40);
    app.training.series.insert(FINISHED.into(), metrics);
    app.training.ended.insert(
        FINISHED.into(),
        Ended {
            lines: vec![
                "train: run 20260920-101500-9f00 succeeded; 400 steps in 13m; loss 0.91; \
                 output in runs/20260920-101500-9f00/output"
                    .into(),
            ],
            error: None,
            ..Ended::default()
        },
    );
    snapshot("training_finished", &mut app)?;
    Ok(())
}

#[test]
fn training_of_a_run_nothing_follows() -> TestResult {
    let mut app = training_app()?;
    app.training.selected = 2;
    snapshot("training_not_followed", &mut app)?;
    Ok(())
}

/// Samples of a pod's machine, one every 10 seconds up to [`NOW`]: the
/// network volume on the whole shared cluster, the container's disk filling
/// up, half its 8 CPUs busy, and `gpus` GPUs, the first nearly out of memory
/// and the second at full use.
fn machine(count: u64, gpus: u32) -> std::collections::VecDeque<crate::system::SystemSample> {
    use std::fmt::Write as _;

    let mut samples = std::collections::VecDeque::new();
    for index in 0..count {
        let mut output = String::from("@gpu\n");
        for gpu in 0..gpus {
            let (utilization, used) = if gpu == 1 {
                (99, 40_960)
            } else {
                (60 + index, 79_000)
            };
            writeln!(
                output,
                "{gpu}, NVIDIA H100 80GB HBM3, {utilization}, {used}, 81920, {}, 312.4, 700",
                60 + gpu
            )
            .ok();
        }
        write!(
            output,
            "@loadavg\n96.40 90.00 80.00 1/1 1\n@uptime\n{uptime}.00 1.00\n\
             @cpu.max\n800000 100000\n@cpu.stat\nusage_usec {usage}\n\
             @nproc\n128\n@meminfo\nMemTotal: 134217728 kB\nMemAvailable: 83886080 kB\n\
             @df.run\nFilesystem Type 1024-blocks Used Available Capacity Mounted on\n\
             mfs#euro-3.runpod.net:9421 fuse.mfs 2343372656 1827830672 515542000 78% /workspace\n\
             @df.root\nFilesystem Type 1024-blocks Used Available Capacity Mounted on\n\
             overlay overlay 20971520 {used} {free} 0% /\n",
            uptime = 1000 + index * 10,
            usage = index * 40_000_000,
            used = 17_150_000 + index * 100_000,
            free = 3_821_520 - index * 100_000,
        )
        .ok();
        let at = crate::tui::snapshots::at(NOW - (count - 1 - index) * 10);
        let sample = crate::system::parse(&output, at, samples.back());
        samples.push_back(sample);
    }
    samples
}

#[test]
fn training_with_the_system_panel() -> TestResult {
    let mut app = training_app()?;
    app.training.system.insert(FOLLOWED.into(), machine(12, 2));
    snapshot_at("training_system_120x40", &mut app, 120, 40)?;
    let shown = text(&draw(&mut app, 120, 40)?).join("\n");
    assert!(shown.contains("gpu1 ██████  99%"), "{shown}");
    // Inside a container: its cores busy, not the host's load.
    assert!(shown.contains("4.0/8.0 cores"), "{shown}");
    assert!(!shown.contains("load"), "{shown}");
    // Too narrow: no panel, as before.
    let narrow = text(&draw(&mut app, 80, 24)?).join("\n");
    assert!(!narrow.contains("system"), "{narrow}");
    Ok(())
}

#[test]
fn the_system_panel_shows_only_where_it_fits() -> TestResult {
    let mut app = training_app()?;
    // Two disks and four GPUs: the tallest panel without folding.
    app.training.system.insert(FOLLOWED.into(), machine(12, 4));
    for (width, height) in [(100, 30), (100, 24), (120, 24), (120, 30)] {
        snapshot_at(
            &format!("training_system_gpus_{width}x{height}"),
            &mut app,
            width,
            height,
        )?;
        let shown = text(&draw(&mut app, width, height)?).join("\n");
        // From 120 columns, and only with rows left for the run's detail.
        let expected = width >= 120 && height >= 30;
        assert_eq!(
            shown.contains("╭ system"),
            expected,
            "{width}x{height}\n{shown}"
        );
    }
    Ok(())
}

#[test]
fn the_system_panel_warns_as_its_gauges_fill() -> TestResult {
    let mut app = training_app()?;
    app.training.system.insert(FOLLOWED.into(), machine(12, 2));
    let terminal = draw(&mut app, 120, 40)?;
    let rows = text(&terminal);
    let buffer = terminal.backend().buffer();
    let style_of = |needle: &str| -> Result<ratatui::style::Style, Box<dyn std::error::Error>> {
        let (y, line) = rows
            .iter()
            .enumerate()
            .find(|(_, line)| line.contains(needle))
            .ok_or(format!("no {needle}"))?;
        let start = line.find(needle).ok_or("gone")?;
        let x = line[..start].chars().count() + needle.chars().count() - 1;
        let cell = buffer
            .cell((u16::try_from(x)?, u16::try_from(y)?))
            .ok_or("no cell")?;
        Ok(cell.style())
    };
    let theme = app.theme;
    // The container's disk at 87%, the second GPU at 99%, the CPU below both.
    assert_eq!(style_of("87%")?.fg, theme.warn.fg);
    // The network volume: the whole cluster's share, dim, marked shared.
    assert_eq!(style_of("78%")?.fg, theme.dim.fg);
    assert_eq!(style_of("shared")?.fg, theme.dim.fg);
    assert_eq!(style_of("99%")?.fg, theme.error.fg);
    assert_eq!(style_of("cpu  ")?.fg, theme.dim.fg);
    // The first GPU's memory is nearly full: its figures turn red.
    assert_eq!(style_of("77/80G")?.fg, theme.error.fg);
    Ok(())
}

#[test]
fn the_system_panel_shows_the_age_of_an_old_sample() -> TestResult {
    let mut app = training_app()?;
    app.training.selected = 2;
    let shown = text(&draw(&mut app, 120, 40)?).join("\n");
    assert!(shown.contains("no sample: not followed"), "{shown}");
    let mut samples = machine(3, 6);
    for sample in &mut samples {
        sample.at -= Duration::from_secs(300);
    }
    app.training.system.insert(LEFT.into(), samples);
    let shown = text(&draw(&mut app, 120, 40)?).join("\n");
    assert!(shown.contains(" system · 5m ago "), "{shown}");
    assert!(shown.contains("rest"), "{shown}");
    assert!(shown.contains("avg of 3"), "{shown}");
    assert!(!shown.contains("gpu3"), "{shown}");
    // Not followed, but sampled a moment ago: no age.
    for sample in app.training.system.get_mut(LEFT).into_iter().flatten() {
        sample.at += Duration::from_secs(300);
    }
    let shown = text(&draw(&mut app, 120, 40)?).join("\n");
    assert!(shown.contains("╭ system ─"), "{shown}");
    // Followed, before the first sample.
    app.training.selected = 0;
    let shown = text(&draw(&mut app, 120, 40)?).join("\n");
    assert!(shown.contains("waiting for the first sample"), "{shown}");
    // Followed, but no sample for two rounds: the age shows.
    let mut late = machine(3, 1);
    for sample in &mut late {
        sample.at -= Duration::from_secs(25);
    }
    app.training.system.insert(FOLLOWED.into(), late);
    let shown = text(&draw(&mut app, 120, 40)?).join("\n");
    assert!(shown.contains(" system · 25s ago "), "{shown}");
    Ok(())
}

/// The pod line of the followed run, whose pod's last event is `ready` while
/// `pod.json` holds `record`, drawn at 120x40.
fn pod_line_of(record: crate::runpod::PodRecord) -> Result<String, Box<dyn std::error::Error>> {
    let mut app = training_app()?;
    app.training.runs[0].pod = Some(record);
    if let Some(follow) = app.training.tasks.get_mut(&TaskId(3)) {
        follow.pod = Some(crate::runpod::PodStatus::Ready {
            pod_id: crate::runpod::PodId::new("k3x9abc")?,
            after: Duration::from_secs(180),
            deadline: Some("2026-09-21T19:32:20Z".into()),
        });
    }
    let rows = text(&draw(&mut app, 120, 40)?);
    let line = rows
        .iter()
        .find(|row| row.trim_start().starts_with("pod k3x9abc"))
        .ok_or("no pod line")?;
    Ok(line.trim().to_string())
}

/// No event marks the job's start, so the pod line reads the state `pod.json`
/// records, not the pod's last event: a pod whose job runs reads `running`.
#[test]
fn the_pod_line_reads_the_recorded_state_over_the_last_event() -> TestResult {
    let mut record = pod(FOLLOWED)?;
    record.state = crate::runpod::PodState::Running;
    let line = pod_line_of(record)?;
    assert!(
        line.starts_with("pod k3x9abc running $0.53/h  up 41m"),
        "{line}"
    );
    Ok(())
}

/// A pod `pod.json` records deleted reads `deleted`, with its final uptime and
/// spend, whatever its last event said.
#[test]
fn a_deleted_pod_reads_deleted_with_its_final_uptime_and_spend() -> TestResult {
    let mut record = pod(FOLLOWED)?;
    record.deleted(crate::runpod::DeletedBy::Watchdog, at(NOW - 60));
    let line = pod_line_of(record)?;
    assert_eq!(
        line,
        "pod k3x9abc deleted at 2026-09-21T14:12:20Z  up 40m  spent about $0.35"
    );
    Ok(())
}

/// A failed Runpod run whose pod line is stale.
const BROKEN: &str = "20260918-080000-dead";

/// `x` asks, then leaves the failed runs out of the list until the TUI
/// restarts: a later read of `runs/` keeps them out. With none, it says so.
#[test]
fn x_clears_the_failed_runs_from_the_list_after_asking() -> TestResult {
    let mut app = training_app()?;
    app.training.runs.push(RunRow {
        record: run(BROKEN, "gpu_cloud", RunState::Failed),
        pod: Some(pod(BROKEN)?),
    });
    let rows = app.training.runs.clone();
    app.training.selected = 3;
    app.on_input(&key(KeyCode::Char('x')));
    snapshot("training_clear_failed", &mut app)?;
    app.on_input(&key(KeyCode::Char('y')));
    let ids = |app: &App| -> Vec<String> {
        app.training
            .runs
            .iter()
            .map(|row| row.record.id.clone())
            .collect()
    };
    assert_eq!(ids(&app), [FOLLOWED, FINISHED, LEFT]);
    assert_eq!(app.training.selected, 2);
    app.training.listed(crate::tui::training::Listing {
        runs: Ok(rows),
        skipped: Vec::new(),
    });
    assert_eq!(ids(&app), [FOLLOWED, FINISHED, LEFT], "still cleared");
    app.on_input(&key(KeyCode::Char('x')));
    assert!(app.overlay.is_none());
    let said = app.status.as_ref().map(|status| status.text.as_str());
    assert_eq!(said, Some("no failed run to clear"));
    Ok(())
}

/// `p` dismisses the pod of a run nothing follows, line and column, and shows
/// it again; a followed run's pod stays, and following a run shows its pod.
#[test]
fn p_dismisses_the_pod_of_a_run_nothing_follows() -> TestResult {
    let mut app = training_app()?;
    let shows = |app: &mut App| -> Result<bool, Infallible> {
        let rows = text(&draw(app, 120, 40)?);
        Ok(rows.iter().any(|row| row.contains("k3x9abc")))
    };
    app.on_input(&key(KeyCode::Char('p')));
    assert!(shows(&mut app)?, "followed: the pod stays");
    app.training.tasks.clear();
    app.on_input(&key(KeyCode::Char('p')));
    assert!(!shows(&mut app)?, "dismissed");
    app.on_input(&key(KeyCode::Char('p')));
    assert!(shows(&mut app)?, "shown again");
    app.on_input(&key(KeyCode::Char('p')));
    app.on_input(&key(KeyCode::Char('a')));
    assert!(shows(&mut app)?, "followed again: the pod shows");
    Ok(())
}

#[test]
fn the_cancel_dialog_and_the_quit_dialog_of_a_followed_run() -> TestResult {
    let mut app = training_app()?;
    app.on_input(&key(KeyCode::Char('c')));
    snapshot("training_cancel", &mut app)?;
    app.overlay = None;
    app.on_input(&key(KeyCode::Char('q')));
    snapshot("training_quit", &mut app)?;
    Ok(())
}

use crate::tui::app::{Action, Confirm};
use crate::tui::start::{RunpodPlan, StartPlan};

/// The plan of a run on the Runpod target `gpu_cloud`.
pub(super) fn runpod_plan() -> StartPlan {
    StartPlan {
        target: "gpu_cloud".into(),
        kind: "runpod, Secure Cloud".into(),
        model: "Qwen/Qwen3-4B, qlora, 3 epochs, lr 2e-4".into(),
        train: 1234,
        eval: 137,
        runpod: Some(Box::new(RunpodPlan::new(crate::tui::start::runpod_spec(
            ListOrAuto::List(vec![
                "NVIDIA GeForce RTX 4090".into(),
                "NVIDIA RTX A6000".into(),
                "NVIDIA A40".into(),
            ]),
            1,
        )))),
        warnings: vec!["qwen2 renders no reasoning_content (see the logs)".into()],
    }
}

#[test]
fn the_start_dialog_of_a_runpod_run_before_and_with_its_prices() -> TestResult {
    let mut app = app();
    app.view = View::Training;
    app.prepared(Ok(runpod_plan()));
    snapshot("start_runpod_looking_up", &mut app)?;
    let mut gpus = gpu_types()?;
    gpus.retain(|gpu| gpu.id != "NVIDIA A40");
    app.start_catalog_read(looked_up(Ok(gpus)));
    snapshot("start_runpod", &mut app)?;
    Ok(())
}

#[test]
fn the_start_dialog_of_an_ssh_run() -> TestResult {
    let mut app = app();
    app.view = View::Training;
    let plan = StartPlan {
        target: "homelab".into(),
        kind: "ssh, docker".into(),
        runpod: None,
        warnings: Vec::new(),
        ..runpod_plan()
    };
    app.overlay = Some(Overlay::Confirm(Confirm {
        title: " Start a training run? ".into(),
        text: crate::tui::start::text(&plan, None),
        yes: "start",
        no: "cancel",
        action: Action::Start(Box::new(plan)),
    }));
    snapshot("start_ssh", &mut app)?;
    Ok(())
}

#[test]
fn quitting_while_a_runpod_run_provisions_offers_to_abandon_it() -> TestResult {
    let mut app = app();
    app.view = View::Training;
    app.training
        .tasks
        .insert(TaskId(4), Follow::new(Job::Start { runpod: true }, ""));
    app.on_input(&key(KeyCode::Char('q')));
    snapshot("quit_starting", &mut app)?;
    app.on_input(&key(KeyCode::Char('y')));
    snapshot("abandon_dialog", &mut app)?;
    Ok(())
}

#[test]
fn c_on_a_starting_runpod_run_offers_to_abandon_it() -> TestResult {
    let mut app = training_app()?;
    app.training.tasks.insert(
        TaskId(3),
        Follow::new(Job::Start { runpod: true }, FOLLOWED),
    );
    // No job yet, so no metric.
    app.training.series.remove(FOLLOWED);
    app.on_input(&key(KeyCode::Char('c')));
    snapshot("abandon_starting_run", &mut app)?;
    Ok(())
}

/// A start dialog switched to `auto`, too tall for 80x24 with its warnings
/// and what auto picks now, keeps what `y` saves and the most the run costs.
#[test]
fn a_changed_start_keeps_what_it_saves_and_its_cost() -> TestResult {
    let mut app = app();
    app.view = View::Training;
    let mut plan = runpod_plan();
    if let Some(runpod) = &mut plan.runpod {
        runpod.choose_gpus(ListOrAuto::Auto);
    }
    plan.warnings = (1..=4)
        .map(|n| format!("warning number {n} about this run"))
        .collect();
    app.prepared(Ok(plan));
    app.start_catalog_read(looked_up(Ok(gpu_types()?)));
    snapshot("start_runpod_changed", &mut app)?;
    // More warnings than fit: the changed line would be cut, it is kept.
    let many = (5..=11).map(|n| format!("warning     warning number {n} about this run"));
    if let Some(Overlay::Confirm(confirm)) = &mut app.overlay {
        confirm.text.splice(7..7, many);
    }
    let rows = text(&draw(&mut app, 80, 24)?).join("\n");
    assert!(!rows.contains("warning number 11"), "cut: {rows}");
    for shown in [
        "changed     gpu_types: saved to overbrainer.toml on y",
        "max_hours   6",
        "…",
        "y start",
    ] {
        assert!(rows.contains(shown), "{shown}\n{rows}");
    }
    Ok(())
}

/// A start dialog taller than the terminal keeps its key line and its
/// most-it-can-cost line, and marks the text it cut; so does a quit dialog.
#[test]
fn a_tall_dialog_keeps_its_keys_and_its_cost_at_80x24() -> TestResult {
    let mut app = app();
    app.view = View::Training;
    let mut plan = runpod_plan();
    let gpus: Vec<String> = (1..=14).map(|n| format!("NVIDIA GPU model {n}")).collect();
    if let Some(runpod) = &mut plan.runpod {
        runpod.spec.gpu_types = ListOrAuto::List(gpus.clone());
    }
    plan.warnings = (1..=4)
        .map(|n| format!("warning number {n} about this run"))
        .collect();
    app.prepared(Ok(plan));
    let listed = gpus
        .iter()
        .map(|gpu| serde_json::json!({"id": gpu, "memory": 48, "price": {"secure": 0.5}}))
        .collect();
    app.start_catalog_read(looked_up(Ok(serde_json::from_value(
        serde_json::Value::Array(listed),
    )?)));
    let rows = text(&draw(&mut app, 80, 24)?).join("\n");
    for shown in [
        "warning     warning number 1 about this run",
        "warning     warning number 4 about this run",
        "y start",
        "n cancel",
        "max_hours   6",
        "…",
        "target      gpu_cloud",
    ] {
        assert!(rows.contains(shown), "{shown}\n{rows}");
    }
    let mut app = self::app();
    for n in 0..20 {
        app.training.tasks.insert(
            TaskId(n),
            Follow::new(Job::Attach, &format!("20260921-1332{n:02}-a1b2")),
        );
    }
    app.on_input(&key(KeyCode::Char('q')));
    let rows = text(&draw(&mut app, 80, 24)?).join("\n");
    for shown in ["y quit", "n stay", "…"] {
        assert!(rows.contains(shown), "{shown}\n{rows}");
    }
    Ok(())
}

/// The help overlay of every view keeps its note whole at the minimum size.
#[test]
fn the_help_note_fits_every_view_at_80x24() -> TestResult {
    let mut app = dataset_app();
    app.overlay = Some(Overlay::Help);
    for view in View::ALL {
        app.view = view;
        let rows = text(&draw(&mut app, 80, 24)?).join("\n");
        assert!(
            rows.contains("e, d, r, A and t are refused"),
            "{view:?}\n{rows}"
        );
        assert!(
            rows.contains("writes to the project meanwhile."),
            "{view:?}\n{rows}"
        );
    }
    Ok(())
}

/// [`training_app`] with its Runpod run ended and followed by no task, its pod
/// changed by `change`.
fn ended_runpod_app(
    change: impl FnOnce(&mut crate::runpod::PodRecord),
) -> Result<App, serde_json::Error> {
    let mut app = training_app()?;
    app.training.tasks.clear();
    app.training.series.clear();
    if let Some(row) = app.training.runs.first_mut() {
        row.record.state = RunState::Succeeded;
        if let Some(pod) = row.pod.as_mut() {
            change(pod);
        }
    }
    Ok(app)
}

#[test]
fn training_of_a_run_whose_pod_was_deleted() -> TestResult {
    let mut app = ended_runpod_app(|pod| {
        pod.deleted(crate::runpod::DeletedBy::Client, at(NOW - 5 * 60));
    })?;
    snapshot("training_deleted_pod", &mut app)?;
    Ok(())
}

#[test]
fn training_of_a_run_whose_pod_is_kept() -> TestResult {
    let mut app = ended_runpod_app(|pod| {
        pod.keep = true;
        pod.state = crate::runpod::PodState::Kept;
    })?;
    snapshot("training_kept_pod", &mut app)?;
    Ok(())
}

/// The pod line of `app` drawn at 120x40: the row that starts with `pod ` and
/// the row under it, joined as one text.
fn pod_text(app: &mut App) -> Result<String, Infallible> {
    let rows = text(&draw(app, 120, 40)?);
    let start = rows
        .iter()
        .position(|row| row.trim_start().starts_with("pod "))
        .unwrap_or(rows.len());
    let words: Vec<&str> = rows
        .iter()
        .skip(start)
        .take(2)
        .map(|row| row.trim())
        .collect();
    Ok(words.join(" "))
}

#[test]
fn a_pod_asked_to_be_kept_is_kept_only_once_its_job_starts() -> TestResult {
    let mut app = training_app()?;
    app.training.tasks.clear();
    let pod = app
        .training
        .runs
        .first_mut()
        .and_then(|row| row.pod.as_mut())
        .ok_or("no pod")?;
    pod.keep = true;
    let line = pod_text(&mut app)?;
    assert!(
        line.contains("kept once its job starts  the watchdog deletes it by 2026-09-21T19:32:20Z"),
        "{line}"
    );
    assert!(!line.contains("no time limit"), "{line}");
    if let Some(pod) = app
        .training
        .runs
        .first_mut()
        .and_then(|row| row.pod.as_mut())
    {
        pod.state = crate::runpod::PodState::Running;
    }
    let line = pod_text(&mut app)?;
    assert!(line.contains("kept, no time limit"), "{line}");
    assert!(!line.contains("watchdog"), "{line}");
    Ok(())
}

/// A history entry of `stage` by `model`: `tokens` in and out, for `cost`.
fn history_entry(
    stage: Stage,
    model: &str,
    (input_tokens, output_tokens): (u64, u64),
    cost: Option<f64>,
) -> crate::history::Entry {
    crate::history::Entry {
        stage,
        started_at: "2026-09-21T12:00:00Z".into(),
        ended_at: "2026-09-21T12:10:00Z".into(),
        status: crate::history::Status::Ok,
        provider: Some("nanogpt".into()),
        model: Some(model.into()),
        done: 10,
        skipped: 0,
        failed: 0,
        excluded: 0,
        input_tokens,
        output_tokens,
        cost,
        split: None,
        backfilled: false,
    }
}

/// The Project view on [`project_config`], with [`dataset`], a history of
/// three stages by two models, and the runs of [`training_app`], none followed.
pub(super) fn project_app() -> Result<App, Box<dyn std::error::Error>> {
    let mut app = training_app()?;
    app.training.tasks.clear();
    app.view = View::Project;
    app.config = Some(project_config()?);
    app.project.topics = topics();
    app.dataset.loaded(dataset(), &app.project.topics);
    app.history = History::of(&[
        history_entry(Stage::Subtopics, "gen", (3000, 750), Some(0.0012)),
        history_entry(Stage::Questions, "gen", (40_210, 9120), Some(0.0123)),
        history_entry(
            Stage::Answers,
            "claude-opus-5-20260901-long-name",
            (412_000, 1_530_000),
            None,
        ),
    ]);
    app.history_cost = app.history.cost;
    Ok(app)
}

#[test]
fn project_view_with_its_configuration_and_stats() -> TestResult {
    let mut app = project_app()?;
    snapshot("project", &mut app)?;
    Ok(())
}

#[test]
fn project_view_while_a_stage_uses_the_configuration() -> TestResult {
    let mut app = project_app()?;
    app.pipeline_task = Some(TaskId(7));
    app.pipeline.started(Command::Answers, 8);
    // Down to `roles.parent.model`, which `answers` uses.
    for _ in 0..19 {
        app.on_input(&key(KeyCode::Char('j')));
    }
    snapshot("project_locked", &mut app)?;
    let terminal = draw(&mut app, 120, 40)?;
    let rows = text(&terminal);
    let y = rows
        .iter()
        .position(|row| row.contains("model") && row.contains("claude-opus-5"))
        .ok_or("no parent model row")?;
    let row = rows.get(y).ok_or("no row")?;
    assert!(row.contains("(used by answers)"), "{row}");
    let x = row
        .find("claude-opus-5")
        .map(|at| row[..at].chars().count());
    let x = u16::try_from(x.ok_or("no value")?)?;
    let cell = terminal
        .backend()
        .buffer()
        .cell((x, u16::try_from(y)?))
        .ok_or("no cell")?;
    assert_eq!(Some(cell.fg), app.theme.dim.fg, "a locked value is dim");
    let text = rows.join("\n");
    assert!(!text.contains("gen (used"), "the generator is not locked");
    assert!(!text.contains("openai (used"), "nor its provider");
    Ok(())
}

/// Every row of the Project view, drawn once each while `j` walks down,
/// shows `set`, `unset` or `vault ref` for a secret, never its value.
#[test]
fn no_secret_is_ever_drawn_in_the_project_view() -> TestResult {
    for (width, height) in [(80, 24), (120, 40)] {
        let mut app = project_app()?;
        let mut seen = String::new();
        for _ in 0..120 {
            seen.push_str(&text(&draw(&mut app, width, height)?).join("\n"));
            app.on_input(&key(KeyCode::Char('j')));
        }
        for (index, secret) in [SECRET, "sk-live", "secret/overbrainer", "vault:"]
            .into_iter()
            .enumerate()
        {
            assert!(
                !seen.contains(secret),
                "secret #{index} drawn at {width}x{height}"
            );
        }
        assert!(seen.contains("vault ref"), "{width}x{height}");
        assert!(
            seen.contains("env only, set OVERBRAINER_"),
            "{width}x{height}"
        );
        assert!(seen.contains("hf_token"), "the last row is reached");
    }
    Ok(())
}
