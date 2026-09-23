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
use super::tasks::{Done, TaskId};
use super::theme::Theme;
use super::ui;
use crate::cli::data::Command;
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

/// An app on the project `rust_expert` at [`NOW`], with the color theme.
pub(super) fn app() -> App {
    let project = Project {
        name: "rust_expert".into(),
        dir: "/nonexistent/rust_expert".into(),
        topics: Vec::new(),
        eval_ratio: 0.1,
        concurrency: 8,
    };
    App::new(project, LogBuffer::new(100), Theme::color(), at(NOW))
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
    let mut app = app();
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

/// The tree path of a question of `ownership`/`Borrowing`, and of its answer.
pub(super) fn path_to(text: &str, answer: bool) -> Vec<Node> {
    let subtopic = Id::subtopic("ownership", "Borrowing");
    let id = Id::question(&subtopic, text);
    let mut path = vec![
        Node::Topic("ownership".into()),
        Node::Subtopic(subtopic),
        Node::Question(id.clone()),
    ];
    if answer {
        path.push(Node::Answer(id));
    }
    path
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
    settings.bind(|| insta::assert_snapshot!(name.to_string(), terminal.backend()));
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
/// shows as reversed.
#[test]
fn the_monochrome_theme_uses_no_color() -> TestResult {
    let mut app = dataset_app();
    logs(&app);
    app.theme = Theme::mono();
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
fn dataset_on_an_answer_with_its_reasoning() -> TestResult {
    let mut app = dataset_app();
    open_to(&mut app, &path_to(MOVED, true));
    snapshot("dataset_answer", &mut app)?;
    Ok(())
}

#[test]
fn dataset_on_an_excluded_question() -> TestResult {
    let mut app = dataset_app();
    open_to(
        &mut app,
        &path_to("Why can't a &mut and a & borrow coexist?", false),
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

#[test]
fn dataset_load_error() -> TestResult {
    let mut app = app();
    let error = "data/answers.jsonl:3: invalid record: expected value at line 1 column 2";
    let Some(Effect::Spawn(id, _)) = app.start().first().cloned() else {
        return Err("no load started".into());
    };
    app.on_done(id, Ok(Done::Loaded(Err(error.into()))));
    app.status = None;
    snapshot("dataset_error", &mut app)?;
    Ok(())
}

#[test]
fn the_delete_dialog_says_what_goes_with_a_subtopic() -> TestResult {
    let mut app = dataset_app();
    open_to(&mut app, &path_to(MOVED, false)[..2]);
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

#[test]
fn the_run_menu() -> TestResult {
    let mut app = app();
    app.on_input(&key(KeyCode::Char('r')));
    app.on_input(&key(KeyCode::Down));
    app.on_input(&key(KeyCode::Down));
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
const FINISHED: &str = "20260920-101500-9f00";
const LEFT: &str = "20260919-090000-c3d4";

/// Three runs: a Runpod run followed live, a finished local run, and a run left
/// running that nothing follows.
fn training_app() -> Result<App, serde_json::Error> {
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
        runpod: Some(RunpodPlan {
            gpu_types: vec![
                "NVIDIA GeForce RTX 4090".into(),
                "NVIDIA RTX A6000".into(),
                "NVIDIA A40".into(),
            ],
            gpu_count: 1,
            max_hours: 6.0,
        }),
        warnings: vec!["qwen2 renders no reasoning_content (see the logs)".into()],
    }
}

#[test]
fn the_start_dialog_of_a_runpod_run_before_and_with_its_prices() -> TestResult {
    let mut app = app();
    app.view = View::Training;
    app.prepared(Ok(runpod_plan()));
    snapshot("start_runpod_looking_up", &mut app)?;
    app.priced(&vec![
        ("NVIDIA GeForce RTX 4090".to_string(), Some(0.74)),
        ("NVIDIA RTX A6000".to_string(), Some(0.79)),
        ("NVIDIA A40".to_string(), None),
    ]);
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

/// A start dialog taller than the terminal keeps its key line and its
/// most-it-can-cost line, and marks the text it cut; so does a quit dialog.
#[test]
fn a_tall_dialog_keeps_its_keys_and_its_cost_at_80x24() -> TestResult {
    let mut app = app();
    app.view = View::Training;
    let mut plan = runpod_plan();
    let gpus: Vec<String> = (1..=14).map(|n| format!("NVIDIA GPU model {n}")).collect();
    if let Some(runpod) = &mut plan.runpod {
        runpod.gpu_types.clone_from(&gpus);
    }
    plan.warnings = (1..=4)
        .map(|n| format!("warning number {n} about this run"))
        .collect();
    app.prepared(Ok(plan));
    app.priced(&gpus.iter().map(|gpu| (gpu.clone(), Some(0.5))).collect());
    let rows = text(&draw(&mut app, 80, 24)?).join("\n");
    for shown in [
        "warning     warning number 1 about this run",
        "warning     warning number 4 about this run",
        "[y] start",
        "[n] cancel",
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
    for shown in ["[y] quit", "[n] stay", "…"] {
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
            rows.contains("e, d, r and t are refused"),
            "{view:?}\n{rows}"
        );
        assert!(rows.contains("is not locked out."), "{view:?}\n{rows}");
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

/// The pod line of `app` drawn at 120x40: rows 9 and 10 of the detail pane,
/// joined as one text.
fn pod_text(app: &mut App) -> Result<String, Infallible> {
    let rows = text(&draw(app, 120, 40)?);
    let words: Vec<&str> = rows
        .iter()
        .skip(9)
        .take(2)
        .map(|row| row.trim_matches('│').trim())
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
        line.contains("kept once its job starts  deleted by the watchdog by 2026-09-21T19:32:20Z"),
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
