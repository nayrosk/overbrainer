use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use overbrainer::config::{EnvSource, Settings, load};
use overbrainer::dataset::{DataFiles, FinishReason, Id, Question, Subtopic, read};
use overbrainer::dedup::{Deduplicator, Lexical};
use overbrainer::events::{Event, EventBus, Stage};
use overbrainer::llm::{Completion, CompletionRequest, LlmClient, LlmError, Reasoning, Usage};
use overbrainer::pipeline::{self, Ctx, RoleClient};
use overbrainer::prompts::Prompts;
use tokio::sync::broadcast;

type TestResult = Result<(), Box<dyn std::error::Error>>;
type Reply = Box<dyn Fn(&CompletionRequest, usize) -> Result<Completion, LlmError> + Send + Sync>;

/// Answers with `reply(request, call_number)` after `delay`, recording requests and the
/// peak number of concurrent calls.
struct FakeLlm {
    reply: Reply,
    delay: Duration,
    requests: Mutex<Vec<CompletionRequest>>,
    in_flight: AtomicUsize,
    peak: AtomicUsize,
}

impl FakeLlm {
    fn new(reply: Reply) -> Self {
        Self {
            reply,
            delay: Duration::ZERO,
            requests: Mutex::new(Vec::new()),
            in_flight: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        }
    }

    fn requests(&self) -> Vec<CompletionRequest> {
        self.requests.lock().map(|r| r.clone()).unwrap_or_default()
    }
}

impl LlmClient for FakeLlm {
    async fn complete(&self, request: CompletionRequest) -> Result<Completion, LlmError> {
        let call = match self.requests.lock() {
            Ok(mut requests) => {
                requests.push(request.clone());
                requests.len()
            },
            Err(_) => 0,
        };
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(self.delay).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        (self.reply)(&request, call)
    }

    async fn embed(&self, _inputs: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        Err(LlmError::Unsupported("embeddings"))
    }
}

fn text(content: &str) -> Completion {
    Completion {
        content: content.to_string(),
        reasoning: Reasoning::none(),
        usage: Usage {
            input_tokens: 10,
            output_tokens: 20,
        },
        finish: FinishReason::Stop,
    }
}

const PROJECT: &str = r#"
[project]
name = "demo"

[[topics]]
name = "ownership"
subtopics = 2
questions_per_subtopic = 3

[providers.fake]
protocol = "openai"

[roles]
generator = { provider = "fake", model = "gen" }
parent = { provider = "fake", model = "parent", reasoning = true }

[pipeline]
concurrency = 2
max_retries = 2
question_batch_size = 2
eval_ratio = 0.5
"#;

/// Two topics, so `--topic` and `--force` selection has something to leave alone.
const TWO_TOPICS: &str = r#"
[project]
name = "demo"

[[topics]]
name = "ownership"
subtopics = 2
questions_per_subtopic = 2

[[topics]]
name = "control_flow"
subtopics = 2
questions_per_subtopic = 2

[providers.fake]
protocol = "openai"

[roles]
generator = { provider = "fake", model = "gen" }
parent = { provider = "fake", model = "parent", reasoning = true }

[pipeline]
concurrency = 2
max_retries = 2
question_batch_size = 2
eval_ratio = 0.5
"#;

struct Project {
    dir: tempfile::TempDir,
    settings: Settings,
    files: DataFiles,
    prompts: Prompts,
    bus: EventBus,
}

impl Project {
    fn with_toml(toml: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join("overbrainer.toml"), toml)?;
        let settings = load(dir.path(), EnvSource::Vars(Vec::new()))?;
        let files = DataFiles::new(dir.path());
        let prompts = Prompts::load(dir.path())?;
        Ok(Self {
            dir,
            settings,
            files,
            prompts,
            bus: EventBus::new(),
        })
    }

    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_toml(PROJECT)
    }

    fn ctx(&self, force: bool) -> Ctx<'_> {
        self.ctx_topic(None, force)
    }

    fn ctx_topic<'a>(&'a self, topic: Option<&'a str>, force: bool) -> Ctx<'a> {
        Ctx {
            settings: &self.settings,
            files: &self.files,
            prompts: &self.prompts,
            bus: &self.bus,
            topic,
            force,
        }
    }

    fn role<C>(&self, client: C, parent: bool) -> RoleClient<C> {
        let model = if parent {
            &self.settings.roles.parent
        } else {
            &self.settings.roles.generator
        };
        RoleClient {
            client,
            model: model.clone(),
            price: None,
        }
    }
}

#[tokio::test]
async fn subtopics_are_generated_once_and_capped() -> TestResult {
    let project = Project::new()?;
    let fake = FakeLlm::new(Box::new(|_, call| {
        Ok(text(if call == 1 {
            "Sure! Here they are"
        } else {
            r#"["Borrowing", "borrowing ", "Lifetimes", "Moves"]"#
        }))
    }));
    let generator = project.role(fake, false);
    let stats = pipeline::subtopics(&project.ctx(false), &generator).await?;
    assert_eq!((stats.done, stats.failed), (1, 0));
    assert_eq!(stats.usage.output_tokens, 40, "both attempts are counted");
    let subtopics: Vec<Subtopic> = read(&project.files.subtopics)?;
    let names: Vec<&str> = subtopics.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["Borrowing", "Lifetimes"]);
    assert!(project.dir.path().join("data/subtopics.jsonl").is_file());

    let again = pipeline::subtopics(&project.ctx(false), &generator).await?;
    assert_eq!((again.done, again.skipped), (0, 1));
    assert_eq!(generator.client.requests().len(), 2);
    Ok(())
}

#[tokio::test]
async fn questions_fill_each_subtopic_in_batches() -> TestResult {
    let project = Project::new()?;
    let mut out = overbrainer::dataset::Appender::open(&project.files.subtopics)?;
    for name in ["Borrowing", "Lifetimes"] {
        out.append(&Subtopic {
            id: Id::subtopic("ownership", name),
            topic: "ownership".into(),
            name: name.into(),
        })?;
    }
    let fake = FakeLlm::new(Box::new(|_, call| {
        Ok(text(&format!(
            r#"["Distinct question number {call} alpha?", "Another angle number {call} beta?"]"#
        )))
    }));
    let generator = project.role(fake, false);
    let stats = pipeline::questions(&project.ctx(false), &generator, || Lexical::new(0.8)).await?;
    assert_eq!(stats.done, 2);
    let questions: Vec<Question> = read(&project.files.questions)?;
    assert_eq!(questions.len(), 6);
    let requests = generator.client.requests();
    assert_eq!(requests.len(), 4, "2 + 1 questions per subtopic");
    assert!(
        requests[1]
            .prompt
            .contains("- Distinct question number 1 alpha?")
    );
    assert!(requests[1].prompt.contains("JSON array of 1 strings"));
    Ok(())
}

#[tokio::test]
async fn questions_stop_when_the_generator_repeats_itself() -> TestResult {
    let project = Project::new()?;
    let mut out = overbrainer::dataset::Appender::open(&project.files.subtopics)?;
    out.append(&Subtopic {
        id: Id::subtopic("ownership", "Borrowing"),
        topic: "ownership".into(),
        name: "Borrowing".into(),
    })?;
    let fake = FakeLlm::new(Box::new(|_, _| {
        Ok(text(r#"["What is a borrow?", "Why borrow at all?"]"#))
    }));
    let generator = project.role(fake, false);
    pipeline::questions(&project.ctx(false), &generator, || Lexical::new(0.8)).await?;
    let questions: Vec<Question> = read(&project.files.questions)?;
    assert_eq!(questions.len(), 2);
    assert_eq!(
        generator.client.requests().len(),
        3,
        "one productive batch, then max_retries = 2 batches without progress"
    );
    Ok(())
}

/// A deduplicator that fails its `fail_at`-th call to `admit`, otherwise passes
/// candidates through unfiltered. `record` always succeeds.
struct FlakyDedup {
    fail_at: usize,
    calls: usize,
    error: fn() -> LlmError,
}

impl FlakyDedup {
    fn new(fail_at: usize, error: fn() -> LlmError) -> Self {
        Self {
            fail_at,
            calls: 0,
            error,
        }
    }
}

impl Deduplicator for FlakyDedup {
    async fn admit(&mut self, candidates: Vec<String>) -> Result<Vec<String>, LlmError> {
        self.calls += 1;
        if self.calls == self.fail_at {
            return Err((self.error)());
        }
        Ok(candidates)
    }

    async fn record(&mut self, _accepted: &[String]) -> Result<(), LlmError> {
        Ok(())
    }
}

fn non_fatal() -> LlmError {
    LlmError::InvalidResponse("boom".to_string())
}

fn fatal() -> LlmError {
    LlmError::Unsupported("embeddings")
}

fn unauthorized() -> LlmError {
    LlmError::Status {
        status: 401,
        message: "nope".to_string(),
        retry_after: None,
    }
}

/// Drains every event currently buffered on `receiver`.
fn drain(receiver: &mut broadcast::Receiver<Event>) -> Vec<Event> {
    let mut events = Vec::new();
    while let Ok(event) = receiver.try_recv() {
        events.push(event);
    }
    events
}

fn started_total(events: &[Event], stage: Stage) -> Option<usize> {
    events.iter().find_map(|event| match event {
        Event::StageStarted { stage: s, total } if *s == stage => Some(*total),
        _ => None,
    })
}

fn item_done_usage(events: &[Event], stage: Stage) -> Option<Usage> {
    events.iter().find_map(|event| match event {
        Event::ItemDone {
            stage: s, usage, ..
        } if *s == stage => *usage,
        _ => None,
    })
}

fn count_item_failed(events: &[Event], retryable: bool) -> usize {
    events
        .iter()
        .filter(|event| matches!(event, Event::ItemFailed { retryable: r, .. } if *r == retryable))
        .count()
}

#[tokio::test]
async fn subtopics_resume_a_partial_topic_and_dedup_against_existing_names() -> TestResult {
    let project = Project::new()?;
    let mut out = overbrainer::dataset::Appender::open(&project.files.subtopics)?;
    out.append(&Subtopic {
        id: Id::subtopic("ownership", "Borrowing"),
        topic: "ownership".into(),
        name: "Borrowing".into(),
    })?;
    let fake = FakeLlm::new(Box::new(|_, _| Ok(text(r#"["Borrowing", "Lifetimes"]"#))));
    let generator = project.role(fake, false);
    let stats = pipeline::subtopics(&project.ctx(false), &generator).await?;
    assert_eq!((stats.done, stats.skipped), (1, 0));
    let requests = generator.client.requests();
    assert_eq!(requests.len(), 1, "only the missing subtopic is requested");
    assert!(requests[0].prompt.contains("JSON array of 1 strings"));
    let subtopics: Vec<Subtopic> = read(&project.files.subtopics)?;
    let names: Vec<&str> = subtopics.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(
        names,
        ["Borrowing", "Lifetimes"],
        "the repeated name is dropped, only the new one is kept"
    );
    Ok(())
}

#[tokio::test]
async fn questions_resume_a_partially_filled_subtopic() -> TestResult {
    let project = Project::new()?;
    let mut out = overbrainer::dataset::Appender::open(&project.files.subtopics)?;
    out.append(&Subtopic {
        id: Id::subtopic("ownership", "Borrowing"),
        topic: "ownership".into(),
        name: "Borrowing".into(),
    })?;
    let subtopic_id = Id::subtopic("ownership", "Borrowing");
    let mut questions = overbrainer::dataset::Appender::open(&project.files.questions)?;
    for existing in ["What is a borrow?", "Why borrow at all?"] {
        questions.append(&Question {
            id: Id::question(&subtopic_id, existing),
            topic: "ownership".into(),
            subtopic_id: subtopic_id.clone(),
            subtopic: "Borrowing".into(),
            text: existing.into(),
        })?;
    }
    let fake = FakeLlm::new(Box::new(|_, _| Ok(text(r#"["A brand new question?"]"#))));
    let generator = project.role(fake, false);
    let stats = pipeline::questions(&project.ctx(false), &generator, || Lexical::new(0.8)).await?;
    assert_eq!(stats.done, 1);
    let requests = generator.client.requests();
    assert_eq!(requests.len(), 1, "only the missing question is requested");
    assert!(
        requests[0].prompt.contains("JSON array of 1 strings"),
        "batch_size (2) is capped by what is missing (1)"
    );
    let all: Vec<Question> = read(&project.files.questions)?;
    assert_eq!(all.len(), 3);
    Ok(())
}

#[tokio::test]
async fn force_only_removes_the_selected_topics_subtopics() -> TestResult {
    let project = Project::with_toml(TWO_TOPICS)?;
    let mut out = overbrainer::dataset::Appender::open(&project.files.subtopics)?;
    for (topic, name) in [
        ("ownership", "Borrowing"),
        ("ownership", "Lifetimes"),
        ("control_flow", "Loops"),
        ("control_flow", "Matches"),
    ] {
        out.append(&Subtopic {
            id: Id::subtopic(topic, name),
            topic: topic.into(),
            name: name.into(),
        })?;
    }
    let fake = FakeLlm::new(Box::new(|_, _| Ok(text(r#"["Moves", "Borrows"]"#))));
    let generator = project.role(fake, false);
    pipeline::subtopics(&project.ctx_topic(Some("ownership"), true), &generator).await?;
    let subtopics: Vec<Subtopic> = read(&project.files.subtopics)?;
    let control_flow: Vec<&str> = subtopics
        .iter()
        .filter(|s| s.topic == "control_flow")
        .map(|s| s.name.as_str())
        .collect();
    assert_eq!(
        control_flow,
        ["Loops", "Matches"],
        "the other topic's subtopics are untouched"
    );
    let ownership: Vec<&str> = subtopics
        .iter()
        .filter(|s| s.topic == "ownership")
        .map(|s| s.name.as_str())
        .collect();
    assert_eq!(
        ownership,
        ["Moves", "Borrows"],
        "the selected topic was regenerated"
    );
    Ok(())
}

#[tokio::test]
async fn topic_option_limits_subtopics_to_the_selected_topic() -> TestResult {
    let project = Project::with_toml(TWO_TOPICS)?;
    let fake = FakeLlm::new(Box::new(|_, _| Ok(text(r#"["Moves", "Borrows"]"#))));
    let generator = project.role(fake, false);
    pipeline::subtopics(&project.ctx_topic(Some("ownership"), false), &generator).await?;
    let subtopics: Vec<Subtopic> = read(&project.files.subtopics)?;
    assert!(subtopics.iter().all(|s| s.topic == "ownership"));
    assert_eq!(generator.client.requests().len(), 1);
    Ok(())
}

#[tokio::test]
async fn subtopics_stage_stops_on_a_fatal_provider_error() -> TestResult {
    let project = Project::new()?;
    let fake = FakeLlm::new(Box::new(|_, _| Err(unauthorized())));
    let generator = project.role(fake, false);
    let result = pipeline::subtopics(&project.ctx(false), &generator).await;
    assert!(matches!(
        result,
        Err(pipeline::PipelineError::Llm {
            stage: Stage::Subtopics,
            source: LlmError::Status { status: 401, .. },
        })
    ));
    Ok(())
}

#[tokio::test]
async fn questions_stage_stops_on_a_fatal_provider_error() -> TestResult {
    let project = Project::new()?;
    let mut out = overbrainer::dataset::Appender::open(&project.files.subtopics)?;
    out.append(&Subtopic {
        id: Id::subtopic("ownership", "Borrowing"),
        topic: "ownership".into(),
        name: "Borrowing".into(),
    })?;
    let fake = FakeLlm::new(Box::new(|_, _| Err(unauthorized())));
    let generator = project.role(fake, false);
    let result = pipeline::questions(&project.ctx(false), &generator, || Lexical::new(0.8)).await;
    assert!(matches!(
        result,
        Err(pipeline::PipelineError::Llm {
            stage: Stage::Questions,
            source: LlmError::Status { status: 401, .. },
        })
    ));
    Ok(())
}

#[tokio::test]
async fn subtopics_started_total_excludes_topics_already_done() -> TestResult {
    let project = Project::with_toml(TWO_TOPICS)?;
    let mut out = overbrainer::dataset::Appender::open(&project.files.subtopics)?;
    for name in ["Borrowing", "Lifetimes"] {
        out.append(&Subtopic {
            id: Id::subtopic("ownership", name),
            topic: "ownership".into(),
            name: name.into(),
        })?;
    }
    let fake = FakeLlm::new(Box::new(|_, _| Ok(text(r#"["Loops", "Matches"]"#))));
    let generator = project.role(fake, false);
    let mut receiver = project.bus.subscribe();
    let stats = pipeline::subtopics(&project.ctx(false), &generator).await?;
    assert_eq!((stats.done, stats.skipped), (1, 1));
    let events = drain(&mut receiver);
    assert_eq!(
        started_total(&events, Stage::Subtopics),
        Some(1),
        "the already-done topic is not counted"
    );
    Ok(())
}

#[tokio::test]
async fn questions_started_total_excludes_subtopics_already_filled() -> TestResult {
    let project = Project::new()?;
    let mut out = overbrainer::dataset::Appender::open(&project.files.subtopics)?;
    for name in ["Borrowing", "Lifetimes"] {
        out.append(&Subtopic {
            id: Id::subtopic("ownership", name),
            topic: "ownership".into(),
            name: name.into(),
        })?;
    }
    let filled_id = Id::subtopic("ownership", "Borrowing");
    let mut questions = overbrainer::dataset::Appender::open(&project.files.questions)?;
    for existing in ["Q1?", "Q2?", "Q3?"] {
        questions.append(&Question {
            id: Id::question(&filled_id, existing),
            topic: "ownership".into(),
            subtopic_id: filled_id.clone(),
            subtopic: "Borrowing".into(),
            text: existing.into(),
        })?;
    }
    let fake = FakeLlm::new(Box::new(|_, call| {
        Ok(text(&format!(r#"["Q{call}a?", "Q{call}b?"]"#)))
    }));
    let generator = project.role(fake, false);
    let mut receiver = project.bus.subscribe();
    pipeline::questions(&project.ctx(false), &generator, || Lexical::new(0.8)).await?;
    let events = drain(&mut receiver);
    assert_eq!(
        started_total(&events, Stage::Questions),
        Some(1),
        "the already-filled subtopic is not counted"
    );
    Ok(())
}

#[tokio::test]
async fn subtopics_item_done_carries_the_summed_usage_of_all_attempts() -> TestResult {
    let project = Project::new()?;
    let fake = FakeLlm::new(Box::new(|_, call| {
        Ok(text(if call == 1 {
            "not json"
        } else {
            r#"["Borrowing", "Lifetimes"]"#
        }))
    }));
    let generator = project.role(fake, false);
    let mut receiver = project.bus.subscribe();
    pipeline::subtopics(&project.ctx(false), &generator).await?;
    let events = drain(&mut receiver);
    assert_eq!(
        item_done_usage(&events, Stage::Subtopics),
        Some(Usage {
            input_tokens: 20,
            output_tokens: 40,
        }),
        "usage of both attempts (the failed parse and the successful one) is summed"
    );
    Ok(())
}

#[tokio::test]
async fn questions_item_done_carries_the_summed_usage_of_all_batches() -> TestResult {
    let project = Project::new()?;
    let mut out = overbrainer::dataset::Appender::open(&project.files.subtopics)?;
    out.append(&Subtopic {
        id: Id::subtopic("ownership", "Borrowing"),
        topic: "ownership".into(),
        name: "Borrowing".into(),
    })?;
    let fake = FakeLlm::new(Box::new(|_, call| {
        Ok(text(&format!(r#"["Q{call}a?", "Q{call}b?"]"#)))
    }));
    let generator = project.role(fake, false);
    let mut receiver = project.bus.subscribe();
    pipeline::questions(&project.ctx(false), &generator, || Lexical::new(0.8)).await?;
    let events = drain(&mut receiver);
    assert_eq!(
        item_done_usage(&events, Stage::Questions),
        Some(Usage {
            input_tokens: 20,
            output_tokens: 40,
        }),
        "two batches were needed (batch size 2, target 3), usage of both is summed"
    );
    Ok(())
}

#[tokio::test]
async fn only_one_final_itemfailed_event_is_published_per_item() -> TestResult {
    let project = Project::new()?;
    let fake = FakeLlm::new(Box::new(|_, _| Ok(text("not json, ever"))));
    let generator = project.role(fake, false);
    let mut receiver = project.bus.subscribe();
    let stats = pipeline::subtopics(&project.ctx(false), &generator).await?;
    assert_eq!(stats.failed, 1);
    let events = drain(&mut receiver);
    assert_eq!(
        count_item_failed(&events, false),
        1,
        "exactly one final, non-retryable failure is reported for the item"
    );
    assert_eq!(
        count_item_failed(&events, true),
        3,
        "one retryable failure per attempt (max_retries = 2, so 3 attempts)"
    );
    Ok(())
}

#[tokio::test]
async fn questions_stage_fails_only_the_subtopic_when_dedup_admit_fails_non_fatally() -> TestResult
{
    let project = Project::new()?;
    let mut out = overbrainer::dataset::Appender::open(&project.files.subtopics)?;
    for name in ["Borrowing", "Lifetimes"] {
        out.append(&Subtopic {
            id: Id::subtopic("ownership", name),
            topic: "ownership".into(),
            name: name.into(),
        })?;
    }
    let fake = FakeLlm::new(Box::new(|_, call| {
        Ok(text(&format!(r#"["Q{call}a?", "Q{call}b?"]"#)))
    }));
    let generator = project.role(fake, false);
    let stats = pipeline::questions(&project.ctx(false), &generator, || {
        FlakyDedup::new(1, non_fatal)
    })
    .await?;
    assert_eq!(
        stats.failed, 1,
        "only the first subtopic's first batch failed"
    );
    assert_eq!(stats.done, 1, "the second subtopic still completed");
    let questions: Vec<Question> = read(&project.files.questions)?;
    assert!(
        questions.iter().all(|q| q.subtopic == "Lifetimes"),
        "the failed subtopic wrote nothing"
    );
    Ok(())
}

#[tokio::test]
async fn questions_stage_stops_when_dedup_admit_fails_fatally() -> TestResult {
    let project = Project::new()?;
    let mut out = overbrainer::dataset::Appender::open(&project.files.subtopics)?;
    out.append(&Subtopic {
        id: Id::subtopic("ownership", "Borrowing"),
        topic: "ownership".into(),
        name: "Borrowing".into(),
    })?;
    let fake = FakeLlm::new(Box::new(|_, call| {
        Ok(text(&format!(r#"["Q{call}a?", "Q{call}b?"]"#)))
    }));
    let generator = project.role(fake, false);
    let result = pipeline::questions(&project.ctx(false), &generator, || {
        FlakyDedup::new(1, fatal)
    })
    .await;
    assert!(matches!(
        result,
        Err(pipeline::PipelineError::Llm {
            stage: Stage::Questions,
            source: LlmError::Unsupported(_),
        })
    ));
    Ok(())
}
