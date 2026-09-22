use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use overbrainer::config::{EnvSource, Settings, load};
use overbrainer::dataset::{DataFiles, FinishReason, Id, Question, Subtopic, read};
use overbrainer::dedup::Lexical;
use overbrainer::events::EventBus;
use overbrainer::llm::{Completion, CompletionRequest, LlmClient, LlmError, Reasoning, Usage};
use overbrainer::pipeline::{self, Ctx, RoleClient};
use overbrainer::prompts::Prompts;

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

struct Project {
    dir: tempfile::TempDir,
    settings: Settings,
    files: DataFiles,
    prompts: Prompts,
    bus: EventBus,
}

impl Project {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join("overbrainer.toml"), PROJECT)?;
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

    fn ctx(&self, force: bool) -> Ctx<'_> {
        Ctx {
            settings: &self.settings,
            files: &self.files,
            prompts: &self.prompts,
            bus: &self.bus,
            topic: None,
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
