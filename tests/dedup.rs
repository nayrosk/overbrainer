use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use overbrainer::dedup::{Deduplicator, Embedding, Layered, Lexical, cosine};
use overbrainer::llm::{Completion, CompletionRequest, LlmClient, LlmError, RetryPolicy};

/// Embeds known texts to fixed vectors and remembers what it was asked to embed.
struct FakeEmbedder {
    vectors: HashMap<String, Vec<f32>>,
    asked: Mutex<Vec<String>>,
}

impl FakeEmbedder {
    fn new(pairs: &[(&str, [f32; 2])]) -> Self {
        Self {
            vectors: pairs
                .iter()
                .map(|(text, vector)| ((*text).to_string(), vector.to_vec()))
                .collect(),
            asked: Mutex::new(Vec::new()),
        }
    }

    fn asked(&self) -> Vec<String> {
        self.asked
            .lock()
            .map(|asked| asked.clone())
            .unwrap_or_default()
    }
}

impl LlmClient for &FakeEmbedder {
    async fn complete(&self, _request: CompletionRequest) -> Result<Completion, LlmError> {
        Err(LlmError::Unsupported("completions"))
    }

    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        if let Ok(mut asked) = self.asked.lock() {
            asked.extend(inputs.iter().cloned());
        }
        inputs
            .iter()
            .map(|input| {
                self.vectors
                    .get(input)
                    .cloned()
                    .ok_or_else(|| LlmError::InvalidResponse(format!("unknown text {input}")))
            })
            .collect()
    }
}

fn texts(items: &[&str]) -> Vec<String> {
    items.iter().map(|item| (*item).to_string()).collect()
}

/// A fast policy for tests: no real waiting.
fn fast_policy(max_retries: u32) -> RetryPolicy {
    RetryPolicy {
        max_retries,
        base: Duration::from_millis(1),
        cap: Duration::from_millis(2),
    }
}

/// Embeds one fixed vector for every input, failing its first `fails` calls with a
/// retryable server error.
struct FlakyEmbedder {
    fails: usize,
    calls: AtomicUsize,
    vector: Vec<f32>,
}

impl FlakyEmbedder {
    fn new(fails: usize, vector: [f32; 2]) -> Self {
        Self {
            fails,
            calls: AtomicUsize::new(0),
            vector: vector.to_vec(),
        }
    }
}

impl LlmClient for &FlakyEmbedder {
    async fn complete(&self, _request: CompletionRequest) -> Result<Completion, LlmError> {
        Err(LlmError::Unsupported("completions"))
    }

    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if call <= self.fails {
            return Err(LlmError::Status {
                status: 503,
                message: String::new(),
                retry_after: None,
            });
        }
        Ok(inputs.iter().map(|_| self.vector.clone()).collect())
    }
}

#[test]
fn cosine_of_parallel_orthogonal_and_zero_vectors() {
    assert!((cosine(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-9);
    assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-9);
    assert!(cosine(&[0.0, 0.0], &[1.0, 0.0]).abs() < 1e-9);
}

#[tokio::test]
async fn embedding_drops_semantic_duplicates() -> Result<(), LlmError> {
    let fake = FakeEmbedder::new(&[
        ("Why can't I move out of a borrow?", [1.0, 0.0]),
        ("How come moving from a reference fails?", [0.99, 0.05]),
        ("What does Pin guarantee?", [0.0, 1.0]),
    ]);
    let mut embedding = Embedding::new(&fake, 0.9, fast_policy(0));
    embedding
        .record(&texts(&["Why can't I move out of a borrow?"]))
        .await?;
    let kept = embedding
        .admit(texts(&[
            "How come moving from a reference fails?",
            "What does Pin guarantee?",
        ]))
        .await?;
    assert_eq!(kept, texts(&["What does Pin guarantee?"]));
    Ok(())
}

#[tokio::test]
async fn layered_embeds_only_lexically_new_candidates() -> Result<(), LlmError> {
    let fake = FakeEmbedder::new(&[
        ("What is a borrow?", [1.0, 0.0]),
        ("Explain lifetimes.", [0.0, 1.0]),
    ]);
    let mut layered = Layered::new(
        Lexical::new(0.8),
        Some(Embedding::new(&fake, 0.9, fast_policy(0))),
    );
    let kept = layered
        .admit(texts(&[
            "What is a borrow?",
            "what is a borrow",
            "Explain lifetimes.",
        ]))
        .await?;
    assert_eq!(kept, texts(&["What is a borrow?", "Explain lifetimes."]));
    assert_eq!(
        fake.asked(),
        texts(&["What is a borrow?", "Explain lifetimes."])
    );
    Ok(())
}

#[tokio::test]
async fn embedding_retries_a_transient_failure_then_succeeds() -> Result<(), LlmError> {
    let fake = FlakyEmbedder::new(1, [1.0, 0.0]);
    let mut embedding = Embedding::new(&fake, 0.9, fast_policy(2));
    embedding.record(&texts(&["a"])).await?;
    assert_eq!(
        fake.calls.load(Ordering::SeqCst),
        2,
        "the first call fails and is retried once"
    );
    Ok(())
}

#[tokio::test]
async fn embedding_gives_up_after_the_retry_budget() {
    let fake = FlakyEmbedder::new(5, [1.0, 0.0]);
    let mut embedding = Embedding::new(&fake, 0.9, fast_policy(1));
    let result = embedding.record(&texts(&["a"])).await;
    assert!(matches!(result, Err(LlmError::Status { status: 503, .. })));
    assert_eq!(
        fake.calls.load(Ordering::SeqCst),
        2,
        "one call plus one retry"
    );
}

/// Embeds every input to the same vector and records the inputs of each request.
struct BatchRecorder {
    batches: Mutex<Vec<Vec<String>>>,
}

impl LlmClient for &BatchRecorder {
    async fn complete(&self, _request: CompletionRequest) -> Result<Completion, LlmError> {
        Err(LlmError::Unsupported("completions"))
    }

    async fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        if let Ok(mut batches) = self.batches.lock() {
            batches.push(inputs.to_vec());
        }
        Ok(inputs.iter().map(|_| vec![1.0, 0.0]).collect())
    }
}

#[tokio::test]
async fn embeddings_are_requested_in_chunks_of_at_most_256_in_order() -> Result<(), LlmError> {
    let recorder = BatchRecorder {
        batches: Mutex::new(Vec::new()),
    };
    let inputs: Vec<String> = (0..600).map(|n| format!("question {n}")).collect();
    let mut embedding = Embedding::new(&recorder, 0.9, fast_policy(0));
    embedding.record(&inputs).await?;
    let batches = recorder
        .batches
        .lock()
        .map(|batches| batches.clone())
        .unwrap_or_default();
    let sizes: Vec<usize> = batches.iter().map(Vec::len).collect();
    assert_eq!(sizes, [256, 256, 88]);
    assert_eq!(batches.concat(), inputs, "chunks keep the input order");
    Ok(())
}
