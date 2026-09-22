use std::collections::HashMap;
use std::sync::Mutex;

use overbrainer::dedup::{Deduplicator, Embedding, Layered, Lexical, cosine};
use overbrainer::llm::{Completion, CompletionRequest, LlmClient, LlmError};

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
    let mut embedding = Embedding::new(&fake, 0.9);
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
    let mut layered = Layered::new(Lexical::new(0.8), Some(Embedding::new(&fake, 0.9)));
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
