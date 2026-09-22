use std::time::Duration;

use super::Deduplicator;
use crate::llm::{LlmClient, LlmError, RetryPolicy, with_retry};

/// Most inputs sent in one embeddings request. Providers cap the batch size (2048 on
/// `OpenAI`), so larger sets are split into several requests.
const MAX_INPUTS_PER_REQUEST: usize = 256;

/// Cosine similarity of embeddings against a threshold.
#[derive(Debug)]
pub struct Embedding<C> {
    client: C,
    threshold: f64,
    policy: RetryPolicy,
    accepted: Vec<Vec<f32>>,
}

impl<C: LlmClient> Embedding<C> {
    /// Texts whose cosine similarity is at least `threshold` are duplicates. Embed
    /// requests are retried under `policy`, so a transient failure of the embeddings
    /// endpoint does not fail the whole call on its own.
    #[must_use]
    pub fn new(client: C, threshold: f64, policy: RetryPolicy) -> Self {
        Self {
            client,
            threshold,
            policy,
            accepted: Vec::new(),
        }
    }

    /// Embeds `texts` in order, [`MAX_INPUTS_PER_REQUEST`] at a time, each request
    /// retried on its own.
    async fn vectors(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        let mut all = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(MAX_INPUTS_PER_REQUEST) {
            let vectors = with_retry(&self.policy, || self.client.embed(chunk), log_retry).await?;
            if vectors.len() != chunk.len() {
                return Err(LlmError::InvalidResponse(format!(
                    "{} embeddings for {} texts",
                    vectors.len(),
                    chunk.len()
                )));
            }
            all.extend(vectors);
        }
        Ok(all)
    }
}

fn log_retry(error: &LlmError, wait: Duration) {
    tracing::debug!(
        "embedding request failed ({error}); retrying in {:.1}s",
        wait.as_secs_f64()
    );
}

impl<C: LlmClient> Deduplicator for Embedding<C> {
    async fn admit(&mut self, candidates: Vec<String>) -> Result<Vec<String>, LlmError> {
        if candidates.is_empty() {
            return Ok(candidates);
        }
        let vectors = self.vectors(&candidates).await?;
        let mut kept = Vec::new();
        for (candidate, vector) in candidates.into_iter().zip(vectors) {
            let duplicate = self
                .accepted
                .iter()
                .any(|other| cosine(&vector, other) >= self.threshold);
            if !duplicate {
                self.accepted.push(vector);
                kept.push(candidate);
            }
        }
        Ok(kept)
    }

    async fn record(&mut self, accepted: &[String]) -> Result<(), LlmError> {
        if accepted.is_empty() {
            return Ok(());
        }
        let vectors = self.vectors(accepted).await?;
        self.accepted.extend(vectors);
        Ok(())
    }
}

/// Cosine similarity of `a` and `b`; 0 when either is a zero vector.
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut norm_a, mut norm_b) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}
