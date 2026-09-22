use super::Deduplicator;
use crate::llm::{LlmClient, LlmError};

/// Cosine similarity of embeddings against a threshold.
#[derive(Debug)]
pub struct Embedding<C> {
    client: C,
    threshold: f64,
    accepted: Vec<Vec<f32>>,
}

impl<C: LlmClient> Embedding<C> {
    /// Texts whose cosine similarity is at least `threshold` are duplicates.
    #[must_use]
    pub fn new(client: C, threshold: f64) -> Self {
        Self {
            client,
            threshold,
            accepted: Vec::new(),
        }
    }

    async fn vectors(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        let vectors = self.client.embed(texts).await?;
        if vectors.len() == texts.len() {
            Ok(vectors)
        } else {
            Err(LlmError::InvalidResponse(format!(
                "{} embeddings for {} texts",
                vectors.len(),
                texts.len()
            )))
        }
    }
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
