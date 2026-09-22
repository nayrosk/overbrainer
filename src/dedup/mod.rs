//! Near-duplicate filtering of generated questions.

mod embedding;
mod lexical;

pub use embedding::{Embedding, cosine};
pub use lexical::{Lexical, jaccard, shingles};

use std::future::Future;

use crate::llm::LlmError;

/// Filters near-duplicates against everything accepted so far.
pub trait Deduplicator: Send {
    /// Keeps the candidates that are neither near-duplicates of accepted items nor of
    /// an earlier kept candidate, records them as accepted, and returns them in order.
    ///
    /// # Errors
    ///
    /// Returns an [`LlmError`] when an embedding request fails.
    fn admit(
        &mut self,
        candidates: Vec<String>,
    ) -> impl Future<Output = Result<Vec<String>, LlmError>> + Send;

    /// Records items accepted earlier (for example read from disk) without filtering.
    ///
    /// # Errors
    ///
    /// Returns an [`LlmError`] when an embedding request fails.
    fn record(&mut self, accepted: &[String]) -> impl Future<Output = Result<(), LlmError>> + Send;
}

/// Lexical filtering first, then the optional embedding filter on what survives, so
/// embeddings are only requested for lexically new candidates.
#[derive(Debug)]
pub struct Layered<E> {
    lexical: Lexical,
    embedding: Option<E>,
}

impl<E: Deduplicator> Layered<E> {
    /// Chains `lexical` and, when set, `embedding`.
    #[must_use]
    pub fn new(lexical: Lexical, embedding: Option<E>) -> Self {
        Self { lexical, embedding }
    }
}

impl<E: Deduplicator> Deduplicator for Layered<E> {
    async fn admit(&mut self, candidates: Vec<String>) -> Result<Vec<String>, LlmError> {
        let novel = self.lexical.novel(&candidates);
        let kept: Vec<String> = candidates
            .into_iter()
            .zip(novel)
            .filter_map(|(candidate, novel)| novel.then_some(candidate))
            .collect();
        let kept = match &mut self.embedding {
            Some(embedding) => embedding.admit(kept).await?,
            None => kept,
        };
        self.lexical.insert(&kept);
        Ok(kept)
    }

    async fn record(&mut self, accepted: &[String]) -> Result<(), LlmError> {
        self.lexical.insert(accepted);
        if let Some(embedding) = &mut self.embedding {
            embedding.record(accepted).await?;
        }
        Ok(())
    }
}
