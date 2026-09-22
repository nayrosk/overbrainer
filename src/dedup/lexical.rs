use std::collections::BTreeSet;

use super::Deduplicator;
use crate::llm::LlmError;

/// Jaccard similarity of word bigrams against a threshold.
#[derive(Debug, Clone)]
pub struct Lexical {
    threshold: f64,
    accepted: Vec<BTreeSet<String>>,
}

impl Lexical {
    /// Texts whose similarity is at least `threshold` are duplicates.
    #[must_use]
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            accepted: Vec::new(),
        }
    }

    /// For each candidate, whether it is new compared with the accepted texts and with
    /// the earlier new candidates of the same batch. Records nothing.
    #[must_use]
    pub fn novel(&self, candidates: &[String]) -> Vec<bool> {
        let mut kept: Vec<BTreeSet<String>> = Vec::new();
        candidates
            .iter()
            .map(|candidate| {
                let shingles = shingles(candidate);
                let duplicate = self
                    .accepted
                    .iter()
                    .chain(&kept)
                    .any(|other| jaccard(&shingles, other) >= self.threshold);
                if !duplicate {
                    kept.push(shingles);
                }
                !duplicate
            })
            .collect()
    }

    /// Records `texts` as accepted.
    pub fn insert(&mut self, texts: &[String]) {
        self.accepted
            .extend(texts.iter().map(|text| shingles(text)));
    }
}

impl Deduplicator for Lexical {
    async fn admit(&mut self, candidates: Vec<String>) -> Result<Vec<String>, LlmError> {
        let novel = self.novel(&candidates);
        let kept: Vec<String> = candidates
            .into_iter()
            .zip(novel)
            .filter_map(|(candidate, novel)| novel.then_some(candidate))
            .collect();
        self.insert(&kept);
        Ok(kept)
    }

    async fn record(&mut self, accepted: &[String]) -> Result<(), LlmError> {
        self.insert(accepted);
        Ok(())
    }
}

/// Word bigrams of `text` after lowercasing, replacing punctuation with spaces and
/// collapsing whitespace. A single word yields itself.
#[must_use]
pub fn shingles(text: &str) -> BTreeSet<String> {
    let cleaned: String = text
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_lowercase().next().unwrap_or(c)
            } else {
                ' '
            }
        })
        .collect();
    let words: Vec<&str> = cleaned.split_whitespace().collect();
    if words.len() < 2 {
        return words.into_iter().map(str::to_string).collect();
    }
    words
        .windows(2)
        .map(|pair| format!("{} {}", pair[0], pair[1]))
        .collect()
}

/// Size of the intersection over size of the union. Two empty sets are identical.
#[must_use]
pub fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    let union = a.union(b).count();
    if union == 0 {
        return 1.0;
    }
    count(a.intersection(b).count()) / count(union)
}

fn count(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_string()).collect()
    }

    #[test]
    fn shingles_ignore_case_and_punctuation() {
        assert_eq!(shingles("What is a borrow?"), shingles("what IS a, borrow"));
        assert_eq!(shingles("Borrowing!").len(), 1);
    }

    #[test]
    fn jaccard_of_identical_and_disjoint_sets() {
        let a = shingles("how do lifetimes work");
        let b = shingles("why use an arena allocator");
        assert!((jaccard(&a, &a) - 1.0).abs() < f64::EPSILON);
        assert!(jaccard(&a, &b).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn near_duplicates_are_dropped() -> Result<(), LlmError> {
        let mut lexical = Lexical::new(0.8);
        lexical
            .record(&texts(&["What is a borrow in Rust?"]))
            .await?;
        let kept = lexical
            .admit(texts(&[
                "what is a borrow in rust",
                "How does the borrow checker handle loops?",
                "How does the borrow checker handle loops",
            ]))
            .await?;
        assert_eq!(kept, texts(&["How does the borrow checker handle loops?"]));
        let again = lexical
            .admit(texts(&["How does the borrow checker handle loops?"]))
            .await?;
        assert!(again.is_empty());
        Ok(())
    }
}
