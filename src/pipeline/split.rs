use std::collections::BTreeMap;

use twox_hash::XxHash3_64;

use super::{Ctx, PipelineError};
use crate::dataset::{Example, Exclusion, read, rewrite};
use crate::events::{Event, Stage, StageStats};

/// Outcome of [`split`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SplitReport {
    /// Examples written to `data/train.jsonl`.
    pub train: usize,
    /// Examples written to `data/eval.jsonl`.
    pub eval: usize,
    /// Examples left out, by reason.
    pub excluded: BTreeMap<Exclusion, usize>,
}

/// Rewrites `data/train.jsonl` and `data/eval.jsonl` from the usable examples of
/// `data/answers.jsonl` (those with `meta.excluded = null`), stratified by subtopic.
/// Both files always hold every topic's usable examples: `--topic` only limits the
/// report and the event counts to that topic. Excluded examples stay in
/// `answers.jsonl`. The split always rewrites both files, so `--force` changes nothing.
///
/// # Errors
///
/// Returns a [`PipelineError`] when a file cannot be read or written.
pub fn split(ctx: &Ctx<'_>) -> Result<SplitReport, PipelineError> {
    let topics = ctx.topics()?;
    let selected = |example: &Example| topics.iter().any(|topic| topic.name == example.topic);
    let examples: Vec<Example> = read(&ctx.files.answers)?;
    ctx.bus.publish(Event::StageStarted {
        stage: Stage::Split,
        total: examples.iter().filter(|example| selected(example)).count(),
    });
    let mut report = SplitReport::default();
    let mut usable = Vec::new();
    for example in examples {
        match example.meta.excluded {
            Some(reason) if selected(&example) => {
                *report.excluded.entry(reason).or_default() += 1;
            },
            Some(_) => {},
            None => usable.push(example),
        }
    }
    let settings = &ctx.settings.pipeline;
    let (train, eval) = stratify(usable, settings.eval_ratio, settings.seed);
    rewrite(&ctx.files.train, &train)?;
    rewrite(&ctx.files.eval, &eval)?;
    report.train = train.iter().filter(|example| selected(example)).count();
    report.eval = eval.iter().filter(|example| selected(example)).count();
    ctx.bus.publish(Event::StageFinished {
        stage: Stage::Split,
        stats: StageStats {
            done: report.train + report.eval,
            excluded: report.excluded.values().sum(),
            ..StageStats::default()
        },
    });
    Ok(report)
}

/// Splits `examples` into `(train, eval)`. Within each (topic, subtopic) group,
/// examples are ordered by a hash of their ID seeded with `seed`, and
/// `round(len * ratio)` of them go to eval. The result depends only on the IDs, the
/// ratio and the seed, not on file order.
#[must_use]
pub fn stratify(examples: Vec<Example>, ratio: f64, seed: u64) -> (Vec<Example>, Vec<Example>) {
    let mut groups: BTreeMap<(String, String), Vec<Example>> = BTreeMap::new();
    for example in examples {
        groups
            .entry((example.topic.clone(), example.subtopic.clone()))
            .or_default()
            .push(example);
    }
    let (mut train, mut eval) = (Vec::new(), Vec::new());
    for mut group in groups.into_values() {
        group.sort_by_key(|example| {
            (
                XxHash3_64::oneshot_with_seed(seed, example.id.as_str().as_bytes()),
                example.id.clone(),
            )
        });
        for (index, example) in group.into_iter().enumerate() {
            let index = f64::from(u32::try_from(index).unwrap_or(u32::MAX));
            if ((index + 1.0) * ratio).round() > (index * ratio).round() {
                eval.push(example);
            } else {
                train.push(example);
            }
        }
    }
    (train, eval)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{FinishReason, Id, Meta, ReasoningKind};

    fn example(subtopic: &str, n: usize) -> Example {
        Example {
            id: Id::of(&[subtopic, &n.to_string()]),
            topic: "t".into(),
            subtopic: subtopic.into(),
            messages: Vec::new(),
            meta: Meta {
                model: "m".into(),
                input_tokens: 0,
                output_tokens: 0,
                finish_reason: FinishReason::Stop,
                reasoning_kind: ReasoningKind::Raw,
                excluded: None,
            },
        }
    }

    fn dataset() -> Vec<Example> {
        (0..20)
            .map(|n| example("a", n))
            .chain((0..10).map(|n| example("b", n)))
            .collect()
    }

    #[test]
    fn each_subtopic_contributes_its_share_to_eval() {
        let (train, eval) = stratify(dataset(), 0.1, 42);
        assert_eq!(train.len() + eval.len(), 30);
        assert_eq!(eval.iter().filter(|e| e.subtopic == "a").count(), 2);
        assert_eq!(eval.iter().filter(|e| e.subtopic == "b").count(), 1);
    }

    #[test]
    fn split_is_deterministic_and_independent_of_input_order() {
        let (_, first) = stratify(dataset(), 0.2, 7);
        let mut reversed = dataset();
        reversed.reverse();
        let (_, second) = stratify(reversed, 0.2, 7);
        assert_eq!(first, second);
    }

    #[test]
    fn the_seed_changes_the_eval_set() {
        let (_, first) = stratify(dataset(), 0.2, 1);
        let (_, second) = stratify(dataset(), 0.2, 2);
        assert_ne!(first, second);
    }
}
