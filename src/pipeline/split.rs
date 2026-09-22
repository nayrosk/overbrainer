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

/// Splits `examples` into `(train, eval)`.
///
/// The eval set holds `max(1, round(n * ratio))` of the `n` examples (none when `n` is
/// below 2), with `ratio` in (0, 1). It is spread over the (topic, subtopic) groups in
/// proportion to their size: each group gets the integer part of its share, and the
/// examples left over go to the groups with the largest remainders, ties broken by
/// group key. Within a group, examples are ordered by a hash of their ID seeded with
/// `seed` and the first ones go to eval. The result depends only on the IDs, the ratio
/// and the seed, not on file order.
#[must_use]
pub fn stratify(examples: Vec<Example>, ratio: f64, seed: u64) -> (Vec<Example>, Vec<Example>) {
    let eval_total = eval_count(examples.len(), ratio);
    let mut groups: BTreeMap<(String, String), Vec<Example>> = BTreeMap::new();
    for example in examples {
        groups
            .entry((example.topic.clone(), example.subtopic.clone()))
            .or_default()
            .push(example);
    }
    let sizes: Vec<usize> = groups.values().map(Vec::len).collect();
    let quotas = allocate(&sizes, eval_total);
    let (mut train, mut eval) = (Vec::new(), Vec::new());
    for (mut group, quota) in groups.into_values().zip(quotas) {
        group.sort_by_key(|example| {
            (
                XxHash3_64::oneshot_with_seed(seed, example.id.as_str().as_bytes()),
                example.id.clone(),
            )
        });
        let rest = group.split_off(quota.min(group.len()));
        eval.extend(group);
        train.extend(rest);
    }
    (train, eval)
}

/// Size of the eval set for `total` usable examples: `max(1, round(total * ratio))`,
/// or 0 below 2 examples.
fn eval_count(total: usize, ratio: f64) -> usize {
    if total < 2 {
        return 0;
    }
    let rounded =
        |count: usize| (f64::from(u32::try_from(count).unwrap_or(u32::MAX)) * ratio).round();
    // round(total * ratio) as an integer, counted as the steps where the rounded share
    // goes up (by one at most, since ratio < 1), which avoids a float-to-integer cast.
    let steps = (0..total)
        .filter(|&index| rounded(index + 1) > rounded(index))
        .count();
    steps.clamp(1, total)
}

/// Spreads `eval_total` over groups of `sizes` by largest remainder, ties going to the
/// earlier group.
fn allocate(sizes: &[usize], eval_total: usize) -> Vec<usize> {
    let total: usize = sizes.iter().sum();
    if total == 0 {
        return vec![0; sizes.len()];
    }
    let mut quotas: Vec<usize> = sizes.iter().map(|size| size * eval_total / total).collect();
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|&group| std::cmp::Reverse(sizes[group] * eval_total % total));
    let left = eval_total.saturating_sub(quotas.iter().sum());
    for &group in order.iter().take(left) {
        quotas[group] += 1;
    }
    quotas
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

    /// `groups` subtopics of `size` examples each.
    fn uniform(groups: usize, size: usize) -> Vec<Example> {
        (0..groups)
            .flat_map(|group| (0..size).map(move |n| example(&format!("s{group}"), n)))
            .collect()
    }

    #[test]
    fn many_small_subtopics_still_give_an_eval_set() {
        let (train, eval) = stratify(uniform(20, 1), 0.1, 42);
        assert_eq!((train.len(), eval.len()), (18, 2));
        let (again_train, again_eval) = stratify(uniform(20, 1), 0.1, 42);
        assert_eq!((again_train, again_eval), (train, eval));
    }

    #[test]
    fn the_ratio_holds_on_the_whole_set() {
        let (train, eval) = stratify(uniform(7, 15), 0.1, 42);
        assert_eq!(eval.len(), 11, "round(105 * 0.1)");
        assert_eq!(train.len(), 94);
        for group in 0..7 {
            let name = format!("s{group}");
            let count = eval.iter().filter(|e| e.subtopic == name).count();
            assert!((1..=2).contains(&count), "{name} got {count}");
        }
    }

    #[test]
    fn at_least_one_eval_example_from_two_usable_ones() {
        assert_eq!(stratify(uniform(1, 1), 0.1, 42).1.len(), 0);
        assert_eq!(stratify(uniform(2, 1), 0.1, 42).1.len(), 1);
        assert_eq!(stratify(Vec::new(), 0.1, 42), (Vec::new(), Vec::new()));
    }

    #[test]
    fn the_seed_changes_the_eval_set() {
        let (_, first) = stratify(dataset(), 0.2, 1);
        let (_, second) = stratify(dataset(), 0.2, 2);
        assert_ne!(first, second);
    }
}
