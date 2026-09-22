use std::collections::{BTreeMap, BTreeSet};

use twox_hash::XxHash3_64;

use super::{Ctx, PipelineError};
use crate::dataset::{Example, Exclusion, Id, Question, read, rewrite};
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
    /// Examples left out because their topic is no longer configured or their
    /// question is no longer in `data/questions.jsonl`.
    pub orphaned: usize,
}

/// Rewrites `data/train.jsonl` and `data/eval.jsonl` from the usable examples of
/// `data/answers.jsonl` (those with `meta.excluded = null`), stratified by subtopic.
///
/// Only examples whose topic is configured in `overbrainer.toml` and whose question is
/// still in `data/questions.jsonl` are used; the others (a removed topic, a question
/// replaced by `questions --force`) are counted as orphaned. Both files always hold
/// every topic's usable examples: `--topic` only limits the report and the event
/// counts to that topic. Excluded and orphaned examples stay in `answers.jsonl`. The
/// split always rewrites both files, so `--force` changes nothing.
///
/// # Errors
///
/// Returns a [`PipelineError`] when a file cannot be read or written.
pub fn split(ctx: &Ctx<'_>) -> Result<SplitReport, PipelineError> {
    ctx.topics()?;
    let counted = |example: &Example| ctx.topic.is_none_or(|name| example.topic == name);
    let examples: Vec<Example> = read(&ctx.files.answers)?;
    let questions: Vec<Question> = read(&ctx.files.questions)?;
    let known: BTreeSet<&Id> = questions.iter().map(|question| &question.id).collect();
    let configured = |example: &Example| {
        ctx.settings
            .topics
            .iter()
            .any(|topic| topic.name == example.topic)
    };
    ctx.bus.publish(Event::StageStarted {
        stage: Stage::Split,
        total: examples.iter().filter(|example| counted(example)).count(),
    });
    let mut report = SplitReport::default();
    let mut usable = Vec::new();
    let mut orphaned_usable = 0;
    for example in examples {
        if !configured(&example) || !known.contains(&example.id) {
            report.orphaned += usize::from(counted(&example));
            orphaned_usable += usize::from(example.meta.excluded.is_none());
            continue;
        }
        match example.meta.excluded {
            Some(reason) if counted(&example) => {
                *report.excluded.entry(reason).or_default() += 1;
            },
            Some(_) => {},
            None => usable.push(example),
        }
    }
    if every_usable_answer_orphaned(orphaned_usable, usable.len()) {
        warn_all_orphaned(orphaned_usable);
    }
    let settings = &ctx.settings.pipeline;
    let (train, eval) = stratify(usable, settings.eval_ratio, settings.seed);
    rewrite(&ctx.files.train, &train)?;
    rewrite(&ctx.files.eval, &eval)?;
    report.train = train.iter().filter(|example| counted(example)).count();
    report.eval = eval.iter().filter(|example| counted(example)).count();
    ctx.bus.publish(Event::StageFinished {
        stage: Stage::Split,
        stats: StageStats {
            done: report.train + report.eval,
            skipped: report.orphaned,
            excluded: report.excluded.values().sum(),
            ..StageStats::default()
        },
    });
    Ok(report)
}

/// Splits `examples` into `(train, eval)`.
///
/// The eval set holds `max(1, round(n * ratio))` of the `n` examples, at most `n - 1`
/// so train is never empty, and none when `n` is below 2 (`ratio` in (0, 1)). It is
/// spread in proportion to size in two steps: across topics, then within each topic
/// across its subtopics. At each step a group gets the integer part of its share, and
/// what is left over goes to the groups with the largest remainders, ties broken by a
/// hash of the group name seeded with `seed`. Within a subtopic, examples are ordered
/// by a hash of their ID seeded with `seed` and the first ones go to eval. The result
/// depends only on the topics, subtopics, IDs, the ratio and the seed, not on file
/// order.
#[must_use]
pub fn stratify(examples: Vec<Example>, ratio: f64, seed: u64) -> (Vec<Example>, Vec<Example>) {
    let eval_total = eval_count(examples.len(), ratio);
    let mut topics: BTreeMap<String, BTreeMap<String, Vec<Example>>> = BTreeMap::new();
    for example in examples {
        topics
            .entry(example.topic.clone())
            .or_default()
            .entry(example.subtopic.clone())
            .or_default()
            .push(example);
    }
    let sizes: Vec<(&str, usize)> = topics
        .iter()
        .map(|(topic, subtopics)| (topic.as_str(), subtopics.values().map(Vec::len).sum()))
        .collect();
    let topic_quotas = allocate(&sizes, eval_total, seed);
    let (mut train, mut eval) = (Vec::new(), Vec::new());
    for (subtopics, topic_quota) in topics.into_values().zip(topic_quotas) {
        let sizes: Vec<(&str, usize)> = subtopics
            .iter()
            .map(|(subtopic, group)| (subtopic.as_str(), group.len()))
            .collect();
        let quotas = allocate(&sizes, topic_quota, seed);
        for (group, quota) in subtopics.into_values().zip(quotas) {
            let (to_eval, to_train) = pick(group, quota, seed);
            eval.extend(to_eval);
            train.extend(to_train);
        }
    }
    (train, eval)
}

/// The first `quota` examples of `group` in seeded hash order, then the others.
fn pick(mut group: Vec<Example>, quota: usize, seed: u64) -> (Vec<Example>, Vec<Example>) {
    group.sort_by_key(|example| {
        (
            XxHash3_64::oneshot_with_seed(seed, example.id.as_str().as_bytes()),
            example.id.clone(),
        )
    });
    let rest = group.split_off(quota.min(group.len()));
    (group, rest)
}

/// Size of the eval set for `total` usable examples: `max(1, round(total * ratio))`
/// capped at `total - 1`, or 0 below 2 examples.
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
    steps.clamp(1, total - 1)
}

/// Whether answers would have been usable but all of them are orphaned.
fn every_usable_answer_orphaned(orphaned_usable: usize, usable: usize) -> bool {
    orphaned_usable > 0 && usable == 0
}

fn warn_all_orphaned(count: usize) {
    tracing::warn!(
        "all {count} usable answers are orphaned (topic no longer configured or question no longer in data/questions.jsonl): train and eval will be empty"
    );
}

/// Spreads `eval_total` over named groups of the given sizes by largest remainder,
/// ties broken by a hash of the name seeded with `seed`, then by name.
fn allocate(groups: &[(&str, usize)], eval_total: usize, seed: u64) -> Vec<usize> {
    let total: usize = groups.iter().map(|(_, size)| size).sum();
    if total == 0 {
        return vec![0; groups.len()];
    }
    let mut quotas: Vec<usize> = groups
        .iter()
        .map(|(_, size)| size * eval_total / total)
        .collect();
    let mut order: Vec<usize> = (0..groups.len()).collect();
    order.sort_by_key(|&index| {
        let (name, size) = groups[index];
        (
            std::cmp::Reverse(size * eval_total % total),
            XxHash3_64::oneshot_with_seed(seed, name.as_bytes()),
            name,
        )
    });
    let left = eval_total.saturating_sub(quotas.iter().sum());
    for &index in order.iter().take(left) {
        quotas[index] += 1;
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
        let sizes = |(train, eval): (Vec<Example>, Vec<Example>)| (train.len(), eval.len());
        assert_eq!(sizes(stratify(uniform(1, 1), 0.1, 42)), (1, 0));
        assert_eq!(sizes(stratify(uniform(2, 1), 0.1, 42)), (1, 1));
        assert_eq!(stratify(Vec::new(), 0.1, 42), (Vec::new(), Vec::new()));
    }

    #[test]
    fn train_is_never_empty() {
        let sizes = |(train, eval): (Vec<Example>, Vec<Example>)| (train.len(), eval.len());
        assert_eq!(sizes(stratify(uniform(2, 1), 0.9, 42)), (1, 1));
        assert_eq!(sizes(stratify(uniform(1, 1), 0.9, 42)), (1, 0));
        assert_eq!(sizes(stratify(uniform(1, 10), 0.99, 42)), (1, 9));
    }

    /// `topics` topics of `subtopics` subtopics of `size` examples each.
    fn grid(topics: &[&str], subtopics: usize, size: usize) -> Vec<Example> {
        let mut examples = Vec::new();
        for topic in topics {
            for subtopic in 0..subtopics {
                for n in 0..size {
                    let mut example = example(&format!("{topic}{subtopic}"), n);
                    example.topic = (*topic).to_string();
                    examples.push(example);
                }
            }
        }
        examples
    }

    fn per_topic(eval: &[Example]) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for example in eval {
            *counts.entry(example.topic.clone()).or_default() += 1;
        }
        counts
    }

    #[test]
    fn every_topic_gets_its_share_of_small_subtopics() {
        let (_, eval) = stratify(grid(&["a", "b", "c", "d"], 10, 5), 0.1, 42);
        assert_eq!(
            per_topic(&eval),
            BTreeMap::from([
                ("a".to_string(), 5),
                ("b".to_string(), 5),
                ("c".to_string(), 5),
                ("d".to_string(), 5),
            ])
        );
        let (_, eval) = stratify(grid(&["a", "b", "c"], 30, 3), 0.1, 42);
        assert_eq!(
            per_topic(&eval),
            BTreeMap::from([
                ("a".to_string(), 9),
                ("b".to_string(), 9),
                ("c".to_string(), 9),
            ])
        );
    }

    #[test]
    fn the_seed_picks_which_small_subtopics_go_to_eval() {
        let subtopics = |seed| -> BTreeSet<String> {
            let (_, eval) = stratify(grid(&["a", "b", "c"], 30, 3), 0.1, seed);
            eval.into_iter().map(|example| example.subtopic).collect()
        };
        assert_eq!(subtopics(1), subtopics(1));
        assert_ne!(subtopics(1), subtopics(2));
    }

    #[test]
    fn a_warning_is_due_only_when_every_usable_answer_is_orphaned() {
        assert!(every_usable_answer_orphaned(3, 0));
        assert!(!every_usable_answer_orphaned(0, 0));
        assert!(!every_usable_answer_orphaned(3, 1));
    }

    #[test]
    fn the_seed_changes_the_eval_set() {
        let (_, first) = stratify(dataset(), 0.2, 1);
        let (_, second) = stratify(dataset(), 0.2, 2);
        assert_ne!(first, second);
    }
}
