//! What the verdicts and the child's timings add up to.

use serde::{Deserialize, Serialize};

use super::{ChildAnswer, Verdict, VerdictLine};

/// A count as a float, for rates and means.
#[must_use]
pub fn count_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}

/// The nearest-rank `percent` percentile of `sorted`, ascending; `None` when empty.
#[must_use]
pub fn percentile(sorted: &[f64], percent: usize) -> Option<f64> {
    let rank = percent.saturating_mul(sorted.len()).div_ceil(100).max(1);
    sorted.get(rank - 1).copied()
}

/// The mean of `values`; `None` when empty.
fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / count_f64(values.len()))
}

/// What a compare's verdicts and the child's timings add up to.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Summary {
    /// Questions asked.
    pub questions: usize,
    /// Child answers the judge preferred.
    pub wins: usize,
    /// Pairs the judge found as good.
    pub ties: usize,
    /// Parent answers the judge preferred.
    pub losses: usize,
    /// Replies of the judge that never parsed: not in the rate.
    pub unparsed: usize,
    /// Questions the child gave no answer to: counted as losses in the rate.
    pub errors: usize,
    /// Child answers cut at the token limit (`finish` is `length`): the
    /// judge sees them as they are, cut.
    #[serde(default)]
    pub truncated: usize,
    /// `(wins + ties) / (questions - unparsed)`, when any question counts.
    pub win_or_tie: Option<f64>,
    /// Median seconds per answer.
    pub latency_p50: Option<f64>,
    /// 95th percentile of the seconds per answer.
    pub latency_p95: Option<f64>,
    /// Median seconds to the first token.
    pub first_token_p50: Option<f64>,
    /// Mean completion tokens per second.
    pub tokens_per_second: Option<f64>,
    /// Mean seconds per answer.
    pub mean_seconds: Option<f64>,
}

/// Sums up `verdicts`, the timings of `answers` (failed answers have none)
/// and the answers cut at the token limit.
#[must_use]
pub fn summarize(verdicts: &[VerdictLine], answers: &[ChildAnswer]) -> Summary {
    let count = |verdict| {
        verdicts
            .iter()
            .filter(|line| line.verdict == verdict)
            .count()
    };
    let (wins, ties, losses, unparsed, errors) = (
        count(Verdict::Win),
        count(Verdict::Tie),
        count(Verdict::Loss),
        count(Verdict::Unparsed),
        count(Verdict::Error),
    );
    let counted = verdicts.len() - unparsed;
    let sorted = |pick: fn(&ChildAnswer) -> Option<f64>| {
        let mut values: Vec<f64> = answers
            .iter()
            .filter(|answer| !answer.is_error())
            .filter_map(pick)
            .filter(|value| value.is_finite())
            .collect();
        values.sort_by(f64::total_cmp);
        values
    };
    let seconds = sorted(|answer| answer.seconds);
    let first = sorted(|answer| answer.first_token_seconds);
    let rates = sorted(|answer| answer.tokens_per_second);
    Summary {
        questions: verdicts.len(),
        wins,
        ties,
        losses,
        unparsed,
        errors,
        truncated: answers
            .iter()
            .filter(|answer| !answer.is_error() && answer.finish.as_deref() == Some("length"))
            .count(),
        win_or_tie: (counted > 0).then(|| count_f64(wins + ties) / count_f64(counted)),
        latency_p50: percentile(&seconds, 50),
        latency_p95: percentile(&seconds, 95),
        first_token_p50: percentile(&first, 50),
        tokens_per_second: mean(&rates),
        mean_seconds: mean(&seconds),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A verdict line for the tests.
    fn line(id: &str, verdict: Verdict) -> VerdictLine {
        VerdictLine {
            id: id.into(),
            verdict,
            reason: None,
            child_first: true,
        }
    }

    /// A child answer that took `seconds`.
    fn answer(id: &str, seconds: f64) -> ChildAnswer {
        ChildAnswer {
            id: id.into(),
            answer: Some("A.".into()),
            seconds: Some(seconds),
            first_token_seconds: Some(seconds / 10.0),
            tokens_per_second: Some(50.0),
            ..ChildAnswer::default()
        }
    }

    /// Whether `value` is `expected` within float noise.
    fn close(value: Option<f64>, expected: f64) -> bool {
        value.is_some_and(|value| (value - expected).abs() < 1e-9)
    }

    /// Nearest-rank percentiles, none of nothing.
    #[test]
    fn percentiles_use_the_nearest_rank() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&sorted, 50), Some(5.0));
        assert_eq!(percentile(&sorted, 95), Some(10.0));
        assert_eq!(percentile(&[3.0], 95), Some(3.0));
        assert_eq!(percentile(&[], 50), None);
    }

    /// Unparsed verdicts are left out of the rate; child errors count as losses.
    #[test]
    fn the_rate_leaves_out_unparsed_and_counts_errors_as_losses() {
        let verdicts = [
            line("q1", Verdict::Win),
            line("q2", Verdict::Tie),
            line("q3", Verdict::Loss),
            line("q4", Verdict::Unparsed),
            line("q5", Verdict::Error),
        ];
        let answers = [
            answer("q1", 1.0),
            answer("q2", 2.0),
            answer("q3", 3.0),
            answer("q4", 4.0),
        ];
        let summary = summarize(&verdicts, &answers);
        assert_eq!(
            (
                summary.questions,
                summary.wins,
                summary.ties,
                summary.losses,
                summary.unparsed,
                summary.errors
            ),
            (5, 1, 1, 1, 1, 1)
        );
        assert!(close(summary.win_or_tie, 0.5));
        assert!(close(summary.latency_p50, 2.0));
        assert!(close(summary.latency_p95, 4.0));
        assert!(close(summary.mean_seconds, 2.5));
        assert!(close(summary.tokens_per_second, 50.0));
    }

    /// Child answers cut at the token limit (`finish` is `length`) are
    /// counted; failed answers and other finish reasons are not.
    #[test]
    fn answers_cut_at_the_token_limit_are_counted() {
        let cut = |id: &str, finish: &str| ChildAnswer {
            finish: Some(finish.into()),
            ..answer(id, 1.0)
        };
        let failed = ChildAnswer {
            id: "q4".into(),
            finish: Some("length".into()),
            error: Some("timeout".into()),
            ..ChildAnswer::default()
        };
        let answers = [
            cut("q1", "length"),
            cut("q2", "length"),
            cut("q3", "stop"),
            failed,
        ];
        let summary = summarize(&[line("q1", Verdict::Loss)], &answers);
        assert_eq!(summary.truncated, 2);
        assert_eq!(summarize(&[], &[]).truncated, 0);
    }

    /// Nothing judged: no rate, no latency.
    #[test]
    fn an_empty_compare_has_no_rate() {
        let summary = summarize(&[], &[]);
        assert_eq!(summary.win_or_tie, None);
        assert_eq!(summary.latency_p50, None);
    }
}
