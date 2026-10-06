//! What a thousand requests cost, parent and child.

use serde::{Deserialize, Serialize};

use super::EvalQuestion;
use super::stats::count_f64;

/// Tokens in a million.
const MILLION: f64 = 1_000_000.0;
/// Requests in a cost row.
const REQUESTS: f64 = 1_000.0;
/// Seconds in an hour.
const HOUR: f64 = 3_600.0;

/// The prices of the cost rows, in dollars.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Prices {
    /// Per million input tokens of the parent.
    pub parent_in: Option<f64>,
    /// Per million output tokens of the parent.
    pub parent_out: Option<f64>,
    /// Per hour of the hardware serving the child.
    pub child_per_hour: Option<f64>,
}

/// The prices of a compare: the parent's from `compare`, the child's from
/// `compare` when set, else from the pod's hourly price.
///
/// On Runpod `child_price_per_hour` defaults to the pod's `cost_per_hour`;
/// elsewhere there is no default, so `pod_price_per_hour` is `None` there.
/// The flag is true only when the child price came from the pod, so a report
/// can say where it comes from.
#[must_use]
pub fn prices(compare: &crate::config::Compare, pod_price_per_hour: Option<f64>) -> (Prices, bool) {
    let from_pod = compare.child_price_per_hour.is_none() && pod_price_per_hour.is_some();
    let prices = Prices {
        parent_in: compare.parent_price_in,
        parent_out: compare.parent_price_out,
        child_per_hour: compare.child_price_per_hour.or(pod_price_per_hour),
    };
    (prices, from_pod)
}

/// Dollars per 1,000 requests.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Costs {
    /// The parent's, from its billed tokens; both parent prices needed.
    pub parent_per_1k: Option<f64>,
    /// The child's, from its mean time per request, requests one at a time:
    /// an upper bound.
    pub child_per_1k: Option<f64>,
    /// Child cost over parent cost, when both are known and the parent's is not 0.
    pub ratio: Option<f64>,
}

/// The cost rows of `questions` answered by the parent, and by the child in
/// `mean_seconds` per request, at `prices`.
///
/// The mean billed tokens are computed in `f64`; a token total above
/// `u32::MAX` is clamped, which no realistic eval set reaches.
#[must_use]
pub fn costs(prices: &Prices, questions: &[EvalQuestion], mean_seconds: Option<f64>) -> Costs {
    let parent_per_1k = match (prices.parent_in, prices.parent_out, questions.len()) {
        (Some(price_in), Some(price_out), count) if count > 0 => {
            let tokens = |pick: fn(&EvalQuestion) -> u64| {
                let total: u64 = questions.iter().map(pick).sum();
                f64::from(u32::try_from(total).unwrap_or(u32::MAX)) / count_f64(count)
            };
            let input = tokens(|question| question.parent_input_tokens);
            let output = tokens(|question| question.parent_output_tokens);
            Some(round6(
                (input * price_in + output * price_out) / MILLION * REQUESTS,
            ))
        },
        _ => None,
    };
    let child_per_1k = match (prices.child_per_hour, mean_seconds) {
        (Some(per_hour), Some(seconds)) => Some(round6(per_hour * seconds / HOUR * REQUESTS)),
        _ => None,
    };
    let ratio = match (child_per_1k, parent_per_1k) {
        (Some(child), Some(parent)) if parent > 0.0 => Some(child / parent),
        _ => None,
    };
    Costs {
        parent_per_1k,
        child_per_1k,
        ratio,
    }
}

/// `value` rounded to 6 decimals: dollar amounts without float noise.
fn round6(value: f64) -> f64 {
    (value * MILLION).round() / MILLION
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Compare;

    /// A question the parent answered with these tokens.
    fn question(input: u64, output: u64) -> EvalQuestion {
        EvalQuestion {
            id: "q".into(),
            topic: "t".into(),
            messages: Vec::new(),
            parent: String::new(),
            parent_input_tokens: input,
            parent_output_tokens: output,
        }
    }

    /// Whether `value` is `expected` within float noise.
    fn close(value: Option<f64>, expected: f64) -> bool {
        value.is_some_and(|value| (value - expected).abs() < 1e-9)
    }

    /// The parent's cost comes from its mean billed tokens, the child's from
    /// its mean time and the hourly price.
    #[test]
    fn costs_per_thousand_requests() {
        let questions = [question(100, 1_000), question(300, 3_000)];
        let prices = Prices {
            parent_in: Some(1.0),
            parent_out: Some(2.0),
            child_per_hour: Some(3.6),
        };
        let costs = costs(&prices, &questions, Some(0.5));
        // Mean 200 in and 2,000 out per request: (200 * 1 + 2000 * 2) / 1e6 * 1000.
        assert!(close(costs.parent_per_1k, 4.2));
        // 3.6 $/h for 0.5 s per request: 0.0005 $ per request.
        assert!(close(costs.child_per_1k, 0.5));
        assert_eq!(
            costs.ratio.map(|ratio| (ratio * 1000.0).round()),
            Some(119.0)
        );
    }

    /// A missing price leaves its row out, and the ratio with it.
    #[test]
    fn missing_prices_leave_rows_out() {
        let costs = costs(&Prices::default(), &[question(1, 1)], Some(1.0));
        assert_eq!(costs, Costs::default());
        let half = Prices {
            parent_in: Some(1.0),
            ..Prices::default()
        };
        assert_eq!(costs_of(&half), None);
    }

    /// The parent cost with `prices` on one question.
    fn costs_of(prices: &Prices) -> Option<f64> {
        costs(prices, &[question(1, 1)], None).parent_per_1k
    }

    /// A config with parent prices and the given child price.
    fn config(child: Option<f64>) -> Compare {
        Compare {
            parent_price_in: Some(3.0),
            parent_price_out: Some(15.0),
            child_price_per_hour: child,
            ..Compare::default()
        }
    }

    /// The configured child price wins over the pod's.
    #[test]
    fn the_configured_child_price_wins_over_the_pod() {
        let (prices, from_pod) = prices(&config(Some(1.5)), Some(0.7));
        assert_eq!(prices.child_per_hour, Some(1.5));
        assert!(!from_pod);
        assert_eq!(prices.parent_in, Some(3.0));
        assert_eq!(prices.parent_out, Some(15.0));
    }

    /// Without a configured price the pod's is used and flagged.
    #[test]
    fn the_pod_price_is_the_default() {
        let (prices, from_pod) = prices(&config(None), Some(0.7));
        assert_eq!(prices.child_per_hour, Some(0.7));
        assert!(from_pod);
    }

    /// Neither a configured nor a pod price: no child price.
    #[test]
    fn no_price_at_all_leaves_the_child_free() {
        let (prices, from_pod) = prices(&config(None), None);
        assert_eq!(prices.child_per_hour, None);
        assert!(!from_pod);
    }
}
