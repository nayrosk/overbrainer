//! Retries with exponential backoff and jitter, shared by the HTTP clients (the LLM
//! providers and the Runpod API).

use std::future::Future;
use std::time::Duration;

/// Longest `Retry-After` honored. A server asking for more is waited on this long.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(600);

/// An error that may be worth trying again.
pub trait Retryable {
    /// Whether the failed operation is worth trying again.
    fn is_retryable(&self) -> bool;

    /// How long the server asked to wait before trying again, when it said so.
    fn retry_after(&self) -> Option<Duration>;
}

/// Exponential backoff with jitter, honoring `Retry-After`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Retries after the first attempt.
    pub max_retries: u32,
    /// Delay before the first retry, before jitter.
    pub base: Duration,
    /// Upper bound of the exponential delay.
    pub cap: Duration,
}

impl RetryPolicy {
    /// One second base, doubling up to one minute.
    #[must_use]
    pub fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            base: Duration::from_secs(1),
            cap: Duration::from_secs(60),
        }
    }

    /// Delay before retry number `attempt` (0 for the first retry).
    ///
    /// `retry_after` wins when present. Otherwise the delay is `base * 2^attempt`,
    /// capped, then scaled by a factor between 0.5 and 1 chosen by `jitter` in [0, 1].
    #[must_use]
    pub fn delay(&self, attempt: u32, retry_after: Option<Duration>, jitter: f64) -> Duration {
        if let Some(wait) = retry_after {
            return wait.min(MAX_RETRY_AFTER);
        }
        let exponential = self
            .base
            .saturating_mul(2_u32.saturating_pow(attempt))
            .min(self.cap);
        exponential.mul_f64(0.5 + 0.5 * jitter.clamp(0.0, 1.0))
    }
}

/// Runs `operation` until it succeeds, fails with a non-retryable error, or has been
/// retried `policy.max_retries` times. `on_retry` is called before each wait.
///
/// # Errors
///
/// Returns the last error when it is not retryable or when retries are exhausted.
pub async fn with_retry<T, E, F, Fut, R>(
    policy: &RetryPolicy,
    mut operation: F,
    mut on_retry: R,
) -> Result<T, E>
where
    E: Retryable,
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    R: FnMut(&E, Duration),
{
    let mut attempt = 0;
    loop {
        match operation().await {
            Ok(value) => return Ok(value),
            Err(error) if error.is_retryable() && attempt < policy.max_retries => {
                let wait = policy.delay(attempt, error.retry_after(), fastrand::f64());
                on_retry(&error, wait);
                tokio::time::sleep(wait).await;
                attempt += 1;
            },
            Err(error) => return Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    /// An error whose retryability the test chooses.
    #[derive(Debug, PartialEq, Eq)]
    struct Failure {
        retryable: bool,
    }

    impl Retryable for Failure {
        fn is_retryable(&self) -> bool {
            self.retryable
        }

        fn retry_after(&self) -> Option<Duration> {
            None
        }
    }

    #[test]
    fn delay_doubles_until_the_cap() {
        let policy = RetryPolicy::new(5);
        assert_eq!(policy.delay(0, None, 1.0), Duration::from_secs(1));
        assert_eq!(policy.delay(3, None, 1.0), Duration::from_secs(8));
        assert_eq!(policy.delay(10, None, 1.0), Duration::from_secs(60));
        assert_eq!(policy.delay(40, None, 1.0), Duration::from_secs(60));
    }

    #[test]
    fn jitter_scales_between_half_and_full() {
        let policy = RetryPolicy::new(5);
        assert_eq!(policy.delay(2, None, 0.0), Duration::from_secs(2));
        assert_eq!(policy.delay(2, None, 0.5), Duration::from_secs(3));
        assert_eq!(policy.delay(2, None, 7.0), Duration::from_secs(4));
    }

    #[test]
    fn retry_after_wins_and_is_bounded() {
        let policy = RetryPolicy::new(5);
        assert_eq!(
            policy.delay(0, Some(Duration::from_secs(30)), 0.0),
            Duration::from_secs(30)
        );
        assert_eq!(
            policy.delay(0, Some(Duration::from_secs(9_999)), 0.0),
            Duration::from_secs(600)
        );
    }

    fn fast(max_retries: u32) -> RetryPolicy {
        RetryPolicy {
            max_retries,
            base: Duration::from_millis(1),
            cap: Duration::from_millis(2),
        }
    }

    #[tokio::test]
    async fn retries_retryable_errors_then_succeeds() -> Result<(), Failure> {
        let calls = AtomicU32::new(0);
        let mut retries = 0;
        let value = with_retry(
            &fast(3),
            || async {
                if calls.fetch_add(1, Ordering::SeqCst) < 2 {
                    Err(Failure { retryable: true })
                } else {
                    Ok(7)
                }
            },
            |_, _| retries += 1,
        )
        .await?;
        assert_eq!(value, 7);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert_eq!(retries, 2);
        Ok(())
    }

    #[tokio::test]
    async fn fatal_errors_are_not_retried() {
        let calls = AtomicU32::new(0);
        let result: Result<(), _> = with_retry(
            &fast(3),
            || async {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(Failure { retryable: false })
            },
            |_, _| {},
        )
        .await;
        assert_eq!(result, Err(Failure { retryable: false }));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn gives_up_after_max_retries() {
        let calls = AtomicU32::new(0);
        let result: Result<(), _> = with_retry(
            &fast(2),
            || async {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(Failure { retryable: true })
            },
            |_, _| {},
        )
        .await;
        assert_eq!(result, Err(Failure { retryable: true }));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }
}
