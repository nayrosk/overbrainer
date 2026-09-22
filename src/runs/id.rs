use std::time::{SystemTime, UNIX_EPOCH};

/// UTC calendar fields of `time`: year, month, day, hour, minute, second. Times
/// before 1970 read as 1970-01-01.
fn utc(time: SystemTime) -> [u64; 6] {
    let seconds = time
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    let (days, rest) = (seconds / 86_400, seconds % 86_400);
    // Days to civil date, from Howard Hinnant's `civil_from_days`, for days >= 0.
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    [year, month, day, rest / 3_600, rest % 3_600 / 60, rest % 60]
}

/// A new run ID: UTC date and time of `now`, then four random hex digits, for example
/// `20260922-143005-a1b2`. IDs sort by creation time.
#[must_use]
pub fn new_run_id(now: SystemTime) -> String {
    let [year, month, day, hour, minute, second] = utc(now);
    format!(
        "{year:04}{month:02}{day:02}-{hour:02}{minute:02}{second:02}-{:04x}",
        fastrand::u16(..)
    )
}

/// `now` in RFC 3339 form, UTC, to the second: `2026-09-22T14:30:05Z`.
#[must_use]
pub fn rfc3339(now: SystemTime) -> String {
    let [year, month, day, hour, minute, second] = utc(now);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Whether `id` can name a run directory: letters, digits and `-` only, so it
/// never leaves `runs/`.
#[must_use]
pub fn is_valid_run_id(id: &str) -> bool {
    !id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn at(seconds: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(seconds)
    }

    #[test]
    fn times_are_utc_calendar_dates() {
        assert_eq!(rfc3339(at(1_790_000_000)), "2026-09-21T14:13:20Z");
        assert_eq!(rfc3339(at(1_709_164_800)), "2024-02-29T00:00:00Z");
        assert_eq!(rfc3339(at(1_790_035_199)), "2026-09-21T23:59:59Z");
        assert_eq!(rfc3339(at(0)), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn run_ids_start_with_the_time() {
        let id = new_run_id(at(1_790_000_000));
        assert!(id.starts_with("20260921-141320-"), "{id}");
        assert_eq!(id.len(), 20);
        assert!(is_valid_run_id(&id));
    }

    #[test]
    fn run_ids_cannot_escape_the_runs_directory() {
        assert!(!is_valid_run_id("../x"));
        assert!(!is_valid_run_id("a/b"));
        assert!(!is_valid_run_id(""));
        assert!(!is_valid_run_id("."));
    }
}
