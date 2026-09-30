use std::time::{Duration, SystemTime, UNIX_EPOCH};

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

/// The time `text` gives in the form [`rfc3339`] writes, `2026-09-22T14:30:05Z`;
/// `None` for any other form, or a year outside 1970 to 9999.
#[must_use]
pub fn parse_rfc3339(text: &str) -> Option<SystemTime> {
    let (date, time) = text.strip_suffix('Z')?.split_once('T')?;
    let fields: Vec<u64> = date
        .split('-')
        .chain(time.split(':'))
        .map(|field| field.parse().ok())
        .collect::<Option<_>>()?;
    let [year, month, day, hour, minute, second] = fields[..] else {
        return None;
    };
    // Four-digit years only, as `rfc3339` writes them: this also keeps the
    // arithmetic below far from overflowing.
    if !(1970..=9999).contains(&year) || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    if hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    // Civil date to days, from Howard Hinnant's `days_from_civil`, for years
    // from 1970.
    let year = if month <= 2 { year - 1 } else { year };
    let era = year / 400;
    let yoe = year - era * 400;
    let doy = (153 * ((month + 9) % 12) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = (era * 146_097 + doe).checked_sub(719_468)?;
    let seconds = days * 86_400 + hour * 3_600 + minute * 60 + second;
    UNIX_EPOCH.checked_add(Duration::from_secs(seconds))
}

/// `now` in compact UTC form, UTC, to the second, for use in file names:
/// `20260922T143005Z`.
#[must_use]
pub fn compact_utc(now: SystemTime) -> String {
    let [year, month, day, hour, minute, second] = utc(now);
    format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z")
}

/// Whether `id` can name a run directory: letters, digits and `-` only, so it
/// never leaves `runs/`.
#[must_use]
pub fn is_valid_run_id(id: &str) -> bool {
    !id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
}

#[cfg(test)]
mod tests {
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
    fn rfc3339_times_read_back() {
        for seconds in [
            0,
            1_709_164_800,
            1_790_000_000,
            1_790_035_199,
            4_102_444_800,
        ] {
            assert_eq!(parse_rfc3339(&rfc3339(at(seconds))), Some(at(seconds)));
        }
        for text in [
            "2026-09-21T14:13:20",
            "2026-09-21 14:13:20Z",
            "2026-13-21T14:13:20Z",
            "2026-09-21T24:00:00Z",
            "1969-12-31T23:59:59Z",
            "10000-01-01T00:00:00Z",
            "18446744073709551615-01-01T00:00:00Z",
            "2026-09-21T14:13Z",
            "",
        ] {
            assert_eq!(parse_rfc3339(text), None, "{text}");
        }
    }

    #[test]
    fn compact_utc_gives_a_sortable_file_name() {
        assert_eq!(compact_utc(at(1_790_000_000)), "20260921T141320Z");
        assert_eq!(compact_utc(at(0)), "19700101T000000Z");
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
