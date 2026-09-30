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

/// Longest project part of a run ID, from [`project_slug`].
const SLUG_MAX: usize = 40;
/// Longest run ID [`is_valid_run_id`] accepts.
pub const RUN_ID_MAX: usize = 64;

/// The project part of a run ID: `name` in lowercase snake case. ASCII letters
/// and digits are kept, every other run of characters becomes one `_`, and the
/// result is trimmed and capped at 40 characters; `run` when nothing is left.
#[must_use]
pub fn project_slug(name: &str) -> String {
    let mut slug = String::new();
    let mut gap = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            if gap && !slug.is_empty() {
                slug.push('_');
            }
            gap = false;
            slug.push(c.to_ascii_lowercase());
        } else {
            gap = true;
        }
    }
    slug.truncate(SLUG_MAX);
    let slug = slug.trim_end_matches('_');
    if slug.is_empty() {
        "run".to_string()
    } else {
        slug.to_string()
    }
}

/// A new run ID: the [`project_slug`] of `project`, then the UTC date and time
/// of `now`, for example `malware_development_20260922-143005`.
#[must_use]
pub fn new_run_id(project: &str, now: SystemTime) -> String {
    let [year, month, day, hour, minute, second] = utc(now);
    format!(
        "{}_{year:04}{month:02}{day:02}-{hour:02}{minute:02}{second:02}",
        project_slug(project)
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
    // Up to 9999, as `rfc3339` writes them: this also keeps the arithmetic
    // below far from overflowing.
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
    let at = UNIX_EPOCH.checked_add(Duration::from_secs(seconds))?;
    // Only the exact form `rfc3339` writes: no sign, no leading zero, no
    // impossible date such as February 31.
    (rfc3339(at) == text).then_some(at)
}

/// `now` in compact UTC form, UTC, to the second, for use in file names:
/// `20260922T143005Z`.
#[must_use]
pub fn compact_utc(now: SystemTime) -> String {
    let [year, month, day, hour, minute, second] = utc(now);
    format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z")
}

/// Whether `id` can name a run directory: ASCII letters, digits, `_` and `-`,
/// starting with a letter or a digit, at most [`RUN_ID_MAX`] characters. Such a
/// name never leaves `runs/`, and is also safe in an ssh alias, a container name
/// and a pod name. IDs of the older `20260922-143005-a1b2` form stay valid.
#[must_use]
pub fn is_valid_run_id(id: &str) -> bool {
    is_safe_name(id, RUN_ID_MAX)
}

/// Whether `name` is non-empty, at most `max` characters of ASCII letters,
/// digits, `_` and `-`, and starts with a letter or a digit.
#[must_use]
pub fn is_safe_name(name: &str, max: usize) -> bool {
    name.len() <= max
        && name.starts_with(|c: char| c.is_ascii_alphanumeric())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
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
            "0000001970-01-01T00:00:00Z",
            "2026-9-21T14:13:20Z",
            "2026-09-21T+4:13:20Z",
            "2026-02-31T00:00:00Z",
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
    fn run_ids_are_the_project_then_the_time() {
        let id = new_run_id("Malware Development", at(1_790_000_000));
        assert_eq!(id, "malware_development_20260921-141320");
        assert!(is_valid_run_id(&id));
    }

    #[test]
    fn project_slugs_keep_lowercase_ascii_letters_and_digits() {
        assert_eq!(project_slug("Malware Development"), "malware_development");
        assert_eq!(project_slug("  GPT-4o mini!! v2 "), "gpt_4o_mini_v2");
        assert_eq!(project_slug("Café Über"), "caf_ber");
        assert_eq!(project_slug("already_snake_case"), "already_snake_case");
        assert_eq!(project_slug("模型"), "run");
        assert_eq!(project_slug("  --  "), "run");
        assert_eq!(project_slug(""), "run");
    }

    #[test]
    fn project_slugs_are_capped_without_a_trailing_underscore() {
        let long = "a".repeat(100);
        assert_eq!(project_slug(&long), "a".repeat(40));
        let cut_at_a_gap = format!("{} b", "a".repeat(39));
        assert_eq!(project_slug(&cut_at_a_gap), "a".repeat(39));
        let longest = new_run_id(&format!("{long}_{long}"), at(1_790_000_000));
        // With the `_99` a collision may add, an ID stays within the limit.
        assert!(is_valid_run_id(&format!("{longest}_99")), "{longest}");
    }

    #[test]
    fn old_run_ids_stay_valid() {
        assert!(is_valid_run_id("20260921-141320-a1b2"));
        assert!(is_valid_run_id("r1"));
        assert!(is_valid_run_id("malware_development_20260921-141320_2"));
        assert!(is_valid_run_id(&"a".repeat(64)));
    }

    #[test]
    fn run_ids_cannot_escape_the_runs_directory() {
        assert!(!is_valid_run_id("../x"));
        assert!(!is_valid_run_id("a/b"));
        assert!(!is_valid_run_id(""));
        assert!(!is_valid_run_id("."));
        assert!(!is_valid_run_id("a.b"));
        assert!(!is_valid_run_id("_x"));
        assert!(!is_valid_run_id("-x"));
        assert!(!is_valid_run_id("a b"));
        assert!(!is_valid_run_id("é"));
        assert!(!is_valid_run_id(&"a".repeat(65)));
    }

    #[test]
    fn safe_names_take_their_own_length_limit() {
        assert!(is_safe_name("overbrainer-a_b", 15));
        assert!(!is_safe_name("overbrainer-a_b", 14));
        assert!(!is_safe_name("", 10));
        assert!(!is_safe_name("_a", 10));
    }
}
