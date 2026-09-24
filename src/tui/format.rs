//! Text forms of times and quantities shared by the views.

use std::time::{SystemTime, UNIX_EPOCH};

/// What stands for running work where a spinner turns once motion is on.
pub(super) const WORKING: &str = "…";

/// Seconds since the Unix epoch, 0 before it.
pub(super) fn unix(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

/// `HH:MM:SS` in UTC.
pub(super) fn clock(time: SystemTime) -> String {
    let seconds = unix(time);
    format!(
        "{:02}:{:02}:{:02}",
        seconds / 3600 % 24,
        seconds / 60 % 60,
        seconds % 60
    )
}

/// A duration as `1h02m`, `41m` or `35s`.
pub(super) fn duration(duration: std::time::Duration) -> String {
    let seconds = duration.as_secs();
    match (seconds / 3600, seconds / 60 % 60) {
        (0, 0) => format!("{seconds}s"),
        (0, minutes) => format!("{minutes}m"),
        (hours, minutes) => format!("{hours}h{minutes:02}m"),
    }
}

/// `text` wrapped at `width` columns between words, its first line indented
/// by `indent` spaces and the others by two more: a hanging indent. Spaces
/// between words are kept; a word longer than a line stays whole.
pub(super) fn hang(text: &str, width: u16, indent: usize) -> Vec<String> {
    let width = usize::from(width);
    let mut lines = Vec::new();
    let mut line = " ".repeat(indent);
    let mut start = indent;
    for word in text.split(' ') {
        let used = line.chars().count();
        let fresh = used == start;
        if !fresh && !word.is_empty() && used + 1 + word.chars().count() > width {
            lines.push(std::mem::replace(&mut line, " ".repeat(indent + 2)));
            start = indent + 2;
        } else if !fresh {
            line.push(' ');
        }
        line.push_str(word);
    }
    lines.push(line);
    lines
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn a_clock_is_utc_hours_minutes_and_seconds() {
        let time = UNIX_EPOCH + Duration::from_secs(1_790_000_000);
        assert_eq!(clock(time), "14:13:20");
        assert_eq!(clock(UNIX_EPOCH), "00:00:00");
    }

    #[test]
    fn a_hanging_indent_wraps_on_words() {
        assert_eq!(hang("a b c", 5, 0), ["a b c"]);
        assert_eq!(hang("aaa bbb ccc", 7, 2), ["  aaa", "    bbb", "    ccc"]);
        assert_eq!(hang("aaa bbb ccc", 11, 2), ["  aaa bbb", "    ccc"]);
        assert_eq!(hang("", 10, 2), ["  "]);
        assert_eq!(hang("a  b", 10, 0), ["a  b"], "spaces between words kept");
        assert_eq!(hang("abcdefgh ij", 4, 0), ["abcdefgh", "  ij"]);
    }

    #[test]
    fn durations_show_hours_and_minutes_or_seconds() {
        assert_eq!(duration(Duration::from_secs(35)), "35s");
        assert_eq!(duration(Duration::from_secs(41 * 60 + 5)), "41m");
        assert_eq!(duration(Duration::from_secs(3720)), "1h02m");
    }
}
