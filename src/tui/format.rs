//! Text forms of times and quantities shared by the views.

use std::time::{SystemTime, UNIX_EPOCH};

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
    fn durations_show_hours_and_minutes_or_seconds() {
        assert_eq!(duration(Duration::from_secs(35)), "35s");
        assert_eq!(duration(Duration::from_secs(41 * 60 + 5)), "41m");
        assert_eq!(duration(Duration::from_secs(3720)), "1h02m");
    }
}
