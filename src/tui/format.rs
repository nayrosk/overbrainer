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
}
