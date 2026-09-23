//! The key table the help overlay shows, as data.

use super::app::View;

/// One row of the help overlay.
pub(super) struct KeyHelp {
    /// The keys.
    pub(super) keys: &'static str,
    /// What they do.
    pub(super) action: &'static str,
}

const fn row(keys: &'static str, action: &'static str) -> KeyHelp {
    KeyHelp { keys, action }
}

/// Keys that work in every view.
pub(super) const GLOBAL: &[KeyHelp] = &[
    row("1 2 3 4, Tab, Shift-Tab", "switch view"),
    row("?", "this help (Esc or ? closes it)"),
    row("q, Ctrl-C", "quit"),
];

const LOGS: &[KeyHelp] = &[
    row("k j, Up Down, PgUp PgDn", "scroll"),
    row("G, End", "follow the newest lines"),
    row(
        "f",
        "cycle the level shown: error, warn, info, debug, trace",
    ),
];

/// Keys of `view`.
pub(super) fn of(view: View) -> &'static [KeyHelp] {
    match view {
        View::Logs => LOGS,
        View::Dataset | View::Pipeline | View::Training => &[],
    }
}
