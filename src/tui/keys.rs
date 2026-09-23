//! The key table the help overlay shows, as data.

use super::app::View;

/// Width of the help overlay, borders included.
pub(super) const HELP_WIDTH: u16 = 76;
/// Width of the overlay's key column.
pub(super) const KEYS_WIDTH: u16 = 26;
/// Room left for an action: the overlay less its two borders, the key column and
/// the space between the columns.
pub(super) const ACTION_WIDTH: u16 = HELP_WIDTH - 2 - KEYS_WIDTH - 1;

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
    row("f", "cycle level: error, warn, info, debug, trace"),
];

/// Keys of `view`.
pub(super) fn of(view: View) -> &'static [KeyHelp] {
    match view {
        View::Logs => LOGS,
        View::Dataset | View::Pipeline | View::Training => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row fits the overlay, so no text is cut at the minimum terminal size
    /// (80 columns, wider than the overlay) and up.
    #[test]
    fn every_row_fits_the_help_overlay() {
        let views = View::ALL.into_iter().flat_map(of);
        for KeyHelp { keys, action } in GLOBAL.iter().chain(views) {
            assert!(keys.chars().count() <= usize::from(KEYS_WIDTH), "{keys}");
            assert!(
                action.chars().count() <= usize::from(ACTION_WIDTH),
                "{action}"
            );
        }
    }
}
