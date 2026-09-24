//! The key table the help overlay shows, as data.

use super::app::View;

/// Width of the help overlay, borders included: it fits 80 columns.
pub(super) const HELP_WIDTH: u16 = 78;
/// Columns of padding inside the overlay's borders, on each side.
pub(super) const HELP_PADDING: u16 = 1;
/// Width of the overlay's key column.
pub(super) const KEYS_WIDTH: u16 = 26;
/// Room left for an action: the overlay less its two borders, its padding, the
/// key column and the space between the columns.
pub(super) const ACTION_WIDTH: u16 = HELP_WIDTH - 2 - 2 * HELP_PADDING - KEYS_WIDTH - 1;

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

/// The note under the keys: what the data lock refuses, and what it does not
/// cover.
pub(super) const NOTE: &str = "e, d, r and t are refused while a stage, an edit or a training \
                               start runs in this TUI; an overbrainer command in another \
                               terminal is not locked out.";

/// Keys that work in every view.
pub(super) const GLOBAL: &[KeyHelp] = &[
    row("1 2 3 4, Tab, Shift-Tab", "switch view"),
    row("?", "this help (Esc, ? or q closes it)"),
    row("q, Ctrl-C", "quit"),
    row("R", "reload the data files and runs"),
    row("r", "run a pipeline stage, or run (asks which)"),
    row("y, n Esc", "in a dialog: confirm, cancel (default: no)"),
];

const DATASET: &[KeyHelp] = &[
    row("k j, Up Down", "move"),
    row("l h, Right Left, Enter", "expand, collapse"),
    row("PgUp PgDn", "scroll the detail pane"),
    row("/", "filter the tree (Enter keeps, Esc clears)"),
    row("s", "stats pane"),
    row("e", "edit in $EDITOR (question, answer, subtopic)"),
    row("d", "delete, with what depends on it (asks first)"),
];

const TRAINING: &[KeyHelp] = &[
    row("k j, Up Down", "select a run"),
    row("a", "attach: follow the selected run again"),
    row("c", "cancel the selected run's job (asks first)"),
    row("c, Runpod run starting", "abandon it instead (asks first)"),
    row("t", "start a training run (asks first)"),
];

const LOGS: &[KeyHelp] = &[
    row("k j, Up Down, PgUp PgDn", "scroll"),
    row("G, End", "follow the newest lines"),
    row("f", "cycle level: error, warn, info, debug, trace"),
];

/// Keys of `view`.
pub(super) fn of(view: View) -> &'static [KeyHelp] {
    match view {
        View::Dataset => DATASET,
        View::Training => TRAINING,
        View::Logs => LOGS,
        View::Pipeline => &[],
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
