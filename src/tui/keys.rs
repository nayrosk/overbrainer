//! The key tables the help overlay and the footer show, as data.

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
    row(
        "y, n Esc Enter",
        "in a dialog: confirm, cancel (the default, n)",
    ),
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

/// One key hint of the footer: a key and what it does, in a word or two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Hint {
    /// The key.
    pub(super) key: &'static str,
    /// What it does.
    pub(super) label: &'static str,
    /// Whether the data lock refuses it: drawn crossed out while it holds.
    pub(super) locks: bool,
}

const fn hint(key: &'static str, label: &'static str) -> Hint {
    Hint {
        key,
        label,
        locks: false,
    }
}

const fn locking(key: &'static str, label: &'static str) -> Hint {
    Hint {
        key,
        label,
        locks: true,
    }
}

/// What separates two hints, and two pieces of work.
pub(super) const SEPARATOR: &str = " · ";
/// The last hint on the right of the footer, always shown.
pub(super) const HELP_HINT: &str = "? help";

/// What the keys act on, which picks the footer's hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Context {
    /// A view, with no overlay.
    View(View),
    /// The Training view on a Runpod run still starting: `c` abandons it.
    Abandon,
    /// The filter being typed.
    Filter,
    /// A dialog answered with `y`, labelled `yes` and `no`.
    Dialog {
        /// What `y` does.
        yes: &'static str,
        /// What `n`, Esc and Enter do.
        no: &'static str,
    },
    /// The help overlay.
    Help,
    /// The `r` menu.
    Menu,
}

const FOOTER_DATASET: &[Hint] = &[
    hint("j/k", "move"),
    hint("l", "open"),
    hint("/", "filter"),
    hint("s", "stats"),
    locking("e", "edit"),
    locking("d", "delete"),
];
const FOOTER_PIPELINE: &[Hint] = &[
    locking("r", "run a stage"),
    hint("1-4", "views"),
    hint("q", "quit"),
];
const FOOTER_TRAINING: &[Hint] = &[
    hint("j/k", "select"),
    hint("a", "attach"),
    hint("c", "cancel"),
    locking("t", "start"),
];
const FOOTER_ABANDON: &[Hint] = &[
    hint("j/k", "select"),
    hint("a", "attach"),
    hint("c", "abandon"),
    locking("t", "start"),
];
const FOOTER_LOGS: &[Hint] = &[
    hint("j/k", "scroll"),
    hint("G", "follow"),
    hint("f", "level"),
];
const FOOTER_FILTER: &[Hint] = &[hint("Enter", "keep"), hint("Esc", "clear")];
const FOOTER_HELP: &[Hint] = &[hint("Esc", "close")];
const FOOTER_MENU: &[Hint] = &[
    hint("j/k", "move"),
    hint("Enter", "run"),
    hint("Esc", "close"),
];

/// The footer's hints in `context`, most useful first: the footer drops them
/// from the end when it runs out of room.
pub(super) fn footer(context: Context) -> Vec<Hint> {
    match context {
        Context::View(View::Dataset) => FOOTER_DATASET.to_vec(),
        Context::View(View::Pipeline) => FOOTER_PIPELINE.to_vec(),
        Context::View(View::Training) => FOOTER_TRAINING.to_vec(),
        Context::View(View::Logs) => FOOTER_LOGS.to_vec(),
        Context::Abandon => FOOTER_ABANDON.to_vec(),
        Context::Filter => FOOTER_FILTER.to_vec(),
        Context::Dialog { yes, no } => vec![hint("y", yes), hint("n, Esc or Enter", no)],
        Context::Help => FOOTER_HELP.to_vec(),
        Context::Menu => FOOTER_MENU.to_vec(),
    }
}

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

    /// Every context's hints fit 80 columns with `? help` and the margins, so
    /// none is dropped when no work is shown.
    #[test]
    fn every_footer_fits_80_columns_with_help() {
        let dialog = Context::Dialog {
            yes: "cancel the run",
            no: "keep it",
        };
        let others = [
            Context::Abandon,
            Context::Filter,
            dialog,
            Context::Help,
            Context::Menu,
        ];
        for context in View::ALL.map(Context::View).into_iter().chain(others) {
            let hints = footer(context);
            let text: Vec<String> = hints
                .iter()
                .map(|hint| format!("{} {}", hint.key, hint.label))
                .collect();
            let line = format!(" {}  {HELP_HINT} ", text.join(SEPARATOR));
            assert!(line.chars().count() <= 80, "{context:?}: {line}");
        }
    }
}
