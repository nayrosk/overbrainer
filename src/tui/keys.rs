//! The key tables the help overlay and the footer show, as data.

use super::app::View;
use super::widgets::picker::Mode;

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

/// The note under the keys: what the data lock refuses, and the project lock the
/// TUI holds.
pub(super) const NOTE: &str = "e, d, r, A, t, h, C and J are refused while a stage, an edit or \
                               a training start runs in this TUI; the TUI also holds the project lock, \
                               so no other overbrainer command writes to the project \
                               meanwhile.";

/// Keys that work in every view.
pub(super) const GLOBAL: &[KeyHelp] = &[
    row("1-6, Tab, Shift-Tab", "switch view"),
    row("?", "this help (Esc, ? or q closes it)"),
    row("q, Ctrl-C", "quit"),
    row("R", "reload the data files and runs"),
    row("g", "open the overbrainer repository in a browser"),
    row("r, A", "run auto or a stage (asks which); A: auto"),
    row(
        "y, n Esc Enter",
        "in a dialog: confirm, cancel (the default, n)",
    ),
];

const PROJECT: &[KeyHelp] = &[
    row(
        "k j, PgUp PgDn, Home End",
        "select a field, by page, the first or last",
    ),
    row("Enter", "edit and save; toggles a bool, cycles a choice"),
    row(
        "t o, in a Runpod picker",
        "type the value instead, sort another way",
    ),
    row("a", "add a topic, a provider or a target"),
    row("d", "delete the selected topic, provider or target"),
    row("u", "undo the last write to overbrainer.toml"),
    row("E", "open overbrainer.toml in $EDITOR"),
];

const DATASET: &[KeyHelp] = &[
    row("k j, Up Down", "move"),
    row("l h, Right Left, Enter", "expand, collapse (Enter toggles)"),
    row(
        "PgUp PgDn, [ ]",
        "scroll the detail pane; [ ] jumps part to part",
    ),
    row("/", "filter the tree (Enter keeps, Esc clears)"),
    row("s", "stats pane"),
    row("e, E", "edit in $EDITOR; E edits the question's answer"),
    row("d", "delete, with what depends on it (asks first)"),
    row("D", "delete the question's answer only (asks first)"),
];

/// The help rows of the training view.
const TRAINING: &[KeyHelp] = &[
    row("k j, Up Down", "select a run"),
    row("a, p", "attach the run again; p hides or shows its pod"),
    row("c", "cancel the job; abandons a Runpod start (asks)"),
    row("t", "start a training run (asks first)"),
    row("s, T", "stop with a snapshot; T resumes it (asks first)"),
    row("x", "hide the failed runs until restart (asks first)"),
    row("h, C", "push to Hugging Face; compare it (both ask)"),
    row(
        "g c, starting on Runpod",
        "choose GPU types, data centers (saved on y)",
    ),
];

const PIPELINE: &[KeyHelp] = &[row(
    "c, auto mode running",
    "cancel its stage and the rest (asks first)",
)];

/// The help rows of the Compare view.
const COMPARE: &[KeyHelp] = &[
    row("k j, Up Down", "move in the focused list"),
    row("h l, Left Right", "focus the compares or the questions"),
    row(
        "PgUp PgDn, [ ]",
        "scroll the detail; [ ] jumps part to part",
    ),
    row("f", "filter: all, losses, ties, wins, errors"),
    row("C", "compare the newest run with a GGUF (asks first)"),
    row("J", "judge the selected compare again (asks first)"),
    row("c", "cancel the compare running (asks first)"),
];

const LOGS: &[KeyHelp] = &[
    row("k j, Up Down, PgUp PgDn", "scroll"),
    row("G, End", "follow the newest lines"),
    row("f", "cycle level: error, warn, info, debug, trace"),
    row("s", "toggle overbrainer / selected run's pod"),
    row("x", "export the shown lines to .overbrainer/"),
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
    /// A value or a name being typed in the Project view's form.
    Form,
    /// A choice being made in the Project view's form.
    Pick,
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
    /// A picker's entries.
    Picker {
        /// How entries are chosen.
        mode: Mode,
        /// Whether `t` types the value instead.
        typed: bool,
    },
    /// A picker whose entries are being read, or cannot be.
    Listing {
        /// Whether `t` types the value instead.
        typed: bool,
    },
    /// The dialog starting a run on a Runpod target: `g` and `c` choose its
    /// GPU types and data centers.
    Start,
}

const FOOTER_PROJECT: &[Hint] = &[
    hint("j/k", "move"),
    hint("Enter", "edit"),
    hint("a", "add"),
    hint("d", "delete"),
    hint("u", "undo"),
    hint("E", "$EDITOR"),
];
const FOOTER_DATASET: &[Hint] = &[
    hint("j/k", "move"),
    hint("l", "open"),
    hint("/", "filter"),
    hint("s", "stats"),
    locking("e", "edit"),
    locking("d", "delete"),
    hint("[ ]", "parts"),
];
const FOOTER_PIPELINE: &[Hint] = &[
    locking("r", "run a stage"),
    locking("A", "auto"),
    hint("q", "quit"),
    hint("1-6", "views"),
];
/// The footer hints of the training view.
const FOOTER_TRAINING: &[Hint] = &[
    hint("j/k", "select"),
    hint("a", "attach"),
    hint("c", "cancel"),
    locking("t", "start"),
    locking("h", "push"),
    hint("x", "clear"),
    hint("p", "pod"),
];
/// The footer hints of the Training view on a Runpod run still starting, where `c` abandons it.
const FOOTER_ABANDON: &[Hint] = &[
    hint("j/k", "select"),
    hint("a", "attach"),
    hint("c", "abandon"),
    locking("t", "start"),
    locking("h", "push"),
    hint("x", "clear"),
    hint("p", "pod"),
];
const FOOTER_LOGS: &[Hint] = &[
    hint("j/k", "scroll"),
    hint("G", "follow"),
    hint("f", "level"),
    hint("s", "source"),
    hint("x", "export"),
];
/// The footer hints of the Compare view.
const FOOTER_COMPARE: &[Hint] = &[
    hint("j/k", "move"),
    hint("h/l", "focus"),
    hint("f", "filter"),
    locking("C", "compare"),
    locking("J", "rejudge"),
    hint("c", "cancel"),
];
const FOOTER_FILTER: &[Hint] = &[hint("Enter", "keep"), hint("Esc", "clear")];
const FOOTER_FORM: &[Hint] = &[hint("Enter", "save"), hint("Esc", "cancel")];
const FOOTER_PICK: &[Hint] = &[
    hint("←/→", "choose"),
    hint("Enter", "pick"),
    hint("Esc", "cancel"),
];
const FOOTER_HELP: &[Hint] = &[hint("Esc", "close")];
const FOOTER_LISTING: &[Hint] = &[hint("t", "type"), hint("Esc", "close")];
const FOOTER_PICKER: &[Hint] = &[
    hint("Space", "toggle"),
    hint("J/K", "order"),
    hint("/", "filter"),
    hint("Enter", "keep"),
    hint("t", "type"),
    hint("Esc", "cancel"),
];
const FOOTER_PICK_ONE: &[Hint] = &[
    hint("j/k", "move"),
    hint("/", "filter"),
    hint("Enter", "pick"),
    hint("t", "type"),
    hint("Esc", "cancel"),
];
const FOOTER_START: &[Hint] = &[
    hint("y", "start"),
    hint("g", "GPU types"),
    hint("c", "data centers"),
    hint("n, Esc or Enter", "cancel"),
];
const FOOTER_MENU: &[Hint] = &[
    hint("j/k", "move"),
    hint("Enter", "run"),
    hint("Esc", "close"),
];

/// The footer's hints in `context`, most useful first: the footer drops them
/// from the end when it runs out of room.
pub(super) fn footer(context: Context) -> Vec<Hint> {
    match context {
        Context::View(View::Project) => FOOTER_PROJECT.to_vec(),
        Context::View(View::Dataset) => FOOTER_DATASET.to_vec(),
        Context::View(View::Pipeline) => FOOTER_PIPELINE.to_vec(),
        Context::View(View::Training) => FOOTER_TRAINING.to_vec(),
        Context::View(View::Logs) => FOOTER_LOGS.to_vec(),
        Context::View(View::Compare) => FOOTER_COMPARE.to_vec(),
        Context::Abandon => FOOTER_ABANDON.to_vec(),
        Context::Filter => FOOTER_FILTER.to_vec(),
        Context::Form => FOOTER_FORM.to_vec(),
        Context::Pick => FOOTER_PICK.to_vec(),
        Context::Dialog { yes, no } => vec![hint("y", yes), hint("n, Esc or Enter", no)],
        Context::Help => FOOTER_HELP.to_vec(),
        Context::Listing { typed } => untyped(FOOTER_LISTING, typed),
        Context::Menu => FOOTER_MENU.to_vec(),
        Context::Picker {
            mode: Mode::Multi,
            typed,
        } => untyped(FOOTER_PICKER, typed),
        Context::Picker {
            mode: Mode::Single,
            typed,
        } => untyped(FOOTER_PICK_ONE, typed),
        Context::Start => FOOTER_START.to_vec(),
    }
}

/// `hints`, without `t` unless `typed`.
fn untyped(hints: &[Hint], typed: bool) -> Vec<Hint> {
    hints
        .iter()
        .filter(|hint| typed || hint.key != "t")
        .copied()
        .collect()
}

/// Keys of `view`.
pub(super) fn of(view: View) -> &'static [KeyHelp] {
    match view {
        View::Project => PROJECT,
        View::Dataset => DATASET,
        View::Training => TRAINING,
        View::Logs => LOGS,
        View::Pipeline => PIPELINE,
        View::Compare => COMPARE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `h` is in the Training view's help and footer, as a locking key.
    #[test]
    fn h_pushes_to_hugging_face() {
        assert!(
            TRAINING
                .iter()
                .any(|row| row.keys == "h, C" && row.action.starts_with("push to Hugging Face")),
            "h in the help"
        );
        for context in [Context::View(View::Training), Context::Abandon] {
            assert!(
                footer(context).contains(&locking("h", "push")),
                "{context:?}"
            );
        }
        assert!(
            NOTE.starts_with("e, d, r, A, t, h, C and J are refused"),
            "{NOTE}"
        );
    }

    /// `C` compares in the Training and Compare views; `C` and `J` are
    /// locking keys of the Compare view, as `h` is of the Training view.
    #[test]
    fn c_compares_and_j_judges_again() {
        assert!(
            TRAINING
                .iter()
                .any(|row| row.keys == "h, C" && row.action.ends_with("compare it (both ask)")),
            "C in the Training help"
        );
        assert!(
            COMPARE.iter().any(|row| row.keys == "J"),
            "J in the Compare help"
        );
        let hints = footer(Context::View(View::Compare));
        assert!(hints.contains(&locking("C", "compare")), "{hints:?}");
        assert!(hints.contains(&locking("J", "rejudge")), "{hints:?}");
    }

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
            Context::Form,
            Context::Pick,
            dialog,
            Context::Help,
            Context::Menu,
            Context::Picker {
                mode: Mode::Multi,
                typed: true,
            },
            Context::Picker {
                mode: Mode::Single,
                typed: true,
            },
            Context::Listing { typed: true },
            Context::Start,
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
