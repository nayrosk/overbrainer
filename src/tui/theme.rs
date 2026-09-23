//! The styles the views use: named ANSI colors, or modifiers only under `NO_COLOR`.

use ratatui::style::{Color, Modifier, Style};
use tracing::Level;

/// The styles of every view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Theme {
    /// Block titles and the header.
    pub(super) title: Style,
    /// The selected tab, row or tree node.
    pub(super) selected: Style,
    /// Secondary text.
    pub(super) dim: Style,
    /// Success and plain information.
    pub(super) ok: Style,
    /// Warnings.
    pub(super) warn: Style,
    /// Errors.
    pub(super) error: Style,
    /// Keys in the help overlay and dialogs.
    pub(super) key: Style,
}

impl Theme {
    /// Named ANSI colors only, which 16-color terminals and SSH sessions show.
    pub(super) fn color() -> Self {
        Self {
            title: Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            selected: Style::new()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
            dim: Style::new().fg(Color::DarkGray),
            ok: Style::new().fg(Color::Green),
            warn: Style::new().fg(Color::Yellow),
            error: Style::new().fg(Color::Red).add_modifier(Modifier::BOLD),
            key: Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        }
    }

    /// Modifiers only, every color left to the terminal (`NO_COLOR`).
    pub(super) fn mono() -> Self {
        Self {
            title: Style::new().add_modifier(Modifier::BOLD),
            selected: Style::new().add_modifier(Modifier::REVERSED),
            dim: Style::new().add_modifier(Modifier::DIM),
            ok: Style::new(),
            warn: Style::new().add_modifier(Modifier::BOLD),
            error: Style::new().add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            key: Style::new().add_modifier(Modifier::BOLD),
        }
    }

    /// [`Theme::mono`] when `NO_COLOR` is set and not empty (no-color.org), else
    /// [`Theme::color`]. Read once, when the TUI starts.
    pub(super) fn detect() -> Self {
        if std::env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty()) {
            Self::mono()
        } else {
            Self::color()
        }
    }

    /// The style of a log level.
    pub(super) fn level(&self, level: Level) -> Style {
        match level {
            Level::ERROR => self.error,
            Level::WARN => self.warn,
            Level::INFO => self.ok,
            _ => self.dim,
        }
    }
}
