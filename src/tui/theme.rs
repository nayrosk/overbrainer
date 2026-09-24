//! The crimson palette and the styles the views use, at the color level the
//! terminal shows: 24-bit, 256 colors, the 16 named colors, or modifiers only
//! under `NO_COLOR`. Crimson means identity and focus, never danger: errors are
//! orange-red, bold and marked `✗`.

use std::ffi::{OsStr, OsString};

use ratatui::style::{Color, Modifier, Style};
use tracing::Level;

/// How many colors the terminal shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ColorLevel {
    /// No color at all, modifiers only (`NO_COLOR`).
    Mono,
    /// The 16 named colors, which follow the terminal's own theme; the
    /// terminal's background shows.
    Named,
    /// The 256-color palette; the background is painted.
    Indexed,
    /// 24-bit colors; the background is painted.
    TrueColor,
}

/// The environment variables the look depends on, read once when the TUI
/// starts and passed in, so tests never set a variable.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct LookEnv {
    /// `NO_COLOR`: monochrome when set and not empty (no-color.org).
    pub(super) no_color: Option<OsString>,
    /// `OVERBRAINER_TUI_COLOR`: `truecolor`, `256` or `16` forces a level.
    pub(super) color: Option<OsString>,
    /// `COLORTERM`: `truecolor` or `24bit` means 24-bit colors.
    pub(super) colorterm: Option<OsString>,
    /// `TERM`: a name containing `256color` means 256 colors.
    pub(super) term: Option<OsString>,
    /// `OVERBRAINER_TUI_MOTION`: `on`, `reduced` or `off`.
    pub(super) motion: Option<OsString>,
    /// Whether `SSH_CONNECTION` or `SSH_TTY` is set: motion is reduced.
    pub(super) ssh: bool,
}

impl LookEnv {
    /// The variables of this process.
    pub(super) fn from_process() -> Self {
        Self {
            no_color: std::env::var_os("NO_COLOR"),
            color: std::env::var_os("OVERBRAINER_TUI_COLOR"),
            colorterm: std::env::var_os("COLORTERM"),
            term: std::env::var_os("TERM"),
            motion: std::env::var_os("OVERBRAINER_TUI_MOTION"),
            ssh: std::env::var_os("SSH_CONNECTION").is_some()
                || std::env::var_os("SSH_TTY").is_some(),
        }
    }

    /// What the TUI logs about these variables when it starts: each value it
    /// does not know, which it ignores.
    pub(super) fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if let Some(value) = filled(self.color.as_deref())
            && ColorLevel::forced(value).is_none()
        {
            warnings.push(format!(
                "OVERBRAINER_TUI_COLOR={} is not truecolor, 256 or 16: ignored",
                value.display()
            ));
        }
        if let Some(value) = filled(self.motion.as_deref())
            && super::motion::MotionLevel::chosen(value).is_none()
        {
            warnings.push(format!(
                "OVERBRAINER_TUI_MOTION={} is not on, reduced or off: ignored",
                value.display()
            ));
        }
        warnings
    }
}

/// `value` when it is set and not empty.
fn filled(value: Option<&OsStr>) -> Option<&OsStr> {
    value.filter(|value| !value.is_empty())
}

impl ColorLevel {
    /// The level for `env`: [`ColorLevel::Mono`] under `NO_COLOR`, else the
    /// level `OVERBRAINER_TUI_COLOR` forces, else 24-bit when `COLORTERM` says
    /// so, 256 colors when `TERM` does, and the 16 named colors otherwise.
    pub(super) fn detect(env: &LookEnv) -> Self {
        if filled(env.no_color.as_deref()).is_some() {
            return Self::Mono;
        }
        if let Some(level) = filled(env.color.as_deref()).and_then(Self::forced) {
            return level;
        }
        let colorterm = env.colorterm.as_deref().and_then(OsStr::to_str);
        if matches!(colorterm, Some("truecolor" | "24bit")) {
            return Self::TrueColor;
        }
        let term = env.term.as_deref().and_then(OsStr::to_str);
        if term.is_some_and(|term| term.contains("256color")) {
            return Self::Indexed;
        }
        Self::Named
    }

    /// The level an `OVERBRAINER_TUI_COLOR` of `value` forces, if it is one.
    fn forced(value: &OsStr) -> Option<Self> {
        match value.to_str()? {
            "truecolor" => Some(Self::TrueColor),
            "256" => Some(Self::Indexed),
            "16" => Some(Self::Named),
            _ => None,
        }
    }
}

/// One color of the palette at each level: 24-bit, its 256-color index, and
/// its named color.
#[derive(Debug, Clone, Copy)]
struct Tone {
    rgb: u32,
    index: u8,
    named: Color,
}

impl Tone {
    /// This tone at `level`; no color at [`ColorLevel::Mono`].
    fn at(self, level: ColorLevel) -> Option<Color> {
        match level {
            ColorLevel::Mono => None,
            ColorLevel::Named => Some(self.named),
            ColorLevel::Indexed => Some(Color::Indexed(self.index)),
            ColorLevel::TrueColor => Some(Color::from_u32(self.rgb)),
        }
    }

    /// This tone at `level` where a painted background is wanted: none at the
    /// 16-color level, which leaves the terminal's background.
    fn painted(self, level: ColorLevel) -> Option<Color> {
        match level {
            ColorLevel::Mono | ColorLevel::Named => None,
            _ => self.at(level),
        }
    }

    /// A style with this tone as its foreground at `level`.
    fn fg(self, level: ColorLevel) -> Style {
        let style = Style::new();
        match self.at(level) {
            Some(color) => style.fg(color),
            None => style,
        }
    }
}

const fn tone(rgb: u32, index: u8, named: Color) -> Tone {
    Tone { rgb, index, named }
}

const BG: Tone = tone(0x0019_1114, 233, Color::Reset);
const SURFACE: Tone = tone(0x0020_1318, 234, Color::Reset);
const SELECTION: Tone = tone(0x004A_1A30, 236, Color::Magenta);
const BORDER: Tone = tone(0x006D_2545, 53, Color::DarkGray);
const BORDER_FOCUS: Tone = tone(0x00B0_436E, 131, Color::Magenta);
const ACCENT: Tone = tone(0x00E9_3D82, 168, Color::Magenta);
const ACCENT_HI: Tone = tone(0x00FF_92AD, 211, Color::LightMagenta);
const ACCENT_DIM: Tone = tone(0x00B0_436E, 131, Color::Magenta);
const TEXT: Tone = tone(0x00E8_D8DF, 254, Color::Reset);
const TEXT_HI: Tone = tone(0x00FD_D3E8, 224, Color::White);
const TEXT_MUTED: Tone = tone(0x00A0_808E, 138, Color::DarkGray);
const OK: Tone = tone(0x00A6_E3A1, 151, Color::Green);
const WARN: Tone = tone(0x00F9_E2AF, 223, Color::Yellow);
const ERROR: Tone = tone(0x00FF_7A5C, 209, Color::LightRed);
const INFO: Tone = tone(0x0089_DCEB, 116, Color::Cyan);
const LR: Tone = tone(0x00B4_BEFE, 147, Color::LightBlue);
const GRAD_NORM: Tone = tone(0x0094_E2D5, 115, Color::Cyan);

/// The styles of every view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Theme {
    /// The color level the styles are for.
    pub(super) level: ColorLevel,
    /// Painted over the whole frame first: the background and the text color
    /// at the 24-bit and 256 levels, nothing otherwise.
    pub(super) base: Style,
    /// Painted over an overlay's rectangle, as `base` is.
    pub(super) surface: Style,
    /// The border of a pane.
    pub(super) border: Style,
    /// The border of the pane the keys act on, and of overlays.
    pub(super) border_focus: Style,
    /// Crimson: identity, focus, running work.
    pub(super) accent: Style,
    /// The dim end of the followed run's pulse.
    pub(super) accent_dim: Style,
    /// Titles and the header: bright crimson, bold.
    pub(super) title: Style,
    /// The selected tab: bright crimson, bold and underlined (reversed in
    /// monochrome).
    pub(super) tab: Style,
    /// The selected row or tree node.
    pub(super) selected: Style,
    /// Dialog text.
    pub(super) text_hi: Style,
    /// Secondary text.
    pub(super) dim: Style,
    /// Success.
    pub(super) ok: Style,
    /// Warnings.
    pub(super) warn: Style,
    /// Errors: orange-red and bold, never crimson.
    pub(super) error: Style,
    /// Plain information.
    pub(super) info: Style,
    /// Keys in hints, the help overlay and dialogs.
    pub(super) key: Style,
    /// A key refused while the data is locked: muted and crossed out.
    pub(super) locked: Style,
    /// The filled part of a progress bar.
    pub(super) gauge: Style,
    /// The training loss line.
    pub(super) loss: Style,
    /// The eval loss points.
    pub(super) eval_loss: Style,
    /// The learning-rate sparkline.
    pub(super) lr: Style,
    /// The gradient-norm sparkline.
    pub(super) grad_norm: Style,
}

impl Theme {
    /// The crimson palette at `level`; [`Theme::mono`] at
    /// [`ColorLevel::Mono`].
    pub(super) fn new(level: ColorLevel) -> Self {
        if level == ColorLevel::Mono {
            return Self::mono();
        }
        let painted = |tone: Tone| {
            let style = TEXT.fg(level);
            match tone.painted(level) {
                Some(color) => style.bg(color),
                None => style,
            }
        };
        let bold = Modifier::BOLD;
        let selected = if level == ColorLevel::Named {
            Style::new().fg(Color::Black).bg(Color::Magenta)
        } else {
            let style = TEXT_HI.fg(level);
            match SELECTION.at(level) {
                Some(color) => style.bg(color),
                None => style,
            }
        };
        Self {
            level,
            base: painted(BG),
            surface: painted(SURFACE),
            border: BORDER.fg(level),
            border_focus: BORDER_FOCUS.fg(level),
            accent: ACCENT.fg(level),
            accent_dim: ACCENT_DIM.fg(level),
            title: ACCENT_HI.fg(level).add_modifier(bold),
            tab: ACCENT_HI
                .fg(level)
                .add_modifier(bold | Modifier::UNDERLINED),
            selected: selected.add_modifier(bold),
            text_hi: TEXT_HI.fg(level),
            dim: TEXT_MUTED.fg(level),
            ok: OK.fg(level),
            warn: WARN.fg(level),
            error: ERROR.fg(level).add_modifier(bold),
            info: INFO.fg(level),
            key: ACCENT_HI.fg(level).add_modifier(bold),
            locked: TEXT_MUTED.fg(level).add_modifier(Modifier::CROSSED_OUT),
            gauge: ACCENT.fg(level),
            loss: ACCENT_HI.fg(level),
            eval_loss: WARN.fg(level),
            lr: LR.fg(level),
            grad_norm: GRAD_NORM.fg(level),
        }
    }

    /// Modifiers only, every color left to the terminal (`NO_COLOR`).
    pub(super) fn mono() -> Self {
        let plain = Style::new();
        let bold = plain.add_modifier(Modifier::BOLD);
        Self {
            level: ColorLevel::Mono,
            base: plain,
            surface: plain,
            border: plain,
            border_focus: bold,
            accent: plain,
            accent_dim: plain,
            title: bold,
            tab: plain.add_modifier(Modifier::REVERSED),
            selected: plain.add_modifier(Modifier::REVERSED),
            text_hi: plain,
            dim: plain.add_modifier(Modifier::DIM),
            ok: plain,
            warn: bold,
            error: plain.add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            info: plain,
            key: bold,
            locked: plain.add_modifier(Modifier::DIM | Modifier::CROSSED_OUT),
            gauge: bold,
            loss: plain,
            eval_loss: bold,
            lr: plain,
            grad_norm: plain,
        }
    }

    /// The style of a log level.
    pub(super) fn level(&self, level: Level) -> Style {
        match level {
            Level::ERROR => self.error,
            Level::WARN => self.warn,
            Level::INFO => self.info,
            _ => self.dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use ratatui::buffer::Buffer;
    use ratatui::layout::Rect;
    use ratatui::text::{Line, Span};
    use ratatui::widgets::Widget;

    use super::*;

    fn env(pairs: &[(&str, &str)]) -> LookEnv {
        let mut env = LookEnv::default();
        for (name, value) in pairs {
            let value = Some(OsString::from(value));
            match *name {
                "NO_COLOR" => env.no_color = value,
                "OVERBRAINER_TUI_COLOR" => env.color = value,
                "COLORTERM" => env.colorterm = value,
                "TERM" => env.term = value,
                _ => {},
            }
        }
        env
    }

    #[test]
    fn no_color_wins_then_the_override_then_the_terminal() {
        let detect = |pairs: &[(&str, &str)]| ColorLevel::detect(&env(pairs));
        assert_eq!(detect(&[]), ColorLevel::Named);
        assert_eq!(detect(&[("NO_COLOR", "1")]), ColorLevel::Mono);
        assert_eq!(detect(&[("NO_COLOR", "0")]), ColorLevel::Mono);
        assert_eq!(detect(&[("NO_COLOR", "")]), ColorLevel::Named);
        assert_eq!(
            detect(&[("NO_COLOR", "1"), ("OVERBRAINER_TUI_COLOR", "truecolor")]),
            ColorLevel::Mono
        );
        assert_eq!(detect(&[("COLORTERM", "truecolor")]), ColorLevel::TrueColor);
        assert_eq!(detect(&[("COLORTERM", "24bit")]), ColorLevel::TrueColor);
        assert_eq!(detect(&[("TERM", "tmux-256color")]), ColorLevel::Indexed);
        assert_eq!(detect(&[("TERM", "xterm")]), ColorLevel::Named);
        let forced = |value| {
            detect(&[
                ("COLORTERM", "truecolor"),
                ("TERM", "xterm-256color"),
                ("OVERBRAINER_TUI_COLOR", value),
            ])
        };
        assert_eq!(forced("16"), ColorLevel::Named);
        assert_eq!(forced("256"), ColorLevel::Indexed);
        assert_eq!(forced("truecolor"), ColorLevel::TrueColor);
        assert_eq!(forced("purple"), ColorLevel::TrueColor, "unknown: ignored");
    }

    #[test]
    fn an_unknown_override_is_warned_about() {
        assert_eq!(
            env(&[("OVERBRAINER_TUI_COLOR", "256")]).warnings(),
            Vec::<String>::new()
        );
        assert_eq!(
            env(&[("OVERBRAINER_TUI_COLOR", "")]).warnings(),
            Vec::<String>::new()
        );
        assert_eq!(
            env(&[("OVERBRAINER_TUI_COLOR", "purple")]).warnings(),
            ["OVERBRAINER_TUI_COLOR=purple is not truecolor, 256 or 16: ignored"]
        );
        let motion = LookEnv {
            motion: Some(OsString::from("fast")),
            ..LookEnv::default()
        };
        assert_eq!(
            motion.warnings(),
            ["OVERBRAINER_TUI_MOTION=fast is not on, reduced or off: ignored"]
        );
    }

    #[test]
    fn each_level_has_its_own_colors() {
        let accent = |level| Theme::new(level).accent.fg;
        assert_eq!(
            accent(ColorLevel::TrueColor),
            Some(Color::Rgb(0xE9, 0x3D, 0x82))
        );
        assert_eq!(accent(ColorLevel::Indexed), Some(Color::Indexed(168)));
        assert_eq!(accent(ColorLevel::Named), Some(Color::Magenta));
        assert_eq!(accent(ColorLevel::Mono), None);
        assert_eq!(Theme::new(ColorLevel::Mono), Theme::mono());
    }

    #[test]
    fn the_background_is_painted_on_truecolor_and_256_only() {
        let bg = |level| Theme::new(level).base.bg;
        assert_eq!(
            bg(ColorLevel::TrueColor),
            Some(Color::Rgb(0x19, 0x11, 0x14))
        );
        assert_eq!(bg(ColorLevel::Indexed), Some(Color::Indexed(233)));
        assert_eq!(bg(ColorLevel::Named), None);
        assert_eq!(bg(ColorLevel::Mono), None);
        let surface = Theme::new(ColorLevel::TrueColor).surface.bg;
        assert_eq!(surface, Some(Color::Rgb(0x20, 0x13, 0x18)));
    }

    /// Styles reach the cells: an error is orange-red and bold, never crimson;
    /// a locked key is crossed out; the 16-color selection is black on magenta.
    #[test]
    fn a_buffer_drawn_with_the_theme_carries_its_styles() {
        let theme = Theme::new(ColorLevel::TrueColor);
        let area = Rect::new(0, 0, 12, 1);
        let mut buffer = Buffer::empty(area);
        Line::from(vec![
            Span::styled("✗ no", theme.error),
            Span::raw(" "),
            Span::styled("e edit", theme.locked),
        ])
        .render(area, &mut buffer);
        let error = &buffer[(0, 0)];
        assert_eq!(error.fg, Color::Rgb(0xFF, 0x7A, 0x5C));
        assert!(error.modifier.contains(Modifier::BOLD));
        let locked = &buffer[(5, 0)];
        assert_eq!(locked.symbol(), "e");
        assert_eq!(locked.fg, Color::Rgb(0xA0, 0x80, 0x8E));
        assert!(locked.modifier.contains(Modifier::CROSSED_OUT));
        let named = Theme::new(ColorLevel::Named).selected;
        assert_eq!(
            (named.fg, named.bg),
            (Some(Color::Black), Some(Color::Magenta))
        );
    }
}
