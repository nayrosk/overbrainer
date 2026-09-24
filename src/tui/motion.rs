//! Motion: what moves on screen, and when the loop must draw a new frame for
//! it. The loop injects the time: [`App::on_frame`] advances the motion clock,
//! and the views read only that clock, never `Instant::now()`. Motion is `on`
//! by default, `reduced` over SSH, and `off` under `NO_COLOR`;
//! `OVERBRAINER_TUI_MOTION` chooses.

use std::ffi::OsStr;
use std::time::Duration;

use super::app::App;
use super::format::WORKING;
use super::theme::{ColorLevel, LookEnv};

/// The frames of the spinner of running work.
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
/// How long each frame of the spinner shows.
pub(super) const SPIN: Duration = Duration::from_millis(80);
/// How long a load runs before the Dataset view says it reads the files.
pub(super) const LOADING_AFTER: Duration = Duration::from_millis(150);

/// How much moves on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MotionLevel {
    /// Nothing moves: a running item shows [`WORKING`], bars their true value.
    Off,
    /// Spinners and smoothed bars, no color effect (the default over SSH).
    Reduced,
    /// Spinners, smoothed bars and, in 24-bit color, fades and the pulse.
    On,
}

impl MotionLevel {
    /// The level for `env` at the color level `color`: off in monochrome
    /// (`NO_COLOR`), else the level `OVERBRAINER_TUI_MOTION` chooses, else
    /// reduced over SSH and on otherwise.
    pub(super) fn detect(env: &LookEnv, color: ColorLevel) -> Self {
        if color == ColorLevel::Mono {
            return Self::Off;
        }
        if let Some(level) = env.motion.as_deref().and_then(Self::chosen) {
            return level;
        }
        if env.ssh { Self::Reduced } else { Self::On }
    }

    /// The level an `OVERBRAINER_TUI_MOTION` of `value` chooses, if it is one.
    pub(super) fn chosen(value: &OsStr) -> Option<Self> {
        match value.to_str()? {
            "on" => Some(Self::On),
            "reduced" => Some(Self::Reduced),
            "off" => Some(Self::Off),
            _ => None,
        }
    }
}

/// What moves on screen, and the clock it moves by.
#[derive(Debug, Clone)]
pub(super) struct Motion {
    /// How much moves.
    level: MotionLevel,
    /// Time since the TUI started, as the loop's frames told it.
    clock: Duration,
}

impl Motion {
    /// Motion at `level`, its clock at zero.
    pub(super) fn new(level: MotionLevel) -> Self {
        Self {
            level,
            clock: Duration::ZERO,
        }
    }

    /// How much moves.
    pub(super) fn level(&self) -> MotionLevel {
        self.level
    }

    /// The motion clock.
    pub(super) fn clock(&self) -> Duration {
        self.clock
    }

    /// The spinner's frame at the clock; [`WORKING`] when motion is off.
    pub(super) fn spinner(&self) -> &'static str {
        if self.level == MotionLevel::Off {
            return WORKING;
        }
        let frame = self.clock.as_millis() / SPIN.as_millis() % 10;
        SPINNER[usize::try_from(frame).unwrap_or(0)]
    }

    /// Whether the load started at `started` on the clock (none known: long
    /// ago) has run long enough to be shown: at once when motion is off.
    pub(super) fn shows_load(&self, started: Option<Duration>) -> bool {
        self.level == MotionLevel::Off
            || started.is_none_or(|at| self.clock.saturating_sub(at) >= LOADING_AFTER)
    }
}

impl App {
    /// How long until the loop must draw the next frame, while something moves
    /// on screen; `None` when nothing does, so an idle TUI never wakes up.
    pub(super) fn frame_period(&self) -> Option<Duration> {
        if self.motion.level() == MotionLevel::Off {
            return None;
        }
        (!self.work().is_empty()).then_some(SPIN)
    }

    /// The loop's frame came `elapsed` after the last one: advances the motion
    /// clock, and marks the app to be drawn when something visible changed.
    pub(super) fn on_frame(&mut self, elapsed: Duration) {
        let spinner = self.motion.spinner();
        let loading = self.motion.shows_load(self.load_at);
        self.motion.clock = self.motion.clock.saturating_add(elapsed);
        let working = !self.work().is_empty();
        if working && self.motion.spinner() != spinner {
            self.dirty = true;
        }
        if self.load.is_some() && self.motion.shows_load(self.load_at) != loading {
            self.dirty = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use super::*;
    use crate::tui::snapshots::{app, draw, pipeline_running, text};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn env(motion: Option<&str>, ssh: bool) -> LookEnv {
        LookEnv {
            motion: motion.map(OsString::from),
            ssh,
            ..LookEnv::default()
        }
    }

    #[test]
    fn motion_is_on_reduced_over_ssh_and_off_without_color() {
        let detect = |env: &LookEnv, color| MotionLevel::detect(env, color);
        let color = ColorLevel::TrueColor;
        assert_eq!(detect(&env(None, false), color), MotionLevel::On);
        assert_eq!(detect(&env(None, true), color), MotionLevel::Reduced);
        assert_eq!(detect(&env(Some("on"), true), color), MotionLevel::On);
        assert_eq!(detect(&env(Some("off"), false), color), MotionLevel::Off);
        assert_eq!(
            detect(&env(Some("reduced"), false), color),
            MotionLevel::Reduced
        );
        assert_eq!(
            detect(&env(Some("fast"), false), color),
            MotionLevel::On,
            "ignored"
        );
        assert_eq!(
            detect(&env(Some("on"), false), ColorLevel::Mono),
            MotionLevel::Off,
            "NO_COLOR wins"
        );
        assert_eq!(
            detect(&env(None, false), ColorLevel::Named),
            MotionLevel::On
        );
    }

    #[test]
    fn the_spinner_turns_every_80_ms() {
        let mut motion = Motion::new(MotionLevel::On);
        assert_eq!(motion.spinner(), "⠋");
        motion.clock = Duration::from_millis(79);
        assert_eq!(motion.spinner(), "⠋");
        motion.clock = Duration::from_millis(80);
        assert_eq!(motion.spinner(), "⠙");
        motion.clock = Duration::from_millis(80 * 13);
        assert_eq!(motion.spinner(), "⠸", "frame 3 of the second turn");
        motion.level = MotionLevel::Off;
        assert_eq!(motion.spinner(), WORKING);
    }

    #[test]
    fn nothing_wakes_the_loop_when_nothing_moves() {
        let mut app = app();
        app.motion = Motion::new(MotionLevel::On);
        assert_eq!(app.frame_period(), None, "idle");
        pipeline_running(&mut app);
        assert_eq!(app.frame_period(), Some(SPIN), "a spinner shows");
        app.motion = Motion::new(MotionLevel::Off);
        assert_eq!(app.frame_period(), None, "motion off");
    }

    #[test]
    fn a_frame_redraws_only_when_the_spinner_moves() {
        let mut app = app();
        app.motion = Motion::new(MotionLevel::Reduced);
        pipeline_running(&mut app);
        app.dirty = false;
        app.on_frame(Duration::from_millis(40));
        assert!(!app.dirty, "same spinner frame");
        app.on_frame(Duration::from_millis(40));
        assert!(app.dirty, "next spinner frame");
        assert_eq!(app.motion.clock(), Duration::from_millis(80));
    }

    #[test]
    fn a_load_is_shown_after_150_ms() -> TestResult {
        let mut app = app();
        app.motion = Motion::new(MotionLevel::On);
        app.start();
        let shown = |app: &mut App| -> Result<bool, Box<dyn std::error::Error>> {
            Ok(text(&draw(app, 80, 24)?)
                .join("\n")
                .contains("reading data/…"))
        };
        assert!(!shown(&mut app)?);
        app.on_frame(Duration::from_millis(80));
        assert!(!shown(&mut app)?);
        app.dirty = false;
        app.on_frame(Duration::from_millis(80));
        assert!(app.dirty);
        assert!(shown(&mut app)?);
        let mut still = self::app();
        still.start();
        assert!(shown(&mut still)?, "at once when motion is off");
        Ok(())
    }
}
