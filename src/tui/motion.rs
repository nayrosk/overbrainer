//! Motion: what moves on screen, and when the loop must draw a new frame for
//! it. The loop injects the time: [`App::on_frame`] advances the motion clock,
//! and the views read only that clock, never `Instant::now()`. Motion is `on`
//! by default, `reduced` over SSH, and `off` under `NO_COLOR`;
//! `OVERBRAINER_TUI_MOTION` chooses.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::time::Duration;

use super::app::App;
use super::format::WORKING;
use super::pipeline::{STAGES, StageState};
use super::theme::{ColorLevel, LookEnv};

/// The frames of the spinner of running work.
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
/// How long each frame of the spinner shows.
pub(super) const SPIN: Duration = Duration::from_millis(80);
/// How long a load runs before the Dataset view says it reads the files.
pub(super) const LOADING_AFTER: Duration = Duration::from_millis(150);
/// The time between frames while something eases.
pub(super) const FAST: Duration = Duration::from_millis(33);
/// How fast a bar closes on its target: a time constant, in seconds.
const TAU: f64 = 0.25;
/// How close a bar gets to its target before it lands on it.
const SETTLED: f64 = 0.001;

/// A progress bar smoothed on screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Bar {
    /// The bar of a pipeline stage, by its position in [`STAGES`].
    Stage(usize),
    /// The step bar of the selected training run.
    Step,
}

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
    /// What each bar on screen shows, easing toward its true value.
    bars: BTreeMap<Bar, f64>,
}

impl Motion {
    /// Motion at `level`, its clock at zero.
    pub(super) fn new(level: MotionLevel) -> Self {
        Self {
            level,
            clock: Duration::ZERO,
            bars: BTreeMap::new(),
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

    /// What bar `key` shows when its true value is `target`: that value when
    /// motion is off or the bar is new, never more than it, else the eased
    /// value.
    pub(super) fn bar(&self, key: Bar, target: f64) -> f64 {
        if self.level == MotionLevel::Off {
            return target;
        }
        self.bars
            .get(&key)
            .map_or(target, |shown| shown.min(target))
    }

    /// Eases each bar of `targets` (a bar, its true value, and whether it
    /// lands on it at once: its stage ended) over `elapsed`. A new bar starts
    /// at its value, a bar whose value dropped lands on it; a bar no longer
    /// shown is forgotten. Returns whether a bar moved.
    fn ease(&mut self, targets: &[(Bar, f64, bool)], elapsed: Duration) -> bool {
        let closed = 1.0 - (-elapsed.as_secs_f64() / TAU).exp();
        let mut moved = false;
        for (key, target, snap) in targets {
            let shown = self.bars.entry(*key).or_insert(*target);
            let next = if *snap || *target <= *shown {
                *target
            } else {
                let eased = *shown + (*target - *shown) * closed;
                if *target - eased < SETTLED {
                    *target
                } else {
                    eased
                }
            };
            moved |= (next - *shown).abs() > f64::EPSILON;
            *shown = next;
        }
        self.bars
            .retain(|key, _| targets.iter().any(|(target, _, _)| target == key));
        moved
    }

    /// Whether a bar of `targets` is still easing toward its value.
    fn easing(&self, targets: &[(Bar, f64, bool)]) -> bool {
        self.level != MotionLevel::Off
            && targets.iter().any(|(key, target, _)| {
                self.bars
                    .get(key)
                    .is_some_and(|shown| target - shown >= SETTLED)
            })
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
    /// on screen: [`FAST`] while a bar eases, [`SPIN`] while a spinner shows;
    /// `None` when nothing moves, so an idle TUI never wakes up.
    pub(super) fn frame_period(&self) -> Option<Duration> {
        if self.motion.level() == MotionLevel::Off {
            return None;
        }
        if self.motion.easing(&self.bar_targets()) {
            return Some(FAST);
        }
        (!self.work().is_empty()).then_some(SPIN)
    }

    /// The true value of every smoothed bar, and whether it lands on it at
    /// once: the stages started (a stage that ended lands), and the selected
    /// run's steps.
    fn bar_targets(&self) -> Vec<(Bar, f64, bool)> {
        let mut targets = Vec::new();
        for (index, stage) in STAGES.iter().enumerate() {
            let row = self.pipeline.row(*stage);
            let ended = match row.state {
                StageState::Running => false,
                StageState::Done | StageState::Stopped => true,
                StageState::Idle | StageState::Pending => continue,
            };
            targets.push((Bar::Stage(index), row.ratio(), ended));
        }
        if let Some(ratio) = self.training.selected_ratio() {
            targets.push((Bar::Step, ratio, false));
        }
        targets
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
        let targets = self.bar_targets();
        if self.motion.ease(&targets, elapsed) {
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
    fn a_bar_eases_toward_its_value_and_lands_on_it() {
        let mut motion = Motion::new(MotionLevel::Reduced);
        let step = |ratio, snap| [(Bar::Step, ratio, snap)];
        assert!(
            !motion.ease(&step(0.2, false), Duration::ZERO),
            "a new bar starts at its value"
        );
        assert!(
            (motion.bar(Bar::Step, 0.6) - 0.2).abs() < 1e-9,
            "not eased yet"
        );
        assert!(motion.easing(&step(0.6, false)));
        assert!(motion.ease(&step(0.6, false), Duration::from_millis(250)));
        let one_tau = 0.2 + 0.4 * (1.0 - (-1.0_f64).exp());
        assert!((motion.bar(Bar::Step, 0.6) - one_tau).abs() < 1e-9);
        motion.ease(&step(0.6, false), Duration::from_secs(10));
        assert!(
            (motion.bar(Bar::Step, 0.6) - 0.6).abs() < f64::EPSILON,
            "landed"
        );
        assert!(!motion.easing(&step(0.6, false)));
        motion.ease(&step(0.1, false), Duration::from_millis(16));
        assert!(
            (motion.bar(Bar::Step, 0.1) - 0.1).abs() < f64::EPSILON,
            "a drop lands at once"
        );
        motion.ease(&step(0.9, true), Duration::from_millis(16));
        assert!(
            (motion.bar(Bar::Step, 0.9) - 0.9).abs() < f64::EPSILON,
            "an ended stage lands"
        );
        motion.ease(&[], Duration::from_millis(16));
        assert!(
            (motion.bar(Bar::Step, 0.4) - 0.4).abs() < f64::EPSILON,
            "forgotten: new again"
        );
        let off = Motion::new(MotionLevel::Off);
        assert!((off.bar(Bar::Step, 0.7) - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn frames_come_fast_while_a_bar_eases() {
        let mut app = app();
        app.motion = Motion::new(MotionLevel::On);
        pipeline_running(&mut app);
        app.on_frame(Duration::ZERO);
        assert_eq!(
            app.frame_period(),
            Some(SPIN),
            "settled: the spinner's pace"
        );
        for n in 120..200 {
            app.pipeline.event(&crate::events::Event::ItemDone {
                stage: crate::events::Stage::Answers,
                id: format!("item{n}"),
                usage: None,
            });
        }
        assert_eq!(app.frame_period(), Some(FAST), "easing to 200/400");
        app.on_frame(Duration::from_secs(10));
        assert_eq!(app.frame_period(), Some(SPIN));
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
