//! Motion: what moves on screen, and when the loop must draw a new frame for
//! it. The loop injects the time: [`App::on_frame`] advances the motion clock,
//! and the views read only that clock, never `Instant::now()`. Motion is `on`
//! by default, `reduced` over SSH, and `off` under `NO_COLOR`;
//! `OVERBRAINER_TUI_MOTION` chooses.
//!
//! The spinner and the smoothed bars are plain state. The color effects (a
//! view or an overlay fading in, a new status fading in, the pulse of the
//! followed run) are tachyonfx effects, applied to the drawn buffer as post
//! processing, in 24-bit color only. Each is kept with the clock time it
//! started at, and applied from a fresh copy at its age: drawing twice
//! between two frames gives the same frame.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::time::{Duration, SystemTime};

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;
use tachyonfx::{Effect as Fx, Interpolation, fx};

use super::app::{App, Overlay, View};
use super::format::WORKING;
use super::pipeline::{STAGES, StageState};
use super::theme::{ColorLevel, LookEnv, Theme};
use super::training::RunActivity;
use super::widgets::bar::round_to;

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
/// How long a view takes to fade in.
const VIEW_FADE: Duration = Duration::from_millis(160);
/// How long an overlay takes to fade in.
const OVERLAY_FADE: Duration = Duration::from_millis(140);
/// How long a new status message takes to fade in.
const TOAST_FADE: Duration = Duration::from_millis(200);
/// Half the pulse of the followed run: from bright to dim crimson. The pulse
/// has no frame period of its own: it moves at the spinner's [`SPIN`], a step
/// too small to see in a swing this slow, and a pulse beside a spinner then
/// never wakes the loop at a second pace.
const PULSE_HALF: Duration = Duration::from_millis(1200);

/// What a color effect is for; one of each runs at most.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Purpose {
    /// The view shown fades in.
    View,
    /// The overlay opened fades in; it closes at once.
    Overlay,
    /// A new status message fades in.
    Toast,
}

/// Where the color effects apply on the frame just drawn.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Areas {
    /// The view, between the header and the footer.
    pub(super) body: Rect,
    /// The overlay, if one is open.
    pub(super) overlay: Option<Rect>,
    /// The status message on the footer, if one shows.
    pub(super) toast: Option<Rect>,
    /// The followed run's `●`, if it shows.
    pub(super) pulse: Option<Rect>,
}

/// The colors the effects fade from and to: the painted background, the
/// overlay surface, and the dim end of the pulse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Colors {
    bg: Color,
    surface: Color,
    dim: Color,
}

/// A color effect, and the clock time it started at.
#[derive(Debug, Clone)]
struct Running {
    fx: Fx,
    started: Duration,
}

/// What the screen showed at the last draw: a change starts an effect.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Seen {
    /// The view, once one was drawn.
    view: Option<View>,
    /// Which overlay was open: its kind, or a dialog's title.
    overlay: Option<String>,
    /// The status message and when it was set.
    status: Option<(String, SystemTime)>,
}

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
    /// The cells of each bar on screen, at its last draw: what it shows moves
    /// on screen only by an eighth of one.
    cells: BTreeMap<Bar, u16>,
    /// The colors of the color effects; `None` when they are off (motion not
    /// `on`, or no 24-bit color).
    colors: Option<Colors>,
    /// The color effects running.
    effects: BTreeMap<Purpose, Running>,
    /// The pulse: from bright crimson to dim, played forth and back.
    pulse: Option<Fx>,
    /// What the screen showed at the last draw.
    seen: Seen,
}

impl Motion {
    /// Motion at `level`, its clock at zero.
    pub(super) fn new(level: MotionLevel) -> Self {
        Self {
            level,
            clock: Duration::ZERO,
            bars: BTreeMap::new(),
            cells: BTreeMap::new(),
            colors: None,
            effects: BTreeMap::new(),
            pulse: None,
            seen: Seen::default(),
        }
    }

    /// This motion with the color effects of `theme`: only when motion is
    /// `on` and `theme` is in 24-bit color.
    pub(super) fn colored(mut self, theme: &Theme) -> Self {
        let colors = match (theme.base.bg, theme.surface.bg, theme.accent_dim.fg) {
            (Some(bg), Some(surface), Some(dim)) => Some(Colors { bg, surface, dim }),
            _ => None,
        };
        let on = self.level == MotionLevel::On && theme.level == ColorLevel::TrueColor;
        self.colors = colors.filter(|_| on);
        self.pulse = self
            .colors
            .map(|colors| fx::fade_to_fg(colors.dim, (PULSE_HALF, Interpolation::SineInOut)));
        self
    }

    /// Whether the pulse shows: color effects are on.
    pub(super) fn pulses(&self) -> bool {
        self.pulse.is_some()
    }

    /// Starts the effect of `purpose` now, replacing one running.
    fn start(&mut self, purpose: Purpose) {
        let Some(colors) = self.colors else {
            return;
        };
        let fx = match purpose {
            Purpose::View => {
                fx::fade_from(colors.bg, colors.bg, (VIEW_FADE, Interpolation::QuadOut))
            },
            Purpose::Overlay => fx::fade_from(
                colors.surface,
                colors.surface,
                (OVERLAY_FADE, Interpolation::QuadOut),
            ),
            Purpose::Toast => fx::fade_from_fg(colors.bg, (TOAST_FADE, Interpolation::QuadOut)),
        };
        self.effects.insert(
            purpose,
            Running {
                fx,
                started: self.clock,
            },
        );
    }

    /// Starts the effect of each change between the last draw and `now`: a
    /// new view, an overlay opened (closing one is instant), a new status.
    fn observe(&mut self, now: Seen) {
        if self.seen.view.is_some() && self.seen.view != now.view {
            self.start(Purpose::View);
        }
        match &now.overlay {
            Some(_) if now.overlay != self.seen.overlay => self.start(Purpose::Overlay),
            Some(_) => {},
            None => {
                self.effects.remove(&Purpose::Overlay);
            },
        }
        if now.status.is_some() && now.status != self.seen.status {
            self.start(Purpose::Toast);
        }
        self.seen = now;
    }

    /// Forgets the effects that ended; returns whether any ran.
    fn settle_effects(&mut self) -> bool {
        let ran = !self.effects.is_empty();
        let clock = self.clock;
        self.effects.retain(|_, running| {
            let length = running
                .fx
                .timer()
                .map_or(Duration::ZERO, |timer| timer.duration());
            clock.saturating_sub(running.started) < length
        });
        ran
    }

    /// The pulse's place in its swing: from 0 (bright) up to [`PULSE_HALF`]
    /// (dim) and back, every two halves of the clock.
    fn pulse_age(&self) -> Duration {
        let period = PULSE_HALF.as_millis() * 2;
        let phase = u64::try_from(self.clock.as_millis() % period).unwrap_or(0);
        let half = u64::try_from(PULSE_HALF.as_millis()).unwrap_or(0);
        Duration::from_millis(if phase <= half {
            phase
        } else {
            2 * half - phase
        })
    }

    /// Applies the color effects running, and the pulse, to `buffer`, each on
    /// its area of `areas`: a fresh copy of each effect played to its age.
    pub(super) fn apply(&self, buffer: &mut Buffer, areas: &Areas) {
        for (purpose, running) in &self.effects {
            let area = match purpose {
                Purpose::View => Some(areas.body),
                Purpose::Overlay => areas.overlay,
                Purpose::Toast => areas.toast,
            };
            if let Some(area) = area {
                let mut fx = running.fx.clone();
                fx.process(self.clock.saturating_sub(running.started), buffer, area);
            }
        }
        if let (Some(pulse), Some(area)) = (&self.pulse, areas.pulse) {
            let mut fx = pulse.clone();
            fx.process(self.pulse_age(), buffer, area);
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

    /// What bar `key`, drawn `cells` wide, shows when its true value is
    /// `target`, as [`Motion::shown`] says; its cells are kept, for
    /// [`App::on_frame`] to redraw it only when it moves on screen.
    pub(super) fn bar(&mut self, key: Bar, target: f64, cells: u16) -> f64 {
        self.cells.insert(key, cells);
        self.shown(key, target)
    }

    /// What bar `key` shows when its true value is `target`: that value when
    /// motion is off or the bar is new, never more than it, else the eased
    /// value.
    fn shown(&self, key: Bar, target: f64) -> f64 {
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
    /// shown is forgotten. Returns whether a bar moved on screen: by an
    /// eighth of a cell once it was drawn, by any amount before.
    fn ease(&mut self, targets: &[(Bar, f64, bool)], elapsed: Duration) -> bool {
        let closed = 1.0 - (-elapsed.as_secs_f64() / TAU).exp();
        let mut moved = false;
        for (key, target, snap) in targets {
            let shown = *self.bars.entry(*key).or_insert(*target);
            let next = if *snap || *target <= shown {
                *target
            } else {
                let eased = shown + (*target - shown) * closed;
                if *target - eased < SETTLED {
                    *target
                } else {
                    eased
                }
            };
            moved |= self.apart(*key, shown, next);
            self.bars.insert(*key, next);
        }
        let shown = |key: &Bar| targets.iter().any(|(target, _, _)| target == key);
        self.bars.retain(|key, _| shown(key));
        self.cells.retain(|key, _| shown(key));
        moved
    }

    /// Whether bar `key` draws `one` and `other` apart: in eighths of a cell
    /// once it was drawn, else as soon as they differ.
    fn apart(&self, key: Bar, one: f64, other: f64) -> bool {
        match self.cells.get(&key) {
            Some(cells) => {
                let eighths = u32::from(*cells) * 8;
                let at = |value: f64| round_to(value.clamp(0.0, 1.0) * f64::from(eighths), eighths);
                at(one) != at(other)
            },
            None => (one - other).abs() >= SETTLED,
        }
    }

    /// Whether a bar of `targets` still shows short of its value.
    fn easing(&self, targets: &[(Bar, f64, bool)]) -> bool {
        self.level != MotionLevel::Off
            && targets.iter().any(|(key, target, _)| {
                self.bars
                    .get(key)
                    .is_some_and(|shown| self.apart(*key, *shown, *target))
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
    /// on screen: [`FAST`] while a fade runs or a bar of the view eases,
    /// [`SPIN`] while a spinner or the pulse shows; `None` when nothing moves,
    /// so an idle TUI never wakes up.
    pub(super) fn frame_period(&self) -> Option<Duration> {
        if self.motion.level() == MotionLevel::Off {
            return None;
        }
        if !self.motion.effects.is_empty() || self.motion.easing(&self.bar_targets()) {
            return Some(FAST);
        }
        (!self.work().is_empty() || self.pulse_shown()).then_some(SPIN)
    }

    /// Whether the followed run's `●` pulses on screen: the Training view on a
    /// run a task follows, with color effects on and no overlay over it.
    pub(super) fn pulse_shown(&self) -> bool {
        self.motion.pulses()
            && self.overlay.is_none()
            && self.view == View::Training
            && self.training.selected_activity() == RunActivity::Followed
    }

    /// Starts the color effects of what changed since the last draw: called
    /// by the render, before it draws.
    pub(super) fn observe_motion(&mut self) {
        let overlay = self.overlay.as_ref().map(|overlay| match overlay {
            Overlay::Help => "help".to_string(),
            Overlay::Menu(_) => "menu".to_string(),
            Overlay::Confirm(confirm) => confirm.title.clone(),
        });
        let status = self
            .status
            .as_ref()
            .map(|status| (status.text.clone(), status.at));
        self.motion.observe(Seen {
            view: Some(self.view),
            overlay,
            status,
        });
    }

    /// The true value of every smoothed bar of the view shown, and whether
    /// it lands on it at once: the stages started on the Pipeline view (a
    /// stage that ended lands), the selected run's steps on the Training
    /// view. A bar out of view is not eased: it shows its value once seen.
    fn bar_targets(&self) -> Vec<(Bar, f64, bool)> {
        let mut targets = Vec::new();
        match self.view {
            View::Pipeline => {},
            View::Training => {
                if let Some(ratio) = self.training.selected_ratio() {
                    targets.push((Bar::Step, ratio, false));
                }
                return targets;
            },
            View::Dataset | View::Logs => return targets,
        }
        for (index, stage) in STAGES.iter().enumerate() {
            let row = self.pipeline.row(*stage);
            let ended = match row.state {
                StageState::Running => false,
                StageState::Done | StageState::Stopped => true,
                StageState::Idle | StageState::Pending => continue,
            };
            targets.push((Bar::Stage(index), row.ratio(), ended));
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
        if self.motion.settle_effects() || self.pulse_shown() {
            self.dirty = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use crossterm::event::KeyCode;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::buffer::Cell;

    use super::*;
    use crate::tui::app::Severity;
    use crate::tui::snapshots::{
        app, dataset_app, draw, key, pipeline_running, text, training_app,
    };

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
            (motion.shown(Bar::Step, 0.6) - 0.2).abs() < 1e-9,
            "not eased yet"
        );
        assert!(motion.easing(&step(0.6, false)));
        assert!(motion.ease(&step(0.6, false), Duration::from_millis(250)));
        let one_tau = 0.2 + 0.4 * (1.0 - (-1.0_f64).exp());
        assert!((motion.shown(Bar::Step, 0.6) - one_tau).abs() < 1e-9);
        motion.ease(&step(0.6, false), Duration::from_secs(10));
        assert!(
            (motion.shown(Bar::Step, 0.6) - 0.6).abs() < f64::EPSILON,
            "landed"
        );
        assert!(!motion.easing(&step(0.6, false)));
        motion.ease(&step(0.1, false), Duration::from_millis(16));
        assert!(
            (motion.shown(Bar::Step, 0.1) - 0.1).abs() < f64::EPSILON,
            "a drop lands at once"
        );
        motion.ease(&step(0.9, true), Duration::from_millis(16));
        assert!(
            (motion.shown(Bar::Step, 0.9) - 0.9).abs() < f64::EPSILON,
            "an ended stage lands"
        );
        motion.ease(&[], Duration::from_millis(16));
        assert!(
            (motion.shown(Bar::Step, 0.4) - 0.4).abs() < f64::EPSILON,
            "forgotten: new again"
        );
        let off = Motion::new(MotionLevel::Off);
        assert!((off.shown(Bar::Step, 0.7) - 0.7).abs() < f64::EPSILON);
    }

    /// 80 more answers done: the answers bar has 200/400 to show.
    fn answers_done(app: &mut App) {
        for n in 120..200 {
            app.pipeline.event(&crate::events::Event::ItemDone {
                stage: crate::events::Stage::Answers,
                id: format!("item{n}"),
                usage: None,
            });
        }
    }

    #[test]
    fn frames_come_fast_while_a_bar_eases() -> TestResult {
        let mut app = app();
        app.motion = Motion::new(MotionLevel::On);
        pipeline_running(&mut app);
        app.view = View::Pipeline;
        app.on_frame(Duration::ZERO);
        draw(&mut app, 80, 24)?;
        assert_eq!(
            app.frame_period(),
            Some(SPIN),
            "settled: the spinner's pace"
        );
        answers_done(&mut app);
        assert_eq!(app.frame_period(), Some(FAST), "easing to 200/400");
        app.on_frame(Duration::from_secs(10));
        assert_eq!(app.frame_period(), Some(SPIN));
        Ok(())
    }

    #[test]
    fn a_bar_out_of_view_never_speeds_the_frames_up() {
        let mut app = app();
        app.motion = Motion::new(MotionLevel::On);
        pipeline_running(&mut app);
        assert_eq!(app.view, View::Dataset);
        app.on_frame(Duration::ZERO);
        answers_done(&mut app);
        app.on_frame(Duration::from_millis(1));
        assert_eq!(app.frame_period(), Some(SPIN), "the spinner's pace only");
        app.view = View::Pipeline;
        app.on_frame(Duration::from_millis(1));
        assert_eq!(app.frame_period(), Some(SPIN), "seen at its value");
    }

    #[test]
    fn a_bar_redraws_only_when_it_moves_by_an_eighth_of_a_cell() -> TestResult {
        let mut app = app();
        app.motion = Motion::new(MotionLevel::On);
        pipeline_running(&mut app);
        app.view = View::Pipeline;
        app.on_frame(Duration::ZERO);
        draw(&mut app, 80, 24)?;
        answers_done(&mut app);
        app.dirty = false;
        // 29 cells: an eighth of one is 1/232 of the bar, and 0.2 of it
        // closes by far less in 10 µs.
        app.on_frame(Duration::from_micros(10));
        assert!(!app.dirty, "the same eighth on screen");
        assert_eq!(app.frame_period(), Some(FAST), "still easing");
        app.on_frame(Duration::from_millis(30));
        assert!(app.dirty, "an eighth further");
        Ok(())
    }

    #[test]
    fn a_load_is_shown_after_150_ms() -> TestResult {
        let mut app = app();
        app.motion = Motion::new(MotionLevel::On);
        app.start();
        let shown = |app: &mut App| -> Result<bool, Box<dyn std::error::Error>> {
            Ok(text(&draw(app, 80, 24)?)
                .join("\n")
                .contains("reading data/"))
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
        let rows = text(&draw(&mut still, 80, 24)?).join("\n");
        assert!(rows.contains("… reading data/ "), "one ellipsis: {rows}");
        Ok(())
    }

    /// `app` with motion on, and the color effects of its 24-bit theme.
    fn moving(mut app: App) -> App {
        app.motion = Motion::new(MotionLevel::On).colored(&app.theme);
        app
    }

    /// The 24-bit background every fade starts from.
    const BG: Color = Color::Rgb(0x19, 0x11, 0x14);

    /// The cell at `x`, `y` of `terminal`.
    fn cell(terminal: &Terminal<TestBackend>, x: u16, y: u16) -> Result<Cell, String> {
        terminal
            .backend()
            .buffer()
            .cell((x, y))
            .cloned()
            .ok_or_else(|| format!("no cell at {x}, {y}"))
    }

    #[test]
    fn a_view_fades_in_from_the_background() -> TestResult {
        let mut app = moving(dataset_app());
        draw(&mut app, 80, 24)?;
        app.on_input(&key(KeyCode::Char('2')));
        let terminal = draw(&mut app, 80, 24)?;
        for y in 1..23 {
            for x in 0..80 {
                let cell = cell(&terminal, x, y)?;
                assert_eq!((cell.fg, cell.bg), (BG, BG), "at {x}, {y}");
            }
        }
        assert_eq!(app.frame_period(), Some(FAST));
        app.on_frame(VIEW_FADE);
        let faded = draw(&mut app, 80, 24)?;
        let mut still = dataset_app();
        still.on_input(&key(KeyCode::Char('2')));
        assert_eq!(
            faded.backend().buffer(),
            draw(&mut still, 80, 24)?.backend().buffer()
        );
        assert_eq!(app.frame_period(), None, "idle once the fade ended");
        Ok(())
    }

    #[test]
    fn an_overlay_fades_in_and_closes_at_once() -> TestResult {
        let mut app = moving(app());
        draw(&mut app, 80, 24)?;
        app.on_input(&key(KeyCode::Char('?')));
        let surface = Color::Rgb(0x20, 0x13, 0x18);
        let rows = text(&draw(&mut app, 80, 24)?);
        let y = rows
            .iter()
            .position(|row| row.contains("Everywhere"))
            .ok_or("no help")?;
        let x = rows
            .get(y)
            .ok_or("no row")?
            .chars()
            .position(|c| c == 'E')
            .ok_or("no E")?;
        let (x, y) = (u16::try_from(x)?, u16::try_from(y)?);
        assert_eq!(cell(&draw(&mut app, 80, 24)?, x, y)?.fg, surface);
        app.on_frame(OVERLAY_FADE);
        let title = app.theme.title.fg.ok_or("no title color")?;
        assert_eq!(cell(&draw(&mut app, 80, 24)?, x, y)?.fg, title);
        app.on_input(&key(KeyCode::Char('?')));
        app.on_input(&key(KeyCode::Esc));
        draw(&mut app, 80, 24)?;
        assert!(
            !app.motion.effects.contains_key(&Purpose::Overlay),
            "closed at once"
        );
        Ok(())
    }

    #[test]
    fn a_new_status_fades_in() -> TestResult {
        let mut app = moving(app());
        draw(&mut app, 80, 24)?;
        app.say(Severity::Info, "deletion saved");
        assert_eq!(cell(&draw(&mut app, 80, 24)?, 1, 23)?.fg, BG);
        app.on_frame(TOAST_FADE);
        let ok = app.theme.ok.fg.ok_or("no ok color")?;
        assert_eq!(cell(&draw(&mut app, 80, 24)?, 1, 23)?.fg, ok);
        Ok(())
    }

    #[test]
    fn the_followed_run_pulses_between_two_crimsons() -> TestResult {
        let mut app = moving(training_app()?);
        let bright = app.theme.title.fg.ok_or("no bright")?;
        let dim = app.theme.accent_dim.fg.ok_or("no dim")?;
        let rows = text(&draw(&mut app, 80, 24)?);
        let y = rows
            .iter()
            .position(|row| row.contains('●'))
            .ok_or("no marker")?;
        let y = u16::try_from(y)?;
        assert_eq!(cell(&draw(&mut app, 80, 24)?, 2, y)?.symbol(), "●");
        assert_eq!(cell(&draw(&mut app, 80, 24)?, 2, y)?.fg, bright);
        app.on_frame(PULSE_HALF);
        assert_eq!(cell(&draw(&mut app, 80, 24)?, 2, y)?.fg, dim);
        app.on_frame(PULSE_HALF);
        assert_eq!(cell(&draw(&mut app, 80, 24)?, 2, y)?.fg, bright);
        assert!(app.pulse_shown());
        app.on_input(&key(KeyCode::Char('c')));
        assert!(app.overlay.is_some(), "the cancel dialog");
        assert!(!app.pulse_shown(), "under an overlay");
        app.on_frame(PULSE_HALF);
        let marker = cell(&draw(&mut app, 80, 24)?, 2, y)?;
        assert_eq!(marker.symbol(), "●");
        assert_eq!(
            Some(marker.fg),
            app.theme.dim.fg,
            "dimmed with the view, never pulsed"
        );
        app.overlay = None;
        app.training.tasks.clear();
        assert!(!app.pulse_shown(), "no run followed");
        Ok(())
    }

    /// Symbols and styles of `terminal`, a spinner frame and [`WORKING`] read
    /// alike.
    fn settled(terminal: &Terminal<TestBackend>) -> Vec<(String, ratatui::style::Style)> {
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| {
                let symbol = if SPINNER.contains(&cell.symbol()) {
                    WORKING
                } else {
                    cell.symbol()
                };
                (symbol.to_string(), cell.style())
            })
            .collect()
    }

    /// Long after every change, motion on draws what motion off draws, the
    /// pulse at its bright end.
    #[test]
    fn motion_on_settles_to_what_motion_off_draws() -> TestResult {
        let changes = |app: &mut App| -> TestResult {
            app.view = View::Dataset;
            draw(app, 80, 24)?;
            app.on_input(&key(KeyCode::Char('3')));
            app.say(Severity::Warn, "refused: a stage is running");
            app.overlay = Some(Overlay::Help);
            draw(app, 80, 24)?;
            Ok(())
        };
        let mut on = moving(training_app()?);
        changes(&mut on)?;
        on.on_frame(PULSE_HALF * 20);
        let mut off = training_app()?;
        changes(&mut off)?;
        assert_eq!(
            settled(&draw(&mut on, 80, 24)?),
            settled(&draw(&mut off, 80, 24)?)
        );
        Ok(())
    }

    /// Every fade changes colors only: halfway through, the symbols are those
    /// motion off draws.
    #[test]
    fn a_fade_never_changes_a_symbol() -> TestResult {
        let changes = |app: &mut App| -> TestResult {
            draw(app, 80, 24)?;
            app.on_input(&key(KeyCode::Char('2')));
            app.say(Severity::Info, "deletion saved");
            app.overlay = Some(Overlay::Help);
            Ok(())
        };
        let mut on = moving(dataset_app());
        changes(&mut on)?;
        let mut off = dataset_app();
        changes(&mut off)?;
        let still = text(&draw(&mut off, 80, 24)?);
        assert_eq!(text(&draw(&mut on, 80, 24)?), still, "as they start");
        on.on_frame(OVERLAY_FADE / 2);
        assert!(!on.motion.effects.is_empty(), "halfway");
        assert_eq!(text(&draw(&mut on, 80, 24)?), still, "halfway");
        Ok(())
    }

    #[test]
    fn nothing_wakes_the_loop_on_a_run_nothing_follows() -> TestResult {
        let mut app = moving(training_app()?);
        draw(&mut app, 80, 24)?;
        assert_eq!(app.frame_period(), Some(SPIN), "the followed run pulses");
        app.training.tasks.clear();
        assert_eq!(app.view, View::Training);
        assert!(app.motion.pulses());
        assert_eq!(app.frame_period(), None, "no run followed: nothing moves");
        Ok(())
    }

    #[test]
    fn color_effects_need_motion_on_and_24_bit_color() {
        let truecolor = Theme::new(ColorLevel::TrueColor);
        assert!(Motion::new(MotionLevel::On).colored(&truecolor).pulses());
        assert!(
            !Motion::new(MotionLevel::Reduced)
                .colored(&truecolor)
                .pulses()
        );
        let indexed = Theme::new(ColorLevel::Indexed);
        assert!(!Motion::new(MotionLevel::On).colored(&indexed).pulses());
        assert!(
            !Motion::new(MotionLevel::On)
                .colored(&Theme::mono())
                .pulses()
        );
    }
}
