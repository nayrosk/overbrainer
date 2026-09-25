//! The event loop: one task selecting over terminal input, background tasks and
//! their messages, process signals, a clock tick and, while something moves on
//! screen, the frames of motion, drawing the app when it changed. It owns the
//! terminal and runs the effects the app asks for.

use std::any::Any;
use std::io::{self, Write};
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::process::{ExitStatus, Stdio};
use std::time::{Duration, SystemTime};

use anyhow::Context;
use crossterm::event::Event as TermEvent;
use ratatui::backend::Backend;
use ratatui::{DefaultTerminal, Terminal};
use tokio::process::Child;
use tokio::signal::unix::{Signal, SignalKind, signal};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::time::{Instant, Interval, MissedTickBehavior};
use tracing::Level;

use super::app::{App, Effect, Exit};
use super::tasks::{Done, Msg, Task, TaskId, Tasks, TrainJob};
use super::terminal::Screen;
use super::ui;

/// Time between two ticks of the app's clock.
const TICK: Duration = Duration::from_millis(250);
/// Shortest time between two draws.
const FRAME: Duration = Duration::from_millis(33);
/// How long an editor asked to end with SIGTERM gets before SIGKILL.
const EDITOR_GRACE: Duration = Duration::from_secs(2);

/// What the real terminal adds to the loop: process signals, and the screen the
/// editor borrows.
pub(super) struct Real {
    signals: Signals,
    screen: Screen,
}

/// Runs the TUI on the real terminal until the app is done.
///
/// # Errors
///
/// Returns an error when drawing fails, the terminal cannot be read, the signals
/// cannot be caught, the terminal cannot be handed to the editor and back, or a
/// process signal ended the TUI.
pub(super) async fn run(terminal: &mut DefaultTerminal, app: &mut App) -> anyhow::Result<()> {
    let signals = Signals::new().context("cannot catch the process signals")?;
    let (events, input) = mpsc::unbounded_channel();
    let real = Real {
        signals,
        screen: Screen::start(events),
    };
    drive(terminal, app, input, Some(real)).await
}

/// The loop itself, on any backend, reading terminal events from `input`; with
/// `real`, also the process signals, and the editor gets the terminal. A read
/// error on `input` ends the loop with that error; `input` closing quits. It
/// starts the app's background tasks, and hands each one's end back to the app.
///
/// # Errors
///
/// Returns an error when drawing fails, the terminal cannot be read, the
/// terminal cannot be handed to the editor and back, or a process signal ended
/// the TUI.
pub(super) async fn drive<B>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    input: UnboundedReceiver<io::Result<TermEvent>>,
    real: Option<Real>,
) -> anyhow::Result<()>
where
    B: Backend,
    B::Error: Send + Sync + 'static,
{
    let mut looping = Loop::new(terminal, input, &app.project.dir);
    if let Some(real) = real {
        looping.signals = Some(real.signals);
        looping.screen = Some(real.screen);
    }
    let result = looping.run(app).await;
    if let Some(mut editor) = looping.editor.take() {
        stop_editor(&mut editor).await;
    }
    looping.settle(app).await;
    if let Some(screen) = looping.screen.take() {
        screen.stop().await;
    }
    result
}

/// What woke the loop while it settles.
enum Settling {
    /// A task ended, or none is left.
    Ended(Option<(TaskId, Result<Done, String>)>),
    /// A process signal.
    Signal,
    /// Time to write the new log lines.
    Tick,
}

/// What woke the loop.
enum Wake {
    Input(Option<io::Result<TermEvent>>),
    Message(Msg),
    Done(TaskId, Result<Done, String>),
    Signal,
    Tick,
    Draw,
    /// A frame of motion is due.
    Frame,
}

struct Loop<'t, B: Backend> {
    terminal: &'t mut Terminal<B>,
    input: UnboundedReceiver<io::Result<TermEvent>>,
    tasks: Tasks,
    messages: UnboundedSender<Msg>,
    inbox: UnboundedReceiver<Msg>,
    tick: Interval,
    /// The process signals, with the real terminal.
    signals: Option<Signals>,
    /// The real terminal's screen, which the editor borrows.
    screen: Option<Screen>,
    /// Where lines are written once the loop ended and the screen was given
    /// back: stderr, with the real terminal.
    console: Option<Box<dyn Write + Send>>,
    /// The sequence number of the last log line written to the console.
    logged: u64,
    /// Whether the editor has the terminal: no input is read, nothing is drawn,
    /// and SIGINT is ignored (it comes from a Ctrl-C typed in the editor).
    suspended: bool,
    /// The editor running, if any.
    editor: Option<Child>,
    last_draw: Option<Instant>,
    /// When the last frame of motion was, while something moves on screen;
    /// `None` while nothing does.
    frame_at: Option<Instant>,
    /// Effects the loop could not run before it ended, for [`Loop::settle`].
    pending: Vec<Effect>,
}

impl<'t, B> Loop<'t, B>
where
    B: Backend,
    B::Error: Send + Sync + 'static,
{
    /// A loop drawing on `terminal`, reading terminal events from `input`, for
    /// the project in `dir`; no signals, no screen to hand over.
    fn new(
        terminal: &'t mut Terminal<B>,
        input: UnboundedReceiver<io::Result<TermEvent>>,
        dir: &std::path::Path,
    ) -> Self {
        let mut tick = tokio::time::interval(TICK);
        tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
        let (messages, inbox) = mpsc::unbounded_channel();
        Self {
            terminal,
            input,
            tasks: Tasks::new(dir, messages.clone()),
            messages,
            inbox,
            tick,
            signals: None,
            screen: None,
            console: None,
            logged: 0,
            suspended: false,
            editor: None,
            last_draw: None,
            frame_at: None,
            pending: Vec::new(),
        }
    }

    async fn run(&mut self, app: &mut App) -> anyhow::Result<()> {
        let mut effects = app.start();
        loop {
            self.apply_all(effects).await?;
            effects = match self.wait(app).await {
                Wake::Input(Some(Ok(event))) => app.on_input(&event),
                Wake::Input(Some(Err(error))) => {
                    return Err(anyhow::Error::new(error).context("cannot read the terminal"));
                },
                Wake::Input(None) => {
                    app.exit = Some(Exit::Quit);
                    Vec::new()
                },
                Wake::Message(message) => self.on_message(app, message)?,
                Wake::Done(id, result) => self.finished(app, id, result)?,
                Wake::Signal => app.on_signal(),
                Wake::Tick => app.on_tick(SystemTime::now()),
                Wake::Draw => Vec::new(),
                Wake::Frame => {
                    let elapsed = self.frame_elapsed();
                    app.on_frame(elapsed);
                    Vec::new()
                },
            };
            if let Err(error) = self.draw(app) {
                self.pending = effects;
                return Err(error);
            }
            if let Some(exit) = app.exit {
                return self.exit(app, effects, exit);
            }
        }
    }

    /// Applies `effects` in order; when one fails, the others are kept for
    /// [`Loop::settle`].
    async fn apply_all(&mut self, effects: Vec<Effect>) -> anyhow::Result<()> {
        let mut effects = effects.into_iter();
        while let Some(effect) = effects.next() {
            if let Err(error) = self.apply(effect).await {
                self.pending.extend(effects);
                return Err(error);
            }
        }
        Ok(())
    }

    /// Ends the loop for `exit`: the messages still in the inbox are handled,
    /// and `effects` are kept for [`Loop::settle`].
    fn exit(&mut self, app: &mut App, effects: Vec<Effect>, exit: Exit) -> anyhow::Result<()> {
        self.pending = effects;
        let more = self.drained(app)?;
        self.pending.extend(more);
        match exit {
            Exit::Quit => Ok(()),
            Exit::Signal => anyhow::bail!("interrupted by signal"),
        }
    }

    /// Task `id` ended with `result`. Every message of a task is in the inbox
    /// before its end: they are handled first, so its last lines are not lost.
    fn finished(
        &mut self,
        app: &mut App,
        id: TaskId,
        result: Result<Done, String>,
    ) -> anyhow::Result<Vec<Effect>> {
        let mut effects = self.drained(app)?;
        effects.extend(app.on_done(id, result));
        Ok(effects)
    }

    /// Handles every message already in the inbox.
    fn drained(&mut self, app: &mut App) -> anyhow::Result<Vec<Effect>> {
        let mut effects = Vec::new();
        while let Ok(message) = self.inbox.try_recv() {
            effects.extend(self.on_message(app, message)?);
        }
        Ok(effects)
    }

    /// Once the loop ended with work left (a terminal error, its input gone, or
    /// effects it could not run): the effects that must not be lost run (a
    /// training task or an edit starts, a token is cancelled), then the tasks
    /// end as on a confirmed quit and are waited for, however long: a training
    /// task is never aborted, since dropped mid-start it would leave a job or a
    /// pod that nothing finds again, and a start is never abandoned without a
    /// signal. With the real terminal, the screen is given back first and
    /// stderr says what is waited for; a signal meanwhile acts as a first one
    /// would. The ends still reach the app, for its exit notes.
    async fn settle(&mut self, app: &mut App) {
        self.logged = app.logs.seq();
        // List prices are only read: nothing waits for them.
        self.tasks.abort_lookups();
        for effect in std::mem::take(&mut self.pending) {
            self.apply_late(app, effect);
        }
        if !self.tasks.is_empty() {
            // Why the loop ended stays what it was.
            let why = app.exit;
            self.wait_tasks(app).await;
            app.exit = why.or(app.exit);
        }
        self.drain_late(app);
        self.write_logs(app);
    }

    /// Ends the tasks as a confirmed quit does and waits for them: a run still
    /// starting is detached once its job started, never abandoned; only a
    /// signal received meanwhile abandons it, as a first one would. With the
    /// real terminal, gives the screen back (so Ctrl-C raises SIGINT) and says
    /// on stderr what is waited for. The messages in the inbox are handled
    /// before each end.
    async fn wait_tasks(&mut self, app: &mut App) {
        for effect in app.on_loop_end() {
            self.apply_late(app, effect);
        }
        let lines = app.waiting_for();
        if !lines.is_empty()
            && let Some(screen) = &mut self.screen
        {
            screen.suspend().await.ok();
            self.console.get_or_insert_with(|| Box::new(io::stderr()));
        }
        self.write(&lines);
        let mut tick = tokio::time::interval(TICK);
        loop {
            let woke = tokio::select! {
                next = self.tasks.next() => Settling::Ended(next),
                () = Signals::recv(self.signals.as_mut(), true) => Settling::Signal,
                _ = tick.tick() => Settling::Tick,
            };
            let effects = match woke {
                Settling::Ended(Some((id, result))) => {
                    self.drain_late(app);
                    app.on_done(id, result)
                },
                Settling::Ended(None) => break,
                Settling::Signal => self.interrupted(app),
                Settling::Tick => Vec::new(),
            };
            for effect in effects {
                self.apply_late(app, effect);
            }
            self.write_logs(app);
        }
    }

    /// Writes the WARN and ERROR lines logged since the last ones written (the
    /// lines the Logs view shows, such as a flow's "remove the pod" warning),
    /// once the screen was given back; nothing before.
    fn write_logs(&mut self, app: &App) {
        let Some(console) = &mut self.console else {
            return;
        };
        for line in app.logs.since(Level::WARN, self.logged) {
            writeln!(console, "{} {}: {}", line.level, line.target, line.message).ok();
            self.logged = line.seq;
        }
    }

    /// A signal while the tasks are waited for: they end as on a signal, and
    /// stderr says so.
    fn interrupted(&mut self, app: &mut App) -> Vec<Effect> {
        let effects = app.on_signal();
        if let Some(status) = &app.status {
            let said = status.text.clone();
            self.write(&[said]);
        }
        effects
    }

    /// Writes `lines` once the screen was given back; nothing before.
    fn write(&mut self, lines: &[String]) {
        if let Some(console) = &mut self.console {
            for line in lines {
                writeln!(console, "{line}").ok();
            }
        }
    }

    /// Runs `effect` once the loop ended: a training task or an edit starts
    /// (never lost, never cut), a token is cancelled, a task abandoned; a
    /// reading, a stage or the editor does not start any more, nor a new run
    /// (the app notes it was not started).
    fn apply_late(&mut self, app: &mut App, effect: Effect) {
        match effect {
            Effect::Spawn(id, Task::Train(TrainJob::Start)) => app.start_dropped(id),
            Effect::Spawn(id, task @ (Task::Train(_) | Task::Edit(_))) => {
                self.tasks.spawn(id, task);
            },
            Effect::Cancel(id) => self.tasks.cancel(id),
            Effect::Abandon(id) => self.tasks.abandon(id),
            Effect::Spawn(..) | Effect::OpenEditor { .. } => {},
        }
    }

    /// Handles every message in the inbox once the loop ended.
    fn drain_late(&mut self, app: &mut App) {
        while let Ok(message) = self.inbox.try_recv() {
            if matches!(message, Msg::EditorExited(_)) {
                continue;
            }
            for effect in app.on_message(message) {
                self.apply_late(app, effect);
            }
        }
    }

    /// The next thing that happens.
    async fn wait(&mut self, app: &App) -> Wake {
        let next_draw = self.next_draw();
        let draw = app.dirty && !self.suspended;
        let frame = self.next_frame(app.frame_period());
        let signals = self.signals.as_mut();
        let interrupt = !self.suspended;
        tokio::select! {
            event = self.input.recv(), if !self.suspended => Wake::Input(event),
            Some(message) = self.inbox.recv() => Wake::Message(message),
            status = wait_editor(self.editor.as_mut()), if self.editor.is_some() => {
                Wake::Message(Msg::EditorExited(status))
            },
            Some((id, result)) = self.tasks.next(), if !self.tasks.is_empty() => {
                Wake::Done(id, result)
            },
            () = Signals::recv(signals, interrupt) => Wake::Signal,
            _ = self.tick.tick() => Wake::Tick,
            () = tokio::time::sleep_until(next_draw), if draw => Wake::Draw,
            () = tokio::time::sleep_until(frame.unwrap_or(next_draw)), if frame.is_some() => {
                Wake::Frame
            },
        }
    }

    /// When the next frame of motion is due: `period` after the last one (the
    /// first one a period from now), and never before the next draw may come,
    /// so each frame that changes the screen draws it; while something moves
    /// and the editor does not have the terminal. `None` otherwise, and the
    /// frames stop.
    fn next_frame(&mut self, period: Option<Duration>) -> Option<Instant> {
        let Some(period) = period.filter(|_| !self.suspended) else {
            self.frame_at = None;
            return None;
        };
        let due = *self.frame_at.get_or_insert_with(Instant::now) + period;
        Some(self.last_draw.map_or(due, |at| due.max(at + FRAME)))
    }

    /// The time since the last frame of motion, which is now.
    fn frame_elapsed(&mut self) -> Duration {
        let now = Instant::now();
        let elapsed = self
            .frame_at
            .map_or(Duration::ZERO, |last| now.saturating_duration_since(last));
        self.frame_at = Some(now);
        elapsed
    }

    fn next_draw(&self) -> Instant {
        self.last_draw.map_or_else(Instant::now, |at| at + FRAME)
    }

    /// Draws `app` when it changed and the shortest time between two draws
    /// passed. A panic while drawing is caught and returned as an error, as a
    /// draw error is: the loop then ends and never draws again, and the tasks
    /// are settled rather than dropped with the loop (a start never cut). The
    /// owner thread's panic hook has restored the terminal by then.
    fn draw(&mut self, app: &mut App) -> anyhow::Result<()> {
        // Never `Instant::now() >= self.next_draw()`: with no draw yet, the
        // second reading of the clock can come after the first, and the first
        // frame would then never be drawn.
        let due = self.last_draw.is_none_or(|at| Instant::now() >= at + FRAME);
        if app.dirty && !self.suspended && due {
            // While frames run, the motion clock catches up with the time
            // since the last one first: an effect this draw starts (a key
            // opened a view or a dialog) starts now, not at that frame.
            if self.frame_at.is_some() {
                let elapsed = self.frame_elapsed();
                app.on_frame(elapsed);
            }
            let terminal = &mut *self.terminal;
            let drawn = panic::catch_unwind(AssertUnwindSafe(|| {
                terminal.draw(|frame| ui::render(frame, app)).map(drop)
            }));
            match drawn {
                Ok(drawn) => drawn?,
                Err(payload) => {
                    let message = panic_message(payload.as_ref());
                    tracing::error!("drawing the TUI panicked: {message}");
                    anyhow::bail!("drawing the TUI panicked: {message}");
                },
            }
            app.dirty = false;
            self.last_draw = Some(Instant::now());
        }
        Ok(())
    }

    async fn apply(&mut self, effect: Effect) -> anyhow::Result<()> {
        match effect {
            Effect::Spawn(id, task) => self.tasks.spawn(id, task),
            Effect::Cancel(id) => self.tasks.cancel(id),
            Effect::Abandon(id) => self.tasks.abandon(id),
            Effect::OpenEditor { command, path } => self.open_editor(&command, path).await?,
        }
        Ok(())
    }

    /// Hands the terminal to the editor: stops reading input, drops the keys
    /// read before (they were typed for the TUI, not the editor) and leaves the
    /// alternate screen, then starts the editor, whose end comes back as
    /// [`Msg::EditorExited`]. Messages and tasks keep being handled meanwhile.
    ///
    /// The editor is started without a shell: its first word is the program, the
    /// others and the file its arguments.
    async fn open_editor(&mut self, command: &[String], path: PathBuf) -> anyhow::Result<()> {
        if let Some(screen) = &mut self.screen {
            screen
                .suspend()
                .await
                .context("cannot hand the terminal to the editor")?;
        }
        while self.input.try_recv().is_ok() {}
        self.suspended = true;
        let started = match command.split_first() {
            Some((program, args)) => tokio::process::Command::new(program)
                .args(args)
                .arg(&path)
                .kill_on_drop(true)
                .spawn(),
            None => Err(io::Error::other("no editor command")),
        };
        match started {
            Ok(child) => self.editor = Some(child),
            Err(error) => {
                self.messages.send(Msg::EditorExited(Err(error))).ok();
            },
        }
        Ok(())
    }

    /// The editor ended: takes the terminal back, repaints it all, then lets the
    /// app check the edited file.
    fn on_message(&mut self, app: &mut App, message: Msg) -> anyhow::Result<Vec<Effect>> {
        match message {
            Msg::EditorExited(status) => {
                self.editor = None;
                if let Some(screen) = &mut self.screen {
                    screen
                        .resume()
                        .context("cannot take the terminal back from the editor")?;
                }
                if let Some(signals) = &mut self.signals {
                    signals
                        .forget_interrupts()
                        .context("cannot catch the process signals")?;
                }
                self.terminal.clear()?;
                self.suspended = false;
                app.dirty = true;
                Ok(app.on_editor_exit(status))
            },
            message => Ok(app.on_message(message)),
        }
    }
}

/// SIGINT, SIGTERM and SIGHUP: in raw mode they only come from outside (`kill`, a
/// closed terminal), since Ctrl-C is a key.
pub(super) struct Signals {
    interrupt: Signal,
    terminate: Signal,
    hangup: Signal,
}

impl Signals {
    fn new() -> io::Result<Self> {
        Ok(Self {
            interrupt: signal(SignalKind::interrupt())?,
            terminate: signal(SignalKind::terminate())?,
            hangup: signal(SignalKind::hangup())?,
        })
    }

    /// Drops a SIGINT received while it was not listened to: a new listener
    /// only sees the ones sent after it.
    fn forget_interrupts(&mut self) -> io::Result<()> {
        self.interrupt = signal(SignalKind::interrupt())?;
        Ok(())
    }

    /// Waits for any of them, SIGINT only when `interrupt`; for ever without
    /// `signals`. A SIGINT sent meanwhile is kept until
    /// [`Signals::forget_interrupts`].
    async fn recv(signals: Option<&mut Self>, interrupt: bool) {
        let Some(signals) = signals else {
            return std::future::pending().await;
        };
        tokio::select! {
            _ = signals.interrupt.recv(), if interrupt => {},
            _ = signals.terminate.recv() => {},
            _ = signals.hangup.recv() => {},
        }
    }
}

/// The message a panic carried, when it is text.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "no message".to_string())
}

/// Waits for the editor to end; for ever without one.
async fn wait_editor(editor: Option<&mut Child>) -> io::Result<ExitStatus> {
    match editor {
        Some(editor) => editor.wait().await,
        None => std::future::pending().await,
    }
}

/// Stops the editor still running when the TUI ends, and waits for it, so it no
/// longer uses the terminal the guard restores: SIGTERM first, so it can put the
/// terminal back as it found it, then SIGKILL after [`EDITOR_GRACE`].
async fn stop_editor(editor: &mut Child) {
    if let Some(pid) = editor.id() {
        let asked = tokio::process::Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
        if asked.is_ok_and(|status| status.success())
            && tokio::time::timeout(EDITOR_GRACE, editor.wait())
                .await
                .is_ok()
        {
            return;
        }
    }
    if let Err(error) = editor.kill().await {
        tracing::error!("cannot stop the editor: {error}");
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use crossterm::event::KeyCode;
    use ratatui::backend::{ClearType, TestBackend, WindowSize};
    use ratatui::buffer::Cell;
    use ratatui::layout::{Position, Size};

    use super::*;
    use crate::tui::snapshots::{app, key};

    /// An upper bound only: a shared CI runner can be far slower than a laptop.
    const LIMIT: Duration = Duration::from_secs(30);

    #[tokio::test]
    async fn q_ends_the_loop_after_a_resize_is_drawn() -> Result<(), Box<dyn std::error::Error>> {
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;
        let mut app = app();
        let (events, input) = mpsc::unbounded_channel();
        terminal.backend_mut().resize(100, 30);
        events.send(Ok(TermEvent::Resize(100, 30)))?;
        events.send(Ok(key(KeyCode::Char('4'))))?;
        events.send(Ok(key(KeyCode::Char('q'))))?;
        tokio::time::timeout(LIMIT, drive(&mut terminal, &mut app, input, None)).await??;
        let area = terminal.backend().buffer().area;
        assert_eq!((area.width, area.height), (100, 30));
        assert_eq!(app.view, crate::tui::app::View::Logs);
        Ok(())
    }

    #[tokio::test]
    async fn the_loop_ends_when_its_input_is_gone() -> Result<(), Box<dyn std::error::Error>> {
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;
        let mut app = app();
        let (events, input) = mpsc::unbounded_channel();
        drop(events);
        tokio::time::timeout(LIMIT, drive(&mut terminal, &mut app, input, None)).await??;
        assert_eq!(app.exit, Some(Exit::Quit));
        Ok(())
    }

    #[tokio::test]
    async fn the_data_is_loaded_when_the_loop_starts() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let files = crate::dataset::DataFiles::new(dir.path());
        crate::dataset::rewrite(
            &files.subtopics,
            &crate::tui::snapshots::dataset().subtopics,
        )?;
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;
        let mut app = app();
        app.project.dir = dir.path().to_path_buf();
        let (events, input) = mpsc::unbounded_channel();
        // No key ends the loop, so it cannot quit before the load ends: it runs
        // for a window far longer than a read of three records, then is dropped.
        let window = Duration::from_secs(2);
        let run = tokio::time::timeout(window, drive(&mut terminal, &mut app, input, None)).await;
        assert!(run.is_err(), "the loop ended on its own: {run:?}");
        drop(events);
        assert_eq!(app.load, None, "the load ended");
        let subtopics = app
            .dataset
            .model
            .as_ref()
            .map(|model| model.data.subtopics.len());
        assert_eq!(subtopics, Some(3));
        Ok(())
    }

    #[tokio::test]
    async fn a_read_error_ends_the_loop_with_that_error() -> Result<(), Box<dyn std::error::Error>>
    {
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;
        let mut app = app();
        let (events, input) = mpsc::unbounded_channel();
        events.send(Err(io::Error::other("the terminal is gone")))?;
        drop(events);
        let result =
            tokio::time::timeout(LIMIT, drive(&mut terminal, &mut app, input, None)).await?;
        let error = result.err().map(|error| format!("{error:#}"));
        assert_eq!(
            error.as_deref(),
            Some("cannot read the terminal: the terminal is gone")
        );
        Ok(())
    }

    /// A terminal error ends the loop, but a training task still running is
    /// waited for, never aborted, and its end reaches the app.
    #[tokio::test]
    async fn a_terminal_error_waits_for_the_training_tasks()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let run = "20260921-133200-a1b2";
        crate::runs::Runs::new(dir.path()).save(&crate::tui::snapshots::run(
            run,
            "homelab",
            crate::runs::RunState::Running,
        ))?;
        let backend = Shared::new();
        let frame = Arc::clone(&backend.frame);
        let mut terminal = Terminal::new(backend)?;
        let mut app = app();
        app.project.dir = dir.path().to_path_buf();
        let (events, input) = mpsc::unbounded_channel();
        let run_loop = drive(&mut terminal, &mut app, input, None);
        let keys = async {
            events.send(Ok(key(KeyCode::Char('3'))))?;
            drawn(&frame, run).await?;
            events.send(Ok(key(KeyCode::Char('a'))))?;
            events.send(Err(io::Error::other("the terminal is gone")))?;
            Ok::<_, Box<dyn std::error::Error>>(())
        };
        let joined = tokio::time::timeout(LIMIT, async { tokio::join!(run_loop, keys) }).await;
        let Ok((result, sent)) = joined else {
            return Err(stalled("waiting for the loop", &app, &frame).into());
        };
        sent?;
        assert!(result.is_err());
        assert!(app.training.tasks.is_empty(), "the task was waited for");
        // No overbrainer.toml: the attach flow fails, and says why.
        let ended = app.training.ended.get(run).and_then(|e| e.error.clone());
        assert!(
            ended
                .as_deref()
                .is_some_and(|e| e.contains("overbrainer.toml")),
            "{ended:?}"
        );
        Ok(())
    }

    /// A test backend that shares the text of its screen after each draw,
    /// whose draws fail once `fail` is set, and unwind, as a view that panics
    /// would, once `unwind` is set.
    struct Shared {
        inner: TestBackend,
        frame: Arc<Mutex<String>>,
        fail: Arc<AtomicBool>,
        unwind: Arc<AtomicBool>,
    }

    impl Shared {
        fn new() -> Self {
            Self {
                inner: TestBackend::new(80, 24),
                frame: Arc::new(Mutex::new(String::new())),
                fail: Arc::new(AtomicBool::new(false)),
                unwind: Arc::new(AtomicBool::new(false)),
            }
        }
    }

    /// The error a [`TestBackend`] never returns.
    fn never(error: Infallible) -> io::Error {
        match error {}
    }

    impl Backend for Shared {
        type Error = io::Error;

        fn draw<'a, I>(&mut self, content: I) -> io::Result<()>
        where
            I: Iterator<Item = (u16, u16, &'a Cell)>,
        {
            if self.unwind.load(Ordering::SeqCst) {
                std::panic::resume_unwind(Box::new("a view failed"));
            }
            if self.fail.load(Ordering::SeqCst) {
                return Err(io::Error::other("the terminal is gone"));
            }
            self.inner.draw(content).map_err(never)?;
            let text: String = self
                .inner
                .buffer()
                .content()
                .iter()
                .map(Cell::symbol)
                .collect();
            if let Ok(mut frame) = self.frame.lock() {
                *frame = text;
            }
            Ok(())
        }

        fn hide_cursor(&mut self) -> io::Result<()> {
            self.inner.hide_cursor().map_err(never)
        }

        fn show_cursor(&mut self) -> io::Result<()> {
            self.inner.show_cursor().map_err(never)
        }

        fn get_cursor_position(&mut self) -> io::Result<Position> {
            self.inner.get_cursor_position().map_err(never)
        }

        fn set_cursor_position<P: Into<Position>>(&mut self, position: P) -> io::Result<()> {
            self.inner.set_cursor_position(position).map_err(never)
        }

        fn clear(&mut self) -> io::Result<()> {
            self.inner.clear().map_err(never)
        }

        fn clear_region(&mut self, clear_type: ClearType) -> io::Result<()> {
            self.inner.clear_region(clear_type).map_err(never)
        }

        fn size(&self) -> io::Result<Size> {
            self.inner.size().map_err(never)
        }

        fn window_size(&mut self) -> io::Result<WindowSize> {
            self.inner.window_size().map_err(never)
        }

        fn flush(&mut self) -> io::Result<()> {
            self.inner.flush().map_err(never)
        }
    }

    /// Waits until `text` is on the screen shared in `frame`, 10 s at most.
    async fn drawn(frame: &Mutex<String>, text: &str) -> Result<(), String> {
        for _ in 0..1000 {
            if frame.lock().is_ok_and(|frame| frame.contains(text)) {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Err(format!("never drawn: {text}"))
    }

    /// Why a test timed out: what `app` was at and the last screen drawn, so a
    /// timeout on a slow runner says where it stopped.
    fn stalled(stage: &str, app: &App, frame: &Mutex<String>) -> String {
        let screen = frame.lock().map(|frame| frame.clone()).unwrap_or_default();
        format!(
            "timed out {stage}: view {:?}, exit {:?}, {} training task(s), status {:?}, screen {screen:?}",
            app.view,
            app.exit,
            app.training.tasks.len(),
            app.status.as_ref().map(|status| status.text.clone()),
        )
    }

    /// Frames of motion come only while something moves: the first one a
    /// period from now, then one period after the last; none once nothing
    /// moves or the editor has the terminal.
    #[tokio::test]
    async fn frames_come_only_while_something_moves() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;
        let mut looping = looping(&mut terminal, dir.path());
        assert_eq!(looping.next_frame(None), None);
        assert_eq!(looping.frame_at, None);
        let period = Duration::from_millis(80);
        let first = looping.next_frame(Some(period)).ok_or("no frame")?;
        let started = looping.frame_at.ok_or("no start")?;
        assert_eq!(first, started + period);
        assert_eq!(looping.next_frame(Some(period)), Some(first), "not moved");
        let elapsed = looping.frame_elapsed();
        assert_eq!(looping.frame_at, Some(started + elapsed));
        looping.suspended = true;
        assert_eq!(looping.next_frame(Some(period)), None);
        assert_eq!(looping.frame_at, None, "the frames stop");
        Ok(())
    }

    /// A frame never comes before the next draw may: each frame that changes
    /// the screen draws it, rather than wake the loop once for the frame and
    /// once more for the draw.
    #[tokio::test]
    async fn a_frame_comes_at_or_after_the_draw_it_needs() -> Result<(), Box<dyn std::error::Error>>
    {
        use crate::tui::motion::FAST;
        let dir = tempfile::tempdir()?;
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;
        let mut looping = looping(&mut terminal, dir.path());
        let started = Instant::now();
        looping.frame_at = Some(started);
        looping.last_draw = Some(started + Duration::from_millis(5));
        let frame = looping.next_frame(Some(FAST)).ok_or("no frame")?;
        assert_eq!(frame, started + Duration::from_millis(5) + FRAME);
        looping.last_draw = Some(started - FRAME);
        let frame = looping.next_frame(Some(FAST)).ok_or("no frame")?;
        assert_eq!(frame, started + FAST, "the draw is due already");
        Ok(())
    }

    /// While frames run, a draw first brings the motion clock to now: an
    /// effect a key starts begins when it was pressed, not at the last frame.
    #[tokio::test]
    async fn a_draw_brings_the_motion_clock_to_now() -> Result<(), Box<dyn std::error::Error>> {
        use crate::tui::motion::{Motion, MotionLevel};
        let dir = tempfile::tempdir()?;
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;
        let mut looping = looping(&mut terminal, dir.path());
        let mut app = app();
        app.motion = Motion::new(MotionLevel::On);
        let gap = Duration::from_millis(50);
        looping.frame_at = Some(Instant::now().checked_sub(gap).ok_or("no instant")?);
        app.dirty = true;
        looping.draw(&mut app)?;
        assert!(app.motion.clock() >= gap, "{:?}", app.motion.clock());
        assert!(!app.dirty, "drawn");
        Ok(())
    }

    /// With motion on, the spinner of running work turns on screen while the
    /// loop waits for nothing else.
    #[tokio::test]
    async fn the_spinner_turns_while_work_runs() -> Result<(), Box<dyn std::error::Error>> {
        use crate::tui::motion::{Motion, MotionLevel};
        let backend = Shared::new();
        let frame = Arc::clone(&backend.frame);
        let mut terminal = Terminal::new(backend)?;
        let mut app = app();
        app.motion = Motion::new(MotionLevel::On);
        crate::tui::snapshots::pipeline_running(&mut app);
        let (events, input) = mpsc::unbounded_channel();
        let run_loop = drive(&mut terminal, &mut app, input, None);
        let watch = async move {
            drawn(&frame, "⠋ answers 120/400").await?;
            drawn(&frame, "⠙ answers 120/400").await?;
            drop(events);
            Ok::<_, String>(())
        };
        let (result, watched) =
            tokio::time::timeout(LIMIT, async { tokio::join!(run_loop, watch) }).await?;
        watched?;
        result?;
        Ok(())
    }

    /// The effects of a wake whose draw failed still run: a confirmed cancel
    /// starts, and is waited for.
    #[tokio::test]
    async fn the_effects_of_a_failed_draw_still_run() -> Result<(), Box<dyn std::error::Error>> {
        let dir = crate::tui::snapshots::project()?;
        let mut record = crate::tui::snapshots::run(RUN, "homelab", crate::runs::RunState::Failed);
        record.job = None;
        crate::runs::Runs::new(dir.path()).save(&record)?;
        let backend = Shared::new();
        let (frame, fail) = (Arc::clone(&backend.frame), Arc::clone(&backend.fail));
        let mut terminal = Terminal::new(backend)?;
        let mut app = app();
        app.project.dir = dir.path().to_path_buf();
        let (events, input) = mpsc::unbounded_channel();
        let run_loop = drive(&mut terminal, &mut app, input, None);
        let keys = async {
            events.send(Ok(key(KeyCode::Char('3'))))?;
            drawn(&frame, RUN).await?;
            events.send(Ok(key(KeyCode::Char('c'))))?;
            drawn(&frame, "Cancel a run?").await?;
            // Past the shortest time between two draws: `y` is drawn at once.
            tokio::time::sleep(FRAME * 2).await;
            fail.store(true, Ordering::SeqCst);
            events.send(Ok(key(KeyCode::Char('y'))))?;
            Ok::<_, Box<dyn std::error::Error>>(())
        };
        let joined = tokio::time::timeout(LIMIT, async { tokio::join!(run_loop, keys) }).await;
        let Ok((result, sent)) = joined else {
            return Err(stalled("waiting for the loop", &app, &frame).into());
        };
        sent?;
        let error = result.err().map(|error| format!("{error:#}"));
        assert_eq!(error.as_deref(), Some("the terminal is gone"));
        let ended = app.training.ended.get(RUN).and_then(|e| e.error.clone());
        assert_eq!(ended, Some(format!("run {RUN} has not started")));
        Ok(())
    }

    /// A loop on `terminal` with no input, for the project in `dir`.
    fn looping<'t>(
        terminal: &'t mut Terminal<TestBackend>,
        dir: &std::path::Path,
    ) -> Loop<'t, TestBackend> {
        Loop::new(terminal, mpsc::unbounded_channel().1, dir)
    }

    const RUN: &str = "20260921-133200-a1b2";

    /// The app, following no run but for task 5, cancelling [`RUN`].
    fn cancelling_app(dir: &std::path::Path) -> App {
        use crate::tui::training::{Follow, Job};
        let mut app = app();
        app.project.dir = dir.to_path_buf();
        app.training
            .tasks
            .insert(TaskId(5), Follow::new(Job::Cancel, RUN));
        app
    }

    #[tokio::test]
    async fn the_last_lines_of_a_task_reach_the_exit_notes()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::cli::front::Report;
        let dir = tempfile::tempdir()?;
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;
        let mut looping = looping(&mut terminal, dir.path());
        let mut app = cancelling_app(dir.path());
        app.leaving = Some(Exit::Quit);
        let line = format!("train: run {RUN} cancelled");
        looping
            .messages
            .send(Msg::Report(TaskId(5), Report::Line(line.clone())))?;
        looping.finished(&mut app, TaskId(5), Ok(Done::Trained(Ok(()))))?;
        assert_eq!(app.exit, Some(Exit::Quit));
        assert_eq!(app.exit_notes, [line]);
        Ok(())
    }

    /// An effect the loop could not run before it ended (here a cancel task)
    /// still runs, and is waited for.
    #[tokio::test]
    async fn a_cancel_left_pending_runs_once_the_loop_ended()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::tui::tasks::{Task, TrainJob};
        let dir = crate::tui::snapshots::project()?;
        let mut record = crate::tui::snapshots::run(RUN, "homelab", crate::runs::RunState::Failed);
        record.job = None;
        crate::runs::Runs::new(dir.path()).save(&record)?;
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;
        let mut looping = looping(&mut terminal, dir.path());
        let mut app = cancelling_app(dir.path());
        app.exit = Some(Exit::Quit);
        looping.pending = vec![Effect::Spawn(
            TaskId(5),
            Task::Train(TrainJob::Cancel(RUN.into())),
        )];
        tokio::time::timeout(LIMIT, looping.settle(&mut app)).await?;
        assert!(app.training.tasks.is_empty());
        let error = app.training.ended.get(RUN).and_then(|e| e.error.clone());
        assert_eq!(error, Some(format!("run {RUN} has not started")));
        assert_eq!(app.exit, Some(Exit::Quit), "why the loop ended is kept");
        Ok(())
    }

    /// A start left pending when the loop ended (a draw error) is never
    /// spawned: nothing would follow it. The exit notes say it was not started.
    #[tokio::test]
    async fn a_start_left_pending_never_runs_once_the_loop_ended()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::tui::tasks::{Task, TrainJob};
        use crate::tui::training::{Follow, Job};
        let dir = tempfile::tempdir()?;
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;
        let mut looping = looping(&mut terminal, dir.path());
        let mut app = app();
        app.project.dir = dir.path().to_path_buf();
        app.training
            .tasks
            .insert(TaskId(4), Follow::new(Job::Start { runpod: true }, ""));
        looping.pending = vec![Effect::Spawn(TaskId(4), Task::Train(TrainJob::Start))];
        tokio::time::timeout(LIMIT, looping.settle(&mut app)).await?;
        assert!(looping.tasks.is_empty(), "never spawned");
        assert!(app.training.tasks.is_empty());
        assert_eq!(
            app.exit_notes,
            [
                "a new training run was not started: the TUI ended first; start it again with \
              `overbrainer train` or t"
            ]
        );
        Ok(())
    }

    /// Once the loop ended, the messages of a task are handled before its end:
    /// its lines come before its error in the exit notes.
    #[tokio::test]
    async fn the_last_lines_of_a_task_waited_for_come_before_its_end()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::cli::front::Report;
        use crate::tui::tasks::{Task, TrainJob};
        let dir = crate::tui::snapshots::project()?;
        let mut record = crate::tui::snapshots::run(RUN, "homelab", crate::runs::RunState::Failed);
        record.job = None;
        crate::runs::Runs::new(dir.path()).save(&record)?;
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;
        let mut looping = looping(&mut terminal, dir.path());
        let mut app = cancelling_app(dir.path());
        app.leaving = Some(Exit::Quit);
        looping.pending = vec![Effect::Spawn(
            TaskId(5),
            Task::Train(TrainJob::Cancel(RUN.into())),
        )];
        looping
            .messages
            .send(Msg::Report(TaskId(5), Report::Line("a line".into())))?;
        tokio::time::timeout(LIMIT, looping.settle(&mut app)).await?;
        assert_eq!(
            app.exit_notes,
            ["a line".to_string(), format!("run {RUN} has not started")]
        );
        Ok(())
    }

    /// The app of the project in `dir`, with a Runpod run still provisioning
    /// as task 4.
    fn provisioning_app(dir: &std::path::Path) -> App {
        use crate::tui::training::{Follow, Job};
        let mut app = app();
        app.project.dir = dir.to_path_buf();
        app.training
            .tasks
            .insert(TaskId(4), Follow::new(Job::Start { runpod: true }, RUN));
        app
    }

    /// A loop whose input fails at once, for the project in `dir`.
    fn failing_input<'t>(
        terminal: &'t mut Terminal<TestBackend>,
        dir: &std::path::Path,
    ) -> Result<Loop<'t, TestBackend>, Box<dyn std::error::Error>> {
        let (events, input) = mpsc::unbounded_channel();
        events.send(Err(io::Error::other("the terminal is gone")))?;
        Ok(Loop::new(terminal, input, dir))
    }

    /// A loop ended by a terminal error waits for a Runpod run still
    /// provisioning, and never abandons it: no signal came.
    #[tokio::test]
    async fn a_terminal_error_never_abandons_a_provisioning_run()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;
        let mut looping = failing_input(&mut terminal, dir.path())?;
        let mut app = provisioning_app(dir.path());
        let release = tokio_util::sync::CancellationToken::new();
        let abandon = looping.tasks.park(TaskId(4), release.clone());
        let result = tokio::time::timeout(LIMIT, looping.run(&mut app)).await?;
        assert!(result.is_err());
        let check = async {
            tokio::time::sleep(Duration::from_millis(200)).await;
            let abandoned = abandon.load(Ordering::SeqCst);
            release.cancel();
            abandoned
        };
        let ((), abandoned) = tokio::time::timeout(LIMIT, async {
            tokio::join!(looping.settle(&mut app), check)
        })
        .await?;
        assert!(!abandoned, "abandoned while it was waited for");
        assert!(!abandon.load(Ordering::SeqCst));
        assert!(app.training.tasks.is_empty(), "it was waited for");
        assert_eq!(app.leaving, Some(Exit::Quit));
        Ok(())
    }

    /// A signal while a loop ended by a terminal error waits abandons a Runpod
    /// run still provisioning, as a first signal would.
    #[tokio::test]
    async fn a_signal_while_the_loop_ends_abandons_a_provisioning_run()
    -> Result<(), Box<dyn std::error::Error>> {
        let _signals = crate::test_support::SIGNALS.lock().await;
        let dir = tempfile::tempdir()?;
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;
        let mut looping = failing_input(&mut terminal, dir.path())?;
        looping.signals = Some(Signals::new()?);
        let mut app = provisioning_app(dir.path());
        let abandon = looping
            .tasks
            .park(TaskId(4), tokio_util::sync::CancellationToken::new());
        let result = tokio::time::timeout(LIMIT, looping.run(&mut app)).await?;
        assert!(result.is_err());
        let check = async {
            tokio::time::sleep(Duration::from_millis(200)).await;
            let abandoned = abandon.load(Ordering::SeqCst);
            raise("INT")?;
            Ok::<_, Box<dyn std::error::Error>>(abandoned)
        };
        let ((), abandoned) = tokio::time::timeout(LIMIT, async {
            tokio::join!(looping.settle(&mut app), check)
        })
        .await?;
        assert!(!abandoned?, "abandoned before the signal");
        assert!(abandon.load(Ordering::SeqCst), "abandoned on the signal");
        assert!(app.training.tasks.is_empty());
        assert_eq!(app.exit, Some(Exit::Signal));
        Ok(())
    }

    /// A panic while drawing ends the loop as a draw error does: the loop
    /// never draws again, and a Runpod run still provisioning is waited for,
    /// never dropped with the loop nor abandoned.
    #[tokio::test]
    async fn a_panic_while_drawing_ends_the_loop_as_a_draw_error()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let backend = Shared::new();
        backend.unwind.store(true, Ordering::SeqCst);
        let frame = Arc::clone(&backend.frame);
        let mut terminal = Terminal::new(backend)?;
        let (_events, input) = mpsc::unbounded_channel();
        let mut looping = Loop::new(&mut terminal, input, dir.path());
        let mut app = provisioning_app(dir.path());
        let release = tokio_util::sync::CancellationToken::new();
        let abandon = looping.tasks.park(TaskId(4), release.clone());
        let ran = tokio::time::timeout(LIMIT, looping.run(&mut app)).await;
        let Ok(result) = ran else {
            return Err(stalled("running the loop", &app, &frame).into());
        };
        let error = result.err().map(|error| format!("{error:#}"));
        assert_eq!(
            error.as_deref(),
            Some("drawing the TUI panicked: a view failed")
        );
        let check = async {
            tokio::time::sleep(Duration::from_millis(200)).await;
            let abandoned = abandon.load(Ordering::SeqCst);
            release.cancel();
            abandoned
        };
        let settled = tokio::time::timeout(LIMIT, async {
            tokio::join!(looping.settle(&mut app), check)
        })
        .await;
        let Ok(((), abandoned)) = settled else {
            return Err(stalled("settling", &app, &frame).into());
        };
        assert!(!abandoned);
        assert!(app.training.tasks.is_empty(), "it was waited for");
        Ok(())
    }

    /// A writer sharing what it gets, standing for stderr.
    #[derive(Clone, Default)]
    struct Console(Arc<Mutex<Vec<u8>>>);

    impl Console {
        fn text(&self) -> String {
            self.0
                .lock()
                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                .unwrap_or_default()
        }
    }

    impl Write for Console {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .map_err(|_| io::Error::other("poisoned"))?
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// A log line of the flows at `level`.
    fn logged(level: tracing::Level, message: &str) -> crate::logging::LogLine {
        crate::logging::LogLine {
            seq: 0,
            level,
            target: "overbrainer::runpod_train".into(),
            time: SystemTime::UNIX_EPOCH,
            message: message.into(),
        }
    }

    /// Once the screen is given back, what is waited for is written, then each
    /// warning or error the flows log as it arrives, and those logged last
    /// before the exit; never the other lines.
    #[tokio::test]
    async fn warnings_logged_while_the_loop_ends_are_written()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;
        let mut looping = failing_input(&mut terminal, dir.path())?;
        let console = Console::default();
        looping.console = Some(Box::new(console.clone()));
        let mut app = provisioning_app(dir.path());
        let logs = app.logs.clone();
        let release = tokio_util::sync::CancellationToken::new();
        looping.tasks.park(TaskId(4), release.clone());
        let result = tokio::time::timeout(LIMIT, looping.run(&mut app)).await?;
        assert!(result.is_err());
        let remove = "remove the pod with `overbrainer pod rm`";
        let check = async {
            logs.push(logged(tracing::Level::INFO, "not written"));
            logs.push(logged(tracing::Level::WARN, remove));
            for _ in 0..500 {
                if console.text().contains(remove) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            let written = console.text().contains(remove);
            logs.push(logged(tracing::Level::ERROR, "the last line"));
            release.cancel();
            written
        };
        let ((), written) = tokio::time::timeout(LIMIT, async {
            tokio::join!(looping.settle(&mut app), check)
        })
        .await?;
        assert!(written, "written as it arrived: {}", console.text());
        let text = console.text();
        assert_eq!(
            text.lines().collect::<Vec<_>>(),
            [
                "waiting for run 20260921-133200-a1b2 to start; Ctrl-C abandons it...",
                &format!("WARN overbrainer::runpod_train: {remove}"),
                "ERROR overbrainer::runpod_train: the last line",
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn the_editor_round_trip_saves_the_edit() -> Result<(), Box<dyn std::error::Error>> {
        let dir = crate::tui::snapshots::project()?;
        let mut terminal = Terminal::new(TestBackend::new(80, 24))?;
        let mut app = app();
        app.project.dir = dir.path().to_path_buf();
        app.project.topics = crate::tui::snapshots::topics();
        app.editor = vec![
            "sh".into(),
            "-c".into(),
            "printf 'When does a borrow end?' > \"$1\"".into(),
            "editor".into(),
        ];
        let (events, input) = mpsc::unbounded_channel();
        let run = drive(&mut terminal, &mut app, input, None);
        let files = crate::dataset::DataFiles::new(dir.path());
        let keys = [
            KeyCode::Enter,
            KeyCode::Char('j'),
            KeyCode::Char('l'),
            KeyCode::Char('j'),
            KeyCode::Char('j'),
            KeyCode::Char('j'),
            KeyCode::Char('e'),
            // Typed before the editor took the terminal: dropped, not replayed.
            KeyCode::Char('4'),
        ];
        let quit = async {
            // The data is loaded by then.
            tokio::time::sleep(Duration::from_millis(300)).await;
            for code in keys {
                events.send(Ok(key(code)))?;
            }
            for _ in 0..100 {
                tokio::time::sleep(Duration::from_millis(50)).await;
                if files.train.exists() {
                    break;
                }
            }
            // The edit writes its files before the loop hears it ended, so `q`
            // may still find it saving and ask to confirm: `y` answers. When
            // `q` quits at once, the loop is gone and `y` is never read.
            events.send(Ok(key(KeyCode::Char('q'))))?;
            events.send(Ok(key(KeyCode::Char('y'))))
        };
        let (ran, sent) = tokio::time::timeout(LIMIT, async { tokio::join!(run, quit) }).await?;
        ran?;
        sent?;
        let questions: Vec<crate::dataset::Question> = crate::dataset::read(&files.questions)?;
        let texts: Vec<&str> = questions.iter().map(|q| q.text.as_str()).collect();
        assert!(texts.contains(&"When does a borrow end?"), "{texts:?}");
        assert!(!texts.contains(&"When does NLL end a borrow?"));
        // Quitting while the edit was saving notes that it was saved; no other note.
        assert!(
            app.exit_notes
                .iter()
                .all(|note| note.starts_with("a change was saved: ")),
            "{:?}",
            app.exit_notes
        );
        assert_eq!(app.view, crate::tui::app::View::Dataset, "4 was dropped");
        Ok(())
    }

    /// Sends `signal` to this test process, through a shell's `kill`.
    fn raise(signal: &str) -> Result<(), Box<dyn std::error::Error>> {
        let status = std::process::Command::new("sh")
            .args(["-c", &format!("kill -{signal} \"$PPID\"")])
            .status()?;
        if !status.success() {
            return Err(format!("kill -{signal}: {status}").into());
        }
        Ok(())
    }

    /// While the editor has the terminal a SIGINT is ignored, and forgotten when
    /// the TUI takes it back; SIGTERM still ends the TUI. The listeners are
    /// installed before any signal is sent, so the test process catches them.
    #[tokio::test]
    async fn sigint_is_ignored_while_the_editor_runs_and_forgotten_after()
    -> Result<(), Box<dyn std::error::Error>> {
        let _signals = crate::test_support::SIGNALS.lock().await;
        let short = Duration::from_millis(300);
        let mut signals = Signals::new()?;
        raise("INT")?;
        let suspended = tokio::time::timeout(short, Signals::recv(Some(&mut signals), false));
        assert!(suspended.await.is_err(), "SIGINT woke a suspended loop");
        signals.forget_interrupts()?;
        let resumed = tokio::time::timeout(short, Signals::recv(Some(&mut signals), true));
        assert!(resumed.await.is_err(), "the old SIGINT was replayed");
        raise("INT")?;
        tokio::time::timeout(LIMIT, Signals::recv(Some(&mut signals), true)).await?;
        raise("TERM")?;
        tokio::time::timeout(LIMIT, Signals::recv(Some(&mut signals), false)).await?;
        Ok(())
    }

    /// An editor, as `sh`, running `on_term` on SIGTERM; returned once its trap
    /// is set, so the signal cannot come first.
    async fn editor_trapping(on_term: &str) -> io::Result<Child> {
        use tokio::io::AsyncBufReadExt;
        let mut editor = tokio::process::Command::new("sh")
            .args([
                "-c",
                &format!("trap '{on_term}' TERM; echo ready; while :; do sleep 0.1; done"),
            ])
            .stdout(Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;
        let Some(stdout) = editor.stdout.take() else {
            return Err(io::Error::other("no stdout"));
        };
        let mut ready = String::new();
        tokio::io::BufReader::new(stdout)
            .read_line(&mut ready)
            .await?;
        Ok(editor)
    }

    #[tokio::test]
    async fn an_editor_left_running_is_asked_to_end_then_reaped()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut editor = tokio::time::timeout(LIMIT, editor_trapping("exit 3")).await??;
        tokio::time::timeout(LIMIT, stop_editor(&mut editor)).await?;
        let status = editor.try_wait()?;
        assert_eq!(
            status.and_then(|status| status.code()),
            Some(3),
            "SIGTERM first"
        );
        Ok(())
    }

    #[tokio::test]
    async fn an_editor_ignoring_sigterm_is_killed_then_reaped()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::os::unix::process::ExitStatusExt;
        let mut editor = tokio::time::timeout(LIMIT, editor_trapping("")).await??;
        tokio::time::timeout(LIMIT, stop_editor(&mut editor)).await?;
        let status = editor.try_wait()?;
        assert_eq!(status.and_then(|status| status.signal()), Some(9));
        Ok(())
    }

    #[tokio::test]
    async fn a_burst_through_a_task_bus_loses_nothing_while_the_loop_draws()
    -> Result<(), Box<dyn std::error::Error>> {
        use crate::events::{Event, EventBus, Stage};
        use crate::tui::tasks::{BUS_CAPACITY, Msg, TaskId, forward};

        let mut terminal = Terminal::new(TestBackend::new(120, 40))?;
        let mut app = app();
        app.view = crate::tui::app::View::Pipeline;
        app.pipeline_task = Some(TaskId(1));
        app.pipeline.started(crate::cli::data::Command::Answers, 8);
        let bus = EventBus::with_capacity(BUS_CAPACITY);
        let (messages, mut inbox) = mpsc::unbounded_channel();
        forward(TaskId(1), bus.subscribe(), messages);
        let received = std::cell::Cell::new(0_usize);
        // Two bursts of 20 000 events, the size of a large tail read, each
        // followed by an await, as `watch` does between two reads.
        let publish = async {
            bus.publish(Event::StageStarted {
                stage: Stage::Answers,
                total: 40_000,
            });
            for burst in 0..2 {
                for n in 0..20_000 {
                    bus.publish(Event::ItemDone {
                        stage: Stage::Answers,
                        id: format!("{burst}-{n}"),
                        usage: None,
                    });
                }
                while received.get() < 1 + (burst + 1) * 20_000 {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            }
            drop(bus);
        };
        let consume = async {
            while let Some(message) = inbox.recv().await {
                if let Msg::Lagged(_, skipped) = message {
                    return Err(format!("{skipped} events skipped"));
                }
                app.on_message(message);
                received.set(received.get() + 1);
                if received.get().is_multiple_of(5_000) {
                    terminal
                        .draw(|frame| ui::render(frame, &mut app))
                        .map_err(|error| error.to_string())?;
                }
            }
            Ok(received.get())
        };
        let ((), received) =
            tokio::time::timeout(LIMIT, async { tokio::join!(publish, consume) }).await?;
        assert_eq!(received?, 40_001);
        let row = app.pipeline.row(Stage::Answers);
        assert_eq!((row.finished, row.total), (40_000, 40_000));
        Ok(())
    }
}
