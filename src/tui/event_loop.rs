//! The event loop: one task selecting over terminal input, background tasks and
//! their messages, process signals and a clock tick, drawing the app when it
//! changed. It owns the terminal and runs the effects the app asks for.

use std::io;
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

use super::app::{App, Effect, Exit};
use super::tasks::{Done, Msg, TaskId, Tasks};
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
    let mut tick = tokio::time::interval(TICK);
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let (messages, inbox) = mpsc::unbounded_channel();
    let mut looping = Loop {
        terminal,
        input,
        tasks: Tasks::new(&app.project.dir),
        messages,
        inbox,
        tick,
        real,
        suspended: false,
        editor: None,
        last_draw: None,
    };
    let result = looping.run(app).await;
    if let Some(mut editor) = looping.editor.take() {
        stop_editor(&mut editor).await;
    }
    if let Some(real) = looping.real {
        real.screen.stop().await;
    }
    result
}

/// What woke the loop.
enum Wake {
    Input(Option<io::Result<TermEvent>>),
    Message(Msg),
    Done(TaskId, Result<Done, String>),
    Signal,
    Tick,
    Draw,
}

struct Loop<'t, B: Backend> {
    terminal: &'t mut Terminal<B>,
    input: UnboundedReceiver<io::Result<TermEvent>>,
    tasks: Tasks,
    messages: UnboundedSender<Msg>,
    inbox: UnboundedReceiver<Msg>,
    tick: Interval,
    real: Option<Real>,
    /// Whether the editor has the terminal: no input is read, nothing is drawn,
    /// and SIGINT is ignored (it comes from a Ctrl-C typed in the editor).
    suspended: bool,
    /// The editor running, if any.
    editor: Option<Child>,
    last_draw: Option<Instant>,
}

impl<B> Loop<'_, B>
where
    B: Backend,
    B::Error: Send + Sync + 'static,
{
    async fn run(&mut self, app: &mut App) -> anyhow::Result<()> {
        let mut effects = app.start();
        loop {
            for effect in effects {
                self.apply(effect).await?;
            }
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
                Wake::Done(id, result) => app.on_done(id, result),
                Wake::Signal => {
                    app.on_signal();
                    Vec::new()
                },
                Wake::Tick => {
                    app.on_tick(SystemTime::now());
                    Vec::new()
                },
                Wake::Draw => Vec::new(),
            };
            self.draw(app)?;
            match app.exit {
                Some(Exit::Quit) => return Ok(()),
                Some(Exit::Signal) => anyhow::bail!("interrupted by signal"),
                None => {},
            }
        }
    }

    /// The next thing that happens.
    async fn wait(&mut self, app: &App) -> Wake {
        let next_draw = self.next_draw();
        let draw = app.dirty && !self.suspended;
        let signals = self.real.as_mut().map(|real| &mut real.signals);
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
        }
    }

    fn next_draw(&self) -> Instant {
        self.last_draw.map_or_else(Instant::now, |at| at + FRAME)
    }

    fn draw(&mut self, app: &mut App) -> anyhow::Result<()> {
        if app.dirty && !self.suspended && Instant::now() >= self.next_draw() {
            self.terminal.draw(|frame| ui::render(frame, app))?;
            app.dirty = false;
            self.last_draw = Some(Instant::now());
        }
        Ok(())
    }

    async fn apply(&mut self, effect: Effect) -> anyhow::Result<()> {
        match effect {
            Effect::Spawn(id, task) => self.tasks.spawn(id, task),
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
        if let Some(real) = &mut self.real {
            real.screen
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
                if let Some(real) = &mut self.real {
                    real.screen
                        .resume()
                        .context("cannot take the terminal back from the editor")?;
                    real.signals
                        .forget_interrupts()
                        .context("cannot catch the process signals")?;
                }
                self.terminal.clear()?;
                self.suspended = false;
                app.dirty = true;
                Ok(app.on_editor_exit(status))
            },
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
    use crossterm::event::KeyCode;
    use ratatui::backend::TestBackend;

    use super::*;
    use crate::tui::snapshots::{app, key};

    const LIMIT: Duration = Duration::from_secs(10);

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
            events.send(Ok(key(KeyCode::Char('q'))))
        };
        let (ran, sent) = tokio::time::timeout(LIMIT, async { tokio::join!(run, quit) }).await?;
        ran?;
        sent?;
        let questions: Vec<crate::dataset::Question> = crate::dataset::read(&files.questions)?;
        let texts: Vec<&str> = questions.iter().map(|q| q.text.as_str()).collect();
        assert!(texts.contains(&"When does a borrow end?"), "{texts:?}");
        assert!(!texts.contains(&"When does NLL end a borrow?"));
        assert!(app.exit_notes.is_empty());
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
}
