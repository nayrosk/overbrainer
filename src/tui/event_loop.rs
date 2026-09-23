//! The event loop: one task selecting over terminal input, background tasks,
//! process signals and a clock tick, drawing the app when it changed.

use std::io;
use std::time::{Duration, SystemTime};

use anyhow::Context;
use crossterm::event::Event as TermEvent;
use ratatui::backend::Backend;
use ratatui::{DefaultTerminal, Terminal};
use tokio::signal::unix::{Signal, SignalKind, signal};
use tokio::sync::mpsc::{self, UnboundedReceiver};
use tokio::time::{Instant, MissedTickBehavior};

use super::app::{App, Effect, Exit};
use super::tasks::Tasks;
use super::terminal::InputTask;
use super::ui;

/// Time between two ticks of the app's clock.
const TICK: Duration = Duration::from_millis(250);
/// Shortest time between two draws.
const FRAME: Duration = Duration::from_millis(33);

/// Runs the TUI on the real terminal until the app is done.
///
/// # Errors
///
/// Returns an error when drawing fails, the terminal cannot be read, the signals
/// cannot be caught, or a process signal ended the TUI.
pub(super) async fn run(terminal: &mut DefaultTerminal, app: &mut App) -> anyhow::Result<()> {
    let signals = Signals::new().context("cannot catch the process signals")?;
    let (events, input) = mpsc::unbounded_channel();
    let reader = InputTask::start(events);
    let result = drive(terminal, app, input, Some(signals)).await;
    reader.stop().await;
    result
}

/// The loop itself, on any backend, reading terminal events from `input` and the
/// process signals from `signals` when given. A read error on `input` ends the
/// loop with that error; `input` closing quits. It starts the app's background
/// tasks, and hands each one's end back to the app.
///
/// # Errors
///
/// Returns an error when drawing fails, the terminal cannot be read, or a
/// process signal ended the TUI.
pub(super) async fn drive<B>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    mut input: UnboundedReceiver<io::Result<TermEvent>>,
    mut signals: Option<Signals>,
) -> anyhow::Result<()>
where
    B: Backend,
    B::Error: Send + Sync + 'static,
{
    let mut tick = tokio::time::interval(TICK);
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut tasks = Tasks::new(&app.project.dir);
    let mut last_draw: Option<Instant> = None;
    let mut effects = app.start();
    loop {
        for effect in effects.drain(..) {
            match effect {
                Effect::Spawn(id, task) => tasks.spawn(id, task),
            }
        }
        let next_draw = last_draw.map_or_else(Instant::now, |at| at + FRAME);
        tokio::select! {
            event = input.recv() => match event {
                Some(Ok(event)) => effects = app.on_input(&event),
                Some(Err(error)) => {
                    return Err(anyhow::Error::new(error).context("cannot read the terminal"));
                },
                None => app.exit = Some(Exit::Quit),
            },
            Some((id, result)) = tasks.next(), if !tasks.is_empty() => {
                effects = app.on_done(id, result);
            },
            () = Signals::recv(signals.as_mut()) => app.on_signal(),
            _ = tick.tick() => app.on_tick(SystemTime::now()),
            () = tokio::time::sleep_until(next_draw), if app.dirty => {},
        }
        if app.dirty && Instant::now() >= next_draw {
            terminal.draw(|frame| ui::render(frame, app))?;
            app.dirty = false;
            last_draw = Some(Instant::now());
        }
        match app.exit {
            Some(Exit::Quit) => return Ok(()),
            Some(Exit::Signal) => anyhow::bail!("interrupted by signal"),
            None => {},
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

    /// Waits for any of them; for ever without `signals`.
    async fn recv(signals: Option<&mut Self>) {
        let Some(signals) = signals else {
            return std::future::pending().await;
        };
        tokio::select! {
            _ = signals.interrupt.recv() => {},
            _ = signals.terminate.recv() => {},
            _ = signals.hangup.recv() => {},
        }
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
}
