//! The event loop: one task selecting over terminal input, process signals and a
//! clock tick, drawing the app when it changed.

use std::io;
use std::time::{Duration, SystemTime};

use crossterm::event::Event as TermEvent;
use ratatui::backend::Backend;
use ratatui::{DefaultTerminal, Terminal};
use tokio::signal::unix::{Signal, SignalKind, signal};
use tokio::sync::mpsc::{self, UnboundedReceiver};
use tokio::time::{Instant, MissedTickBehavior};

use super::app::{App, Exit};
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
/// Returns an error when drawing fails, the signals cannot be caught, or a
/// process signal ended the TUI.
pub(super) async fn run(terminal: &mut DefaultTerminal, app: &mut App) -> anyhow::Result<()> {
    let (events, input) = mpsc::unbounded_channel();
    let reader = InputTask::start(events);
    let result = drive(terminal, app, input).await;
    reader.stop().await;
    result
}

/// The loop itself, on any backend, reading terminal events from `input`.
///
/// # Errors
///
/// See [`run`].
pub(super) async fn drive<B>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    mut input: UnboundedReceiver<TermEvent>,
) -> anyhow::Result<()>
where
    B: Backend,
    B::Error: Send + Sync + 'static,
{
    let mut signals = Signals::new()?;
    let mut tick = tokio::time::interval(TICK);
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut last_draw: Option<Instant> = None;
    loop {
        let next_draw = last_draw.map_or_else(Instant::now, |at| at + FRAME);
        tokio::select! {
            event = input.recv() => match event {
                Some(event) => app.on_input(&event),
                None => app.exit = Some(Exit::Quit),
            },
            () = signals.recv() => app.on_signal(),
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
struct Signals {
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

    /// Waits for any of them.
    async fn recv(&mut self) {
        tokio::select! {
            _ = self.interrupt.recv() => {},
            _ = self.terminate.recv() => {},
            _ = self.hangup.recv() => {},
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
        events.send(TermEvent::Resize(100, 30))?;
        events.send(key(KeyCode::Char('4')))?;
        events.send(key(KeyCode::Char('q')))?;
        tokio::time::timeout(LIMIT, drive(&mut terminal, &mut app, input)).await??;
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
        tokio::time::timeout(LIMIT, drive(&mut terminal, &mut app, input)).await??;
        assert_eq!(app.exit, Some(Exit::Quit));
        Ok(())
    }
}
