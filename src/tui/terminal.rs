//! The real terminal: setting it up, restoring it on every exit path, reading its
//! input, and handing it to the editor and back.

use std::io::{self, Write};
use std::mem::ManuallyDrop;
use std::panic::{self, PanicHookInfo};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use crossterm::cursor::{Hide, Show};
use crossterm::event::{self, Event, EventStream};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::{DefaultTerminal, Terminal};
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// A panic hook, shared so the guard can put it back.
type Hook = dyn Fn(&PanicHookInfo<'_>) + Send + Sync + 'static;

/// Restores the terminal when dropped, and meanwhile owns the panic hook.
///
/// Built before the terminal is set up, so a setup that fails halfway (raw mode
/// on, alternate screen not entered) is undone too, as is an error returned from
/// the loop.
///
/// The hook restores the terminal, then runs the previous hook, only for a panic
/// on the thread that owns the terminal: that panic ends the TUI. A panic on any
/// other thread is a background task's, which tokio catches and the TUI survives;
/// the hook only logs it at ERROR, into the Logs view, since the previous hook
/// would print over the alternate screen.
pub(super) struct TerminalGuard {
    previous: Arc<Hook>,
}

impl TerminalGuard {
    /// Installs the hook for the current thread, which will own the terminal.
    pub(super) fn enter() -> Self {
        let previous: Arc<Hook> = Arc::from(panic::take_hook());
        let owner = thread::current().id();
        let chained = Arc::clone(&previous);
        panic::set_hook(Box::new(move |info| {
            if thread::current().id() == owner {
                restore();
                chained(info);
            } else {
                tracing::error!("a background task panicked: {info}");
            }
        }));
        Self { previous }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore();
        // Hooks cannot be changed while this thread panics; the process ends then.
        if !thread::panicking() {
            drop(panic::take_hook());
            let previous = Arc::clone(&self.previous);
            panic::set_hook(Box::new(move |info| previous(info)));
        }
    }
}

/// Enables raw mode and enters the alternate screen, like `ratatui::try_init`
/// without the panic hook it installs (the guard's hook replaces it).
///
/// The terminal is never dropped. ratatui's `Drop for Terminal` shows a hidden
/// cursor again and, when that fails on a closed terminal, reports it with
/// `eprintln!`, which panics on a dead stderr and aborts the process during an
/// unwind. [`restore`] shows the cursor instead, from the guard and the panic
/// hook, and the terminal's memory is left to the end of the process.
///
/// # Errors
///
/// Returns an error when the terminal cannot be set up.
pub(super) fn init() -> io::Result<ManuallyDrop<DefaultTerminal>> {
    RESTORED.store(false, Ordering::SeqCst);
    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen)?;
    Terminal::new(CrosstermBackend::new(io::stdout())).map(ManuallyDrop::new)
}

/// Set while [`Screen::suspend`] has put the terminal back in its normal mode
/// (the editor has it, or the TUI waits for its work once its loop ended):
/// [`restore`] then has nothing to undo.
static RESTORED: AtomicBool = AtomicBool::new(false);

/// Leaves the alternate screen and raw mode, and shows the cursor, unless
/// [`Screen::suspend`] already did. A failure is written to stderr, ignoring a
/// failed write: the terminal may be gone.
fn restore() {
    if RESTORED.load(Ordering::SeqCst) {
        return;
    }
    if let Err(error) = ratatui::try_restore() {
        writeln!(io::stderr(), "cannot restore the terminal: {error}").ok();
    }
    execute!(io::stdout(), Show).ok();
}

/// The screen of the real terminal, with the task reading its input.
///
/// While the editor has the terminal, no task reads it and the terminal is in
/// its normal mode; the guard's panic hook still restores it then, which leaving
/// the alternate screen and raw mode a second time does no harm to.
pub(super) struct Screen {
    reader: Option<InputTask>,
    events: UnboundedSender<io::Result<Event>>,
}

impl Screen {
    /// Starts reading the terminal, sending each event on `events`.
    pub(super) fn start(events: UnboundedSender<io::Result<Event>>) -> Self {
        Self {
            reader: Some(InputTask::start(events.clone())),
            events,
        }
    }

    /// Hands the terminal over: stops reading it first and waits until the
    /// `EventStream` is dropped (so its reader thread takes no keystroke from the
    /// editor), drops the keys read but not handled, then leaves the alternate
    /// screen and raw mode, and shows the cursor. The terminal is put back in
    /// its normal mode even when the keys cannot be dropped.
    ///
    /// # Errors
    ///
    /// Returns an error when the keys cannot be dropped or the terminal cannot
    /// be switched back.
    pub(super) async fn suspend(&mut self) -> io::Result<()> {
        if let Some(reader) = self.reader.take() {
            reader.stop().await;
        }
        // Keys crossterm already parsed stay in its global buffer, and would come
        // back after the editor: drop them, with any the tty still holds.
        let dropped = drop_keys();
        let left = execute!(io::stdout(), LeaveAlternateScreen, Show);
        let normal = disable_raw_mode();
        if left.is_ok() && normal.is_ok() {
            RESTORED.store(true, Ordering::SeqCst);
        }
        dropped.and(left).and(normal)
    }

    /// Takes the terminal back: raw mode, the alternate screen and a hidden
    /// cursor, then reads it again with a new `EventStream`. The caller clears
    /// the terminal so the next draw repaints it all.
    ///
    /// # Errors
    ///
    /// Returns an error when the terminal cannot be set up again.
    pub(super) fn resume(&mut self) -> io::Result<()> {
        RESTORED.store(false, Ordering::SeqCst);
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen, Hide)?;
        self.reader = Some(InputTask::start(self.events.clone()));
        Ok(())
    }

    /// Stops reading the terminal.
    pub(super) async fn stop(self) {
        if let Some(reader) = self.reader {
            reader.stop().await;
        }
    }
}

/// Reads and drops every key crossterm already parsed or the tty holds.
fn drop_keys() -> io::Result<()> {
    while event::poll(Duration::ZERO)? {
        event::read()?;
    }
    Ok(())
}

/// The task owning crossterm's `EventStream`, forwarding its events, then its
/// error if reading fails.
struct InputTask {
    token: CancellationToken,
    handle: JoinHandle<()>,
}

impl InputTask {
    /// Starts reading the terminal, sending each event on `events`. A read error
    /// is sent too, and ends the task.
    fn start(events: UnboundedSender<io::Result<Event>>) -> Self {
        let token = CancellationToken::new();
        let stop = token.clone();
        let handle = tokio::spawn(async move {
            let mut stream = EventStream::new();
            loop {
                tokio::select! {
                    () = stop.cancelled() => break,
                    event = stream.next() => match event {
                        Some(Ok(event)) => {
                            if events.send(Ok(event)).is_err() {
                                break;
                            }
                        },
                        Some(Err(error)) => {
                            events.send(Err(error)).ok();
                            break;
                        },
                        None => break,
                    },
                }
            }
        });
        Self { token, handle }
    }

    /// Stops reading and waits until the `EventStream` is dropped, so its reader
    /// thread no longer takes keystrokes.
    async fn stop(self) {
        self.token.cancel();
        self.handle.await.ok();
    }
}
