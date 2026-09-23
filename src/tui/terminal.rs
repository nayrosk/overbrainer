//! The real terminal: setting it up, restoring it on every exit path, and reading
//! its input.

use std::io::{self, Write};
use std::panic::{self, PanicHookInfo};
use std::sync::Arc;
use std::thread;

use crossterm::event::{Event, EventStream};
use crossterm::execute;
use crossterm::terminal::{EnterAlternateScreen, enable_raw_mode};
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
/// # Errors
///
/// Returns an error when the terminal cannot be set up.
pub(super) fn init() -> io::Result<DefaultTerminal> {
    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen)?;
    Terminal::new(CrosstermBackend::new(io::stdout()))
}

/// Leaves the alternate screen and raw mode. A failure is written to stderr,
/// ignoring a failed write: the terminal may be gone.
fn restore() {
    if let Err(error) = ratatui::try_restore() {
        writeln!(io::stderr(), "cannot restore the terminal: {error}").ok();
    }
}

/// The task owning crossterm's `EventStream`, forwarding its events, then its
/// error if reading fails.
pub(super) struct InputTask {
    token: CancellationToken,
    handle: JoinHandle<()>,
}

impl InputTask {
    /// Starts reading the terminal, sending each event on `events`. A read error
    /// is sent too, and ends the task.
    pub(super) fn start(events: UnboundedSender<io::Result<Event>>) -> Self {
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
    pub(super) async fn stop(self) {
        self.token.cancel();
        self.handle.await.ok();
    }
}
