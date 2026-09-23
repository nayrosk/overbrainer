//! The real terminal: restoring it on every exit path, and reading its input.

use crossterm::event::{Event, EventStream};
use futures::StreamExt;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// Restores the terminal when dropped: built only once `ratatui::try_init`
/// succeeded, so an error returned from the loop restores it too. A panic is
/// covered by the hook `try_init` installs.
pub(super) struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        if let Err(error) = ratatui::try_restore() {
            eprintln!("cannot restore the terminal: {error}");
        }
    }
}

/// The task owning crossterm's `EventStream`, forwarding its events.
pub(super) struct InputTask {
    token: CancellationToken,
    handle: JoinHandle<()>,
}

impl InputTask {
    /// Starts reading the terminal, sending each event on `events`.
    pub(super) fn start(events: UnboundedSender<Event>) -> Self {
        let token = CancellationToken::new();
        let stop = token.clone();
        let handle = tokio::spawn(async move {
            let mut stream = EventStream::new();
            loop {
                tokio::select! {
                    () = stop.cancelled() => break,
                    event = stream.next() => match event {
                        Some(Ok(event)) => {
                            if events.send(event).is_err() {
                                break;
                            }
                        },
                        Some(Err(error)) => {
                            tracing::error!("cannot read the terminal: {error}");
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
