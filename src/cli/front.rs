//! What differs between the front ends that run the pipeline and training flows:
//! where their events go, what interrupts them, and where their summary lines go.

use std::future::Future;
use std::io;
use std::pin::{Pin, pin};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context as TaskContext, Poll, Waker};

use anyhow::Context;
use tokio::signal::unix::{SignalKind, signal};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::events::EventBus;

/// A front end of the flows in `cli::data` and `cli::train`.
pub(crate) enum Frontend {
    /// The command line: a new bus rendered to stderr, Ctrl-C (SIGINT), stdout.
    Cli,
    /// The terminal UI: the task's own bus, the token that detaches it, the flag
    /// that abandons a Runpod provisioning, and a sink into the app.
    Tui {
        /// The task's bus, which the app forwards.
        bus: EventBus,
        /// Cancelled to interrupt the task's flow (a raced watch detaches).
        detach: CancellationToken,
        /// Set to make Runpod provisioning stop and delete its pod.
        abandon: Arc<AtomicBool>,
        /// Receives what the command line would print.
        report: Arc<dyn Fn(Report) + Send + Sync>,
    },
}

/// What a flow tells its front end besides events.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Report {
    /// A line the command line prints on stdout.
    Line(String),
    /// A training run was created, with this ID (the command line says nothing).
    RunCreated(String),
}

impl Frontend {
    /// The bus a flow publishes on, with what renders it.
    pub(crate) fn open_bus(&self) -> BusGuard {
        match self {
            Self::Cli => {
                let bus = EventBus::new();
                let renderer = tokio::spawn(super::progress::render(bus.subscribe()));
                BusGuard {
                    bus,
                    renderer: Some(renderer),
                }
            },
            Self::Tui { bus, .. } => BusGuard {
                bus: bus.clone(),
                renderer: None,
            },
        }
    }

    /// What stops a raced flow and is noted by a shielded one, caught from now on.
    pub(crate) fn interrupt(&self) -> Interrupt {
        match self {
            Self::Cli => Interrupt::catch(),
            Self::Tui { detach, .. } => {
                let detach = detach.clone();
                Interrupt::on(Box::pin(async move {
                    detach.cancelled_owned().await;
                    Ok(())
                }))
            },
        }
    }

    /// The flag Runpod provisioning checks between its steps, set from now on.
    ///
    /// # Errors
    ///
    /// Returns an error when the Ctrl-C handler cannot be installed.
    pub(crate) fn provisioning_flag(&self) -> anyhow::Result<Flag> {
        match self {
            Self::Cli => {
                let mut sigint = signal(SignalKind::interrupt()).context("cannot catch Ctrl-C")?;
                let interrupted = Arc::new(AtomicBool::new(false));
                let seen = Arc::clone(&interrupted);
                let watcher = tokio::spawn(async move {
                    if sigint.recv().await.is_some() {
                        seen.store(true, Ordering::SeqCst);
                    }
                });
                Ok(Flag {
                    interrupted,
                    watcher: Some(watcher),
                })
            },
            Self::Tui { abandon, .. } => Ok(Flag {
                interrupted: Arc::clone(abandon),
                watcher: None,
            }),
        }
    }

    /// Emits a summary line: the command line prints it on stdout, the TUI gets
    /// it as a [`Report::Line`].
    pub(crate) fn line(&self, line: &str) {
        match self {
            Self::Cli => println!("{line}"),
            Self::Tui { report, .. } => report(Report::Line(line.to_string())),
        }
    }

    /// Says that training run `id` was created: the TUI binds its task to it.
    pub(crate) fn run_created(&self, id: &str) {
        match self {
            Self::Cli => {},
            Self::Tui { report, .. } => report(Report::RunCreated(id.to_string())),
        }
    }
}

/// The bus of a flow, and the task rendering it if any.
pub(crate) struct BusGuard {
    /// Where the flow publishes.
    pub(crate) bus: EventBus,
    renderer: Option<JoinHandle<()>>,
}

impl BusGuard {
    /// Drops the bus, then waits for the renderer to show what is left.
    pub(crate) async fn close(self) {
        drop(self.bus);
        if let Some(renderer) = self.renderer {
            renderer.await.ok();
        }
    }
}

/// The interrupted flag of Runpod provisioning, and the task that sets it if any.
pub(crate) struct Flag {
    /// Set once the flow must stop provisioning.
    pub(crate) interrupted: Arc<AtomicBool>,
    watcher: Option<JoinHandle<()>>,
}

impl Flag {
    /// Stops watching for the interruption.
    pub(crate) fn close(self) {
        if let Some(watcher) = self.watcher {
            watcher.abort();
        }
    }
}

type Signal = Pin<Box<dyn Future<Output = io::Result<()>> + Send>>;

/// An interruption (Ctrl-C on the command line), caught from its creation on so
/// that it no longer stops the process: a flow either runs to its end regardless
/// ([`Interrupt::shield`]) or stops at the first interruption ([`Interrupt::race`]).
pub(crate) enum Interrupt {
    /// Waiting for the interruption.
    Listening(Signal),
    /// The interruption came.
    Caught,
    /// The interruption cannot be caught: Ctrl-C stops the process as usual.
    Off,
}

impl Interrupt {
    /// Interrupted when `signal` resolves.
    fn on(signal: Signal) -> Self {
        Self::Listening(signal)
    }

    /// Starts catching Ctrl-C now.
    fn catch() -> Self {
        let mut signal: Signal = Box::pin(tokio::signal::ctrl_c());
        // The handler is installed on the first poll; a later poll registers the
        // real waker.
        let mut cx = TaskContext::from_waker(Waker::noop());
        match signal.as_mut().poll(&mut cx) {
            Poll::Pending => Self::Listening(signal),
            Poll::Ready(result) => Self::after(result),
        }
    }

    fn after(result: io::Result<()>) -> Self {
        match result {
            Ok(()) => Self::Caught,
            Err(error) => {
                tracing::warn!("cannot catch Ctrl-C: {error}");
                Self::Off
            },
        }
    }

    /// Whether the interruption came.
    pub(crate) fn caught(&self) -> bool {
        matches!(self, Self::Caught)
    }

    /// Runs `flow` to its end, noting an interruption meanwhile.
    pub(crate) async fn shield<T>(&mut self, flow: impl Future<Output = T>) -> T {
        let mut flow = pin!(flow);
        let result = match self {
            Self::Listening(signal) => tokio::select! {
                output = &mut flow => return output,
                result = signal.as_mut() => result,
            },
            Self::Caught | Self::Off => return flow.await,
        };
        *self = Self::after(result);
        flow.await
    }

    /// Runs `flow`, or `None` when the interruption comes first (or already came).
    pub(crate) async fn race<T>(&mut self, flow: impl Future<Output = T>) -> Option<T> {
        let mut flow = pin!(flow);
        let result = match self {
            Self::Listening(signal) => tokio::select! {
                output = &mut flow => return Some(output),
                result = signal.as_mut() => result,
            },
            Self::Caught => return None,
            Self::Off => return Some(flow.await),
        };
        *self = Self::after(result);
        if self.caught() {
            None
        } else {
            Some(flow.await)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::signal::unix::{SignalKind, signal};

    use super::*;
    use crate::events::{Event, Stage};

    /// Sends SIGINT to this process, as Ctrl-C does.
    async fn ctrl_c() -> io::Result<()> {
        let status = tokio::process::Command::new("kill")
            .args(["-INT", &std::process::id().to_string()])
            .status()
            .await?;
        if status.success() {
            Ok(())
        } else {
            Err(io::Error::other("kill failed"))
        }
    }

    #[tokio::test]
    async fn ctrl_c_never_stops_a_shielded_flow_and_stops_a_raced_one()
    -> Result<(), Box<dyn std::error::Error>> {
        let _signals = crate::test_support::SIGNALS.lock().await;
        let limit = Duration::from_secs(10);
        let mut interrupt = Frontend::Cli.interrupt();
        assert!(matches!(interrupt, Interrupt::Listening(_)));
        // Ctrl-C arrives while the flow waits: it still ends, and Ctrl-C is noted.
        let mut received = signal(SignalKind::interrupt())?;
        let shielded = async {
            ctrl_c().await?;
            received.recv().await;
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok::<_, io::Error>("ended")
        };
        let ended = tokio::time::timeout(limit, interrupt.shield(shielded)).await??;
        assert_eq!(ended, "ended");
        assert!(interrupt.caught());
        // Once Ctrl-C was pressed, a race does not even start its flow.
        assert_eq!(interrupt.race(std::future::ready(())).await, None);

        let mut interrupt = Frontend::Cli.interrupt();
        let raced = async {
            ctrl_c().await?;
            std::future::pending::<()>().await;
            Ok::<_, io::Error>(())
        };
        let raced = tokio::time::timeout(limit, interrupt.race(raced)).await?;
        assert!(raced.is_none());
        assert!(interrupt.caught());
        Ok(())
    }

    fn tui(detach: &CancellationToken) -> Frontend {
        Frontend::Tui {
            bus: EventBus::with_capacity(8),
            detach: detach.clone(),
            abandon: Arc::new(AtomicBool::new(false)),
            report: Arc::new(|_| {}),
        }
    }

    #[tokio::test]
    async fn a_detach_token_never_stops_a_shielded_flow_and_stops_a_raced_one()
    -> Result<(), Box<dyn std::error::Error>> {
        let limit = Duration::from_secs(10);
        let token = CancellationToken::new();
        let mut interrupt = tui(&token).interrupt();
        assert!(matches!(interrupt, Interrupt::Listening(_)));
        let shielded = async {
            token.cancel();
            tokio::time::sleep(Duration::from_millis(50)).await;
            "ended"
        };
        assert_eq!(
            tokio::time::timeout(limit, interrupt.shield(shielded)).await?,
            "ended"
        );
        assert!(interrupt.caught());
        assert_eq!(interrupt.race(std::future::ready(())).await, None);

        let token = CancellationToken::new();
        let mut interrupt = tui(&token).interrupt();
        let raced = async {
            token.cancel();
            std::future::pending::<()>().await;
        };
        assert!(
            tokio::time::timeout(limit, interrupt.race(raced))
                .await?
                .is_none()
        );
        assert!(interrupt.caught());
        Ok(())
    }

    #[tokio::test]
    async fn a_tui_front_end_shares_its_bus_flag_and_reports() -> anyhow::Result<()> {
        let lines = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = Arc::clone(&lines);
        let abandon = Arc::new(AtomicBool::new(false));
        let front = Frontend::Tui {
            bus: EventBus::with_capacity(8),
            detach: CancellationToken::new(),
            abandon: Arc::clone(&abandon),
            report: Arc::new(move |report| {
                if let Ok(mut lines) = seen.lock() {
                    lines.push(report);
                }
            }),
        };
        let Frontend::Tui { bus, .. } = &front else {
            anyhow::bail!("not a TUI front end");
        };
        let mut events = bus.subscribe();
        let guard = front.open_bus();
        guard.bus.publish(Event::StageStarted {
            stage: Stage::Split,
            total: 2,
        });
        assert!(events.try_recv().is_ok(), "the task's own bus");
        guard.close().await;
        let flag = front.provisioning_flag()?;
        abandon.store(true, Ordering::SeqCst);
        assert!(flag.interrupted.load(Ordering::SeqCst));
        flag.close();
        front.line("split: 1 train, 1 eval, 0 excluded, 0 orphaned");
        let reported = lines.lock().map(|lines| lines.clone()).unwrap_or_default();
        assert_eq!(
            reported,
            [Report::Line(
                "split: 1 train, 1 eval, 0 excluded, 0 orphaned".into()
            )]
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_cli_bus_is_rendered_until_it_is_closed() -> Result<(), tokio::time::error::Elapsed> {
        let guard = Frontend::Cli.open_bus();
        let mut extra = guard.bus.subscribe();
        guard.bus.publish(Event::StageStarted {
            stage: Stage::Split,
            total: 1,
        });
        assert!(extra.try_recv().is_ok());
        tokio::time::timeout(Duration::from_secs(10), guard.close()).await
    }

    #[tokio::test]
    async fn a_cli_provisioning_flag_starts_clear() -> anyhow::Result<()> {
        let flag = Frontend::Cli.provisioning_flag()?;
        assert!(!flag.interrupted.load(Ordering::SeqCst));
        flag.close();
        Ok(())
    }
}
