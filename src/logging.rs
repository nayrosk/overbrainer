//! Tracing setup: formatted lines on stderr for the command line, an in-memory
//! buffer for the terminal UI.

use std::collections::VecDeque;
use std::fmt::{self, Write as _};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::SystemTime;

use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// Env variable holding the tracing filter, for example `info` or `overbrainer=debug`.
pub const LOG_ENV: &str = "OVERBRAINER_LOG";

/// Filter used when `OVERBRAINER_LOG` is unset or invalid: our own logs at `info`,
/// dependencies at `warn`. vaultrs and rustify log every failed request at ERROR
/// level before overbrainer reports the same failure with its full cause, so they
/// are silenced; `OVERBRAINER_LOG` brings them back when debugging.
pub const DEFAULT_FILTER: &str = "overbrainer=info,vaultrs=off,rustify=off,warn";

/// Lines the terminal UI keeps in memory; older ones are dropped.
pub const LOG_LINES: usize = 5000;

/// Where logs go.
#[derive(Debug, Clone)]
pub enum LogMode {
    /// Formatted lines on stderr: every command but `tui`.
    Stderr,
    /// Kept in memory for the terminal UI's Logs view; nothing reaches stderr.
    Tui(LogBuffer),
}

/// Installs the global subscriber for `mode`, filtered by `OVERBRAINER_LOG`, else
/// [`DEFAULT_FILTER`]. [`LogMode::Stderr`] honors `NO_COLOR`. A second call does
/// nothing.
pub fn init(mode: &LogMode) {
    let filter =
        EnvFilter::try_from_env(LOG_ENV).unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER));
    match mode {
        LogMode::Stderr => {
            let ansi = std::env::var_os("NO_COLOR").is_none();
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_writer(std::io::stderr)
                .with_ansi(ansi)
                .with_target(false)
                .try_init()
                .ok();
        },
        LogMode::Tui(buffer) => {
            tracing_subscriber::registry()
                .with(filter)
                .with(buffer.layer())
                .try_init()
                .ok();
        },
    }
}

/// One log line kept by a [`LogBuffer`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogLine {
    /// Position of the line among every line pushed, from 1.
    pub seq: u64,
    /// Severity.
    pub level: Level,
    /// Module path or explicit target of the event.
    pub target: String,
    /// When the event was recorded.
    pub time: SystemTime,
    /// The message, then each other field as ` key=value`.
    pub message: String,
}

/// Lines selected by [`LogBuffer::window`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Window {
    /// The selected lines, oldest first.
    pub lines: Vec<LogLine>,
    /// Lines kept at the requested level, selected or not.
    pub matching: usize,
}

/// A bounded, shared buffer of log lines. Cloning shares the same buffer.
#[derive(Debug, Clone)]
pub struct LogBuffer {
    ring: Arc<Mutex<Ring>>,
}

#[derive(Debug)]
struct Ring {
    lines: VecDeque<LogLine>,
    capacity: usize,
    seq: u64,
}

impl LogBuffer {
    /// An empty buffer keeping at most `capacity` lines.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            ring: Arc::new(Mutex::new(Ring {
                lines: VecDeque::with_capacity(capacity.min(LOG_LINES)),
                capacity: capacity.max(1),
                seq: 0,
            })),
        }
    }

    /// The ring, even when a thread panicked while holding it: the lines stay valid.
    fn lock(&self) -> MutexGuard<'_, Ring> {
        self.ring.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Adds `line`, numbered after the previous one, dropping the oldest line when
    /// full.
    pub fn push(&self, mut line: LogLine) {
        let mut ring = self.lock();
        ring.seq += 1;
        line.seq = ring.seq;
        if ring.lines.len() == ring.capacity {
            ring.lines.pop_front();
        }
        ring.lines.push_back(line);
    }

    /// Lines pushed so far, dropped ones included.
    #[must_use]
    pub fn seq(&self) -> u64 {
        self.lock().seq
    }

    /// At most `count` lines at `min` or more severe, ending `offset` matching lines
    /// before the newest one (clamped so the window stays full when it can), oldest
    /// first. Copies only those lines.
    #[must_use]
    pub fn window(&self, min: Level, count: usize, offset: usize) -> Window {
        let ring = self.lock();
        let matching: Vec<&LogLine> = ring.lines.iter().filter(|line| line.level <= min).collect();
        let end = matching
            .len()
            .saturating_sub(offset)
            .max(count.min(matching.len()));
        let start = end.saturating_sub(count);
        Window {
            lines: matching[start..end]
                .iter()
                .map(|line| (*line).clone())
                .collect(),
            matching: matching.len(),
        }
    }

    /// The newest line at `min` or more severe.
    #[must_use]
    pub fn latest(&self, min: Level) -> Option<LogLine> {
        self.lock()
            .lines
            .iter()
            .rev()
            .find(|line| line.level <= min)
            .cloned()
    }

    /// A tracing layer pushing every event it sees into this buffer.
    #[must_use]
    pub fn layer(&self) -> BufferLayer {
        BufferLayer {
            buffer: self.clone(),
        }
    }
}

/// Tracing layer of a [`LogBuffer`], see [`LogBuffer::layer`].
#[derive(Debug)]
pub struct BufferLayer {
    buffer: LogBuffer,
}

impl<S: Subscriber> Layer<S> for BufferLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut fields = Fields::default();
        event.record(&mut fields);
        let metadata = event.metadata();
        self.buffer.push(LogLine {
            seq: 0,
            level: *metadata.level(),
            target: metadata.target().to_string(),
            time: SystemTime::now(),
            message: fields.message + &fields.rest,
        });
    }
}

/// An event's fields: the message, then the others as ` key=value`.
#[derive(Default)]
struct Fields {
    message: String,
    rest: String,
}

impl Visit for Fields {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message.push_str(value);
        } else {
            write!(self.rest, " {}={value}", field.name()).ok();
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if field.name() == "message" {
            write!(self.message, "{value:?}").ok();
        } else {
            write!(self.rest, " {}={value:?}", field.name()).ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use tracing_subscriber::layer::SubscriberExt;

    use super::*;

    fn line(level: Level, message: &str) -> LogLine {
        LogLine {
            seq: 0,
            level,
            target: "overbrainer::test".to_string(),
            time: SystemTime::UNIX_EPOCH,
            message: message.to_string(),
        }
    }

    fn messages(lines: &[LogLine]) -> Vec<&str> {
        lines.iter().map(|line| line.message.as_str()).collect()
    }

    #[test]
    fn the_buffer_keeps_the_newest_lines_up_to_its_capacity() {
        let buffer = LogBuffer::new(3);
        for n in 0..5 {
            buffer.push(line(Level::INFO, &n.to_string()));
        }
        assert_eq!(buffer.seq(), 5);
        let window = buffer.window(Level::TRACE, 10, 0);
        assert_eq!(messages(&window.lines), ["2", "3", "4"]);
        assert_eq!(window.matching, 3);
        assert_eq!(window.lines[0].seq, 3);
    }

    #[test]
    fn a_window_filters_by_level_and_counts_back_from_the_newest() {
        let buffer = LogBuffer::new(10);
        buffer.push(line(Level::ERROR, "e1"));
        buffer.push(line(Level::INFO, "i1"));
        buffer.push(line(Level::WARN, "w1"));
        buffer.push(line(Level::DEBUG, "d1"));
        buffer.push(line(Level::ERROR, "e2"));
        let warn = buffer.window(Level::WARN, 10, 0);
        assert_eq!(messages(&warn.lines), ["e1", "w1", "e2"]);
        assert_eq!(warn.matching, 3);
        assert_eq!(
            messages(&buffer.window(Level::TRACE, 2, 0).lines),
            ["d1", "e2"]
        );
        assert_eq!(
            messages(&buffer.window(Level::TRACE, 2, 1).lines),
            ["w1", "d1"]
        );
        assert_eq!(
            messages(&buffer.window(Level::TRACE, 2, 9).lines),
            ["e1", "i1"]
        );
        assert_eq!(
            buffer.latest(Level::WARN).map(|line| line.message),
            Some("e2".to_string())
        );
    }

    #[test]
    fn the_layer_records_level_target_message_and_fields() {
        let buffer = LogBuffer::new(10);
        let subscriber = tracing_subscriber::registry().with(buffer.layer());
        tracing::subscriber::with_default(subscriber, || {
            tracing::warn!(attempt = 2, "retrying {}", "answers");
            tracing::info!(target: "elsewhere", "plain");
        });
        let window = buffer.window(Level::TRACE, 10, 0);
        assert_eq!(
            messages(&window.lines),
            ["retrying answers attempt=2", "plain"]
        );
        assert_eq!(window.lines[0].level, Level::WARN);
        assert_eq!(window.lines[0].target, "overbrainer::logging::tests");
        assert_eq!(window.lines[1].target, "elsewhere");
    }
}
