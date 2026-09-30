//! The pod's own logs: Runpod's log stream of a pod (`GET /pods/{id}/logs`, an
//! event stream of its container and system lines), read while the pod exists
//! and kept in the run directory, so they outlive the pod.
//!
//! The stream never ends by itself and has no end-of-replay marker: a
//! [`snapshot`] ends after a silence, a [`follow`] reconnects with
//! `Last-Event-ID` (exclusive, so no line comes twice). Every line is redacted
//! before anyone sees it: see [`crate::secrets::redact_line`].
//!
//! The capture starts once overbrainer follows the job; its first connection
//! replays the last 5000 lines, which covers the pod's start (image pull,
//! bootstrap). A pod deleted before its job started (a failed start) has its
//! log read by the [`drain`] that runs before every delete.

mod kept;
mod sse;

use std::future::Future;
use std::io;
use std::time::Duration;

use serde::{Deserialize, Serialize};

pub use self::kept::{
    Capture, keep_file, kept_cursor, kept_lines, open_kept, parse_kept_line, read_kept,
    read_kept_from,
};
use self::sse::{Frame, SseReader};
use super::client::{ApiError, RunpodClient};
use super::provision::PodCtx;
use super::record::{PodRecord, PodState};
use super::types::PodId;
use crate::runs::Runs;
use crate::secrets::Redactor;

/// The pod's logs, in a run directory: JSON lines `{ts, source, line}`.
pub const POD_LOG: &str = ".pod/pod.log";

/// The last event ID of the stream kept in [`POD_LOG`], in a run directory.
pub const POD_LOG_CURSOR: &str = ".pod/pod.log.cursor";

/// The bootstrap's log on the pod, in the run directory on the pod.
pub const BOOTSTRAP_LOG: &str = ".pod/bootstrap.log";

/// Largest [`POD_LOG`] kept, in bytes: past it, one line says so and the
/// capture stops.
pub const POD_LOG_CAP: u64 = 20 * 1024 * 1024;

/// Most lines Runpod replays (`tail`); more is refused.
pub const TAIL_MAX: u32 = 5000;

/// How long the last read before a delete waits for the lines not read yet.
pub const DRAIN_WAIT: Duration = Duration::from_secs(5);

/// The source of the line saying the kept log was capped.
const CAPPED_SOURCE: &str = "overbrainer";

/// A silence this long ends a snapshot: the replay comes in one burst.
const SILENCE: Duration = Duration::from_millis(750);

/// First and longest wait before reconnecting.
const BACKOFF_MIN: Duration = Duration::from_millis(500);
const BACKOFF_MAX: Duration = Duration::from_secs(15);

/// Longest wait a `Retry-After` gets.
const RETRY_AFTER_MAX: Duration = Duration::from_secs(60);

/// Longest cursor kept; Runpod's are about 35 characters.
const MAX_CURSOR: usize = 200;

/// Where a line of a pod's log comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum LogSource {
    /// The container's own output.
    Container,
    /// Runpod's account of the pod: image pull, container start.
    System,
}

impl LogSource {
    /// Its name in the API.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Container => "container",
            Self::System => "system",
        }
    }
}

/// One line of a pod's log, as the stream gives it and [`POD_LOG`] keeps it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PodLogLine {
    /// When the pod wrote it (RFC 3339), empty when unknown.
    pub ts: String,
    /// `container`, `system`, `raw` for an event that was not the documented
    /// JSON object (kept whole), or `overbrainer` for the line saying the kept
    /// log was capped.
    pub source: String,
    /// The line, redacted.
    pub line: String,
}

impl PodLogLine {
    /// The source in three letters: `ctr`, `sys`, `raw`, or `obr` for
    /// overbrainer's own line.
    #[must_use]
    pub fn short_source(&self) -> &'static str {
        match self.source.as_str() {
            "container" => "ctr",
            "system" => "sys",
            CAPPED_SOURCE => "obr",
            _ => "raw",
        }
    }

    /// Whether the line comes from `source`.
    #[must_use]
    pub fn is_from(&self, source: LogSource) -> bool {
        self.source == source.name()
    }

    /// This line with only what can be shown safely around the line itself:
    /// a `ts` that is not an RFC 3339 time becomes empty, an unknown source
    /// becomes `raw`.
    #[must_use]
    pub fn cleaned(mut self) -> Self {
        if !valid_ts(&self.ts) {
            self.ts = String::new();
        }
        if !matches!(
            self.source.as_str(),
            "container" | "system" | "raw" | CAPPED_SOURCE
        ) {
            self.source = "raw".to_string();
        }
        self
    }

    /// The line for a terminal or a text file: `<ts> <sys|ctr> <line>`, the
    /// line redacted again and on one line (see [`one_line`]).
    #[must_use]
    pub fn display(&self) -> String {
        let clean = self.clone().cleaned();
        let line = crate::secrets::redact_line(&clean.line, &[]);
        format!("{} {} {}", clean.ts, clean.short_source(), one_line(&line))
    }
}

/// Whether `ts` has the shape of an RFC 3339 time: `YYYY-MM-DDTHH:MM:SS`,
/// then fractions and a zone.
fn valid_ts(ts: &str) -> bool {
    let bytes = ts.as_bytes();
    if bytes.len() < 20 || bytes.len() > 40 {
        return false;
    }
    let shape = b"0000-00-00T00:00:00";
    let head = bytes[..shape.len()]
        .iter()
        .zip(shape)
        .all(|(byte, want)| match want {
            b'0' => byte.is_ascii_digit(),
            other => byte == other,
        });
    head && bytes[shape.len()..]
        .iter()
        .all(|byte| byte.is_ascii_digit() || matches!(byte, b'.' | b':' | b'+' | b'-' | b'Z'))
}

/// `text` on one line: a newline and a carriage return are written `\n` and
/// `\r`, other control characters but tab, and the bidirectional overrides
/// and isolates, `\u{..}`.
#[must_use]
pub fn one_line(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push('\t'),
            '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}' => {
                escaped.extend(c.escape_unicode());
            },
            c if c.is_control() => escaped.extend(c.escape_unicode()),
            c => escaped.push(c),
        }
    }
    escaped
}

/// What to ask of the log stream.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LogQuery {
    /// One source only, or both.
    pub source: Option<LogSource>,
    /// Lines to replay (at most [`TAIL_MAX`]); Runpod's default is 100.
    pub tail: Option<u32>,
    /// Lines from this time on (RFC 3339); wins over `tail`.
    pub since: Option<String>,
    /// Resume after this event ID; wins over `since` and `tail`.
    pub cursor: Option<String>,
}

/// Why reading a pod's log stopped.
#[derive(Debug, thiserror::Error)]
pub enum LogError {
    /// The API refused for good (400, 401, 403, 404, 422), or the client
    /// cannot be used.
    #[error(transparent)]
    Api(#[from] ApiError),
    /// The lines could not be written.
    #[error("cannot write the pod's log lines")]
    Sink(#[source] io::Error),
}

/// Whether `error` would come back the same on every reconnect.
fn permanent(error: &ApiError) -> bool {
    match error {
        ApiError::Status { status, .. } => matches!(status, 400 | 401 | 403 | 404 | 422),
        ApiError::Client(_) | ApiError::InvalidApiKey => true,
        ApiError::Transport(_) | ApiError::InvalidResponse(_) => false,
    }
}

/// The line an event's data holds: the documented JSON object (cleaned, see
/// [`PodLogLine::cleaned`]), or the data itself under the `raw` source.
/// `None` for an empty event (a keepalive).
fn parse_payload(data: &str, truncated: bool) -> Option<PodLogLine> {
    if data.trim().is_empty() {
        return None;
    }
    let mut line = match serde_json::from_str::<PodLogLine>(data) {
        Ok(line) if line != PodLogLine::default() => line.cleaned(),
        _ => PodLogLine {
            ts: String::new(),
            source: "raw".to_string(),
            line: data.to_string(),
        },
    };
    if truncated {
        line.line.push_str(" [truncated]");
    }
    Some(line)
}

/// A cursor safe to keep and send back: printable ASCII, not too long.
fn valid_cursor(cursor: &str) -> bool {
    !cursor.is_empty() && cursor.len() <= MAX_CURSOR && cursor.bytes().all(|b| b.is_ascii_graphic())
}

/// What a connection of the stream ended with.
enum End {
    /// The server closed it.
    Closed,
    /// A snapshot's silence passed.
    Quiet,
}

/// One reader of a pod's log stream, over as many connections as it takes.
struct Stream<'a, S> {
    client: &'a RunpodClient,
    id: &'a PodId,
    query: LogQuery,
    redactor: Redactor,
    sink: &'a mut S,
    /// Whether an event arrived on the last connection.
    delivered: bool,
    /// Whether a refused cursor was already given up for a time.
    fell_back: bool,
}

impl<'a, S> Stream<'a, S>
where
    S: FnMut(&[PodLogLine], Option<&str>) -> io::Result<()>,
{
    fn new(client: &'a RunpodClient, id: &'a PodId, query: LogQuery, sink: &'a mut S) -> Self {
        Self {
            client,
            id,
            query,
            redactor: Redactor::new(client.log_secrets()),
            sink,
            delivered: false,
            fell_back: false,
        }
    }

    /// Reads one connection until it ends, or, with `silence`, until no event
    /// arrived for that long after the first one.
    async fn pump(&mut self, silence: Option<Duration>) -> Result<End, LogError> {
        self.delivered = false;
        let mut response = self.client.open_pod_logs(self.id, &self.query).await?;
        let mut reader = SseReader::default();
        loop {
            let next = match silence {
                Some(quiet) if self.delivered => {
                    match tokio::time::timeout(quiet, response.chunk()).await {
                        Ok(next) => next,
                        Err(_) => return Ok(End::Quiet),
                    }
                },
                _ => response.chunk().await,
            };
            let Some(chunk) = next.map_err(ApiError::Transport)? else {
                let frames = reader.finish();
                self.deliver(frames)?;
                return Ok(End::Closed);
            };
            let frames = reader.feed(&chunk);
            self.deliver(frames)?;
        }
    }

    /// Hands the lines of `frames` to the sink, redacted, with the cursor they
    /// end at, then moves the cursor there.
    fn deliver(&mut self, frames: Vec<Frame>) -> Result<(), LogError> {
        if frames.is_empty() {
            return Ok(());
        }
        let mut cursor = None;
        let mut lines = Vec::new();
        for frame in frames {
            if let Some(mut line) = frame
                .data
                .as_deref()
                .and_then(|data| parse_payload(data, frame.truncated))
            {
                line.line = self.redactor.line(&line.line);
                lines.push(line);
            }
            if let Some(id) = frame.id.filter(|id| valid_cursor(id)) {
                cursor = Some(id);
            }
        }
        self.delivered = true;
        (self.sink)(&lines, cursor.as_deref()).map_err(LogError::Sink)?;
        if cursor.is_some() {
            self.query.cursor = cursor;
        }
        Ok(())
    }

    /// After `error`, whether to try once more without the cursor: Runpod
    /// refuses a cursor it no longer knows with a 400 or a 422. The stream
    /// then starts at the cursor's own time (a few lines may come twice), or
    /// with the longest replay when the cursor holds no time.
    fn fall_back(&mut self, error: &LogError) -> bool {
        let LogError::Api(api) = error else {
            return false;
        };
        if self.fell_back || !matches!(api.status(), Some(400 | 422)) {
            return false;
        }
        let Some(cursor) = self.query.cursor.take() else {
            return false;
        };
        self.fell_back = true;
        self.query.since = cursor
            .split_once('/')
            .map(|(ts, _)| ts)
            .filter(|ts| valid_ts(ts))
            .map(str::to_string);
        if self.query.since.is_none() {
            self.query.tail = Some(TAIL_MAX);
        }
        tracing::debug!("the pod log stream refused its cursor ({api}); starting again");
        true
    }
}

/// Reads what the log stream of the pod `id` gives for `query` and stops once
/// it has been silent for 750 ms after its first event, once the server closes
/// it, or after `max_wait` in all (Runpod sends nothing, headers included,
/// until it has a line: a pod with nothing new ends there). `sink` gets each
/// batch of lines, redacted, with the event ID they end at. Returns the last
/// event ID seen, or `query`'s own.
///
/// # Errors
///
/// Returns [`LogError::Api`] when the API refuses or cannot be reached, and
/// [`LogError::Sink`] when `sink` fails.
pub async fn snapshot<S>(
    client: &RunpodClient,
    id: &PodId,
    query: LogQuery,
    max_wait: Duration,
    sink: &mut S,
) -> Result<Option<String>, LogError>
where
    S: FnMut(&[PodLogLine], Option<&str>) -> io::Result<()>,
{
    let mut stream = Stream::new(client, id, query, sink);
    let read = async {
        loop {
            match stream.pump(Some(SILENCE)).await {
                Err(error) if stream.fall_back(&error) => {},
                ended => return ended,
            }
        }
    };
    match tokio::time::timeout(max_wait, read).await {
        Ok(Err(error)) => Err(error),
        Ok(Ok(End::Closed | End::Quiet)) | Err(_) => Ok(stream.query.cursor),
    }
}

/// The waits between two connections of [`follow`].
#[derive(Debug)]
struct Backoff {
    next: Duration,
}

impl Backoff {
    fn new() -> Self {
        Self { next: BACKOFF_MIN }
    }

    /// The wait after a connection that `delivered` events or not and ended
    /// with `error`, if any: 0.5 s after a connection that delivered, then
    /// doubling up to 15 s, or a longer `Retry-After` (up to a minute).
    fn wait(&mut self, delivered: bool, error: Option<&ApiError>) -> Duration {
        if delivered {
            self.next = BACKOFF_MIN;
        }
        let wait = match error {
            Some(ApiError::Status {
                retry_after: Some(after),
                ..
            }) => (*after).min(RETRY_AFTER_MAX).max(self.next),
            _ => self.next,
        };
        self.next = (self.next * 2).min(BACKOFF_MAX);
        wait
    }
}

/// Follows the log stream of the pod `id` for `query`, forever: `sink` gets
/// each batch of lines as [`snapshot`] says. A dropped connection is opened
/// again from the last event ID, after 0.5 s doubling up to 15 s (reset once
/// an event arrives), or Runpod's `Retry-After`. Returns only why it stopped:
/// a refusal that would come back on every reconnect (400, 401, 403, 404, 422;
/// a 404 once the pod is deleted), or a failed `sink`.
pub async fn follow<S>(client: &RunpodClient, id: &PodId, query: LogQuery, sink: &mut S) -> LogError
where
    S: FnMut(&[PodLogLine], Option<&str>) -> io::Result<()>,
{
    let mut stream = Stream::new(client, id, query, sink);
    let mut backoff = Backoff::new();
    loop {
        let error = match stream.pump(None).await {
            Ok(End::Closed | End::Quiet) => None,
            Err(error) if stream.fall_back(&error) => continue,
            Err(LogError::Api(error)) if !permanent(&error) => Some(error),
            Err(error) => return error,
        };
        let wait = backoff.wait(stream.delivered, error.as_ref());
        let why = error
            .as_ref()
            .map_or_else(|| "closed".to_string(), ToString::to_string);
        tracing::debug!("the log stream of pod {id} ended ({why}); reconnecting");
        tokio::time::sleep(wait).await;
    }
}

/// Opens the capture of the run `run_id`, or says why not.
fn open_capture(runs: &Runs, run_id: &str) -> Option<Capture> {
    let dir = runs.run_dir(run_id).ok()?;
    match Capture::open(&dir) {
        Ok(capture) if capture.capped() => None,
        Ok(capture) => Some(capture),
        Err(error) => {
            cannot_keep(run_id, &error);
            None
        },
    }
}

/// Keeps the log of the pod `pod_id` in the directory of the run `run_id`
/// while the pod exists: follows its stream from the kept cursor (or the
/// longest replay), until the pod is gone, the cap is reached or the API
/// refuses. Best effort: failures are logged, never returned.
pub async fn capture(client: &RunpodClient, runs: &Runs, run_id: &str, pod_id: &PodId) {
    let Some(mut capture) = open_capture(runs, run_id) else {
        return;
    };
    let query = capture.query();
    let mut sink = |lines: &[PodLogLine], cursor: Option<&str>| {
        capture.take(lines, cursor)?;
        if capture.capped() {
            return Err(io::Error::other("capped"));
        }
        Ok(())
    };
    let error = follow(client, pod_id, query, &mut sink).await;
    if capture.capped() {
        tracing::debug!("the pod's log of run {run_id} reached its cap");
    } else {
        note_end(run_id, &error);
    }
    if let Err(error) = capture.finish() {
        cannot_keep(run_id, &error);
    }
}

/// The last read of the log of the pod `pod_id` into the directory of the run
/// `run_id`, before the pod is deleted: what the stream gives from the kept
/// cursor within [`DRAIN_WAIT`]. Best effort.
pub async fn drain(client: &RunpodClient, runs: &Runs, run_id: &str, pod_id: &PodId) {
    let Some(mut capture) = open_capture(runs, run_id) else {
        return;
    };
    let query = capture.query();
    let mut sink = |lines: &[PodLogLine], cursor: Option<&str>| capture.take(lines, cursor);
    if let Err(error) = snapshot(client, pod_id, query, DRAIN_WAIT, &mut sink).await {
        note_end(run_id, &error);
    }
    if let Err(error) = capture.finish() {
        cannot_keep(run_id, &error);
    }
}

/// Says, at debug level unless a write failed, why the pod's log of the run
/// `run_id` is no longer kept.
fn note_end(run_id: &str, error: &LogError) {
    match error {
        LogError::Sink(source) => cannot_keep(run_id, source),
        LogError::Api(_) => {
            tracing::debug!("the pod's log of run {run_id} is no longer read: {error}");
        },
    }
}

/// Warns that the pod's log of the run `run_id` cannot be kept.
fn cannot_keep(run_id: &str, error: &io::Error) {
    tracing::warn!("cannot keep the pod's log of run {run_id}: {error}");
}

/// Runs `work` while the log of the pod of `pod` is kept in its run
/// directory; the capture never ends `work` early. Only `work` when no pod is
/// recorded or it is deleted.
pub async fn with_pod_logs<F: Future>(ctx: &PodCtx<'_>, pod: &PodRecord, work: F) -> F::Output {
    let Some(pod_id) = pod
        .pod_id
        .as_ref()
        .filter(|_| pod.state != PodState::Deleted)
    else {
        return work.await;
    };
    let kept = async {
        capture(ctx.client, ctx.runs, &pod.run_id, pod_id).await;
        std::future::pending::<F::Output>().await
    };
    tokio::select! {
        output = work => output,
        output = Box::pin(kept) => output,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payloads_are_parsed_cleaned_or_kept_raw() {
        assert_eq!(
            parse_payload(
                r#"{"ts":"2026-06-01T12:02:03.5Z","source":"container","line":"hi"}"#,
                false
            ),
            Some(PodLogLine {
                ts: "2026-06-01T12:02:03.5Z".into(),
                source: "container".into(),
                line: "hi".into()
            })
        );
        assert_eq!(
            parse_payload(
                "{\"ts\":\"\\u001b[2Jnow\",\"source\":\"\\u001b]0;x\",\"line\":\"hi\"}",
                false
            ),
            Some(PodLogLine {
                ts: String::new(),
                source: "raw".into(),
                line: "hi".into()
            })
        );
        assert_eq!(
            parse_payload("not json", true),
            Some(PodLogLine {
                ts: String::new(),
                source: "raw".into(),
                line: "not json [truncated]".into()
            })
        );
        assert_eq!(
            parse_payload("null", false).map(|line| line.source),
            Some("raw".into())
        );
        assert_eq!(parse_payload("  ", false), None);
    }

    #[test]
    fn cursors_must_be_printable() {
        assert!(valid_cursor("2026-06-01T12:02:03Z/000000000001"));
        assert!(!valid_cursor(""));
        assert!(!valid_cursor("a b"));
        assert!(!valid_cursor("a\nb"));
        assert!(!valid_cursor(&"a".repeat(MAX_CURSOR + 1)));
    }

    #[test]
    fn times_must_look_like_rfc_3339() {
        assert!(valid_ts("2026-06-01T12:02:03Z"));
        assert!(valid_ts("2026-06-01T12:02:03.123456+02:00"));
        assert!(!valid_ts("2026-06-01 12:02:03Z"));
        assert!(!valid_ts("2026-06-01T12:02:03Z\u{1b}[0m"));
        assert!(!valid_ts(""));
    }

    #[test]
    fn a_line_is_shown_redacted_on_one_line_without_terminal_controls() {
        let line = PodLogLine {
            ts: "2026-06-01T12:02:03Z".into(),
            source: "system".into(),
            line: "a\r\nb\tc\u{1b}[0m \u{202E}evil HF_TOKEN=x".into(),
        };
        assert_eq!(
            line.display(),
            "2026-06-01T12:02:03Z sys a\\r\\nb\tc\\u{1b}[0m \\u{202e}evil HF_TOKEN=***"
        );
        let forged = PodLogLine {
            ts: "\u{1b}[31m".into(),
            source: "\u{1b}[2J".into(),
            line: "x".into(),
        };
        assert_eq!(forged.display(), " raw x");
    }

    #[test]
    fn short_sources() {
        let line = |source: &str| PodLogLine {
            source: source.into(),
            ..PodLogLine::default()
        };
        assert_eq!(line("container").short_source(), "ctr");
        assert_eq!(line("system").short_source(), "sys");
        assert_eq!(line("raw").short_source(), "raw");
        assert_eq!(line("other").short_source(), "raw");
        assert!(line("system").is_from(LogSource::System));
    }

    fn unavailable(retry_after: Option<Duration>) -> ApiError {
        ApiError::Status {
            status: 503,
            message: String::new(),
            retry_after,
            capacity: false,
        }
    }

    #[test]
    fn the_backoff_grows_until_a_connection_delivers() {
        let mut backoff = Backoff::new();
        let error = unavailable(None);
        let waits: Vec<f64> = (0..7)
            .map(|_| backoff.wait(false, Some(&error)).as_secs_f64())
            .collect();
        assert_eq!(waits, [0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 15.0]);
        assert_eq!(backoff.wait(true, None), BACKOFF_MIN);
        assert_eq!(backoff.wait(false, Some(&error)), Duration::from_secs(1));
    }

    #[test]
    fn the_backoff_honours_retry_after_within_a_minute() {
        let mut backoff = Backoff::new();
        let later = unavailable(Some(Duration::from_secs(5)));
        assert_eq!(backoff.wait(false, Some(&later)), Duration::from_secs(5));
        let far = unavailable(Some(Duration::from_secs(3600)));
        assert_eq!(backoff.wait(false, Some(&far)), RETRY_AFTER_MAX);
        // Shorter than the backoff: the backoff.
        let soon = unavailable(Some(Duration::from_millis(1)));
        assert_eq!(backoff.wait(false, Some(&soon)), Duration::from_secs(2));
    }
}
