//! The pod's own logs: Runpod's log stream of a pod (`GET /pods/{id}/logs`, an
//! event stream of its container and system lines), read while the pod exists
//! and kept in the run directory, so they outlive the pod.
//!
//! The stream never ends by itself and has no end-of-replay marker: a
//! [`snapshot`] ends after a silence, a [`follow`] reconnects with
//! `Last-Event-ID` (exclusive, so no line comes twice). Every line is redacted
//! before anyone sees it: see [`crate::secrets::redact_line`].

use std::fs::{self, File, OpenOptions};
use std::future::Future;
use std::io::{self, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use secrecy::SecretString;
use serde::{Deserialize, Serialize};

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

/// Room kept under the cap for the line saying the log was capped.
const CAP_RESERVE: u64 = 512;

/// A silence this long ends a snapshot: the replay comes in one burst.
const SILENCE: Duration = Duration::from_millis(750);

/// Bytes kept of one line of the stream; the rest of the line is dropped.
const MAX_LINE: usize = 1 << 20;

/// Bytes kept of one event's data; the rest is dropped.
const MAX_FRAME: usize = 4 << 20;

/// First and longest wait before reconnecting.
const BACKOFF_MIN: Duration = Duration::from_millis(500);
const BACKOFF_MAX: Duration = Duration::from_secs(15);

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
    /// `container`, `system`, or `raw` for an event that was not the documented
    /// JSON object, kept whole.
    pub source: String,
    /// The line, redacted.
    pub line: String,
}

impl PodLogLine {
    /// The source in three letters: `ctr`, `sys`, or the source itself.
    #[must_use]
    pub fn short_source(&self) -> &str {
        match self.source.as_str() {
            "container" => "ctr",
            "system" => "sys",
            other => other,
        }
    }

    /// Whether the line comes from `source`.
    #[must_use]
    pub fn is_from(&self, source: LogSource) -> bool {
        self.source == source.name()
    }
}

impl PodLogLine {
    /// The line for a terminal or a text file: `<ts> <sys|ctr> <line>`, the
    /// line redacted again and on one line (a newline or a carriage return is
    /// written `\n` or `\r`, another control character `\u{..}`).
    #[must_use]
    pub fn display(&self) -> String {
        let line = crate::secrets::redact_line(&self.line, &[]);
        format!("{} {} {}", self.ts, self.short_source(), one_line(&line))
    }
}

/// `text` on one line: a newline and a carriage return are written `\n` and
/// `\r`, other control characters but tab `\u{..}`.
#[must_use]
pub fn one_line(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push('\t'),
            c if c.is_control() => escaped.extend(c.escape_unicode()),
            c => escaped.push(c),
        }
    }
    escaped
}

/// The event ID the log kept in the run directory `run_dir` ends at, if any.
#[must_use]
pub fn kept_cursor(run_dir: &Path) -> Option<String> {
    fs::read_to_string(run_dir.join(POD_LOG_CURSOR))
        .ok()
        .map(|text| text.trim().to_string())
        .filter(|cursor| valid_cursor(cursor))
}

/// The lines of the log kept in the run directory `run_dir`, oldest first;
/// a line that cannot be read is skipped. Empty when none is kept.
///
/// # Errors
///
/// Returns the I/O error of reading an existing log.
pub fn kept_lines(run_dir: &Path) -> io::Result<Vec<PodLogLine>> {
    let text = match fs::read(run_dir.join(POD_LOG)) {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };
    Ok(text
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect())
}

/// What to ask of the log stream.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LogQuery {
    /// One source only, or both.
    pub source: Option<LogSource>,
    /// Lines to replay (at most [`TAIL_MAX`]); Runpod's default is 100.
    pub tail: Option<u32>,
    /// Resume after this event ID; wins over `tail`.
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

/// One event of the stream.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Frame {
    id: Option<String>,
    data: Option<String>,
    truncated: bool,
}

/// A reader of `text/event-stream` bytes: only `data:` and `id:` are read,
/// comments and other fields are skipped, `\r\n` is taken for `\n`. A line and
/// an event are capped ([`MAX_LINE`], [`MAX_FRAME`]), since the stream never ends.
#[derive(Debug, Default)]
struct SseReader {
    pending: Vec<u8>,
    skipping: bool,
    data: String,
    has_data: bool,
    id: Option<String>,
    truncated: bool,
}

impl SseReader {
    /// Reads `chunk`, returning the events it completes.
    fn feed(&mut self, chunk: &[u8]) -> Vec<Frame> {
        let mut frames = Vec::new();
        let mut rest = chunk;
        while let Some(at) = rest.iter().position(|byte| *byte == b'\n') {
            self.append(&rest[..at]);
            self.line(&mut frames);
            rest = &rest[at + 1..];
        }
        self.append(rest);
        frames
    }

    /// Ends the stream: a last line without newline and a last event without
    /// blank line still count.
    fn finish(&mut self) -> Vec<Frame> {
        let mut frames = Vec::new();
        if !self.pending.is_empty() {
            self.line(&mut frames);
        }
        self.flush(&mut frames);
        frames
    }

    fn append(&mut self, piece: &[u8]) {
        if self.skipping {
            return;
        }
        let room = MAX_LINE.saturating_sub(self.pending.len());
        if piece.len() > room {
            self.pending.extend_from_slice(&piece[..room]);
            self.skipping = true;
            self.truncated = true;
        } else {
            self.pending.extend_from_slice(piece);
        }
    }

    fn line(&mut self, frames: &mut Vec<Frame>) {
        let bytes = std::mem::take(&mut self.pending);
        self.skipping = false;
        let text = String::from_utf8_lossy(&bytes);
        let text = text.strip_suffix('\r').unwrap_or(&text);
        if text.is_empty() {
            self.flush(frames);
        } else if let Some(value) = field(text, "data") {
            if self.data.len() + value.len() + 1 > MAX_FRAME {
                self.truncated = true;
            } else {
                if self.has_data {
                    self.data.push('\n');
                }
                self.data.push_str(value);
                self.has_data = true;
            }
        } else if let Some(value) = field(text, "id") {
            self.id = Some(value.to_string());
        }
    }

    fn flush(&mut self, frames: &mut Vec<Frame>) {
        let data = std::mem::take(&mut self.data);
        let has_data = std::mem::take(&mut self.has_data);
        let id = self.id.take();
        let truncated = std::mem::take(&mut self.truncated);
        if id.is_none() && !has_data {
            return;
        }
        frames.push(Frame {
            id,
            data: has_data.then_some(data),
            truncated,
        });
    }
}

/// The value of the field `name` on `line`, one space after the colon dropped.
fn field<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let value = line.strip_prefix(name)?.strip_prefix(':')?;
    Some(value.strip_prefix(' ').unwrap_or(value))
}

/// The line an event's data holds: the documented JSON object, or the data
/// itself under the `raw` source. `None` for an empty event (a keepalive).
fn parse_payload(data: &str, truncated: bool) -> Option<PodLogLine> {
    if data.trim().is_empty() {
        return None;
    }
    let mut line = match serde_json::from_str::<PodLogLine>(data) {
        Ok(line) if line != PodLogLine::default() => line,
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
    /// Whether an event arrived since the last connect.
    delivered: bool,
}

impl<'a, S> Stream<'a, S>
where
    S: FnMut(&[PodLogLine], Option<&str>) -> io::Result<()>,
{
    fn new(client: &'a RunpodClient, id: &'a PodId, query: LogQuery, sink: &'a mut S) -> Self {
        let known: Vec<SecretString> = vec![client.api_key().clone()];
        Self {
            client,
            id,
            query,
            redactor: Redactor::new(known),
            sink,
            delivered: false,
        }
    }

    /// Reads one connection until it ends, or, with `silence`, until no event
    /// arrived for that long after the first one.
    async fn pump(&mut self, silence: Option<Duration>) -> Result<End, LogError> {
        let mut response = self.client.open_pod_logs(self.id, &self.query).await?;
        self.delivered = false;
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
    match tokio::time::timeout(max_wait, stream.pump(Some(SILENCE))).await {
        Ok(Err(error)) => Err(error),
        Ok(Ok(End::Closed | End::Quiet)) | Err(_) => Ok(stream.query.cursor),
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
    let mut backoff = BACKOFF_MIN;
    loop {
        let ended = stream.pump(None).await;
        if stream.delivered {
            backoff = BACKOFF_MIN;
        }
        let wait = match reconnect_wait(ended, backoff) {
            Ok(wait) => wait,
            Err(error) => return error,
        };
        tracing::debug!("the log stream of pod {id} ended; reconnecting");
        tokio::time::sleep(wait).await;
        backoff = (backoff * 2).min(BACKOFF_MAX);
    }
}

/// How long to wait before reconnecting after a connection `ended`, or the
/// error that ends the follow.
fn reconnect_wait(ended: Result<End, LogError>, backoff: Duration) -> Result<Duration, LogError> {
    match ended {
        Ok(End::Closed | End::Quiet) => Ok(backoff),
        Err(LogError::Api(error)) if !permanent(&error) => {
            tracing::debug!("the pod log stream failed: {error}");
            Ok(error_wait(&error, backoff))
        },
        Err(error) => Err(error),
    }
}

/// How long to wait after `error`: its `Retry-After` when longer than `backoff`.
fn error_wait(error: &ApiError, backoff: Duration) -> Duration {
    match error {
        ApiError::Status {
            retry_after: Some(after),
            ..
        } => (*after).max(backoff).min(BACKOFF_MAX * 4),
        _ => backoff,
    }
}

/// The pod's log kept in a run directory ([`POD_LOG`], mode 0600) and its
/// cursor ([`POD_LOG_CURSOR`]). Past [`POD_LOG_CAP`], one line says the log
/// was capped and nothing more is kept.
#[derive(Debug)]
pub struct Capture {
    log: PathBuf,
    cursor_path: PathBuf,
    file: File,
    size: u64,
    cap: u64,
    capped: bool,
    cursor: Option<String>,
}

impl Capture {
    /// Opens the kept log of the run directory `run_dir`, created when absent,
    /// with its cursor.
    ///
    /// # Errors
    ///
    /// Returns the I/O error of creating `.pod/` or opening the log.
    pub fn open(run_dir: &Path) -> io::Result<Self> {
        let log = run_dir.join(POD_LOG);
        if let Some(dir) = log.parent() {
            fs::create_dir_all(dir)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .open(&log)?;
        let size = file.metadata()?.len();
        let cursor_path = run_dir.join(POD_LOG_CURSOR);
        let cursor = kept_cursor(run_dir);
        let mut capture = Self {
            log,
            cursor_path,
            file,
            size,
            cap: POD_LOG_CAP,
            capped: false,
            cursor,
        };
        capture.capped = capture.full(0) || ends_capped(&capture.file, size);
        Ok(capture)
    }

    /// This capture with another cap, in bytes (tests use small ones).
    #[must_use]
    pub fn with_cap(mut self, cap: u64) -> Self {
        self.cap = cap;
        self.capped = self.full(0);
        self
    }

    /// The event ID the kept log ends at.
    #[must_use]
    pub fn cursor(&self) -> Option<&str> {
        self.cursor.as_deref()
    }

    /// Whether the cap was reached: nothing more is kept.
    #[must_use]
    pub fn capped(&self) -> bool {
        self.capped
    }

    /// Where the lines of the log stream start for this capture: after its
    /// cursor, or the longest replay without one.
    #[must_use]
    pub fn query(&self) -> LogQuery {
        LogQuery {
            source: None,
            tail: self.cursor.is_none().then_some(TAIL_MAX),
            cursor: self.cursor.clone(),
        }
    }

    fn full(&self, adding: u64) -> bool {
        self.size + adding > self.cap.saturating_sub(CAP_RESERVE)
    }

    /// Appends `lines`, then keeps `cursor`. Reaching the cap writes one line
    /// saying so instead of the rest.
    ///
    /// # Errors
    ///
    /// Returns the I/O error of a write.
    pub fn take(&mut self, lines: &[PodLogLine], cursor: Option<&str>) -> io::Result<()> {
        if self.capped {
            return Ok(());
        }
        let mut text = String::new();
        for line in lines {
            let json = serde_json::to_string(line).map_err(io::Error::other)?;
            let adding = u64::try_from(text.len() + json.len() + 1).unwrap_or(u64::MAX);
            if self.full(adding) {
                self.capped = true;
                let marker = PodLogLine {
                    ts: crate::runs::rfc3339(std::time::SystemTime::now()),
                    source: CAPPED_SOURCE.to_string(),
                    line: format!(
                        "the pod's log reached {} MiB: later lines are not kept",
                        self.cap / (1024 * 1024)
                    ),
                };
                text.push_str(&serde_json::to_string(&marker).map_err(io::Error::other)?);
                text.push('\n');
                break;
            }
            text.push_str(&json);
            text.push('\n');
        }
        self.file.write_all(text.as_bytes())?;
        self.size += u64::try_from(text.len()).unwrap_or(u64::MAX);
        if let Some(cursor) = cursor
            && !self.capped
        {
            self.save_cursor(cursor)?;
        }
        Ok(())
    }

    /// Writes `cursor` to its file, whole or not at all.
    fn save_cursor(&mut self, cursor: &str) -> io::Result<()> {
        let temporary = self.cursor_path.with_extension("cursor.tmp");
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .mode(0o600)
            .open(&temporary)?;
        file.write_all(cursor.as_bytes())?;
        fs::rename(&temporary, &self.cursor_path)?;
        self.cursor = Some(cursor.to_string());
        Ok(())
    }

    /// The kept log's path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.log
    }
}

/// Whether the kept log `file`, of `size` bytes, ends with the line saying it
/// was capped.
fn ends_capped(file: &File, size: u64) -> bool {
    use std::io::{Read, Seek, SeekFrom};
    let from = size.saturating_sub(1024);
    let mut tail = Vec::new();
    let mut reader = file;
    if reader.seek(SeekFrom::Start(from)).is_err() || reader.read_to_end(&mut tail).is_err() {
        return false;
    }
    String::from_utf8_lossy(&tail)
        .lines()
        .last()
        .and_then(|last| serde_json::from_str::<PodLogLine>(last).ok())
        .is_some_and(|last| last.source == CAPPED_SOURCE)
}

/// Opens the capture of the run `run_id`, or says why not, at debug level.
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

    fn frames(chunks: &[&str]) -> Vec<Frame> {
        let mut reader = SseReader::default();
        let mut frames: Vec<Frame> = chunks
            .iter()
            .flat_map(|chunk| reader.feed(chunk.as_bytes()))
            .collect();
        frames.extend(reader.finish());
        frames
    }

    fn frame(id: Option<&str>, data: Option<&str>) -> Frame {
        Frame {
            id: id.map(str::to_string),
            data: data.map(str::to_string),
            truncated: false,
        }
    }

    #[test]
    fn events_are_read_across_chunks() {
        assert_eq!(
            frames(&[
                "id: a/1\nda",
                "ta: {\"x\":1}\n\nid: a/2\r\ndata: two\r\n",
                "\r\n"
            ]),
            [
                frame(Some("a/1"), Some("{\"x\":1}")),
                frame(Some("a/2"), Some("two"))
            ]
        );
    }

    #[test]
    fn comments_and_other_fields_are_skipped_and_keepalives_move_the_cursor() {
        assert_eq!(
            frames(&[": ping\n\nevent: log\nretry: 5\nid: k/9\n\n"]),
            [frame(Some("k/9"), None)]
        );
    }

    #[test]
    fn several_data_lines_join_and_a_last_event_without_blank_line_counts() {
        assert_eq!(
            frames(&["data: a\ndata:b\ndata:  c"]),
            [frame(None, Some("a\nb\n c"))]
        );
    }

    #[test]
    fn an_oversized_line_is_cut() {
        let long = "x".repeat(MAX_LINE + 10);
        let events = frames(&["data: ", &long, "\n\ndata: next\n\n"]);
        assert_eq!(events.len(), 2);
        assert!(events[0].truncated);
        assert_eq!(events[0].data.as_ref().map(String::len), Some(MAX_LINE - 6));
        assert_eq!(events[1].data.as_deref(), Some("next"));
    }

    #[test]
    fn payloads_are_parsed_or_kept_raw() {
        assert_eq!(
            parse_payload(
                r#"{"ts":"2026-06-01T12:02:03Z","source":"container","line":"hi"}"#,
                false
            ),
            Some(PodLogLine {
                ts: "2026-06-01T12:02:03Z".into(),
                source: "container".into(),
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
    fn a_line_is_shown_redacted_on_one_line() {
        let line = PodLogLine {
            ts: "2026-06-01T12:02:03Z".into(),
            source: "system".into(),
            line: "a\r\nb\tc\u{1b}[0m HF_TOKEN=x".into(),
        };
        assert_eq!(
            line.display(),
            "2026-06-01T12:02:03Z sys a\\r\\nb\tc\\u{1b}[0m HF_TOKEN=***"
        );
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
        assert!(line("system").is_from(LogSource::System));
    }

    #[test]
    fn a_capture_keeps_lines_and_its_cursor_then_stops_at_its_cap() -> io::Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir()?;
        let line = PodLogLine {
            ts: "t".into(),
            source: "container".into(),
            line: "x".repeat(100),
        };
        let mut capture = Capture::open(dir.path())?.with_cap(CAP_RESERVE + 300);
        assert_eq!(capture.query().tail, Some(TAIL_MAX));
        capture.take(std::slice::from_ref(&line), Some("c/1"))?;
        assert_eq!(capture.cursor(), Some("c/1"));
        let reopened = Capture::open(dir.path())?.with_cap(CAP_RESERVE + 300);
        assert_eq!(reopened.cursor(), Some("c/1"));
        assert_eq!(
            reopened.query(),
            LogQuery {
                source: None,
                tail: None,
                cursor: Some("c/1".into())
            }
        );
        capture.take(&[line.clone(), line.clone(), line.clone()], Some("c/4"))?;
        assert!(capture.capped());
        assert_eq!(capture.cursor(), Some("c/1"));
        capture.take(std::slice::from_ref(&line), Some("c/5"))?;
        let kept = fs::read_to_string(dir.path().join(POD_LOG))?;
        let lines: Vec<PodLogLine> = kept
            .lines()
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()
            .map_err(io::Error::other)?;
        assert_eq!(lines.len(), 3, "{kept}");
        assert_eq!(lines[1], line);
        assert_eq!(lines[2].source, "overbrainer");
        assert!(lines[2].line.contains("later lines are not kept"));
        let mode = fs::metadata(dir.path().join(POD_LOG))?.permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
        assert!(
            Capture::open(dir.path())?
                .with_cap(CAP_RESERVE + 300)
                .capped()
        );
        Ok(())
    }
}
