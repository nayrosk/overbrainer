//! The pod source of the Logs view: the Runpod pod's log of a run, as kept in
//! `runs/<run-id>/.pod/pod.log` by the command following it, read by a task
//! from where the last read stopped, every [`REFRESH`] while it is shown.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;
use std::time::SystemTime;

use tracing::Level;

use super::app::{App, Effect, Severity, View};
use super::follow::REFRESH;
use super::tasks::{Task, TaskId};
use crate::logging::{LOG_LINES, LogBuffer, LogLine};
use crate::runpod::{POD_LOG, PodLogLine, one_line};
use crate::runs::{Runs, parse_rfc3339};
use crate::secrets::redact_line;

/// Which log the Logs view shows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum LogSource {
    /// overbrainer's own log lines.
    #[default]
    Overbrainer,
    /// The pod's log of a run.
    Pod,
}

/// The pod's log of one run, as far as it was read.
#[derive(Debug)]
pub(super) struct PodLogs {
    /// The run, once `s` chose one.
    pub(super) run: Option<String>,
    /// Its lines: the source in the target, `sys` or `ctr`.
    pub(super) lines: LogBuffer,
    /// Bytes of the kept log read so far.
    offset: u64,
    /// The read running, if any.
    reading: Option<TaskId>,
    /// When the last read started.
    read_at: Option<SystemTime>,
    /// Why the last read failed.
    pub(super) error: Option<String>,
}

impl Default for PodLogs {
    fn default() -> Self {
        Self {
            run: None,
            lines: LogBuffer::new(LOG_LINES),
            offset: 0,
            reading: None,
            read_at: None,
            error: None,
        }
    }
}

/// What a read of a kept pod log found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PodLogRead {
    /// The run.
    pub(super) run: String,
    /// Where the next read starts.
    pub(super) offset: u64,
    /// Whether the log was found shorter than before: read again from its
    /// start, so the lines shown so far are replaced.
    pub(super) restarted: bool,
    /// The new lines, oldest first.
    pub(super) lines: Vec<PodLogLine>,
    /// Why the log could not be read.
    pub(super) error: Option<String>,
}

/// Reads the pod log kept for the run `run` of the project in `dir`, from
/// `offset` to its last complete line. No log yet reads as nothing.
pub(super) fn read_pod_log(dir: &Path, run: &str, offset: u64) -> PodLogRead {
    let mut read = PodLogRead {
        run: run.to_string(),
        offset,
        restarted: false,
        lines: Vec::new(),
        error: None,
    };
    let path = match Runs::new(dir).run_dir(run) {
        Ok(run_dir) => run_dir.join(POD_LOG),
        Err(error) => {
            read.error = Some(error.to_string());
            return read;
        },
    };
    match read_from(&path, offset) {
        Ok((bytes, start)) => {
            read.restarted = start != offset;
            let complete = bytes
                .iter()
                .rposition(|byte| *byte == b'\n')
                .map_or(0, |at| at + 1);
            let text = String::from_utf8_lossy(&bytes[..complete]);
            read.lines = text
                .lines()
                .filter_map(|line| serde_json::from_str(line).ok())
                .collect();
            read.offset = start + u64::try_from(complete).unwrap_or(0);
        },
        Err(error) if error.kind() == io::ErrorKind::NotFound => {},
        Err(error) => read.error = Some(format!("cannot read {}: {error}", path.display())),
    }
    read
}

/// The bytes of `path` from `offset`, or from its start when it is now
/// shorter, with where they start.
fn read_from(path: &Path, offset: u64) -> io::Result<(Vec<u8>, u64)> {
    let mut file = File::open(path)?;
    let start = if file.metadata()?.len() < offset {
        0
    } else {
        offset
    };
    file.seek(SeekFrom::Start(start))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok((bytes, start))
}

/// A pod log line as the Logs view keeps it.
fn log_line(line: &PodLogLine, now: SystemTime) -> LogLine {
    let time = line
        .ts
        .split_once('.')
        .map_or_else(
            || parse_rfc3339(&line.ts),
            |(seconds, _)| parse_rfc3339(&format!("{seconds}Z")),
        )
        .unwrap_or(now);
    LogLine {
        seq: 0,
        level: Level::INFO,
        target: line.short_source().to_string(),
        time,
        message: one_line(&redact_line(&line.line, &[])),
    }
}

/// `line` of the pod source formatted for the Logs export file:
/// `2026-09-28T01:02:03Z ctr message`.
pub(super) fn export_pod_line(line: &LogLine) -> String {
    format!(
        "{} {} {}",
        crate::runs::rfc3339(line.time),
        line.target,
        line.message
    )
}

impl App {
    /// `s` in the Logs view: switches between overbrainer's log and the pod's
    /// log of the run selected in the Training view, read at once.
    pub(super) fn toggle_log_source(&mut self) -> Vec<Effect> {
        self.log_view.anchor = None;
        if self.log_view.source == LogSource::Pod {
            self.log_view.source = LogSource::Overbrainer;
            return Vec::new();
        }
        let Some(run) = self
            .training
            .selected_run()
            .map(|row| row.record.id.clone())
        else {
            self.say(
                Severity::Warn,
                "no run selected: pick one in Training (4) first",
            );
            return Vec::new();
        };
        self.log_view.source = LogSource::Pod;
        if self.pod_logs.run.as_deref() != Some(run.as_str()) {
            self.pod_logs = PodLogs {
                run: Some(run),
                ..PodLogs::default()
            };
        }
        self.read_pod_log()
    }

    /// Reads the shown pod log again when it is shown and it is time.
    pub(super) fn read_pod_log_when_due(&mut self) -> Vec<Effect> {
        let recent = self
            .pod_logs
            .read_at
            .is_some_and(|at| matches!(self.now.duration_since(at), Ok(since) if since < REFRESH));
        if self.view == View::Logs && self.log_view.source == LogSource::Pod && !recent {
            return self.read_pod_log();
        }
        Vec::new()
    }

    /// Reads the shown pod log from where the last read stopped, in a task;
    /// one at a time.
    fn read_pod_log(&mut self) -> Vec<Effect> {
        let Some(run) = self.pod_logs.run.clone() else {
            return Vec::new();
        };
        if self.pod_logs.reading.is_some() {
            return Vec::new();
        }
        self.pod_logs.read_at = Some(self.now);
        let id = self.task_id();
        self.pod_logs.reading = Some(id);
        vec![Effect::Spawn(
            id,
            Task::PodLog {
                run,
                offset: self.pod_logs.offset,
            },
        )]
    }

    /// Read `id` found `read`: its lines are added when it is the read
    /// running and of the run shown.
    pub(super) fn pod_log_read(&mut self, id: TaskId, read: PodLogRead) -> Vec<Effect> {
        if self.pod_logs.reading != Some(id) {
            return Vec::new();
        }
        self.pod_logs.reading = None;
        if self.pod_logs.run.as_deref() != Some(read.run.as_str()) {
            return Vec::new();
        }
        if read.restarted {
            self.pod_logs.lines = LogBuffer::new(LOG_LINES);
        }
        for line in &read.lines {
            self.pod_logs.lines.push(log_line(line, self.now));
        }
        self.pod_logs.offset = read.offset;
        self.pod_logs.error = read.error;
        self.dirty = true;
        Vec::new()
    }

    /// Read task `id` failed (a panic): it no longer counts.
    pub(super) fn pod_log_failed(&mut self, id: TaskId, error: String) -> bool {
        if self.pod_logs.reading != Some(id) {
            return false;
        }
        self.pod_logs.reading = None;
        self.pod_logs.error = Some(error);
        true
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use super::*;

    const RUN: &str = "20260922-143005-a1b2";

    fn json_line(ts: &str, source: &str, line: &str) -> String {
        format!(
            "{}\n",
            serde_json::json!({"ts": ts, "source": source, "line": line})
        )
    }

    #[test]
    fn a_kept_log_is_read_by_complete_lines_from_an_offset() -> io::Result<()> {
        let dir = tempfile::tempdir()?;
        let pod = dir.path().join("runs").join(RUN).join(".pod");
        std::fs::create_dir_all(&pod)?;
        let first = json_line("2026-09-22T14:30:10Z", "system", "one");
        std::fs::write(pod.join("pod.log"), format!("{first}{{\"ts\":"))?;
        let read = read_pod_log(dir.path(), RUN, 0);
        assert_eq!(read.lines.len(), 1);
        assert_eq!(read.offset, first.len() as u64);
        assert!(!read.restarted);
        let again = read_pod_log(dir.path(), RUN, read.offset);
        assert!(again.lines.is_empty());
        // Shorter than the offset: read from the start again.
        let restarted = read_pod_log(dir.path(), RUN, 10_000);
        assert!(restarted.restarted);
        assert_eq!(restarted.lines.len(), 1);
        // No log yet: nothing, and no error.
        let none = read_pod_log(dir.path(), "20260101-000000-dead", 0);
        assert_eq!((none.lines.len(), none.error), (0, None));
        Ok(())
    }

    fn pod_lines() -> Vec<PodLogLine> {
        [
            (
                "2026-09-21T14:10:00Z",
                "system",
                "pulling image overbrainer/axolotl",
            ),
            (
                "2026-09-21T14:12:30Z",
                "container",
                "bootstrap: run 20260922-143005-a1b2",
            ),
            (
                "2026-09-21T14:13:10Z",
                "container",
                "export HF_TOKEN=hf_abcdefghijklmnop",
            ),
        ]
        .into_iter()
        .map(|(ts, source, line)| PodLogLine {
            ts: ts.into(),
            source: source.into(),
            line: line.into(),
        })
        .collect()
    }

    #[test]
    fn s_shows_the_kept_pod_log_of_the_selected_run() -> Result<(), Box<dyn std::error::Error>> {
        use crossterm::event::KeyCode;

        use crate::runs::RunState;
        use crate::tui::snapshots::{app, key, pod, run, snapshot};
        use crate::tui::training::RunRow;

        let mut app = app();
        app.view = View::Logs;
        // No run selected: nothing to show, the source stays.
        assert!(app.on_input(&key(KeyCode::Char('s'))).is_empty());
        assert_eq!(app.log_view.source, LogSource::Overbrainer);
        app.training.runs = vec![RunRow {
            record: run(RUN, "gpu_cloud", RunState::Running),
            pod: Some(pod(RUN)?),
        }];
        let effects = app.on_input(&key(KeyCode::Char('s')));
        let [Effect::Spawn(id, Task::PodLog { run, offset: 0 })] = effects.as_slice() else {
            return Err(format!("{effects:?}").into());
        };
        assert_eq!(run, RUN);
        assert_eq!(app.log_view.source, LogSource::Pod);
        // One read at a time.
        assert!(app.read_pod_log_when_due().is_empty());
        let read = PodLogRead {
            run: RUN.into(),
            offset: 300,
            restarted: false,
            lines: pod_lines(),
            error: None,
        };
        app.pod_log_read(*id, read);
        app.status = None;
        snapshot("logs_pod", &mut app)?;
        let effects = app.on_input(&key(KeyCode::Char('x')));
        let [Effect::ExportLogs { name, lines }] = effects.as_slice() else {
            return Err(format!("{effects:?}").into());
        };
        assert!(name.starts_with(&format!("logs-pod-{RUN}-")), "{name}");
        assert_eq!(
            lines.last().map(String::as_str),
            Some("2026-09-21T14:13:10Z ctr export HF_TOKEN=***")
        );
        // Due again once the refresh passed: the next read starts at the offset.
        app.now += REFRESH;
        let effects = app.read_pod_log_when_due();
        assert!(
            matches!(
                effects.as_slice(),
                [Effect::Spawn(_, Task::PodLog { offset: 300, .. })]
            ),
            "{effects:?}"
        );
        // `s` again: overbrainer's own log.
        app.on_input(&key(KeyCode::Char('s')));
        assert_eq!(app.log_view.source, LogSource::Overbrainer);
        Ok(())
    }

    #[test]
    fn a_pod_line_keeps_its_time_and_source_redacted_on_one_line() {
        let now = UNIX_EPOCH;
        let line = PodLogLine {
            ts: "2026-09-21T14:13:20.250Z".into(),
            source: "container".into(),
            line: "a\nb HF_TOKEN=x".into(),
        };
        let kept = log_line(&line, now);
        assert_eq!(kept.time, UNIX_EPOCH + Duration::from_secs(1_790_000_000));
        assert_eq!(kept.target, "ctr");
        assert_eq!(kept.message, "a\\nb HF_TOKEN=***");
        assert_eq!(
            export_pod_line(&kept),
            "2026-09-21T14:13:20Z ctr a\\nb HF_TOKEN=***"
        );
        let unknown = PodLogLine::default();
        assert_eq!(log_line(&unknown, now).time, now);
    }
}
