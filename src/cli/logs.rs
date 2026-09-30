//! `overbrainer runs logs <run-id>`: the job's log of a run, or with `--pod`
//! the logs of its Runpod pod, kept in the run directory and, while the pod
//! exists, read from Runpod. It never writes: the capture belongs to the
//! command following the run.

use std::fs;
use std::io;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, bail};

use super::LogsArgs;
use crate::config::EnvSource;
use crate::exec::JOB_LOG;
use crate::runpod::{
    BOOTSTRAP_LOG, LogError, LogQuery, LogSource, PodId, PodLogLine, PodRecord, PodState,
    RunpodClient, TAIL_MAX, WATCHDOG_LOG, follow_logs, kept_cursor, kept_lines, one_line, snapshot,
};
use crate::runs::Runs;
use crate::secrets::redact_line;

/// How long the catch-up of a live pod's log waits for its lines.
const CATCH_UP: Duration = Duration::from_secs(10);

/// Runs `overbrainer runs logs`.
///
/// # Errors
///
/// Returns an error when the run does not exist, a kept log cannot be read,
/// or Runpod refuses the log stream of a pod that exists.
pub async fn run(project_dir: &Path, args: &LogsArgs) -> anyhow::Result<()> {
    let runs = Runs::new(project_dir);
    let dir = runs.run_dir(&args.run_id)?;
    if !dir.is_dir() {
        bail!("no run {} in runs/", args.run_id);
    }
    if !args.pod {
        return job_log(&dir, &args.run_id, args.tail);
    }
    let shown = kept(&dir, args)?;
    let live = live_pod(&runs, &args.run_id);
    let Some(pod_id) = live else {
        if !shown {
            println!("runs: no pod log kept for run {}", args.run_id);
        }
        return Ok(());
    };
    let client = match client(project_dir).await {
        Ok(client) => client,
        Err(error) => {
            eprintln!("runs: cannot read the logs of pod {pod_id} from Runpod: {error:#}");
            return Ok(());
        },
    };
    let query = LogQuery {
        source: args.source,
        tail: Some(args.tail.unwrap_or(TAIL_MAX).min(TAIL_MAX)),
        cursor: kept_cursor(&dir),
    };
    stream(&client, &pod_id, query, args.follow).await
}

/// Prints the run's `job.log`, its last `tail` lines only when given.
fn job_log(dir: &Path, run_id: &str, tail: Option<u32>) -> anyhow::Result<()> {
    let path = dir.join(JOB_LOG);
    let text = match fs::read(&path) {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            println!("runs: no {JOB_LOG} for run {run_id} yet");
            return Ok(());
        },
        Err(error) => {
            return Err(error).with_context(|| format!("cannot read {}", path.display()));
        },
    };
    let lines: Vec<&str> = text.lines().collect();
    for line in last(&lines, tail) {
        println!("{}", redact_line(line, &[]));
    }
    Ok(())
}

/// The last `tail` of `items`, or all of them.
fn last<T>(items: &[T], tail: Option<u32>) -> &[T] {
    let keep = tail.map_or(items.len(), |tail| {
        usize::try_from(tail).unwrap_or(usize::MAX)
    });
    &items[items.len().saturating_sub(keep)..]
}

/// Prints what the run directory keeps of the pod's logs: the bootstrap's and
/// the watchdog's own logs (unless only system lines are asked for), then the
/// pod's log. Whether anything was printed.
fn kept(dir: &Path, args: &LogsArgs) -> anyhow::Result<bool> {
    let mut shown = false;
    if args.source != Some(LogSource::System) {
        for name in [BOOTSTRAP_LOG, WATCHDOG_LOG] {
            let path = dir.join(name);
            let Ok(bytes) = fs::read(&path) else {
                continue;
            };
            let text = String::from_utf8_lossy(&bytes);
            let lines: Vec<&str> = text.lines().collect();
            println!("== {name} ==");
            for line in last(&lines, args.tail) {
                println!("{}", one_line(&redact_line(line, &[])));
            }
            shown = true;
        }
    }
    let lines: Vec<PodLogLine> = kept_lines(dir)
        .with_context(|| format!("cannot read the pod log of run {}", args.run_id))?
        .into_iter()
        .filter(|line| args.source.is_none_or(|source| line.is_from(source)))
        .collect();
    if !lines.is_empty() {
        if shown {
            println!("== pod ==");
        }
        for line in last(&lines, args.tail) {
            println!("{}", line.display());
        }
        shown = true;
    }
    Ok(shown)
}

/// The pod of the run `run_id` that may still exist: recorded, not deleted.
fn live_pod(runs: &Runs, run_id: &str) -> Option<PodId> {
    match PodRecord::load(runs, run_id) {
        Ok(Some(record)) if record.state != PodState::Deleted => record.pod_id,
        Ok(_) => None,
        Err(error) => {
            eprintln!("runs: cannot read the pod record of run {run_id}: {error}");
            None
        },
    }
}

/// A Runpod client from the project's configuration.
async fn client(project_dir: &Path) -> anyhow::Result<RunpodClient> {
    let settings = crate::config::load(project_dir, EnvSource::Process)?;
    super::pod::client(&settings).await
}

/// Prints the pod's lines Runpod has after what is kept, then, with `follow`,
/// the new ones until Ctrl-C.
async fn stream(
    client: &RunpodClient,
    pod_id: &PodId,
    query: LogQuery,
    follow: bool,
) -> anyhow::Result<()> {
    let mut print = |lines: &[PodLogLine], _: Option<&str>| {
        for line in lines {
            println!("{}", line.display());
        }
        Ok(())
    };
    let ended = if follow {
        tokio::select! {
            error = follow_logs(client, pod_id, query, &mut print) => Err(error),
            _ = tokio::signal::ctrl_c() => Ok(None),
        }
    } else {
        snapshot(client, pod_id, query, CATCH_UP, &mut print).await
    };
    match ended {
        Ok(_) => Ok(()),
        Err(LogError::Api(error)) if error.status() == Some(404) => {
            eprintln!("runs: pod {pod_id} no longer exists, so Runpod has no more of its logs");
            Ok(())
        },
        Err(error) => Err(anyhow::Error::new(error)
            .context(format!("cannot read the logs of pod {pod_id} from Runpod"))),
    }
}
