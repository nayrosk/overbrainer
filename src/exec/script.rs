//! POSIX shell snippets shared by the executors. They run with `sh` on the target.

use super::{CANCEL_FILE, Container, EXIT_FILE, JobStatus, PID_FILE};

/// `text` as one single-quoted shell word.
#[must_use]
pub fn quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', r"'\''"))
}

/// Wraps the job's `script` so that it records its process ID first and its exit
/// code last, in the current directory. The exit code is written through a rename,
/// so a reader never sees a partial file.
#[must_use]
pub fn job_script(script: &str) -> String {
    format!(
        "echo $$ > {PID_FILE}\n( {script} )\ncode=$?\necho \"$code\" > {EXIT_FILE}.tmp && mv -f {EXIT_FILE}.tmp {EXIT_FILE}\nexit \"$code\"\n"
    )
}

/// Prints one word describing the job in `dir` with session leader `pid`:
/// `cancelled`, `exited <code>`, `running` or `lost`.
#[must_use]
pub fn status_script(dir: &str, pid: u32) -> String {
    format!(
        "cd -- {dir} 2>/dev/null || {{ echo lost; exit 0; }}\n\
         if [ -f {CANCEL_FILE} ]; then echo cancelled\n\
         elif [ -f {EXIT_FILE} ]; then echo \"exited $(cat {EXIT_FILE})\"\n\
         elif kill -s 0 {pid} 2>/dev/null; then echo running\n\
         else echo lost; fi\n",
        dir = quote(dir)
    )
}

/// Reads the output of [`status_script`].
#[must_use]
pub fn parse_status(output: &str) -> Option<JobStatus> {
    let output = output.trim();
    match output {
        "running" => Some(JobStatus::Running),
        "cancelled" => Some(JobStatus::Cancelled),
        "lost" => Some(JobStatus::Lost),
        _ => output
            .strip_prefix("exited ")
            .and_then(|code| code.trim().parse().ok())
            .map(JobStatus::Exited),
    }
}

/// Marks the job in `dir` as cancelled, stops its container, then its process group:
/// `SIGTERM`, and `SIGKILL` if the group is still alive 10 seconds later.
#[must_use]
pub fn cancel_script(dir: &str, pid: u32, container: Option<&Container>) -> String {
    let stop = container.map_or_else(String::new, |container| {
        format!(
            "{} stop -t 30 {} >/dev/null 2>&1\n",
            container.engine.command(),
            quote(&container.name)
        )
    });
    format!(
        "cd -- {dir} || exit 1\n\
         : > {CANCEL_FILE}\n\
         {stop}\
         kill -s TERM -{pid} 2>/dev/null\n\
         i=0\n\
         while kill -s 0 -{pid} 2>/dev/null && [ \"$i\" -lt 10 ]; do sleep 1; i=$((i + 1)); done\n\
         kill -s KILL -{pid} 2>/dev/null\n\
         exit 0\n",
        dir = quote(dir)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Engine;

    #[test]
    fn quoting_survives_single_quotes() {
        assert_eq!(quote("a b"), "'a b'");
        assert_eq!(quote("it's"), r"'it'\''s'");
        assert_eq!(quote(""), "''");
    }

    #[test]
    fn statuses_parse() {
        assert_eq!(parse_status("running\n"), Some(JobStatus::Running));
        assert_eq!(parse_status("exited 0\n"), Some(JobStatus::Exited(0)));
        assert_eq!(parse_status("exited 137"), Some(JobStatus::Exited(137)));
        assert_eq!(parse_status("cancelled"), Some(JobStatus::Cancelled));
        assert_eq!(parse_status("lost"), Some(JobStatus::Lost));
        assert_eq!(parse_status("exited x"), None);
        assert_eq!(parse_status(""), None);
    }

    #[test]
    fn cancel_stops_the_container_first() {
        let container = Container {
            engine: Engine::Podman,
            name: "overbrainer-r1".to_string(),
        };
        let script = cancel_script("/w/r1", 42, Some(&container));
        let stop = script.find("podman stop -t 30 'overbrainer-r1'");
        let term = script.find("kill -s TERM -42");
        assert!(stop.is_some() && term.is_some() && stop < term, "{script}");
        assert!(!cancel_script("/w/r1", 42, None).contains(" stop "));
    }

    #[test]
    fn job_script_records_pid_and_exit_code() {
        assert_eq!(
            job_script("true && false"),
            "echo $$ > job.pid\n( true && false )\ncode=$?\necho \"$code\" > exit_code.tmp && mv -f exit_code.tmp exit_code\nexit \"$code\"\n"
        );
    }
}
