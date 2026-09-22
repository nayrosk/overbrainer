//! POSIX shell snippets shared by the executors. They run with `sh` on the target,
//! which is `bash` on Arch and macOS, `dash` on Debian and Ubuntu and `busybox ash`
//! on Alpine, so nothing here may rely on one shell's extensions.

use super::{CANCEL_FILE, CANCELLING_FILE, Container, EXIT_FILE, JobStatus, PID_FILE};

/// `text` as one single-quoted shell word.
#[must_use]
pub fn quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', r"'\''"))
}

/// Shell function sending signal `$1` to the process group led by `$2`, defined once
/// at the top of every script below that signals a group.
///
/// Shells disagree on how the negative process ID naming a group reaches `kill`.
/// The `dash` of Debian and Ubuntu, where `/bin/sh` usually points, rejects
/// `kill -s 0 -1234` as `Illegal option -1`, and needs `-- -1234`. The `busybox ash`
/// of Alpine images rejects the `--` as `invalid number` and needs the bare
/// `-1234`. `bash`, the `/bin/sh` of Arch and macOS, takes both. Trying the two
/// forms in turn therefore works under every `sh` a target or a client runs, and
/// keeps the exit code of a plain `kill`: zero when the group was signalled, non
/// zero when it is gone. Both forms print nothing.
const GROUP_SIGNAL: &str = "group_signal() { kill -s \"$1\" -- -\"$2\" 2>/dev/null || kill -s \"$1\" -\"$2\" 2>/dev/null; }\n";

/// Wraps the job's `script` so that it records its process ID first and its exit
/// code last, in the current directory. Both are written through a rename, so a
/// reader never sees a partial file. The script runs in a subshell on lines of its
/// own, so a trailing comment or heredoc terminator inside it cannot swallow the
/// wrapper's closing parenthesis.
#[must_use]
pub fn job_script(script: &str) -> String {
    format!(
        "echo $$ > {PID_FILE}.tmp && mv -f {PID_FILE}.tmp {PID_FILE}\n(\n{script}\n)\ncode=$?\necho \"$code\" > {EXIT_FILE}.tmp && mv -f {EXIT_FILE}.tmp {EXIT_FILE}\nexit \"$code\"\n"
    )
}

/// Prints one word describing the job in `dir` with session leader `pid`:
/// `cancelled`, `exited <code>`, `running` or `lost`. Checks the whole process
/// group, not only its leader, so a job whose leader has exited but whose children
/// linger is still `running`.
///
/// The group's liveness is sampled once, before any marker file is checked, so a
/// cancel that runs to completion between that sample and the marker checks cannot
/// make this read `lost`. The markers are then read in this order:
///
/// - [`CANCEL_FILE`] wins unconditionally: once it exists the result is always
///   `cancelled`, even if the process group's ID has since been recycled by an
///   unrelated, live process;
/// - otherwise, while [`CANCELLING_FILE`] exists and the group is still alive, the
///   result is `running`, ahead of [`EXIT_FILE`]: stopping a job's container makes
///   its wrapper write an exit code while the cancel is still stopping the group,
///   and that must not stick as `exited <code>`;
/// - failing that, an existing [`EXIT_FILE`] wins;
/// - otherwise [`CANCELLING_FILE`] with a group that is gone reads `cancelled`,
///   which covers the window between the group's death and [`CANCEL_FILE`];
/// - failing all that, the sampled liveness decides `running` or `lost`.
///
/// [`EXIT_FILE`] coming ahead of a [`CANCELLING_FILE`] whose group is gone is what
/// keeps an abandoned marker (a cancel killed after writing it) from labelling a
/// job that then finished normally as cancelled for ever. Its cost is a short
/// window during a container cancel: between the wrapper writing [`EXIT_FILE`] and
/// [`CANCEL_FILE`] landing, roughly a second (up to the `<engine> stop -t 30`
/// timeout), this reads `exited <code>` instead of `running`. [`CANCEL_FILE`] still
/// wins afterwards, since [`cancel_script`] waits for the group to be gone before
/// writing it.
///
/// Exits `1` without printing or signalling anything when `pid` is `0` or `1`.
#[must_use]
pub fn status_script(dir: &str, pid: u32) -> String {
    format!(
        "{GROUP_SIGNAL}\
         cd -- {dir} 2>/dev/null || {{ echo lost; exit 0; }}\n\
         pid={pid}\n\
         [ \"$pid\" -gt 1 ] || exit 1\n\
         alive=0\n\
         group_signal 0 \"$pid\" && alive=1\n\
         if [ -f {CANCEL_FILE} ]; then echo cancelled\n\
         elif [ -f {CANCELLING_FILE} ] && [ \"$alive\" -eq 1 ]; then echo running\n\
         elif [ -f {EXIT_FILE} ]; then echo \"exited $(cat {EXIT_FILE})\"\n\
         elif [ -f {CANCELLING_FILE} ]; then echo cancelled\n\
         elif [ \"$alive\" -eq 1 ]; then echo running\n\
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
/// [`CANCELLING_FILE`] is written, atomically, before anything is signalled;
/// [`CANCEL_FILE`] is written only once the group is confirmed gone. Together with
/// [`status_script`]'s marker priority, this keeps `status` reporting `running`
/// while a cancel is in progress and `cancelled` as soon as the group dies, never
/// `lost`, and makes a repeated cancel a no-op. Stopping a container is the one
/// case where `status` can read `exited <code>` mid-cancel, for as long as the
/// group takes to die after the wrapper has written its exit code; see
/// [`status_script`]. Does nothing but exit `0`, without signalling anything, when the
/// job already has an exit code or is already marked cancelled: a finished job keeps
/// its real result, and a cancel is never sent to a process group whose ID may since
/// have been recycled by the system. Exits `1` without signalling anything when
/// `pid` is `0` or `1`, or when the job directory is gone.
///
/// After the `SIGKILL`, it waits (up to 10 more seconds) until the group is really
/// gone before writing [`CANCEL_FILE`]: a killed leader stays a member of its group
/// as a zombie until its parent reaps it. If the group outlives that wait, it exits
/// `1` with a message and without [`CANCEL_FILE`]; [`CANCELLING_FILE`] stays, so
/// `status` keeps following the group's liveness and a later cancel tries again.
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
        "{GROUP_SIGNAL}\
         cd -- {dir} || exit 1\n\
         pid={pid}\n\
         [ \"$pid\" -gt 1 ] || exit 1\n\
         if [ -f {EXIT_FILE} ] || [ -f {CANCEL_FILE} ]; then exit 0; fi\n\
         : > {CANCELLING_FILE}.tmp && mv -f {CANCELLING_FILE}.tmp {CANCELLING_FILE}\n\
         {stop}\
         group_signal TERM \"$pid\"\n\
         i=0\n\
         while group_signal 0 \"$pid\" && [ \"$i\" -lt 10 ]; do sleep 1; i=$((i + 1)); done\n\
         group_signal KILL \"$pid\"\n\
         i=0\n\
         while group_signal 0 \"$pid\"; do\n\
         [ \"$i\" -lt 10 ] || {{ echo \"process group $pid is still present after SIGKILL (its leader may be waiting for its parent process to reap it); the job stays marked cancelling, retry the cancel\" >&2; exit 1; }}\n\
         sleep 1; i=$((i + 1)); done\n\
         : > {CANCEL_FILE}\n\
         exit 0\n",
        dir = quote(dir)
    )
}

#[cfg(test)]
mod tests {
    use std::fmt;
    use std::fs;
    use std::os::unix::process::CommandExt;
    use std::process::Command;
    use std::sync::OnceLock;

    use tempfile::tempdir;

    use crate::config::Engine;

    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// A shell the generated scripts must run under.
    #[derive(Clone, Copy)]
    struct Shell {
        /// Program to run.
        program: &'static str,
        /// Arguments before `-c`, for a multi-call binary such as `busybox`.
        args: &'static [&'static str],
    }

    impl Shell {
        /// The command running `script` with this shell.
        fn run(self, script: &str) -> Command {
            let mut command = Command::new(self.program);
            command.args(self.args).arg("-c").arg(script);
            command
        }
    }

    impl fmt::Display for Shell {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "{}", self.program)?;
            for arg in self.args {
                write!(formatter, " {arg}")?;
            }
            Ok(())
        }
    }

    /// The shells the scripts are tested under. `sh` is `bash` on Arch and macOS and
    /// `dash` on Debian and Ubuntu, and `busybox ash` is what Alpine targets run;
    /// they disagree on how `kill` takes a process group, so a script that works
    /// under one can be broken under another.
    const CANDIDATES: [Shell; 3] = [
        Shell {
            program: "sh",
            args: &[],
        },
        Shell {
            program: "dash",
            args: &[],
        },
        Shell {
            program: "busybox",
            args: &["sh"],
        },
    ];

    /// Every shell of [`CANDIDATES`] installed here, probed once. A shell that is
    /// missing is reported once and skipped, so a machine without `dash` or `busybox`
    /// runs the coverage it can instead of failing.
    fn shells() -> &'static [Shell] {
        static SHELLS: OnceLock<Vec<Shell>> = OnceLock::new();
        SHELLS.get_or_init(|| {
            CANDIDATES
                .into_iter()
                .filter(|shell| {
                    let available = shell.run(":").status().is_ok_and(|status| status.success());
                    if !available {
                        eprintln!("skipped: {shell} is not installed");
                    }
                    available
                })
                .collect()
        })
    }

    /// Whether the process group led by `pid` is still alive, asked through the same
    /// helper the scripts use.
    fn group_alive(shell: Shell, pid: u32) -> Result<bool, Box<dyn std::error::Error>> {
        Ok(shell
            .run(&format!("{GROUP_SIGNAL}group_signal 0 {pid}"))
            .status()?
            .success())
    }

    /// Sends `SIGKILL` to the process group led by `pid`.
    fn kill_group(shell: Shell, pid: u32) -> TestResult {
        shell
            .run(&format!("{GROUP_SIGNAL}group_signal KILL {pid}"))
            .status()?;
        Ok(())
    }

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
        let term = script.find("group_signal TERM \"$pid\"");
        assert!(stop.is_some() && term.is_some() && stop < term, "{script}");
        assert!(!cancel_script("/w/r1", 42, None).contains(" stop "));
    }

    #[test]
    fn job_script_wraps_the_script_and_records_pid_and_exit_code() {
        assert_eq!(
            job_script("true && false"),
            "echo $$ > job.pid.tmp && mv -f job.pid.tmp job.pid\n(\ntrue && false\n)\ncode=$?\necho \"$code\" > exit_code.tmp && mv -f exit_code.tmp exit_code\nexit \"$code\"\n"
        );
    }

    #[test]
    fn job_script_runs_with_a_trailing_comment_in_the_script() -> TestResult {
        for &shell in shells() {
            let dir = tempdir()?;
            let script = job_script("true # a trailing comment, not a closing paren");
            let output = shell.run(&script).current_dir(dir.path()).output()?;
            assert!(
                output.status.success(),
                "{shell}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let pid = fs::read_to_string(dir.path().join(PID_FILE))?;
            assert!(pid.trim().parse::<u32>().is_ok(), "{shell}: {pid:?}");
            assert_eq!(fs::read_to_string(dir.path().join(EXIT_FILE))?.trim(), "0");
        }
        Ok(())
    }

    #[test]
    fn job_script_records_a_non_zero_exit_code() -> TestResult {
        for &shell in shells() {
            let dir = tempdir()?;
            let output = shell
                .run(&job_script("exit 7"))
                .current_dir(dir.path())
                .output()?;
            assert_eq!(output.status.code(), Some(7), "{shell}");
            assert_eq!(fs::read_to_string(dir.path().join(EXIT_FILE))?.trim(), "7");
        }
        Ok(())
    }

    #[test]
    fn status_reports_lost_when_the_directory_is_gone() -> TestResult {
        for &shell in shells() {
            let script = status_script("/no/such/dir/overbrainer-test", 2);
            let output = shell.run(&script).output()?;
            assert!(
                output.status.success(),
                "{shell}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(
                parse_status(&String::from_utf8_lossy(&output.stdout)),
                Some(JobStatus::Lost),
                "{shell}"
            );
        }
        Ok(())
    }

    #[test]
    fn status_rejects_pid_0_and_1_without_signalling() -> TestResult {
        for &shell in shells() {
            let dir = tempdir()?;
            for pid in [0, 1] {
                let script = status_script(&dir.path().display().to_string(), pid);
                let output = shell.run(&script).output()?;
                assert_eq!(output.status.code(), Some(1), "{shell}");
                assert!(output.stdout.is_empty(), "{shell}");
            }
        }
        Ok(())
    }

    #[test]
    fn status_reports_running_for_a_live_process_group() -> TestResult {
        for &shell in shells() {
            let mut child = Command::new("sleep").arg("5").process_group(0).spawn()?;
            let pid = child.id();
            let dir = tempdir()?;
            let output = shell
                .run(&status_script(&dir.path().display().to_string(), pid))
                .output()?;
            let status = parse_status(&String::from_utf8_lossy(&output.stdout));
            child.kill()?;
            child.wait()?;
            assert_eq!(status, Some(JobStatus::Running), "{shell}");
        }
        Ok(())
    }

    #[test]
    fn status_reports_exited_with_no_cancel_markers() -> TestResult {
        for &shell in shells() {
            let mut child = Command::new("true").process_group(0).spawn()?;
            let pid = child.id();
            child.wait()?;

            let dir = tempdir()?;
            fs::write(dir.path().join(EXIT_FILE), "42\n")?;

            let output = shell
                .run(&status_script(&dir.path().display().to_string(), pid))
                .output()?;
            assert_eq!(
                parse_status(&String::from_utf8_lossy(&output.stdout)),
                Some(JobStatus::Exited(42)),
                "{shell}"
            );
        }
        Ok(())
    }

    #[test]
    fn status_stays_running_when_an_exit_code_appears_mid_cancel() -> TestResult {
        // A container-cancel race: `<engine> stop` makes the foreground
        // `docker run --rm` exit, so the job's wrapper writes exit_code while the
        // group is still being stopped. `cancelling` exists, `exit_code` exists,
        // `cancelled` does not exist yet, and the group is still alive.
        for &shell in shells() {
            let mut child = Command::new("sleep").arg("5").process_group(0).spawn()?;
            let pid = child.id();

            let dir = tempdir()?;
            fs::write(dir.path().join(CANCELLING_FILE), "")?;
            fs::write(dir.path().join(EXIT_FILE), "143\n")?;

            let output = shell
                .run(&status_script(&dir.path().display().to_string(), pid))
                .output()?;
            let status = parse_status(&String::from_utf8_lossy(&output.stdout));
            child.kill()?;
            child.wait()?;
            assert_eq!(status, Some(JobStatus::Running), "{shell}");
        }
        Ok(())
    }

    #[test]
    fn status_reports_exited_for_an_abandoned_cancelling_marker() -> TestResult {
        // A cancel killed between writing `cancelling` and signalling anything:
        // the marker stays on disk for ever while the job finishes normally. Its
        // real exit code must win, not the abandoned marker.
        for &shell in shells() {
            let mut child = Command::new("true").process_group(0).spawn()?;
            let pid = child.id();
            child.wait()?;

            let dir = tempdir()?;
            fs::write(dir.path().join(CANCELLING_FILE), "")?;
            fs::write(dir.path().join(EXIT_FILE), "0\n")?;

            let output = shell
                .run(&status_script(&dir.path().display().to_string(), pid))
                .output()?;
            assert_eq!(
                parse_status(&String::from_utf8_lossy(&output.stdout)),
                Some(JobStatus::Exited(0)),
                "{shell}"
            );
        }
        Ok(())
    }

    #[test]
    fn status_prefers_cancelled_even_with_a_live_recycled_pid() -> TestResult {
        for &shell in shells() {
            let mut child = Command::new("sleep").arg("5").process_group(0).spawn()?;
            let pid = child.id();

            let dir = tempdir()?;
            fs::write(dir.path().join(CANCEL_FILE), "")?;

            let output = shell
                .run(&status_script(&dir.path().display().to_string(), pid))
                .output()?;
            let status = parse_status(&String::from_utf8_lossy(&output.stdout));
            child.kill()?;
            child.wait()?;
            assert_eq!(status, Some(JobStatus::Cancelled), "{shell}");
        }
        Ok(())
    }

    #[test]
    fn cancel_rejects_pid_0_and_1_without_signalling() -> TestResult {
        for &shell in shells() {
            let dir = tempdir()?;
            for pid in [0, 1] {
                let script = cancel_script(&dir.path().display().to_string(), pid, None);
                let output = shell.run(&script).output()?;
                assert_eq!(output.status.code(), Some(1), "{shell}");
                assert!(!dir.path().join(CANCEL_FILE).exists(), "{shell}");
            }
        }
        Ok(())
    }

    #[test]
    fn cancel_exits_without_signalling_when_the_directory_is_gone() -> TestResult {
        for &shell in shells() {
            let script = cancel_script("/no/such/dir/overbrainer-test", 42, None);
            let output = shell.run(&script).output()?;
            assert_eq!(output.status.code(), Some(1), "{shell}");
        }
        Ok(())
    }

    #[test]
    fn cancel_does_nothing_when_the_job_already_exited() -> TestResult {
        for &shell in shells() {
            let dir = tempdir()?;
            fs::write(dir.path().join(EXIT_FILE), "0\n")?;
            let mut child = Command::new("sleep").arg("5").spawn()?;
            let pid = child.id();
            let script = cancel_script(&dir.path().display().to_string(), pid, None);
            let output = shell.run(&script).output()?;
            let untouched = child.try_wait()?.is_none();
            child.kill()?;
            child.wait()?;
            assert!(
                output.status.success(),
                "{shell}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(!dir.path().join(CANCEL_FILE).exists(), "{shell}");
            assert!(
                untouched,
                "{shell}: cancel must not signal a finished job's possibly recycled pid"
            );
        }
        Ok(())
    }

    #[test]
    fn cancel_does_nothing_when_already_cancelled() -> TestResult {
        for &shell in shells() {
            let child = Command::new("sleep").arg("5").process_group(0).spawn()?;
            let pid = child.id();
            let reaper = std::thread::spawn(move || {
                let mut child = child;
                child.wait()
            });

            let dir = tempdir()?;
            fs::write(dir.path().join(CANCEL_FILE), "")?;
            let script = cancel_script(&dir.path().display().to_string(), pid, None);
            let output = shell.run(&script).output()?;
            let still_alive = group_alive(shell, pid)?;

            kill_group(shell, pid)?;
            reaper.join().map_err(|_| "reaper thread panicked")??;
            assert!(
                output.status.success(),
                "{shell}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                still_alive,
                "{shell}: cancel must not re-signal an already cancelled job's process group"
            );
        }
        Ok(())
    }

    #[test]
    fn status_stays_running_while_a_cancel_is_in_progress() -> TestResult {
        // The job ignores SIGTERM, so it only dies to the SIGKILL sent after the 10
        // second grace period: this gives the test a window to observe `status`
        // mid-cancel.
        for &shell in shells() {
            let child = shell
                .run("trap '' TERM; sleep 30")
                .process_group(0)
                .spawn()?;
            let pid = child.id();
            let reaper = std::thread::spawn(move || {
                let mut child = child;
                child.wait()
            });

            let dir = tempdir()?;
            let cancel_dir = dir.path().display().to_string();
            let canceller = std::thread::spawn(move || {
                shell.run(&cancel_script(&cancel_dir, pid, None)).output()
            });

            std::thread::sleep(std::time::Duration::from_secs(2));
            let mid_output = shell
                .run(&status_script(&dir.path().display().to_string(), pid))
                .output()?;
            assert_eq!(
                parse_status(&String::from_utf8_lossy(&mid_output.stdout)),
                Some(JobStatus::Running),
                "{shell}: status must stay running while the group is still being stopped"
            );

            let cancel_output = canceller
                .join()
                .map_err(|_| "canceller thread panicked")??;
            assert!(
                cancel_output.status.success(),
                "{shell}: {}",
                String::from_utf8_lossy(&cancel_output.stderr)
            );
            assert!(dir.path().join(CANCEL_FILE).exists(), "{shell}");
            reaper.join().map_err(|_| "reaper thread panicked")??;
        }
        Ok(())
    }

    #[test]
    fn cancel_kills_the_process_group_before_the_cancelled_marker_appears() -> TestResult {
        // The job ignores SIGTERM, so it dies to the SIGKILL sent after the 10 second
        // grace period. Its parent (this test) only reaps it at 12 seconds: until then
        // the killed leader is a zombie, which still counts as a member of its process
        // group. The marker must wait for that, not follow the SIGKILL at once.
        for &shell in shells() {
            let child = shell
                .run("trap '' TERM; sleep 30")
                .process_group(0)
                .spawn()?;
            let pid = child.id();
            let reaper = std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_secs(12));
                let mut child = child;
                child.wait()
            });

            let dir = tempdir()?;
            let cancel_dir = dir.path().display().to_string();
            let canceller = std::thread::spawn(move || {
                shell.run(&cancel_script(&cancel_dir, pid, None)).output()
            });

            // The moment the marker is seen, the group must already be gone. Checking
            // it then, rather than comparing when each was first seen by separate
            // polls, leaves no window between the two observations to race in.
            let marker_path = dir.path().join(CANCEL_FILE);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            while !marker_path.exists() {
                if std::time::Instant::now() > deadline {
                    return Err("timed out waiting for the cancelled marker".into());
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            let alive = group_alive(shell, pid)?;

            let cancel_output = canceller
                .join()
                .map_err(|_| "canceller thread panicked")??;
            reaper.join().map_err(|_| "reaper thread panicked")??;
            assert!(
                !alive,
                "{shell}: the cancelled marker appeared while the process group still existed"
            );
            assert!(
                cancel_output.status.success(),
                "{shell}: {}",
                String::from_utf8_lossy(&cancel_output.stderr)
            );
        }
        Ok(())
    }

    #[test]
    fn status_never_reports_lost_during_a_normal_cancel() -> TestResult {
        // The job obeys SIGTERM (the default action), so the group can die well within
        // the grace period: this is the race window finding 1 fixes. After the group
        // dies but before the final `cancelled` marker lands, a fast poller must read
        // `cancelled` (thanks to the `cancelling` marker written up front), never
        // `lost`.
        for &shell in shells() {
            let child = Command::new("sleep").arg("10").process_group(0).spawn()?;
            let pid = child.id();
            let reaper = std::thread::spawn(move || {
                let mut child = child;
                child.wait()
            });

            let dir = tempdir()?;
            let cancel_dir = dir.path().display().to_string();
            let canceller = std::thread::spawn(move || {
                shell.run(&cancel_script(&cancel_dir, pid, None)).output()
            });

            let status_dir = dir.path().display().to_string();
            let mut saw_cancelled = false;
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
            loop {
                if std::time::Instant::now() > deadline {
                    return Err("timed out waiting for the cancel to finish".into());
                }
                let output = shell.run(&status_script(&status_dir, pid)).output()?;
                let status = parse_status(&String::from_utf8_lossy(&output.stdout));
                assert_ne!(
                    status,
                    Some(JobStatus::Lost),
                    "{shell}: status must never read lost while a cancel is in progress"
                );
                if status == Some(JobStatus::Cancelled) {
                    saw_cancelled = true;
                }
                if canceller.is_finished() && saw_cancelled {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }

            let cancel_output = canceller
                .join()
                .map_err(|_| "canceller thread panicked")??;
            assert!(
                cancel_output.status.success(),
                "{shell}: {}",
                String::from_utf8_lossy(&cancel_output.stderr)
            );
            reaper.join().map_err(|_| "reaper thread panicked")??;
        }
        Ok(())
    }
}
