//! POSIX shell snippets shared by the executors. They run with `sh` on the target.

use super::{CANCEL_FILE, CANCELLING_FILE, Container, EXIT_FILE, JobStatus, PID_FILE};

/// `text` as one single-quoted shell word.
#[must_use]
pub fn quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', r"'\''"))
}

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
/// - otherwise, while [`CANCELLING_FILE`] exists, the result follows the sampled
///   liveness (`running` if it was alive, `cancelled` if not), ahead of
///   [`EXIT_FILE`]: stopping a job's container can make its wrapper write an exit
///   code while the cancel is still in flight, and that must not stick as
///   `exited <code>` forever;
/// - only once neither marker exists does an existing [`EXIT_FILE`] win;
/// - failing that, the sampled liveness decides `running` or `lost`.
///
/// Exits `1` without printing or signalling anything when `pid` is `0` or `1`.
#[must_use]
pub fn status_script(dir: &str, pid: u32) -> String {
    format!(
        "cd -- {dir} 2>/dev/null || {{ echo lost; exit 0; }}\n\
         pid={pid}\n\
         [ \"$pid\" -gt 1 ] || exit 1\n\
         alive=0\n\
         kill -s 0 -\"$pid\" 2>/dev/null && alive=1\n\
         if [ -f {CANCEL_FILE} ]; then echo cancelled\n\
         elif [ -f {CANCELLING_FILE} ]; then\n\
         if [ \"$alive\" -eq 1 ]; then echo running; else echo cancelled; fi\n\
         elif [ -f {EXIT_FILE} ]; then echo \"exited $(cat {EXIT_FILE})\"\n\
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
/// `lost` and never stuck on an exit code that stopping the container causes the
/// job's wrapper to write while the cancel is still in flight, and makes a repeated
/// cancel a no-op. Does nothing but exit `0`, without signalling anything, when the
/// job already has an exit code or is already marked cancelled: a finished job keeps
/// its real result, and a cancel is never sent to a process group whose ID may since
/// have been recycled by the system. Exits `1` without signalling anything when
/// `pid` is `0` or `1`, or when the job directory is gone.
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
         pid={pid}\n\
         [ \"$pid\" -gt 1 ] || exit 1\n\
         if [ -f {EXIT_FILE} ] || [ -f {CANCEL_FILE} ]; then exit 0; fi\n\
         : > {CANCELLING_FILE}.tmp && mv -f {CANCELLING_FILE}.tmp {CANCELLING_FILE}\n\
         {stop}\
         kill -s TERM -\"$pid\" 2>/dev/null\n\
         i=0\n\
         while kill -s 0 -\"$pid\" 2>/dev/null && [ \"$i\" -lt 10 ]; do sleep 1; i=$((i + 1)); done\n\
         kill -s KILL -\"$pid\" 2>/dev/null\n\
         : > {CANCEL_FILE}\n\
         exit 0\n",
        dir = quote(dir)
    )
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::process::CommandExt;
    use std::process::Command;

    use tempfile::tempdir;

    use crate::config::Engine;

    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// `sh` is required by every test below; they skip (not fail) without it.
    fn sh_available() -> bool {
        Command::new("sh")
            .arg("-c")
            .arg(":")
            .status()
            .is_ok_and(|status| status.success())
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
        let term = script.find("kill -s TERM -\"$pid\"");
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
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        let dir = tempdir()?;
        let script = job_script("true # a trailing comment, not a closing paren");
        let output = Command::new("sh")
            .arg("-c")
            .arg(&script)
            .current_dir(dir.path())
            .output()?;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let pid = fs::read_to_string(dir.path().join(PID_FILE))?;
        assert!(pid.trim().parse::<u32>().is_ok(), "{pid:?}");
        assert_eq!(fs::read_to_string(dir.path().join(EXIT_FILE))?.trim(), "0");
        Ok(())
    }

    #[test]
    fn job_script_records_a_non_zero_exit_code() -> TestResult {
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        let dir = tempdir()?;
        let output = Command::new("sh")
            .arg("-c")
            .arg(job_script("exit 7"))
            .current_dir(dir.path())
            .output()?;
        assert_eq!(output.status.code(), Some(7));
        assert_eq!(fs::read_to_string(dir.path().join(EXIT_FILE))?.trim(), "7");
        Ok(())
    }

    #[test]
    fn status_reports_lost_when_the_directory_is_gone() -> TestResult {
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        let script = status_script("/no/such/dir/overbrainer-test", 2);
        let output = Command::new("sh").arg("-c").arg(&script).output()?;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            parse_status(&String::from_utf8_lossy(&output.stdout)),
            Some(JobStatus::Lost)
        );
        Ok(())
    }

    #[test]
    fn status_rejects_pid_0_and_1_without_signalling() -> TestResult {
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        let dir = tempdir()?;
        for pid in [0, 1] {
            let script = status_script(&dir.path().display().to_string(), pid);
            let output = Command::new("sh").arg("-c").arg(&script).output()?;
            assert_eq!(output.status.code(), Some(1));
            assert!(output.stdout.is_empty());
        }
        Ok(())
    }

    #[test]
    fn status_reports_running_for_a_live_process_group() -> TestResult {
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        let mut child = Command::new("sleep").arg("5").process_group(0).spawn()?;
        let pid = child.id();
        let dir = tempdir()?;
        let output = Command::new("sh")
            .arg("-c")
            .arg(status_script(&dir.path().display().to_string(), pid))
            .output()?;
        assert_eq!(
            parse_status(&String::from_utf8_lossy(&output.stdout)),
            Some(JobStatus::Running)
        );
        child.kill()?;
        child.wait()?;
        Ok(())
    }

    #[test]
    fn status_reports_exited_with_no_cancel_markers() -> TestResult {
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        let mut child = Command::new("true").process_group(0).spawn()?;
        let pid = child.id();
        child.wait()?;

        let dir = tempdir()?;
        fs::write(dir.path().join(EXIT_FILE), "42\n")?;

        let output = Command::new("sh")
            .arg("-c")
            .arg(status_script(&dir.path().display().to_string(), pid))
            .output()?;
        assert_eq!(
            parse_status(&String::from_utf8_lossy(&output.stdout)),
            Some(JobStatus::Exited(42))
        );
        Ok(())
    }

    #[test]
    fn status_prefers_cancelled_when_exit_code_appears_mid_cancel() -> TestResult {
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        // A container-cancel race: `<engine> stop` makes the foreground
        // `docker run --rm` exit, so the job's wrapper writes exit_code before the
        // group TERM finishes it off. `cancelling` exists, `exit_code` exists,
        // `cancelled` does not exist yet, and the group is already gone.
        let mut child = Command::new("true").process_group(0).spawn()?;
        let pid = child.id();
        child.wait()?;

        let dir = tempdir()?;
        fs::write(dir.path().join(CANCELLING_FILE), "")?;
        fs::write(dir.path().join(EXIT_FILE), "143\n")?;

        let output = Command::new("sh")
            .arg("-c")
            .arg(status_script(&dir.path().display().to_string(), pid))
            .output()?;
        assert_eq!(
            parse_status(&String::from_utf8_lossy(&output.stdout)),
            Some(JobStatus::Cancelled)
        );
        Ok(())
    }

    #[test]
    fn status_prefers_cancelled_even_with_a_live_recycled_pid() -> TestResult {
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        let mut child = Command::new("sleep").arg("5").process_group(0).spawn()?;
        let pid = child.id();

        let dir = tempdir()?;
        fs::write(dir.path().join(CANCEL_FILE), "")?;

        let output = Command::new("sh")
            .arg("-c")
            .arg(status_script(&dir.path().display().to_string(), pid))
            .output()?;
        assert_eq!(
            parse_status(&String::from_utf8_lossy(&output.stdout)),
            Some(JobStatus::Cancelled)
        );

        child.kill()?;
        child.wait()?;
        Ok(())
    }

    #[test]
    fn cancel_rejects_pid_0_and_1_without_signalling() -> TestResult {
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        let dir = tempdir()?;
        for pid in [0, 1] {
            let script = cancel_script(&dir.path().display().to_string(), pid, None);
            let output = Command::new("sh").arg("-c").arg(&script).output()?;
            assert_eq!(output.status.code(), Some(1));
            assert!(!dir.path().join(CANCEL_FILE).exists());
        }
        Ok(())
    }

    #[test]
    fn cancel_exits_without_signalling_when_the_directory_is_gone() -> TestResult {
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        let script = cancel_script("/no/such/dir/overbrainer-test", 42, None);
        let output = Command::new("sh").arg("-c").arg(&script).output()?;
        assert_eq!(output.status.code(), Some(1));
        Ok(())
    }

    #[test]
    fn cancel_does_nothing_when_the_job_already_exited() -> TestResult {
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        let dir = tempdir()?;
        fs::write(dir.path().join(EXIT_FILE), "0\n")?;
        let mut child = Command::new("sleep").arg("5").spawn()?;
        let pid = child.id();
        let script = cancel_script(&dir.path().display().to_string(), pid, None);
        let output = Command::new("sh").arg("-c").arg(&script).output()?;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!dir.path().join(CANCEL_FILE).exists());
        assert!(
            child.try_wait()?.is_none(),
            "cancel must not signal a finished job's possibly recycled pid"
        );
        child.kill()?;
        child.wait()?;
        Ok(())
    }

    #[test]
    fn cancel_does_nothing_when_already_cancelled() -> TestResult {
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        let child = Command::new("sleep").arg("5").process_group(0).spawn()?;
        let pid = child.id();
        let reaper = std::thread::spawn(move || {
            let mut child = child;
            child.wait()
        });

        let dir = tempdir()?;
        fs::write(dir.path().join(CANCEL_FILE), "")?;
        let script = cancel_script(&dir.path().display().to_string(), pid, None);
        let output = Command::new("sh").arg("-c").arg(&script).output()?;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );

        let still_alive = Command::new("sh")
            .arg("-c")
            .arg(format!("kill -s 0 -{pid} 2>/dev/null"))
            .status()?;
        assert!(
            still_alive.success(),
            "cancel must not re-signal an already cancelled job's process group"
        );

        Command::new("sh")
            .arg("-c")
            .arg(format!("kill -s KILL -{pid} 2>/dev/null"))
            .status()?;
        reaper.join().map_err(|_| "reaper thread panicked")??;
        Ok(())
    }

    #[test]
    fn status_stays_running_while_a_cancel_is_in_progress() -> TestResult {
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        // Ignores SIGTERM, so it only dies to the SIGKILL sent after the 10 second
        // grace period: this gives the test a window to observe `status` mid-cancel.
        let child = Command::new("sh")
            .args(["-c", "trap '' TERM; sleep 30"])
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
            Command::new("sh")
                .arg("-c")
                .arg(cancel_script(&cancel_dir, pid, None))
                .output()
        });

        std::thread::sleep(std::time::Duration::from_secs(2));
        let mid_output = Command::new("sh")
            .arg("-c")
            .arg(status_script(&dir.path().display().to_string(), pid))
            .output()?;
        assert_eq!(
            parse_status(&String::from_utf8_lossy(&mid_output.stdout)),
            Some(JobStatus::Running),
            "status must stay running while the group is still being stopped"
        );

        let cancel_output = canceller
            .join()
            .map_err(|_| "canceller thread panicked")??;
        assert!(
            cancel_output.status.success(),
            "{}",
            String::from_utf8_lossy(&cancel_output.stderr)
        );
        assert!(dir.path().join(CANCEL_FILE).exists());
        reaper.join().map_err(|_| "reaper thread panicked")??;
        Ok(())
    }

    #[test]
    fn cancel_kills_the_process_group_before_the_cancelled_marker_appears() -> TestResult {
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        // Ignores SIGTERM, so death is driven by the unconditional SIGKILL at the
        // end of the grace period: this makes the group's death deterministic
        // rather than depending on how fast an ordinary process reacts to SIGTERM.
        let child = Command::new("sh")
            .args(["-c", "trap '' TERM; sleep 30"])
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
            Command::new("sh")
                .arg("-c")
                .arg(cancel_script(&cancel_dir, pid, None))
                .output()
        });

        // Poll the group's liveness and the marker's existence side by side, and
        // record which this process observes first. Both are external observations
        // of the same script (a different process), so this cannot prove they
        // happen in the same instant the script sees them, but it does prove this
        // process never observes the marker before the group is already gone.
        let marker_path = dir.path().join(CANCEL_FILE);
        let mut group_dead_at = None;
        let mut marker_seen_at = None;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while group_dead_at.is_none() || marker_seen_at.is_none() {
            if std::time::Instant::now() > deadline {
                return Err(
                    "timed out waiting for the group to die and the marker to appear".into(),
                );
            }
            if group_dead_at.is_none() {
                let alive = Command::new("sh")
                    .arg("-c")
                    .arg(format!("kill -s 0 -{pid} 2>/dev/null"))
                    .status()?
                    .success();
                if !alive {
                    group_dead_at = Some(std::time::Instant::now());
                }
            }
            if marker_seen_at.is_none() && marker_path.exists() {
                marker_seen_at = Some(std::time::Instant::now());
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }

        let cancel_output = canceller
            .join()
            .map_err(|_| "canceller thread panicked")??;
        assert!(
            cancel_output.status.success(),
            "{}",
            String::from_utf8_lossy(&cancel_output.stderr)
        );
        reaper.join().map_err(|_| "reaper thread panicked")??;

        let (Some(group_dead_at), Some(marker_seen_at)) = (group_dead_at, marker_seen_at) else {
            return Err("both the group's death and the marker must have been observed".into());
        };
        assert!(
            group_dead_at <= marker_seen_at,
            "the process group must be observed gone no later than the cancelled marker"
        );
        Ok(())
    }

    #[test]
    fn status_never_reports_lost_during_a_normal_cancel() -> TestResult {
        if !sh_available() {
            eprintln!("skipped: sh is not installed");
            return Ok(());
        }
        // Obeys SIGTERM (the default action), so the group can die well within the
        // grace period: this is the race window finding 1 fixes. After the group
        // dies but before the final `cancelled` marker lands, a fast poller must
        // read `cancelled` (thanks to the `cancelling` marker written up front),
        // never `lost`.
        let child = Command::new("sleep").arg("10").process_group(0).spawn()?;
        let pid = child.id();
        let reaper = std::thread::spawn(move || {
            let mut child = child;
            child.wait()
        });

        let dir = tempdir()?;
        let cancel_dir = dir.path().display().to_string();
        let canceller = std::thread::spawn(move || {
            Command::new("sh")
                .arg("-c")
                .arg(cancel_script(&cancel_dir, pid, None))
                .output()
        });

        let status_dir = dir.path().display().to_string();
        let mut saw_cancelled = false;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            if std::time::Instant::now() > deadline {
                return Err("timed out waiting for the cancel to finish".into());
            }
            let output = Command::new("sh")
                .arg("-c")
                .arg(status_script(&status_dir, pid))
                .output()?;
            let status = parse_status(&String::from_utf8_lossy(&output.stdout));
            assert_ne!(
                status,
                Some(JobStatus::Lost),
                "status must never read lost while a cancel is in progress"
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
            "{}",
            String::from_utf8_lossy(&cancel_output.stderr)
        );
        reaper.join().map_err(|_| "reaper thread panicked")??;
        Ok(())
    }
}
