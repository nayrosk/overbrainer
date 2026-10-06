//! overbrainer: distill a parent LLM into a smaller child model.

pub mod cli;
pub mod compare;
pub mod config;
pub mod dataset;
pub mod dedup;
pub mod events;
pub mod exec;
pub mod export;
pub mod history;
pub mod hub;
pub mod llm;
pub mod logging;
pub mod metrics;
pub mod pipeline;
pub mod pricing;
pub mod project_format;
pub mod project_lock;
pub mod prompts;
pub mod retry;
pub mod runpod;
pub mod runs;
pub mod secrets;
pub mod system;
pub mod train;
pub mod tui;
pub mod update;

/// What the unit tests share across modules.
#[cfg(test)]
pub(crate) mod test_support {
    /// Held by every test that sends a real signal to the test process. All unit
    /// tests run in one process, so a signal one test raises reaches the
    /// listeners of another running at the same time and breaks its timing.
    /// Bind it to a name (`let _signals = SIGNALS.lock().await;`): `_` would
    /// release it at once.
    pub(crate) static SIGNALS: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    /// Makes `path` executable.
    pub(crate) fn make_executable(path: &std::path::Path) -> std::io::Result<()> {
        use std::os::unix::fs::PermissionsExt as _;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
    }

    /// Whether `sh` runs here.
    pub(crate) fn sh_available() -> bool {
        std::process::Command::new("sh")
            .arg("-c")
            .arg(":")
            .status()
            .is_ok_and(|status| status.success())
    }

    /// Whether `sh`, `python3` and `tar` run here: what the job scripts need.
    pub(crate) fn tools_available() -> bool {
        sh_available()
            && ["python3", "tar"].iter().all(|program| {
                std::process::Command::new(program)
                    .arg("--version")
                    .output()
                    .is_ok_and(|output| output.status.success())
            })
    }

    /// `tar -czf archive -C dir top`.
    pub(crate) fn tar(
        dir: &std::path::Path,
        archive: &std::path::Path,
        top: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let status = std::process::Command::new("tar")
            .arg("-czf")
            .arg(archive)
            .arg("-C")
            .arg(dir)
            .arg(top)
            .status()?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("tar failed: {status}").into())
        }
    }

    /// The stdout then the stderr of `output`, as text.
    pub(crate) fn text(output: &std::process::Output) -> String {
        format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    }
}
