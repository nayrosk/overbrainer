//! The pod's command and environment: `bootstrap.sh` with `watchdog.sh`
//! embedded, and the `env` of the create call that both read.

use std::collections::BTreeMap;
use std::time::Duration;

use crate::exec::GROUP_SIGNAL;

/// The bootstrap's functions, embedded in the binary.
const BOOTSTRAP: &str = include_str!("bootstrap.sh");
/// The watchdog, embedded in the binary, without its `group_signal` helper.
const WATCHDOG: &str = include_str!("watchdog.sh");
/// End marker of the here-document carrying the watchdog in the pod's command.
const HEREDOC_END: &str = "OVERBRAINER_WATCHDOG_END";

/// Where the bootstrap writes the jobs' environment on the pod.
pub const JOB_ENV: &str = "/etc/overbrainer/job.env";

/// The create call's variable holding the base64 of the pod's private host key.
pub const HOST_KEY_ENV: &str = "OVERBRAINER_HOST_KEY";

/// The watchdog as the pod runs it: the `group_signal` helper of the job scripts,
/// then `watchdog.sh`.
#[must_use]
pub fn watchdog_script() -> String {
    format!("{GROUP_SIGNAL}{WATCHDOG}")
}

/// The bootstrap's shell functions, without the call that runs them.
#[must_use]
pub fn bootstrap_functions() -> &'static str {
    BOOTSTRAP
}

/// The pod's `cmd`: `bash -c` with the bootstrap, a `write_watchdog` function
/// holding the watchdog in a quoted here-document, and the call to
/// `bootstrap_main`. `write_watchdog` writes to a sibling `.tmp` file and only
/// `mv -f`s it into place once the write itself succeeded, so a write that fails
/// partway (a full disk) can never leave `fail` a truncated watchdog to exec into.
#[must_use]
pub fn pod_command() -> Vec<String> {
    let script = format!(
        "{BOOTSTRAP}\nwrite_watchdog() {{\n  mkdir -p \"$(dirname \"$1\")\" || return 1\n  cat > \"$1.tmp\" <<'{HEREDOC_END}' || return 1\n{}{HEREDOC_END}\n  mv -f \"$1.tmp\" \"$1\"\n}}\n\nbootstrap_main\n",
        watchdog_script()
    );
    vec!["bash".to_string(), "-c".to_string(), script]
}

/// What a pod's bootstrap and watchdog need to know, sent in its `env`.
#[derive(Debug, Clone)]
pub struct PodSettings<'a> {
    /// The run, also the marker that finds the pod again.
    pub run_id: &'a str,
    /// Directory of the run directories on the pod.
    pub workdir: &'a str,
    /// When the watchdog deletes the pod, in Unix seconds; `None` when kept.
    pub deadline_unix: Option<u64>,
    /// How long the watchdog waits for a job to start.
    pub boot_grace: Duration,
    /// How long the watchdog keeps a pod whose ended job was not retrieved.
    pub retrieve_grace: Duration,
    /// `--keep-pod`: once the job exists, the watchdog never deletes the pod
    /// (before that, the boot grace and a failed bootstrap still do).
    pub keep: bool,
    /// Base URL of the Runpod API, for the watchdog.
    pub api_url: &'a str,
    /// The run's client public key.
    pub authorized_key: &'a str,
}

/// The pod's plain `env` (the host key is added by [`CreateEnv`](super::CreateEnv)).
/// It never holds `HF_TOKEN` (the job gets it over SSH) nor the account API key
/// (the pod has its own pod-scoped key).
#[must_use]
pub fn pod_env(settings: &PodSettings<'_>) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    let mut set = |name: &str, value: String| {
        env.insert(name.to_string(), value);
    };
    set("OVERBRAINER_RUN_ID", settings.run_id.to_string());
    set("OVERBRAINER_WORKDIR", settings.workdir.to_string());
    set(
        "OVERBRAINER_RUN_DIR",
        format!("{}/{}", settings.workdir, settings.run_id),
    );
    if let Some(deadline) = settings.deadline_unix {
        set("OVERBRAINER_DEADLINE", deadline.to_string());
    }
    set(
        "OVERBRAINER_BOOT_GRACE",
        settings.boot_grace.as_secs().to_string(),
    );
    set(
        "OVERBRAINER_RETRIEVE_GRACE",
        settings.retrieve_grace.as_secs().to_string(),
    );
    set(
        "OVERBRAINER_KEEP_POD",
        if settings.keep { "1" } else { "0" }.to_string(),
    );
    set("OVERBRAINER_API_URL", settings.api_url.to_string());
    set(
        "OVERBRAINER_AUTHORIZED_KEY",
        settings.authorized_key.to_string(),
    );
    set("OVERBRAINER_VERSION", env!("CARGO_PKG_VERSION").to_string());
    set("JUPYTER_DISABLE", "1".to_string());
    env
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use super::*;

    #[test]
    fn the_command_carries_the_whole_watchdog_and_stays_small() {
        let command = pod_command();
        assert_eq!(command[..2], ["bash", "-c"]);
        let script = &command[2];
        assert!(script.len() < 100 * 1024, "{} bytes", script.len());
        assert!(script.contains(&watchdog_script()));
        assert!(script.ends_with("\nbootstrap_main\n"));
        assert!(
            !WATCHDOG.lines().any(|line| line == HEREDOC_END),
            "the watchdog would end its own here-document"
        );
    }

    #[test]
    fn every_shell_parses_the_command() {
        let script = &pod_command()[2];
        for shell in [&["sh"][..], &["dash"], &["busybox", "sh"], &["bash"]] {
            let Ok(output) = Command::new(shell[0])
                .args(&shell[1..])
                .arg("-n")
                .arg("-c")
                .arg(script)
                .output()
            else {
                eprintln!("skipped: {} is not installed", shell.join(" "));
                continue;
            };
            assert!(
                output.status.success(),
                "{shell:?}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    #[test]
    fn the_env_holds_the_watchdog_settings_and_no_secret() {
        let settings = PodSettings {
            run_id: "r1",
            workdir: "/workspace/overbrainer",
            deadline_unix: Some(1_790_021_600),
            boot_grace: Duration::from_secs(1800),
            retrieve_grace: Duration::from_secs(3600),
            keep: false,
            api_url: "https://api.runpod.io/v2",
            authorized_key: "ssh-ed25519 AAAAclient overbrainer-r1",
        };
        let env = pod_env(&settings);
        let get = |name: &str| env.get(name).map(String::as_str);
        assert_eq!(
            get("OVERBRAINER_RUN_DIR"),
            Some("/workspace/overbrainer/r1")
        );
        assert_eq!(get("OVERBRAINER_DEADLINE"), Some("1790021600"));
        assert_eq!(get("OVERBRAINER_BOOT_GRACE"), Some("1800"));
        assert_eq!(get("OVERBRAINER_KEEP_POD"), Some("0"));
        assert_eq!(get("JUPYTER_DISABLE"), Some("1"));
        assert!(!env.contains_key(HOST_KEY_ENV));
        assert!(!env.contains_key("HF_TOKEN"));
        assert!(!env.contains_key("PUBLIC_KEY"));
        assert!(env.len() <= 50);
        let kept = pod_env(&PodSettings {
            keep: true,
            deadline_unix: None,
            ..settings
        });
        assert_eq!(
            kept.get("OVERBRAINER_KEEP_POD").map(String::as_str),
            Some("1")
        );
        assert!(!kept.contains_key("OVERBRAINER_DEADLINE"));
    }
}
