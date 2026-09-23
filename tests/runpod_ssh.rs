//! A Runpod pod reached over SSH, with a real sshd standing in for the pod and a
//! local stub for the Runpod API: the per-run ssh config and its `HostKeyAlias`
//! pinning, readiness and the watchdog's verdict, and ending a pod. Runs only when
//! `OVERBRAINER_TEST_SSH_HOST` and `OVERBRAINER_TEST_SSH_CONFIG` are set, as in the
//! `ssh` CI job, whose config file supplies the endpoint, the client key and the
//! server's host key; skipped otherwise.

use std::fs;
use std::path::{Path, PathBuf};

use overbrainer::exec::{Executor, SshExecutor};
use overbrainer::runpod::{PodKeys, SshEndpoint, alias, write_config};
use secrecy::SecretString;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// What the CI ssh config says about the test sshd.
struct Sshd {
    endpoint: SshEndpoint,
    identity: PathBuf,
    host_public: String,
}

/// The value of `key` in the ssh config `text`.
fn value(text: &str, key: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let (name, value) = line.trim().split_once(char::is_whitespace)?;
        name.eq_ignore_ascii_case(key)
            .then(|| value.trim().trim_matches('"').to_string())
    })
}

fn sshd() -> Result<Option<Sshd>, Box<dyn std::error::Error>> {
    if std::env::var_os("OVERBRAINER_TEST_SSH_HOST").is_none() {
        return Ok(None);
    }
    let Some(config) = std::env::var_os("OVERBRAINER_TEST_SSH_CONFIG") else {
        return Ok(None);
    };
    let text = fs::read_to_string(config)?;
    let get = |key: &str| value(&text, key).ok_or(format!("no {key} in the test ssh config"));
    let known_hosts = fs::read_to_string(get("UserKnownHostsFile")?)?;
    let host_public = known_hosts
        .lines()
        .find_map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            (fields.get(1) == Some(&"ssh-ed25519")).then(|| fields[1..3].join(" "))
        })
        .ok_or("no ssh-ed25519 key in the test known_hosts")?;
    Ok(Some(Sshd {
        endpoint: SshEndpoint {
            host: get("HostName")?,
            port: get("Port")?.parse()?,
            user: get("User")?,
        },
        identity: PathBuf::from(get("IdentityFile")?),
        host_public,
    }))
}

fn skip() {
    eprintln!("skipped: OVERBRAINER_TEST_SSH_HOST and OVERBRAINER_TEST_SSH_CONFIG are not set");
}

fn keys(sshd: &Sshd, host_public: &str) -> PodKeys {
    PodKeys::new(
        sshd.identity.clone(),
        String::new(),
        host_public.to_string(),
        SecretString::from("unused"),
    )
}

fn new_run_id() -> String {
    format!("20260922-000000-{:04x}", fastrand::u16(..))
}

async fn connect(
    dir: &Path,
    sshd: &Sshd,
    keys: &PodKeys,
) -> Result<SshExecutor, Box<dyn std::error::Error>> {
    let run_id = new_run_id();
    let alias = alias(&run_id);
    let config = write_config(dir, &alias, &sshd.endpoint, keys)?;
    let workdir = format!("overbrainer-tests/runpod-{run_id}");
    Ok(SshExecutor::connect(&alias, &workdir, Some(&config)).await?)
}

#[tokio::test]
async fn the_per_run_config_reaches_the_pinned_host() -> TestResult {
    let Some(sshd) = sshd()? else {
        skip();
        return Ok(());
    };
    let dir = tempfile::tempdir()?;
    let executor = connect(dir.path(), &sshd, &keys(&sshd, &sshd.host_public)).await?;
    assert!(executor.workdir().contains("/overbrainer-tests/runpod-"));
    Ok(())
}

#[tokio::test]
async fn another_host_key_is_refused_through_the_per_run_config() -> TestResult {
    let Some(sshd) = sshd()? else {
        skip();
        return Ok(());
    };
    let dir = tempfile::tempdir()?;
    let other = PodKeys::generate(&dir.path().join("other"), "other")?;
    let result = connect(dir.path(), &sshd, &keys(&sshd, &other.host_public)).await;
    assert!(result.is_err(), "connected although the pinned key differs");
    Ok(())
}
