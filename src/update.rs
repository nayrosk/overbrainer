//! Looks up the latest overbrainer release on crates.io, at most once a day.
//!
//! Every failure (network, parse, cache) is logged at debug and gives no answer:
//! the check must never get in the way of the command it runs beside.

use std::cmp::Ordering;
use std::error::Error;
use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

/// The crates.io API entry for this crate.
pub const CRATES_IO_URL: &str = "https://crates.io/api/v1/crates/overbrainer";

/// crates.io asks every client for a contact.
const USER_AGENT: &str = concat!(
    "overbrainer/",
    env!("CARGO_PKG_VERSION"),
    " (https://github.com/nayrosk/overbrainer)"
);
const TIMEOUT: Duration = Duration::from_secs(3);
/// A cache younger than this skips the request.
const MAX_AGE: Duration = Duration::from_secs(24 * 3600);
const CACHE_FILE: &str = "latest-version.json";
/// The most the check reads from crates.io (its answer is a few KiB).
const MAX_BODY: usize = 64 * 1024;
/// The most the check reads from its cache file.
const MAX_CACHE: usize = 4096;
/// The longest version the check accepts.
const MAX_VERSION: usize = 64;

/// A release newer than the running binary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Newer {
    pub latest: String,
    pub current: &'static str,
}

impl fmt::Display for Newer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "overbrainer {} is available (you have {}): cargo install overbrainer",
            self.latest, self.current
        )
    }
}

/// What the check reads from the environment.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CheckEnv {
    /// `OVERBRAINER_NO_UPDATE_CHECK` is set and not empty.
    pub disabled: bool,
    /// Where the cache file lives; `None` fetches every time.
    pub cache_dir: Option<PathBuf>,
}

impl CheckEnv {
    /// Reads `OVERBRAINER_NO_UPDATE_CHECK`, `XDG_CACHE_HOME` and `HOME`.
    #[must_use]
    pub fn from_process() -> Self {
        Self::from_vars(|name| std::env::var_os(name))
    }

    fn from_vars(var: impl Fn(&str) -> Option<OsString>) -> Self {
        let set = |name| var(name).filter(|value| !value.is_empty());
        // The XDG spec says to ignore a relative XDG_CACHE_HOME.
        let base = set("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .filter(|dir| dir.is_absolute())
            .or_else(|| set("HOME").map(|home| PathBuf::from(home).join(".cache")));
        Self {
            disabled: set("OVERBRAINER_NO_UPDATE_CHECK").is_some(),
            cache_dir: base.map(|dir| dir.join("overbrainer")),
        }
    }
}

/// Whether `version` is short and only `[0-9A-Za-z.+-]`: anything else, such
/// as an escape sequence, is never printed.
fn plain_version(version: &str) -> bool {
    version.len() <= MAX_VERSION
        && version
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b".+-".contains(&byte))
}

/// `major.minor.patch` of a version, and whether it carries a pre-release.
fn parse(version: &str) -> Option<([u64; 3], bool)> {
    let version = version.split('+').next().unwrap_or_default();
    let (numbers, pre) = match version.split_once('-') {
        Some((numbers, _)) => (numbers, true),
        None => (version, false),
    };
    let mut parts = numbers.split('.').map(|part| part.parse::<u64>().ok());
    let triple = [parts.next()??, parts.next()??, parts.next()??];
    parts.next().is_none().then_some((triple, pre))
}

/// Whether `latest` is a newer release than `current`. Versions compare by
/// their numbers; on a tie a release beats a pre-release. An unparsable version
/// is never newer.
#[must_use]
pub fn is_newer(latest: &str, current: &str) -> bool {
    let (Some((latest, latest_pre)), Some((current, current_pre))) =
        (parse(latest), parse(current))
    else {
        return false;
    };
    match latest.cmp(&current) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => current_pre && !latest_pre,
    }
}

/// The latest release when it is newer than this binary, from a cache younger
/// than a day or else from `url`, a crates.io crate endpoint.
pub async fn check(env: &CheckEnv, url: &str, now: SystemTime) -> Option<Newer> {
    let current = env!("CARGO_PKG_VERSION");
    let latest = latest(env, url, now, TIMEOUT).await?;
    is_newer(&latest, current).then_some(Newer { latest, current })
}

#[derive(Serialize, Deserialize)]
struct Cache {
    /// Unix seconds.
    checked_at: u64,
    latest: String,
}

async fn latest(env: &CheckEnv, url: &str, now: SystemTime, timeout: Duration) -> Option<String> {
    if env.disabled {
        return None;
    }
    let dir = env.cache_dir.as_deref();
    if let Some(latest) = dir.and_then(|dir| read_cache(dir, now)) {
        return Some(latest);
    }
    let latest = fetch(url, timeout)
        .await
        .map_err(|error| tracing::debug!("update check: {error}"))
        .ok()?;
    if let Some(dir) = dir {
        write_cache(dir, now, &latest);
    }
    Some(latest)
}

/// The cached version when the cache is readable and younger than a day.
/// Anything but a regular file, or a file past [`MAX_CACHE`], is ignored.
fn read_cache(dir: &Path, now: SystemTime) -> Option<String> {
    let path = dir.join(CACHE_FILE);
    let text = read_small(&path)
        .map_err(|error| tracing::debug!("update check: cannot read {}: {error}", path.display()))
        .ok()?;
    let cache: Cache = serde_json::from_str(&text)
        .map_err(|error| tracing::debug!("update check: cannot parse {}: {error}", path.display()))
        .ok()?;
    let checked_at = SystemTime::UNIX_EPOCH.checked_add(Duration::from_secs(cache.checked_at))?;
    // A cache from the future (a clock set back) fails here and is refreshed.
    let age = now.duration_since(checked_at).ok()?;
    (age < MAX_AGE && plain_version(&cache.latest)).then_some(cache.latest)
}

fn read_small(path: &Path) -> std::io::Result<String> {
    use rustix::fs::{Mode, OFlags};
    use std::io::{Error, Read as _};
    // NOFOLLOW refuses a symlink; NONBLOCK keeps a FIFO from hanging the open.
    let flags = OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC;
    let file = std::fs::File::from(rustix::fs::open(path, flags, Mode::empty())?);
    if !file.metadata()?.is_file() {
        return Err(Error::other("not a regular file"));
    }
    let mut text = String::new();
    file.take(MAX_CACHE as u64 + 1).read_to_string(&mut text)?;
    if text.len() > MAX_CACHE {
        return Err(Error::other(format!("larger than {MAX_CACHE} bytes")));
    }
    Ok(text)
}

fn write_cache(dir: &Path, now: SystemTime, latest: &str) {
    let checked_at = now
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    let cache = Cache {
        checked_at,
        latest: latest.to_owned(),
    };
    let written = serde_json::to_vec(&cache)
        .map_err(|error| error.to_string())
        .and_then(|json| {
            std::fs::create_dir_all(dir).map_err(|error| error.to_string())?;
            crate::runs::write_atomic(dir, CACHE_FILE, &json).map_err(|error| error.to_string())
        });
    if let Err(error) = written {
        tracing::debug!(
            "update check: cannot write the cache in {}: {error}",
            dir.display()
        );
    }
}

#[derive(Deserialize)]
struct CrateResponse {
    #[serde(rename = "crate")]
    krate: CrateInfo,
}

#[derive(Deserialize)]
struct CrateInfo {
    max_stable_version: String,
}

async fn fetch(url: &str, timeout: Duration) -> Result<String, Box<dyn Error + Send + Sync>> {
    let client = reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(timeout)
        // Only crates.io answers: a redirect is a failure, never followed.
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let mut response = client.get(url).send().await?.error_for_status()?;
    if !response.status().is_success() {
        return Err(format!("unexpected status {}", response.status()).into());
    }
    let too_large = || format!("the answer is larger than {MAX_BODY} bytes");
    if response
        .content_length()
        .is_some_and(|length| length > MAX_BODY as u64)
    {
        return Err(too_large().into());
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if body.len() + chunk.len() > MAX_BODY {
            return Err(too_large().into());
        }
        body.extend_from_slice(&chunk);
    }
    let body: CrateResponse = serde_json::from_slice(&body)?;
    let version = body.krate.max_stable_version;
    if !plain_version(&version) {
        return Err(format!("not a version: {version:?}").into());
    }
    Ok(version)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    const HOUR: Duration = Duration::from_secs(3600);
    /// A trimmed crates.io answer: the newest version is a pre-release.
    const BODY: &str = r#"{"crate":{"id":"overbrainer","name":"overbrainer",
        "max_version":"0.5.0-rc.1","max_stable_version":"0.4.2",
        "newest_version":"0.5.0-rc.1","downloads":1234},
        "versions":[{"num":"0.5.0-rc.1"},{"num":"0.4.2"}]}"#;

    fn now() -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(1_800_000_000)
    }

    fn unix(time: SystemTime) -> u64 {
        time.duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0, |since| since.as_secs())
    }

    fn env(dir: &Path) -> CheckEnv {
        CheckEnv {
            disabled: false,
            cache_dir: Some(dir.to_path_buf()),
        }
    }

    fn write_cache(dir: &Path, checked_at: SystemTime, latest: &str) -> std::io::Result<()> {
        let json = serde_json::json!({"checked_at": unix(checked_at), "latest": latest});
        std::fs::write(dir.join(CACHE_FILE), json.to_string())
    }

    fn read_cache(dir: &Path) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(dir.join(CACHE_FILE))?;
        Ok(serde_json::from_str(&text)?)
    }

    async fn serve(template: ResponseTemplate, expected: u64) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/v1/crates/overbrainer"))
            .respond_with(template)
            .expect(expected)
            .mount(&server)
            .await;
        server
    }

    fn url(server: &MockServer) -> String {
        format!("{}/api/v1/crates/overbrainer", server.uri())
    }

    #[test]
    fn versions_compare_by_their_numbers() {
        for (latest, current, newer) in [
            ("0.4.1", "0.4.0", true),
            ("0.4.0", "0.4.0", false),
            ("0.3.9", "0.4.0", false),
            ("0.10.0", "0.9.9", true),
            ("1.0.0", "0.99.99", true),
            ("0.4.0", "0.4.0-rc.1", true),
            ("0.4.1", "0.4.0-rc.1", true),
            ("0.3.9", "0.4.0-rc.1", false),
            ("0.4.0+build.7", "0.4.0", false),
            ("garbage", "0.4.0", false),
            ("0.4.1", "garbage", false),
            ("0.4", "0.3.0", false),
            ("0.4.1.2", "0.4.0", false),
            ("", "0.4.0", false),
        ] {
            assert_eq!(is_newer(latest, current), newer, "{latest} vs {current}");
        }
    }

    #[test]
    fn the_message_names_both_versions_and_the_install_command() {
        let newer = Newer {
            latest: "0.4.1".into(),
            current: "0.4.0",
        };
        assert_eq!(
            newer.to_string(),
            "overbrainer 0.4.1 is available (you have 0.4.0): cargo install overbrainer"
        );
    }

    #[test]
    fn the_environment_picks_the_cache_directory_and_the_opt_out() {
        let vars = |pairs: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                pairs
                    .iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, value)| OsString::from(value))
            }
        };
        let xdg = CheckEnv::from_vars(vars(&[("XDG_CACHE_HOME", "/x"), ("HOME", "/h")]));
        assert_eq!(xdg.cache_dir, Some(PathBuf::from("/x/overbrainer")));
        assert!(!xdg.disabled);
        let home = CheckEnv::from_vars(vars(&[("XDG_CACHE_HOME", ""), ("HOME", "/h")]));
        assert_eq!(home.cache_dir, Some(PathBuf::from("/h/.cache/overbrainer")));
        let relative = CheckEnv::from_vars(vars(&[("XDG_CACHE_HOME", "rel"), ("HOME", "/h")]));
        assert_eq!(
            relative.cache_dir,
            Some(PathBuf::from("/h/.cache/overbrainer"))
        );
        assert_eq!(CheckEnv::from_vars(vars(&[])).cache_dir, None);
        assert!(CheckEnv::from_vars(vars(&[("OVERBRAINER_NO_UPDATE_CHECK", "1")])).disabled);
        assert!(!CheckEnv::from_vars(vars(&[("OVERBRAINER_NO_UPDATE_CHECK", "")])).disabled);
    }

    #[tokio::test]
    async fn a_fresh_cache_answers_without_a_request() -> TestResult {
        let dir = tempfile::tempdir()?;
        write_cache(dir.path(), now() - HOUR, "0.4.7")?;
        let server = serve(ResponseTemplate::new(200).set_body_string(BODY), 0).await;
        let found = latest(&env(dir.path()), &url(&server), now(), TIMEOUT).await;
        assert_eq!(found.as_deref(), Some("0.4.7"));
        Ok(())
    }

    #[tokio::test]
    async fn a_stale_cache_is_fetched_again_and_rewritten() -> TestResult {
        let dir = tempfile::tempdir()?;
        write_cache(dir.path(), now() - 25 * HOUR, "0.4.0")?;
        let server = serve(ResponseTemplate::new(200).set_body_string(BODY), 1).await;
        let found = latest(&env(dir.path()), &url(&server), now(), TIMEOUT).await;
        assert_eq!(found.as_deref(), Some("0.4.2"));
        let cache = read_cache(dir.path())?;
        assert_eq!(cache["latest"], "0.4.2");
        assert_eq!(cache["checked_at"], unix(now()));
        Ok(())
    }

    #[tokio::test]
    async fn a_missing_cache_is_fetched_and_its_directory_created() -> TestResult {
        let dir = tempfile::tempdir()?;
        let nested = dir.path().join("cache").join("overbrainer");
        let server = serve(ResponseTemplate::new(200).set_body_string(BODY), 1).await;
        let found = latest(&env(&nested), &url(&server), now(), TIMEOUT).await;
        assert_eq!(found.as_deref(), Some("0.4.2"));
        assert_eq!(read_cache(&nested)?["latest"], "0.4.2");
        let leftovers = std::fs::read_dir(&nested)?.count();
        assert_eq!(leftovers, 1, "only the cache file remains");
        Ok(())
    }

    #[tokio::test]
    async fn a_cache_that_does_not_parse_is_fetched() -> TestResult {
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join(CACHE_FILE), "not json")?;
        let server = serve(ResponseTemplate::new(200).set_body_string(BODY), 1).await;
        let found = latest(&env(dir.path()), &url(&server), now(), TIMEOUT).await;
        assert_eq!(found.as_deref(), Some("0.4.2"));
        assert_eq!(read_cache(dir.path())?["latest"], "0.4.2");
        Ok(())
    }

    #[tokio::test]
    async fn a_cache_that_cannot_be_read_is_fetched() -> TestResult {
        let dir = tempfile::tempdir()?;
        std::fs::create_dir(dir.path().join(CACHE_FILE))?;
        let server = serve(ResponseTemplate::new(200).set_body_string(BODY), 1).await;
        let found = latest(&env(dir.path()), &url(&server), now(), TIMEOUT).await;
        assert_eq!(found.as_deref(), Some("0.4.2"));
        Ok(())
    }

    #[tokio::test]
    async fn an_oversized_cache_is_fetched() -> TestResult {
        let dir = tempfile::tempdir()?;
        let padding = " ".repeat(8192);
        let json = format!(
            r#"{{"checked_at": {}, {padding}"latest": "0.4.7"}}"#,
            unix(now())
        );
        std::fs::write(dir.path().join(CACHE_FILE), json)?;
        let server = serve(ResponseTemplate::new(200).set_body_string(BODY), 1).await;
        let found = latest(&env(dir.path()), &url(&server), now(), TIMEOUT).await;
        assert_eq!(found.as_deref(), Some("0.4.2"));
        Ok(())
    }

    #[tokio::test]
    async fn a_huge_checked_at_is_fetched_without_panicking() -> TestResult {
        let dir = tempfile::tempdir()?;
        let json = format!(r#"{{"checked_at": {}, "latest": "0.4.7"}}"#, u64::MAX);
        std::fs::write(dir.path().join(CACHE_FILE), json)?;
        let server = serve(ResponseTemplate::new(200).set_body_string(BODY), 1).await;
        let found = latest(&env(dir.path()), &url(&server), now(), TIMEOUT).await;
        assert_eq!(found.as_deref(), Some("0.4.2"));
        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_symlinked_cache_is_ignored_and_replaced() -> TestResult {
        let dir = tempfile::tempdir()?;
        let target = dir.path().join("elsewhere.json");
        let json = serde_json::json!({"checked_at": unix(now()), "latest": "0.4.7"}).to_string();
        std::fs::write(&target, &json)?;
        std::os::unix::fs::symlink(&target, dir.path().join(CACHE_FILE))?;
        let server = serve(ResponseTemplate::new(200).set_body_string(BODY), 1).await;
        let found = latest(&env(dir.path()), &url(&server), now(), TIMEOUT).await;
        assert_eq!(found.as_deref(), Some("0.4.2"));
        assert_eq!(
            std::fs::read_to_string(&target)?,
            json,
            "the target is untouched"
        );
        let written = std::fs::symlink_metadata(dir.path().join(CACHE_FILE))?;
        assert!(written.is_file(), "the symlink was replaced by a file");
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn a_fifo_cache_is_fetched_without_blocking() -> TestResult {
        let dir = tempfile::tempdir()?;
        rustix::fs::mkfifoat(
            rustix::fs::CWD,
            dir.path().join(CACHE_FILE),
            rustix::fs::Mode::from_raw_mode(0o600),
        )?;
        let server = serve(ResponseTemplate::new(200).set_body_string(BODY), 1).await;
        let found = latest(&env(dir.path()), &url(&server), now(), TIMEOUT).await;
        assert_eq!(found.as_deref(), Some("0.4.2"));
        Ok(())
    }

    #[tokio::test]
    async fn a_cache_from_the_future_is_fetched_again() -> TestResult {
        let dir = tempfile::tempdir()?;
        write_cache(dir.path(), now() + HOUR, "0.4.0")?;
        let server = serve(ResponseTemplate::new(200).set_body_string(BODY), 1).await;
        let found = latest(&env(dir.path()), &url(&server), now(), TIMEOUT).await;
        assert_eq!(found.as_deref(), Some("0.4.2"));
        Ok(())
    }

    #[tokio::test]
    async fn no_cache_directory_still_fetches() {
        let server = serve(ResponseTemplate::new(200).set_body_string(BODY), 1).await;
        let found = latest(&CheckEnv::default(), &url(&server), now(), TIMEOUT).await;
        assert_eq!(found.as_deref(), Some("0.4.2"));
    }

    #[tokio::test]
    async fn the_request_names_the_client() -> TestResult {
        let dir = tempfile::tempdir()?;
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(header("user-agent", USER_AGENT))
            .respond_with(ResponseTemplate::new(200).set_body_string(BODY))
            .expect(1)
            .mount(&server)
            .await;
        let found = latest(&env(dir.path()), &url(&server), now(), TIMEOUT).await;
        assert_eq!(found.as_deref(), Some("0.4.2"));
        assert!(USER_AGENT.starts_with(concat!("overbrainer/", env!("CARGO_PKG_VERSION"), " (")));
        Ok(())
    }

    #[tokio::test]
    async fn failures_give_nothing_and_leave_no_cache() -> TestResult {
        let cases = [
            ResponseTemplate::new(500),
            ResponseTemplate::new(200).set_body_string("{\"crate\":"),
            ResponseTemplate::new(200).set_body_string("{\"crate\":{}}"),
            ResponseTemplate::new(200)
                .set_body_string(BODY)
                .set_delay(Duration::from_millis(500)),
        ];
        for template in cases {
            let dir = tempfile::tempdir()?;
            let server = serve(template, 1).await;
            let timeout = Duration::from_millis(100);
            let found = latest(&env(dir.path()), &url(&server), now(), timeout).await;
            assert_eq!(found, None);
            assert!(!dir.path().join(CACHE_FILE).exists());
        }
        Ok(())
    }

    #[tokio::test]
    async fn a_version_that_is_not_plain_text_is_unknown() -> TestResult {
        let escape = "9.9.9-\u{1b}[2J";
        let long = format!("9.9.9-{}", "a".repeat(100));
        for version in [escape, long.as_str()] {
            let body = serde_json::json!({"crate": {"max_stable_version": version}}).to_string();
            let dir = tempfile::tempdir()?;
            let server = serve(ResponseTemplate::new(200).set_body_string(body), 1).await;
            let found = latest(&env(dir.path()), &url(&server), now(), TIMEOUT).await;
            assert_eq!(found, None, "{version:?} from crates.io");
            let dir = tempfile::tempdir()?;
            write_cache(dir.path(), now() - HOUR, version)?;
            let server = serve(ResponseTemplate::new(500), 1).await;
            let found = latest(&env(dir.path()), &url(&server), now(), TIMEOUT).await;
            assert_eq!(found, None, "{version:?} from the cache");
        }
        let longest = format!("9.9.9-{}", "a".repeat(58));
        assert!(plain_version(&longest));
        assert!(!plain_version(&format!("{longest}a")));
        Ok(())
    }

    #[tokio::test]
    async fn a_redirect_is_not_followed() -> TestResult {
        let server = MockServer::start().await;
        let elsewhere = format!("{}/elsewhere", server.uri());
        Mock::given(method("GET"))
            .and(path("/api/v1/crates/overbrainer"))
            .respond_with(ResponseTemplate::new(302).insert_header("location", elsewhere.as_str()))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/elsewhere"))
            .respond_with(ResponseTemplate::new(200).set_body_string(BODY))
            .expect(0)
            .mount(&server)
            .await;
        let found = fetch(&url(&server), TIMEOUT).await;
        assert!(found.is_err(), "{found:?}");
        Ok(())
    }

    #[test]
    fn the_check_can_be_spawned() {
        fn spawnable<F: std::future::Future + Send + 'static>(_: F) {}
        let env = CheckEnv::default();
        spawnable(async move { check(&env, CRATES_IO_URL, now()).await });
    }

    #[tokio::test]
    async fn an_oversized_body_gives_nothing() -> TestResult {
        let padding = " ".repeat(MAX_BODY + 1);
        let big = format!(r#"{{"crate":{{"max_stable_version":"0.4.2"}}{padding}}}"#);
        let dir = tempfile::tempdir()?;
        let server = serve(ResponseTemplate::new(200).set_body_string(big), 1).await;
        let found = latest(&env(dir.path()), &url(&server), now(), TIMEOUT).await;
        assert_eq!(found, None);
        Ok(())
    }

    #[tokio::test]
    async fn an_oversized_body_without_a_length_gives_nothing() -> TestResult {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let body = format!(
            r#"{{"crate":{{"max_stable_version":"0.4.2"}}{}}}"#,
            " ".repeat(MAX_BODY + 1)
        );
        let serving = tokio::spawn(async move {
            use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
            let (mut stream, _) = listener.accept().await?;
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request).await?;
            let head =
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n";
            stream.write_all(head.as_bytes()).await?;
            // No length: the body ends when the connection closes.
            let _ = stream.write_all(body.as_bytes()).await;
            std::io::Result::Ok(())
        });
        let found = fetch(&format!("http://{address}/"), TIMEOUT).await;
        assert!(found.is_err(), "{found:?}");
        serving.abort();
        Ok(())
    }

    #[tokio::test]
    async fn an_unreachable_server_gives_nothing() {
        let found = check(&CheckEnv::default(), "http://127.0.0.1:1/", now()).await;
        assert_eq!(found, None);
    }

    #[tokio::test]
    async fn disabled_makes_no_request() -> TestResult {
        let dir = tempfile::tempdir()?;
        let server = serve(ResponseTemplate::new(200).set_body_string(BODY), 0).await;
        let disabled = CheckEnv {
            disabled: true,
            cache_dir: Some(dir.path().to_path_buf()),
        };
        assert_eq!(check(&disabled, &url(&server), now()).await, None);
        assert!(!dir.path().join(CACHE_FILE).exists());
        Ok(())
    }

    #[tokio::test]
    async fn check_reports_only_a_newer_release() -> TestResult {
        let dir = tempfile::tempdir()?;
        let server = serve(ResponseTemplate::new(200).set_body_string(BODY), 0).await;
        write_cache(dir.path(), now() - HOUR, "999.0.0")?;
        let newer = check(&env(dir.path()), &url(&server), now()).await;
        assert_eq!(
            newer,
            Some(Newer {
                latest: "999.0.0".into(),
                current: env!("CARGO_PKG_VERSION"),
            })
        );
        write_cache(dir.path(), now() - HOUR, "0.0.1")?;
        assert_eq!(check(&env(dir.path()), &url(&server), now()).await, None);
        Ok(())
    }
}
