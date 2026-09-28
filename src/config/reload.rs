//! Reading `overbrainer.toml` and `.env` again while overbrainer runs: which
//! keys `.env` set at start, the environment a reload uses, and a stamp of both
//! files that tells when they changed.
//!
//! `.env` was loaded into the process environment at start, and nothing may
//! write it again (`set_var` is never called). A reload therefore builds its
//! own environment: the process one without the keys `.env` set, plus what
//! `.env` holds now, as [`EnvSource::Vars`].

use std::collections::BTreeSet;
use std::io;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use super::{CONFIG_FILE, ConfigError, EnvSource, Settings, load_str};

/// Name of the file of environment variables, in the project directory.
pub const DOTENV_FILE: &str = ".env";

/// The keys `.env` set in the process environment at start: those it holds
/// that the shell did not export (`dotenvy` never overrides a variable).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DotenvKeys(BTreeSet<String>);

impl DotenvKeys {
    /// The keys of the `.env` at `path` that `process`, the environment before
    /// `.env` is loaded, does not hold. A missing file sets none.
    ///
    /// # Errors
    ///
    /// Returns why `.env` cannot be read or parsed; the message never holds
    /// its content.
    pub fn record(
        path: &Path,
        process: impl Iterator<Item = (String, String)>,
    ) -> Result<Self, DotenvError> {
        let exported: BTreeSet<String> = process.map(|(key, _)| key).collect();
        Ok(Self(
            read_dotenv(path)?
                .into_iter()
                .map(|(key, _)| key)
                .filter(|key| !exported.contains(key))
                .collect(),
        ))
    }

    /// Whether `.env` set `key` at start.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.0.contains(key)
    }
}

/// Why `.env` cannot be used. Never holds its content: a line of `.env`
/// usually holds a secret.
#[derive(Debug, thiserror::Error)]
pub enum DotenvError {
    /// A syntax error, on this line when it is known.
    #[error("cannot parse .env (syntax error{})", line.map(|line| format!(" at line {line}")).unwrap_or_default())]
    Syntax {
        /// The line, from 1.
        line: Option<usize>,
    },
    /// The file cannot be read.
    #[error("cannot read .env: {0}")]
    Read(io::Error),
}

/// Why a reload keeps the settings it had.
#[derive(Debug, thiserror::Error)]
pub enum ReloadError {
    /// `overbrainer.toml` cannot be read or does not validate.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// `.env` cannot be read or parsed.
    #[error(transparent)]
    Dotenv(#[from] DotenvError),
}

/// The configuration read again.
#[derive(Debug)]
pub struct Reloaded {
    /// The text of `overbrainer.toml`.
    pub text: String,
    /// Its settings, layered with `env`.
    pub settings: Settings,
    /// The environment they were read with.
    pub env: EnvSource,
}

/// The environment a reload uses: every variable of `process` but the keys
/// `.env` set at start, then each entry of `current`, the `.env` of now, whose
/// key the shell does not export. The first entry of a key repeated in
/// `current` wins, as when `.env` is loaded.
#[must_use]
pub fn merged_env(
    process: impl Iterator<Item = (String, String)>,
    dotenv: &DotenvKeys,
    current: Vec<(String, String)>,
) -> EnvSource {
    let mut vars: Vec<(String, String)> =
        process.filter(|(key, _)| !dotenv.contains(key)).collect();
    let mut taken: BTreeSet<String> = vars.iter().map(|(key, _)| key.clone()).collect();
    for (key, value) in current {
        if taken.insert(key.clone()) {
            vars.push((key, value));
        }
    }
    EnvSource::Vars(vars)
}

/// What tells a change of `overbrainer.toml` or `.env`: the modification
/// time and size of each, `None` for a file that is not there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stamp {
    config: Option<(SystemTime, u64)>,
    dotenv: Option<(SystemTime, u64)>,
}

/// The [`Stamp`] of the files of the project in `project_dir` now.
#[must_use]
pub fn stamp(project_dir: &Path) -> Stamp {
    let of = |name: &str| {
        let metadata = std::fs::metadata(project_dir.join(name)).ok()?;
        Some((metadata.modified().unwrap_or(UNIX_EPOCH), metadata.len()))
    };
    Stamp {
        config: of(CONFIG_FILE),
        dotenv: of(DOTENV_FILE),
    }
}

/// Reads `overbrainer.toml` and `.env` of `project_dir` again, with the process
/// environment merged as [`merged_env`] does.
///
/// # Errors
///
/// Returns [`ReloadError::Dotenv`] when `.env` cannot be read or parsed, and
/// [`ReloadError::Config`] when `overbrainer.toml` cannot be read or does not
/// validate.
pub fn reload(project_dir: &Path, dotenv: &DotenvKeys) -> Result<Reloaded, ReloadError> {
    reload_from(project_dir, dotenv, process_vars())
}

/// [`reload`] with `process` as the process environment.
fn reload_from(
    project_dir: &Path,
    dotenv: &DotenvKeys,
    process: impl Iterator<Item = (String, String)>,
) -> Result<Reloaded, ReloadError> {
    let current = read_dotenv(&project_dir.join(DOTENV_FILE))?;
    let env = merged_env(process, dotenv, current);
    let path = project_dir.join(CONFIG_FILE);
    let text = std::fs::read_to_string(&path).map_err(|error| ConfigError::Read { path, error })?;
    let settings = load_str(&text, env.clone())?;
    Ok(Reloaded {
        text,
        settings,
        env,
    })
}

/// The process environment, as [`EnvSource::Process`] reads it: variables
/// whose name or value is not UTF-8 are left out.
fn process_vars() -> impl Iterator<Item = (String, String)> {
    std::env::vars_os()
        .filter_map(|(key, value)| Some((key.into_string().ok()?, value.into_string().ok()?)))
}

/// The entries of the `.env` at `path`, in order, the first of a repeated key
/// only; none for a missing file.
fn read_dotenv(path: &Path) -> Result<Vec<(String, String)>, DotenvError> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(DotenvError::Read(error)),
    };
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    let mut lines = Lines {
        rest: text.as_bytes(),
        line: 0,
        at_start: true,
    };
    let mut entries: Vec<(String, String)> = Vec::new();
    for entry in dotenvy::from_read_iter(&mut lines) {
        match entry {
            Ok((key, value)) => {
                if !entries.iter().any(|(seen, _)| *seen == key) {
                    entries.push((key, value));
                }
            },
            Err(dotenvy::Error::Io(error)) => return Err(DotenvError::Read(error)),
            // The error holds the offending text, often a secret: only the
            // line it was read from is kept.
            Err(_) => {
                return Err(DotenvError::Syntax {
                    line: Some(lines.line),
                });
            },
        }
    }
    Ok(entries)
}

/// A reader over a text that gives at most one line per read, and counts
/// the lines it started giving: when `dotenvy` fails on an entry, the last
/// one is the line it failed on, whatever the error holds.
struct Lines<'a> {
    /// What is left to give.
    rest: &'a [u8],
    /// The lines given so far, the last maybe in part.
    line: usize,
    /// Whether the next byte starts a line.
    at_start: bool,
}

impl io::Read for Lines<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.rest.is_empty() || buf.is_empty() {
            return Ok(0);
        }
        if self.at_start {
            self.line += 1;
        }
        let line_end = self
            .rest
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(self.rest.len(), |at| at + 1);
        let count = line_end.min(buf.len());
        let (given, rest) = self.rest.split_at(count);
        buf[..count].copy_from_slice(given);
        self.at_start = given.last() == Some(&b'\n');
        self.rest = rest;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn pairs(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect()
    }

    fn keys(keys: &[&str]) -> DotenvKeys {
        DotenvKeys(keys.iter().map(|key| (*key).to_string()).collect())
    }

    fn value<'a>(env: &'a EnvSource, key: &str) -> Option<&'a str> {
        let EnvSource::Vars(vars) = env else {
            return None;
        };
        vars.iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.as_str())
    }

    #[test]
    fn a_shell_key_wins_over_the_env_file() {
        let env = merged_env(
            pairs(&[("SHELL_KEY", "shell")]).into_iter(),
            &keys(&[]),
            pairs(&[("SHELL_KEY", "file")]),
        );
        assert_eq!(value(&env, "SHELL_KEY"), Some("shell"));
    }

    #[test]
    fn a_key_the_env_file_set_takes_its_new_value() {
        let env = merged_env(
            pairs(&[("FILE_KEY", "old")]).into_iter(),
            &keys(&["FILE_KEY"]),
            pairs(&[("FILE_KEY", "new")]),
        );
        assert_eq!(env, EnvSource::Vars(pairs(&[("FILE_KEY", "new")])));
    }

    #[test]
    fn a_key_removed_from_the_env_file_goes() {
        let env = merged_env(
            pairs(&[("GONE", "old"), ("PATH", "/bin")]).into_iter(),
            &keys(&["GONE"]),
            Vec::new(),
        );
        assert_eq!(env, EnvSource::Vars(pairs(&[("PATH", "/bin")])));
    }

    #[test]
    fn a_new_env_file_key_is_added_once() {
        let env = merged_env(
            std::iter::empty(),
            &keys(&[]),
            pairs(&[("NEW", "first"), ("NEW", "second")]),
        );
        assert_eq!(env, EnvSource::Vars(pairs(&[("NEW", "first")])));
    }

    #[test]
    fn record_skips_the_keys_the_shell_exports() -> TestResult {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join(DOTENV_FILE);
        fs::write(&path, "# comment\nFROM_FILE=a\nEXPORTED=b\n")?;
        let recorded = DotenvKeys::record(&path, pairs(&[("EXPORTED", "shell")]).into_iter())?;
        assert_eq!(recorded, keys(&["FROM_FILE"]));
        let missing = DotenvKeys::record(&dir.path().join("none"), std::iter::empty())?;
        assert_eq!(missing, DotenvKeys::default());
        Ok(())
    }

    #[test]
    fn the_stamp_changes_with_the_content_and_with_the_env_file() -> TestResult {
        let dir = tempfile::tempdir()?;
        fs::write(dir.path().join(CONFIG_FILE), "a")?;
        let first = stamp(dir.path());
        assert_eq!(stamp(dir.path()), first, "nothing changed");
        fs::write(dir.path().join(CONFIG_FILE), "ab")?;
        let edited = stamp(dir.path());
        assert_ne!(edited, first);
        fs::write(dir.path().join(DOTENV_FILE), "A=1\n")?;
        let created = stamp(dir.path());
        assert_ne!(created, edited);
        fs::remove_file(dir.path().join(DOTENV_FILE))?;
        assert_eq!(stamp(dir.path()), edited, "removed");
        Ok(())
    }

    const VALID: &str = r#"
[project]
name = "demo"

[[topics]]
name = "t"
subtopics = 1
questions_per_subtopic = 1

[providers.p]
protocol = "openai"

[roles]
generator = { provider = "p", model = "gen" }
parent = { provider = "p", model = "parent" }
"#;

    #[test]
    fn a_reload_reads_both_files_with_the_merged_environment() -> TestResult {
        let dir = tempfile::tempdir()?;
        fs::write(dir.path().join(CONFIG_FILE), VALID)?;
        fs::write(
            dir.path().join(DOTENV_FILE),
            "OVERBRAINER_PROJECT__NAME=from-file\n",
        )?;
        let process = pairs(&[("OVERBRAINER_PROJECT__NAME", "at-start")]);
        let dotenv = keys(&["OVERBRAINER_PROJECT__NAME"]);
        let reloaded = reload_from(dir.path(), &dotenv, process.into_iter())?;
        assert_eq!(reloaded.settings.project.name, "from-file");
        assert_eq!(reloaded.text, VALID);
        assert_eq!(
            value(&reloaded.env, "OVERBRAINER_PROJECT__NAME"),
            Some("from-file")
        );
        Ok(())
    }

    #[test]
    fn an_invalid_file_is_an_error() -> TestResult {
        let dir = tempfile::tempdir()?;
        fs::write(
            dir.path().join(CONFIG_FILE),
            VALID.replace("subtopics = 1", "subtopics = 0"),
        )?;
        let error = reload_from(dir.path(), &keys(&[]), std::iter::empty());
        assert!(
            matches!(error, Err(ReloadError::Config(ConfigError::Invalid(_)))),
            "{error:?}"
        );
        Ok(())
    }

    #[test]
    fn a_syntax_error_names_the_exact_line() -> TestResult {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join(DOTENV_FILE);
        for (text, line) in [
            ("A=1\nB=two words\n", 2),
            ("A=1\n# c\n\nB=\"bad\\q\"\n", 4),
            ("A=\"multi\nline\"\nC=x y\n", 3),
            ("A=1\nB=\"open\nstill open\n", 3),
            ("two=1\nX=two words\n", 2),
        ] {
            fs::write(&path, text)?;
            let error = read_dotenv(&path).err().ok_or("must fail")?;
            assert_eq!(
                error.to_string(),
                format!("cannot parse .env (syntax error at line {line})"),
                "{text:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn an_env_file_syntax_error_names_its_line_only() -> TestResult {
        let dir = tempfile::tempdir()?;
        fs::write(dir.path().join(CONFIG_FILE), VALID)?;
        fs::write(
            dir.path().join(DOTENV_FILE),
            "# keys\nGOOD=1\nBROKEN sk-secret-value\n",
        )?;
        let Err(error) = reload_from(dir.path(), &keys(&[]), std::iter::empty()) else {
            return Err("a broken .env must fail the reload".into());
        };
        let message = error.to_string();
        assert_eq!(message, "cannot parse .env (syntax error at line 3)");
        assert!(!format!("{error:?}").contains("sk-secret"));
        Ok(())
    }
}
