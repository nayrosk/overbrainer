use std::path::{Path, PathBuf};

use config::{Config, Environment, File, FileFormat};

use super::{Settings, validate};

/// Name of the project-level configuration file, relative to the project directory.
pub const CONFIG_FILE: &str = "overbrainer.toml";
/// Prefix required on every environment variable read into the configuration.
pub const ENV_PREFIX: &str = "OVERBRAINER";

/// Where [`load`] reads `OVERBRAINER_*` environment variable overrides from.
#[derive(Debug, Clone)]
pub enum EnvSource {
    /// Read from the current process environment. This is what a running binary uses.
    Process,
    /// Use these key-value pairs instead of the process environment. Tests use this
    /// to inject environment variables without calling `std::env::set_var`.
    Vars(Vec<(String, String)>),
}

/// Everything that can go wrong while loading configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The configuration file could not be read.
    #[error("cannot read {}: {source}", path.display())]
    Read {
        /// Path to the file that could not be read.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The file or environment could not be parsed into `Settings`. The message names
    /// the key and the expected type, never the offending value, which may be a secret
    /// set in the wrong variable.
    #[error("invalid configuration: {0}")]
    Parse(String),
    /// The parsed settings failed semantic validation.
    #[error("invalid configuration:\n  {}", .0.join("\n  "))]
    Invalid(Vec<String>),
}

impl From<config::ConfigError> for ConfigError {
    fn from(source: config::ConfigError) -> Self {
        Self::Parse(describe(&source))
    }
}

/// Describes a `config` error without the offending value.
///
/// Type errors keep their key and expected type. Custom messages from serde quote the
/// value (`invalid value: string "..."`, `unknown variant ...`), so only the part from
/// `expected` onward is kept. Unknown field names are keys, not values, and are kept.
/// TOML syntax errors are reported by location only, never with source snippets.
fn describe(error: &config::ConfigError) -> String {
    match error {
        config::ConfigError::Type { key, expected, .. } => {
            let key = key.as_deref().unwrap_or("a value");
            format!("`{key}` has the wrong type, expected {expected}")
        },
        config::ConfigError::At { error, key, .. } => match key {
            Some(key) => format!("`{key}`: {}", describe(error)),
            None => describe(error),
        },
        config::ConfigError::Message(message) => redact(message),
        config::ConfigError::FileParse { .. } => redact_toml_error(error),
        _ => "invalid configuration".to_string(),
    }
}

/// Extracts location info from TOML parse errors, never including source snippets.
fn redact_toml_error(error: &config::ConfigError) -> String {
    let message = error.to_string();
    let first_line = message.lines().next().unwrap_or("");

    if let Some(pos) = first_line.find("at line ") {
        if let Some(end) = first_line[pos..].find('\n') {
            let location = &first_line[pos + 3..pos + end];
            return format!("TOML syntax error in overbrainer.toml {location}");
        }
        let location = &first_line[pos + 3..];
        return format!("TOML syntax error in overbrainer.toml {location}");
    }

    "TOML syntax error in overbrainer.toml".to_string()
}

/// Keeps what a serde message says was expected, dropping the value it quotes.
fn redact(message: &str) -> String {
    if message.starts_with("unknown field") || message.starts_with("missing field") {
        return message.to_string();
    }
    if let Some(index) = message.find("expected") {
        return format!("invalid value, {}", &message[index..]);
    }
    if let Some((head, _)) = message.split_once(" does not have variant constructor") {
        return format!("{head}: unknown variant");
    }
    "invalid value".to_string()
}

/// Loads `<project_dir>/overbrainer.toml` layered with `OVERBRAINER_*` variables.
///
/// `env: EnvSource::Process` reads the process environment; tests pass
/// `EnvSource::Vars` with an explicit list of key-value pairs.
///
/// # Errors
///
/// Returns [`ConfigError::Read`] when the file cannot be read, [`ConfigError::Parse`]
/// when the file or environment cannot be parsed into [`Settings`], and
/// [`ConfigError::Invalid`] when the parsed settings fail semantic validation or
/// when an env-only key is set in the file.
pub fn load(project_dir: &Path, env: EnvSource) -> Result<Settings, ConfigError> {
    let path = project_dir.join(CONFIG_FILE);
    let content = std::fs::read_to_string(&path).map_err(|source| ConfigError::Read {
        path: path.clone(),
        source,
    })?;
    let env: Option<config::Map<String, String>> = match env {
        EnvSource::Process => None,
        EnvSource::Vars(pairs) => Some(pairs.into_iter().collect()),
    };

    let file_only = Config::builder()
        .add_source(File::from_str(&content, FileFormat::Toml))
        .build()?;
    let mut problems = validate::env_only_in_file(&file_only);

    let settings: Settings = Config::builder()
        .add_source(File::from_str(&content, FileFormat::Toml))
        .add_source(
            Environment::with_prefix(ENV_PREFIX)
                .prefix_separator("_")
                .separator("__")
                .try_parsing(false)
                .source(env),
        )
        .build()?
        .try_deserialize()?;

    problems.extend(validate::check(&settings));
    if problems.is_empty() {
        Ok(settings)
    } else {
        Err(ConfigError::Invalid(problems))
    }
}
