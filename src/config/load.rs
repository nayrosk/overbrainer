use std::collections::HashMap;
use std::hash::BuildHasher;
use std::path::{Path, PathBuf};

use config::{Config, Environment, File, FileFormat};

use super::{Settings, validate};

/// Name of the project-level configuration file, relative to the project directory.
pub const CONFIG_FILE: &str = "overbrainer.toml";
/// Prefix required on every environment variable read into the configuration.
pub const ENV_PREFIX: &str = "OVERBRAINER";

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
    /// The file or environment could not be parsed into `Settings`.
    #[error("invalid configuration: {0}")]
    Parse(Box<config::ConfigError>),
    /// The parsed settings failed semantic validation.
    #[error("invalid configuration:\n  {}", .0.join("\n  "))]
    Invalid(Vec<String>),
}

impl From<config::ConfigError> for ConfigError {
    fn from(source: config::ConfigError) -> Self {
        Self::Parse(Box::new(source))
    }
}

/// Loads `<project_dir>/overbrainer.toml` layered with `OVERBRAINER_*` variables.
///
/// `env: None` reads the process environment; tests pass an explicit map. The hasher
/// is generic so callers are not forced into the default one; it is converted to
/// `config`'s own map type internally.
///
/// # Errors
///
/// Returns [`ConfigError::Read`] when the file cannot be read, [`ConfigError::Parse`]
/// when the file or environment cannot be parsed into [`Settings`], and
/// [`ConfigError::Invalid`] when the parsed settings fail semantic validation or
/// when an env-only key is set in the file.
pub fn load<S: BuildHasher>(
    project_dir: &Path,
    env: Option<HashMap<String, String, S>>,
) -> Result<Settings, ConfigError> {
    let path = project_dir.join(CONFIG_FILE);
    let content = std::fs::read_to_string(&path).map_err(|source| ConfigError::Read {
        path: path.clone(),
        source,
    })?;
    let env: Option<config::Map<String, String>> = env.map(|map| map.into_iter().collect());

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
