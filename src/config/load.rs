use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use config::{Config, Environment, File, FileFormat};
use serde_json::Value;

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

/// Keeps only the location of a TOML parse error ("at line N, column M"), never the
/// source snippet that follows it on the next lines.
fn redact_toml_error(error: &config::ConfigError) -> String {
    let message = error.to_string();
    let first_line = message.lines().next().unwrap_or("");
    match first_line.find("at line ") {
        Some(pos) => format!(
            "TOML syntax error in overbrainer.toml {}",
            first_line[pos..].trim_end()
        ),
        None => "TOML syntax error in overbrainer.toml".to_string(),
    }
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

    let mut settings: Settings = Config::builder()
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

    if let Some(training) = settings.training.as_mut() {
        let file_extra: BTreeMap<String, Value> =
            file_only.get("training.axolotl_extra").unwrap_or_default();
        coerce_env_scalars(&mut training.axolotl_extra, &file_extra);
    }

    problems.extend(validate::check(&settings));
    if problems.is_empty() {
        Ok(settings)
    } else {
        Err(ConfigError::Invalid(problems))
    }
}

/// Gives env-provided values of `training.axolotl_extra` a type.
///
/// The `Environment` source above is built with `try_parsing(false)`, the config-wide
/// default that keeps every env value a string (so a secret such as `0123` is never
/// silently type-parsed). This function is where `axolotl_extra` alone deliberately
/// relaxes that default: every value set through
/// `OVERBRAINER_TRAINING__AXOLOTL_EXTRA__*` arrives as a string, and a string that is
/// not the file's own value for the same key came from env and becomes a boolean
/// (`true`, `false`), a canonical integer, or a finite float when it parses as one
/// (see [`scalar`]); any other text stays a string. Values from `overbrainer.toml`
/// keep their TOML type. A value that must stay a string, such as a revision with
/// leading zeros (`"0123"`) or a version-like `"3.14"`, belongs in the file; env has
/// no way to know a key means "always a string", so an override of that key with
/// different text is still coerced by this function.
fn coerce_env_scalars(extra: &mut BTreeMap<String, Value>, file: &BTreeMap<String, Value>) {
    for (key, value) in extra.iter_mut() {
        coerce(value, file.get(key));
    }
}

fn coerce(value: &mut Value, file: Option<&Value>) {
    if let Value::Object(map) = value {
        for (key, inner) in map.iter_mut() {
            coerce(inner, file.and_then(|file| file.get(key)));
        }
        return;
    }
    let parsed = match &*value {
        Value::String(text) if file.and_then(Value::as_str) != Some(text.as_str()) => scalar(text),
        _ => None,
    };
    if let Some(parsed) = parsed {
        *value = parsed;
    }
}

/// `text` as a boolean, an integer or a finite float, when it is one.
///
/// An integer is only recognized when `text` is its own canonical rendering (no
/// leading zero, no leading `+`), so a padded value such as a revision (`"0123"`)
/// is not silently read as `123`. When `text` parses as an integer but is not
/// canonical, it is left as a string outright, it is not tried as a float either.
/// Floats have no such check: any text that parses as a finite `f64` is accepted.
fn scalar(text: &str) -> Option<Value> {
    match text {
        "true" => return Some(Value::Bool(true)),
        "false" => return Some(Value::Bool(false)),
        _ => {},
    }
    if let Ok(integer) = text.parse::<i64>() {
        return (integer.to_string() == text).then_some(Value::from(integer));
    }
    text.parse::<f64>()
        .ok()
        .and_then(serde_json::Number::from_f64)
        .map(Value::Number)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_parse_only_plain_values() {
        assert_eq!(scalar("true"), Some(Value::Bool(true)));
        assert_eq!(scalar("10"), Some(Value::from(10)));
        assert_eq!(scalar("0.05"), Some(Value::from(0.05)));
        assert_eq!(scalar("nan"), None);
        assert_eq!(scalar("inf"), None);
        assert_eq!(scalar("qwen3"), None);
        assert_eq!(scalar("True"), None);
    }

    #[test]
    fn non_canonical_integers_stay_strings() {
        assert_eq!(scalar("0123"), None, "leading zero");
        assert_eq!(scalar("+5"), None, "leading plus");
        assert_eq!(scalar("007"), None, "leading zeros");
    }

    #[test]
    fn canonical_integers_and_floats_still_parse() {
        assert_eq!(scalar("0"), Some(Value::from(0)));
        assert_eq!(scalar("-3"), Some(Value::from(-3)));
        assert_eq!(scalar("1e-7"), Some(Value::from(1e-7)));
    }
}
