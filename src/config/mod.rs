//! Typed configuration loaded from `overbrainer.toml` and `OVERBRAINER_*` env vars.

mod load;
mod types;
mod validate;

pub use load::{CONFIG_FILE, ConfigError, ENV_PREFIX, EnvSource, env_keys, load, load_str};
pub use types::*;
