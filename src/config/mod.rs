//! Typed configuration loaded from `overbrainer.toml` and `OVERBRAINER_*` env vars.

mod load;
mod types;
mod validate;

pub use load::{CONFIG_FILE, ConfigError, ENV_PREFIX, EnvSource, load};
pub use types::*;
