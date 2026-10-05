//! Typed configuration loaded from `overbrainer.toml` and `OVERBRAINER_*` env vars.

pub mod edit;
pub mod fields;
mod load;
mod reload;
mod types;
pub(crate) mod validate;

pub use load::{
    CONFIG_FILE, ConfigError, ENV_PREFIX, EnvSource, Source, env_keys, load, load_str,
    process_client, ssh_client_env,
};
pub use reload::{
    DOTENV_FILE, DotenvError, DotenvKeys, ReloadError, Reloaded, Stamp, merged_env, reload, stamp,
};
pub use types::*;
pub(crate) use validate::is_valid_name;
