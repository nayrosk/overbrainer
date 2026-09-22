//! Tracing setup for the command line interface.

use tracing_subscriber::EnvFilter;

/// Env variable holding the tracing filter, for example `info` or `overbrainer=debug`.
pub const LOG_ENV: &str = "OVERBRAINER_LOG";

/// Sends logs to stderr. Honors `NO_COLOR`. Defaults to `info`.
pub fn init() {
    let filter = EnvFilter::try_from_env(LOG_ENV).unwrap_or_else(|_| EnvFilter::new("info"));
    let ansi = std::env::var_os("NO_COLOR").is_none();
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(ansi)
        .with_target(false)
        .try_init()
        .ok();
}
