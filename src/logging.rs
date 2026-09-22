//! Tracing setup for the command line interface.

use tracing_subscriber::EnvFilter;

/// Env variable holding the tracing filter, for example `info` or `overbrainer=debug`.
pub const LOG_ENV: &str = "OVERBRAINER_LOG";

/// Filter used when `OVERBRAINER_LOG` is unset or invalid: our own logs at `info`,
/// dependencies at `warn`. vaultrs and rustify log every failed request at ERROR
/// level before overbrainer reports the same failure with its full cause, so they
/// are silenced; `OVERBRAINER_LOG` brings them back when debugging.
pub const DEFAULT_FILTER: &str = "overbrainer=info,vaultrs=off,rustify=off,warn";

/// Sends logs to stderr. Honors `NO_COLOR`. Defaults to [`DEFAULT_FILTER`].
pub fn init() {
    let filter =
        EnvFilter::try_from_env(LOG_ENV).unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER));
    let ansi = std::env::var_os("NO_COLOR").is_none();
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(ansi)
        .with_target(false)
        .try_init()
        .ok();
}
