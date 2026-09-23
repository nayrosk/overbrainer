//! overbrainer: distill a parent LLM into a smaller child model.

pub mod cli;
pub mod config;
pub mod dataset;
pub mod dedup;
pub mod events;
pub mod exec;
pub mod llm;
pub mod logging;
pub mod pipeline;
pub mod pricing;
pub mod prompts;
pub mod retry;
pub mod runpod;
pub mod runs;
pub mod secrets;
pub mod train;
pub mod tui;

/// What the unit tests share across modules.
#[cfg(test)]
pub(crate) mod test_support {
    /// Held by every test that sends a real signal to the test process. All unit
    /// tests run in one process, so a signal one test raises reaches the
    /// listeners of another running at the same time and breaks its timing.
    /// Bind it to a name (`let _signals = SIGNALS.lock().await;`): `_` would
    /// release it at once.
    pub(crate) static SIGNALS: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
}
