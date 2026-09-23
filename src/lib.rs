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
