//! overbrainer: distill a parent LLM into a smaller child model.

pub mod cli;
pub mod config;
pub mod dataset;
pub mod dedup;
pub mod events;
pub mod llm;
pub mod logging;
pub mod pipeline;
pub mod pricing;
pub mod prompts;
pub mod secrets;
pub mod train;
