//! Every integration test, built as one binary to keep link time, memory and disk low.

mod cli;
mod cli_complete;
mod cli_metrics;
mod cli_pipeline;
mod cli_pod;
mod cli_runpod;
mod cli_runs_logs;
mod cli_skill;
mod cli_train;
#[allow(dead_code)]
mod common;
mod config_load;
mod dataset_jsonl;
mod dedup;
mod exec_local;
mod exec_ssh;
mod llm_anthropic;
mod llm_openai;
mod metrics_plugin;
mod pipeline;
mod plugin;
mod pricing;
mod runpod_catalog;
mod runpod_client;
mod runpod_disk;
mod runpod_flow;
mod runpod_logs;
mod runpod_orphans;
mod runpod_provision;
mod runpod_ssh;
mod runpod_watchdog;
mod runs_train;
mod secrets_vault;
mod train_axolotl;
