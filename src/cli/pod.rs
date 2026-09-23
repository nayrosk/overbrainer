//! `overbrainer pod ls` and `pod rm <run-id>`: the Runpod pods overbrainer created.

use std::path::Path;
use std::sync::atomic::AtomicBool;

use anyhow::Context;

use super::PodCommand;
use super::progress::duration_words;
use crate::config::{DEFAULT_RUNPOD_BASE_URL, EnvSource, Settings};
use crate::events::EventBus;
use crate::runpod::{PodCtx, RunpodClient, Timing, pod_rows, remove_run_pods, table};
use crate::runs::Runs;

/// Runs `overbrainer pod ls` or `pod rm`.
///
/// # Errors
///
/// Returns an error when the configuration, the API key or the Runpod API fails,
/// or when `pod rm` refuses a running run.
pub async fn run(project_dir: &Path, command: &PodCommand) -> anyhow::Result<()> {
    let settings = crate::config::load(project_dir, EnvSource::Process)?;
    let client = client(&settings).await?;
    let runs = Runs::new(project_dir);
    let bus = EventBus::new();
    let renderer = tokio::spawn(super::progress::render(bus.subscribe()));
    let timing = Timing::standard();
    let interrupted = AtomicBool::new(false);
    let ctx = PodCtx {
        client: &client,
        runs: &runs,
        bus: &bus,
        timing: &timing,
        interrupted: &interrupted,
    };
    let result = match command {
        PodCommand::Ls => ls(&ctx).await,
        PodCommand::Rm { run_id, force } => rm(&ctx, run_id, *force).await,
    };
    drop(bus);
    renderer.await.ok();
    result
}

/// A client of the Runpod API with the account key, resolved only now.
///
/// # Errors
///
/// Returns an error when the key is not set or cannot be resolved.
pub(super) async fn client(settings: &Settings) -> anyhow::Result<RunpodClient> {
    let key = settings
        .runpod
        .api_key
        .as_ref()
        .context("no Runpod API key: set OVERBRAINER_RUNPOD__API_KEY")?;
    let key = super::resolver()
        .resolve(key)
        .await
        .context("cannot resolve runpod.api_key")?;
    let base_url = settings
        .runpod
        .base_url
        .as_deref()
        .unwrap_or(DEFAULT_RUNPOD_BASE_URL);
    Ok(RunpodClient::new(base_url, &key)?)
}

async fn ls(ctx: &PodCtx<'_>) -> anyhow::Result<()> {
    for line in table(&pod_rows(ctx).await?) {
        println!("{line}");
    }
    Ok(())
}

async fn rm(ctx: &PodCtx<'_>, run_id: &str, force: bool) -> anyhow::Result<()> {
    let removed = remove_run_pods(ctx, run_id, force).await?;
    if removed.is_empty() {
        println!("pod: no pod for run {run_id}");
    }
    for pod in removed {
        let after = pod.uptime.map_or_else(String::new, |uptime| {
            format!(" after {}", duration_words(uptime))
        });
        let spend = pod
            .estimated_spend
            .map_or_else(String::new, |spend| format!(", about ${spend:.2}"));
        println!("pod: {} deleted{after}{spend}", pod.pod_id);
    }
    Ok(())
}
