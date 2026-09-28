//! `overbrainer pod`: `ls` and `rm <run-id>` for the Runpod pods overbrainer
//! created, and the read-only catalog listings (`gpus`, `datacenters`,
//! `volumes`, `templates`).

use std::path::Path;
use std::sync::atomic::AtomicBool;

use anyhow::Context;

use super::progress::duration_words;
use super::{GpuArgs, PodCommand};
use crate::config::{DEFAULT_RUNPOD_BASE_URL, EnvSource, Settings};
use crate::events::EventBus;
use crate::runpod::{
    GpuFilter, PodCtx, RunpodClient, Timing, data_center_table, gpu_table, pod_rows,
    remove_run_pods, select_gpus, table, template_table, volume_table,
};
use crate::runs::Runs;

/// Runs an `overbrainer pod` subcommand.
///
/// # Errors
///
/// Returns an error when the configuration, the API key or the Runpod API fails,
/// or when `pod rm` refuses a running run.
pub async fn run(project_dir: &Path, command: &PodCommand) -> anyhow::Result<()> {
    let settings = crate::config::load(project_dir, EnvSource::Process)?;
    let client = client(&settings).await?;
    match command {
        PodCommand::Gpus(args) => return gpus(&client, args).await,
        PodCommand::Datacenters => return datacenters(&client).await,
        PodCommand::Volumes => return volumes(&client).await,
        PodCommand::Templates => return templates(&client).await,
        PodCommand::Ls | PodCommand::Rm { .. } => {},
    }
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
        // The catalog commands returned above; never reached.
        PodCommand::Gpus(_)
        | PodCommand::Datacenters
        | PodCommand::Volumes
        | PodCommand::Templates => Ok(()),
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
pub(crate) async fn client(settings: &Settings) -> anyhow::Result<RunpodClient> {
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
    let rows = pod_rows(ctx).await?;
    if rows.is_empty() {
        println!("pod: no overbrainer pod on this account");
    }
    for line in table(&rows) {
        println!("{line}");
    }
    Ok(())
}

async fn rm(ctx: &PodCtx<'_>, run_id: &str, force: bool) -> anyhow::Result<()> {
    let removal = remove_run_pods(ctx, run_id, force).await;
    if removal.removed.is_empty() && removal.result.is_ok() {
        println!("pod: no pod for run {run_id}");
    }
    // Every deleted pod is printed, even when the command fails overall.
    for pod in &removal.removed {
        let after = pod.uptime.map_or_else(String::new, |uptime| {
            format!(" after {}", duration_words(uptime))
        });
        let spend = pod
            .estimated_spend
            .map_or_else(String::new, |spend| format!(", about ${spend:.2}"));
        println!("pod: {} deleted{after}{spend}", pod.pod_id);
    }
    Ok(removal.result?)
}

/// Runs `overbrainer pod gpus`: the Secure Cloud GPU types, filtered by
/// `args` and sorted cheapest first.
async fn gpus(client: &RunpodClient, args: &GpuArgs) -> anyhow::Result<()> {
    // `gpu_count` is not asked on the command line: the listing is for one GPU.
    let gpus = client.list_gpu_types(1).await?;
    let filter = GpuFilter {
        min_vram_gb: args.min_vram,
        max_price: args.max_price,
        data_center: args.data_center.clone(),
        in_stock: args.in_stock,
        gpu_count: None,
    };
    let selected = select_gpus(&gpus, &filter);
    if selected.is_empty() {
        println!("pod: no GPU type matches");
    }
    for line in gpu_table(&selected, args.data_center.as_deref()) {
        println!("{line}");
    }
    Ok(())
}

/// Runs `overbrainer pod datacenters`: every data center of the catalog.
async fn datacenters(client: &RunpodClient) -> anyhow::Result<()> {
    let data_centers = client.list_data_centers().await?;
    if data_centers.is_empty() {
        println!("pod: no data center found");
    }
    for line in data_center_table(&data_centers) {
        println!("{line}");
    }
    Ok(())
}

/// Runs `overbrainer pod volumes`: the account's network volumes.
async fn volumes(client: &RunpodClient) -> anyhow::Result<()> {
    let volumes = client.list_network_volumes().await?;
    if volumes.is_empty() {
        println!("pod: no network volume on this account");
    }
    for line in volume_table(&volumes) {
        println!("{line}");
    }
    Ok(())
}

/// Runs `overbrainer pod templates`: the account's pod templates.
async fn templates(client: &RunpodClient) -> anyhow::Result<()> {
    let templates = client.list_templates().await?;
    if templates.is_empty() {
        println!("pod: no pod template on this account");
    }
    for line in template_table(&templates) {
        println!("{line}");
    }
    Ok(())
}
