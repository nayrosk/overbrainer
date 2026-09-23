//! The `overbrainer config check` subcommand.

use std::path::Path;

use anyhow::Context;
use secrecy::SecretString;

use crate::config::{
    DEFAULT_IMAGE, DEFAULT_RUNPOD_BASE_URL, DEFAULT_RUNPOD_IMAGE, DEFAULT_RUNPOD_VENV,
    DEFAULT_WORKDIR, Engine, EnvSource, Runtime, Settings, Target,
};

/// Prints the resolved configuration with secrets masked. With `resolve`, also
/// resolves every secret (testing Vault access) and fails on the first error.
///
/// # Errors
///
/// Returns an error if the configuration cannot be loaded, or, when `resolve` is
/// set, if any secret cannot be resolved (for example a `vault:` reference with no
/// Vault configured, or a Vault request that fails).
pub async fn run(project_dir: &Path, resolve: bool) -> anyhow::Result<()> {
    let settings = crate::config::load(project_dir, EnvSource::Process)?;
    for line in describe(&settings) {
        println!("{line}");
    }
    if resolve {
        let resolver = super::resolver();
        for (key, secret) in secrets(&settings) {
            resolver
                .resolve(secret)
                .await
                .with_context(|| format!("cannot resolve {key}"))?;
            println!("{key}: resolved");
        }
    }
    Ok(())
}

fn masked(secret: Option<&SecretString>) -> &'static str {
    if secret.is_some() { "***" } else { "(unset)" }
}

fn secrets(settings: &Settings) -> Vec<(String, &SecretString)> {
    let mut found = Vec::new();
    for (name, provider) in &settings.providers {
        if let Some(key) = &provider.api_key {
            found.push((format!("providers.{name}.api_key"), key));
        }
    }
    if let Some(key) = &settings.runpod.api_key {
        found.push(("runpod.api_key".to_string(), key));
    }
    if let Some(token) = &settings.hf_token {
        found.push(("hf_token".to_string(), token));
    }
    found
}

fn describe(settings: &Settings) -> Vec<String> {
    let mut lines = vec![format!("project.name = {}", settings.project.name)];
    for topic in &settings.topics {
        lines.push(format!(
            "topics.{} = {} subtopics x {} questions",
            topic.name, topic.subtopics, topic.questions_per_subtopic
        ));
    }
    for (name, provider) in &settings.providers {
        lines.push(format!(
            "providers.{name}.protocol = {:?}",
            provider.protocol
        ));
        lines.push(format!(
            "providers.{name}.base_url = {}",
            provider.base_url.as_deref().unwrap_or("(unset)")
        ));
        lines.push(format!(
            "providers.{name}.api_key = {}",
            masked(provider.api_key.as_ref())
        ));
    }
    for (role, model) in settings.roles.all() {
        lines.push(format!(
            "roles.{role} = {}/{} (reasoning: {}, max_tokens: {})",
            model.provider, model.model, model.reasoning, model.max_tokens
        ));
    }
    lines.push(format!("pipeline = {:?}", settings.pipeline));
    if let Some(training) = &settings.training {
        lines.push(format!(
            "training = {} on {} ({:?})",
            training.base_model, training.target, training.adapter
        ));
    }
    for (name, target) in &settings.targets {
        lines.push(format!("targets.{name} = {}", target_summary(target)));
    }
    lines.push(format!(
        "runpod.api_key = {}",
        masked(settings.runpod.api_key.as_ref())
    ));
    lines.push(format!(
        "runpod.base_url = {}",
        settings
            .runpod
            .base_url
            .as_deref()
            .unwrap_or(DEFAULT_RUNPOD_BASE_URL)
    ));
    lines.push(format!("hf_token = {}", masked(settings.hf_token.as_ref())));
    lines
}

fn target_summary(target: &Target) -> String {
    match target {
        Target::Local {
            runtime,
            engine,
            image,
            venv,
        } => format!(
            "local, {}",
            runtime_summary(*runtime, *engine, image.as_deref(), venv.as_deref())
        ),
        Target::Ssh {
            runtime,
            host,
            workdir,
            engine,
            image,
            venv,
        } => format!(
            "ssh {} in {}, {}",
            host.as_deref().unwrap_or("(host unset)"),
            workdir.as_deref().unwrap_or(DEFAULT_WORKDIR),
            runtime_summary(*runtime, *engine, image.as_deref(), venv.as_deref())
        ),
        Target::Runpod { .. } => runpod_summary(target),
    }
}

/// A runpod target on one line, defaults applied.
fn runpod_summary(target: &Target) -> String {
    let Target::Runpod {
        gpu_types,
        gpu_count,
        image,
        venv,
        container_disk_gb,
        max_hours,
        boot_grace_minutes,
        retrieve_grace_minutes,
        data_center_ids,
        network_volume_id,
    } = target
    else {
        return String::new();
    };
    let data_centers = if data_center_ids.is_empty() {
        "any".to_string()
    } else {
        data_center_ids.join(", ")
    };
    format!(
        "runpod {gpu_count}x [{}], max {max_hours}h, image {}, venv {}, disk {container_disk_gb} GB, \
         boot grace {boot_grace_minutes} min, retrieve grace {retrieve_grace_minutes} min, \
         data centers {data_centers}, network volume {}",
        gpu_types.join(", "),
        image.as_deref().unwrap_or(DEFAULT_RUNPOD_IMAGE),
        venv.as_deref().unwrap_or(DEFAULT_RUNPOD_VENV),
        network_volume_id.as_deref().unwrap_or("none"),
    )
}

fn runtime_summary(
    runtime: Runtime,
    engine: Option<Engine>,
    image: Option<&str>,
    venv: Option<&str>,
) -> String {
    match runtime {
        Runtime::Docker => format!(
            "{} {}",
            engine.unwrap_or(Engine::Docker).command(),
            image.unwrap_or(DEFAULT_IMAGE)
        ),
        Runtime::Native => format!("native, venv {}", venv.unwrap_or("(axolotl on PATH)")),
    }
}
