//! The `overbrainer config check` subcommand.

use std::path::Path;

use anyhow::Context;
use secrecy::SecretString;

use crate::config::{EnvSource, Settings, Target};

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
        let summary = match target {
            Target::Local { runtime, .. } => format!("local, {runtime:?}"),
            Target::Ssh { runtime, host, .. } => {
                format!(
                    "ssh {}, {runtime:?}",
                    host.as_deref().unwrap_or("(host unset)")
                )
            },
            Target::Runpod {
                gpu_type,
                gpu_count,
                max_hours,
                ..
            } => format!("runpod {gpu_count}x {gpu_type}, max {max_hours}h"),
        };
        lines.push(format!("targets.{name} = {summary}"));
    }
    lines.push(format!(
        "runpod.api_key = {}",
        masked(settings.runpod.api_key.as_ref())
    ));
    lines.push(format!("hf_token = {}", masked(settings.hf_token.as_ref())));
    lines
}
