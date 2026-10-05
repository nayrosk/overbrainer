//! The `overbrainer config check` subcommand.

use std::path::Path;

use anyhow::Context;
use secrecy::SecretString;

use crate::config::{
    DEFAULT_IMAGE, DEFAULT_RUNPOD_BASE_URL, DEFAULT_RUNPOD_IMAGE, DEFAULT_RUNPOD_VENV,
    DEFAULT_WORKDIR, Engine, EnvSource, HAS_BUILTIN_SSH, Runtime, SSH_CLIENT_ENV, Settings,
    SshClient, Target, effective_client, ssh_client_env,
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
    let ssh_env = ssh_client_env(&EnvSource::Process);
    for line in describe(&settings, ssh_env.as_deref()).map_err(anyhow::Error::msg)? {
        println!("{line}");
    }
    for deprecation in crate::config::validate::deprecations(&settings) {
        eprintln!("warning: {deprecation}");
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

/// The lines of `config check`. `ssh_env` is the value of `OVERBRAINER_SSH_CLIENT`.
///
/// # Errors
///
/// Returns a message when `ssh_env` is invalid, or names the built-in client in a
/// build without it.
fn describe(settings: &Settings, ssh_env: Option<&str>) -> Result<Vec<String>, String> {
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
        if let Some(configured) = configured_client(target) {
            lines.push(format!(
                "targets.{name}.ssh_client = {}",
                client_summary(configured, ssh_env)?
            ));
        }
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
    Ok(lines)
}

/// The `ssh_client` of an SSH or Runpod target.
fn configured_client(target: &Target) -> Option<SshClient> {
    match target {
        Target::Ssh { ssh_client, .. } | Target::Runpod { ssh_client, .. } => Some(*ssh_client),
        Target::Local { .. } => None,
    }
}

/// The effective client, where it comes from, and how to use the other one.
fn client_summary(configured: SshClient, ssh_env: Option<&str>) -> Result<String, String> {
    let client = effective_client(configured, ssh_env)?;
    let from_env = ssh_env.is_some_and(|value| !value.trim().is_empty());
    let mut summary = client.name().to_string();
    if from_env {
        summary = format!("{summary} ({SSH_CLIENT_ENV})");
    }
    if client == SshClient::Openssh {
        summary.push_str(if HAS_BUILTIN_SSH {
            "; set ssh_client = \"builtin\" (or OVERBRAINER_SSH_CLIENT=builtin) to use the built-in client"
        } else {
            "; a build with the builtin-ssh feature (the release binaries have it) can use the built-in client"
        });
    }
    Ok(summary)
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
            ..
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
        min_vram_gb,
        max_price_per_hour,
        gpu_count,
        image,
        venv,
        container_disk_gb,
        max_hours,
        max_cost_usd,
        boot_grace_minutes,
        retrieve_grace_minutes,
        data_center_ids,
        network_volume_id,
        max_volume_gb,
        ..
    } = target
    else {
        return String::new();
    };
    let data_centers = if data_center_ids.is_any() {
        "any".to_string()
    } else {
        data_center_ids.to_string()
    };
    let mut limits = Vec::new();
    if let Some(gb) = min_vram_gb {
        limits.push(format!("at least {gb} GB"));
    }
    if let Some(price) = max_price_per_hour {
        limits.push(format!("at most ${price}/h per GPU"));
    }
    let limits = if limits.is_empty() {
        String::new()
    } else {
        format!(" ({})", limits.join(", "))
    };
    let cost = max_cost_usd.map_or_else(String::new, |usd| format!(", max ${usd}"));
    let grow = max_volume_gb.map_or_else(String::new, |gb| format!(" (grows up to {gb} GB)"));
    format!(
        "runpod {gpu_count}x [{gpu_types}]{limits}, max {max_hours}h{cost}, image {}, venv {}, disk {container_disk_gb} GB, \
         boot grace {boot_grace_minutes} min, retrieve grace {retrieve_grace_minutes} min, \
         data centers {data_centers}, network volume {}{grow}",
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::load_str;

    /// A project with an ssh target `box` and a runpod target `cloud`, `extra`
    /// appended to both.
    fn project(extra: &str) -> Result<Settings, crate::config::ConfigError> {
        let text = format!(
            r#"
[project]
name = "demo"
[providers.p]
protocol = "openai"
[roles.generator]
provider = "p"
model = "m"
[roles.parent]
provider = "p"
model = "m"
[targets.local]
kind = "local"
runtime = "native"
[targets.box]
kind = "ssh"
runtime = "native"
{extra}
[targets.cloud]
kind = "runpod"
gpu_types = ["NVIDIA A40"]
max_hours = 6
{extra}
"#
        );
        load_str(&text, EnvSource::Vars(Vec::new()))
    }

    /// The line of `lines` that starts with `prefix`.
    fn line<'a>(lines: &'a [String], prefix: &str) -> Option<&'a String> {
        lines.iter().find(|line| line.starts_with(prefix))
    }

    #[test]
    fn config_check_prints_the_effective_client_of_ssh_and_runpod_targets()
    -> Result<(), Box<dyn std::error::Error>> {
        let lines = describe(&project("")?, None)?;
        for target in ["box", "cloud"] {
            let found = line(&lines, &format!("targets.{target}.ssh_client = openssh"));
            assert!(found.is_some(), "{lines:?}");
        }
        assert!(line(&lines, "targets.local.ssh_client").is_none());
        Ok(())
    }

    #[cfg(feature = "builtin-ssh")]
    #[test]
    fn config_check_suggests_builtin_in_a_build_with_the_feature()
    -> Result<(), Box<dyn std::error::Error>> {
        let lines = describe(&project("")?, None)?;
        let found = line(&lines, "targets.box.ssh_client").ok_or("no line")?;
        assert!(
            found.contains("ssh_client = \"builtin\"")
                && found.contains("OVERBRAINER_SSH_CLIENT=builtin"),
            "{found}"
        );
        let lines = describe(&project("ssh_client = \"builtin\"")?, None)?;
        assert!(
            line(&lines, "targets.box.ssh_client = builtin").is_some(),
            "{lines:?}"
        );
        Ok(())
    }

    #[cfg(feature = "builtin-ssh")]
    #[test]
    fn the_environment_wins_in_config_check() -> Result<(), Box<dyn std::error::Error>> {
        let lines = describe(&project("")?, Some("builtin"))?;
        assert!(
            line(
                &lines,
                "targets.cloud.ssh_client = builtin (OVERBRAINER_SSH_CLIENT)"
            )
            .is_some(),
            "{lines:?}"
        );
        Ok(())
    }

    #[cfg(not(feature = "builtin-ssh"))]
    #[test]
    fn config_check_points_to_the_feature_build_without_the_feature()
    -> Result<(), Box<dyn std::error::Error>> {
        let lines = describe(&project("")?, None)?;
        let found = line(&lines, "targets.box.ssh_client").ok_or("no line")?;
        assert!(found.contains("builtin-ssh feature"), "{found}");
        assert_eq!(
            describe(&project("")?, Some("builtin")).err(),
            Some(crate::config::BUILTIN_SSH_REFUSED.to_string())
        );
        Ok(())
    }

    #[test]
    fn config_check_refuses_an_invalid_client_variable() -> Result<(), Box<dyn std::error::Error>> {
        let error = describe(&project("")?, Some("putty")).err();
        assert!(
            error
                .as_deref()
                .is_some_and(|error| error.contains("OVERBRAINER_SSH_CLIENT")),
            "{error:?}"
        );
        Ok(())
    }
}
