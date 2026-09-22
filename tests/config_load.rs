use std::fs;

use overbrainer::config::{ConfigError, Engine, EnvSource, Target, load};
use secrecy::ExposeSecret;

const BASE: &str = r#"
[project]
name = "demo"

[providers.nanogpt]
protocol = "openai"

[roles]
generator = { provider = "nanogpt", model = "m1" }
parent = { provider = "nanogpt", model = "m2" }
"#;

fn project(toml: &str) -> Result<tempfile::TempDir, std::io::Error> {
    let dir = tempfile::tempdir()?;
    fs::write(dir.path().join("overbrainer.toml"), toml)?;
    Ok(dir)
}

fn env(pairs: &[(&str, &str)]) -> EnvSource {
    EnvSource::Vars(
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect(),
    )
}

#[test]
fn env_overrides_file_values() -> Result<(), Box<dyn std::error::Error>> {
    let dir = project(BASE)?;
    let settings = load(
        dir.path(),
        env(&[("OVERBRAINER_PIPELINE__CONCURRENCY", "16")]),
    )?;
    assert_eq!(settings.pipeline.concurrency, 16);
    Ok(())
}

#[test]
fn secrets_from_env_keep_their_exact_text() -> Result<(), Box<dyn std::error::Error>> {
    let dir = project(BASE)?;
    let settings = load(
        dir.path(),
        env(&[
            ("OVERBRAINER_PROVIDERS__NANOGPT__API_KEY", "0123"),
            (
                "OVERBRAINER_PROVIDERS__NANOGPT__BASE_URL",
                "https://nano-gpt.com/api/v1",
            ),
        ]),
    )?;
    let provider = settings
        .providers
        .get("nanogpt")
        .ok_or("provider missing")?;
    let key = provider.api_key.as_ref().ok_or("api_key missing")?;
    assert_eq!(key.expose_secret(), "0123");
    assert_eq!(
        provider.base_url.as_deref(),
        Some("https://nano-gpt.com/api/v1")
    );
    Ok(())
}

#[test]
fn env_only_keys_in_file_are_rejected() -> Result<(), Box<dyn std::error::Error>> {
    // `hf_token` must land in the document's root table, not inside the last
    // `[roles]` table, so it is prepended rather than appended.
    let toml = "hf_token = \"hf_leak\"\n".to_string()
        + &BASE.replace(
            "protocol = \"openai\"",
            "protocol = \"openai\"\napi_key = \"sk-leak\"\nbase_url = \"https://x\"",
        );
    let dir = project(&toml)?;
    match load(dir.path(), env(&[])) {
        Err(ConfigError::Invalid(problems)) => {
            assert!(
                problems.contains(
                    &"providers.nanogpt.api_key: must be set through env, not in overbrainer.toml"
                        .to_string()
                )
            );
            assert!(
                problems.contains(
                    &"providers.nanogpt.base_url: must be set through env, not in overbrainer.toml"
                        .to_string()
                )
            );
            assert!(problems.contains(
                &"hf_token: must be set through env, not in overbrainer.toml".to_string()
            ));
            let joined = problems.join("\n");
            assert!(
                !joined.contains("sk-leak") && !joined.contains("hf_leak"),
                "values must not leak"
            );
            Ok(())
        },
        other => Err(format!("expected Invalid, got {other:?}").into()),
    }
}

#[test]
fn unknown_env_variable_with_prefix_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
    let dir = project(BASE)?;
    assert!(matches!(
        load(dir.path(), env(&[("OVERBRAINER_TYPO", "1")])),
        Err(ConfigError::Parse(_))
    ));
    Ok(())
}

#[test]
fn missing_file_reports_its_path() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    match load(dir.path(), env(&[])) {
        Err(ConfigError::Read { path, .. }) => {
            assert!(path.ends_with("overbrainer.toml"));
            Ok(())
        },
        other => Err(format!("expected Read, got {other:?}").into()),
    }
}

const TARGETS: &str = r#"
[targets.gpu]
kind = "runpod"
gpu_type = "NVIDIA A40"
image = "axolotl:latest"
max_hours = 1.5

[targets.box]
kind = "ssh"
runtime = "native"
"#;

#[test]
fn env_overrides_numeric_and_string_fields_of_targets() -> Result<(), Box<dyn std::error::Error>> {
    let dir = project(&format!("{BASE}{TARGETS}"))?;
    let settings = load(
        dir.path(),
        env(&[
            ("OVERBRAINER_TARGETS__GPU__MAX_HOURS", "2"),
            ("OVERBRAINER_TARGETS__GPU__GPU_COUNT", "4"),
            ("OVERBRAINER_TARGETS__GPU__CONTAINER_DISK_GB", "120"),
            ("OVERBRAINER_TARGETS__BOX__HOST", "trainer@gpu-box"),
        ]),
    )?;
    match settings.targets.get("gpu") {
        Some(Target::Runpod {
            max_hours,
            gpu_count,
            container_disk_gb,
            ..
        }) => {
            assert!(
                (max_hours - 2.0).abs() < f64::EPSILON,
                "max_hours = {max_hours}"
            );
            assert_eq!(*gpu_count, 4);
            assert_eq!(*container_disk_gb, 120);
        },
        other => return Err(format!("expected runpod target, got {other:?}").into()),
    }
    match settings.targets.get("box") {
        Some(Target::Ssh { host, .. }) => {
            assert_eq!(host.as_deref(), Some("trainer@gpu-box"));
        },
        other => return Err(format!("expected ssh target, got {other:?}").into()),
    }
    Ok(())
}

#[test]
fn target_numbers_from_the_file_and_defaults_still_apply() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = project(&format!("{BASE}{TARGETS}"))?;
    let settings = load(dir.path(), env(&[]))?;
    match settings.targets.get("gpu") {
        Some(Target::Runpod {
            max_hours,
            gpu_count,
            container_disk_gb,
            ..
        }) => {
            assert!(
                (max_hours - 1.5).abs() < f64::EPSILON,
                "max_hours = {max_hours}"
            );
            assert_eq!(*gpu_count, 1);
            assert_eq!(*container_disk_gb, 50);
            Ok(())
        },
        other => Err(format!("expected runpod target, got {other:?}").into()),
    }
}

#[test]
fn non_numeric_env_value_for_a_numeric_target_field_is_rejected()
-> Result<(), Box<dyn std::error::Error>> {
    let dir = project(&format!("{BASE}{TARGETS}"))?;
    assert!(matches!(
        load(
            dir.path(),
            env(&[("OVERBRAINER_TARGETS__GPU__GPU_COUNT", "many")])
        ),
        Err(ConfigError::Parse(_))
    ));
    Ok(())
}

fn env_only_problems(toml: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let dir = project(toml)?;
    match load(dir.path(), env(&[])) {
        Err(ConfigError::Invalid(problems)) => Ok(problems),
        other => Err(format!("expected Invalid, got {other:?}").into()),
    }
}

#[test]
fn target_host_in_file_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
    let toml = format!(
        "{BASE}{}",
        TARGETS.replace(
            "kind = \"ssh\"",
            "kind = \"ssh\"\nhost = \"leak@private-host\""
        )
    );
    let problems = env_only_problems(&toml)?;
    assert!(
        problems.contains(
            &"targets.box.host: must be set through env, not in overbrainer.toml".to_string()
        ),
        "{problems:?}"
    );
    assert!(
        !problems.join("\n").contains("private-host"),
        "values must not leak"
    );
    Ok(())
}

#[test]
fn runpod_api_key_in_file_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
    let toml = format!("{BASE}\n[runpod]\napi_key = \"rp-leak\"\n");
    let problems = env_only_problems(&toml)?;
    assert!(
        problems.contains(
            &"runpod.api_key: must be set through env, not in overbrainer.toml".to_string()
        ),
        "{problems:?}"
    );
    assert!(
        !problems.join("\n").contains("rp-leak"),
        "values must not leak"
    );
    Ok(())
}

#[test]
fn log_in_file_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
    // Prepended so that `log` lands in the root table.
    let problems = env_only_problems(&format!("log = \"debug\"\n{BASE}"))?;
    assert!(
        problems.contains(&"log: must be set through env, not in overbrainer.toml".to_string()),
        "{problems:?}"
    );
    Ok(())
}

#[test]
fn log_from_env_is_accepted() -> Result<(), Box<dyn std::error::Error>> {
    let dir = project(BASE)?;
    let settings = load(dir.path(), env(&[("OVERBRAINER_LOG", "debug")]))?;
    assert_eq!(settings.log.as_deref(), Some("debug"));
    Ok(())
}

#[test]
fn type_errors_name_the_key_but_not_the_value() -> Result<(), Box<dyn std::error::Error>> {
    let dir = project(&format!("{BASE}{TARGETS}"))?;
    for (key, path) in [
        ("OVERBRAINER_PIPELINE__CONCURRENCY", "pipeline.concurrency"),
        (
            "OVERBRAINER_ROLES__PARENT__REASONING",
            "roles.parent.reasoning",
        ),
        (
            "OVERBRAINER_ROLES__PARENT__MAX_TOKENS",
            "roles.parent.max_tokens",
        ),
        (
            "OVERBRAINER_PROVIDERS__NANOGPT__PROTOCOL",
            "providers.nanogpt.protocol",
        ),
        ("OVERBRAINER_TARGETS__GPU__GPU_COUNT", "targets.gpu"),
        ("OVERBRAINER_TARGETS__GPU__KIND", "targets.gpu.kind"),
    ] {
        match load(dir.path(), env(&[(key, "sk-leak-7")])) {
            Err(error @ ConfigError::Parse(_)) => {
                let text = error.to_string();
                assert!(text.contains(path), "{key}: {text}");
                assert!(!text.contains("sk-leak-7"), "{key} leaked: {text}");
                assert!(
                    !format!("{error:?}").contains("sk-leak-7"),
                    "{key} leaked in Debug"
                );
            },
            other => return Err(format!("{key}: expected Parse, got {other:?}").into()),
        }
    }
    Ok(())
}

#[test]
fn toml_syntax_errors_never_echo_source() -> Result<(), Box<dyn std::error::Error>> {
    let malformed_toml = r#"
[project]
name = "demo"

not valid toml here = SK-LEAK-MARKER-999 !!broken!!
"#;
    let dir = project(malformed_toml)?;
    match load(dir.path(), env(&[])) {
        Err(error @ ConfigError::Parse(_)) => {
            let text = error.to_string();
            let debug = format!("{error:?}");
            assert!(
                !text.contains("SK-LEAK-MARKER-999"),
                "marker leaked in Display: {text}"
            );
            assert!(
                !debug.contains("SK-LEAK-MARKER-999"),
                "marker leaked in Debug: {debug}"
            );
            assert_eq!(
                text,
                "invalid configuration: TOML syntax error in overbrainer.toml at line 5, column 5",
                "the error names the location only"
            );
            Ok(())
        },
        other => Err(format!("expected Parse, got {other:?}").into()),
    }
}

#[test]
fn ssh_target_options_come_from_the_file_and_env() -> Result<(), Box<dyn std::error::Error>> {
    let toml = format!(
        "{BASE}{}",
        TARGETS.replace(
            "runtime = \"native\"",
            "runtime = \"docker\"\nengine = \"podman\""
        )
    );
    let dir = project(&toml)?;
    let settings = load(
        dir.path(),
        env(&[("OVERBRAINER_TARGETS__BOX__WORKDIR", "/data/overbrainer")]),
    )?;
    match settings.targets.get("box") {
        Some(Target::Ssh {
            engine,
            workdir,
            image,
            ..
        }) => {
            assert_eq!(*engine, Some(Engine::Podman));
            assert_eq!(workdir.as_deref(), Some("/data/overbrainer"));
            assert_eq!(*image, None);
            Ok(())
        },
        other => Err(format!("expected ssh target, got {other:?}").into()),
    }
}

const TRAINING: &str = r#"
[training]
target = "box"
base_model = "Qwen/Qwen3-4B"
adapter = "qlora"

[training.axolotl_extra]
chat_template = "qwen3"
revision = "0123"
special_tokens = { pad_token = "<|endoftext|>" }

[targets.box]
kind = "ssh"
runtime = "native"
"#;

#[test]
fn env_values_in_axolotl_extra_get_their_types() -> Result<(), Box<dyn std::error::Error>> {
    let dir = project(&format!("{BASE}{TRAINING}"))?;
    let settings = load(
        dir.path(),
        env(&[
            (
                "OVERBRAINER_TRAINING__AXOLOTL_EXTRA__GRADIENT_CHECKPOINTING",
                "true",
            ),
            ("OVERBRAINER_TRAINING__AXOLOTL_EXTRA__WARMUP_STEPS", "10"),
            ("OVERBRAINER_TRAINING__AXOLOTL_EXTRA__WEIGHT_DECAY", "0.01"),
            (
                "OVERBRAINER_TRAINING__AXOLOTL_EXTRA__ATTN_IMPLEMENTATION",
                "sdpa",
            ),
            (
                "OVERBRAINER_TRAINING__AXOLOTL_EXTRA__SPECIAL_TOKENS__EOS_TOKEN",
                "<|im_end|>",
            ),
        ]),
    )?;
    let extra = &settings.training.ok_or("training missing")?.axolotl_extra;
    assert_eq!(
        serde_json::to_value(extra)?,
        serde_json::json!({
            "attn_implementation": "sdpa",
            "chat_template": "qwen3",
            "gradient_checkpointing": true,
            "revision": "0123",
            "special_tokens": {"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"},
            "warmup_steps": 10,
            "weight_decay": 0.01
        })
    );
    Ok(())
}
