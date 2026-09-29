use std::fs;

use overbrainer::config::{
    ConfigError, Engine, EnvSource, ListOrAuto, Target, env_keys, load, load_str,
};
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
fn the_tui_variables_are_not_configuration_keys() -> Result<(), Box<dyn std::error::Error>> {
    let dir = project(BASE)?;
    load(
        dir.path(),
        env(&[
            ("OVERBRAINER_TUI_COLOR", "256"),
            ("OVERBRAINER_TUI_MOTION", "off"),
        ]),
    )?;
    Ok(())
}

#[test]
fn a_lower_case_tui_variable_is_not_a_configuration_key() -> Result<(), Box<dyn std::error::Error>>
{
    // `config::Environment` lower-cases keys, so the TUI prefix must match in any case.
    load_str(
        BASE,
        env(&[
            ("overbrainer_tui_color", "256"),
            ("Overbrainer_Tui_Motion", "off"),
        ]),
    )?;
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

/// The message with its chain of causes, as `{:#}` prints it at the top level.
fn chain(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut cause = error.source();
    while let Some(next) = cause {
        text = format!("{text}: {next}");
        cause = next.source();
    }
    text
}

#[test]
fn a_read_error_names_its_cause_once() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let Err(error) = load(dir.path(), env(&[])) else {
        return Err("expected an error".into());
    };
    for text in [error.to_string(), chain(&error)] {
        assert!(text.starts_with("cannot read "), "{text}");
        assert_eq!(text.matches("(os error 2)").count(), 1, "{text}");
    }
    Ok(())
}

const TARGETS: &str = r#"
[targets.gpu]
kind = "runpod"
gpu_types = ["NVIDIA A40"]
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

#[test]
fn runpod_lists_come_from_a_comma_separated_env_value() -> Result<(), Box<dyn std::error::Error>> {
    let dir = project(&format!("{BASE}{TARGETS}"))?;
    let settings = load(
        dir.path(),
        env(&[
            (
                "OVERBRAINER_TARGETS__GPU__GPU_TYPES",
                "NVIDIA GeForce RTX 4090, NVIDIA A40",
            ),
            ("OVERBRAINER_TARGETS__GPU__DATA_CENTER_IDS", "EU-RO-1"),
            ("OVERBRAINER_TARGETS__GPU__NETWORK_VOLUME_ID", "vol123"),
            ("OVERBRAINER_TARGETS__GPU__BOOT_GRACE_MINUTES", "45"),
            ("OVERBRAINER_TARGETS__GPU__RETRIEVE_GRACE_MINUTES", "90"),
        ]),
    )?;
    match settings.targets.get("gpu") {
        Some(Target::Runpod {
            gpu_types,
            data_center_ids,
            network_volume_id,
            boot_grace_minutes,
            retrieve_grace_minutes,
            ..
        }) => {
            assert_eq!(gpu_types.list(), &["NVIDIA GeForce RTX 4090", "NVIDIA A40"]);
            assert_eq!(data_center_ids.list(), &["EU-RO-1"]);
            assert_eq!(network_volume_id.as_deref(), Some("vol123"));
            assert_eq!(*boot_grace_minutes, 45);
            assert_eq!(*retrieve_grace_minutes, 90);
            Ok(())
        },
        other => Err(format!("expected runpod target, got {other:?}").into()),
    }
}

#[test]
fn the_runpod_base_url_comes_from_env_and_must_be_https() -> Result<(), Box<dyn std::error::Error>>
{
    let dir = project(BASE)?;
    for url in [
        "https://api.runpod.io/v2",
        "http://127.0.0.1:8080/v2",
        "http://localhost:1",
    ] {
        let settings = load(dir.path(), env(&[("OVERBRAINER_RUNPOD__BASE_URL", url)]))?;
        assert_eq!(settings.runpod.base_url.as_deref(), Some(url));
    }
    for url in ["http://api.runpod.io/v2", "ftp://127.0.0.1/", "not a url"] {
        match load(dir.path(), env(&[("OVERBRAINER_RUNPOD__BASE_URL", url)])) {
            Err(ConfigError::Invalid(problems)) => assert_eq!(
                problems,
                vec![
                    "runpod.base_url: must be an https URL (http only on a loopback host)"
                        .to_string()
                ],
                "{url}"
            ),
            other => return Err(format!("{url}: expected Invalid, got {other:?}").into()),
        }
    }
    Ok(())
}

#[test]
fn runpod_base_url_in_file_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
    let toml = format!("{BASE}\n[runpod]\nbase_url = \"https://api.runpod.io/v2\"\n");
    let problems = env_only_problems(&toml)?;
    assert!(
        problems.contains(
            &"runpod.base_url: must be set through env, not in overbrainer.toml".to_string()
        ),
        "{problems:?}"
    );
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

/// Runs both `load` (reading `toml` from a temp file) and `load_str` (given `toml`
/// directly) with the same `env`, and asserts they agree: same `Ok`/`Err` shape, and
/// the same `Debug` rendering (which, for `Settings`, never includes a secret's value:
/// `SecretString`'s `Debug` impl always prints a redacted placeholder).
fn assert_load_and_load_str_agree(
    toml: &str,
    pairs: EnvSource,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = project(toml)?;
    let content = fs::read_to_string(dir.path().join("overbrainer.toml"))?;
    let from_load = load(dir.path(), pairs.clone());
    let from_load_str = load_str(&content, pairs);
    assert_eq!(
        format!("{from_load:?}"),
        format!("{from_load_str:?}"),
        "load and load_str disagree for this fixture"
    );
    Ok(())
}

#[test]
fn load_str_agrees_with_load_on_a_valid_document() -> Result<(), Box<dyn std::error::Error>> {
    assert_load_and_load_str_agree(BASE, env(&[("OVERBRAINER_PIPELINE__CONCURRENCY", "16")]))
}

#[test]
fn load_str_agrees_with_load_on_targets_and_axolotl_extra() -> Result<(), Box<dyn std::error::Error>>
{
    assert_load_and_load_str_agree(
        &format!("{BASE}{TARGETS}{TRAINING}"),
        env(&[
            ("OVERBRAINER_TARGETS__GPU__MAX_HOURS", "2"),
            ("OVERBRAINER_TARGETS__BOX__HOST", "trainer@gpu-box"),
            (
                "OVERBRAINER_TRAINING__AXOLOTL_EXTRA__GRADIENT_CHECKPOINTING",
                "true",
            ),
        ]),
    )
}

#[test]
fn load_str_agrees_with_load_on_an_env_only_key_leaked_in_the_file()
-> Result<(), Box<dyn std::error::Error>> {
    let toml = format!("{BASE}\n[runpod]\napi_key = \"rp-leak\"\n");
    assert_load_and_load_str_agree(&toml, env(&[]))
}

#[test]
fn load_str_agrees_with_load_on_malformed_toml() -> Result<(), Box<dyn std::error::Error>> {
    let malformed = "not valid toml here = !!broken!!\n";
    assert_load_and_load_str_agree(malformed, env(&[]))
}

#[test]
fn load_str_of_an_invalid_document_returns_dotted_path_messages()
-> Result<(), Box<dyn std::error::Error>> {
    // No file, no temp dir: `load_str` validates the text directly.
    let toml = format!("{BASE}\n[runpod]\napi_key = \"rp-leak\"\n");
    match load_str(&toml, env(&[])) {
        Err(ConfigError::Invalid(problems)) => {
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
        },
        other => Err(format!("expected Invalid, got {other:?}").into()),
    }
}

#[test]
fn load_str_of_a_valid_document_matches_load() -> Result<(), Box<dyn std::error::Error>> {
    let settings = load_str(BASE, env(&[]))?;
    assert_eq!(settings.project.name, "demo");
    Ok(())
}

#[test]
fn env_keys_from_vars_maps_dotted_lower_case_keys() {
    let keys = env_keys(&env(&[
        ("OVERBRAINER_PROVIDERS__OPENROUTER__API_KEY", "sk-x"),
        ("OVERBRAINER_LOG", "debug"),
        ("OVERBRAINER_TUI_COLOR", "256"),
        ("OVERBRAINER_TARGETS__GPU__MAX_HOURS", "2"),
        ("UNRELATED_VAR", "1"),
    ]));
    assert_eq!(
        keys,
        std::collections::BTreeSet::from([
            "providers.openrouter.api_key".to_string(),
            "log".to_string(),
            "targets.gpu.max_hours".to_string(),
        ])
    );
}

#[test]
fn env_keys_agrees_with_load_str_on_a_lower_case_variable_name()
-> Result<(), Box<dyn std::error::Error>> {
    // `overbrainer_log` (lower case): `load_str` applies it because `config::Environment`
    // lower-cases every key before matching its prefix. `env_keys` must report it too.
    let pairs = env(&[("overbrainer_log", "debug")]);
    let settings = load_str(BASE, pairs.clone())?;
    assert_eq!(settings.log.as_deref(), Some("debug"));
    assert_eq!(
        env_keys(&pairs),
        std::collections::BTreeSet::from(["log".to_string()])
    );
    Ok(())
}

#[test]
fn env_keys_agrees_with_load_str_on_a_mixed_case_nested_key()
-> Result<(), Box<dyn std::error::Error>> {
    let pairs = env(&[("Overbrainer_Providers__Nanogpt__Api_Key", "sk-mixed-case")]);
    let settings = load_str(BASE, pairs.clone())?;
    let provider = settings
        .providers
        .get("nanogpt")
        .ok_or("provider missing")?;
    let key = provider.api_key.as_ref().ok_or("api_key missing")?;
    assert_eq!(key.expose_secret(), "sk-mixed-case");
    assert_eq!(
        env_keys(&pairs),
        std::collections::BTreeSet::from(["providers.nanogpt.api_key".to_string()])
    );
    Ok(())
}

#[test]
fn runpod_auto_comes_from_env_and_from_the_file() -> Result<(), Box<dyn std::error::Error>> {
    let dir = project(&format!("{BASE}{TARGETS}"))?;
    let settings = load(
        dir.path(),
        env(&[
            ("OVERBRAINER_TARGETS__GPU__GPU_TYPES", "auto"),
            ("OVERBRAINER_TARGETS__GPU__DATA_CENTER_IDS", " auto "),
            ("OVERBRAINER_TARGETS__GPU__MIN_VRAM_GB", "48"),
            ("OVERBRAINER_TARGETS__GPU__MAX_PRICE_PER_HOUR", "1.25"),
        ]),
    )?;
    match settings.targets.get("gpu") {
        Some(Target::Runpod {
            gpu_types,
            data_center_ids,
            min_vram_gb,
            max_price_per_hour,
            ..
        }) => {
            assert_eq!(*gpu_types, ListOrAuto::Auto);
            assert_eq!(*data_center_ids, ListOrAuto::Auto);
            assert_eq!(*min_vram_gb, Some(48));
            assert_eq!(*max_price_per_hour, Some(1.25));
        },
        other => return Err(format!("expected runpod target, got {other:?}").into()),
    }
    let file = TARGETS.replace(
        "gpu_types = [\"NVIDIA A40\"]",
        "gpu_types = \"auto\"\ndata_center_ids = \"auto\"\nmin_vram_gb = 24",
    );
    let dir = project(&format!("{BASE}{file}"))?;
    let settings = load(dir.path(), env(&[]))?;
    match settings.targets.get("gpu") {
        Some(Target::Runpod {
            gpu_types,
            data_center_ids,
            min_vram_gb,
            max_price_per_hour,
            ..
        }) => {
            assert_eq!(*gpu_types, ListOrAuto::Auto);
            assert_eq!(*data_center_ids, ListOrAuto::Auto);
            assert_eq!(*min_vram_gb, Some(24));
            assert_eq!(*max_price_per_hour, None);
            Ok(())
        },
        other => Err(format!("expected runpod target, got {other:?}").into()),
    }
}

#[test]
fn only_lower_case_auto_is_auto() -> Result<(), Box<dyn std::error::Error>> {
    let dir = project(&format!("{BASE}{TARGETS}"))?;
    let settings = load(
        dir.path(),
        env(&[("OVERBRAINER_TARGETS__GPU__GPU_TYPES", "Auto")]),
    );
    // A list holding it, which validation then refuses.
    let Err(ConfigError::Invalid(problems)) = settings else {
        return Err(format!("expected invalid settings, got {settings:?}").into());
    };
    assert!(
        problems.iter().any(|problem| problem
            == "targets.gpu.gpu_types: write gpu_types = \"auto\", not a list holding it"),
        "{problems:?}"
    );
    Ok(())
}

#[test]
fn metrics_listen_comes_from_the_file_or_env() -> Result<(), Box<dyn std::error::Error>> {
    let dir = project(BASE)?;
    assert_eq!(load(dir.path(), env(&[]))?.metrics.listen, None);
    let dir = project(&format!("{BASE}\n[metrics]\nlisten = \"127.0.0.1:9464\"\n"))?;
    let settings = load(dir.path(), env(&[]))?;
    assert_eq!(settings.metrics.listen, Some("127.0.0.1:9464".parse()?));
    let dir = project(BASE)?;
    let settings = load(
        dir.path(),
        env(&[("OVERBRAINER_METRICS__LISTEN", "[::1]:9000")]),
    )?;
    assert_eq!(settings.metrics.listen, Some("[::1]:9000".parse()?));
    Ok(())
}

#[test]
fn an_invalid_metrics_address_is_refused_with_its_path() -> Result<(), Box<dyn std::error::Error>> {
    for (file, vars) in [
        ("\n[metrics]\nlisten = \"localhost\"\n", &[][..]),
        ("", &[("OVERBRAINER_METRICS__LISTEN", "9464")][..]),
    ] {
        let dir = project(&format!("{BASE}{file}"))?;
        let error = load(dir.path(), env(vars))
            .err()
            .ok_or("an invalid address was accepted")?;
        assert!(error.to_string().contains("metrics.listen"), "{error}");
    }
    Ok(())
}
