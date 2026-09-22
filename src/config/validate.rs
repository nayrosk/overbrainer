use super::types::{Adapter, Protocol, Runtime, Settings, Target, Training};

/// Highest `pipeline.concurrency`: far above what providers allow, and well within
/// what a semaphore can hold.
const MAX_CONCURRENCY: usize = 1024;

/// Lowest `thinking_budget`: the smallest `budget_tokens` the `anthropic` protocol accepts.
const MIN_THINKING_BUDGET: u32 = 1024;

/// Returns true when `name` matches `^[a-z0-9_]+$`.
pub(crate) fn is_valid_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// Semantic checks that serde cannot express. Returns one message per problem.
pub(crate) fn check(settings: &Settings) -> Vec<String> {
    let mut problems = Vec::new();
    check_names(settings, &mut problems);
    check_roles(settings, &mut problems);
    check_role_params(settings, &mut problems);
    check_topics(settings, &mut problems);
    check_pipeline(settings, &mut problems);
    check_training(settings, &mut problems);
    check_targets(settings, &mut problems);
    problems
}

/// Provider and target names must be usable in env variable names.
fn check_names(settings: &Settings, problems: &mut Vec<String>) {
    let names = settings
        .providers
        .keys()
        .map(|name| ("providers", name))
        .chain(settings.targets.keys().map(|name| ("targets", name)));
    for (table, name) in names {
        if !is_valid_name(name) {
            problems.push(format!("{table}.{name}: name must match ^[a-z0-9_]+$"));
        }
    }
}

/// Every role must reference a declared provider; the embedder needs embeddings.
fn check_roles(settings: &Settings, problems: &mut Vec<String>) {
    for (role, model) in settings.roles.all() {
        if !settings.providers.contains_key(&model.provider) {
            problems.push(format!(
                "roles.{role}: unknown provider `{}`",
                model.provider
            ));
        }
    }
    if let Some(embedder) = &settings.roles.embedder
        && settings
            .providers
            .get(&embedder.provider)
            .is_some_and(|provider| provider.protocol == Protocol::Anthropic)
    {
        problems.push(
            "roles.embedder: the anthropic protocol has no embeddings, use an openai provider"
                .to_string(),
        );
    }
}

/// Per-role request parameters are within range.
fn check_role_params(settings: &Settings, problems: &mut Vec<String>) {
    for (role, model) in settings.roles.all() {
        if model.max_tokens == 0 {
            problems.push(format!("roles.{role}.max_tokens: must be at least 1"));
        }
        if let Some(temperature) = model.temperature
            && !(0.0..=2.0).contains(&temperature)
        {
            problems.push(format!("roles.{role}.temperature: must be in [0, 2]"));
        }
        if model.reasoning_effort.is_some() && !model.reasoning {
            problems.push(format!(
                "roles.{role}.reasoning_effort: requires reasoning = true"
            ));
        }
    }
    check_thinking_temperature(settings, problems);
    check_thinking_budget(settings, problems);
}

/// `thinking_budget` requires reasoning, a value in `[1024, max_tokens)`, no
/// `reasoning_effort`, and the `anthropic` protocol (the only one that has it).
fn check_thinking_budget(settings: &Settings, problems: &mut Vec<String>) {
    for (role, model) in settings.roles.all() {
        let Some(budget) = model.thinking_budget else {
            continue;
        };
        if !model.reasoning {
            problems.push(format!(
                "roles.{role}.thinking_budget: requires reasoning = true"
            ));
        }
        if model.reasoning_effort.is_some() {
            problems.push(format!(
                "roles.{role}.thinking_budget: cannot be combined with reasoning_effort"
            ));
        }
        let anthropic = settings
            .providers
            .get(&model.provider)
            .is_some_and(|provider| provider.protocol == Protocol::Anthropic);
        if !anthropic {
            problems.push(format!(
                "roles.{role}.thinking_budget: only valid on the anthropic protocol"
            ));
        }
        if budget < MIN_THINKING_BUDGET {
            problems.push(format!(
                "roles.{role}.thinking_budget: must be at least {MIN_THINKING_BUDGET}"
            ));
        }
        if budget >= model.max_tokens {
            problems.push(format!(
                "roles.{role}.thinking_budget: must be less than max_tokens"
            ));
        }
    }
}

/// The `anthropic` protocol rejects a temperature when thinking is enabled.
fn check_thinking_temperature(settings: &Settings, problems: &mut Vec<String>) {
    for (role, model) in settings.roles.all() {
        let anthropic = settings
            .providers
            .get(&model.provider)
            .is_some_and(|provider| provider.protocol == Protocol::Anthropic);
        if anthropic && model.reasoning && model.temperature.is_some() {
            problems.push(format!(
                "roles.{role}.temperature: the anthropic protocol does not accept a temperature with reasoning = true"
            ));
        }
    }
}

/// Topic names are unique and their counts are at least 1.
fn check_topics(settings: &Settings, problems: &mut Vec<String>) {
    let mut seen = std::collections::BTreeSet::new();
    for topic in &settings.topics {
        if !seen.insert(topic.name.as_str()) {
            problems.push(format!("topics: duplicate name `{}`", topic.name));
        }
        if topic.subtopics == 0 {
            problems.push(format!(
                "topics.{}.subtopics: must be at least 1",
                topic.name
            ));
        }
        if topic.questions_per_subtopic == 0 {
            problems.push(format!(
                "topics.{}.questions_per_subtopic: must be at least 1",
                topic.name
            ));
        }
    }
}

/// Pipeline counts and ratios are within range. NaN is always out of range.
fn check_pipeline(settings: &Settings, problems: &mut Vec<String>) {
    let pipeline = &settings.pipeline;
    if pipeline.concurrency == 0 {
        problems.push("pipeline.concurrency: must be at least 1".to_string());
    }
    if pipeline.concurrency > MAX_CONCURRENCY {
        problems.push(format!(
            "pipeline.concurrency: must be at most {MAX_CONCURRENCY}"
        ));
    }
    if !(pipeline.eval_ratio > 0.0 && pipeline.eval_ratio < 1.0) {
        problems.push("pipeline.eval_ratio: must be in (0, 1)".to_string());
    }
    if !(pipeline.dedup_threshold > 0.0 && pipeline.dedup_threshold <= 1.0) {
        problems.push("pipeline.dedup_threshold: must be in (0, 1]".to_string());
    }
    if !(pipeline.embedding_threshold > 0.0 && pipeline.embedding_threshold <= 1.0) {
        problems.push("pipeline.embedding_threshold: must be in (0, 1]".to_string());
    }
    if pipeline.question_batch_size == 0 {
        problems.push("pipeline.question_batch_size: must be at least 1".to_string());
    }
    if pipeline.request_timeout_secs == 0 {
        problems.push("pipeline.request_timeout_secs: must be at least 1".to_string());
    }
}

/// Axolotl keys set from a typed `[training]` key, with the key to use instead.
const TYPED_AXOLOTL_KEYS: [(&str, &str); 16] = [
    ("base_model", "base_model"),
    ("adapter", "adapter"),
    ("num_epochs", "epochs"),
    ("learning_rate", "learning_rate"),
    ("lora_r", "lora_r"),
    ("lora_alpha", "lora_alpha"),
    ("lora_dropout", "lora_dropout"),
    ("sequence_len", "sequence_len"),
    ("micro_batch_size", "micro_batch_size"),
    ("gradient_accumulation_steps", "gradient_accumulation_steps"),
    ("optimizer", "optimizer"),
    ("lr_scheduler", "lr_scheduler"),
    ("sample_packing", "sample_packing"),
    ("evals_per_epoch", "evals_per_epoch"),
    ("saves_per_epoch", "saves_per_epoch"),
    ("hub_model_id", "hub_model_id"),
];

/// Axolotl keys that overbrainer manages: the run layout and the metrics plugin
/// depend on them.
const MANAGED_AXOLOTL_KEYS: [&str; 6] = [
    "datasets",
    "test_datasets",
    "val_set_size",
    "output_dir",
    "dataset_prepared_path",
    "plugins",
];

/// The training section points at a declared target and its values are in range.
fn check_training(settings: &Settings, problems: &mut Vec<String>) {
    let Some(training) = &settings.training else {
        return;
    };
    if !settings.targets.contains_key(&training.target) {
        problems.push(format!(
            "training.target: unknown target `{}`",
            training.target
        ));
    }
    check_training_counts(training, problems);
    check_training_rates(training, problems);
    check_axolotl_extra(training, problems);
}

/// Counts are at least 1 and names are not empty.
fn check_training_counts(training: &Training, problems: &mut Vec<String>) {
    let counts = [
        ("epochs", training.epochs),
        ("lora_r", training.lora_r),
        ("lora_alpha", training.lora_alpha),
        ("sequence_len", training.sequence_len),
        ("micro_batch_size", training.micro_batch_size),
        (
            "gradient_accumulation_steps",
            training.gradient_accumulation_steps,
        ),
        ("evals_per_epoch", training.evals_per_epoch),
        ("saves_per_epoch", training.saves_per_epoch),
    ];
    for (key, value) in counts {
        if value == 0 {
            problems.push(format!("training.{key}: must be at least 1"));
        }
    }
    for (key, value) in [
        ("optimizer", &training.optimizer),
        ("lr_scheduler", &training.lr_scheduler),
    ] {
        if value.trim().is_empty() {
            problems.push(format!("training.{key}: must not be empty"));
        }
    }
    if training
        .hub_model_id
        .as_ref()
        .is_some_and(|id| id.trim().is_empty())
    {
        problems.push("training.hub_model_id: must not be empty".to_string());
    }
}

/// The learning rate and dropout are in range, and `merge` has an adapter to merge.
fn check_training_rates(training: &Training, problems: &mut Vec<String>) {
    if !(training.learning_rate > 0.0 && training.learning_rate.is_finite()) {
        problems.push("training.learning_rate: must be greater than 0".to_string());
    }
    if !(0.0..1.0).contains(&training.lora_dropout) {
        problems.push("training.lora_dropout: must be in [0, 1)".to_string());
    }
    if training.merge && training.adapter == Adapter::Full {
        problems.push(
            "training.merge: requires adapter = \"lora\" or \"qlora\" (a full fine-tune has no adapter to merge)"
                .to_string(),
        );
    }
}

/// `axolotl_extra` must not set a key that has a typed equivalent or that
/// overbrainer manages.
fn check_axolotl_extra(training: &Training, problems: &mut Vec<String>) {
    for key in training.axolotl_extra.keys() {
        if let Some((_, typed)) = TYPED_AXOLOTL_KEYS.iter().find(|(name, _)| name == key) {
            problems.push(format!(
                "training.axolotl_extra.{key}: set training.{typed} instead"
            ));
        }
        if MANAGED_AXOLOTL_KEYS.contains(&key.as_str()) {
            problems.push(format!(
                "training.axolotl_extra.{key}: managed by overbrainer, cannot be overridden"
            ));
        }
    }
}

/// Per-kind target requirements.
fn check_targets(settings: &Settings, problems: &mut Vec<String>) {
    for (name, target) in &settings.targets {
        match target {
            Target::Local {
                runtime,
                engine,
                image,
                venv,
            } => {
                check_runtime(
                    name,
                    *runtime,
                    engine.is_some() || image.is_some(),
                    venv.is_some(),
                    problems,
                );
                check_target_image(name, image.as_deref(), problems);
                check_target_venv(name, venv.as_deref(), problems);
            },
            Target::Ssh {
                runtime,
                engine,
                image,
                venv,
                workdir,
                ..
            } => {
                check_runtime(
                    name,
                    *runtime,
                    engine.is_some() || image.is_some(),
                    venv.is_some(),
                    problems,
                );
                check_target_image(name, image.as_deref(), problems);
                check_target_venv(name, venv.as_deref(), problems);
                check_target_workdir(name, workdir.as_deref(), problems);
            },
            Target::Runpod {
                max_hours,
                gpu_count,
                ..
            } => {
                if max_hours.is_nan() || *max_hours <= 0.0 {
                    problems.push(format!("targets.{name}.max_hours: must be greater than 0"));
                }
                if *gpu_count == 0 {
                    problems.push(format!("targets.{name}.gpu_count: must be at least 1"));
                }
            },
        }
    }
}

/// `engine` and `image` only apply to the `docker` runtime, `venv` only to `native`.
fn check_runtime(
    name: &str,
    runtime: Runtime,
    container: bool,
    venv: bool,
    problems: &mut Vec<String>,
) {
    match runtime {
        Runtime::Docker if venv => {
            problems.push(format!(
                "targets.{name}.venv: only with runtime = \"native\""
            ));
        },
        Runtime::Native if container => problems.push(format!(
            "targets.{name}: engine and image only apply with runtime = \"docker\""
        )),
        Runtime::Docker | Runtime::Native => {},
    }
}

/// `workdir` must not be empty and, when set, must be a safe path (see
/// [`check_safe_path`]). It reaches remote shell commands, so unsafe characters are
/// rejected here in addition to callers quoting it.
fn check_target_workdir(name: &str, workdir: Option<&str>, problems: &mut Vec<String>) {
    let Some(workdir) = workdir else {
        return;
    };
    if workdir.trim().is_empty() {
        problems.push(format!("targets.{name}.workdir: must not be empty"));
        return;
    }
    check_safe_path(name, "workdir", workdir, problems);
}

/// `venv`, when set, must be a safe path (see [`check_safe_path`]). It reaches remote
/// shell commands, so unsafe characters are rejected here in addition to callers
/// quoting it.
fn check_target_venv(name: &str, venv: Option<&str>, problems: &mut Vec<String>) {
    if let Some(venv) = venv {
        check_safe_path(name, "venv", venv, problems);
    }
}

/// Characters allowed in a `workdir` or `venv` value.
fn is_path_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | '~' | '-')
}

/// `value` uses only `[A-Za-z0-9._/~-]`, has no `..` path segment, does not start
/// with `-`, and uses `~` only as the first character, alone or followed by `/`.
fn check_safe_path(name: &str, field: &str, value: &str, problems: &mut Vec<String>) {
    if !value.chars().all(is_path_char) {
        problems.push(format!(
            "targets.{name}.{field}: only letters, digits and . _ / ~ - are allowed"
        ));
        return;
    }
    if value.starts_with('-') {
        problems.push(format!("targets.{name}.{field}: must not start with -"));
    }
    if value.split('/').any(|segment| segment == "..") {
        problems.push(format!("targets.{name}.{field}: must not contain .."));
    }
    if !valid_tilde_placement(value) {
        problems.push(format!(
            "targets.{name}.{field}: ~ is only allowed as the first character, alone or followed by /"
        ));
    }
}

/// `~` appears at most once, and only as the first character, alone or followed by `/`.
fn valid_tilde_placement(value: &str) -> bool {
    match value.matches('~').count() {
        0 => true,
        1 => value.starts_with('~') && (value.len() == 1 || value.as_bytes()[1] == b'/'),
        _ => false,
    }
}

/// `image`, when set, must use only `[A-Za-z0-9._/:@-]` and not start with `-`. It
/// reaches remote shell commands, so unsafe characters are rejected here in addition
/// to callers quoting it.
fn check_target_image(name: &str, image: Option<&str>, problems: &mut Vec<String>) {
    let Some(image) = image else {
        return;
    };
    if !image.chars().all(is_image_char) {
        problems.push(format!(
            "targets.{name}.image: only letters, digits and . _ / : @ - are allowed"
        ));
        return;
    }
    if image.starts_with('-') {
        problems.push(format!("targets.{name}.image: must not start with -"));
    }
}

/// Characters allowed in an `image` value.
fn is_image_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | ':' | '@' | '-')
}

/// Lists env-only keys that appear in the TOML file. Values are never included.
pub(crate) fn env_only_in_file(file: &config::Config) -> Vec<String> {
    let mut keys = Vec::new();
    let table_names = |table: &str| -> Vec<String> {
        file.get_table(table)
            .map(|map| map.into_keys().collect())
            .unwrap_or_default()
    };
    for name in table_names("providers") {
        keys.push(format!("providers.{name}.base_url"));
        keys.push(format!("providers.{name}.api_key"));
    }
    for name in table_names("targets") {
        keys.push(format!("targets.{name}.host"));
    }
    keys.push("runpod.api_key".to_string());
    keys.push("hf_token".to_string());
    keys.push("log".to_string());

    keys.into_iter()
        .filter(|key| file.get::<config::Value>(key).is_ok())
        .map(|key| format!("{key}: must be set through env, not in overbrainer.toml"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(toml: &str) -> Result<Settings, config::ConfigError> {
        config::Config::builder()
            .add_source(config::File::from_str(toml, config::FileFormat::Toml))
            .build()?
            .try_deserialize()
    }

    const VALID: &str = r#"
        [project]
        name = "demo"
        [[topics]]
        name = "ownership"
        subtopics = 3
        questions_per_subtopic = 5
        [providers.nanogpt]
        protocol = "openai"
        [roles]
        generator = { provider = "nanogpt", model = "m1" }
        parent = { provider = "nanogpt", model = "m2", reasoning = true }
        [training]
        target = "local"
        base_model = "Qwen/Qwen3-4B"
        adapter = "qlora"
        [targets.local]
        kind = "local"
        runtime = "native"
    "#;

    #[test]
    fn names_follow_the_allowed_charset() {
        assert!(is_valid_name("gpu_cloud2"));
        assert!(!is_valid_name(""));
        assert!(!is_valid_name("gpu-cloud"));
        assert!(!is_valid_name("GPU"));
    }

    #[test]
    fn valid_settings_have_no_problems() -> Result<(), config::ConfigError> {
        assert_eq!(check(&settings(VALID)?), Vec::<String>::new());
        Ok(())
    }

    #[test]
    fn role_must_reference_a_known_provider() -> Result<(), config::ConfigError> {
        let toml = VALID.replace(r#"model = "m1""#, r#"model = "m1", provider_x = 1"#);
        assert!(
            settings(&toml).is_err(),
            "unknown field must be rejected by serde"
        );
        let toml = VALID.replace(
            r#"{ provider = "nanogpt", model = "m1" }"#,
            r#"{ provider = "missing", model = "m1" }"#,
        );
        let problems = check(&settings(&toml)?);
        assert_eq!(
            problems,
            vec!["roles.generator: unknown provider `missing`".to_string()]
        );
        Ok(())
    }

    #[test]
    fn training_target_must_exist() -> Result<(), config::ConfigError> {
        let toml = VALID.replace(r#"target = "local""#, r#"target = "cloud""#);
        let problems = check(&settings(&toml)?);
        assert_eq!(
            problems,
            vec!["training.target: unknown target `cloud`".to_string()]
        );
        Ok(())
    }

    #[test]
    fn runpod_max_hours_must_be_a_positive_number() -> Result<(), config::ConfigError> {
        for value in ["0.0", "-1.0", "nan"] {
            let toml = format!(
                "{VALID}\n[targets.cloud]\nkind = \"runpod\"\ngpu_type = \"g\"\nimage = \"i\"\nmax_hours = {value}\n"
            );
            let problems = check(&settings(&toml)?);
            assert_eq!(
                problems,
                vec!["targets.cloud.max_hours: must be greater than 0".to_string()],
                "max_hours = {value}"
            );
        }
        Ok(())
    }

    #[test]
    fn runtime_options_match_the_runtime() -> Result<(), config::ConfigError> {
        let toml = VALID.replace(r#"runtime = "native""#, r#"runtime = "docker""#);
        assert_eq!(check(&settings(&toml)?), Vec::<String>::new());
        let toml = VALID.replace(
            r#"runtime = "native""#,
            "runtime = \"docker\"\nvenv = \"/opt/venv\"",
        );
        assert_eq!(
            check(&settings(&toml)?),
            vec!["targets.local.venv: only with runtime = \"native\"".to_string()]
        );
        let toml = VALID.replace(
            r#"runtime = "native""#,
            "runtime = \"native\"\nengine = \"podman\"",
        );
        assert_eq!(
            check(&settings(&toml)?),
            vec![
                "targets.local: engine and image only apply with runtime = \"docker\"".to_string()
            ]
        );
        Ok(())
    }

    #[test]
    fn ssh_workdir_must_not_be_empty() -> Result<(), config::ConfigError> {
        let toml = format!(
            "{VALID}\n[targets.box]\nkind = \"ssh\"\nruntime = \"native\"\nworkdir = \" \"\n"
        );
        assert_eq!(
            check(&settings(&toml)?),
            vec!["targets.box.workdir: must not be empty".to_string()]
        );
        Ok(())
    }

    #[test]
    fn target_paths_reject_unsafe_characters() -> Result<(), config::ConfigError> {
        for value in [
            "/opt/venv;rm",
            "/opt/venv`id`",
            "/opt/venv$(id)",
            "/opt/venv here",
        ] {
            let toml = VALID.replace(
                r#"runtime = "native""#,
                &format!("runtime = \"native\"\nvenv = \"{value}\""),
            );
            assert_eq!(
                check(&settings(&toml)?),
                vec![
                    "targets.local.venv: only letters, digits and . _ / ~ - are allowed"
                        .to_string()
                ],
                "venv = {value}"
            );
        }
        Ok(())
    }

    #[test]
    fn target_paths_reject_dot_dot_segments() -> Result<(), config::ConfigError> {
        for value in ["/opt/../etc", "..", "foo/../bar", "../foo"] {
            let toml = VALID.replace(
                r#"runtime = "native""#,
                &format!("runtime = \"native\"\nvenv = \"{value}\""),
            );
            assert_eq!(
                check(&settings(&toml)?),
                vec!["targets.local.venv: must not contain ..".to_string()],
                "venv = {value}"
            );
        }
        Ok(())
    }

    #[test]
    fn target_paths_reject_a_leading_dash() -> Result<(), config::ConfigError> {
        let toml = VALID.replace(
            r#"runtime = "native""#,
            "runtime = \"native\"\nvenv = \"-rf\"",
        );
        assert_eq!(
            check(&settings(&toml)?),
            vec!["targets.local.venv: must not start with -".to_string()]
        );
        Ok(())
    }

    #[test]
    fn target_paths_reject_a_misplaced_tilde() -> Result<(), config::ConfigError> {
        for value in ["a~b", "~foo", "~/foo~"] {
            let toml = VALID.replace(
                r#"runtime = "native""#,
                &format!("runtime = \"native\"\nvenv = \"{value}\""),
            );
            assert_eq!(
                check(&settings(&toml)?),
                vec![
                    "targets.local.venv: ~ is only allowed as the first character, alone or followed by /"
                        .to_string()
                ],
                "venv = {value}"
            );
        }
        Ok(())
    }

    #[test]
    fn target_paths_accept_safe_values() -> Result<(), config::ConfigError> {
        for value in ["/opt/axolotl-venv", "relative/path_1.2", "~", "~/venv"] {
            let toml = VALID.replace(
                r#"runtime = "native""#,
                &format!("runtime = \"native\"\nvenv = \"{value}\""),
            );
            assert_eq!(
                check(&settings(&toml)?),
                Vec::<String>::new(),
                "venv = {value}"
            );
        }
        Ok(())
    }

    #[test]
    fn ssh_workdir_rejects_dot_dot_segments() -> Result<(), config::ConfigError> {
        let toml = format!(
            "{VALID}\n[targets.box]\nkind = \"ssh\"\nruntime = \"native\"\nworkdir = \"/data/../etc\"\n"
        );
        assert_eq!(
            check(&settings(&toml)?),
            vec!["targets.box.workdir: must not contain ..".to_string()]
        );
        Ok(())
    }

    #[test]
    fn target_image_rejects_unsafe_characters() -> Result<(), config::ConfigError> {
        for value in [
            "repo/img;rm",
            "repo/img`id`",
            "repo/img$(id)",
            "repo/img here",
        ] {
            let toml = VALID.replace(
                r#"runtime = "native""#,
                &format!("runtime = \"docker\"\nimage = \"{value}\""),
            );
            assert_eq!(
                check(&settings(&toml)?),
                vec![
                    "targets.local.image: only letters, digits and . _ / : @ - are allowed"
                        .to_string()
                ],
                "image = {value}"
            );
        }
        Ok(())
    }

    #[test]
    fn target_image_rejects_a_leading_dash() -> Result<(), config::ConfigError> {
        let toml = VALID.replace(
            r#"runtime = "native""#,
            "runtime = \"docker\"\nimage = \"-x\"",
        );
        assert_eq!(
            check(&settings(&toml)?),
            vec!["targets.local.image: must not start with -".to_string()]
        );
        Ok(())
    }

    #[test]
    fn target_image_accepts_a_reference_with_a_digest() -> Result<(), config::ConfigError> {
        let toml = VALID.replace(
            r#"runtime = "native""#,
            "runtime = \"docker\"\nimage = \"axolotlai/axolotl:0.19.0-py3.12-cu130-2.12.1@sha256:9de7c7a5b8830480a7d2eb3b6d49759586615f5f8eb1126d5df29f8bd9fa324b\"",
        );
        assert_eq!(check(&settings(&toml)?), Vec::<String>::new());
        Ok(())
    }

    #[test]
    fn training_defaults_are_set() -> Result<(), Box<dyn std::error::Error>> {
        let settings = settings(VALID)?;
        let training = settings.training.ok_or("training missing")?;
        assert_eq!(training.lora_alpha, 32);
        assert!((training.lora_dropout - 0.05).abs() < f64::EPSILON);
        assert_eq!(training.micro_batch_size, 2);
        assert_eq!(training.gradient_accumulation_steps, 4);
        assert_eq!(training.optimizer, "adamw_torch_fused");
        assert_eq!(training.lr_scheduler, "cosine");
        assert!(training.sample_packing);
        assert_eq!(training.evals_per_epoch, 4);
        assert_eq!(training.saves_per_epoch, 1);
        Ok(())
    }

    #[test]
    fn training_values_are_checked() -> Result<(), config::ConfigError> {
        let toml = VALID.replace(
            "adapter = \"qlora\"",
            "adapter = \"full\"\nmerge = true\nepochs = 0\nmicro_batch_size = 0\nlearning_rate = -1.0\nlora_dropout = 1.0\noptimizer = \"\"\nhub_model_id = \"\"",
        );
        assert_eq!(
            check(&settings(&toml)?),
            vec![
                "training.epochs: must be at least 1".to_string(),
                "training.micro_batch_size: must be at least 1".to_string(),
                "training.optimizer: must not be empty".to_string(),
                "training.hub_model_id: must not be empty".to_string(),
                "training.learning_rate: must be greater than 0".to_string(),
                "training.lora_dropout: must be in [0, 1)".to_string(),
                "training.merge: requires adapter = \"lora\" or \"qlora\" (a full fine-tune has no adapter to merge)".to_string(),
            ]
        );
        Ok(())
    }

    #[test]
    fn axolotl_extra_cannot_replace_typed_or_managed_keys() -> Result<(), config::ConfigError> {
        let toml = format!(
            "{VALID}\n[training.axolotl_extra]\nnum_epochs = 5\noutput_dir = \"/tmp/x\"\nwarmup_ratio = 0.05\n"
        );
        assert_eq!(
            check(&settings(&toml)?),
            vec![
                "training.axolotl_extra.num_epochs: set training.epochs instead".to_string(),
                "training.axolotl_extra.output_dir: managed by overbrainer, cannot be overridden"
                    .to_string(),
            ]
        );
        Ok(())
    }

    #[test]
    fn invalid_names_and_ranges_are_reported() -> Result<(), config::ConfigError> {
        let toml = format!(
            "{}\n[pipeline]\nconcurrency = 0\neval_ratio = 1.5\ndedup_threshold = 0.0\n",
            VALID
                .replace("[providers.nanogpt]", "[providers.Nano-GPT]")
                .replace(r#"provider = "nanogpt""#, r#"provider = "Nano-GPT""#)
        );
        let problems = check(&settings(&toml)?);
        assert!(problems.contains(&"providers.Nano-GPT: name must match ^[a-z0-9_]+$".to_string()));
        assert!(problems.contains(&"pipeline.concurrency: must be at least 1".to_string()));
        assert!(problems.contains(&"pipeline.eval_ratio: must be in (0, 1)".to_string()));
        assert!(problems.contains(&"pipeline.dedup_threshold: must be in (0, 1]".to_string()));
        Ok(())
    }

    #[test]
    fn role_parameters_are_checked() -> Result<(), config::ConfigError> {
        let toml = VALID.replace(
            r#"{ provider = "nanogpt", model = "m1" }"#,
            r#"{ provider = "nanogpt", model = "m1", max_tokens = 0, temperature = 2.5, reasoning_effort = "high" }"#,
        );
        let problems = check(&settings(&toml)?);
        assert_eq!(
            problems,
            vec![
                "roles.generator.max_tokens: must be at least 1".to_string(),
                "roles.generator.temperature: must be in [0, 2]".to_string(),
                "roles.generator.reasoning_effort: requires reasoning = true".to_string(),
            ]
        );
        Ok(())
    }

    #[test]
    fn embedder_must_use_the_openai_protocol() -> Result<(), config::ConfigError> {
        let toml = VALID.replace(
            "[roles]",
            "[providers.claude]\nprotocol = \"anthropic\"\n[roles]\nembedder = { provider = \"claude\", model = \"e\" }",
        );
        let problems = check(&settings(&toml)?);
        assert_eq!(
            problems,
            vec![
                "roles.embedder: the anthropic protocol has no embeddings, use an openai provider"
                    .to_string()
            ]
        );
        Ok(())
    }

    #[test]
    fn role_parameters_have_defaults() -> Result<(), config::ConfigError> {
        let settings = settings(VALID)?;
        assert_eq!(settings.roles.parent.max_tokens, 16_384);
        assert_eq!(settings.roles.parent.temperature, None);
        assert_eq!(settings.roles.parent.reasoning_effort, None);
        assert_eq!(settings.pipeline.question_batch_size, 10);
        assert_eq!(settings.pipeline.request_timeout_secs, 600);
        Ok(())
    }

    #[test]
    fn new_pipeline_fields_are_checked() -> Result<(), config::ConfigError> {
        let toml = format!(
            "{VALID}\n[pipeline]\nembedding_threshold = 1.5\nquestion_batch_size = 0\nrequest_timeout_secs = 0\n"
        );
        let problems = check(&settings(&toml)?);
        assert_eq!(
            problems,
            vec![
                "pipeline.embedding_threshold: must be in (0, 1]".to_string(),
                "pipeline.question_batch_size: must be at least 1".to_string(),
                "pipeline.request_timeout_secs: must be at least 1".to_string(),
            ]
        );
        Ok(())
    }

    #[test]
    fn concurrency_is_capped() -> Result<(), config::ConfigError> {
        let toml = format!("{VALID}\n[pipeline]\nconcurrency = 1025\n");
        assert_eq!(
            check(&settings(&toml)?),
            vec!["pipeline.concurrency: must be at most 1024".to_string()]
        );
        let toml = format!("{VALID}\n[pipeline]\nconcurrency = 1024\n");
        assert_eq!(check(&settings(&toml)?), Vec::<String>::new());
        Ok(())
    }

    #[test]
    fn temperature_with_reasoning_is_rejected_on_the_anthropic_protocol()
    -> Result<(), config::ConfigError> {
        let with_claude = |parent: &str| {
            VALID.replace(
                r#"parent = { provider = "nanogpt", model = "m2", reasoning = true }"#,
                &format!("parent = {parent}\n[providers.claude]\nprotocol = \"anthropic\""),
            )
        };
        let toml = with_claude(
            r#"{ provider = "claude", model = "m2", reasoning = true, temperature = 0.7 }"#,
        );
        assert_eq!(
            check(&settings(&toml)?),
            vec![
                "roles.parent.temperature: the anthropic protocol does not accept a temperature with reasoning = true"
                    .to_string()
            ]
        );
        for parent in [
            r#"{ provider = "claude", model = "m2", temperature = 0.7 }"#,
            r#"{ provider = "claude", model = "m2", reasoning = true }"#,
            r#"{ provider = "nanogpt", model = "m2", reasoning = true, temperature = 0.7 }"#,
        ] {
            assert_eq!(
                check(&settings(&with_claude(parent))?),
                Vec::<String>::new(),
                "{parent}"
            );
        }
        Ok(())
    }

    /// Applies `generator` in place of the default `nanogpt`-backed generator, and
    /// declares a `claude` provider on the `anthropic` protocol for it to reference.
    fn with_anthropic_generator(generator: &str) -> String {
        format!(
            "{}\n[providers.claude]\nprotocol = \"anthropic\"",
            VALID.replace(
                r#"generator = { provider = "nanogpt", model = "m1" }"#,
                &format!("generator = {generator}"),
            )
        )
    }

    #[test]
    fn thinking_budget_requires_reasoning() -> Result<(), config::ConfigError> {
        let toml = with_anthropic_generator(
            r#"{ provider = "claude", model = "m1", thinking_budget = 2048 }"#,
        );
        let problems = check(&settings(&toml)?);
        assert_eq!(
            problems,
            vec!["roles.generator.thinking_budget: requires reasoning = true".to_string()]
        );
        Ok(())
    }

    #[test]
    fn thinking_budget_has_a_minimum() -> Result<(), config::ConfigError> {
        let toml = with_anthropic_generator(
            r#"{ provider = "claude", model = "m1", reasoning = true, thinking_budget = 512, max_tokens = 4096 }"#,
        );
        let problems = check(&settings(&toml)?);
        assert_eq!(
            problems,
            vec!["roles.generator.thinking_budget: must be at least 1024".to_string()]
        );
        Ok(())
    }

    #[test]
    fn thinking_budget_must_be_below_max_tokens() -> Result<(), config::ConfigError> {
        let toml = with_anthropic_generator(
            r#"{ provider = "claude", model = "m1", reasoning = true, thinking_budget = 4096, max_tokens = 4096 }"#,
        );
        let problems = check(&settings(&toml)?);
        assert_eq!(
            problems,
            vec!["roles.generator.thinking_budget: must be less than max_tokens".to_string()]
        );
        Ok(())
    }

    #[test]
    fn thinking_budget_cannot_combine_with_reasoning_effort() -> Result<(), config::ConfigError> {
        let toml = with_anthropic_generator(
            r#"{ provider = "claude", model = "m1", reasoning = true, thinking_budget = 2048, reasoning_effort = "high", max_tokens = 4096 }"#,
        );
        let problems = check(&settings(&toml)?);
        assert_eq!(
            problems,
            vec![
                "roles.generator.thinking_budget: cannot be combined with reasoning_effort"
                    .to_string()
            ]
        );
        Ok(())
    }

    #[test]
    fn thinking_budget_is_rejected_on_the_openai_protocol() -> Result<(), config::ConfigError> {
        let toml = VALID.replace(
            r#"generator = { provider = "nanogpt", model = "m1" }"#,
            r#"generator = { provider = "nanogpt", model = "m1", reasoning = true, thinking_budget = 2048 }"#,
        );
        let problems = check(&settings(&toml)?);
        assert_eq!(
            problems,
            vec![
                "roles.generator.thinking_budget: only valid on the anthropic protocol".to_string()
            ]
        );
        Ok(())
    }

    #[test]
    fn thinking_budget_is_accepted_when_valid() -> Result<(), config::ConfigError> {
        let toml = with_anthropic_generator(
            r#"{ provider = "claude", model = "m1", reasoning = true, thinking_budget = 2048, max_tokens = 4096 }"#,
        );
        assert_eq!(check(&settings(&toml)?), Vec::<String>::new());
        Ok(())
    }

    #[test]
    fn topics_must_be_unique_and_non_zero() -> Result<(), config::ConfigError> {
        let toml = format!(
            "{VALID}\n[[topics]]\nname = \"ownership\"\nsubtopics = 0\nquestions_per_subtopic = 1\n"
        );
        let problems = check(&settings(&toml)?);
        assert!(problems.contains(&"topics: duplicate name `ownership`".to_string()));
        assert!(problems.contains(&"topics.ownership.subtopics: must be at least 1".to_string()));
        Ok(())
    }
}
