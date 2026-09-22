use super::types::{Protocol, Runtime, Settings, Target};

/// Highest `pipeline.concurrency`: far above what providers allow, and well within
/// what a semaphore can hold.
const MAX_CONCURRENCY: usize = 1024;

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

/// The training section must point at a declared target.
fn check_training(settings: &Settings, problems: &mut Vec<String>) {
    if let Some(training) = &settings.training
        && !settings.targets.contains_key(&training.target)
    {
        problems.push(format!(
            "training.target: unknown target `{}`",
            training.target
        ));
    }
}

/// Per-kind target requirements.
fn check_targets(settings: &Settings, problems: &mut Vec<String>) {
    for (name, target) in &settings.targets {
        match target {
            Target::Local { runtime, image, .. } | Target::Ssh { runtime, image, .. } => {
                if *runtime == Runtime::Docker && image.is_none() {
                    problems.push(format!("targets.{name}: runtime `docker` requires `image`"));
                }
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
    fn docker_runtime_requires_an_image() -> Result<(), config::ConfigError> {
        let toml = VALID.replace(r#"runtime = "native""#, r#"runtime = "docker""#);
        let problems = check(&settings(&toml)?);
        assert_eq!(
            problems,
            vec!["targets.local: runtime `docker` requires `image`".to_string()]
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
