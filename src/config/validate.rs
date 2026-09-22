use super::types::{Runtime, Settings, Target};

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

    for name in settings.providers.keys() {
        if !is_valid_name(name) {
            problems.push(format!("providers.{name}: name must match ^[a-z0-9_]+$"));
        }
    }
    for name in settings.targets.keys() {
        if !is_valid_name(name) {
            problems.push(format!("targets.{name}: name must match ^[a-z0-9_]+$"));
        }
    }

    let roles = [
        ("generator", Some(&settings.roles.generator)),
        ("parent", Some(&settings.roles.parent)),
        ("embedder", settings.roles.embedder.as_ref()),
    ];
    for (role, model) in roles {
        if let Some(model) = model
            && !settings.providers.contains_key(&model.provider)
        {
            problems.push(format!(
                "roles.{role}: unknown provider `{}`",
                model.provider
            ));
        }
    }

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

    let pipeline = &settings.pipeline;
    if pipeline.concurrency == 0 {
        problems.push("pipeline.concurrency: must be at least 1".to_string());
    }
    if !(pipeline.eval_ratio > 0.0 && pipeline.eval_ratio < 1.0) {
        problems.push("pipeline.eval_ratio: must be in (0, 1)".to_string());
    }
    if !(pipeline.dedup_threshold > 0.0 && pipeline.dedup_threshold <= 1.0) {
        problems.push("pipeline.dedup_threshold: must be in (0, 1]".to_string());
    }

    if let Some(training) = &settings.training
        && !settings.targets.contains_key(&training.target)
    {
        problems.push(format!(
            "training.target: unknown target `{}`",
            training.target
        ));
    }

    for (name, target) in &settings.targets {
        match target {
            Target::Local { runtime, image, .. } | Target::Ssh { runtime, image, .. } => {
                if *runtime == Runtime::Docker && image.is_none() {
                    problems.push(format!("targets.{name}: runtime `docker` requires `image`"));
                }
            }
            Target::Runpod {
                max_hours,
                gpu_count,
                ..
            } => {
                if *max_hours <= 0.0 {
                    problems.push(format!("targets.{name}.max_hours: must be greater than 0"));
                }
                if *gpu_count == 0 {
                    problems.push(format!("targets.{name}.gpu_count: must be at least 1"));
                }
            }
        }
    }

    problems
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
