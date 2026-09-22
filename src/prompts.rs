//! Prompt templates: built-in defaults, overridable per project in `prompts/`.

use std::io;
use std::path::{Path, PathBuf};

use minijinja::{AutoEscape, Environment, UndefinedBehavior};
use serde::Serialize;

/// Directory of the project that holds template overrides.
pub const DIR: &str = "prompts";
/// Template asking the generator for the subtopics of a topic.
pub const SUBTOPICS: &str = "subtopics.txt";
/// Template asking the generator for questions about a subtopic.
pub const QUESTIONS: &str = "questions.txt";
/// System prompt sent to the parent with each question.
pub const ANSWER_SYSTEM: &str = "answer_system.txt";

/// Built-in templates, by file name. `init` writes them to `prompts/`.
pub const DEFAULTS: [(&str, &str); 3] = [
    (
        SUBTOPICS,
        include_str!("../templates/prompts/subtopics.txt"),
    ),
    (
        QUESTIONS,
        include_str!("../templates/prompts/questions.txt"),
    ),
    (
        ANSWER_SYSTEM,
        include_str!("../templates/prompts/answer_system.txt"),
    ),
];

/// Errors from loading or rendering templates.
#[derive(Debug, thiserror::Error)]
pub enum PromptError {
    /// An override file exists but cannot be read.
    #[error("cannot read {}", path.display())]
    Read {
        /// The override file.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// A template does not parse.
    #[error("invalid template {name}")]
    Template {
        /// Template file name.
        name: String,
        /// Parse error with line information.
        #[source]
        source: minijinja::Error,
    },
    /// A template failed to render, for example because it uses an undefined variable.
    #[error("cannot render template {name}")]
    Render {
        /// Template file name.
        name: String,
        /// Render error with line information.
        #[source]
        source: minijinja::Error,
    },
}

/// The loaded templates. Undefined variables are errors; output is never escaped.
#[derive(Debug)]
pub struct Prompts {
    env: Environment<'static>,
}

impl Prompts {
    /// Loads each template from `<project_dir>/prompts/<name>` when that file exists,
    /// otherwise from the built-in default.
    ///
    /// # Errors
    ///
    /// Returns [`PromptError::Read`] if an override exists but cannot be read, and
    /// [`PromptError::Template`] if a template does not parse.
    pub fn load(project_dir: &Path) -> Result<Self, PromptError> {
        let mut env = Environment::new();
        env.set_undefined_behavior(UndefinedBehavior::Strict);
        env.set_auto_escape_callback(|_| AutoEscape::None);
        for (name, default) in DEFAULTS {
            let source = read_override(&project_dir.join(DIR).join(name))?
                .unwrap_or_else(|| default.to_string());
            env.add_template_owned(name, source)
                .map_err(|source| PromptError::Template {
                    name: name.to_string(),
                    source,
                })?;
        }
        Ok(Self { env })
    }

    /// No template at all: every render fails. For stages that render none, such as
    /// the split, so a broken override cannot stop them.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            env: Environment::new(),
        }
    }

    /// Renders template `name` with `context`.
    ///
    /// # Errors
    ///
    /// Returns [`PromptError::Render`] if the template is unknown or fails to render.
    pub fn render<S: Serialize>(&self, name: &str, context: S) -> Result<String, PromptError> {
        let error = |source| PromptError::Render {
            name: name.to_string(),
            source,
        };
        self.env
            .get_template(name)
            .map_err(error)?
            .render(context)
            .map_err(error)
    }
}

fn read_override(path: &Path) -> Result<Option<String>, PromptError> {
    match std::fs::read_to_string(path) {
        Ok(source) => Ok(Some(source)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(PromptError::Read {
            path: path.to_path_buf(),
            source,
        }),
    }
}

#[cfg(test)]
mod tests {
    use minijinja::context;

    use super::*;

    #[test]
    fn defaults_render_with_their_documented_variables() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let prompts = Prompts::load(dir.path())?;
        let text = prompts.render(
            SUBTOPICS,
            context! { topic => "ownership", description => None::<String>, count => 3 },
        )?;
        assert!(text.contains("\"ownership\""));
        assert!(text.contains("JSON array of 3 strings"));
        assert!(!text.contains("Topic description"));

        let text = prompts.render(
            QUESTIONS,
            context! {
                topic => "ownership", description => "Rust ownership", subtopic => "borrowing",
                count => 2, accepted => vec!["What is a borrow?"],
            },
        )?;
        assert!(text.contains("- What is a borrow?"));
        assert!(text.contains("Topic description: Rust ownership"));

        let text = prompts.render(
            ANSWER_SYSTEM,
            context! { topic => "ownership", description => None::<String> },
        )?;
        assert!(text.starts_with("You are an expert in ownership."));
        Ok(())
    }

    #[test]
    fn empty_prompts_render_nothing() {
        let rendered = Prompts::empty().render(SUBTOPICS, context! { topic => "t" });
        assert!(matches!(rendered, Err(PromptError::Render { .. })));
    }

    #[test]
    fn overrides_replace_defaults_without_escaping() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        std::fs::create_dir(dir.path().join(DIR))?;
        std::fs::write(
            dir.path().join(DIR).join(ANSWER_SYSTEM),
            "Expert in <{{ topic }}> & more",
        )?;
        let prompts = Prompts::load(dir.path())?;
        let text = prompts.render(ANSWER_SYSTEM, context! { topic => "a&b" })?;
        assert_eq!(text, "Expert in <a&b> & more");
        Ok(())
    }

    #[test]
    fn undefined_variables_are_errors() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let prompts = Prompts::load(dir.path())?;
        let result = prompts.render(ANSWER_SYSTEM, context! {});
        assert!(matches!(result, Err(PromptError::Render { .. })));
        Ok(())
    }

    #[test]
    fn a_broken_override_names_the_template() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        std::fs::create_dir(dir.path().join(DIR))?;
        std::fs::write(dir.path().join(DIR).join(QUESTIONS), "{% for %}")?;
        match Prompts::load(dir.path()) {
            Err(PromptError::Template { name, .. }) => {
                assert_eq!(name, QUESTIONS);
                Ok(())
            },
            other => Err(format!("expected Template, got {other:?}").into()),
        }
    }
}
