//! The `overbrainer migrate` subcommand: brings a project from an older format
//! to [`project_format::CURRENT`].

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context as _, bail};

use super::init::{GITIGNORE, add_gitignore_entries};
use crate::config::CONFIG_FILE;
use crate::config::edit::{ConfigDoc, FieldPath};
use crate::config::fields::FieldValue;
use crate::dataset::{DataFiles, Example};
use crate::events::Stage;
use crate::history::{self, Entry, Status};
use crate::project_format;
use crate::runs::rfc3339;

/// What every command on a project from before 0.4.0 says once.
pub const HINT: &str = "this project predates overbrainer 0.4.0: run overbrainer migrate";

/// The `.gitignore` entry of the state directory.
const STATE_ENTRY: &str = "/.overbrainer/";

/// The `.gitignore` lines that already ignore the state directory.
const STATE_ENTRIES: [&str; 4] = [
    ".overbrainer",
    ".overbrainer/",
    "/.overbrainer",
    STATE_ENTRY,
];

/// One change a migration makes.
#[derive(Debug, Clone, PartialEq)]
enum Change {
    /// Add [`STATE_ENTRY`] to `.gitignore`.
    Gitignore,
    /// Append this `answers` entry, rebuilt from `data/answers.jsonl`.
    Backfill(Entry),
    /// Write `.overbrainer/version`.
    Version,
    /// Move the deprecated `training.hub_model_id` to `[hub]`. A `[hub] repo`
    /// already set stays, and the old key is only dropped.
    HubModelId {
        value: String,
        hub_repo: Option<String>,
    },
}

impl Change {
    /// The line that tells this change: what was done, or with `dry_run`
    /// what would be.
    fn line(&self, dry_run: bool) -> String {
        let (done, planned, what) = match self {
            Self::HubModelId {
                value,
                hub_repo: None,
            } => (
                "moved",
                "move",
                format!(
                    "training.hub_model_id to [hub] repo = \"{value}\" (private, after_training)"
                ),
            ),
            Self::HubModelId { .. } => ("removed", "remove", "training.hub_model_id".to_string()),
            Self::Gitignore => ("added", "add", format!("{STATE_ENTRY} to {GITIGNORE}")),
            Self::Backfill(entry) => (
                "backfilled",
                "backfill",
                format!(
                    "the answers history of {}: {} answer(s), {} excluded; tokens {} in, {} out",
                    entry.model.as_deref().unwrap_or("-"),
                    entry.done,
                    entry.excluded,
                    entry.input_tokens,
                    entry.output_tokens
                ),
            ),
            Self::Version => (
                "wrote",
                "write",
                format!(
                    ".overbrainer/{}: format {}",
                    project_format::VERSION_FILE,
                    project_format::CURRENT
                ),
            ),
        };
        if dry_run {
            format!("would {planned} {what}")
        } else {
            format!("{done} {what}")
        }
    }
}

/// What a migration of a project would do.
#[derive(Debug, Default)]
struct Plan {
    changes: Vec<Change>,
    /// Steps left out, and why.
    notes: Vec<String>,
}

/// Migrates the project in `project_dir`, or with `dry_run` only says what
/// it would do, and prints one line per change, or `nothing to migrate`.
///
/// # Errors
///
/// Returns an error when the project uses a newer format than this
/// overbrainer knows, or when a file cannot be read or written.
pub fn run(project_dir: &Path, dry_run: bool) -> anyhow::Result<()> {
    for line in migrate(project_dir, dry_run)? {
        println!("{line}");
    }
    Ok(())
}

/// The lines [`run`] prints, after migrating unless `dry_run`.
fn migrate(project_dir: &Path, dry_run: bool) -> anyhow::Result<Vec<String>> {
    let plan = plan(project_dir)?;
    let mut lines = plan.notes.clone();
    if plan.changes.is_empty() {
        if lines.is_empty() {
            lines.push("nothing to migrate".to_string());
        }
        return Ok(lines);
    }
    if !dry_run {
        apply(project_dir, &plan.changes)?;
    }
    lines.extend(plan.changes.iter().map(|change| change.line(dry_run)));
    Ok(lines)
}

/// What migrating `project_dir` takes. The version is written last, so an
/// interrupted migration runs again.
fn plan(project_dir: &Path) -> anyhow::Result<Plan> {
    let version = project_format::read(project_dir).context("cannot read the project format")?;
    if let Some(version) = version
        && version > project_format::CURRENT
    {
        bail!(
            "this project uses format {version}, newer than the format {} this overbrainer \
             knows: upgrade overbrainer",
            project_format::CURRENT
        );
    }
    let mut plan = Plan::default();
    if let Some(change) = hub_model_id(project_dir)? {
        if let Change::HubModelId {
            value,
            hub_repo: Some(hub_repo),
        } = &change
        {
            plan.notes.push(format!(
                "kept [hub] repo = \"{hub_repo}\"; dropped training.hub_model_id = \"{value}\""
            ));
        }
        plan.changes.push(change);
    }
    if !ignores_state(&project_dir.join(GITIGNORE))? {
        plan.changes.push(Change::Gitignore);
    }
    if version.is_none() {
        let entries = history::read(project_dir)
            .with_context(|| format!("cannot read {}", history::path(project_dir).display()))?;
        let answers = entries.iter().filter(|entry| entry.stage == Stage::Answers);
        if answers.clone().any(|entry| !entry.backfilled) {
            plan.notes.push(format!(
                "not backfilling the answers history: {} already has answers entries",
                history::path(project_dir).display()
            ));
        } else {
            // A migration that stopped part way left some models backfilled:
            // only the others are.
            let done: Vec<&str> = answers.filter_map(|entry| entry.model.as_deref()).collect();
            plan.changes.extend(
                backfill(project_dir)?
                    .into_iter()
                    .filter(|entry| !done.contains(&entry.model.as_deref().unwrap_or_default()))
                    .map(Change::Backfill),
            );
        }
        plan.changes.push(Change::Version);
    }
    Ok(plan)
}

/// Makes `changes`, in order. The backfill lines go in one append, so a crash
/// leaves all of them or none.
fn apply(project_dir: &Path, changes: &[Change]) -> anyhow::Result<()> {
    if changes.contains(&Change::Gitignore) {
        add_gitignore_entries(&project_dir.join(GITIGNORE), &format!("{STATE_ENTRY}\n"))?;
    }
    let backfill: Vec<Entry> = changes
        .iter()
        .filter_map(|change| match change {
            Change::Backfill(entry) => Some(entry.clone()),
            Change::Gitignore | Change::Version | Change::HubModelId { .. } => None,
        })
        .collect();
    if !backfill.is_empty() {
        history::append_all(project_dir, &backfill)
            .with_context(|| format!("cannot write {}", history::path(project_dir).display()))?;
    }
    if changes
        .iter()
        .any(|change| matches!(change, Change::HubModelId { .. }))
    {
        move_hub_model_id(project_dir)?;
    }
    if changes.contains(&Change::Version) {
        project_format::write_current(project_dir).context("cannot write the project format")?;
    }
    Ok(())
}

/// The deprecated `training.hub_model_id` of `overbrainer.toml`, when set. A
/// missing or unparsable file has none: `config check` reports those.
fn hub_model_id(project_dir: &Path) -> anyhow::Result<Option<Change>> {
    let Some(doc) = read_config(project_dir)? else {
        return Ok(None);
    };
    Ok(doc
        .get(&FieldPath::Training("hub_model_id"))
        .map(|value| Change::HubModelId {
            value,
            hub_repo: doc.get(&FieldPath::Hub("repo")),
        }))
}

fn read_config(project_dir: &Path) -> anyhow::Result<Option<ConfigDoc>> {
    let path = project_dir.join(CONFIG_FILE);
    match std::fs::read_to_string(&path) {
        Ok(text) => Ok(ConfigDoc::parse(&text).ok()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("cannot read {}", path.display())),
    }
}

/// Moves `training.hub_model_id` to `[hub]`, keeping the comments, and writes
/// `overbrainer.toml` atomically. `[hub]` keys that are already set stay.
fn move_hub_model_id(project_dir: &Path) -> anyhow::Result<()> {
    let Some(mut doc) = read_config(project_dir)? else {
        return Ok(());
    };
    let Some(value) = doc.get(&FieldPath::Training("hub_model_id")) else {
        return Ok(());
    };
    let edit = |error: crate::config::edit::EditError| anyhow::anyhow!("{error}");
    doc.unset(&FieldPath::Training("hub_model_id"))
        .map_err(edit)?;
    // With a `[hub] repo` already there, that table is the user's: leave it be.
    if doc.get(&FieldPath::Hub("repo")).is_none() {
        doc.set(&FieldPath::Hub("repo"), FieldValue::Text(value))
            .map_err(edit)?;
        for key in ["private", "after_training"] {
            let path = FieldPath::Hub(key);
            if doc.get(&path).is_none() {
                doc.set(&path, FieldValue::Bool(true)).map_err(edit)?;
            }
        }
    }
    crate::runs::write_atomic(project_dir, CONFIG_FILE, doc.text().as_bytes())
        .with_context(|| format!("cannot write {CONFIG_FILE}"))
}

/// Whether `.gitignore` at `path` has a line in [`STATE_ENTRIES`]; not when
/// it does not exist.
fn ignores_state(path: &Path) -> anyhow::Result<bool> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(text
            .lines()
            .any(|line| STATE_ENTRIES.contains(&line.trim()))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e).with_context(|| format!("cannot read {}", path.display())),
    }
}

/// One `answers` entry per model of `data/answers.jsonl`, dated by the file's
/// modification time: answers keep no time of their own, nor a provider.
fn backfill(project_dir: &Path) -> anyhow::Result<Vec<Entry>> {
    let path = DataFiles::new(project_dir).answers;
    let answers: Vec<Example> = crate::dataset::read(&path)?;
    if answers.is_empty() {
        return Ok(Vec::new());
    }
    let modified = std::fs::metadata(&path)
        .and_then(|metadata| metadata.modified())
        .with_context(|| format!("cannot read the modification time of {}", path.display()))?;
    let at = rfc3339(modified);
    let mut models: BTreeMap<&str, Entry> = BTreeMap::new();
    for answer in &answers {
        let meta = &answer.meta;
        let entry = models
            .entry(meta.model.as_str())
            .or_insert_with(|| empty_entry(&meta.model, &at));
        // As a recorded run counts them: kept for training, or excluded.
        if meta.excluded.is_some() {
            entry.excluded += 1;
        } else {
            entry.done += 1;
        }
        entry.input_tokens += meta.input_tokens;
        entry.output_tokens += meta.output_tokens;
    }
    Ok(models.into_values().collect())
}

fn empty_entry(model: &str, at: &str) -> Entry {
    Entry {
        stage: Stage::Answers,
        started_at: at.to_string(),
        ended_at: at.to_string(),
        status: Status::Ok,
        provider: None,
        model: Some(model.to_string()),
        done: 0,
        skipped: 0,
        failed: 0,
        excluded: 0,
        input_tokens: 0,
        output_tokens: 0,
        cost: None,
        split: None,
        backfilled: true,
    }
}

/// Whether `command` in `project_dir` should end with [`HINT`]: a project
/// from before 0.4.0, and a command that works on it and that the hint does
/// not concern (`init`, `migrate`, `skill`), outside the TUI, which shows it
/// itself.
pub(super) fn hints(command: &super::Command, project_dir: &Path) -> bool {
    use super::Command;
    !matches!(
        command,
        Command::Init { .. } | Command::Migrate(_) | Command::Skill { .. } | Command::Tui
    ) && project_format::predates_versions(project_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Cli;
    use crate::dataset::{Exclusion, FinishReason, Id, Meta, ReasoningKind};
    use crate::project_lock::STATE_DIR;
    use clap::Parser as _;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn answer(question: &str, model: &str, tokens: (u64, u64), excluded: bool) -> Example {
        Example {
            id: Id::of(&[question]),
            topic: "t".into(),
            subtopic: "s".into(),
            messages: Vec::new(),
            meta: Meta {
                model: model.into(),
                input_tokens: tokens.0,
                output_tokens: tokens.1,
                finish_reason: FinishReason::Stop,
                reasoning_kind: ReasoningKind::Raw,
                excluded: excluded.then_some(Exclusion::Truncated),
            },
        }
    }

    /// A project as 0.3 left it: no version, `.gitignore` without the state
    /// directory, answers by two models.
    fn old_project() -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join(crate::config::CONFIG_FILE), "")?;
        std::fs::write(dir.path().join(GITIGNORE), ".env\n/data/\n/runs/")?;
        std::fs::create_dir(dir.path().join("data"))?;
        crate::dataset::rewrite(
            &DataFiles::new(dir.path()).answers,
            &[
                answer("a", "parent", (10, 100), false),
                answer("b", "other", (1, 2), false),
                answer("c", "parent", (20, 200), true),
            ],
        )?;
        Ok(dir)
    }

    #[test]
    fn an_old_project_is_migrated_once() -> TestResult {
        let dir = old_project()?;
        assert_eq!(
            migrate(dir.path(), false)?,
            [
                "added /.overbrainer/ to .gitignore",
                "backfilled the answers history of other: 1 answer(s), 0 excluded; tokens 1 in, 2 out",
                "backfilled the answers history of parent: 1 answer(s), 1 excluded; tokens 30 in, 300 out",
                "wrote .overbrainer/version: format 1",
            ]
        );
        assert_eq!(project_format::read(dir.path())?, Some(1));
        assert_eq!(
            std::fs::read_to_string(dir.path().join(GITIGNORE))?,
            ".env\n/data/\n/runs/\n/.overbrainer/\n"
        );
        let entries = history::read(dir.path())?;
        let [other, parent] = entries.as_slice() else {
            return Err(format!("expected two entries, got {entries:?}").into());
        };
        assert_eq!(other.model.as_deref(), Some("other"));
        assert_eq!(
            (parent.stage, parent.status, parent.done, parent.excluded),
            (Stage::Answers, Status::Ok, 1, 1)
        );
        assert_eq!((parent.input_tokens, parent.output_tokens), (30, 300));
        assert_eq!((parent.cost, parent.provider.as_deref()), (None, None));
        assert!(parent.backfilled && other.backfilled);
        assert_eq!(parent.started_at, parent.ended_at);

        assert_eq!(migrate(dir.path(), false)?, ["nothing to migrate"]);
        assert_eq!(history::read(dir.path())?.len(), 2);
        Ok(())
    }

    #[test]
    fn a_dry_run_says_the_same_and_writes_nothing() -> TestResult {
        let dir = old_project()?;
        let gitignore = std::fs::read_to_string(dir.path().join(GITIGNORE))?;
        assert_eq!(
            migrate(dir.path(), true)?,
            [
                "would add /.overbrainer/ to .gitignore",
                "would backfill the answers history of other: 1 answer(s), 0 excluded; tokens 1 in, 2 out",
                "would backfill the answers history of parent: 1 answer(s), 1 excluded; tokens 30 in, 300 out",
                "would write .overbrainer/version: format 1",
            ]
        );
        assert!(!dir.path().join(STATE_DIR).exists());
        assert_eq!(
            std::fs::read_to_string(dir.path().join(GITIGNORE))?,
            gitignore
        );
        Ok(())
    }

    #[test]
    fn an_answers_history_is_never_backfilled_twice() -> TestResult {
        let dir = old_project()?;
        let recorded = Entry {
            backfilled: false,
            ..empty_entry("parent", "2026-09-27T10:00:00Z")
        };
        history::append(dir.path(), &recorded)?;
        let lines = migrate(dir.path(), false)?;
        assert_eq!(
            lines.first().map(String::as_str),
            Some(
                format!(
                    "not backfilling the answers history: {} already has answers entries",
                    history::path(dir.path()).display()
                )
                .as_str()
            )
        );
        assert_eq!(lines.len(), 3, "{lines:?}");
        assert_eq!(history::read(dir.path())?, [recorded]);
        assert_eq!(project_format::read(dir.path())?, Some(1));
        Ok(())
    }

    #[test]
    fn a_stopped_backfill_is_finished_with_the_missing_models_only() -> TestResult {
        let dir = old_project()?;
        let at = "2026-09-27T10:00:00Z";
        history::append(dir.path(), &empty_entry("other", at))?;
        // The line of the next model was cut by the crash.
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(history::path(dir.path()))?;
        std::io::Write::write_all(&mut file, b"{\"stage\": \"answ")?;
        drop(file);
        assert_eq!(
            migrate(dir.path(), false)?,
            [
                "added /.overbrainer/ to .gitignore",
                "backfilled the answers history of parent: 1 answer(s), 1 excluded; tokens 30 in, 300 out",
                "wrote .overbrainer/version: format 1",
            ]
        );
        let models: Vec<_> = history::read(dir.path())?
            .into_iter()
            .map(|entry| (entry.model, entry.backfilled))
            .collect();
        assert_eq!(
            models,
            [(Some("other".into()), true), (Some("parent".into()), true)]
        );
        assert_eq!(migrate(dir.path(), false)?, ["nothing to migrate"]);
        Ok(())
    }

    #[test]
    fn a_project_without_answers_gets_no_backfill() -> TestResult {
        let dir = tempfile::tempdir()?;
        assert_eq!(
            migrate(dir.path(), false)?,
            [
                "added /.overbrainer/ to .gitignore",
                "wrote .overbrainer/version: format 1"
            ]
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join(GITIGNORE))?,
            "/.overbrainer/\n"
        );
        assert_eq!(history::read(dir.path())?, [] as [crate::history::Entry; 0]);
        Ok(())
    }

    #[test]
    fn any_line_ignoring_the_state_directory_is_kept() -> TestResult {
        for line in [
            ".overbrainer",
            ".overbrainer/",
            "/.overbrainer",
            " /.overbrainer/ ",
        ] {
            let dir = tempfile::tempdir()?;
            let text = format!(".env\n{line}\n");
            std::fs::write(dir.path().join(GITIGNORE), &text)?;
            assert_eq!(
                migrate(dir.path(), false)?,
                ["wrote .overbrainer/version: format 1"],
                "{line:?}"
            );
            assert_eq!(std::fs::read_to_string(dir.path().join(GITIGNORE))?, text);
        }
        Ok(())
    }

    #[test]
    fn a_newer_format_is_refused() -> TestResult {
        let dir = old_project()?;
        std::fs::create_dir(dir.path().join(STATE_DIR))?;
        std::fs::write(
            dir.path()
                .join(STATE_DIR)
                .join(project_format::VERSION_FILE),
            "2\n",
        )?;
        match migrate(dir.path(), false) {
            Err(error) => assert_eq!(
                error.to_string(),
                "this project uses format 2, newer than the format 1 this overbrainer knows: \
                 upgrade overbrainer"
            ),
            Ok(lines) => return Err(format!("expected a refusal, got {lines:?}").into()),
        }
        assert_eq!(history::read(dir.path())?, [] as [crate::history::Entry; 0]);
        Ok(())
    }

    /// A project already at the current format, with this `[training]` key and
    /// the commented rest of a config.
    fn project_with_toml(toml: &str) -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join(crate::config::CONFIG_FILE), toml)?;
        std::fs::write(dir.path().join(GITIGNORE), format!("{STATE_ENTRY}\n"))?;
        project_format::write_current(dir.path())?;
        Ok(dir)
    }

    /// A valid commented config whose `[training]` table also holds `extra`.
    fn with_training_key(extra: &str) -> String {
        format!(
            "# my project\n[project]\nname = \"demo\"\n\n[[topics]]\nname = \"ownership\"\n\
             subtopics = 3\nquestions_per_subtopic = 5\n\n[providers.nanogpt]\nprotocol = \"openai\"\n\n\
             [roles]\ngenerator = {{ provider = \"nanogpt\", model = \"m1\" }}\n\
             parent = {{ provider = \"nanogpt\", model = \"m2\" }}\n\n\
             [training]\n# the base\ntarget = \"local\"\nbase_model = \"Qwen/Qwen3-4B\"\n\
             adapter = \"qlora\"\n{extra}\n\n[targets.local]\nkind = \"local\"\nruntime = \"native\"\n"
        )
    }

    fn read_toml(dir: &Path) -> Result<String, Box<dyn std::error::Error>> {
        Ok(std::fs::read_to_string(
            dir.join(crate::config::CONFIG_FILE),
        )?)
    }

    fn load_text(text: &str) -> Result<crate::config::Settings, Box<dyn std::error::Error>> {
        Ok(crate::config::load_str(
            text,
            crate::config::EnvSource::Vars(Vec::new()),
        )?)
    }

    #[test]
    fn hub_model_id_moves_to_the_hub_section() -> TestResult {
        let dir = project_with_toml(&with_training_key("hub_model_id = \"me/mentor\""))?;
        assert_eq!(
            migrate(dir.path(), false)?,
            ["moved training.hub_model_id to [hub] repo = \"me/mentor\" (private, after_training)"]
        );
        let text = read_toml(dir.path())?;
        assert!(!text.contains("hub_model_id"), "{text}");
        assert!(text.contains("# my project") && text.contains("# the base"));
        let settings = load_text(&text)?;
        assert_eq!(settings.hub.repo.as_deref(), Some("me/mentor"));
        assert!(settings.hub.private && settings.hub.after_training);
        assert_eq!(migrate(dir.path(), false)?, ["nothing to migrate"]);
        Ok(())
    }

    #[test]
    fn dry_run_lists_the_hub_move_without_writing() -> TestResult {
        let dir = project_with_toml(&with_training_key("hub_model_id = \"me/mentor\""))?;
        let before = read_toml(dir.path())?;
        assert_eq!(
            migrate(dir.path(), true)?,
            [
                "would move training.hub_model_id to [hub] repo = \"me/mentor\" (private, after_training)"
            ]
        );
        assert_eq!(read_toml(dir.path())?, before);
        Ok(())
    }

    #[test]
    fn an_existing_hub_repo_is_not_overwritten() -> TestResult {
        let toml = format!(
            "{}\n[hub]\nrepo = \"me/kept\"\nprivate = false\n",
            with_training_key("hub_model_id = \"me/old\"")
        );
        let dir = project_with_toml(&toml)?;
        assert_eq!(
            migrate(dir.path(), false)?,
            [
                "kept [hub] repo = \"me/kept\"; dropped training.hub_model_id = \"me/old\"",
                "removed training.hub_model_id"
            ]
        );
        let text = read_toml(dir.path())?;
        assert!(!text.contains("hub_model_id"), "{text}");
        let settings = load_text(&text)?;
        assert_eq!(settings.hub.repo.as_deref(), Some("me/kept"));
        assert!(!settings.hub.private && !settings.hub.after_training);
        Ok(())
    }

    #[test]
    fn a_hub_table_that_sets_the_flags_keeps_them() -> TestResult {
        let toml = format!(
            "{}\n[hub]\nprivate = false\nafter_training = false\n",
            with_training_key("hub_model_id = \"me/mentor\"")
        );
        let dir = project_with_toml(&toml)?;
        migrate(dir.path(), false)?;
        let settings = load_text(&read_toml(dir.path())?)?;
        assert_eq!(settings.hub.repo.as_deref(), Some("me/mentor"));
        assert!(!settings.hub.private && !settings.hub.after_training);
        Ok(())
    }

    #[test]
    fn a_missing_or_unparsable_config_has_nothing_to_move() -> TestResult {
        let dir = project_with_toml("")?;
        std::fs::remove_file(dir.path().join(crate::config::CONFIG_FILE))?;
        assert_eq!(migrate(dir.path(), false)?, ["nothing to migrate"]);
        assert!(!dir.path().join(crate::config::CONFIG_FILE).exists());
        let broken = "[training\nhub_model_id = \"me/x\"\n";
        let dir = project_with_toml(broken)?;
        assert_eq!(migrate(dir.path(), false)?, ["nothing to migrate"]);
        assert_eq!(read_toml(dir.path())?, broken);
        Ok(())
    }

    #[test]
    fn a_config_without_the_key_is_left_alone() -> TestResult {
        let toml = with_training_key("epochs = 2");
        let dir = project_with_toml(&toml)?;
        assert_eq!(migrate(dir.path(), false)?, ["nothing to migrate"]);
        assert_eq!(read_toml(dir.path())?, toml);
        Ok(())
    }

    #[test]
    fn only_commands_on_an_old_project_hint() -> TestResult {
        let dir = old_project()?;
        let parse = |args: &[&str]| {
            Cli::try_parse_from(std::iter::once("overbrainer").chain(args.iter().copied()))
                .map(|cli| cli.command)
        };
        for args in [
            &["history"][..],
            &["run"],
            &["config", "check"],
            &["pod", "ls"],
        ] {
            assert!(hints(&parse(args)?, dir.path()), "{args:?}");
        }
        for args in [&["init"][..], &["migrate"], &["skill", "install"], &["tui"]] {
            assert!(!hints(&parse(args)?, dir.path()), "{args:?}");
        }
        project_format::write_current(dir.path())?;
        assert!(!hints(&parse(&["history"])?, dir.path()));
        let empty = tempfile::tempdir()?;
        assert!(!hints(&parse(&["history"])?, empty.path()));
        Ok(())
    }
}
