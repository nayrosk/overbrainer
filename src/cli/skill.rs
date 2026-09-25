//! The `overbrainer skill` subcommand: install the agent skill.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};

use super::{SkillCommand, SkillInstallArgs};

/// The agent skill, `skills/overbrainer/SKILL.md`, as shipped with this version.
pub(crate) const SKILL: &str = include_str!("../../skills/overbrainer/SKILL.md");

/// Directory of the skill inside a skills directory.
const SKILL_DIR: &str = "overbrainer";
/// File of the skill inside its directory.
const SKILL_FILE: &str = "SKILL.md";

/// Runs an `overbrainer skill` subcommand for the project in `project_dir`.
///
/// # Errors
///
/// Returns an error when the skills directory cannot be found or written, or when an
/// installed skill differs and `--force` was not given.
pub(crate) fn run(project_dir: &Path, command: &SkillCommand) -> anyhow::Result<()> {
    match command {
        SkillCommand::Install(args) => {
            let home = std::env::var_os("HOME").map(PathBuf::from);
            let base = skills_dir(project_dir, args, home)?;
            let (path, outcome) = install(&base, args.force)?;
            match outcome {
                Outcome::Installed => println!("installed {}", path.display()),
                Outcome::Replaced => println!("replaced {}", path.display()),
                Outcome::UpToDate => println!("{} is up to date", path.display()),
            }
            Ok(())
        },
    }
}

/// The skills directory `args` select.
fn skills_dir(
    project_dir: &Path,
    args: &SkillInstallArgs,
    home: Option<PathBuf>,
) -> anyhow::Result<PathBuf> {
    if let Some(dir) = &args.dir {
        return Ok(dir.clone());
    }
    let root = if args.global {
        home.context("--global needs HOME to find ~/.claude/skills")?
    } else {
        project_dir.to_path_buf()
    };
    Ok(root.join(".claude").join("skills"))
}

/// What [`install`] did.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Installed,
    Replaced,
    UpToDate,
}

/// Writes [`SKILL`] to `<base>/overbrainer/SKILL.md`, atomically. An identical file
/// is left alone; a different one is replaced only with `force`.
fn install(base: &Path, force: bool) -> anyhow::Result<(PathBuf, Outcome)> {
    let dir = base.join(SKILL_DIR);
    let path = dir.join(SKILL_FILE);
    let outcome = match std::fs::read(&path) {
        Ok(existing) if existing == SKILL.as_bytes() => return Ok((path, Outcome::UpToDate)),
        Ok(_) if !force => bail!(
            "{} differs from the skill of overbrainer {}; use --force to replace it",
            path.display(),
            env!("CARGO_PKG_VERSION")
        ),
        Ok(_) => Outcome::Replaced,
        Err(e) if e.kind() == ErrorKind::NotFound => Outcome::Installed,
        Err(e) => return Err(e).with_context(|| format!("cannot read {}", path.display())),
    };
    std::fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
    crate::runs::write_atomic(&dir, SKILL_FILE, SKILL.as_bytes())?;
    Ok((path, outcome))
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::*;
    use crate::cli::Cli;

    /// The frontmatter field `key`, on one line.
    fn field<'a>(key: &str) -> Option<&'a str> {
        let rest = SKILL.strip_prefix("---\n")?;
        let (frontmatter, _) = rest.split_once("\n---\n")?;
        frontmatter
            .lines()
            .find_map(|line| line.strip_prefix(key)?.strip_prefix(": "))
    }

    #[test]
    fn frontmatter_follows_the_agent_skills_specification() -> Result<(), String> {
        assert_eq!(field("name"), Some("overbrainer"));
        let description = field("description").ok_or("no description")?;
        assert!(!description.is_empty());
        assert!(description.chars().count() <= 1024);
        // A plain YAML scalar cannot hold ": ".
        assert!(!description.contains(": "));
        Ok(())
    }

    /// Every command line in `SKILL.md`: the words after `overbrainer` in a
    /// backticked span, or on a line that starts with `overbrainer`.
    fn command_lines() -> Vec<Vec<&'static str>> {
        let mut lines = Vec::new();
        for line in SKILL.lines() {
            if let Some(rest) = line.trim_start().strip_prefix("overbrainer ") {
                lines.push(words(rest));
            }
            for (_, rest) in line
                .match_indices("`overbrainer ")
                .map(|(i, m)| (i, &line[i + m.len()..]))
            {
                lines.push(words(rest.split('`').next().unwrap_or_default()));
            }
        }
        lines
    }

    fn words(text: &str) -> Vec<&str> {
        text.split('#')
            .next()
            .unwrap_or_default()
            .split_whitespace()
            .collect()
    }

    /// Checks that the subcommands and `--flags` in `words` exist.
    fn check(root: &clap::Command, words: &[&str]) -> Result<(), String> {
        let mut command = root;
        let mut words = words.iter();
        while let Some(&word) = words.next() {
            if let Some(flag) = word.strip_prefix("--") {
                let arg = command
                    .get_arguments()
                    .find(|arg| arg.get_long() == Some(flag))
                    .ok_or_else(|| format!("{} has no --{flag}", command.get_name()))?;
                if arg.get_action().takes_values() {
                    words.next();
                }
            } else if word == "-C" {
                words.next();
            } else if command.has_subcommands() && word.chars().all(|c| c.is_ascii_lowercase()) {
                command = command
                    .find_subcommand(word)
                    .ok_or_else(|| format!("{} has no subcommand {word}", command.get_name()))?;
            }
        }
        Ok(())
    }

    #[test]
    fn every_command_in_the_skill_exists() -> Result<(), String> {
        let mut root = Cli::command();
        root.build();
        let lines = command_lines();
        assert!(lines.len() > 20, "found only {} command lines", lines.len());
        for words in lines {
            check(&root, &words).map_err(|e| format!("`overbrainer {}`: {e}", words.join(" ")))?;
        }
        Ok(())
    }

    #[test]
    fn docs_links_point_at_this_version() {
        let tag = format!("blob/v{}/", env!("CARGO_PKG_VERSION"));
        let links: Vec<&str> = SKILL
            .match_indices("https://github.com/nayrosk/overbrainer/blob/")
            .map(|(i, _)| &SKILL[i..])
            .collect();
        assert!(!links.is_empty());
        for link in links {
            let link = &link[..link.find(')').unwrap_or(link.len())];
            assert!(link.contains(&tag), "{link}");
        }
    }

    fn args(global: bool, dir: Option<&str>) -> SkillInstallArgs {
        SkillInstallArgs {
            global,
            dir: dir.map(PathBuf::from),
            force: false,
        }
    }

    #[test]
    fn picks_the_skills_directory() -> anyhow::Result<()> {
        let home = Some(PathBuf::from("/home/u"));
        let project = Path::new("/p");
        assert_eq!(
            skills_dir(project, &args(false, None), home.clone())?,
            Path::new("/p/.claude/skills")
        );
        assert_eq!(
            skills_dir(project, &args(true, None), home.clone())?,
            Path::new("/home/u/.claude/skills")
        );
        assert_eq!(
            skills_dir(project, &args(false, Some("/x")), home)?,
            Path::new("/x")
        );
        assert!(skills_dir(project, &args(true, None), None).is_err());
        Ok(())
    }
}
