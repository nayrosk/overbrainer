//! The `overbrainer skill` subcommand: install the agent skill.

use std::io::{ErrorKind, Write as _};
use std::path::{Path, PathBuf};

use anyhow::{Context, anyhow, bail};

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
        home.filter(|home| !home.as_os_str().is_empty())
            .context("--global needs HOME to find ~/.claude/skills")?
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
    match outcome {
        // Found missing: never replace a skill another process wrote meanwhile.
        Outcome::Installed => create_atomic(&dir, SKILL_FILE, SKILL.as_bytes())?,
        _ => crate::runs::write_atomic(&dir, SKILL_FILE, SKILL.as_bytes())
            .with_context(|| format!("cannot write {}", path.display()))?,
    }
    Ok((path, outcome))
}

/// Creates `dir/name` with `content`, atomically and only if it does not exist yet:
/// the content goes to a temporary file first, then a hard link puts it in place,
/// which fails rather than replace a file created in the meantime.
fn create_atomic(dir: &Path, name: &str, content: &[u8]) -> anyhow::Result<()> {
    let tmp = dir.join(format!(".{name}.{:016x}.tmp", fastrand::u64(..)));
    let path = dir.join(name);
    let created = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .and_then(|mut file| file.write_all(content))
        .with_context(|| format!("cannot write {}", path.display()))
        .and_then(|()| {
            std::fs::hard_link(&tmp, &path).map_err(|e| {
                if e.kind() == ErrorKind::AlreadyExists {
                    anyhow!(
                        "{} appeared while installing; run the command again",
                        path.display()
                    )
                } else {
                    anyhow::Error::new(e).context(format!("cannot write {}", path.display()))
                }
            })
        });
    if let Err(error) = std::fs::remove_file(&tmp)
        && error.kind() != ErrorKind::NotFound
    {
        tracing::warn!("cannot remove {}: {error}", tmp.display());
    }
    created
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

    /// Every command line in `text`: the words after `overbrainer` in a
    /// backticked span, or on a line that starts with `overbrainer`.
    fn command_lines(text: &str) -> Vec<Vec<&str>> {
        let mut lines = Vec::new();
        for line in text.lines() {
            if let Some(rest) = line.trim_start().strip_prefix("overbrainer ") {
                lines.extend(chained(rest));
            }
            for (i, m) in line.match_indices("`overbrainer ") {
                let span = line[i + m.len()..].split('`').next().unwrap_or_default();
                lines.extend(chained(span));
            }
        }
        lines
    }

    /// The words of `text`, the rest of a command after `overbrainer`, and of every
    /// `&& overbrainer ...` chained after it.
    fn chained(text: &str) -> Vec<Vec<&str>> {
        let mut segments = text.split("&&");
        let first = segments.next().unwrap_or_default();
        std::iter::once(first)
            .chain(segments.filter_map(|segment| segment.trim_start().strip_prefix("overbrainer ")))
            .map(words)
            .collect()
    }

    fn words(text: &str) -> Vec<&str> {
        text.split('#')
            .next()
            .unwrap_or_default()
            .split_whitespace()
            .collect()
    }

    /// A placeholder such as `RUN_ID` or `NAME`.
    fn is_placeholder(word: &str) -> bool {
        word.chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
    }

    /// Checks that the subcommands and `--flags` in `words` exist, and that the
    /// command does not stop where a subcommand is required.
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
            } else if command.has_subcommands() {
                if is_placeholder(word) {
                    // Stands for any subcommand: nothing more to check.
                    return Ok(());
                }
                command = command
                    .find_subcommand(word)
                    .ok_or_else(|| format!("{} has no subcommand {word}", command.get_name()))?;
            }
        }
        if command.is_subcommand_required_set() {
            return Err(format!("{} needs a subcommand", command.get_name()));
        }
        Ok(())
    }

    #[test]
    fn every_command_in_the_skill_exists() -> Result<(), String> {
        let root = built_cli();
        let lines = command_lines(SKILL);
        assert!(lines.len() > 20, "found only {} command lines", lines.len());
        for words in lines {
            check(&root, &words).map_err(|e| format!("`overbrainer {}`: {e}", words.join(" ")))?;
        }
        Ok(())
    }

    fn built_cli() -> clap::Command {
        let mut root = Cli::command();
        root.build();
        root
    }

    #[test]
    fn check_accepts_real_commands() -> Result<(), String> {
        let root = built_cli();
        for line in [
            "train",
            "train --target NAME",
            "train attach RUN_ID",
            "runs ls",
            "pod rm RUN_ID --force",
            "config check --resolve",
            "-C DIR answers --topic NAME",
        ] {
            check(&root, &words(line)).map_err(|e| format!("`{line}`: {e}"))?;
        }
        Ok(())
    }

    #[test]
    fn check_rejects_wrong_commands() {
        let root = built_cli();
        for line in [
            "runs-ls",
            "runs",
            "pod",
            "config",
            "skill",
            "config chek",
            "pod rm RUN_ID --nope",
        ] {
            assert!(check(&root, &words(line)).is_err(), "`{line}` passed");
        }
    }

    #[test]
    fn command_lines_include_every_chained_command() {
        let text = "overbrainer init x && cd x && overbrainer runs ls\n\
                    Run `overbrainer pod ls && overbrainer config check`.";
        assert_eq!(
            command_lines(text),
            [
                vec!["init", "x"],
                vec!["runs", "ls"],
                vec!["pod", "ls"],
                vec!["config", "check"],
            ]
        );
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

    #[cfg(unix)]
    #[test]
    fn a_failed_write_names_the_skill_file() -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        // Root writes through a read-only directory: nothing to test then.
        let probe = tempfile::tempdir()?;
        std::fs::set_permissions(probe.path(), std::fs::Permissions::from_mode(0o500))?;
        let root = std::fs::write(probe.path().join("x"), b"x").is_ok();
        std::fs::set_permissions(probe.path(), std::fs::Permissions::from_mode(0o700))?;
        if root {
            eprintln!("skipped: running as root, permissions cannot force a save to fail");
            return Ok(());
        }

        let base = tempfile::tempdir()?;
        let dir = base.path().join(SKILL_DIR);
        std::fs::create_dir(&dir)?;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o555))?;
        let result = install(base.path(), false);
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755))?;
        let Err(error) = result else {
            anyhow::bail!("writing into a read-only directory succeeded");
        };
        assert_eq!(
            error.to_string(),
            format!("cannot write {}", dir.join(SKILL_FILE).display())
        );
        Ok(())
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
        // An empty HOME would put a global install in the current directory.
        assert!(skills_dir(project, &args(true, None), Some(PathBuf::new())).is_err());
        Ok(())
    }

    #[test]
    fn a_skill_created_meanwhile_is_not_replaced() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(dir.path().join(SKILL_FILE), "written by someone else\n")?;
        let error = create_atomic(dir.path(), SKILL_FILE, SKILL.as_bytes())
            .err()
            .ok_or_else(|| anyhow::anyhow!("an existing file was replaced"))?;
        assert!(
            format!("{error:#}").contains("appeared while installing"),
            "{error:#}"
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join(SKILL_FILE))?,
            "written by someone else\n"
        );
        let left: Vec<_> = std::fs::read_dir(dir.path())?.collect::<Result<_, _>>()?;
        assert_eq!(left.len(), 1, "temporary file left behind");
        Ok(())
    }
}
