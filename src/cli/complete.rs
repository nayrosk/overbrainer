//! Dynamic shell completion: candidates read from the project named on the command
//! line being completed.
//!
//! The shell runs `overbrainer` with `COMPLETE=<shell>`, then `--` and the command
//! line. Nothing here prints or logs, since the candidates go to stdout: a project
//! that cannot be read yields no candidates.

use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

use clap_complete::engine::CompletionCandidate;
use config::{Config, File, FileFormat};
use serde::Deserialize;
use serde_json::Value;

use crate::config::CONFIG_FILE;
use crate::runs::Runs;

/// The project directory of the command line being completed: the value of the
/// last `-C` or `--project-dir` after the `--` that follows the completer, or `.`.
///
/// `home` is `HOME`, used to expand a leading `~`; the caller reads it from the
/// environment so this function stays pure.
pub(crate) fn project_dir_from(args: &[OsString], home: Option<&OsStr>) -> PathBuf {
    // Skip the completer's own arguments, the `--`, then the program name.
    let mut words = args.iter().skip_while(|word| *word != "--").skip(2);
    let mut dir = PathBuf::from(".");
    while let Some(word) = words.next() {
        let Some(text) = word.to_str() else {
            continue;
        };
        if text == "-C" || text == "--project-dir" {
            if let Some(value) = words.next() {
                dir = match value.to_str() {
                    Some(text) => resolve(text, home),
                    None => PathBuf::from(value),
                };
            }
        } else if let Some(value) = text.strip_prefix("--project-dir=") {
            dir = resolve(value, home);
        } else if let Some(value) = text.strip_prefix("-C") {
            dir = resolve(value.strip_prefix('=').unwrap_or(value), home);
        }
    }
    dir
}

/// A completed `-C` value, unquoted and with a leading `~` expanded: shells pass
/// the word as typed, so `~/proj` and `'a b'` reach us unexpanded and still quoted.
fn resolve(text: &str, home: Option<&OsStr>) -> PathBuf {
    expand_tilde(strip_quotes(text), home)
}

/// Strips one layer of matching surrounding `'...'` or `"..."`, if present.
fn strip_quotes(text: &str) -> &str {
    let bytes = text.as_bytes();
    let quoted = bytes.len() >= 2
        && ((bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\'')
            || (bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"'));
    if quoted {
        &text[1..text.len() - 1]
    } else {
        text
    }
}

/// Expands a leading `~` or `~/` using `home`. Without a `home`, or for a named
/// user's home such as `~user/p`, the text is left as is.
fn expand_tilde(text: &str, home: Option<&OsStr>) -> PathBuf {
    let Some(home) = home else {
        return PathBuf::from(text);
    };
    if text == "~" {
        return PathBuf::from(home);
    }
    match text.strip_prefix("~/") {
        Some(rest) => Path::new(home).join(rest),
        None => PathBuf::from(text),
    }
}

/// The project directory of the command line being completed.
fn project_dir() -> PathBuf {
    let args: Vec<OsString> = std::env::args_os().collect();
    project_dir_from(&args, std::env::var_os("HOME").as_deref())
}

/// Run IDs in the project's `runs/`, with their recorded state and target.
pub(crate) fn run_ids() -> Vec<CompletionCandidate> {
    run_ids_in(&project_dir())
}

fn run_ids_in(dir: &Path) -> Vec<CompletionCandidate> {
    Runs::new(dir)
        .list_with(|_, _| {})
        .unwrap_or_default()
        .into_iter()
        .map(|record| {
            let help = format!("{}, {}", record.state.name(), record.target);
            CompletionCandidate::new(record.id).help(Some(help.into()))
        })
        .collect()
}

/// The names completion needs from `overbrainer.toml`, read without validation:
/// `config::load` fails when the provider keys are in `.env`, which completion
/// does not load.
///
/// `targets` values are ignored, but `IgnoredAny` is zero-sized and clippy denies a
/// map with a zero-sized value type; `Value` is the same escape hatch
/// `config::load` already uses for `training.axolotl_extra`.
#[derive(Deserialize)]
struct Names {
    #[serde(default)]
    topics: Vec<TopicName>,
    #[serde(default)]
    targets: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct TopicName {
    name: String,
}

fn names(dir: &Path) -> Option<Names> {
    let content = std::fs::read_to_string(dir.join(CONFIG_FILE)).ok()?;
    Config::builder()
        .add_source(File::from_str(&content, FileFormat::Toml))
        .build()
        .ok()?
        .try_deserialize()
        .ok()
}

/// Topic names in the project's `overbrainer.toml`.
pub(crate) fn topics() -> Vec<CompletionCandidate> {
    names(&project_dir())
        .map(|names| {
            names
                .topics
                .into_iter()
                .map(|t| CompletionCandidate::new(t.name))
                .collect()
        })
        .unwrap_or_default()
}

/// Target names in the project's `overbrainer.toml`.
pub(crate) fn targets() -> Vec<CompletionCandidate> {
    names(&project_dir())
        .map(|names| {
            names
                .targets
                .into_keys()
                .map(CompletionCandidate::new)
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir_of(line: &[&str]) -> PathBuf {
        dir_of_with_home(line, None)
    }

    fn dir_of_with_home(line: &[&str], home: Option<&str>) -> PathBuf {
        let args: Vec<OsString> = line.iter().map(OsString::from).collect();
        project_dir_from(&args, home.map(OsStr::new))
    }

    #[test]
    fn reads_names_from_an_invalid_but_parsable_config() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join(crate::config::CONFIG_FILE),
            "[[topics]]\nname = \"ownership\"\n[[topics]]\nname = \"traits\"\nunknown = 1\n\
             [targets.local]\nkind = \"local\"\n[targets.gpu_cloud]\nkind = \"runpod\"\n",
        )?;
        let names = names(dir.path()).ok_or("no names")?;
        let topics: Vec<&str> = names.topics.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(topics, ["ownership", "traits"]);
        assert_eq!(
            names.targets.keys().collect::<Vec<_>>(),
            ["gpu_cloud", "local"]
        );
        Ok(())
    }

    #[test]
    fn has_no_names_without_a_readable_config() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        assert!(names(dir.path()).is_none());
        std::fs::write(dir.path().join(crate::config::CONFIG_FILE), "[[topics]\n")?;
        assert!(names(dir.path()).is_none());
        Ok(())
    }

    #[test]
    fn defaults_to_the_current_directory() {
        assert_eq!(
            dir_of(&["ob", "--", "overbrainer", "train", "attach", ""]),
            Path::new(".")
        );
    }

    #[test]
    fn reads_every_form_of_the_flag() {
        for line in [
            ["ob", "--", "overbrainer", "-C", "p", "runs"].as_slice(),
            &["ob", "--", "overbrainer", "-Cp", "runs"],
            &["ob", "--", "overbrainer", "-C=p", "runs"],
            &["ob", "--", "overbrainer", "--project-dir", "p", "runs"],
            &["ob", "--", "overbrainer", "--project-dir=p", "runs"],
            &["ob", "--", "overbrainer", "runs", "-C", "p"],
        ] {
            assert_eq!(dir_of(line), Path::new("p"), "{line:?}");
        }
    }

    #[test]
    fn the_last_flag_wins() {
        assert_eq!(
            dir_of(&["ob", "--", "overbrainer", "-C", "a", "-C", "b", "runs"]),
            Path::new("b")
        );
    }

    #[test]
    fn ignores_words_before_the_separator() {
        assert_eq!(
            dir_of(&["ob", "-C", "x", "--", "overbrainer", "runs"]),
            Path::new(".")
        );
        assert_eq!(dir_of(&["ob", "-C", "x"]), Path::new("."));
    }

    #[test]
    fn a_flag_being_completed_has_no_value_yet() {
        assert_eq!(dir_of(&["ob", "--", "overbrainer", "-C"]), Path::new("."));
    }

    #[test]
    fn ignores_a_separator_inside_the_command_line() {
        assert_eq!(
            dir_of(&["ob", "--", "overbrainer", "-C", "p", "train", "--", "x"]),
            Path::new("p")
        );
    }

    #[test]
    fn expands_a_leading_tilde_from_home() {
        assert_eq!(
            dir_of_with_home(
                &["ob", "--", "overbrainer", "-C", "~", "runs"],
                Some("/home/nay")
            ),
            Path::new("/home/nay")
        );
        assert_eq!(
            dir_of_with_home(
                &["ob", "--", "overbrainer", "-C", "~/p", "runs"],
                Some("/home/nay")
            ),
            Path::new("/home/nay/p")
        );
    }

    #[test]
    fn leaves_a_named_users_home_alone() {
        assert_eq!(
            dir_of_with_home(
                &["ob", "--", "overbrainer", "-C", "~user/p", "runs"],
                Some("/home/nay")
            ),
            Path::new("~user/p")
        );
    }

    #[test]
    fn leaves_a_tilde_as_is_without_home() {
        assert_eq!(
            dir_of(&["ob", "--", "overbrainer", "-C", "~", "runs"]),
            Path::new("~")
        );
        assert_eq!(
            dir_of(&["ob", "--", "overbrainer", "-C", "~/p", "runs"]),
            Path::new("~/p")
        );
    }

    #[test]
    fn strips_one_layer_of_matching_quotes() {
        assert_eq!(
            dir_of(&["ob", "--", "overbrainer", "-C", "'p q'", "runs"]),
            Path::new("p q")
        );
        assert_eq!(
            dir_of(&["ob", "--", "overbrainer", "-C", "\"p q\"", "runs"]),
            Path::new("p q")
        );
    }
}
