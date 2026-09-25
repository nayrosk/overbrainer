//! Dynamic shell completion: candidates read from the project named on the command
//! line being completed.
//!
//! The shell runs `overbrainer` with `COMPLETE=<shell>`, then `--` and the command
//! line. Nothing here prints or logs, since the candidates go to stdout: a project
//! that cannot be read yields no candidates.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use clap_complete::engine::CompletionCandidate;

use crate::runs::Runs;

/// The project directory of the command line being completed: the value of the
/// last `-C` or `--project-dir` after the `--` that follows the completer, or `.`.
pub(crate) fn project_dir_from(args: &[OsString]) -> PathBuf {
    // Skip the completer's own arguments, the `--`, then the program name.
    let mut words = args.iter().skip_while(|word| *word != "--").skip(2);
    let mut dir = PathBuf::from(".");
    while let Some(word) = words.next() {
        let Some(text) = word.to_str() else {
            continue;
        };
        if text == "-C" || text == "--project-dir" {
            if let Some(value) = words.next() {
                dir = PathBuf::from(value);
            }
        } else if let Some(value) = text.strip_prefix("--project-dir=") {
            dir = PathBuf::from(value);
        } else if let Some(value) = text.strip_prefix("-C") {
            dir = PathBuf::from(value.strip_prefix('=').unwrap_or(value));
        }
    }
    dir
}

/// The project directory of the command line being completed.
fn project_dir() -> PathBuf {
    let args: Vec<OsString> = std::env::args_os().collect();
    project_dir_from(&args)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn dir_of(line: &[&str]) -> PathBuf {
        let args: Vec<OsString> = line.iter().map(OsString::from).collect();
        project_dir_from(&args)
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
}
