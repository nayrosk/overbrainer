//! Editing a question, an answer or a subtopic name in `$EDITOR`: the temp file it
//! edits, and what its edited text means.

use std::ffi::OsString;
use std::fs::{self, DirBuilder, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use crate::dataset::{AnswerText, Id, LINE_BREAKS};

/// The line before an answer's reasoning in the temp file.
pub(super) const REASONING_MARKER: &str = "=== overbrainer: reasoning (leave empty for none) ===";
/// The line before an answer's content in the temp file.
pub(super) const ANSWER_MARKER: &str = "=== overbrainer: answer ===";

/// What is being edited, as it was when the editor opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Target {
    /// The text of question `id`.
    Question {
        /// The question.
        id: Id,
        /// Its text.
        text: String,
    },
    /// The assistant message of answer `id`.
    Answer {
        /// The answer.
        id: Id,
        /// Its reasoning and content.
        before: AnswerText,
    },
    /// The name of subtopic `id`.
    Subtopic {
        /// The subtopic.
        id: Id,
        /// Its name.
        name: String,
    },
}

/// An edit the user made, checked, ready to apply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Edited {
    /// A question's new text.
    Question {
        /// The question.
        id: Id,
        /// Its text before.
        before: String,
        /// Its new text.
        text: String,
    },
    /// An answer's new assistant message.
    Answer {
        /// The answer.
        id: Id,
        /// Before.
        before: AnswerText,
        /// After.
        after: AnswerText,
    },
    /// A subtopic's new name.
    Subtopic {
        /// The subtopic.
        id: Id,
        /// Its name before.
        before: String,
        /// Its new name.
        name: String,
    },
}

/// Why an edited file changes nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Refusal {
    /// Why, for the status line.
    pub(super) message: String,
    /// Whether the temp file is kept, since it holds typed work.
    pub(super) keep: bool,
}

impl Refusal {
    fn drop(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            keep: false,
        }
    }

    fn keep(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            keep: true,
        }
    }
}

/// Most bytes an edited file may hold when it is read back.
const EDIT_MAX: u64 = 16 << 20;

/// The name of the temp file in its session directory.
const EDIT_FILE: &str = "edit.txt";

/// An open edit: the temp file and what it edits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Session {
    /// The temp file, `data/.edit-<16 hex>/edit.txt`, in a directory only its
    /// owner can enter.
    pub(super) path: PathBuf,
    /// What it edits.
    pub(super) target: Target,
}

impl Session {
    /// The edited text, read back as a regular file only, never through a
    /// link, from a session directory still ours and closed to others, and at
    /// most [`EDIT_MAX`] bytes. An editor that saves by writing a new file and
    /// renaming it over the old one is fine.
    ///
    /// # Errors
    ///
    /// Returns the error of reading the file.
    pub(super) fn read(&self) -> io::Result<Vec<u8>> {
        let dir = self.path.parent().unwrap_or_else(|| Path::new("."));
        crate::files::read_private(dir, EDIT_FILE, EDIT_MAX)
    }

    /// Removes the temp file and its directory, with whatever the editor left
    /// beside it (a backup, a swap file). Best effort: a leftover is harmless.
    pub(super) fn remove(&self) {
        if let Some(dir) = self.path.parent() {
            fs::remove_dir_all(dir).ok();
        }
    }
}

/// The text of `target` in the temp file.
///
/// # Errors
///
/// Returns why an answer whose text holds a marker line cannot be edited here.
pub(super) fn text_of(target: &Target) -> Result<String, String> {
    match target {
        Target::Question { text, .. } => Ok(format!("{text}\n")),
        Target::Subtopic { name, .. } => Ok(format!("{name}\n")),
        Target::Answer { before, .. } => {
            let reasoning = before.reasoning.as_deref().unwrap_or_default();
            let marked = |text: &str| {
                text.lines()
                    .any(|line| line == REASONING_MARKER || line == ANSWER_MARKER)
            };
            if marked(reasoning) || marked(&before.content) {
                return Err(
                    "this answer holds an overbrainer marker line: edit data/answers.jsonl by hand"
                        .to_string(),
                );
            }
            Ok(format!(
                "{REASONING_MARKER}\n{reasoning}\n{ANSWER_MARKER}\n{}\n",
                before.content
            ))
        },
    }
}

/// Writes `text` to a new temp file in a new directory of `data_dir` that only
/// its owner can enter, so another writer of `data/` cannot swap the file for a
/// link to one of the owner's files before it is read back.
///
/// # Errors
///
/// Returns the I/O error when the directory or the file cannot be created or
/// written.
pub(super) fn open(data_dir: &Path, target: Target, text: &str) -> io::Result<Session> {
    fs::create_dir_all(data_dir)?;
    loop {
        let dir = data_dir.join(format!(".edit-{:016x}", fastrand::u64(..)));
        match DirBuilder::new().mode(0o700).create(&dir) {
            Ok(()) => {},
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
        let path = dir.join(EDIT_FILE);
        let written = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)
            .and_then(|mut file| file.write_all(text.as_bytes()));
        if let Err(error) = written {
            fs::remove_dir_all(&dir).ok();
            return Err(error);
        }
        return Ok(Session { path, target });
    }
}

/// The editor command: `$VISUAL`, else `$EDITOR`, else `vi`, split on whitespace
/// (so `code --wait` works; no shell).
pub(super) fn command(visual: Option<OsString>, editor: Option<OsString>) -> Vec<String> {
    let words = |value: Option<OsString>| {
        value
            .map(|value| {
                value
                    .to_string_lossy()
                    .split_whitespace()
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .filter(|words| !words.is_empty())
    };
    words(visual)
        .or_else(|| words(editor))
        .unwrap_or_else(|| vec!["vi".to_string()])
}

/// What the edited `bytes` of `session`'s file change.
///
/// # Errors
///
/// Returns a [`Refusal`] when the text is not valid UTF-8, empty, unchanged,
/// has its marker lines changed, or names a subtopic on several lines.
pub(super) fn check(session: &Session, bytes: Vec<u8>) -> Result<Edited, Refusal> {
    let text = String::from_utf8(bytes)
        .map_err(|_| Refusal::drop("not valid UTF-8; nothing changed"))?
        .replace("\r\n", "\n");
    match &session.target {
        Target::Question { id, text: before } => {
            let text = one_text(&text, before)?;
            Ok(Edited::Question {
                id: id.clone(),
                before: before.clone(),
                text,
            })
        },
        Target::Subtopic { id, name: before } => {
            let name = one_text(&text, before)?;
            if name.contains(LINE_BREAKS) {
                return Err(Refusal::keep(
                    "a subtopic name must be one line; nothing changed",
                ));
            }
            Ok(Edited::Subtopic {
                id: id.clone(),
                before: before.clone(),
                name,
            })
        },
        Target::Answer { id, before } => {
            let after = answer(&text)?;
            // The original goes through the same parse, so the newlines the
            // sections lose there do not count as a change.
            let original = text_of(&session.target)
                .ok()
                .and_then(|text| answer(&text).ok());
            if &after == before || original.as_ref() == Some(&after) {
                return Err(Refusal::drop("unchanged"));
            }
            Ok(Edited::Answer {
                id: id.clone(),
                before: before.clone(),
                after,
            })
        },
    }
}

/// A question or subtopic name: the whole text, trimmed.
fn one_text(text: &str, before: &str) -> Result<String, Refusal> {
    let text = text.trim();
    if text.is_empty() {
        return Err(Refusal::drop("empty; nothing changed"));
    }
    if text == before.trim() {
        return Err(Refusal::drop("unchanged"));
    }
    Ok(text.to_string())
}

/// An answer's two sections, between and after the marker lines.
fn answer(text: &str) -> Result<AnswerText, Refusal> {
    let lines: Vec<&str> = text.split('\n').collect();
    let at = |marker: &str| {
        let found: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, line)| **line == marker)
            .map(|(index, _)| index)
            .collect();
        match found.as_slice() {
            [index] => Some(*index),
            _ => None,
        }
    };
    let (Some(reasoning_at), Some(answer_at)) = (at(REASONING_MARKER), at(ANSWER_MARKER)) else {
        return Err(Refusal::keep(
            "the marker lines were changed; nothing changed",
        ));
    };
    if reasoning_at > answer_at {
        return Err(Refusal::keep(
            "the marker lines were changed; nothing changed",
        ));
    }
    let section = |lines: &[&str]| lines.join("\n").trim_matches('\n').to_string();
    let reasoning = section(&lines[reasoning_at + 1..answer_at]);
    let content = section(&lines[answer_at + 1..]);
    if content.trim().is_empty() {
        return Err(Refusal::drop("empty; nothing changed"));
    }
    Ok(AnswerText {
        reasoning: (!reasoning.is_empty()).then_some(reasoning),
        content,
    })
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use super::*;

    fn answer_target(reasoning: Option<&str>, content: &str) -> Target {
        Target::Answer {
            id: Id::of(&["a"]),
            before: AnswerText {
                reasoning: reasoning.map(str::to_string),
                content: content.to_string(),
            },
        }
    }

    fn session(target: Target) -> Session {
        Session {
            path: PathBuf::from("unused"),
            target,
        }
    }

    fn question() -> Session {
        session(Target::Question {
            id: Id::of(&["q"]),
            text: "Why borrow?".into(),
        })
    }

    /// An edited file swapped for a link is not read back through it.
    #[test]
    fn a_session_file_swapped_for_a_link_is_refused() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let outside = dir.path().join("private");
        fs::write(&outside, "secret")?;
        let session = open(&dir.path().join("data"), question().target, "Why?\n")?;
        fs::remove_file(&session.path)?;
        std::os::unix::fs::symlink(&outside, &session.path)?;
        let error = session.read().err().ok_or("a link was read back")?;
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        Ok(())
    }

    /// An editor that saves by writing a new file and renaming it over the
    /// edited one is read back.
    #[test]
    fn a_save_by_rename_is_read_back() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let session = open(&dir.path().join("data"), question().target, "Why?\n")?;
        let saved = session.path.with_extension("swp");
        fs::write(&saved, "Why borrow?\n")?;
        fs::rename(&saved, &session.path)?;
        assert_eq!(session.read()?, b"Why borrow?\n");
        Ok(())
    }

    /// Each session is a new directory under `data/` only its owner enters,
    /// holding a new file only its owner reads, and it is removed whole.
    #[test]
    fn a_temp_file_is_new_private_and_under_data() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let data = dir.path().join("data");
        let target = Target::Subtopic {
            id: Id::of(&["s"]),
            name: "Borrowing".into(),
        };
        let first = open(&data, target.clone(), "Borrowing\n")?;
        let second = open(&data, target, "Borrowing\n")?;
        assert_ne!(first.path, second.path);
        let dir = first.path.parent().ok_or("no session directory")?;
        let name = dir.file_name().map(|n| n.to_string_lossy().into_owned());
        let name = name.unwrap_or_default();
        assert_eq!((name.get(..6), name.len()), (Some(".edit-"), 22));
        assert_eq!(dir.parent(), Some(data.as_path()));
        assert_eq!(fs::metadata(dir)?.permissions().mode() & 0o777, 0o700);
        assert_eq!(
            fs::metadata(&first.path)?.permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(first.read()?, b"Borrowing\n");
        first.remove();
        assert!(!dir.exists(), "the session directory is removed");
        Ok(())
    }

    #[test]
    fn the_editor_is_visual_then_editor_then_vi() {
        let some = |text: &str| Some(OsString::from(text));
        assert_eq!(
            command(some("code --wait"), some("nano")),
            ["code", "--wait"]
        );
        assert_eq!(command(some("  "), some("nano -w")), ["nano", "-w"]);
        assert_eq!(command(None, None), ["vi"]);
    }

    #[test]
    fn a_question_is_its_trimmed_text() {
        let edited = check(&question(), b"  Why borrow at all?\r\n\n".to_vec());
        assert_eq!(
            edited,
            Ok(Edited::Question {
                id: Id::of(&["q"]),
                before: "Why borrow?".into(),
                text: "Why borrow at all?".into(),
            })
        );
    }

    #[test]
    fn empty_unchanged_and_invalid_text_changes_nothing_and_drops_the_file() {
        for (bytes, message) in [
            (b" \n\n".to_vec(), "empty; nothing changed"),
            (b"Why borrow?\n".to_vec(), "unchanged"),
            (vec![0xff, 0xfe], "not valid UTF-8; nothing changed"),
        ] {
            assert_eq!(check(&question(), bytes), Err(Refusal::drop(message)));
        }
    }

    #[test]
    fn a_subtopic_name_must_be_one_line() {
        let subtopic = session(Target::Subtopic {
            id: Id::of(&["s"]),
            name: "Borrowing".into(),
        });
        for name in [
            "Loans\nand more\n",
            "Loans\rand more",
            "Loans\u{2028}and more",
        ] {
            assert_eq!(
                check(&subtopic, name.as_bytes().to_vec()),
                Err(Refusal::keep(
                    "a subtopic name must be one line; nothing changed"
                ))
            );
        }
    }

    #[test]
    fn an_answer_round_trips_through_its_markers() -> Result<(), String> {
        let target = answer_target(Some("Let me think.\nAgain."), "Because.");
        let text = text_of(&target)?;
        assert_eq!(
            text,
            format!("{REASONING_MARKER}\nLet me think.\nAgain.\n{ANSWER_MARKER}\nBecause.\n")
        );
        let session = session(target);
        assert_eq!(
            check(&session, text.clone().into_bytes()),
            Err(Refusal::drop("unchanged"))
        );
        let edited = text.replace("Because.", "\n\nBecause of moves.\n\n");
        let Ok(Edited::Answer { after, .. }) = check(&session, edited.into_bytes()) else {
            return Err("not an answer edit".into());
        };
        assert_eq!(after.content, "Because of moves.");
        assert_eq!(after.reasoning.as_deref(), Some("Let me think.\nAgain."));
        Ok(())
    }

    #[test]
    fn an_answer_opened_and_closed_as_is_is_unchanged() -> Result<(), String> {
        let session = session(answer_target(Some("\nLet me think."), "Because.\n"));
        let text = text_of(&session.target)?;
        assert_eq!(
            check(&session, text.into_bytes()),
            Err(Refusal::drop("unchanged"))
        );
        Ok(())
    }

    #[test]
    fn an_empty_reasoning_section_means_none() -> Result<(), String> {
        let session = session(answer_target(None, "Because."));
        let text = format!("{REASONING_MARKER}\n\n\n{ANSWER_MARKER}\nNew.\n");
        let Ok(Edited::Answer { after, .. }) = check(&session, text.into_bytes()) else {
            return Err("not an answer edit".into());
        };
        assert_eq!(after.reasoning, None);
        Ok(())
    }

    #[test]
    fn changed_marker_lines_are_refused_and_the_file_kept() {
        let session = session(answer_target(None, "Because."));
        let changed = Refusal::keep("the marker lines were changed; nothing changed");
        for text in [
            format!("{ANSWER_MARKER}\nNew.\n"),
            format!("{REASONING_MARKER}\n{REASONING_MARKER}\n{ANSWER_MARKER}\nNew.\n"),
            format!("{ANSWER_MARKER}\nNew.\n{REASONING_MARKER}\n"),
        ] {
            assert_eq!(check(&session, text.into_bytes()), Err(changed.clone()));
        }
        let empty = format!("{REASONING_MARKER}\nWhy.\n{ANSWER_MARKER}\n\n");
        assert_eq!(
            check(&session, empty.into_bytes()),
            Err(Refusal::drop("empty; nothing changed"))
        );
    }

    #[test]
    fn an_answer_holding_a_marker_line_is_not_opened() {
        let target = answer_target(Some(ANSWER_MARKER), "Because.");
        assert!(text_of(&target).is_err_and(|error| error.contains("marker line")));
    }
}
