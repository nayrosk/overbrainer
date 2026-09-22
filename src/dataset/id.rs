use std::fmt;

use serde::{Deserialize, Serialize};

/// Separator between the parts hashed into an [`Id`]. It cannot appear in normalized text.
const SEPARATOR: char = '\u{1f}';

/// Stable identifier: xxh3 128-bit hash of normalized content, as 32 lowercase hex chars.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Id(String);

impl Id {
    /// Hashes `parts`, joined by a separator that normalized text never contains.
    #[must_use]
    pub fn of(parts: &[&str]) -> Self {
        let joined = parts.join(&SEPARATOR.to_string());
        Self(format!(
            "{:032x}",
            twox_hash::XxHash3_128::oneshot(joined.as_bytes())
        ))
    }

    /// ID of a subtopic: topic name plus normalized subtopic name.
    #[must_use]
    pub fn subtopic(topic: &str, name: &str) -> Self {
        Self::of(&["subtopic", topic, &normalize(name)])
    }

    /// ID of a question: subtopic ID plus normalized question text.
    #[must_use]
    pub fn question(subtopic: &Id, text: &str) -> Self {
        Self::of(&["question", subtopic.as_str(), &normalize(text)])
    }

    /// The 32-char hex form.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Lowercases `text`, trims it and collapses every run of whitespace (and control
/// characters) into one space.
#[must_use]
pub fn normalize(text: &str) -> String {
    text.split(|c: char| c.is_whitespace() || c.is_control())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_32_lowercase_hex_chars() {
        let id = Id::of(&["a"]);
        assert_eq!(id.as_str().len(), 32);
        assert!(
            id.as_str()
                .chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
        );
    }

    #[test]
    fn ids_are_stable_across_versions() {
        // Pinned value: changing the hash or the joining breaks resume of existing data.
        assert_eq!(
            Id::subtopic("ownership", "Borrowing").as_str(),
            Id::subtopic("ownership", "  borrowing ").as_str()
        );
        assert_eq!(
            Id::subtopic("ownership", "borrowing").as_str(),
            "d410fa975affdd150daf206922cfd8cc"
        );
    }

    #[test]
    fn parts_do_not_run_together() {
        assert_ne!(Id::of(&["ab", "c"]), Id::of(&["a", "bc"]));
    }

    #[test]
    fn question_ids_depend_on_the_subtopic() {
        let first = Id::subtopic("t", "one");
        let second = Id::subtopic("t", "two");
        assert_ne!(
            Id::question(&first, "What is x?"),
            Id::question(&second, "What is x?")
        );
        assert_eq!(
            Id::question(&first, "What is  X?"),
            Id::question(&first, "what is x?")
        );
    }

    #[test]
    fn normalize_collapses_whitespace_and_case() {
        assert_eq!(normalize("  Hello\n\tWORLD  "), "hello world");
        assert_eq!(normalize(""), "");
    }
}
