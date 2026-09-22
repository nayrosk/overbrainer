use serde::Deserialize;

/// Extracts a JSON array of strings from a model answer, tolerating prose or a code
/// fence around it. Tries each `[` in the text in turn and keeps the first one that
/// starts a valid JSON array of strings, so a count or a citation marker in brackets
/// elsewhere in the text does not prevent a match. Items are trimmed and empty ones
/// dropped. Returns `None` when the answer holds no such array.
#[must_use]
pub fn string_array(text: &str) -> Option<Vec<String>> {
    text.char_indices()
        .filter(|&(_, c)| c == '[')
        .find_map(|(start, _)| array_at(&text[start..]))
}

/// Parses a JSON array of strings from the start of `slice`, ignoring anything after
/// its closing bracket.
fn array_at(slice: &str) -> Option<Vec<String>> {
    let mut de = serde_json::Deserializer::from_str(slice);
    let items = Vec::<String>::deserialize(&mut de).ok()?;
    Some(
        items
            .into_iter()
            .map(|item| item.trim().to_string())
            .filter(|item| !item.is_empty())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_fenced_and_wrapped_arrays_parse() {
        let expected = Some(vec!["a".to_string(), "b".to_string()]);
        assert_eq!(string_array(r#"["a", " b "]"#), expected);
        assert_eq!(string_array("```json\n[\"a\", \"b\", \"\"]\n```"), expected);
        assert_eq!(string_array("Here you go: [\"a\",\"b\"] Enjoy."), expected);
    }

    #[test]
    fn anything_else_is_rejected() {
        assert_eq!(string_array("no array"), None);
        assert_eq!(string_array("] backwards ["), None);
        assert_eq!(string_array("[1, 2]"), None);
        assert_eq!(string_array("[\"unterminated"), None);
    }

    #[test]
    fn a_count_in_brackets_before_the_array_is_skipped() {
        assert_eq!(
            string_array(r#"Here are [3] questions: ["a"]"#),
            Some(vec!["a".to_string()])
        );
    }

    #[test]
    fn a_trailing_reference_marker_does_not_prevent_the_match() {
        assert_eq!(
            string_array(r#"["a", "b"] see reference [1]"#),
            Some(vec!["a".to_string(), "b".to_string()])
        );
    }
}
