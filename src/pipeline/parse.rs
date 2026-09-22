/// Extracts a JSON array of strings from a model answer, tolerating prose or a code
/// fence around it. Items are trimmed and empty ones dropped. Returns `None` when the
/// answer holds no such array.
#[must_use]
pub fn string_array(text: &str) -> Option<Vec<String>> {
    let start = text.find('[')?;
    let end = text.rfind(']')?;
    if end < start {
        return None;
    }
    let items: Vec<String> = serde_json::from_str(&text[start..=end]).ok()?;
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
}
