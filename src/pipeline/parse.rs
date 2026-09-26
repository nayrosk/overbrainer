//! Extracting a list of strings from a model answer, tolerant of the shapes
//! different providers return.

/// Extracts a list of strings from a model answer, tolerating prose or a code fence
/// around it. In order, it tries: a JSON array of strings at any `[` in the text,
/// salvaging the complete leading items of an array the token limit cut short; then,
/// when no array yields an item, a numbered or bulleted plain-text list. An array
/// wrapped in an object (`{"items": [...]}`) is matched by the array scan, since the
/// inner `[` is tried like any other. Items are trimmed and empty ones dropped.
/// Returns `None` when the answer holds no list.
#[must_use]
pub fn string_array(text: &str) -> Option<Vec<String>> {
    text.char_indices()
        .filter(|&(_, c)| c == '[')
        .find_map(|(start, _)| array_from(&text[start..]))
        .or_else(|| list_items(text))
}

/// Reads the string elements of a JSON array starting at `slice` (which begins with
/// `[`), stopping at its closing bracket or, when the array is truncated, at the first
/// element that is not a complete JSON string. Returns the complete items when there
/// is at least one, so a truncated array still yields what finished; `None` when the
/// array holds no complete string (a number, a `]` right away, or an unterminated
/// first string).
fn array_from(slice: &str) -> Option<Vec<String>> {
    let bytes = slice.as_bytes();
    let mut i = 1; // past the opening '['
    let mut items = Vec::new();
    loop {
        while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b',') {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] == b']' || bytes[i] != b'"' {
            break;
        }
        let Some((item, len)) = take_json_string(&slice[i..]) else {
            break;
        };
        let item = item.trim();
        if !item.is_empty() {
            items.push(item.to_string());
        }
        i += len;
    }
    (!items.is_empty()).then_some(items)
}

/// Reads one JSON string from the start of `slice` (which begins with `"`). Returns
/// the unescaped value and the number of bytes it spans, or `None` when the string is
/// not terminated (the answer was cut short).
fn take_json_string(slice: &str) -> Option<(String, usize)> {
    let bytes = slice.as_bytes();
    let mut i = 1; // past the opening quote
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2, // skip the escape and its next byte
            b'"' => {
                let end = i + 1;
                let value: String = serde_json::from_str(&slice[..end]).ok()?;
                return Some((value, end));
            },
            _ => i += 1,
        }
    }
    None
}

/// Reads a numbered or bulleted plain-text list: lines starting with `-`, `*`, `•`, or
/// a number followed by `.` or `)`. Returns the item texts when at least one line
/// carries a marker, else `None`.
fn list_items(text: &str) -> Option<Vec<String>> {
    let items: Vec<String> = text
        .lines()
        .filter_map(strip_marker)
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .collect();
    (!items.is_empty()).then_some(items)
}

/// The text after a leading list marker on `line`, or `None` when the line has none.
fn strip_marker(line: &str) -> Option<&str> {
    let line = line.trim_start();
    for marker in ['-', '*', '•'] {
        if let Some(rest) = line.strip_prefix(marker)
            && (rest.is_empty() || rest.starts_with(char::is_whitespace))
        {
            return Some(rest.trim_start());
        }
    }
    let digits = line.chars().take_while(char::is_ascii_digit).count();
    if digits > 0 {
        let rest = &line[digits..];
        if let Some(rest) = rest.strip_prefix('.').or_else(|| rest.strip_prefix(')')) {
            return Some(rest.trim_start());
        }
    }
    None
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

    #[test]
    fn an_array_wrapped_in_an_object_is_matched() {
        assert_eq!(
            string_array(r#"{"questions": ["a", "b"]}"#),
            Some(vec!["a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn a_truncated_array_keeps_its_complete_items() {
        assert_eq!(
            string_array(r#"["a", "b", "cut off"#),
            Some(vec!["a".to_string(), "b".to_string()])
        );
        assert_eq!(
            string_array(r#"["only one", "#),
            Some(vec!["only one".to_string()])
        );
    }

    #[test]
    fn escapes_inside_strings_survive() {
        assert_eq!(
            string_array(r#"["a \"quote\" and \\ slash"]"#),
            Some(vec![r#"a "quote" and \ slash"#.to_string()])
        );
    }

    #[test]
    fn a_numbered_list_is_read_when_there_is_no_array() {
        assert_eq!(
            string_array("1. first\n2) second\n3. third"),
            Some(vec![
                "first".to_string(),
                "second".to_string(),
                "third".to_string(),
            ])
        );
    }

    #[test]
    fn a_bulleted_list_is_read_when_there_is_no_array() {
        assert_eq!(
            string_array("Here they are:\n- one\n* two\n\u{2022} three\nthanks"),
            Some(vec![
                "one".to_string(),
                "two".to_string(),
                "three".to_string(),
            ])
        );
    }

    #[test]
    fn prose_without_markers_is_not_a_list() {
        assert_eq!(string_array("I cannot help with that request."), None);
    }
}
