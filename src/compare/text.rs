//! The answer of a model without its reasoning.

/// Opens a reasoning block.
const OPEN: &str = "<think>";
/// Closes a reasoning block.
const CLOSE: &str = "</think>";

/// `text` without its reasoning, trimmed: every `<think>...</think>` block is
/// removed, an opening `<think>` never closed drops the rest, and a
/// `</think>` before any `<think>` (a chat template that opened the block in
/// the prompt) drops what comes before it.
#[must_use]
pub fn strip_reasoning(text: &str) -> String {
    let mut rest = match (text.find(OPEN), text.find(CLOSE)) {
        (open, Some(close)) if open.is_none_or(|open| open > close) => &text[close + CLOSE.len()..],
        _ => text,
    };
    let mut kept = String::with_capacity(rest.len());
    let mut joined = false;
    while let Some(start) = rest.find(OPEN) {
        push_segment(&mut kept, &rest[..start], joined);
        joined = true;
        rest = match rest[start..].find(CLOSE) {
            Some(end) => &rest[start + end + CLOSE.len()..],
            None => "",
        };
    }
    push_segment(&mut kept, rest, joined);
    kept.trim().to_string()
}

/// Appends `segment` to `kept`. After a removed block (`joined`), leading
/// spaces of the segment are dropped when `kept` already ends with a space, so
/// the join leaves no doubled space; indentation elsewhere is untouched.
fn push_segment(kept: &mut String, segment: &str, joined: bool) {
    if joined && kept.ends_with(' ') {
        kept.push_str(segment.trim_start_matches(' '));
    } else {
        kept.push_str(segment);
    }
}

#[cfg(test)]
/// Tests of reasoning removal.
mod tests {
    use super::*;

    /// Reasoning blocks go, wherever they are; an unclosed one drops the rest,
    /// and a close without an open drops what comes before it.
    #[test]
    fn reasoning_is_removed_from_an_answer() {
        assert_eq!(strip_reasoning("  Borrow it.  "), "Borrow it.");
        assert_eq!(
            strip_reasoning("<think>why</think>\nBorrow it."),
            "Borrow it."
        );
        assert_eq!(
            strip_reasoning("A <think>x</think>B<think>y</think> C"),
            "A B C"
        );
        assert_eq!(strip_reasoning("Answer.<think>never closed"), "Answer.");
        assert_eq!(
            strip_reasoning("thinking first</think>\nAnswer."),
            "Answer."
        );
    }

    /// Code indentation survives, with or without a reasoning block before it.
    #[test]
    fn indentation_is_kept() {
        let code = "def f():\n    return 1";
        assert_eq!(strip_reasoning(code), code);
        assert_eq!(strip_reasoning(&format!("<think>x</think>\n{code}")), code);
        assert_eq!(strip_reasoning(&format!("{code}<think>x</think>")), code);
    }
}
