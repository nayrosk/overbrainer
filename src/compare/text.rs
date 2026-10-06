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
    while let Some(start) = rest.find(OPEN) {
        kept.push_str(&rest[..start]);
        rest = match rest[start..].find(CLOSE) {
            Some(end) => &rest[start + end + CLOSE.len()..],
            None => "",
        };
    }
    kept.push_str(rest);
    collapse_spaces(kept.trim())
}

/// `text` with runs of spaces left by a removed block made one space; line
/// breaks are kept.
fn collapse_spaces(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space = false;
    for c in text.chars() {
        if c == ' ' {
            if !space {
                out.push(c);
            }
            space = true;
        } else {
            out.push(c);
            space = false;
        }
    }
    out
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
}
