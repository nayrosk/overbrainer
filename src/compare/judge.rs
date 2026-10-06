//! The judge of a compare: the order it sees each pair in, its prompt, and
//! what its reply says.

use serde_json::Value;
use twox_hash::XxHash3_64;

use super::{CompareError, EvalQuestion, Verdict};
use crate::config::RoleModel;
use crate::prompts::{JUDGE, Prompts};

/// Whether the child's answer is shown first, as answer A, for question `id`:
/// the low bit of its XXH3-64 with `seed`. About half the questions each way,
/// the same order on every rejudge.
#[must_use]
pub fn child_first(seed: u64, id: &str) -> bool {
    XxHash3_64::oneshot_with_seed(seed, id.as_bytes()) & 1 == 1
}

/// What the judge picked, before it is mapped to the child.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pick {
    /// Answer A.
    A,
    /// Answer B.
    B,
    /// Neither.
    Tie,
}

/// The judge's reply parsed: its pick and its reason, trimmed. `None` when
/// the reply holds no JSON object whose `verdict` is A, B or tie (any case).
#[must_use]
pub fn parse_reply(reply: &str) -> Option<(Pick, String)> {
    let start = reply.find('{')?;
    let end = reply.rfind('}')?;
    if end < start {
        return None;
    }
    let value: Value = serde_json::from_str(&reply[start..=end]).ok()?;
    let pick = match value
        .get("verdict")?
        .as_str()?
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "a" => Pick::A,
        "b" => Pick::B,
        "tie" => Pick::Tie,
        _ => return None,
    };
    let reason = value
        .get("reason")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    Some((pick, reason))
}

/// The child's verdict for `pick`, its answer shown first or not.
#[must_use]
pub fn verdict_of(pick: Pick, child_first: bool) -> Verdict {
    match (pick, child_first) {
        (Pick::Tie, _) => Verdict::Tie,
        (Pick::A, true) | (Pick::B, false) => Verdict::Win,
        (Pick::A, false) | (Pick::B, true) => Verdict::Loss,
    }
}

/// The key of a judge: 16 hex digits of the XXH3-64 of its provider, model
/// and prompt template. The verdicts of one key share a file.
#[must_use]
pub fn judge_key(role: &RoleModel, prompt: &str) -> String {
    let text = format!("{}\u{1f}{}\u{1f}{prompt}", role.provider, role.model);
    format!("{:016x}", XxHash3_64::oneshot(text.as_bytes()))
}

/// The verdicts file of judge `key`: `verdicts-<key>.jsonl`.
#[must_use]
pub fn verdicts_file(key: &str) -> String {
    format!("verdicts-{key}.jsonl")
}

/// The judge prompt for `question`, answers `(a, b)` in that order.
///
/// # Errors
///
/// Returns [`CompareError::Prompt`] when the template fails to render.
pub fn render_judge(
    prompts: &Prompts,
    question: &EvalQuestion,
    (answer_a, answer_b): (&str, &str),
) -> Result<String, CompareError> {
    Ok(prompts.render(
        JUDGE,
        minijinja::context! {
            question => question.text(),
            answer_a => answer_a,
            answer_b => answer_b,
        },
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::{ChatMessage, EvalQuestion, Verdict};
    use crate::config::RoleModel;
    use crate::prompts::Prompts;

    /// Result type of the tests.
    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// Where the judge prompt snapshot is stored.
    const SNAPSHOTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/snapshots/compare");

    /// The question the prompt tests ask.
    fn question() -> EvalQuestion {
        EvalQuestion {
            id: "q1".into(),
            topic: "t".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: "Why borrow?".into(),
            }],
            parent: "Parent says.".into(),
            parent_input_tokens: 1,
            parent_output_tokens: 1,
        }
    }

    /// The order is fixed by the seed and the question, and mixed over many questions.
    #[test]
    fn the_order_is_seeded_and_mixed() {
        let firsts: Vec<bool> = (0..200)
            .map(|i| child_first(42, &format!("q{i}")))
            .collect();
        let shown_first = firsts.iter().filter(|first| **first).count();
        assert!((60..=140).contains(&shown_first), "{shown_first} of 200");
        assert_eq!(child_first(42, "q7"), child_first(42, "q7"));
        assert!(
            (0..50).any(|i| child_first(1, &format!("q{i}")) != child_first(2, &format!("q{i}")))
        );
    }

    /// The JSON object is found in the reply, whatever is around it.
    #[test]
    fn replies_parse_with_noise_around_the_object() {
        assert_eq!(
            parse_reply(r#"{"verdict": "A", "reason": "More complete."}"#),
            Some((Pick::A, "More complete.".to_string()))
        );
        assert_eq!(
            parse_reply("Sure.\n```json\n{\"verdict\": \"b\", \"reason\": \" x \"}\n```"),
            Some((Pick::B, "x".to_string()))
        );
        assert_eq!(
            parse_reply(r#"{"verdict": "TIE"}"#),
            Some((Pick::Tie, String::new()))
        );
        assert_eq!(parse_reply(r#"{"verdict": "C"}"#), None);
        assert_eq!(parse_reply("A is better"), None);
        assert_eq!(parse_reply("} {"), None);
    }

    /// A pick for A or B is mapped back to the child through the order.
    #[test]
    fn a_pick_maps_back_to_the_child() {
        assert_eq!(verdict_of(Pick::A, true), Verdict::Win);
        assert_eq!(verdict_of(Pick::A, false), Verdict::Loss);
        assert_eq!(verdict_of(Pick::B, true), Verdict::Loss);
        assert_eq!(verdict_of(Pick::B, false), Verdict::Win);
        assert_eq!(verdict_of(Pick::Tie, false), Verdict::Tie);
    }

    /// Another judge model or prompt gets another verdicts file.
    #[test]
    fn the_judge_key_follows_model_and_prompt() -> TestResult {
        let role: RoleModel = serde_json::from_value(serde_json::json!(
            {"provider": "p", "model": "m"}
        ))?;
        let other: RoleModel = serde_json::from_value(serde_json::json!(
            {"provider": "p", "model": "n"}
        ))?;
        let key = judge_key(&role, "prompt");
        assert_eq!(key.len(), 16);
        assert_eq!(key, judge_key(&role, "prompt"));
        assert_ne!(key, judge_key(&other, "prompt"));
        assert_ne!(key, judge_key(&role, "prompt 2"));
        assert_eq!(verdicts_file(&key), format!("verdicts-{key}.jsonl"));
        Ok(())
    }

    /// The default prompt holds the question and both answers, in order.
    #[test]
    fn the_judge_prompt_shows_both_answers() -> TestResult {
        let dir = tempfile::tempdir()?;
        let prompts = Prompts::load(dir.path())?;
        let text = render_judge(&prompts, &question(), ("First answer.", "Second answer."))?;
        let (question_at, a, b) = (
            text.find("Why borrow?").ok_or("no question")?,
            text.find("First answer.").ok_or("no A")?,
            text.find("Second answer.").ok_or("no B")?,
        );
        assert!(question_at < a && a < b);
        assert!(text.contains("\"verdict\""));
        assert!(prompts.source(crate::prompts::JUDGE).is_some());
        Ok(())
    }

    /// The rendered judge prompt is stable.
    #[test]
    fn the_judge_prompt_snapshot() -> TestResult {
        let dir = tempfile::tempdir()?;
        let prompts = Prompts::load(dir.path())?;
        let text = render_judge(&prompts, &question(), ("First answer.", "Second answer."))?;
        let mut settings = insta::Settings::clone_current();
        settings.set_snapshot_path(SNAPSHOTS);
        settings.set_prepend_module_to_snapshot(false);
        settings.set_omit_expression(true);
        settings.bind(|| insta::assert_snapshot!("judge_prompt", text));
        Ok(())
    }
}
