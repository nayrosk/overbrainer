//! The judge of a compare: the order it sees each pair in, its prompt, and
//! what its reply says.

use std::collections::HashMap;
use std::path::Path;
use std::pin::Pin;

use futures::stream::{self, StreamExt as _};
use serde_json::Value;
use twox_hash::XxHash3_64;

use super::{
    ChildAnswer, CompareError, CompareSetup, EvalQuestion, Verdict, VerdictLine, append_verdict,
    read_verdicts,
};
use crate::config::RoleModel;
use crate::events::{Event, EventBus};
use crate::llm::{CompletionRequest, LlmClient, RetryPolicy, with_retry};
use crate::prompts::{JUDGE, Prompts};

/// One question being judged, boxed so the pool holds a concrete future
/// type (see [`judge_all`]).
type Judging<'a> = Pin<Box<dyn Future<Output = Result<VerdictLine, CompareError>> + Send + 'a>>;

/// Times a reply that does not parse is asked for, in all.
const ASKS: usize = 2;

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
/// Each `{` is tried in order, so braces in the text around the object do
/// not hide it: the first object with a valid verdict wins.
#[must_use]
pub fn parse_reply(reply: &str) -> Option<(Pick, String)> {
    reply
        .match_indices('{')
        .find_map(|(start, _)| parse_object(&reply[start..]))
}

/// The pick and reason of the JSON object `text` starts with, whatever
/// follows it; `None` when it starts with no object with a valid verdict.
fn parse_object(text: &str) -> Option<(Pick, String)> {
    let value = serde_json::Deserializer::from_str(text)
        .into_iter::<Value>()
        .next()?
        .ok()?;
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

/// What a judge run needs.
pub struct JudgeRun<'a, C> {
    /// The judge's client.
    pub client: &'a C,
    /// The judge's model and parameters.
    pub role: &'a RoleModel,
    /// The prompts, the judge's among them.
    pub prompts: &'a Prompts,
    /// Retries of a failed request.
    pub policy: RetryPolicy,
    /// Requests at once.
    pub concurrency: usize,
    /// Where the progress goes.
    pub bus: &'a EventBus,
    /// The compare's ID, for the progress events.
    pub compare: &'a str,
}

/// `n` as a `u64`.
fn count_u64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// Judges every question of `setup` that `file` has no verdict for yet,
/// `run.concurrency` at once, appending each verdict to `file` as it comes:
/// an interrupted run resumes where it stopped. A question the child has no
/// answer to in `answers` gets [`Verdict::Error`] without asking the judge.
/// Publishes [`Event::Judged`] after each verdict. Returns every verdict, in
/// the order of `setup`; a verdict in `file` for a question not in `setup`
/// is left out and not counted.
///
/// # Errors
///
/// Returns [`CompareError::Judge`] when a request fails for good (the
/// verdicts before it stay in `file`), [`CompareError::Prompt`] when the
/// prompt does not render, and [`CompareError::Io`] or
/// [`CompareError::Dataset`] when `file` cannot be read or written.
pub async fn judge_all<C: LlmClient>(
    run: &JudgeRun<'_, C>,
    setup: &CompareSetup,
    answers: &[ChildAnswer],
    file: &Path,
) -> Result<Vec<VerdictLine>, CompareError> {
    let mut done: HashMap<String, VerdictLine> = read_verdicts(file)?
        .into_iter()
        .map(|line| (line.id.clone(), line))
        .collect();
    let by_id: HashMap<&str, &ChildAnswer> = answers
        .iter()
        .map(|answer| (answer.id.as_str(), answer))
        .collect();
    let total = count_u64(setup.questions.len());
    let (judged_before, todo): (Vec<&EvalQuestion>, Vec<&EvalQuestion>) = setup
        .questions
        .iter()
        .partition(|question| done.contains_key(&question.id));
    let mut count = count_u64(judged_before.len());
    // Boxed, and made before the pool, so the pool holds a concrete future
    // type and no closure: a closure returning `impl Future` there made the
    // compare's own future lose its `Send` bound (needed to spawn it, in the
    // TUI) for a higher-ranked lifetime `rustc` cannot solve, as in the
    // questions stage.
    let judgings: Vec<Judging<'_>> = todo
        .into_iter()
        .map(|question| {
            let child = by_id
                .get(question.id.as_str())
                .copied()
                .filter(|answer| !answer.is_error())
                .and_then(|answer| answer.answer.as_deref());
            Box::pin(judge_one(run, setup.seed, question, child)) as Judging<'_>
        })
        .collect();
    let mut judged = stream::iter(judgings).buffer_unordered(run.concurrency.max(1));
    while let Some(result) = judged.next().await {
        let line = result?;
        append_verdict(file, &line)?;
        done.insert(line.id.clone(), line);
        count += 1;
        run.bus.publish(Event::Judged {
            compare_id: run.compare.to_string(),
            done: count,
            total,
        });
    }
    Ok(setup
        .questions
        .iter()
        .filter_map(|question| done.remove(&question.id))
        .collect())
}

/// The verdict of `question` against the child's answer `child`, its order
/// from `seed`: [`Verdict::Error`] without an answer, [`Verdict::Unparsed`]
/// when the reply does not parse, asked [`ASKS`] times.
async fn judge_one<C: LlmClient>(
    run: &JudgeRun<'_, C>,
    seed: u64,
    question: &EvalQuestion,
    child: Option<&str>,
) -> Result<VerdictLine, CompareError> {
    let first = child_first(seed, &question.id);
    let line = |verdict, reason| VerdictLine {
        id: question.id.clone(),
        verdict,
        reason,
        child_first: first,
    };
    let Some(child) = child else {
        return Ok(line(Verdict::Error, None));
    };
    let child = super::strip_reasoning(child);
    let parent = question.parent.as_str();
    let pair = if first {
        (child.as_str(), parent)
    } else {
        (parent, child.as_str())
    };
    let prompt = render_judge(run.prompts, question, pair)?;
    let request = CompletionRequest::for_role(run.role, None, prompt);
    for _ in 0..ASKS {
        let completion = with_retry(
            &run.policy,
            || run.client.complete(request.clone()),
            |error, wait| {
                tracing::warn!(
                    "compare: judging {} failed: {error}; retrying in {:.1}s",
                    question.id,
                    wait.as_secs_f64()
                );
            },
        )
        .await
        .map_err(|source| CompareError::Judge {
            id: question.id.clone(),
            source,
        })?;
        if let Some((pick, reason)) = parse_reply(&completion.content) {
            let reason = (!reason.is_empty()).then_some(reason);
            return Ok(line(verdict_of(pick, first), reason));
        }
    }
    Ok(line(Verdict::Unparsed, None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::compare::{
        ChatMessage, ChildAnswer, CompareSetup, EvalQuestion, Verdict, VerdictLine, append_verdict,
        read_verdicts,
    };
    use crate::config::RoleModel;
    use crate::dataset::FinishReason;
    use crate::events::{Event, EventBus};
    use crate::llm::{Completion, LlmError, Reasoning, Usage};
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

    /// Braces in the text before or after the JSON object do not hide it:
    /// the first object with a valid verdict wins.
    #[test]
    fn replies_parse_with_braces_around_the_object() {
        assert_eq!(
            parse_reply(
                "Answer A uses {braces} and a set {1, 2}.\n{\"verdict\": \"A\", \"reason\": \"Fuller.\"}"
            ),
            Some((Pick::A, "Fuller.".to_string()))
        );
        assert_eq!(
            parse_reply("{\"verdict\": \"B\", \"reason\": \"Right.\"}\nNote: see {x}."),
            Some((Pick::B, "Right.".to_string()))
        );
        assert_eq!(
            parse_reply(
                "Like {\"verdict\": \"C\"} is wrong; {\"verdict\": \"tie\", \"reason\": \"Same.\"} {end}"
            ),
            Some((Pick::Tie, "Same.".to_string()))
        );
        assert_eq!(parse_reply("{a} {\"verdict\": \"maybe\"} }"), None);
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

    /// A judge that prefers the answer holding `good`; replies garbage to a
    /// question holding `garbage`, and fails one holding `fail`.
    struct FakeJudge {
        /// Requests received.
        calls: AtomicUsize,
    }

    impl FakeJudge {
        /// A judge that has received nothing.
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }

        /// Its reply to `prompt`.
        fn reply(prompt: &str) -> Result<String, LlmError> {
            if prompt.contains("fail") {
                return Err(LlmError::InvalidResponse("refused".into()));
            }
            if prompt.contains("garbage") {
                return Ok("I cannot decide.".into());
            }
            let a = prompt.split("Answer A:\n").nth(1).unwrap_or_default();
            let a = a.split("Answer B:\n").next().unwrap_or_default();
            let pick = if a.contains("good") { "A" } else { "B" };
            Ok(format!(
                r#"{{"verdict": "{pick}", "reason": "{pick} is good."}}"#
            ))
        }
    }

    /// The fake judge as a client: completions only.
    impl LlmClient for FakeJudge {
        /// Counts the call and answers with [`FakeJudge::reply`].
        fn complete(
            &self,
            request: CompletionRequest,
        ) -> impl std::future::Future<Output = Result<Completion, LlmError>> + Send {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let reply = Self::reply(&request.prompt);
            std::future::ready(reply.map(|content| Completion {
                content,
                reasoning: Reasoning::none(),
                usage: Usage::default(),
                finish: FinishReason::Stop,
            }))
        }

        /// Unsupported: the judge never embeds.
        fn embed(
            &self,
            _inputs: &[String],
        ) -> impl std::future::Future<Output = Result<Vec<Vec<f32>>, LlmError>> + Send {
            std::future::ready(Err(LlmError::Unsupported("embeddings")))
        }
    }

    /// A setup of these questions; each parent answer is `Parent.`.
    fn setup_of(texts: &[&str]) -> CompareSetup {
        CompareSetup {
            run: "r1".into(),
            compare: "c1".into(),
            gguf: "output/gguf/r1-Q4_K_M.gguf".into(),
            gguf_sha256: "0".repeat(64),
            quantize: "Q4_K_M".into(),
            llama_cpp: "b11320".into(),
            base_model: None,
            seed: 7,
            created: "2026-10-06T12:00:00Z".into(),
            questions: texts
                .iter()
                .enumerate()
                .map(|(i, text)| EvalQuestion {
                    id: format!("q{i}"),
                    topic: "t".into(),
                    messages: vec![ChatMessage {
                        role: "user".into(),
                        content: (*text).to_string(),
                    }],
                    parent: "Parent.".into(),
                    parent_input_tokens: 1,
                    parent_output_tokens: 1,
                })
                .collect(),
        }
    }

    /// The child's answer `answer` to question `i`, or a failed request.
    fn child(i: usize, answer: Option<&str>) -> ChildAnswer {
        ChildAnswer {
            id: format!("q{i}"),
            answer: answer.map(str::to_string),
            error: answer.is_none().then(|| "timed out".to_string()),
            ..ChildAnswer::default()
        }
    }

    /// The judge's role.
    fn role() -> Result<RoleModel, serde_json::Error> {
        serde_json::from_value(serde_json::json!({"provider": "p", "model": "j"}))
    }

    /// Wins, losses, a failed child and an unparsed reply; then a second run
    /// asks the judge nothing.
    #[tokio::test]
    async fn every_question_is_judged_once_and_resumed() -> TestResult {
        let dir = tempfile::tempdir()?;
        let prompts = Prompts::load(dir.path())?;
        let role = role()?;
        let setup = setup_of(&["Q0?", "Q1?", "Q2?", "garbage Q3?"]);
        // q1 is a Loss only because the child is shown first for this seed
        // (the weak answer is then A, and the fake judge picks B, the
        // parent). A change of the hash fails here, not in the verdicts.
        assert!(child_first(setup.seed, "q1"));
        let answers = [
            child(0, Some("A good answer.")),
            child(1, Some("<think>x</think>A weak answer.")),
            child(2, None),
            child(3, Some("Whatever.")),
        ];
        let judge = FakeJudge::new();
        let bus = EventBus::new();
        let mut events = bus.subscribe();
        let run = JudgeRun {
            client: &judge,
            role: &role,
            prompts: &prompts,
            policy: RetryPolicy::new(0),
            concurrency: 2,
            bus: &bus,
            compare: "c1",
        };
        let file = dir.path().join("verdicts-x.jsonl");
        let verdicts = judge_all(&run, &setup, &answers, &file).await?;
        let got: Vec<(&str, Verdict)> = verdicts
            .iter()
            .map(|line| (line.id.as_str(), line.verdict))
            .collect();
        assert_eq!(
            got,
            [
                ("q0", Verdict::Win),
                ("q1", Verdict::Loss),
                ("q2", Verdict::Error),
                ("q3", Verdict::Unparsed),
            ]
        );
        // q0 and q1 once, q3 twice (asked again once), q2 never.
        assert_eq!(judge.calls.load(Ordering::SeqCst), 4);
        assert_eq!(read_verdicts(&file)?.len(), 4);
        let mut last = None;
        while let Ok(event) = events.try_recv() {
            if let Event::Judged { done, total, .. } = event {
                last = Some((done, total));
            }
        }
        assert_eq!(last, Some((4, 4)));
        let again = judge_all(&run, &setup, &answers, &file).await?;
        assert_eq!(again, verdicts);
        assert_eq!(
            judge.calls.load(Ordering::SeqCst),
            4,
            "nothing judged twice"
        );
        Ok(())
    }

    /// A verdict line for a question not in the setup stays out of the
    /// result and of the progress count; nothing is written for it.
    #[tokio::test]
    async fn a_stray_verdict_is_ignored() -> TestResult {
        let dir = tempfile::tempdir()?;
        let prompts = Prompts::load(dir.path())?;
        let role = role()?;
        let setup = setup_of(&["Q0?", "Q1?"]);
        let answers = [child(0, Some("good")), child(1, Some("good"))];
        let file = dir.path().join("verdicts-x.jsonl");
        let stray = VerdictLine {
            id: "gone".into(),
            verdict: Verdict::Win,
            reason: None,
            child_first: true,
        };
        append_verdict(&file, &stray)?;
        let judge = FakeJudge::new();
        let bus = EventBus::new();
        let mut events = bus.subscribe();
        let run = JudgeRun {
            client: &judge,
            role: &role,
            prompts: &prompts,
            policy: RetryPolicy::new(0),
            concurrency: 1,
            bus: &bus,
            compare: "c1",
        };
        let verdicts = judge_all(&run, &setup, &answers, &file).await?;
        let ids: Vec<&str> = verdicts.iter().map(|line| line.id.as_str()).collect();
        assert_eq!(ids, ["q0", "q1"]);
        assert_eq!(judge.calls.load(Ordering::SeqCst), 2);
        let mut last = None;
        while let Ok(event) = events.try_recv() {
            if let Event::Judged { done, total, .. } = event {
                last = Some((done, total));
            }
        }
        assert_eq!(last, Some((2, 2)));
        Ok(())
    }

    /// A judge error stops the run; what was judged stays for the next one.
    #[tokio::test]
    async fn a_judge_error_keeps_what_was_judged() -> TestResult {
        let dir = tempfile::tempdir()?;
        let prompts = Prompts::load(dir.path())?;
        let role = role()?;
        let setup = setup_of(&["Q0?", "fail Q1?"]);
        let answers = [child(0, Some("good")), child(1, Some("x"))];
        let judge = FakeJudge::new();
        let bus = EventBus::new();
        let run = JudgeRun {
            client: &judge,
            role: &role,
            prompts: &prompts,
            policy: RetryPolicy::new(0),
            concurrency: 1,
            bus: &bus,
            compare: "c1",
        };
        let file = dir.path().join("verdicts-x.jsonl");
        let error = judge_all(&run, &setup, &answers, &file)
            .await
            .err()
            .ok_or("judged despite the failure")?;
        assert!(
            matches!(error, CompareError::Judge { ref id, .. } if id == "q1"),
            "{error}"
        );
        assert_eq!(read_verdicts(&file)?.len(), 1);
        Ok(())
    }
}
