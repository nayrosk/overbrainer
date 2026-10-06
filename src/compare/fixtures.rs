//! Fixtures shared by the tests of the compare report and of the views that show it.

use super::{
    ChatMessage, ChildAnswer, CompareSetup, EvalQuestion, Hardware, JudgeInfo, Parts, Prices,
    Report, Verdict, VerdictLine, build_report,
};
use crate::config::RoleModel;

/// Eight questions: four wins, a tie, two losses (one a child error), an unparsed verdict.
pub(crate) fn setup() -> CompareSetup {
    let questions = (1..=8)
        .map(|i| EvalQuestion {
            id: format!("q{i}"),
            topic: "ownership".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: format!("Question {i}: why does the borrow checker\nrefuse this?"),
            }],
            parent: format!("Parent answer {i}."),
            parent_input_tokens: 50,
            parent_output_tokens: 400,
        })
        .collect();
    CompareSetup {
        run: "demo_20261006-100000".into(),
        compare: "compare_20261006-120000".into(),
        gguf: "output/gguf/demo_20261006-100000-Q4_K_M.gguf".into(),
        gguf_sha256: "ab".repeat(32),
        quantize: "Q4_K_M".into(),
        llama_cpp: "b11320".into(),
        base_model: Some("Qwen/Qwen3-4B".into()),
        seed: 42,
        created: "2026-10-06T12:00:00Z".into(),
        questions,
    }
}

/// The child's answers to [`setup`]: q7 failed.
pub(crate) fn answers() -> Vec<ChildAnswer> {
    (1..=8)
        .map(|i| {
            if i == 7 {
                return ChildAnswer {
                    id: "q7".into(),
                    error: Some("timed out".into()),
                    ..ChildAnswer::default()
                };
            }
            ChildAnswer {
                id: format!("q{i}"),
                answer: Some(format!("Child answer {i}.")),
                finish: Some("stop".into()),
                input_tokens: Some(50),
                output_tokens: Some(120),
                seconds: Some(0.25 * f64::from(i)),
                first_token_seconds: Some(0.05),
                tokens_per_second: Some(60.0),
                error: None,
            }
        })
        .collect()
}

/// The verdicts of the eight questions of [`setup`].
pub(crate) fn verdicts() -> Vec<VerdictLine> {
    let verdict = |i: u32| match i {
        1..=4 => Verdict::Win,
        5 => Verdict::Tie,
        6 => Verdict::Loss,
        7 => Verdict::Error,
        _ => Verdict::Unparsed,
    };
    (1..=8)
        .map(|i| VerdictLine {
            id: format!("q{i}"),
            verdict: verdict(i),
            reason: (i <= 6).then(|| format!("Reason {i}.")),
            child_first: i % 2 == 0,
        })
        .collect()
}

/// The judge of the tests: the parent's model.
pub(crate) fn judge() -> Result<RoleModel, serde_json::Error> {
    serde_json::from_value(
        serde_json::json!({"provider": "openrouter", "model": "deepseek/deepseek-r1"}),
    )
}

/// The report of the eight questions, with prices, on one GPU, judged by the parent.
pub(crate) fn sample_report() -> Result<Report, serde_json::Error> {
    let (setup, answers, verdicts, role) = (setup(), answers(), verdicts(), judge()?);
    Ok(build_report(&Parts {
        setup: &setup,
        answers: &answers,
        verdicts: &verdicts,
        hardware: Some(Hardware {
            build: "ubuntu-cuda-13.4-x64".into(),
            gpus: vec!["NVIDIA A40".into()],
            cpu: None,
        }),
        judge: JudgeInfo {
            role: &role,
            is_parent: true,
            verdicts_file: "verdicts-0123456789abcdef.jsonl",
        },
        prices: Prices {
            parent_in: Some(0.55),
            parent_out: Some(2.19),
            child_per_hour: Some(0.40),
        },
        child_price_from_pod: true,
    }))
}
