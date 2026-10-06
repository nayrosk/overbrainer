//! The report of a compare: `compare.json` with every question, and
//! `compare.md`, what a reader gets.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::types::{read_json, write_json};
use super::{
    ChildAnswer, CompareError, CompareSetup, Costs, Hardware, Prices, REPORT_JSON, REPORT_MD,
    Summary, Verdict, VerdictLine, costs, summarize,
};
use crate::config::RoleModel;

/// Examples shown per kind in the Markdown report.
const SHOWN: usize = 5;
/// Questions under which the report says the sample is small.
const SMALL_SAMPLE: usize = 100;
/// Characters of the GGUF SHA-256 shown.
const SHA_SHOWN: usize = 12;

/// One question of the report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuestionResult {
    /// ID of the question.
    pub id: String,
    /// Topic name.
    pub topic: String,
    /// The question.
    pub question: String,
    /// The child's answer, reasoning removed; `None` when it failed.
    pub child: Option<String>,
    /// The parent's answer, reasoning removed.
    pub parent: String,
    /// The verdict for the child.
    pub verdict: Verdict,
    /// The judge's reason.
    pub reason: Option<String>,
    /// Seconds the child took.
    pub seconds: Option<f64>,
    /// Seconds the child took to the first token.
    pub first_token_seconds: Option<f64>,
    /// Completion tokens per second of the child's answer.
    pub tokens_per_second: Option<f64>,
    /// Why the child gave no answer.
    pub error: Option<String>,
}

/// The report of a compare, as `compare.json` holds it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    /// The run compared.
    pub run: String,
    /// The compare's ID.
    pub compare: String,
    /// The run's base model, when known.
    pub base_model: Option<String>,
    /// The GGUF's llama-quantize type.
    pub quantize: String,
    /// SHA-256 of the GGUF.
    pub gguf_sha256: String,
    /// The llama.cpp release that served it.
    pub llama_cpp: String,
    /// The machine that served it, when it said.
    pub hardware: Option<Hardware>,
    /// The judge, `provider/model`.
    pub judge: String,
    /// Whether the judge is the parent model itself.
    pub judge_is_parent: bool,
    /// The verdicts file the report read.
    pub verdicts_file: String,
    /// When the compare was created, RFC 3339 UTC.
    pub created: String,
    /// Seed of the order of each pair.
    pub seed: u64,
    /// The counts, rates and timings.
    pub summary: Summary,
    /// Dollars per 1,000 requests.
    pub costs: Costs,
    /// Whether the child's hourly price is the Runpod pod's.
    pub child_price_from_pod: bool,
    /// Every question, in eval order.
    pub questions: Vec<QuestionResult>,
}

impl Report {
    /// Reads `compare.json` of the compare directory `dir`.
    ///
    /// # Errors
    ///
    /// Returns [`CompareError::Io`] when it cannot be read, and
    /// [`CompareError::Json`] when it does not parse.
    pub fn load(dir: &Path) -> Result<Self, CompareError> {
        read_json(&dir.join(REPORT_JSON))
    }
}

/// The judge of a report.
#[derive(Debug, Clone, Copy)]
pub struct JudgeInfo<'a> {
    /// Its model.
    pub role: &'a RoleModel,
    /// Whether it is the parent.
    pub is_parent: bool,
    /// The verdicts file it wrote.
    pub verdicts_file: &'a str,
}

/// What a report is built from.
#[derive(Debug, Clone)]
pub struct Parts<'a> {
    /// What was compared.
    pub setup: &'a CompareSetup,
    /// The child's answers.
    pub answers: &'a [ChildAnswer],
    /// The judge's verdicts.
    pub verdicts: &'a [VerdictLine],
    /// The machine that served the child.
    pub hardware: Option<Hardware>,
    /// The judge.
    pub judge: JudgeInfo<'a>,
    /// The prices of the cost rows.
    pub prices: Prices,
    /// Whether `prices.child_per_hour` is the Runpod pod's.
    pub child_price_from_pod: bool,
}

/// The report of `parts`: questions in eval order, those without a verdict
/// left out of the counts (an interrupted judge).
#[must_use]
pub fn build(parts: &Parts<'_>) -> Report {
    let setup = parts.setup;
    let answers: HashMap<&str, &ChildAnswer> = parts
        .answers
        .iter()
        .map(|answer| (answer.id.as_str(), answer))
        .collect();
    let verdicts: HashMap<&str, &VerdictLine> = parts
        .verdicts
        .iter()
        .map(|line| (line.id.as_str(), line))
        .collect();
    let questions: Vec<QuestionResult> = setup
        .questions
        .iter()
        .filter_map(|question| {
            let line = verdicts.get(question.id.as_str())?;
            let answer = answers.get(question.id.as_str());
            Some(QuestionResult {
                id: question.id.clone(),
                topic: question.topic.clone(),
                question: question.text().to_string(),
                child: answer
                    .filter(|answer| !answer.is_error())
                    .and_then(|answer| answer.answer.as_deref())
                    .map(super::strip_reasoning),
                parent: question.parent.clone(),
                verdict: line.verdict,
                reason: line.reason.clone(),
                seconds: answer.and_then(|answer| answer.seconds),
                first_token_seconds: answer.and_then(|answer| answer.first_token_seconds),
                tokens_per_second: answer.and_then(|answer| answer.tokens_per_second),
                error: answer.and_then(|answer| answer.error.clone()),
            })
        })
        .collect();
    let summary = summarize(parts.verdicts, parts.answers);
    let costs = costs(&parts.prices, &setup.questions, summary.mean_seconds);
    Report {
        run: setup.run.clone(),
        compare: setup.compare.clone(),
        base_model: setup.base_model.clone(),
        quantize: setup.quantize.clone(),
        gguf_sha256: setup.gguf_sha256.clone(),
        llama_cpp: setup.llama_cpp.clone(),
        hardware: parts.hardware.clone(),
        judge: format!("{}/{}", parts.judge.role.provider, parts.judge.role.model),
        judge_is_parent: parts.judge.is_parent,
        verdicts_file: parts.judge.verdicts_file.to_string(),
        created: setup.created.clone(),
        seed: setup.seed,
        summary,
        costs,
        child_price_from_pod: parts.child_price_from_pod,
        questions,
    }
}

/// Writes `compare.json` and `compare.md` into the compare directory `dir`;
/// returns the path of `compare.md`.
///
/// # Errors
///
/// Returns [`CompareError::Io`] when a file cannot be written.
pub fn write(dir: &Path, report: &Report) -> Result<PathBuf, CompareError> {
    write_json(&dir.join(REPORT_JSON), report)?;
    let markdown = dir.join(REPORT_MD);
    std::fs::write(&markdown, render_markdown(report))
        .map_err(super::error::io_error(&markdown))?;
    Ok(markdown)
}

/// `rate` as a percentage, one decimal.
fn percent(rate: f64) -> String {
    format!("{:.1}%", rate * 100.0)
}

/// `seconds` with two decimals.
fn secs(seconds: f64) -> String {
    format!("{seconds:.2} s")
}

/// `dollars` with four decimals.
fn dollars(dollars: f64) -> String {
    format!("${dollars:.4}")
}

/// `value` rendered, or `-` when unknown.
fn or_dash<T>(value: Option<T>, render: impl Fn(T) -> String) -> String {
    value.map_or_else(|| "-".to_string(), render)
}

/// `text` on one line, cut to `max` characters.
fn one_line(text: &str, max: usize) -> String {
    let line: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if line.chars().count() <= max {
        return line;
    }
    let cut: String = line.chars().take(max.saturating_sub(3)).collect();
    format!("{cut}...")
}

/// `text` as a Markdown quote.
fn quoted(text: &str) -> String {
    text.lines()
        .map(|line| {
            if line.is_empty() {
                ">".to_string()
            } else {
                format!("> {line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The Markdown report: header, summary, the first losses and wins, limits.
#[must_use]
pub fn render_markdown(report: &Report) -> String {
    let mut text = String::new();
    // Writing to a `String` cannot fail.
    let _ = writeln!(text, "# Compare of run {}\n", report.run);
    render_header(&mut text, report);
    render_summary(&mut text, report);
    render_examples(&mut text, report);
    render_limits(&mut text, report);
    text
}

/// The table of what was compared.
fn render_header(text: &mut String, report: &Report) {
    let sha: String = report.gguf_sha256.chars().take(SHA_SHOWN).collect();
    let judge = if report.judge_is_parent {
        format!("{} (the parent)", report.judge)
    } else {
        report.judge.clone()
    };
    let rows = [
        ("Compare", report.compare.clone()),
        (
            "Base model",
            report
                .base_model
                .clone()
                .unwrap_or_else(|| "unknown".into()),
        ),
        (
            "GGUF",
            format!(
                "{}, llama.cpp {}, SHA-256 {sha}",
                report.quantize, report.llama_cpp
            ),
        ),
        (
            "Hardware",
            report
                .hardware
                .as_ref()
                .map_or_else(|| "unknown".into(), Hardware::describe),
        ),
        ("Judge", judge),
        ("Questions", report.summary.questions.to_string()),
        ("Date", report.created.clone()),
    ];
    let _ = writeln!(text, "| | |\n|---|---|");
    for (name, value) in rows {
        let _ = writeln!(text, "| {name} | {value} |");
    }
    text.push('\n');
}

/// The summary table.
fn render_summary(text: &mut String, report: &Report) {
    let summary = &report.summary;
    let costs = &report.costs;
    let unset = "not computed: set [compare] prices";
    let child_cost = costs.child_per_1k.map_or_else(
        || unset.to_string(),
        |cost| format!("{} (upper bound)", dollars(cost)),
    );
    let rows = [
        ("Win or tie", or_dash(summary.win_or_tie, percent)),
        (
            "Wins / ties / losses",
            format!(
                "{} / {} / {}",
                summary.wins,
                summary.ties,
                summary.losses + summary.errors
            ),
        ),
        (
            "Unparsed verdicts (not counted)",
            summary.unparsed.to_string(),
        ),
        (
            "Child errors (counted as losses)",
            summary.errors.to_string(),
        ),
        (
            "Child answers cut at the token limit",
            summary.truncated.to_string(),
        ),
        (
            "Latency p50 / p95",
            format!(
                "{} / {}",
                or_dash(summary.latency_p50, secs),
                or_dash(summary.latency_p95, secs)
            ),
        ),
        (
            "Time to first token p50",
            or_dash(summary.first_token_p50, secs),
        ),
        (
            "Output tokens per second",
            or_dash(summary.tokens_per_second, |rate| format!("{rate:.1}")),
        ),
        (
            "Parent cost per 1,000 requests",
            costs
                .parent_per_1k
                .map_or_else(|| unset.to_string(), dollars),
        ),
        ("Child cost per 1,000 requests", child_cost),
        ("Child cost / parent cost", or_dash(costs.ratio, percent)),
    ];
    let _ = writeln!(text, "## Summary\n\n| Measure | Value |\n|---|---|");
    for (name, value) in rows {
        let _ = writeln!(text, "| {name} | {value} |");
    }
    text.push('\n');
}

/// The first losses (child errors first) and the first wins, each with both answers.
fn render_examples(text: &mut String, report: &Report) {
    let errors = report
        .questions
        .iter()
        .filter(|question| question.verdict == Verdict::Error);
    let losses = report
        .questions
        .iter()
        .filter(|question| question.verdict == Verdict::Loss);
    let lost: Vec<&QuestionResult> = errors.chain(losses).collect();
    let won: Vec<&QuestionResult> = report
        .questions
        .iter()
        .filter(|question| question.verdict == Verdict::Win)
        .collect();
    for (title, list) in [("Losses", lost), ("Wins", won)] {
        if list.is_empty() {
            continue;
        }
        let _ = writeln!(
            text,
            "## {title} (first {} of {})\n",
            list.len().min(SHOWN),
            list.len()
        );
        for question in list.into_iter().take(SHOWN) {
            render_example(text, question);
        }
    }
}

/// One question with both answers and the judge's reason.
fn render_example(text: &mut String, question: &QuestionResult) {
    let _ = writeln!(
        text,
        "### {}: {}\n",
        question.id,
        one_line(&question.question, 80)
    );
    let child = match (&question.child, &question.error) {
        (Some(child), _) => quoted(child),
        (None, Some(error)) => format!("> (no answer: {error})"),
        (None, None) => "> (no answer)".to_string(),
    };
    let _ = writeln!(text, "Child:\n\n{child}\n");
    let _ = writeln!(text, "Parent:\n\n{}\n", quoted(&question.parent));
    if let Some(reason) = &question.reason {
        let _ = writeln!(text, "Judge: {reason}\n");
    }
}

/// The limits of the numbers, those that apply.
fn render_limits(text: &mut String, report: &Report) {
    let summary = &report.summary;
    let mut limits = Vec::new();
    if report.judge_is_parent {
        limits.push(
            "The judge is the parent model: it may favor its own answers. Set `roles.judge` to \
             another model and run `overbrainer compare --rejudge <compare-id>`."
                .to_string(),
        );
    }
    if summary.questions < SMALL_SAMPLE {
        limits.push(format!(
            "{} questions: a small sample, the rates move by several points from one set to \
             another.",
            summary.questions
        ));
    }
    limits.push(
        "Requests were sent one at a time: the latency is that of one user, and the child cost \
         is an upper bound (a server under load answers several requests at once)."
            .to_string(),
    );
    if report.child_price_from_pod {
        limits.push("The child's hourly price is the Runpod pod's.".to_string());
    }
    if summary.truncated > 0 {
        let answers = if summary.truncated == 1 {
            "child answer"
        } else {
            "child answers"
        };
        limits.push(format!(
            "{} {answers} hit the token limit; raise [compare] max_tokens or the context.",
            summary.truncated
        ));
    }
    if report
        .hardware
        .as_ref()
        .is_some_and(|hardware| !hardware.has_gpu())
    {
        limits.push("The child ran on a CPU.".to_string());
    }
    let _ = writeln!(text, "## Limits\n");
    for limit in limits {
        let _ = writeln!(text, "- {limit}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compare::fixtures::{answers, judge, sample_report, setup, verdicts};

    /// What a test returns.
    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// Where the report snapshots are stored.
    const SNAPSHOTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/snapshots/compare");

    /// Compares `text` with the stored snapshot `name`.
    fn snapshot(name: &str, text: &str) {
        let mut settings = insta::Settings::clone_current();
        settings.set_snapshot_path(SNAPSHOTS);
        settings.set_prepend_module_to_snapshot(false);
        settings.set_omit_expression(true);
        settings.bind(|| insta::assert_snapshot!(name.to_string(), text));
    }

    /// With prices and a GPU: every row, and the limits that apply.
    #[test]
    fn report_with_prices() -> TestResult {
        let report = sample_report()?;
        assert_eq!(report.summary.win_or_tie, Some(5.0 / 7.0));
        snapshot("report_with_prices", &render_markdown(&report));
        Ok(())
    }

    /// Without prices, on a CPU, with two child answers cut at the token
    /// limit: the cost rows say why they are missing, and the limits name
    /// the CPU and the cut answers.
    #[test]
    fn report_without_prices_on_cpu() -> TestResult {
        let (setup, mut answers, verdicts, role) = (setup(), answers(), verdicts(), judge()?);
        for answer in answers
            .iter_mut()
            .filter(|answer| answer.id == "q5" || answer.id == "q6")
        {
            answer.finish = Some("length".into());
        }
        let report = build(&Parts {
            setup: &setup,
            answers: &answers,
            verdicts: &verdicts,
            hardware: Some(Hardware {
                build: "ubuntu-x64".into(),
                gpus: Vec::new(),
                cpu: Some("AMD EPYC 7413".into()),
            }),
            judge: JudgeInfo {
                role: &role,
                is_parent: false,
                verdicts_file: "verdicts-0123456789abcdef.jsonl",
            },
            prices: Prices::default(),
            child_price_from_pod: false,
        });
        assert_eq!(report.summary.truncated, 2);
        snapshot("report_without_prices_on_cpu", &render_markdown(&report));
        Ok(())
    }

    /// The JSON holds every question with its timings and no secret-like
    /// field; it reads back to the same report.
    #[test]
    fn the_json_report_reads_back() -> TestResult {
        let dir = tempfile::tempdir()?;
        let report = sample_report()?;
        let markdown = write(dir.path(), &report)?;
        assert!(markdown.ends_with(crate::compare::REPORT_MD));
        let back = Report::load(dir.path())?;
        assert_eq!(back.questions.len(), 8);
        assert_eq!(
            serde_json::to_string_pretty(&back)?,
            serde_json::to_string_pretty(&report)?
        );
        let first = back.questions.first().ok_or("no question")?;
        assert_eq!(first.id, "q1");
        assert_eq!(first.verdict, Verdict::Win);
        let first_token = first.first_token_seconds.ok_or("no first token time")?;
        assert!((first_token - 0.05).abs() < 1e-9);
        let rate = first.tokens_per_second.ok_or("no rate")?;
        assert!((rate - 60.0).abs() < 1e-9);
        let failed = back.questions.get(6).ok_or("no q7")?;
        assert_eq!(failed.tokens_per_second, None);
        let json = std::fs::read_to_string(dir.path().join(crate::compare::REPORT_JSON))?;
        assert!(!json.contains("api_key") && !json.contains("base_url"));
        Ok(())
    }
}
