//! The `overbrainer history` subcommand.

use std::path::Path;

use anyhow::Context as _;

use super::HistoryArgs;
use crate::history::{self, Cost, Entry, Total};

/// Prints the totals per stage and overall, or with `--all` every execution.
///
/// # Errors
///
/// Returns an error when the history exists but cannot be read.
pub fn run(project_dir: &Path, args: &HistoryArgs) -> anyhow::Result<()> {
    let entries = history::read(project_dir)
        .with_context(|| format!("cannot read {}", history::path(project_dir).display()))?;
    if entries.is_empty() {
        println!("no stage has run yet");
        return Ok(());
    }
    if args.all {
        for entry in &entries {
            println!("{}", entry_line(entry));
        }
        return Ok(());
    }
    let (per_stage, all) = history::totals(&entries);
    for (stage, total) in &per_stage {
        println!("{}", total_line(stage.name(), total));
    }
    println!("{}", total_line("total", &all));
    Ok(())
}

fn total_line(name: &str, total: &Total) -> String {
    format!(
        "{name:<9}  {} run(s), {} done, {} failed; tokens {} in, {} out; {}",
        total.runs, total.done, total.failed, total.input_tokens, total.output_tokens, total.cost
    )
}

fn entry_line(entry: &Entry) -> String {
    if let Some(split) = &entry.split {
        return format!(
            "{}  {:<9}  {:<11}  train {}, eval {}, excluded {}, orphaned {}",
            entry.started_at,
            entry.stage.name(),
            entry.status.name(),
            split.train,
            split.eval,
            entry.excluded,
            split.orphaned,
        );
    }
    let model = entry.model.as_deref().unwrap_or("-");
    let cost = entry.cost.map_or(Cost::Unknown, Cost::Known);
    let note = if entry.backfilled {
        " (backfilled)"
    } else {
        ""
    };
    format!(
        "{}  {:<9}  {:<11}  {model}  {} done, {} skipped, {} failed, {} excluded; tokens {} in, {} out; {cost}{note}",
        entry.started_at,
        entry.stage.name(),
        entry.status.name(),
        entry.done,
        entry.skipped,
        entry.failed,
        entry.excluded,
        entry.input_tokens,
        entry.output_tokens,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::Stage;
    use crate::history::Status;

    #[test]
    fn a_backfilled_entry_says_so() {
        let mut entry = Entry {
            stage: Stage::Answers,
            started_at: "2026-09-27T10:00:00Z".into(),
            ended_at: "2026-09-27T10:00:00Z".into(),
            status: Status::Ok,
            provider: None,
            model: Some("parent".into()),
            done: 3,
            skipped: 0,
            failed: 0,
            excluded: 1,
            input_tokens: 30,
            output_tokens: 15,
            cost: None,
            split: None,
            backfilled: false,
        };
        assert!(!entry_line(&entry).contains("backfilled"));
        entry.backfilled = true;
        assert!(
            entry_line(&entry).ends_with("cost unknown (backfilled)"),
            "{}",
            entry_line(&entry)
        );
    }
}
