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
    let model = entry.model.as_deref().unwrap_or("-");
    let cost = entry.cost.map_or(Cost::Unknown, Cost::Known);
    format!(
        "{}  {:<9}  {:<11}  {model}  {} done, {} skipped, {} failed, {} excluded; tokens {} in, {} out; {cost}",
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
