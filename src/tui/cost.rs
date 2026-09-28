//! What the project cost so far: the stages the history recorded, the stage
//! running, and the Runpod pods of the runs.

use super::app::App;
use super::pipeline::{Row, StageState};
use super::training::estimated_spend;
use crate::history::{self, Cost, Entry};
use crate::llm::Usage;

/// `total` plus one more expense, `None` when its amount is unknown; a `None`
/// total is nothing spent yet.
fn add(total: Option<Cost>, cost: Option<f64>) -> Cost {
    total.map_or_else(|| cost.map_or(Cost::Unknown, Cost::Known), |t| t.plus(cost))
}

/// Whether `row` spent anything yet.
fn spent(row: &Row) -> bool {
    row.cost.is_some() || row.usage != Usage::default()
}

/// What `rows` spent, as the Pipeline view shows it.
pub(super) fn rows_cost(rows: &[Row]) -> Cost {
    rows.iter()
        .filter(|row| spent(row))
        .fold(None, |total, row| Some(add(total, row.cost)))
        .unwrap_or_default()
}

/// The cost of the stages recorded in `entries`, `None` when none spent anything:
/// `split` and a stage with nothing to do never make the sum partial.
pub(super) fn history_cost(entries: &[Entry]) -> Option<Cost> {
    let spent: Vec<Entry> = entries
        .iter()
        .filter(|e| e.cost.is_some() || e.input_tokens > 0 || e.output_tokens > 0)
        .cloned()
        .collect();
    (!spent.is_empty()).then(|| history::totals(&spent).1.cost)
}

/// The project's cost so far: the history, the stage running (a finished one is
/// in the history once reloaded), and every Runpod pod, live or deleted.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the footer shows it from the next commit")
)]
pub(super) fn project_cost(app: &App) -> Cost {
    let mut total = app.history_cost;
    for row in &app.pipeline.rows {
        if row.state == StageState::Running && spent(row) {
            total = Some(add(total, row.cost));
        }
    }
    for run in &app.training.runs {
        // A pod never created spent nothing.
        if run.pod.as_ref().is_some_and(|pod| pod.pod_id.is_some()) {
            total = Some(add(total, estimated_spend(run, app.now)));
        }
    }
    total.unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use crate::cli::data::Command;
    use crate::events::{Event, Stage, StageStats};
    use crate::history::{Cost, Entry, Span, Status};
    use crate::llm::Usage;
    use crate::runpod::PodState;
    use crate::runs::RunState;
    use crate::tui::snapshots::{app, pod, run};
    use crate::tui::training::RunRow;

    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// An item of `stage` done with `cost`.
    fn done(stage: Stage, cost: Option<f64>) -> Event {
        Event::ItemDone {
            stage,
            id: "x".into(),
            usage: Some(Usage {
                input_tokens: 10,
                output_tokens: 5,
            }),
            cost,
        }
    }

    /// An app running the answers stage, with one item done at `cost`.
    fn running(cost: Option<f64>) -> crate::tui::app::App {
        let mut app = app();
        app.pipeline.started(Command::Answers, 4);
        app.pipeline.event(&Event::StageStarted {
            stage: Stage::Answers,
            total: 10,
        });
        app.pipeline.event(&done(Stage::Answers, cost));
        app
    }

    fn entry(stage: Stage, tokens: u64, cost: Option<f64>) -> Entry {
        let stats = StageStats {
            usage: Usage {
                input_tokens: tokens,
                output_tokens: 0,
            },
            cost,
            ..StageStats::default()
        };
        let span = Span {
            started_at: "2026-09-27T10:00:00Z".into(),
            ended_at: "2026-09-27T10:01:00Z".into(),
        };
        Entry::from_stats(stage, span, Status::Ok, None, &stats)
    }

    #[test]
    fn nothing_at_all_is_unknown() {
        assert_eq!(project_cost(&app()), Cost::Unknown);
    }

    #[test]
    fn the_history_alone() {
        let mut app = app();
        app.history_cost = Some(Cost::Known(1.5));
        assert_eq!(project_cost(&app), Cost::Known(1.5));
        app.history_cost = Some(Cost::Unknown);
        assert_eq!(project_cost(&app), Cost::Unknown);
    }

    #[test]
    fn a_running_stage_adds_its_cost_so_far() {
        let mut app = running(Some(0.25));
        assert_eq!(project_cost(&app), Cost::Known(0.25));
        app.history_cost = Some(Cost::Known(1.5));
        assert_eq!(project_cost(&app), Cost::Known(1.75));
        // A history whose every cost is unknown makes the sum partial.
        app.history_cost = Some(Cost::Unknown);
        assert_eq!(project_cost(&app), Cost::Partial(0.25));
    }

    #[test]
    fn a_running_stage_that_spent_at_an_unknown_price_makes_it_partial() {
        let mut app = running(None);
        app.history_cost = Some(Cost::Known(1.5));
        assert_eq!(project_cost(&app), Cost::Partial(1.5));
    }

    #[test]
    fn a_running_stage_that_spent_nothing_yet_changes_nothing() {
        let mut app = app();
        app.pipeline.started(Command::Answers, 4);
        app.pipeline.event(&Event::StageStarted {
            stage: Stage::Answers,
            total: 10,
        });
        app.history_cost = Some(Cost::Known(1.5));
        assert_eq!(project_cost(&app), Cost::Known(1.5));
    }

    #[test]
    fn a_finished_stage_is_counted_once_by_the_history() {
        let mut app = running(Some(0.25));
        app.pipeline.event(&Event::StageFinished {
            stage: Stage::Answers,
            stats: StageStats {
                cost: Some(0.25),
                ..StageStats::default()
            },
        });
        app.history_cost = Some(Cost::Known(1.75));
        assert_eq!(project_cost(&app), Cost::Known(1.75));
    }

    #[test]
    fn a_deleted_pod_adds_its_recorded_spend() -> TestResult {
        let mut app = app();
        let mut record = pod("r1")?;
        record.state = PodState::Deleted;
        record.estimated_spend = Some(0.4);
        app.training.runs = vec![RunRow {
            record: run("r1", "gpu_cloud", RunState::Succeeded),
            pod: Some(record),
        }];
        assert_eq!(project_cost(&app), Cost::Known(0.4));
        app.history_cost = Some(Cost::Known(1.5));
        assert_eq!(project_cost(&app), Cost::Known(1.9));
        Ok(())
    }

    #[test]
    fn a_live_pod_adds_its_rate_times_its_uptime() -> TestResult {
        let mut app = app();
        app.training.runs = vec![RunRow {
            record: run("r1", "gpu_cloud", RunState::Running),
            pod: Some(pod("r1")?),
        }];
        // $0.53/h for 41 minutes.
        let Cost::Known(spend) = project_cost(&app) else {
            return Err("the spend is not known".into());
        };
        assert!((spend - 0.53 * 41.0 / 60.0).abs() < 1e-9, "{spend}");
        Ok(())
    }

    #[test]
    fn a_pod_at_an_unknown_rate_makes_it_partial_and_other_runs_add_nothing() -> TestResult {
        let mut app = app();
        let mut record = pod("r1")?;
        record.cost_per_hour = None;
        app.training.runs = vec![
            RunRow {
                record: run("r1", "gpu_cloud", RunState::Running),
                pod: Some(record),
            },
            RunRow {
                record: run("r2", "homelab", RunState::Succeeded),
                pod: None,
            },
        ];
        app.history_cost = Some(Cost::Known(1.5));
        assert_eq!(project_cost(&app), Cost::Partial(1.5));
        app.training.runs.remove(0);
        assert_eq!(project_cost(&app), Cost::Known(1.5));
        Ok(())
    }

    #[test]
    fn a_pod_never_created_spent_nothing() {
        let mut app = app();
        app.training.runs = vec![RunRow {
            record: run("r1", "gpu_cloud", RunState::Failed),
            pod: Some(crate::runpod::PodRecord::new(
                "r1",
                false,
                1,
                "ssh-ed25519 AAAAhost",
            )),
        }];
        assert_eq!(project_cost(&app), Cost::Unknown);
    }

    #[test]
    fn the_history_counts_only_the_stages_that_spent() {
        assert_eq!(history_cost(&[]), None);
        // `split` spends nothing: it never makes the sum partial.
        let split = entry(Stage::Split, 0, None);
        assert_eq!(history_cost(std::slice::from_ref(&split)), None);
        let answers = entry(Stage::Answers, 100, Some(0.5));
        assert_eq!(
            history_cost(&[answers.clone(), split]),
            Some(Cost::Known(0.5))
        );
        let unpriced = entry(Stage::Questions, 100, None);
        assert_eq!(
            history_cost(&[answers, unpriced.clone()]),
            Some(Cost::Partial(0.5))
        );
        assert_eq!(history_cost(&[unpriced]), Some(Cost::Unknown));
    }
}
