//! Where compares and the GGUF they serve are: the exports of a run, and the
//! compares of every run.

use std::path::Path;

use super::{CompareError, Report};
use crate::export::{EXPORT_FILE, ExportRecord};
use crate::runs::{RECORD_FILE, RunRecord, Runs, RunsError};

/// The newest export of run `run_id` whose GGUF is in its `output/gguf/`:
/// from `export.json` in the run directory (an export in the training job)
/// and in each `exports/<id>/` (an export on its own), by `created`.
#[must_use]
pub fn latest_export(runs: &Runs, run_id: &str) -> Option<ExportRecord> {
    let run_dir = runs.run_dir(run_id).ok()?;
    let mut records = Vec::new();
    records.extend(read_export(&run_dir.join(EXPORT_FILE)));
    if let Ok(exports) = runs.exports(run_id)
        && let Ok(entries) = std::fs::read_dir(exports.dir())
    {
        for entry in entries.filter_map(Result::ok) {
            records.extend(read_export(&entry.path().join(EXPORT_FILE)));
        }
    }
    records
        .into_iter()
        .filter(|record| run_dir.join(&record.file).is_file())
        .max_by(|a, b| a.created.cmp(&b.created))
}

/// The export record at `path`, when it is there and parses.
fn read_export(path: &Path) -> Option<ExportRecord> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// The newest run with a GGUF ([`latest_export`]), if any.
///
/// # Errors
///
/// Returns [`RunsError`] when the runs cannot be listed.
pub fn newest_exported_run(runs: &Runs) -> Result<Option<RunRecord>, RunsError> {
    Ok(runs
        .list()?
        .into_iter()
        .rev()
        .find(|record| latest_export(runs, &record.id).is_some()))
}

/// One compare, as listings show it.
#[derive(Debug, Clone, PartialEq)]
pub struct CompareEntry {
    /// The run compared.
    pub run: String,
    /// The compare's job record.
    pub record: RunRecord,
    /// Its report, once written.
    pub report: Option<Report>,
}

/// Every compare of every run, newest first, with the warnings of the
/// records that could not be read.
///
/// # Errors
///
/// Returns [`RunsError`] when the runs cannot be listed.
pub fn list_compares(runs: &Runs) -> Result<(Vec<CompareEntry>, Vec<String>), RunsError> {
    let mut entries = Vec::new();
    let mut warnings = Vec::new();
    for run in runs.list()? {
        let compares = runs.compares(&run.id)?;
        let listed = compares.list_with(|id, error| {
            warnings.push(format!(
                "skipping unreadable compare record {id} of run {}: {error}",
                run.id
            ));
        })?;
        for record in listed {
            let report = compares
                .run_dir(&record.id)
                .ok()
                .and_then(|dir| Report::load(&dir).ok());
            entries.push(CompareEntry {
                run: run.id.clone(),
                record,
                report,
            });
        }
    }
    entries.sort_by(|a, b| b.record.created.cmp(&a.record.created));
    Ok((entries, warnings))
}

/// The compares holding compare `id`: of run `run` when given, else of
/// whichever run has it.
///
/// # Errors
///
/// Returns [`CompareError::NotFound`] when no run has it.
pub fn find_compare(runs: &Runs, run: Option<&str>, id: &str) -> Result<Runs, CompareError> {
    let holds = |compares: &Runs| {
        compares
            .run_dir(id)
            .is_ok_and(|dir| dir.join(RECORD_FILE).is_file())
    };
    let found = match run {
        Some(run) => runs.compares(run).ok().filter(holds),
        None => runs
            .list()
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|record| runs.compares(&record.id).ok())
            .find(holds),
    };
    found.ok_or_else(|| CompareError::NotFound(id.to_string()))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::export::{EXPORT_FILE, ExportRecord, GGUF_DIR};
    use crate::runs::{RunState, Runs, create};

    /// Result type of the tests.
    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// Writes an `export.json` of `file` created at `created` into `dir`, and the file.
    fn exported(run_dir: &Path, dir: &Path, file: &str, created: &str) -> TestResult {
        std::fs::create_dir_all(run_dir.join(GGUF_DIR))?;
        std::fs::write(run_dir.join(GGUF_DIR).join(file), "gguf")?;
        std::fs::create_dir_all(dir)?;
        let record = ExportRecord {
            quantize: "Q4_K_M".into(),
            llama_cpp: "b11320".into(),
            file: format!("{GGUF_DIR}/{file}"),
            sha256: "0".repeat(64),
            size: 4,
            created: created.into(),
        };
        std::fs::write(dir.join(EXPORT_FILE), serde_json::to_string(&record)?)?;
        Ok(())
    }

    /// The newest export wins, in the run directory or an export's; a record
    /// whose GGUF is gone does not count.
    #[test]
    fn the_latest_export_of_a_run_is_found() -> TestResult {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let run_dir = runs.run_dir("r1")?;
        assert!(latest_export(&runs, "r1").is_none());
        exported(&run_dir, &run_dir, "r1-Q4_K_M.gguf", "2026-10-01T10:00:00Z")?;
        let job = runs.exports("r1")?.run_dir("export_20261002-100000")?;
        exported(&run_dir, &job, "r1-Q8_0.gguf", "2026-10-02T10:00:00Z")?;
        let latest = latest_export(&runs, "r1").ok_or("no export")?;
        assert_eq!(latest.file, "output/gguf/r1-Q8_0.gguf");
        std::fs::remove_file(run_dir.join("output/gguf/r1-Q8_0.gguf"))?;
        let latest = latest_export(&runs, "r1").ok_or("no export")?;
        assert_eq!(latest.file, "output/gguf/r1-Q4_K_M.gguf");
        Ok(())
    }

    /// Compares are listed newest first, with their report once written, and
    /// found by ID, with or without their run.
    #[test]
    fn compares_are_listed_and_found() -> TestResult {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let mut run = create(&runs, "demo", "/w", "box")?;
        run.state = RunState::Succeeded;
        runs.save(&run)?;
        let compares = runs.compares(&run.id)?;
        let first = create(&compares, crate::compare::COMPARE_PREFIX, "/w", "box")?;
        let (listed, warnings) = list_compares(&runs)?;
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].run, run.id);
        assert!(listed[0].report.is_none());
        let found = find_compare(&runs, None, &first.id)?;
        assert_eq!(found.dir(), compares.dir());
        assert_eq!(
            find_compare(&runs, Some(&run.id), &first.id)?.dir(),
            compares.dir()
        );
        assert!(matches!(
            find_compare(&runs, None, "compare_19700101-000000"),
            Err(CompareError::NotFound(_))
        ));
        Ok(())
    }
}
