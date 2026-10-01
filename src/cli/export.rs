//! `overbrainer export RUN_ID`: the model of a finished run to GGUF, with an
//! Ollama Modelfile, on the run's own target. On a local or SSH target the job
//! runs in `runs/<run-id>/exports/<export-id>/` beside the run's files; on a
//! Runpod target, on a new pod (see `runpod_train::export`).
//!
//! Ctrl-C cancels the export: unlike a training job, it is not left running.

use std::path::Path;

use anyhow::{Context, bail};

use super::ExportArgs;
use super::front::Frontend;
use super::train::{POLL, executor, prepare, secrets, warn};
use crate::config::{EnvSource, Settings, Source, Target};
use crate::exec::{Executor, JobRuntime, quote};
use crate::export::{
    Delivered, EXPORT_FILE, EXPORTS_DIR, ExportJob, ExportRecord, GGUF_DIR, Ollama, SCRIPT_FILE,
    deliver, discard_staged, latest_gguf, ollama_command, ollama_create,
};
use crate::runpod::RunpodTarget;
use crate::runs::{
    Launch, Outcome, RUNS_DIR, RunCtx, RunRecord, RunState, Runs, artifacts_missing, cancel,
    create, start_mounted, watch,
};
use crate::train::{CONFIG_FILE, OUTPUT_DIR, Trainer as _};

/// The project part of an export's ID: `export_YYYYMMDD-HHMMSS`.
pub(super) const EXPORT_PREFIX: &str = "export";

/// The Ollama command line tool.
const OLLAMA: &str = "ollama";

/// What to export, decided before anything runs.
pub(super) struct Plan<'a> {
    /// The run whose model is exported.
    pub(super) run: &'a RunRecord,
    /// The model, relative to the run directory: `output`, or a stopped
    /// run's checkpoint.
    pub(super) model: &'a str,
    /// The llama-quantize type.
    pub(super) quantize: &'a str,
    /// The Ollama model to create, if any.
    pub(super) ollama: Option<&'a str>,
}

/// Runs `overbrainer export`.
///
/// # Errors
///
/// Returns an error when the run cannot be exported, the target cannot be
/// used, or the export fails or is interrupted.
pub(super) async fn run(
    project_dir: &Path,
    args: &ExportArgs,
    front: &Frontend,
) -> anyhow::Result<()> {
    let settings = Source::from(EnvSource::Process).load(project_dir)?;
    let runs = Runs::new(project_dir);
    let record = runs.load(&args.run_id)?;
    let model = model_of(&runs, &record)?;
    let quantize = args
        .quantize
        .clone()
        .unwrap_or_else(|| settings.export.quantize.clone());
    let ollama = args
        .ollama
        .clone()
        .or_else(|| settings.export.ollama_name.clone());
    let target = settings.targets.get(&record.target).with_context(|| {
        format!(
            "target `{}` of run {} is no longer in overbrainer.toml",
            record.target, record.id
        )
    })?;
    if let Some(snapshot) = record
        .snapshot
        .as_ref()
        .filter(|_| record.state == RunState::Stopped)
    {
        warn(&format!(
            "run {} stopped at step {}: its export holds the partial model of that step",
            record.id, snapshot.step
        ));
    }
    let plan = Plan {
        run: &record,
        model: &model,
        quantize: &quantize,
        ollama: ollama.as_deref(),
    };
    if let Some(spec) = RunpodTarget::from_target(target) {
        return Box::pin(super::runpod_train::export(
            project_dir,
            &settings,
            (&spec, args.keep_pod),
            &plan,
            front,
        ))
        .await;
    }
    if args.keep_pod {
        bail!("--keep-pod only applies to a runpod target");
    }
    Box::pin(on_target(project_dir, &settings, target, &plan, front)).await
}

/// The model of `record` to export, relative to its run directory: its
/// `output/` once it succeeded; for a stopped run, `output/` when Axolotl
/// saved the partial model there, else its snapshot's checkpoint.
///
/// # Errors
///
/// Returns an error for a run neither succeeded nor stopped, one whose files
/// were not retrieved, or one without a model or `axolotl.yaml` locally.
pub(super) fn model_of(runs: &Runs, record: &RunRecord) -> anyhow::Result<String> {
    let id = &record.id;
    let dir = runs.run_dir(id)?;
    if !matches!(record.state, RunState::Succeeded | RunState::Stopped) {
        bail!(
            "run {id} is {}: only a succeeded or stopped run can be exported",
            record.state.name()
        );
    }
    if artifacts_missing(record) {
        bail!(
            "the results of run {id} were not retrieved: retrieve them with `overbrainer train \
             attach {id}`, then export"
        );
    }
    if !dir.join(CONFIG_FILE).is_file() {
        bail!("{RUNS_DIR}/{id}/{CONFIG_FILE} is missing: the export needs it");
    }
    let holds_model = |model: &str| {
        ["adapter_config.json", "config.json"]
            .iter()
            .any(|file| dir.join(model).join(file).is_file())
    };
    if holds_model(OUTPUT_DIR) {
        return Ok(OUTPUT_DIR.to_string());
    }
    match &record.snapshot {
        Some(snapshot)
            if record.state == RunState::Stopped && holds_model(&snapshot.checkpoint) =>
        {
            Ok(snapshot.checkpoint.clone())
        },
        _ => bail!("{RUNS_DIR}/{id}/{OUTPUT_DIR} holds no model: nothing to export"),
    }
}

/// The export of `plan` on the run's local or SSH target, `target`.
async fn on_target(
    project_dir: &Path,
    settings: &Settings,
    target: &Target,
    plan: &Plan<'_>,
    front: &Frontend,
) -> anyhow::Result<()> {
    let run = plan.run;
    let runtime = JobRuntime::from_target(target)
        .with_context(|| format!("target `{}` cannot run an export", run.target))?;
    // Caught from before the preparation: Ctrl-C stops it without an export.
    let mut interrupt = front.interrupt();
    let (secrets, executor) = prepare(&mut interrupt, async {
        let secrets = secrets(settings).await?;
        let executor = executor(project_dir, &run.target, target).await?;
        Ok((secrets, executor))
    })
    .await?;
    let runs = Runs::new(project_dir);
    let exports = runs.exports(&run.id)?;
    let jobs = format!("{}/{EXPORTS_DIR}", run.remote_dir);
    let record = create(&exports, EXPORT_PREFIX, &jobs, &run.target)?;
    let id = record.id.clone();
    front.line(&started_line(&id, plan));
    let script = format!(
        "{}/{EXPORTS_DIR}/{id}/{SCRIPT_FILE}",
        runtime.root(&run.remote_dir)
    );
    let job = ExportJob::in_place(&run.id, plan.quantize, plan.model, script);
    let guard = front.open_bus();
    let ctx = RunCtx {
        runs: &exports,
        executor: &executor,
        bus: &guard.bus,
        poll: POLL,
    };
    let launch = Launch {
        runtime: &runtime,
        secrets,
    };
    // Starting is never interrupted: a job spawned and not recorded could not
    // be found again.
    let started = interrupt
        .shield(async {
            let local_run = runs.run_dir(&run.id)?;
            let stage = exports.run_dir(&id)?.join(".upload");
            if let Err(error) = upload_missing(&executor, run, plan.model, &local_run, &stage).await
            {
                fail(&exports, record, &error);
                return Err(error);
            }
            Ok(start_mounted(&ctx, &job, launch, record, &run.remote_dir).await?)
        })
        .await;
    let result = match started {
        Err(error) => Err(error),
        Ok(record) => {
            let follow = interrupt.race(watch(&ctx, &job, record.clone())).await;
            if let Some(outcome) = follow {
                outcome.map_err(anyhow::Error::from)
            } else {
                interrupt
                    .shield(cancel(&exports, &executor, &job, record))
                    .await?;
                Err(anyhow::anyhow!(
                    "interrupted: export {id} of run {} cancelled",
                    run.id
                ))
            }
        },
    };
    guard.close().await;
    let delivery = Delivery {
        runs: &runs,
        exports: &exports,
        plan,
        file: &job.file_name(),
    };
    finish_export(&delivery, &result?, front)
}

/// Where an export ended and where its GGUF goes.
pub(super) struct Delivery<'a> {
    /// The project's runs.
    pub(super) runs: &'a Runs,
    /// The exports of the run, holding this one.
    pub(super) exports: &'a Runs,
    /// What was exported.
    pub(super) plan: &'a Plan<'a>,
    /// The GGUF file the export writes.
    pub(super) file: &'a str,
}

/// What the export of `plan` says when it starts.
pub(super) fn started_line(id: &str, plan: &Plan<'_>) -> String {
    format!(
        "export: {id}: {RUNS_DIR}/{}/{} to GGUF {} on target `{}`",
        plan.run.id, plan.model, plan.quantize, plan.run.target
    )
}

/// Records the export `record` failed with `error`, best effort.
fn fail(exports: &Runs, mut record: RunRecord, error: &anyhow::Error) {
    record.state = RunState::Failed;
    record.message = Some(format!("{error:#}"));
    if let Err(save_error) = exports.save(&record) {
        warn(&format!(
            "cannot record export {} as failed: {save_error}",
            record.id
        ));
    }
}

/// Uploads the model of `run` (`model`, relative to its run directory) and
/// its `axolotl.yaml` from the local run directory `local_run` into its run
/// directory on the target, when a read-only look finds them gone there; they
/// are first hard-linked into `stage`, removed afterwards. Never the
/// checkpoints but `model` itself.
async fn upload_missing<E: Executor>(
    executor: &E,
    run: &RunRecord,
    model: &str,
    local_run: &Path,
    stage: &Path,
) -> anyhow::Result<()> {
    let remote = &run.remote_dir;
    let at = |path: &str| quote(&format!("{remote}/{path}"));
    let probe = format!(
        "if [ -f {config} ] && {{ [ -f {adapter} ] || [ -f {full} ]; }}; then echo present; \
         else echo missing; fi\n",
        config = at(CONFIG_FILE),
        adapter = at(&format!("{model}/adapter_config.json")),
        full = at(&format!("{model}/config.json")),
    );
    let answer = executor.probe(&probe).await?;
    if String::from_utf8_lossy(&answer).trim() == "present" {
        return Ok(());
    }
    tracing::info!(
        "export: {remote} no longer holds the model of run {}: uploading it",
        run.id
    );
    ExportJob::staged(&run.id, "", local_run, model).prepare(stage, "")?;
    let script = stage.join(SCRIPT_FILE);
    std::fs::remove_file(&script).with_context(|| format!("cannot remove {}", script.display()))?;
    let uploaded = executor.upload(stage, remote, &[]).await;
    if let Err(error) = std::fs::remove_dir_all(stage) {
        warn(&format!("cannot remove {}: {error}", stage.display()));
    }
    Ok(uploaded?)
}

/// Reports how the export of `delivery` ended with `outcome` and, once it
/// succeeded, delivers its GGUF into the run's `output/gguf/` with its
/// Modelfile, then creates the Ollama model of its plan.
///
/// # Errors
///
/// Returns an error when the export did not succeed, or cannot be delivered.
pub(super) fn finish_export(
    delivery: &Delivery<'_>,
    outcome: &Outcome,
    front: &Frontend,
) -> anyhow::Result<()> {
    let Delivery {
        runs,
        exports,
        plan,
        file,
    } = delivery;
    let run = plan.run;
    let record = &outcome.record;
    let id = &record.id;
    let job_dir = format!("{RUNS_DIR}/{}/{EXPORTS_DIR}/{id}", run.id);
    match record.state {
        RunState::Succeeded => {
            let delivered = deliver(&runs.run_dir(&run.id)?, &exports.run_dir(id)?, file)?;
            report_delivered(&delivered, &run.id, plan.ollama, front, "export");
            Ok(())
        },
        RunState::Cancelled => {
            discard(runs, exports, &run.id, id);
            bail!("export {id} of run {} was cancelled", run.id)
        },
        _ => {
            discard(runs, exports, &run.id, id);
            let message = record
                .message
                .as_deref()
                .unwrap_or("it failed")
                .replace(&format!("{RUNS_DIR}/{id}/"), &format!("{job_dir}/"));
            bail!("export {id} of run {} failed: {message}", run.id)
        },
    }
}

/// Removes what was staged for the export `id` of run `run_id`, which did not
/// succeed; a failure is only warned about.
fn discard(runs: &Runs, exports: &Runs, run_id: &str, id: &str) {
    let (Ok(run_dir), Ok(job_dir)) = (runs.run_dir(run_id), exports.run_dir(id)) else {
        return;
    };
    if let Err(error) = discard_staged(&run_dir, &job_dir) {
        warn(&format!("cannot clean the export {id}: {error:#}"));
    }
}

/// Says where the GGUF of `delivered` and its Modelfile are, then creates the
/// Ollama model `ollama` from them, or says how to.
pub(super) fn report_delivered(
    delivered: &Delivered,
    run_id: &str,
    ollama: Option<&str>,
    front: &Frontend,
    label: &str,
) {
    let dir = format!("{RUNS_DIR}/{run_id}/{GGUF_DIR}");
    let record = &delivered.record;
    let file = record.file.rsplit('/').next().unwrap_or(&record.file);
    front.line(&format!(
        "{label}: GGUF in {dir}/{file} ({}, {}), Modelfile beside it",
        record.quantize,
        size_words(record.size)
    ));
    let gguf_dir = delivered.modelfile.parent().unwrap_or(Path::new("."));
    let Some(name) = ollama else {
        front.line(&format!(
            "{label}: run it with Ollama after `{}`",
            ollama_command(Path::new(&dir), "<name>")
        ));
        return;
    };
    match ollama_create(OLLAMA, gguf_dir, name) {
        Ollama::Created => front.line(&format!(
            "{label}: Ollama model {name} created: `ollama run {name}`"
        )),
        Ollama::NotFound(_) => warn(&format!(
            "ollama is not on PATH: create the model with `{}`",
            ollama_command(Path::new(&dir), name)
        )),
        Ollama::Failed(message) => warn(&format!(
            "ollama create {name} failed: {message}; create it with `{}`",
            ollama_command(Path::new(&dir), name)
        )),
    }
}

/// `bytes` in MB or GB, one decimal.
fn size_words(bytes: u64) -> String {
    let mb = bytes / 100_000;
    if mb >= 10_000 {
        format!("{}.{} GB", mb / 10_000, mb % 10_000 / 1_000)
    } else {
        format!("{}.{} MB", mb / 10, mb % 10)
    }
}

/// Delivers the GGUF an export in the training job of `record` wrote, once it
/// is back: `export.json` and the Modelfile, then the Ollama model `ollama`.
/// Nothing when the run has no GGUF, or its newest one was delivered already.
/// A failure is only warned about: the run itself succeeded.
pub(super) fn deliver_in_job(
    runs: &Runs,
    record: &RunRecord,
    ollama: Option<&str>,
    front: &Frontend,
) {
    let Ok(dir) = runs.run_dir(&record.id) else {
        return;
    };
    let Some(file) = latest_gguf(&dir, &record.id) else {
        return;
    };
    if delivered_already(&dir, &file) {
        return;
    }
    match deliver(&dir, &dir, &file) {
        Ok(delivered) => report_delivered(&delivered, &record.id, ollama, front, "train"),
        Err(error) => warn(&format!(
            "cannot record the export of run {}: {error:#}",
            record.id
        )),
    }
}

/// Whether `export.json` in `dir` records `file` at its current size.
fn delivered_already(dir: &Path, file: &str) -> bool {
    let Ok(text) = std::fs::read_to_string(dir.join(EXPORT_FILE)) else {
        return false;
    };
    let Ok(record) = serde_json::from_str::<ExportRecord>(&text) else {
        return false;
    };
    let path = dir.join(GGUF_DIR).join(file);
    record.file == format!("{GGUF_DIR}/{file}")
        && std::fs::metadata(path).is_ok_and(|meta| meta.len() == record.size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runs::{Snapshot, SnapshotReason};

    fn record(state: RunState) -> RunRecord {
        RunRecord {
            id: "r1".into(),
            target: "box".into(),
            created: "2026-10-01T12:00:00Z".into(),
            remote_dir: "/w/r1".into(),
            job: None,
            state,
            message: None,
            snapshot: None,
            resumed_from: None,
            snapshots: true,
        }
    }

    #[test]
    fn only_a_finished_run_with_a_model_is_exported() -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let dir = runs.run_dir("r1")?;
        std::fs::create_dir_all(dir.join("output/checkpoint-40"))?;
        let refused = |state, why: &str| -> Result<(), Box<dyn std::error::Error>> {
            let error = model_of(&runs, &record(state)).err().ok_or("accepted")?;
            assert!(error.to_string().contains(why), "{error}");
            Ok(())
        };
        refused(
            RunState::Running,
            "run r1 is running: only a succeeded or stopped run",
        )?;
        refused(RunState::Failed, "is failed")?;
        refused(RunState::Succeeded, "axolotl.yaml is missing")?;
        std::fs::write(dir.join("axolotl.yaml"), "base_model: m\n")?;
        refused(RunState::Succeeded, "runs/r1/output holds no model")?;
        let mut missing = record(RunState::Succeeded);
        missing.message = Some("artifacts not retrieved: download failed".into());
        let error = model_of(&runs, &missing).err().ok_or("accepted")?;
        assert!(error.to_string().contains("train attach r1"), "{error}");

        let mut stopped = record(RunState::Stopped);
        stopped.snapshot = Some(Snapshot {
            checkpoint: "output/checkpoint-40".into(),
            step: 40,
            reason: SnapshotReason::Requested,
        });
        std::fs::write(dir.join("output/checkpoint-40/adapter_config.json"), "{}")?;
        assert_eq!(model_of(&runs, &stopped)?, "output/checkpoint-40");
        std::fs::write(dir.join("output/adapter_config.json"), "{}")?;
        assert_eq!(model_of(&runs, &stopped)?, "output");
        assert_eq!(model_of(&runs, &record(RunState::Succeeded))?, "output");
        Ok(())
    }

    #[tokio::test]
    async fn the_model_is_uploaded_only_where_it_is_gone() -> Result<(), Box<dyn std::error::Error>>
    {
        let local = tempfile::tempdir()?;
        let target = tempfile::tempdir()?;
        let executor = crate::exec::LocalExecutor::new(target.path())?;
        std::fs::create_dir_all(local.path().join("output/checkpoint-10"))?;
        std::fs::write(local.path().join("output/adapter_config.json"), "{}")?;
        std::fs::write(
            local
                .path()
                .join("output/checkpoint-10/adapter_config.json"),
            "{}",
        )?;
        std::fs::write(local.path().join("axolotl.yaml"), "base_model: m\n")?;
        let remote = target.path().join("r1");
        let mut run = record(RunState::Succeeded);
        run.remote_dir = remote.to_string_lossy().into_owned();
        let stage = local.path().join("exports/e1/.upload");
        upload_missing(&executor, &run, "output", local.path(), &stage).await?;
        assert!(remote.join("output/adapter_config.json").is_file());
        assert!(remote.join("axolotl.yaml").is_file());
        assert!(
            !remote.join("output/checkpoint-10").exists(),
            "no checkpoint"
        );
        assert!(!remote.join(SCRIPT_FILE).exists());
        assert!(!stage.exists(), "the stage is removed");
        // Present now: nothing is uploaded again.
        std::fs::remove_file(local.path().join("axolotl.yaml"))?;
        upload_missing(&executor, &run, "output", local.path(), &stage).await?;
        Ok(())
    }

    #[test]
    fn sizes_read_in_mb_or_gb() {
        assert_eq!(size_words(412_345_678), "412.3 MB");
        assert_eq!(size_words(1_234_567_890), "1.2 GB");
        assert_eq!(size_words(0), "0.0 MB");
    }

    #[test]
    fn an_in_job_export_is_delivered_once() -> Result<(), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let dir = runs.run_dir("r1")?;
        std::fs::create_dir_all(dir.join(GGUF_DIR))?;
        let front = Frontend::Cli(None);
        deliver_in_job(&runs, &record(RunState::Succeeded), None, &front);
        assert!(
            !dir.join(EXPORT_FILE).exists(),
            "no GGUF, nothing to deliver"
        );
        std::fs::write(dir.join(GGUF_DIR).join("r1-Q4_K_M.gguf"), "gguf")?;
        deliver_in_job(&runs, &record(RunState::Succeeded), None, &front);
        assert!(delivered_already(&dir, "r1-Q4_K_M.gguf"));
        assert!(dir.join(GGUF_DIR).join("Modelfile").is_file());
        std::fs::write(dir.join(GGUF_DIR).join("r1-Q4_K_M.gguf"), "longer gguf")?;
        assert!(!delivered_already(&dir, "r1-Q4_K_M.gguf"));
        Ok(())
    }
}
