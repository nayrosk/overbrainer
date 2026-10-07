//! `overbrainer compare`: the child against the parent on the eval set. The
//! job runs on the run's own target, as an export does: on a local or SSH
//! target in `runs/<run-id>/compares/<compare-id>/`, serving the run's GGUF
//! where the run's files are; on a Runpod target on a new pod, the GGUF
//! uploaded with the job (see `runpod_train::compare`). The judge then runs
//! here.
//!
//! Ctrl-C cancels the job; while the judge runs, it stops it and keeps the
//! verdicts so far.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, bail};

use super::CompareArgs;
use super::export::{fail, holds_on_target, upload_stage};
use super::front::Frontend;
use super::train::{POLL, executor, prepare};
use crate::compare::{
    COMPARE_PREFIX, COMPARES_DIR, ChildSettings, CompareJob, CompareSetup, EvalQuestion, JudgeInfo,
    JudgeRun, ModelSource, Parts, SCRIPT_FILE, build_report, find_compare, judge_all, judge_key,
    latest_export, newest_exported_run, prices, read_answers, read_eval, read_hardware,
    verdicts_file, write_report,
};
use crate::config::{Settings, Source, Target};
use crate::dataset::DataFiles;
use crate::exec::{Executor, JobRuntime, quote};
use crate::export::{ExportRecord, link_file};
use crate::llm::RetryPolicy;
use crate::prompts::{JUDGE, Prompts};
use crate::runpod::{GpuOffer, PodRecord, RunpodClient, RunpodTarget, cheapest_in_stock};
use crate::runs::{
    Launch, Outcome, RunCtx, RunRecord, RunState, Runs, cancel, create, rfc3339, start_mounted,
    watch,
};
use crate::train::CONFIG_FILE;

/// What a compare compares, decided before anything runs.
#[derive(Debug, Clone)]
pub(crate) struct Plan {
    /// The run.
    pub(crate) run: RunRecord,
    /// Its GGUF, locally.
    pub(crate) gguf: PathBuf,
    /// What its `export.json` records of it.
    pub(crate) export: ExportRecord,
    /// The questions.
    pub(crate) questions: Vec<EvalQuestion>,
    /// The server's context size: the run's `sequence_len`, 0 when unknown.
    pub(crate) context: u32,
    /// The run's base model, when its `axolotl.yaml` says.
    pub(crate) base_model: Option<String>,
}

/// The plan of a compare of run `run` (else the newest run with a GGUF) on
/// the first `limit` questions of `data/eval.jsonl`.
///
/// # Errors
///
/// Returns an error when there is no such run, it has no GGUF, or the eval
/// set has no question.
pub(crate) fn plan(
    runs: &Runs,
    project_dir: &Path,
    run: Option<&str>,
    limit: Option<u32>,
) -> anyhow::Result<Plan> {
    let record = match run {
        Some(id) => runs.load(id)?,
        None => newest_exported_run(runs)?.context(
            "no run has a GGUF: export one with `overbrainer export <run-id>`, then compare",
        )?,
    };
    let export = latest_export(runs, &record.id).with_context(|| {
        format!(
            "run {id} has no GGUF; run `overbrainer export {id}` first",
            id = record.id
        )
    })?;
    let run_dir = runs.run_dir(&record.id)?;
    let limit = limit.map(|limit| usize::try_from(limit).unwrap_or(usize::MAX));
    let questions = read_eval(&DataFiles::new(project_dir).eval, limit)?;
    let config = run_dir.join(CONFIG_FILE);
    Ok(Plan {
        gguf: run_dir.join(&export.file),
        context: crate::export::sequence_len(&config).unwrap_or(0),
        base_model: std::fs::read_to_string(&config)
            .ok()
            .and_then(|text| crate::train::top_level_scalar(&text, "base_model")),
        run: record,
        export,
        questions,
    })
}

/// The setup of compare `compare` of `plan`, its order drawn from `seed`.
pub(super) fn setup_of(plan: &Plan, compare: &str, seed: u64) -> CompareSetup {
    CompareSetup {
        run: plan.run.id.clone(),
        compare: compare.to_string(),
        gguf: plan.export.file.clone(),
        gguf_sha256: plan.export.sha256.clone(),
        quantize: plan.export.quantize.clone(),
        llama_cpp: crate::export::LLAMA_CPP_TAG.to_string(),
        base_model: plan.base_model.clone(),
        seed,
        created: rfc3339(SystemTime::now()),
        questions: plan.questions.clone(),
    }
}

/// The compare job of `plan` serving the GGUF from `model`, with the
/// settings of `[compare]`.
///
/// # Errors
///
/// Returns an error when `model` is a GGUF on the target not named by an
/// absolute path.
pub(super) fn job_of(
    plan: &Plan,
    settings: &Settings,
    model: ModelSource,
) -> anyhow::Result<CompareJob> {
    Ok(CompareJob::new(
        model,
        plan.questions.clone(),
        ChildSettings::from_config(&settings.compare, plan.context),
    )?)
}

/// What the pod of a Runpod compare costs an hour, before it starts.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PodPrice {
    /// The cheapest GPU type in stock it may get, as `overbrainer pod gpus`
    /// prices it, for the target's GPU count.
    InStock(GpuOffer),
    /// The catalog shows none in stock with a price: the target's
    /// `max_price_per_hour`, a cap.
    Cap(f64),
    /// Neither a price in stock nor a cap.
    Unknown,
}

impl PodPrice {
    /// What the price is, as a sentence ending a line.
    pub(crate) fn words(&self) -> String {
        match self {
            Self::InStock(offer) => format!(
                "at about ${:.2}/h ({}, the cheapest GPU type in stock)",
                offer.per_hour, offer.gpu_type
            ),
            Self::Cap(cap) => format!(
                "at most ${cap:.2}/h (max_price_per_hour: the catalog shows no price in stock)"
            ),
            Self::Unknown => "at a price the catalog does not show (no GPU type in stock with \
                              a price, and no max_price_per_hour)"
                .to_string(),
        }
    }
}

/// The price of a pod of `spec` for a compare, its `auto` GPU types kept to
/// `floor_gb` of VRAM: the cheapest in stock in the catalog `client` lists,
/// else the target's cap. A catalog that cannot be read is only logged. The
/// command line says it before the pod starts; the TUI's confirmation can
/// too, with the client of [`super::pod::client`] and the floor of
/// [`compare_vram_floor`].
pub(crate) async fn pod_price(
    client: &RunpodClient,
    spec: &RunpodTarget,
    floor_gb: Option<u32>,
) -> PodPrice {
    match client.list_gpu_types(spec.gpu_count).await {
        Ok(gpus) => {
            if let Some(offer) = cheapest_in_stock(spec, &gpus, floor_gb) {
                return PodPrice::InStock(offer);
            }
        },
        Err(error) => tracing::debug!("cannot read the GPU catalog for the price: {error}"),
    }
    spec.max_price_per_hour
        .map_or(PodPrice::Unknown, PodPrice::Cap)
}

/// The VRAM floor of the `auto` GPU types of `spec` for a compare of a GGUF
/// of `bytes`: the GGUF and its cache must fit; none when `spec` lists its
/// GPU types or sets `min_vram_gb`.
pub(crate) fn compare_vram_floor(spec: &RunpodTarget, bytes: u64) -> Option<u32> {
    (spec.gpu_types.is_auto() && spec.min_vram_gb.is_none())
        .then(|| super::runpod_train::gguf_vram_floor(bytes))
}

/// What a compare started now would do, for the TUI's confirmation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Preview {
    /// The run.
    pub(crate) run: String,
    /// The GGUF, relative to the run directory.
    pub(crate) gguf: String,
    /// Questions asked.
    pub(crate) questions: usize,
    /// The target's name.
    pub(crate) target: String,
    /// On a Runpod target, the pod and what it costs an hour, as a sentence.
    pub(crate) runpod: Option<String>,
}

/// What a compare of run `run` (else the newest with a GGUF) would do, with
/// the settings of `source`. The local files are read off the async
/// threads; on a Runpod target the GPU catalog prices the pod.
///
/// # Errors
///
/// As [`plan`], and when the settings cannot be loaded or the run's target
/// is gone from them.
pub(crate) async fn preview(
    project_dir: &Path,
    source: &Source,
    run: Option<&str>,
) -> anyhow::Result<Preview> {
    let (dir, source, run) = (
        project_dir.to_path_buf(),
        source.clone(),
        run.map(str::to_string),
    );
    let (settings, plan) = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let settings = source.load(&dir)?;
        let plan = plan(&Runs::new(&dir), &dir, run.as_deref(), None)?;
        Ok((settings, plan))
    })
    .await
    .context("cannot read the run to compare")??;
    let target = target_of(&settings, &plan.run)?;
    let runpod = match RunpodTarget::from_target(target) {
        Some(spec) => {
            let client = super::pod::client(&settings).await;
            Some(runpod_line(client, &spec, plan.export.size).await)
        },
        None => None,
    };
    Ok(Preview {
        run: plan.run.id.clone(),
        gguf: plan.export.file.clone(),
        questions: plan.questions.len(),
        target: plan.run.target.clone(),
        runpod,
    })
}

/// The preview's sentence on the pod of a compare on `spec`, of a GGUF of
/// `gguf_bytes`: the cheapest GPU type in stock in the catalog of `client`;
/// without one (no client, the catalog unread, none in stock with a price),
/// the target's `max_price_per_hour`, else that the price is picked when the
/// pod starts.
async fn runpod_line(
    client: anyhow::Result<RunpodClient>,
    spec: &RunpodTarget,
    gguf_bytes: u64,
) -> String {
    let price = match client {
        Ok(client) => pod_price(&client, spec, compare_vram_floor(spec, gguf_bytes)).await,
        Err(error) => {
            tracing::debug!("cannot price the compare's pod: {error:#}");
            spec.max_price_per_hour
                .map_or(PodPrice::Unknown, PodPrice::Cap)
        },
    };
    match price {
        PodPrice::InStock(_) => format!("a new Runpod pod, {}", price.words()),
        PodPrice::Cap(cap) => format!("a new Runpod pod, at most ${cap:.2}/h"),
        PodPrice::Unknown => "a new Runpod pod, at the price picked when it starts".to_string(),
    }
}

/// The target of `run` in `settings`.
fn target_of<'a>(settings: &'a Settings, run: &RunRecord) -> anyhow::Result<&'a Target> {
    settings.targets.get(&run.target).with_context(|| {
        format!(
            "target `{}` of run {} is no longer in overbrainer.toml",
            run.target, run.id
        )
    })
}

/// Runs `overbrainer compare` with the settings of `source`, for `front`.
///
/// # Errors
///
/// Returns an error when the run cannot be compared, the target cannot be
/// used, the job or the judge fails, or it is interrupted.
pub(crate) async fn run(
    project_dir: &Path,
    args: &CompareArgs,
    front: &Frontend,
    source: &Source,
) -> anyhow::Result<()> {
    let settings = source.load(project_dir)?;
    let runs = Runs::new(project_dir);
    if let Some(id) = &args.rejudge {
        let compares = find_compare(&runs, args.run.as_deref(), id)?;
        let judging = Judging {
            project_dir,
            settings: &settings,
            compares: &compares,
            id,
            front,
        };
        let report = judge_and_report(&judging).await?;
        front.line(&report.to_string_lossy());
        return Ok(());
    }
    let plan = plan(&runs, project_dir, args.run.as_deref(), args.limit)?;
    let target = target_of(&settings, &plan.run)?;
    let spec = RunpodTarget::from_target(target);
    if spec.is_none() && args.keep_pod {
        bail!("--keep-pod only applies to a runpod target");
    }
    tracing::info!(
        "compare: run {} ({}) on {} questions, on target `{}`",
        plan.run.id,
        plan.export.quantize,
        plan.questions.len(),
        plan.run.target
    );
    let (compares, id) = if let Some(spec) = spec {
        Box::pin(super::runpod_train::compare(
            project_dir,
            &settings,
            (&spec, args.keep_pod),
            &plan,
            front,
        ))
        .await?
    } else {
        Box::pin(on_target(project_dir, &settings, target, &plan, front)).await?
    };
    let judging = Judging {
        project_dir,
        settings: &settings,
        compares: &compares,
        id: &id,
        front,
    };
    let report = judge_and_report(&judging).await?;
    front.line(&report.to_string_lossy());
    Ok(())
}

/// The compare job of `plan` on the run's local or SSH target `target`,
/// serving the run's GGUF where the run's files are on the target (uploaded
/// there first when missing or different); returns the compares of the run
/// and the compare's ID once its answers are back.
async fn on_target(
    project_dir: &Path,
    settings: &Settings,
    target: &Target,
    plan: &Plan,
    front: &Frontend,
) -> anyhow::Result<(Runs, String)> {
    let run = &plan.run;
    let runtime = JobRuntime::from_target(target)
        .with_context(|| format!("target `{}` cannot run a compare", run.target))?;
    // The GGUF as the job sees it: the job's container mounts the run directory.
    let model = format!("{}/{}", runtime.root(&run.remote_dir), plan.export.file);
    let job = job_of(plan, settings, ModelSource::OnTarget(model))?;
    // Caught from before the preparation: Ctrl-C stops it without a compare.
    let mut interrupt = front.interrupt();
    let executor = prepare(&mut interrupt, executor(project_dir, &run.target, target)).await?;
    let runs = Runs::new(project_dir);
    let compares = runs.compares(&run.id)?;
    let jobs = format!("{}/{COMPARES_DIR}", run.remote_dir);
    let record = create(&compares, COMPARE_PREFIX, &jobs, &run.target)?;
    let id = record.id.clone();
    // The script as the job sees it: a container starts in the run directory.
    let job = job.with_script(format!(
        "{}/{COMPARES_DIR}/{id}/{SCRIPT_FILE}",
        runtime.root(&run.remote_dir)
    ));
    let job_dir = compares.run_dir(&id)?;
    tracing::info!("compare: {id}: {}", compares.relative_dir(&id));
    let guard = front.open_bus();
    let ctx = RunCtx {
        runs: &compares,
        executor: &executor,
        bus: &guard.bus,
        poll: POLL,
    };
    let launch = Launch {
        runtime: &runtime,
        secrets: Vec::new(),
    };
    // Starting is never interrupted: a job spawned and not recorded could not
    // be found again.
    let started = interrupt
        .shield(async {
            setup_of(plan, &id, settings.pipeline.seed).save(&job_dir)?;
            let stage = job_dir.join(".upload");
            gguf_on_target(&executor, plan, &runs.run_dir(&run.id)?, &stage).await?;
            Ok(start_mounted(&ctx, &job, launch, record, &run.remote_dir).await?)
        })
        .await
        .inspect_err(|error| fail(&compares, &id, error));
    let result = match started {
        Err(error) => Err(error),
        Ok(record) => {
            if let Some(outcome) = interrupt.race(watch(&ctx, &job, record.clone())).await {
                outcome.map_err(|error| {
                    anyhow::Error::from(error).context(format!(
                        "compare {id} of run {} may still run on target `{}`; {}",
                        run.id,
                        run.target,
                        compares.follow_hint(&id)
                    ))
                })
            } else {
                interrupt
                    .shield(cancel(&compares, &executor, &job, record))
                    .await?;
                Err(anyhow::anyhow!(
                    "interrupted: compare {id} of run {} cancelled",
                    run.id
                ))
            }
        },
    };
    guard.close().await;
    generated(&compares, &result?)?;
    Ok((compares, id))
}

/// Makes sure the run directory of `plan` on the target of `executor` holds
/// the run's GGUF as exported (the SHA-256 its `export.json` records):
/// uploads it from the local run directory `local_run` when it is missing or
/// different there, hard-linked into `stage` first, removed afterwards.
///
/// # Errors
///
/// Returns an error when the target cannot be reached, or the GGUF cannot be
/// staged or uploaded.
async fn gguf_on_target<E: Executor>(
    executor: &E,
    plan: &Plan,
    local_run: &Path,
    stage: &Path,
) -> anyhow::Result<()> {
    let remote = &plan.run.remote_dir;
    let file = &plan.export.file;
    let path = quote(&format!("{remote}/{file}"));
    if holds_on_target(executor, &format!("[ -f {path} ]")).await? {
        let digests = executor
            .manifest(remote, std::slice::from_ref(file), &[])
            .await?;
        if digests
            .iter()
            .any(|digest| digest.path == *file && digest.sha256 == plan.export.sha256)
        {
            return Ok(());
        }
    }
    tracing::info!(
        "compare: {remote} does not hold the GGUF of run {} as exported: uploading it",
        plan.run.id
    );
    link_file(&local_run.join(file), &stage.join(file))?;
    upload_stage(executor, stage, remote).await
}

/// Says how the job of a compare ended, as its `outcome` records it: the
/// answers are back once it succeeded.
///
/// # Errors
///
/// Returns an error when the job was cancelled or failed.
pub(super) fn generated(compares: &Runs, outcome: &Outcome) -> anyhow::Result<()> {
    let record = &outcome.record;
    let subject = compares.subject(&record.id);
    match record.state {
        RunState::Succeeded => Ok(()),
        RunState::Cancelled => bail!("{subject} was cancelled"),
        _ => {
            let message = record.message.as_deref().unwrap_or("it failed");
            bail!(
                "{subject} failed: {message}; its server log is {}/{}",
                compares.relative_dir(&record.id),
                crate::compare::SERVER_LOG
            )
        },
    }
}

/// What the judge of a compare needs.
pub(super) struct Judging<'a> {
    /// The project.
    pub(super) project_dir: &'a Path,
    /// Its settings: the judge, the prices, the concurrency.
    pub(super) settings: &'a Settings,
    /// The compares holding it.
    pub(super) compares: &'a Runs,
    /// The compare.
    pub(super) id: &'a str,
    /// The front end: progress, interruption.
    pub(super) front: &'a Frontend,
}

/// Judges compare `judging.id` (resuming its judge's verdicts file), then
/// writes its report from the verdicts of the questions it compared; returns
/// the path of `compare.md`.
///
/// # Errors
///
/// Returns an error when its files cannot be read, the judge cannot be
/// reached or fails, the report cannot be written, or it is interrupted.
pub(super) async fn judge_and_report(judging: &Judging<'_>) -> anyhow::Result<PathBuf> {
    let Judging {
        project_dir,
        settings,
        compares,
        id,
        front,
    } = judging;
    let dir = compares.run_dir(id)?;
    let setup = CompareSetup::load(&dir)?;
    let answers = read_answers(&dir)?;
    if answers.is_empty() {
        bail!(
            "compare {id} has no answers of the child: run `overbrainer compare --run {}` again",
            setup.run
        );
    }
    let prompts = Prompts::load(project_dir)?;
    let role = settings.roles.judge_model();
    let key = judge_key(role, &prompts.source(JUDGE).unwrap_or_default());
    let file_name = verdicts_file(&key);
    let file = dir.join(&file_name);
    let client = crate::llm::connect(settings, role, &super::resolver()).await?;
    let guard = front.open_bus();
    let run = JudgeRun {
        client: &client,
        role,
        prompts: &prompts,
        policy: RetryPolicy::new(settings.pipeline.max_retries),
        concurrency: settings.pipeline.concurrency,
        bus: &guard.bus,
        compare: id,
    };
    tracing::info!(
        "compare: {id}: judging with {}/{}",
        role.provider,
        role.model
    );
    let mut interrupt = front.interrupt();
    let judged = interrupt
        .race(judge_all(&run, &setup, &answers, &file))
        .await;
    guard.close().await;
    let Some(verdicts) = judged else {
        bail!(
            "interrupted: the verdicts so far are kept; resume with `overbrainer compare \
             --rejudge {id}`"
        );
    };
    let verdicts = verdicts.with_context(|| {
        format!("the verdicts so far are kept; resume with `overbrainer compare --rejudge {id}`")
    })?;
    let pod_price = PodRecord::load(compares, id)?.and_then(|pod| pod.cost_per_hour);
    let (prices, child_price_from_pod) = prices(&settings.compare, pod_price);
    let report = build_report(&Parts {
        setup: &setup,
        answers: &answers,
        verdicts: &verdicts,
        hardware: read_hardware(&dir),
        judge: JudgeInfo {
            role,
            is_parent: settings.roles.judge.is_none(),
            verdicts_file: &file_name,
        },
        prices,
        child_price_from_pod,
    });
    Ok(write_report(&dir, &report)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ListOrAuto;

    /// A Runpod target of `gpu_types` capped at `cap`.
    fn spec(gpu_types: ListOrAuto, cap: Option<f64>) -> RunpodTarget {
        RunpodTarget {
            gpu_types,
            min_vram_gb: None,
            max_price_per_hour: cap,
            gpu_count: 1,
            image: "img".into(),
            venv: Some("/venv".into()),
            container_disk_gb: 50,
            max_hours: 1.0,
            max_cost_usd: None,
            boot_grace: std::time::Duration::from_mins(30),
            retrieve_grace: std::time::Duration::from_secs(3600),
            data_center_ids: ListOrAuto::default(),
            network_volume_id: None,
            max_volume_gb: None,
            ssh_client: crate::config::SshClient::Openssh,
        }
    }

    /// The GGUF of run `r1`, relative to its run directory.
    const GGUF: &str = "output/gguf/r1-Q4_K_M.gguf";

    /// The plan of a compare of run `r1`, its remote directory `remote_dir`,
    /// whose GGUF [`GGUF`] has the SHA-256 `sha256`.
    fn plan_of(remote_dir: &Path, sha256: String) -> Plan {
        Plan {
            run: RunRecord {
                id: "r1".into(),
                target: "box".into(),
                created: "2026-10-06T10:00:00Z".into(),
                remote_dir: remote_dir.to_string_lossy().into_owned(),
                job: None,
                state: RunState::Succeeded,
                message: None,
                snapshot: None,
                resumed_from: None,
                snapshots: true,
            },
            gguf: PathBuf::from(GGUF),
            export: ExportRecord {
                quantize: "Q4_K_M".into(),
                llama_cpp: crate::export::LLAMA_CPP_TAG.into(),
                file: GGUF.into(),
                sha256,
                size: 4,
                created: "2026-10-06T11:00:00Z".into(),
            },
            questions: Vec::new(),
            context: 0,
            base_model: None,
        }
    }

    /// The GGUF is uploaded only where it is missing or different, then the
    /// stage is removed.
    #[tokio::test]
    async fn the_gguf_is_uploaded_only_where_it_is_missing_or_different()
    -> Result<(), Box<dyn std::error::Error>> {
        let local = tempfile::tempdir()?;
        let target = tempfile::tempdir()?;
        let executor = crate::exec::LocalExecutor::new(target.path())?;
        let remote = target.path().join("r1");
        let gguf = local.path().join(GGUF);
        std::fs::create_dir_all(gguf.parent().ok_or("no gguf dir")?)?;
        std::fs::write(&gguf, "gguf")?;
        let plan = plan_of(&remote, crate::exec::sha256_file(&gguf)?);
        let stage = local.path().join("compares/c1/.upload");
        gguf_on_target(&executor, &plan, local.path(), &stage).await?;
        let uploaded = remote.join(&plan.export.file);
        assert_eq!(std::fs::read_to_string(&uploaded)?, "gguf");
        assert!(!stage.exists(), "the stage is removed");
        // Different on the target: uploaded again.
        std::fs::write(&uploaded, "stale")?;
        gguf_on_target(&executor, &plan, local.path(), &stage).await?;
        assert_eq!(std::fs::read_to_string(&uploaded)?, "gguf");
        // The same: nothing is staged, so no local GGUF is needed.
        std::fs::remove_file(&gguf)?;
        gguf_on_target(&executor, &plan, local.path(), &stage).await?;
        assert!(!stage.exists());
        Ok(())
    }

    /// The price of a compare's pod is the cheapest GPU type in stock, else
    /// the cap, else unknown; each says so.
    #[tokio::test]
    async fn the_pod_price_is_the_cheapest_in_stock_else_the_cap()
    -> Result<(), Box<dyn std::error::Error>> {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v2/catalog/gpus"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"gpus": [
                    {"id": "NVIDIA A40", "memory": 48, "secure": true,
                     "price": {"secure": 0.4}, "maxCount": {"secure": 8},
                     "availability": "HIGH"},
                    {"id": "NVIDIA H100", "memory": 80, "secure": true,
                     "price": {"secure": 2.5}, "maxCount": {"secure": 8},
                     "availability": "NONE"}
                ]})),
            )
            .mount(&server)
            .await;
        let client = RunpodClient::new(
            &format!("{}/v2", server.uri()),
            &secrecy::SecretString::from("k"),
        )?;
        let auto = pod_price(&client, &spec(ListOrAuto::Auto, None), None).await;
        assert_eq!(
            auto.words(),
            "at about $0.40/h (NVIDIA A40, the cheapest GPU type in stock)"
        );
        let sold_out = spec(ListOrAuto::List(vec!["NVIDIA H100".into()]), Some(3.0));
        let capped = pod_price(&client, &sold_out, None).await;
        assert_eq!(capped, PodPrice::Cap(3.0));
        assert!(
            capped.words().starts_with("at most $3.00/h"),
            "{}",
            capped.words()
        );
        let unknown = spec(ListOrAuto::List(vec!["NVIDIA H100".into()]), None);
        assert_eq!(pod_price(&client, &unknown, None).await, PodPrice::Unknown);
        let line = runpod_line(Ok(client), &spec(ListOrAuto::Auto, None), 4).await;
        assert_eq!(
            line,
            "a new Runpod pod, at about $0.40/h (NVIDIA A40, the cheapest GPU type in stock)"
        );
        Ok(())
    }

    /// Without a price in stock, because the catalog cannot be read or no
    /// client can be made, the preview says the cap, else that the price is
    /// picked when the pod starts.
    #[tokio::test]
    async fn the_preview_falls_back_to_the_cap() -> Result<(), Box<dyn std::error::Error>> {
        use wiremock::matchers::method;
        use wiremock::{Mock, MockServer, ResponseTemplate};
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let client = RunpodClient::new(
            &format!("{}/v2", server.uri()),
            &secrecy::SecretString::from("k"),
        )?;
        let capped = spec(ListOrAuto::Auto, Some(0.5));
        assert_eq!(
            runpod_line(Ok(client), &capped, 4).await,
            "a new Runpod pod, at most $0.50/h"
        );
        let no_client = || Err(anyhow::anyhow!("no Runpod API key"));
        assert_eq!(
            runpod_line(no_client(), &capped, 4).await,
            "a new Runpod pod, at most $0.50/h"
        );
        assert_eq!(
            runpod_line(no_client(), &spec(ListOrAuto::Auto, None), 4).await,
            "a new Runpod pod, at the price picked when it starts"
        );
        Ok(())
    }
}
