//! What `t` shows before a training run starts: the target, the model, the data,
//! and for Runpod the VRAM the run needs per GPU (estimated from the model's
//! shape on Hugging Face), the GPU types (configured, chosen with `g`, or what
//! `auto` picks now) with their list price, VRAM, stock and fit, the data
//! centers, and the most `max_hours` can cost.

use std::path::Path;
use std::time::Duration;

use serde::de::IgnoredAny;

use crate::config::{Adapter, CONFIG_FILE, ListOrAuto, Runtime, Settings, Source, Target};
use crate::dataset::{DataFiles, read};
use crate::runpod::{
    Availability, GpuType, ResolveError, RunpodTarget, resolve_with_floor, short_cap_warning,
};
use crate::runs::Runs;
use crate::train::sizing::{Estimate, Fit, HF_URL, estimate_model, fit};
use crate::train::{Outputs, reasoning_template_warning};

/// Total time the GPU catalog may take; the dialog never waits for it.
pub(super) const START_CATALOG_TIMEOUT: Duration = Duration::from_secs(10);

/// What a training run started now would use.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct StartPlan {
    /// `training.target`.
    pub(super) target: String,
    /// What kind of target it is, for example `runpod, Secure Cloud`.
    pub(super) kind: String,
    /// Base model, adapter, epochs and learning rate.
    pub(super) model: String,
    /// Examples in `data/train.jsonl`.
    pub(super) train: usize,
    /// Examples in `data/eval.jsonl`.
    pub(super) eval: usize,
    /// The pod of a Runpod target.
    pub(super) runpod: Option<Box<RunpodPlan>>,
    /// What the flow would warn about.
    pub(super) warnings: Vec<String>,
    /// The stopped run it resumes from, if any.
    pub(super) resume: Option<ResumePlan>,
}

/// The stopped run a new run resumes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ResumePlan {
    /// The stopped run.
    pub(super) run_id: String,
    /// The step of its snapshot.
    pub(super) step: u64,
    /// Its checkpoint, relative to its run directory.
    pub(super) checkpoint: String,
}

/// The pod a Runpod run would ask for.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct RunpodPlan {
    /// The target as the run would use it, with the GPU types and data
    /// centers chosen in the dialog.
    pub(super) spec: RunpodTarget,
    /// `gpu_types` as the settings have it.
    pub(super) file_gpu_types: ListOrAuto,
    /// `data_center_ids` as the settings have it.
    pub(super) file_data_center_ids: ListOrAuto,
    /// `min_vram_gb` as the settings have it.
    pub(super) file_min_vram_gb: Option<u32>,
    /// `max_price_per_hour` as the settings have it.
    pub(super) file_max_price_per_hour: Option<f64>,
    /// The VRAM the run needs per GPU, once the start dialog estimated it.
    pub(super) need: Option<Need>,
}

impl RunpodPlan {
    /// The plan of `spec`, nothing chosen yet.
    pub(super) fn new(spec: RunpodTarget) -> Self {
        Self {
            file_gpu_types: spec.gpu_types.clone(),
            file_data_center_ids: spec.data_center_ids.clone(),
            file_min_vram_gb: spec.min_vram_gb,
            file_max_price_per_hour: spec.max_price_per_hour,
            need: None,
            spec,
        }
    }

    /// The fields chosen in the dialog that differ from the settings, which
    /// `y` saves before the run starts: the `auto` limits among them are
    /// unset.
    pub(super) fn changed(&self) -> Vec<&'static str> {
        let mut changed = Vec::new();
        if self.spec.gpu_types != self.file_gpu_types {
            changed.push(GPU_TYPES);
        }
        if self.spec.data_center_ids != self.file_data_center_ids {
            changed.push(DATA_CENTER_IDS);
        }
        changed.extend(self.removed());
        changed
    }

    /// The `auto` limits the settings have and the plan dropped.
    fn removed(&self) -> Vec<&'static str> {
        let [vram, price] = AUTO_LIMITS;
        let mut removed = Vec::new();
        if self.file_min_vram_gb.is_some() && self.spec.min_vram_gb.is_none() {
            removed.push(vram);
        }
        if self.file_max_price_per_hour.is_some() && self.spec.max_price_per_hour.is_none() {
            removed.push(price);
        }
        removed
    }

    /// The GPU types chosen: listed ones drop the `auto` limits, `auto` gets
    /// those of the settings back.
    pub(super) fn choose_gpus(&mut self, gpu_types: ListOrAuto) {
        if gpu_types.is_auto() {
            self.spec.min_vram_gb = self.file_min_vram_gb;
            self.spec.max_price_per_hour = self.file_max_price_per_hour;
        } else {
            self.spec.min_vram_gb = None;
            self.spec.max_price_per_hour = None;
        }
        self.spec.gpu_types = gpu_types;
    }

    /// What the dialog's `changed` line lists: `gpu_types (min_vram_gb and
    /// max_price_per_hour removed), data_center_ids`; none when nothing
    /// changed.
    fn changed_text(&self) -> Option<String> {
        let mut shown = Vec::new();
        if self.spec.gpu_types != self.file_gpu_types {
            let removed = self.removed();
            shown.push(if removed.is_empty() {
                GPU_TYPES.to_string()
            } else {
                format!("{GPU_TYPES} ({} removed)", removed.join(" and "))
            });
        }
        if self.spec.data_center_ids != self.file_data_center_ids {
            shown.push(DATA_CENTER_IDS.to_string());
        }
        (!shown.is_empty()).then(|| shown.join(", "))
    }
}

/// The GPU types field of a Runpod target.
pub(super) const GPU_TYPES: &str = "gpu_types";
/// The data centers field of a Runpod target.
pub(super) const DATA_CENTER_IDS: &str = "data_center_ids";
/// The limits of a Runpod target's `auto` GPU types: unset when the types are
/// listed.
pub(super) const AUTO_LIMITS: [&str; 2] = ["min_vram_gb", "max_price_per_hour"];

/// The Secure Cloud GPU types listed for the run's GPU count, or why they
/// cannot be.
pub(super) type Gpus = Result<Vec<GpuType>, String>;

/// The VRAM a run needs per GPU, or why it is unknown.
pub(super) type Need = Result<Estimate, String>;

/// What the start dialog looks up in the background.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Catalog {
    /// The GPU types.
    pub(super) gpus: Gpus,
    /// The VRAM the run needs per GPU.
    pub(super) need: Need,
}

impl Catalog {
    /// The least VRAM `auto` GPU types need, when the estimate is known.
    fn floor_gb(&self) -> Option<u32> {
        self.need.as_ref().ok().map(Estimate::floor_gb)
    }
}

/// Entries an `auto` choice shows at most; the others are counted.
const AUTO_SHOWN: usize = 4;

/// The start of the line saying what the run costs at most.
const COST_LABEL: &str = "max_hours   ";

/// The start of the line saying what the run may spend (`max_cost_usd`).
const COST_CAP_LABEL: &str = "max_cost    ";

/// The start of the line saying what `y` saves first.
const CHANGED_LABEL: &str = "changed     ";

/// What a run started now in the project in `dir` would use, from its settings
/// (read from `source`) and data files only.
///
/// # Errors
///
/// Returns why no run can start: the settings cannot be loaded, there is no
/// `[training]` section, its target is unknown, or the data cannot be read.
pub(super) fn prepare(dir: &Path, source: &Source) -> Result<StartPlan, String> {
    let settings = source.load(dir).map_err(|error| format!("{error:#}"))?;
    plan(&settings, &DataFiles::new(dir))
}

/// What a run resuming the stopped run `run_id` of the project in `dir` would
/// use, from its settings (read from `source`) and the stopped run's own data.
///
/// # Errors
///
/// Returns why no run can start, or why `run_id` cannot be resumed: it is not
/// stopped with its snapshot here, or the training settings changed since.
pub(super) fn prepare_resume(
    dir: &Path,
    source: &Source,
    run_id: &str,
) -> Result<StartPlan, String> {
    let settings = source.load(dir).map_err(|error| format!("{error:#}"))?;
    let training = settings
        .training
        .as_ref()
        .ok_or("no [training] section in overbrainer.toml")?;
    let runs = Runs::new(dir);
    let trainer = crate::cli::train::trainer(dir, training, &runs, Some(run_id))
        .map_err(|error| format!("{error:#}"))?;
    let resume = trainer
        .resume()
        .ok_or_else(|| format!("run {run_id} cannot be resumed"))?;
    let step = runs
        .load(run_id)
        .ok()
        .and_then(|record| record.snapshot)
        .map_or(0, |snapshot| snapshot.step);
    let mut plan = plan(&settings, &DataFiles::new(&resume.dir))?;
    plan.resume = Some(ResumePlan {
        run_id: run_id.to_string(),
        step,
        checkpoint: resume.checkpoint.clone(),
    });
    Ok(plan)
}

/// What a run started now would use, from `settings` and the data `files`.
///
/// # Errors
///
/// Returns why no run can start: there is no `[training]` section, its
/// target is unknown, or the data cannot be read.
fn plan(settings: &Settings, files: &DataFiles) -> Result<StartPlan, String> {
    let training = settings
        .training
        .as_ref()
        .ok_or("no [training] section in overbrainer.toml")?;
    let target = settings
        .targets
        .get(&training.target)
        .ok_or_else(|| format!("unknown target `{}`", training.target))?;
    let count = |path: &Path| {
        read::<IgnoredAny>(path)
            .map(|lines| lines.len())
            .map_err(|error| format!("{:#}", anyhow::Error::from(error)))
    };
    let adapter = match training.adapter {
        Adapter::Lora => "lora",
        Adapter::Qlora => "qlora",
        Adapter::Full => "full",
    };
    let mut warnings: Vec<String> = reasoning_template_warning(training).into_iter().collect();
    if training.hub_model_id.is_some() && settings.hf_token.is_none() {
        warnings.push(
            "training.hub_model_id is set but OVERBRAINER_HF_TOKEN is not: the push will fail"
                .to_string(),
        );
    }
    Ok(StartPlan {
        target: training.target.clone(),
        kind: kind(target),
        model: format!(
            "{}, {adapter}, {} epochs, lr {:e}",
            training.base_model, training.epochs, training.learning_rate
        ),
        train: count(&files.train)?,
        eval: count(&files.eval)?,
        runpod: RunpodTarget::from_target(target).map(|spec| Box::new(RunpodPlan::new(spec))),
        warnings,
        resume: None,
    })
}

/// What auto mode would run after split.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct AutoPlan {
    /// The run it starts, and what that run leaves; `None` without
    /// `[training]`: auto mode stops after split.
    pub(super) run: Option<(Box<StartPlan>, Outputs)>,
}

/// What auto mode run now in the project in `dir` would do after split, from
/// its settings (read from `source`).
///
/// # Errors
///
/// Returns why it cannot run: the settings cannot be loaded, the training
/// target is unknown, or the data cannot be read.
pub(super) fn prepare_auto(dir: &Path, source: &Source) -> Result<AutoPlan, String> {
    let settings = source.load(dir).map_err(|error| format!("{error:#}"))?;
    let Some(training) = &settings.training else {
        return Ok(AutoPlan { run: None });
    };
    let outputs = Outputs::of(training);
    let run = plan(&settings, &DataFiles::new(dir))?;
    Ok(AutoPlan {
        run: Some((Box::new(run), outputs)),
    })
}

/// The kind of `target`, never its host nor any key.
fn kind(target: &Target) -> String {
    let runtime = |runtime: &Runtime| match runtime {
        Runtime::Docker => "docker",
        Runtime::Native => "native",
    };
    match target {
        Target::Local { runtime: r, .. } => format!("local, {}", runtime(r)),
        Target::Ssh { runtime: r, .. } => format!("ssh, {}", runtime(r)),
        Target::Runpod { .. } => "runpod, Secure Cloud".to_string(),
    }
}

/// The GPU types for `gpu_count` GPUs (see [`list_gpus`]) and the VRAM a run
/// needs per GPU (see [`estimate_need`]), looked up together.
pub(super) async fn look_up(dir: &Path, source: Source, gpu_count: u32) -> Catalog {
    let (gpus, need) = tokio::join!(
        list_gpus(dir, source.clone(), gpu_count),
        estimate_need(dir, source, HF_URL, START_CATALOG_TIMEOUT),
    );
    Catalog { gpus, need }
}

/// The VRAM a run of the project in `dir` (its settings from `source`) needs
/// per GPU, its model's shape read from the Hugging Face Hub at `base_url`
/// with the project's token, giving up after `limit`. Why it is unknown is
/// logged too, never with the token.
pub(super) async fn estimate_need(
    dir: &Path,
    source: Source,
    base_url: &str,
    limit: Duration,
) -> Need {
    let lookup = async {
        let settings = source.load(dir).map_err(|error| format!("{error:#}"))?;
        let training = settings
            .training
            .as_ref()
            .ok_or("no [training] section in overbrainer.toml")?;
        let token = match &settings.hf_token {
            Some(token) => Some(
                crate::cli::train::hf_token(token)
                    .await
                    .map_err(|error| format!("{error:#}"))?,
            ),
            None => None,
        };
        estimate_model(training, token.as_ref(), base_url, limit)
            .await
            .map_err(|error| error.to_string())
    };
    let need = tokio::time::timeout(limit, lookup)
        .await
        .unwrap_or_else(|_| Err("Hugging Face took too long to answer".to_string()));
    if let Err(error) = &need {
        tracing::warn!("cannot estimate the VRAM the run needs: {error}");
    }
    need
}

/// The Secure Cloud GPU types for `gpu_count` GPUs on the Runpod account of
/// the project in `dir` (its settings from `source`), giving up after
/// [`START_CATALOG_TIMEOUT`]. Why they cannot be read is logged too, with only the
/// client's fixed messages.
pub(super) async fn list_gpus(dir: &Path, source: Source, gpu_count: u32) -> Gpus {
    let gpus = lookup_gpus(dir, source, gpu_count, START_CATALOG_TIMEOUT).await;
    if let Err(error) = &gpus {
        tracing::warn!("{error}");
    }
    gpus
}

/// [`list_gpus`] without logging, giving up after `limit`.
async fn lookup_gpus(dir: &Path, source: Source, gpu_count: u32, limit: Duration) -> Gpus {
    let lookup = async {
        let settings = source.load(dir)?;
        let client = crate::cli::pod::client(&settings).await?;
        Ok::<_, anyhow::Error>(client.list_gpu_types(gpu_count).await?)
    };
    match tokio::time::timeout(limit, lookup).await {
        Ok(Ok(gpus)) => Ok(gpus),
        Ok(Err(error)) => Err(format!("cannot read the Runpod catalog: {error:#}")),
        Err(_) => Err("the Runpod catalog took too long to answer".to_string()),
    }
}

/// Stock of `gpu` for the run: its best in `centers`, or overall when they
/// are any.
fn stock(gpu: &GpuType, centers: &[String]) -> Availability {
    let rank = |band: &Availability| match band {
        Availability::High => 3,
        Availability::Medium => 2,
        Availability::Low => 1,
        Availability::None | Availability::Unknown => 0,
    };
    if centers.is_empty() {
        return gpu.availability;
    }
    centers
        .iter()
        .map(|center| gpu.stock_in(center))
        .max_by_key(rank)
        .unwrap_or(Availability::None)
}

/// The confirmation text of `plan`, with the GPU types of the catalog once
/// looked up.
pub(super) fn text(plan: &StartPlan, catalog: Option<&Catalog>) -> Vec<String> {
    let data = match &plan.resume {
        Some(resume) => format!(
            "data        runs/{id}/data/train.jsonl {} examples, runs/{id}/data/eval.jsonl {}",
            plan.train,
            plan.eval,
            id = resume.run_id
        ),
        None => format!(
            "data        data/train.jsonl {} examples, data/eval.jsonl {}",
            plan.train, plan.eval
        ),
    };
    lines(plan, catalog, data)
}

/// [`text`] for a run started once split has rebuilt the data, which it
/// cannot count yet.
pub(super) fn text_after_split(plan: &StartPlan, catalog: Option<&Catalog>) -> Vec<String> {
    let data = "data        rebuilt by split just before the run".to_string();
    lines(plan, catalog, data)
}

fn lines(plan: &StartPlan, catalog: Option<&Catalog>, data: String) -> Vec<String> {
    let mut text = vec![
        format!("target      {} ({})", plan.target, plan.kind),
        format!("model       {}", plan.model),
    ];
    if let Some(resume) = &plan.resume {
        text.push(format!(
            "resume      run {} from step {}: its snapshot runs/{}/{} and its data",
            resume.run_id, resume.step, resume.run_id, resume.checkpoint
        ));
    }
    text.push(data);
    // Before the GPU list: a dialog too tall for the terminal cuts after them.
    for warning in &plan.warnings {
        text.push(format!("warning     {warning}"));
    }
    if let Some(runpod) = &plan.runpod {
        if let Some(changed) = runpod.changed_text() {
            text.push(format!(
                "{CHANGED_LABEL}{changed}: saved to {CONFIG_FILE} on y, then the run starts"
            ));
        }
        text.extend(runpod_lines(&runpod.spec, catalog));
    }
    text.push(
        "The run keeps going when you leave this view or quit; attach again here or with \
         `overbrainer train attach <run-id>`."
            .to_string(),
    );
    text
}

/// The positions in `text`, a start dialog's, of the lines a dialog too
/// tall for the terminal keeps: what `y` saves first, and what a Runpod run
/// costs at most (`max_hours`, `max_cost_usd`).
pub(super) fn pinned(text: &[String]) -> Vec<usize> {
    text.iter()
        .enumerate()
        .filter(|(_, line)| {
            [CHANGED_LABEL, COST_LABEL, COST_CAP_LABEL]
                .iter()
                .any(|label| line.starts_with(label))
        })
        .map(|(at, _)| at)
        .collect()
}

/// The VRAM the run needs per GPU; then the GPU types of `spec`, or those
/// `auto` picks now from the catalog, each with its list price times the GPU
/// count, VRAM, stock and fit; the data centers; then `max_hours` with the
/// most it can cost at the highest listed rate.
fn runpod_lines(spec: &RunpodTarget, catalog: Option<&Catalog>) -> Vec<String> {
    let count = spec.gpu_count;
    let listed = catalog.and_then(|catalog| catalog.gpus.as_deref().ok());
    // The estimate is the floor of `auto` only without `min_vram_gb`.
    let floor = catalog
        .and_then(Catalog::floor_gb)
        .filter(|_| spec.min_vram_gb.is_none());
    let (header, chosen) = match (&spec.gpu_types, listed) {
        (ListOrAuto::List(ids), _) => (
            format!("GPU types   tried in order, list price x {count} GPU, VRAM, stock, fit:"),
            Ok(ids.clone()),
        ),
        (ListOrAuto::Auto, None) => (
            format!(
                "GPU types   auto: those in stock for {count} GPU, cheapest first, chosen at \
                 the start"
            ),
            Ok(Vec::new()),
        ),
        (ListOrAuto::Auto, Some(listed)) => match auto_gpus(spec, listed, floor) {
            Ok(ids) => (
                format!("GPU types   auto picks now, list price x {count} GPU, VRAM, stock, fit:"),
                Ok(ids),
            ),
            Err(error) => (format!("GPU types   auto: {error}"), Err(())),
        },
    };
    let auto_floor = floor.filter(|_| spec.gpu_types.is_auto());
    let mut lines = vec![vram_line(catalog, auto_floor), header];
    let ids = chosen.clone().unwrap_or_default();
    let shown = if spec.gpu_types.is_auto() {
        AUTO_SHOWN
    } else {
        ids.len()
    };
    let centers = spec.data_center_ids.list();
    let shown_ids: Vec<String> = ids.iter().take(shown).cloned().collect();
    lines.extend(gpu_lines(&shown_ids, count, centers, catalog));
    if ids.len() > shown {
        lines.push(format!("- and {} more", ids.len() - shown));
    }
    if let (ListOrAuto::List(chosen), Some(catalog)) = (&spec.gpu_types, catalog) {
        lines.extend(small_line(chosen, catalog));
    }
    if let Some(Catalog {
        gpus: Err(error), ..
    }) = catalog
    {
        lines.push(format!("catalog     {error}"));
    }
    lines.push(center_line(spec, listed, floor, chosen.is_ok()));
    lines.push(max_hours_line(spec, listed, &ids));
    if let Some(usd) = spec.max_cost_usd {
        let short = highest_rate(listed, &ids)
            .0
            .and_then(|rate| short_cap_warning(usd, rate * f64::from(count)))
            .map_or_else(String::new, |warning| {
                format!("; at the highest listed rate, {warning}")
            });
        lines.push(format!(
            "{COST_CAP_LABEL}${usd:.2}: the job stops with a snapshot at {:.0}% ({} min before \
             the cap at the latest); the pod is deleted at 100%, with a snapshot nobody \
             collected by then{short}",
            crate::runpod::SNAPSHOT_SHARE * 100.0,
            crate::runpod::SNAPSHOT_LEAD.as_secs() / 60
        ));
    }
    lines
}

/// The highest list price of the GPU types `ids` once `gpus` are read, and
/// whether some of their prices are unknown.
fn highest_rate(gpus: Option<&[GpuType]>, ids: &[String]) -> (Option<f64>, bool) {
    let rates: Vec<Option<f64>> = ids
        .iter()
        .map(|id| {
            gpus.and_then(|listed| listed.iter().find(|gpu| gpu.id == *id))
                .and_then(GpuType::secure_price)
        })
        .collect();
    (
        rates.iter().flatten().copied().reduce(f64::max),
        rates.contains(&None),
    )
}

/// A warning naming the `chosen` GPU types with less VRAM than the estimate
/// of `catalog`, if any: they stay chosen, but their pod may run out of
/// memory.
fn small_line(chosen: &[String], catalog: &Catalog) -> Option<String> {
    let (Ok(listed), Ok(need)) = (&catalog.gpus, &catalog.need) else {
        return None;
    };
    let small: Vec<&str> = chosen
        .iter()
        .filter(|id| {
            listed
                .iter()
                .find(|gpu| gpu.id == **id)
                .is_some_and(|gpu| fit(gpu.memory, Some(need)) == Fit::Small)
        })
        .map(String::as_str)
        .collect();
    let (verb, names) = match small.as_slice() {
        [] => return None,
        [one] => ("has", (*one).to_string()),
        several => ("have", several.join(", ")),
    };
    Some(format!(
        "warning     {names} {verb} less VRAM than the estimate: the run may run out of memory"
    ))
}

/// The VRAM the run needs per GPU, once estimated, and the `floor` of `auto`
/// GPU types it gives; or why it is unknown.
fn vram_line(catalog: Option<&Catalog>, floor: Option<u32>) -> String {
    let need = match catalog.map(|catalog| &catalog.need) {
        None => "estimating from the model...".to_string(),
        Some(Ok(need)) => match floor {
            Some(gb) => format!("{need} per GPU (estimate): auto keeps >= {gb} GB"),
            None => format!("{need} per GPU (estimate)"),
        },
        Some(Err(error)) => format!("unknown: {error}"),
    };
    format!("vram        {need}")
}

/// The GPU types `auto` picks now from `gpus`, with at least `floor` GB of
/// VRAM when given, in the data centers listed if any.
fn auto_gpus(
    spec: &RunpodTarget,
    gpus: &[GpuType],
    floor: Option<u32>,
) -> Result<Vec<String>, ResolveError> {
    let mut alone = spec.clone();
    if alone.data_center_ids.is_auto() {
        alone.data_center_ids = ListOrAuto::default();
    }
    resolve_with_floor(&alone, gpus, floor).map(|resolved| resolved.gpu_types.list().to_vec())
}

/// The cells of the GPU type `id` once `catalog` is read: its list price for
/// `count` GPUs, its VRAM, its stock in `centers` and whether it holds the
/// run; or what is known instead.
fn gpu_cells(
    id: &str,
    count: u32,
    centers: &[String],
    catalog: Option<&Catalog>,
) -> Result<[String; 4], &'static str> {
    let Some(catalog) = catalog else {
        return Err("looking up the catalog...");
    };
    let listed = catalog.gpus.as_ref().map_err(|_| "catalog unread")?;
    let gpu = listed
        .iter()
        .find(|gpu| gpu.id == id)
        .ok_or("not in the catalog")?;
    let price = gpu.secure_price().map_or_else(
        || "price unknown".to_string(),
        |rate| format!("${:.2}/h", rate * f64::from(count)),
    );
    Ok([
        price,
        format!("{:>3} GB", gpu.memory),
        stock(gpu, centers).name().to_string(),
        fit(gpu.memory, catalog.need.as_ref().ok())
            .name()
            .to_string(),
    ])
}

/// The lines of the GPU types `ids`, each column as wide as its widest cell.
fn gpu_lines(
    ids: &[String],
    count: u32,
    centers: &[String],
    catalog: Option<&Catalog>,
) -> Vec<String> {
    let rows: Vec<(&String, Result<[String; 4], &str>)> = ids
        .iter()
        .map(|id| (id, gpu_cells(id, count, centers, catalog)))
        .collect();
    let widest = |column: usize| {
        rows.iter()
            .filter_map(|(_, cells)| cells.as_ref().ok())
            .map(|cells| cells[column].chars().count())
            .max()
            .unwrap_or(0)
    };
    let id_width = ids.iter().map(|id| id.chars().count()).max().unwrap_or(0);
    let (price_width, stock_width) = (widest(0), widest(2));
    rows.iter()
        .map(|(id, cells)| match cells {
            Ok([price, vram, stock, fit]) => format!(
                "- {id:<id_width$}  {price:<price_width$}  {vram}  {stock:<stock_width$}  {fit}"
            ),
            Err(about) => format!("- {id:<id_width$}  {about}"),
        })
        .collect()
}

/// The data centers of `spec`, or those `auto` picks now from `gpus` (with
/// GPU types of at least `floor` GB) when `auto` found GPU types (`placed`).
fn center_line(
    spec: &RunpodTarget,
    gpus: Option<&[GpuType]>,
    floor: Option<u32>,
    placed: bool,
) -> String {
    let centers = match (&spec.data_center_ids, gpus) {
        (ListOrAuto::List(ids), _) if ids.is_empty() => "any".to_string(),
        (ListOrAuto::List(ids), _) => ids.join(", "),
        (ListOrAuto::Auto, Some(listed)) if placed => {
            match resolve_with_floor(spec, listed, floor) {
                Ok(resolved) => {
                    let ids = resolved.data_center_ids.list();
                    let more = ids.len().saturating_sub(AUTO_SHOWN);
                    let shown = ids
                        .iter()
                        .take(AUTO_SHOWN)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ");
                    if more > 0 {
                        format!("auto picks now: {shown} and {more} more")
                    } else {
                        format!("auto picks now: {shown}")
                    }
                },
                Err(error) => format!("auto: {error}"),
            }
        },
        (ListOrAuto::Auto, _) => {
            "auto: those with a chosen GPU type in stock, chosen at the start".to_string()
        },
    };
    format!("datacenters {centers}")
}

/// `max_hours` of `spec`, with the most the run can cost at the highest list
/// price of the GPU types `ids` once `gpus` are read.
fn max_hours_line(spec: &RunpodTarget, gpus: Option<&[GpuType]>, ids: &[String]) -> String {
    let (highest, unknown) = highest_rate(gpus, ids);
    let some_unknown = if unknown {
        " (some prices unknown)"
    } else {
        ""
    };
    let count = f64::from(spec.gpu_count);
    let most = highest.map_or_else(String::new, |rate| {
        format!(
            ", about ${:.2} at most at the highest listed rate{some_unknown}",
            rate * count * spec.max_hours
        )
    });
    format!(
        "{COST_LABEL}{}: the watchdog deletes the pod by then{most}",
        spec.max_hours
    )
}

#[cfg(test)]
/// A Runpod target trying `gpus` in order, `count` per pod, for 6 hours.
pub(super) fn runpod_spec(gpus: ListOrAuto, count: u32) -> RunpodTarget {
    RunpodTarget {
        gpu_types: gpus,
        min_vram_gb: None,
        max_price_per_hour: None,
        gpu_count: count,
        image: crate::config::DEFAULT_RUNPOD_IMAGE.to_string(),
        venv: crate::config::DEFAULT_RUNPOD_VENV.to_string(),
        container_disk_gb: 50,
        max_hours: 6.0,
        max_cost_usd: None,
        boot_grace: Duration::from_secs(1800),
        retrieve_grace: Duration::from_secs(3600),
        data_center_ids: ListOrAuto::default(),
        network_volume_id: None,
        max_volume_gb: None,
    }
}

#[cfg(test)]
mod tests {
    use crate::config::EnvSource;
    use wiremock::matchers::{header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::tui::snapshots::{gpu_types, looked_up};

    /// The position of the `max_hours` line in `text`.
    fn cost_line(text: &[String]) -> Option<usize> {
        text.iter().position(|line| line.starts_with(COST_LABEL))
    }

    fn list(ids: &[&str]) -> ListOrAuto {
        ListOrAuto::List(ids.iter().map(|id| (*id).to_string()).collect())
    }

    fn plan(runpod: bool) -> StartPlan {
        StartPlan {
            target: "gpu_cloud".into(),
            kind: "runpod, Secure Cloud".into(),
            model: "Qwen/Qwen3-4B, qlora, 3 epochs, lr 2e-4".into(),
            train: 1234,
            eval: 137,
            runpod: runpod.then(|| {
                Box::new(RunpodPlan::new(runpod_spec(
                    list(&["NVIDIA GeForce RTX 4090", "NVIDIA B300"]),
                    2,
                )))
            }),
            warnings: Vec::new(),
            resume: None,
        }
    }

    /// [`plan`] on `spec`, as the settings have it.
    fn planned(spec: RunpodTarget) -> StartPlan {
        StartPlan {
            runpod: Some(Box::new(RunpodPlan::new(spec))),
            ..plan(false)
        }
    }

    fn runpod(plan: &mut StartPlan) -> Result<&mut RunpodPlan, String> {
        plan.runpod
            .as_deref_mut()
            .ok_or_else(|| "no Runpod plan".to_string())
    }

    fn fixture() -> Result<Catalog, serde_json::Error> {
        gpu_types().map(|gpus| looked_up(Ok(gpus)))
    }

    #[test]
    fn auto_gpu_types_say_they_are_chosen_at_the_start() -> Result<(), String> {
        let mut auto = plan(true);
        runpod(&mut auto)?.spec.gpu_types = ListOrAuto::Auto;
        let lines = text(&auto, None);
        assert!(
            lines.iter().any(|line| line
                == "GPU types   auto: those in stock for 2 GPU, cheapest first, chosen at the start"),
            "{lines:?}"
        );
        assert!(lines.contains(&"datacenters any".to_string()), "{lines:?}");
        let line = cost_line(&lines).and_then(|at| lines.get(at).cloned());
        assert_eq!(
            line.as_deref(),
            Some("max_hours   6: the watchdog deletes the pod by then")
        );
        assert_eq!(cost_line(&text(&plan(false), None)), None);
        Ok(())
    }

    #[test]
    fn listed_gpus_show_price_vram_stock_and_fit_and_bound_the_run()
    -> Result<(), Box<dyn std::error::Error>> {
        let gpus = fixture()?;
        let lines = text(&plan(true), Some(&gpus));
        assert_eq!(lines[3], "vram        about 20.4 GB per GPU (estimate)");
        assert_eq!(
            lines[4],
            "GPU types   tried in order, list price x 2 GPU, VRAM, stock, fit:"
        );
        assert_eq!(
            lines[5],
            "- NVIDIA GeForce RTX 4090  $1.38/h   24 GB  MEDIUM  ok"
        );
        assert_eq!(lines[6], "- NVIDIA B300              not in the catalog");
        assert_eq!(lines[7], "datacenters any");
        assert_eq!(
            lines[8],
            "max_hours   6: the watchdog deletes the pod by then, about $8.28 at most at the \
             highest listed rate (some prices unknown)"
        );
        assert_eq!(cost_line(&lines), Some(8));
        let waiting = text(&plan(true), None);
        assert_eq!(waiting[3], "vram        estimating from the model...");
        assert!(
            waiting[5].ends_with("looking up the catalog..."),
            "{waiting:?}"
        );
        Ok(())
    }

    #[test]
    fn an_unknown_estimate_says_why_and_fits_nothing() -> Result<(), Box<dyn std::error::Error>> {
        let catalog = Catalog {
            gpus: Ok(gpu_types()?),
            need: Err("Hugging Face answered 401 for Qwen/Qwen3-4B".into()),
        };
        let lines = text(&plan(true), Some(&catalog));
        assert_eq!(
            lines[3],
            "vram        unknown: Hugging Face answered 401 for Qwen/Qwen3-4B"
        );
        assert!(lines[5].ends_with("MEDIUM  ?"), "{lines:?}");
        let auto = runpod_spec(ListOrAuto::Auto, 1);
        let lines = text(&planned(auto), Some(&catalog));
        assert_eq!(
            lines[4],
            "GPU types   auto picks now, list price x 1 GPU, VRAM, stock, fit:"
        );
        assert!(
            lines[5].starts_with("- NVIDIA RTX 2000 Ada Generation"),
            "no floor: {lines:?}"
        );
        Ok(())
    }

    #[test]
    fn chosen_gpu_types_below_the_estimate_are_warned_about()
    -> Result<(), Box<dyn std::error::Error>> {
        let small = "NVIDIA RTX 2000 Ada Generation";
        let chosen = planned(runpod_spec(list(&[small, "NVIDIA A40"]), 1));
        let lines = text(&chosen, Some(&fixture()?));
        assert!(lines[5].ends_with("16 GB  HIGH  small"), "{lines:?}");
        assert_eq!(
            lines[7],
            format!(
                "warning     {small} has less VRAM than the estimate: the run may run out of \
                 memory"
            )
        );
        let both = planned(runpod_spec(list(&[small, "NVIDIA A40", small]), 1));
        let lines = text(&both, Some(&fixture()?));
        assert!(lines[8].contains(&format!("{small}, {small} have less VRAM")));
        let fine = text(&plan(true), Some(&fixture()?));
        assert!(
            !fine.iter().any(|line| line.contains("less VRAM")),
            "{fine:?}"
        );
        let unknown = Catalog {
            gpus: Ok(gpu_types()?),
            need: Err("unknown".into()),
        };
        let lines = text(&chosen, Some(&unknown));
        assert!(
            !lines.iter().any(|line| line.contains("less VRAM")),
            "{lines:?}"
        );
        Ok(())
    }

    #[test]
    fn a_price_of_zero_or_less_is_unknown_and_never_bounds_the_run()
    -> Result<(), Box<dyn std::error::Error>> {
        let gpus: Vec<GpuType> = serde_json::from_value(serde_json::json!([
            {"id": "NVIDIA GeForce RTX 4090", "memory": 24, "price": {"secure": 0.0},
             "availability": "HIGH"},
            {"id": "NVIDIA B300", "memory": 288, "price": {"secure": -1.0}}
        ]))?;
        let lines = text(&plan(true), Some(&looked_up(Ok(gpus))));
        assert_eq!(
            lines[5],
            "- NVIDIA GeForce RTX 4090  price unknown   24 GB  HIGH     ok"
        );
        assert_eq!(
            lines[6],
            "- NVIDIA B300              price unknown  288 GB  UNKNOWN  ok"
        );
        assert_eq!(
            lines[8],
            "max_hours   6: the watchdog deletes the pod by then"
        );
        Ok(())
    }

    #[test]
    fn auto_shows_what_it_picks_now_above_the_estimate_and_bounds_the_run_by_all_of_it()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut auto = runpod_spec(ListOrAuto::Auto, 1);
        let lines = text(&planned(auto.clone()), Some(&fixture()?));
        assert_eq!(
            lines[3],
            "vram        about 20.4 GB per GPU (estimate): auto keeps >= 21 GB"
        );
        assert_eq!(
            lines[4],
            "GPU types   auto picks now, list price x 1 GPU, VRAM, stock, fit:"
        );
        // The 16 GB RTX 2000 Ada, cheapest, is below the estimate.
        assert_eq!(
            lines[5],
            "- NVIDIA A40               $0.40/h   48 GB  HIGH    ok"
        );
        assert_eq!(
            lines[6],
            "- NVIDIA L4                $0.43/h   24 GB  HIGH    ok"
        );
        assert_eq!(lines[9], "- and 2 more");
        // The dearest picked, the H100, bounds the run even though not shown.
        assert!(
            lines[11].ends_with("about $17.94 at most at the highest listed rate"),
            "{lines:?}"
        );
        // An explicit min_vram_gb wins over the estimate.
        auto.min_vram_gb = Some(10);
        let lines = text(&planned(auto.clone()), Some(&fixture()?));
        assert_eq!(lines[3], "vram        about 20.4 GB per GPU (estimate)");
        assert!(
            lines[5].starts_with("- NVIDIA RTX 2000 Ada Generation")
                && lines[5].ends_with("HIGH  small"),
            "{lines:?}"
        );
        auto.min_vram_gb = Some(500);
        let lines = text(&planned(auto), Some(&fixture()?));
        assert!(
            lines[4].starts_with("GPU types   auto: no GPU type in stock"),
            "{lines:?}"
        );
        assert_eq!(lines[5], "datacenters any");
        Ok(())
    }

    /// Two GPU types, stocked per data center.
    fn stocked() -> Result<Vec<GpuType>, serde_json::Error> {
        serde_json::from_value(serde_json::json!([
            {"id": "NVIDIA A40", "memory": 48, "price": {"secure": 0.4},
             "maxCount": {"secure": 8}, "availability": "HIGH",
             "dataCenters": [{"id": "EU-RO-1", "availability": "LOW"},
                             {"id": "US-TX-3", "availability": "NONE"}]},
            {"id": "NVIDIA L40S", "memory": 48, "price": {"secure": 0.86},
             "maxCount": {"secure": 8}, "availability": "MEDIUM",
             "dataCenters": [{"id": "US-TX-3", "availability": "MEDIUM"},
                             {"id": "EU-SE-1", "availability": "HIGH"}]}
        ]))
    }

    #[test]
    fn stock_is_judged_in_the_data_centers_chosen_and_auto_ones_are_shown()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut listed = runpod_spec(list(&["NVIDIA A40", "NVIDIA L40S"]), 2);
        listed.data_center_ids = list(&["US-TX-3"]);
        let lines = text(&planned(listed.clone()), Some(&looked_up(Ok(stocked()?))));
        assert!(lines[5].ends_with("48 GB  NONE    ok"), "{lines:?}");
        assert!(lines[6].ends_with("48 GB  MEDIUM  ok"), "{lines:?}");
        assert_eq!(lines[7], "datacenters US-TX-3");
        listed.data_center_ids = ListOrAuto::Auto;
        let listed = planned(listed);
        let lines = text(&listed, Some(&looked_up(Ok(stocked()?))));
        assert_eq!(
            lines[7],
            "datacenters auto picks now: EU-RO-1, EU-SE-1, US-TX-3"
        );
        assert!(lines[5].ends_with("HIGH    ok"), "overall stock: {lines:?}");
        let waiting = text(&listed, None);
        assert_eq!(
            waiting[7],
            "datacenters auto: those with a chosen GPU type in stock, chosen at the start"
        );
        Ok(())
    }

    #[test]
    fn the_gpu_column_is_as_wide_as_the_longest_id_shown() {
        let long = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition";
        let failed: Gpus = Err("unread".into());
        let lines = text(
            &planned(runpod_spec(list(&[long, "NVIDIA A40"]), 1)),
            Some(&looked_up(failed)),
        );
        assert_eq!(lines[5], format!("- {long}  catalog unread"));
        assert_eq!(
            lines[6],
            format!(
                "- {:<width$}  catalog unread",
                "NVIDIA A40",
                width = long.len()
            )
        );
    }

    #[test]
    fn an_unread_catalog_says_why_and_bounds_nothing() {
        let failed: Gpus = Err("cannot read the Runpod catalog: no Runpod API key".into());
        let lines = text(&plan(true), Some(&looked_up(failed)));
        assert_eq!(lines[5], "- NVIDIA GeForce RTX 4090  catalog unread");
        assert_eq!(
            lines[7],
            "catalog     cannot read the Runpod catalog: no Runpod API key"
        );
        assert_eq!(
            lines[9],
            "max_hours   6: the watchdog deletes the pod by then"
        );
    }

    #[test]
    fn a_resumed_run_says_where_it_starts_and_a_cost_cap_is_kept_in_view() {
        let mut spec = runpod_spec(list(&["NVIDIA A40"]), 1);
        spec.max_cost_usd = Some(20.0);
        let plan = StartPlan {
            resume: Some(ResumePlan {
                run_id: "r0".into(),
                step: 120,
                checkpoint: "output/checkpoint-120".into(),
            }),
            ..planned(spec)
        };
        let lines = text(&plan, None);
        assert_eq!(
            lines.get(2).map(String::as_str),
            Some(
                "resume      run r0 from step 120: its snapshot runs/r0/output/checkpoint-120 and its data"
            )
        );
        assert_eq!(
            lines.get(3).map(String::as_str),
            Some("data        runs/r0/data/train.jsonl 1234 examples, runs/r0/data/eval.jsonl 137")
        );
        let cap = lines.iter().position(|line| {
            line == "max_cost    $20.00: the job stops with a snapshot at 95% (15 min before \
                         the cap at the latest); the pod is deleted at 100%, with a snapshot \
                         nobody collected by then"
        });
        assert!(cap.is_some(), "{lines:?}");
        assert!(cap.is_some_and(|at| pinned(&lines).contains(&at)));
    }

    #[test]
    fn a_cost_cap_too_small_for_the_listed_rate_is_warned_about()
    -> Result<(), Box<dyn std::error::Error>> {
        let gpus = fixture()?;
        let mut small = plan(true);
        // $0.50 at $1.38/h buys about 22 min: 7 of training before the lead.
        runpod(&mut small)?.spec.max_cost_usd = Some(0.5);
        let lines = text(&small, Some(&gpus));
        let cap = lines.iter().find(|line| line.starts_with(COST_CAP_LABEL));
        assert_eq!(
            cap.map(String::as_str),
            Some(
                "max_cost    $0.50: the job stops with a snapshot at 95% (15 min before the cap \
                 at the latest); the pod is deleted at 100%, with a snapshot nobody collected by \
                 then; at the highest listed rate, max_cost_usd $0.50 buys 22 min at $1.38/h: \
                 the job is stopped with a snapshot 15 min before the cap, so it trains 7 min at \
                 most; set max_cost_usd to $0.69 or more"
            )
        );
        runpod(&mut small)?.spec.max_cost_usd = Some(0.69);
        let lines = text(&small, Some(&gpus));
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with(COST_CAP_LABEL) && line.ends_with("by then")),
            "{lines:?}"
        );
        // No price yet: nothing to warn about.
        runpod(&mut small)?.spec.max_cost_usd = Some(0.5);
        let lines = text(&small, None);
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with(COST_CAP_LABEL) && line.ends_with("by then")),
            "{lines:?}"
        );
        Ok(())
    }

    #[test]
    fn a_choice_made_in_the_dialog_says_it_is_saved_first() -> Result<(), String> {
        let mut chosen = plan(true);
        assert_eq!(runpod(&mut chosen)?.changed(), Vec::<&str>::new());
        runpod(&mut chosen)?.spec.data_center_ids = ListOrAuto::Auto;
        runpod(&mut chosen)?.spec.gpu_types = list(&["NVIDIA A40"]);
        assert_eq!(
            runpod(&mut chosen)?.changed(),
            ["gpu_types", "data_center_ids"]
        );
        let lines = text(&chosen, None);
        assert_eq!(
            lines[3],
            "changed     gpu_types, data_center_ids: saved to overbrainer.toml on y, then the \
             run starts"
        );
        assert_eq!(cost_line(&lines), Some(8));
        assert_eq!(pinned(&lines), [3, 8]);
        Ok(())
    }

    #[test]
    fn a_plan_reads_the_training_section_and_the_split_files()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = crate::tui::snapshots::project()?;
        let none = prepare(dir.path(), &EnvSource::Vars(Vec::new()).into());
        assert_eq!(
            none,
            Err("no [training] section in overbrainer.toml".to_string())
        );
        let config = format!(
            "{}\n[training]\ntarget = \"homelab\"\nbase_model = \"Qwen/Qwen3-4B\"\nadapter = \"qlora\"\n\
             hub_model_id = \"me/model\"\n\n[targets.homelab]\nkind = \"ssh\"\nruntime = \"docker\"\n",
            crate::tui::snapshots::CONFIG
        );
        std::fs::write(dir.path().join("overbrainer.toml"), config)?;
        let files = DataFiles::new(dir.path());
        std::fs::write(&files.train, "{}\n{}\n")?;
        let env = EnvSource::Vars(vec![(
            "OVERBRAINER_TARGETS__HOMELAB__HOST".into(),
            "gpu.example".into(),
        )]);
        let plan = prepare(dir.path(), &env.into())?;
        assert_eq!((plan.train, plan.eval), (2, 0));
        assert_eq!(plan.kind, "ssh, docker");
        assert_eq!(plan.model, "Qwen/Qwen3-4B, qlora, 3 epochs, lr 2e-4");
        assert_eq!(plan.runpod, None);
        assert!(
            plan.warnings
                .iter()
                .any(|w| w.contains("OVERBRAINER_HF_TOKEN"))
        );
        assert!(
            !format!("{plan:?}").contains("gpu.example"),
            "never the host"
        );
        Ok(())
    }

    /// The key of the stub account; it must never show.
    const KEY: &str = "rp_tui_key_5150";

    /// A project on a Runpod target, with the API at `server` and [`KEY`].
    fn runpod_project(
        server: Option<&MockServer>,
    ) -> Result<(tempfile::TempDir, EnvSource), Box<dyn std::error::Error>> {
        let dir = crate::tui::snapshots::project()?;
        let config = format!(
            "{}\n[training]\ntarget = \"gpu_cloud\"\nbase_model = \"Qwen/Qwen3-4B\"\n\
             adapter = \"qlora\"\n\n[targets.gpu_cloud]\nkind = \"runpod\"\n\
             gpu_types = [\"NVIDIA GeForce RTX 4090\", \"NVIDIA A40\"]\ngpu_count = 2\nmax_hours = 6\n",
            crate::tui::snapshots::CONFIG
        );
        std::fs::write(dir.path().join("overbrainer.toml"), config)?;
        let vars = server.map_or_else(Vec::new, |server| {
            vec![
                ("OVERBRAINER_RUNPOD__API_KEY".to_string(), KEY.to_string()),
                (
                    "OVERBRAINER_RUNPOD__BASE_URL".to_string(),
                    format!("{}/v2", server.uri()),
                ),
            ]
        });
        Ok((dir, EnvSource::Vars(vars)))
    }

    #[tokio::test]
    async fn the_catalog_is_read_for_the_gpu_count() -> Result<(), Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v2/catalog/gpus"))
            .and(query_param("count", "2"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"gpus": [
                    {"id": "NVIDIA GeForce RTX 4090", "memory": 24, "price": {"secure": 0.74},
                     "maxCount": {"secure": 8}, "availability": "LOW"}
                ]})),
            )
            .expect(1)
            .mount(&server)
            .await;
        let (dir, env) = runpod_project(Some(&server))?;
        let plan = prepare(dir.path(), &env.clone().into())?;
        let runpod = plan.runpod.clone().ok_or("no Runpod plan")?;
        assert_eq!((runpod.spec.gpu_count, runpod.spec.max_hours), (2, 6.0));
        assert_eq!(runpod.changed(), Vec::<&str>::new());
        let gpus = list_gpus(dir.path(), env.into(), runpod.spec.gpu_count).await;
        let shown = text(&plan, Some(&looked_up(gpus))).join("\n");
        assert!(
            shown.contains("- NVIDIA GeForce RTX 4090  $1.48/h   24 GB  LOW  ok"),
            "{shown}"
        );
        assert!(shown.contains("- NVIDIA A40               not in the catalog"));
        assert!(!shown.contains(KEY) && !format!("{plan:?}").contains(KEY));
        Ok(())
    }

    /// A failed read says the client's fixed message, never the key the error
    /// body echoes.
    #[tokio::test]
    async fn a_failed_catalog_read_never_shows_the_key() -> Result<(), Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v2/catalog/gpus"))
            .respond_with(
                ResponseTemplate::new(404).set_body_string(format!("{{\"detail\": \"{KEY}\"}}")),
            )
            .mount(&server)
            .await;
        let (dir, env) = runpod_project(Some(&server))?;
        let error = lookup_gpus(dir.path(), env.into(), 2, START_CATALOG_TIMEOUT)
            .await
            .err()
            .ok_or("read")?;
        assert!(
            error.starts_with("cannot read the Runpod catalog: Runpod answered 404"),
            "{error}"
        );
        assert!(!error.contains(KEY), "the key shows");
        Ok(())
    }

    #[tokio::test]
    async fn the_estimate_reads_the_model_of_the_training_with_the_token()
    -> Result<(), Box<dyn std::error::Error>> {
        let hub = MockServer::start().await;
        let token = "hf_start_token_77";
        let bearer = format!("Bearer {token}");
        Mock::given(method("GET"))
            .and(path("/api/models/Qwen/Qwen3-4B"))
            .and(header("authorization", bearer.as_str()))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(
                    serde_json::json!({"safetensors": {"total": 4_022_468_096_u64}}),
                ),
            )
            .mount(&hub)
            .await;
        Mock::given(method("GET"))
            .and(path("/Qwen/Qwen3-4B/resolve/main/config.json"))
            .and(header("authorization", bearer.as_str()))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "hidden_size": 2560, "num_hidden_layers": 36, "vocab_size": 151_936
            })))
            .mount(&hub)
            .await;
        let (dir, _) = runpod_project(None)?;
        let env = EnvSource::Vars(vec![("OVERBRAINER_HF_TOKEN".into(), token.into())]);
        let need = estimate_need(dir.path(), env.into(), &hub.uri(), START_CATALOG_TIMEOUT).await?;
        // qlora on Qwen3-4B, the project's settings.
        assert_eq!(need.floor_gb(), 14, "{need}");
        let none = estimate_need(
            dir.path(),
            EnvSource::Vars(Vec::new()).into(),
            &hub.uri(),
            START_CATALOG_TIMEOUT,
        )
        .await;
        assert_eq!(
            none,
            Err("Hugging Face answered 404 for Qwen/Qwen3-4B".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_lookup_past_its_time_gives_up() -> Result<(), Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v2/catalog/gpus"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"gpus": []}))
                    .set_delay(Duration::from_secs(30)),
            )
            .mount(&server)
            .await;
        let (dir, env) = runpod_project(Some(&server))?;
        let started = std::time::Instant::now();
        let gpus = lookup_gpus(dir.path(), env.into(), 2, Duration::from_secs(1)).await;
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "gave up in time"
        );
        assert_eq!(
            gpus,
            Err("the Runpod catalog took too long to answer".to_string())
        );
        Ok(())
    }

    #[tokio::test]
    async fn without_an_api_key_the_catalog_is_unread() -> Result<(), Box<dyn std::error::Error>> {
        let (dir, env) = runpod_project(None)?;
        let gpus = list_gpus(dir.path(), env.into(), 1).await;
        assert!(
            gpus.as_ref()
                .err()
                .is_some_and(|error| error.contains("no Runpod API key")),
            "{gpus:?}"
        );
        Ok(())
    }
}
