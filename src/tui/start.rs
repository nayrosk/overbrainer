//! What `t` shows before a training run starts: the target, the model, the data,
//! and for Runpod the GPU types (configured, chosen with `g`, or what `auto`
//! picks now) with their list price, VRAM and stock, the data centers, and the
//! most `max_hours` can cost.

use std::path::Path;
use std::time::Duration;

use serde::de::IgnoredAny;

use crate::config::{Adapter, CONFIG_FILE, EnvSource, ListOrAuto, Runtime, Target};
use crate::dataset::{DataFiles, read};
use crate::runpod::{Availability, GpuType, RunpodTarget, resolve};
use crate::train::reasoning_template_warning;

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
}

impl RunpodPlan {
    /// The plan of `spec`, nothing chosen yet.
    pub(super) fn new(spec: RunpodTarget) -> Self {
        Self {
            file_gpu_types: spec.gpu_types.clone(),
            file_data_center_ids: spec.data_center_ids.clone(),
            file_min_vram_gb: spec.min_vram_gb,
            file_max_price_per_hour: spec.max_price_per_hour,
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

/// Entries an `auto` choice shows at most; the others are counted.
const AUTO_SHOWN: usize = 4;

/// The start of the line saying what the run costs at most.
const COST_LABEL: &str = "max_hours   ";

/// The start of the line saying what `y` saves first.
const CHANGED_LABEL: &str = "changed     ";

/// What a run started now in the project in `dir` would use, from its settings
/// (read with `env`) and data files only.
///
/// # Errors
///
/// Returns why no run can start: the settings cannot be loaded, there is no
/// `[training]` section, its target is unknown, or the data cannot be read.
pub(super) fn prepare(dir: &Path, env: EnvSource) -> Result<StartPlan, String> {
    let settings = crate::config::load(dir, env).map_err(|error| format!("{error:#}"))?;
    let training = settings
        .training
        .as_ref()
        .ok_or("no [training] section in overbrainer.toml")?;
    let target = settings
        .targets
        .get(&training.target)
        .ok_or_else(|| format!("unknown target `{}`", training.target))?;
    let files = DataFiles::new(dir);
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

/// The Secure Cloud GPU types for `gpu_count` GPUs on the Runpod account of
/// the project in `dir` (its settings read with `env`), giving up after
/// [`START_CATALOG_TIMEOUT`]. Why they cannot be read is logged too, with only the
/// client's fixed messages.
pub(super) async fn list_gpus(dir: &Path, env: EnvSource, gpu_count: u32) -> Gpus {
    let gpus = lookup_gpus(dir, env, gpu_count, START_CATALOG_TIMEOUT).await;
    if let Err(error) = &gpus {
        tracing::warn!("{error}");
    }
    gpus
}

/// [`list_gpus`] without logging, giving up after `limit`.
async fn lookup_gpus(dir: &Path, env: EnvSource, gpu_count: u32, limit: Duration) -> Gpus {
    let lookup = async {
        let settings = crate::config::load(dir, env)?;
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
pub(super) fn text(plan: &StartPlan, gpus: Option<&Gpus>) -> Vec<String> {
    let mut text = vec![
        format!("target      {} ({})", plan.target, plan.kind),
        format!("model       {}", plan.model),
        format!(
            "data        data/train.jsonl {} examples, data/eval.jsonl {}",
            plan.train, plan.eval
        ),
    ];
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
        text.extend(runpod_lines(&runpod.spec, gpus));
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
/// costs at most (`max_hours`).
pub(super) fn pinned(text: &[String]) -> Vec<usize> {
    text.iter()
        .enumerate()
        .filter(|(_, line)| line.starts_with(CHANGED_LABEL) || line.starts_with(COST_LABEL))
        .map(|(at, _)| at)
        .collect()
}

/// The GPU types of `spec`, or those `auto` picks now from `gpus`, each with
/// its list price times the GPU count, VRAM and stock; the data centers; then
/// `max_hours` with the most it can cost at the highest listed rate.
fn runpod_lines(spec: &RunpodTarget, gpus: Option<&Gpus>) -> Vec<String> {
    let count = spec.gpu_count;
    let catalog = match gpus {
        Some(Ok(listed)) => Some(listed.as_slice()),
        _ => None,
    };
    let (header, chosen) = match (&spec.gpu_types, catalog) {
        (ListOrAuto::List(ids), _) => (
            format!("GPU types   tried in order, list price x {count} GPU, VRAM, stock:"),
            Ok(ids.clone()),
        ),
        (ListOrAuto::Auto, None) => (
            format!(
                "GPU types   auto: those in stock for {count} GPU, cheapest first, chosen at \
                 the start"
            ),
            Ok(Vec::new()),
        ),
        (ListOrAuto::Auto, Some(listed)) => match auto_gpus(spec, listed) {
            Ok(ids) => (
                format!("GPU types   auto picks now, list price x {count} GPU, VRAM, stock:"),
                Ok(ids),
            ),
            Err(error) => (format!("GPU types   auto: {error}"), Err(())),
        },
    };
    let mut lines = vec![header];
    let ids = chosen.clone().unwrap_or_default();
    let shown = if spec.gpu_types.is_auto() {
        AUTO_SHOWN
    } else {
        ids.len()
    };
    let centers = spec.data_center_ids.list();
    let width = ids
        .iter()
        .take(shown)
        .map(|id| id.chars().count())
        .max()
        .unwrap_or(0);
    for id in ids.iter().take(shown) {
        lines.push(gpu_line(id, width, count, centers, gpus));
    }
    if ids.len() > shown {
        lines.push(format!("- and {} more", ids.len() - shown));
    }
    if let Some(Err(error)) = gpus {
        lines.push(format!("catalog     {error}"));
    }
    lines.push(center_line(spec, catalog, chosen.is_ok()));
    lines.push(max_hours_line(spec, catalog, &ids));
    lines
}

/// The GPU types `auto` picks now from `gpus`, in the data centers listed if
/// any.
fn auto_gpus(spec: &RunpodTarget, gpus: &[GpuType]) -> Result<Vec<String>, String> {
    let mut alone = spec.clone();
    if alone.data_center_ids.is_auto() {
        alone.data_center_ids = ListOrAuto::default();
    }
    resolve(&alone, gpus).map(|resolved| resolved.gpu_types.list().to_vec())
}

/// The line of the GPU type `id`, padded to `width`: its list price for
/// `count` GPUs, its VRAM and its stock in `centers`, once `gpus` are read.
fn gpu_line(id: &str, width: usize, count: u32, centers: &[String], gpus: Option<&Gpus>) -> String {
    let about = match gpus {
        None => "looking up the catalog...".to_string(),
        Some(Err(_)) => "catalog unread".to_string(),
        Some(Ok(listed)) => match listed.iter().find(|gpu| gpu.id == id) {
            None => "not in the catalog".to_string(),
            Some(gpu) => {
                let price = gpu.secure_price().map_or_else(
                    || "price unknown".to_string(),
                    |rate| format!("${:.2}/h", rate * f64::from(count)),
                );
                format!(
                    "{price:<13} {:>3} GB  {}",
                    gpu.memory,
                    stock(gpu, centers).name()
                )
            },
        },
    };
    format!("- {id:<width$}  {about}")
}

/// The data centers of `spec`, or those `auto` picks now from `gpus` when
/// `auto` found GPU types (`placed`).
fn center_line(spec: &RunpodTarget, gpus: Option<&[GpuType]>, placed: bool) -> String {
    let centers = match (&spec.data_center_ids, gpus) {
        (ListOrAuto::List(ids), _) if ids.is_empty() => "any".to_string(),
        (ListOrAuto::List(ids), _) => ids.join(", "),
        (ListOrAuto::Auto, Some(listed)) if placed => match resolve(spec, listed) {
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
    let rates: Vec<Option<f64>> = ids
        .iter()
        .map(|id| {
            gpus.and_then(|listed| listed.iter().find(|gpu| gpu.id == *id))
                .and_then(GpuType::secure_price)
        })
        .collect();
    let highest = rates.iter().flatten().copied().reduce(f64::max);
    let some_unknown = if rates.contains(&None) {
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
        boot_grace: Duration::from_secs(1800),
        retrieve_grace: Duration::from_secs(3600),
        data_center_ids: ListOrAuto::default(),
        network_volume_id: None,
    }
}

#[cfg(test)]
mod tests {
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::tui::snapshots::gpu_types;

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

    fn fixture() -> Result<Gpus, serde_json::Error> {
        gpu_types().map(Ok)
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
    fn listed_gpus_show_price_vram_and_stock_and_bound_the_run()
    -> Result<(), Box<dyn std::error::Error>> {
        let gpus = fixture()?;
        let lines = text(&plan(true), Some(&gpus));
        assert_eq!(
            lines[3],
            "GPU types   tried in order, list price x 2 GPU, VRAM, stock:"
        );
        assert_eq!(
            lines[4],
            "- NVIDIA GeForce RTX 4090  $1.38/h        24 GB  MEDIUM"
        );
        assert_eq!(lines[5], "- NVIDIA B300              not in the catalog");
        assert_eq!(lines[6], "datacenters any");
        assert_eq!(
            lines[7],
            "max_hours   6: the watchdog deletes the pod by then, about $8.28 at most at the \
             highest listed rate (some prices unknown)"
        );
        assert_eq!(cost_line(&lines), Some(7));
        let waiting = text(&plan(true), None);
        assert!(
            waiting[4].ends_with("looking up the catalog..."),
            "{waiting:?}"
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
        let lines = text(&plan(true), Some(&Ok(gpus)));
        assert_eq!(
            lines[4],
            "- NVIDIA GeForce RTX 4090  price unknown  24 GB  HIGH"
        );
        assert_eq!(
            lines[5],
            "- NVIDIA B300              price unknown 288 GB  UNKNOWN"
        );
        assert_eq!(
            lines[7],
            "max_hours   6: the watchdog deletes the pod by then"
        );
        Ok(())
    }

    #[test]
    fn auto_shows_what_it_picks_now_and_bounds_the_run_by_all_of_it()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut auto = runpod_spec(ListOrAuto::Auto, 1);
        let lines = text(&planned(auto.clone()), Some(&fixture()?));
        assert_eq!(
            lines[3],
            "GPU types   auto picks now, list price x 1 GPU, VRAM, stock:"
        );
        assert_eq!(
            lines[4],
            "- NVIDIA RTX 2000 Ada Generation  $0.24/h        16 GB  HIGH"
        );
        assert_eq!(
            lines[5],
            "- NVIDIA A40                      $0.40/h        48 GB  HIGH"
        );
        assert_eq!(lines[8], "- and 3 more");
        // The dearest picked, the H100, bounds the run even though not shown.
        assert!(
            lines[10].ends_with("about $17.94 at most at the highest listed rate"),
            "{lines:?}"
        );
        auto.min_vram_gb = Some(500);
        let lines = text(&planned(auto), Some(&fixture()?));
        assert!(
            lines[3].starts_with("GPU types   auto: no GPU type in stock"),
            "{lines:?}"
        );
        assert_eq!(lines[4], "datacenters any");
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
        let lines = text(&planned(listed.clone()), Some(&Ok(stocked()?)));
        assert!(lines[4].ends_with("48 GB  NONE"), "{lines:?}");
        assert!(lines[5].ends_with("48 GB  MEDIUM"), "{lines:?}");
        assert_eq!(lines[6], "datacenters US-TX-3");
        listed.data_center_ids = ListOrAuto::Auto;
        let listed = planned(listed);
        let lines = text(&listed, Some(&Ok(stocked()?)));
        assert_eq!(
            lines[6],
            "datacenters auto picks now: EU-RO-1, EU-SE-1, US-TX-3"
        );
        assert!(lines[4].ends_with("HIGH"), "overall stock: {lines:?}");
        let waiting = text(&listed, None);
        assert_eq!(
            waiting[6],
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
            Some(&failed),
        );
        assert_eq!(lines[4], format!("- {long}  catalog unread"));
        assert_eq!(
            lines[5],
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
        let lines = text(&plan(true), Some(&failed));
        assert_eq!(lines[4], "- NVIDIA GeForce RTX 4090  catalog unread");
        assert_eq!(
            lines[6],
            "catalog     cannot read the Runpod catalog: no Runpod API key"
        );
        assert_eq!(
            lines[8],
            "max_hours   6: the watchdog deletes the pod by then"
        );
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
        assert_eq!(cost_line(&lines), Some(7));
        assert_eq!(pinned(&lines), [3, 7]);
        Ok(())
    }

    #[test]
    fn a_plan_reads_the_training_section_and_the_split_files()
    -> Result<(), Box<dyn std::error::Error>> {
        let dir = crate::tui::snapshots::project()?;
        let none = prepare(dir.path(), EnvSource::Vars(Vec::new()));
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
        let plan = prepare(dir.path(), env)?;
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
        let plan = prepare(dir.path(), env.clone())?;
        let runpod = plan.runpod.clone().ok_or("no Runpod plan")?;
        assert_eq!((runpod.spec.gpu_count, runpod.spec.max_hours), (2, 6.0));
        assert_eq!(runpod.changed(), Vec::<&str>::new());
        let gpus = list_gpus(dir.path(), env, runpod.spec.gpu_count).await;
        let shown = text(&plan, Some(&gpus)).join("\n");
        assert!(
            shown.contains("- NVIDIA GeForce RTX 4090  $1.48/h        24 GB  LOW"),
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
        let error = lookup_gpus(dir.path(), env, 2, START_CATALOG_TIMEOUT)
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
        let gpus = lookup_gpus(dir.path(), env, 2, Duration::from_secs(1)).await;
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
        let gpus = list_gpus(dir.path(), env, 1).await;
        assert!(
            gpus.as_ref()
                .err()
                .is_some_and(|error| error.contains("no Runpod API key")),
            "{gpus:?}"
        );
        Ok(())
    }
}
