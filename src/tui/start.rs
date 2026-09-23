//! What `t` shows before a training run starts: the target, the model, the data,
//! and for Runpod the GPU types with their list prices and the most `max_hours`
//! can cost.

use std::path::Path;
use std::time::Duration;

use serde::de::IgnoredAny;

use crate::config::{Adapter, EnvSource, Runtime, Target};
use crate::dataset::{DataFiles, read};
use crate::train::reasoning_template_warning;

/// Total time the list prices may take; the dialog never waits for them.
pub(super) const PRICES_TIMEOUT: Duration = Duration::from_secs(10);

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
    pub(super) runpod: Option<RunpodPlan>,
    /// What the flow would warn about.
    pub(super) warnings: Vec<String>,
}

/// The pod a Runpod run would ask for.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct RunpodPlan {
    /// GPU types, tried in order.
    pub(super) gpu_types: Vec<String>,
    /// GPUs per pod.
    pub(super) gpu_count: u32,
    /// When the watchdog deletes the pod.
    pub(super) max_hours: f64,
}

/// The list price of each GPU type, per GPU and hour, when it could be read.
pub(super) type Prices = Vec<(String, Option<f64>)>;

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
        runpod: crate::runpod::RunpodTarget::from_target(target).map(|spec| RunpodPlan {
            gpu_types: spec.gpu_types,
            gpu_count: spec.gpu_count,
            max_hours: spec.max_hours,
        }),
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

/// The list price of each of `gpu_types` on the Runpod account of the project
/// in `dir` (its settings read with `env`), best effort: a type whose price
/// cannot be read has none, and the whole lookup gives up after
/// [`PRICES_TIMEOUT`]. Only the client's fixed messages reach the logs.
pub(super) async fn list_prices(dir: &Path, env: EnvSource, gpu_types: Vec<String>) -> Prices {
    list_prices_within(dir, env, gpu_types, PRICES_TIMEOUT).await
}

/// [`list_prices`], giving up after `limit`: the prices read by then are
/// kept, the types still unread have none.
async fn list_prices_within(
    dir: &Path,
    env: EnvSource,
    gpu_types: Vec<String>,
    limit: Duration,
) -> Prices {
    let mut prices = Vec::new();
    let lookup = async {
        let settings = crate::config::load(dir, env)?;
        let client = crate::cli::pod::client(&settings).await?;
        read_prices(&client, &gpu_types, &mut prices).await;
        Ok::<_, anyhow::Error>(())
    };
    let failure = match tokio::time::timeout(limit, lookup).await {
        Ok(Ok(())) => None,
        Ok(Err(error)) => Some(format!("cannot look up list prices: {error:#}")),
        Err(_) => Some("list prices took too long".to_string()),
    };
    if let Some(failure) = failure {
        tracing::warn!("{failure}");
    }
    let unread = gpu_types.into_iter().skip(prices.len());
    prices.extend(unread.map(|gpu| (gpu, None)));
    prices
}

/// Reads the list price of each of `gpu_types` into `prices`, in order, `None`
/// for one that cannot be read.
async fn read_prices(
    client: &crate::runpod::RunpodClient,
    gpu_types: &[String],
    prices: &mut Prices,
) {
    for gpu in gpu_types {
        let price = client.gpu_list_price(gpu).await.unwrap_or_else(|error| {
            tracing::warn!("cannot read the list price of {gpu}: {error}");
            None
        });
        prices.push((gpu.clone(), price));
    }
}

/// The usable list price of `gpu` in `prices`: none when it is not there, or
/// not a positive number.
fn listed(prices: &Prices, gpu: &str) -> Option<f64> {
    prices
        .iter()
        .find(|(name, _)| name == gpu)
        .and_then(|(_, price)| *price)
        .filter(|price| price.is_finite() && *price > 0.0)
}

/// The confirmation text of `plan`, with the list `prices` once looked up.
pub(super) fn text(plan: &StartPlan, prices: Option<&Prices>) -> Vec<String> {
    let mut text = vec![
        format!("target      {} ({})", plan.target, plan.kind),
        format!("model       {}", plan.model),
        format!(
            "data        data/train.jsonl {} examples, data/eval.jsonl {}",
            plan.train, plan.eval
        ),
    ];
    if let Some(runpod) = &plan.runpod {
        text.extend(gpu_lines(runpod, prices));
    }
    for warning in &plan.warnings {
        text.push(format!("warning     {warning}"));
    }
    text.push(
        "The run keeps going when you leave this view or quit; attach again here or with \
         `overbrainer train attach <run-id>`."
            .to_string(),
    );
    text
}

/// The GPU types of `runpod`, each with its list price times the GPU count
/// once `prices` are known, then `max_hours` with the most it can cost at the
/// highest listed rate.
fn gpu_lines(runpod: &RunpodPlan, prices: Option<&Prices>) -> Vec<String> {
    let count = f64::from(runpod.gpu_count);
    let mut lines = vec![format!(
        "GPU types   tried in order, list price x {} GPU:",
        runpod.gpu_count
    )];
    let rates: Vec<Option<f64>> = runpod
        .gpu_types
        .iter()
        .map(|gpu| prices.and_then(|prices| listed(prices, gpu)))
        .collect();
    for (gpu, rate) in runpod.gpu_types.iter().zip(&rates) {
        let price = match (prices, rate) {
            (None, _) => "looking up list prices...".to_string(),
            (Some(_), Some(rate)) => format!("${:.2}/h", rate * count),
            (Some(_), None) => "list price unknown".to_string(),
        };
        lines.push(format!("- {gpu:<26} {price}"));
    }
    let highest = rates.iter().flatten().copied().reduce(f64::max);
    let some_unknown = if rates.contains(&None) {
        " (some prices unknown)"
    } else {
        ""
    };
    let most = highest.map_or_else(String::new, |rate| {
        format!(
            ", about ${:.2} at most at the highest listed rate{some_unknown}",
            rate * count * runpod.max_hours
        )
    });
    lines.push(format!(
        "max_hours   {}: the watchdog deletes the pod by then{most}",
        runpod.max_hours
    ));
    lines
}

#[cfg(test)]
mod tests {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    fn plan(runpod: bool) -> StartPlan {
        StartPlan {
            target: "gpu_cloud".into(),
            kind: "runpod, Secure Cloud".into(),
            model: "Qwen/Qwen3-4B, qlora, 3 epochs, lr 2e-4".into(),
            train: 1234,
            eval: 137,
            runpod: runpod.then(|| RunpodPlan {
                gpu_types: vec!["NVIDIA GeForce RTX 4090".into(), "NVIDIA A40".into()],
                gpu_count: 2,
                max_hours: 6.0,
            }),
            warnings: Vec::new(),
        }
    }

    #[test]
    fn list_prices_count_every_gpu_and_bound_the_run() {
        let prices = vec![
            ("NVIDIA GeForce RTX 4090".to_string(), Some(0.74)),
            ("NVIDIA A40".to_string(), None),
        ];
        let text = text(&plan(true), Some(&prices));
        assert_eq!(text[4], "- NVIDIA GeForce RTX 4090    $1.48/h");
        assert_eq!(text[5], "- NVIDIA A40                 list price unknown");
        assert_eq!(
            text[6],
            "max_hours   6: the watchdog deletes the pod by then, about $8.88 at most at the \
             highest listed rate (some prices unknown)"
        );
        let all = vec![
            ("NVIDIA GeForce RTX 4090".to_string(), Some(0.74)),
            ("NVIDIA A40".to_string(), Some(0.44)),
        ];
        assert!(super::text(&plan(true), Some(&all))[6].ends_with("at the highest listed rate"));
        let waiting = super::text(&plan(true), None);
        assert!(waiting[4].ends_with("looking up list prices..."));
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
    async fn list_prices_are_read_from_the_catalog_best_effort()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v2/catalog/gpus/NVIDIA%20GeForce%20RTX%204090"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "NVIDIA GeForce RTX 4090",
                "price": {"community": 0.34, "secure": 0.74}
            })))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v2/catalog/gpus/NVIDIA%20A40"))
            .respond_with(ResponseTemplate::new(404).set_body_string(format!("no, {KEY}")))
            .expect(1)
            .mount(&server)
            .await;
        let (dir, env) = runpod_project(Some(&server))?;
        let plan = prepare(dir.path(), env.clone())?;
        let runpod = plan.runpod.clone().ok_or("no Runpod plan")?;
        assert_eq!((runpod.gpu_count, runpod.max_hours), (2, 6.0));
        let prices = list_prices(dir.path(), env, runpod.gpu_types).await;
        assert_eq!(
            prices,
            [
                ("NVIDIA GeForce RTX 4090".to_string(), Some(0.74)),
                ("NVIDIA A40".to_string(), None),
            ]
        );
        let shown = text(&plan, Some(&prices)).join("\n");
        assert!(shown.contains("$1.48/h") && shown.contains("list price unknown"));
        assert!(!shown.contains(KEY) && !format!("{plan:?}").contains(KEY));
        Ok(())
    }

    #[test]
    fn a_price_of_zero_or_less_is_unknown_and_never_bounds_the_run() {
        let prices = vec![
            ("NVIDIA GeForce RTX 4090".to_string(), Some(0.0)),
            ("NVIDIA A40".to_string(), Some(-1.0)),
        ];
        let text = text(&plan(true), Some(&prices));
        assert_eq!(text[4], "- NVIDIA GeForce RTX 4090    list price unknown");
        assert_eq!(text[5], "- NVIDIA A40                 list price unknown");
        assert_eq!(
            text[6],
            "max_hours   6: the watchdog deletes the pod by then"
        );
        let prices = vec![
            ("NVIDIA GeForce RTX 4090".to_string(), Some(f64::NAN)),
            ("NVIDIA A40".to_string(), Some(0.44)),
        ];
        let text = super::text(&plan(true), Some(&prices));
        assert_eq!(text[4], "- NVIDIA GeForce RTX 4090    list price unknown");
        assert!(
            text[6]
                .contains("about $5.28 at most at the highest listed rate (some prices unknown)"),
            "{}",
            text[6]
        );
    }

    /// A catalog answering `price` for `gpu`, after `delay`.
    async fn priced(server: &MockServer, gpu: &str, price: f64, delay: Duration) {
        Mock::given(method("GET"))
            .and(path(format!(
                "/v2/catalog/gpus/{}",
                gpu.replace(' ', "%20")
            )))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"id": gpu, "price": {"secure": price}}))
                    .set_delay(delay),
            )
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn a_lookup_past_its_time_keeps_the_prices_read_in_time()
    -> Result<(), Box<dyn std::error::Error>> {
        let server = MockServer::start().await;
        priced(&server, "NVIDIA GeForce RTX 4090", 0.74, Duration::ZERO).await;
        priced(&server, "NVIDIA A40", 0.44, Duration::from_secs(30)).await;
        let (dir, env) = runpod_project(Some(&server))?;
        let gpus = vec![
            "NVIDIA GeForce RTX 4090".to_string(),
            "NVIDIA A40".to_string(),
            "NVIDIA RTX A6000".to_string(),
        ];
        let started = std::time::Instant::now();
        let prices = list_prices_within(dir.path(), env, gpus, Duration::from_secs(2)).await;
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "gave up in time"
        );
        assert_eq!(
            prices,
            [
                ("NVIDIA GeForce RTX 4090".to_string(), Some(0.74)),
                ("NVIDIA A40".to_string(), None),
                ("NVIDIA RTX A6000".to_string(), None),
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_failed_price_read_never_logs_the_key() -> Result<(), Box<dyn std::error::Error>> {
        use tracing_subscriber::layer::SubscriberExt;

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v2/catalog/gpus/NVIDIA%20A40"))
            .respond_with(
                ResponseTemplate::new(404).set_body_string(format!("{{\"detail\": \"{KEY}\"}}")),
            )
            .mount(&server)
            .await;
        let (dir, env) = runpod_project(Some(&server))?;
        let logs = crate::logging::LogBuffer::new(100);
        let subscriber = tracing_subscriber::registry().with(logs.layer());
        let guard = tracing::subscriber::set_default(subscriber);
        let prices = list_prices(dir.path(), env, vec!["NVIDIA A40".into()]).await;
        drop(guard);
        assert_eq!(prices, [("NVIDIA A40".to_string(), None)]);
        let lines = logs.window(tracing::Level::TRACE, 1000, 0).lines;
        assert!(
            lines.iter().any(|line| line
                .message
                .contains("cannot read the list price of NVIDIA A40")),
            "{lines:?}"
        );
        for line in &lines {
            assert!(!line.message.contains(KEY), "{line:?}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn without_an_api_key_every_list_price_is_unknown()
    -> Result<(), Box<dyn std::error::Error>> {
        let (dir, env) = runpod_project(None)?;
        let prices = list_prices(dir.path(), env, vec!["NVIDIA A40".into()]).await;
        assert_eq!(prices, [("NVIDIA A40".to_string(), None)]);
        Ok(())
    }
}
