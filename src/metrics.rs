//! Prometheus metrics of a project, served at `GET /metrics` while a command holds
//! the project. Counters start from the stage history, then follow the events of
//! every bus the command builds with them as its [`Observer`], so they are
//! cumulative per project. Retries are not in the history: they restart at zero
//! with each process.

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::fmt;
use std::io;
use std::net::SocketAddr;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, SystemTime};

use hyper::body::Incoming;
use hyper::header::{CONTENT_TYPE, HeaderValue};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::registry::Registry;
use tokio::net::TcpListener;
use tokio::task::{JoinHandle, JoinSet};

use crate::events::{Event, Observer, Stage};
use crate::history::Entry;
use crate::llm::Usage;
use crate::runpod::{PodRecord, PodState};
use crate::runs::Runs;
use crate::train::TrainMetric;

/// Content type of the exposition `prometheus_client` writes.
pub const CONTENT_TYPE_VALUE: &str = "application/openmetrics-text; version=1.0.0; charset=utf-8";

/// Model label of tokens and cost whose stage named no model.
const UNKNOWN_MODEL: &str = "unknown";

/// Pause after a failed accept (out of file descriptors), so the loop never spins.
const ACCEPT_PAUSE: Duration = Duration::from_millis(100);

type FloatCounter = Counter<f64, AtomicU64>;
type FloatGauge = Gauge<f64, AtomicU64>;

#[derive(Debug, Clone, Hash, PartialEq, Eq, EncodeLabelSet)]
struct ItemLabels {
    stage: &'static str,
    result: &'static str,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, EncodeLabelSet)]
struct TokenLabels {
    stage: &'static str,
    model: String,
    direction: &'static str,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, EncodeLabelSet)]
struct ModelLabels {
    stage: &'static str,
    model: String,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, EncodeLabelSet)]
struct StageLabels {
    stage: &'static str,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, EncodeLabelSet)]
struct RunLabels {
    run_id: String,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, EncodeLabelSet)]
struct VersionLabels {
    version: &'static str,
}

/// What one bus said so far that labels its next events.
#[derive(Debug, Default)]
struct Followed {
    /// The model of each stage, from [`Event::StageModel`].
    models: BTreeMap<Stage, String>,
    /// Stages started and not finished yet.
    running: Vec<Stage>,
    /// The run whose metrics come next, from [`Event::RunWatched`].
    run_id: Option<String>,
}

/// The metric families of a project and their registry.
#[derive(Debug)]
pub struct Metrics {
    registry: Registry,
    items: Family<ItemLabels, Counter>,
    tokens: Family<TokenLabels, Counter>,
    cost: Family<ModelLabels, FloatCounter>,
    running: Family<StageLabels, Gauge>,
    retries: Family<StageLabels, Counter>,
    train_step: Family<RunLabels, Gauge>,
    train_loss: Family<RunLabels, FloatGauge>,
    eval_loss: Family<RunLabels, FloatGauge>,
    learning_rate: Family<RunLabels, FloatGauge>,
    spend: Family<RunLabels, FloatGauge>,
    /// Where the Runpod spend is read at each scrape; none leaves it out.
    runs: Option<Runs>,
    /// Per bus ID.
    buses: Mutex<BTreeMap<usize, Followed>>,
}

impl Metrics {
    /// The metrics, with the items, tokens and cost of `history` already counted.
    #[must_use]
    pub fn new(history: &[Entry]) -> Self {
        let metrics = Self {
            registry: Registry::default(),
            items: Family::default(),
            tokens: Family::default(),
            cost: Family::default(),
            running: Family::default(),
            retries: Family::default(),
            train_step: Family::default(),
            train_loss: Family::default(),
            eval_loss: Family::default(),
            learning_rate: Family::default(),
            spend: Family::default(),
            runs: None,
            buses: Mutex::default(),
        };
        let mut registry = Registry::default();
        registry.register(
            "overbrainer_stage_items",
            "Items of the pipeline stages, by result",
            metrics.items.clone(),
        );
        registry.register(
            "overbrainer_tokens",
            "Tokens sent (in) and received (out) by the pipeline stages",
            metrics.tokens.clone(),
        );
        registry.register(
            "overbrainer_cost_usd",
            "Cost of the pipeline stages in USD, when the model price is known",
            metrics.cost.clone(),
        );
        registry.register(
            "overbrainer_stage_running",
            "Pipeline stages running now",
            metrics.running.clone(),
        );
        registry.register(
            "overbrainer_item_retries",
            "Failed attempts retried, since this process started",
            metrics.retries.clone(),
        );
        registry.register(
            "overbrainer_train_step",
            "Last optimizer step of a training run",
            metrics.train_step.clone(),
        );
        registry.register(
            "overbrainer_train_loss",
            "Last training loss of a run",
            metrics.train_loss.clone(),
        );
        registry.register(
            "overbrainer_eval_loss",
            "Last evaluation loss of a run",
            metrics.eval_loss.clone(),
        );
        registry.register(
            "overbrainer_learning_rate",
            "Last learning rate of a run",
            metrics.learning_rate.clone(),
        );
        registry.register(
            "overbrainer_runpod_spend_usd",
            "Estimated Runpod spend of a run in USD: rate times uptime",
            metrics.spend.clone(),
        );
        let build = Family::<VersionLabels, Gauge>::default();
        build
            .get_or_create(&VersionLabels {
                version: env!("CARGO_PKG_VERSION"),
            })
            .set(1);
        registry.register("overbrainer_build_info", "Version of overbrainer", build);
        // The families are shared: the registry holds clones of the same series.
        let metrics = Self {
            registry,
            ..metrics
        };
        // Every stage shows, running or not.
        for stage in [
            Stage::Subtopics,
            Stage::Questions,
            Stage::Answers,
            Stage::Split,
        ] {
            metrics.stage_running(stage).set(0);
        }
        for entry in history {
            metrics.seed(entry);
        }
        metrics
    }

    /// Also shows the Runpod spend of the pods recorded in `runs`, read at each
    /// scrape.
    #[must_use]
    pub fn with_runs(mut self, runs: Runs) -> Self {
        self.runs = Some(runs);
        self
    }

    fn seed(&self, entry: &Entry) {
        let stage = entry.stage.name();
        for (result, count) in [
            ("done", entry.done),
            ("skipped", entry.skipped),
            ("failed", entry.failed),
            ("excluded", entry.excluded),
        ] {
            if count > 0 {
                self.item(stage, result, count);
            }
        }
        if let Some(model) = &entry.model {
            let usage = Usage {
                input_tokens: entry.input_tokens,
                output_tokens: entry.output_tokens,
            };
            self.spent(stage, model, usage, entry.cost);
        }
    }

    fn item(&self, stage: &'static str, result: &'static str, count: usize) {
        self.items
            .get_or_create(&ItemLabels { stage, result })
            .inc_by(u64::try_from(count).unwrap_or(u64::MAX));
    }

    fn spent(&self, stage: &'static str, model: &str, usage: Usage, cost: Option<f64>) {
        for (direction, count) in [("in", usage.input_tokens), ("out", usage.output_tokens)] {
            self.tokens
                .get_or_create(&TokenLabels {
                    stage,
                    model: model.to_string(),
                    direction,
                })
                .inc_by(count);
        }
        if let Some(cost) = cost {
            self.cost
                .get_or_create(&ModelLabels {
                    stage,
                    model: model.to_string(),
                })
                .inc_by(cost);
        }
    }

    fn follow(&self, followed: &mut Followed, event: &Event) {
        match event {
            Event::StageStarted { stage, .. } => {
                self.stage_running(*stage).inc();
                followed.running.push(*stage);
            },
            Event::StageModel { stage, model } => {
                followed.models.insert(*stage, model.clone());
            },
            Event::ItemDone {
                stage, usage, cost, ..
            } => match usage {
                Some(usage) => {
                    self.item(stage.name(), "done", 1);
                    let model = followed
                        .models
                        .get(stage)
                        .map_or(UNKNOWN_MODEL, String::as_str);
                    self.spent(stage.name(), model, *usage, *cost);
                },
                None => self.item(stage.name(), "skipped", 1),
            },
            Event::ItemFailed {
                stage, retryable, ..
            } => {
                if *retryable {
                    self.retries
                        .get_or_create(&StageLabels {
                            stage: stage.name(),
                        })
                        .inc();
                } else {
                    self.item(stage.name(), "failed", 1);
                }
            },
            Event::StageFinished { stage, stats } => {
                if stats.excluded > 0 {
                    self.item(stage.name(), "excluded", stats.excluded);
                }
                if let Some(index) = followed.running.iter().position(|s| s == stage) {
                    followed.running.remove(index);
                    self.stage_running(*stage).dec();
                }
            },
            Event::RunWatched { run_id } => followed.run_id = Some(run_id.clone()),
            Event::Metric(metric) => {
                if let Some(run_id) = &followed.run_id {
                    self.train(run_id, metric);
                }
            },
            Event::JobStatus(_) | Event::PodStatus(_) => {},
        }
    }

    fn stage_running(&self, stage: Stage) -> impl std::ops::Deref<Target = Gauge> + '_ {
        self.running.get_or_create(&StageLabels {
            stage: stage.name(),
        })
    }

    fn train(&self, run_id: &str, metric: &TrainMetric) {
        let labels = RunLabels {
            run_id: run_id.to_string(),
        };
        self.train_step
            .get_or_create(&labels)
            .set(i64::try_from(metric.step).unwrap_or(i64::MAX));
        for (family, value) in [
            (&self.train_loss, metric.loss),
            (&self.eval_loss, metric.eval_loss),
            (&self.learning_rate, metric.learning_rate),
        ] {
            if let Some(value) = value {
                family.get_or_create(&labels).set(value);
            }
        }
    }

    /// Sets the Runpod spend of every recorded pod as of `now`.
    fn read_spend(&self, now: SystemTime) {
        let Some(runs) = &self.runs else {
            return;
        };
        for (run_id, spend) in pod_spends(runs, now) {
            self.spend.get_or_create(&RunLabels { run_id }).set(spend);
        }
    }

    /// The text exposition of every family, the Runpod spend read now.
    ///
    /// # Errors
    ///
    /// Returns an error when a family cannot be written.
    pub fn encode(&self) -> Result<String, fmt::Error> {
        self.read_spend(SystemTime::now());
        let mut text = String::new();
        prometheus_client::encoding::text::encode(&mut text, &self.registry)?;
        Ok(text)
    }
}

impl Observer for Metrics {
    fn event(&self, bus: usize, event: &Event) {
        let mut buses = self.buses.lock().unwrap_or_else(PoisonError::into_inner);
        self.follow(buses.entry(bus).or_default(), event);
    }

    fn closed(&self, bus: usize) {
        let mut buses = self.buses.lock().unwrap_or_else(PoisonError::into_inner);
        // A stage that failed or was interrupted never finished.
        for stage in buses.remove(&bus).unwrap_or_default().running {
            self.stage_running(stage).dec();
        }
    }
}

/// The Runpod spend at `now` of each run of `runs` with a pod record. A record
/// that cannot be read is skipped.
fn pod_spends(runs: &Runs, now: SystemTime) -> Vec<(String, f64)> {
    let skipped = |id: &str, error: &crate::runs::RunsError| {
        tracing::debug!("metrics: skipping run {id}: {error}");
    };
    let records = runs.list_with(skipped).unwrap_or_else(|error| {
        tracing::debug!("metrics: cannot list the runs: {error}");
        Vec::new()
    });
    records
        .into_iter()
        .filter_map(|record| {
            let pod = PodRecord::load(runs, &record.id)
                .map_err(|error| skipped(&record.id, &error))
                .ok()??;
            Some((record.id, spend(&pod, now)?))
        })
        .collect()
}

/// The spend of `pod` at `now`: its estimate once deleted, else its rate times
/// its uptime so far.
fn spend(pod: &PodRecord, now: SystemTime) -> Option<f64> {
    if pod.state == PodState::Deleted {
        return pod.estimated_spend;
    }
    Some(pod.cost_per_hour? * pod.uptime(now)?.as_secs_f64() / 3600.0)
}

/// The running endpoint. Dropping it stops the server and closes its connections.
#[derive(Debug)]
pub struct MetricsServer {
    address: SocketAddr,
    server: JoinHandle<()>,
}

impl MetricsServer {
    /// The address it listens on, with the port the system chose for port 0.
    #[must_use]
    pub fn address(&self) -> SocketAddr {
        self.address
    }
}

impl Drop for MetricsServer {
    fn drop(&mut self) {
        self.server.abort();
    }
}

/// Serves `metrics` at `GET /metrics` on `address`. The metrics count what the
/// buses observed by them publish.
///
/// # Errors
///
/// Returns an error when `address` cannot be bound.
pub async fn serve(address: SocketAddr, metrics: Arc<Metrics>) -> io::Result<MetricsServer> {
    let listener = TcpListener::bind(address).await?;
    let address = listener.local_addr()?;
    let server = tokio::spawn(accept(listener, metrics));
    Ok(MetricsServer { address, server })
}

/// Serves each connection in its own task, all of them dropped with this one.
async fn accept(listener: TcpListener, metrics: Arc<Metrics>) {
    let mut connections = JoinSet::new();
    loop {
        tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => {
                    connections.spawn(connection(stream, Arc::clone(&metrics)));
                },
                Err(error) => {
                    tracing::debug!("cannot accept a metrics connection: {error}");
                    tokio::time::sleep(ACCEPT_PAUSE).await;
                },
            },
            Some(_) = connections.join_next(), if !connections.is_empty() => {},
        }
    }
}

async fn connection(stream: tokio::net::TcpStream, metrics: Arc<Metrics>) {
    let service = service_fn(move |request| {
        let response = respond(&request, &metrics);
        async move { Ok::<_, Infallible>(response) }
    });
    if let Err(error) = http1::Builder::new()
        .serve_connection(TokioIo::new(stream), service)
        .await
    {
        tracing::debug!("metrics connection: {error}");
    }
}

fn respond(request: &Request<Incoming>, metrics: &Metrics) -> Response<String> {
    if request.method() != Method::GET || request.uri().path() != "/metrics" {
        return status(StatusCode::NOT_FOUND, "not found\n");
    }
    match metrics.encode() {
        Ok(text) => {
            let mut response = Response::new(text);
            response
                .headers_mut()
                .insert(CONTENT_TYPE, HeaderValue::from_static(CONTENT_TYPE_VALUE));
            response
        },
        Err(error) => {
            tracing::debug!("cannot encode the metrics: {error}");
            status(
                StatusCode::INTERNAL_SERVER_ERROR,
                "cannot encode the metrics\n",
            )
        },
    }
}

fn status(code: StatusCode, text: &str) -> Response<String> {
    let mut response = Response::new(text.to_string());
    *response.status_mut() = code;
    response
}

#[cfg(test)]
mod tests {
    use std::time::UNIX_EPOCH;

    use super::*;
    use crate::events::{EventBus, StageStats};
    use crate::history::Status;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// The value of the sample `series` (name and labels, as written) in `text`.
    fn value(text: &str, series: &str) -> Option<f64> {
        text.lines()
            .find_map(|line| line.strip_prefix(series)?.strip_prefix(' '))
            .and_then(|value| value.parse().ok())
    }

    fn assert_value(text: &str, series: &str, expected: f64) {
        let found = value(text, series);
        assert!(
            found.is_some_and(|found| (found - expected).abs() < 1e-9),
            "{series}: expected {expected}, got {found:?} in\n{text}"
        );
    }

    fn entry(stage: Stage, model: Option<&str>, cost: Option<f64>) -> Entry {
        Entry {
            stage,
            started_at: "2026-09-28T10:00:00Z".into(),
            ended_at: "2026-09-28T10:01:00Z".into(),
            status: Status::Ok,
            provider: model.map(|_| "mock".into()),
            model: model.map(str::to_string),
            done: 3,
            skipped: 1,
            failed: 1,
            excluded: 2,
            input_tokens: 100,
            output_tokens: 40,
            cost,
            split: None,
        }
    }

    fn metric(step: u64, loss: Option<f64>, eval_loss: Option<f64>) -> TrainMetric {
        TrainMetric {
            time: 0.0,
            step,
            epoch: None,
            max_steps: None,
            loss,
            eval_loss,
            learning_rate: Some(2e-5),
            grad_norm: None,
        }
    }

    #[test]
    fn the_history_seeds_items_tokens_and_cost() -> TestResult {
        let metrics = Metrics::new(&[
            entry(Stage::Answers, Some("parent"), Some(0.5)),
            entry(Stage::Answers, Some("parent"), None),
            entry(Stage::Questions, Some("gen"), Some(0.25)),
            entry(Stage::Split, None, None),
        ]);
        let text = metrics.encode()?;
        let answers = r#"stage="answers",model="parent""#;
        assert_value(
            &text,
            &format!("overbrainer_tokens_total{{{answers},direction=\"in\"}}"),
            200.0,
        );
        assert_value(
            &text,
            &format!("overbrainer_tokens_total{{{answers},direction=\"out\"}}"),
            80.0,
        );
        // An unknown cost adds nothing.
        assert_value(
            &text,
            &format!("overbrainer_cost_usd_total{{{answers}}}"),
            0.5,
        );
        assert_value(
            &text,
            r#"overbrainer_cost_usd_total{stage="questions",model="gen"}"#,
            0.25,
        );
        for (result, count) in [
            ("done", 6.0),
            ("skipped", 2.0),
            ("failed", 2.0),
            ("excluded", 4.0),
        ] {
            assert_value(
                &text,
                &format!("overbrainer_stage_items_total{{stage=\"answers\",result=\"{result}\"}}"),
                count,
            );
        }
        // Split has no model: items only.
        assert_value(
            &text,
            r#"overbrainer_stage_items_total{stage="split",result="excluded"}"#,
            2.0,
        );
        assert!(
            !text.contains(r#"overbrainer_tokens_total{stage="split""#),
            "{text}"
        );
        assert_value(
            &text,
            &format!(
                "overbrainer_build_info{{version=\"{}\"}}",
                env!("CARGO_PKG_VERSION")
            ),
            1.0,
        );
        Ok(())
    }

    #[test]
    fn events_move_every_counter_and_gauge() -> TestResult {
        let metrics = Metrics::new(&[]);
        let usage = Usage {
            input_tokens: 10,
            output_tokens: 4,
        };
        for event in [
            Event::StageStarted {
                stage: Stage::Answers,
                total: 3,
            },
            Event::StageModel {
                stage: Stage::Answers,
                model: "parent".into(),
            },
            Event::ItemDone {
                stage: Stage::Answers,
                id: "a".into(),
                usage: Some(usage),
                cost: Some(0.125),
            },
            Event::ItemDone {
                stage: Stage::Answers,
                id: "b".into(),
                usage: Some(usage),
                cost: None,
            },
            Event::ItemDone {
                stage: Stage::Answers,
                id: "c".into(),
                usage: None,
                cost: None,
            },
            Event::ItemFailed {
                stage: Stage::Answers,
                id: "d".into(),
                error: "busy".into(),
                retryable: true,
            },
            Event::ItemFailed {
                stage: Stage::Answers,
                id: "d".into(),
                error: "busy".into(),
                retryable: false,
            },
        ] {
            metrics.event(1, &event);
        }
        let text = metrics.encode()?;
        assert_value(&text, r#"overbrainer_stage_running{stage="answers"}"#, 1.0);
        for (result, count) in [("done", 2.0), ("skipped", 1.0), ("failed", 1.0)] {
            assert_value(
                &text,
                &format!("overbrainer_stage_items_total{{stage=\"answers\",result=\"{result}\"}}"),
                count,
            );
        }
        assert_value(
            &text,
            r#"overbrainer_tokens_total{stage="answers",model="parent",direction="in"}"#,
            20.0,
        );
        assert_value(
            &text,
            r#"overbrainer_cost_usd_total{stage="answers",model="parent"}"#,
            0.125,
        );
        assert_value(
            &text,
            r#"overbrainer_item_retries_total{stage="answers"}"#,
            1.0,
        );
        metrics.event(
            1,
            &Event::StageFinished {
                stage: Stage::Answers,
                stats: StageStats {
                    excluded: 2,
                    ..StageStats::default()
                },
            },
        );
        let text = metrics.encode()?;
        assert_value(&text, r#"overbrainer_stage_running{stage="answers"}"#, 0.0);
        assert_value(
            &text,
            r#"overbrainer_stage_items_total{stage="answers",result="excluded"}"#,
            2.0,
        );
        Ok(())
    }

    #[test]
    fn training_metrics_carry_the_run_their_bus_watches() -> TestResult {
        let metrics = Metrics::new(&[]);
        // Before its bus names the run, a metric has no run to go to.
        metrics.event(2, &Event::Metric(metric(1, Some(9.0), None)));
        metrics.event(
            2,
            &Event::RunWatched {
                run_id: "20260928-100000-a1b2".into(),
            },
        );
        metrics.event(2, &Event::Metric(metric(5, Some(1.5), None)));
        metrics.event(2, &Event::Metric(metric(6, None, Some(1.25))));
        // Another bus watches no run.
        metrics.event(3, &Event::Metric(metric(7, Some(9.0), None)));
        let text = metrics.encode()?;
        let run = r#"{run_id="20260928-100000-a1b2"}"#;
        assert_value(&text, &format!("overbrainer_train_step{run}"), 6.0);
        assert_value(&text, &format!("overbrainer_train_loss{run}"), 1.5);
        assert_value(&text, &format!("overbrainer_eval_loss{run}"), 1.25);
        assert_value(&text, &format!("overbrainer_learning_rate{run}"), 2e-5);
        assert_eq!(text.matches("overbrainer_train_step{").count(), 1, "{text}");
        Ok(())
    }

    #[test]
    fn a_burst_far_larger_than_any_channel_is_fully_counted() -> TestResult {
        const BURST: usize = 30_000;
        let metrics = Arc::new(Metrics::new(&[]));
        let bus = EventBus::observed(8, Some(Arc::clone(&metrics) as Arc<dyn Observer>));
        let _behind = bus.subscribe();
        for index in 0..BURST {
            bus.publish(Event::ItemDone {
                stage: Stage::Answers,
                id: index.to_string(),
                usage: None,
                cost: None,
            });
        }
        let text = metrics.encode()?;
        assert_value(
            &text,
            r#"overbrainer_stage_items_total{stage="answers",result="skipped"}"#,
            30_000.0,
        );
        Ok(())
    }

    #[test]
    fn a_bus_that_ends_mid_stage_stops_it_running() -> TestResult {
        let metrics = Metrics::new(&[]);
        let started = Event::StageStarted {
            stage: Stage::Questions,
            total: 1,
        };
        metrics.event(1, &started);
        metrics.event(2, &started);
        metrics.closed(1);
        assert_value(
            &metrics.encode()?,
            r#"overbrainer_stage_running{stage="questions"}"#,
            1.0,
        );
        metrics.closed(2);
        metrics.closed(2);
        assert_value(
            &metrics.encode()?,
            r#"overbrainer_stage_running{stage="questions"}"#,
            0.0,
        );
        Ok(())
    }

    #[test]
    fn the_runpod_spend_comes_from_the_pod_records() -> TestResult {
        let dir = tempfile::tempdir()?;
        let runs = Runs::new(dir.path());
        let gone = crate::runs::create(&runs, "/workspace", "cloud")?;
        let mut pod = PodRecord::new(&gone.id, false, 1, "ssh-ed25519 AAAA");
        pod.state = PodState::Deleted;
        pod.estimated_spend = Some(0.64);
        pod.save(&runs)?;
        let live = crate::runs::create(&runs, "/workspace", "cloud")?;
        let mut pod = PodRecord::new(&live.id, false, 1, "ssh-ed25519 AAAA");
        pod.state = PodState::Running;
        pod.cost_per_hour = Some(0.5);
        pod.created_unix = Some(1000);
        pod.save(&runs)?;
        let local = crate::runs::create(&runs, "/work", "local")?;
        let metrics = Metrics::new(&[]).with_runs(runs.clone());
        let now = UNIX_EPOCH + Duration::from_secs(1000 + 2 * 3600);
        metrics.read_spend(now);
        let mut text = String::new();
        prometheus_client::encoding::text::encode(&mut text, &metrics.registry)?;
        let series = |id: &str| format!("overbrainer_runpod_spend_usd{{run_id=\"{id}\"}}");
        assert_value(&text, &series(&gone.id), 0.64);
        assert_value(&text, &series(&live.id), 1.0);
        assert_eq!(value(&text, &series(&local.id)), None);
        Ok(())
    }

    #[tokio::test]
    async fn the_endpoint_serves_the_families_and_nothing_else() -> TestResult {
        let history = [entry(Stage::Answers, Some("parent"), Some(0.5))];
        let metrics = Arc::new(Metrics::new(&history));
        let server = serve("127.0.0.1:0".parse()?, Arc::clone(&metrics)).await?;
        let base = format!("http://{}", server.address());
        let client = reqwest::Client::new();
        let response = client.get(format!("{base}/metrics")).send().await?;
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .map(HeaderValue::as_bytes),
            Some(CONTENT_TYPE_VALUE.as_bytes())
        );
        let text = response.text().await?;
        // A family shows once it has a series: these have one from the start.
        for family in [
            "overbrainer_stage_items",
            "overbrainer_tokens",
            "overbrainer_cost_usd",
            "overbrainer_stage_running",
            "overbrainer_build_info",
        ] {
            assert!(
                text.contains(&format!("# TYPE {family} ")),
                "{family}: {text}"
            );
        }
        assert!(text.ends_with("# EOF\n"), "{text}");
        for response in [
            client.get(format!("{base}/")).send().await?,
            client.get(format!("{base}/metrics/x")).send().await?,
            client.post(format!("{base}/metrics")).send().await?,
        ] {
            assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND);
        }
        // What an observed bus publishes shows at the next scrape.
        let running = r#"overbrainer_stage_running{stage="split"}"#;
        assert_value(&text, running, 0.0);
        let bus = EventBus::observed(1, Some(metrics));
        bus.publish(Event::StageStarted {
            stage: Stage::Split,
            total: 0,
        });
        let text = client
            .get(format!("{base}/metrics"))
            .send()
            .await?
            .text()
            .await?;
        assert_value(&text, running, 1.0);
        Ok(())
    }

    #[tokio::test]
    async fn dropping_the_server_frees_its_port() -> TestResult {
        let server = serve("127.0.0.1:0".parse()?, Arc::new(Metrics::new(&[]))).await?;
        let address = server.address();
        assert!(
            TcpListener::bind(address).await.is_err(),
            "the port is free"
        );
        drop(server);
        let mut bound = Err(io::Error::other("never tried"));
        for _ in 0..100 {
            bound = TcpListener::bind(address).await;
            if bound.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        bound?;
        Ok(())
    }
}
