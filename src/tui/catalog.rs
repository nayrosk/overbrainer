//! What the pickers list from the Runpod catalog and account, with the columns
//! of `overbrainer pod`, read in the background with the project's settings.

use std::path::Path;
use std::time::Duration;

use super::widgets::picker::{Entry, Mode, Spec};
use crate::config::{DEFAULT_RUNPOD_IMAGE, EnvSource, ListOrAuto};
use crate::runpod::{
    ApiError, Availability, DataCenter, GpuFilter, GpuType, NetworkVolume, RunpodClient, Template,
    select_gpus,
};

/// Total time a listing may take: templates can take several pages.
pub(super) const CATALOG_TIMEOUT: Duration = Duration::from_secs(20);

/// The ID of the volume picker's `none` entry: picking it unsets the volume.
pub(super) const NO_VOLUME: &str = "";

/// The ID of the template picker's `default` entry: picking it unsets the
/// image, so the pinned default applies.
pub(super) const DEFAULT_IMAGE: &str = "";

/// The column of a volume entry holding its data center.
const VOLUME_DATA_CENTER: usize = 3;

/// What a picker asks the catalog for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Query {
    /// What is listed.
    pub(super) kind: CatalogKind,
    /// GPUs per pod: stock is for that count.
    pub(super) gpu_count: u32,
    /// The GPU types chosen, whose stock a data center shows; none for `auto`.
    pub(super) gpu_types: Vec<String>,
}

/// What a listing read.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Listed {
    /// The picker's entries.
    pub(super) entries: Vec<Entry>,
    /// The GPU types, when the listing read them, for the field hints.
    pub(super) gpus: Vec<GpuType>,
}

/// What a picker lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CatalogKind {
    /// Secure Cloud GPU types, cheapest first.
    Gpus,
    /// Data centers, by ID.
    DataCenters,
    /// The account's network volumes, by name.
    Volumes,
    /// The account's pod templates, by name; picking one gives its image.
    Templates,
}

impl CatalogKind {
    /// What its picker shows besides the entries.
    pub(super) fn spec(self) -> Spec {
        match self {
            Self::Gpus => Spec {
                title: "GPU types",
                header: &["ID", "VRAM GB", "$/H", "MAX COUNT", "STOCK"],
                id_column: 0,
                mode: Mode::Multi,
                auto: Some("cheapest in stock"),
                empty: "no GPU type on Runpod's Secure Cloud",
            },
            Self::DataCenters => Spec {
                title: "Data centers",
                header: &["ID", "NAME", "REGION", "GPU STOCK"],
                id_column: 0,
                mode: Mode::Multi,
                auto: Some("GPUs in stock"),
                empty: "no data center in the catalog",
            },
            Self::Volumes => Spec {
                title: "Network volume",
                header: &["ID", "NAME", "SIZE GB", "DATA CENTER"],
                id_column: 0,
                mode: Mode::Single,
                auto: None,
                empty: "no network volume on this account",
            },
            Self::Templates => Spec {
                title: "Template image",
                header: &["ID", "NAME", "IMAGE"],
                id_column: 2,
                mode: Mode::Single,
                auto: None,
                empty: "no pod template on this account",
            },
        }
    }
}

/// What `query` asks, read from the Runpod account of the project in `dir`
/// (its settings read with `env`).
///
/// # Errors
///
/// Returns why nothing can be listed: the settings, no API key, the API, or
/// [`CATALOG_TIMEOUT`] passed. Only the client's fixed messages, never the key.
pub(super) async fn fetch(dir: &Path, env: EnvSource, query: Query) -> Result<Listed, String> {
    fetch_within(dir, env, query, CATALOG_TIMEOUT).await
}

/// [`fetch`], giving up after `limit`.
async fn fetch_within(
    dir: &Path,
    env: EnvSource,
    query: Query,
    limit: Duration,
) -> Result<Listed, String> {
    let lookup = async {
        let settings = crate::config::load(dir, env)?;
        let client = crate::cli::pod::client(&settings).await?;
        Ok::<_, anyhow::Error>(read(&client, &query).await?)
    };
    match tokio::time::timeout(limit, lookup).await {
        Ok(Ok(entries)) => Ok(entries),
        Ok(Err(error)) => Err(format!("cannot read the Runpod catalog: {error:#}")),
        Err(_) => Err("the Runpod catalog took too long to answer".to_string()),
    }
}

/// What `query` asks, listed with `client`.
async fn read(client: &RunpodClient, query: &Query) -> Result<Listed, ApiError> {
    let count = query.gpu_count;
    let listed = |entries| Listed {
        entries,
        gpus: Vec::new(),
    };
    Ok(match query.kind {
        CatalogKind::Gpus => {
            let gpus = client.list_gpu_types(count).await?;
            Listed {
                entries: gpu_entries(&gpus, count),
                gpus,
            }
        },
        CatalogKind::DataCenters => {
            let centers = client.list_data_centers().await?;
            let gpus = client.list_gpu_types(count).await?;
            Listed {
                entries: data_center_entries(&centers, &gpus, &query.gpu_types),
                gpus,
            }
        },
        CatalogKind::Volumes => listed(volume_entries(&client.list_network_volumes().await?)),
        CatalogKind::Templates => listed(template_entries(&client.list_templates().await?)),
    })
}

/// The Secure Cloud GPU types, cheapest first; one whose pod maximum is below
/// `gpu_count` cannot be chosen.
pub(super) fn gpu_entries(gpus: &[GpuType], gpu_count: u32) -> Vec<Entry> {
    select_gpus(gpus, &GpuFilter::default())
        .into_iter()
        .map(|gpu| Entry {
            columns: vec![
                gpu.id.clone(),
                gpu.memory.to_string(),
                gpu.secure_price()
                    .map_or_else(|| "-".to_string(), |price| format!("{price:.2}")),
                gpu.max_count.secure.to_string(),
                gpu.availability.name().to_string(),
            ],
            selectable: gpu.max_count.secure >= gpu_count,
            id: gpu.id,
        })
        .collect()
}

/// The data centers, by ID, with the stock of each `chosen` GPU type there in
/// chosen order (`NONE / LOW`), or without any, how many GPU types are in
/// stock there. Stock is each GPU's own, as `gpus` was listed for the pod's
/// GPU count, never the data center listing's, which ignores the count.
pub(super) fn data_center_entries(
    data_centers: &[DataCenter],
    gpus: &[GpuType],
    chosen: &[String],
) -> Vec<Entry> {
    let mut rows: Vec<&DataCenter> = data_centers.iter().collect();
    rows.sort_by(|a, b| a.id.cmp(&b.id));
    rows.into_iter()
        .map(|center| {
            let stock = if chosen.is_empty() {
                let count = gpus
                    .iter()
                    .filter(|gpu| gpu.on_secure_cloud() && gpu.stock_in(&center.id).is_in_stock())
                    .count();
                if count == 1 {
                    "1 GPU type".to_string()
                } else {
                    format!("{count} GPU types")
                }
            } else {
                chosen
                    .iter()
                    .map(|id| {
                        gpus.iter()
                            .find(|gpu| gpu.id == *id)
                            .map_or(Availability::None, |gpu| gpu.stock_in(&center.id))
                            .name()
                    })
                    .collect::<Vec<_>>()
                    .join(" / ")
            };
            Entry {
                id: center.id.clone(),
                columns: vec![
                    center.id.clone(),
                    center.name.clone(),
                    center.region.clone(),
                    stock,
                ],
                selectable: true,
            }
        })
        .collect()
}

/// The network volumes, by name, after a `none` entry that unsets it.
pub(super) fn volume_entries(volumes: &[NetworkVolume]) -> Vec<Entry> {
    let mut rows: Vec<&NetworkVolume> = volumes.iter().collect();
    rows.sort_by(|a, b| a.name.cmp(&b.name));
    let none = Entry {
        id: NO_VOLUME.to_string(),
        columns: vec![
            "none".to_string(),
            "no network volume".to_string(),
            String::new(),
            String::new(),
        ],
        selectable: true,
    };
    std::iter::once(none)
        .chain(rows.into_iter().map(|volume| Entry {
            id: volume.id.clone(),
            columns: vec![
                volume.id.clone(),
                volume.name.clone(),
                volume.size.to_string(),
                volume.data_center.clone(),
            ],
            selectable: true,
        }))
        .collect()
}

/// The data center of a volume entry; none for the `none` entry, or for a
/// volume the catalog does not list (the picker shows `-` there).
pub(super) fn volume_data_center(entry: &Entry) -> Option<&str> {
    entry
        .columns
        .get(VOLUME_DATA_CENTER)
        .map(String::as_str)
        .filter(|center| entry.id != NO_VOLUME && !center.is_empty() && *center != "-")
}

/// The pod templates, by name, after a `default` entry that unsets the image;
/// picking one gives its image, so one without an image cannot be chosen (its
/// entry's ID is then the template's).
pub(super) fn template_entries(templates: &[Template]) -> Vec<Entry> {
    let mut rows: Vec<&Template> = templates.iter().collect();
    rows.sort_by(|a, b| a.name.cmp(&b.name));
    let default = Entry {
        id: DEFAULT_IMAGE.to_string(),
        columns: vec![
            "default".to_string(),
            "the pinned Axolotl image".to_string(),
            DEFAULT_RUNPOD_IMAGE.to_string(),
        ],
        selectable: true,
    };
    std::iter::once(default)
        .chain(rows.into_iter().map(|template| Entry {
            id: if template.image.is_empty() {
                template.id.clone()
            } else {
                template.image.clone()
            },
            columns: vec![
                template.id.clone(),
                template.name.clone(),
                template.image.clone(),
            ],
            selectable: !template.image.is_empty(),
        }))
        .collect()
}

/// A Runpod target's fields as the Project view shows them, for the hints.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Sizing {
    /// `gpu_types`: `auto`, or IDs comma-separated.
    pub(super) gpu_types: String,
    /// `gpu_count`.
    pub(super) gpu_count: u32,
    /// `max_hours`, when set.
    pub(super) max_hours: Option<f64>,
    /// `max_price_per_hour`, when set.
    pub(super) max_price: Option<f64>,
}

impl Sizing {
    /// `gpu_types` as a list, or `auto`.
    fn gpu_types(&self) -> ListOrAuto {
        parse_list(&self.gpu_types)
    }
}

/// `text` as the Project view shows a list-or-auto field: `auto`, or items
/// comma-separated.
pub(super) fn parse_list(text: &str) -> ListOrAuto {
    if text.trim() == ListOrAuto::AUTO {
        return ListOrAuto::Auto;
    }
    ListOrAuto::List(
        text.split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(str::to_string)
            .collect(),
    )
}

/// The chosen GPU types `gpus` lists; none when one is not listed, since the
/// hint would then be wrong.
fn chosen_gpus<'a>(gpus: &'a [GpuType], ids: &[String]) -> Option<Vec<&'a GpuType>> {
    let chosen: Option<Vec<&GpuType>> = ids
        .iter()
        .map(|id| gpus.iter().find(|gpu| gpu.id == *id))
        .collect();
    chosen.filter(|chosen| !chosen.is_empty())
}

/// The hint of `gpu_count`: the most GPUs a pod can have with every chosen
/// type, or with `auto`, with any Secure Cloud type.
pub(super) fn gpu_count_hint(gpus: &[GpuType], sizing: &Sizing) -> Option<String> {
    match sizing.gpu_types() {
        ListOrAuto::Auto => gpus
            .iter()
            .filter(|gpu| gpu.on_secure_cloud())
            .map(|gpu| gpu.max_count.secure)
            .max()
            .map(|most| format!("at most {most} on Runpod's Secure Cloud")),
        ListOrAuto::List(ids) => chosen_gpus(gpus, &ids)?
            .iter()
            .map(|gpu| gpu.max_count.secure)
            .min()
            .map(|most| format!("at most {most} with the chosen types")),
    }
}

/// The hint of `max_hours`: the most the run can cost, `max_hours` at the
/// dearest chosen price for `gpu_count` GPUs; with `auto`, at
/// `max_price_per_hour`. None when a price is not known.
pub(super) fn cost_hint(gpus: &[GpuType], sizing: &Sizing) -> Option<String> {
    let hours = sizing.max_hours?;
    let (price, at) = match sizing.gpu_types() {
        ListOrAuto::Auto => (sizing.max_price?, "max_price_per_hour"),
        ListOrAuto::List(ids) => {
            let prices: Option<Vec<f64>> = chosen_gpus(gpus, &ids)?
                .iter()
                .map(|gpu| gpu.secure_price())
                .collect();
            (prices?.into_iter().fold(0.0, f64::max), "the chosen prices")
        },
    };
    let count = sizing.gpu_count;
    let most = hours * f64::from(count) * price;
    Some(format!(
        "at most ${most:.2} at {at} ({count} × ${price:.2}/h × {hours} h)"
    ))
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// The key of the stub account; it must never show.
    const KEY: &str = "rp_catalog_key_7331";

    /// A project whose API is at `server` with [`KEY`], or has no key.
    fn project(
        server: Option<&MockServer>,
    ) -> Result<(tempfile::TempDir, EnvSource), Box<dyn std::error::Error>> {
        let dir = crate::tui::snapshots::project()?;
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

    fn query(kind: CatalogKind, gpu_count: u32, gpu_types: &[&str]) -> Query {
        Query {
            kind,
            gpu_count,
            gpu_types: gpu_types.iter().map(|id| (*id).to_string()).collect(),
        }
    }

    fn ids(entries: &[Entry]) -> Vec<&str> {
        entries.iter().map(|entry| entry.id.as_str()).collect()
    }

    #[tokio::test]
    async fn gpus_are_read_for_the_gpu_count_cheapest_first() -> TestResult {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v2/catalog/gpus"))
            .and(query_param("count", "2"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"gpus": [
                {"id": "NVIDIA A40", "memory": 48, "price": {"secure": 0.4},
                 "maxCount": {"secure": 8}, "availability": "HIGH"},
                {"id": "NVIDIA L4", "memory": 24, "price": {"secure": 0.2},
                 "maxCount": {"secure": 1}, "availability": "LOW"},
                {"id": "Community only", "memory": 24, "secure": false}
            ]})))
            .expect(1)
            .mount(&server)
            .await;
        let (dir, env) = project(Some(&server))?;
        let listed = fetch(dir.path(), env, query(CatalogKind::Gpus, 2, &[])).await?;
        let entries = listed.entries;
        assert_eq!(ids(&entries), ["NVIDIA L4", "NVIDIA A40"]);
        assert_eq!(listed.gpus.len(), 3, "the GPU types are kept for the hints");
        assert_eq!(
            entries[1].columns,
            ["NVIDIA A40", "48", "0.40", "8", "HIGH"]
        );
        assert!(!entries[0].selectable, "one GPU at most per pod");
        assert!(entries[1].selectable);
        Ok(())
    }

    #[tokio::test]
    async fn volumes_and_templates_are_listed_in_order() -> TestResult {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v2/network-volumes"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"networkVolumes": [
                    {"id": "v2", "name": "zeta", "size": 50, "dataCenterId": "EU-RO-1"},
                    {"id": "v1", "name": "alpha", "size": 100, "dataCenterId": "US-KS-2"}
                ]})),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v2/templates"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "templates": [
                    {"id": "t2", "name": "zeta", "image": "img/z:1"},
                    {"id": "t1", "name": "alpha", "image": ""},
                    {"id": "s1", "name": "worker", "image": "img/s:1", "serverless": true}
                ],
                "pagination": {"hasNextPage": false}
            })))
            .mount(&server)
            .await;
        let (dir, env) = project(Some(&server))?;
        let volumes = fetch(dir.path(), env.clone(), query(CatalogKind::Volumes, 1, &[]))
            .await?
            .entries;
        assert_eq!(ids(&volumes), [NO_VOLUME, "v1", "v2"], "none first");
        assert_eq!(volumes[0].columns, ["none", "no network volume", "", ""]);
        assert_eq!(volumes[1].columns, ["v1", "alpha", "100", "US-KS-2"]);
        assert_eq!(volume_data_center(&volumes[1]), Some("US-KS-2"));
        assert_eq!(volume_data_center(&volumes[0]), None);
        let templates = fetch(dir.path(), env, query(CatalogKind::Templates, 1, &[]))
            .await?
            .entries;
        assert_eq!(
            ids(&templates),
            [DEFAULT_IMAGE, "t1", "img/z:1"],
            "the default first, then an image is picked"
        );
        assert_eq!(templates[0].columns[0], "default");
        assert!(templates[0].selectable);
        assert!(!templates[1].selectable, "no image, nothing to pick");
        Ok(())
    }

    #[tokio::test]
    async fn without_a_key_the_error_says_how_to_set_it() -> TestResult {
        let (dir, env) = project(None)?;
        let error = fetch(dir.path(), env, query(CatalogKind::Volumes, 1, &[]))
            .await
            .err()
            .ok_or("listed without a key")?;
        assert!(
            error.contains("no Runpod API key: set OVERBRAINER_RUNPOD__API_KEY"),
            "{error}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn a_refused_listing_never_shows_the_key() -> TestResult {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v2/network-volumes"))
            .respond_with(
                ResponseTemplate::new(401).set_body_string(format!("{{\"detail\": \"{KEY}\"}}")),
            )
            .mount(&server)
            .await;
        let (dir, env) = project(Some(&server))?;
        let error = fetch(dir.path(), env, query(CatalogKind::Volumes, 1, &[]))
            .await
            .err()
            .ok_or("listed on a 401")?;
        assert!(error.starts_with("cannot read the Runpod catalog: "));
        assert!(!error.contains(KEY), "the key shows in the error");
        Ok(())
    }

    #[tokio::test]
    async fn a_listing_past_its_time_gives_up() -> TestResult {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v2/catalog/datacenters"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(json!({"dataCenters": []}))
                    .set_delay(Duration::from_secs(30)),
            )
            .mount(&server)
            .await;
        let (dir, env) = project(Some(&server))?;
        let limit = Duration::from_millis(500);
        let error = fetch_within(
            dir.path(),
            env,
            query(CatalogKind::DataCenters, 1, &[]),
            limit,
        )
        .await
        .map(|listed| listed.entries);
        assert_eq!(
            error,
            Err("the Runpod catalog took too long to answer".to_string())
        );
        Ok(())
    }

    /// Two data centers, and the GPU types listed for `count=2` with their own
    /// stock per data center.
    async fn data_center_server() -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v2/catalog/datacenters"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"dataCenters": [
                    {"id": "US-KS-2", "name": "US Kansas 2", "region": "NORTH_AMERICA",
                     "gpuAvailability": [{"id": "NVIDIA A40", "availability": "HIGH"},
                                         {"id": "NVIDIA L4", "availability": "HIGH"}]},
                    {"id": "EU-RO-1", "name": "EU Romania 1", "region": "EUROPE"}
                ]})),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v2/catalog/gpus"))
            .and(query_param("count", "2"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"gpus": [
                {"id": "NVIDIA A40", "memory": 48, "price": {"secure": 0.4},
                 "maxCount": {"secure": 8}, "availability": "HIGH",
                 "dataCenters": [{"id": "US-KS-2", "availability": "LOW"},
                                 {"id": "EU-RO-1", "availability": "HIGH"}]},
                {"id": "NVIDIA L4", "memory": 24, "price": {"secure": 0.2},
                 "maxCount": {"secure": 8}, "availability": "NONE",
                 "dataCenters": [{"id": "US-KS-2", "availability": "NONE"}]}
            ]})))
            .expect(1)
            .mount(&server)
            .await;
        server
    }

    #[tokio::test]
    async fn data_centers_show_the_stock_of_the_chosen_gpus_for_the_count() -> TestResult {
        let server = data_center_server().await;
        let (dir, env) = project(Some(&server))?;
        let chosen = query(CatalogKind::DataCenters, 2, &["NVIDIA L4", "NVIDIA A40"]);
        let listed = fetch(dir.path(), env, chosen).await?;
        let centers = listed.entries;
        assert_eq!(ids(&centers), ["EU-RO-1", "US-KS-2"]);
        // The GPU's own count-scoped stock, not the data center listing's.
        assert_eq!(
            centers[1].columns,
            ["US-KS-2", "US Kansas 2", "NORTH_AMERICA", "NONE / LOW"]
        );
        assert_eq!(centers[0].columns[3], "NONE / HIGH");
        assert_eq!(listed.gpus.len(), 2);
        Ok(())
    }

    #[tokio::test]
    async fn without_chosen_gpus_data_centers_count_the_types_in_stock() -> TestResult {
        let server = data_center_server().await;
        let (dir, env) = project(Some(&server))?;
        let centers = fetch(dir.path(), env, query(CatalogKind::DataCenters, 2, &[]))
            .await?
            .entries;
        assert_eq!(centers[0].columns[3], "1 GPU type");
        assert_eq!(centers[1].columns[3], "1 GPU type", "L4 has none for 2");
        Ok(())
    }

    fn gpus() -> Result<Vec<GpuType>, serde_json::Error> {
        serde_json::from_value(json!([
            {"id": "A", "memory": 48, "price": {"secure": 0.5}, "maxCount": {"secure": 8}},
            {"id": "B", "memory": 80, "price": {"secure": 2.0}, "maxCount": {"secure": 4}},
            {"id": "C", "memory": 24, "maxCount": {"secure": 1}}
        ]))
    }

    fn sizing(gpu_types: &str, gpu_count: u32, max_hours: Option<f64>) -> Sizing {
        Sizing {
            gpu_types: gpu_types.to_string(),
            gpu_count,
            max_hours,
            max_price: None,
        }
    }

    #[test]
    fn the_gpu_count_hint_is_the_most_every_chosen_type_allows() -> TestResult {
        let gpus = gpus()?;
        let hint = |types: &str| gpu_count_hint(&gpus, &sizing(types, 1, None));
        assert_eq!(
            hint("A").as_deref(),
            Some("at most 8 with the chosen types")
        );
        assert_eq!(
            hint("A, B").as_deref(),
            Some("at most 4 with the chosen types")
        );
        assert_eq!(hint("Z"), None, "not in the catalog");
        assert_eq!(
            hint("auto").as_deref(),
            Some("at most 8 on Runpod's Secure Cloud")
        );
        Ok(())
    }

    #[test]
    fn the_max_hours_hint_is_the_most_the_run_can_cost() -> TestResult {
        let gpus = gpus()?;
        let hint = |sizing: &Sizing| cost_hint(&gpus, sizing);
        assert_eq!(
            hint(&sizing("A, B", 2, Some(6.0))).as_deref(),
            Some("at most $24.00 at the chosen prices (2 × $2.00/h × 6 h)")
        );
        assert_eq!(hint(&sizing("A", 1, None)), None, "no max_hours");
        assert_eq!(hint(&sizing("C", 1, Some(6.0))), None, "no price");
        assert_eq!(hint(&sizing("auto", 1, Some(2.0))), None, "no price limit");
        let mut capped = sizing("auto", 1, Some(2.0));
        capped.max_price = Some(1.5);
        assert_eq!(
            hint(&capped).as_deref(),
            Some("at most $3.00 at max_price_per_hour (1 × $1.50/h × 2 h)")
        );
        Ok(())
    }
}
