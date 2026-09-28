//! What the pickers list from the Runpod catalog and account, with the columns
//! of `overbrainer pod`, read in the background with the project's settings.

use std::path::Path;
use std::time::Duration;

use super::widgets::picker::{Entry, Mode, Spec};
use crate::config::EnvSource;
use crate::runpod::{
    ApiError, DataCenter, GpuFilter, GpuType, NetworkVolume, RunpodClient, Template, select_gpus,
};

/// Total time a listing may take: templates can take several pages.
pub(super) const CATALOG_TIMEOUT: Duration = Duration::from_secs(20);

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
                header: &["ID", "NAME", "REGION", "GPU TYPES IN STOCK"],
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

/// The entries of a `kind` picker, read from the Runpod account of the project
/// in `dir` (its settings read with `env`); GPU stock is for `gpu_count` GPUs.
///
/// # Errors
///
/// Returns why nothing can be listed: the settings, no API key, the API, or
/// [`CATALOG_TIMEOUT`] passed. Only the client's fixed messages, never the key.
pub(super) async fn fetch(
    dir: &Path,
    env: EnvSource,
    kind: CatalogKind,
    gpu_count: u32,
) -> Result<Vec<Entry>, String> {
    fetch_within(dir, env, kind, gpu_count, CATALOG_TIMEOUT).await
}

/// [`fetch`], giving up after `limit`.
async fn fetch_within(
    dir: &Path,
    env: EnvSource,
    kind: CatalogKind,
    gpu_count: u32,
    limit: Duration,
) -> Result<Vec<Entry>, String> {
    let lookup = async {
        let settings = crate::config::load(dir, env)?;
        let client = crate::cli::pod::client(&settings).await?;
        Ok::<_, anyhow::Error>(read(&client, kind, gpu_count).await?)
    };
    match tokio::time::timeout(limit, lookup).await {
        Ok(Ok(entries)) => Ok(entries),
        Ok(Err(error)) => Err(format!("cannot read the Runpod catalog: {error:#}")),
        Err(_) => Err("the Runpod catalog took too long to answer".to_string()),
    }
}

/// The entries of `kind`, listed with `client`.
async fn read(
    client: &RunpodClient,
    kind: CatalogKind,
    gpu_count: u32,
) -> Result<Vec<Entry>, ApiError> {
    Ok(match kind {
        CatalogKind::Gpus => gpu_entries(&client.list_gpu_types(gpu_count).await?, gpu_count),
        CatalogKind::DataCenters => data_center_entries(&client.list_data_centers().await?),
        CatalogKind::Volumes => volume_entries(&client.list_network_volumes().await?),
        CatalogKind::Templates => template_entries(&client.list_templates().await?),
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

/// The data centers, by ID, with how many GPU types they have in stock.
pub(super) fn data_center_entries(data_centers: &[DataCenter]) -> Vec<Entry> {
    let mut rows: Vec<&DataCenter> = data_centers.iter().collect();
    rows.sort_by(|a, b| a.id.cmp(&b.id));
    rows.into_iter()
        .map(|center| {
            let in_stock = center
                .gpu_availability
                .iter()
                .filter(|entry| entry.availability.is_in_stock())
                .count();
            Entry {
                id: center.id.clone(),
                columns: vec![
                    center.id.clone(),
                    center.name.clone(),
                    center.region.clone(),
                    in_stock.to_string(),
                ],
                selectable: true,
            }
        })
        .collect()
}

/// The network volumes, by name.
pub(super) fn volume_entries(volumes: &[NetworkVolume]) -> Vec<Entry> {
    let mut rows: Vec<&NetworkVolume> = volumes.iter().collect();
    rows.sort_by(|a, b| a.name.cmp(&b.name));
    rows.into_iter()
        .map(|volume| Entry {
            id: volume.id.clone(),
            columns: vec![
                volume.id.clone(),
                volume.name.clone(),
                volume.size.to_string(),
                volume.data_center.clone(),
            ],
            selectable: true,
        })
        .collect()
}

/// The pod templates, by name; picking one gives its image, so one without
/// an image cannot be chosen.
pub(super) fn template_entries(templates: &[Template]) -> Vec<Entry> {
    let mut rows: Vec<&Template> = templates.iter().collect();
    rows.sort_by(|a, b| a.name.cmp(&b.name));
    rows.into_iter()
        .map(|template| Entry {
            id: template.image.clone(),
            columns: vec![
                template.id.clone(),
                template.name.clone(),
                template.image.clone(),
            ],
            selectable: !template.image.is_empty(),
        })
        .collect()
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
        let entries = fetch(dir.path(), env, CatalogKind::Gpus, 2).await?;
        assert_eq!(ids(&entries), ["NVIDIA L4", "NVIDIA A40"]);
        assert_eq!(
            entries[1].columns,
            ["NVIDIA A40", "48", "0.40", "8", "HIGH"]
        );
        assert!(!entries[0].selectable, "one GPU at most per pod");
        assert!(entries[1].selectable);
        Ok(())
    }

    #[tokio::test]
    async fn data_centers_volumes_and_templates_are_listed_in_order() -> TestResult {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v2/catalog/datacenters"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"dataCenters": [
                    {"id": "US-KS-2", "name": "US Kansas 2", "region": "NORTH_AMERICA",
                     "gpuAvailability": [{"id": "NVIDIA A40", "availability": "HIGH"},
                                         {"id": "NVIDIA L4", "availability": "NONE"}]},
                    {"id": "EU-RO-1", "name": "EU Romania 1", "region": "EUROPE"}
                ]})),
            )
            .mount(&server)
            .await;
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
        let centers = fetch(dir.path(), env.clone(), CatalogKind::DataCenters, 1).await?;
        assert_eq!(ids(&centers), ["EU-RO-1", "US-KS-2"]);
        assert_eq!(
            centers[1].columns,
            ["US-KS-2", "US Kansas 2", "NORTH_AMERICA", "1"]
        );
        let volumes = fetch(dir.path(), env.clone(), CatalogKind::Volumes, 1).await?;
        assert_eq!(ids(&volumes), ["v1", "v2"]);
        assert_eq!(volumes[0].columns, ["v1", "alpha", "100", "US-KS-2"]);
        let templates = fetch(dir.path(), env, CatalogKind::Templates, 1).await?;
        assert_eq!(ids(&templates), ["", "img/z:1"], "an image is picked");
        assert!(!templates[0].selectable, "no image, nothing to pick");
        Ok(())
    }

    #[tokio::test]
    async fn without_a_key_the_error_says_how_to_set_it() -> TestResult {
        let (dir, env) = project(None)?;
        let error = fetch(dir.path(), env, CatalogKind::Volumes, 1)
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
        let error = fetch(dir.path(), env, CatalogKind::Volumes, 1)
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
        let error = fetch_within(dir.path(), env, CatalogKind::DataCenters, 1, limit).await;
        assert_eq!(
            error,
            Err("the Runpod catalog took too long to answer".to_string())
        );
        Ok(())
    }
}
