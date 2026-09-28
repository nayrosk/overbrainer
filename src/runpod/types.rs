//! The subset of the Runpod REST API (v2) that overbrainer sends and reads.

use std::collections::BTreeMap;
use std::fmt;

use secrecy::{ExposeSecret, SecretString};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A pod ID as Runpod assigns it: letters and digits only, so it is safe in a URL
/// path and a file.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct PodId(String);

/// A string rejected by [`PodId::new`]: empty, or holding anything but ASCII
/// letters and digits.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("`{0}` is not a Runpod pod ID")]
pub struct InvalidPodId(String);

impl PodId {
    /// Validates `id`.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidPodId`] when `id` is empty or holds anything but ASCII
    /// letters and digits.
    pub fn new(id: &str) -> Result<Self, InvalidPodId> {
        if !id.is_empty() && id.chars().all(|c| c.is_ascii_alphanumeric()) {
            Ok(Self(id.to_string()))
        } else {
            Err(InvalidPodId(id.to_string()))
        }
    }

    /// The ID.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PodId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for PodId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::new(&raw).map_err(serde::de::Error::custom)
    }
}

/// What Runpod says a pod is doing (`desiredStatus` in v1, `status` in v2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RemoteStatus {
    /// Being placed on a host.
    Provisioning,
    /// Starting its container.
    Starting,
    /// Running (the container may still be pulling its image).
    Running,
    /// Stopped: the container exited or the pod was stopped.
    Exited,
    /// Failed to start.
    Error,
    /// Terminated.
    Terminated,
    /// Anything this version of overbrainer does not know.
    #[default]
    #[serde(other)]
    Unknown,
}

impl RemoteStatus {
    /// The status as Runpod writes it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Provisioning => "PROVISIONING",
            Self::Starting => "STARTING",
            Self::Running => "RUNNING",
            Self::Exited => "EXITED",
            Self::Error => "ERROR",
            Self::Terminated => "TERMINATED",
            Self::Unknown => "UNKNOWN",
        }
    }

    /// Whether the pod will never serve SSH again without a restart.
    #[must_use]
    pub fn is_dead(self) -> bool {
        matches!(self, Self::Exited | Self::Error | Self::Terminated)
    }
}

/// A pod, as `GET /pods/{id}`, `GET /pods` and `POST /pods` return it. Only the
/// fields overbrainer uses are read.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Pod {
    /// Its ID.
    pub id: PodId,
    /// Its name, `overbrainer-<run-id>-<attempt>` for pods overbrainer creates.
    #[serde(default)]
    pub name: String,
    /// What it is doing.
    #[serde(default)]
    pub status: RemoteStatus,
    /// USD per hour actually billed; 0 once it no longer runs.
    #[serde(default)]
    pub cost: Option<f64>,
    /// The data center it was placed in.
    #[serde(default)]
    pub data_center_id: Option<String>,
    /// Its GPUs.
    #[serde(default)]
    pub gpu: Option<PodGpu>,
    /// How to reach its SSH server.
    #[serde(default)]
    pub ssh: Option<PodSsh>,
    /// Its environment, of which only overbrainer's run marker is read: the rest,
    /// which holds the pod's private host key, never enters overbrainer's memory.
    #[serde(default)]
    pub env: Option<PodEnv>,
    /// When it was created, as Runpod writes it.
    #[serde(default)]
    pub created_at: Option<String>,
}

impl Pod {
    /// The run marker of the pod, when overbrainer created it.
    #[must_use]
    pub fn run_id(&self) -> Option<&str> {
        self.env.as_ref()?.run_id.as_deref()
    }

    /// The public SSH endpoint, once Runpod has mapped port 22.
    #[must_use]
    pub fn direct(&self) -> Option<&SshDirect> {
        self.ssh.as_ref()?.direct.as_ref()
    }

    /// The GPU type, when known.
    #[must_use]
    pub fn gpu_type(&self) -> Option<&str> {
        self.gpu.as_ref()?.id.as_deref()
    }

    /// The hourly rate, when Runpod reports a non-zero one.
    #[must_use]
    pub fn rate(&self) -> Option<f64> {
        self.cost.filter(|cost| *cost > 0.0)
    }
}

/// The GPUs of a pod.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct PodGpu {
    /// GPU type ID.
    #[serde(default)]
    pub id: Option<String>,
    /// Number of GPUs.
    #[serde(default)]
    pub count: Option<u32>,
}

/// SSH access to a pod.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct PodSsh {
    /// The public IP and port mapped to the pod's port 22; null until assigned.
    #[serde(default)]
    pub direct: Option<SshDirect>,
}

/// The public endpoint of a pod's SSH server.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SshDirect {
    /// Public IP or host name.
    pub host: String,
    /// Public port, which changes when the pod is reset.
    pub port: u16,
    /// User to log in as.
    pub username: String,
}

/// The part of a pod's environment overbrainer reads back.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct PodEnv {
    /// `OVERBRAINER_RUN_ID`: the run the pod was created for.
    #[serde(rename = "OVERBRAINER_RUN_ID", default)]
    pub run_id: Option<String>,
}

/// One page of `GET /pods`.
#[derive(Debug, Clone, Deserialize)]
pub struct PodPage {
    /// The pods of this page.
    #[serde(default)]
    pub pods: Vec<Pod>,
    /// Where the next page starts.
    #[serde(default)]
    pub pagination: Option<Pagination>,
}

/// Paging of `GET /pods` and `GET /templates`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Pagination {
    /// Whether another page follows.
    #[serde(default)]
    pub has_next_page: bool,
    /// The cursor of the next page.
    #[serde(default)]
    pub next_cursor: Option<String>,
}

/// Deserializes a field that may be `null` as its type's default, so a catalog
/// entry with a `null` Runpod did not fill in still parses.
fn nullable<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Option::<T>::deserialize(deserializer).map(Option::unwrap_or_default)
}

/// A catalog stock band, as Runpod writes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Availability {
    /// Out of stock.
    None,
    /// Little stock.
    Low,
    /// Some stock.
    Medium,
    /// Plenty of stock.
    High,
    /// Absent, or a band this version of overbrainer does not know.
    #[default]
    #[serde(other)]
    Unknown,
}

impl Availability {
    /// The band as Runpod writes it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::Low => "LOW",
            Self::Medium => "MEDIUM",
            Self::High => "HIGH",
            Self::Unknown => "UNKNOWN",
        }
    }

    /// Whether some stock is reported: `LOW`, `MEDIUM` or `HIGH`. An unknown
    /// band is not stock.
    #[must_use]
    pub fn is_in_stock(self) -> bool {
        matches!(self, Self::Low | Self::Medium | Self::High)
    }
}

/// A GPU type of the catalog (`GET /catalog/gpus`).
/// Only the fields overbrainer uses are read; every one tolerates `null`.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuType {
    /// GPU type ID, as a pod create asks for it.
    #[serde(default, deserialize_with = "nullable")]
    pub id: String,
    /// Display name.
    #[serde(default, deserialize_with = "nullable")]
    pub name: String,
    /// `NVIDIA`, `AMD` or `UNKNOWN`.
    #[serde(default, deserialize_with = "nullable")]
    pub manufacturer: String,
    /// VRAM of one GPU, in GB.
    #[serde(default, deserialize_with = "nullable")]
    pub memory: u32,
    /// Whether Secure Cloud offers it; `None` when not said.
    #[serde(default)]
    pub secure: Option<bool>,
    /// List prices of one GPU.
    #[serde(default, deserialize_with = "nullable")]
    pub price: GpuPrice,
    /// Most GPUs of this type on one pod.
    #[serde(default, deserialize_with = "nullable")]
    pub max_count: GpuMaxCount,
    /// Overall pod stock for the requested GPU count.
    #[serde(default, deserialize_with = "nullable")]
    pub availability: Availability,
    /// Pod stock per data center offering it.
    #[serde(default, deserialize_with = "nullable")]
    pub data_centers: Vec<Stock>,
    /// CUDA versions its machines offer.
    #[serde(default, deserialize_with = "nullable")]
    pub cuda_versions: Vec<CudaVersion>,
}

impl GpuType {
    /// The Secure Cloud list price of one GPU, in USD per hour; none when it
    /// is not a positive number, which Runpod sends for a type it does not price.
    #[must_use]
    pub fn secure_price(&self) -> Option<f64> {
        self.price
            .secure
            .filter(|price| price.is_finite() && *price > 0.0)
    }

    /// Whether Secure Cloud offers it (true unless Runpod says otherwise).
    #[must_use]
    pub fn on_secure_cloud(&self) -> bool {
        self.secure != Some(false)
    }

    /// The stock of this GPU in `data_center`: `None` when it is not offered
    /// there.
    #[must_use]
    pub fn stock_in(&self, data_center: &str) -> Availability {
        self.data_centers
            .iter()
            .find(|entry| entry.id == data_center)
            .map_or(Availability::None, |entry| entry.availability)
    }
}

/// List prices of one GPU, in USD per hour.
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
pub struct GpuPrice {
    /// Secure Cloud.
    #[serde(default)]
    pub secure: Option<f64>,
    /// Community Cloud.
    #[serde(default)]
    pub community: Option<f64>,
}

/// Most GPUs of one type a pod can have, per cloud: a ceiling, not stock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
pub struct GpuMaxCount {
    /// Secure Cloud.
    #[serde(default, deserialize_with = "nullable")]
    pub secure: u32,
    /// Community Cloud.
    #[serde(default, deserialize_with = "nullable")]
    pub community: u32,
}

/// The stock of one catalog entry (a GPU type in a data center, or a data
/// center for a GPU type).
#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
pub struct Stock {
    /// The entry's ID.
    #[serde(default, deserialize_with = "nullable")]
    pub id: String,
    /// Its display name.
    #[serde(default, deserialize_with = "nullable")]
    pub name: String,
    /// Its stock band.
    #[serde(default, deserialize_with = "nullable")]
    pub availability: Availability,
}

/// A CUDA version offered by a GPU type's machines.
#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
pub struct CudaVersion {
    /// `major.minor`.
    #[serde(default, deserialize_with = "nullable")]
    pub version: String,
    /// Whether a machine on it has free capacity now.
    #[serde(default, deserialize_with = "nullable")]
    pub available: bool,
}

/// `GET /catalog/gpus`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GpuTypeList {
    /// The GPU types.
    #[serde(default, deserialize_with = "nullable")]
    pub gpus: Vec<GpuType>,
}

/// A data center of the catalog (`GET /catalog/datacenters`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DataCenter {
    /// Its ID, as `data_center_ids` names it.
    #[serde(default, deserialize_with = "nullable")]
    pub id: String,
    /// Display name.
    #[serde(default, deserialize_with = "nullable")]
    pub name: String,
    /// Continental region, such as `EUROPE`.
    #[serde(default, deserialize_with = "nullable")]
    pub region: String,
    /// Stock of each GPU type it offers.
    #[serde(default, deserialize_with = "nullable")]
    pub gpu_availability: Vec<Stock>,
}

/// `GET /catalog/datacenters`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DataCenterList {
    /// The data centers.
    #[serde(default, deserialize_with = "nullable")]
    pub data_centers: Vec<DataCenter>,
}

/// A network volume of the account (`GET /network-volumes`).
#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NetworkVolume {
    /// Its ID, as `network_volume_id` names it.
    #[serde(default, deserialize_with = "nullable")]
    pub id: String,
    /// Its name.
    #[serde(default, deserialize_with = "nullable")]
    pub name: String,
    /// Size in GB.
    #[serde(default, deserialize_with = "nullable")]
    pub size: u32,
    /// The data center it lives in.
    #[serde(default, alias = "dataCenterId", deserialize_with = "nullable")]
    pub data_center: String,
}

/// `GET /network-volumes`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NetworkVolumeList {
    /// The volumes.
    #[serde(default, deserialize_with = "nullable")]
    pub network_volumes: Vec<NetworkVolume>,
}

/// A template of the account (`GET /templates`). Its `env`, which may hold
/// secrets, is never read.
#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
pub struct Template {
    /// Its ID.
    #[serde(default, deserialize_with = "nullable")]
    pub id: String,
    /// Its name.
    #[serde(default, deserialize_with = "nullable")]
    pub name: String,
    /// Its container image.
    #[serde(default, deserialize_with = "nullable")]
    pub image: String,
    /// Whether it is for serverless workers rather than pods.
    #[serde(default, deserialize_with = "nullable")]
    pub serverless: bool,
}

/// One page of `GET /templates`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TemplatePage {
    /// The templates of this page.
    #[serde(default, deserialize_with = "nullable")]
    pub templates: Vec<Template>,
    /// Where the next page starts.
    #[serde(default)]
    pub pagination: Option<Pagination>,
}

/// The body of `POST /pods`: exactly the v2 fields overbrainer sets (the API
/// rejects any unknown field with a 422).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CreatePod {
    /// `overbrainer-<run-id>-<attempt>`.
    pub name: String,
    /// Container image, pinned by digest.
    pub image: String,
    /// Always `SECURE`.
    pub cloud: &'static str,
    /// The GPU asked for.
    pub gpu: GpuRequest,
    /// Container disk, in GB.
    pub disk: u32,
    /// Always `["22/tcp"]`.
    pub ports: Vec<String>,
    /// Always false: the bootstrap starts sshd with the per-run keys.
    pub start_ssh: bool,
    /// Data centers the pod may be placed in; any when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_center_ids: Option<Vec<String>>,
    /// The network volume, when one is configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mounts: Option<Mounts>,
    /// The pod's environment.
    pub env: CreateEnv,
    /// The pod's command, run after the image's entrypoint.
    pub cmd: Vec<String>,
}

/// The GPU part of [`CreatePod`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuRequest {
    /// GPU type ID.
    pub id: String,
    /// Number of GPUs.
    pub count: u32,
    /// Lowest CUDA version the host driver must support.
    pub min_cuda_version: &'static str,
}

/// Volumes of [`CreatePod`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Mounts {
    /// Network volumes.
    pub network: Vec<NetworkMount>,
}

/// A network volume and where it is mounted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NetworkMount {
    /// Volume ID.
    pub volume_id: String,
    /// Mount path in the container.
    pub path: String,
}

/// The environment of a new pod: plain variables, plus the pod's private host key,
/// which only the JSON serializer ever sees. `Debug` shows variable names only.
#[derive(Clone)]
pub struct CreateEnv {
    /// Variables that hold no secret.
    pub plain: BTreeMap<String, String>,
    /// Name of the variable holding the host key.
    pub host_key_name: &'static str,
    /// The base64 of the OpenSSH private host key.
    pub host_key: SecretString,
}

impl Serialize for CreateEnv {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.plain.len() + 1))?;
        for (name, value) in &self.plain {
            map.serialize_entry(name, value)?;
        }
        map.serialize_entry(self.host_key_name, self.host_key.expose_secret())?;
        map.end()
    }
}

impl fmt::Debug for CreateEnv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self
            .plain
            .keys()
            .map(String::as_str)
            .chain([self.host_key_name])
            .collect();
        f.debug_struct("CreateEnv")
            .field("names", &names)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn request() -> CreatePod {
        CreatePod {
            name: "overbrainer-r1-1".into(),
            image: "img@sha256:abc".into(),
            cloud: "SECURE",
            gpu: GpuRequest {
                id: "NVIDIA A40".into(),
                count: 1,
                min_cuda_version: "13.0",
            },
            disk: 50,
            ports: vec!["22/tcp".into()],
            start_ssh: false,
            data_center_ids: None,
            mounts: None,
            env: CreateEnv {
                plain: BTreeMap::from([("OVERBRAINER_RUN_ID".into(), "r1".into())]),
                host_key_name: "OVERBRAINER_HOST_KEY",
                host_key: SecretString::from("c2VjcmV0LWhvc3Qta2V5"),
            },
            cmd: vec!["bash".into(), "-c".into(), "true".into()],
        }
    }

    #[test]
    fn the_request_uses_the_v2_field_names_only() -> TestResult {
        let json = serde_json::to_value(request())?;
        let keys: Vec<&str> = json
            .as_object()
            .ok_or("not an object")?
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            vec![
                "cloud", "cmd", "disk", "env", "gpu", "image", "name", "ports", "startSsh"
            ]
        );
        assert_eq!(json["gpu"]["minCudaVersion"], "13.0");
        assert_eq!(json["env"]["OVERBRAINER_HOST_KEY"], "c2VjcmV0LWhvc3Qta2V5");
        for v1 in [
            "imageName",
            "gpuTypeIds",
            "interruptible",
            "supportPublicIp",
            "cloudType",
        ] {
            assert!(json.get(v1).is_none(), "{v1}");
        }
        let mut with_volume = request();
        with_volume.data_center_ids = Some(vec!["EU-RO-1".into()]);
        with_volume.mounts = Some(Mounts {
            network: vec![NetworkMount {
                volume_id: "vol1".into(),
                path: "/workspace/data".into(),
            }],
        });
        let json = serde_json::to_value(with_volume)?;
        assert_eq!(json["dataCenterIds"][0], "EU-RO-1");
        assert_eq!(json["mounts"]["network"][0]["volumeId"], "vol1");
        Ok(())
    }

    #[test]
    fn debug_never_shows_the_host_key() {
        let text = format!("{:?}", request());
        assert!(!text.contains("c2VjcmV0LWhvc3Qta2V5"), "{text}");
        assert!(text.contains("OVERBRAINER_HOST_KEY"), "{text}");
    }

    #[test]
    fn a_pod_reads_only_the_run_marker_from_its_env() -> TestResult {
        let pod: Pod = serde_json::from_str(
            r#"{"id": "k3x9abc", "name": "overbrainer-r1-1", "status": "RUNNING", "cost": 0.53,
                "dataCenterId": "EU-RO-1", "gpu": {"id": "NVIDIA A40", "count": 1},
                "ssh": {"direct": {"host": "203.0.113.7", "port": 40122, "username": "root"}},
                "env": {"OVERBRAINER_RUN_ID": "r1", "OVERBRAINER_HOST_KEY": "c2VjcmV0"},
                "createdAt": "2026-09-22T14:30:08Z", "somethingNew": true}"#,
        )?;
        assert_eq!(pod.id.as_str(), "k3x9abc");
        assert_eq!(pod.run_id(), Some("r1"));
        assert_eq!(pod.rate(), Some(0.53));
        assert_eq!(pod.gpu_type(), Some("NVIDIA A40"));
        assert_eq!(pod.direct().map(|direct| direct.port), Some(40122));
        assert!(!format!("{pod:?}").contains("c2VjcmV0"));
        Ok(())
    }

    #[test]
    fn unknown_statuses_and_missing_fields_are_tolerated() -> TestResult {
        let pod: Pod = serde_json::from_str(
            r#"{"id": "p1", "status": "MIGRATING", "ssh": {"direct": null}, "cost": 0.0}"#,
        )?;
        assert_eq!(pod.status, RemoteStatus::Unknown);
        assert_eq!(pod.direct(), None);
        assert_eq!(pod.rate(), None);
        assert_eq!(pod.run_id(), None);
        Ok(())
    }

    #[test]
    fn pod_ids_are_letters_and_digits() {
        assert!(PodId::new("k3x9abc").is_ok());
        for bad in ["", "../x", "a/b", "a b", "a-b"] {
            assert!(PodId::new(bad).is_err(), "{bad}");
        }
        assert!(serde_json::from_str::<Pod>(r#"{"id": "../etc"}"#).is_err());
    }

    #[test]
    fn a_rejected_pod_id_is_an_error_of_its_own() {
        let error: Option<InvalidPodId> = PodId::new("a-b").err();
        let source: Option<&dyn std::error::Error> =
            error.as_ref().map(|e| e as &dyn std::error::Error);
        assert_eq!(
            source.map(ToString::to_string).as_deref(),
            Some("`a-b` is not a Runpod pod ID")
        );
    }
}
