//! A `runpod` target with its defaults applied.

use std::time::Duration;

use crate::config::{DEFAULT_RUNPOD_IMAGE, DEFAULT_RUNPOD_VENV, Target};
use crate::exec::JobRuntime;

use super::JOB_ENV;

/// Directory of the run directories on a pod without a network volume. Nothing
/// else in the image uses it, and `/workspace` itself is never a mount point.
pub const WORKDIR: &str = "/workspace/overbrainer";
/// Where a network volume is mounted.
pub const VOLUME_MOUNT: &str = "/workspace/data";
/// Directory of the run directories on a network volume, so they outlive the pod.
pub const VOLUME_WORKDIR: &str = "/workspace/data/overbrainer";
/// Lowest CUDA version of the host driver, for [`DEFAULT_RUNPOD_IMAGE`]'s CUDA 13.
pub const MIN_CUDA_VERSION: &str = "13.0";

/// A `runpod` target of `overbrainer.toml`, defaults applied.
#[derive(Debug, Clone, PartialEq)]
pub struct RunpodTarget {
    /// GPU types, tried in order.
    pub gpu_types: Vec<String>,
    /// GPUs per pod.
    pub gpu_count: u32,
    /// Container image.
    pub image: String,
    /// Virtual environment holding `bin/axolotl` on the pod.
    pub venv: String,
    /// Container disk, in GB.
    pub container_disk_gb: u32,
    /// Hours after which the watchdog deletes the pod.
    pub max_hours: f64,
    /// How long the watchdog waits for a job to start.
    pub boot_grace: Duration,
    /// How long the watchdog keeps a pod whose ended job was not retrieved.
    pub retrieve_grace: Duration,
    /// Allowed data centers; any when empty.
    pub data_center_ids: Vec<String>,
    /// Network volume, if any.
    pub network_volume_id: Option<String>,
}

impl RunpodTarget {
    /// The target, when it is a `runpod` one.
    #[must_use]
    pub fn from_target(target: &Target) -> Option<Self> {
        let Target::Runpod {
            gpu_types,
            gpu_count,
            image,
            venv,
            container_disk_gb,
            max_hours,
            boot_grace_minutes,
            retrieve_grace_minutes,
            data_center_ids,
            network_volume_id,
        } = target
        else {
            return None;
        };
        Some(Self {
            gpu_types: gpu_types.clone(),
            gpu_count: *gpu_count,
            image: image
                .clone()
                .unwrap_or_else(|| DEFAULT_RUNPOD_IMAGE.to_string()),
            venv: venv
                .clone()
                .unwrap_or_else(|| DEFAULT_RUNPOD_VENV.to_string()),
            container_disk_gb: *container_disk_gb,
            max_hours: *max_hours,
            boot_grace: Duration::from_secs(u64::from(*boot_grace_minutes) * 60),
            retrieve_grace: Duration::from_secs(u64::from(*retrieve_grace_minutes) * 60),
            data_center_ids: data_center_ids.clone(),
            network_volume_id: network_volume_id.clone(),
        })
    }

    /// Directory of the run directories on the pod.
    #[must_use]
    pub fn workdir(&self) -> &'static str {
        if self.network_volume_id.is_some() {
            VOLUME_WORKDIR
        } else {
            WORKDIR
        }
    }

    /// How jobs run on the pod: the image's virtual environment, after the
    /// environment file the bootstrap writes.
    #[must_use]
    pub fn runtime(&self) -> JobRuntime {
        JobRuntime::Native {
            venv: Some(self.venv.clone()),
            env_file: Some(JOB_ENV.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_target(volume: Option<&str>) -> Target {
        Target::Runpod {
            gpu_types: vec!["NVIDIA A40".into()],
            gpu_count: 1,
            image: None,
            venv: None,
            container_disk_gb: 50,
            max_hours: 6.0,
            boot_grace_minutes: 30,
            retrieve_grace_minutes: 60,
            data_center_ids: vec!["EU-RO-1".into()],
            network_volume_id: volume.map(str::to_string),
        }
    }

    #[test]
    fn defaults_are_applied() -> Result<(), &'static str> {
        let target = RunpodTarget::from_target(&config_target(None)).ok_or("not runpod")?;
        assert_eq!(target.image, DEFAULT_RUNPOD_IMAGE);
        assert_eq!(target.venv, "/workspace/axolotl-venv");
        assert_eq!(target.boot_grace, Duration::from_secs(1800));
        assert_eq!(target.retrieve_grace, Duration::from_secs(3600));
        assert_eq!(target.workdir(), "/workspace/overbrainer");
        assert_eq!(
            target.runtime(),
            JobRuntime::Native {
                venv: Some("/workspace/axolotl-venv".into()),
                env_file: Some("/etc/overbrainer/job.env".into()),
            }
        );
        let on_volume =
            RunpodTarget::from_target(&config_target(Some("vol1"))).ok_or("not runpod")?;
        assert_eq!(on_volume.workdir(), "/workspace/data/overbrainer");
        Ok(())
    }
}
