//! The gauges of the machine a watched run is on, from its [`SystemSample`]s:
//! labelled `run_id`, in base units, gone once the run's bus closes.

use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::Gauge;
use prometheus_client::registry::Registry;

use super::{FloatGauge, RunLabels, register};
use crate::system::{SystemSample, float};

#[derive(Debug, Clone, Hash, PartialEq, Eq, EncodeLabelSet)]
struct MountLabels {
    run_id: String,
    mount: String,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, EncodeLabelSet)]
struct GpuLabels {
    run_id: String,
    gpu: String,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, EncodeLabelSet)]
struct GpuInfoLabels {
    run_id: String,
    gpu: String,
    name: String,
}

/// The series one sample of a run set, to remove them before the next one or
/// once its bus closes.
#[derive(Debug, Default)]
pub(super) struct SystemSeries {
    run_id: String,
    mounts: Vec<String>,
    /// Index and name of each GPU.
    gpus: Vec<(String, String)>,
}

/// The families of the target's machine.
#[derive(Debug)]
pub(super) struct SystemFamilies {
    disk_used: Family<MountLabels, FloatGauge>,
    disk_size: Family<MountLabels, FloatGauge>,
    cpu_usage: Family<RunLabels, FloatGauge>,
    cpu_load1: Family<RunLabels, FloatGauge>,
    cpus: Family<RunLabels, FloatGauge>,
    memory_used: Family<RunLabels, FloatGauge>,
    memory_limit: Family<RunLabels, FloatGauge>,
    sampled: Family<RunLabels, FloatGauge>,
    gpu_utilization: Family<GpuLabels, FloatGauge>,
    gpu_memory_used: Family<GpuLabels, FloatGauge>,
    gpu_memory_total: Family<GpuLabels, FloatGauge>,
    gpu_temperature: Family<GpuLabels, FloatGauge>,
    gpu_power: Family<GpuLabels, FloatGauge>,
    gpu_power_limit: Family<GpuLabels, FloatGauge>,
    gpu_info: Family<GpuInfoLabels, Gauge>,
}

impl SystemFamilies {
    /// The families, registered in `registry`.
    pub(super) fn new(registry: &mut Registry) -> Self {
        Self {
            disk_used: register(
                registry,
                "overbrainer_target_disk_used_bytes",
                "Bytes used on a file system of a run's target: its run directory's, and /",
            ),
            disk_size: register(
                registry,
                "overbrainer_target_disk_size_bytes",
                "Size of a file system of a run's target",
            ),
            cpu_usage: register(
                registry,
                "overbrainer_target_cpu_usage_ratio",
                "Share of the CPUs of a run's target busy since the previous sample",
            ),
            cpu_load1: register(
                registry,
                "overbrainer_target_cpu_load1",
                "Load average over one minute on a run's target",
            ),
            cpus: register(
                registry,
                "overbrainer_target_cpus",
                "CPUs of a run's target: its container's quota, else nproc",
            ),
            memory_used: register(
                registry,
                "overbrainer_target_memory_used_bytes",
                "Memory used on a run's target, page cache that can be dropped left out",
            ),
            memory_limit: register(
                registry,
                "overbrainer_target_memory_limit_bytes",
                "Memory of a run's target: its container's limit, else the machine's",
            ),
            sampled: register(
                registry,
                "overbrainer_target_sample_timestamp_seconds",
                "When the target of a run was last sampled, in Unix seconds",
            ),
            gpu_utilization: register(
                registry,
                "overbrainer_gpu_utilization_ratio",
                "Share of time a GPU of a run's target ran a kernel",
            ),
            gpu_memory_used: register(
                registry,
                "overbrainer_gpu_memory_used_bytes",
                "Memory used on a GPU of a run's target",
            ),
            gpu_memory_total: register(
                registry,
                "overbrainer_gpu_memory_total_bytes",
                "Memory of a GPU of a run's target",
            ),
            gpu_temperature: register(
                registry,
                "overbrainer_gpu_temperature_celsius",
                "Temperature of a GPU of a run's target",
            ),
            gpu_power: register(
                registry,
                "overbrainer_gpu_power_watts",
                "Power drawn by a GPU of a run's target",
            ),
            gpu_power_limit: register(
                registry,
                "overbrainer_gpu_power_limit_watts",
                "Power limit of a GPU of a run's target",
            ),
            gpu_info: register(
                registry,
                "overbrainer_gpu_info",
                "A GPU of a run's target, with its model name",
            ),
        }
    }

    /// Replaces the series `before` set with those of `sample`, taken on the
    /// target of run `run_id`; returns them.
    pub(super) fn set(
        &self,
        run_id: &str,
        sample: &SystemSample,
        before: Option<&SystemSeries>,
    ) -> SystemSeries {
        if let Some(before) = before {
            self.remove(before);
        }
        let run = RunLabels {
            run_id: run_id.to_string(),
        };
        let mut series = SystemSeries {
            run_id: run_id.to_string(),
            ..SystemSeries::default()
        };
        for disk in &sample.disks {
            let labels = MountLabels {
                run_id: run_id.to_string(),
                mount: disk.mount.clone(),
            };
            self.disk_used
                .get_or_create(&labels)
                .set(float(disk.used_bytes));
            self.disk_size
                .get_or_create(&labels)
                .set(float(disk.size_bytes));
            series.mounts.push(disk.mount.clone());
        }
        if let Some(cpu) = &sample.cpu {
            for (family, value) in [
                (&self.cpu_usage, cpu.usage),
                (&self.cpu_load1, cpu.load1),
                (&self.cpus, cpu.cpus),
            ] {
                if let Some(value) = value {
                    family.get_or_create(&run).set(value);
                }
            }
        }
        if let Some(memory) = &sample.memory {
            self.memory_used
                .get_or_create(&run)
                .set(float(memory.used_bytes));
            self.memory_limit
                .get_or_create(&run)
                .set(float(memory.limit_bytes));
        }
        if let Ok(since) = sample.at.duration_since(std::time::UNIX_EPOCH) {
            self.sampled.get_or_create(&run).set(since.as_secs_f64());
        }
        for gpu in &sample.gpus {
            let index = gpu.index.to_string();
            let labels = GpuLabels {
                run_id: run_id.to_string(),
                gpu: index.clone(),
            };
            for (family, value) in [
                (&self.gpu_utilization, gpu.utilization),
                (&self.gpu_memory_used, gpu.memory_used_bytes.map(float)),
                (&self.gpu_memory_total, gpu.memory_total_bytes.map(float)),
                (&self.gpu_temperature, gpu.temperature_celsius),
                (&self.gpu_power, gpu.power_watts),
                (&self.gpu_power_limit, gpu.power_limit_watts),
            ] {
                if let Some(value) = value {
                    family.get_or_create(&labels).set(value);
                }
            }
            self.gpu_info
                .get_or_create(&GpuInfoLabels {
                    run_id: run_id.to_string(),
                    gpu: index.clone(),
                    name: gpu.name.clone(),
                })
                .set(1);
            series.gpus.push((index, gpu.name.clone()));
        }
        series
    }

    /// Removes the series in `series`.
    pub(super) fn remove(&self, series: &SystemSeries) {
        let run = RunLabels {
            run_id: series.run_id.clone(),
        };
        for family in [
            &self.cpu_usage,
            &self.cpu_load1,
            &self.cpus,
            &self.memory_used,
            &self.memory_limit,
            &self.sampled,
        ] {
            family.remove(&run);
        }
        for mount in &series.mounts {
            let labels = MountLabels {
                run_id: series.run_id.clone(),
                mount: mount.clone(),
            };
            self.disk_used.remove(&labels);
            self.disk_size.remove(&labels);
        }
        for (gpu, name) in &series.gpus {
            let labels = GpuLabels {
                run_id: series.run_id.clone(),
                gpu: gpu.clone(),
            };
            for family in [
                &self.gpu_utilization,
                &self.gpu_memory_used,
                &self.gpu_memory_total,
                &self.gpu_temperature,
                &self.gpu_power,
                &self.gpu_power_limit,
            ] {
                family.remove(&labels);
            }
            self.gpu_info.remove(&GpuInfoLabels {
                run_id: series.run_id.clone(),
                gpu: gpu.clone(),
                name: name.clone(),
            });
        }
    }
}
