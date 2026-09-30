//! What the machine of a training run looks like: disk, CPU, memory and GPUs,
//! read by one shell script in one round trip ([`probe_script`]) and parsed into
//! a [`SystemSample`] ([`parse`]).
//!
//! Inside a container (a Runpod pod), `/proc/meminfo` and `nproc` describe the
//! host, so the cgroup v2 files of the container come first; `/proc` is the
//! fallback where there is no cgroup limit (a plain host, cgroup v1).

use std::collections::BTreeMap;
use std::time::SystemTime;

use crate::exec::quote;

/// Bytes in a MiB, the unit of `nvidia-smi`'s memory figures.
const MIB: u64 = 1024 * 1024;

/// What the probe found on the target at one moment. A part the target does not
/// have or would not tell (no GPU, no `/proc`) is empty or `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemSample {
    /// When the probe's answer arrived, on this machine's clock.
    pub at: SystemTime,
    /// The file system of the run directory first, then `/` when it is another.
    pub disks: Vec<Disk>,
    /// The CPU.
    pub cpu: Option<Cpu>,
    /// The memory.
    pub memory: Option<Memory>,
    /// The GPUs, as `nvidia-smi` lists them.
    pub gpus: Vec<Gpu>,
}

impl SystemSample {
    /// The file system holding the run directory.
    #[must_use]
    pub fn run_disk(&self) -> Option<&Disk> {
        self.disks.first()
    }
}

/// A mounted file system, as `df -Pk` reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Disk {
    /// Where it is mounted.
    pub mount: String,
    /// Its size.
    pub size_bytes: u64,
    /// Bytes in use.
    pub used_bytes: u64,
    /// Bytes an unprivileged user can still write.
    pub available_bytes: u64,
}

impl Disk {
    /// The share in use, as `df` computes its capacity: used over used plus
    /// available, so blocks reserved for root count as full. `None` for an
    /// empty file system.
    #[must_use]
    pub fn used_ratio(&self) -> Option<f64> {
        ratio(
            self.used_bytes,
            self.used_bytes.saturating_add(self.available_bytes),
        )
    }
}

/// The CPU of the target, or of its container.
#[derive(Debug, Clone, PartialEq)]
pub struct Cpu {
    /// Share of the CPUs busy since the previous sample, between 0 and 1; `None`
    /// on the first sample.
    pub usage: Option<f64>,
    /// Load average over the last minute.
    pub load1: Option<f64>,
    /// CPUs available: the container's quota when it has one, else `nproc`.
    pub cpus: Option<f64>,
    /// The counters the next sample's usage is computed from.
    pub(crate) times: Option<CpuTimes>,
}

/// CPU time counters, in the same unit for `busy` and `total`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CpuTimes {
    /// Time the CPUs were busy.
    busy: f64,
    /// Time the CPUs could have been busy.
    total: f64,
    /// Whether they come from the cgroup (`cpu.stat` and the uptime) rather than
    /// `/proc/stat`: counters from different sources are never compared.
    cgroup: bool,
}

/// The memory of the target, or the limit of its container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Memory {
    /// Bytes in use, the page cache the kernel can drop left out.
    pub used_bytes: u64,
    /// Bytes available in all.
    pub limit_bytes: u64,
}

impl Memory {
    /// The share in use; `None` without a limit.
    #[must_use]
    pub fn ratio(&self) -> Option<f64> {
        ratio(self.used_bytes, self.limit_bytes)
    }
}

/// A GPU, as `nvidia-smi` reports it. A figure it does not support is `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct Gpu {
    /// Its index on the target.
    pub index: u32,
    /// Its model, control characters dropped.
    pub name: String,
    /// Share of time a kernel ran, between 0 and 1.
    pub utilization: Option<f64>,
    /// Memory in use.
    pub memory_used_bytes: Option<u64>,
    /// Memory in all.
    pub memory_total_bytes: Option<u64>,
    /// Temperature in degrees Celsius.
    pub temperature_celsius: Option<f64>,
    /// Power drawn, in watts.
    pub power_watts: Option<f64>,
    /// Power limit, in watts.
    pub power_limit_watts: Option<f64>,
}

impl Gpu {
    /// The share of its memory in use, when both figures are known.
    #[must_use]
    pub fn memory_ratio(&self) -> Option<f64> {
        ratio(self.memory_used_bytes?, self.memory_total_bytes?)
    }
}

/// `part` over `whole`, `None` when `whole` is zero.
fn ratio(part: u64, whole: u64) -> Option<f64> {
    (whole > 0).then(|| float(part) / float(whole))
}

/// `value` as a float, without a lossy cast: exact up to 2^53.
#[must_use]
pub fn float(value: u64) -> f64 {
    let high = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let low = u32::try_from(value & u64::from(u32::MAX)).unwrap_or(u32::MAX);
    f64::from(high) * 4_294_967_296.0 + f64::from(low)
}

/// The script printing everything [`parse`] reads, one `@name` line before each
/// part. Each command's errors are dropped and the script always succeeds, so a
/// target without `nvidia-smi` or `/proc` still answers the rest. `run_dir` is
/// the run directory on the target, whose file system comes first.
#[must_use]
pub fn probe_script(run_dir: &str) -> String {
    format!(
        r#"part() {{ printf '\n@%s\n' "$1"; }}
part gpu
if command -v nvidia-smi >/dev/null 2>&1; then
  limit=
  command -v timeout >/dev/null 2>&1 && limit='timeout 5'
  $limit nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null
fi
part loadavg; cat /proc/loadavg 2>/dev/null
part stat; head -n 1 /proc/stat 2>/dev/null
part uptime; cat /proc/uptime 2>/dev/null
part meminfo; grep -E '^(MemTotal|MemAvailable):' /proc/meminfo 2>/dev/null
part memory.max; cat /sys/fs/cgroup/memory.max 2>/dev/null
part memory.current; cat /sys/fs/cgroup/memory.current 2>/dev/null
part memory.stat; grep '^inactive_file ' /sys/fs/cgroup/memory.stat 2>/dev/null
part cpu.max; cat /sys/fs/cgroup/cpu.max 2>/dev/null
part cpu.stat; grep '^usage_usec ' /sys/fs/cgroup/cpu.stat 2>/dev/null
part nproc; nproc 2>/dev/null
part df.run; df -Pk -- {dir} 2>/dev/null
part df.root; df -Pk -- / 2>/dev/null
exit 0
"#,
        dir = quote(run_dir)
    )
}

/// The sample in `output`, the answer of [`probe_script`] received at `at`. The
/// CPU usage is computed from the counters of `previous`, the sample before it.
#[must_use]
pub fn parse(output: &str, at: SystemTime, previous: Option<&SystemSample>) -> SystemSample {
    let parts = parts(output);
    let part = |name: &str| parts.get(name).map_or(&[][..], Vec::as_slice);
    let mut disks = Vec::new();
    for name in ["df.run", "df.root"] {
        if let Some(disk) = disk(part(name))
            && !disks.iter().any(|seen: &Disk| seen.mount == disk.mount)
        {
            disks.push(disk);
        }
    }
    let before = previous
        .and_then(|sample| sample.cpu.as_ref())
        .and_then(|cpu| cpu.times);
    SystemSample {
        at,
        disks,
        cpu: cpu(&part, before),
        memory: memory(&part),
        gpus: part("gpu").iter().filter_map(|line| gpu(line)).collect(),
    }
}

/// The non-blank lines of each `@name` part of `output`.
fn parts(output: &str) -> BTreeMap<&str, Vec<&str>> {
    let mut parts: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    let mut current = None;
    for line in output.lines().map(str::trim) {
        if let Some(name) = line.strip_prefix('@') {
            current = Some(name);
            parts.entry(name).or_default();
        } else if let Some(name) = current
            && !line.is_empty()
        {
            parts.entry(name).or_default().push(line);
        }
    }
    parts
}

/// The first number of the first line of `lines`.
fn first_number<T: std::str::FromStr>(lines: &[&str]) -> Option<T> {
    lines.first()?.split_whitespace().next()?.parse().ok()
}

/// The file system `df -Pk` describes in `lines`: its header, then one line.
fn disk(lines: &[&str]) -> Option<Disk> {
    let fields: Vec<&str> = lines.get(1)?.split_whitespace().collect();
    let kib =
        |index: usize| -> Option<u64> { fields.get(index)?.parse::<u64>().ok()?.checked_mul(1024) };
    // A mount point may hold spaces: it is everything after the capacity.
    let mount = fields.get(5..).filter(|rest| !rest.is_empty())?.join(" ");
    Some(Disk {
        mount,
        size_bytes: kib(1)?,
        used_bytes: kib(2)?,
        available_bytes: kib(3)?,
    })
}

/// The memory: the cgroup's limit and use when it has a limit, else
/// `/proc/meminfo`.
fn memory<'a>(part: &impl Fn(&str) -> &'a [&'a str]) -> Option<Memory> {
    let cgroup = || {
        let limit: u64 = first_number(part("memory.max"))?;
        let current: u64 = first_number(part("memory.current"))?;
        let inactive: u64 = part("memory.stat")
            .first()
            .and_then(|line| line.split_whitespace().nth(1)?.parse().ok())
            .unwrap_or(0);
        Some(Memory {
            used_bytes: current.saturating_sub(inactive),
            limit_bytes: limit,
        })
    };
    cgroup().or_else(|| {
        let field = |name: &str| -> Option<u64> {
            let line = part("meminfo").iter().find(|line| line.starts_with(name))?;
            line.split_whitespace()
                .nth(1)?
                .parse::<u64>()
                .ok()?
                .checked_mul(1024)
        };
        let total = field("MemTotal:")?;
        let available = field("MemAvailable:")?;
        Some(Memory {
            used_bytes: total.saturating_sub(available),
            limit_bytes: total,
        })
    })
}

/// The CPU: its count and load, and its usage since the counters `before`.
fn cpu<'a>(part: &impl Fn(&str) -> &'a [&'a str], before: Option<CpuTimes>) -> Option<Cpu> {
    let load1: Option<f64> = first_number(part("loadavg"));
    let nproc: Option<f64> = first_number(part("nproc"));
    let quota = part("cpu.max").first().and_then(|line| {
        let mut fields = line.split_whitespace();
        let quota: f64 = fields.next()?.parse().ok()?;
        let period: f64 = fields.next()?.parse().ok()?;
        (period > 0.0).then_some(quota / period)
    });
    let cpus = quota.or(nproc);
    let times = cgroup_times(part, cpus).or_else(|| proc_times(part("stat")));
    if load1.is_none() && cpus.is_none() && times.is_none() {
        return None;
    }
    let usage = match (before, times) {
        (Some(before), Some(now)) if before.cgroup == now.cgroup => {
            let total = now.total - before.total;
            (total > 0.0).then(|| ((now.busy - before.busy) / total).clamp(0.0, 1.0))
        },
        _ => None,
    };
    Some(Cpu {
        usage,
        load1,
        cpus,
        times,
    })
}

/// The cgroup's CPU time and the time `cpus` CPUs had since boot, in seconds;
/// `None` outside a cgroup with a `cpu.max` file.
fn cgroup_times<'a>(part: &impl Fn(&str) -> &'a [&'a str], cpus: Option<f64>) -> Option<CpuTimes> {
    part("cpu.max").first()?;
    let usage: f64 = part("cpu.stat")
        .first()?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    let uptime: f64 = first_number(part("uptime"))?;
    Some(CpuTimes {
        busy: usage / 1e6,
        total: uptime * cpus?,
        cgroup: true,
    })
}

/// The busy and total jiffies of the `cpu` line of `/proc/stat`: idle and
/// I/O wait are not busy.
fn proc_times(lines: &[&str]) -> Option<CpuTimes> {
    let mut fields = lines.first()?.split_whitespace();
    if fields.next()? != "cpu" {
        return None;
    }
    let values: Vec<f64> = fields.filter_map(|field| field.parse().ok()).collect();
    // user nice system idle iowait irq softirq steal: guest time is already in
    // user and nice.
    let counted = values.get(..8).unwrap_or(&values);
    let total: f64 = counted.iter().sum();
    let idle = values.get(3).copied().unwrap_or(0.0) + values.get(4).copied().unwrap_or(0.0);
    (total > 0.0).then_some(CpuTimes {
        busy: total - idle,
        total,
        cgroup: false,
    })
}

/// A GPU from one CSV line of `nvidia-smi`.
fn gpu(line: &str) -> Option<Gpu> {
    let fields: Vec<&str> = line.split(',').map(str::trim).collect();
    // The name sits between the index and six figures: a comma in it stays.
    let figures = fields.len().checked_sub(6).filter(|at| *at >= 2)?;
    let number = |index: usize| -> Option<f64> {
        fields
            .get(index)?
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
    };
    let mib =
        |index: usize| -> Option<u64> { fields.get(index)?.parse::<u64>().ok()?.checked_mul(MIB) };
    let name: String = fields
        .get(1..figures)?
        .join(",")
        .chars()
        .filter(|c| !c.is_control())
        .collect();
    Some(Gpu {
        index: fields.first()?.parse().ok()?,
        name,
        utilization: number(figures).map(|percent| (percent / 100.0).clamp(0.0, 1.0)),
        memory_used_bytes: mib(figures + 1),
        memory_total_bytes: mib(figures + 2),
        temperature_celsius: number(figures + 3),
        power_watts: number(figures + 4),
        power_limit_watts: number(figures + 5),
    })
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use super::*;

    fn at(seconds: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(seconds)
    }

    /// A Runpod pod: two GPUs, a cgroup v2 limit, the volume and `/` apart.
    const POD: &str = "
@gpu
0, NVIDIA A100-SXM4-80GB, 87, 40960, 81920, 64, 312.45, 400.00
1, NVIDIA A100-SXM4-80GB, [N/A], 1024, 81920, 41, [N/A], 400.00

@loadavg
3.25 2.10 1.05 4/812 12345

@stat
cpu  1000 0 1000 7000 1000 0 0 0 0 0

@uptime
100.00 1500.00

@meminfo
MemTotal:       2097152000 kB
MemAvailable:   1048576000 kB

@memory.max
68719476736

@memory.current
34359738368

@memory.stat
inactive_file 1073741824

@cpu.max
800000 100000

@cpu.stat
usage_usec 200000000

@nproc
128

@df.run
Filesystem     1024-blocks      Used Available Capacity Mounted on
mfs#runpod   104857600  83886080  20971520      80% /workspace

@df.root
Filesystem     1024-blocks      Used Available Capacity Mounted on
overlay         52428800  10485760  41943040      20% /
";

    /// A plain host: no GPU, no cgroup limit, the run directory on `/`.
    const HOST: &str = "
@gpu

@loadavg
0.50 0.40 0.30 1/200 999
@stat
cpu  2000 0 1000 6000 1000 0 0 0 0 0
@uptime
5000.00 9000.00
@meminfo
MemTotal:       16384000 kB
MemAvailable:   12288000 kB
@memory.max
@memory.current
@memory.stat
@cpu.max
@cpu.stat
@nproc
8
@df.run
Filesystem     1024-blocks      Used Available Capacity Mounted on
/dev/nvme0n1p2  1000000  400000  600000      40% /
@df.root
Filesystem     1024-blocks      Used Available Capacity Mounted on
/dev/nvme0n1p2  1000000  400000  600000      40% /
";

    #[test]
    fn a_pod_reads_its_gpus_and_its_cgroup() -> Result<(), String> {
        let sample = parse(POD, at(10), None);
        assert_eq!(sample.at, at(10));
        let [first, second] = sample.gpus.as_slice() else {
            return Err(format!("{:?}", sample.gpus));
        };
        assert_eq!(
            *first,
            Gpu {
                index: 0,
                name: "NVIDIA A100-SXM4-80GB".into(),
                utilization: Some(0.87),
                memory_used_bytes: Some(40960 * MIB),
                memory_total_bytes: Some(81920 * MIB),
                temperature_celsius: Some(64.0),
                power_watts: Some(312.45),
                power_limit_watts: Some(400.0),
            }
        );
        assert_eq!((second.index, second.utilization), (1, None));
        assert_eq!(second.power_watts, None);
        assert_eq!(first.memory_ratio(), Some(0.5));
        // The cgroup's limit, the page cache it can drop left out; not the host.
        assert_eq!(
            sample.memory,
            Some(Memory {
                used_bytes: 31 * 1024 * MIB,
                limit_bytes: 64 * 1024 * MIB,
            })
        );
        let cpu = sample.cpu.ok_or("no cpu")?;
        assert_eq!(
            (cpu.load1, cpu.cpus, cpu.usage),
            (Some(3.25), Some(8.0), None)
        );
        let mounts: Vec<&str> = sample.disks.iter().map(|d| d.mount.as_str()).collect();
        assert_eq!(mounts, ["/workspace", "/"]);
        let run = sample.disks.first().ok_or("no disk")?;
        assert_eq!(run.size_bytes, 104_857_600 * 1024);
        assert_eq!(run.available_bytes, 20_971_520 * 1024);
        assert_eq!(run.used_ratio(), Some(0.8));
        Ok(())
    }

    #[test]
    fn a_host_without_gpu_or_cgroup_falls_back_to_proc() -> Result<(), String> {
        let sample = parse(HOST, at(0), None);
        assert!(sample.gpus.is_empty());
        assert_eq!(
            sample.memory,
            Some(Memory {
                used_bytes: 4_096_000 * 1024,
                limit_bytes: 16_384_000 * 1024,
            })
        );
        let cpu = sample.cpu.as_ref().ok_or("no cpu")?;
        assert_eq!((cpu.load1, cpu.cpus), (Some(0.5), Some(8.0)));
        // One file system for the run directory and `/`: listed once.
        assert_eq!(sample.disks.len(), 1);
        assert_eq!(sample.run_disk().map(|d| d.mount.as_str()), Some("/"));
        Ok(())
    }

    #[test]
    fn an_unlimited_cgroup_reads_proc_meminfo() {
        let output = HOST.replace("@memory.max", "@memory.max\nmax\n@memory.current\n5\n@x");
        let sample = parse(&output, at(0), None);
        assert_eq!(
            sample.memory.map(|m| m.limit_bytes),
            Some(16_384_000 * 1024)
        );
    }

    #[test]
    fn cpu_usage_is_the_busy_share_since_the_previous_sample() -> Result<(), String> {
        let first = parse(HOST, at(0), None);
        // 1000 more busy jiffies out of 4000.
        let later = HOST.replace("cpu  2000 0 1000 6000 1000", "cpu  2500 0 1500 8000 2000");
        let second = parse(&later, at(10), Some(&first));
        let usage = second
            .cpu
            .as_ref()
            .and_then(|cpu| cpu.usage)
            .ok_or("no usage")?;
        assert!((usage - 0.25).abs() < 1e-9, "{usage}");
        Ok(())
    }

    #[test]
    fn a_container_usage_is_its_cgroup_time_over_its_quota() -> Result<(), String> {
        let first = parse(POD, at(0), None);
        // 40 CPU seconds over 10 seconds of 8 CPUs.
        let later = POD
            .replace("usage_usec 200000000", "usage_usec 240000000")
            .replace("100.00 1500.00", "110.00 1500.00");
        let second = parse(&later, at(10), Some(&first));
        let usage = second
            .cpu
            .as_ref()
            .and_then(|cpu| cpu.usage)
            .ok_or("no usage")?;
        assert!((usage - 0.5).abs() < 1e-9, "{usage}");
        // Counters from another source are never compared.
        let host = parse(HOST, at(20), Some(&second));
        assert_eq!(host.cpu.and_then(|cpu| cpu.usage), None);
        Ok(())
    }

    #[test]
    fn an_empty_answer_is_an_empty_sample() {
        let sample = parse("", at(0), None);
        assert!(sample.disks.is_empty() && sample.gpus.is_empty());
        assert_eq!((sample.cpu, sample.memory), (None, None));
    }

    #[test]
    fn a_mount_point_keeps_its_spaces_and_a_name_its_commas() -> Result<(), String> {
        let disk = disk(&["header", "/dev/sdb1 100 25 75 25% /mnt/my data"]).ok_or("no disk")?;
        assert_eq!(disk.mount, "/mnt/my data");
        let odd = gpu("2, Odd, GPU\u{7}, 5, 1, 2, 30, 50.5, 100").ok_or("no gpu")?;
        assert_eq!((odd.index, odd.name.as_str()), (2, "Odd,GPU"));
        assert_eq!(gpu("garbage"), None);
        Ok(())
    }

    #[test]
    fn floats_of_large_counts_are_exact() {
        assert!((float(0) - 0.0).abs() < f64::EPSILON);
        assert!((float(81_920 * MIB) - 85_899_345_920.0).abs() < f64::EPSILON);
        assert!((float(u64::from(u32::MAX) + 1) - 4_294_967_296.0).abs() < f64::EPSILON);
    }

    #[test]
    fn the_script_quotes_the_run_directory() {
        let script = probe_script("/w/it's");
        assert!(script.contains("df -Pk -- '/w/it'\\''s'"), "{script}");
        assert!(script.ends_with("exit 0\n"));
    }
}
