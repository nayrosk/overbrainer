//! The disk policy of a Runpod run: while overbrainer follows the job, it
//! watches how full the run's disk is, warns as it fills, and before it is full
//! grows the network volume (when `max_volume_gb` allows it) or stops the job
//! with a snapshot.
//!
//! On a pod without a network volume, the run's disk is the container disk, read
//! with `df` by the system sampler ([`Event::System`]). On a network volume, `df`
//! shows the whole shared cluster instead, so the volume's own use is read with
//! `du` and weighed against its size as the API reports it, less a margin: the
//! quota refuses writes a little before the size in GB. That `du`, and the size
//! of the newest checkpoint, come from a second, cheaper probe that runs at most
//! every [`DISK_PROBE_EVERY`].

use std::convert::Infallible;
use std::time::{Duration, Instant};

use tokio::sync::broadcast::Receiver;
use tokio::sync::broadcast::error::RecvError;

use super::RunpodClient;
use crate::events::Event;
use crate::exec::{Executor, quote};
use crate::runs::{RunRecord, SnapshotReason, request_snapshot};
use crate::system::{self, Disk, SystemSample};
use crate::train::OUTPUT_DIR;

/// Use, in percent, of the first warning.
pub const WARN_PERCENT: u64 = 85;
/// Percent between two warnings, after the first one.
pub const WARN_STEP: u64 = 5;
/// Use, in percent, at which the disk is grown or the job stopped.
pub const ACT_PERCENT: u64 = 92;
/// The disk is also acted on when its free space is less than this many times
/// the size of the newest checkpoint, in tenths: the next save would not fit.
const CHECKPOINT_ROOM_TENTHS: u64 = 15;
/// Share of a network volume's size, in percent, that can really be written:
/// Runpod's quota refuses writes a little before the size in GB.
pub const VOLUME_USABLE_PERCENT: u64 = 94;
/// Where the client writes the network volume's size in GB, in the run
/// directory, for the watchdog's own disk rule.
pub const VOLUME_SIZE_FILE: &str = ".pod/volume_gb";
/// How often, at most, the disk probe (`du`) runs.
pub const DISK_PROBE_EVERY: Duration = Duration::from_secs(60);
/// Samples after a grow within which the API must report the new size.
const GROW_CHECKS: u32 = 2;
/// Most a grow adds at least, in GB.
const GROW_MIN_GB: u32 = 50;
/// Longest wait for the disk probe.
const PROBE_TIMEOUT: Duration = Duration::from_secs(60);
/// Bytes in a GB, as Runpod sizes volumes.
const GB: u64 = 1_000_000_000;

/// How much of the run's disk is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Usage {
    /// Bytes in use.
    pub used_bytes: u64,
    /// Bytes that can be written in all.
    pub capacity_bytes: u64,
}

impl Usage {
    /// A file system as `df` reports it: used over used plus available, so
    /// blocks reserved for root count as full.
    #[must_use]
    pub fn of_disk(disk: &Disk) -> Self {
        Self {
            used_bytes: disk.used_bytes,
            capacity_bytes: disk.used_bytes.saturating_add(disk.available_bytes),
        }
    }

    /// A network volume of `size_gb` holding `used_bytes`: only
    /// [`VOLUME_USABLE_PERCENT`] of its size counts.
    #[must_use]
    pub fn of_volume(used_bytes: u64, size_gb: u32) -> Self {
        Self {
            used_bytes,
            capacity_bytes: u64::from(size_gb).saturating_mul(GB / 100 * VOLUME_USABLE_PERCENT),
        }
    }

    /// Bytes still free.
    #[must_use]
    pub fn free_bytes(&self) -> u64 {
        self.capacity_bytes.saturating_sub(self.used_bytes)
    }

    /// Whether at least `percent` of the disk is used; never for an empty one.
    fn at_least(&self, percent: u64) -> bool {
        self.capacity_bytes > 0
            && u128::from(self.used_bytes) * 100
                >= u128::from(percent) * u128::from(self.capacity_bytes)
    }
}

/// Why the disk must be acted on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Critical {
    /// [`ACT_PERCENT`] of it is used.
    Full,
    /// The next checkpoint would not fit.
    NoRoomForCheckpoint,
}

/// What one look at the disk calls for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Assessment {
    /// A warning, at this percent: the highest step reached past the last
    /// warning.
    pub warn: Option<u64>,
    /// An action.
    pub act: Option<Critical>,
}

/// What `usage` calls for: a warning at [`WARN_PERCENT`] and at each further
/// [`WARN_STEP`] above `warned` (the last warning's percent), and an action at
/// [`ACT_PERCENT`] or when the free space is less than 1.5 times
/// `checkpoint_bytes`, the size of the newest checkpoint when known.
#[must_use]
pub fn assess(usage: Usage, checkpoint_bytes: Option<u64>, warned: Option<u64>) -> Assessment {
    let level = (0..=(100 - WARN_PERCENT) / WARN_STEP)
        .map(|step| WARN_PERCENT + step * WARN_STEP)
        .take_while(|percent| usage.at_least(*percent))
        .last();
    let warn = level.filter(|level| warned.is_none_or(|warned| *level > warned));
    let no_room = checkpoint_bytes.is_some_and(|checkpoint| {
        u128::from(usage.free_bytes()) * 10
            < u128::from(checkpoint) * u128::from(CHECKPOINT_ROOM_TENTHS)
    });
    let act = if usage.at_least(ACT_PERCENT) {
        Some(Critical::Full)
    } else if usage.capacity_bytes > 0 && no_room {
        Some(Critical::NoRoomForCheckpoint)
    } else {
        None
    };
    Assessment { warn, act }
}

/// The size, in GB, to grow a network volume of `current_gb` to: half again
/// its size, and at least [`GROW_MIN_GB`] more, but never past `max_gb`.
/// `None` when that is no growth at all.
#[must_use]
pub fn grown_size(current_gb: u32, max_gb: u32) -> Option<u32> {
    let wanted = current_gb
        .saturating_add(current_gb.div_ceil(2))
        .max(current_gb.saturating_add(GROW_MIN_GB));
    let size = wanted.min(max_gb);
    (size > current_gb).then_some(size)
}

/// The script printing the size of the newest checkpoint of the run in
/// `run_dir`, and with a network volume mounted at `volume_dir`, what the
/// volume holds, both in KiB, each after its `@name` line. It always succeeds.
#[must_use]
pub fn disk_probe_script(run_dir: &str, volume_dir: Option<&str>) -> String {
    let output = quote(&format!("{run_dir}/{OUTPUT_DIR}"));
    let volume = volume_dir.map_or_else(String::new, |volume| {
        format!("part volume\ndu -sk -- {} 2>/dev/null\n", quote(volume))
    });
    format!(
        "part() {{ printf '\\n@%s\\n' \"$1\"; }}\n\
         part checkpoint\n\
         last=$(ls -1d -- {output}/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -n 1)\n\
         [ -n \"$last\" ] && du -sk -- {output}/checkpoint-\"$last\" 2>/dev/null\n\
         {volume}exit 0\n"
    )
}

/// What [`disk_probe_script`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DiskProbe {
    /// Size of the newest checkpoint, when there is one.
    pub checkpoint_bytes: Option<u64>,
    /// What the network volume holds, when there is one.
    pub volume_used_bytes: Option<u64>,
}

/// Reads the answer of [`disk_probe_script`].
#[must_use]
pub fn parse_disk_probe(output: &str) -> DiskProbe {
    let parts = system::parts(output);
    let kib = |name: &str| {
        parts
            .get(name)
            .and_then(|lines| system::first_number::<u64>(lines))
            .and_then(|kib| kib.checked_mul(1024))
    };
    DiskProbe {
        checkpoint_bytes: kib("checkpoint"),
        volume_used_bytes: kib("volume"),
    }
}

/// The network volume of a pod.
#[derive(Debug, Clone, Copy)]
pub struct VolumeDisk<'a> {
    /// Its ID.
    pub id: &'a str,
    /// Where it is mounted on the pod.
    pub mount: &'a str,
    /// The target's `max_volume_gb`: without it, the volume never grows.
    pub max_gb: Option<u32>,
}

/// A grow asked for, not seen yet.
#[derive(Debug, Clone, Copy)]
struct Growing {
    to_gb: u32,
    checks_left: u32,
}

/// The disk policy of one followed run: see the [module](self).
pub struct DiskWatch<'a, E> {
    executor: &'a E,
    client: &'a RunpodClient,
    run: &'a RunRecord,
    volume: Option<VolumeDisk<'a>>,
    /// The volume's size in GB, as the API last reported it.
    size_gb: Option<u32>,
    probe: DiskProbe,
    probed_at: Option<Instant>,
    warned: Option<u64>,
    growing: Option<Growing>,
    /// Whether the snapshot was asked for: nothing more is done then.
    stopped: bool,
}

impl<'a, E: Executor> DiskWatch<'a, E> {
    /// The policy of the running run `run`, reached through `executor`, on a
    /// pod with the network volume `volume` when it has one.
    #[must_use]
    pub fn new(
        executor: &'a E,
        client: &'a RunpodClient,
        run: &'a RunRecord,
        volume: Option<VolumeDisk<'a>>,
    ) -> Self {
        Self {
            executor,
            client,
            run,
            volume,
            size_gb: None,
            probe: DiskProbe::default(),
            probed_at: None,
            warned: None,
            growing: None,
            stopped: false,
        }
    }

    /// Follows the samples on `events` until the bus closes, then waits
    /// forever: it never ends the watch.
    pub async fn run(mut self, mut events: Receiver<Event>) -> Infallible {
        loop {
            match events.recv().await {
                Ok(Event::System(sample)) => self.sampled(&sample).await,
                Ok(_) | Err(RecvError::Lagged(_)) => {},
                Err(RecvError::Closed) => break,
            }
        }
        std::future::pending().await
    }

    /// Takes the sample `sample` into account.
    async fn sampled(&mut self, sample: &SystemSample) {
        if self.stopped {
            return;
        }
        if self.volume.is_some() && self.size_gb.is_none() {
            self.read_size().await;
        }
        self.probe_if_due().await;
        if let Some(usage) = self.usage(sample) {
            self.check(usage, self.probe.checkpoint_bytes).await;
        }
    }

    /// The run disk's use: the volume's `du` against its size, or the run
    /// directory's file system from `sample`.
    fn usage(&self, sample: &SystemSample) -> Option<Usage> {
        if self.volume.is_some() {
            return Some(Usage::of_volume(
                self.probe.volume_used_bytes?,
                self.size_gb?,
            ));
        }
        sample.run_disk().map(Usage::of_disk)
    }

    /// Runs the disk probe when [`DISK_PROBE_EVERY`] passed since the last one;
    /// a failure keeps what the last one found.
    async fn probe_if_due(&mut self) {
        if self
            .probed_at
            .is_some_and(|at| at.elapsed() < DISK_PROBE_EVERY)
        {
            return;
        }
        self.probed_at = Some(Instant::now());
        let script =
            disk_probe_script(&self.run.remote_dir, self.volume.map(|volume| volume.mount));
        let answer = match tokio::time::timeout(PROBE_TIMEOUT, self.executor.probe(&script)).await {
            Ok(answer) => answer.map_err(|error| error.to_string()),
            Err(_) => Err("no answer in time".to_string()),
        };
        match answer {
            Ok(output) => self.probe = parse_disk_probe(&String::from_utf8_lossy(&output)),
            Err(error) => tracing::debug!("cannot measure the pod's disk: {error}"),
        }
    }

    /// Reads the volume's size from the API and, when it changed, hands it to
    /// the watchdog. Returns it, or `None` without a volume or when it cannot
    /// be read: the failure is logged, and the next sample tries again.
    pub async fn read_size(&mut self) -> Option<u32> {
        let volume = self.volume?;
        let size = match self.client.get_network_volume(volume.id).await {
            Ok(Some(found)) => Ok(found.size),
            Ok(None) => Err("not found".to_string()),
            Err(error) => Err(error.to_string()),
        };
        let size = size
            .inspect_err(|error| {
                tracing::debug!("cannot read network volume {}: {error}", volume.id);
            })
            .ok()?;
        if self.size_gb != Some(size) {
            self.size_gb = Some(size);
            self.hand_size(size).await;
        }
        Some(size)
    }

    /// Writes the volume's `size` where the watchdog reads it.
    async fn hand_size(&self, size: u32) {
        let path = format!("{}/{VOLUME_SIZE_FILE}", self.run.remote_dir);
        if let Err(error) = self.executor.put_file(&path, &size.to_string()).await {
            tracing::debug!("cannot hand the volume size to the watchdog: {error}");
        }
    }

    /// Warns and acts on `usage`, with the newest checkpoint of
    /// `checkpoint_bytes`. While a grow is pending, it only checks the grow.
    pub async fn check(&mut self, usage: Usage, checkpoint_bytes: Option<u64>) {
        if self.stopped {
            return;
        }
        if self.growing.is_some() {
            self.check_grow().await;
            return;
        }
        let assessment = assess(usage, checkpoint_bytes, self.warned);
        if let Some(percent) = assessment.warn {
            self.warned = Some(percent);
            tracing::warn!(
                "the pod's {} is {percent}% full ({} free)",
                self.disk_name(),
                gb(usage.free_bytes())
            );
        }
        let Some(critical) = assessment.act else {
            return;
        };
        let why = match critical {
            Critical::Full => format!("the pod's {} is {ACT_PERCENT}% full", self.disk_name()),
            Critical::NoRoomForCheckpoint => format!(
                "the pod's {} has {} free, too little for the next checkpoint",
                self.disk_name(),
                gb(usage.free_bytes())
            ),
        };
        if !self.grow(&why).await {
            self.stop(&why).await;
        }
    }

    /// The disk's name in a warning.
    fn disk_name(&self) -> &'static str {
        if self.volume.is_some() {
            "network volume"
        } else {
            "disk"
        }
    }

    /// Asks Runpod to grow the volume, when `max_volume_gb` leaves room;
    /// whether it did.
    async fn grow(&mut self, why: &str) -> bool {
        let (Some(volume), Some(size)) = (self.volume, self.size_gb) else {
            return false;
        };
        let Some(to_gb) = volume.max_gb.and_then(|max| grown_size(size, max)) else {
            return false;
        };
        let resized = self.client.resize_network_volume(volume.id, to_gb).await;
        let note = match &resized {
            Ok(_) => format!(
                "{why}: growing network volume {} from {size} to {to_gb} GB (max_volume_gb)",
                volume.id
            ),
            Err(error) => format!("cannot grow network volume {}: {error}", volume.id),
        };
        tracing::warn!("{note}");
        if resized.is_ok() {
            self.growing = Some(Growing {
                to_gb,
                checks_left: GROW_CHECKS,
            });
        }
        resized.is_ok()
    }

    /// Whether the pending grow shows in the API: then the new size counts;
    /// after [`GROW_CHECKS`] samples without it, the job is stopped.
    async fn check_grow(&mut self) {
        let Some(mut growing) = self.growing.take() else {
            return;
        };
        if self
            .read_size()
            .await
            .is_some_and(|size| size >= growing.to_gb)
        {
            self.warned = None;
            tracing::warn!("the network volume grew to {} GB", growing.to_gb);
            return;
        }
        growing.checks_left = growing.checks_left.saturating_sub(1);
        if growing.checks_left == 0 {
            self.stop("the network volume did not grow").await;
        } else {
            self.growing = Some(growing);
        }
    }

    /// Stops the job with a snapshot for `why`; tried again at the next sample
    /// when the request cannot be written.
    async fn stop(&mut self, why: &str) {
        let asked = request_snapshot(self.executor, self.run, SnapshotReason::Disk).await;
        self.stopped = asked.is_ok();
        let note = match asked {
            Ok(()) => format!("{why}: the job is stopped with a snapshot"),
            Err(error) => format!("{why}, but the snapshot cannot be asked for: {error}"),
        };
        tracing::warn!("{note}");
    }
}

/// `bytes` in GB, one decimal.
fn gb(bytes: u64) -> String {
    format!("{:.1} GB", system::float(bytes) / system::float(GB))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn usage(percent: u64) -> Usage {
        Usage {
            used_bytes: percent * GIB,
            capacity_bytes: 100 * GIB,
        }
    }

    #[test]
    fn warnings_start_at_85_percent_then_come_every_5() {
        assert_eq!(assess(usage(84), None, None), Assessment::default());
        assert_eq!(assess(usage(85), None, None).warn, Some(85));
        assert_eq!(assess(usage(89), None, Some(85)).warn, None);
        assert_eq!(assess(usage(90), None, Some(85)).warn, Some(90));
        // A jump warns once, at the highest step reached.
        assert_eq!(assess(usage(97), None, Some(85)).warn, Some(95));
        assert_eq!(assess(usage(100), None, Some(95)).warn, Some(100));
        assert_eq!(assess(usage(100), None, Some(100)).warn, None);
    }

    #[test]
    fn the_disk_is_acted_on_at_92_percent() {
        assert_eq!(assess(usage(91), None, None).act, None);
        assert_eq!(assess(usage(92), None, None).act, Some(Critical::Full));
        // Just under: 91.99%.
        let under = Usage {
            used_bytes: 9199,
            capacity_bytes: 10_000,
        };
        assert_eq!(assess(under, None, None).act, None);
        let empty = Usage {
            used_bytes: 0,
            capacity_bytes: 0,
        };
        assert_eq!(assess(empty, Some(GIB), None), Assessment::default());
    }

    #[test]
    fn the_next_checkpoint_must_fit_one_and_a_half_times() {
        // 40 GiB free: a 26 GiB checkpoint fits 1.5 times (39), a 27 GiB one not.
        assert_eq!(assess(usage(60), Some(26 * GIB), None).act, None);
        assert_eq!(
            assess(usage(60), Some(27 * GIB), None).act,
            Some(Critical::NoRoomForCheckpoint)
        );
        // Unknown checkpoint size: only the percentage counts.
        assert_eq!(assess(usage(60), None, None).act, None);
    }

    #[test]
    fn a_volume_counts_94_percent_of_its_size() {
        let volume = Usage::of_volume(9_400_000_000, 10);
        assert_eq!(volume.capacity_bytes, 9_400_000_000);
        assert_eq!(volume.free_bytes(), 0);
        assert_eq!(assess(volume, None, None).act, Some(Critical::Full));
        let disk = Usage::of_disk(&Disk {
            mount: "/".into(),
            fstype: None,
            shared: false,
            size_bytes: 100,
            used_bytes: 40,
            available_bytes: 50,
        });
        assert_eq!((disk.used_bytes, disk.capacity_bytes), (40, 90));
    }

    #[test]
    fn the_watchdog_acts_at_the_same_share() {
        let script = crate::runpod::watchdog_script();
        assert!(script.contains(&format!("\nDISK_ACT={ACT_PERCENT}\n")));
        assert!(script.contains(&format!("\nVOLUME_USABLE={VOLUME_USABLE_PERCENT}\n")));
        assert!(script.contains(&format!(
            "pod_number {}",
            VOLUME_SIZE_FILE.trim_start_matches(".pod/")
        )));
    }

    #[test]
    fn a_volume_grows_by_half_at_least_50_gb_up_to_its_cap() {
        assert_eq!(grown_size(200, 1000), Some(300));
        assert_eq!(grown_size(20, 1000), Some(70));
        assert_eq!(grown_size(200, 250), Some(250));
        assert_eq!(grown_size(250, 250), None);
        assert_eq!(grown_size(300, 250), None);
    }

    #[test]
    fn the_disk_probe_reads_the_newest_checkpoint_and_the_volume() {
        let found =
            parse_disk_probe("\n@checkpoint\n2048\t/r/output/checkpoint-20\n\n@volume\n10\t/v\n");
        assert_eq!(
            found,
            DiskProbe {
                checkpoint_bytes: Some(2048 * 1024),
                volume_used_bytes: Some(10 * 1024),
            }
        );
        assert_eq!(parse_disk_probe("\n@checkpoint\n"), DiskProbe::default());
        let script = disk_probe_script("/w/it's", None);
        assert!(
            script.contains("'/w/it'\\''s/output'/checkpoint-*"),
            "{script}"
        );
        assert!(!script.contains("@volume") && !script.contains("part volume"));
        let script = disk_probe_script("/w", Some("/workspace/data"));
        assert!(
            script.contains("part volume\ndu -sk -- '/workspace/data'"),
            "{script}"
        );
    }

    #[test]
    fn the_disk_probe_finds_the_newest_checkpoint_by_step() -> Result<(), Box<dyn std::error::Error>>
    {
        let dir = tempfile::tempdir()?;
        let run = dir.path().join("run");
        for (step, bytes) in [(9, 1), (10, 1024 * 1024), (100, 64 * 1024)] {
            let checkpoint = run.join(format!("output/checkpoint-{step}"));
            std::fs::create_dir_all(&checkpoint)?;
            std::fs::write(checkpoint.join("weights"), vec![1u8; bytes])?;
        }
        let run = run.to_str().ok_or("path")?;
        let output = std::process::Command::new("sh")
            .args(["-c", &disk_probe_script(run, Some(run))])
            .output()?;
        assert!(output.status.success());
        let found = parse_disk_probe(&String::from_utf8_lossy(&output.stdout));
        // checkpoint-100, not checkpoint-9 (the last by name).
        let checkpoint = found.checkpoint_bytes.ok_or("no checkpoint")?;
        assert!(
            (64 * 1024..512 * 1024).contains(&checkpoint),
            "{checkpoint}"
        );
        assert!(
            found
                .volume_used_bytes
                .is_some_and(|used| used >= 12 * 1024)
        );
        Ok(())
    }
}
