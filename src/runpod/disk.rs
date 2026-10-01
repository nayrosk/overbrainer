//! The disk policy of a Runpod run: while overbrainer follows the job, it
//! watches how full the run's disk is, warns as it fills, and before it is full
//! grows the network volume (when `max_volume_gb` allows it) or stops the job
//! with a snapshot.
//!
//! On a pod without a network volume, the run's disk is the container disk, read
//! with `df` by the system sampler ([`Event::System`]); a disk the sampler marks
//! shared is never judged by its `df`. On a network volume, `df`
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
use crate::runs::{RunError, RunRecord, SnapshotReason, request_snapshot};
use crate::system::{self, Disk, SystemSample};
use crate::train::{OUTPUT_DIR, SNAPSHOT_REQUEST};

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
/// Use, in percent, at which the watchdog asks for a snapshot on its own: later
/// than [`ACT_PERCENT`], so a grow overbrainer is making comes first.
pub const WATCHDOG_ACT_PERCENT: u64 = 97;
/// How long the API may keep showing the old size after a grow before the job
/// is stopped.
pub const GROW_WINDOW: Duration = Duration::from_secs(120);
/// Disk probes in a row that cannot measure the volume before a warning.
const UNMEASURED_WARN: u32 = 3;
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
/// its size, and at least 50 GB more, but never past `max_gb`.
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
        format!(
            "part volume\n$limit du -sk -- {} 2>/dev/null\n",
            quote(volume)
        )
    });
    // A `du` stuck on a network mount is killed rather than left running.
    format!(
        "part() {{ printf '\\n@%s\\n' \"$1\"; }}\n\
         limit=\n\
         command -v timeout >/dev/null 2>&1 && limit='timeout -k 5 50'\n\
         part checkpoint\n\
         last=$(ls -1d -- {output}/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -n 1)\n\
         [ -n \"$last\" ] && $limit du -sk -- {output}/checkpoint-\"$last\" 2>/dev/null\n\
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

/// A grow Runpod accepted whose new size the API did not report yet.
#[derive(Debug, Clone, Copy)]
struct Growing {
    to_gb: u32,
    since: Instant,
}

/// What a read of the volume's size after a grow shows.
enum GrowCheck {
    /// The new size.
    Grown(u32),
    /// A smaller size, or no answer, still within the time allowed.
    Waiting,
    /// No new size in time: why.
    Failed(&'static str),
}

/// The disk policy of one followed run: it warns from [`WARN_PERCENT`] on,
/// and at [`ACT_PERCENT`], or when the next checkpoint would not fit, grows the
/// network volume within `max_volume_gb` or stops the job with a snapshot.
pub struct DiskWatch<'a, E> {
    executor: &'a E,
    client: &'a RunpodClient,
    run: &'a RunRecord,
    volume: Option<VolumeDisk<'a>>,
    /// The volume's size in GB, as the API last reported it.
    size_gb: Option<u32>,
    /// The size last written to [`VOLUME_SIZE_FILE`].
    handed_gb: Option<u32>,
    probe: DiskProbe,
    probed_at: Option<Instant>,
    /// Disk probes in a row that could not measure the volume.
    unmeasured: u32,
    warned: Option<u64>,
    growing: Option<Growing>,
    grow_window: Duration,
    /// Whether Runpod refused a grow: the volume is not grown again.
    cannot_grow: bool,
    /// Whether it said the job cannot save a snapshot (an older overbrainer
    /// started it): said once, as it only warns then.
    said_no_snapshot: bool,
    /// Whether it does nothing more: the snapshot was asked for, or the disk
    /// cannot be measured.
    idle: bool,
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
            handed_gb: None,
            probe: DiskProbe::default(),
            probed_at: None,
            unmeasured: 0,
            warned: None,
            growing: None,
            grow_window: GROW_WINDOW,
            cannot_grow: false,
            said_no_snapshot: false,
            idle: false,
        }
    }

    /// The same policy, waiting `window` rather than [`GROW_WINDOW`] for a grow
    /// to show in the API.
    #[must_use]
    pub fn with_grow_window(mut self, window: Duration) -> Self {
        self.grow_window = window;
        self
    }

    /// A policy that does nothing: for a run whose disk cannot be measured,
    /// such as one on a network volume whose ID `pod.json` does not hold.
    #[must_use]
    pub fn idle(mut self) -> Self {
        self.idle = true;
        self
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
        if self.idle {
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
    /// directory's file system from `sample`, unless it is shared: `df` then
    /// shows the whole cluster, which says nothing of what the run may write.
    fn usage(&self, sample: &SystemSample) -> Option<Usage> {
        if self.volume.is_some() {
            return Some(Usage::of_volume(
                self.probe.volume_used_bytes?,
                self.size_gb?,
            ));
        }
        sample
            .run_disk()
            .filter(|disk| !disk.shared)
            .map(Usage::of_disk)
    }

    /// When [`DISK_PROBE_EVERY`] passed since the last time, reads the volume's
    /// size again (it may have been resized elsewhere) and runs the disk
    /// probe; a failed probe keeps what the last one found.
    async fn probe_if_due(&mut self) {
        if self
            .probed_at
            .is_some_and(|at| at.elapsed() < DISK_PROBE_EVERY)
        {
            return;
        }
        self.probed_at = Some(Instant::now());
        if self.volume.is_some() {
            self.read_size().await;
        }
        let script =
            disk_probe_script(&self.run.remote_dir, self.volume.map(|volume| volume.mount));
        let answer = match tokio::time::timeout(PROBE_TIMEOUT, self.executor.probe(&script)).await {
            Ok(answer) => answer
                .map(|output| parse_disk_probe(&String::from_utf8_lossy(&output)))
                .map_err(|error| error.to_string()),
            Err(_) => Err("no answer in time".to_string()),
        };
        self.probed(answer);
    }

    /// Keeps what the disk probe found; warns once the volume could not be
    /// measured [`UNMEASURED_WARN`] times in a row.
    fn probed(&mut self, answer: Result<DiskProbe, String>) {
        let why = match answer {
            Ok(found) => {
                self.probe.checkpoint_bytes = found.checkpoint_bytes;
                if self.volume.is_none() || found.volume_used_bytes.is_some() {
                    self.probe.volume_used_bytes = found.volume_used_bytes;
                    self.unmeasured = 0;
                    return;
                }
                "du gave no size".to_string()
            },
            Err(error) => error,
        };
        tracing::debug!("cannot measure the pod's disk: {why}");
        self.unmeasured_again(&why);
    }

    /// Counts one more probe that could not measure the disk, for `why`, and
    /// warns once the volume went unmeasured [`UNMEASURED_WARN`] times in a row.
    fn unmeasured_again(&mut self, why: &str) {
        self.unmeasured = self.unmeasured.saturating_add(1);
        if self.volume.is_none() || self.unmeasured != UNMEASURED_WARN {
            return;
        }
        tracing::warn!(
            "cannot measure the network volume {UNMEASURED_WARN} times in a row ({why}): \
             the disk policy uses its last known use until it can"
        );
    }

    /// The volume's size in GB from the API; a size of 0 is no answer.
    async fn fetch_size(&self, id: &str) -> Result<u32, String> {
        match self.client.get_network_volume(id).await {
            Ok(Some(found)) if found.size > 0 => Ok(found.size),
            Ok(Some(_)) => Err("Runpod gave no size".to_string()),
            Ok(None) => Err("not found".to_string()),
            Err(error) => Err(error.to_string()),
        }
    }

    /// Reads the volume's size from the API and hands it to the watchdog
    /// (during a grow, never less than the size asked for). Returns it, or
    /// `None` without a volume or when it cannot be read: the failure is
    /// logged, and the next probe tries again.
    pub async fn read_size(&mut self) -> Option<u32> {
        let volume = self.volume?;
        let size = self
            .fetch_size(volume.id)
            .await
            .inspect_err(|error| {
                tracing::debug!("cannot read network volume {}: {error}", volume.id);
            })
            .ok()?;
        self.size_gb = Some(size);
        let handed = self.growing.map_or(size, |growing| size.max(growing.to_gb));
        self.hand_size(handed).await;
        Some(size)
    }

    /// Writes `size` where the watchdog reads the volume's size, unless it is
    /// there already; a failure is tried again the next time.
    async fn hand_size(&mut self, size: u32) {
        if self.handed_gb == Some(size) {
            return;
        }
        let path = format!("{}/{VOLUME_SIZE_FILE}", self.run.remote_dir);
        match self.executor.put_file(&path, &size.to_string()).await {
            Ok(()) => self.handed_gb = Some(size),
            Err(error) => tracing::debug!("cannot hand the volume size to the watchdog: {error}"),
        }
    }

    /// Warns and acts on `usage`, with the newest checkpoint of
    /// `checkpoint_bytes`. While a grow is pending, it only checks the grow.
    pub async fn check(&mut self, usage: Usage, checkpoint_bytes: Option<u64>) {
        if self.idle {
            return;
        }
        if self.growing.is_some() {
            self.check_grow().await;
            return;
        }
        let assessment = assess(usage, checkpoint_bytes, self.warned);
        if let Some(percent) = assessment.warn {
            self.warn_full(percent, usage);
        }
        let Some(critical) = assessment.act else {
            return;
        };
        if self.resized_elsewhere().await {
            return;
        }
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

    /// Warns that the disk is `percent` full.
    fn warn_full(&mut self, percent: u64, usage: Usage) {
        self.warned = Some(percent);
        tracing::warn!(
            "the pod's {} is {percent}% full ({} free)",
            self.disk_name(),
            gb(usage.free_bytes())
        );
    }

    /// Whether the volume's size changed since it was last read, as when it is
    /// grown elsewhere: the next sample weighs it against its new size.
    async fn resized_elsewhere(&mut self) -> bool {
        let known = self.size_gb;
        self.volume.is_some()
            && self
                .read_size()
                .await
                .is_some_and(|size| Some(size) != known)
    }

    /// The disk's name in a warning.
    fn disk_name(&self) -> &'static str {
        if self.volume.is_some() {
            "network volume"
        } else {
            "disk"
        }
    }

    /// Asks Runpod to grow the volume, when `max_volume_gb` leaves room and
    /// Runpod never refused; whether it accepted. The size asked for goes to
    /// the watchdog at once, so its own rule weighs the volume against it.
    async fn grow(&mut self, why: &str) -> bool {
        let (Some(volume), Some(size)) = (self.volume, self.size_gb) else {
            return false;
        };
        let Some(to_gb) = volume.max_gb.and_then(|max| grown_size(size, max)) else {
            return false;
        };
        if self.cannot_grow {
            return false;
        }
        let resized = self.client.resize_network_volume(volume.id, to_gb).await;
        let note = match &resized {
            Ok(_) => format!(
                "{why}: growing network volume {} from {size} to {to_gb} GB (max_volume_gb); \
                 a volume never shrinks back: it stays billed at {to_gb} GB after the run, \
                 for every pod using it",
                volume.id
            ),
            Err(error) => format!("cannot grow network volume {}: {error}", volume.id),
        };
        tracing::warn!("{note}");
        let Ok(answer) = resized else {
            self.cannot_grow = true;
            return false;
        };
        self.growing = Some(Growing {
            to_gb,
            since: Instant::now(),
        });
        self.hand_size(to_gb).await;
        if answer.size >= to_gb {
            self.grown(answer.size);
        }
        true
    }

    /// The volume is now `size_gb`.
    fn grown(&mut self, size_gb: u32) {
        self.growing = None;
        self.size_gb = Some(size_gb);
        self.warned = None;
    }

    /// Whether the pending grow shows in the API: then the new size counts.
    /// The job is stopped once the API kept showing a smaller size for the
    /// grow window, or could not be read for three times as long.
    async fn check_grow(&mut self) {
        let (Some(growing), Some(volume)) = (self.growing, self.volume) else {
            return;
        };
        let waited = growing.since.elapsed();
        let check = match self.fetch_size(volume.id).await {
            Ok(size) if size >= growing.to_gb => GrowCheck::Grown(size),
            Ok(_) if waited >= self.grow_window => {
                GrowCheck::Failed("the network volume did not grow")
            },
            Err(_) if waited >= self.grow_window.saturating_mul(3) => {
                GrowCheck::Failed("the size of the network volume cannot be read after its grow")
            },
            Ok(_) | Err(_) => GrowCheck::Waiting,
        };
        match check {
            GrowCheck::Grown(size) => {
                self.grown(size);
                tracing::warn!("network volume {} grew to {size} GB", volume.id);
            },
            GrowCheck::Failed(why) => {
                self.growing = None;
                self.stop(why).await;
            },
            GrowCheck::Waiting => {},
        }
    }

    /// Stops the job with a snapshot for `why`. The job of a run an older
    /// overbrainer started cannot save one: it is only warned about, once, and
    /// keeps running.
    async fn stop(&mut self, why: &str) {
        if self.run.snapshots {
            self.ask_snapshot(why).await;
        } else if !self.said_no_snapshot {
            self.said_no_snapshot = true;
            tracing::warn!("{why}, but {}", RunError::NoSnapshots(self.run.id.clone()));
        }
    }

    /// Asks the job for a snapshot for `why`, unless one was asked for
    /// already (its reason stays); tried again at the next sample when the
    /// request cannot be written.
    async fn ask_snapshot(&mut self, why: &str) {
        let path = format!("{}/{SNAPSHOT_REQUEST}", self.run.remote_dir);
        let asked = self
            .executor
            .read_from(&path, 0, 64)
            .await
            .is_ok_and(|content| !content.iter().all(u8::is_ascii_whitespace));
        let note = if asked {
            self.idle = true;
            format!("{why}; a snapshot was already asked for")
        } else {
            match request_snapshot(self.executor, self.run, SnapshotReason::Disk).await {
                Ok(()) => {
                    self.idle = true;
                    format!("{why}: the job is stopped with a snapshot")
                },
                Err(error) => format!("{why}, but the snapshot cannot be asked for: {error}"),
            }
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
        assert!(script.contains(&format!("\nDISK_ACT={WATCHDOG_ACT_PERCENT}\n")));
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
            script.contains("part volume\n$limit du -sk -- '/workspace/data'"),
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
