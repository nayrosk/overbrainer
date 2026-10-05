//! `overbrainer push RUN_ID`: the model of a finished run, with a generated
//! model card, to a Hugging Face model repo in one commit.
//!
//! Ctrl-C cancels the push: nothing is committed before every upload
//! finished, and a rerun resumes.

use std::future::{Future, ready};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context as _, anyhow, bail};
use tokio::sync::mpsc;

use super::PushArgs;
pub(crate) use super::export::size_words;
use super::export::{Action, model_of};
use super::front::Frontend;
use super::train::{hf_token, warn};
use crate::config::{EnvSource, Settings, Source};
use crate::events::{Event, EventBus};
use crate::hub::record::{HUB_DIR, PushRecord};
use crate::hub::{
    Commit, CommitRequest, HfHub, Hub, HubError, Progress, ProgressSink, RepoId, RepoState,
    UploadFile, card, files,
};
use crate::runs::{RUNS_DIR, RunRecord, Runs, rfc3339, write_atomic};
use crate::train::OUTPUT_DIR;

/// The card in the repo, and the copy a dry run writes in `runs/<id>/hub/`.
const CARD_FILE: &str = "README.md";

/// What a push stopped by Ctrl-C (or the TUI) says.
pub(crate) const PUSH_CANCELLED: &str =
    "push cancelled before its commit finished; run it again to resume";

/// At most one progress event per interval.
const PROGRESS_EVERY: Duration = Duration::from_secs(1);

/// What `overbrainer push` was asked, besides the run.
#[derive(Debug, Clone, Default)]
pub(crate) struct PushOptions {
    /// `NAMESPACE/NAME`, overriding `[hub] repo`.
    pub repo: Option<String>,
    /// Create the repo public whatever `[hub] private` says.
    pub public: bool,
    /// Replace a README.md overbrainer did not write.
    pub overwrite_card: bool,
    /// Show what would be pushed, push nothing.
    pub dry_run: bool,
}

/// The run a push sends: the runs of the project, its settings and the run's id.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PushTarget<'a> {
    /// The project's runs.
    pub runs: &'a Runs,
    /// The project's settings.
    pub settings: &'a Settings,
    /// The run to push.
    pub run_id: &'a str,
}

/// What a push did, or would do on a dry run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Pushed {
    /// The repo pushed to.
    pub repo: RepoId,
    /// The commit URL; `None` on a dry run.
    pub url: Option<String>,
    /// The run's files pushed, the card aside.
    pub files: usize,
    /// Their total size.
    pub bytes: u64,
    /// Whether the repo's own README.md was kept.
    pub card_kept: bool,
}

/// Runs `overbrainer push`.
///
/// # Errors
///
/// Returns an error when the run cannot be pushed, the token is missing, or
/// the push fails or is interrupted.
pub(super) async fn run(
    project_dir: &Path,
    args: &PushArgs,
    front: &Frontend,
) -> anyhow::Result<()> {
    let settings = Source::from(EnvSource::Process).load(project_dir)?;
    let runs = Runs::new(project_dir);
    let opts = PushOptions {
        repo: args.repo.clone(),
        public: args.public,
        overwrite_card: args.overwrite_card,
        dry_run: args.dry_run,
    };
    let push = async {
        if settings.hf_token.is_none() && opts.dry_run {
            let target = PushTarget {
                runs: &runs,
                settings: &settings,
                run_id: &args.run_id,
            };
            push_run(&Offline, target, &opts, front).await
        } else {
            push_with_token(&runs, &settings, &args.run_id, &opts, front).await
        }
    };
    // A push whose commit is done wins over a Ctrl-C that came with it.
    tokio::select! {
        biased;
        pushed = push => pushed.map(|_| ()),
        signal = tokio::signal::ctrl_c() => {
            signal.context("cannot catch Ctrl-C")?;
            bail!("{PUSH_CANCELLED}")
        },
    }
}

/// What a push of a run would send, read from its files alone: no network.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PushPlan {
    /// The run.
    pub run_id: String,
    /// `[hub] repo`, else `<you>/<project>` with `_` as `-`.
    pub repo: String,
    /// Whether the repo is created private.
    pub private: bool,
    /// The run's files pushed, the card aside.
    pub files: usize,
    /// Their total size.
    pub bytes: u64,
}

/// What a push of run `run_id` with the default options would send, for the
/// TUI's confirmation.
///
/// # Errors
///
/// Returns the same refusals as [`push_run`] for a run that cannot be pushed.
pub(crate) fn plan(runs: &Runs, settings: &Settings, run_id: &str) -> anyhow::Result<PushPlan> {
    let (_, _, files) = pushable(runs, run_id)?;
    let repo = settings
        .hub
        .repo
        .clone()
        .unwrap_or_else(|| format!("<you>/{}", settings.project.name.replace('_', "-")));
    Ok(PushPlan {
        run_id: run_id.to_string(),
        repo,
        private: settings.hub.private,
        files: files.len(),
        bytes: files.iter().map(|file| file.size).sum(),
    })
}

/// The record of run `run_id`, its directory and the files of its `output/`
/// to push.
///
/// # Errors
///
/// Returns an error when the run cannot be pushed: its state, its files, a
/// checkpoint without a model in `output/`, or nothing to push.
fn pushable(runs: &Runs, run_id: &str) -> anyhow::Result<(RunRecord, PathBuf, Vec<UploadFile>)> {
    let record = runs.load(run_id)?;
    let model = model_of(runs, &record, Action::Push)?;
    if model != OUTPUT_DIR {
        bail!(
            "run {run_id} has no model in {OUTPUT_DIR}/ (only checkpoint {model}): export or \
             resume it first"
        );
    }
    let run_dir = runs.run_dir(run_id)?;
    let files = files::select(&run_dir.join(OUTPUT_DIR))?;
    if files.is_empty() {
        bail!("run {run_id} has nothing to push in {OUTPUT_DIR}/");
    }
    Ok((record, run_dir, files))
}

/// Pushes run `run_id` of `runs` to the Hugging Face Hub with the token of
/// `settings`, resolved only now; shared by the command and the TUI task.
///
/// # Errors
///
/// Returns an error when the token is missing or cannot be resolved, the run
/// cannot be pushed, or the Hub refuses it.
pub(crate) async fn push_with_token(
    runs: &Runs,
    settings: &Settings,
    run_id: &str,
    opts: &PushOptions,
    front: &Frontend,
) -> anyhow::Result<Pushed> {
    let Some(token) = &settings.hf_token else {
        bail!(
            "OVERBRAINER_HF_TOKEN is not set: a Hugging Face token with write access is needed \
             to push"
        );
    };
    let token = hf_token(token).await?;
    let hub = HfHub::new(settings.hub.base_url.as_deref(), &token)?;
    let target = PushTarget {
        runs,
        settings,
        run_id,
    };
    push_run(&hub, target, opts, front).await
}

/// Pushes run `target` to `hub`; shared by the command, the TUI task and the
/// push after training. Dropping the future cancels the push before its
/// commit.
///
/// # Errors
///
/// Returns an error when the run cannot be pushed, or the Hub refuses it.
pub(crate) async fn push_run<H: Hub>(
    hub: &H,
    target: PushTarget<'_>,
    opts: &PushOptions,
    front: &Frontend,
) -> anyhow::Result<Pushed> {
    let PushTarget {
        runs,
        settings,
        run_id,
    } = target;
    let (record, run_dir, files) = pushable(runs, run_id)?;
    let repo = repo_of(hub, settings, opts).await?;
    let private = !opts.public && settings.hub.private;
    let mut input = card::gather(runs, &record, settings, &repo, &files)?;
    // A local base model has no license to look up.
    if !Path::new(&input.base_model).exists() {
        input.license = hub
            .license_of(&input.base_model)
            .await
            .unwrap_or_else(|error| {
                warn(&format!("leaving the license out of the card: {error}"));
                None
            });
    }
    let card = card::render(&input);
    let count = files.len();
    let bytes = files.iter().map(|file| file.size).sum();
    let pushed = Pushed {
        repo,
        url: None,
        files: count,
        bytes,
        card_kept: false,
    };
    if opts.dry_run {
        let staged = DryRun {
            run_dir: &run_dir,
            run_id,
            files: &files,
            card: &card,
            private,
        };
        dry_run(&staged, &pushed, front)?;
        return Ok(pushed);
    }

    let repo = &pushed.repo;
    let private = ensure_repo(hub, repo, private, front).await?;
    let remote = hub.remote_card(repo).await.map_err(hub_error)?;
    // A kept README.md is left out of the commit: re-uploading the copy read
    // above could overwrite an edit made during the upload.
    let card = match remote {
        Some(remote) if !opts.overwrite_card && !card::replaceable(Some(&remote)) => {
            front.line("kept the repo's own README.md (use --overwrite-card to replace it)");
            None
        },
        _ => Some(card),
    };
    let card_kept = card.is_none();
    let paths: Vec<String> = files
        .iter()
        .map(|file| file.path_in_repo.clone())
        .chain(card.is_some().then(|| CARD_FILE.to_string()))
        .collect();
    let message = format!(
        "Upload run {run_id} with overbrainer {}",
        env!("CARGO_PKG_VERSION")
    );
    let guard = front.open_bus();
    let request = CommitRequest {
        files,
        card,
        message,
    };
    let commit = upload(hub, repo, request, run_id, &guard.bus).await;
    guard.close().await;
    let commit = commit.map_err(hub_error)?;
    PushRecord {
        repo: repo.to_string(),
        commit: commit.oid,
        url: commit.url.clone(),
        files: paths,
        private,
        pushed: rfc3339(SystemTime::now()),
    }
    .save(&run_dir)?;
    front.line(&format!(
        "push: {} ({}, {})",
        commit.url,
        files_word(count),
        size_words(bytes)
    ));
    Ok(Pushed {
        url: Some(commit.url),
        card_kept,
        ..pushed
    })
}

/// Makes sure `repo` exists, created with `private` when it does not; its
/// visibility then. An existing repo keeps its own, said when it differs.
async fn ensure_repo<H: Hub>(
    hub: &H,
    repo: &RepoId,
    private: bool,
    front: &Frontend,
) -> anyhow::Result<bool> {
    let state = match hub.ensure_repo(repo, private).await {
        Ok(state) => state,
        Err(HubError::Forbidden { namespace }) => {
            // A failed whoami keeps the plain message.
            let owner = hub.whoami().await.ok();
            return Err(anyhow!(forbidden_message(
                &namespace,
                &repo.name,
                owner.as_deref()
            )));
        },
        Err(other) => return Err(hub_error(other)),
    };
    Ok(match state {
        RepoState::Created { private } => private,
        RepoState::Existing { private: kept } => {
            if kept != private {
                front.line(&format!(
                    "repo {repo} exists and stays {}",
                    visibility(kept)
                ));
            }
            kept
        },
    })
}

/// Why a token cannot write to `namespace`, given the user it belongs to when
/// known. Hub namespaces are case-sensitive, so a name that differs only in
/// case gets its own advice.
fn forbidden_message(namespace: &str, name: &str, owner: Option<&str>) -> String {
    match owner {
        Some(owner) if owner == namespace => None,
        Some(owner) if owner.eq_ignore_ascii_case(namespace) => Some(format!(
            "the Hugging Face namespace is case-sensitive: the token belongs to {owner}, use \
             --repo {owner}/{name} (or [hub] repo)"
        )),
        Some(owner) => Some(format!(
            "the Hugging Face token belongs to {owner}, which cannot write to {namespace}: use \
             a namespace it can write to, or a token with write access to {namespace}"
        )),
        None => None,
    }
    .unwrap_or_else(|| {
        format!(
            "the Hugging Face token needs write access to {namespace}: set OVERBRAINER_HF_TOKEN \
             to a token that has it"
        )
    })
}

/// The repo of a push: `--repo`, else `[hub] repo`, else `<whoami>/<project>`
/// with `_` as `-`. A dry run without a token says `<you>` for the user.
async fn repo_of<H: Hub>(
    hub: &H,
    settings: &Settings,
    opts: &PushOptions,
) -> anyhow::Result<RepoId> {
    let text = if let Some(text) = opts.repo.as_ref().or(settings.hub.repo.as_ref()) {
        text.clone()
    } else {
        let name = settings.project.name.replace('_', "-");
        match hub.whoami().await {
            Ok(user) => format!("{user}/{name}"),
            Err(HubError::Auth) if opts.dry_run => {
                return Ok(RepoId {
                    namespace: "<you>".to_string(),
                    name,
                });
            },
            Err(error) => return Err(hub_error(error)),
        }
    };
    RepoId::parse(&text)
        .with_context(|| format!("`{text}` is not a Hugging Face repo: use NAMESPACE/NAME"))
}

/// What a dry run of a push says and writes, at hand in [`push_run`].
struct DryRun<'a> {
    /// The run's directory.
    run_dir: &'a Path,
    /// The run.
    run_id: &'a str,
    /// The run's files that would be pushed.
    files: &'a [UploadFile],
    /// The card that would be pushed.
    card: &'a str,
    /// Whether the repo would be created private.
    private: bool,
}

/// Says what `pushed` would send and writes the card of `staged` to
/// `runs/<id>/hub/`.
fn dry_run(staged: &DryRun<'_>, pushed: &Pushed, front: &Frontend) -> anyhow::Result<()> {
    let DryRun {
        run_dir,
        run_id,
        files,
        card,
        private,
    } = *staged;
    front.line(&format!(
        "push: {}, {} when created; nothing is sent (--dry-run)",
        pushed.repo,
        visibility(private)
    ));
    for file in files {
        front.line(&format!("{}  {}", file.path_in_repo, size_words(file.size)));
    }
    let dir = run_dir.join(HUB_DIR);
    std::fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
    write_atomic(&dir, CARD_FILE, card.as_bytes())?;
    front.line(&format!(
        "push: {}, {}; card written to {RUNS_DIR}/{run_id}/{HUB_DIR}/{CARD_FILE}",
        files_word(pushed.files),
        size_words(pushed.bytes)
    ));
    Ok(())
}

/// The commit of `request` (no card: README.md stays) to `repo`, its progress
/// published on `bus` as [`Event::Push`] of run `run_id` at most once per
/// [`PROGRESS_EVERY`].
async fn upload<H: Hub>(
    hub: &H,
    repo: &RepoId,
    request: CommitRequest,
    run_id: &str,
    bus: &EventBus,
) -> Result<Commit, HubError> {
    let (sink, mut progress): (ProgressSink, _) = mpsc::unbounded_channel();
    let mut upload = std::pin::pin!(hub.upload(repo, request, sink));
    let mut last: Option<Instant> = None;
    loop {
        tokio::select! {
            result = &mut upload => return result,
            Some(Progress { done, total }) = progress.recv() => {
                if last.is_none_or(|at| at.elapsed() >= PROGRESS_EVERY) {
                    last = Some(Instant::now());
                    bus.publish(Event::Push {
                        run_id: run_id.to_string(),
                        done,
                        total,
                    });
                }
            },
        }
    }
}

/// `error` as the user reads it: an auth error names the token's variable,
/// never its value.
fn hub_error(error: HubError) -> anyhow::Error {
    match error {
        HubError::Auth => anyhow!(
            "Hugging Face refused the token: set OVERBRAINER_HF_TOKEN to a valid token with \
             write access"
        ),
        HubError::Forbidden { namespace } => anyhow!(
            "the Hugging Face token needs write access to {namespace}: set OVERBRAINER_HF_TOKEN \
             to a token that has it"
        ),
        other => other.into(),
    }
}

/// `private` or `public`, as the push says it.
fn visibility(private: bool) -> &'static str {
    if private { "private" } else { "public" }
}

/// `1 file`, `2 files`.
pub(crate) fn files_word(count: usize) -> String {
    if count == 1 {
        "1 file".to_string()
    } else {
        format!("{count} files")
    }
}

/// The Hub of a dry run without a token: it knows no user and reaches nothing.
struct Offline;

impl Hub for Offline {
    /// Reports no user and refuses every call that would reach the network.
    fn whoami(&self) -> impl Future<Output = Result<String, HubError>> + Send {
        ready(Err(HubError::Auth))
    }

    /// Refuses with an authentication error: a dry run has no token.
    fn ensure_repo(
        &self,
        _repo: &RepoId,
        _private: bool,
    ) -> impl Future<Output = Result<RepoState, HubError>> + Send {
        ready(Err(HubError::Auth))
    }

    /// Refuses with an authentication error: a dry run has no token.
    fn remote_card(
        &self,
        _repo: &RepoId,
    ) -> impl Future<Output = Result<Option<String>, HubError>> + Send {
        ready(Err(HubError::Auth))
    }

    /// Refuses with an authentication error: a dry run has no token.
    fn license_of(
        &self,
        _model: &str,
    ) -> impl Future<Output = Result<Option<String>, HubError>> + Send {
        ready(Ok(None))
    }

    /// Refuses with an authentication error: a dry run has no token.
    fn upload(
        &self,
        _repo: &RepoId,
        _commit: CommitRequest,
        _progress: ProgressSink,
    ) -> impl Future<Output = Result<Commit, HubError>> + Send {
        ready(Err(HubError::Auth))
    }
}

/// Shared by the tests of the push and of the push after training.
#[cfg(test)]
pub(crate) mod fixtures {
    use std::future::{Future, ready};
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Mutex};

    use tokio_util::sync::CancellationToken;

    use crate::cli::front::{Frontend, Report};
    use crate::config::{EnvSource, Settings, load_str};
    use crate::events::EventBus;
    use crate::hub::{
        Commit, CommitRequest, Hub, HubError, Progress, ProgressSink, RepoId, RepoState,
    };
    use crate::runs::{RunRecord, RunState, Runs, Snapshot, SnapshotReason};

    /// The id of the run the fake Hub tests push.
    pub(crate) const RUN: &str = "r1";

    /// `(repo, paths, card)` of an upload; no card when README.md is kept.
    pub(crate) type Upload = (String, Vec<String>, Option<String>);

    /// A Hub that records what it is asked and answers as it was set up to.
    #[derive(Default)]
    pub(crate) struct FakeHub {
        /// The visibility of the repo when it exists already.
        pub existing: Option<bool>,
        pub remote_card: Option<String>,
        pub fail_auth: bool,
        /// The namespace `ensure_repo` refuses with `Forbidden`, when set.
        pub forbidden: Option<String>,
        /// The user `whoami` names; `me` when unset.
        pub owner: Option<String>,
        /// Progress reports per upload, at least one.
        pub progress_steps: u64,
        /// `(repo, private)` of each `ensure_repo`.
        pub ensured: Mutex<Vec<(String, bool)>>,
        /// `(repo, paths, card)` of each upload.
        pub uploads: Mutex<Vec<Upload>>,
    }

    impl Hub for FakeHub {
        /// Answers from the fields, and records the repo and the files of each upload.
        fn whoami(&self) -> impl Future<Output = Result<String, HubError>> + Send {
            ready(if self.fail_auth {
                Err(HubError::Auth)
            } else {
                Ok(self.owner.clone().unwrap_or_else(|| "me".into()))
            })
        }

        /// Records the repo and its visibility, and answers from the fields.
        fn ensure_repo(
            &self,
            repo: &RepoId,
            private: bool,
        ) -> impl Future<Output = Result<RepoState, HubError>> + Send {
            if let Ok(mut ensured) = self.ensured.lock() {
                ensured.push((repo.to_string(), private));
            }
            if let Some(namespace) = &self.forbidden {
                return ready(Err(HubError::Forbidden {
                    namespace: namespace.clone(),
                }));
            }
            ready(Ok(match self.existing {
                Some(private) => RepoState::Existing { private },
                None => RepoState::Created { private },
            }))
        }

        /// Answers with the remote card set on the fake.
        fn remote_card(
            &self,
            _repo: &RepoId,
        ) -> impl Future<Output = Result<Option<String>, HubError>> + Send {
            ready(Ok(self.remote_card.clone()))
        }

        /// Answers with a fixed license.
        fn license_of(
            &self,
            _model: &str,
        ) -> impl Future<Output = Result<Option<String>, HubError>> + Send {
            ready(Ok(Some("apache-2.0".into())))
        }

        /// Records the upload and reports progress in the steps set on the fake.
        fn upload(
            &self,
            repo: &RepoId,
            CommitRequest { files, card, .. }: CommitRequest,
            progress: ProgressSink,
        ) -> impl Future<Output = Result<Commit, HubError>> + Send {
            let total: u64 = files.iter().map(|f| f.size).sum();
            let paths = files.into_iter().map(|f| f.path_in_repo).collect();
            if let Ok(mut uploads) = self.uploads.lock() {
                uploads.push((repo.to_string(), paths, card));
            }
            let steps = self.progress_steps.max(1);
            async move {
                for step in 1..=steps {
                    let _ = progress.send(Progress {
                        done: total * step / steps,
                        total,
                    });
                    tokio::task::yield_now().await;
                }
                Ok(Commit {
                    url: "https://hf.co/me/x/commit/1".into(),
                    oid: "1".into(),
                })
            }
        }
    }

    impl FakeHub {
        /// The uploads made so far.
        pub(crate) fn uploads(&self) -> Vec<Upload> {
            self.uploads.lock().map(|u| u.clone()).unwrap_or_default()
        }

        /// The repos ensured so far, with their visibility.
        pub(crate) fn ensured(&self) -> Vec<(String, bool)> {
            self.ensured.lock().map(|e| e.clone()).unwrap_or_default()
        }
    }

    /// A minimal project config, with a mock provider and a local target.
    pub(crate) const CONFIG: &str = r#"[project]
name = "my_proj"

[providers.mock]
protocol = "openai"

[roles]
generator = { provider = "mock", model = "gen" }
parent = { provider = "mock", model = "parent" }
"#;

    /// Settings from `CONFIG` plus `extra`, with a Hugging Face token in the environment.
    pub(crate) fn settings(extra: &str) -> Result<Settings, Box<dyn std::error::Error>> {
        let env = vec![("OVERBRAINER_HF_TOKEN".to_string(), "hf_test".to_string())];
        Ok(load_str(&format!("{CONFIG}{extra}"), EnvSource::Vars(env))?)
    }

    /// A run record in `state`, with no model recorded yet.
    pub(crate) fn record(state: RunState) -> RunRecord {
        RunRecord {
            id: RUN.into(),
            target: "box".into(),
            created: "2026-10-01T12:00:00Z".into(),
            remote_dir: "/w/r1".into(),
            job: None,
            state,
            message: None,
            snapshot: None,
            resumed_from: None,
            snapshots: true,
        }
    }

    /// A project with run [`RUN`] in `state`, its adapter in `output/` unless
    /// `checkpoint_only`, beside Axolotl's README.md and a checkpoint.
    pub(crate) fn project(
        state: RunState,
        checkpoint_only: bool,
    ) -> Result<(tempfile::TempDir, Runs), Box<dyn std::error::Error>> {
        let project = tempfile::tempdir()?;
        let runs = Runs::new(project.path());
        let mut run = record(state);
        if checkpoint_only {
            run.snapshot = Some(Snapshot {
                checkpoint: "output/checkpoint-10".into(),
                step: 10,
                reason: SnapshotReason::Requested,
            });
        }
        runs.save(&run)?;
        let dir = runs.run_dir(RUN)?;
        std::fs::create_dir_all(dir.join("output/checkpoint-10"))?;
        std::fs::write(
            dir.join("axolotl.yaml"),
            "base_model: Qwen/Qwen3-0.6B\nadapter: qlora\n",
        )?;
        std::fs::write(dir.join("output/checkpoint-10/adapter_config.json"), "{}")?;
        std::fs::write(dir.join("output/README.md"), "axolotl")?;
        if !checkpoint_only {
            std::fs::write(dir.join("output/adapter_config.json"), "{}")?;
            std::fs::write(dir.join("output/adapter_model.safetensors"), "weights")?;
        }
        Ok((project, runs))
    }

    /// A TUI front end whose lines land in the returned vector.
    pub(crate) fn front() -> (Frontend, Arc<Mutex<Vec<String>>>) {
        front_with(CancellationToken::new())
    }

    /// [`front`], interrupted when `detach` is cancelled.
    pub(crate) fn front_with(detach: CancellationToken) -> (Frontend, Arc<Mutex<Vec<String>>>) {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&lines);
        let front = Frontend::Tui {
            bus: EventBus::new(),
            detach,
            abandon: Arc::new(AtomicBool::new(false)),
            report: Arc::new(move |report| {
                if let (Report::Line(line), Ok(mut lines)) = (report, seen.lock()) {
                    lines.push(line);
                }
            }),
        };
        (front, lines)
    }

    /// The lines the front end said so far.
    pub(crate) fn said(lines: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
        lines.lock().map(|l| l.clone()).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::{FakeHub, RUN, front, project, record, said, settings};
    use super::*;
    use crate::config::load_str;
    use crate::events::EventBus;
    use crate::hub::card::MARKER;
    use crate::runs::{RunRecord, RunState};

    /// The result of a test that can fail with any error.
    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// Runs `push_run` against `hub` and returns its result with the lines said.
    async fn push(
        hub: &FakeHub,
        runs: &Runs,
        settings: &Settings,
        opts: &PushOptions,
    ) -> (anyhow::Result<Pushed>, Vec<String>) {
        let (front, lines) = front();
        let target = PushTarget {
            runs,
            settings,
            run_id: RUN,
        };
        let pushed = push_run(hub, target, opts, &front).await;
        (pushed, said(&lines))
    }

    /// A running run is refused and nothing is uploaded.
    #[tokio::test]
    async fn refuses_a_running_run() -> TestResult {
        let (_project, runs) = project(RunState::Running, false)?;
        let hub = FakeHub::default();
        let (pushed, _) = push(&hub, &runs, &settings("")?, &PushOptions::default()).await;
        let error = pushed.err().ok_or("pushed a running run")?;
        assert_eq!(
            error.to_string(),
            "run r1 is running: only a succeeded or stopped run can be pushed"
        );
        let refusal = |record: &RunRecord| -> Result<String, Box<dyn std::error::Error>> {
            runs.save(record)?;
            let error = model_of(&runs, record, Action::Push)
                .err()
                .ok_or("accepted")?;
            Ok(error.to_string())
        };
        let mut missing = record(RunState::Succeeded);
        missing.message = Some("artifacts not retrieved: download failed".into());
        assert_eq!(
            refusal(&missing)?,
            "the results of run r1 were not retrieved: retrieve them with `overbrainer train \
             attach r1`, then push"
        );
        let dir = runs.run_dir(RUN)?;
        std::fs::remove_file(dir.join("output/adapter_config.json"))?;
        assert_eq!(
            refusal(&record(RunState::Succeeded))?,
            "runs/r1/output holds no model: nothing to push"
        );
        std::fs::remove_file(dir.join("axolotl.yaml"))?;
        assert_eq!(
            refusal(&record(RunState::Succeeded))?,
            "runs/r1/axolotl.yaml is missing: the push needs it"
        );
        assert_eq!(hub.uploads().len(), 0, "nothing uploaded");
        Ok(())
    }

    /// A stopped run that has only a checkpoint is refused.
    #[tokio::test]
    async fn refuses_a_checkpoint_only_stopped_run() -> TestResult {
        let (_project, runs) = project(RunState::Stopped, true)?;
        let hub = FakeHub::default();
        let (pushed, _) = push(&hub, &runs, &settings("")?, &PushOptions::default()).await;
        let error = pushed.err().ok_or("pushed a checkpoint")?;
        assert_eq!(
            error.to_string(),
            "run r1 has no model in output/ (only checkpoint output/checkpoint-10): export or \
             resume it first"
        );
        assert_eq!(hub.uploads().len(), 0, "nothing uploaded");
        Ok(())
    }

    /// Without `--repo` or `[hub].repo`, the repo is the user name and the project name.
    #[tokio::test]
    async fn default_repo_is_whoami_slash_project() -> TestResult {
        let (_project, runs) = project(RunState::Succeeded, false)?;
        let hub = FakeHub::default();
        let (pushed, lines) = push(&hub, &runs, &settings("")?, &PushOptions::default()).await;
        let pushed = pushed?;
        assert_eq!(pushed.repo.to_string(), "me/my-proj");
        assert_eq!(pushed.url.as_deref(), Some("https://hf.co/me/x/commit/1"));
        assert_eq!((pushed.files, pushed.bytes), (2, 9));
        let uploads = hub.uploads();
        let [(repo, paths, Some(card))] = uploads.as_slice() else {
            return Err("one upload with a card expected".into());
        };
        assert_eq!(repo, "me/my-proj");
        assert_eq!(paths, &["adapter_config.json", "adapter_model.safetensors"]);
        assert!(card.contains("license: apache-2.0"), "{card}");
        assert!(
            lines.contains(&"push: https://hf.co/me/x/commit/1 (2 files, 0.0 MB)".to_string()),
            "{lines:?}"
        );
        // A configured repo wins over whoami, `--repo` over both.
        let configured = settings("\n[hub]\nrepo = \"org/model\"\n")?;
        let (pushed, _) = push(&hub, &runs, &configured, &PushOptions::default()).await;
        assert_eq!(pushed?.repo.to_string(), "org/model");
        let opts = PushOptions {
            repo: Some("org/other".into()),
            ..PushOptions::default()
        };
        let (pushed, _) = push(&hub, &runs, &configured, &opts).await;
        assert_eq!(pushed?.repo.to_string(), "org/other");
        let opts = PushOptions {
            repo: Some("not a repo".into()),
            ..PushOptions::default()
        };
        let (pushed, _) = push(&hub, &runs, &configured, &opts).await;
        let error = pushed.err().ok_or("pushed to a bad repo")?;
        assert!(error.to_string().contains("not a repo"), "{error}");
        Ok(())
    }

    /// `--public` wins over `[hub].private`.
    #[tokio::test]
    async fn public_flag_wins_over_config() -> TestResult {
        let (_project, runs) = project(RunState::Succeeded, false)?;
        let hub = FakeHub::default();
        let private = settings("\n[hub]\nprivate = true\n")?;
        let opts = PushOptions {
            public: true,
            ..PushOptions::default()
        };
        push(&hub, &runs, &private, &opts).await.0?;
        push(&hub, &runs, &private, &PushOptions::default())
            .await
            .0?;
        let public = settings("\n[hub]\nprivate = false\n")?;
        push(&hub, &runs, &public, &PushOptions::default())
            .await
            .0?;
        assert_eq!(
            hub.ensured(),
            [
                ("me/my-proj".to_string(), false),
                ("me/my-proj".to_string(), true),
                ("me/my-proj".to_string(), false),
            ]
        );
        Ok(())
    }

    /// A dry run writes the card and uploads nothing.
    #[tokio::test]
    async fn dry_run_writes_the_card_and_uploads_nothing() -> TestResult {
        let (_project, runs) = project(RunState::Succeeded, false)?;
        let hub = FakeHub::default();
        let opts = PushOptions {
            dry_run: true,
            ..PushOptions::default()
        };
        let (pushed, lines) = push(&hub, &runs, &settings("")?, &opts).await;
        let pushed = pushed?;
        assert_eq!(pushed.url, None);
        assert_eq!((pushed.files, pushed.bytes), (2, 9));
        assert!(hub.uploads().is_empty() && hub.ensured().is_empty());
        let card = std::fs::read_to_string(runs.run_dir(RUN)?.join("hub/README.md"))?;
        assert!(card.contains(MARKER), "{card}");
        assert!(!runs.run_dir(RUN)?.join("hub/push.json").exists());
        assert_eq!(
            lines,
            [
                "push: me/my-proj, private when created; nothing is sent (--dry-run)",
                "adapter_config.json  0.0 MB",
                "adapter_model.safetensors  0.0 MB",
                "push: 2 files, 0.0 MB; card written to runs/r1/hub/README.md",
            ]
        );
        Ok(())
    }

    /// A card that overbrainer did not write is kept unless the flag is given.
    #[tokio::test]
    async fn foreign_card_is_kept_without_the_flag() -> TestResult {
        let (_project, runs) = project(RunState::Succeeded, false)?;
        let hub = FakeHub {
            existing: Some(false),
            remote_card: Some("# my own card\n".into()),
            ..FakeHub::default()
        };
        let (pushed, lines) = push(&hub, &runs, &settings("")?, &PushOptions::default()).await;
        assert!(pushed?.card_kept);
        let uploads = hub.uploads();
        let [(_, paths, card)] = uploads.as_slice() else {
            return Err("one upload expected".into());
        };
        assert_eq!(card, &None, "the kept card is not uploaded again");
        assert_eq!(paths, &["adapter_config.json", "adapter_model.safetensors"]);
        let text = std::fs::read_to_string(runs.run_dir(RUN)?.join("hub/push.json"))?;
        let saved: crate::hub::record::PushRecord = serde_json::from_str(&text)?;
        assert_eq!(
            saved.files,
            ["adapter_config.json", "adapter_model.safetensors"],
            "the record lists what the commit wrote"
        );
        assert!(
            lines.contains(&"repo me/my-proj exists and stays public".to_string()),
            "{lines:?}"
        );
        assert!(
            lines.contains(
                &"kept the repo's own README.md (use --overwrite-card to replace it)".to_string()
            ),
            "{lines:?}"
        );
        Ok(())
    }

    /// A card that carries the overbrainer marker is replaced.
    #[tokio::test]
    async fn marked_card_is_replaced() -> TestResult {
        let (_project, runs) = project(RunState::Succeeded, false)?;
        let old = format!("old card\n{MARKER}\n");
        let hub = FakeHub {
            existing: Some(true),
            remote_card: Some(old.clone()),
            ..FakeHub::default()
        };
        let (pushed, lines) = push(&hub, &runs, &settings("")?, &PushOptions::default()).await;
        assert!(!pushed?.card_kept);
        let uploads = hub.uploads();
        let [(_, _, Some(card))] = uploads.as_slice() else {
            return Err("one upload with a card expected".into());
        };
        assert_ne!(card, &old);
        assert!(
            card.contains(MARKER) && card.contains("me/my-proj"),
            "{card}"
        );
        assert!(
            !lines.iter().any(|line| line.contains("exists and stays")),
            "same visibility, nothing to say: {lines:?}"
        );
        Ok(())
    }

    /// `--overwrite-card` replaces a card that overbrainer did not write.
    #[tokio::test]
    async fn overwrite_card_replaces_a_foreign_card() -> TestResult {
        let (_project, runs) = project(RunState::Succeeded, false)?;
        let hub = FakeHub {
            existing: Some(true),
            remote_card: Some("# my own card\n".into()),
            ..FakeHub::default()
        };
        let opts = PushOptions {
            overwrite_card: true,
            ..PushOptions::default()
        };
        let (pushed, _) = push(&hub, &runs, &settings("")?, &opts).await;
        assert!(!pushed?.card_kept);
        let uploads = hub.uploads();
        let [(_, _, Some(card))] = uploads.as_slice() else {
            return Err("one upload with a card expected".into());
        };
        assert!(card.contains(MARKER), "{card}");
        Ok(())
    }

    /// An authentication error names the token variable and never its value.
    #[tokio::test]
    async fn auth_error_names_the_token_variable_not_its_value() -> TestResult {
        let (_project, runs) = project(RunState::Succeeded, false)?;
        let hub = FakeHub {
            fail_auth: true,
            ..FakeHub::default()
        };
        let (pushed, lines) = push(&hub, &runs, &settings("")?, &PushOptions::default()).await;
        let error = pushed.err().ok_or("pushed with a refused token")?;
        let text = format!("{error:#}");
        assert!(
            text.contains("OVERBRAINER_HF_TOKEN"),
            "the error must name the variable"
        );
        assert!(
            !text.contains("hf_test"),
            "the error must not hold the token"
        );
        assert!(
            !lines.iter().any(|line| line.contains("hf_test")),
            "no line may hold the token"
        );
        assert_eq!(hub.uploads().len(), 0, "nothing uploaded");
        Ok(())
    }

    /// The message of a push to `nayrosk/x` refused for `nayrosk`, with the
    /// token owned by `owner` (`None`: whoami fails).
    async fn forbidden_push(owner: Option<&str>) -> Result<String, Box<dyn std::error::Error>> {
        let (_project, runs) = project(RunState::Succeeded, false)?;
        let hub = FakeHub {
            forbidden: Some("nayrosk".into()),
            owner: owner.map(Into::into),
            fail_auth: owner.is_none(),
            ..FakeHub::default()
        };
        let opts = PushOptions {
            repo: Some("nayrosk/x".into()),
            ..PushOptions::default()
        };
        let (pushed, _) = push(&hub, &runs, &settings("")?, &opts).await;
        let error = pushed.err().ok_or("pushed to a forbidden namespace")?;
        assert_eq!(hub.uploads().len(), 0, "nothing uploaded");
        Ok(format!("{error:#}"))
    }

    /// A refused namespace that differs from the user's only in case says so.
    #[tokio::test]
    async fn forbidden_namespace_differing_in_case_says_so() -> TestResult {
        let text = forbidden_push(Some("Nayrosk")).await?;
        assert_eq!(
            text,
            "the Hugging Face namespace is case-sensitive: the token belongs to Nayrosk, use \
             --repo Nayrosk/x (or [hub] repo)"
        );
        Ok(())
    }

    /// A refused namespace of another user names the owner.
    #[tokio::test]
    async fn forbidden_namespace_of_another_user_names_the_owner() -> TestResult {
        let text = forbidden_push(Some("someone")).await?;
        assert_eq!(
            text,
            "the Hugging Face token belongs to someone, which cannot write to nayrosk: use a \
             namespace it can write to, or a token with write access to nayrosk"
        );
        Ok(())
    }

    /// A refused namespace keeps the plain message when `whoami` fails.
    #[tokio::test]
    async fn forbidden_namespace_keeps_the_plain_message_when_whoami_fails() -> TestResult {
        let text = forbidden_push(None).await?;
        assert_eq!(
            text,
            "the Hugging Face token needs write access to nayrosk: set OVERBRAINER_HF_TOKEN to a \
             token that has it"
        );
        Ok(())
    }

    /// Progress is published at most once a second.
    #[tokio::test]
    async fn progress_is_published_at_most_once_a_second() -> TestResult {
        let hub = FakeHub {
            progress_steps: 5,
            ..FakeHub::default()
        };
        let bus = EventBus::new();
        let mut events = bus.subscribe();
        let repo = RepoId::parse("me/x").ok_or("bad repo")?;
        let files = vec![UploadFile {
            local: "a".into(),
            path_in_repo: "a".into(),
            size: 100,
        }];
        let request = CommitRequest {
            files,
            card: None,
            message: String::new(),
        };
        upload(&hub, &repo, request, RUN, &bus).await?;
        let mut pushes = Vec::new();
        while let Ok(event) = events.try_recv() {
            pushes.push(event);
        }
        assert_eq!(
            pushes,
            [Event::Push {
                run_id: RUN.into(),
                done: 20,
                total: 100
            }],
            "five reports within a second make one event"
        );
        Ok(())
    }

    /// The plan says what a push sends without any network call.
    #[test]
    fn plan_says_what_a_push_sends_without_the_network() -> TestResult {
        let (_project, runs) = project(RunState::Succeeded, false)?;
        let planned = plan(&runs, &settings("")?, RUN)?;
        assert_eq!(
            planned,
            PushPlan {
                run_id: RUN.into(),
                repo: "<you>/my-proj".into(),
                private: true,
                files: 2,
                bytes: 9,
            }
        );
        let configured = settings("\n[hub]\nrepo = \"org/model\"\nprivate = false\n")?;
        let planned = plan(&runs, &configured, RUN)?;
        assert_eq!(
            (planned.repo.as_str(), planned.private),
            ("org/model", false)
        );
        runs.save(&record(RunState::Running))?;
        let error = plan(&runs, &configured, RUN)
            .err()
            .ok_or("planned a running run")?;
        assert_eq!(
            error.to_string(),
            "run r1 is running: only a succeeded or stopped run can be pushed"
        );
        let (_project, runs) = project(RunState::Stopped, true)?;
        let error = plan(&runs, &configured, RUN)
            .err()
            .ok_or("planned a checkpoint")?;
        assert_eq!(
            error.to_string(),
            "run r1 has no model in output/ (only checkpoint output/checkpoint-10): export or \
             resume it first"
        );
        Ok(())
    }

    /// `push_with_token` refuses when no token is set.
    #[tokio::test]
    async fn push_with_token_refuses_without_a_token() -> TestResult {
        let (_project, runs) = project(RunState::Succeeded, false)?;
        let settings = load_str(super::fixtures::CONFIG, EnvSource::Vars(Vec::new()))?;
        let (front, _) = front();
        let error = push_with_token(&runs, &settings, RUN, &PushOptions::default(), &front)
            .await
            .err()
            .ok_or("pushed without a token")?;
        assert_eq!(
            error.to_string(),
            "OVERBRAINER_HF_TOKEN is not set: a Hugging Face token with write access is needed \
             to push"
        );
        Ok(())
    }

    /// A finished push is saved in the run's push record.
    #[tokio::test]
    async fn push_record_is_saved() -> TestResult {
        let (_project, runs) = project(RunState::Succeeded, false)?;
        let hub = FakeHub::default();
        push(&hub, &runs, &settings("")?, &PushOptions::default())
            .await
            .0?;
        let text = std::fs::read_to_string(runs.run_dir(RUN)?.join("hub/push.json"))?;
        let saved: crate::hub::record::PushRecord = serde_json::from_str(&text)?;
        assert_eq!(saved.repo, "me/my-proj");
        assert_eq!(saved.commit, "1");
        assert_eq!(saved.url, "https://hf.co/me/x/commit/1");
        assert_eq!(
            saved.files,
            [
                "adapter_config.json",
                "adapter_model.safetensors",
                "README.md"
            ]
        );
        assert!(saved.private);
        assert!(crate::runs::parse_rfc3339(&saved.pushed).is_some());
        Ok(())
    }
}
