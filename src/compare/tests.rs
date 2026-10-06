//! The compare job and its script against a fake llama.cpp release: the fake
//! `llama-server` answers streamed chat completions.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::*;
use crate::export::{LLAMA_CPP_CUDA, LLAMA_CPP_TAG};
use crate::test_support::{make_executable, tar, text, tools_available};
use crate::train::{METRICS_FILE, Trainer as _};

/// What a test returns.
type TestResult = Result<(), Box<dyn std::error::Error>>;

/// The one fake `llama-server` of these tests. It checks that it serves
/// `$FAKE_LLAMA_MODEL` on 127.0.0.1 (exit 4 otherwise), writes its
/// `LD_LIBRARY_PATH` to `server-env.json` in its directory, serves `/health`
/// (503 forever when `$FAKE_LLAMA_MODE` is `never`) and streamed completions
/// answering `Child says: <question>` after a reasoning block, with usage in
/// the last chunk. A question with `crash` in it kills the server.
const FAKE_SERVER: &str = r#"#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
if args[args.index("-m") + 1] != os.environ.get("FAKE_LLAMA_MODEL"):
    sys.exit(f"fake llama-server: unexpected model in {args}")
if args[args.index("--host") + 1] != "127.0.0.1":
    sys.exit(f"fake llama-server: unexpected host in {args}")
with open("server-env.json", "w") as file:
    json.dump({"LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "")}, file)
mode = os.environ.get("FAKE_LLAMA_MODE", "ok")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(503 if mode == "never" else 200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        question = body["messages"][-1]["content"]
        if "crash" in question:
            os._exit(3)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "hm"}}]},
            {"choices": [{"delta": {"content": "Child says: "}}]},
            {"choices": [{"delta": {"content": question}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
        ]
        for chunk in chunks:
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")


HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"#;

/// A stub `uname` printing `Linux x86_64`.
const UNAME_LINUX_X64: &str = "#!/bin/sh\necho 'Linux x86_64'\n";

/// A stub `nvidia-smi` that fails, as on a machine with no working driver.
const NO_GPU: &str = "#!/bin/sh\nexit 9\n";

/// A stub `nvidia-smi` of a driver supporting CUDA `version`: one GPU named
/// `Fake GPU` for `--query-gpu`, the header line otherwise.
fn gpu(version: &str) -> String {
    format!(
        "#!/bin/sh\ncase \"$*\" in\n*query-gpu*) echo 'Fake GPU' ;;\n\
         *) echo '| NVIDIA-SMI 580.00   Driver Version: 580.00   CUDA Version: {version}     |' ;;\n\
         esac\n"
    )
}

/// Writes the executable `name` with `body` into the directory `dir`.
fn stub(dir: &Path, name: &str, body: &str) -> TestResult {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(name);
    std::fs::write(&path, body)?;
    make_executable(&path)?;
    Ok(())
}

/// The settings of every test job.
const SETTINGS: ChildSettings = ChildSettings {
    max_tokens: 64,
    temperature: 0.0,
    server_start_secs: 10,
    context: 0,
};

/// Eval questions asking `texts`, IDs `q0`, `q1`...
fn questions(texts: &[&str]) -> Vec<EvalQuestion> {
    texts
        .iter()
        .enumerate()
        .map(|(i, text)| EvalQuestion {
            id: format!("q{i}"),
            topic: "t".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: (*text).to_string(),
            }],
            parent: "Parent.".into(),
            parent_input_tokens: 1,
            parent_output_tokens: 1,
        })
        .collect()
}

/// A fake release, a job directory prepared by a `CompareJob`, stub `uname`
/// and `nvidia-smi` (no GPU), and a cache.
struct Fixture {
    /// Holds everything.
    root: tempfile::TempDir,
    /// The job, as prepared.
    job: CompareJob,
    /// The path `llama-server` must get after `-m`.
    model: String,
    /// SHA-256 of every llama.cpp archive.
    bin_sha: String,
    /// SHA-256 of every cudart archive.
    cudart_sha: String,
}

impl Fixture {
    /// Builds the fake release and prepares the job for `texts`, its GGUF
    /// uploaded, or already on the target when `on_target`.
    fn new(texts: &[&str], on_target: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let path = root.path();
        let build = path.join("build");
        let top = format!("llama-{LLAMA_CPP_TAG}");
        stub(&build.join(&top), "llama-server", FAKE_SERVER)?;
        let download = path.join("remote/releases/download").join(LLAMA_CPP_TAG);
        std::fs::create_dir_all(&download)?;
        let bin_tar = download.join("bin.tar.gz");
        tar(&build, &bin_tar, &top)?;
        for asset in ["ubuntu-x64", &format!("ubuntu-cuda-{LLAMA_CPP_CUDA}-x64")] {
            std::fs::copy(
                &bin_tar,
                download.join(format!("llama-{LLAMA_CPP_TAG}-bin-{asset}.tar.gz")),
            )?;
        }
        let cudart = format!("cudart-llama-{LLAMA_CPP_TAG}-bin-ubuntu-cuda-{LLAMA_CPP_CUDA}-x64");
        std::fs::create_dir_all(build.join(&cudart))?;
        std::fs::write(build.join(&cudart).join("libcudart.so.13"), "")?;
        let cudart_tar = download.join(format!("{cudart}.tar.gz"));
        tar(&build, &cudart_tar, &cudart)?;
        let stubs = path.join("stubs");
        stub(&stubs, "uname", UNAME_LINUX_X64)?;
        stub(&stubs, "nvidia-smi", NO_GPU)?;
        let gguf = path.join("child.gguf");
        std::fs::write(&gguf, "gguf")?;
        let (source, model) = if on_target {
            let model = gguf.to_string_lossy().into_owned();
            (ModelSource::OnTarget(model.clone()), model)
        } else {
            (ModelSource::Upload(gguf), MODEL_FILE.to_string())
        };
        let job = CompareJob::new(source, questions(texts), SETTINGS);
        let job_dir = path.join("job");
        job.prepare(&job_dir, &job_dir.to_string_lossy())?;
        Ok(Self {
            bin_sha: crate::exec::sha256_file(&bin_tar)?,
            cudart_sha: crate::exec::sha256_file(&cudart_tar)?,
            root,
            job,
            model,
        })
    }

    /// The job directory.
    fn job_dir(&self) -> PathBuf {
        self.root.path().join("job")
    }

    /// The directory of the stub `uname` and `nvidia-smi`.
    fn stubs(&self) -> PathBuf {
        self.root.path().join("stubs")
    }

    /// Runs the script as the job would, with the fake release's URL and
    /// digests, the stubs first on `PATH`, and the fake server in `mode`.
    fn run(&self, mode: &str) -> std::io::Result<std::process::Output> {
        let path = self.root.path();
        let mut command = Command::new("sh");
        command
            .arg(self.job_dir().join(SCRIPT_FILE))
            .current_dir(path);
        for (name, value) in self.job.env(&self.job_dir().to_string_lossy()) {
            command.env(name, value);
        }
        command
            .env(
                "OVERBRAINER_LLAMA_CPP_URL",
                format!("file://{}", path.join("remote").display()),
            )
            .env("OVERBRAINER_CACHE", path.join("cache"))
            .env("FAKE_LLAMA_MODE", mode)
            .env("FAKE_LLAMA_MODEL", &self.model)
            .env(
                "PATH",
                format!(
                    "{}:{}",
                    self.stubs().display(),
                    std::env::var("PATH").unwrap_or_default()
                ),
            );
        for name in ["UBUNTU_X64", "UBUNTU_CUDA_X64"] {
            command.env(
                format!("OVERBRAINER_LLAMA_CPP_{name}_SHA256"),
                &self.bin_sha,
            );
        }
        command.env("OVERBRAINER_LLAMA_CPP_CUDART_X64_SHA256", &self.cudart_sha);
        command.output()
    }
}

/// An uploaded job's directory holds the script, the client, the questions
/// and the model, which `discard_model` removes.
#[test]
fn a_prepared_job_holds_what_the_target_needs() -> TestResult {
    let dir = tempfile::tempdir()?;
    let gguf = dir.path().join("child.gguf");
    std::fs::write(&gguf, "gguf")?;
    let job_dir = dir.path().join("job");
    CompareJob::new(ModelSource::Upload(gguf), questions(&["Why?"]), SETTINGS)
        .prepare(&job_dir, "/w/job")?;
    for file in [SCRIPT_FILE, CLIENT_FILE, QUESTIONS_FILE, MODEL_FILE] {
        assert!(job_dir.join(file).is_file(), "{file} missing");
    }
    let line = std::fs::read_to_string(job_dir.join(QUESTIONS_FILE))?;
    assert!(
        line.starts_with(r#"{"id":"q0","messages":[{"role":"user","content":"Why?"}]}"#),
        "{line}"
    );
    discard_model(&job_dir)?;
    assert!(!job_dir.join(MODEL_FILE).exists());
    discard_model(&job_dir)?;
    Ok(())
}

/// A job whose GGUF is on the target already uploads no model and names it
/// to the script.
#[test]
fn a_job_with_its_model_on_the_target_uploads_none() -> TestResult {
    let dir = tempfile::tempdir()?;
    let job = CompareJob::new(
        ModelSource::OnTarget("/w/r1/output/gguf/r1-Q4_K_M.gguf".into()),
        questions(&["Why?"]),
        SETTINGS,
    );
    job.prepare(dir.path(), "/w/job")?;
    assert!(dir.path().join(SCRIPT_FILE).is_file());
    assert!(!dir.path().join(MODEL_FILE).exists());
    assert!(job.env("/w/job").contains(&(
        "OVERBRAINER_COMPARE_MODEL".to_string(),
        "/w/r1/output/gguf/r1-Q4_K_M.gguf".to_string()
    )));
    Ok(())
}

/// Every question is answered and measured; progress lines go to the metrics
/// file; the uploaded model is gone from the job directory after, and the
/// release is cached.
#[test]
fn every_question_is_answered_and_measured() -> TestResult {
    if !tools_available() {
        eprintln!("skipped: needs sh, python3 and tar");
        return Ok(());
    }
    let fixture = Fixture::new(&["Why borrow?", "What is a lifetime?"], false)?;
    let output = fixture.run("ok")?;
    assert!(output.status.success(), "{}", text(&output));
    let job_dir = fixture.job_dir();
    let answers = read_answers(&job_dir)?;
    assert_eq!(answers.len(), 2);
    assert_eq!(
        answers[0].answer.as_deref(),
        Some("Child says: Why borrow?")
    );
    assert_eq!(answers[0].output_tokens, Some(3));
    assert!(answers[0].seconds.is_some_and(|seconds| seconds > 0.0));
    assert!(answers[0].first_token_seconds.is_some());
    let hardware = read_hardware(&job_dir).ok_or("no hardware.json")?;
    assert_eq!(hardware.build, "ubuntu-x64");
    assert!(hardware.gpus.is_empty(), "{:?}", hardware.gpus);
    let metrics = std::fs::read_to_string(job_dir.join(METRICS_FILE))?;
    assert!(metrics.contains(r#""step": 2, "total": 2"#), "{metrics}");
    assert!(!job_dir.join(MODEL_FILE).exists(), "the model is removed");
    assert!(job_dir.join(SERVER_LOG).is_file());
    // The release is cached: a second run downloads nothing, then fails on
    // the model it removed.
    std::fs::remove_dir_all(fixture.root.path().join("remote"))?;
    let again = fixture.run("ok")?;
    assert!(!again.status.success(), "no model: {}", text(&again));
    assert!(!text(&again).contains("downloading"), "{}", text(&again));
    assert!(
        text(&again).contains("model.gguf is missing"),
        "{}",
        text(&again)
    );
    Ok(())
}

/// A GGUF already on the target is served where it is, and left there.
#[test]
fn a_model_on_the_target_is_served_where_it_is_and_kept() -> TestResult {
    if !tools_available() {
        eprintln!("skipped: needs sh, python3 and tar");
        return Ok(());
    }
    let fixture = Fixture::new(&["Why?"], true)?;
    let output = fixture.run("ok")?;
    assert!(output.status.success(), "{}", text(&output));
    assert_eq!(read_answers(&fixture.job_dir())?.len(), 1);
    assert!(Path::new(&fixture.model).is_file(), "the model is kept");
    assert!(!fixture.job_dir().join(MODEL_FILE).exists());
    Ok(())
}

/// On a GPU whose driver supports the pinned CUDA, the CUDA build runs with
/// the cudart libraries first on `LD_LIBRARY_PATH`.
#[test]
fn a_gpu_host_runs_the_cuda_build_with_its_runtime_first() -> TestResult {
    if !tools_available() {
        eprintln!("skipped: needs sh, python3 and tar");
        return Ok(());
    }
    let fixture = Fixture::new(&["Why?"], false)?;
    stub(&fixture.stubs(), "nvidia-smi", &gpu("13.0"))?;
    let output = fixture.run("ok")?;
    assert!(output.status.success(), "{}", text(&output));
    let hardware = read_hardware(&fixture.job_dir()).ok_or("no hardware.json")?;
    assert_eq!(hardware.build, format!("ubuntu-cuda-{LLAMA_CPP_CUDA}-x64"));
    assert_eq!(hardware.gpus, ["Fake GPU"]);
    let seen: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        fixture.job_dir().join("server-env.json"),
    )?)?;
    let library_path = seen["LD_LIBRARY_PATH"]
        .as_str()
        .ok_or("no LD_LIBRARY_PATH")?;
    let first = library_path.split(':').next().ok_or("empty")?;
    let cudart = format!("cudart-llama-{LLAMA_CPP_TAG}-bin-ubuntu-cuda-{LLAMA_CPP_CUDA}-x64");
    assert!(first.ends_with(&cudart), "{library_path}");
    assert!(Path::new(first).join("libcudart.so.13").is_file());
    Ok(())
}

/// A server never ready fails the job in time, with the end of its log.
#[test]
fn a_server_never_ready_fails_the_job() -> TestResult {
    if !tools_available() {
        eprintln!("skipped: needs sh, python3 and tar");
        return Ok(());
    }
    let fixture = Fixture::new(&["Why?"], false)?;
    let started = std::time::Instant::now();
    let output = fixture.run("never")?;
    assert!(!output.status.success());
    assert!(
        text(&output).contains("llama-server was not ready after 10 s"),
        "{}",
        text(&output)
    );
    assert!(started.elapsed() < std::time::Duration::from_secs(60));
    assert!(
        !fixture.job_dir().join(MODEL_FILE).exists(),
        "the model is removed"
    );
    Ok(())
}

/// A server that dies while answering fails the job, saying so.
#[test]
fn a_server_that_dies_fails_the_job() -> TestResult {
    if !tools_available() {
        eprintln!("skipped: needs sh, python3 and tar");
        return Ok(());
    }
    let fixture = Fixture::new(&["Fine?", "Please crash now"], false)?;
    let output = fixture.run("ok")?;
    assert!(!output.status.success());
    assert!(
        text(&output).contains("llama-server exited while answering"),
        "{}",
        text(&output)
    );
    Ok(())
}

/// A release whose digest differs is never extracted.
#[test]
fn a_release_with_another_digest_is_never_used() -> TestResult {
    if !tools_available() {
        eprintln!("skipped: needs sh, python3 and tar");
        return Ok(());
    }
    let mut fixture = Fixture::new(&["Why?"], false)?;
    fixture.bin_sha = "0".repeat(64);
    let output = fixture.run("ok")?;
    assert!(!output.status.success());
    assert!(text(&output).contains("not used"), "{}", text(&output));
    let cache = fixture
        .root
        .path()
        .join("cache/llama.cpp")
        .join(LLAMA_CPP_TAG);
    assert!(!cache.join("ubuntu-x64").exists());
    Ok(())
}

/// A llama.cpp build `pick_build` chooses.
#[derive(Debug, PartialEq, Eq)]
struct Build {
    /// The release asset.
    asset: String,
    /// Its digest.
    digest: String,
    /// The cudart archive, empty for a CPU build.
    cudart: String,
    /// Its digest, empty for a CPU build.
    cudart_digest: String,
}

impl Build {
    /// The CPU build `asset`, of digest `digest`.
    fn cpu(asset: &str, digest: &str) -> Self {
        Self {
            asset: asset.to_string(),
            digest: digest.to_string(),
            cudart: String::new(),
            cudart_digest: String::new(),
        }
    }

    /// The CUDA build of `arch` (`x64` or `arm64`), of digest `digest`, with
    /// its cudart archive of digest `cudart_digest`.
    fn cuda(arch: &str, digest: &str, cudart_digest: &str) -> Self {
        Self {
            asset: format!("ubuntu-cuda-{LLAMA_CPP_CUDA}-{arch}"),
            digest: digest.to_string(),
            cudart: format!("cudart-llama-{LLAMA_CPP_TAG}-bin-ubuntu-cuda-{LLAMA_CPP_CUDA}-{arch}"),
            cudart_digest: cudart_digest.to_string(),
        }
    }
}

/// Runs the script's `pick_build` with `uname` printing `platform` and
/// `nvidia-smi` as `smi` (absent when `None`), on a `PATH` holding only
/// those stubs: the build, and what it printed on stderr.
fn pick_build(
    platform: &str,
    smi: Option<&str>,
) -> Result<(Build, String), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let stubs = dir.path().join("stubs");
    stub(&stubs, "uname", &format!("#!/bin/sh\necho '{platform}'\n"))?;
    if let Some(body) = smi {
        stub(&stubs, "nvidia-smi", body)?;
    }
    // The script's functions without its last line, which runs `main`.
    let functions = SCRIPT
        .strip_suffix("main \"$@\"\n")
        .ok_or("compare.sh does not end with main \"$@\"")?;
    let library = dir.path().join("functions.sh");
    std::fs::write(&library, functions)?;
    let output = Command::new("/bin/sh")
        .arg("-c")
        .arg(r#". "$1"; pick_build; printf '%s\n' "$asset" "$digest" "$cudart" "$cudart_digest""#)
        .arg("sh")
        .arg(&library)
        .env_clear()
        .env("PATH", &stubs)
        .envs(crate::export::llama_cpp_env())
        .output()?;
    if !output.status.success() {
        return Err(text(&output).into());
    }
    let stdout = String::from_utf8(output.stdout)?;
    let mut lines = stdout.lines().map(str::to_string);
    let mut next = || lines.next().ok_or("pick_build printed too little");
    let build = Build {
        asset: next()?,
        digest: next()?,
        cudart: next()?,
        cudart_digest: next()?,
    };
    Ok((build, String::from_utf8(output.stderr)?))
}

/// Each platform and GPU case gets its build: CUDA with its runtime on a
/// driver supporting the pinned CUDA, the CPU build otherwise, with a
/// warning when the driver is too old.
#[test]
fn the_build_matches_the_platform_and_the_gpu() -> TestResult {
    use crate::export::{
        CUDART_ARM64_SHA256, CUDART_X64_SHA256, MACOS_ARM64_SHA256, UBUNTU_ARM64_SHA256,
        UBUNTU_CUDA_ARM64_SHA256, UBUNTU_CUDA_X64_SHA256, UBUNTU_X64_SHA256,
    };
    if !crate::test_support::sh_available() {
        eprintln!("skipped: needs sh");
        return Ok(());
    }
    let driver_13 = gpu("13.0");
    let driver_12 = gpu("12.8");
    let cpu_x64 = || Build::cpu("ubuntu-x64", UBUNTU_X64_SHA256);
    let cases = [
        ("Linux x86_64", None, cpu_x64(), false),
        (
            "Linux aarch64",
            None,
            Build::cpu("ubuntu-arm64", UBUNTU_ARM64_SHA256),
            false,
        ),
        ("Linux x86_64", Some(NO_GPU), cpu_x64(), false),
        (
            "Linux x86_64",
            Some(driver_13.as_str()),
            Build::cuda("x64", UBUNTU_CUDA_X64_SHA256, CUDART_X64_SHA256),
            false,
        ),
        (
            "Linux aarch64",
            Some(driver_13.as_str()),
            Build::cuda("arm64", UBUNTU_CUDA_ARM64_SHA256, CUDART_ARM64_SHA256),
            false,
        ),
        ("Linux x86_64", Some(driver_12.as_str()), cpu_x64(), true),
        (
            "Darwin arm64",
            None,
            Build::cpu("macos-arm64", MACOS_ARM64_SHA256),
            false,
        ),
    ];
    for (platform, smi, expected, warns) in cases {
        let (build, stderr) = pick_build(platform, smi)?;
        let case = format!("{platform}, nvidia-smi {smi:?}");
        assert_eq!(build, expected, "{case}");
        assert_eq!(stderr.contains("warning"), warns, "{case}: {stderr}");
        if warns {
            assert_eq!(stderr.lines().count(), 1, "{stderr}");
            assert!(stderr.contains("CUDA 12.8"), "{stderr}");
        }
    }
    let other = pick_build("Linux riscv64", None);
    assert!(
        other.is_err_and(|error| error
            .to_string()
            .contains("no prebuilt llama-server for Linux riscv64")),
        "an unknown platform fails"
    );
    Ok(())
}

/// The trainer side: its command, environment, stage and artifacts.
#[test]
fn the_job_runs_compare_sh_with_its_settings() {
    let job = CompareJob::new(
        ModelSource::Upload(PathBuf::from("/m.gguf")),
        Vec::new(),
        ChildSettings {
            max_tokens: 512,
            temperature: 0.2,
            server_start_secs: 120,
            context: 4096,
        },
    );
    let env = job.env("/w/r1/compares/c1");
    let get = |name: &str| {
        env.iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    };
    assert_eq!(
        get(crate::train::METRICS_ENV),
        Some("/w/r1/compares/c1/metrics.jsonl")
    );
    assert_eq!(get("OVERBRAINER_COMPARE_MODEL"), Some(MODEL_FILE));
    assert_eq!(get("OVERBRAINER_COMPARE_MAX_TOKENS"), Some("512"));
    assert_eq!(get("OVERBRAINER_COMPARE_TEMPERATURE"), Some("0.2"));
    assert_eq!(get("OVERBRAINER_COMPARE_START_SECS"), Some("120"));
    assert_eq!(get("OVERBRAINER_COMPARE_CTX"), Some("4096"));
    assert_eq!(get("OVERBRAINER_LLAMA_CPP_CUDA"), Some(LLAMA_CPP_CUDA));
    assert_eq!(job.stages(), [crate::train::JobStage::Compare]);
    assert_eq!(
        job.commands(),
        [["python3", "-c", crate::export::TRAMPOLINE, SCRIPT_FILE].map(str::to_string)]
    );
    let artifacts = job.artifacts();
    assert_eq!(artifacts.required.as_deref(), Some(ANSWERS_FILE));
    for entry in [ANSWERS_FILE, SERVER_LOG, HARDWARE_FILE, METRICS_FILE] {
        assert!(artifacts.entries.contains(&entry.to_string()), "{entry}");
    }
    assert!(job.caches_tools());
    assert!(!job.metrics_required());
}

/// `[compare]` and the run's context give the child's settings.
#[test]
fn the_settings_come_from_the_config() {
    let settings = ChildSettings::from_config(&crate::config::Compare::default(), 2048);
    assert_eq!(
        settings,
        ChildSettings {
            max_tokens: 4096,
            temperature: 0.0,
            server_start_secs: 300,
            context: 2048,
        }
    );
}
