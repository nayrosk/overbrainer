//! The compare job and its script against a fake llama.cpp release: the fake
//! `llama-server` answers streamed chat completions.

use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::*;
use crate::export::{LLAMA_CPP_CUDA, LLAMA_CPP_ROCM, LLAMA_CPP_TAG};
use crate::test_support::{make_executable, tar, text, tools_available};
use crate::train::{METRICS_FILE, Trainer as _};

/// What a test returns.
type TestResult = Result<(), Box<dyn std::error::Error>>;

/// The one fake `llama-server` of these tests. It checks that it serves
/// `$FAKE_LLAMA_MODEL` on 127.0.0.1 (exit 4 otherwise), writes its
/// `LD_LIBRARY_PATH` and the path of its own file (`SERVER`, which names the
/// build it came from) to `server-env.json` in its directory, serves `/health`
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
    json.dump({"LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
               "SERVER": os.path.realpath(__file__)}, file)
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

/// A stub that prints nothing and fails: a missing AMD tool.
const FAILS: &str = "#!/bin/sh\nexit 9\n";

/// A stub `ldconfig -p` listing the three libraries of the `ROCm` 7 runtime.
const LDCONFIG_ROCM: &str = "#!/bin/sh\n\
    echo '\tlibamdhip64.so.7 (libc6,x86-64) => /usr/lib/libamdhip64.so.7'\n\
    echo '\tlibrocblas.so.5 (libc6,x86-64) => /usr/lib/librocblas.so.5'\n\
    echo '\tlibhipblas.so.3 (libc6,x86-64) => /usr/lib/libhipblas.so.3'\n";

/// A stub `ldconfig -p` listing the `ROCm` 7 runtime but `librocblas.so.5`.
const LDCONFIG_NO_ROCBLAS: &str = "#!/bin/sh\n\
    echo '\tlibamdhip64.so.7 (libc6,x86-64) => /usr/lib/libamdhip64.so.7'\n\
    echo '\tlibhipblas.so.3 (libc6,x86-64) => /usr/lib/libhipblas.so.3'\n";

/// A stub `ldconfig -p` of a machine without the `ROCm` runtime.
const LDCONFIG_PLAIN: &str = "#!/bin/sh\necho '\tlibc.so.6 (libc6,x86-64) => /usr/lib/libc.so.6'\n";

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

/// A fake release, a job directory prepared by a `CompareJob`, stub `uname`,
/// `nvidia-smi` (no GPU), `ldconfig` (no `ROCm`) and `getconf` (glibc 2.39,
/// whatever this machine has), no `/dev/kfd`, and a cache.
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
        for asset in [
            "ubuntu-x64",
            &format!("ubuntu-cuda-{LLAMA_CPP_CUDA}-x64"),
            &format!("ubuntu-rocm-{LLAMA_CPP_ROCM}-x64"),
        ] {
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
        stub(&stubs, "ldconfig", LDCONFIG_PLAIN)?;
        Glibc::Getconf("2.39").stub_into(&stubs)?;
        let gguf = path.join("child.gguf");
        std::fs::write(&gguf, "gguf")?;
        let (source, model) = if on_target {
            let model = gguf.to_string_lossy().into_owned();
            (ModelSource::OnTarget(model.clone()), model)
        } else {
            (ModelSource::Upload(gguf), MODEL_FILE.to_string())
        };
        let job = CompareJob::new(source, questions(texts), SETTINGS)?;
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

    /// The file standing for `/dev/kfd`: absent until `make_amd` creates it.
    fn kfd(&self) -> PathBuf {
        self.root.path().join("kfd")
    }

    /// Makes the host an AMD one with the `ROCm` 7 runtime: `/dev/kfd` exists
    /// and `ldconfig` lists the libraries. `amd_tools` are the stub AMD
    /// tools to add, each as a name and its body.
    fn make_amd(&self, amd_tools: StubTools<'_>) -> TestResult {
        std::fs::write(self.kfd(), "")?;
        stub(&self.stubs(), "ldconfig", LDCONFIG_ROCM)?;
        for (name, body) in amd_tools {
            stub(&self.stubs(), name, body)?;
        }
        Ok(())
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
            .env("OVERBRAINER_COMPARE_KFD", self.kfd())
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
        for name in ["UBUNTU_X64", "UBUNTU_CUDA_X64", "UBUNTU_ROCM_X64"] {
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
    let job = CompareJob::new(ModelSource::Upload(gguf), questions(&["Why?"]), SETTINGS)?;
    job.prepare(&job_dir, "/w/job")?;
    assert!(job.env("/w/job").contains(&(
        "OVERBRAINER_COMPARE_DISCARD_MODEL".to_string(),
        "1".to_string()
    )));
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
    )?;
    job.prepare(dir.path(), "/w/job")?;
    assert!(
        job.env("/w/job")
            .iter()
            .all(|(name, _)| name != "OVERBRAINER_COMPARE_DISCARD_MODEL"),
        "a model on the target is never discarded"
    );
    assert!(dir.path().join(SCRIPT_FILE).is_file());
    assert!(!dir.path().join(MODEL_FILE).exists());
    assert!(job.env("/w/job").contains(&(
        "OVERBRAINER_COMPARE_MODEL".to_string(),
        "/w/r1/output/gguf/r1-Q4_K_M.gguf".to_string()
    )));
    Ok(())
}

/// Checks that `answer` is the fake server's answer to `question`, with
/// its tokens, finish reason and timings.
fn assert_measured(answer: &ChildAnswer, question: &str) {
    assert_eq!(
        answer.answer.as_deref(),
        Some(format!("Child says: {question}").as_str())
    );
    assert_eq!(answer.output_tokens, Some(3));
    assert_eq!(answer.input_tokens, Some(7));
    assert_eq!(answer.finish.as_deref(), Some("stop"));
    assert!(answer.tokens_per_second.is_some_and(|rate| rate > 0.0));
    assert!(answer.seconds.is_some_and(|seconds| seconds > 0.0));
    assert!(answer.first_token_seconds.is_some());
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
    assert_measured(&answers[0], "Why borrow?");
    assert_measured(&answers[1], "What is a lifetime?");
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

/// On an NVIDIA host whose driver supports CUDA 12 only, the CPU build
/// serves the child: hardware.json lists no GPU and the report says the
/// child ran on a CPU.
#[test]
fn an_old_nvidia_driver_falls_back_to_the_cpu_and_says_so() -> TestResult {
    if !tools_available() {
        eprintln!("skipped: needs sh, python3 and tar");
        return Ok(());
    }
    let fixture = Fixture::new(&["Why?"], false)?;
    stub(&fixture.stubs(), "nvidia-smi", &gpu("12.8"))?;
    let output = fixture.run("ok")?;
    assert!(output.status.success(), "{}", text(&output));
    let hardware = read_hardware(&fixture.job_dir()).ok_or("no hardware.json")?;
    assert_eq!(hardware.build, "ubuntu-x64");
    assert!(hardware.gpus.is_empty(), "{:?}", hardware.gpus);
    assert!(!hardware.has_gpu());
    let (setup, answers, verdicts) = (
        super::fixtures::setup(),
        super::fixtures::answers(),
        super::fixtures::verdicts(),
    );
    let role = super::fixtures::judge()?;
    let report = build_report(&Parts {
        setup: &setup,
        answers: &answers,
        verdicts: &verdicts,
        hardware: Some(hardware),
        judge: JudgeInfo {
            role: &role,
            is_parent: false,
            verdicts_file: "verdicts-0123456789abcdef.jsonl",
        },
        prices: Prices::default(),
        child_price_from_pod: false,
    });
    let markdown = render_markdown(&report);
    assert!(markdown.contains("- The child ran on a CPU."), "{markdown}");
    assert!(markdown.contains(", CPU (ubuntu-x64)"), "{markdown}");
    Ok(())
}

/// On an NVIDIA host whose driver supports CUDA 13 but whose glibc is
/// Ubuntu 22.04's, the CPU build serves the child, with a warning in the
/// job's output: hardware.json lists no GPU.
#[test]
fn an_old_glibc_serves_the_child_on_the_cpu() -> TestResult {
    if !tools_available() {
        eprintln!("skipped: needs sh, python3 and tar");
        return Ok(());
    }
    let fixture = Fixture::new(&["Why?"], false)?;
    stub(&fixture.stubs(), "nvidia-smi", &gpu("13.0"))?;
    Glibc::Getconf("2.35").stub_into(&fixture.stubs())?;
    let output = fixture.run("ok")?;
    assert!(output.status.success(), "{}", text(&output));
    let stderr = String::from_utf8(output.stderr)?;
    assert!(
        stderr.contains("compare: warning: glibc 2.35 is older than the glibc 2.38"),
        "{stderr}"
    );
    let hardware = read_hardware(&fixture.job_dir()).ok_or("no hardware.json")?;
    assert_eq!(hardware.build, "ubuntu-x64");
    assert!(!hardware.has_gpu(), "{:?}", hardware.gpus);
    Ok(())
}

/// Runs the client's `write_hardware` for `build` in a fresh directory,
/// with `nvidia-smi`, `rocm-smi`, `amd-smi`, `lspci` and `sysctl` stubbed
/// (failing unless `tools` gives a body, each a name and its body), and
/// returns hardware.json.
fn hardware_of(build: &str, tools: StubTools<'_>) -> Result<Hardware, Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let stubs = dir.path().join("stubs");
    for name in ["nvidia-smi", "rocm-smi", "amd-smi", "lspci", "sysctl"] {
        stub(&stubs, name, FAILS)?;
    }
    for (name, body) in tools {
        stub(&stubs, name, body)?;
    }
    std::fs::write(dir.path().join(CLIENT_FILE), super::job::CLIENT)?;
    let output = Command::new("python3")
        .args([
            "-c",
            "import sys, compare_client; compare_client.write_hardware(sys.argv[1])",
            build,
        ])
        .current_dir(dir.path())
        .env(
            "PATH",
            format!(
                "{}:{}",
                stubs.display(),
                std::env::var("PATH").unwrap_or_default()
            ),
        )
        .output()?;
    if !output.status.success() {
        return Err(text(&output).into());
    }
    read_hardware(dir.path()).ok_or_else(|| "no hardware.json".into())
}

/// The GPUs of hardware.json follow the build that served the child: the
/// NVIDIA ones for CUDA, the AMD ones for `ROCm` (an integrated "AMD Radeon
/// Graphics" left out beside a discrete GPU), the Apple chip for Metal, and
/// none for a CPU build, even with an NVIDIA GPU in the machine.
#[test]
fn hardware_lists_the_gpus_of_the_build_that_served() -> TestResult {
    if !tools_available() {
        eprintln!("skipped: needs sh, python3 and tar");
        return Ok(());
    }
    let nvidia = gpu("13.0");
    let rocm_smi = "#!/bin/sh\n\
        echo 'GPU[0]\t\t: Card Series: \t\tAMD Radeon RX 7800 XT'\n\
        echo 'GPU[1]\t\t: Card Series: \t\tAMD Radeon Graphics'\n";
    let igpu_only = "#!/bin/sh\necho 'GPU[0]\t\t: Card Series: \t\tAMD Radeon Graphics'\n";
    let sysctl = "#!/bin/sh\necho 'Apple M2 Pro'\n";
    let cuda = format!("ubuntu-cuda-{LLAMA_CPP_CUDA}-x64");
    let rocm = format!("ubuntu-rocm-{LLAMA_CPP_ROCM}-x64");
    let cases: [(&str, StubTools<'_>, Vec<&str>); 6] = [
        (&cuda, &[("nvidia-smi", &nvidia)], vec!["Fake GPU"]),
        (
            &rocm,
            &[("rocm-smi", rocm_smi)],
            vec!["AMD Radeon RX 7800 XT"],
        ),
        (
            &rocm,
            &[("rocm-smi", igpu_only)],
            vec!["AMD Radeon Graphics"],
        ),
        (
            "macos-arm64",
            &[("sysctl", sysctl)],
            vec!["Apple M2 Pro (Metal)"],
        ),
        ("ubuntu-x64", &[("nvidia-smi", &nvidia)], vec![]),
        ("ubuntu-arm64", &[("rocm-smi", rocm_smi)], vec![]),
    ];
    for (build, tools, expected) in cases {
        let hardware = hardware_of(build, tools)?;
        assert_eq!(hardware.build, build);
        assert_eq!(hardware.gpus, expected, "{build}");
        assert_eq!(hardware.has_gpu(), !expected.is_empty(), "{build}");
    }
    Ok(())
}

/// Stub tools of a host, each a name and its body.
type StubTools<'a> = &'a [(&'a str, &'a str)];

/// On an AMD host with the `ROCm` 7 runtime, the `ROCm` build serves the model,
/// and hardware.json names the GPU the AMD tool lists, whichever tool answers:
/// `rocm-smi`, `amd-smi`, `lspci`, or none (a generic name).
#[test]
fn an_amd_host_runs_the_rocm_build_and_names_its_gpu() -> TestResult {
    if !tools_available() {
        eprintln!("skipped: needs sh, python3 and tar");
        return Ok(());
    }
    let rocm_smi = "#!/bin/sh\necho 'GPU[0]\t\t: Card Series: \t\tFake Radeon'\n\
                    echo 'GPU[1]\t\t: Card Series: \t\tFake Radeon'\n";
    let amd_smi = "#!/bin/sh\necho '        MARKET_NAME: Fake Instinct'\n";
    let lspci = "#!/bin/sh\n\
        echo '03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Fake Navi'\n\
        echo '00:02.0 VGA compatible controller: Intel Corporation Fake Graphics'\n";
    let cases: [(StubTools<'_>, Vec<&str>); 4] = [
        (
            &[("rocm-smi", rocm_smi), ("amd-smi", FAILS), ("lspci", FAILS)],
            vec!["Fake Radeon", "Fake Radeon"],
        ),
        (
            &[("rocm-smi", FAILS), ("amd-smi", amd_smi), ("lspci", FAILS)],
            vec!["Fake Instinct"],
        ),
        (
            &[("rocm-smi", FAILS), ("amd-smi", FAILS), ("lspci", lspci)],
            vec!["Fake Navi"],
        ),
        (
            &[("rocm-smi", FAILS), ("amd-smi", FAILS), ("lspci", FAILS)],
            vec!["AMD GPU"],
        ),
    ];
    for (tools, expected) in cases {
        let fixture = Fixture::new(&["Why?"], false)?;
        fixture.make_amd(tools)?;
        let output = fixture.run("ok")?;
        assert!(output.status.success(), "{}", text(&output));
        let hardware = read_hardware(&fixture.job_dir()).ok_or("no hardware.json")?;
        assert_eq!(hardware.build, format!("ubuntu-rocm-{LLAMA_CPP_ROCM}-x64"));
        assert_eq!(hardware.gpus, expected);
        assert!(hardware.has_gpu());
        let seen: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
            fixture.job_dir().join("server-env.json"),
        )?)?;
        let server = seen["SERVER"].as_str().ok_or("no SERVER")?;
        assert!(
            server.contains(&format!("ubuntu-rocm-{LLAMA_CPP_ROCM}-x64")),
            "{server}"
        );
    }
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
    let said = text(&output);
    assert!(said.contains("compare: file://"), "{said}");
    assert!(
        said.contains(&format!("expected {}: not used", "0".repeat(64))),
        "{said}"
    );
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

    /// The `ROCm` build of Linux `x86_64`, of digest `digest`: it has no
    /// runtime archive, the system provides the libraries.
    fn rocm(digest: &str) -> Self {
        Self::cpu(&format!("ubuntu-rocm-{LLAMA_CPP_ROCM}-x64"), digest)
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

/// What `pick_build` finds of an AMD GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Amd {
    /// No `/dev/kfd`.
    Absent,
    /// `/dev/kfd`, usable, and the `ROCm` 7 runtime installed.
    WithRuntime,
    /// `/dev/kfd`, usable, and no `ROCm` runtime.
    WithoutRuntime,
    /// `/dev/kfd`, usable, and the `ROCm` runtime but `librocblas.so.5`.
    WithoutRocblas,
    /// `/dev/kfd`, the runtime, and no write access to the device.
    Unwritable,
}

/// How the host says which glibc it has, to `pick_build`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Glibc<'a> {
    /// It does not: neither `getconf` nor `ldd`.
    Unknown,
    /// `getconf GNU_LIBC_VERSION` prints `glibc <version>`.
    Getconf(&'a str),
    /// Only `ldd --version` says it, as Ubuntu's does, on its first line.
    Ldd(&'a str),
}

impl Glibc<'_> {
    /// Writes the stub `getconf` or `ldd` saying the version into `stubs`.
    fn stub_into(self, stubs: &Path) -> TestResult {
        match self {
            Self::Unknown => Ok(()),
            Self::Getconf(version) => stub(
                stubs,
                "getconf",
                &format!(
                    "#!/bin/sh\n[ \"$1\" = GNU_LIBC_VERSION ] || exit 1\necho 'glibc {version}'\n"
                ),
            ),
            Self::Ldd(version) => stub(
                stubs,
                "ldd",
                &format!(
                    "#!/bin/sh\n[ \"$1\" = --version ] || exit 1\n\
                     echo 'ldd (Ubuntu GLIBC {version}-0ubuntu3.8) {version}'\n\
                     echo 'Copyright (C) 2022 Free Software Foundation, Inc.'\n"
                ),
            ),
        }
    }
}

/// Runs the script's `pick_build` with `uname` printing `platform`,
/// `nvidia-smi` as `smi` (absent when `None`), the AMD host `amd` and the
/// glibc `glibc`, on a `PATH` holding only those stubs: the build, and what
/// it printed on stderr.
fn pick_build(
    platform: &str,
    smi: Option<&str>,
    amd: Amd,
    glibc: Glibc<'_>,
) -> Result<(Build, String), Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let stubs = dir.path().join("stubs");
    stub(&stubs, "uname", &format!("#!/bin/sh\necho '{platform}'\n"))?;
    if let Some(body) = smi {
        stub(&stubs, "nvidia-smi", body)?;
    }
    glibc.stub_into(&stubs)?;
    let libs = match amd {
        Amd::WithoutRuntime => LDCONFIG_PLAIN,
        Amd::WithoutRocblas => LDCONFIG_NO_ROCBLAS,
        Amd::Absent | Amd::WithRuntime | Amd::Unwritable => LDCONFIG_ROCM,
    };
    stub(&stubs, "ldconfig", libs)?;
    let kfd = dir.path().join("kfd");
    if amd != Amd::Absent {
        std::fs::write(&kfd, "")?;
    }
    if amd == Amd::Unwritable {
        let mut permissions = std::fs::metadata(&kfd)?.permissions();
        permissions.set_mode(0o444);
        std::fs::set_permissions(&kfd, permissions)?;
    }
    // Sourced with OVERBRAINER_COMPARE_SOURCED set, the script defines its
    // functions and does not run `main`.
    let library = dir.path().join("functions.sh");
    std::fs::write(&library, SCRIPT)?;
    let output = Command::new("/bin/sh")
        .arg("-c")
        .arg(r#". "$1"; pick_build; printf '%s\n' "$asset" "$digest" "$cudart" "$cudart_digest""#)
        .arg("sh")
        .arg(&library)
        .env_clear()
        .env("PATH", &stubs)
        .env("OVERBRAINER_COMPARE_SOURCED", "1")
        .env("OVERBRAINER_COMPARE_KFD", &kfd)
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

/// One `pick_build` case: the platform, `nvidia-smi`, the AMD host, the
/// glibc, and the build and the one warning (a part of it) expected.
struct BuildCase<'a> {
    /// What `uname` prints.
    platform: &'a str,
    /// The stub `nvidia-smi`, absent when `None`.
    smi: Option<&'a str>,
    /// The AMD host.
    amd: Amd,
    /// The host's glibc.
    glibc: Glibc<'a>,
    /// The build `pick_build` must choose.
    expected: Build,
    /// A part of the only warning it must print, none when it prints none.
    warning: Option<&'a str>,
}

impl BuildCase<'_> {
    /// Runs `pick_build` and checks the build and the warning; an unwritable
    /// device case is skipped for a user who can write anything.
    fn check(&self) -> TestResult {
        if self.amd == Amd::Unwritable && !kfd_is_unwritable()? {
            eprintln!("skipped the unwritable case: the user can write anything");
            return Ok(());
        }
        let (build, stderr) = pick_build(self.platform, self.smi, self.amd, self.glibc)?;
        let case = format!(
            "{}, nvidia-smi {:?}, {:?}, {:?}",
            self.platform, self.smi, self.amd, self.glibc
        );
        assert_eq!(build, self.expected, "{case}");
        assert_eq!(
            stderr.contains("warning"),
            self.warning.is_some(),
            "{case}: {stderr}"
        );
        if let Some(part) = self.warning {
            assert_eq!(stderr.lines().count(), 1, "{case}: {stderr}");
            assert!(stderr.contains(part), "{case}: {stderr}");
        }
        Ok(())
    }
}

/// Each platform and NVIDIA case gets its build: CUDA with its runtime on a
/// driver supporting the pinned CUDA, the CPU build otherwise, with a warning
/// when the driver is too old.
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
    let case = |platform, smi, expected, warning| BuildCase {
        platform,
        smi,
        amd: Amd::Absent,
        glibc: Glibc::Unknown,
        expected,
        warning,
    };
    let cpu_x64 = || Build::cpu("ubuntu-x64", UBUNTU_X64_SHA256);
    let cuda_x64 = || Build::cuda("x64", UBUNTU_CUDA_X64_SHA256, CUDART_X64_SHA256);
    let cases = [
        case("Linux x86_64", None, cpu_x64(), None),
        case(
            "Linux aarch64",
            None,
            Build::cpu("ubuntu-arm64", UBUNTU_ARM64_SHA256),
            None,
        ),
        case("Linux x86_64", Some(NO_GPU), cpu_x64(), None),
        case("Linux x86_64", Some(driver_13.as_str()), cuda_x64(), None),
        case(
            "Linux aarch64",
            Some(driver_13.as_str()),
            Build::cuda("arm64", UBUNTU_CUDA_ARM64_SHA256, CUDART_ARM64_SHA256),
            None,
        ),
        case(
            "Linux x86_64",
            Some(driver_12.as_str()),
            cpu_x64(),
            Some("CUDA 12.8"),
        ),
        case(
            "Darwin arm64",
            None,
            Build::cpu("macos-arm64", MACOS_ARM64_SHA256),
            None,
        ),
    ];
    for case in cases {
        case.check()?;
    }
    let other = pick_build("Linux riscv64", None, Amd::Absent, Glibc::Unknown);
    assert!(
        other.is_err_and(|error| error
            .to_string()
            .contains("no prebuilt llama-server for Linux riscv64")),
        "an unknown platform fails"
    );
    Ok(())
}

/// An AMD host with the `ROCm` 7 runtime gets the `ROCm` build; without the
/// runtime, without write access to the device, or on arm64 it gets the CPU
/// build and one warning saying why; NVIDIA comes first.
#[test]
fn an_amd_host_gets_the_rocm_build_or_the_reason_it_does_not() -> TestResult {
    use crate::export::{
        CUDART_X64_SHA256, UBUNTU_ARM64_SHA256, UBUNTU_CUDA_X64_SHA256, UBUNTU_ROCM_X64_SHA256,
        UBUNTU_X64_SHA256,
    };
    if !crate::test_support::sh_available() {
        eprintln!("skipped: needs sh");
        return Ok(());
    }
    let driver_13 = gpu("13.0");
    let case = |platform, smi, amd, expected, warning| BuildCase {
        platform,
        smi,
        amd,
        glibc: Glibc::Unknown,
        expected,
        warning,
    };
    let cpu_x64 = || Build::cpu("ubuntu-x64", UBUNTU_X64_SHA256);
    let cases = [
        case(
            "Linux x86_64",
            Some(NO_GPU),
            Amd::WithRuntime,
            Build::rocm(UBUNTU_ROCM_X64_SHA256),
            None,
        ),
        case(
            "Linux x86_64",
            None,
            Amd::WithoutRuntime,
            cpu_x64(),
            Some("apt install libamdhip64-7 librocblas5 libhipblas3"),
        ),
        case(
            "Linux x86_64",
            None,
            Amd::WithoutRocblas,
            cpu_x64(),
            Some(
                "missing (librocblas.so.5); using the CPU build. On Ubuntu: apt install librocblas5",
            ),
        ),
        case(
            "Linux x86_64",
            None,
            Amd::Unwritable,
            cpu_x64(),
            Some("render"),
        ),
        case(
            "Linux aarch64",
            None,
            Amd::WithRuntime,
            Build::cpu("ubuntu-arm64", UBUNTU_ARM64_SHA256),
            Some("no ROCm build for arm64"),
        ),
        case(
            "Linux x86_64",
            Some(driver_13.as_str()),
            Amd::WithRuntime,
            Build::cuda("x64", UBUNTU_CUDA_X64_SHA256, CUDART_X64_SHA256),
            None,
        ),
    ];
    for case in cases {
        case.check()?;
    }
    let (_, stderr) = pick_build("Linux x86_64", None, Amd::WithoutRuntime, Glibc::Unknown)?;
    assert!(
        stderr.contains("missing (libamdhip64.so.7, librocblas.so.5, libhipblas.so.3)"),
        "{stderr}"
    );
    let (_, stderr) = pick_build("Linux x86_64", None, Amd::WithoutRocblas, Glibc::Unknown)?;
    for present in ["libamdhip64", "libhipblas"] {
        assert!(
            !stderr.contains(present),
            "only the missing library: {stderr}"
        );
    }
    Ok(())
}

/// A GPU build needs a newer glibc than Ubuntu 22.04's 2.35: on an older
/// one, said by `getconf` or by `ldd`, the CPU build runs with one warning
/// naming both versions; on a recent enough one, or one nothing says, the GPU
/// build runs as before.
#[test]
fn an_old_glibc_gets_the_cpu_build_and_says_so() -> TestResult {
    use crate::export::{
        CUDART_X64_SHA256, LLAMA_CPP_GPU_GLIBC, UBUNTU_CUDA_X64_SHA256, UBUNTU_ROCM_X64_SHA256,
        UBUNTU_X64_SHA256,
    };
    if !crate::test_support::sh_available() {
        eprintln!("skipped: needs sh");
        return Ok(());
    }
    let driver_13 = gpu("13.0");
    let nvidia = Some(driver_13.as_str());
    let case = |smi, amd, glibc, expected, warning| BuildCase {
        platform: "Linux x86_64",
        smi,
        amd,
        glibc,
        expected,
        warning,
    };
    let cpu_x64 = || Build::cpu("ubuntu-x64", UBUNTU_X64_SHA256);
    let cuda_x64 = || Build::cuda("x64", UBUNTU_CUDA_X64_SHA256, CUDART_X64_SHA256);
    let too_old = format!(
        "glibc 2.35 is older than the glibc {LLAMA_CPP_GPU_GLIBC} the ubuntu-cuda-{LLAMA_CPP_CUDA}-x64 build"
    );
    let rocm_too_old = format!(
        "glibc 2.35 is older than the glibc {LLAMA_CPP_GPU_GLIBC} the ubuntu-rocm-{LLAMA_CPP_ROCM}-x64 build"
    );
    let cases = [
        case(
            nvidia,
            Amd::Absent,
            Glibc::Getconf("2.35"),
            cpu_x64(),
            Some(too_old.as_str()),
        ),
        case(
            nvidia,
            Amd::Absent,
            Glibc::Ldd("2.35"),
            cpu_x64(),
            Some(too_old.as_str()),
        ),
        case(
            None,
            Amd::WithRuntime,
            Glibc::Getconf("2.35"),
            cpu_x64(),
            Some(rocm_too_old.as_str()),
        ),
        case(
            nvidia,
            Amd::Absent,
            Glibc::Getconf("2.39"),
            cuda_x64(),
            None,
        ),
        case(nvidia, Amd::Absent, Glibc::Ldd("2.39"), cuda_x64(), None),
        case(
            nvidia,
            Amd::Absent,
            Glibc::Getconf(LLAMA_CPP_GPU_GLIBC),
            cuda_x64(),
            None,
        ),
        case(nvidia, Amd::Absent, Glibc::Getconf("3.0"), cuda_x64(), None),
        case(
            None,
            Amd::WithRuntime,
            Glibc::Ldd("2.41"),
            Build::rocm(UBUNTU_ROCM_X64_SHA256),
            None,
        ),
    ];
    for case in cases {
        case.check()?;
    }
    let (_, stderr) = pick_build("Linux x86_64", nvidia, Amd::Absent, Glibc::Getconf("2.35"))?;
    assert!(
        stderr.contains("using the CPU build ubuntu-x64"),
        "{stderr}"
    );
    Ok(())
}

/// Whether a read-only file is closed to writing for the current user: false
/// for root, which writes anything.
fn kfd_is_unwritable() -> Result<bool, Box<dyn std::error::Error>> {
    let dir = tempfile::tempdir()?;
    let file = dir.path().join("probe");
    std::fs::write(&file, "")?;
    let mut permissions = std::fs::metadata(&file)?.permissions();
    permissions.set_mode(0o444);
    std::fs::set_permissions(&file, permissions)?;
    Ok(std::fs::OpenOptions::new().write(true).open(&file).is_err())
}

/// The trainer side: its command, environment, stage and artifacts.
#[test]
fn the_job_runs_compare_sh_with_its_settings() -> TestResult {
    let job = CompareJob::new(
        ModelSource::Upload(PathBuf::from("/m.gguf")),
        Vec::new(),
        ChildSettings {
            max_tokens: 512,
            temperature: 0.2,
            server_start_secs: 120,
            context: 4096,
        },
    )?;
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
    Ok(())
}

/// A compare in a container mounting the run directory names its script by
/// its path there, as an export in place does.
#[test]
fn a_mounted_compare_names_its_script_by_its_path() -> TestResult {
    let job = CompareJob::new(
        ModelSource::OnTarget("/workspace/run/output/gguf/r1-Q4_K_M.gguf".into()),
        Vec::new(),
        ChildSettings {
            max_tokens: 512,
            temperature: 0.2,
            server_start_secs: 120,
            context: 0,
        },
    )?
    .with_script("/workspace/run/compares/c1/compare.sh".into());
    assert_eq!(
        job.commands()
            .first()
            .and_then(|command| command.last())
            .map(String::as_str),
        Some("/workspace/run/compares/c1/compare.sh")
    );
    Ok(())
}

/// A GGUF on the target is named by its absolute path: a relative one is
/// refused, as the job could not tell where it is.
#[test]
fn a_relative_model_path_on_the_target_is_refused() {
    let job = CompareJob::new(
        ModelSource::OnTarget("output/gguf/r1-Q4_K_M.gguf".into()),
        Vec::new(),
        SETTINGS,
    );
    assert!(
        matches!(job, Err(CompareError::RelativeModel(ref path)) if path == "output/gguf/r1-Q4_K_M.gguf"),
        "{job:?}"
    );
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
