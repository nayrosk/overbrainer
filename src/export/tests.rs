use std::path::Path;
use std::process::Command;

use super::*;

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[test]
fn an_in_job_export_reads_the_run_s_output_and_config() {
    let job = ExportJob::in_job("r1", "Q4_K_M");
    assert_eq!(job.file_name(), "r1-Q4_K_M.gguf");
    let env = job.env("/w/r1");
    let get = |name: &str| {
        env.iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    };
    assert_eq!(get(METRICS_ENV), Some("/w/r1/metrics.jsonl"));
    assert_eq!(get("OVERBRAINER_EXPORT_NAME"), Some("r1"));
    assert_eq!(get("OVERBRAINER_EXPORT_QUANTIZE"), Some("Q4_K_M"));
    assert_eq!(get("OVERBRAINER_EXPORT_MODEL"), Some("output"));
    assert_eq!(get("OVERBRAINER_EXPORT_CONFIG"), Some("axolotl.yaml"));
    assert_eq!(get("OVERBRAINER_LLAMA_CPP"), Some(LLAMA_CPP_TAG));
    assert_eq!(
        job.command(),
        ["python3", "-c", TRAMPOLINE, "export.sh"].map(str::to_string)
    );
    assert!(job.caches_tools() && !job.metrics_required());
    assert_eq!(job.stages(), [JobStage::Export]);
    assert_eq!(job.stop_marker(), None);
    let artifacts = job.artifacts();
    assert_eq!(artifacts.entries, ["output/gguf", "metrics.jsonl"]);
    assert_eq!(artifacts.required.as_deref(), Some("output/gguf"));
}

#[test]
fn an_in_place_export_reads_two_directories_up() {
    let job = ExportJob::in_place(
        "r1",
        "Q8_0",
        "output/checkpoint-40",
        "/workspace/run/exports/e1/export.sh".into(),
    );
    let env = job.script_env();
    assert!(env.contains(&(
        "OVERBRAINER_EXPORT_MODEL".into(),
        "../../output/checkpoint-40".into()
    )));
    assert!(env.contains(&(
        "OVERBRAINER_EXPORT_CONFIG".into(),
        "../../axolotl.yaml".into()
    )));
    assert_eq!(
        job.command().last().map(String::as_str),
        Some("/workspace/run/exports/e1/export.sh")
    );
}

#[test]
fn a_staged_export_links_the_model_but_not_the_checkpoints() -> TestResult {
    let run = tempfile::tempdir()?;
    let output = run.path().join("output");
    std::fs::create_dir_all(output.join("checkpoint-20"))?;
    std::fs::create_dir_all(output.join("gguf"))?;
    std::fs::create_dir_all(output.join("merged"))?;
    std::fs::write(output.join("adapter_model.safetensors"), "weights")?;
    std::fs::write(
        output.join("checkpoint-20/adapter_model.safetensors"),
        "old",
    )?;
    std::fs::write(output.join("gguf/r1-Q8_0.gguf"), "gguf")?;
    std::fs::write(output.join("merged/config.json"), "{}")?;
    std::fs::write(run.path().join("axolotl.yaml"), "base_model: m\n")?;
    let job_dir = tempfile::tempdir()?;
    ExportJob::staged("r1", "Q4_K_M", run.path(), "output").prepare(job_dir.path(), "/w/e1")?;
    let staged = job_dir.path();
    assert_eq!(
        std::fs::read_to_string(staged.join(SCRIPT_FILE))?,
        EXPORT_SCRIPT
    );
    assert!(staged.join("axolotl.yaml").is_file());
    assert!(staged.join("output/adapter_model.safetensors").is_file());
    assert!(staged.join("output/merged/config.json").is_file());
    assert!(!staged.join("output/checkpoint-20").exists());
    assert!(!staged.join("output/gguf").exists());

    let checkpoint = tempfile::tempdir()?;
    ExportJob::staged("r1", "Q4_K_M", run.path(), "output/checkpoint-20")
        .prepare(checkpoint.path(), "/w/e1")?;
    assert!(
        checkpoint
            .path()
            .join("output/checkpoint-20/adapter_model.safetensors")
            .is_file()
    );
    assert!(!checkpoint.path().join("output/merged").exists());
    Ok(())
}

#[test]
fn a_delivered_export_lands_beside_its_modelfile() -> TestResult {
    let run = tempfile::tempdir()?;
    std::fs::write(
        run.path().join("axolotl.yaml"),
        "base_model: m\nsequence_len: 2048\n",
    )?;
    let job_dir = run.path().join("exports/export-1");
    std::fs::create_dir_all(job_dir.join("output/gguf"))?;
    std::fs::write(job_dir.join("output/gguf/r1-Q8_0.gguf"), "gguf")?;
    std::fs::write(job_dir.join("output/adapter_model.safetensors"), "staged")?;
    std::fs::write(job_dir.join("axolotl.yaml"), "staged")?;
    let delivered = deliver(run.path(), &job_dir, "r1-Q8_0.gguf")?;
    assert_eq!(delivered.gguf, run.path().join("output/gguf/r1-Q8_0.gguf"));
    assert_eq!(std::fs::read_to_string(&delivered.gguf)?, "gguf");
    assert!(!job_dir.join("output").exists() && !job_dir.join("axolotl.yaml").exists());
    assert_eq!(
        std::fs::read_to_string(&delivered.modelfile)?,
        "FROM ./r1-Q8_0.gguf\nPARAMETER num_ctx 2048\n"
    );
    let record: ExportRecord =
        serde_json::from_str(&std::fs::read_to_string(job_dir.join(EXPORT_FILE))?)?;
    assert_eq!(record, delivered.record);
    assert_eq!(record.quantize, "Q8_0");
    assert_eq!(record.llama_cpp, "b11320");
    assert_eq!(record.file, "output/gguf/r1-Q8_0.gguf");
    assert_eq!(record.size, 4);
    assert_eq!(record.sha256, crate::exec::sha256_file(&delivered.gguf)?);
    let missing = deliver(run.path(), &job_dir, "r1-Q4_K_M.gguf");
    assert!(
        matches!(missing, Err(ExportError::NoGguf(_))),
        "{missing:?}"
    );
    Ok(())
}

#[test]
fn an_in_job_export_is_delivered_where_it_is() -> TestResult {
    let run = tempfile::tempdir()?;
    std::fs::create_dir_all(run.path().join("output/gguf"))?;
    std::fs::write(run.path().join("output/gguf/r1-F16.gguf"), "gguf")?;
    std::fs::write(run.path().join("output/adapter_model.safetensors"), "kept")?;
    assert_eq!(
        latest_gguf(run.path(), "r1").as_deref(),
        Some("r1-F16.gguf")
    );
    assert_eq!(latest_gguf(run.path(), "r2"), None);
    let delivered = deliver(run.path(), run.path(), "r1-F16.gguf")?;
    assert!(
        run.path()
            .join("output/adapter_model.safetensors")
            .is_file()
    );
    assert!(run.path().join(EXPORT_FILE).is_file());
    // No axolotl.yaml to read the sequence length from: no num_ctx.
    assert_eq!(
        std::fs::read_to_string(delivered.modelfile)?,
        "FROM ./r1-F16.gguf\n"
    );
    Ok(())
}

#[test]
fn ollama_is_run_in_the_gguf_directory_or_its_command_given() -> TestResult {
    let dir = tempfile::tempdir()?;
    let missing = ollama_create("overbrainer-no-such-ollama", dir.path(), "mentor");
    let Ollama::NotFound(command) = missing else {
        return Err(format!("{missing:?}").into());
    };
    assert!(
        command.ends_with("&& ollama create mentor -f Modelfile"),
        "{command}"
    );
    if !sh_available() {
        return Ok(());
    }
    let fake = dir.path().join("ollama");
    std::fs::write(
        &fake,
        "#!/bin/sh\nprintf '%s ' \"$@\" > args\n[ -f Modelfile ] || { echo 'no Modelfile' >&2; exit 1; }\n",
    )?;
    make_executable(&fake)?;
    let program = fake.to_string_lossy().into_owned();
    assert!(matches!(
        ollama_create(&program, dir.path(), "mentor"),
        Ollama::Failed(message) if message == "no Modelfile"
    ));
    std::fs::write(dir.path().join(MODELFILE), "FROM ./x.gguf\n")?;
    assert_eq!(
        ollama_create(&program, dir.path(), "mentor"),
        Ollama::Created
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join("args"))?,
        "create mentor -f Modelfile "
    );
    Ok(())
}

fn make_executable(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
}

fn sh_available() -> bool {
    Command::new("sh")
        .arg("-c")
        .arg(":")
        .status()
        .is_ok_and(|status| status.success())
}

fn tools_available() -> bool {
    sh_available()
        && ["python3", "tar"].iter().all(|program| {
            Command::new(program)
                .arg("--version")
                .output()
                .is_ok_and(|output| output.status.success())
        })
}

/// The release asset name of this machine, as the script picks it.
fn host_asset() -> Option<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Some("ubuntu-x64"),
        ("linux", "aarch64") => Some("ubuntu-arm64"),
        ("macos", "aarch64") => Some("macos-arm64"),
        _ => None,
    }
}

/// A fake llama.cpp release, Python modules and `axolotl`, and a run.
struct Fixture {
    root: tempfile::TempDir,
    source_sha: String,
    bin_sha: String,
}

/// The fake converter: writes its `--outtype` into `--outfile`, and fails
/// when `gguf-py` is not on `PYTHONPATH`.
const FAKE_CONVERT: &str = r#"import os, sys
args = sys.argv[1:]
if "gguf-py" not in os.environ.get("PYTHONPATH", ""):
    sys.exit("gguf-py is not on PYTHONPATH")
model = args[0]
assert os.path.isfile(os.path.join(model, "config.json")), model
outtype = args[args.index("--outtype") + 1]
with open(args[args.index("--outfile") + 1], "w") as file:
    file.write(f"{outtype} {model}")
"#;

/// The fake `axolotl merge-lora`: what it was given, then a merged model in
/// `<output_dir>/merged`. The config is JSON, which is YAML too.
const FAKE_AXOLOTL: &str = r#"#!/bin/sh
[ "$1" = merge-lora ] || exit 2
cp "$2" seen-merge.yaml
out=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["output_dir"])' "$2")
mkdir -p "$out/merged"
echo '{}' > "$out/merged/config.json"
"#;

/// A `yaml` module of JSON, so the tests never depend on `PyYAML`.
const FAKE_YAML: &str = "import json\n\
def safe_load(file):\n    return json.load(file)\n\
def safe_dump(data, file):\n    json.dump(data, file)\n";

impl Fixture {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let path = root.path();
        let build = path.join("build");
        let source = build.join(format!("llama.cpp-{LLAMA_CPP_TAG}"));
        std::fs::create_dir_all(source.join("gguf-py/gguf"))?;
        std::fs::write(source.join("convert_hf_to_gguf.py"), FAKE_CONVERT)?;
        let bin = build.join(format!("llama-{LLAMA_CPP_TAG}"));
        std::fs::create_dir_all(&bin)?;
        let quantize = bin.join("llama-quantize");
        std::fs::write(
            &quantize,
            "#!/bin/sh\ncp \"$1\" \"$2\" && printf ' %s' \"$3\" >> \"$2\"\n",
        )?;
        make_executable(&quantize)?;
        let remote = path.join("remote");
        let archive = remote.join("archive/refs/tags");
        let download = remote.join(format!("releases/download/{LLAMA_CPP_TAG}"));
        std::fs::create_dir_all(&archive)?;
        std::fs::create_dir_all(&download)?;
        let source_tar = archive.join(format!("{LLAMA_CPP_TAG}.tar.gz"));
        tar(&build, &source_tar, &format!("llama.cpp-{LLAMA_CPP_TAG}"))?;
        let bin_tar = download.join("bin.tar.gz");
        tar(&build, &bin_tar, &format!("llama-{LLAMA_CPP_TAG}"))?;
        for asset in ["ubuntu-x64", "ubuntu-arm64", "macos-arm64"] {
            std::fs::copy(
                &bin_tar,
                download.join(format!("llama-{LLAMA_CPP_TAG}-bin-{asset}.tar.gz")),
            )?;
        }
        let stubs = path.join("stubs");
        for module in ["torch", "numpy", "transformers", "yaml"] {
            std::fs::create_dir_all(stubs.join(module))?;
            let body = if module == "yaml" { FAKE_YAML } else { "" };
            std::fs::write(stubs.join(module).join("__init__.py"), body)?;
        }
        let tools = path.join("bin");
        std::fs::create_dir_all(&tools)?;
        std::fs::write(tools.join("axolotl"), FAKE_AXOLOTL)?;
        make_executable(&tools.join("axolotl"))?;
        let run = path.join("run");
        std::fs::create_dir_all(run.join("output"))?;
        ExportJob::write_script(&run)?;
        Ok(Self {
            source_sha: crate::exec::sha256_file(&source_tar)?,
            bin_sha: crate::exec::sha256_file(&bin_tar)?,
            root,
        })
    }

    fn run(&self) -> std::path::PathBuf {
        self.root.path().join("run")
    }

    /// Runs the script in the run directory as an export in a training job.
    fn export(
        &self,
        quantize: &str,
        edit: impl Fn(&mut Command),
    ) -> std::io::Result<std::process::Output> {
        let path = self.root.path();
        let mut command = Command::new("sh");
        command.arg(self.run().join(SCRIPT_FILE)).current_dir(path);
        for (name, value) in ExportJob::in_job("r1", quantize).env(&self.run().to_string_lossy()) {
            command.env(name, value);
        }
        let url = format!("file://{}", path.join("remote").display());
        command
            .env("OVERBRAINER_LLAMA_CPP_URL", url)
            .env("OVERBRAINER_LLAMA_CPP_SOURCE_SHA256", &self.source_sha)
            .env("OVERBRAINER_LLAMA_CPP_UBUNTU_X64_SHA256", &self.bin_sha)
            .env("OVERBRAINER_LLAMA_CPP_UBUNTU_ARM64_SHA256", &self.bin_sha)
            .env("OVERBRAINER_LLAMA_CPP_MACOS_ARM64_SHA256", &self.bin_sha)
            .env("OVERBRAINER_CACHE", path.join("cache"))
            .env("PYTHONPATH", path.join("stubs"))
            .env(
                "PATH",
                format!(
                    "{}:{}",
                    path.join("bin").display(),
                    std::env::var("PATH").unwrap_or_default()
                ),
            );
        edit(&mut command);
        command.output()
    }
}

fn tar(dir: &Path, archive: &Path, top: &str) -> Result<(), Box<dyn std::error::Error>> {
    let status = Command::new("tar")
        .arg("-czf")
        .arg(archive)
        .arg("-C")
        .arg(dir)
        .arg(top)
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("tar failed: {status}").into())
    }
}

fn text(output: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn a_full_model_is_converted_quantized_and_the_tools_cached() -> TestResult {
    if !tools_available() || host_asset().is_none() {
        eprintln!("skipped: needs sh, python3 and tar on a supported platform");
        return Ok(());
    }
    let fixture = Fixture::new()?;
    let run = fixture.run();
    std::fs::write(run.join("output/config.json"), "{}")?;
    let output = fixture.export("Q4_K_M", |_| {})?;
    assert!(output.status.success(), "{}", text(&output));
    let gguf = std::fs::read_to_string(run.join("output/gguf/r1-Q4_K_M.gguf"))?;
    assert_eq!(gguf, "auto output Q4_K_M");
    assert!(!run.join("export-work").exists());
    // The job writes the stage event before the script: the script writes none.
    assert!(!run.join("metrics.jsonl").exists());
    let cache = fixture
        .root
        .path()
        .join("cache/llama.cpp")
        .join(LLAMA_CPP_TAG);
    assert!(cache.join("source/convert_hf_to_gguf.py").is_file());
    let asset = host_asset().ok_or("no asset")?;
    assert!(cache.join(asset).join("llama-quantize").is_file());
    // Cached: the release is not downloaded again.
    std::fs::remove_dir_all(fixture.root.path().join("remote"))?;
    let again = fixture.export("Q4_K_M", |_| {})?;
    assert!(again.status.success(), "{}", text(&again));
    Ok(())
}

#[test]
fn an_adapter_is_merged_first_and_f16_is_not_quantized() -> TestResult {
    if !tools_available() {
        eprintln!("skipped: needs sh, python3 and tar");
        return Ok(());
    }
    let fixture = Fixture::new()?;
    let run = fixture.run();
    std::fs::write(run.join("output/adapter_config.json"), "{}")?;
    std::fs::write(
        run.join("axolotl.yaml"),
        r#"{"base_model": "m", "plugins": ["p"], "output_dir": "/w/r1/output"}"#,
    )?;
    // F16 needs no llama-quantize: its digest is never checked.
    let output = fixture.export("F16", |command| {
        command.env("OVERBRAINER_LLAMA_CPP_UBUNTU_X64_SHA256", "0");
    })?;
    assert!(output.status.success(), "{}", text(&output));
    assert_eq!(
        std::fs::read_to_string(run.join("output/gguf/r1-F16.gguf"))?,
        "f16 export-work/merged"
    );
    let merge: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(run.join("seen-merge.yaml"))?)?;
    assert_eq!(
        merge,
        serde_json::json!({
            "base_model": "m",
            "output_dir": "export-work",
            "lora_model_dir": "output",
        })
    );
    assert!(!run.join("export-work").exists());
    Ok(())
}

#[test]
fn a_release_with_another_digest_is_never_used() -> TestResult {
    if !tools_available() {
        eprintln!("skipped: needs sh, python3 and tar");
        return Ok(());
    }
    let fixture = Fixture::new()?;
    let run = fixture.run();
    std::fs::write(run.join("output/config.json"), "{}")?;
    let wrong = "0".repeat(64);
    let output = fixture.export("F16", |command| {
        command.env("OVERBRAINER_LLAMA_CPP_SOURCE_SHA256", &wrong);
    })?;
    assert!(!output.status.success());
    let said = text(&output);
    assert!(
        said.contains(&format!("expected {wrong}: not used")),
        "{said}"
    );
    let cache = fixture
        .root
        .path()
        .join("cache/llama.cpp")
        .join(LLAMA_CPP_TAG);
    assert!(!cache.join("source").exists());
    assert!(!cache.join("source.tar.gz").exists());
    assert!(!run.join("output/gguf/r1-F16.gguf").exists());
    Ok(())
}

#[test]
fn a_missing_module_or_platform_is_named() -> TestResult {
    if !tools_available() {
        eprintln!("skipped: needs sh, python3 and tar");
        return Ok(());
    }
    let fixture = Fixture::new()?;
    std::fs::write(fixture.run().join("output/config.json"), "{}")?;
    std::fs::remove_dir_all(fixture.root.path().join("stubs/torch"))?;
    let output = fixture.export("F16", |_| {})?;
    // A torch installed on this machine would stand in for the stub.
    let torch_installed = Command::new("python3")
        .args(["-c", "import torch"])
        .output()
        .is_ok_and(|output| output.status.success());
    if !torch_installed {
        assert!(!output.status.success());
        let said = text(&output);
        assert!(said.contains("needs the Python module torch"), "{said}");
    }

    let uname = fixture.root.path().join("bin/uname");
    std::fs::write(&uname, "#!/bin/sh\necho 'Linux riscv64'\n")?;
    make_executable(&uname)?;
    let output = fixture.export("Q4_K_M", |_| {})?;
    assert!(!output.status.success());
    assert!(
        text(&output).contains("has no prebuilt llama-quantize for Linux riscv64"),
        "{}",
        text(&output)
    );
    Ok(())
}

#[test]
fn the_trampoline_runs_the_script_with_python_s_directory_first_on_path() -> TestResult {
    if !tools_available() {
        eprintln!("skipped: needs sh and python3");
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let script = dir.path().join("probe.sh");
    std::fs::write(&script, "printf '%s' \"${PATH%%:*}\"\n")?;
    let output = Command::new("python3")
        .args(["-c", TRAMPOLINE])
        .arg(&script)
        .output()?;
    assert!(output.status.success(), "{}", text(&output));
    let python = Command::new("python3")
        .args([
            "-c",
            "import os, sys; print(os.path.dirname(sys.executable), end='')",
        ])
        .output()?;
    assert_eq!(output.stdout, python.stdout);
    Ok(())
}

/// The phase the lines of the metrics file `path` give a running job.
fn phase_of(path: &Path) -> Result<crate::train::Phase, Box<dyn std::error::Error>> {
    let mut phases = crate::train::Phases::default();
    for line in std::fs::read_to_string(path)?.lines() {
        phases.line(&crate::train::parse_line(line)?);
    }
    Ok(phases.phase(false))
}

#[test]
fn an_export_in_the_training_job_shows_as_exporting_gguf() -> TestResult {
    if !sh_available() {
        eprintln!("skipped: needs sh");
        return Ok(());
    }
    let training: crate::config::Training = serde_json::from_value(serde_json::json!({
        "target": "local", "base_model": "m", "adapter": "lora", "merge": true
    }))?;
    let project = tempfile::tempdir()?;
    let trainer =
        crate::train::Axolotl::new(&training, &crate::dataset::DataFiles::new(project.path()))
            .exporting("r1", "Q4_K_M");
    let stages = trainer.stages();
    assert_eq!(stages.last(), Some(&JobStage::Export));
    // The job's own chain, each command standing in for the real one.
    let commands = vec![vec!["true".to_string()]; trainer.commands().len()];
    let job = crate::exec::JobRuntime::Native {
        venv: None,
        env_file: None,
    }
    .job(crate::exec::JobSpec {
        run_id: "r1",
        run_dir: "/unused",
        cache_dir: "/unused",
        commands: &commands,
        env: &[],
        stop_marker: None,
        stages: Some(crate::exec::Stages {
            file: METRICS_FILE,
            names: &stages,
        }),
        secrets: Vec::new(),
        tools_dir: None,
    });
    let dir = tempfile::tempdir()?;
    let status = Command::new("sh")
        .arg("-c")
        .arg(&job.script)
        .current_dir(dir.path())
        .status()?;
    assert!(status.success(), "{}", job.script);
    let phase = phase_of(&dir.path().join(METRICS_FILE))?;
    assert_eq!(phase.label(), "exporting GGUF");
    Ok(())
}

#[tokio::test]
async fn a_standalone_export_shows_as_exporting_gguf() -> TestResult {
    use crate::exec::{Executor as _, JobRuntime, LocalExecutor};
    use crate::runs::{Launch, RunCtx, Runs, create, start_mounted};
    if !tools_available() {
        eprintln!("skipped: needs sh and python3");
        return Ok(());
    }
    let project = tempfile::tempdir()?;
    let runs = Runs::new(project.path());
    let executor = LocalExecutor::new(runs.dir())?;
    let run_dir = format!("{}/r1", executor.workdir());
    std::fs::create_dir_all(&run_dir)?;
    let exports = runs.exports("r1")?;
    let bus = crate::events::EventBus::new();
    let ctx = RunCtx {
        runs: &exports,
        executor: &executor,
        bus: &bus,
        poll: std::time::Duration::from_millis(50),
    };
    let record = create(&exports, "export", &format!("{run_dir}/exports"), "box")?;
    let script = format!("{}/{SCRIPT_FILE}", record.remote_dir);
    // No model: the script fails once its stage is written.
    let job = ExportJob::in_place("r1", "F16", "output", script);
    let runtime = JobRuntime::Native {
        venv: None,
        env_file: None,
    };
    let launch = Launch {
        runtime: &runtime,
        secrets: Vec::new(),
    };
    let started = start_mounted(&ctx, &job, launch, record, &run_dir).await?;
    let id = started.job.ok_or("no job")?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while !executor.status(&id).await?.is_finished() {
        if std::time::Instant::now() > deadline {
            return Err("the export job did not end".into());
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    let metrics = Path::new(&started.remote_dir).join(METRICS_FILE);
    assert_eq!(phase_of(&metrics)?.label(), "exporting GGUF");
    assert!(!Path::new(&run_dir).join(METRICS_FILE).exists());
    Ok(())
}
