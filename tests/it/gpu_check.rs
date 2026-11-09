//! The GPU check script run against stub `torch` packages with the local
//! `python3`; skipped when it is not installed.

use std::fs;
use std::path::Path;
use std::process::Command;

use overbrainer::train::{GPU_CHECK_SCRIPT, GpuCheck};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Whether `python3` runs here.
fn python_available() -> bool {
    Command::new("python3")
        .arg("-c")
        .arg("pass")
        .output()
        .is_ok_and(|output| output.status.success())
}

/// A stub `torch` package in `dir`: `count` CUDA devices, and an MPS backend
/// that is available when `mps` is set.
fn stub_torch(dir: &Path, count: usize, mps: bool) -> std::io::Result<()> {
    let package = dir.join("torch");
    fs::create_dir_all(&package)?;
    let py_bool = |value: bool| if value { "True" } else { "False" };
    fs::write(
        package.join("__init__.py"),
        format!(
            "__version__ = '2.9.0'\n\
             class version:\n    cuda = '13.0'\n    hip = None\n\
             class cuda:\n    \
                 @staticmethod\n    def is_available(): return {available}\n    \
                 @staticmethod\n    def device_count(): return {count}\n    \
                 @staticmethod\n    def get_device_name(i): return 'NVIDIA L4'\n\
             class backends:\n    class mps:\n        \
                 @staticmethod\n        def is_available(): return {mps}\n",
            available = py_bool(count > 0),
            mps = py_bool(mps),
        ),
    )
}

/// Runs the script with `torch` from `path` (none when `None`), requiring
/// `min_gpus`, and returns its exit code and parsed result.
fn run(
    dir: &Path,
    path: Option<&Path>,
    min_gpus: u32,
    platform: &str,
) -> Result<(Option<i32>, Option<GpuCheck>), Box<dyn std::error::Error>> {
    let script = dir.join("gpu_check.py");
    fs::write(&script, GPU_CHECK_SCRIPT)?;
    let result = dir.join("gpu_check.json");
    let _ = fs::remove_file(&result);
    // Runs the script with `sys.platform` set, so the macOS branch is tested
    // on any machine.
    let launcher = format!(
        "import runpy, sys; sys.platform = {platform:?}; sys.argv = sys.argv[1:]; \
         runpy.run_path(sys.argv[0], run_name='__main__')"
    );
    let mut command = Command::new("python3");
    command
        .args(["-S", "-c", &launcher])
        .arg(&script)
        .arg(&result)
        .arg(min_gpus.to_string())
        .env_remove("PYTHONHOME");
    match path {
        Some(path) => command.env("PYTHONPATH", path),
        None => command.env("PYTHONPATH", dir.join("empty")),
    };
    let status = command.status()?;
    let check = fs::read(&result)
        .ok()
        .and_then(|content| GpuCheck::parse(&content));
    Ok((status.code(), check))
}

/// Enough CUDA devices pass; none, too few or no torch fail with their codes.
#[test]
fn the_check_follows_the_devices_torch_sees() -> TestResult {
    if !python_available() {
        eprintln!("skipped: python3 is not installed");
        return Ok(());
    }
    let dir = tempfile::tempdir()?;

    let two = dir.path().join("two");
    stub_torch(&two, 2, false)?;
    let (code, check) = run(dir.path(), Some(&two), 2, "linux")?;
    assert_eq!(code, Some(0));
    let check = check.ok_or("no result")?;
    assert_eq!(check.code, 0);
    assert_eq!(check.devices, ["NVIDIA L4", "NVIDIA L4"]);

    let (code, check) = run(dir.path(), Some(&two), 4, "linux")?;
    assert_eq!(code, Some(5));
    assert_eq!(check.ok_or("no result")?.required, Some(4));

    let none = dir.path().join("none");
    stub_torch(&none, 0, false)?;
    let (code, check) = run(dir.path(), Some(&none), 1, "linux")?;
    assert_eq!(code, Some(4));
    assert_eq!(check.ok_or("no result")?.backend.as_deref(), Some("cuda"));

    let (code, check) = run(dir.path(), None, 1, "linux")?;
    assert_eq!(code, Some(3));
    assert_eq!(check.ok_or("no result")?.code, 3);
    Ok(())
}

/// On macOS the check asks for an MPS device, not CUDA.
#[test]
fn on_macos_the_check_asks_for_mps() -> TestResult {
    if !python_available() {
        eprintln!("skipped: python3 is not installed");
        return Ok(());
    }
    let dir = tempfile::tempdir()?;
    let with = dir.path().join("with");
    stub_torch(&with, 0, true)?;
    let (code, check) = run(dir.path(), Some(&with), 1, "darwin")?;
    assert_eq!(code, Some(0));
    assert_eq!(check.ok_or("no result")?.backend.as_deref(), Some("mps"));

    let without = dir.path().join("without");
    stub_torch(&without, 2, false)?;
    let (code, _) = run(dir.path(), Some(&without), 1, "darwin")?;
    assert_eq!(code, Some(4));
    Ok(())
}
