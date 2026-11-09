//! The GPU check a training job runs before its trainer: the trainer's `PyTorch`
//! must see enough CUDA or `ROCm` devices (an MPS device on macOS), or the run
//! stops before any GPU time is spent on a CPU fallback.

use serde::Deserialize;

use crate::exec::Preflight;

/// The check's script, written into every run directory.
pub const GPU_CHECK_SCRIPT: &str = include_str!("gpu_check.py");
/// The script's file name in the run directory.
pub const GPU_CHECK_FILE: &str = "gpu_check.py";
/// The file the script writes its result to, in the run directory.
pub const GPU_CHECK_RESULT: &str = "gpu_check.json";
/// The job's exit code when the check fails.
pub const GPU_CHECK_CODE: u8 = 90;

/// The script's code when torch cannot be imported.
const IMPORT_FAILED: u8 = 3;
/// The script's code when torch sees no device.
const NO_DEVICE: u8 = 4;
/// The script's code when torch sees fewer devices than required.
const TOO_FEW: u8 = 5;

/// The preflight check of a job whose run directory is `root` as the job sees
/// it: at least `min_gpus` devices.
#[must_use]
pub fn gpu_preflight(root: &str, min_gpus: u32) -> Preflight {
    Preflight {
        args: vec![
            format!("{root}/{GPU_CHECK_FILE}"),
            format!("{root}/{GPU_CHECK_RESULT}"),
            min_gpus.to_string(),
        ],
        code: GPU_CHECK_CODE,
    }
}

/// What the check wrote to [`GPU_CHECK_RESULT`].
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct GpuCheck {
    /// 0 when the check passed, else the reason's code.
    pub code: u8,
    /// `cuda` (CUDA or `ROCm`) or `mps`; absent when torch could not be imported.
    #[serde(default)]
    pub backend: Option<String>,
    /// The names of the devices torch sees.
    #[serde(default)]
    pub devices: Vec<String>,
    /// Least devices the run needs.
    #[serde(default)]
    pub required: Option<u32>,
}

impl GpuCheck {
    /// Parses the check's result, `None` when it is not one.
    #[must_use]
    pub fn parse(content: &[u8]) -> Option<Self> {
        serde_json::from_slice(content).ok()
    }

    /// Why the check failed, as the run's message ending with `log` (where the
    /// job log is), or `None` when it passed or gave a code it does not know.
    #[must_use]
    pub fn failure(&self, log: &str) -> Option<String> {
        let why = match self.code {
            IMPORT_FAILED => "the trainer's Python cannot import torch".to_string(),
            NO_DEVICE if self.backend.as_deref() == Some("mps") => {
                "PyTorch sees no MPS device".to_string()
            },
            NO_DEVICE => "PyTorch sees no CUDA or ROCm device: check the image's CUDA version \
                          against the host driver, or set `require_gpu = false` on a local or \
                          SSH target for a CPU run"
                .to_string(),
            TOO_FEW => format!(
                "PyTorch sees {} GPU(s), the target asks for {}",
                self.devices.len(),
                self.required.unwrap_or(1)
            ),
            _ => return None,
        };
        Some(format!("the GPU check failed: {why} ({log})"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The preflight runs the script on the run directory with the GPU count.
    #[test]
    fn the_preflight_runs_the_script_with_the_gpu_count() {
        let check = gpu_preflight("/workspace/run", 2);
        assert_eq!(
            check.args,
            [
                "/workspace/run/gpu_check.py",
                "/workspace/run/gpu_check.json",
                "2"
            ]
        );
        assert_eq!(check.code, GPU_CHECK_CODE);
    }

    /// Each failure code names its cause; a pass or an unknown code names none.
    #[test]
    fn each_failure_names_its_cause() -> Result<(), String> {
        let read = |json: &str| GpuCheck::parse(json.as_bytes()).ok_or(json.to_string());
        let log = "see runs/r1/job.log";
        let failure =
            |json: &str| -> Result<Option<String>, String> { Ok(read(json)?.failure(log)) };
        assert_eq!(
            failure(r#"{"code":3,"error":"ModuleNotFoundError"}"#)?.as_deref(),
            Some(
                "the GPU check failed: the trainer's Python cannot import torch (see runs/r1/job.log)"
            )
        );
        assert!(
            failure(r#"{"code":4,"backend":"cuda","devices":[]}"#)?
                .is_some_and(|message| message.contains("no CUDA or ROCm device")
                    && message.contains("require_gpu = false"))
        );
        assert_eq!(
            failure(r#"{"code":4,"backend":"mps","devices":[]}"#)?.as_deref(),
            Some("the GPU check failed: PyTorch sees no MPS device (see runs/r1/job.log)")
        );
        assert_eq!(
            failure(r#"{"code":5,"backend":"cuda","devices":["NVIDIA L4"],"required":2}"#)?
                .as_deref(),
            Some(
                "the GPU check failed: PyTorch sees 1 GPU(s), the target asks for 2 (see runs/r1/job.log)"
            )
        );
        assert_eq!(failure(r#"{"code":0,"devices":["NVIDIA L4"]}"#)?, None);
        assert_eq!(failure(r#"{"code":7}"#)?, None);
        assert_eq!(GpuCheck::parse(b"not json"), None);
        Ok(())
    }
}
