//! The llama.cpp release an export uses: one pinned tag, its source tarball and
//! the prebuilt binaries for each platform (`llama-quantize` for an export,
//! `llama-server` for a compare, with a CUDA build and its runtime for NVIDIA
//! GPUs and a `ROCm` build for AMD GPUs), each with its SHA-256. The scripts
//! download them on the target, check them against these digests before
//! extracting anything, and cache them there.
//!
//! To move to a newer release, see "Bumping llama.cpp" in `docs/training.md`.

/// The release tag.
pub const TAG: &str = "b11320";

/// The repository the source tarball and the release assets come from.
pub const REPO_URL: &str = "https://github.com/ggml-org/llama.cpp";

/// SHA-256 of `archive/refs/tags/<TAG>.tar.gz`, the source tree holding
/// `convert_hf_to_gguf.py`, its `conversion/` package and `gguf-py/`.
pub const SOURCE_SHA256: &str = "edd777f61c1d3704456f01bbb9a77b4bd71a1b39d9b8fe0109c3d65c2c4ece74";

/// SHA-256 of `llama-<TAG>-bin-ubuntu-x64.tar.gz` (Linux `x86_64`, glibc 2.34).
pub const UBUNTU_X64_SHA256: &str =
    "ef1856938dc1434138ce53688791eb0d2d64cf46e309a0942a12bba3366c0919";

/// SHA-256 of `llama-<TAG>-bin-ubuntu-arm64.tar.gz` (Linux `aarch64`, glibc 2.38).
pub const UBUNTU_ARM64_SHA256: &str =
    "88589b963d8e2ffd2d4df2f542ed7e301fb636b637c99081f5d58646aee20a9a";

/// SHA-256 of `llama-<TAG>-bin-macos-arm64.tar.gz` (macOS on Apple silicon).
pub const MACOS_ARM64_SHA256: &str =
    "f6f337fc7d2ff9260f53177cf4fe6bbf6b0f7faa75a49fb224aaf66885a5c956";

/// The CUDA version of the CUDA builds a compare runs on an NVIDIA GPU.
pub const CUDA: &str = "13.4";

/// SHA-256 of `llama-<TAG>-bin-ubuntu-cuda-<CUDA>-x64.tar.gz` (Linux `x86_64`, NVIDIA GPU).
pub const UBUNTU_CUDA_X64_SHA256: &str =
    "4c6b7cf40228efb96930f6694e7faae43a7e01993baba8e7894e2a3fc43e8a31";

/// SHA-256 of `llama-<TAG>-bin-ubuntu-cuda-<CUDA>-arm64.tar.gz` (Linux `aarch64`, NVIDIA GPU).
pub const UBUNTU_CUDA_ARM64_SHA256: &str =
    "df2ea0d5c3bd5492dda70cd86f2a384709f7aa7ceec123b159e20b2844ce2f46";

/// SHA-256 of `cudart-llama-<TAG>-bin-ubuntu-cuda-<CUDA>-x64.tar.gz`: the CUDA
/// runtime and cuBLAS the x64 CUDA build loads, when the machine lacks them.
pub const CUDART_X64_SHA256: &str =
    "249a86f9156641b98d586c6c6c649335f691d3a27f9fdaaca699af9cbfbb3ca2";

/// SHA-256 of `cudart-llama-<TAG>-bin-ubuntu-cuda-<CUDA>-arm64.tar.gz`.
pub const CUDART_ARM64_SHA256: &str =
    "e0895065d9244b3f8a50c2d85e196605e8a48c1f06a53f2f52db089fbd29d8bb";

/// The `ROCm` label of the `ROCm` build a compare runs on an AMD GPU.
pub const ROCM: &str = "10.0";

/// SHA-256 of `llama-<TAG>-bin-ubuntu-rocm-<ROCM>-x64.tar.gz` (Linux `x86_64`,
/// AMD GPU). It does not bundle the `ROCm` 7 runtime (`libamdhip64.so.7`,
/// `librocblas.so.5`, `libhipblas.so.3`): the system provides it. There is no
/// `ROCm` build for `aarch64`.
pub const UBUNTU_ROCM_X64_SHA256: &str =
    "d3f610c849bc8d365c31a62dba9c1a309a783202fe5d86d85f6a64b34cecb195";

/// The oldest glibc the GPU builds run on: the CUDA builds (`x86_64` and
/// `aarch64`) and the `ROCm` build all need `GLIBC_2.38` (Ubuntu 24.04), while
/// the `x86_64` CPU build needs 2.34 only. A compare on an older glibc (Ubuntu
/// 22.04 has 2.35) runs the CPU build instead, with a warning.
pub const GPU_GLIBC: &str = "2.38";

/// The environment the scripts read the pin from: the tag, the CUDA and `ROCm`
/// versions, the glibc the GPU builds need, the repository, and each digest.
#[must_use]
pub fn env() -> Vec<(String, String)> {
    [
        ("OVERBRAINER_LLAMA_CPP", TAG),
        ("OVERBRAINER_LLAMA_CPP_URL", REPO_URL),
        ("OVERBRAINER_LLAMA_CPP_SOURCE_SHA256", SOURCE_SHA256),
        ("OVERBRAINER_LLAMA_CPP_UBUNTU_X64_SHA256", UBUNTU_X64_SHA256),
        (
            "OVERBRAINER_LLAMA_CPP_UBUNTU_ARM64_SHA256",
            UBUNTU_ARM64_SHA256,
        ),
        (
            "OVERBRAINER_LLAMA_CPP_MACOS_ARM64_SHA256",
            MACOS_ARM64_SHA256,
        ),
        ("OVERBRAINER_LLAMA_CPP_CUDA", CUDA),
        (
            "OVERBRAINER_LLAMA_CPP_UBUNTU_CUDA_X64_SHA256",
            UBUNTU_CUDA_X64_SHA256,
        ),
        (
            "OVERBRAINER_LLAMA_CPP_UBUNTU_CUDA_ARM64_SHA256",
            UBUNTU_CUDA_ARM64_SHA256,
        ),
        ("OVERBRAINER_LLAMA_CPP_CUDART_X64_SHA256", CUDART_X64_SHA256),
        ("OVERBRAINER_LLAMA_CPP_ROCM", ROCM),
        ("OVERBRAINER_LLAMA_CPP_GPU_GLIBC", GPU_GLIBC),
        (
            "OVERBRAINER_LLAMA_CPP_UBUNTU_ROCM_X64_SHA256",
            UBUNTU_ROCM_X64_SHA256,
        ),
        (
            "OVERBRAINER_LLAMA_CPP_CUDART_ARM64_SHA256",
            CUDART_ARM64_SHA256,
        ),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_string(), value.to_string()))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pinned digest is 64 lowercase hex digits.
    #[test]
    fn every_digest_is_a_sha256_in_lowercase_hex() {
        for digest in [
            SOURCE_SHA256,
            UBUNTU_X64_SHA256,
            UBUNTU_ARM64_SHA256,
            MACOS_ARM64_SHA256,
            UBUNTU_CUDA_X64_SHA256,
            UBUNTU_CUDA_ARM64_SHA256,
            CUDART_X64_SHA256,
            CUDART_ARM64_SHA256,
            UBUNTU_ROCM_X64_SHA256,
        ] {
            assert_eq!(digest.len(), 64, "{digest}");
            assert!(
                digest
                    .chars()
                    .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
                "{digest}"
            );
        }
    }

    /// The script's environment holds the release tag, the CUDA version, the
    /// glibc of the GPU builds and every build's digest.
    #[test]
    fn the_script_gets_the_tag_and_every_digest() {
        let env = env();
        let get = |name: &str| {
            env.iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.as_str())
        };
        assert_eq!(get("OVERBRAINER_LLAMA_CPP"), Some(TAG));
        assert_eq!(
            get("OVERBRAINER_LLAMA_CPP_MACOS_ARM64_SHA256"),
            Some(MACOS_ARM64_SHA256)
        );
        assert_eq!(get("OVERBRAINER_LLAMA_CPP_CUDA"), Some(CUDA));
        assert_eq!(
            get("OVERBRAINER_LLAMA_CPP_UBUNTU_CUDA_X64_SHA256"),
            Some(UBUNTU_CUDA_X64_SHA256)
        );
        assert_eq!(
            get("OVERBRAINER_LLAMA_CPP_CUDART_ARM64_SHA256"),
            Some(CUDART_ARM64_SHA256)
        );
        assert_eq!(
            get("OVERBRAINER_LLAMA_CPP_UBUNTU_CUDA_ARM64_SHA256"),
            Some(UBUNTU_CUDA_ARM64_SHA256)
        );
        assert_eq!(
            get("OVERBRAINER_LLAMA_CPP_CUDART_X64_SHA256"),
            Some(CUDART_X64_SHA256)
        );
        assert_eq!(get("OVERBRAINER_LLAMA_CPP_ROCM"), Some(ROCM));
        assert_eq!(
            get("OVERBRAINER_LLAMA_CPP_UBUNTU_ROCM_X64_SHA256"),
            Some(UBUNTU_ROCM_X64_SHA256)
        );
        assert_eq!(get("OVERBRAINER_LLAMA_CPP_GPU_GLIBC"), Some(GPU_GLIBC));
        assert_eq!(env.len(), 14);
    }
}
