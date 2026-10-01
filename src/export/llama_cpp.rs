//! The llama.cpp release an export uses: one pinned tag, its source tarball and
//! the prebuilt `llama-quantize` for each platform, each with its SHA-256. The
//! export script downloads them on the target, checks them against these
//! digests before extracting anything, and caches them there.
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

/// The environment the export script reads the pin from: the tag, the
/// repository, and each digest.
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
    ]
    .into_iter()
    .map(|(name, value)| (name.to_string(), value.to_string()))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_digest_is_a_sha256_in_lowercase_hex() {
        for digest in [
            SOURCE_SHA256,
            UBUNTU_X64_SHA256,
            UBUNTU_ARM64_SHA256,
            MACOS_ARM64_SHA256,
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
        assert_eq!(env.len(), 6);
    }
}
