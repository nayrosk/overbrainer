//! SHA-256 manifests of a run's files: computed on the target with `sha256sum`,
//! and here with the `sha2` crate, so a download can be checked file by file.

use std::fs::{self, File};
use std::io::{self, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

use super::{ExecError, quote};

/// A regular file of a run directory and the SHA-256 of its content.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct FileDigest {
    /// Path relative to the run directory, with `/` separators, for example
    /// `output/adapter_model.safetensors`.
    pub path: String,
    /// SHA-256 of the content, 64 lowercase hexadecimal digits.
    pub sha256: String,
}

/// Prints one NUL-terminated `<hash> <path>` record for every regular file under
/// the `entries` of `remote` that exist, leaving out names matching an `exclude`
/// pattern at any depth (the same rule as the `tar --exclude` of a download), or
/// nothing when no entry exists. Fails, saying so, when `remote` itself is not a
/// directory. Paths are printed with a leading `./`, so an entry can never read
/// as an option.
///
/// Each file is hashed through `sha256sum`'s standard input, so a file name never
/// reaches `sha256sum` and never has to survive its own escaping: GNU coreutils
/// escapes a name holding a backslash or a line break, busybox does not, and a
/// script working with one would not work with the other. The record itself is
/// framed by this script, not by `sha256sum`, and terminated with a NUL byte,
/// which cannot appear in a POSIX path, so a name can hold anything else,
/// including a line break, a backslash or a quote, without ambiguity.
///
/// Needs `find` and `sha256sum`, which GNU coreutils and busybox both provide.
#[must_use]
pub fn manifest_script(remote: &str, entries: &[String], exclude: &[String]) -> String {
    let candidates: Vec<String> = entries.iter().map(|entry| quote(entry)).collect();
    let prune = if exclude.is_empty() {
        String::new()
    } else {
        let names: Vec<String> = exclude
            .iter()
            .map(|pattern| format!("-name {}", quote(pattern)))
            .collect();
        format!("\\( {} \\) -prune -o ", names.join(" -o "))
    };
    format!(
        "[ -d {dir} ] || {{ printf '%s does not exist\\n' {dir} >&2; exit 1; }}\n\
         cd -- {dir} || exit 1\n\
         for entry in {candidates}; do\n\
         [ -e \"$entry\" ] || continue\n\
         find \"./$entry\" {prune}-type f -exec sh -c '\n\
         for f; do\n\
         hash=$(sha256sum < \"$f\") || exit 1\n\
         printf \"%s %s\\0\" \"${{hash%% *}}\" \"$f\"\n\
         done\n\
         ' sh {{}} + || exit 1\n\
         done\n",
        dir = quote(remote),
        candidates = candidates.join(" "),
    )
}

/// Reads the output of [`manifest_script`], sorted by path.
///
/// # Errors
///
/// Returns [`ExecError::Protocol`] when non-empty `output` does not end with the
/// NUL that terminates every record: without it, an output cut short mid-record
/// would otherwise read as a shorter, wrong path. Also returns
/// [`ExecError::Protocol`] for a record that is not `<64 hex digits>`, a space
/// and a path, or whose path is absolute or holds a `..` component (a line break
/// in a name stays legal: only `/`-separated components are checked).
pub fn parse_manifest(output: &str) -> Result<Vec<FileDigest>, ExecError> {
    if !output.is_empty() && !output.ends_with('\0') {
        return Err(ExecError::Protocol(
            "the manifest was cut short: it does not end with its final NUL".to_string(),
        ));
    }
    let mut digests = Vec::new();
    for record in output.split('\0').filter(|record| !record.is_empty()) {
        digests.push(parse_record(record)?);
    }
    digests.sort();
    Ok(digests)
}

/// One `<hash> <path>` record of [`manifest_script`], without its trailing NUL.
fn parse_record(record: &str) -> Result<FileDigest, ExecError> {
    let invalid = || ExecError::Protocol(format!("unexpected manifest record {record:?}"));
    let hash = record.get(..64).ok_or_else(invalid)?;
    if !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid());
    }
    let name = record
        .get(64..)
        .and_then(|tail| tail.strip_prefix(' '))
        .ok_or_else(invalid)?;
    let path = name.strip_prefix("./").unwrap_or(name).to_string();
    if path.starts_with('/') || path.split('/').any(|part| part == "..") {
        return Err(ExecError::Protocol(format!(
            "unsafe manifest path {path:?}"
        )));
    }
    Ok(FileDigest {
        path,
        sha256: hash.to_ascii_lowercase(),
    })
}

/// SHA-256 of the file at `path`, as 64 lowercase hexadecimal digits.
///
/// # Errors
///
/// Returns the I/O error when the file cannot be read.
pub fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 256 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from(DIGITS[usize::from(byte >> 4)]));
        out.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    out
}

/// The manifest of the `entries` of the local directory `dir`, computed here with
/// the same rules as [`manifest_script`]: regular files only (symbolic links are
/// not followed), names matching an `exclude` pattern skipped at any depth,
/// missing entries skipped.
///
/// # Errors
///
/// Returns [`ExecError::Command`] (action `manifest`) when `dir` is not a
/// directory, and [`ExecError::Io`] when a file cannot be read.
pub fn local_manifest(
    dir: &Path,
    entries: &[String],
    exclude: &[String],
) -> Result<Vec<FileDigest>, ExecError> {
    if !dir.is_dir() {
        // Debug-quoted (through its already-lossy display text, to sidestep
        // clippy's own Path-Debug lint) so a control character in `dir` cannot
        // reach the terminal raw.
        let dir = dir.display().to_string();
        return Err(ExecError::Command {
            action: "manifest",
            message: format!("{dir:?} does not exist"),
        });
    }
    let mut digests = Vec::new();
    for entry in entries {
        walk(dir, entry, exclude, &mut digests)?;
    }
    digests.sort();
    Ok(digests)
}

/// Adds the regular files under `relative` (inside `root`) to `digests`.
fn walk(
    root: &Path,
    relative: &str,
    exclude: &[String],
    digests: &mut Vec<FileDigest>,
) -> Result<(), ExecError> {
    let path = root.join(relative);
    let name = Path::new(relative)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    if exclude.iter().any(|pattern| glob_match(pattern, &name)) {
        return Ok(());
    }
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(source) => return Err(ExecError::Io { path, source }),
    };
    if metadata.is_file() {
        let sha256 = sha256_file(&path).map_err(|source| ExecError::Io {
            path: path.clone(),
            source,
        })?;
        digests.push(FileDigest {
            path: relative.to_string(),
            sha256,
        });
    } else if metadata.is_dir() {
        let children = fs::read_dir(&path).map_err(|source| ExecError::Io {
            path: path.clone(),
            source,
        })?;
        for child in children {
            let child = child.map_err(|source| ExecError::Io {
                path: path.clone(),
                source,
            })?;
            let child = format!("{relative}/{}", child.file_name().to_string_lossy());
            walk(root, &child, exclude, digests)?;
        }
    }
    Ok(())
}

/// Whether `name` matches the shell pattern `pattern`, where `*` matches any run
/// of characters and `?` exactly one; every other character matches itself.
#[must_use]
pub fn glob_match(pattern: &str, name: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let name: Vec<char> = name.chars().collect();
    let (mut p, mut n) = (0, 0);
    let mut backtrack: Option<(usize, usize)> = None;
    while n < name.len() {
        match pattern.get(p) {
            Some('*') => {
                backtrack = Some((p, n));
                p += 1;
            },
            Some('?') => {
                p += 1;
                n += 1;
            },
            Some(&c) if c == name[n] => {
                p += 1;
                n += 1;
            },
            _ => match backtrack {
                Some((star, matched)) => {
                    p = star + 1;
                    n = matched + 1;
                    backtrack = Some((star, matched + 1));
                },
                None => return false,
            },
        }
    }
    pattern[p..].iter().all(|&c| c == '*')
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    /// SHA-256 of `b"w"`, for the fixture files below.
    const W: &str = "50e721e49c013f00c62cf59f2163542a9d8df02464efeb615d31051b0fddc326";

    #[test]
    fn digests_match_known_values() -> TestResult {
        let dir = tempfile::tempdir()?;
        let empty = dir.path().join("empty");
        fs::write(&empty, "")?;
        assert_eq!(
            sha256_file(&empty)?,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let w = dir.path().join("w");
        fs::write(&w, "w")?;
        assert_eq!(sha256_file(&w)?, W);
        Ok(())
    }

    #[test]
    fn patterns_match_like_the_shell() {
        assert!(glob_match("checkpoint-*", "checkpoint-5"));
        assert!(glob_match("checkpoint-*", "checkpoint-"));
        assert!(!glob_match("checkpoint-*", "my-checkpoint-5"));
        assert!(glob_match("*.bin", "a.b.bin"));
        assert!(glob_match("a?c", "abc"));
        assert!(!glob_match("a?c", "ac"));
        assert!(glob_match("*", ""));
        assert!(!glob_match("", "a"));
    }

    #[test]
    fn manifest_records_survive_any_name_once_nul_terminated() -> TestResult {
        let output = format!("{W} output/b.bin\0{W} output/odd\\name\"'*?\n.bin\0");
        assert_eq!(
            parse_manifest(&output)?,
            vec![
                FileDigest {
                    path: "output/b.bin".into(),
                    sha256: W.into()
                },
                FileDigest {
                    path: "output/odd\\name\"'*?\n.bin".into(),
                    sha256: W.into()
                },
            ]
        );
        assert!(parse_manifest("nothex a\0").is_err());
        assert!(parse_manifest(&format!("{W}noSpace\0")).is_err());
        Ok(())
    }

    #[test]
    fn a_manifest_cut_short_of_its_final_nul_is_rejected() -> TestResult {
        assert_eq!(parse_manifest("")?, Vec::new());
        let cut_short = format!("{W} output/a.bin\0{W} output/b");
        assert!(parse_manifest(&cut_short).is_err());
        Ok(())
    }

    #[test]
    fn absolute_paths_and_dot_dot_components_are_rejected() {
        assert!(parse_manifest(&format!("{W} /etc/passwd\0")).is_err());
        assert!(parse_manifest(&format!("{W} a/../b\0")).is_err());
        assert!(parse_manifest(&format!("{W} ..\0")).is_err());
        assert!(parse_manifest(&format!("{W} ../output/a.bin\0")).is_err());
        // A line break inside a name stays legal: only `/`-separated components
        // are checked, not the raw bytes of the name.
        assert!(parse_manifest(&format!("{W} a\nb\0")).is_ok());
    }

    #[test]
    fn the_missing_directory_message_is_debug_quoted() -> TestResult {
        let root = tempfile::tempdir()?;
        // A name a naive `{}` format would print raw, hiding the control character.
        let gone = root.path().join("go\ne");
        let error = local_manifest(&gone, &entries(), &[])
            .err()
            .ok_or("expected an error")?;
        let display = gone.display().to_string();
        assert_eq!(
            error.to_string(),
            format!("manifest failed: {display:?} does not exist")
        );
        Ok(())
    }

    /// A run directory with an adapter, an excluded checkpoint, a nested
    /// excluded checkpoint, a log, a symbolic link, and names that would trip a
    /// naive line- or escape-based parser: a line break, a backslash, a quote and
    /// two consecutive spaces.
    fn fixture() -> Result<tempfile::TempDir, Box<dyn std::error::Error>> {
        let root = tempfile::tempdir()?;
        let dir = root.path();
        fs::create_dir_all(dir.join("output/checkpoint-5"))?;
        fs::create_dir_all(dir.join("output/nested/checkpoint-9"))?;
        fs::write(dir.join("output/adapter.bin"), "w")?;
        fs::write(dir.join("output/nested/config.json"), "w")?;
        fs::write(dir.join("output/checkpoint-5/state"), "w")?;
        fs::write(dir.join("output/nested/checkpoint-9/state"), "w")?;
        fs::write(dir.join("job.log"), "w")?;
        fs::write(dir.join("output/weird\nname.bin"), "w")?;
        fs::write(dir.join("output/back\\slash.bin"), "w")?;
        fs::write(dir.join("output/quote\"mark.bin"), "w")?;
        fs::write(dir.join("output/two  spaces.bin"), "w")?;
        std::os::unix::fs::symlink("adapter.bin", dir.join("output/link.bin"))?;
        Ok(root)
    }

    fn expected() -> Vec<FileDigest> {
        let mut digests: Vec<FileDigest> = [
            "job.log",
            "output/adapter.bin",
            "output/nested/config.json",
            "output/weird\nname.bin",
            "output/back\\slash.bin",
            "output/quote\"mark.bin",
            "output/two  spaces.bin",
        ]
        .into_iter()
        .map(|path| FileDigest {
            path: path.to_string(),
            sha256: W.to_string(),
        })
        .collect();
        digests.sort();
        digests
    }

    fn entries() -> Vec<String> {
        vec!["output".into(), "missing".into(), "job.log".into()]
    }

    #[test]
    fn the_local_manifest_skips_excluded_names_links_and_missing_entries() -> TestResult {
        let root = fixture()?;
        let manifest = local_manifest(root.path(), &entries(), &["checkpoint-*".into()])?;
        assert_eq!(manifest, expected());
        let gone = local_manifest(&root.path().join("gone"), &entries(), &[]);
        assert!(matches!(
            gone,
            Err(ExecError::Command {
                action: "manifest",
                ..
            })
        ));
        Ok(())
    }

    /// Shells the script is run under: `sh`, and `dash` and `busybox sh` when
    /// installed.
    fn shells() -> Vec<Vec<&'static str>> {
        [vec!["sh"], vec!["dash"], vec!["busybox", "sh"]]
            .into_iter()
            .filter(|shell| {
                let available = Command::new(shell[0])
                    .args(&shell[1..])
                    .args(["-c", ":"])
                    .status()
                    .is_ok_and(|status| status.success());
                if !available {
                    eprintln!("skipped: {} is not installed", shell.join(" "));
                }
                available
            })
            .collect()
    }

    #[test]
    fn the_script_agrees_with_the_local_manifest_under_every_shell() -> TestResult {
        let root = fixture()?;
        let remote = root.path().to_string_lossy().into_owned();
        let script = manifest_script(&remote, &entries(), &["checkpoint-*".into()]);
        for shell in shells() {
            let output = Command::new(shell[0])
                .args(&shell[1..])
                .arg("-c")
                .arg(&script)
                .output()?;
            assert!(
                output.status.success(),
                "{shell:?}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let manifest = parse_manifest(&String::from_utf8(output.stdout)?)?;
            assert_eq!(manifest, expected(), "{shell:?}");
        }
        let missing = manifest_script(&format!("{remote}/gone"), &entries(), &[]);
        let output = Command::new("sh").arg("-c").arg(&missing).output()?;
        assert_eq!(output.status.code(), Some(1));
        assert_eq!(
            String::from_utf8(output.stderr)?,
            format!("{remote}/gone does not exist\n")
        );
        let nothing = manifest_script(&remote, &["missing".into()], &[]);
        let output = Command::new("sh").arg("-c").arg(&nothing).output()?;
        assert!(output.status.success() && output.stdout.is_empty());
        Ok(())
    }
}
