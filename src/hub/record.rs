//! `runs/<id>/hub/push.json`: what the last push of a run sent where.

use std::fs;
use std::path::Path;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

use crate::runs::write_atomic;

/// The directory of a run that holds what concerns the Hub.
pub const HUB_DIR: &str = "hub";
/// The record of the last push, in [`HUB_DIR`].
pub const PUSH_FILE: &str = "push.json";

/// The last push of a run.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct PushRecord {
    /// `NAMESPACE/NAME`.
    pub repo: String,
    /// The commit id.
    pub commit: String,
    /// The commit URL.
    pub url: String,
    /// The paths in the repo the commit wrote.
    pub files: Vec<String>,
    /// The repo's visibility after the push.
    pub private: bool,
    /// When it was pushed, RFC 3339 UTC.
    pub pushed: String,
}

impl PushRecord {
    /// Writes the record to `run_dir/hub/push.json`, atomically like `run.json`.
    ///
    /// # Errors
    ///
    /// Returns an error when the directory or the file cannot be written.
    pub fn save(&self, run_dir: &Path) -> anyhow::Result<()> {
        let dir = run_dir.join(HUB_DIR);
        fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
        let mut content = serde_json::to_vec_pretty(self)?;
        content.push(b'\n');
        write_atomic(&dir, PUSH_FILE, &content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_saved_record_reads_back() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let record = PushRecord {
            repo: "me/x".into(),
            commit: "1".into(),
            url: "https://hf.co/me/x/commit/1".into(),
            files: vec!["adapter_config.json".into(), "README.md".into()],
            private: true,
            pushed: "2026-10-04T12:00:00Z".into(),
        };
        record.save(dir.path())?;
        let text = fs::read_to_string(dir.path().join("hub/push.json"))?;
        assert_eq!(serde_json::from_str::<PushRecord>(&text)?, record);
        Ok(())
    }
}
