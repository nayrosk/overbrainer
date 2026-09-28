//! `overbrainer run` reading its configuration again between its stages, when
//! `overbrainer.toml` or `.env` changed since it was last read.

use std::path::{Path, PathBuf};

use anyhow::Context as _;

use crate::config::{DotenvKeys, EnvSource, Settings, Stamp, reload, stamp};

/// The configuration files of a run, as last read.
#[derive(Debug)]
pub(crate) struct Reloader {
    /// The project directory.
    dir: PathBuf,
    /// The keys `.env` set at start.
    dotenv: DotenvKeys,
    /// The stamp of the files as last read.
    stamp: Stamp,
    /// The environment the settings are read with: the process one until a
    /// reload, then the one it merged.
    env: EnvSource,
}

impl Reloader {
    /// The files of the project in `project_dir`, about to be read with the
    /// process environment; `dotenv` are the keys `.env` set at start. The
    /// stamp is taken first, so a change during that read is not missed.
    pub(crate) fn new(project_dir: &Path, dotenv: DotenvKeys) -> Self {
        Self {
            dir: project_dir.to_path_buf(),
            dotenv,
            stamp: stamp(project_dir),
            env: EnvSource::Process,
        }
    }

    /// The environment the settings are read with now.
    pub(crate) fn env(&self) -> &EnvSource {
        &self.env
    }

    /// The settings read again when the files changed since they were last
    /// read, said on stderr; `None` when they did not.
    ///
    /// # Errors
    ///
    /// Returns why the changed files cannot be used: the run stops rather than
    /// go on with a configuration being changed.
    pub(crate) fn changed(&mut self) -> anyhow::Result<Option<Settings>> {
        let now = stamp(&self.dir);
        if now == self.stamp {
            return Ok(None);
        }
        let reloaded = reload(&self.dir, &self.dotenv)
            .context("the configuration changed and cannot be used; the run stops")?;
        self.stamp = now;
        self.env = reloaded.env;
        eprintln!("config reloaded");
        Ok(Some(reloaded.settings))
    }
}
