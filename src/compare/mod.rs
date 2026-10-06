//! `overbrainer compare`: the child against the parent on the eval set.
//!
//! A compare is a job of a run, like an export, kept in
//! `runs/<run-id>/compares/<compare-id>/`. On the run's target it serves the
//! run's GGUF with `llama-server` and asks it every question; back here a
//! judge model compares each child answer with the parent's, and the report
//! is written beside them.

/// Directory of a run holding its compare jobs, one directory each.
pub const COMPARES_DIR: &str = "compares";

/// The project part of a compare's ID: `compare_YYYYMMDD-HHMMSS`.
pub const COMPARE_PREFIX: &str = "compare";
