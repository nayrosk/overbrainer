//! The `pre_train` hook: a command of the user's a training job runs after the
//! GPU check and before its trainer, which stops the run when it exits with a
//! code the target does not allow or runs out of time.

use serde::Deserialize;

use crate::config::PreTrain;
use crate::exec::{Hook, Preflight};

/// The file that gets the hook's output, in the run directory.
pub const PRE_TRAIN_LOG: &str = "pre_train.log";
/// The file the hook's exit code goes to, in the run directory.
pub const PRE_TRAIN_RESULT: &str = "pre_train.json";
/// The job's exit code when the hook exits with a code it does not allow.
pub const PRE_TRAIN_CODE: u8 = 91;
/// The job's exit code when the hook runs out of time.
pub const PRE_TRAIN_TIMEOUT_CODE: u8 = 92;

/// The exit code of a shell whose command was not found.
const NOT_FOUND: i32 = 127;

/// The preflight check running `pre_train` in a job whose run directory is
/// `root` as the job sees it. Codes outside 0 to 255, which validation
/// refuses, are left out.
#[must_use]
pub fn pre_train_preflight(root: &str, pre_train: &PreTrain) -> Preflight {
    Preflight::Hook(Hook {
        command: pre_train.command.clone(),
        ok_exit_codes: pre_train
            .ok_exit_codes
            .iter()
            .filter_map(|code| u8::try_from(*code).ok())
            .collect(),
        timeout_seconds: pre_train.timeout_seconds,
        log: format!("{root}/{PRE_TRAIN_LOG}"),
        result: format!("{root}/{PRE_TRAIN_RESULT}"),
        failed: PRE_TRAIN_CODE,
        timed_out: PRE_TRAIN_TIMEOUT_CODE,
    })
}

/// What the job wrote to [`PRE_TRAIN_RESULT`] once the hook ended.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct HookResult {
    /// The hook's exit code.
    pub code: i32,
    /// The codes the target allows.
    #[serde(default)]
    pub allowed: Vec<u8>,
    /// The hook's time limit, in seconds.
    #[serde(default)]
    pub timeout: u32,
    /// Whether `timeout` stopped the hook.
    #[serde(default)]
    pub timed_out: bool,
}

impl HookResult {
    /// Parses the hook's result, `None` when it is not one.
    #[must_use]
    pub fn parse(content: &[u8]) -> Option<Self> {
        serde_json::from_slice(content).ok()
    }

    /// Why the hook stopped the run, as the run's message ending with `log`
    /// (where the hook's log is), or `None` when its code is allowed.
    #[must_use]
    pub fn failure(&self, log: &str) -> Option<String> {
        if self.timed_out {
            return Some(format!(
                "the pre_train hook ran past {} s: raise `timeout_seconds` or check what it waits for ({log})",
                self.timeout
            ));
        }
        if u8::try_from(self.code).is_ok_and(|code| self.allowed.contains(&code)) {
            return None;
        }
        let allowed: Vec<String> = self.allowed.iter().map(u8::to_string).collect();
        let hint = if self.code == NOT_FOUND {
            ": its program is not on the target's PATH; install it in the image or the venv"
        } else {
            ""
        };
        Some(format!(
            "the pre_train hook exited with code {} (allowed: {}){hint} ({log})",
            self.code,
            allowed.join(", ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The preflight carries the command, the codes, the limit and the files
    /// in the run directory.
    #[test]
    fn the_preflight_runs_the_command_with_its_codes() {
        let pre_train = PreTrain {
            command: vec!["computefence".into(), "doctor".into()],
            ok_exit_codes: vec![0, 1, 300],
            timeout_seconds: 30,
        };
        assert_eq!(
            pre_train_preflight("/w/r1", &pre_train),
            Preflight::Hook(Hook {
                command: vec!["computefence".into(), "doctor".into()],
                ok_exit_codes: vec![0, 1],
                timeout_seconds: 30,
                log: "/w/r1/pre_train.log".into(),
                result: "/w/r1/pre_train.json".into(),
                failed: 91,
                timed_out: 92,
            })
        );
    }

    /// A refused code, a missing program and a timeout each name their cause;
    /// an allowed code names none.
    #[test]
    fn each_failure_names_its_cause() -> Result<(), String> {
        let failure = |json: &str| -> Result<Option<String>, String> {
            Ok(HookResult::parse(json.as_bytes())
                .ok_or(json.to_string())?
                .failure("see runs/r1/pre_train.log"))
        };
        assert_eq!(
            failure(r#"{"code":2,"allowed":[0,1],"timeout":120,"timed_out":false}"#)?.as_deref(),
            Some(
                "the pre_train hook exited with code 2 (allowed: 0, 1) (see runs/r1/pre_train.log)"
            )
        );
        assert!(
            failure(r#"{"code":127,"allowed":[0],"timeout":120,"timed_out":false}"#)?
                .is_some_and(|message| message.contains("not on the target's PATH"))
        );
        assert_eq!(
            failure(r#"{"code":124,"allowed":[0],"timeout":120,"timed_out":true}"#)?.as_deref(),
            Some(
                "the pre_train hook ran past 120 s: raise `timeout_seconds` or check what it waits for (see runs/r1/pre_train.log)"
            )
        );
        assert_eq!(
            failure(r#"{"code":1,"allowed":[0,1],"timeout":120,"timed_out":false}"#)?,
            None
        );
        assert_eq!(HookResult::parse(b"{}"), None);
        Ok(())
    }
}
