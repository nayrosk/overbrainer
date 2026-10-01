//! The run's pod host key as a Runpod secret: the create call's `env` only
//! refers to it (`{{ RUNPOD_SECRET_<name> }}`), so the pod metadata Runpod
//! returns never holds the key. The secret lives as long as a pod of the run
//! may: a restarted container runs its bootstrap again and needs the key.

use secrecy::SecretString;

use super::client::SECRETS_FORBIDDEN_MESSAGE;
use super::flow::forget_client_key;
use super::{ApiError, NewSecret, PodCtx, PodError, PodRecord, PodState, RunpodClient};
use crate::runs::{Runs, is_valid_run_id};

/// Prefix of the name of every run's host key secret.
pub const HOST_KEY_SECRET_PREFIX: &str = "overbrainer_host_key_";

/// The name of the host key secret of the run `run_id`. A run ID is ASCII
/// letters, digits, `_` and `-`, at most 64 characters, so the name meets
/// Runpod's rules (a letter first, at most 191 characters).
#[must_use]
pub fn host_key_secret(run_id: &str) -> String {
    format!("{HOST_KEY_SECRET_PREFIX}{run_id}")
}

/// What the pod's `env` holds instead of the host key: Runpod replaces it with
/// the secret's value when the pod boots.
#[must_use]
pub fn host_key_placeholder(run_id: &str) -> String {
    format!("{{{{ RUNPOD_SECRET_{} }}}}", host_key_secret(run_id))
}

/// The run of the host key secret `name`, or `None` when `name` is not one.
#[must_use]
pub fn host_key_run(name: &str) -> Option<&str> {
    name.strip_prefix(HOST_KEY_SECRET_PREFIX)
        .filter(|run_id| is_valid_run_id(run_id))
}

/// Stores `value`, the base64 of the pod's private host key, as the host key
/// secret of the run `run_id`: created, or, when the name is taken (a create
/// retried after an answer that got lost), found by name and given the value.
/// From then on the client redacts the value too.
///
/// # Errors
///
/// Returns [`PodError::HostKeySecret`] with the client's fixed message when
/// Runpod refuses or fails: it never holds anything Runpod said.
pub async fn store_host_key(
    client: &RunpodClient,
    run_id: &str,
    value: &SecretString,
) -> Result<(), PodError> {
    let name = host_key_secret(run_id);
    let secret = NewSecret {
        name: name.clone(),
        value: value.clone(),
        description: format!("SSH host key of the pod of overbrainer run {run_id}"),
    };
    client.redact_also(value.clone());
    let stored = match client.create_secret(&secret).await {
        Ok(_) => Ok(()),
        Err(error) if error.status() == Some(409) => replace(client, &name, value).await,
        Err(error) => Err(error),
    };
    stored.map_err(|error| PodError::HostKeySecret(secret_error(&error)))
}

/// Gives the existing secret `name` the value `value`.
async fn replace(client: &RunpodClient, name: &str, value: &SecretString) -> Result<(), ApiError> {
    let found = client.list_secrets(Some(name)).await?;
    let Some(secret) = found.iter().find(|secret| secret.name == name) else {
        return Err(ApiError::InvalidResponse(format!(
            "Runpod says the secret {name} exists but does not list it"
        )));
    };
    client.update_secret_value(&secret.id, value).await?;
    Ok(())
}

/// What a failed store says: the 403 message alone names the permission; any
/// other error keeps the client's text, which never quotes a secret write's
/// answer.
fn secret_error(error: &ApiError) -> String {
    match error.status() {
        Some(403) => SECRETS_FORBIDDEN_MESSAGE.to_string(),
        _ => error.to_string(),
    }
}

/// Deletes the host key secret of the run `run_id`, found by name; nothing
/// when there is none.
///
/// # Errors
///
/// Returns an [`ApiError`] when the secrets cannot be listed or one cannot be
/// deleted.
pub async fn drop_host_key(client: &RunpodClient, run_id: &str) -> Result<(), ApiError> {
    let name = host_key_secret(run_id);
    for secret in client.list_secrets(Some(&name)).await? {
        // Runpod matches the name without regard to case: only ours goes.
        if secret.name == name {
            client.delete_secret(&secret.id).await?;
        }
    }
    Ok(())
}

/// [`drop_host_key`], a failure only warned about with how to finish it.
pub async fn drop_host_key_or_warn(client: &RunpodClient, run_id: &str) {
    if let Err(error) = drop_host_key(client, run_id).await {
        tracing::warn!(
            "cannot delete the Runpod secret {} ({error}); `overbrainer pod rm {run_id}` tries again",
            host_key_secret(run_id)
        );
    }
}

/// Removes the private client key of the run `run_id` and deletes its host key
/// secret, best effort, once its `pod.json` shows no pod of it that may still
/// exist (no pod, or one confirmed deleted, and no stray): a stray pod
/// restarted later boots with the same secret. Callers save `pod.json` first, strays included. Keys kept here
/// go once the strays are deleted: by `overbrainer pod rm`, or by the sweep of
/// the secrets of ended runs at the next start.
pub async fn forget_keys(ctx: &PodCtx<'_>, run_id: &str) {
    if !no_pod_left(ctx.runs, run_id) {
        tracing::debug!("keeping the keys of run {run_id}: a pod of it may still exist");
        return;
    }
    forget_client_key(ctx.runs, run_id);
    drop_host_key_or_warn(ctx.client, run_id).await;
}

/// Whether `pod.json` shows no pod of the run that may still exist: no pod, or
/// one confirmed deleted, and no stray. A `pod.json` that cannot be read shows
/// nothing for sure; none at all means no pod was ever asked for.
fn no_pod_left(runs: &Runs, run_id: &str) -> bool {
    match PodRecord::load(runs, run_id) {
        Ok(Some(record)) => {
            record.stray_pods.is_empty()
                && (record.pod_id.is_none() || record.state == PodState::Deleted)
        },
        Ok(None) => true,
        Err(error) => {
            tracing::warn!("cannot read the pod record of run {run_id}: {error}");
            false
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_secret_name_and_its_reference_follow_the_run() {
        assert_eq!(
            host_key_secret("20260922-143005-a1b2"),
            "overbrainer_host_key_20260922-143005-a1b2"
        );
        assert_eq!(
            host_key_placeholder("r1"),
            "{{ RUNPOD_SECRET_overbrainer_host_key_r1 }}"
        );
        assert_eq!(host_key_run("overbrainer_host_key_r1"), Some("r1"));
        for other in [
            "hf-token",
            "overbrainer_host_key_",
            "overbrainer_host_key_../x",
            "overbrainer_host_key_a/b",
        ] {
            assert_eq!(host_key_run(other), None, "{other}");
        }
    }

    #[test]
    fn every_name_meets_the_runpod_rules() {
        let longest = "a".repeat(crate::runs::RUN_ID_MAX);
        let name = host_key_secret(&longest);
        assert!(name.len() <= 191);
        assert!(name.starts_with(|c: char| c.is_ascii_alphabetic()));
        assert!(!name.to_ascii_uppercase().starts_with("RUNPOD"));
    }
}
