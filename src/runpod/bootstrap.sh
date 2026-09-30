# overbrainer pod bootstrap: the pod's cmd, run by bash after the image's
# entrypoint. It installs the run's host key and authorized key, writes the job's
# environment file and the watchdog, starts sshd, then becomes the watchdog.
#
# The run directory is claimed FIRST, before anything is written in it: on a
# network volume, another checkout of the project may have started a run with
# the same ID, whose files must stay untouched. Then the watchdog file is
# written, before anything else: if a later step fails, `fail` can still exec
# into the watchdog it already wrote, so the pod always ends up guarded, even
# one that never reaches sshd. A handful of the paths bootstrap_main writes to are overridable through
# OVERBRAINER_* variables, defaulting to the real pod's paths, so tests can point
# them at a temporary directory instead of the real /etc and /root.
#
# Only functions are defined here, so tests can source this file and call them on
# temporary directories; overbrainer appends write_watchdog and the call to
# bootstrap_main when it builds the pod's command. POSIX sh.

# log MESSAGE: on stdout, which Runpod keeps in the pod's logs, and in
# bootstrap_log_dir/bootstrap.log once bootstrap_main set it: the run's .pod/
# once the run directory is claimed (overbrainer copies it with the results),
# or the pod's own directory for a refused run. Never into a directory this pod
# did not claim.
log() {
  line="$(date -u +%Y-%m-%dT%H:%M:%SZ) bootstrap: $*"
  printf '%s\n' "$line"
  if [ -n "${bootstrap_log_dir:-}" ]; then
    printf '%s\n' "$line" >> "$bootstrap_log_dir/bootstrap.log" 2>/dev/null || true
  fi
}

# The watchdog file's path: OVERBRAINER_WATCHDOG_FILE when set (tests), the real
# pod's path otherwise. Shared by bootstrap_main and fail so both agree on it.
watchdog_path() {
  printf '%s' "${OVERBRAINER_WATCHDOG_FILE:-/etc/overbrainer/watchdog.sh}"
}

# fail REASON: logs REASON, records it in .pod/bootstrap_failed (plain text, never
# a secret: every caller passes a fixed literal), then, if the watchdog file has
# already been written, execs into it with OVERBRAINER_BOOT_FAILED=1 so the pod
# still gets a proof attempt, a verdict and gets deleted, kept or not. Only when
# the watchdog itself could not be written does this exit without a guard, as
# before: nothing is left that could run it.
fail() {
  log "failed: $*"
  if [ -n "${OVERBRAINER_RUN_DIR:-}" ]; then
    { mkdir -p "$OVERBRAINER_RUN_DIR/.pod" &&
        printf '%s\n' "$*" > "$OVERBRAINER_RUN_DIR/.pod/bootstrap_failed"; } 2>/dev/null || true
  fi
  watchdog_file=$(watchdog_path)
  if [ -f "$watchdog_file" ]; then
    OVERBRAINER_BOOT_FAILED=1 exec bash "$watchdog_file"
  fi
  exit 1
}

# claim_run_dir DIR OWNER: creates DIR and its parents, then DIR/.claim holding
# OWNER with `set -C` (an O_EXCL open), as overbrainer's own claim does. Returns
# 0 when it made the marker or the marker already holds OWNER (an earlier pod of
# the same run), 2 when it holds anything else (another run owns DIR), 1 when
# neither can be made.
claim_run_dir() {
  mkdir -p "$1" || return 1
  if (set -C; printf '%s\n' "$2" > "$1/.claim") 2>/dev/null; then return 0; fi
  [ -f "$1/.claim" ] || return 1
  [ "$(cat "$1/.claim")" = "$2" ] || return 2
}

# install_host_key DIR: replaces every host key in DIR with the run's ed25519 key
# (OVERBRAINER_HOST_KEY, the base64 of its OpenSSH private key file).
install_host_key() {
  mkdir -p "$1" || return 1
  rm -f "$1"/ssh_host_* || return 1
  printf '%s' "$OVERBRAINER_HOST_KEY" | base64 -d > "$1/ssh_host_ed25519_key" || return 1
  chmod 600 "$1/ssh_host_ed25519_key" || return 1
  ssh-keygen -y -f "$1/ssh_host_ed25519_key" > "$1/ssh_host_ed25519_key.pub" || return 1
}

# install_authorized_key DIR: authorizes the run's client key
# (OVERBRAINER_AUTHORIZED_KEY) and nothing else.
install_authorized_key() {
  mkdir -p "$1" || return 1
  chmod 700 "$1" || return 1
  printf '%s\n' "$OVERBRAINER_AUTHORIZED_KEY" > "$1/authorized_keys" || return 1
  chmod 600 "$1/authorized_keys" || return 1
}

# export_line NAME VALUE: an `export` line that a POSIX shell reads back verbatim.
export_line() {
  printf "export %s='%s'\n" "$1" "$(printf '%s' "$2" | sed "s/'/'\\\\''/g")"
}

# write_job_env FILE CUDA_SCRIPT: the environment every job of the pod starts with,
# since an SSH session lacks the image's Docker ENV: PATH, the CUDA libraries the
# image's entrypoint adds (CUDA_SCRIPT, sourced when present) and HF_HOME.
# Nothing else: never RUNPOD_API_KEY.
write_job_env() {
  if [ -f "$2" ]; then
    case $- in *u*) nounset=1 ;; *) nounset=0 ;; esac
    set +u
    . "$2"
    if [ "$nounset" = 1 ]; then set -u; fi
  fi
  mkdir -p "$(dirname "$1")" || return 1
  {
    export_line PATH "$PATH"
    if [ -n "${LD_LIBRARY_PATH:-}" ]; then export_line LD_LIBRARY_PATH "$LD_LIBRARY_PATH"; fi
    export_line HF_HOME "$OVERBRAINER_WORKDIR/.hf-cache"
  } > "$1.tmp" || return 1
  mv -f "$1.tmp" "$1" || return 1
}

bootstrap_main() {
  set -eu
  umask 077
  bootstrap_log_dir=
  watchdog_file=$(watchdog_path)
  etc_ssh_dir=${OVERBRAINER_ETC_SSH_DIR:-/etc/ssh}
  authorized_keys_dir=${OVERBRAINER_AUTHORIZED_KEYS_DIR:-/root/.ssh}
  job_env_file=${OVERBRAINER_JOB_ENV_FILE:-/etc/overbrainer/job.env}
  cuda_env_script=${OVERBRAINER_CUDA_ENV_SCRIPT:-/workspace/axolotl/scripts/cuda13_env.sh}
  claimed=0
  claim_run_dir "$OVERBRAINER_RUN_DIR" "${OVERBRAINER_CLAIM:?OVERBRAINER_CLAIM is not set}" ||
    claimed=$?
  if [ "$claimed" = 2 ]; then
    # Another run owns the directory: nothing is written there. The watchdog
    # and the failure go to a directory of this pod's own, and the watchdog
    # deletes the pod.
    OVERBRAINER_RUN_DIR=${OVERBRAINER_REFUSED_RUN_DIR:-/root/overbrainer-refused-run}
    export OVERBRAINER_RUN_DIR
    mkdir -p "$OVERBRAINER_RUN_DIR/.pod" || fail "cannot create the run directory"
    bootstrap_log_dir="$OVERBRAINER_RUN_DIR/.pod"
    write_watchdog "$watchdog_file" || fail "cannot write the watchdog"
    fail "the run directory belongs to another run"
  fi
  [ "$claimed" = 0 ] || fail "cannot create the run directory"
  mkdir -p "$OVERBRAINER_RUN_DIR/.pod" || fail "cannot create the run directory"
  bootstrap_log_dir="$OVERBRAINER_RUN_DIR/.pod"
  log "run ${OVERBRAINER_RUN_ID:-unknown}"
  # A verdict left by an earlier pod (a network volume outlives it) must never be
  # read as this pod's: it goes before sshd can serve it.
  rm -f "$OVERBRAINER_RUN_DIR/.pod/watchdog" || fail "cannot remove a stale verdict"
  write_watchdog "$watchdog_file" || fail "cannot write the watchdog"
  install_host_key "$etc_ssh_dir" || fail "cannot install the host key"
  install_authorized_key "$authorized_keys_dir" || fail "cannot install the authorized key"
  write_job_env "$job_env_file" "$cuda_env_script" ||
    fail "cannot write the job environment"
  unset OVERBRAINER_HOST_KEY OVERBRAINER_AUTHORIZED_KEY
  service ssh restart || fail "cannot start sshd"
  exec bash "$watchdog_file"
}
