# overbrainer pod bootstrap: the pod's cmd, run by bash after the image's
# entrypoint. It installs the run's host key and authorized key, writes the job's
# environment file and the watchdog, starts sshd, then becomes the watchdog.
#
# The run directory and the watchdog file are created FIRST, before anything else
# is touched: if a later step fails, `fail` can still exec into the watchdog it
# already wrote, so the pod always ends up guarded, even one that never reaches
# sshd. A handful of the paths bootstrap_main writes to are overridable through
# OVERBRAINER_* variables, defaulting to the real pod's paths, so tests can point
# them at a temporary directory instead of the real /etc and /root.
#
# Only functions are defined here, so tests can source this file and call them on
# temporary directories; overbrainer appends write_watchdog and the call to
# bootstrap_main when it builds the pod's command. POSIX sh.

log() { printf '%s bootstrap: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

# The watchdog file's path: OVERBRAINER_WATCHDOG_FILE when set (tests), the real
# pod's path otherwise. Shared by bootstrap_main and fail so both agree on it.
watchdog_path() {
  printf '%s' "${OVERBRAINER_WATCHDOG_FILE:-/etc/overbrainer/watchdog.sh}"
}

# fail REASON: logs REASON, records it in .pod/bootstrap_failed (plain text, never
# a secret: every caller passes a fixed literal), then, if the watchdog file has
# already been written, execs into it with OVERBRAINER_BOOT_FAILED=1 so the pod
# still gets a proof attempt, a verdict and, unless kept, gets deleted. Only when
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
  log "run ${OVERBRAINER_RUN_ID:-unknown}"
  watchdog_file=$(watchdog_path)
  etc_ssh_dir=${OVERBRAINER_ETC_SSH_DIR:-/etc/ssh}
  authorized_keys_dir=${OVERBRAINER_AUTHORIZED_KEYS_DIR:-/root/.ssh}
  job_env_file=${OVERBRAINER_JOB_ENV_FILE:-/etc/overbrainer/job.env}
  cuda_env_script=${OVERBRAINER_CUDA_ENV_SCRIPT:-/workspace/axolotl/scripts/cuda13_env.sh}
  mkdir -p "$OVERBRAINER_RUN_DIR/.pod" || fail "cannot create the run directory"
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
