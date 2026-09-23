# overbrainer pod bootstrap: the pod's cmd, run by bash after the image's
# entrypoint. It installs the run's host key and authorized key, writes the job's
# environment file and the watchdog, starts sshd, then becomes the watchdog.
#
# Only functions are defined here, so tests can source this file and call them on
# temporary directories; overbrainer appends write_watchdog and the call to
# bootstrap_main when it builds the pod's command. POSIX sh.

log() { printf '%s bootstrap: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

fail() {
  log "failed: $*"
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
  install_host_key /etc/ssh || fail "cannot install the host key"
  install_authorized_key /root/.ssh || fail "cannot install the authorized key"
  write_job_env /etc/overbrainer/job.env /workspace/axolotl/scripts/cuda13_env.sh ||
    fail "cannot write the job environment"
  mkdir -p "$OVERBRAINER_RUN_DIR/.pod" || fail "cannot create the run directory"
  write_watchdog /etc/overbrainer/watchdog.sh || fail "cannot write the watchdog"
  unset OVERBRAINER_HOST_KEY OVERBRAINER_AUTHORIZED_KEY
  service ssh restart || fail "cannot start sshd"
  exec bash /etc/overbrainer/watchdog.sh
}
