# overbrainer pod watchdog. The bootstrap execs it as the process the container's
# init (docker-init) runs, so it lives exactly as long as the container. At startup
# it proves that the pod's own API key reaches the pod, and writes the verdict
# where the client reads it; then, every interval, it deletes the pod when a rule
# says so. Keep mode (--keep-pod) holds only once the job exists (job.pid): from
# then on nothing deletes the pod, but before that a kept pod is as unguarded as
# any other, so every rule applies to it, the boot grace included. The API key
# only ever reaches curl on its standard input: it is never in a command line, a
# file or a log.
#
# When the bootstrap itself failed (OVERBRAINER_BOOT_FAILED=1, set by bootstrap.sh's
# `fail` once the watchdog file exists), this still runs: the pod's own key is
# already installed by then, so the proof can still be attempted, but the verdict
# always reports the bootstrap failure and the pod is deleted at once, kept or
# not: no job will ever start on it. This is what stops a pod that failed early,
# before sshd even started, from sitting unguarded and billing for nothing.
#
# The deadline (max_hours) is held off while a client following the job keeps its
# lease fresh: overbrainer touches .pod/lease every few minutes while the job makes
# progress, and the deadline applies again once the lease is LEASE_TTL old.
#
# Before the pod is lost, the job is asked for a snapshot (snapshot.request, which
# the trainer's plugin reads: it saves a checkpoint and stops): SNAPSHOT_LEAD
# before the deadline while the lease is stale, and at .pod/snapshot_at, the time
# the client wrote for 95% of max_cost_usd. A request the watchdog made holds the
# deadline off for up to SNAPSHOT_WAIT; once a snapshot ended the job, the
# deadline gives way to the retrieve grace, capped at the deadline plus
# RETRIEVE_GRACE, so `train attach` can still collect it. At .pod/cost_cap_at
# (100% of max_cost_usd) the pod is deleted, whatever the lease.
#
# POSIX sh: it runs under bash on the pod and is tested under sh, dash and busybox.
# group_signal comes from overbrainer's job scripts and is prepended to this file.

RUN_DIR=${OVERBRAINER_RUN_DIR:?OVERBRAINER_RUN_DIR is not set}
POD_DIR=$RUN_DIR/.pod
INTERVAL=${OVERBRAINER_INTERVAL:-60}
PROBE_TRIES=${OVERBRAINER_PROBE_TRIES:-5}
PROBE_WAIT=${OVERBRAINER_PROBE_WAIT:-10}
BOOT_GRACE=${OVERBRAINER_BOOT_GRACE:-1800}
RETRIEVE_GRACE=${OVERBRAINER_RETRIEVE_GRACE:-3600}
KEEP=${OVERBRAINER_KEEP_POD:-0}
DEADLINE=${OVERBRAINER_DEADLINE:-}
LEASE_TTL=${OVERBRAINER_LEASE_TTL:-900}
SNAPSHOT_LEAD=${OVERBRAINER_SNAPSHOT_LEAD:-900}
SNAPSHOT_WAIT=${OVERBRAINER_SNAPSHOT_WAIT:-900}
BOOT_FAILED=${OVERBRAINER_BOOT_FAILED:-0}
API=${OVERBRAINER_API_URL:-https://api.runpod.io/v2}
AGENT="overbrainer-watchdog/${OVERBRAINER_VERSION:-unknown}"

now() { date +%s; }

# Succeeds while the client following the job renewed its lease (touched
# .pod/lease, on the pod's own clock) less than LEASE_TTL seconds before $1.
lease_fresh() {
  touched=$(stat -c %Y "$POD_DIR/lease" 2>/dev/null) || return 1
  [ $(($1 - touched)) -lt "$LEASE_TTL" ]
}

# Prints the Unix time the client wrote in .pod/$1, nothing when there is none.
pod_time() {
  value=$(cat "$POD_DIR/$1" 2>/dev/null)
  case $value in '' | *[!0-9]*) value= ;; esac
  printf '%s' "$value"
}

# Asks the job for a snapshot for reason $1, unless one was asked for already.
request_snapshot() {
  [ -f "$RUN_DIR/snapshot.request" ] && return 0
  printf '%s' "$1" > "$RUN_DIR/snapshot.request.tmp" &&
    mv -f "$RUN_DIR/snapshot.request.tmp" "$RUN_DIR/snapshot.request" || return 1
  requested_at=$n
  log "snapshot requested reason=$1"
}

# Whether the deadline deletes the pod at $1, once passed with the lease stale:
# not while a snapshot the watchdog asked for is being taken (SNAPSHOT_WAIT at
# most), and, once a snapshot ended the job, not before DEADLINE + RETRIEVE_GRACE.
deadline_due() {
  if [ -n "$ended_at" ] && [ -f "$RUN_DIR/snapshot.json" ]; then
    [ "$1" -ge $((DEADLINE + RETRIEVE_GRACE)) ]
    return
  fi
  if [ -n "$requested_at" ] && [ -z "$ended_at" ] && [ $(($1 - requested_at)) -lt "$SNAPSHOT_WAIT" ]; then
    return 1
  fi
  return 0
}

log() {
  line="$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"
  printf '%s\n' "$line"
  printf '%s\n' "$line" >> "$POD_DIR/watchdog.log" 2>/dev/null
}

# api METHOD PATH [BODY]: prints the HTTP status of the call, 000 without answer.
api() {
  if [ -n "${3:-}" ]; then
    code=$(printf 'Authorization: Bearer %s\n' "${RUNPOD_API_KEY:-}" |
      curl -s -o /dev/null -w '%{http_code}' --max-time 10 -A "$AGENT" -H @- \
        -H 'Content-Type: application/json' -X "$1" --data "$3" "$API$2" 2>/dev/null)
  else
    code=$(printf 'Authorization: Bearer %s\n' "${RUNPOD_API_KEY:-}" |
      curl -s -o /dev/null -w '%{http_code}' --max-time 10 -A "$AGENT" -H @- \
        -X "$1" "$API$2" 2>/dev/null)
  fi
  printf '%s' "${code:-000}"
}

# Prints `ready` when the pod's key can read the pod, `failed <reason>` otherwise.
probe() {
  if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${RUNPOD_POD_ID:-}" ]; then
    echo "failed missing_key"
    return
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "failed no_curl"
    return
  fi
  tries=0
  while :; do
    code=$(api GET "/pods/$RUNPOD_POD_ID")
    case $code in
      200) echo ready; return ;;
      000 | 408 | 429 | 5??) ;;
      *) echo "failed http_$code"; return ;;
    esac
    tries=$((tries + 1))
    if [ "$tries" -ge "$PROBE_TRIES" ]; then
      if [ "$code" = 000 ]; then echo "failed unreachable"; else echo "failed http_$code"; fi
      return
    fi
    sleep "$PROBE_WAIT"
  done
}

verdict() {
  printf '%s\n' "$1" > "$POD_DIR/watchdog.tmp" && mv -f "$POD_DIR/watchdog.tmp" "$POD_DIR/watchdog"
}

# Whether the job has ended: an exit code or a cancel marker, or a recorded
# process group that is gone (a job killed before writing its exit code).
#
# group_signal's `kill -0` reads as "gone" only on a signal failure: on the pod
# everything runs as root, so a live process group can never refuse the signal
# with EPERM, the one case where that would wrongly read as ended.
job_ended() {
  [ -f "$RUN_DIR/exit_code" ] && return 0
  [ -f "$RUN_DIR/cancelled" ] && return 0
  [ -f "$RUN_DIR/job.pid" ] || return 1
  pid=$(cat "$RUN_DIR/job.pid" 2>/dev/null)
  case $pid in '' | *[!0-9]*) return 1 ;; esac
  [ "$pid" -gt 1 ] || return 1
  if group_signal 0 "$pid"; then return 1; fi
  return 0
}

# Deletes the pod, falling back to terminate, then stop, then runpodctl. Returns
# non-zero when every way failed; the next tick tries again. Sets DELETE_METHOD to
# which one worked, since only `stop` leaves the pod's volume billed until the
# owner removes it: the caller logs that case differently.
delete_pod() {
  log "delete reason=$1"
  code=$(api DELETE "/pods/$RUNPOD_POD_ID")
  log "DELETE $code"
  case $code in 200 | 202 | 204 | 404) DELETE_METHOD=delete; return 0 ;; esac
  code=$(api POST "/pods/$RUNPOD_POD_ID/action" '{"action":"terminate"}')
  log "terminate $code"
  case $code in 200 | 202 | 204) DELETE_METHOD=terminate; return 0 ;; esac
  code=$(api POST "/pods/$RUNPOD_POD_ID/action" '{"action":"stop"}')
  log "stop $code"
  case $code in 200 | 202 | 204) DELETE_METHOD=stop; return 0 ;; esac
  if command -v runpodctl >/dev/null 2>&1; then
    if runpodctl pod delete "$RUNPOD_POD_ID" >/dev/null 2>&1; then
      log "runpodctl delete ok"
      DELETE_METHOD=runpodctl
      return 0
    fi
    log "runpodctl delete failed"
  fi
  return 1
}

trap 'log "stopping"; exit 0' TERM INT

mkdir -p "$POD_DIR"
start=$(now)
log "start deadline=${DEADLINE:-none} boot_grace=$BOOT_GRACE retrieve_grace=$RETRIEVE_GRACE keep=$KEEP boot_failed=$BOOT_FAILED"
result=$(probe)
if [ "$BOOT_FAILED" = 1 ]; then
  boot_reason=$(cat "$POD_DIR/bootstrap_failed" 2>/dev/null)
  result="failed bootstrap: ${boot_reason:-unknown}"
fi
verdict "$result"
log "probe $result"

started=0
ended_at=
retrieved=0
requested_at=
while :; do
  n=$(now)
  if [ "$started" = 0 ] && [ -f "$RUN_DIR/job.pid" ]; then
    started=1
    log "job started"
  fi
  if [ -z "$ended_at" ] && job_ended; then
    ended_at=$n
    log "job ended"
  fi
  if [ "$retrieved" = 0 ] && [ -f "$POD_DIR/retrieved" ]; then
    retrieved=1
    log "retrieved marker seen"
  fi
  if [ -n "$DEADLINE" ] && [ "$n" -ge "$DEADLINE" ] && [ -z "${held:-}" ] && lease_fresh "$n"; then
    held=1
    log "deadline passed, lease held by the client"
  fi
  snapshot_at=$(pod_time snapshot_at)
  cost_cap_at=$(pod_time cost_cap_at)
  if [ "$started" = 1 ] && [ -z "$ended_at" ] && [ "$KEEP" != 1 ] && [ -z "$requested_at" ]; then
    if [ -n "$snapshot_at" ] && [ "$n" -ge "$snapshot_at" ]; then
      request_snapshot cost
    elif [ -n "$DEADLINE" ] && [ "$n" -ge $((DEADLINE - SNAPSHOT_LEAD)) ] && ! lease_fresh "$n"; then
      request_snapshot deadline
    fi
  fi
  # Keep mode stays the cost cap, deadline, retrieved and abandoned rules only
  # once the job started; a failed bootstrap deletes the pod whatever the mode.
  reason=
  if [ "$BOOT_FAILED" = 1 ]; then
    reason=bootstrap_failed
  elif [ "$KEEP" = 1 ] && [ "$started" = 1 ]; then
    :
  elif [ -n "$cost_cap_at" ] && [ "$n" -ge "$cost_cap_at" ]; then
    reason=cost_cap
  elif [ -n "$DEADLINE" ] && [ "$n" -ge "$DEADLINE" ] && ! lease_fresh "$n" && deadline_due "$n"; then
    reason=deadline
  elif [ "$retrieved" = 1 ]; then
    reason=retrieved
  elif [ -n "$ended_at" ]; then
    if [ $((n - ended_at)) -ge "$RETRIEVE_GRACE" ]; then reason=abandoned; fi
  elif [ "$started" = 0 ] && [ $((n - start)) -ge "$BOOT_GRACE" ]; then
    reason=never_started
  fi
  if [ -n "$reason" ] && delete_pod "$reason"; then
    if [ "$DELETE_METHOD" = stop ]; then
      log "stopped (the volume keeps billing until overbrainer pod rm)"
    else
      log "deleted"
    fi
    exit 0
  fi
  sleep "$INTERVAL"
done
