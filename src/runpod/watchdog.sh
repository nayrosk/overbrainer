# overbrainer pod watchdog. The bootstrap execs it as the process the container's
# init (docker-init) runs, so it lives exactly as long as the container. At startup
# it proves that the pod's own API key reaches the pod, and writes the verdict
# where the client reads it; then, every interval, it deletes the pod when a rule
# says so. Keep mode (--keep-pod) never deletes. The API key only ever reaches
# curl on its standard input: it is never in a command line, a file or a log.
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
API=${OVERBRAINER_API_URL:-https://api.runpod.io/v2}
AGENT="overbrainer-watchdog/${OVERBRAINER_VERSION:-unknown}"

now() { date +%s; }

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
# non-zero when every way failed; the next tick tries again.
delete_pod() {
  log "delete reason=$1"
  code=$(api DELETE "/pods/$RUNPOD_POD_ID")
  log "DELETE $code"
  case $code in 200 | 202 | 204 | 404) return 0 ;; esac
  code=$(api POST "/pods/$RUNPOD_POD_ID/action" '{"action":"terminate"}')
  log "terminate $code"
  case $code in 200 | 202 | 204) return 0 ;; esac
  code=$(api POST "/pods/$RUNPOD_POD_ID/action" '{"action":"stop"}')
  log "stop $code"
  case $code in 200 | 202 | 204) return 0 ;; esac
  if command -v runpodctl >/dev/null 2>&1; then
    if runpodctl pod delete "$RUNPOD_POD_ID" >/dev/null 2>&1; then
      log "runpodctl delete ok"
      return 0
    fi
    log "runpodctl delete failed"
  fi
  return 1
}

trap 'log "stopping"; exit 0' TERM INT

mkdir -p "$POD_DIR"
start=$(now)
log "start deadline=${DEADLINE:-none} boot_grace=$BOOT_GRACE retrieve_grace=$RETRIEVE_GRACE keep=$KEEP"
result=$(probe)
verdict "$result"
log "probe $result"

started=0
ended_at=
retrieved=0
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
  reason=
  if [ "$KEEP" != 1 ]; then
    if [ -n "$DEADLINE" ] && [ "$n" -ge "$DEADLINE" ]; then
      reason=deadline
    elif [ "$retrieved" = 1 ]; then
      reason=retrieved
    elif [ -n "$ended_at" ]; then
      if [ $((n - ended_at)) -ge "$RETRIEVE_GRACE" ]; then reason=abandoned; fi
    elif [ "$started" = 0 ] && [ $((n - start)) -ge "$BOOT_GRACE" ]; then
      reason=never_started
    fi
  fi
  if [ -n "$reason" ] && delete_pod "$reason"; then
    log "deleted"
    exit 0
  fi
  sleep "$INTERVAL"
done
