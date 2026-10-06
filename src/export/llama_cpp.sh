# The llama.cpp helpers export.sh and compare.sh share. overbrainer writes
# this code at the top of each of them, as one script: nothing else is
# uploaded. Every message starts with the script's name without `.sh`
# (`export` or `compare`).
set -eu

log_prefix=${0##*/}
log_prefix=${log_prefix%.sh}

# Prints `<prefix>: $*` on stderr and ends the script.
fail() {
    printf '%s: %s\n' "$log_prefix" "$*" >&2
    exit 1
}

# Prints `<prefix>: $*` on stdout.
say() {
    printf '%s: %s\n' "$log_prefix" "$*"
}

# Downloads $1 to $3 and checks its SHA-256 against $2 before anything uses it.
fetch() {
    python3 - "$1" "$2" "$3" "$log_prefix" <<'PY'
import hashlib
import os
import sys
import urllib.request

url, expected, path, prefix = sys.argv[1:5]
digest = hashlib.sha256()
with urllib.request.urlopen(url, timeout=120) as answer, open(path, "wb") as file:
    while True:
        chunk = answer.read(1 << 20)
        if not chunk:
            break
        digest.update(chunk)
        file.write(chunk)
if digest.hexdigest() != expected:
    os.remove(path)
    sys.exit(f"{prefix}: {url} has SHA-256 {digest.hexdigest()}, expected {expected}: not used")
PY
}

# Extracts the archive at $2, whose top directory is $3, into $1 through a
# temporary directory, so a reader never sees half of it.
unpack() {
    rm -rf "$1.tmp"
    mkdir -p "$1.tmp"
    tar -xzf "$2" -C "$1.tmp"
    [ -d "$1.tmp/$3" ] || fail "$2 has no $3 directory"
    rm -rf "$1"
    mv "$1.tmp/$3" "$1"
    rm -rf "$1.tmp" "$2"
}

