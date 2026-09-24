#!/usr/bin/env bash
# Records the demo GIFs of docs/assets with VHS, entirely in containers.
#
# Usage: docs/assets/record.sh DEMO_PROJECT [TAPE...]
#
# DEMO_PROJECT is an overbrainer project with data/ and runs/ to show, for
# example one that went through `overbrainer run` and a training run. It is
# copied without .env, .env.*, runs/*/ssh/, runs/*/output/ and the Hugging
# Face cache, so no key reaches the recording. Pod IDs and hosts in
# runs/*/pod.json do show in the Training view and in `runs ls`: use a project
# whose pods are deleted, or edit them in the copy.
#
# Steps:
# 1. Build overbrainer in a Rust image on the same Debian release as the VHS
#    image (trixie), so the binary runs inside it.
# 2. Run each tape (default: every *.tape here) in the VHS image, with the
#    binary at /usr/local/bin/overbrainer and the demo copy at /demo.
# 3. Optimize each GIF losslessly with `gifsicle -O3`.
#
# Environment: ENGINE (default podman), CACHE_DIR (default
# ${XDG_CACHE_HOME:-$HOME/.cache}/overbrainer-demo).
#
# VHS is pinned to v0.11.0: v0.12.0 renders nothing, because it runs ffmpeg
# with a context it has already cancelled.
set -euo pipefail

RUST_IMAGE="docker.io/library/rust:1.98@sha256:a8a5f0a1e5fe7dfe1d352591e4a1c7dd2c08fd70475cae872cf3458ba0df0546"
VHS_IMAGE="ghcr.io/charmbracelet/vhs:v0.11.0@sha256:9d5fc3dc0c160b0fb1d2212baff07e6bdf3fa9438c504a3237484567302fcf93"
GIFSICLE_IMAGE="docker.io/library/alpine:3.22@sha256:5291449c3df73caf6ed85e649dec1b9e818b39a5d8c871e97afc13e9cd5e8fa8"

if [ "$#" -lt 1 ]; then
  echo "usage: $0 DEMO_PROJECT [TAPE...]" >&2
  exit 2
fi
demo_src="$(cd "$1" && pwd -P)"
shift

assets="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$assets/../.." && pwd)"
engine="${ENGINE:-podman}"
cache="${CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/overbrainer-demo}"

if [ "$#" -gt 0 ]; then
  tapes=("$@")
else
  tapes=()
  for tape in "$assets"/*.tape; do
    tapes+=("$(basename "$tape")")
  done
fi

# Rootless podman: the host user becomes root in the container, so files come
# out owned by you and Chromium in the VHS image can run without its sandbox.
user_args=()
if [ "$engine" = podman ]; then
  user_args=('--userns=keep-id:uid=0,gid=0')
fi
run=("$engine" run --rm "${user_args[@]}" --security-opt label=disable)

mkdir -p "$cache/target" "$cache/registry"
echo "building overbrainer" >&2
"${run[@]}" \
  -v "$repo":/src:ro \
  -v "$cache/target":/target \
  -v "$cache/registry":/usr/local/cargo/registry \
  -e CARGO_TARGET_DIR=/target \
  -w /src \
  "$RUST_IMAGE" cargo build --release --locked
binary="$cache/target/release/overbrainer"

demo="$cache/demo"
mkdir -p "$demo"
if [ "$demo_src" = "$(cd "$demo" && pwd -P)" ]; then
  echo "the demo project must not be $demo, which this script empties" >&2
  exit 2
fi
rm -rf "$demo"
mkdir -p "$demo"
(
  cd "$demo_src"
  tar -cf - \
    --exclude='./.env' --exclude='./.env.*' \
    --exclude='./runs/*/ssh' --exclude='./runs/*/output' --exclude='./runs/.hf-cache' \
    .
) | tar -xf - -C "$demo"

for tape in "${tapes[@]}"; do
  echo "recording $tape" >&2
  "${run[@]}" \
    -v "$assets":/vhs \
    -v "$demo":/demo \
    -v "$binary":/usr/local/bin/overbrainer:ro \
    -w /vhs \
    "$VHS_IMAGE" "$tape"
done

gifs=()
for tape in "${tapes[@]}"; do
  gifs+=("${tape%.tape}.gif")
done
echo "optimizing ${gifs[*]}" >&2
"${run[@]}" -v "$assets":/assets -w /assets "$GIFSICLE_IMAGE" \
  sh -c 'apk add --no-cache --quiet gifsicle && gifsicle -O3 --batch "$@"' sh "${gifs[@]}"
ls -l "${gifs[@]/#/$assets/}"
