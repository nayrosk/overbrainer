# Serves a run's GGUF with llama-server and asks it every question of the
# eval set, one at a time. Written by overbrainer, run by `sh` in the job's
# runtime, in the job's own directory, which holds questions.jsonl and
# compare_client.py, and model.gguf when the GGUF was uploaded with the job.
# The job writes the compare's stage event before it starts.
# Read from the environment:
#   OVERBRAINER_COMPARE_MODEL        the GGUF: model.gguf when uploaded with the
#                                    job, else an absolute path on the target
#   OVERBRAINER_COMPARE_DISCARD_MODEL set when the GGUF was uploaded with the
#                                    job: it is removed once the job ends
#   OVERBRAINER_COMPARE_CTX          context size of the server, 0 for the model's own
#   OVERBRAINER_COMPARE_MAX_TOKENS   limit of each answer
#   OVERBRAINER_COMPARE_TEMPERATURE  sampling temperature
#   OVERBRAINER_COMPARE_START_SECS   how long the server may take to be ready
#   OVERBRAINER_METRICS              the job's metrics file, for progress lines
#   OVERBRAINER_CACHE                cache holding llama.cpp, shared by the runs
#   OVERBRAINER_COMPARE_KFD          the AMD compute device, /dev/kfd by default
#   OVERBRAINER_LLAMA_CPP*           the pinned llama.cpp release and its digests
#
# overbrainer puts the helpers of llama_cpp.sh (fail, say, fetch, unpack)
# before this code. Everything is a function; the last line runs `main`,
# unless OVERBRAINER_COMPARE_SOURCED is set (the tests source the functions).
set -eu

# Prints `compare: warning: $*` on stderr: the job goes on.
warn() {
    printf 'compare: warning: %s\n' "$*" >&2
}

server_pid=
model=

# Stops the server (TERM, then KILL after 10 s) and removes the GGUF uploaded
# with the job (OVERBRAINER_COMPARE_DISCARD_MODEL set): it never stays on the
# target after the job. A GGUF that was on the target already is left there.
cleanup() {
    if [ -n "$server_pid" ]; then
        kill "$server_pid" 2>/dev/null || true
        i=0
        while kill -0 "$server_pid" 2>/dev/null; do
            i=$((i + 1))
            if [ "$i" -gt 50 ]; then
                kill -9 "$server_pid" 2>/dev/null || true
                break
            fi
            sleep 0.2
        done
    fi
    if [ -n "${OVERBRAINER_COMPARE_DISCARD_MODEL:-}" ] && [ -n "$model" ]; then
        rm -f -- "$model"
    fi
}

# Prints the CUDA version the NVIDIA driver supports, from the
# "CUDA Version: X.Y" header of nvidia-smi; prints nothing when the header
# does not say. Fails when there is no working nvidia-smi: no NVIDIA GPU.
driver_cuda() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    smi=$(nvidia-smi 2>/dev/null) || return 1
    case $smi in
    *"CUDA Version:"*) ;;
    *) return 0 ;;
    esac
    version=${smi#*CUDA Version:}
    version=${version#"${version%%[! ]*}"}
    printf '%s\n' "${version%%[!0-9.]*}"
}

# Prints the output of `ldconfig -p`, the libraries the dynamic linker knows;
# prints nothing when there is no ldconfig. It lives in /sbin for most users.
linker_libraries() {
    if command -v ldconfig >/dev/null 2>&1; then
        ldconfig -p 2>/dev/null || true
    elif [ -x /sbin/ldconfig ]; then
        /sbin/ldconfig -p 2>/dev/null || true
    fi
}

# Picks the ROCm build for an AMD GPU, after pick_build chose the CPU build:
# only a Linux x86_64 host with /dev/kfd, the ROCm 7 runtime in the linker's
# cache and read and write access to the device gets it. When there is a
# /dev/kfd and the build stays on the CPU, warns why. Does nothing, quietly,
# on a host without /dev/kfd.
pick_amd() {
    kfd=${OVERBRAINER_COMPARE_KFD:-/dev/kfd}
    [ -e "$kfd" ] || return 0
    if [ "$arch" != x64 ]; then
        warn "AMD GPU found but llama.cpp $tag has no ROCm build for $arch: using the CPU build $asset"
        return 0
    fi
    known=$(linker_libraries)
    missing=
    packages=
    # Each library of the ROCm 7 runtime, with the Ubuntu package holding it.
    for pair in libamdhip64.so.7:libamdhip64-7 librocblas.so.5:librocblas5 libhipblas.so.3:libhipblas3; do
        library=${pair%%:*}
        case $known in
        *"$library "*) ;;
        *)
            missing="${missing:+$missing, }$library"
            packages="${packages:+$packages }${pair#*:}"
            ;;
        esac
    done
    if [ -n "$missing" ]; then
        warn "AMD GPU found but the ROCm 7 runtime is missing ($missing); using the CPU build. On Ubuntu: apt install $packages"
        return 0
    fi
    if [ ! -r "$kfd" ] || [ ! -w "$kfd" ]; then
        warn "AMD GPU found but $kfd is not readable and writable by this user (add it to the render group); using the CPU build $asset"
        return 0
    fi
    asset=ubuntu-rocm-$rocm-x64 digest=$OVERBRAINER_LLAMA_CPP_UBUNTU_ROCM_X64_SHA256
}

# Picks the llama.cpp build for this machine: sets asset and digest, and
# cudart and cudart_digest (the CUDA runtime archive) for a CUDA build, empty
# for a CPU or ROCm one. A Linux host with a working nvidia-smi gets the CUDA
# build, unless its driver supports a CUDA major version below the pinned one:
# then the CPU build, with a warning. Without nvidia-smi, an AMD host gets the
# ROCm build (see pick_amd).
pick_build() {
    tag=$OVERBRAINER_LLAMA_CPP
    cuda=$OVERBRAINER_LLAMA_CPP_CUDA
    rocm=$OVERBRAINER_LLAMA_CPP_ROCM
    cudart=
    cudart_digest=
    platform=$(uname -sm)
    case $platform in
    "Linux x86_64")
        arch=x64
        asset=ubuntu-x64 digest=$OVERBRAINER_LLAMA_CPP_UBUNTU_X64_SHA256
        cuda_digest=$OVERBRAINER_LLAMA_CPP_UBUNTU_CUDA_X64_SHA256
        runtime_digest=$OVERBRAINER_LLAMA_CPP_CUDART_X64_SHA256
        ;;
    "Linux aarch64")
        arch=arm64
        asset=ubuntu-arm64 digest=$OVERBRAINER_LLAMA_CPP_UBUNTU_ARM64_SHA256
        cuda_digest=$OVERBRAINER_LLAMA_CPP_UBUNTU_CUDA_ARM64_SHA256
        runtime_digest=$OVERBRAINER_LLAMA_CPP_CUDART_ARM64_SHA256
        ;;
    "Darwin arm64")
        asset=macos-arm64 digest=$OVERBRAINER_LLAMA_CPP_MACOS_ARM64_SHA256
        return 0
        ;;
    *) fail "llama.cpp $tag has no prebuilt llama-server for $platform (Linux x86_64, Linux aarch64 and Darwin arm64 only)" ;;
    esac
    if ! driver=$(driver_cuda); then
        pick_amd
        return 0
    fi
    driver_major=${driver%%.*}
    case $driver_major in
    '' | *[!0-9]*)
        warn "nvidia-smi does not say which CUDA version the driver supports: using the CPU build $asset"
        return 0
        ;;
    esac
    if [ "$driver_major" -lt "${cuda%%.*}" ]; then
        warn "the NVIDIA driver supports CUDA $driver, below the CUDA $cuda of llama.cpp $tag: using the CPU build $asset"
        return 0
    fi
    asset=ubuntu-cuda-$cuda-$arch digest=$cuda_digest
    cudart=cudart-llama-$tag-bin-ubuntu-cuda-$cuda-$arch cudart_digest=$runtime_digest
}

# Fetches the build pick_build chose into the cache, with its CUDA runtime
# (the ROCm runtime comes from the system),
# starts llama-server on 127.0.0.1 and runs the client.
main() {
    cd "$(dirname -- "$0")"
    model=$OVERBRAINER_COMPARE_MODEL
    trap cleanup EXIT
    trap 'exit 143' TERM INT

    pick_build
    cache=$OVERBRAINER_CACHE/llama.cpp/$tag
    mkdir -p "$cache"
    bin_dir=$cache/$asset
    if [ ! -x "$bin_dir/llama-server" ]; then
        say "downloading llama.cpp $tag ($asset)"
        fetch "$OVERBRAINER_LLAMA_CPP_URL/releases/download/$tag/llama-$tag-bin-$asset.tar.gz" \
            "$digest" "$cache/$asset.tar.gz"
        unpack "$bin_dir" "$cache/$asset.tar.gz" "llama-$tag"
    fi
    [ -x "$bin_dir/llama-server" ] || fail "llama.cpp $tag ($asset) has no llama-server"

    # On a CUDA host the runtime the build was made with comes first, whatever
    # the machine has.
    lib_path=$bin_dir
    if [ -n "$cudart" ]; then
        cudart_dir=$cache/$cudart
        if [ ! -f "$cudart_dir/libcudart.so.${cuda%%.*}" ]; then
            say "downloading the CUDA $cuda runtime of llama.cpp $tag"
            fetch "$OVERBRAINER_LLAMA_CPP_URL/releases/download/$tag/$cudart.tar.gz" \
                "$cudart_digest" "$cache/$cudart.tar.gz"
            unpack "$cudart_dir" "$cache/$cudart.tar.gz" "$cudart"
        fi
        lib_path=$cudart_dir:$bin_dir
    fi

    [ -f "$model" ] || fail "$model is missing from $(pwd)"
    port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
    say "starting llama-server ($asset) on 127.0.0.1:$port"
    LD_LIBRARY_PATH="$lib_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$bin_dir/llama-server" -m "$model" --host 127.0.0.1 --port "$port" \
        -ngl 999 -c "$OVERBRAINER_COMPARE_CTX" -np 1 --no-webui >server.log 2>&1 &
    server_pid=$!

    python3 compare_client.py --port "$port" --server-pid "$server_pid" --build "$asset"
}

[ -n "${OVERBRAINER_COMPARE_SOURCED:-}" ] || main "$@"
