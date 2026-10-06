# Exports the model of a run to GGUF with llama.cpp. Written by overbrainer,
# run by `sh` in the job's runtime (container or virtual environment), so
# torch and transformers come from the Axolotl image.
#
# It works in its own directory: model and config paths are relative to it.
# The job writes the export's stage event before it starts.
# overbrainer puts the helpers of llama_cpp.sh (fail, say, fetch, unpack)
# before this code.
# Read from the environment:
#   OVERBRAINER_EXPORT_NAME      name of the GGUF file (the run ID)
#   OVERBRAINER_EXPORT_QUANTIZE  llama-quantize type; F16 and BF16 skip it
#   OVERBRAINER_EXPORT_MODEL     the adapter, or the whole model of a full fine-tune
#   OVERBRAINER_EXPORT_CONFIG    the run's axolotl.yaml
#   OVERBRAINER_CACHE            cache holding llama.cpp, shared by the runs
#   OVERBRAINER_LLAMA_CPP*       the pinned llama.cpp release and its digests
set -eu

cd "$(dirname -- "$0")"

name=$OVERBRAINER_EXPORT_NAME
quantize=$OVERBRAINER_EXPORT_QUANTIZE
model=$OVERBRAINER_EXPORT_MODEL
config=$OVERBRAINER_EXPORT_CONFIG
tag=$OVERBRAINER_LLAMA_CPP
cache=$OVERBRAINER_CACHE/llama.cpp/$tag
work=export-work
out=output/gguf
target=$out/$name-$quantize.gguf

rm -rf "$work"
trap 'rm -rf "$work" "$target.part"' EXIT
mkdir -p "$work" "$out" "$cache"

# The Hugging Face model to convert: a merged copy the run already has, the
# adapter merged into its base here, or the whole model of a full fine-tune.
if [ -f "$model/merged/config.json" ]; then
    hf=$model/merged
    say "converting the merged model in $hf"
elif [ -f "$model/adapter_config.json" ]; then
    [ -f "$config" ] || fail "$config is missing: the adapter cannot be merged without it"
    say "merging the adapter in $model into its base model"
    python3 - "$config" "$work" "$model" <<'EOF'
import sys

import yaml

source, work, adapter = sys.argv[1:4]
with open(source, encoding="utf-8") as file:
    config = yaml.safe_load(file)
# Only the merge is run: no plugin, no resume, no push.
for key in ("plugins", "resume_from_checkpoint", "hub_model_id", "hub_strategy"):
    config.pop(key, None)
config["output_dir"] = work
config["lora_model_dir"] = adapter
with open(f"{work}/merge.yaml", "w", encoding="utf-8") as file:
    yaml.safe_dump(config, file)
EOF
    axolotl merge-lora "$work/merge.yaml"
    hf=$work/merged
    [ -f "$hf/config.json" ] || fail "axolotl merge-lora left no model in $hf"
elif [ -f "$model/config.json" ]; then
    hf=$model
    say "converting the model in $hf"
else
    fail "$model holds neither an adapter nor a model"
fi

source_dir=$cache/source
if [ ! -f "$source_dir/convert_hf_to_gguf.py" ]; then
    say "downloading llama.cpp $tag (source)"
    fetch "$OVERBRAINER_LLAMA_CPP_URL/archive/refs/tags/$tag.tar.gz" \
        "$OVERBRAINER_LLAMA_CPP_SOURCE_SHA256" "$cache/source.tar.gz"
    unpack "$source_dir" "$cache/source.tar.gz" "llama.cpp-$tag"
fi

case $quantize in
F16 | BF16) ;;
*)
    platform=$(uname -sm)
    case $platform in
    "Linux x86_64") asset=ubuntu-x64 digest=$OVERBRAINER_LLAMA_CPP_UBUNTU_X64_SHA256 ;;
    "Linux aarch64") asset=ubuntu-arm64 digest=$OVERBRAINER_LLAMA_CPP_UBUNTU_ARM64_SHA256 ;;
    "Darwin arm64") asset=macos-arm64 digest=$OVERBRAINER_LLAMA_CPP_MACOS_ARM64_SHA256 ;;
    *) fail "llama.cpp $tag has no prebuilt llama-quantize for $platform (Linux x86_64, Linux aarch64 and Darwin arm64 only)" ;;
    esac
    bin_dir=$cache/$asset
    if [ ! -x "$bin_dir/llama-quantize" ]; then
        say "downloading llama.cpp $tag ($asset)"
        fetch "$OVERBRAINER_LLAMA_CPP_URL/releases/download/$tag/llama-$tag-bin-$asset.tar.gz" \
            "$digest" "$cache/$asset.tar.gz"
        unpack "$bin_dir" "$cache/$asset.tar.gz" "llama-$tag"
    fi
    ;;
esac

python3 - torch numpy transformers yaml <<'EOF'
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    sys.exit(
        "export: llama.cpp's convert_hf_to_gguf.py needs the Python module "
        + ", ".join(missing)
        + ", missing from the job's environment"
    )
EOF

convert() {
    PYTHONPATH="$source_dir/gguf-py${PYTHONPATH:+:$PYTHONPATH}" \
        python3 "$source_dir/convert_hf_to_gguf.py" "$hf" --outtype "$1" --outfile "$2"
}
case $quantize in
F16)
    say "converting to $target"
    convert f16 "$target.part"
    ;;
BF16)
    say "converting to $target"
    convert bf16 "$target.part"
    ;;
*)
    say "converting to $work/model.gguf"
    convert auto "$work/model.gguf"
    say "quantizing to $target ($quantize)"
    "$bin_dir/llama-quantize" "$work/model.gguf" "$target.part" "$quantize"
    ;;
esac
mv -f "$target.part" "$target"
say "wrote $target"
