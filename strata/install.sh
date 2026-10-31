#!/usr/bin/env bash
# Install the pinned Strata runtime for FlashNext into STRATA_ROOT (default
# $STRATA_ROOT or ~/strata-runtime; put it on a fast NVMe disk with ~90 GB free).
# Idempotent: verified downloads and models are reused, never re-fetched.
#
#   strata/install.sh [--skip-models] [STRATA_ROOT]
#
# 1. Official Strata v0.2.9 Linux release (Control 0.1.55 + CUDA 13 engine) and
#    llama.cpp gguf-py, each checked against a pinned SHA256.
# 2. Pinned ISTA-DASLab IQ3_XXS GGUFs (~76 GB), SHA256-verified (download_models.py).
# 3. CPU-only preparation: MTP draft tensors, native pack (prepare-cpu.sh).
# 4. Installs launch.sh and env.sh into the root and renders both engine configs
#    from the templates for that root. Re-run after editing them here.
# Never starts the engine or touches the GPU. Run as the workspace user.
set -euo pipefail

SKIP_MODELS=0
if [[ ${1:-} == --skip-models ]]; then SKIP_MODELS=1; shift; fi
ROOT=$(realpath -m -- "${1:-${STRATA_ROOT:-$HOME/strata-runtime}}")
SRC=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)

RELEASE=https://github.com/Niko1221/Strata/releases/download/v0.2.9
CONTROL=strata-control-0.1.55-linux-x86_64.tar.gz
CONTROL_SHA=50783d2fa6c6e98e9bdc8d4f66ad070706b1e1f3e5e2782cc59fd5dabd9488e8
ENGINE=strata-engine-0.1.55-linux-x86_64-cuda-cuda13.tar.gz
ENGINE_SHA=ce86b23b26a4521a1871113057534b8a34c01df0ac7b916adccf9b1c8c068f92
ENGINE_DIR=components/engine/0.1.55-linux-x86_64-cuda-cuda13
LLAMA_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d
LLAMA=llama.cpp-3cf0325.tar.gz
LLAMA_SHA=c076d7534afa0e5d0ec2a0d425b11e791c16f3de0d727221aea071cef156a280
HF_REPO=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
HF_REV=ed59f92082b1e93c0e96d60a8b11aab089b52f09

mkdir -p "$ROOT"/{downloads,components/engine,tools,models,logs,results,home,data,config,state,cache,bin}
cd "$ROOT"

fetch() {  # fetch URL FILE SHA256: download unless a verified copy exists
  local url=$1 file=downloads/$2 sha=$3
  if [[ -f $file ]] && echo "$sha  $file" | sha256sum -c --status; then
    echo "verified $file"; return
  fi
  curl -fL --retry 3 -o "$file.part" "$url"
  echo "$sha  $file.part" | sha256sum -c --status || { echo "SHA256 mismatch: $url" >&2; rm -f "$file.part"; exit 1; }
  mv "$file.part" "$file"; echo "downloaded $file"
}

fetch "$RELEASE/$CONTROL" "$CONTROL" "$CONTROL_SHA"
fetch "$RELEASE/$ENGINE" "$ENGINE" "$ENGINE_SHA"
fetch "https://github.com/ggml-org/llama.cpp/archive/$LLAMA_COMMIT.tar.gz" "$LLAMA" "$LLAMA_SHA"

if [[ ! -x control/bin/strata ]]; then
  mkdir -p control && tar -xzf "downloads/$CONTROL" -C control
fi
if [[ ! -x $ENGINE_DIR/strata ]]; then
  mkdir -p "$ENGINE_DIR" && tar -xzf "downloads/$ENGINE" -C "$ENGINE_DIR"
fi
if [[ ! -d tools/llama.cpp-$LLAMA_COMMIT/gguf-py ]]; then
  tar -xzf "downloads/$LLAMA" -C tools "llama.cpp-$LLAMA_COMMIT/gguf-py"
fi

cat > bin/strata <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/../env.sh"
exec "$STRATA_ROOT/control/bin/strata" "$@"
EOF
chmod +x bin/strata

install -m 0755 "$SRC/launch.sh" launch.sh
install -m 0644 "$SRC/env.sh" env.sh
for c in 128k 64k; do
  sed "s|__RUNTIME_ROOT__|$ROOT|g" "$SRC/strata-iq3-$c.json.tmpl" > "strata-iq3-$c.json"
done

if (( SKIP_MODELS )); then
  echo "runtime installed in $ROOT (models skipped)"; exit 0
fi

if [[ ! -s downloads/hf-ista-metadata.json ]]; then
  curl -fL --retry 3 -o downloads/hf-ista-metadata.json \
    "https://huggingface.co/api/models/$HF_REPO/revision/$HF_REV?blobs=true"
fi
control/python/bin/python3 "$SRC/download_models.py" "$ROOT"
if [[ ! -s downloads/prepared-artifacts.json ]]; then
  bash "$SRC/prepare-cpu.sh" "$ROOT"
fi
echo "Strata runtime ready in $ROOT; the flashnext gateway starts it on demand (--root $ROOT)."
