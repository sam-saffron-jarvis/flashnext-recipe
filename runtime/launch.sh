#!/usr/bin/env bash
# NOT RUN during preparation. This starts inference and DOES use the GPU.
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
CONFIG=strata-iq3-128k.json
case "${1:-}" in
  --help|-h)
    echo 'Starts Strata on 127.0.0.1:18199 (128K context); requires STRATA_GPU_CONFIRMED_FREE=1.'
    echo 'First arrange a separate approved GPU-free benchmark window. Never stops any other service.'
    echo 'Example: STRATA_GPU_CONFIRMED_FREE=1 ./launch.sh [--context 64k|128k]'
    exit 0;;
  --context)
    [[ $# == 2 && ( "$2" == 64k || "$2" == 128k ) ]] || { echo 'Expected --context 64k|128k' >&2; exit 2; }
    CONFIG="strata-iq3-$2.json";;
  '') ;;
  *) echo "Unknown argument: $1" >&2; exit 2;;
esac
if [[ ${STRATA_GPU_CONFIRMED_FREE:-} != 1 ]]; then
  echo 'Refusing GPU launch. Confirm the active task has finished and GPU is free, then set STRATA_GPU_CONFIRMED_FREE=1.' >&2
  exit 2
fi
source "$ROOT/env.sh"
# Read-only guard, executed ONLY at a later explicitly confirmed launch.
# It does not unload, stop, restart, or modify any existing process/service.
used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)
if ! [[ "$used" =~ ^[0-9]+$ ]] || (( used > 2048 )); then
  echo "Refusing launch: GPU 0 already uses ${used} MiB (limit 2048 MiB; host display holds ~1.2 GB). No process was changed." >&2
  exit 2
fi
"$ROOT/control/python/bin/python3" - "$ROOT" "$CONFIG" <<'PY'
import json,pathlib,socket,sys
r=pathlib.Path(sys.argv[1]); assert (r/'downloads/prepared-artifacts.json').is_file(), 'CPU preparation incomplete'
c=json.loads((r/sys.argv[2]).read_text())
s=socket.socket(); s.bind((c['host'],c['port'])); s.close()
PY
cd "$ROOT/control"
exec "$ROOT/control/bin/strata" serve --engine strata --config "$ROOT/$CONFIG" \
  --host 127.0.0.1 --port 18199 --parallel 1 --max-queue 1 --offline
