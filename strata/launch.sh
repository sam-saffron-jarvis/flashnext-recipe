#!/usr/bin/env bash
# Start the Strata engine for FlashNext in the foreground (loopback 127.0.0.1:18199).
#
# Normally the flashnext gateway runs this on the first chat request and stops it
# after the idle timeout; run it by hand only with the gateway stopped.
# install.sh copies it into the runtime root; ROOT is the directory it runs from.
set -euo pipefail
ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
CONFIG=strata-iq3-128k.json
case "${1:-}" in
  --help|-h)
    echo 'Starts Strata on 127.0.0.1:18199 (128K context by default); requires STRATA_GPU_CONFIRMED_FREE=1.'
    echo 'Never stops any other service; refuses to start if the GPU is already in use.'
    echo 'Example: STRATA_GPU_CONFIRMED_FREE=1 ./launch.sh [--context 64k|128k]'
    exit 0;;
  --context)
    [[ $# == 2 && ( "$2" == 64k || "$2" == 128k ) ]] || { echo 'Expected --context 64k|128k' >&2; exit 2; }
    CONFIG="strata-iq3-$2.json";;
  '') ;;
  *) echo "Unknown argument: $1" >&2; exit 2;;
esac
if [[ ${STRATA_GPU_CONFIRMED_FREE:-} != 1 ]]; then
  echo 'Refusing GPU launch: set STRATA_GPU_CONFIRMED_FREE=1 once the GPU is known to be free.' >&2
  exit 2
fi
# shellcheck source-path=SCRIPTDIR source=env.sh
source "$ROOT/env.sh"
# Read-only guard: never unloads, stops or modifies another process.
used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)
if ! [[ "$used" =~ ^[0-9]+$ ]] || (( used > 2048 )); then
  echo "Refusing launch: GPU 0 already uses ${used} MiB (limit 2048 MiB; the host display holds ~1.2 GB). No process was changed." >&2
  exit 2
fi
# Preparation must be complete and the port free. SO_REUSEADDR so sockets left in
# TIME_WAIT by a just-stopped engine do not block a restart for ~60s.
"$ROOT/control/python/bin/python3" - "$ROOT" "$CONFIG" <<'PY'
import json, pathlib, socket, sys
r = pathlib.Path(sys.argv[1])
assert (r / 'downloads/prepared-artifacts.json').is_file(), 'CPU preparation incomplete'
c = json.loads((r / sys.argv[2]).read_text())
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind((c["host"], c["port"])); s.close()
PY
cd "$ROOT/control"
exec "$ROOT/control/bin/strata" serve --engine strata --config "$ROOT/$CONFIG" \
  --host 127.0.0.1 --port 18199 --parallel 1 --max-queue 1 --offline
