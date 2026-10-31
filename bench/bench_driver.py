#!/usr/bin/env python3
"""Drive a full benchmark window against a live FlashNext (Strata) deployment.

This is the script that produced bench/results/run-*.jsonl. It is deliberately
conservative: it never stops, restarts, or unloads an existing service. It waits
for the lazy gateway engine to release the GPU on its own, then benchmarks only a
process it owns.

  export STRATA_ROOT=/path/to/strata-runtime        # as passed to install.sh
  python3 bench/bench_driver.py [max_seconds_to_wait_for_an_idle_window]

Sequence
  A  wait for the gateway engine to idle out (default 700 s)
  B  cold ladder: benchmark.py --measure-load (fresh engine process, 3 rounds)
  C  gateway cold probe: first token through :18098 from a fully unloaded engine
  D  gateway warm probes: ~100K and ~125K prefill, then 32K/1000-token decode

If phase A never gets a free window it does NOT fake a cold run: it emits
`cold_ladder_skipped` and falls back to warm probes through the gateway, which
are labelled as such in the JSONL.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.request
from types import SimpleNamespace

ROOT = pathlib.Path(os.environ.get("STRATA_ROOT") or os.environ.get("STRATA_RUNTIME") or pathlib.Path.home() / "strata-runtime")
HERE = pathlib.Path(__file__).resolve().parent
GATEWAY = os.environ.get("FLASHNEXT_GATEWAY", "http://127.0.0.1:18098")
ENGINE_PORT = int(os.environ.get("FLASHNEXT_ENGINE_PORT", "18199"))

spec = importlib.util.spec_from_file_location("fnbench", HERE / "benchmark.py")
bench = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(bench)

STAMP = time.strftime("%Y%m%d-%H%M%S")
OUT = ROOT / "results" / f"bench-{STAMP}.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)
ROWS = []


def emit(row):
    row = {"at_unix": time.time(), **row}
    ROWS.append(row)
    line = json.dumps(row, ensure_ascii=False)
    print(line, flush=True)
    OUT.open("a").write(line + "\n")


def gpu_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout.split()
    return max(int(x) for x in out)


def mem_available():
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return -1


def gateway_status():
    try:
        with urllib.request.urlopen(GATEWAY + "/status", timeout=5) as r:
            return json.load(r)
    except Exception as exc:  # gateway unreachable or restarting
        return {"error": str(exc)}


def port_free(port):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def main():
    wait_idle = int(sys.argv[1]) if len(sys.argv) > 1 else 700
    emit({"label": "pre_state", "gpu_used_mib": gpu_used(), "mem_available_mib": mem_available(),
          "gateway": gateway_status(), "engine_port_free": port_free(ENGINE_PORT),
          "config": (ROOT / "strata-iq3-128k.json").read_text()})

    # ---- A: wait for the lazy engine to release the GPU on its own ----
    unloaded_at = None
    deadline = time.monotonic() + wait_idle
    while time.monotonic() < deadline:
        status = gateway_status()
        if not status.get("loaded") and port_free(ENGINE_PORT) and gpu_used() < 2048:
            unloaded_at = time.monotonic()
            break
        time.sleep(15)
    emit({"label": "idle_window", "unloaded": unloaded_at is not None,
          "waited_s": round((unloaded_at or deadline) - (deadline - wait_idle), 1),
          "gpu_used_mib": gpu_used(), "gateway": gateway_status()})

    used = gpu_used()
    cold_ladder_ok = unloaded_at is not None and used < 2048

    # ---- B: cold ladder from a fresh engine process (owned child only) ----
    if cold_ladder_ok:
        cmd = [str(ROOT / "control/python/bin/python3"), str(HERE / "benchmark.py"), "--root", str(ROOT),
               "--run", "--i-confirm-gpu-is-free", "--measure-load", "--launch-context", "128k",
               "--rounds", "3", "--smoke", "--output", str(OUT)]
        t0 = time.monotonic()
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1500)
        emit({"label": "cold_ladder", "returncode": proc.returncode,
              "elapsed_s": round(time.monotonic() - t0, 1), "stderr_tail": proc.stderr[-1500:]})
    else:
        emit({"label": "cold_ladder_skipped", "reason": f"GPU busy ({used} MiB) or port busy; warm fallback used"})

    # ---- warm fallback ladder through the gateway if the cold window never opened ----
    if not cold_ladder_ok:
        args = SimpleNamespace(url=GATEWAY + "/v1/chat/completions", model="flashnext", timeout=1800)
        for trial in range(3):
            for target in (2048, 8192, 16384, 32768, 54000):
                prompt, lines = bench.facts(target, "Reply with exactly: READY")
                emit({**bench.call(args, {"messages": [{"role": "user", "content": prompt}], "max_tokens": 1},
                                f"gateway_prefill_{target}_r{trial + 1}"),
                      "target_prompt_tokens": target, "synthetic_lines": lines})

    # ---- C: gateway cold probe (engine load + first token, engine process cold) ----
    gate_free = False
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if port_free(ENGINE_PORT) and gpu_used() < 2048:
            gate_free = True
            break
        time.sleep(5)
    args = SimpleNamespace(url=GATEWAY + "/v1/chat/completions", model="flashnext", timeout=1800)
    emit({"label": "gateway_before_cold_probe", "gateway": gateway_status(),
          "gpu_used_mib": gpu_used(), "engine_port_free": gate_free})
    emit({**bench.call(args, {"messages": [{"role": "user", "content": "Reply with exactly: READY"}],
                           "max_tokens": 8}, "gateway_cold_probe"),
          "engine_was_loaded": not port_free(ENGINE_PORT)})

    # ---- D: warm gateway probes: ~100K and ~125K prefill, then 32K/1000 decode ----
    for target in (100000, 122000):
        prompt, lines = bench.facts(target, "Reply with exactly: READY")
        emit({**bench.call(args, {"messages": [{"role": "user", "content": prompt}], "max_tokens": 1},
                        f"gateway_prefill_{target}"), "target_prompt_tokens": target, "synthetic_lines": lines})
    prompt, lines = bench.facts(32768, "Write a long numbered list of observations about these facts. "
                                       "Continue for at least 2000 tokens. Do not summarize or conclude early.")
    emit({**bench.call(args, {"messages": [{"role": "user", "content": prompt}], "max_tokens": 1000},
                    "gateway_decode_32768"), "target_prompt_tokens": 32768, "synthetic_lines": lines})

    emit({"label": "post_state", "gpu_used_mib": gpu_used(), "mem_available_mib": mem_available(),
          "gateway": gateway_status()})
    print(f"\nEVIDENCE {OUT} rows={len(ROWS)}")


if __name__ == "__main__":
    main()
