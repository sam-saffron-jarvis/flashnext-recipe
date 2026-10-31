# FlashNext on one RTX 4090 — the recipe I actually run

**What this is:** serving [Qwen3.8-Flash-Next](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(a ~80B-parameter MoE, 512 routed experts per layer across 48 layers, 10 active per token)
on a **single 24 GB RTX 4090**, quantised by ETH Zurich's IST-DASLab to **IQ3_XXS** with
GSQ + RCO, run by [Strata](https://github.com/Niko1221/Strata) (MIT) with experts streamed
from system RAM/SSD, behind a small lazy-loading OpenAI-compatible gateway that unloads the
engine when nobody is using it.

Every number below is mine, measured on the appliance that runs this stack, on 2026-11-01
(two independent 3-round cold runs, ~6.5 h apart). Raw evidence is in `bench/`.

## The numbers

### Prefill (prompt processing), fresh engine process, `--max-context 131072`

Synthetic salted prompts, `max_tokens: 1`, greedy, thinking off. Rates are Strata's own
engine clock (`timings.prompt_per_second`), not a client estimate. 3 rounds; median and range shown.

| actual prompt tokens | median prefill tok/s | min–max | median time to first token |
|---|---|---|---|
| 1,960 – 1,965 | **789** | 766 – 789 | 2.49 s |
| 7,681 – 8,163 | **2,739** | 2,670 – 2,808 | 2.89 s |
| 15,998 – 16,482 | **2,823** | 2,700 – 2,827 | 5.67 s |
| 31,190 – 32,158 | **2,857** | 2,782 – 2,866 | 11.22 s |
| 54,294 – 54,298 | **2,763** | 2,757 – 2,763 | 19.65 s |

The second run (different process, ~6.5 h earlier) gives medians **782 / 2,721 / 2,747 /
2,852 / 2,757** for the same ladder — the two runs agree within ~4 %.

At small prompts the per-token rate collapses because there is roughly **1.8 s of fixed
first-token cost** per request (derived: 2,491 ms at 1,960 tokens vs the ~0.36 ms/token the
same engine sustains at 32k). Quoting "prefill tok/s" without saying which prompt size is
meaningless, so both are given here.

### Decode (generation), 1,000 requested tokens, thinking off

| prompt context | run | median decode tok/s | min–max | MTP drafts accepted |
|---|---|---|---|---|
| ~2.0k | B | **168** | 165 – 177 | 569 – 610 of 850 – 882 |
| ~32.2k | B | **180** | 169 – 184 | 592 – 631 of 819 – 861 |
| ~2.0k | A | **178** | 154 – 179 | 561 – 610 of 862 – 886 |
| ~31.2 – 32.2k | A | **177** | 170 – 180 | 592 – 624 of 837 – 863 |

So: **~2.7–2.9k tok/s prefill, ~170 tok/s decode**, with draft acceptance of 63–77 %
(usually ~70 %). Decode is *faster* at 32k context than at 2k, which is what a MoE with
streamed experts looks like: the bottleneck is expert I/O and drafting, not attention.

### Full 128K window, through the gateway (warm engine)

| prompt tokens | prefill tok/s | time to first token |
|---|---|---|
| 102,373 | 2,745 | 37.35 s |
| 125,359 (95.6 % of the 131,072 window) | 2,714 | 46.24 s |

Prefill does not fall off at the top of the window — `--kv int8` plus KV streaming
(`--kv-resident 32768`) is what makes 128K affordable on 24 GB at all.

### Cold start, and what the gateway costs

| measurement | seconds |
|---|---|
| fresh engine process → `/health` `loaded:true` (run B, page cache warm from a just-exited engine) | **16.0** |
| fresh engine process → `/health` `loaded:true` (run A, ~6.5 h after the previous engine) | **26.5** |
| gateway cold probe: POST to an unloaded gateway, 17-token prompt, first token | **15.2** |
| same probe, completion of 2 tokens | 15.3 |

Gateway overhead is essentially zero: the same 32,188-token / 1,000-token request measured
11.39 s client-side to first token vs 11.36 s engine-side `prompt_ms`, 174.0 tok/s engine
decode vs 174.3 tok/s client estimate, 17.12 s wall vs 17.10 s in the engine's own timings.

For comparison, the previous llama.cpp-based setup on this same card managed ~300 tok/s
prefill and ~29 tok/s decode. Same GPU, same model family, different runtime and
quantisation — see [docs/legacy-llamacpp.md](docs/legacy-llamacpp.md).

## Hardware and why it fits

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 564 MiB |
| Driver | 615.78.08 (Strata's engine requires ≥ 580.65) |
| CPU | Intel i9-14900K (24 cores / 32 threads) |
| RAM | 62 GB + 63 GB swap |
| Model storage | NVMe, ext4 |
| Weight files | 75 839 998 528 B (75.84 GB / 70.63 GiB) across 2 shards |
| VRAM in use while loaded | ~23.3 GB measured (`nvidia-smi`), with `--vram-reserve-mib 1024` |

The trick is that only the dense part, attention, drafting and a slice of KV are pinned to
the GPU. The 512-expert-per-layer MoE is served from host memory with an adaptive resident
expert cache, and the second shard — a **28.8 GB PLE / n-gram lookup table** — is read
straight off SSD with `--ple-io direct` instead of being resident anywhere. Engine log
excerpt from production traffic:

```
strata serve: asynchronous adaptive tier: 183 rounds, 10983 experts swapped in, 16.6 ms per round
cache hit rate: 95.0% (1791867 hits / 1885487 lookups); 2476 more read by the GPU over PCIe ...
```

95 % expert-cache hits and ~10k expert swaps per generation window is where the ~170 tok/s
comes from.

## Pinned provenance

Model — `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`, Apache-2.0 (see `docs/model-card.ISTA-DASLab.md`):

| file | bytes | SHA-256 |
|---|---|---|
| `IQ3_XXS/…-00001-of-00002.gguf` | 47 039 860 096 | `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15` |
| `IQ3_XXS/…-00002-of-00002.gguf` | 28 800 138 432 | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

Draft head — `Qwen/Qwen3.8-Flash-Next`, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`,
quantised locally to Q2_0 (889 017 216 B, `a22206a4d74eafbf31063a84610292535fb3b49af9a339963817fd04aed81e4d`).

Runtime — Strata upstream release **v0.2.9**, Control **0.1.55** (`c62d5e3a15b40ea51c96aae9e8d36928965f53ab`),
engine **0.1.55-linux-x86_64-cuda-cuda13**, build id `99866a47a2b8+cuda13.0`, archs
`sm_75/86/89/120` (4090 = sm_89). The engine binary and its bundled `libcudart/libcublas/
libcublasLt` are hashed in `runtime/provenance/engine-component.json`; the CUDA 13 libraries
ship inside the component, so no CUDA toolkit, nvcc, Docker or pip wheels are needed.
Hashes of the recipe scripts themselves are in `runtime/downloads/runtime-files-sha256.txt`.

## Setup

1. Install Strata v0.2.9 into your runtime root (upstream installer), then copy these files
   into it: `env.sh`, `launch.sh`, `bin/strata`, `download_models.py`, `prepare-cpu.sh`,
   `verify-cpu.py`, `benchmark.py`, `test_benchmark.py`, and the two config templates.

2. `env.sh` keeps all state under the runtime root and forbids surprise downloads:

   ```sh
   export STRATA_ROOT=/path/to/strata-runtime          # edit in env.sh or export before sourcing
   export STRATA_COMPONENTS_DIR=$STRATA_ROOT/components STRATA_DATA_DIR=$STRATA_ROOT/data
   export STRATA_INSTALL_DIR=$STRATA_ROOT STRATA_ENGINE_BACKEND=cuda
   export STRATA_NO_ACQUIRE=1 HF_HUB_OFFLINE=1
   ```

3. Download the pinned shards (sequential, one transfer at a time, SHA-256 verified against
   the pinned HF metadata in `runtime/downloads/hf-ista-metadata.json`):

   ```sh
   cd "$STRATA_ROOT" && control/python/bin/python3 download_models.py
   # writes downloads/model-verification.json; aborts on any size or hash mismatch
   ```

4. CPU-only preparation — pack the native expert blob, extract the tokenizer, fetch and
   quantise the MTP draft head, build the draft runtime. It never touches CUDA or a server:

   ```sh
   ./prepare-cpu.sh          # ~16 GB of derived artifacts, CPU/network only
   ./control/python/bin/python3 verify-cpu.py
   # writes downloads/prepared-artifacts.json with hashes of everything it generated
   ```

5. Materialise the engine profile (the templates use `__RUNTIME_ROOT__` placeholders):

   ```sh
   sed "s|__RUNTIME_ROOT__|$STRATA_ROOT|g" strata-iq3-128k.json.tmpl > strata-iq3-128k.json
   sed "s|__RUNTIME_ROOT__|$STRATA_ROOT|g" strata-iq3-64k.json.tmpl  > strata-iq3-64k.json
   ```

6. Sanity-check the launch without touching the GPU (`strata serve --plan` validates and
   prints the effective launch):

   ```sh
   cd "$STRATA_ROOT/control" && ./bin/strata serve --engine strata \
     --config ../strata-iq3-128k.json --host 127.0.0.1 --port 18199 --parallel 1 --max-queue 1 \
     --offline --plan
   ```

## Engine launch — every flag

`launch.sh` is what the gateway runs. It refuses to start unless a human has confirmed the
GPU is free, and it never touches anyone else's process:

```sh
STRATA_GPU_CONFIRMED_FREE=1 ./launch.sh [--context 64k|128k]
```

Guards, in order: `STRATA_GPU_CONFIRMED_FREE` must equal `1`; `nvidia-smi` must report
≤ 2 048 MiB in use on GPU 0 (the host display holds ~1.2 GB); `downloads/prepared-artifacts.json`
must exist; port 18199 must be free. Read-only checks only — it does not unload, stop,
restart or modify an existing service.

Profile (`strata-iq3-128k.json`, args array) and why each one is there:

| flag | why |
|---|---|
| `--pack models/pack/iq3_xxs` | dense weights + tokenizer prepared from shard 1 by `prepare-cpu.sh` |
| `--native …-00001-of-00002.gguf` | expert weights read in their native quantisation |
| `--ple-gguf …-00002-of-00002.gguf` | the 28.8 GB PLE / n-gram lookup table |
| `--ple-io direct` | unbuffered SSD reads for that table instead of 28.8 GB of extra resident RAM — mandatory at 62 GB system RAM |
| `--expert-profile control/data/expert-profile.bin` | routing profile used to decide which experts to keep resident |
| `--expert-cache auto` | grow the resident expert cache only into VRAM left after dense/head/draft/KV/workspaces. Note it is a count of "largest expert blob" units, **not MiB**; use `auto` until you have measured startup |
| `--vram-reserve-mib 1024` | leave 1 GiB of VRAM unallocated; this is the difference between "loads" and "loads sometimes" |
| `--prefill auto` | keep the documented CUDA chunk defaults (cap 8 192); chunks larger than that are opt-in and not bit-exact on CUDA |
| `--spec 4 --spec-min-p 0.5` | speculative draft depth 4, accept threshold 0.5; proposals are always verified by the target model |
| `--mtp models/mtp/rt` | the locally prepared MTP draft runtime (~0.7–1 GiB VRAM plus host draft data) |
| `--max-context 131072` | 128K window; 128K is the documented upper recommendation for IQ3_XXS on a 64 GB machine |
| `--kv int8` | quantised KV — unavoidable at 128K on 24 GB |
| `--kv-resident 32768` | KV streaming: only ~32K tokens' KV lives in VRAM (~0.45 GB of cells), the rest (~1.8 GB at 128K) is hosted |
| `--pool-workers 15` | expert I/O pool: P-cores minus one plus half the E-cores for a 14900K-class chip; do not blindly use all 32 logical threads |
| `--prompt-cache 1` | one small prefix checkpoint (this is why repeated prompts show `cache_n > 0`) |
| `--conversation-cache-mib 0` | disable the multi-conversation parking budget, whose automatic default can eat several GiB |
| JSON `min_free_vram_mib: 23000` | the server itself refuses to load unless 23 GB is actually free |
| JSON `reasoning_effort: medium`, `reasoning_budget_tokens: 2048` | default thinking budget; the gateway overrides the effort per request |
| JSON `mtp: true`, `draft_vocab: cjk` | enable drafting with the prepared draft vocabulary |
| JSON `fit_max_tokens: false` | an over-long request gets HTTP 400 instead of being silently shortened |
| `--parallel 1 --max-queue 1` | one inference slot, one queued call; predictable memory on a shared card |
| `--offline` | no remote image fetches, no MCP/tool subprocesses |

`strata-iq3-64k.json` is identical except `--max-context 65536`; use it if you want a
like-for-like context ceiling with the old llama.cpp setup.

## Gateway

`gateway/server.py` (~270 lines of `aiohttp`) is the only thing bound to the network. It is
a proxy plus a lifecycle manager, and it is deliberately paranoid about the GPU:

- **Lazy load.** Nothing is loaded at boot. `/v1/models` and `/status` never load weights.
  The first inference request spawns `launch.sh`.
- **Idle unload.** After `--idle` seconds without traffic the engine is terminated and VRAM
  is released. Deployed with `--idle 600`; the script default is 120.
- **Admission guard (503).** Before loading: refuse if `nvidia-smi` reports ≥ 2 000 MiB used
  by any process, or `MemAvailable` < 8 192 MiB.
- **Watchdog.** During operation the engine is killed if `MemAvailable` < 3 072 MiB, or on idle.
- **Single slot.** One in-flight request; anything concurrent gets `429 {"error":"FlashNext busy"}`
  (observed in my own run: a follow-up request fired 1 ms after the previous completion got 429).
- **GPU lease.** `flock` on `/tmp/local-llm-optional-gpu.lock` so a second gateway cannot start.
- **Model id `flashnext`.** Backend profiles are selected by `--backend`:

  | backend | engine port | context | max output | thinking budgets | answer reserve |
  |---|---|---|---|---|---|
  | `strata` (current) | 18199 | 131 072 | 16 384 | low 1 024 / medium 4 096 / xhigh 8 192 | 4 096 |
  | `llamacpp` (legacy) | 18099 | 65 536 | 4 096 | low 512 / medium 1 024 / xhigh 2 560 | 1 536 |

- **Reasoning efforts.** `none`, `low`, `medium`, `xhigh`; `high` aliases to `xhigh`; default
  `none`. The budget is injected as `reasoning_budget_tokens` (Strata does its own wrap-up
  injection, so the gateway's `reasoning_budget_message` hack is only used for llama.cpp),
  and it is capped so `max_tokens - reserve` is never exceeded — otherwise thinking eats the
  whole answer.
- **Request cap.** `client_max_size` is 1 MiB; a 100k-token synthetic prompt is ~180 KB, so
  long-context clients are fine, giant JSON bodies are not.
- **Unauthenticated, LAN-trusted.** It binds `0.0.0.0` in the deployed config. Keep it on a
  trusted network; do not expose it.

## runit service

`runit/run` — enabled at image build time, but a provisioning gate keeps it down until the
runtime root actually exists:

```sh
config=__STATE_DIR__/root                    # a small file whose only content is $STRATA_ROOT
test -s "$config" || { echo 'not provisioned'; exec sleep 60; }
root=$(cat "$config")
exec chpst -u __USER__ __VENV__/bin/python -m flashnext.server \
    --backend strata --root "$root" --host 0.0.0.0 --idle 600
```

(Set `STRATA_ROOT` in `gateway/server.py` too: it is the path used for the engine log and for
finding `launch.sh`, and it must match `--root`.)

## term-llm provider

`term-llm/provider.yaml` — the live entry, matching the strata profile:

```yaml
providers:
  flashnext:
    type: openai_compatible
    base_url: http://<host>:18098/v1
    model: flashnext
    context_window: 131072
    max_output_tokens: 16384
    models:
      - id: flashnext
        reasoning_efforts: [none, low, medium, xhigh]
        default_reasoning_effort: none
```

Worth knowing: this assistant runs on that endpoint, on the box it is describing, with a
single inference slot. So it interrupts itself whenever it talks to you.

## Reproducing the benchmark

`bench/BENCHMARK.md` is the honest methodology; read it before quoting anything. Short version:

```sh
cd "$STRATA_ROOT"
STRATA_RUNTIME="$PWD" python3 bench/bench_driver.py 700
```

`bench_driver.py` waits for the lazy gateway engine to unload on its own (it will not stop a
running service), then runs `benchmark.py --measure-load`, which starts `launch.sh` as its own
child, polls `/health` until `loaded:true`, runs the ladder, and SIGTERMs only its own child.
If no free window appears it says so in the JSONL (`cold_ladder_skipped`) and falls back to
warm gateway probes rather than pretending the numbers were cold.

Two runs are included: `results-A-20261031T1306Z.jsonl` and `results-B-20261031T1941Z.jsonl`
(2026-11-01 00:06 and 06:41 Sydney), plus the engine stdout/cold-load log for run A.

Things these numbers are not:

- **Not** a disk-cold measurement. "Cold load" means a new process; the OS page cache is
  uncontrolled (no `drop_caches`, no reboot). That is why 16.0 s and 26.5 s differ.
- `prompt_tokens_over_ttft_estimate` is **not** pure prefill throughput — it includes HTTP,
  templating, scheduling and first-output overhead. The tables above use the engine's
  `prompt_per_second`.
- `client_decode_tok_s_estimate` is an estimate (SSE chunks are not tokens, MTP bursts,
  detokeniser buffering). Server timings are authoritative.
- Cached prompt tokens are disclosed separately (`timings.cache_n`,
  `usage.prompt_tokens_details.cached_tokens`) and are not counted as newly processed prefill.
- IQ3_XXS vs the legacy UD-IQ4_XS is **not** a like-for-like quality comparison. Faster here
  does not mean better; different quantisation, different context ceiling, different runtime.
- Part of the production traffic on this engine is this assistant's own session, which is why
  the driver insists on a verified free window before timing anything.

## Smoke tests

`benchmark.py --smoke` (part of both recorded runs) sends a bounded-thinking request
(`reasoning_effort medium`, 128-token budget) and a streamed `echo` function call, reconstructs
the argument fragments and checks the final tool envelope without executing the tool. Both
passed in both runs: `reasoning_content_present: true`, `tool_format_check_passed: true`
(`{"text":"READY"}`, `finish_reason: tool_calls`).

```sh
python3 -m pytest gateway/test_server.py bench/test_benchmark.py
```

## File map

```
runtime/            env.sh, launch.sh, bin/strata, config templates, download/prepare/verify scripts
runtime/downloads/  pinned HF metadata, model verification, prepared-artifact hashes, script hashes
runtime/provenance/ Strata Control VERSION+COMMIT, BUNDLE.json, engine BUILD.json + file hashes
gateway/            server.py, test_server.py, launch-llamacpp.sh (legacy backend)
runit/run           service definition with the provisioning gate
term-llm/           provider.yaml entry
bench/              benchmark.py, bench_driver.py, BENCHMARK.md, results-A/B JSONL, engine logs
models/             expected.sha256 (strata), expected-llamacpp.sha256 (legacy)
docs/               legacy-llamacpp.md, model-card.ISTA-DASLab.md
```

Host-specific paths are replaced with `__RUNTIME_ROOT__`, `__STATE_DIR__`, `__USER__`,
`__VENV__`, `__APP_DIR__` — substitute your own.

## Licences

Strata is MIT (`LICENSE.Strata`, upstream `Niko1221/Strata`, v0.2.9); the bundled engine
component hashes are just provenance. The model and its card are Apache-2.0 per the
ISTA-DASLab repo. `gguf-py` used by `prepare-cpu.sh` comes from llama.cpp commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d` (MIT). The gateway, launch and benchmark scripts
are mine — no licence file is asserted for them here, so consider them "all rights reserved"
unless I add one.
