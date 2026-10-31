# FlashNext on one RTX 4090 — hosting recipe

A reproducible recipe for serving **Qwen3.8-Flash-Next** (3-shard IQ4_XS MoE GGUF,
~88 GB total) on a **single 24 GB RTX 4090**, with a small lazy-loading OpenAI-compatible
gateway in front of `llama-server`. Numbers below are my own measurements from the
appliance this runs on.

## Hardware

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 564 MiB |
| Driver | 615.78.08 |
| CPU | Intel i9-14900K |
| RAM | 62 GB |
| CUDA libs | cu13 (`runtime/tools/lib/python3.12/site-packages/nvidia/cu13/lib`) |

## Why it fits

The IQ4_XS quantisation is ~88 GB and does not fit in 24 GB VRAM. llama.cpp's
`--n-cpu-moe 38` spills 38 MoE expert layers to system RAM while `-ngl 99` keeps
everything else (attention, shared experts, KV cache) on the GPU. With a 65 536-token
context the working set peaks at ~21.2 GB VRAM — leaving ~3 GB headroom.

## Model download

The HF repo id is **not** recorded on the source appliance, so it is not asserted here.
Obtain the three GGUF shards plus the MTP head and verify the SHA-256 digests:

```
sha256sum -c models/expected.sha256
```

Expected files under `models/UD-IQ4_XS/`:

| file | SHA-256 (first 16 hex) |
|---|---|
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` | `5ce89370720f8bf9` |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf` | `577a38a2392b40ca` |
| `Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf` | `d4634e6d84f0ebb0` |
| `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` | `58a47d6a216307e9` |

## llama.cpp build

```sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 74bca3950054eed1795f525cb9cf06bfe796492d
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j --config Release
```

CUDA 13 runtime libraries must be on `LD_LIBRARY_PATH` (see `launch.sh`).

## Engine launch — every flag explained

`launch.sh` (run by the gateway, not usually by hand):

```sh
export FLASHNEXT_ROOT=<state-dir>   # contains models/UD-IQ4_XS/ and runtime/llama.cpp/
./launch.sh
```

| Flag | Rationale |
|---|---|
| `--model …/…-00001-of-00003.gguf` | First shard; llama.cpp auto-loads sibling shards |
| `--alias flashnext` | Stable model name surfaced at `/v1/models` |
| `--host 127.0.0.1 --port 18099` | Engine stays loopback-only; gateway is the only client |
| `--offline` | No telemetry / outbound calls |
| `--no-webui` | Drop the bundled web UI; saves memory, reduces surface |
| `-fa on` | Flash attention — required to fit 64k context in VRAM |
| `-ngl 99` | Offload all non-MoE layers to GPU |
| `-t 24` | 24 CPU threads (leaves 8 of 32 for the rest of the box) |
| `-C 0xFFFF5555` | Pin inference threads to P-cores / avoid hyperthread siblings (hex affinity mask, i9-14900K topology) |
| `--cpu-strict 1` | Fail rather than silently run with wrong thread affinity |
| `-c 65536` | 64k context window |
| `-np 1` | Single slot — one request at a time |
| `--n-cpu-moe 38` | **Key flag.** 38 MoE expert layers in RAM; GPU keeps ~21 GB working set. 36 OOMed at ~55k context; 38 leaves ~3 GB headroom |
| `-ub 1024` | 1024-token physical batches; controls prompt-eval chunking and VRAM for batch buffers |
| `-ctk f16 -ctv f16` | KV cache in FP16 (largest single VRAM consumer at 64k; quantising KV would save VRAM but hurt quality) |
| `--jinja` | Use the model's chat template (correct tool-calling and thinking tags) |

## Gateway behaviour

`gateway/server.py` is a small `aiohttp` proxy that manages the engine lifecycle:

- **Lazy load** — engine is not started at boot; the first inference request spawns it via `launch.sh`. `/status` and `/v1/models` never load weights.
- **Idle unload** — after `--idle` seconds without traffic the engine is killed and VRAM is released. The deployed service uses `--idle 600` (server default is 120).
- **Admission checks** (503 before loading): refuse if any GPU already reports ≥2 000 MiB used, or if `MemAvailable` < 8 192 MiB. Prevents squeezing onto a contended GPU or OOMing the box.
- **Watchdog** — if `MemAvailable` drops below 3 072 MiB during operation the engine is killed.
- **Lease** — `flock` on `/tmp/local-llm-optional-gpu.lock` so a second gateway instance cannot start.
- **Single slot** — concurrent requests get 429 while the engine is busy.
- **Reasoning efforts** — `none`, `low`, `medium`, `xhigh` (`high` aliases to `xhigh`; default `none`). Thinking is budgeted per effort: low 512, medium 1 024, xhigh 2 560 tokens. The budget is deducted from `max_tokens`, which also reserves space for the visible answer.
- **Output cap** — 4 096 tokens total (`max_tokens` is clamped, not ignored).
- **Context guard** — requests whose prompt exceeds 65 536 tokens are rejected with 400 before the engine sees them.
- **Unauthenticated, LAN-trusted** — binds `0.0.0.0` in the deployed config. Put it behind your own network controls; do not expose it.

Rationale: on a 24 GB card that also hosts other work, "always loaded" is the wrong default. Lazy load + idle unload gives a 22-second first-token cost after a quiet period in exchange for the full GPU the rest of the time.

## runit service

`runit/run`:

```sh
#!/bin/sh
# provisioning gate: service stays down until the runtime root file exists
root=<state-dir>/root
test -s "$root" || { echo "flashnext not provisioned"; sleep 30; exit 1; }
export HOME=<home>
cd /opt/flashnext-app
exec chpst -u <user> /opt/duplex-venv/bin/python -m flashnext.server \
    --root "$(cat "$root")" --host 0.0.0.0 --idle 600
```

The provisioning gate means the service can be enabled at image-build time and only
starts once the model and runtime root are actually present.

## term-llm provider

`term-llm/provider.yaml` — merge after endpoint smoke verification; do not make it the default provider:

```yaml
providers:
  flashnext:
    type: openai_compatible
    base_url: http://<appliance>:18098/v1
    model: flashnext
    context_window: 65536
    max_output_tokens: 4096
    models:
      - id: flashnext
        reasoning_efforts: [none, low, medium, xhigh]
        default_reasoning_effort: none
```

## Benchmarks

Methodology (`bench/bench.py`, results in `bench/results.jsonl`):

- Synthetic prompts built from ~44-token lines with a **fresh random salt per request**, so no line ever hits a prefix cache.
- `stream: true`; time-to-first-token measured from request send to first streamed chunk.
- Prefill rate = `usage.prompt_tokens / TTFT` (gateway-reported token count, not a local estimate).
- Cross-checked against llama.cpp's own engine log (`prompt eval time = X ms / N tokens`); agreement within ~1 %.
- Two runs per row, run back-to-back on an already-loaded engine except where noted.
- The benchmark ran on the appliance while it was also serving this assistant, so figures include real background load rather than a clean-room idle box.

| Scenario | Prompt tokens | Prefill tok/s | Decode tok/s | TTFT (s) | Total (s) |
|---|---|---|---|---|---|
| prefill ~2.4k | 2 434 | 199.7 | — | 12.19 | 12.2 |
| prefill ~2.4k | 2 433 | 261.3 | — | 9.31 | 9.4 |
| prefill ~10k | 10 158 | 307.8 | — | 33.00 | 33.0 |
| prefill ~10k | 9 937 | 314.1 | — | 31.63 | 31.7 |
| prefill ~20.5k | 20 557 | 299.9 | — | 68.55 | 68.6 |
| prefill ~20.5k | 20 542 | 300.4 | — | 68.39 | 68.4 |
| prefill ~41.4k | 41 373 | 292.1 | — | 141.64 | 141.7 |
| prefill ~41.4k | 41 364 | 293.0 | — | 141.19 | 141.2 |
| prefill ~54.4k | 54 434 | 294.9 | — | 184.56 | 184.6 |
| prefill ~54.4k | 54 443 | 303.1 | — | 179.65 | 179.7 |

(Prefill runs request a 2-token completion — decode rates from them are meaningless and omitted.)

| decode, short prompt | 80 | — | 28.8 / 29.1 | 1.57 / 1.54 | 36.3 / 35.9 |
| decode @41.4k ctx | 41 438 | — | 28.6 | 135.23 | 170.1 |
| decode @41.4k ctx | 41 429 | — | 29.6 | 136.12 | 169.9 |
| cold load (idle-unloaded) | 64 max | — | — | 22.13 | 22.7 |

Summary: **~300 tok/s prefill** from 10k up to 54k context, **~29 tok/s decode**
essentially independent of context length up to 41k, **22 s** cold-start after idle
unload (page cache is warm; a true cold read of 88 GB from disk is much worse), and
**21 659 MiB peak VRAM**.

Prefill below ~2.4k tokens is noisy and slower per token (199–261 tok/s) — small
batches don't amortise the fixed request and scheduling cost, and these rows include
gateway routing on a busy box.

## Lessons learned

- **`--n-cpu-moe 36` OOMed at ~55k context on this 24 GB card.** Moving to 38 spilled
  just enough experts to RAM and left roughly 3 GB of headroom. This single integer is
  the difference between "64k context" and "engine dies at 55k".
- **A default 1 024-token output cap silently truncated answers.** Real assistant turns
  with thinking need more. The gateway now caps at 4 096.
- **Unbudgeted `xhigh` thinking ate the whole output.** Once thinking was enabled, long
  reasoning consumed all available `max_tokens` and the visible answer got cut off.
  Per-effort thinking budgets (512 / 1 024 / 2 560), deducted from `max_tokens` with an
  explicit answer reserve, fixed it.
- **Measure with a salted prompt.** Any repeated synthetic prompt hits the prefix cache
  and produces prefill numbers that are fiction.
- **Cross-check against the engine log.** `prompt eval time` from llama.cpp agreed with
  gateway-derived rates to ~1 %, which is what makes these numbers trustworthy.
- **Lazy loading is worth the 22-second wake-up** on a shared 24 GB card. An always-on
  88 GB MoE would starve everything else on the box.

## Tests

```sh
python -m pytest gateway/test_server.py
```

## Not included

Weights (see SHA-256 above), the runtime-root provisioning step, and the GPU host's
network setup. The gateway is unauthenticated — keep it on a trusted network.
