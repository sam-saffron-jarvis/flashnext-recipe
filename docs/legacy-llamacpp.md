# Legacy variant: llama.cpp + UD-IQ4_XS

This is the setup this recipe replaced on 2026-10-31. It still works, it is still the
`--backend llamacpp` path in `gateway/server.py` (engine on 127.0.0.1:18099, launcher
`gateway/launch-llamacpp.sh`), and it is kept as a rollback because it uses a *higher*
bit-width quantisation. It is also dramatically slower.

## What it was

`Qwen3.8-Flash-Next` in **UD-IQ4_XS**, 3 GGUF shards, ~88 GB total, served by llama.cpp at
commit `74bca3950054eed1795f525cb9cf06bfe796492d` with `-c 65536`.

The IQ4_XS quantisation does not fit in 24 GB VRAM. llama.cpp's `--n-cpu-moe 38` spills 38
MoE expert layers into system RAM while `-ngl 99` keeps everything else (attention, shared
experts, KV in FP16) on the GPU. Peak VRAM measured: **21 659 MiB**.

Key flags (full launcher in `launch-llamacpp.sh`):

| flag | why |
|---|---|
| `--n-cpu-moe 38` | **the whole recipe.** 36 OOMed at ~55k context; 38 spills just enough experts to RAM and leaves ~3 GB headroom |
| `-ngl 99 -fa on` | offload everything non-MoE; flash attention required to fit 64k context |
| `-c 65536 -np 1` | 64k window, single slot |
| `-t 24 -C 0xFFFF5555 --cpu-strict 1` | 24 threads pinned to P-cores, avoiding hyperthread siblings; fail rather than run with wrong affinity |
| `-ub 1024` | 1024-token physical batches: controls prompt-eval chunking and batch-buffer VRAM |
| `-ctk f16 -ctv f16` | FP16 KV — the largest single VRAM consumer at 64k; quantising it would save VRAM at a quality cost |
| `--jinja` | model chat template, so tool-calling and thinking tags are correct |
| `--no-webui --offline` | drop the bundled UI, no outbound calls |

CUDA 13 runtime libraries must be on `LD_LIBRARY_PATH` (see the launcher).

Weights (three shards plus the MTP head) are recorded in `models/expected-llamacpp.sha256`;
the HF repository id was not recorded on the source appliance at the time, so it is not
asserted here.

## Measured numbers (2026-10-31, method in `bench/legacy-llamacpp/bench.py`)

Client-derived prefill (`usage.prompt_tokens / TTFT`), cross-checked against llama.cpp's own
`prompt eval time` log lines — agreement within ~1 %. Salted synthetic prompts, two runs per
row, engine already loaded unless noted.

| scenario | prompt tokens | prefill tok/s | decode tok/s | TTFT (s) | total (s) |
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
| decode, short prompt | 80 | — | 28.8 / 29.1 | 1.57 / 1.54 | 36.3 / 35.9 |
| decode @41.4k ctx | 41 438 | — | 28.6 | 135.23 | 170.1 |
| decode @41.4k ctx | 41 429 | — | 29.6 | 136.12 | 169.9 |
| cold load (idle-unloaded) | 64 max | — | — | 22.13 | 22.7 |

(Prefill rows request a 2-token completion, so decode rates from them are meaningless and
omitted.) Summary: **~300 tok/s prefill** from 10k up to 54k, **~29 tok/s decode** almost
independent of context length up to 41k, **22 s** cold start after idle unload (warm page
cache; a genuinely cold read of 88 GB from disk is much worse), **21 659 MiB peak VRAM**.

Prefill below ~2.4k tokens is slower per token (199–261 tok/s) — small batches do not
amortise fixed request and scheduling cost, and these rows include gateway routing.

## Side by side

Same card, same box, both measured by me:

| | llama.cpp UD-IQ4_XS (legacy) | Strata IQ3_XXS (current) |
|---|---|---|
| weight files | ~88 GB, 3 shards | 75.84 GB, 2 shards |
| context | 65 536 | 131 072 |
| prefill (10k–54k) | ~292–314 tok/s | ~2,721–2,857 tok/s |
| prefill at ~102k | n/a | 2,745 tok/s |
| decode | 28.6–29.6 tok/s | 165–184 tok/s |
| cold start to first token | 22.1 s | 15.2 s (16.0 s process load, warm page cache) |
| peak VRAM | 21 659 MiB | ~23.3 GiB in use |
| time to first token at ~41k prompt | 135–142 s | ~14 s (interpolated from the 32k/54k ladder: 11.2 s / 19.7 s) |

Roughly **9× prefill and 6× decode**, which is why the gateway's `--backend` now defaults to
strata in the deployed service. This is *not* a quality comparison: IQ3_XXS is a lower
average bit-width than UD-IQ4_XS and the two runs differ in context ceiling, KV precision and
runtime. If you want the higher-bit variant, benchmark it yourself on your own box rather
than assuming the speed ratio transfers.

## Lessons from the llama.cpp era that still hold

- **Measure with a salted prompt.** A repeated synthetic prompt hits the prefix cache and the
  prefill numbers become fiction. Still the first rule in `bench/BENCHMARK.md`.
- **Cross-check against the engine log.** llama.cpp's `prompt eval time` agreed with
  gateway-derived rates to ~1 %; that is what made them trustworthy, and it is why the strata
  runner prefers `timings.*` over client estimates.
- **Lazy loading is worth the wake-up cost** on a 24 GB card shared with other work. An
  always-on 88 GB MoE starves everything else.
- **Unbudgeted `xhigh` thinking eats the entire output.** The per-effort budgets and the
  explicit answer reserve in `gateway/server.py` exist because of exactly this failure.
- **A 1 024-token output cap silently truncated answers**; that is why the legacy cap became
  4 096 and the strata cap is 16 384.
