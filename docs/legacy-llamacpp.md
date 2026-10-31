# Legacy backend: llama.cpp + UD-IQ4_XS

The setup this recipe replaced on 2026-10-31. It still works as
`--backend llamacpp` (engine on `127.0.0.1:18099`, launcher
[`flashnext/launch.sh`](../flashnext/launch.sh)) and is kept as a rollback
because it uses a higher bit-width quantisation. It is much slower.

## Recipe

- Weights: `unsloth/Qwen3.8-Flash-Next-GGUF` at revision
  `766911a6b7369840a91dbcd95f9f997acaab6cd6`, `UD-IQ4_XS/` (3 shards, ~88 GB);
  hashes in [`provenance/expected-llamacpp.sha256`](../provenance/expected-llamacpp.sha256).
- llama.cpp at commit `74bca3950054eed1795f525cb9cf06bfe796492d` (the launcher
  refuses any other), built with CUDA 13. Its runtime libraries must be on
  `LD_LIBRARY_PATH`; the launcher sets the paths it was built with.
- Runtime root layout: `runtime/llama.cpp/build/bin/`, `models/UD-IQ4_XS/`.

| Flag | Why |
|---|---|
| `--n-cpu-moe 38` | keeps 38 layers' experts in system RAM. At 36 the engine ran out of CUDA memory near 55K context (a 1.7 GB compute buffer failed at 23.8 GB used); 38 peaks at 21.5 GB on the same prompt |
| `-ngl 99 -fa on` | everything else on the GPU; flash attention is needed to fit 64K |
| `-c 65536 -np 1` | 64K window, one slot |
| `-t 24 -C 0xFFFF5555 --cpu-strict 1` | 24 threads on an i9-14900K: one thread on each of the 8 P-cores (no hyperthread siblings) plus all 16 E-cores; fail rather than run with other affinity |
| `-ub 1024` | 1,024-token physical batches |
| `-ctk f16 -ctv f16` | FP16 KV, the largest VRAM consumer at 64K |
| `--jinja` | the model's chat template, so tool calls and thinking tags are right |
| `--offline --no-webui` | no outbound calls, no bundled UI |

## Measurements (2026-10-31)

Method: [`bench/legacy-llamacpp/bench.py`](../bench/legacy-llamacpp/bench.py),
salted synthetic prompts, engine loaded; client-side rates agreed with
llama.cpp's own `prompt eval time` log lines within ~1%.

| Scenario | Prompt tokens | Prefill tok/s | Decode tok/s | First token |
|---|---|---|---|---|
| prefill | ~2,400 | 200 / 261 | | 12.2 / 9.3 s |
| prefill | ~10,000 | 308 / 314 | | 33.0 / 31.6 s |
| prefill | ~20,500 | 300 / 300 | | 68.6 / 68.4 s |
| prefill | ~41,400 | 292 / 293 | | 141.6 / 141.2 s |
| prefill | ~54,400 | 295 / 303 | | 184.6 / 179.7 s |
| decode | 80 | | 28.8 / 29.1 | 1.6 s |
| decode | ~41,400 | | 28.6 / 29.6 | 135–136 s |
| cold start after idle unload | short | | | 22.1 s |

Peak VRAM 21.7 GB. Against Strata on the same card that is about 9× slower
prefill and 5–6× slower decode, with half the context. That is a speed
comparison only: the quantisations, KV precision and runtimes differ.
