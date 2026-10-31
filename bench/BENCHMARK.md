# Benchmark later — not executed during preparation

Only run after the long FlashNext task is finished and an **exclusive GPU window** has been separately arranged. Neither script stops/unloads/restarts any existing service. If FlashNext's engine remains resident, `launch.sh` refuses; do not work around that while the live session is active.

## Start Strata manually, then measure

On the appliance:

```bash
cd __RUNTIME_ROOT__
STRATA_GPU_CONFIRMED_FREE=1 ./launch.sh > logs/server.log 2>&1
```

In a second appliance shell:

```bash
cd __RUNTIME_ROOT__
./control/python/bin/python3 benchmark.py --run --i-confirm-gpu-is-free --rounds 3 --smoke
```

The foreground server is managed by that shell, not runit. End only this Strata server with Ctrl-C when finished. Default context is **128K**, the documented upper recommendation for IQ3_XXS on a 64 GB PC; KV streaming keeps the VRAM window at 32K. Use `./launch.sh --context 64k` for a conservative/like-for-like context-limit comparison with the old llama.cpp setup.

## Measure a fresh engine process and benchmark it

With port 18199 unused and the GPU free:

```bash
cd __RUNTIME_ROOT__
./control/python/bin/python3 benchmark.py --run --i-confirm-gpu-is-free \
  --measure-load --rounds 3 --smoke
```

This starts **only** `launch.sh` as an owned child, polls `/health` until `loaded=true`, records startup seconds, runs the trials sequentially, then sends SIGTERM **only to its own child API process**, whose normal cleanup owns its engine. Add `--launch-context 64k` to use the fallback preset; default is 128K. It refuses an occupied port and never terminates any pre-existing process. Its log is `logs/benchmark-cold-load-<timestamp>.log`.

“Cold load” here means a new engine process. OS disk/page cache is uncontrolled. No `drop_caches`, reboot, model eviction, or competing workload modification is performed. For a genuinely disk-cold result use a separately arranged post-reboot quiet window and document that condition; this script must not reboot the appliance.

## Workload and measurements

- Prefill: approximately 2,048 / 8,192 / 16,384 / 32,768 / 54,000 prompt tokens, one generated token. Synthetic `Fact i: item ... salt ... code ... day ...` lines follow the previous FlashNext repro, estimated at ~34 tokens/line. A random leading nonce and salt make each request unique. A small fixed template prefix may still be reused; `cached_prompt_tokens` / `timings.cache_n` and `prompt_reuse_detected` disclose this. Do not count cached tokens as newly processed prefill.
- Decode: approximately 2K and 32K prompts, requesting 1,000 output tokens, greedy, thinking disabled. `early_stop` and `finish_reason` expose early EOS; a request is not claimed as a full 1,000-token decode unless actual completion usage agrees.
- `usage.prompt_tokens` is the actual context length, not the line-count target. `timings.prompt_n`, `prompt_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`, `predicted_per_second` are Strata's engine-clock measurements. Prefer those over client estimates.
- TTFT is measured from POST start to first non-empty output/reasoning/tool delta (not the initial assistant-role chunk). It includes HTTP, tokenizer/template, scheduling, prefill, and first output overhead. `prompt_tokens_over_ttft_estimate` is **not pure prefill throughput**.
- `client_decode_tok_s_estimate` uses `(completion_tokens-1)/(last_output-first_output)`; it is an estimate because SSE chunks are not necessarily tokens, MTP can burst output, and detokenization can buffer. The server's decode rate is authoritative. No fake per-token timestamps are inferred from chunks.
- The optional `--smoke` sends a 128-token bounded-thinking request (512 total max tokens) and a streamed `echo` function-call request, reconstructs argument fragments, and checks the final tool-call envelope. It does not execute the declared tool. Both recorded runs executed it: bounded thinking returned `reasoning_content_present: true`, and the streamed tool call passed the envelope check.
- Results append as JSONL under `results/benchmark-<time>.jsonl`. Each record includes actual usage, timing details, finish reason, and errors. Use `--output FILE` to override. HTTP errors are recorded and make the command fail; there are no silent retries.

During preparation no inference requests, cold loads, or benchmarks were executed; the only
checks run were `--help`, the default dry plan, `--self-test`, Python compilation, and mocked
API unit tests. A dry plan does not even contact the API. The two recorded runs in this
directory (`results-A-20261031T1306Z.jsonl`, `results-B-20261031T1941Z.jsonl`) were executed
later, each after a separately arranged exclusive-GPU window was confirmed by
`bench_driver.py` waiting for the lazy engine to unload itself.

## Comparison cautions

Compare the same model quantization and sampling/thinking/context/cache policy. This IQ3_XXS run is not a like-for-like quality comparison with the previous llama.cpp UD-IQ4_XS. Upstream 4090 measurements are not a guarantee for this box, CPU, memory channels, SSD, or current release. Save the complete config, engine version, raw JSONL, server logs, process-cold/warm state, and competing workload status. Repeat/interleave trials rather than selecting the best result. Avoid API monitors during timed trials.
