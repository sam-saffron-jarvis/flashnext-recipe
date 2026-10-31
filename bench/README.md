# Benchmarks

How the numbers in the top-level README were produced, and what they do and do
not mean. Raw results are in [`results/`](results/).

## Running it

The engine needs the whole GPU and port 18199, so stop the gateway first:

```sh
export STRATA_ROOT=/srv/strata-runtime          # as passed to strata/install.sh
sv stop flashnext
"$STRATA_ROOT/control/python/bin/python3" bench/benchmark.py --run \
  --i-confirm-gpu-is-free --measure-load --rounds 3 --smoke
sv start flashnext
```

Without `--run` it prints the plan and sends nothing. `--measure-load` starts the
engine as its own child, times it until `/health` reports `loaded: true`, runs the
trials and stops only that child. Results append to
`$STRATA_ROOT/results/benchmark-<time>.jsonl`.

[`decode_by_content.py`](decode_by_content.py) measures decode through a running
gateway for different kinds of output (code, JSON, a tool call, prose, Hebrew,
reasoning, long context), because speculative decoding makes the rate depend on
how predictable the output is:

```sh
python3 bench/decode_by_content.py http://localhost:18098 > decode.jsonl
```

[`bench_driver.py`](bench_driver.py) produced run B on a machine where the
gateway could not be stopped: it waits for the gateway's engine
to idle out on its own, runs the cold ladder above, then adds probes through the
gateway (cold first token, ~100K and ~125K prompts, a 32K/1,000-token decode). If
no free window appears it records `cold_ladder_skipped` instead of passing warm
numbers off as cold.

## Workload

- **Prefill:** ~2K, 8K, 16K, 32K and 54K prompt tokens, `max_tokens: 1`.
  Synthetic numbered lines with a random nonce and salt per request, so nothing
  can come from the prompt cache.
- **Decode:** ~2K and ~32K prompts, 1,000 output tokens, greedy, thinking off.
  A request only counts as a full decode if `usage.completion_tokens` agrees.
- **Smoke (`--smoke`):** a bounded-thinking request and a streamed function call
  whose argument fragments are reassembled and checked.

## What is measured

- Rates come from Strata's engine clock (`timings.prompt_per_second`,
  `timings.predicted_per_second`), not from the client.
- `usage.prompt_tokens` is the real context length; the line counts are only
  targets.
- Time to first token runs from the POST to the first non-empty content,
  reasoning or tool delta, so it includes HTTP, templating and scheduling.

## Cautions

- **"Cold" means a new process**, not a cold disk. The page cache is not dropped,
  which is why a fresh engine took 16.0 s in one run and 26.5 s in the other.
- **Small prompts look slow** because each request pays ~1.8 s of fixed
  first-token cost. Always quote prompt size next to a prefill rate.
- **Decode depends on the output.** The synthetic ladder decodes at ~170–180
  tok/s. Real output ranges from ~215 (JSON) through ~180–200 (code, tool calls)
  to ~145 (English prose) and ~105 (Hebrew), following draft acceptance; real
  mixed use averaged 156 tok/s.
- `prompt_tokens_over_ttft_estimate` and `client_decode_tok_s_estimate` in the
  JSONL are client-side estimates (SSE chunks are not tokens); the engine's own
  timings are authoritative.
- Cached tokens (`timings.cache_n`, `cached_tokens`) are reported separately and
  never counted as prefill.
- IQ3_XXS on Strata against UD-IQ4_XS on llama.cpp is a **speed** comparison:
  different quantisation, runtime and context ceiling. It says nothing about
  quality.

## Recorded runs

| File | When (UTC) | Notes |
|---|---|---|
| `results/run-A-20261031T1306Z.jsonl` | 2026-10-31 13:06 | `benchmark.py --measure-load`; engine log in `run-A-cold-load-…log` |
| `results/run-B-20261031T1941Z.jsonl` | 2026-10-31 19:41 | `bench_driver.py`: cold ladder plus gateway probes |
| `results/decode-by-content-20261101.jsonl` | 2026-11-01 | `decode_by_content.py` through the gateway, warm engine |
| `legacy-llamacpp/results.jsonl` | 2026-10-31 | the llama.cpp backend, method in `legacy-llamacpp/bench.py` |

The CPU-only tests run with `cd bench && python3 -m unittest test_benchmark`.
