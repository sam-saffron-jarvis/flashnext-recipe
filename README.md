# FlashNext on one RTX 4090

[![CI](https://github.com/sam-saffron-jarvis/flashnext-recipe/actions/workflows/ci.yml/badge.svg)](https://github.com/sam-saffron-jarvis/flashnext-recipe/actions/workflows/ci.yml)

Serving [Qwen3.8-Flash-Next](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
on a single 24 GB RTX 4090 with 128K context, at ~2,800 tok/s prefill and
~170 tok/s decode. The model is ISTA-DASLab's IQ3_XXS quantisation, run by
[Strata](https://github.com/Niko1221/Strata) with the experts in system RAM,
behind a small gateway that loads it on demand, queues requests, unloads it when
idle and can hand the GPU to Ollama and back.

This is the code and configuration of a running deployment that serves as a
coding-agent backend, with its raw benchmark data.

## Numbers

Measured 2026-10-31 on this hardware. Two independent three-round runs from a
fresh engine process; rates are Strata's own engine timings, thinking off. Method
and caveats: [bench/README.md](bench/README.md).

| Prompt tokens | Prefill tok/s, median (range) | First token |
|---|---|---|
| ~2,000 | 789 (766–789) | 2.5 s |
| ~8,000 | 2,739 (2,670–2,808) | 2.9 s |
| ~16,000 | 2,823 (2,700–2,827) | 5.7 s |
| ~32,000 | 2,857 (2,782–2,866) | 11.3 s |
| ~54,000 | 2,763 (2,757–2,763) | 19.7 s |
| 102,373 (through the gateway) | 2,745 | 37.4 s |
| 125,359 (through the gateway) | 2,714 | 46.2 s |

The second run's medians (782 / 2,721 / 2,747 / 2,852 / 2,757) agree within 3%.
Every request pays ~1.8 s of fixed first-token cost, which dominates below ~8K tokens.

| Decode, 1,000 tokens | tok/s |
|---|---|
| synthetic prompt, ~2K context | 168 (run B), 178 (run A) |
| synthetic prompt, ~32K context | 180 (run B), 177 (run A) |
| natural code / prose prompt | 186 / 158 |

Synthetic text is easy for the MTP drafter (63–77% of drafts accepted), so expect
~160–185 tok/s on real work. A fresh engine is ready in 16–27 s; a request to an
unloaded gateway gets its first token in ~15 s.

Under agent load:

| | Result |
|---|---|
| short side request between turns of a 78K-token conversation | next turn 0.28 s with the conversation cache, 28.4 s without |
| request arriving while another generates | waits its turn (was an instant 429) |
| Ollama `qwen3.8:27b` request while FlashNext is loaded | FlashNext unloads, Ollama loads fully on the GPU, first reply 7.9–8.8 s |

For comparison, llama.cpp with Unsloth's UD-IQ4_XS on the same card managed ~300
tok/s prefill and ~29 tok/s decode at 64K context
([docs/legacy-llamacpp.md](docs/legacy-llamacpp.md)). Different quantisation, so
that is a speed comparison, not a quality one.

## How it fits in 24 GB

| | |
|---|---|
| GPU | RTX 4090 24 GB; NVIDIA driver ≥ 580.65 (tested 615.78) |
| CPU / RAM | i9-14900K, 62 GB RAM |
| Disk | NVMe, ~87 GB for the runtime and weights |
| While loaded | 23.4 GB VRAM (including ~1.2 GB for the host display), ~10 GB RAM free |

Flash-Next is a sparse mixture of experts: 48 layers of 512 routed experts, 10
active per token. Strata keeps the dense weights, attention, the MTP draft head
and the most recent 32K tokens of int8 KV on the GPU. The experts sit in RAM with
a learned cache of the busy ones in VRAM, and the 28.8 GB per-layer n-gram lookup
table (shard 2) is read straight from the SSD.

## Install

Everything runs as one unprivileged user (here `flashnext`).

1. **Runtime.** On an NVMe disk with ~90 GB free:

   ```sh
   strata/install.sh /srv/strata-runtime
   ```

   It downloads the pinned Strata v0.2.9 release and the IQ3_XXS weights,
   verifying every file against a pinned SHA-256, runs Strata's CPU-only
   preparation (MTP draft head, native pack), installs `launch.sh` and `env.sh`,
   and renders both engine configs for that root. It never touches the GPU and
   is safe to re-run.

2. **Gateway.** Python 3.11+ with `aiohttp>=3.10`, then install the package and
   the runit service:

   ```sh
   python3 -m venv /opt/flashnext-venv && /opt/flashnext-venv/bin/pip install 'aiohttp>=3.10'
   sudo mkdir -p /opt/flashnext-app && sudo cp -r flashnext /opt/flashnext-app/
   sudo cp -r runit/flashnext /etc/service/
   ```

   [`runit/flashnext/run`](runit/flashnext/run) reads `STRATA_ROOT`,
   `FLASHNEXT_USER`, `FLASHNEXT_PYTHON`, `FLASHNEXT_APP_DIR` and
   `FLASHNEXT_IDLE_SECONDS` from its environment; the defaults match the commands
   above. It stays down until the runtime exists.

3. **Check it.** `curl http://localhost:18098/status` never loads the model. The
   real-model gate runs completion, a streamed tool call and idle unload through
   its own gateway, so stop the service first:

   ```sh
   sv stop flashnext
   cd /opt/flashnext-app && sudo -u flashnext /opt/flashnext-venv/bin/python -m flashnext.smoke \
     --root /srv/strata-runtime --evidence /tmp/smoke.json
   sv start flashnext
   ```

4. **Client.** Merge [`term-llm/provider.yaml`](term-llm/provider.yaml) into your
   [term-llm](https://github.com/SamSaffron/term-llm) config. Any OpenAI-compatible
   client works with model `flashnext`.

## Engine configuration

[`strata/strata-iq3-128k.json.tmpl`](strata/strata-iq3-128k.json.tmpl); the 64K
variant differs only in `--max-context`.

| Setting | Why |
|---|---|
| `--pack`, `--native`, `--ple-gguf` | dense weights and tokenizer packed from shard 1; experts read from shard 1 in their native quantisation; shard 2 is the n-gram lookup table |
| `--ple-io direct` | read the lookup table from the SSD instead of holding 28.8 GB more in RAM |
| `--expert-cache auto` | grow the VRAM expert cache into whatever is left after weights, draft head, KV and workspaces |
| `--vram-reserve-mib 1024` | leave 1 GiB free; without it loading is unreliable |
| `--spec 4 --spec-min-p 0.5`, `--mtp` | MTP speculative decoding, depth 4; every draft is verified by the full model |
| `--max-context 131072` | 128K, the upstream recommendation for IQ3_XXS on a 64 GB machine |
| `--kv int8 --kv-resident 32768` | int8 KV with only the latest 32K tokens in VRAM; the rest streams from RAM |
| `--pool-workers 15` | expert I/O threads: P-cores minus one plus half the E-cores on a 14900K |
| `--conversation-cache-mib 4096`, `-slots 3`, `-min-tokens 2048` | park up to three long conversations in RAM so a short side request (a Guardian review, a title, a second client) does not evict the main one |
| `--conversation-cache-min-free-mib 4096` | stop parking before free RAM nears the gateway's 3 GB safety limit |
| `fit_max_tokens: false` | an over-long request gets HTTP 400 instead of being silently cut |
| `min_free_vram_mib: 23000` | the engine refuses to load onto a busy GPU |

## Gateway behaviour

[`flashnext/server.py`](flashnext/server.py), an `aiohttp` app on port 18098 and
the only thing exposed to the network.

- **Lazy.** The first chat request starts the engine; it stops after the idle
  timeout (600 s in the service). `/v1/models` and `/status` never load anything.
- **Queue.** One request runs at a time. Others wait in order for up to 30 s,
  at most 4 of them, then get `429` with `Retry-After`.
- **Reasoning.** Efforts `none` (default), `low`, `medium`, `xhigh` (`high` is
  an alias). Thinking shares `max_tokens` with the answer, so each effort gets a
  thinking budget (1K / 4K / 8K), cut further when needed to keep 4K of
  `max_tokens` for the reply. A budget of 0 is never sent: Strata reads it as
  unlimited.
- **Safety.** It loads only if the GPU has under 2 GB in use and 8 GB of RAM is
  available, and stops the engine if available RAM falls below 3 GB. A
  disconnecting client cancels its request but keeps the engine and its cache.
- **Size.** Request bodies are capped at 1 MiB. Real code and prose run ~4 bytes
  per token, so a full 128K window is about 0.5 MiB.
- **Trust.** No authentication, like Ollama. Keep it on a trusted LAN.

## Sharing the GPU with Ollama (optional)

FlashNext and a large Ollama model do not fit on the card together, so whichever
is asked last gets the GPU. Ollama decides how many layers fit when it loads, so
the swap has to happen before the request reaches it:

1. Move the GPU Ollama to loopback (`OLLAMA_HOST=127.0.0.1:11436`) and install
   [`runit/ollama-proxy`](runit/ollama-proxy/run), which takes over port 11434.
2. Before an inference request, the proxy calls the gateway's loopback-only
   `POST /release`. An idle engine unloads at once, a busy one after its current
   request, and the proxy waits for the VRAM before forwarding unchanged.
   Listings, pulls and uploads pass straight through, and a gateway failure
   never blocks Ollama.
3. In the other direction the gateway asks Ollama to unload its GPU models
   before loading (`--ollama-url`, default `http://127.0.0.1:11436`; empty
   disables).

## Lessons from running agents on it

- **Output limits matter more than speed.** With a 4K output cap an agent writing
  a 150-line file ran out of tokens three times; 16K fixed it.
- **Keep the conversation cache on.** With one prompt cache, any side request
  costs the agent a full re-read of its context.
- **Queue, don't reject.** Agents and their helpers overlap; a 429 for a request
  that would have waited ten seconds just causes retries.
- **Restarts need `SO_REUSEADDR`.** Without it, sockets in TIME_WAIT from a
  just-stopped engine made the port check fail for about a minute.

## Repository

```
flashnext/   gateway (server.py), Ollama proxy, smoke test, llama.cpp launcher, tests
strata/      install.sh, launch.sh, env.sh, engine config templates, download and prepare scripts
runit/       service scripts for the gateway and the optional Ollama proxy
term-llm/    provider entry
bench/       benchmark harness, raw results, methodology
provenance/  pinned metadata, weight and artifact hashes, Strata build info
docs/        the llama.cpp backend this replaced
```

`make check` runs the unit tests, script syntax checks and shellcheck without a
GPU, model or network; CI runs it on every push.

## Provenance

| | Pin |
|---|---|
| Weights | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` @ `ed59f92`, `IQ3_XXS/`, 75.84 GB in 2 shards |
| MTP draft source | `Qwen/Qwen3.8-Flash-Next` @ `de4b8e4`, quantised locally to Q2_0 |
| Strata | v0.2.9: Control 0.1.55, engine 0.1.55 CUDA 13 (bundles its CUDA libraries) |
| gguf-py | llama.cpp `3cf0325` (packing tools only) |

Full hashes are in `strata/install.sh` and [`provenance/`](provenance/).

## Licence

MIT for this repository ([LICENSE](LICENSE)). Strata is MIT, the ISTA-DASLab
weights are Apache-2.0, and llama.cpp's `gguf-py` is MIT.
