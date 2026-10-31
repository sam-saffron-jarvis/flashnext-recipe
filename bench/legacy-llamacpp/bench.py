#!/usr/bin/env python3
"""FlashNext appliance benchmark. Gateway-only (no /tokenize, no calibration).

Prefill sweep: synthetic ~34-token lines, fresh salt per request (no prompt-cache
hits), max_tokens=8, reasoning_effort=none; prefill = usage.prompt_tokens / TTFT.
Decode: ~1000 generated tokens at short context and ~32k context.
Cold load: last step; gateway idle-unloads after 600s so no service restart is
needed (restart would kill the shell session the bench runs in).
"""
import json, random, subprocess, threading, time, urllib.error, urllib.request

GW = "http://127.0.0.1:18098"
LOG = "<state-dir>/assets/appliance-engine.log"
PEAK = [0]


def sampler(stop):
    while not stop.is_set():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            PEAK[0] = max([PEAK[0]] + [int(x) for x in out.split()])
        except Exception:
            pass
        time.sleep(0.3)


def line(salt, i):
    return (f"Fact {i}: item {i} salt {salt} code X{i * 7919 % 100000} "
            f"day {i % 97} ledger {pow(i + salt, 3, 999983)} status "
            f"{'settled pending cleared'[3 * (i % 4):3 * (i % 4) + 3].strip()}")


def run(label, prompt, max_tokens, retries=8):
    body = {"model": "flashnext",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": True, "reasoning_effort": "none",
            "stream_options": {"include_usage": True}}
    off = __import__("os").path.getsize(LOG)
    req = urllib.request.Request(GW + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = ttft = None
    chunks = usage = {}
    finish = err = None
    for attempt in range(retries):
        t0 = time.monotonic(); ttft = None; chunks = 0; usage = {}; finish = None
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                for raw in r:
                    if not raw.startswith(b"data: "):
                        continue
                    p = raw[6:].strip()
                    if p == b"[DONE]":
                        break
                    d = json.loads(p)
                    if d.get("usage"):
                        usage = d["usage"]
                    ch = (d.get("choices") or [{}])[0]
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    if ((ch.get("delta") or {}).get("content")
                            or (ch.get("delta") or {}).get("reasoning_content")):
                        chunks += 1
                        if ttft is None:
                            ttft = time.monotonic() - t0
            err = None
            break
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}"
            if e.code != 429:
                break
            time.sleep(15)
        except Exception as e:
            err = repr(e)
            time.sleep(5)
    total = time.monotonic() - t0
    eng_ms = eng_tok = None
    try:
        import re
        tail = open(LOG, "rb").read()[off:].decode(errors="replace")
        m = re.findall(r"prompt eval time =\s*(\d+\.\d+) ms\s*/\s*(\d+) tokens", tail)
        if m:
            eng_ms, eng_tok = m[-1]
            eng_ms, eng_tok = float(eng_ms), int(eng_tok)
    except Exception:
        pass
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens") or chunks
    res = {"label": label, "err": err, "built_lines": prompt.count("\n") + 1,
           "prompt_tokens": pt, "completion_tokens": ct, "finish": finish,
           "ttft_s": round(ttft, 2) if ttft else None,
           "total_s": round(total, 1), "peak_mib": PEAK[0],
           "prefill_tok_s": round(pt / ttft, 1) if pt and ttft else None,
           "decode_tok_s": round((ct - 1) / (total - ttft), 1) if ttft and ct > 1 and total > ttft else None,
           "eng_prompt_ms": eng_ms, "eng_prompt_tokens": eng_tok}
    print(json.dumps(res), flush=True)
    time.sleep(2)
    return res


def main():
    stop = threading.Event()
    th = threading.Thread(target=sampler, args=(stop,), daemon=True)
    th.start()
    rnd = random.Random()
    # prefill sweep, 2 runs each, ~8-token generations
    for target in (2000, 8000, 16000, 32000, 54000):
        for rep in range(2):
            salt = rnd.randrange(10 ** 9)
            n = max(int(target / 34), 1)
            prompt = "\n".join(line(salt, i) for i in range(n)) + \
                "\n\nReply with exactly: READY"
            run(f"prefill_{target // 1000}k_r{rep}", prompt, 8)
    # decode ~1000 tokens at short context
    for rep in range(2):
        salt = rnd.randrange(10 ** 9)
        run(f"decode_short_r{rep}",
            "Write a detailed technical survey of the design tradeoffs in ten "
            f"unrelated systems (salt {salt}): database indexes, hash routers, "
            "garbage collectors, video codecs, journaling filesystems, DNS "
            "resolvers, ray tracing, CRDTs, consensus protocols, cache "
            "coherence. Number each section. Be concrete.", 1000)
    # decode ~1000 tokens at ~32k context with a retrieval marker
    for rep in range(2):
        salt = rnd.randrange(10 ** 9)
        n = int(32000 / 34)
        parts = [line(salt, i) for i in range(n)]
        parts.insert(n // 2, f"IMPORTANT MARKER: the launch code for salt {salt} "
                             f"is ZEBRA-{salt % 99991}-TULIP.")
        prompt = "\n".join(parts) + ("\n\nEarlier in this document a launch code "
                                     "was recorded. State the code, then explain in "
                                     "detail how you located and verified it, and "
                                     "why long-context retrieval matters for agents. "
                                     "Be verbose.")
        run(f"decode_32k_r{rep}", prompt, 1000)
    # cold load: gateway idle-unloads after 600s; wait for unloaded (or 700s cap)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 700:
        try:
            if not json.load(urllib.request.urlopen(GW + "/status", timeout=5))["loaded"]:
                break
        except Exception:
            break
        time.sleep(10)
    print(json.dumps({"label": "cold_load_note", "waited_s": round(time.monotonic() - t0),
                      "loaded_before": json.load(urllib.request.urlopen(GW + "/status", timeout=5))["loaded"]}),
          flush=True)
    r = run("cold_load", "Say hello in exactly five words.", 64)
    r["was_loaded_before"] = None
    stop.set()
    print(json.dumps({"label": "summary", "overall_peak_mib": PEAK[0]}), flush=True)


if __name__ == "__main__":
    main()
