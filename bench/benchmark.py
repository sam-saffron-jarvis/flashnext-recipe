#!/usr/bin/env python3
"""Opt-in sequential Strata benchmark against a directly launched engine.

Default is a dry plan. --run AND --i-confirm-gpu-is-free authorize requests.
--measure-load additionally starts the runtime's launch.sh as an owned child,
and closes only that child afterward. It never unloads or changes any other
service, so stop the flashnext gateway first (it owns the GPU and port 18199).
Logs and results are written under the runtime root, never the checkout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(os.environ.get("STRATA_ROOT", Path.home() / "strata-runtime"))


def facts(target: int, instruction: str) -> tuple[str, int]:
    # Same synthetic-line approach as the FlashNext repro. Approximation only;
    # actual usage.prompt_tokens and timings.prompt_n are authoritative.
    count = max(1, math.ceil(target / 34))
    salt = secrets.randbelow(1 << 30)
    nonce = secrets.token_hex(16)
    text = f"Unique trial {nonce}; salt {salt}.\n" + "\n".join(
        f"Fact {i}: item {i} salt {salt} code {'ABCDEFGH'[i % 8]}{(i * 7919 + salt) % 9973} day {i % 28}."
        for i in range(count)
    )
    return text + "\n\n" + instruction, count


def events(response):
    """Decode SSE events, including multi-line data. No inference in this helper."""
    data = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                payload = "\n".join(data)
                data = []
                if payload == "[DONE]":
                    return
                yield json.loads(payload)
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data and "\n".join(data) != "[DONE]":
        yield json.loads("\n".join(data))


def call(args, body: dict, label: str) -> dict:
    body = {"model": args.model, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0,
            "reasoning_effort": "none", **body}
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("STRATA_API_KEY"):
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(args.url, data=json.dumps(body).encode(), headers=headers)
    start = time.monotonic()
    first = last = None
    usage, timings, finish = {}, {}, None
    chunks, chars, reasoning_chars, tools = 0, 0, 0, {}
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for event in events(response):
                if "error" in event:
                    raise RuntimeError(str(event["error"]))
                usage = event.get("usage") or usage
                timings = event.get("timings") or timings
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    content, reasoning = delta.get("content") or "", delta.get("reasoning_content") or ""
                    if content or reasoning or delta.get("tool_calls"):
                        now = time.monotonic()
                        first = first or now
                        last = now
                        chunks += 1
                    chars += len(content)
                    reasoning_chars += len(reasoning)
                    finish = choice.get("finish_reason") or finish
                    for tc in delta.get("tool_calls") or []:
                        cur = tools.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                        cur["id"] = tc.get("id") or cur["id"]
                        fn = tc.get("function") or {}
                        cur["name"] += fn.get("name") or ""
                        cur["arguments"] += fn.get("arguments") or ""
        elapsed = time.monotonic() - start
        completed = usage.get("completion_tokens")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        row = {"label": label, "elapsed_s": elapsed,
               "ttft_s": first - start if first else None,
               "usage": usage, "timings": timings, "finish_reason": finish,
               "requested_max_tokens": body.get("max_tokens"),
               "early_stop": completed < body["max_tokens"] if completed is not None else None,
               "content_chars": chars, "reasoning_chars": reasoning_chars,
               "output_sse_chunks": chunks, "tool_calls": list(tools.values()),
               "cached_prompt_tokens": cached,
               "prompt_reuse_detected": (cached or timings.get("cache_n") or 0) > 0,
               "client_decode_tok_s_estimate": ((completed - 1) / (last - first)
                   if completed and completed > 1 and first and last and last > first else None)}
        if "prompt_tokens" in usage:
            row["prompt_tokens_over_ttft_estimate"] = usage["prompt_tokens"] / (first - start) if first else None
        return row
    except urllib.error.HTTPError as exc:
        return {"label": label, "elapsed_s": time.monotonic() - start,
                "error": f"HTTP {exc.code}", "body": exc.read().decode(errors="replace")[:2000]}
    except (OSError, ValueError, RuntimeError) as exc:
        return {"label": label, "elapsed_s": time.monotonic() - start, "error": str(exc)}


def origin(url: str) -> str:
    u = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((u.scheme, u.netloc, "", "", ""))


def measure_load(args, emit):
    # Only used under BOTH explicit run confirmations. Refuse an occupied port;
    # never attach to or stop an already running API/service.
    import socket
    u = urllib.parse.urlsplit(args.url)
    if u.hostname != "127.0.0.1" or u.port != 18199:
        raise SystemExit("--measure-load only supports the prepared 127.0.0.1:18199 launch")
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 18199))
    finally:
        s.close()
    log = (ROOT / "logs" / f"benchmark-cold-load-{int(time.time())}.log").open("w")
    start = time.monotonic()
    proc = subprocess.Popen([str(ROOT / "launch.sh"), "--context", args.launch_context], cwd=ROOT,
                            env={**os.environ, "STRATA_GPU_CONFIRMED_FREE": "1"},
                            stdout=log, stderr=subprocess.STDOUT)
    try:
        while time.monotonic() - start < args.timeout:
            if proc.poll() is not None:
                raise RuntimeError(f"owned launch exited {proc.returncode}; see {log.name}")
            try:
                with urllib.request.urlopen(origin(args.url) + "/health", timeout=3) as response:
                    status = json.load(response)
                if status.get("status") == "ok" and status.get("loaded") is True:
                    emit({"label": "process_cold_load", "elapsed_s": time.monotonic() - start,
                          "disk_cache": "uncontrolled; no global cache drop", "health": status,
                          "server_log": str(log.name)})
                    return proc, log
            except (OSError, ValueError):
                pass
            time.sleep(0.5)
        raise RuntimeError("timed out waiting for owned server readiness")
    except BaseException:
        proc.terminate()
        proc.wait(timeout=90)
        log.close()
        raise


def main():
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT, help="Strata runtime root (default $STRATA_ROOT or %(default)s)")
    ap.add_argument("--url", default="http://127.0.0.1:18199/v1/chat/completions")
    ap.add_argument("--model", default="strata")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--i-confirm-gpu-is-free", action="store_true")
    ap.add_argument("--measure-load", action="store_true")
    ap.add_argument("--launch-context", choices=["64k", "128k"], default="128k",
                    help="context preset used only with --measure-load")
    ap.add_argument("--smoke", action="store_true", help="also check streamed bounded thinking and tools")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--output", type=Path, help="JSONL (default results/benchmark-<time>.jsonl)")
    ap.add_argument("--self-test", action="store_true", help="pure CPU SSE parser/synthetic-prompt checks")
    args = ap.parse_args()
    ROOT = args.root
    if args.self_test:
        import io
        got = list(events(io.BytesIO(b'data: {"choices":[]}\n\ndata: [DONE]\n\n')))
        assert got == [{"choices": []}]
        p, n = facts(2048, "READY")
        q, _ = facts(2048, "READY")
        assert p != q and n == 61 and "Fact 60:" in p
        print("CPU-only self-test passed; no network or GPU used")
        return
    if args.rounds < 1:
        ap.error("--rounds must be positive")
    plan = {"prefill_target_tokens": [2048, 8192, 16384, 32768, 54000],
            "decode_prompt_target_tokens": [2048, 32768], "decode_max_tokens": 1000,
            "rounds": args.rounds, "thinking_for_speed": "none",
            "note": "targets use ~34 tokens/line; actual usage and server timings are authoritative"}
    if not args.run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2))
        return
    if not args.i_confirm_gpu_is_free:
        ap.error("--run also requires --i-confirm-gpu-is-free; do not benchmark during the live task")
    output = args.output or ROOT / "results" / f"benchmark-{int(time.time())}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    proc = log = None
    failures = 0
    with output.open("a") as result:
        def emit(row):
            row = {"at_unix": time.time(), **row}
            line = json.dumps(row, ensure_ascii=False)
            print(line, flush=True)
            result.write(line + "\n")
            result.flush()
        emit({"label": "metadata", "plan": plan, "url": args.url, "model": args.model})
        try:
            if args.measure_load:
                proc, log = measure_load(args, emit)
            for trial in range(args.rounds):
                for target in plan["prefill_target_tokens"]:
                    prompt, lines = facts(target, "Reply with exactly: READY")
                    row = call(args, {"messages": [{"role": "user", "content": prompt}], "max_tokens": 1}, f"prefill_{target}_r{trial+1}")
                    emit({**row, "target_prompt_tokens": target, "synthetic_lines": lines,
                          "prompt_header": prompt.splitlines()[0]})
                    failures += "error" in row
                for target in plan["decode_prompt_target_tokens"]:
                    prompt, lines = facts(target, "Write a long numbered list of observations about these facts. Continue for at least 2000 tokens. Do not summarize or conclude early.")
                    row = call(args, {"messages": [{"role": "user", "content": prompt}], "max_tokens": 1000}, f"decode_{target}_r{trial+1}")
                    emit({**row, "target_prompt_tokens": target, "synthetic_lines": lines,
                          "prompt_header": prompt.splitlines()[0]})
                    failures += "error" in row
            if args.smoke:
                row = call(args, {"messages": [{"role": "user", "content": "What is 17 times 23? Think briefly, then answer."}], "max_tokens": 512,
                                  "reasoning_effort": "medium", "reasoning_budget_tokens": 128}, "bounded_thinking_smoke")
                emit({**row, "reasoning_content_present": row.get("reasoning_chars", 0) > 0})
                failures += "error" in row
                row = call(args, {"messages": [{"role": "user", "content": "Call echo with text READY. Do not answer directly."}], "max_tokens": 256,
                                  "tools": [{"type": "function", "function": {"name": "echo", "description": "Echo text", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}],
                                  "tool_choice": "required"}, "streamed_tool_smoke")
                calls = row.get("tool_calls", [])
                try:
                    ok = row.get("finish_reason") == "tool_calls" and bool(calls) and calls[0]["name"] == "echo" and json.loads(calls[0]["arguments"]) == {"text": "READY"}
                except ValueError:
                    ok = False
                emit({**row, "tool_format_check_passed": ok})
                failures += not ok
        finally:
            if proc is not None:
                proc.terminate()  # ONLY the process this script started; server owns/cleans its engine.
                try:
                    proc.wait(timeout=90)
                finally:
                    log.close()
    if failures:
        raise SystemExit(f"{failures} request/smoke failures; inspect {output}")


if __name__ == "__main__":
    main()
