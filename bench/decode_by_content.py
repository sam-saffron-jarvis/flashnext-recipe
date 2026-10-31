"""Decode rate by kind of output, through the gateway.

Speculative decoding makes the rate depend on how predictable the output is, so
one synthetic number is misleading. For each prompt this streams 1,200 tokens and
records the engine's own rate and an independent client-side one:
(completion_tokens - 1) / (time of last delta - time of first delta).

  python3 bench/decode_by_content.py [gateway-url] > results.jsonl
"""
import json, sys, time, urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18098").rstrip("/") + "/v1/chat/completions"
WRITE_TOOL = [{"type": "function", "function": {"name": "write_file", "description": "Write a file",
               "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                              "required": ["path", "content"]}}}]
FILLER = "\n".join(f"// file pkg/{i}.go\nfunc Handler{i}(w http.ResponseWriter, r *http.Request) {{\n\tif err := validate(r); err != nil {{\n\t\thttp.Error(w, err.Error(), 400)\n\t\treturn\n\t}}\n\tjson.NewEncoder(w).Encode(map[string]int{{\"id\": {i}}})\n}}" for i in range(900))

CASES = [
    ("code (python module)", "none", [{"role": "user", "content": "Write a complete Python module implementing a thread-safe LRU cache with per-entry TTL, with docstrings, type hints and a unittest suite. Output only code."}], None),
    ("prose (essay)", "none", [{"role": "user", "content": "Write a long, detailed essay on the history of bridge engineering from Roman aqueducts to modern cable-stayed bridges."}], None),
    ("tool call (write_file, Go server)", "none", [{"role": "user", "content": "Use write_file to create server.go: a complete Go HTTP server with routing, JSON handlers for CRUD on a users resource, middleware for logging and graceful shutdown."}], WRITE_TOOL),
    ("JSON data", "none", [{"role": "user", "content": "Output a JSON array of 40 fictional users, each with id, name, email, address object, signup date and a list of 3 tags. JSON only."}], None),
    ("Hebrew prose", "none", [{"role": "user", "content": "כתוב חיבור ארוך ומפורט בעברית על ההיסטוריה של ירושלים."}], None),
    ("reasoning (xhigh, thinking on)", "xhigh", [{"role": "user", "content": "A 4-digit number has distinct digits, is divisible by 11, and its digits sum to 20. How many such numbers exist? Show your reasoning, then give the count."}], None),
    ("code with ~68K tokens of context", "none", [{"role": "user", "content": FILLER + "\n\nNow write a complete Python module implementing a thread-safe LRU cache with per-entry TTL, with docstrings, type hints and a unittest suite. Output only code."}], None),
]


def run(label, effort, messages, tools, max_tokens=1200):
    body = {"model": "flashnext", "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "reasoning_effort": effort, "temperature": 0}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic(); first = last = None; usage = timings = None; finish = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data: {"):
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("timings"):
                timings = d["timings"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                    now = time.monotonic(); first = first or now; last = now
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    n = (usage or {}).get("completion_tokens", 0)
    client = (n - 1) / (last - first) if first and last and last > first and n > 1 else None
    t = timings or {}
    acc = f"{t.get('draft_n_accepted')}/{t.get('draft_n')}" if t.get("draft_n") else "-"
    print(json.dumps({"case": label, "tokens": n, "finish": finish, "prompt_tokens": (usage or {}).get("prompt_tokens"),
                      "engine_tok_s": t.get("predicted_per_second"), "client_tok_s": round(client, 1) if client else None,
                      "draft_accept": acc, "wall_s": round(time.monotonic() - t0, 1)}, ensure_ascii=False), flush=True)


for case in CASES:
    run(*case)
