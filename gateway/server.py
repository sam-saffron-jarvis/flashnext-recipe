"""Lazy, bounded OpenAI chat proxy. Listing models never loads weights."""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import time

from aiohttp import ClientConnectionResetError, ClientSession, ClientTimeout, web

MODEL = "flashnext"
CONTEXT_WINDOW = 65536
MAX_OUTPUT_TOKENS = 4096
REASONING_EFFORTS = ("none", "low", "medium", "xhigh")
DEFAULT_REASONING_EFFORT = "none"
# Thinking-token ceilings per effort; always leave ANSWER_RESERVE_TOKENS of
# max_tokens for the answer or tool call.
REASONING_BUDGETS = {"low": 512, "medium": 1024, "xhigh": 2560}
ANSWER_RESERVE_TOKENS = 1536
REASONING_BUDGET_MESSAGE = "\n\n(Thinking budget reached. Stop deliberating; give the answer or make the next tool call now.)\n"
UPSTREAM = "http://127.0.0.1:18099"
CLIENT = web.AppKey("client", ClientSession)


def available_memory():
    return int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))) // 1024


async def resource_check():
    proc = await asyncio.create_subprocess_exec("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE)
    out, _ = await asyncio.wait_for(proc.communicate(), 5)
    if proc.returncode or max(map(int, out.split())) >= 2000 or available_memory() < 8192:
        raise RuntimeError("GPU busy or insufficient RAM; no other service will be evicted")


class Engine:
    def __init__(self, root, idle=120):
        self.root, self.idle = Path(root), idle
        self.process = None
        self.log = None
        self.lease = None
        self.busy = False
        self.last = time.monotonic()

    async def stop(self):
        if self.process:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.process.wait(), 10)
            except asyncio.TimeoutError:
                pass
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await self.process.wait()
            self.process = None
        if self.log:
            self.log.close()
            self.log = None

        if self.lease:
            self.lease.close()
            self.lease = None

    async def start(self, session):
        if self.process and self.process.returncode is None:
            return
        await self.stop()
        # Refuse *any* existing listener, not just healthy engines.
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 18099))
        self.lease = open("/tmp/local-llm-optional-gpu.lock", "a")
        fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        await resource_check()
        self.log = self.root.joinpath("appliance-engine.log").open("ab")
        self.process = await asyncio.create_subprocess_exec(
            "sh", str(Path(__file__).with_name("launch.sh")),
            env=dict(os.environ, FLASHNEXT_ROOT=str(self.root)),
            start_new_session=True, stdout=self.log, stderr=self.log)
        for _ in range(300):
            if self.process.returncode is not None:
                break
            try:
                async with session.get(UPSTREAM + "/v1/models", timeout=ClientTimeout(total=2)) as r:
                    data = await r.json()
                    if r.status == 200 and any(m["id"] == MODEL for m in data["data"]):
                        return
            except (OSError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(1)
        raise RuntimeError("engine did not load expected model")

    async def monitor(self):
        while True:
            await asyncio.sleep(1)
            if self.process and (available_memory() < 3072 or (not self.busy and time.monotonic() - self.last >= self.idle)):
                await self.stop()



def configure_reasoning(body):
    """Match the retained template's real controls, not a generic effort ladder.

    none preserves the deployed non-thinking default. low/medium/xhigh enable
    thinking with the template's own instructions. high is only a legacy alias
    for xhigh, never advertised as an additional distinct mode.
    """
    effort = body.get("reasoning_effort", DEFAULT_REASONING_EFFORT)
    if not isinstance(effort, str):
        raise ValueError("reasoning_effort must be a string: none, low, medium, xhigh")
    effort = effort.strip().lower() or DEFAULT_REASONING_EFFORT
    if effort == "high":
        effort = "xhigh"
    if effort not in REASONING_EFFORTS:
        raise ValueError("supported reasoning_effort values: none, low, medium, xhigh (high aliases xhigh)")
    kwargs = body.get("chat_template_kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("chat_template_kwargs must be an object")
    enabled = effort != "none"
    if "enable_thinking" in kwargs and kwargs["enable_thinking"] is not enabled:
        raise ValueError("enable_thinking conflicts with reasoning_effort; select one listed effort")
    if "reasoning_effort" in kwargs:
        nested = kwargs["reasoning_effort"]
        if not isinstance(nested, str) or ("xhigh" if nested == "high" else nested) != effort:
            raise ValueError("chat_template_kwargs.reasoning_effort conflicts with reasoning_effort")
    if any(key in body for key in ("enable_thinking", "thinking", "reasoning")):
        raise ValueError("use reasoning_effort to select this endpoint's thinking mode")
    body["reasoning_effort"] = effort
    body["chat_template_kwargs"] = dict(kwargs, enable_thinking=enabled, reasoning_effort=effort)
    if enabled and "reasoning_budget_tokens" not in body:
        # Thinking shares max_tokens with the visible answer/tool call. Without
        # a budget, xhigh can spend the whole allowance thinking and the client
        # gets finish_reason=length with nothing usable. llama.cpp force-closes
        # the think block at the budget and injects the message first.
        limit = body.get("max_tokens", MAX_OUTPUT_TOKENS)
        budget = min(REASONING_BUDGETS[effort], max(0, limit - ANSWER_RESERVE_TOKENS))
        body["reasoning_budget_tokens"] = budget
        body.setdefault("reasoning_budget_message", REASONING_BUDGET_MESSAGE)
    return effort

def application(engine):
    app = web.Application(client_max_size=1024 * 1024)

    async def models(request):
        return web.json_response({"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "local-llm", "context_length": CONTEXT_WINDOW, "top_provider": {"max_completion_tokens": MAX_OUTPUT_TOKENS}, "reasoning_efforts": list(REASONING_EFFORTS), "default_reasoning_effort": DEFAULT_REASONING_EFFORT}]})

    async def status(request):
        return web.json_response({"loaded": bool(engine.process and engine.process.returncode is None), "busy": engine.busy, "idle_seconds": engine.idle, "configured_context_window": CONTEXT_WINDOW})

    async def chat(request):
        try:
            body = await request.json()
            if body.get("model") != MODEL:
                return web.json_response({"error": "unknown model"}, status=404)
            if not isinstance(body.get("messages"), list) or not body["messages"]:
                raise ValueError()
            limit = body.get("max_tokens", body.get("max_completion_tokens", MAX_OUTPUT_TOKENS))
            if "max_completion_tokens" in body and body["max_completion_tokens"] != limit:
                raise ValueError()
            body.pop("max_completion_tokens", None)
            if type(limit) is not int or not 1 <= limit <= MAX_OUTPUT_TOKENS:
                raise ValueError()
            body["max_tokens"] = limit
        except (ValueError, TypeError, AttributeError):
            return web.json_response({"error": "invalid chat request"}, status=422)
        try:
            configure_reasoning(body)
        except ValueError as error:
            return web.json_response({"error": {"message": str(error), "type": "invalid_request_error", "param": "reasoning_effort"}}, status=422)
        if engine.busy:
            return web.json_response({"error": "FlashNext busy"}, status=429)
        engine.busy = True
        response = None
        try:
            await engine.start(app[CLIENT])
            async with app[CLIENT].post(UPSTREAM + "/v1/chat/completions", json=body) as upstream:
                response = web.StreamResponse(status=upstream.status, headers={"Content-Type": upstream.headers.get("Content-Type", "application/json")})
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    try:
                        await response.write(chunk)
                    except (ConnectionError, ClientConnectionResetError):
                        # The client went away (cancel, reload). Closing the
                        # upstream request cancels the llama.cpp task; the
                        # loaded engine and its prompt cache stay usable.
                        return response
                await response.write_eof()
                return response
        except Exception:
            await engine.stop()
            if response is not None and response.prepared:
                request.transport.close() if request.transport else None
                return response
            return web.json_response({"error": "FlashNext unavailable; check GPU contention and appliance-engine.log"}, status=503)
        finally:
            engine.busy = False
            engine.last = time.monotonic()

    async def lifecycle(app):
        async with ClientSession(timeout=ClientTimeout(total=600)) as client:
            app[CLIENT] = client
            task = asyncio.create_task(engine.monitor())
            try:
                yield
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await engine.stop()

    app.cleanup_ctx.append(lifecycle)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/status", status)
    app.router.add_post("/v1/chat/completions", chat)
    return app


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18098)
    p.add_argument("--idle", type=int, default=120)
    args = p.parse_args()
    if args.idle < 1:
        p.error("idle must be positive")
    web.run_app(application(Engine(args.root, args.idle)), host=args.host, port=args.port, access_log=None)
