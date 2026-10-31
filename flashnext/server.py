"""Lazy, bounded OpenAI chat gateway for FlashNext. Listing models never loads weights."""
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
BACKEND = "llamacpp"
# Engine profiles. llamacpp is the original UD-IQ4_XS recipe (kept for
# rollback); strata serves ISTA-DASLab IQ3_XXS ~5x faster decode and ~9x
# faster prefill on the same card, with 128K context.
BACKENDS = {
    "llamacpp": {"upstream": "http://127.0.0.1:18099", "port": 18099, "context": 65536, "max_output": 4096,
                 "budgets": {"low": 512, "medium": 1024, "xhigh": 2560}, "reserve": 1536},
    "strata": {"upstream": "http://127.0.0.1:18199", "port": 18199, "context": 131072, "max_output": 16384,
               "budgets": {"low": 1024, "medium": 4096, "xhigh": 8192}, "reserve": 4096},
}


def select_backend(name):
    global BACKEND, UPSTREAM, CONTEXT_WINDOW, MAX_OUTPUT_TOKENS, REASONING_BUDGETS, ANSWER_RESERVE_TOKENS
    b = BACKENDS[name]
    BACKEND, UPSTREAM = name, b["upstream"]
    CONTEXT_WINDOW, MAX_OUTPUT_TOKENS = b["context"], b["max_output"]
    REASONING_BUDGETS, ANSWER_RESERVE_TOKENS = dict(b["budgets"]), b["reserve"]
CLIENT = web.AppKey("client", ClientSession)


def available_memory():
    return int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))) // 1024


GPU_FREE_BELOW_MIB = 2000   # the host display alone holds ~1.2 GB


async def gpu_used_mib():
    proc = await asyncio.create_subprocess_exec("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE)
    out, _ = await asyncio.wait_for(proc.communicate(), 5)
    if proc.returncode:
        raise RuntimeError("nvidia-smi failed")
    return max(map(int, out.split()))


async def resource_check():
    if await gpu_used_mib() >= GPU_FREE_BELOW_MIB or available_memory() < 8192:
        raise RuntimeError("GPU busy or insufficient RAM; no other service will be evicted")


OLLAMA_GPU_URL = "http://127.0.0.1:11436"   # the GPU Ollama, behind ollama_proxy; "" disables


async def free_gpu_from_ollama(session, url=None, timeout=30):
    """Ask the GPU Ollama to unload whatever it holds on the GPU, then wait for the VRAM.

    Ollama unloads a keep_alive:0 model only after its in-flight requests finish, so
    a running answer is never cut off; we just wait (bounded) for the memory.
    Talks to Ollama directly, not through the proxy, so it cannot trigger a
    release of FlashNext itself. Returns True when the GPU is free afterwards.
    """
    url = OLLAMA_GPU_URL if url is None else url
    if not url:
        return False
    try:
        async with session.get(url + "/api/ps", timeout=ClientTimeout(total=5)) as r:
            models = (await r.json()).get("models") or []
    except (OSError, asyncio.TimeoutError, ValueError):
        return False
    held = [m["name"] for m in models if m.get("size_vram")]
    if not held:
        return False
    for name in held:
        try:
            async with session.post(url + "/api/generate", json={"model": name, "keep_alive": 0},
                                    timeout=ClientTimeout(total=10)) as r:
                await r.read()
        except (OSError, asyncio.TimeoutError):
            pass
    print(f"flashnext: asked Ollama to unload {', '.join(held)}", flush=True)
    return await wait_for_gpu_free(timeout)


async def wait_for_gpu_free(timeout=15):
    """After stopping the engine, wait until the driver has actually released its VRAM."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if await gpu_used_mib() < GPU_FREE_BELOW_MIB:
                return True
        except (RuntimeError, OSError, asyncio.TimeoutError):
            return False
        await asyncio.sleep(0.5)
    return False


class Engine:
    def __init__(self, root, idle=120, queue_seconds=30, max_queue=4):
        self.root, self.idle = Path(root), idle
        self.process = None
        self.log = None
        self.lease = None
        self.busy = False
        self.last = time.monotonic()
        # One request runs at a time. Others wait FIFO (asyncio.Lock wakes
        # waiters in order) for up to queue_seconds, at most max_queue of them,
        # instead of failing instantly with 429. Short side requests such as
        # term-llm Guardian reviews then run between agent turns.
        self.slot = asyncio.Lock()
        self.queue_seconds, self.max_queue = queue_seconds, max_queue
        self.waiting = 0

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
            sock.bind(("127.0.0.1", BACKENDS[BACKEND]["port"]))
        self.lease = open("/tmp/local-llm-optional-gpu.lock", "a")
        fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if await gpu_used_mib() >= GPU_FREE_BELOW_MIB:
                # Newest request wins the single GPU: evict Ollama's models
                # (after any answer in progress), as Ollama's proxy evicts us.
                await free_gpu_from_ollama(session)
        except (RuntimeError, OSError, asyncio.TimeoutError):
            pass
        await resource_check()
        if BACKEND == "strata":
            self.log = self.root.joinpath("logs", "gateway-engine.log").open("ab")
            # launch.sh keeps its own read-only GPU guard; admission above already
            # established that the GPU is free.
            argv, env, ready = ["bash", str(self.root / "launch.sh")], dict(os.environ, STRATA_GPU_CONFIRMED_FREE="1"), "/health"
        else:
            self.log = self.root.joinpath("appliance-engine.log").open("ab")
            argv, env, ready = ["sh", str(Path(__file__).with_name("launch.sh"))], dict(os.environ, FLASHNEXT_ROOT=str(self.root)), "/v1/models"
        self.process = await asyncio.create_subprocess_exec(
            *argv, env=env, start_new_session=True, stdout=self.log, stderr=self.log)
        for _ in range(300):
            if self.process.returncode is not None:
                break
            try:
                async with session.get(UPSTREAM + ready, timeout=ClientTimeout(total=2)) as r:
                    data = await r.json()
                    if r.status == 200 and (data.get("loaded") is True if BACKEND == "strata"
                                            else any(m["id"] == MODEL for m in data["data"])):
                        return
            except (OSError, asyncio.TimeoutError, ValueError, KeyError):
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
    if BACKEND == "strata":
        # Strata's own effort ladder is none/low/medium/high; its high is the
        # template's xhigh. It reads reasoning_budget_tokens natively and does
        # its own wrap-up injection, so no budget message is sent.
        body["reasoning_effort"] = "high" if effort == "xhigh" else effort
        if kwargs:
            body["chat_template_kwargs"] = kwargs
    else:
        body["reasoning_effort"] = effort
        body["chat_template_kwargs"] = dict(kwargs, enable_thinking=enabled, reasoning_effort=effort)
    if enabled and "reasoning_budget_tokens" not in body:
        # Thinking shares max_tokens with the visible answer/tool call. Without
        # a budget, xhigh can spend the whole allowance thinking and the client
        # gets finish_reason=length with nothing usable. llama.cpp force-closes
        # the think block at the budget and injects the message first.
        limit = body.get("max_tokens", MAX_OUTPUT_TOKENS)
        room = limit - ANSWER_RESERVE_TOKENS
        if room < 256:
            # Never send 0: Strata reads 0 as "uncapped", llama.cpp as "no thinking".
            room = max(1, limit // 2)
        budget = min(REASONING_BUDGETS[effort], room)
        body["reasoning_budget_tokens"] = budget
        if BACKEND != "strata":
            body.setdefault("reasoning_budget_message", REASONING_BUDGET_MESSAGE)
    return effort

def application(engine):
    app = web.Application(client_max_size=1024 * 1024)

    async def models(request):
        return web.json_response({"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "local-llm", "context_length": CONTEXT_WINDOW, "top_provider": {"max_completion_tokens": MAX_OUTPUT_TOKENS}, "reasoning_efforts": list(REASONING_EFFORTS), "default_reasoning_effort": DEFAULT_REASONING_EFFORT}]})

    async def status(request):
        return web.json_response({"loaded": bool(engine.process and engine.process.returncode is None), "busy": engine.busy, "queued": engine.waiting, "queue_seconds": engine.queue_seconds, "max_queue": engine.max_queue, "idle_seconds": engine.idle, "configured_context_window": CONTEXT_WINDOW, "backend": BACKEND, "max_output_tokens": MAX_OUTPUT_TOKENS})

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
        if engine.slot.locked() and engine.waiting >= engine.max_queue:
            return web.json_response({"error": f"FlashNext busy; queue full ({engine.max_queue} waiting)"},
                                     status=429, headers={"Retry-After": "5"})
        engine.waiting += 1
        try:
            await asyncio.wait_for(engine.slot.acquire(), engine.queue_seconds)
        except asyncio.TimeoutError:
            return web.json_response({"error": f"FlashNext busy; waited {engine.queue_seconds:g}s in queue"},
                                     status=429, headers={"Retry-After": "5"})
        finally:
            engine.waiting -= 1
        if request.transport is None or request.transport.is_closing():
            # The client gave up while queued; do not spend the GPU on it.
            engine.slot.release()
            return web.Response(status=499)
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
            engine.slot.release()

    async def release(request):
        """Free the GPU for another local service (the Ollama proxy calls this).

        Loopback only. Idle engine: unload now. Busy engine: wait up to
        ?wait= seconds (default queue_seconds) for the slot, then unload.
        Holding the slot while stopping keeps a racing chat request from
        reloading mid-stop; it simply queues and reloads afterwards.
        """
        if request.remote not in ("127.0.0.1", "::1"):
            return web.json_response({"error": "release is loopback-only"}, status=403)
        try:
            wait = float(request.query.get("wait", engine.queue_seconds))
        except ValueError:
            return web.json_response({"error": "wait must be a number"}, status=422)
        if not (engine.process and engine.process.returncode is None):
            return web.json_response({"released": False, "loaded": False})
        try:
            await asyncio.wait_for(engine.slot.acquire(), max(0.0, wait))
        except asyncio.TimeoutError:
            return web.json_response({"released": False, "loaded": True, "busy": True}, status=409)
        try:
            loaded = bool(engine.process and engine.process.returncode is None)
            if loaded:
                await engine.stop()
                await wait_for_gpu_free()
            return web.json_response({"released": loaded, "loaded": False})
        finally:
            engine.last = time.monotonic()
            engine.slot.release()

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
    app.router.add_post("/release", release)
    return app


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18098)
    p.add_argument("--idle", type=int, default=120)
    p.add_argument("--backend", choices=sorted(BACKENDS), default="strata")
    p.add_argument("--queue-seconds", type=float, default=30, help="how long a request may wait for the single slot")
    p.add_argument("--max-queue", type=int, default=4, help="requests allowed to wait at once; more get 429")
    p.add_argument("--ollama-url", default=OLLAMA_GPU_URL,
                   help="GPU Ollama to evict before loading (empty string disables)")
    args = p.parse_args()
    OLLAMA_GPU_URL = args.ollama_url.rstrip("/")
    if args.idle < 1:
        p.error("idle must be positive")
    if args.queue_seconds < 0 or args.max_queue < 0:
        p.error("queue settings must be non-negative")
    select_backend(args.backend)
    web.run_app(application(Engine(args.root, args.idle, args.queue_seconds, args.max_queue)),
                host=args.host, port=args.port, access_log=None)
