"""Ollama front door: frees the GPU from an idle FlashNext before inference.

Ollama decides how many layers fit on the GPU at load time and has no pre-load
hook, so unloading FlashNext after Ollama starts is too late: the model would
land mostly on CPU. This proxy owns Ollama's public port. For inference
requests it first asks the FlashNext gateway to release the GPU (immediate if
idle, after the running request if busy), then forwards the request
byte-for-byte. Everything else streams straight through. A FlashNext failure
never blocks Ollama.
"""
import argparse
import json
import time

from aiohttp import ClientConnectionResetError, ClientSession, ClientTimeout, web

INFERENCE_PATHS = frozenset({
    "/api/generate", "/api/chat", "/api/embed", "/api/embeddings",
    "/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/v1/responses", "/v1/messages",
})
HOP_BY_HOP = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
                        "trailers", "transfer-encoding", "upgrade", "host", "content-length"})
CLIENT = web.AppKey("client", ClientSession)
SETTINGS = web.AppKey("settings", dict)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} ollama-proxy: {msg}", flush=True)


def wants_gpu(path, raw):
    """True when this request may make Ollama load or run a model."""
    if path not in INFERENCE_PATHS:
        return False
    try:
        body = json.loads(raw or b"{}")
    except ValueError:
        return False            # Ollama will reject it; do not evict anything for that
    if not isinstance(body, dict):
        return False
    # `{"model": m, "keep_alive": 0}` with no input is Ollama's unload request.
    unload = body.get("keep_alive") in (0, "0", "0s", "0m")
    has_input = any(body.get(k) for k in ("prompt", "messages", "input"))
    return not (unload and not has_input)


async def release_flashnext(app):
    s = app[SETTINGS]
    t0 = time.monotonic()
    try:
        async with app[CLIENT].post(s["release_url"], params={"wait": str(s["release_wait"])},
                                    timeout=ClientTimeout(total=s["release_wait"] + 30)) as r:
            info = await r.json(content_type=None)
        if info.get("released"):
            log(f"released FlashNext GPU in {time.monotonic() - t0:.1f}s")
        elif info.get("busy"):
            log("FlashNext still busy after wait; forwarding anyway")
        return info
    except Exception as error:   # FlashNext down/unreachable must not block Ollama
        log(f"release check failed ({type(error).__name__}: {error}); forwarding anyway")
        return None


async def proxy(request):
    app = request.app
    path = request.rel_url.path
    data = None
    if request.method == "POST" and path in INFERENCE_PATHS:
        data = await request.read()
        if wants_gpu(path, data):
            await release_flashnext(app)
    elif request.can_read_body:
        data = request.content    # stream uploads such as /api/blobs without buffering
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    async with app[CLIENT].request(request.method, app[SETTINGS]["upstream"] + str(request.rel_url),
                                   headers=headers, data=data, allow_redirects=False) as upstream:
        response = web.StreamResponse(status=upstream.status, reason=upstream.reason)
        for k, v in upstream.headers.items():
            if k.lower() not in HOP_BY_HOP:
                response.headers.add(k, v)
        if upstream.content_length is not None:
            response.content_length = upstream.content_length
        await response.prepare(request)
        try:
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
        except (ConnectionError, ClientConnectionResetError):
            return response       # client left; leaving the block cancels Ollama's request
        await response.write_eof()
        return response


def application(upstream, release_url, release_wait=30.0):
    app = web.Application(client_max_size=512 * 1024 * 1024)   # inference bodies can carry base64 images
    app[SETTINGS] = {"upstream": upstream.rstrip("/"), "release_url": release_url, "release_wait": release_wait}

    async def lifecycle(app):
        async with ClientSession(timeout=ClientTimeout(total=None, sock_connect=10), auto_decompress=False) as client:
            app[CLIENT] = client
            yield

    app.cleanup_ctx.append(lifecycle)
    app.router.add_route("*", "/{tail:.*}", proxy)
    return app


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=11434)
    p.add_argument("--upstream", default="http://127.0.0.1:11436")
    p.add_argument("--release-url", default="http://127.0.0.1:18098/release")
    p.add_argument("--release-wait", type=float, default=30.0,
                   help="seconds to wait for a busy FlashNext request to finish before forwarding anyway")
    a = p.parse_args()
    log(f"listening {a.host}:{a.port} -> {a.upstream}; release via {a.release_url}")
    web.run_app(application(a.upstream, a.release_url, a.release_wait), host=a.host, port=a.port,
                access_log=None, print=None)
