import asyncio
import json
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from .ollama_proxy import application, wants_gpu


class WantsGpuTests(unittest.TestCase):
    def test_inference_paths(self):
        self.assertTrue(wants_gpu("/api/chat", b'{"model":"m","messages":[{"role":"user","content":"hi"}]}'))
        self.assertTrue(wants_gpu("/v1/chat/completions", b'{"model":"m","messages":[]}'))
        self.assertTrue(wants_gpu("/api/generate", b'{"model":"m"}'))   # preload still needs the GPU
        self.assertTrue(wants_gpu("/api/embed", b'{"model":"m","input":"x"}'))

    def test_non_inference_and_unload(self):
        self.assertFalse(wants_gpu("/api/tags", b""))
        self.assertFalse(wants_gpu("/api/show", b'{"model":"m"}'))
        self.assertFalse(wants_gpu("/api/generate", b'{"model":"m","keep_alive":0}'))
        self.assertFalse(wants_gpu("/api/chat", b'{"model":"m","keep_alive":"0s","messages":[]}'))
        self.assertFalse(wants_gpu("/api/chat", b"not json"))


class ProxyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.events = []
        self.release_status = 200
        self.release_body = {"released": True, "loaded": False}

        async def ollama(request):
            body = await request.read()
            self.events.append(("ollama", request.method, request.path_qs, len(body),
                                request.headers.get("X-Test")))
            if request.path == "/api/chat":
                resp = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
                await resp.prepare(request)
                for i in range(3):
                    await resp.write(json.dumps({"i": i}).encode() + b"\n")
                    await asyncio.sleep(0.01)
                await resp.write_eof()
                return resp
            return web.json_response({"path": request.path, "bytes": len(body)}, headers={"X-Upstream": "yes"})

        async def release(request):
            self.events.append(("release", request.query.get("wait")))
            return web.json_response(self.release_body, status=self.release_status)

        up = web.Application(client_max_size=64 * 1024 * 1024)
        up.router.add_route("*", "/{tail:.*}", ollama)
        rel = web.Application()
        rel.router.add_post("/release", release)
        self.up = TestServer(up)
        self.rel = TestServer(rel)
        await self.up.start_server()
        await self.rel.start_server()
        self.client = TestClient(TestServer(application(str(self.up.make_url("")), str(self.rel.make_url("/release")), 7.0)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        await self.up.close()
        await self.rel.close()

    async def test_inference_releases_first_then_streams(self):
        r = await self.client.post("/api/chat", json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
                                   headers={"X-Test": "kept"})
        self.assertEqual(r.status, 200)
        lines = (await r.read()).decode().splitlines()
        self.assertEqual([json.loads(x)["i"] for x in lines], [0, 1, 2])
        self.assertEqual(self.events[0], ("release", "7.0"))
        self.assertEqual(self.events[1][:3], ("ollama", "POST", "/api/chat"))
        self.assertEqual(self.events[1][4], "kept")

    async def test_metadata_passes_through_without_release(self):
        r = await self.client.get("/api/tags?x=1")
        self.assertEqual(r.status, 200)
        self.assertEqual(r.headers.get("X-Upstream"), "yes")
        self.assertEqual(self.events, [("ollama", "GET", "/api/tags?x=1", 0, None)])

    async def test_unload_request_does_not_evict_flashnext(self):
        r = await self.client.post("/api/generate", json={"model": "qwen", "keep_alive": 0})
        self.assertEqual(r.status, 200)
        self.assertEqual([e[0] for e in self.events], ["ollama"])

    async def test_release_failure_still_forwards(self):
        self.release_status, self.release_body = 500, {"error": "boom"}
        r = await self.client.post("/v1/chat/completions", json={"model": "qwen", "messages": []})
        self.assertEqual(r.status, 200)
        self.assertEqual([e[0] for e in self.events], ["release", "ollama"])

    async def test_release_unreachable_still_forwards(self):
        await self.rel.close()
        r = await self.client.post("/api/embed", json={"model": "e", "input": "x"})
        self.assertEqual(r.status, 200)
        self.assertEqual([e[0] for e in self.events], ["ollama"])

    async def test_large_upload_streams_through(self):
        blob = b"x" * (5 * 1024 * 1024)
        r = await self.client.post("/api/blobs/sha256:abc", data=blob)
        self.assertEqual(r.status, 200)
        self.assertEqual((await r.json())["bytes"], len(blob))
        self.assertEqual([e[0] for e in self.events], ["ollama"])
