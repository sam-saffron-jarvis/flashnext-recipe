import unittest
from unittest.mock import patch
from aiohttp.test_utils import TestClient, TestServer
from .server import Engine, application, resource_check, configure_reasoning, select_backend


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = Engine("/nonexistent", idle=1)
        self.client = TestClient(TestServer(application(self.engine)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_listing_does_not_load(self):
        r = await self.client.get("/v1/models")
        self.assertEqual((await r.json())["data"][0]["id"], "flashnext")
        self.assertIsNone(self.engine.process)
        r = await self.client.get("/v1/models")
        self.assertEqual((await r.json())["data"][0]["context_length"], 65536)
        r = await self.client.get("/status")
        status = await r.json()
        self.assertEqual(status["configured_context_window"], 65536)
        self.assertFalse(status["loaded"])
        r = await self.client.get("/v1/models")
        model = (await r.json())["data"][0]
        self.assertEqual(model["reasoning_efforts"], ["none", "low", "medium", "xhigh"])
        self.assertEqual(model["default_reasoning_effort"], "none")

    async def test_invalid_requests_rejected_before_load(self):
        for body, status in [({"model": "wrong"}, 404),
                             ({"model": "flashnext", "messages": [{}], "max_tokens": 99999}, 422),
                             ({"model": "flashnext", "messages": [{}], "max_completion_tokens": 99999}, 422),
                             ({"model": "flashnext", "messages": [{}], "max_tokens": 128, "max_completion_tokens": 99999}, 422)]:
            r = await self.client.post("/v1/chat/completions", json=body)
            self.assertEqual(r.status, status)
        self.assertIsNone(self.engine.process)

    async def test_unsupported_efforts_rejected_without_loading(self):
        for effort in ("minimal", "max", "off", "false", "invented", 1, None):
            r = await self.client.post("/v1/chat/completions", json={"model": "flashnext", "messages": [{}], "reasoning_effort": effort})
            self.assertEqual(r.status, 422)
            self.assertIn("reasoning_effort", (await r.json())["error"]["message"])
        self.assertIsNone(self.engine.process)

    async def test_contention_waits_then_times_out(self):
        import time
        self.engine.queue_seconds = 0.3
        await self.engine.slot.acquire()
        try:
            t0 = time.monotonic()
            r = await self.client.post("/v1/chat/completions", json={"model": "flashnext", "messages": [{}]})
            self.assertEqual(r.status, 429)
            self.assertEqual(r.headers.get("Retry-After"), "5")
            self.assertGreaterEqual(time.monotonic() - t0, 0.3)
            self.assertEqual(self.engine.waiting, 0)
        finally:
            self.engine.slot.release()

    async def test_queued_request_runs_when_slot_frees(self):
        import asyncio, time
        self.engine.queue_seconds = 5
        await self.engine.slot.acquire()
        with patch.object(self.engine, "start", side_effect=RuntimeError("stub engine")):
            t0 = time.monotonic()
            task = asyncio.create_task(self.client.post("/v1/chat/completions", json={"model": "flashnext", "messages": [{}]}))
            await asyncio.sleep(0.3)
            self.assertEqual(self.engine.waiting, 1)
            self.engine.slot.release()
            r = await task
            # 503 from the stubbed load proves it got past the queue rather than being refused.
            self.assertEqual(r.status, 503)
            self.assertGreaterEqual(time.monotonic() - t0, 0.3)
        self.assertFalse(self.engine.slot.locked())

    async def test_queued_requests_are_fifo(self):
        import asyncio
        self.engine.queue_seconds = 5
        done = []

        async def fake_start(session):
            await asyncio.sleep(0.1)   # hold the slot long enough to separate completions
            raise RuntimeError("stub engine")

        await self.engine.slot.acquire()
        with patch.object(self.engine, "start", side_effect=fake_start):
            tasks = []
            for i in range(3):
                t = asyncio.create_task(self.client.post(
                    "/v1/chat/completions", json={"model": "flashnext", "messages": [{"role": "user", "content": str(i)}]}))
                t.add_done_callback(lambda _t, i=i: done.append(i))
                tasks.append(t)
                await asyncio.sleep(0.05)
            self.assertEqual(self.engine.waiting, 3)
            self.engine.slot.release()
            statuses = [(await t).status for t in tasks]
        self.assertEqual(statuses, [503, 503, 503])
        self.assertEqual(done, [0, 1, 2])

    async def test_full_queue_rejects_immediately(self):
        import time
        self.engine.queue_seconds, self.engine.max_queue = 5, 0
        await self.engine.slot.acquire()
        try:
            t0 = time.monotonic()
            r = await self.client.post("/v1/chat/completions", json={"model": "flashnext", "messages": [{}]})
            self.assertEqual(r.status, 429)
            self.assertLess(time.monotonic() - t0, 1)
        finally:
            self.engine.slot.release()

    async def test_release_when_not_loaded_is_a_noop(self):
        r = await self.client.post("/release")
        self.assertEqual(await r.json(), {"released": False, "loaded": False})

    async def test_release_unloads_idle_engine(self):
        from unittest.mock import AsyncMock

        class Proc:
            returncode = None
        self.engine.process = Proc()

        async def fake_stop():
            self.engine.process = None
        with patch.object(self.engine, "stop", side_effect=fake_stop) as stop, \
             patch("flashnext.server.wait_for_gpu_free", new_callable=AsyncMock) as freed:
            r = await self.client.post("/release")
            self.assertEqual(await r.json(), {"released": True, "loaded": False})
            stop.assert_called_once()
            freed.assert_awaited_once()
        self.assertFalse(self.engine.slot.locked())

    async def test_release_waits_for_busy_engine_then_gives_up(self):
        class Proc:
            returncode = None
        self.engine.process = Proc()
        await self.engine.slot.acquire()
        try:
            with patch.object(self.engine, "stop") as stop:
                r = await self.client.post("/release?wait=0.2")
                self.assertEqual(r.status, 409)
                self.assertTrue((await r.json())["busy"])
                stop.assert_not_called()
        finally:
            self.engine.slot.release()
            self.engine.process = None

    async def test_release_runs_after_busy_request_finishes(self):
        import asyncio
        from unittest.mock import AsyncMock

        class Proc:
            returncode = None
        self.engine.process = Proc()

        async def fake_stop():
            self.engine.process = None
        await self.engine.slot.acquire()
        with patch.object(self.engine, "stop", side_effect=fake_stop), \
             patch("flashnext.server.wait_for_gpu_free", new_callable=AsyncMock):
            task = asyncio.create_task(self.client.post("/release?wait=5"))
            await asyncio.sleep(0.2)
            self.assertFalse(task.done())
            self.engine.slot.release()
            r = await task
            self.assertEqual((await r.json())["released"], True)

    async def test_failed_load_releases_busy(self):
        with patch.object(self.engine, "start", side_effect=RuntimeError("busy GPU")):
            r = await self.client.post("/v1/chat/completions", json={"model": "flashnext", "messages": [{}]})
            self.assertEqual(r.status, 503)
            self.assertFalse(self.engine.busy)
            self.assertFalse(self.engine.slot.locked())

    async def test_idle_reaps(self):
        import asyncio
        self.engine.process = object()
        self.engine.last = 0
        from unittest.mock import AsyncMock
        with patch.object(self.engine, "stop", new_callable=AsyncMock) as stop:
            task = asyncio.create_task(self.engine.monitor())
            await asyncio.sleep(1.1)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            stop.assert_awaited()
        self.engine.process = None


class OllamaEvictionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from aiohttp import web, ClientSession
        self.calls = []
        self.models = [{"name": "qwen3.8:27b-instruct", "size": 18, "size_vram": 18},
                       {"name": "cpu-only", "size": 5, "size_vram": 0}]

        async def ps(request):
            self.calls.append(("ps",))
            return web.json_response({"models": self.models})

        async def generate(request):
            body = await request.json()
            self.calls.append(("generate", body))
            return web.json_response({"done": True, "done_reason": "unload"})

        app = web.Application()
        app.router.add_get("/api/ps", ps)
        app.router.add_post("/api/generate", generate)
        self.server = TestServer(app)
        await self.server.start_server()
        self.url = str(self.server.make_url("")).rstrip("/")
        self.session = ClientSession()

    async def asyncTearDown(self):
        await self.session.close()
        await self.server.close()

    async def test_unloads_gpu_models_only_then_waits(self):
        from unittest.mock import AsyncMock
        from flashnext.server import free_gpu_from_ollama
        with patch("flashnext.server.wait_for_gpu_free", new_callable=AsyncMock, return_value=True) as wait:
            self.assertTrue(await free_gpu_from_ollama(self.session, self.url, timeout=7))
            wait.assert_awaited_once_with(7)
        self.assertEqual(self.calls, [("ps",), ("generate", {"model": "qwen3.8:27b-instruct", "keep_alive": 0})])

    async def test_nothing_on_gpu_means_no_unload(self):
        from flashnext.server import free_gpu_from_ollama
        self.models = [{"name": "cpu-only", "size": 5, "size_vram": 0}]
        self.assertFalse(await free_gpu_from_ollama(self.session, self.url))
        self.assertEqual(self.calls, [("ps",)])

    async def test_unreachable_or_disabled_ollama_is_harmless(self):
        from flashnext.server import free_gpu_from_ollama
        await self.server.close()
        self.assertFalse(await free_gpu_from_ollama(self.session, self.url))
        self.assertFalse(await free_gpu_from_ollama(self.session, ""))


class ProfileTests(unittest.TestCase):
    def test_launcher_uses_64k_and_explicit_f16_kv(self):
        import os
        from pathlib import Path
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "runtime/llama.cpp/build/bin/llama-server"
            binary.parent.mkdir(parents=True)
            binary.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
            binary.chmod(0o755)
            git = root / "git"
            git.write_text("#!/bin/sh\necho 74bca3950054eed1795f525cb9cf06bfe796492d\n")
            git.chmod(0o755)
            models = root / "models/UD-IQ4_XS"
            models.mkdir(parents=True)
            for shard in ("00001", "00002", "00003"):
                (models / f"Qwen3.8-Flash-Next-UD-IQ4_XS-{shard}-of-00003.gguf").write_bytes(b"test")
            args = subprocess.check_output(["sh", str(Path(__file__).with_name("launch.sh"))],
                    env=dict(os.environ, FLASHNEXT_ROOT=tmp, PATH=tmp + ":" + os.environ["PATH"]), text=True).splitlines()
            self.assertEqual(args[args.index("-c") + 1], "65536")
            self.assertEqual(args[args.index("-ctk") + 1], "f16")
            self.assertEqual(args[args.index("-ctv") + 1], "f16")
            self.assertEqual(args[args.index("-np") + 1], "1")


class ReasoningTests(unittest.TestCase):
    def test_every_canonical_effort_maps_to_native_template(self):
        for effort in ("none", "low", "medium", "xhigh"):
            body = {"reasoning_effort": effort}
            self.assertEqual(configure_reasoning(body), effort)
            self.assertEqual(body["reasoning_effort"], effort)
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": effort != "none", "reasoning_effort": effort})

    def test_default_remains_non_thinking(self):
        for body in ({}, {"reasoning_effort": ""}):
            self.assertEqual(configure_reasoning(body), "none")
            self.assertFalse(body["chat_template_kwargs"]["enable_thinking"])

    def test_high_is_only_an_alias_for_xhigh(self):
        body = {"reasoning_effort": "high"}
        self.assertEqual(configure_reasoning(body), "xhigh")
        self.assertEqual(body["chat_template_kwargs"]["reasoning_effort"], "xhigh")

    def test_conflicting_or_alternate_controls_rejected(self):
        bodies = [
            {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": True}},
            {"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": False}},
            {"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "medium"}},
            {"chat_template_kwargs": None},
            {"enable_thinking": True},
            {"reasoning": {"effort": "low"}},
        ]
        for body in bodies:
            with self.assertRaises(ValueError):
                configure_reasoning(body)

    def test_thinking_budget_leaves_room_for_answer(self):
        body = {"reasoning_effort": "xhigh", "max_tokens": 4096}
        configure_reasoning(body)
        self.assertEqual(body["reasoning_budget_tokens"], 2560)
        self.assertIn("Thinking budget reached", body["reasoning_budget_message"])
        body = {"reasoning_effort": "xhigh", "max_tokens": 2048}
        configure_reasoning(body)
        self.assertEqual(body["reasoning_budget_tokens"], 512)
        body = {"reasoning_effort": "low", "max_tokens": 4096}
        configure_reasoning(body)
        self.assertEqual(body["reasoning_budget_tokens"], 512)

    def test_no_budget_when_thinking_disabled_or_client_set(self):
        body = {"reasoning_effort": "none", "max_tokens": 4096}
        configure_reasoning(body)
        self.assertNotIn("reasoning_budget_tokens", body)
        body = {"reasoning_effort": "medium", "max_tokens": 4096, "reasoning_budget_tokens": 99}
        configure_reasoning(body)
        self.assertEqual(body["reasoning_budget_tokens"], 99)


class StrataBackendTests(unittest.TestCase):
    def setUp(self):
        select_backend("strata")

    def tearDown(self):
        select_backend("llamacpp")

    def test_xhigh_maps_to_strata_high_with_large_budget(self):
        body = {"reasoning_effort": "xhigh", "max_tokens": 16384}
        self.assertEqual(configure_reasoning(body), "xhigh")
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertEqual(body["reasoning_budget_tokens"], 8192)
        self.assertNotIn("reasoning_budget_message", body)
        self.assertNotIn("chat_template_kwargs", body)

    def test_budget_still_leaves_answer_room_under_small_caps(self):
        body = {"reasoning_effort": "medium", "max_tokens": 4096}
        configure_reasoning(body)
        self.assertEqual(body["reasoning_budget_tokens"], 2048)   # never 0 (uncapped on Strata)

    def test_none_disables_thinking(self):
        body = {"reasoning_effort": "none", "max_tokens": 16384}
        configure_reasoning(body)
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertNotIn("reasoning_budget_tokens", body)


class StrataGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        select_backend("strata")
        self.engine = Engine("/nonexistent", idle=1)
        self.client = TestClient(TestServer(application(self.engine)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        select_backend("llamacpp")

    async def test_listing_advertises_strata_limits_without_loading(self):
        data = (await (await self.client.get("/v1/models")).json())["data"][0]
        self.assertEqual(data["context_length"], 131072)
        self.assertEqual(data["top_provider"]["max_completion_tokens"], 16384)
        status = await (await self.client.get("/status")).json()
        self.assertEqual(status["backend"], "strata")
        self.assertIsNone(self.engine.process)
