"""Bounded real-model chat/tool/SSE/idle gate; no persistent listener.

Runs its own in-process gateway, so stop the flashnext service first (it would
hold the GPU and the engine port).
"""
import argparse
import asyncio
import json
from pathlib import Path
from aiohttp.test_utils import TestClient, TestServer
from . import server
from .server import Engine, application, available_memory, CLIENT


async def run(root, evidence):
    engine = Engine(root, idle=2)
    results = {"model": "flashnext", "backend": server.BACKEND, "checks": [], "minimum_available_ram_mib": available_memory()}
    async def sample():
        while True:
            results["minimum_available_ram_mib"] = min(results["minimum_available_ram_mib"], available_memory())
            await asyncio.sleep(0.2)
    sampler = asyncio.create_task(sample())
    try:
        async with TestClient(TestServer(application(engine))) as client:
            response = await client.get("/v1/models")
            assert (await response.json())["data"][0]["id"] == "flashnext"
            assert engine.process is None
            prompt = {"model": "flashnext", "max_tokens": 128, "temperature": 0,
                      "messages": [{"role": "user", "content": "Say hello from FlashNext in one short sentence."}]}
            response = await client.post("/v1/chat/completions", json=prompt)
            data = await response.json()
            assert response.status == 200, data
            assert data["model"] == "flashnext" and data["choices"][0]["message"]["content"]
            results["checks"].append({"completion": data})
            http = client.server.app[CLIENT]
            if server.BACKEND == "strata":
                async with http.get(server.UPSTREAM + "/health") as response:
                    health = await response.json()
                assert health["loaded"] is True and health["max_context"] == server.CONTEXT_WINDOW, health
                results["runtime_health"] = health
            else:
                async with http.get(server.UPSTREAM + "/slots") as response:
                    slots = await response.json()
                async with http.get(server.UPSTREAM + "/v1/models") as response:
                    identity = await response.json()
                assert len(slots) == 1 and slots[0]["n_ctx"] == server.CONTEXT_WINDOW, slots
                assert identity["data"][0]["meta"]["n_ctx"] == server.CONTEXT_WINDOW, identity
                results["runtime_slots"] = slots
                results["runtime_model_identity"] = identity
            tools = [{"type": "function", "function": {"name": "get_temperature", "description": "Get city temperature",
                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
            messages = [{"role": "user", "content": "Use get_temperature to check the temperature in Sydney."}]
            prompt.update(messages=messages, tools=tools, tool_choice={"type": "function", "function": {"name": "get_temperature"}}, stream=True)
            response = await client.post("/v1/chat/completions", json=prompt)
            sse = await response.text()
            assert response.status == 200 and "data: [DONE]" in sse, sse
            call = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            for line in sse.splitlines():
                if not line.startswith("data: {"):
                    continue
                chunk = json.loads(line[6:])
                assert chunk["model"] == "flashnext"
                for delta in chunk["choices"][0].get("delta", {}).get("tool_calls", []):
                    call["id"] += delta.get("id", "")
                    for key in ("name", "arguments"):
                        call["function"][key] += delta.get("function", {}).get(key, "")
            assert call["id"] and call["function"]["name"] == "get_temperature", sse
            assert json.loads(call["function"]["arguments"])["city"] == "Sydney", sse
            results["checks"].append({"streamed_tool_call": call, "sse": sse})
            messages += [{"role": "assistant", "content": "", "tool_calls": [call]},
                         {"role": "tool", "tool_call_id": call["id"], "content": '{"temperature_c":22}'}]
            prompt.update(messages=messages, tool_choice="auto")
            response = await client.post("/v1/chat/completions", json=prompt)
            sse = await response.text()
            content = "".join(json.loads(line[6:])["choices"][0].get("delta", {}).get("content") or "" for line in sse.splitlines() if line.startswith("data: {"))
            assert response.status == 200 and "22" in content and "data: [DONE]" in sse, sse
            results["checks"].append({"streamed_tool_result_followup": sse})
            await asyncio.sleep(4)
            assert engine.process is None and engine.lease is None
            results["idle_unloaded"] = True
    finally:
        sampler.cancel()
        await asyncio.gather(sampler, return_exceptions=True)
        await engine.stop()
    Path(evidence).write_text(json.dumps(results, indent=2))
    print(json.dumps({"passed": True, "evidence": evidence, "minimum_available_ram_mib": results["minimum_available_ram_mib"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=sorted(server.BACKENDS), default="strata")
    p.add_argument("--root", required=True, help="Strata runtime root, or the llama.cpp asset root")
    p.add_argument("--evidence", required=True)
    args = p.parse_args()
    server.select_backend(args.backend)
    asyncio.run(asyncio.wait_for(run(args.root, args.evidence), 400))
