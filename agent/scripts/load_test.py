"""Load test of the service layer: a real HTTP server and real tools, with simulated model latency.

    .venv\\Scripts\\python agent\\scripts\\load_test.py

Models are replaced by a provider that sleeps for typical latencies (routing 0.4 s, a specialist's
tool loop 1.2 s, the answer streamed as 20 chunks over 1 s), so the test costs nothing and measures
the server, not a model: does async I/O overlap requests, do the optimizer's solves (run in worker
threads) block streaming, and how does the in-flight cap queue the excess. Each level sends
`concurrency` requests at once, three rounds; results go to agent/logs/load_test.json.
"""

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

import httpx
import uvicorn

os.environ["PARK_AGENT_NO_APP"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from park_agent.api import create_app, percentile  # noqa: E402
from park_agent.config import LOGS  # noqa: E402
from park_agent.data import ParkData  # noqa: E402
from park_agent.gateway import Gateway  # noqa: E402
from park_agent.harness import Harness  # noqa: E402
from park_agent.tools import ParkTools  # noqa: E402
from park_agent.types import ToolCall, ToolLoopResult  # noqa: E402

PORT, LEVELS, ROUNDS, MAX_IN_FLIGHT = 8766, [1, 4, 8, 16], 3, 8


class SlowModel:
    """Answers every call the same way, after a sleep, and runs the tool calls for real."""
    name = "anthropic"

    def __init__(self, day):
        self.day = day

    async def complete_json(self, model, system, prompt, schema, effort, usage):
        await asyncio.sleep(0.4)
        usage.add(300, 40)
        return {"route": "both", "reason": "load test"}

    async def run_tools(self, model, system, prompt, tools, execute, max_rounds, effort, usage):
        await asyncio.sleep(1.2)
        usage.add(1500, 200)
        name = tools[0].name
        args = ({"park": "Islands of Adventure", "start_date": self.day, "end_date": self.day}
                if name == "get_forecast" else {"park": "Islands of Adventure", "date": self.day})
        if name == "get_forecast":
            name, args = "explain_day", {"park": "Islands of Adventure", "date": self.day}
        result, is_error = await execute(name, args)
        return ToolLoopResult("notes", [ToolCall(name, args, result, is_error)], 2)

    async def stream_text(self, model, system, prompt, effort, usage):
        for i in range(20):
            await asyncio.sleep(0.05)
            yield "word "
        usage.add(2000, 150)


async def one(client, question):
    started, first = time.perf_counter(), None
    async with client.stream("POST", "/ask", json={"question": question}) as r:
        if r.status_code != 200:
            return {"status": r.status_code}
        async for line in r.aiter_lines():
            if first is None and line == "event: token":
                first = time.perf_counter() - started
    return {"status": 200, "ttft": first, "total": time.perf_counter() - started}


async def run_levels():
    results = []
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}", timeout=120) as client:
        for level in LEVELS:
            rows = []
            for _ in range(ROUNDS):
                rows += await asyncio.gather(*(one(client, f"How busy is IOA, question {i}?") for i in range(level)))
            ok = [r for r in rows if r["status"] == 200]
            results.append({
                "concurrency": level, "requests": len(rows), "ok": len(ok),
                "rejected": len(rows) - len(ok),
                "ttft_p50_s": round(percentile([r["ttft"] for r in ok], 50), 2),
                "ttft_p95_s": round(percentile([r["ttft"] for r in ok], 95), 2),
                "total_p50_s": round(percentile([r["total"] for r in ok], 50), 2),
                "total_p95_s": round(percentile([r["total"] for r in ok], 95), 2)})
            print(results[-1])
    return results


def main():
    data = ParkData.load()
    day = str(data.forecast_dates()[2])
    harness = Harness(Gateway(providers={"anthropic": SlowModel(day)}), ParkTools(data))
    app = create_app(harness, rate_per_minute=10_000, max_in_flight=MAX_IN_FLIGHT)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    try:
        results = asyncio.run(run_levels())
    finally:
        server.should_exit = True
        thread.join(timeout=5)
    LOGS.mkdir(exist_ok=True)
    (LOGS / "load_test.json").write_text(json.dumps({"max_in_flight": MAX_IN_FLIGHT, "levels": results}, indent=1),
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
