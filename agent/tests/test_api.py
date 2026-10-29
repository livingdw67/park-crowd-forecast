import json
import os

import httpx
import pytest

os.environ["PARK_AGENT_NO_APP"] = "1"

from park_agent.api import RateLimiter, create_app, percentile  # noqa: E402
from park_agent.gateway import Gateway  # noqa: E402
from park_agent.harness import Harness  # noqa: E402
from park_agent.providers.fake import FakeProvider  # noqa: E402

pytestmark = pytest.mark.anyio


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def harness(tools, script):
    return Harness(Gateway(providers={"anthropic": FakeProvider("anthropic", script)}), tools)


async def test_ask_streams_events_as_sse(tools):
    script = [{"json": {"route": "forecast", "reason": "busy"}},
              {"tool_calls": [("explain_day", {"park": "Islands of Adventure", "date": "2026-11-27"})], "text": "notes"},
              {"chunks": ["Friday: ", "65."]}]
    async with client(create_app(harness(tools, script))) as c:
        r = await c.post("/ask", json={"question": "How busy is IOA on Black Friday?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert [e for e, _ in events] == ["start", "route", "tool", "token", "token", "validation", "done"]
    assert events[0][1]["request_id"] == r.headers["x-request-id"]


async def test_health_ready_and_metrics(tools):
    app = create_app(harness(tools, [{"json": {"route": "out_of_scope", "reason": "x"}}]))
    async with client(app) as c:
        assert (await c.get("/health")).json() == {"status": "ok"}
        ready = await c.get("/ready")
        assert ready.status_code == 200 and ready.json()["providers"] == ["anthropic"]
        await c.post("/ask", json={"question": "What's the weather in Paris?"})
        metrics = (await c.get("/metrics")).json()
    assert metrics["calls"] == 1 and metrics["latency_ms"]["route"]["n"] == 1


async def test_not_ready_without_a_provider(tools):
    async with client(create_app(Harness(Gateway(providers={}), tools))) as c:
        r = await c.get("/ready")
    assert r.status_code == 503 and "no model provider configured" in r.json()["problems"]


async def test_api_key_and_rate_limit(tools):
    app = create_app(harness(tools, [{"json": {"route": "out_of_scope", "reason": "x"}}] * 3),
                     api_key="secret", rate_per_minute=2)
    async with client(app) as c:
        assert (await c.post("/ask", json={"question": "hi"})).status_code == 401
        headers = {"X-API-Key": "secret"}
        assert (await c.post("/ask", json={"question": "hi"}, headers=headers)).status_code == 200
        assert (await c.post("/ask", json={"question": "hi"}, headers=headers)).status_code == 200
        assert (await c.post("/ask", json={"question": "hi"}, headers=headers)).status_code == 429


def test_rate_limiter_window_and_percentile():
    now = [0.0]
    limiter = RateLimiter(2, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a") and limiter.allow("b")
    now[0] = 61
    assert limiter.allow("a")
    assert percentile([5, 1, 3, 2, 4], 50) == 3 and percentile([], 95) is None
