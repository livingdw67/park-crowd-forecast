"""HTTP service for the agent.

    .venv\\Scripts\\python -m uvicorn park_agent.api:app --app-dir agent --port 8000

POST /ask streams the harness's events as Server-Sent Events (`event: <type>` + `data: <json>`), so a
client shows tool calls as they happen and the answer token by token. GET /health says the process
is up (liveness); GET /ready says it can answer (readiness: data loaded, a provider configured, budget
left) and returns 503 when it can't. GET /metrics summarizes the ledger: spend by provider, latency
percentiles by task, failovers.

Protections: an X-API-Key header when AGENT_API_KEY is set, a per-client rate limit, and a cap on
requests in flight; past the cap a request waits up to QUEUE_SECONDS, then gets a 503.
"""

import asyncio
import json
import os
import time
import uuid
from collections import defaultdict, deque

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from park_agent.gateway import RequestBudget

RATE_PER_MINUTE = int(os.getenv("RATE_PER_MINUTE", "20"))
MAX_IN_FLIGHT = int(os.getenv("MAX_IN_FLIGHT", "8"))
QUEUE_SECONDS = float(os.getenv("QUEUE_SECONDS", "10"))
REQUEST_CAP_USD = float(os.getenv("REQUEST_CAP_USD", "0.25"))


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class RateLimiter:
    """Sliding one-minute window per client."""

    def __init__(self, per_minute, clock=time.monotonic):
        self.per_minute, self.clock, self.hits = per_minute, clock, defaultdict(deque)

    def allow(self, client):
        now, hits = self.clock(), self.hits[client]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= self.per_minute:
            return False
        hits.append(now)
        return True


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round(q / 100 * (len(values) - 1)))]


def create_app(harness, api_key=None, rate_per_minute=RATE_PER_MINUTE, max_in_flight=MAX_IN_FLIGHT):
    app = FastAPI(title="Park planning agent")
    limiter = RateLimiter(rate_per_minute)
    slots = asyncio.Semaphore(max_in_flight)

    def guard(request: Request, x_api_key: str | None = Header(default=None)):
        if api_key and x_api_key != api_key:
            raise HTTPException(401, "missing or wrong X-API-Key")
        if not limiter.allow(request.client.host if request.client else "unknown"):
            raise HTTPException(429, f"limit is {rate_per_minute} questions a minute")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        gw, data = harness.gateway, harness.tools.data
        problems = []
        if not gw.providers:
            problems.append("no model provider configured")
        if gw.monthly_cap_usd - gw.month_spent_usd <= 0 and "local" not in gw.providers:
            problems.append("monthly budget used up and no local model")
        if data.forecast.empty:
            problems.append("no forecast data")
        body = {"ready": not problems, "problems": problems, "providers": sorted(gw.providers),
                "month_spent_usd": round(gw.month_spent_usd, 4), "monthly_cap_usd": gw.monthly_cap_usd,
                "snapshot_exported_at": data.exported_at, "forecast_from": str(data.today())}
        return JSONResponse(body, status_code=200 if not problems else 503)

    @app.get("/metrics", dependencies=[Depends(guard)])
    def metrics():
        gw = harness.gateway
        by_task = defaultdict(list)
        for e in gw.ledger:
            by_task[e["task"]].append(e["latency_ms"])
        return {"calls": len(gw.ledger), "month_spent_usd": round(gw.month_spent_usd, 4),
                "provider_spent_usd": {k: round(v, 4) for k, v in gw.provider_spent_usd.items()},
                "failovers": sum(e["fallback"] for e in gw.ledger),
                "errors": sum(e["outcome"] == "error" for e in gw.ledger),
                "latency_ms": {task: {"p50": percentile(v, 50), "p95": percentile(v, 95), "n": len(v)}
                               for task, v in by_task.items()}}

    @app.post("/ask", dependencies=[Depends(guard)])
    async def ask(body: Ask):
        try:
            await asyncio.wait_for(slots.acquire(), QUEUE_SECONDS)
        except asyncio.TimeoutError:
            raise HTTPException(503, "busy; try again shortly") from None
        request_id = uuid.uuid4().hex[:12]

        async def events():
            try:
                yield _sse("start", {"request_id": request_id})
                async for event in harness.run(body.question, request_id, RequestBudget(REQUEST_CAP_USD)):
                    yield _sse(event["type"], {k: v for k, v in event.items() if k != "type"})
            finally:
                slots.release()

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Request-Id": request_id})

    return app


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _default_app():
    from park_agent.config import build_harness
    return create_app(build_harness(), api_key=os.getenv("AGENT_API_KEY") or None)


app = _default_app() if os.getenv("PARK_AGENT_NO_APP") is None else None
