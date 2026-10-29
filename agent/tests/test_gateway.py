import json
from datetime import datetime, timezone

import pytest

from park_agent.gateway import Gateway, RequestBudget, month_key
from park_agent.providers.fake import FakeProvider
from park_agent.types import BudgetExceeded, ProviderError, Tool

pytestmark = pytest.mark.anyio
ANSWER = {"text": "ok"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def echo(name, args):
    return json.dumps({"tool": name, **args}), False


def gateway(**providers):
    return Gateway(providers=providers)


async def test_local_model_goes_first():
    local, claude = FakeProvider("local", [ANSWER]), FakeProvider("anthropic", [])
    gw = gateway(local=local, anthropic=claude)
    await gw.run_tools("analyze", "sys", "q", [], echo, "r1")
    assert local.calls == [("run_tools", "qwen2.5:7b")] and claude.calls == []


async def test_cheapest_paid_model_first_when_no_local():
    claude, gpt = FakeProvider("anthropic", [ANSWER]), FakeProvider("openai", [])
    gw = gateway(anthropic=claude, openai=gpt)
    assert [m.id for m in gw.candidates("analyze")] == ["claude-haiku-5-5", "gpt-5.4-mini"]
    assert [m.id for m in gw.candidates("write")] == ["claude-opus-5-5", "gpt-5.5"]


async def test_retryable_failure_fails_over_and_is_logged():
    claude = FakeProvider("anthropic", [ProviderError("overloaded", retryable=True)])
    gpt = FakeProvider("openai", [{"tool_calls": [("get_forecast", {"park": "USF"})], "text": "busy"}])
    gw = gateway(anthropic=claude, openai=gpt)
    result = await gw.run_tools("analyze", "sys", "q", [Tool("get_forecast", "", {})], echo, "r1")
    assert result.text == "busy" and result.calls[0].name == "get_forecast"
    assert [(e["model"], e["outcome"], e["fallback"]) for e in gw.ledger] == [
        ("claude-haiku-5-5", "error", False), ("gpt-5.4-mini", "ok", True)]


async def test_non_retryable_failure_is_raised():
    claude = FakeProvider("anthropic", [ProviderError("bad request", retryable=False)])
    gpt = FakeProvider("openai", [ANSWER])
    with pytest.raises(ProviderError, match="bad request"):
        await gateway(anthropic=claude, openai=gpt).run_tools("analyze", "sys", "q", [], echo, "r1")
    assert gpt.calls == []


async def test_breaker_skips_a_failing_provider_for_a_while():
    now = [0.0]
    claude = FakeProvider("anthropic", [ProviderError("down", retryable=True)] * 3)
    gpt = FakeProvider("openai", [ANSWER] * 4)
    gw = Gateway(providers={"anthropic": claude, "openai": gpt}, breaker_failures=3, breaker_seconds=60,
                 clock=lambda: now[0])
    for _ in range(3):
        await gw.run_tools("analyze", "sys", "q", [], echo, "r")
    assert [m.provider for m in gw.candidates("analyze")] == ["openai"]
    now[0] = 61
    assert [m.provider for m in gw.candidates("analyze")] == ["anthropic", "openai"]


async def test_request_budget_keeps_paid_calls_out_but_local_still_runs():
    tight = RequestBudget(cap_usd=0.001)       # a Haiku call's worst case is about $0.008
    with pytest.raises(BudgetExceeded):
        await gateway(anthropic=FakeProvider("anthropic", [ANSWER])).run_tools(
            "analyze", "sys", "q", [], echo, "r1", budget=tight)
    local = FakeProvider("local", [ANSWER])
    await gateway(local=local, anthropic=FakeProvider("anthropic", [])).run_tools(
        "analyze", "sys", "q", [], echo, "r2", budget=tight)
    assert local.calls


async def test_monthly_cap_is_read_back_from_the_ledger(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    rows = [{"month": month_key(), "cost_usd": 9.999}, {"month": "2000-01", "cost_usd": 50.0}]
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    gw = Gateway(providers={"anthropic": FakeProvider("anthropic", [ANSWER])}, ledger_path=ledger, monthly_cap_usd=10)
    assert gw.month_spent_usd == pytest.approx(9.999)
    with pytest.raises(BudgetExceeded):
        await gw.run_tools("analyze", "sys", "q", [], echo, "r1")


async def test_provider_cap_skips_only_that_provider(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"month": month_key(), "provider": "openai", "cost_usd": 2.99}) + "\n",
                      encoding="utf-8")
    claude = FakeProvider("anthropic", [ProviderError("overloaded", retryable=True)])
    gpt = FakeProvider("openai", [ANSWER])
    gw = Gateway(providers={"anthropic": claude, "openai": gpt}, ledger_path=ledger,
                 provider_caps_usd={"openai": 3.00})
    with pytest.raises(ProviderError, match="all providers failed"):
        await gw.run_tools("analyze", "sys", "q", [], echo, "r1")
    assert gpt.calls == []          # openai's worst case (~$0.07) doesn't fit its last $0.01


async def test_ledger_prices_calls_from_usage(tmp_path):
    claude = FakeProvider("anthropic", [{"text": "ok", "usage": (1000, 200)}])
    gw = Gateway(providers={"anthropic": claude}, ledger_path=tmp_path / "ledger.jsonl")
    await gw.run_tools("analyze", "sys", "q", [], echo, "r1")
    assert gw.request_cost("r1") == pytest.approx((1000 * 0.10 + 200 * 0.50) / 1e6)
    saved = json.loads((tmp_path / "ledger.jsonl").read_text(encoding="utf-8"))
    assert saved["month"] == datetime.now(timezone.utc).strftime("%Y-%m") and saved["model"] == "claude-haiku-5-5"


async def test_stream_fails_over_before_the_first_chunk_only():
    claude = FakeProvider("anthropic", [ProviderError("overloaded", retryable=True)])
    gpt = FakeProvider("openai", [{"chunks": ["Busy ", "day."]}])
    gw = gateway(anthropic=claude, openai=gpt)
    assert [c async for c in gw.stream("write", "sys", "q", "r1")] == ["Busy ", "day."]

    claude = FakeProvider("anthropic", [{"chunks": ["Busy ", ProviderError("cut off", retryable=True)]}])
    gw = gateway(anthropic=claude, openai=FakeProvider("openai", [{"chunks": ["x"]}]))
    with pytest.raises(ProviderError, match="cut off"):
        [c async for c in gw.stream("write", "sys", "q", "r2")]
