"""AI gateway: the one place the agents reach language models.

Routing: each call names a task, the task lists tiers in order of preference (local first, so paid
models only run when the local one is missing or fails), and within a tier the cheapest configured
provider goes first.

Failover: a retryable failure (rate limit, overload, network, refusal) moves on to the next
candidate; a provider that fails `breaker_failures` times in a row is skipped for `breaker_seconds`.
A streamed answer can fail over only before its first chunk, since text already sent can't be taken
back.

Budget: every call is priced from its token usage. Before a call, its worst case (estimated input
plus the full output allowance) must fit in what is left of the request's cap, the month's cap, and
the provider's own monthly cap when it has one (a prepaid balance); a paid model that doesn't fit is
skipped, and when nothing fits the call raises BudgetExceeded. The month's spend is read back from
the ledger file, so the caps hold across restarts.

Ledger: one record per call (time, request id, task, model, tokens, cost, latency, outcome), kept in
memory and appended to a JSONL file when one is configured.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from park_agent.types import BudgetExceeded, ModelSpec, ProviderError, Usage

MODELS_FILE = Path(__file__).with_name("models.json")
TASK_ROUTES = {"route": ["local", "fast"], "analyze": ["local", "fast"], "write": ["local", "strong"]}
TASK_EFFORT = {"route": "low", "analyze": "low", "write": "medium"}
MAX_OUTPUT_TOKENS = 16000


def load_models(path=MODELS_FILE):
    return [ModelSpec(**m) for m in json.loads(Path(path).read_text(encoding="utf-8"))["models"]]


def estimate_tokens(*texts):
    """Rough input size: about four characters per token. Only used for the pre-call budget check;
    the ledger records the provider's actual counts."""
    return sum(len(t or "") for t in texts) // 4 + 1


def month_key(ts=None):
    return (ts or datetime.now(timezone.utc)).strftime("%Y-%m")


@dataclass
class RequestBudget:
    cap_usd: float
    spent_usd: float = 0.0

    @property
    def remaining(self):
        return self.cap_usd - self.spent_usd


@dataclass
class _Breaker:
    failures: int = 0
    open_until: float = 0.0


@dataclass
class Gateway:
    providers: dict                       # provider name -> adapter
    models: list = field(default_factory=load_models)
    routes: dict = field(default_factory=lambda: dict(TASK_ROUTES))
    monthly_cap_usd: float = 10.0
    provider_caps_usd: dict = field(default_factory=lambda: {"openai": 3.00})
    ledger_path: Path | None = None
    breaker_failures: int = 3
    breaker_seconds: float = 60.0
    clock: callable = time.monotonic

    def __post_init__(self):
        self.ledger = []
        self._breakers = {name: _Breaker() for name in self.providers}
        self.provider_spent_usd = self._spent_this_month()
        self.month_spent_usd = sum(self.provider_spent_usd.values())

    def _spent_this_month(self):
        spent = {}
        if not self.ledger_path or not Path(self.ledger_path).exists():
            return spent
        this_month = month_key()
        with open(self.ledger_path, encoding="utf-8") as f:
            for e in map(json.loads, f):
                if e.get("month") == this_month:
                    spent[e.get("provider")] = spent.get(e.get("provider"), 0.0) + e["cost_usd"]
        return spent

    # --- routing ---------------------------------------------------------------------------------

    def candidates(self, task):
        """Configured models for a task: tiers in the task's order, cheapest first within a tier,
        skipping providers whose breaker is open."""
        now = self.clock()
        out = []
        for tier in self.routes[task]:
            usable = [m for m in self.models if m.tier == tier and m.provider in self.providers
                      and self._breakers[m.provider].open_until <= now]
            out += sorted(usable, key=lambda m: m.usd_in + m.usd_out)
        return out

    def _plan(self, task, budget, input_text):
        """The models to try, in order: those whose worst-case cost fits what is left."""
        models = self.candidates(task)
        if not models:
            raise ProviderError(f"no provider configured for task {task!r}", retryable=False)
        left = self.monthly_cap_usd - self.month_spent_usd
        if budget is not None:
            left = min(left, budget.remaining)
        est_in = estimate_tokens(input_text)

        def room(m):
            cap = self.provider_caps_usd.get(m.provider)
            return left if cap is None else min(left, cap - self.provider_spent_usd.get(m.provider, 0.0))
        fits = [m for m in models if m.cost(est_in, MAX_OUTPUT_TOKENS) <= room(m)]
        if not fits:
            raise BudgetExceeded(f"{task}: ${max(left, 0):.4f} left this request/month, not enough for a paid call")
        return fits

    # --- the three call shapes -------------------------------------------------------------------

    async def run_tools(self, task, system, prompt, tools, execute, request_id, budget=None, max_rounds=6):
        """A tool-using step: the model calls tools until it can answer."""
        async def call(provider, model, usage):
            return await provider.run_tools(model.id, system, prompt, tools, execute, max_rounds,
                                            TASK_EFFORT[task], usage)
        return await self._with_failover(task, system + prompt, request_id, budget, call)

    async def complete_json(self, task, system, prompt, schema, request_id, budget=None):
        """One call whose answer must match a JSON schema (the supervisor's routing decision)."""
        async def call(provider, model, usage):
            return await provider.complete_json(model.id, system, prompt, schema, TASK_EFFORT[task], usage)
        return await self._with_failover(task, system + prompt, request_id, budget, call)

    async def stream(self, task, system, prompt, request_id, budget=None):
        """Yield the answer text as it is generated."""
        errors = []
        for model in self._plan(task, budget, system + prompt):
            provider, usage, started, sent = self.providers[model.provider], Usage(), self.clock(), False
            try:
                async for chunk in provider.stream_text(model.id, system, prompt, TASK_EFFORT[task], usage):
                    sent = True
                    yield chunk
            except ProviderError as e:
                self._record(request_id, task, model, usage, started, "error", str(e), errors, budget)
                self._failed(model.provider)
                if sent or not e.retryable:
                    raise
                errors.append(f"{model.id}: {e}")
                continue
            self._record(request_id, task, model, usage, started, "ok", None, errors, budget)
            self._succeeded(model.provider)
            return
        raise ProviderError("all providers failed: " + "; ".join(errors), retryable=False)

    async def _with_failover(self, task, input_text, request_id, budget, call):
        errors = []
        for model in self._plan(task, budget, input_text):
            usage, started = Usage(), self.clock()
            try:
                result = await call(self.providers[model.provider], model, usage)
            except ProviderError as e:
                self._record(request_id, task, model, usage, started, "error", str(e), errors, budget)
                self._failed(model.provider)
                if not e.retryable:
                    raise
                errors.append(f"{model.id}: {e}")
                continue
            self._record(request_id, task, model, usage, started, "ok", None, errors, budget)
            self._succeeded(model.provider)
            return result
        raise ProviderError("all providers failed: " + "; ".join(errors), retryable=False)

    # --- breaker and ledger ----------------------------------------------------------------------

    def _failed(self, provider):
        b = self._breakers[provider]
        b.failures += 1
        if b.failures >= self.breaker_failures:
            b.open_until = self.clock() + self.breaker_seconds

    def _succeeded(self, provider):
        self._breakers[provider] = _Breaker()

    def _record(self, request_id, task, model, usage, started, outcome, error, prior_errors, budget):
        cost = model.cost(usage.input_tokens, usage.output_tokens)
        self.month_spent_usd += cost
        self.provider_spent_usd[model.provider] = self.provider_spent_usd.get(model.provider, 0.0) + cost
        if budget is not None:
            budget.spent_usd += cost
        now = datetime.now(timezone.utc)
        entry = {"time": now.isoformat(timespec="seconds"), "month": month_key(now), "request_id": request_id,
                 "task": task, "provider": model.provider, "model": model.id,
                 "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                 "cost_usd": round(cost, 6), "latency_ms": round((self.clock() - started) * 1000),
                 "outcome": outcome, "error": error, "fallback": bool(prior_errors)}
        self.ledger.append(entry)
        if self.ledger_path:
            Path(self.ledger_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def request_cost(self, request_id):
        return round(sum(e["cost_usd"] for e in self.ledger if e["request_id"] == request_id), 6)
