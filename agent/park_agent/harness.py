"""The agent harness: supervisor, two specialists, a writer and a validator, as an explicit state machine.

    question -> guardrails -> supervisor (route) -> specialists (tools; in parallel when both are
    needed) -> writer (streams the answer) -> validator (numbers vs. tool output) -> checkpoint

Each step is a gateway call with its own task, so routing, failover and budget apply per step. The
run yields events for the API to stream: route, tool, token, validation, error, done. Every turn's
state is written to logs/turns/<request id>.json.
"""

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path

from park_agent.guardrails import Blocked, check_answer, check_question
from park_agent.types import BudgetExceeded, ProviderError

ROUTES = ["forecast", "staffing", "both", "out_of_scope"]
ROUTE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["route", "reason"],
                "properties": {"route": {"type": "string", "enum": ROUTES}, "reason": {"type": "string"}}}
OUT_OF_SCOPE = ("I can answer questions about crowd forecasts and staffing plans for Universal Studios Florida, "
                "Islands of Adventure, Epic Universe and Volcano Bay. That question is outside what I cover.")


@dataclass(frozen=True)
class Specialist:
    name: str
    tools: tuple
    brief: str


SPECIALISTS = {
    "forecast": Specialist(
        "forecast analyst", ("get_forecast", "explain_day", "get_epic_impact"),
        "You analyze crowd forecasts. Look up the days asked about, explain what drives them (weekday, "
        "holidays, school breaks, events), and say how uncertain the forecast is."),
    "staffing": Specialist(
        "staffing planner", ("get_staffing_plan", "what_if", "get_forecast"),
        "You plan ride and food staffing. Look up the optimizer's plan for the day, and run what_if when "
        "the question changes the crowd, the park hours or the number of workers."),
}


def _context(data):
    today = data.today()
    dates = data.forecast_dates()
    week = ", ".join(f"{(today + timedelta(days=i)):%A} {today + timedelta(days=i)}" for i in range(7))
    return (f"Today is {today:%A} {today}. The next seven days: {week}. Forecasts cover {dates[0]} to "
            f"{dates[-1]}. Parks: Universal Studios Florida, Islands of Adventure, Epic Universe, Volcano Bay.")


@dataclass
class Turn:
    request_id: str
    question: str
    route: str = ""
    route_reason: str = ""
    findings: dict = field(default_factory=dict)      # specialist -> its notes
    tool_calls: list = field(default_factory=list)    # {agent, name, args, result, is_error}
    answer: str = ""
    validation: dict = field(default_factory=dict)
    error: str = ""
    cost_usd: float = 0.0


@dataclass
class Harness:
    gateway: object
    tools: object                                     # ParkTools
    turns_dir: Path | None = None

    async def run(self, question, request_id, budget=None):
        turn = Turn(request_id, question)
        try:
            async for event in self._steps(turn, budget):
                yield event
        except Blocked as e:
            turn.error = f"blocked: {e}"
            yield {"type": "error", "message": f"Blocked: {e}"}
        except BudgetExceeded as e:
            turn.error = f"budget: {e}"
            yield {"type": "error", "message": "The AI budget for this request or month is used up."}
        except ProviderError as e:
            turn.error = f"provider: {e}"
            yield {"type": "error", "message": "No model was available to answer. Try again shortly."}
        turn.cost_usd = self.gateway.request_cost(request_id)
        self._checkpoint(turn)
        yield {"type": "done", "cost_usd": turn.cost_usd, "route": turn.route,
               "models": sorted({e["model"] for e in self.gateway.ledger if e["request_id"] == request_id})}

    async def _steps(self, turn, budget):
        question = check_question(turn.question)
        context = _context(self.tools.data)

        decision = await self.gateway.complete_json(
            "route", "You route questions for a theme park planning assistant. forecast: how busy a park or "
            "day will be, why, or the Epic Universe study. staffing: crew plans, labor cost, waits, what-if "
            "scenarios. both: needs both. out_of_scope: anything else, including other parks. A question about "
            "these parks on a date outside the forecast range is still forecast: the analyst explains the range. "
            + context,
            question, ROUTE_SCHEMA, turn.request_id, budget)
        turn.route, turn.route_reason = decision["route"], decision["reason"]
        yield {"type": "route", "route": turn.route, "reason": turn.route_reason}
        if turn.route == "out_of_scope":
            turn.answer = OUT_OF_SCOPE
            yield {"type": "token", "text": OUT_OF_SCOPE}
            return

        names = ["forecast", "staffing"] if turn.route == "both" else [turn.route]
        results = await asyncio.gather(*(self._specialist(n, question, context, turn, budget) for n in names))
        for name, result in zip(names, results):
            turn.findings[name] = result.text
            for call in result.calls:
                record = {"agent": name, "name": call.name, "args": call.args, "result": call.result,
                          "is_error": call.is_error}
                turn.tool_calls.append(record)
                yield {"type": "tool", **{k: record[k] for k in ("agent", "name", "args", "is_error")}}

        # numbers an answer may quote: tool results and errors, tool arguments, and the question itself
        outputs = [c["result"] for c in turn.tool_calls] + [c["args"] for c in turn.tool_calls] + [question]
        async for chunk in self.gateway.stream("write", WRITER_SYSTEM, self._writer_prompt(question, context, turn),
                                               turn.request_id, budget):
            turn.answer += chunk
            yield {"type": "token", "text": chunk}

        check = check_answer(turn.answer, outputs)
        turn.validation = asdict(check)
        yield {"type": "validation", "ok": check.ok, "ungrounded": check.ungrounded}

    async def _specialist(self, name, question, context, turn, budget):
        spec = SPECIALISTS[name]
        system = (f"You are the {spec.name} on a theme park planning team. {spec.brief} Use the tools for every "
                  f"number; never estimate one yourself. Reply with short factual notes for the writer, citing "
                  f"the numbers the tools returned. {context}")
        return await self.gateway.run_tools("analyze", system, question, self.tools.specs(spec.tools),
                                            self.tools.execute, turn.request_id, budget)

    @staticmethod
    def _writer_prompt(question, context, turn):
        results = [{"tool": c["name"], "args": c["args"], "result": json.loads(c["result"])}
                   for c in turn.tool_calls if not c["is_error"]]
        errors = [{"tool": c["name"], "args": c["args"], "error": json.loads(c["result"]).get("error")}
                  for c in turn.tool_calls if c["is_error"]]
        notes = "\n\n".join(f"{name} notes:\n{text}" for name, text in turn.findings.items())
        return (f"{context}\n\nQuestion: {question}\n\n{notes}\n\nTool results:\n{json.dumps(results, default=str)}"
                f"\n\nTool errors:\n{json.dumps(errors, default=str)}")

    def _checkpoint(self, turn):
        if self.turns_dir:
            Path(self.turns_dir).mkdir(parents=True, exist_ok=True)
            (Path(self.turns_dir) / f"{turn.request_id}.json").write_text(
                json.dumps(asdict(turn), indent=1, default=str), encoding="utf-8")


WRITER_SYSTEM = (
    "You write the final answer for a theme park planning assistant, from the team's notes and the tool "
    "results. Lead with the direct answer in one sentence. Then at most four short bullets with the key "
    "numbers. Use only numbers that appear in the tool results; don't compute new ones except simple "
    "differences. Crowd scores are on a 0-100 scale; mention the typical forecast error when you give a "
    "forecast. Staffing costs and waits come from a planning model with assumed demand; say so when you "
    "quote them. When a tool returned an error, say plainly what isn't available and why, quoting the error's "
    "facts (such as the dates the forecast covers); never offer estimates the tools didn't produce. Plain text, "
    "no headings.")
