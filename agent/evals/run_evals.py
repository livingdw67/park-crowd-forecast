"""Run the agent on the eval questions and grade each run without a model judge.

    .venv\\Scripts\\python agent\\evals\\run_evals.py --max-usd 0.50

Checks per question:
- route: the supervisor's route is one of the allowed ones ("blocked" = refused by the input guardrail)
- tools: for each group in `tools`, at least one of its tools was called without error
  (skipped for questions marked "decline", where the tool error is the right outcome)
- grounded: the validator found every number in the answer in tool output (when there is an answer)
- no_error: the run didn't end in an error event (blocked questions excepted)

A question passes when every check applies and holds. The run stops early once the ledger shows
`--max-usd` spent. Results go to agent/evals/results/<timestamp>.json.
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from park_agent.config import build_harness  # noqa: E402
from park_agent.gateway import RequestBudget  # noqa: E402

HERE = Path(__file__).resolve().parent


def grade(case, events):
    route = next((e["route"] for e in events if e["type"] == "route"), None)
    errors = [e["message"] for e in events if e["type"] == "error"]
    if route is None and errors and errors[0].startswith("Blocked"):
        route = "blocked"
    calls = [e for e in events if e["type"] == "tool"]
    ok_tools = {e["name"] for e in calls if not e["is_error"]}
    validation = next((e for e in events if e["type"] == "validation"), None)
    checks = {"route": route in case["route"]}
    if case.get("expect") != "decline" and case["tools"] and route != "blocked":
        checks["tools"] = all(ok_tools & set(group) for group in case["tools"])
    if validation is not None:
        checks["grounded"] = validation["ok"]
    if route != "blocked":
        checks["no_error"] = not errors
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    return {"id": case["id"], "question": case["question"], "passed": all(checks.values()), "checks": checks,
            "route": route, "tools_called": [(e["agent"], e["name"], e["is_error"]) for e in calls],
            "ungrounded": validation["ungrounded"] if validation else [], "answer": answer, "errors": errors}


async def main(max_usd, only):
    harness = build_harness()
    cases = [json.loads(line) for line in (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if only:
        cases = [c for c in cases if c["id"] in only]
    started_spend, results = harness.gateway.month_spent_usd, []
    for case in cases:
        if harness.gateway.month_spent_usd - started_spend >= max_usd:
            print(f"stopping: ${max_usd} spent")
            break
        request_id = f"eval-{case['id']}-{int(time.time())}"
        t0 = time.perf_counter()
        events = [e async for e in harness.run(case["question"], request_id, RequestBudget(cap_usd=0.10))]
        result = grade(case, events)
        result.update(seconds=round(time.perf_counter() - t0, 1), cost_usd=harness.gateway.request_cost(request_id),
                      models=next((e["models"] for e in events if e["type"] == "done"), []))
        results.append(result)
        failed = [k for k, v in result["checks"].items() if not v]
        print(f"{case['id']} {'PASS' if result['passed'] else 'FAIL ' + ','.join(failed)}  "
              f"{result['seconds']}s ${result['cost_usd']:.4f}")
    summary = {"questions": len(results), "passed": sum(r["passed"] for r in results),
               "cost_usd": round(sum(r["cost_usd"] for r in results), 4),
               "by_check": {k: f"{sum(r['checks'].get(k, True) for r in results)}/{sum(k in r['checks'] for r in results)}"
                            for k in ("route", "tools", "grounded", "no_error")},
               "providers": sorted(harness.gateway.providers)}
    print(json.dumps(summary, indent=1))
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"{datetime.now():%Y%m%d-%H%M%S}.json").write_text(json.dumps({"summary": summary, "results": results},
                                                                          indent=1), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-usd", type=float, default=0.50)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    asyncio.run(main(args.max_usd, args.only))
