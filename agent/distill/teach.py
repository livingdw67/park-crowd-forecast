"""Have a teacher model write the answer for each template question: the only paid step.

    .venv\\Scripts\\python agent\\distill\\teach.py --teacher gpt-5.4-mini --train 300 --heldout 60 --max-usd 1.50
    .venv\\Scripts\\python agent\\distill\\teach.py --teacher qwen2.5:7b --train 40 --heldout 10   # free pipeline test

The teacher gets exactly what the agent's writer gets (WRITER_SYSTEM plus the writer prompt) and goes
through the agent's gateway, so the monthly and per-provider caps and the ledger apply. Questions are
picked round-robin across templates so every question type is covered. Each answer is checked with the
agent's number validator; ungrounded answers are kept in the file but marked, and training skips them.
Resumable: questions already answered by this teacher are skipped.

Writes agent/distill/data/teacher-<model>.jsonl (git-ignored).
"""

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from park_agent import gateway as gw  # noqa: E402
from park_agent.config import LOGS, build_providers  # noqa: E402
from park_agent.guardrails import check_answer  # noqa: E402
from park_agent.harness import WRITER_SYSTEM  # noqa: E402
from park_agent.types import BudgetExceeded  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"


def pick(rows, split, n, seed=11):
    """n questions from a split, round-robin over templates so rare types aren't crowded out."""
    rng = random.Random(seed)
    by_template = {}
    for r in rows:
        if r["split"] == split:
            by_template.setdefault(r["template"], []).append(r)
    for group in by_template.values():
        rng.shuffle(group)
    picked, templates = [], sorted(by_template)
    while len(picked) < n and any(by_template.values()):
        for t in templates:
            if by_template[t] and len(picked) < n:
                picked.append(by_template[t].pop())
    return picked


def tool_outputs(row):
    """What the validator lets an answer quote: the tool results, their arguments, the question."""
    results = json.loads(row["writer_prompt"].split("Tool results:\n", 1)[1].split("\n\nTool errors:\n")[0])
    errors = json.loads(row["writer_prompt"].split("\n\nTool errors:\n", 1)[1])
    return ([json.dumps(r["result"]) for r in results] + [json.dumps({"error": e["error"]}) for e in errors]
            + [json.dumps(c["args"]) for c in row["tool_calls"]] + [row["question"]])


async def main(args):
    rows = [json.loads(line) for line in (DATA / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    chosen = pick(rows, "train", args.train) + pick(rows, "heldout", args.heldout)
    out = DATA / f"teacher-{args.teacher.replace(':', '-')}.jsonl"
    done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()

    gateway = gw.Gateway(providers=build_providers(), ledger_path=LOGS / "ledger.jsonl")
    model = next((m for m in gateway.models if m.id == args.teacher), None)
    if model is None or model.provider not in gateway.providers:
        sys.exit(f"teacher {args.teacher!r} is not a configured model")
    gateway.models, gateway.routes = [model], {"teach": [model.tier]}
    gw.TASK_EFFORT["teach"] = args.effort
    import os
    gateway.provider_caps_usd["openai"] = float(os.getenv("OPENAI_CAP_USD", "3.00"))

    start = gateway.month_spent_usd
    todo = [r for r in chosen if r["id"] not in done]
    print(f"{len(todo)} to answer ({len(chosen) - len(todo)} already done) with {args.teacher}, effort {args.effort}")
    with out.open("a", encoding="utf-8") as f:
        for i, row in enumerate(todo, 1):
            if gateway.month_spent_usd - start >= args.max_usd:
                print(f"stopping: ${args.max_usd} spent")
                break
            request_id = f"teach-{row['id']}-{int(time.time())}"
            t0 = time.perf_counter()
            try:
                answer = "".join([c async for c in gateway.stream("teach", WRITER_SYSTEM, row["writer_prompt"], request_id)])
            except BudgetExceeded as e:
                print(f"stopping: {e}")
                break
            check = check_answer(answer, tool_outputs(row))
            record = {"id": row["id"], "split": row["split"], "template": row["template"], "teacher": args.teacher,
                      "answer": answer, "grounded": check.ok, "ungrounded": check.ungrounded,
                      "cost_usd": gateway.request_cost(request_id), "seconds": round(time.perf_counter() - t0, 1)}
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(f"{i}/{len(todo)} {row['id']} {row['template']:<22} grounded={check.ok} "
                  f"${record['cost_usd']:.4f} {record['seconds']}s")
    spent = gateway.month_spent_usd - start
    print(f"spent ${spent:.4f} this run; month total on this machine ${gateway.month_spent_usd:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--train", type=int, default=300)
    parser.add_argument("--heldout", type=int, default=60)
    parser.add_argument("--effort", default="low", choices=["none", "low", "medium", "high"])
    parser.add_argument("--max-usd", type=float, default=1.0)
    asyncio.run(main(parser.parse_args()))
