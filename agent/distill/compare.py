"""Score writers against the teacher on held-out questions, without a model judge.

    .venv\\Scripts\\python agent\\distill\\compare.py --teacher gpt-5.4-mini --writers park-writer-1.5b qwen2.5-1.5b-base qwen2.5:7b

Every writer gets the same held-out writer prompts (weeks no training example came from) through
Ollama, greedy. Checks per answer:
- grounded: the agent's validator finds every number in the tool output
- format: follows the writer rules: no headings or bold, at most four bullets
- teacher_numbers: share of the numbers in the teacher's answer that this answer also states
  (did it pick the same key facts)
- states_error: quotes the typical forecast error when the tools returned one
Also words and seconds per answer. The teacher's own answers are scored the same way, as the
reference row. Writes agent/distill/results/<timestamp>.json (scores and every answer; no raw tool
output) and prints a table.
"""

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import openai

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from park_agent.guardrails import _numbers, check_answer  # noqa: E402
from park_agent.harness import WRITER_SYSTEM  # noqa: E402
from teach import tool_outputs  # noqa: E402


def numbers(text):
    return {value for _, value, _ in _numbers(text)}


def score(answer, row, teacher_answer):
    bullets = [line for line in answer.splitlines() if re.match(r"\s*([-*•]|\d+\.)\s", line)]
    teacher_nums = numbers(teacher_answer)
    has_error_figure = '"typical_error_points"' in row["writer_prompt"].split("Tool results:", 1)[1]
    return {
        "grounded": check_answer(answer, tool_outputs(row)).ok,
        "format": len(bullets) <= 4 and "**" not in answer and not re.search(r"^\s*#", answer, re.M),
        "teacher_numbers": len(teacher_nums & numbers(answer)) / len(teacher_nums) if teacher_nums else None,
        "states_error": (bool(re.search(r"\b15\b", answer)) if has_error_figure else None),
        "words": len(answer.split()),
    }


def summarize(scores):
    out = {}
    for key in ("grounded", "format", "teacher_numbers", "states_error", "words", "seconds"):
        values = [s[key] for s in scores if s.get(key) is not None]
        out[key] = round(statistics.mean(values), 3) if values else None
    out["n"] = len(scores)
    return out


def main(args):
    questions = {r["id"]: r for r in map(json.loads, (HERE / "data" / "questions.jsonl").read_text(encoding="utf-8").splitlines())}
    teacher = [a for a in map(json.loads, (HERE / "data" / f"teacher-{args.teacher.replace(':', '-')}.jsonl")
                              .read_text(encoding="utf-8").splitlines()) if a["split"] == "heldout"]
    client = openai.OpenAI(base_url="http://localhost:11434/v1", api_key="local")
    results = {"teacher": args.teacher, "questions": len(teacher), "writers": {}}

    reference = [{**score(t["answer"], questions[t["id"]], t["answer"]), "id": t["id"], "answer": t["answer"]} for t in teacher]
    results["writers"][f"teacher ({args.teacher})"] = {"summary": summarize(reference), "answers": reference}
    for writer in args.writers:
        rows = []
        for t in teacher:
            q = questions[t["id"]]
            t0 = time.perf_counter()
            r = client.chat.completions.create(
                model=writer, temperature=0, max_tokens=600,
                messages=[{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": q["writer_prompt"]}],
                extra_body={"options": {"num_ctx": 8192, "repeat_penalty": 1.0}})
            answer = r.choices[0].message.content or ""
            rows.append({**score(answer, q, t["answer"]), "seconds": round(time.perf_counter() - t0, 2),
                         "id": t["id"], "template": t["template"], "answer": answer})
        results["writers"][writer] = {"summary": summarize(rows), "answers": rows}
        print(f"{writer}: {results['writers'][writer]['summary']}")

    print(f"\n{len(teacher)} held-out questions, teacher {args.teacher}")
    print(f"{'writer':<34}{'grounded':>9}{'format':>8}{'teacher #s':>11}{'states ±15':>11}{'words':>7}{'sec':>6}")
    for name, w in results["writers"].items():
        s = w["summary"]
        cell = lambda v, pct=True: "" if v is None else (f"{v:.0%}" if pct else f"{v:g}")  # noqa: E731
        print(f"{name:<34}{cell(s['grounded']):>9}{cell(s['format']):>8}{cell(s['teacher_numbers']):>11}"
              f"{cell(s['states_error']):>11}{cell(s['words'], False):>7}{cell(s['seconds'], False):>6}")
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"compare-{datetime.now():%Y%m%d-%H%M%S}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--writers", nargs="+", required=True)
    main(parser.parse_args())
