"""Checks on what goes in and what comes out.

Input: a length cap and a screen for instruction-override attempts, run before any model sees the
question. Output: every number in an answer must trace to tool output, either directly, as the
difference between two tool numbers, or as the percent change between them. Dates are checked the
same way through their parts (year, month, day), and 12-hour times are converted first ("9pm" is 21).
Integers from 0 to 10 pass as counting words ("the 3 busiest days").
"""

import json
import re
from dataclasses import dataclass, field

MAX_QUESTION_CHARS = 1000
INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts)|disregard (the|your) (rules|instructions)"
    r"|(reveal|print|show) (me )?(your|the) (system )?prompt|you are now|developer mode|jailbreak"
    r"|<\s*/?\s*(system|script)\b", re.IGNORECASE)
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
_TIME = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


class Blocked(Exception):
    """The question was refused before reaching a model."""


def check_question(question):
    q = (question or "").strip()
    if not q:
        raise Blocked("empty question")
    if len(q) > MAX_QUESTION_CHARS:
        raise Blocked(f"questions are limited to {MAX_QUESTION_CHARS} characters")
    if INJECTION.search(q):
        raise Blocked("the question tries to change the assistant's instructions")
    return q


def _numbers(text):
    """Numbers in free text, with 12-hour times turned into 24-hour hours."""
    def to_24h(m):
        hour = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "p" else 0)
        return f" {hour} "
    text = _TIME.sub(to_24h, text)
    found = []
    for raw in _NUMBER.findall(text):
        value = raw.replace(",", "")
        found.append((raw, float(value), len(value.split(".")[1]) if "." in value else 0))
    return found


def tool_numbers(outputs):
    """Every number in tool output: values, numbers inside keys, and the parts of ISO dates."""
    values = set()

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                values.update(float(n) for n in re.findall(r"\d+", str(k)))
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, bool) or node is None:
            return
        elif isinstance(node, (int, float)):
            values.add(float(node))
        elif isinstance(node, str):
            for y, m, d in re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", node):
                values.update({float(y), float(m), float(d)})
            values.update(n for _, n, _ in _numbers(node))

    for out in outputs:
        walk(json.loads(out) if isinstance(out, str) else out)
    return values


@dataclass
class Check:
    ok: bool
    ungrounded: list = field(default_factory=list)


def check_answer(answer, outputs):
    """Every number in `answer` must trace to the tool outputs (see the module docstring)."""
    allowed = sorted(tool_numbers(outputs))
    text = answer
    for month, day in re.findall(r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2})\b", answer, re.IGNORECASE):
        if float(day) in allowed:
            text = re.sub(rf"\b{month}\s+{day}\b", month, text, count=1, flags=re.IGNORECASE)
    ungrounded = []
    for raw, value, places in _numbers(text):
        if value.is_integer() and 0 <= value <= 10:
            continue
        if not _traceable(abs(value), places, allowed):
            ungrounded.append(raw)
    return Check(ok=not ungrounded, ungrounded=ungrounded)


def _traceable(value, places, allowed):
    def same(x):
        return round(x, places) == round(value, places)
    if any(same(a) or same(abs(a)) for a in allowed):
        return True
    for i, a in enumerate(allowed):
        for b in allowed[i + 1:]:
            if same(b - a) or (a and same(abs(b - a) / abs(a) * 100)) or (b and same(abs(b - a) / abs(b) * 100)):
                return True
    return False
