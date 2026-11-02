"""A scripted provider for tests and offline evals: no network, no cost.

`script` is a list of steps consumed one per call. A step is either an exception to raise, or a dict:
  {"tool_calls": [(name, args), ...], "text": "...", "json": {...}, "chunks": [...], "usage": (in, out)}
For run_tools, a step's tool calls run through the real executor before its text is returned.
"""

from park_agent.types import ToolCall, ToolLoopResult


class FakeProvider:
    def __init__(self, name, script):
        self.name, self.script, self.calls = name, list(script), []

    def _next(self, kind, model):
        self.calls.append((kind, model))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    async def run_tools(self, model, system, prompt, tools, execute, max_rounds, effort, usage):
        step = self._next("run_tools", model)
        usage.add(*step.get("usage", (100, 20)))
        calls = []
        for name, args in step.get("tool_calls", []):
            result, is_error = await execute(name, args)
            calls.append(ToolCall(name, args, result, is_error))
        return ToolLoopResult(text=step.get("text", ""), calls=calls, rounds=1 + bool(calls))

    async def complete_json(self, model, system, prompt, schema, effort, usage):
        step = self._next("complete_json", model)
        usage.add(*step.get("usage", (50, 10)))
        return step["json"]

    async def stream_text(self, model, system, prompt, effort, usage):
        step = self._next("stream_text", model)
        usage.add(*step.get("usage", (200, 50)))
        for chunk in step.get("chunks", []):
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk
