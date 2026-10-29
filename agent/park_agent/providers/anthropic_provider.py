"""Claude through the official Anthropic SDK.

Opus requests opt into server-side fallback (`fallbacks: "default"`), so a safety decline is re-run
on Anthropic's recommended model inside the same call; Haiku has no server-side fallback. Thinking
is adaptive on both, with depth set by `output_config.effort`. The tool loop appends each response's
full content (thinking blocks included) unchanged, and returns all of a turn's tool results in one
user message.
"""

import asyncio
import json

import anthropic

from park_agent.types import ProviderError, ToolCall, ToolLoopResult

MAX_TOKENS = 16000
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _error(e):
    if isinstance(e, (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError)):
        return ProviderError(f"{type(e).__name__}: {e}", retryable=True)
    if isinstance(e, anthropic.APIStatusError):
        return ProviderError(f"HTTP {e.status_code}: {e.message}", retryable=e.status_code >= 500)
    return ProviderError(str(e), retryable=False)


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, client=None):
        self.client = client or anthropic.AsyncAnthropic(max_retries=1)

    def _messages(self, model):
        """The messages API for a model, with server-side fallback on Opus."""
        if model.startswith("claude-opus"):
            return self.client.beta.messages, {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        return self.client.messages, {}

    async def _create(self, model, **params):
        api, extra = self._messages(model)
        try:
            response = await api.create(model=model, max_tokens=MAX_TOKENS, **extra, **params)
        except anthropic.APIError as e:
            raise _error(e) from e
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            raise ProviderError(f"refusal ({category})", retryable=True)
        if response.stop_reason == "max_tokens":
            raise ProviderError("response hit max_tokens", retryable=False)
        return response

    async def run_tools(self, model, system, prompt, tools, execute, max_rounds, effort, usage):
        specs = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        messages = [{"role": "user", "content": prompt}]
        calls = []
        for round_ in range(1, max_rounds + 1):
            response = await self._create(model, system=system, tools=specs, messages=messages,
                                          output_config={"effort": effort})
            usage.add(response.usage.input_tokens, response.usage.output_tokens)
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                text = "".join(b.text for b in response.content if b.type == "text")
                return ToolLoopResult(text=text, calls=calls, rounds=round_)
            outcomes = await asyncio.gather(*(execute(b.name, b.input) for b in tool_uses))
            results = []
            for block, (result, is_error) in zip(tool_uses, outcomes):
                calls.append(ToolCall(block.name, block.input, result, is_error))
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": result,
                                "is_error": is_error})
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": results})
        raise ProviderError(f"no answer after {max_rounds} tool rounds", retryable=False)

    async def complete_json(self, model, system, prompt, schema, effort, usage):
        response = await self._create(
            model, system=system, messages=[{"role": "user", "content": prompt}],
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}})
        usage.add(response.usage.input_tokens, response.usage.output_tokens)
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ProviderError(f"structured output did not parse: {e}", retryable=True) from e

    async def stream_text(self, model, system, prompt, effort, usage):
        api, extra = self._messages(model)
        try:
            async with api.stream(model=model, max_tokens=MAX_TOKENS, system=system, **extra,
                                  messages=[{"role": "user", "content": prompt}],
                                  output_config={"effort": effort}) as stream:
                async for text in stream.text_stream:
                    yield text
                final = await stream.get_final_message()
        except anthropic.APIError as e:
            raise _error(e) from e
        usage.add(final.usage.input_tokens, final.usage.output_tokens)
        if final.stop_reason == "refusal":
            raise ProviderError("refusal mid-stream", retryable=True)
