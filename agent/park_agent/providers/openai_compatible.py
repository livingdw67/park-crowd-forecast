"""Chat Completions through the official OpenAI SDK: OpenAI itself, or a local server that speaks the
same API (Ollama at http://localhost:11434/v1).

Local servers reject OpenAI-only options, so `hosted=False` drops `reasoning_effort` and sends
`max_tokens` instead of `max_completion_tokens`.
"""

import asyncio
import json

import openai

from park_agent.types import ProviderError, ToolCall, ToolLoopResult

MAX_TOKENS = 16000


def _error(e):
    if isinstance(e, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)):
        return ProviderError(f"{type(e).__name__}: {e}", retryable=True)
    if isinstance(e, openai.APIStatusError):
        return ProviderError(f"HTTP {e.status_code}: {e.message}", retryable=e.status_code >= 500)
    return ProviderError(str(e), retryable=False)


class OpenAICompatibleProvider:
    def __init__(self, name, client, hosted=True):
        self.name, self.client, self.hosted = name, client, hosted

    @classmethod
    def openai(cls, api_key=None):
        return cls("openai", openai.AsyncOpenAI(api_key=api_key, max_retries=1))

    @classmethod
    def local(cls, base_url="http://localhost:11434/v1"):
        return cls("local", openai.AsyncOpenAI(base_url=base_url, api_key="local", max_retries=0), hosted=False)

    def _options(self, effort):
        if self.hosted:
            return {"max_completion_tokens": MAX_TOKENS, "reasoning_effort": effort}
        return {"max_tokens": MAX_TOKENS}

    async def _create(self, **params):
        try:
            response = await self.client.chat.completions.create(**params)
        except openai.APIError as e:
            raise _error(e) from e
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ProviderError("response hit the token limit", retryable=False)
        if choice.finish_reason == "content_filter":
            raise ProviderError("refusal (content filter)", retryable=True)
        return response, choice.message

    async def run_tools(self, model, system, prompt, tools, execute, max_rounds, effort, usage):
        specs = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                   "parameters": t.parameters}} for t in tools]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        calls = []
        for round_ in range(1, max_rounds + 1):
            response, message = await self._create(model=model, messages=messages, tools=specs, **self._options(effort))
            if response.usage:
                usage.add(response.usage.prompt_tokens, response.usage.completion_tokens)
            if not message.tool_calls:
                return ToolLoopResult(text=message.content or "", calls=calls, rounds=round_)
            messages.append(message.model_dump(exclude_none=True))
            parsed = []
            for tc in message.tool_calls:
                try:
                    parsed.append((tc, json.loads(tc.function.arguments or "{}"), None))
                except json.JSONDecodeError:
                    parsed.append((tc, {}, f"arguments were not valid JSON: {tc.function.arguments!r}"))
            outcomes = await asyncio.gather(*(execute(tc.function.name, args) if err is None else _failed(err)
                                              for tc, args, err in parsed))
            for (tc, args, _), (result, is_error) in zip(parsed, outcomes):
                calls.append(ToolCall(tc.function.name, args, result, is_error))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        raise ProviderError(f"no answer after {max_rounds} tool rounds", retryable=False)

    async def complete_json(self, model, system, prompt, schema, effort, usage):
        response, message = await self._create(
            model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": {"name": "result", "schema": schema, "strict": True}},
            **self._options(effort))
        if response.usage:
            usage.add(response.usage.prompt_tokens, response.usage.completion_tokens)
        try:
            return json.loads(message.content or "")
        except json.JSONDecodeError as e:
            raise ProviderError(f"structured output did not parse: {e}", retryable=True) from e

    async def stream_text(self, model, system, prompt, effort, usage):
        try:
            stream = await self.client.chat.completions.create(
                model=model, stream=True, stream_options={"include_usage": True},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                **self._options(effort))
            async for chunk in stream:
                if chunk.usage:
                    usage.add(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.APIError as e:
            raise _error(e) from e


async def _failed(message):
    return message, True
