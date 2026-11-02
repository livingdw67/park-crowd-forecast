"""Provider-neutral types shared by the gateway, the provider adapters and the agents."""

from dataclasses import dataclass, field
from typing import Awaitable, Callable


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    tier: str              # "fast" or "strong"
    usd_in: float          # per 1M input tokens
    usd_out: float         # per 1M output tokens

    def cost(self, input_tokens, output_tokens):
        return (input_tokens * self.usd_in + output_tokens * self.usd_out) / 1e6


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict       # JSON schema of the arguments


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens, output_tokens):
        self.input_tokens += input_tokens or 0
        self.output_tokens += output_tokens or 0


@dataclass
class ToolCall:
    name: str
    args: dict
    result: str
    is_error: bool = False


@dataclass
class ToolLoopResult:
    text: str
    calls: list = field(default_factory=list)   # ToolCall, in the order they ran
    rounds: int = 0


# Runs one tool call for a provider's loop: (name, args) -> (result text, is_error)
Executor = Callable[[str, dict], Awaitable[tuple]]


class ProviderError(Exception):
    """A failed model call. `retryable` means another provider may succeed (rate limit, overload,
    network, refusal); a non-retryable error is a bug in the request and is raised as is."""

    def __init__(self, message, retryable):
        super().__init__(message)
        self.retryable = retryable


class BudgetExceeded(Exception):
    """The request's spending cap would be passed by the next call."""
