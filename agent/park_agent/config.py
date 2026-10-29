"""Builds the gateway and harness from agent/.env: each provider is enabled when its setting exists.

    ANTHROPIC_API_KEY=...            Claude (Haiku for everyday steps, Opus as teacher)
    OPENAI_API_KEY=...               OpenAI (fallback; capped at $3/month)
    LOCAL_LLM_BASE_URL=http://localhost:11434/v1     a local Ollama server, tried first
    MONTHLY_CAP_USD=10
    OPENAI_CAP_USD=3.00              OpenAI's monthly cap on this machine. The ledger is per machine,
                                     so lower it by what the other machine has spent from the same
                                     prepaid balance.
    LOCAL_WRITE_MODEL=park-writer-1.5b   optional: an Ollama model for the writer step only
"""

import os
from pathlib import Path

from park_agent.data import ParkData
from park_agent.gateway import Gateway
from park_agent.harness import Harness
from park_agent.tools import ParkTools
from park_agent.types import ModelSpec

AGENT = Path(__file__).resolve().parents[1]
LOGS = AGENT / "logs"


def load_env(path=AGENT / ".env"):
    """KEY=value lines into os.environ, without overriding what is already set."""
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#"):
                os.environ.setdefault(key.strip(), value.strip())


def build_providers():
    load_env()
    providers = {}
    if os.getenv("LOCAL_LLM_BASE_URL"):
        from park_agent.providers.openai_compatible import OpenAICompatibleProvider
        providers["local"] = OpenAICompatibleProvider.local(os.environ["LOCAL_LLM_BASE_URL"])
    if os.getenv("ANTHROPIC_API_KEY"):
        from park_agent.providers.anthropic_provider import AnthropicProvider
        providers["anthropic"] = AnthropicProvider()
    if os.getenv("OPENAI_API_KEY"):
        from park_agent.providers.openai_compatible import OpenAICompatibleProvider
        providers["openai"] = OpenAICompatibleProvider.openai()
    return providers


def build_harness(data=None):
    gateway = Gateway(providers=build_providers(), ledger_path=LOGS / "ledger.jsonl",
                      monthly_cap_usd=float(os.getenv("MONTHLY_CAP_USD", "10")),
                      provider_caps_usd={"openai": float(os.getenv("OPENAI_CAP_USD", "3.00"))})
    if os.getenv("LOCAL_WRITE_MODEL") and "local" in gateway.providers:
        # A distilled writer (agent/distill): only the answer-writing step goes to it; routing and
        # tool calls stay on the general local model.
        gateway.models.append(ModelSpec(os.environ["LOCAL_WRITE_MODEL"], "local", "local_write", 0.0, 0.0))
        gateway.routes["write"] = ["local_write", *gateway.routes["write"]]
    return Harness(gateway, ParkTools(data or ParkData.load()), turns_dir=LOGS / "turns")
