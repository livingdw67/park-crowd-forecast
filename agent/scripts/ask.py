"""Ask the agent one question from the command line and print its events.

    .venv\\Scripts\\python agent\\scripts\\ask.py "How busy is Islands of Adventure on Saturday?"
"""

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from park_agent.config import build_harness  # noqa: E402
from park_agent.gateway import RequestBudget  # noqa: E402


async def main(question):
    harness = build_harness()
    print("providers:", ", ".join(harness.gateway.providers) or "none")
    async for event in harness.run(question, uuid.uuid4().hex[:12], RequestBudget(cap_usd=0.25)):
        if event["type"] == "token":
            print(event["text"], end="", flush=True)
        else:
            print(f"\n[{event['type']}] {({k: v for k, v in event.items() if k != 'type'})}")


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "How busy is Islands of Adventure on Saturday?"))
