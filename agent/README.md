# Park agent

Answers questions about the crowd forecast and staffing plans in plain English: "How busy will
Universal Studios be on Saturday?", "Why is next Friday so busy?", "What would it cost to staff
Islands of Adventure for a crowd score of 80?". Every number in an answer comes from a tool call
against the project's own data and optimizer, and a validator checks that before the answer is
final.

```
question -> guardrails -> supervisor (route) -> specialists (tools; in parallel when both are
needed) -> writer (streams the answer) -> validator (numbers vs. tool output) -> checkpoint
```

## Pieces

| Piece | File | What it does |
|---|---|---|
| AI gateway | [gateway.py](park_agent/gateway.py) | The one place models are called. Routes each task local first, then the cheapest configured paid model in the tier; fails over on rate limits, overloads and network errors; a circuit breaker skips a provider after repeated failures; per-request, monthly and per-provider spending caps, enforced before each call from its worst-case cost; a JSONL ledger of every call |
| Providers | [providers/](park_agent/providers/) | Anthropic SDK, OpenAI, and any OpenAI-compatible server (Ollama for local models); a fake provider for tests and the load test |
| Tools | [tools.py](park_agent/tools.py) | `get_forecast`, `explain_day`, `get_epic_impact`, `get_staffing_plan`, `what_if`. Staffing tools run the OR-Tools optimizer from `parks_lib`, the same code as the daily job |
| Harness | [harness.py](park_agent/harness.py) | Supervisor routes the question to a forecast analyst, a staffing planner, both (run in parallel) or out of scope; a writer streams the answer; every turn's state is saved as a checkpoint |
| Guardrails | [guardrails.py](park_agent/guardrails.py) | Input: length cap and a screen for instruction-override attempts, before any model sees the question. Output: every number must trace to tool output (directly, as a difference, or as a percent change) |
| API | [api.py](park_agent/api.py) | FastAPI service: `POST /ask` streams events as Server-Sent Events; `/health`, `/ready` (503 when no provider or budget is left), `/metrics` (spend, latency percentiles, failovers). Optional API key, per-client rate limit, in-flight cap with a short queue |
| Data | [data.py](park_agent/data.py) | A snapshot of the Databricks tables plus the repo's seed files. Staffing plans are recomputed rather than read, since they are a pure function of the forecast and the seeds |

The harness is a small explicit state machine rather than LangGraph: each step is a gateway call
with its own task, so routing, failover and budget apply per step, and the run yields events
(route, tool, token, validation, error, done) for the API to stream.

## Model routing

Models and prices live in [models.json](park_agent/models.json). Routine steps (route, analyze,
write) try a local model first, then the fast tier (Claude Haiku or GPT mini, whichever is cheaper
and configured), at about $0.0005-0.006 a question. The strong tier (Claude Opus or GPT-5.5) only runs
as the `teach` task, writing reference answers for distillation.

## Evals

[evals/questions.jsonl](evals/questions.jsonl) has 30 questions: forecast (10), staffing (8), both
specialists (3), edge cases such as dates past the forecast (2), out of scope (4) and prompt
injection (3). [run_evals.py](evals/run_evals.py) grades each run without a model judge:

- **route:** the supervisor picked an allowed route (or the input guardrail blocked it)
- **tools:** the expected tools were called without error
- **grounded:** the validator found every number of the answer in tool output
- **no_error:** the run finished without an error event

The first full run on gpt-5.4-mini passed 29/30 and showed three problems: a question about a date
past the forecast was routed out of scope instead of answered, the writer couldn't see tool errors,
and the validator rejected dates the user had typed. After the fixes, the run passed 28/30 under the
old validator and 30/30 once regraded from the saved turn checkpoints with the fixed one. A run costs
about $0.17. Every run is kept in [evals/results/](evals/results/).

## Load test

[scripts/load_test.py](scripts/load_test.py) runs the real HTTP server and tools with models
replaced by a provider that sleeps for typical latencies, so it measures the service rather than a
model. First token arrived in 1.7 s and the full answer in 2.8 s at every level from 1 to 8
concurrent requests; at 16 the full answer took 5.5 s and none were rejected.

## Distillation (in progress)

[distill/](distill/) trains a small local model to do the writer step, so routine answers need no
paid model:

1. [make_questions.py](distill/make_questions.py) builds writer inputs from question templates by
   calling the tools directly, holding out three forecast weeks so test answers never share tool
   results with training ones.
2. [teach.py](distill/teach.py) has a teacher model answer them through the gateway (the only paid
   step; caps apply). Ungrounded answers are marked and skipped in training.
3. [train_lora.py](distill/train_lora.py) LoRA fine-tunes Qwen2.5-1.5B on the teacher's grounded
   answers, loss on answer tokens only.
4. [export_gguf.py](distill/export_gguf.py) writes the merged model as GGUF and registers it with
   Ollama; set `LOCAL_WRITE_MODEL` to use it for the writer step.
5. [compare.py](distill/compare.py) scores student, base model and teacher on the held-out weeks:
   grounded, format, share of the teacher's numbers stated, and whether the forecast error is quoted.

The pipeline has run end to end with a local qwen2.5:7b as a stand-in teacher. The paid teacher run
and the student's scores are next.

## Run it

From the repo root, with the project venv ([../README.md](../README.md#local-development)):

```
.venv\Scripts\python -m pip install -r agent\requirements.txt
```

Settings go in `agent\.env` (git-ignored); each provider is enabled when its setting exists:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
MONTHLY_CAP_USD=10
OPENAI_CAP_USD=3.00
```

Export the data snapshot (needs a Databricks CLI profile for the workspace; the snapshot is
git-ignored because it holds queue-times data):

```
databricks auth login --host <your workspace URL> --profile free
.venv\Scripts\python agent\scripts\export_snapshot.py
```

Ask a question, run the service, or run the evals:

```
.venv\Scripts\python agent\scripts\ask.py "How busy is Islands of Adventure on Saturday?"
.venv\Scripts\python -m uvicorn park_agent.api:app --app-dir agent --port 8000
.venv\Scripts\python agent\evals\run_evals.py --max-usd 0.50
```

Tests (25, no network or API keys needed):

```
cd agent
..\.venv\Scripts\python -m pytest -q tests
```
