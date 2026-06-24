# wiki-agent — Wikipedia-grounded Q&A with Claude

A small system that answers questions by searching English Wikipedia through a
single `search_wikipedia(query)` tool, plus an eval suite that measures how well
it works across question types.

- **Agent model:** Claude Haiku 4.5 (`claude-haiku-4-5`) — deliberately small, so
  the *system prompt* carries the behaviour.
- **Judge model:** Claude Opus 4.8 (`claude-opus-4-8`).
- **Retrieval:** live MediaWiki Action API (no key, always fresh). No hosted
  search / RAG tools are used — we own the retrieval loop.

## Setup

You need an Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # required for the agent + judge
```

**Recommended — [uv](https://docs.astral.sh/uv/) (one command, auto-creates the env):**

```bash
uv sync          # optional; `uv run` does this on first use anyway
```

**Fallback — stdlib venv + pip (no uv required):**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Below, prefix commands with `uv run` (recommended) or just run `python` directly
inside an activated venv.

## Run the prototype

```bash
# Demo mode — one question per category, shows whether search was used:
uv run python -m wiki_agent --demo

# Ask anything:
uv run python -m wiki_agent "What is the capital of the country that hosted the 2016 Olympics?"

# Try a different agent model:
uv run python -m wiki_agent --model claude-sonnet-4-6 "Who painted the Mona Lisa?"
```

Each run prints whether search was used, the queries issued and articles found,
and the grounded answer with citations.

## Run the evals

```bash
uv run python -m wiki_agent.evals.run_evals              # full suite (~30 cases)
uv run python -m wiki_agent.evals.run_evals --limit 5    # quick smoke run
uv run python -m wiki_agent.evals.run_evals --category false_premise
uv run python -m wiki_agent.evals.run_evals --no-judge   # agent + programmatic checks only (cheaper)
```

> Using the venv fallback instead of uv? Drop the `uv run` prefix and call
> `python ...` directly after `source .venv/bin/activate`.

The harness reports programmatic metrics (tool-use appropriateness, abstention
accuracy) and LLM-judge metrics (correctness, faithfulness, citation) broken
down by dimension and category, and writes a full per-case JSON to
`wiki_agent/evals/results_*.json`.

## Layout

```
wiki_agent/
  prompts.py          system prompt + search_wikipedia tool definition
  wikipedia.py        MediaWiki client + result formatting
  agent.py            manual agentic loop (ToolCall / AgentResult dataclasses)
  cli.py              `python -m wiki_agent`
  evals/
    cases.jsonl       test taxonomy (8 categories)
    judge.py          Opus-4.8 LLM judge (rubric + JSON output)
    run_evals.py      harness + aggregation
RATIONALE.md          design rationale & eval findings
```
