"""The Wikipedia-grounded answering agent: system prompt, tool definition, loop.

We run a *manual* agentic loop rather than the SDK tool-runner so we capture the
full trace (which queries were issued, what came back). The eval harness needs
that trace for two things the assignment cares about: showing whether search was
used, and grading *faithfulness* (was the answer supported by retrieved text?).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import anthropic

from .wikipedia import format_results, search_wikipedia

# Claude Haiku 4.5 — the agent model. Deliberately a small model: it makes the
# system prompt do the work, which is the point of the exercise. Note Haiku 4.5
# does NOT support adaptive thinking or the effort param (those are 4.6+), so
# the loop runs without thinking and the prompt carries all the behaviour.
DEFAULT_MODEL = "claude-haiku-4-5"
MAX_TURNS = 6
MAX_TOKENS = 2048

SYSTEM_PROMPT = """\
You are a careful research assistant that answers questions using English \
Wikipedia as your source of truth. You have one tool, search_wikipedia(query), \
which returns the top matching Wikipedia articles with their introductions.

How to work:

1. SEARCH FIRST. For any question about facts, people, places, events, dates, \
numbers, or anything that may have changed over time, search Wikipedia before \
answering. Do not answer specific facts from memory — your training data may be \
outdated or wrong. The only exception is trivial reasoning (e.g. simple \
arithmetic) that needs no external source; for those, just answer.

2. SEARCH WELL. Query with concise entity or topic terms, not the user's whole \
sentence. For multi-step questions, break them into parts and search for each in \
turn — first find the entity, then search again for the follow-up fact. If the \
first results don't contain the answer, refine the query and search again.

3. GROUND YOUR ANSWER. Base your answer only on the content the tool returned. \
Do not add facts that aren't supported by what you retrieved. If the retrieved \
articles don't actually contain the answer, say so rather than filling the gap \
from memory.

4. CITE. Name the Wikipedia article(s) you used so the user can verify.

5. BE HONEST AND CALIBRATED.
   - If Wikipedia doesn't have the answer after a genuine search, say you \
couldn't find it on Wikipedia instead of guessing.
   - If the question rests on a false or mistaken premise, point out the \
discrepancy rather than playing along.
   - If the question is ambiguous (an entity with several meanings), state which \
interpretation you're answering and briefly name the alternatives.

6. BE CONCISE. Lead with the direct answer, then a sentence or two of support, \
then your citation.

You cannot answer questions about private, personal, real-time, or future \
information that Wikipedia would not contain — say so plainly."""

TOOL_DEF = {
    "name": "search_wikipedia",
    "description": (
        "Search English Wikipedia and return the top matching articles, each "
        "with its title, URL, and a plain-text extract of the article's "
        "introduction. Call this to look up factual, current, or specific "
        "information before answering. Use concise entity/topic search terms. "
        "You may call it multiple times to refine a query or to follow up on a "
        "different entity in a multi-step question."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Concise search terms (an entity or topic), not a full sentence.",
            }
        },
        "required": ["query"],
    },
}


def _text_of(content_blocks: List[Any]) -> str:
    return "\n".join(b.text for b in content_blocks if getattr(b, "type", None) == "text").strip()


def answer_question(
    question: str,
    model: str = DEFAULT_MODEL,
    client: Optional[anthropic.Anthropic] = None,
) -> Dict[str, Any]:
    """Answer ``question`` with Wikipedia grounding.

    Returns a dict with the answer plus a trace:
        answer, searched, tool_calls (query/result_titles/result_text),
        retrieved_context, n_model_calls, stop_reason.
    """
    if client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Export it before running.")
        client = anthropic.Anthropic()

    system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    messages: List[Dict[str, Any]] = [{"role": "user", "content": question}]
    tool_calls: List[Dict[str, Any]] = []
    n_model_calls = 0
    last_text = ""

    for _ in range(MAX_TURNS):
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=[TOOL_DEF],
            messages=messages,
        )
        n_model_calls += 1
        last_text = _text_of(resp.content) or last_text

        if resp.stop_reason != "tool_use":
            return _result(last_text, tool_calls, n_model_calls, resp.stop_reason)

        # Execute every tool_use block in this turn, return all results together.
        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use" or block.name != "search_wikipedia":
                continue
            query = (block.input or {}).get("query", "")
            try:
                hits = search_wikipedia(query)
                result_text = format_results(query, hits)
                titles = [h.title for h in hits]
                is_error = False
            except Exception as exc:  # network/API hiccup — tell the model, let it adapt
                result_text = f"search_wikipedia failed: {exc}"
                titles = []
                is_error = True
            tool_calls.append({"query": query, "result_titles": titles, "result_text": result_text})
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    # Ran out of turns; return whatever text we last produced.
    return _result(last_text, tool_calls, n_model_calls, "max_turns")


def _result(answer, tool_calls, n_model_calls, stop_reason) -> Dict[str, Any]:
    return {
        "answer": answer,
        "searched": len(tool_calls) > 0,
        "tool_calls": tool_calls,
        "retrieved_context": "\n\n".join(tc["result_text"] for tc in tool_calls),
        "n_model_calls": n_model_calls,
        "stop_reason": stop_reason,
    }
