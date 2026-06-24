"""The Wikipedia-grounded answering agent: system prompt, tool definition, loop.

We run a *manual* agentic loop rather than the SDK tool-runner so we capture the
full trace (which queries were issued, what came back). The eval harness needs
that trace for two things the assignment cares about: showing whether search was
used, and grading *faithfulness* (was the answer supported by retrieved text?).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic

from .prompts import SYSTEM_PROMPT, TOOL_DEF
from .wikipedia import format_results, search_wikipedia

# Claude Haiku 4.5 — the agent model. Deliberately a small model: it makes the
# system prompt do the work, which is the point of the exercise. Note Haiku 4.5
# does NOT support adaptive thinking or the effort param (those are 4.6+), so
# the loop runs without thinking and the prompt carries all the behaviour.
DEFAULT_MODEL = "claude-haiku-4-5"
MAX_TURNS = 6
MAX_TOKENS = 4096


@dataclass
class ToolCall:
    """One search_wikipedia invocation and what it returned."""

    query: str
    result_titles: List[str]
    result_text: str
    is_error: bool = False


@dataclass
class Usage:
    """Token usage accumulated across every model call in one answer."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def add(self, u: Any) -> None:
        """Add one response's ``usage`` object (fields may be missing/None)."""
        self.input_tokens += getattr(u, "input_tokens", 0) or 0
        self.output_tokens += getattr(u, "output_tokens", 0) or 0
        self.cache_read_input_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
        self.cache_creation_input_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0


@dataclass
class AgentResult:
    """An answer plus the trace needed to display and grade it."""

    answer: str
    tool_calls: List[ToolCall]
    n_model_calls: int
    stop_reason: str
    usage: Usage = field(default_factory=Usage)

    @property
    def searched(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def retrieved_context(self) -> str:
        """Concatenated tool output — what faithfulness is graded against."""
        return "\n\n".join(tc.result_text for tc in self.tool_calls)


def _text_of(content_blocks: List[Any]) -> str:
    return "\n".join(b.text for b in content_blocks if getattr(b, "type", None) == "text").strip()


def answer_question(
    question: str,
    model: str = DEFAULT_MODEL,
    client: Optional[anthropic.Anthropic] = None,
) -> AgentResult:
    """Answer ``question`` with Wikipedia grounding, returning an ``AgentResult``."""
    if client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Export it before running.")
        client = anthropic.Anthropic()

    # system / messages / tool_results stay plain dicts: they are the Anthropic
    # SDK's wire format, not our own data model.
    system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    messages: List[Dict[str, Any]] = [{"role": "user", "content": question}]
    tool_calls: List[ToolCall] = []
    usage = Usage()
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
        usage.add(resp.usage)
        last_text = _text_of(resp.content) or last_text

        if resp.stop_reason != "tool_use":
            return AgentResult(last_text, tool_calls, n_model_calls, resp.stop_reason, usage)

        # Execute every tool_use block in this turn, return all results together.
        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use" or block.name != "search_wikipedia":
                continue
            query = (block.input or {}).get("query", "")
            try:
                hits = search_wikipedia(query)
                call = ToolCall(query, [h.title for h in hits], format_results(query, hits))
            except Exception as exc:  # network/API hiccup — tell the model, let it adapt
                call = ToolCall(query, [], f"search_wikipedia failed: {exc}", is_error=True)
            tool_calls.append(call)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": call.result_text,
                    "is_error": call.is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    # Turn budget exhausted while the model still wanted to search. Tell it the
    # search limit is reached, then force one final tool-free call so we always
    # synthesize an answer from what we gathered, instead of returning a
    # leftover preamble (or nothing). (Haiku 4.5 doesn't support mid-conversation
    # system messages, so this goes in a user turn.)
    messages.append(
        {
            "role": "user",
            "content": (
                "You have reached the search limit and cannot search again. Give "
                "your best final answer using only the information already "
                "retrieved above. If it is insufficient, say what you found and "
                "what remains unconfirmed. Do not propose further searches."
            ),
        }
    )
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        tools=[TOOL_DEF],
        tool_choice={"type": "none"},
        messages=messages,
    )
    n_model_calls += 1
    usage.add(resp.usage)
    return AgentResult(_text_of(resp.content) or last_text, tool_calls, n_model_calls, "max_turns", usage)
