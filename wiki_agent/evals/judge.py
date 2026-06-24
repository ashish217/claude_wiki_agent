"""LLM-as-judge grading via Pydantic-typed structured outputs.

We grade three dimensions an exact-match check can't capture (correctness,
faithfulness, citation) plus two behavioural flags (abstained, premise_handled).
The output shape is enforced by the Messages API structured-outputs feature
against the ``Judgment`` schema below — no freeform JSON parsing, no manual
validation. The per-field descriptions are the rubric: they're sent to the judge
as part of the schema.

Judge model: Claude Opus 4.8 (strongest grader, low volume; a different/stronger
model than the agent reduces self-grading bias).
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import anthropic
from pydantic import BaseModel, Field

from ..agent import AgentResult

JUDGE_MODEL = "claude-opus-4-8"


class Judgment(BaseModel):
    """Schema-enforced verdict for one answer."""

    correctness: Literal["correct", "partial", "incorrect"] = Field(
        description=(
            "'correct' if the answer conveys the gold key facts and is not wrong on "
            "the main point; 'partial' if incomplete or partly right; 'incorrect' if "
            "the main point is wrong or missing. When the expected behaviour is to "
            "abstain, treat an appropriate 'I couldn't find this / can't know that' "
            "as 'correct'."
        )
    )
    faithfulness: Literal["grounded", "partial", "unsupported", "not_applicable"] = Field(
        description=(
            "Whether the answer's claims are supported by the RETRIEVED TEXT. "
            "'unsupported' if the answer asserts specifics absent from the retrieved "
            "text (a hallucination risk), even if those specifics happen to be true. "
            "'not_applicable' when no search was needed/done (e.g. arithmetic) or for "
            "a pure abstention."
        )
    )
    citation: Literal["present", "weak", "missing"] = Field(
        description=(
            "'present' if it names the Wikipedia article(s) used; 'weak' if vague "
            "('according to Wikipedia'); 'missing' if none. For abstentions, 'missing' "
            "is acceptable and should not be penalised elsewhere."
        )
    )
    abstained: bool = Field(
        description=(
            "True if the system declined to give a substantive answer because the "
            "information is unknowable or not on Wikipedia."
        )
    )
    premise_handled: Literal["yes", "no", "not_applicable"] = Field(
        description=(
            "For a question containing a FALSE premise: 'yes' if the answer "
            "flags/corrects it, 'no' if it plays along. 'not_applicable' if the "
            "question has no false premise."
        )
    )
    rationale: str = Field(description="One or two sentences justifying the grades.")


_SYSTEM = (
    "You are a strict evaluator of a Wikipedia-grounded question-answering system. "
    "You are given a question, the gold key facts, the expected behaviour, the "
    "Wikipedia text the system retrieved, and its answer. Grade only what is in "
    "front of you, applying the definitions in the output schema."
)


def judge_answer(
    case: Dict[str, Any],
    result: AgentResult,
    client: Optional[anthropic.Anthropic] = None,
) -> Optional[Judgment]:
    """Grade one answer, returning a validated ``Judgment`` (or ``None`` on failure)."""
    client = client or anthropic.Anthropic()

    user = (
        f"QUESTION:\n{case['question']}\n\n"
        f"GOLD KEY FACTS:\n{', '.join(case.get('key_facts', [])) or '(none)'}\n\n"
        f"EXPECTED BEHAVIOUR: should_search={case.get('should_search')}, "
        f"should_abstain={case.get('should_abstain')}, category={case.get('category')}\n\n"
        f"RETRIEVED WIKIPEDIA TEXT:\n{result.retrieved_context or '(no search was performed)'}\n\n"
        f"SYSTEM ANSWER:\n{result.answer or '(empty)'}"
    )

    try:
        resp = client.messages.parse(
            model=JUDGE_MODEL,
            max_tokens=1024,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user}],
            output_format=Judgment,
        )
        return resp.parsed_output
    except Exception as exc:  # don't let one judge hiccup abort the whole run
        print(f"  [judge error on {case.get('id')}: {exc}]")
        return None
