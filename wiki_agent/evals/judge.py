"""LLM-as-judge grading via Pydantic-typed structured outputs.

The judge returns a single ``Judgment`` whose shape is enforced by the Messages
API structured-outputs feature — no freeform JSON parsing. The harness
(run_evals.py) turns this into metrics; in particular **faithfulness is computed
from the claim list**, not asked for as a single score:

    faithfulness = (# claims labelled "supported") / (# total claims)

decomposing the answer into atomic claims and labelling each against the
retrieved text is the only reliable way to measure whether the model actually
*used* what it retrieved (vs. smuggling in parametric knowledge).

Judge model: Claude Opus 4.8 (strongest grader, low volume; a different/stronger
model than the agent reduces self-grading bias).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import anthropic
from pydantic import BaseModel, Field

from ..agent import AgentResult

JUDGE_MODEL = "claude-opus-4-8"


class Claim(BaseModel):
    """One atomic factual claim from the answer, labelled against retrieved text."""

    text: str = Field(description="A single atomic factual claim restated from the system's answer.")
    label: Literal["supported", "unsupported", "contradicted"] = Field(
        description=(
            "'supported' = the RETRIEVED TEXT directly backs this claim. "
            "'contradicted' = the retrieved text states otherwise. "
            "'unsupported' = the retrieved text neither backs nor contradicts it "
            "(e.g. the claim came from the model's own knowledge, not the search)."
        )
    )


class Judgment(BaseModel):
    """Schema-enforced verdict for one answer."""

    correctness: Literal["correct", "partial", "incorrect"] = Field(
        description=(
            "'correct' if the answer conveys the gold key facts and is not wrong on the "
            "main point; 'partial' if incomplete or partly right; 'incorrect' if the main "
            "point is wrong or missing. When the expected behaviour is to abstain, treat an "
            "appropriate 'I couldn't find this / can't know that' as 'correct'."
        )
    )
    claims: List[Claim] = Field(
        description=(
            "Every distinct factual claim the answer makes, each labelled against the "
            "RETRIEVED TEXT. Decompose into atomic, separately-checkable claims. Return an "
            "empty list if the answer makes no verifiable factual claims (e.g. a pure "
            "abstention or refusal)."
        )
    )
    query_quality: Literal["good", "adequate", "poor", "not_applicable"] = Field(
        description=(
            "Judge the SEARCH QUERIES ISSUED: did they retrieve (or correctly establish the "
            "absence of) the content needed to answer? 'good' = well-targeted, surfaced what "
            "was needed; 'adequate' = eventually got there but clumsy/indirect; 'poor' = "
            "badly formed or missed retrievable content. Do NOT penalise good queries when "
            "the information genuinely isn't on Wikipedia. 'not_applicable' if no search was "
            "performed."
        )
    )
    abstained: bool = Field(
        description="True if the system declined to give a substantive answer because the information is unknowable or not on Wikipedia."
    )
    disambiguated: Literal["yes", "no", "not_applicable"] = Field(
        description=(
            "For an ambiguous term with several distinct meanings: 'yes' if the answer "
            "acknowledges the main alternative meanings before answering one; 'no' if it "
            "silently answers a single meaning. 'not_applicable' if the term is not ambiguous."
        )
    )
    premise_handled: Literal["yes", "no", "not_applicable"] = Field(
        description=(
            "For a question containing a FALSE premise: 'yes' if the answer flags/corrects it, "
            "'no' if it plays along. 'not_applicable' if the question has no false premise."
        )
    )
    citation: Literal["present", "weak", "missing"] = Field(
        description=(
            "'present' if it names the Wikipedia article(s) used; 'weak' if vague ('according "
            "to Wikipedia'); 'missing' if none. For abstentions/no-search answers, 'missing' "
            "is acceptable."
        )
    )
    rationale: str = Field(description="One or two sentences justifying the grades.")


_SYSTEM = (
    "You are a strict evaluator of a Wikipedia-grounded question-answering system. You are "
    "given a question, the gold key facts, the expected behaviour, the search queries the "
    "system issued, the Wikipedia text it retrieved, and its answer. Grade only what is in "
    "front of you, applying the definitions in the output schema. For the claim list, "
    "decompose the answer into atomic factual claims and label each strictly against the "
    "retrieved text."
)


def judge_answer(
    case: Dict[str, Any],
    result: AgentResult,
    client: Optional[anthropic.Anthropic] = None,
) -> Optional[Judgment]:
    """Grade one answer, returning a validated ``Judgment`` (or ``None`` on failure)."""
    client = client or anthropic.Anthropic()

    queries = [tc.query for tc in result.tool_calls]
    user = (
        f"QUESTION:\n{case['question']}\n\n"
        f"GOLD KEY FACTS:\n{', '.join(case.get('key_facts', [])) or '(none)'}\n\n"
        f"EXPECTED BEHAVIOUR: category={case.get('category')}, "
        f"should_search={case.get('should_search')}, should_abstain={case.get('should_abstain')}\n\n"
        f"SEARCH QUERIES ISSUED:\n{queries or '(none — no search performed)'}\n\n"
        f"RETRIEVED WIKIPEDIA TEXT:\n{result.retrieved_context or '(no search was performed)'}\n\n"
        f"SYSTEM ANSWER:\n{result.answer or '(empty)'}"
    )

    try:
        resp = client.messages.parse(
            model=JUDGE_MODEL,
            max_tokens=2048,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user}],
            output_format=Judgment,
        )
        return resp.parsed_output
    except Exception as exc:  # don't let one judge hiccup abort the whole run
        print(f"  [judge error on {case.get('id')}: {exc}]")
        return None
