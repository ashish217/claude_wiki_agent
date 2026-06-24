"""LLM-as-judge grading for open-ended quality dimensions.

We grade three dimensions an exact-match check can't capture:
  - correctness   : is the answer factually right vs. the gold key facts?
  - faithfulness  : is the answer supported by the retrieved Wikipedia text
                    (rather than smuggled in from the model's parametric memory)?
  - citation      : did it attribute the Wikipedia article(s) it used?
plus two behavioural flags the negative-case categories depend on:
  - abstained        : did it decline to answer / say it couldn't find it?
  - premise_handled  : did it flag a false premise in the question?

Judge model: Claude Opus 4.8 (the strongest grader, low volume). Using a
different/stronger model than the agent reduces self-grading bias. We ask for a
strict JSON object and validate it — version-agnostic across SDK releases.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import anthropic

JUDGE_MODEL = "claude-opus-4-8"

_RUBRIC = """\
You are a strict evaluator of a Wikipedia-grounded question-answering system.
You are given a question, the gold key facts, what the system was expected to do,
the Wikipedia text the system retrieved, and the system's answer. Grade only what
is in front of you.

Return ONLY a JSON object (no prose, no code fences) with exactly these keys:
{
  "correctness": "correct" | "partial" | "incorrect",
  "faithfulness": "grounded" | "partial" | "unsupported" | "not_applicable",
  "citation": "present" | "weak" | "missing",
  "abstained": true | false,
  "premise_handled": "yes" | "no" | "not_applicable",
  "rationale": "<one or two sentences>"
}

Guidance:
- correctness: "correct" if the answer conveys the gold key facts and is not
  wrong on the main point; "partial" if incomplete or partly right; "incorrect"
  if the main point is wrong or missing. For a question whose expected behaviour
  is to abstain, treat an appropriate "I couldn't find this / can't know that"
  as "correct".
- faithfulness: judge whether the answer's claims are supported by the RETRIEVED
  TEXT. "unsupported" if the answer asserts specifics absent from the retrieved
  text (a hallucination risk), even if those specifics happen to be true.
  "not_applicable" when no search was needed/done (e.g. arithmetic) or for a
  pure abstention.
- citation: "present" if it names the Wikipedia article(s) used; "weak" if vague
  ("according to Wikipedia"); "missing" if none. For abstentions, "missing" is
  acceptable and should not be penalised elsewhere.
- abstained: true if the system declined to give a substantive answer because the
  information is unknowable or not on Wikipedia.
- premise_handled: for a question containing a FALSE premise, "yes" if the answer
  flags/corrects the false premise, "no" if it plays along. "not_applicable" if
  the question has no false premise.
"""

_VALID = {
    "correctness": {"correct", "partial", "incorrect"},
    "faithfulness": {"grounded", "partial", "unsupported", "not_applicable"},
    "citation": {"present", "weak", "missing"},
    "premise_handled": {"yes", "no", "not_applicable"},
}


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _text_of(content_blocks) -> str:
    return "".join(b.text for b in content_blocks if getattr(b, "type", None) == "text").strip()


def judge_answer(
    case: Dict[str, Any],
    result: Dict[str, Any],
    client: Optional[anthropic.Anthropic] = None,
) -> Dict[str, Any]:
    client = client or anthropic.Anthropic()

    user = (
        f"QUESTION:\n{case['question']}\n\n"
        f"GOLD KEY FACTS:\n{', '.join(case.get('key_facts', [])) or '(none)'}\n\n"
        f"EXPECTED BEHAVIOUR: should_search={case.get('should_search')}, "
        f"should_abstain={case.get('should_abstain')}, category={case.get('category')}\n\n"
        f"RETRIEVED WIKIPEDIA TEXT:\n{result.get('retrieved_context') or '(no search was performed)'}\n\n"
        f"SYSTEM ANSWER:\n{result.get('answer') or '(empty)'}"
    )

    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1024,
        system=_RUBRIC,
        messages=[{"role": "user", "content": user}],
    )
    raw = _text_of(resp.content)
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return {"_error": "judge returned non-JSON", "_raw": raw}

    # Light validation so a malformed label doesn't silently skew aggregates.
    for key, allowed in _VALID.items():
        if data.get(key) not in allowed:
            data[key] = data.get(key) or "missing"
    data["abstained"] = bool(data.get("abstained", False))
    return data
