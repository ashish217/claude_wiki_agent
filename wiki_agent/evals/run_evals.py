"""Eval harness: run the agent over the test cases, grade, and report.

    python -m wiki_agent.evals.run_evals                # full suite
    python -m wiki_agent.evals.run_evals --limit 5      # quick smoke run
    python -m wiki_agent.evals.run_evals --category temporal
    python -m wiki_agent.evals.run_evals --no-judge     # agent + programmatic only

Reports two layers:
  - programmatic (deterministic, from the trace): tool-use appropriateness,
    abstention accuracy, avg # searches.
  - LLM judge: correctness / faithfulness / citation by dimension and category.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import anthropic

from ..agent import DEFAULT_MODEL, answer_question
from .judge import judge_answer

CASES_PATH = Path(__file__).with_name("cases.jsonl")
_CORRECT = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
_FAITH = {"grounded": 1.0, "partial": 0.5, "unsupported": 0.0}


def load_cases(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pct(num: float, den: float) -> str:
    return f"{(100 * num / den):.0f}%" if den else "  -"


def run(args) -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set.")

    cases = load_cases(Path(args.cases))
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    if args.limit:
        cases = cases[: args.limit]

    client = anthropic.Anthropic()
    rows: List[Dict[str, Any]] = []

    print(f"Running {len(cases)} case(s) | agent={args.model} | judge={'off' if args.no_judge else 'claude-opus-4-8'}\n")
    for case in cases:
        result = answer_question(case["question"], model=args.model, client=client)
        judgment = {} if args.no_judge else judge_answer(case, result, client=client)

        # Programmatic check: did tool-use match expectations?
        exp = case.get("should_search")
        search_ok = None if exp is None else (result["searched"] == exp)
        abstain_ok = bool(judgment.get("abstained")) == bool(case.get("should_abstain")) if judgment else None

        row = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "answer": result["answer"],
            "searched": result["searched"],
            "n_searches": len(result["tool_calls"]),
            "queries": [tc["query"] for tc in result["tool_calls"]],
            "search_ok": search_ok,
            "abstain_ok": abstain_ok,
            "judgment": judgment,
        }
        rows.append(row)

        c = judgment.get("correctness", "?")
        flags = []
        if search_ok is False:
            flags.append("SEARCH-MISMATCH")
        if abstain_ok is False:
            flags.append("ABSTAIN-MISMATCH")
        flag_str = ("  ⚠ " + ",".join(flags)) if flags else ""
        print(f"  [{case['id']:<16}] correctness={c:<9} searches={len(result['tool_calls'])}{flag_str}")

    _report(rows, args)

    out = Path(args.out) if args.out else CASES_PATH.with_name(
        f"results_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out.write_text(json.dumps({"model": args.model, "rows": rows}, indent=2))
    print(f"\nFull results written to {out}")


def _report(rows: List[Dict[str, Any]], args) -> None:
    print("\n" + "=" * 70 + "\nPROGRAMMATIC (from trace)\n" + "-" * 70)
    avg_searches = sum(r["n_searches"] for r in rows) / len(rows)
    search_checked = [r for r in rows if r["search_ok"] is not None]
    search_pass = sum(1 for r in search_checked if r["search_ok"])
    abstain_checked = [r for r in rows if r["abstain_ok"] is not None]
    abstain_pass = sum(1 for r in abstain_checked if r["abstain_ok"])
    print(f"  avg searches/question      : {avg_searches:.2f}")
    print(f"  tool-use appropriateness   : {_pct(search_pass, len(search_checked))}  ({search_pass}/{len(search_checked)})")
    print(f"  abstention accuracy        : {_pct(abstain_pass, len(abstain_checked))}  ({abstain_pass}/{len(abstain_checked)})")

    if args.no_judge:
        return

    judged = [r for r in rows if r["judgment"] and "correctness" in r["judgment"]]
    if not judged:
        print("\n(no valid judgments)")
        return

    print("\n" + "=" * 70 + "\nJUDGE — overall\n" + "-" * 70)
    corr = sum(_CORRECT.get(r["judgment"]["correctness"], 0) for r in judged) / len(judged)
    faith_rows = [r for r in judged if r["judgment"].get("faithfulness") in _FAITH]
    faith = (sum(_FAITH[r["judgment"]["faithfulness"]] for r in faith_rows) / len(faith_rows)) if faith_rows else 0
    cite_rows = [r for r in judged if not r["judgment"].get("abstained")]
    cite_present = sum(1 for r in cite_rows if r["judgment"].get("citation") == "present")
    fp_rows = [r for r in judged if r["category"] == "false_premise"]
    fp_pass = sum(1 for r in fp_rows if r["judgment"].get("premise_handled") == "yes")
    print(f"  correctness (weighted)     : {corr * 100:.0f}%")
    print(f"  faithfulness (where applic): {faith * 100:.0f}%  (n={len(faith_rows)})")
    print(f"  citation present           : {_pct(cite_present, len(cite_rows))}  ({cite_present}/{len(cite_rows)})")
    print(f"  false-premise handled      : {_pct(fp_pass, len(fp_rows))}  ({fp_pass}/{len(fp_rows)})")

    print("\n" + "=" * 70 + "\nJUDGE — correctness by category\n" + "-" * 70)
    by_cat: Dict[str, List[float]] = defaultdict(list)
    for r in judged:
        by_cat[r["category"]].append(_CORRECT.get(r["judgment"]["correctness"], 0))
    for cat in sorted(by_cat):
        scores = by_cat[cat]
        print(f"  {cat:<22} {sum(scores) / len(scores) * 100:>4.0f}%  (n={len(scores)})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the Wikipedia-agent eval suite.")
    p.add_argument("--cases", default=str(CASES_PATH))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--category", default=None)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
