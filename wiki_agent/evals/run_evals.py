"""Eval harness: run the agent over the test cases, grade, and report.

    python -m wiki_agent.evals.run_evals                # full suite
    python -m wiki_agent.evals.run_evals --limit 5      # quick smoke run
    python -m wiki_agent.evals.run_evals --category temporal
    python -m wiki_agent.evals.run_evals --no-judge     # agent + programmatic only

Each run writes two files (shared timestamp) to wiki_agent/evals/results/:
    run_<ts>.json     per-case detail (answer, trace, judgment)
    report_<ts>.json  aggregated metrics (diff these across iterations)

Reports two layers:
  - programmatic (deterministic, from the trace): tool-use appropriateness,
    abstention accuracy, avg # searches, avg tokens.
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
RESULTS_DIR = Path(__file__).with_name("results")
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
    if not cases:
        raise SystemExit("No cases matched (check --category / --cases / --limit).")

    client = anthropic.Anthropic()
    rows: List[Dict[str, Any]] = []

    print(f"Running {len(cases)} case(s) | agent={args.model} | judge={'off' if args.no_judge else 'claude-opus-4-8'}\n")
    for case in cases:
        result = answer_question(case["question"], model=args.model, client=client)
        judgment = None if args.no_judge else judge_answer(case, result, client=client)

        # Programmatic check: did tool-use match expectations?
        exp = case.get("should_search")
        search_ok = None if exp is None else (result.searched == exp)
        abstain_ok = (bool(judgment.abstained) == bool(case.get("should_abstain"))) if judgment else None

        row = {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "answer": result.answer,
            "searched": result.searched,
            "n_searches": len(result.tool_calls),
            "queries": [tc.query for tc in result.tool_calls],
            "tokens": {
                "input": result.usage.input_tokens,
                "output": result.usage.output_tokens,
                "cache_read": result.usage.cache_read_input_tokens,
            },
            "search_ok": search_ok,
            "abstain_ok": abstain_ok,
            "judgment": judgment.model_dump() if judgment else None,
        }
        rows.append(row)

        c = judgment.correctness if judgment else "?"
        flags = []
        if search_ok is False:
            flags.append("SEARCH-MISMATCH")
        if abstain_ok is False:
            flags.append("ABSTAIN-MISMATCH")
        flag_str = ("  ⚠ " + ",".join(flags)) if flags else ""
        print(f"  [{case['id']:<16}] correctness={c:<9} searches={len(result.tool_calls)}{flag_str}")

    aggregates = _report(rows, args)
    _write_outputs(rows, aggregates, args)


def _report(rows: List[Dict[str, Any]], args) -> Dict[str, Any]:
    """Print the report and return the aggregated metrics (single source of truth)."""
    n = len(rows)
    avg_searches = sum(r["n_searches"] for r in rows) / n
    avg_in = sum(r["tokens"]["input"] for r in rows) / n
    avg_out = sum(r["tokens"]["output"] for r in rows) / n
    search_checked = [r for r in rows if r["search_ok"] is not None]
    search_pass = sum(1 for r in search_checked if r["search_ok"])
    abstain_checked = [r for r in rows if r["abstain_ok"] is not None]
    abstain_pass = sum(1 for r in abstain_checked if r["abstain_ok"])

    print("\n" + "=" * 70 + "\nPROGRAMMATIC (from trace)\n" + "-" * 70)
    print(f"  avg searches/question      : {avg_searches:.2f}")
    print(f"  avg tokens/question        : {avg_in:.0f} in / {avg_out:.0f} out")
    print(f"  tool-use appropriateness   : {_pct(search_pass, len(search_checked))}  ({search_pass}/{len(search_checked)})")
    print(f"  abstention accuracy        : {_pct(abstain_pass, len(abstain_checked))}  ({abstain_pass}/{len(abstain_checked)})")

    agg: Dict[str, Any] = {
        "n_cases": n,
        "programmatic": {
            "avg_searches": round(avg_searches, 2),
            "avg_input_tokens": round(avg_in),
            "avg_output_tokens": round(avg_out),
            "tool_use_appropriateness": {"pass": search_pass, "n": len(search_checked)},
            "abstention_accuracy": {"pass": abstain_pass, "n": len(abstain_checked)},
        },
        "judge": None,
    }

    if args.no_judge:
        return agg

    judged = [r for r in rows if r["judgment"] and "correctness" in r["judgment"]]
    if not judged:
        print("\n(no valid judgments)")
        return agg

    corr = sum(_CORRECT.get(r["judgment"]["correctness"], 0) for r in judged) / len(judged)
    faith_rows = [r for r in judged if r["judgment"].get("faithfulness") in _FAITH]
    faith = (sum(_FAITH[r["judgment"]["faithfulness"]] for r in faith_rows) / len(faith_rows)) if faith_rows else 0
    cite_rows = [r for r in judged if not r["judgment"].get("abstained")]
    cite_present = sum(1 for r in cite_rows if r["judgment"].get("citation") == "present")
    fp_rows = [r for r in judged if r["category"] == "false_premise"]
    fp_pass = sum(1 for r in fp_rows if r["judgment"].get("premise_handled") == "yes")

    print("\n" + "=" * 70 + "\nJUDGE — overall\n" + "-" * 70)
    print(f"  correctness (weighted)     : {corr * 100:.0f}%")
    print(f"  faithfulness (where applic): {faith * 100:.0f}%  (n={len(faith_rows)})")
    print(f"  citation present           : {_pct(cite_present, len(cite_rows))}  ({cite_present}/{len(cite_rows)})")
    print(f"  false-premise handled      : {_pct(fp_pass, len(fp_rows))}  ({fp_pass}/{len(fp_rows)})")

    print("\n" + "=" * 70 + "\nJUDGE — correctness by category\n" + "-" * 70)
    by_cat: Dict[str, List[float]] = defaultdict(list)
    for r in judged:
        by_cat[r["category"]].append(_CORRECT.get(r["judgment"]["correctness"], 0))
    cat_scores = {}
    for cat in sorted(by_cat):
        scores = by_cat[cat]
        cat_scores[cat] = {"score": round(sum(scores) / len(scores), 3), "n": len(scores)}
        print(f"  {cat:<22} {sum(scores) / len(scores) * 100:>4.0f}%  (n={len(scores)})")

    agg["judge"] = {
        "n_judged": len(judged),
        "correctness_weighted": round(corr, 3),
        "faithfulness": {"score": round(faith, 3), "n": len(faith_rows)},
        "citation_present": {"pass": cite_present, "n": len(cite_rows)},
        "false_premise_handled": {"pass": fp_pass, "n": len(fp_rows)},
        "correctness_by_category": cat_scores,
    }
    return agg


def _write_outputs(rows: List[Dict[str, Any]], aggregates: Dict[str, Any], args) -> None:
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    run_path = out_dir / f"run_{ts}.json"
    report_path = out_dir / f"report_{ts}.json"
    meta = {"timestamp": ts, "model": args.model, "n_cases": len(rows), "judge": not args.no_judge}
    run_path.write_text(json.dumps({**meta, "rows": rows}, indent=2))
    report_path.write_text(json.dumps({**meta, "report": aggregates}, indent=2))
    print(f"\nPer-case results : {run_path}")
    print(f"Aggregated report: {report_path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the Wikipedia-agent eval suite.")
    p.add_argument("--cases", default=str(CASES_PATH))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--category", default=None)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--out-dir", default=None, help="directory for result files (default: evals/results)")
    args = p.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
