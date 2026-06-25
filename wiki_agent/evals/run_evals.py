"""Eval harness: run the agent over the test cases, grade, and report.

    python -m wiki_agent.evals.run_evals                # full suite
    python -m wiki_agent.evals.run_evals --limit 5      # quick smoke run
    python -m wiki_agent.evals.run_evals --category temporal
    python -m wiki_agent.evals.run_evals --no-judge     # agent + programmatic only

Metrics (note the DIFFERENT denominators — each metric is scoped to the cases
where it is meaningful):

  pass_rate                 passing cases / all judged cases
  grounding_violation_rate  cases that didn't search / cases where should_search=true
  query_quality             mean(good=1/adequate=.5/poor=0) / cases that searched
  faithfulness (micro)      Σ supported claims / Σ total claims, over cases that
                            searched AND answered substantively (claim-level)

A case "passes" when it does everything right for its category: grounded when it
should be, correct, and (per category) disambiguated / premise-corrected /
abstained, with no contradicted claims and faithfulness >= threshold.

Each run writes run_<ts>.json (per-case detail) and report_<ts>.json (aggregates)
to wiki_agent/evals/results/.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic

from ..agent import DEFAULT_MODEL, answer_question
from .judge import judge_answer

CASES_PATH = Path(__file__).with_name("cases.jsonl")
RESULTS_DIR = Path(__file__).with_name("results")

_CORRECT = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
_QQ = {"good": 1.0, "adequate": 0.5, "poor": 0.0}
# Per-sample faithfulness pass bar. Set to 0.7 (tolerates one unsupported claim in
# a 4+-claim answer) to absorb judge-side claim-decomposition noise on short
# answers, where a single flipped label moves the score ~0.2. Contradicted claims
# are still a hard fail via a separate gate, so this only loosens tolerance for
# *unsupported* (not hallucinated-false) claims.
_FAITH_PASS_THRESHOLD = 0.7


def load_cases(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pct(num: float, den: float) -> str:
    return f"{(100 * num / den):.0f}%" if den else "  n/a"


def _mean(xs: List[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def score_case(case: Dict[str, Any], result, judgment: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-case metrics. ``judgment`` is the judge's model_dump dict (or None)."""
    searched = result.searched
    should_search = bool(case.get("should_search"))
    should_abstain = bool(case.get("should_abstain"))

    # Grounding: only a violation when the case *should* have searched and didn't.
    grounding_applicable = should_search
    grounding_violation = should_search and not searched

    qq_score = None
    faithfulness = None
    supported = contradicted = total_claims = 0
    faith_applicable = False
    passed = None
    fail_reasons = None
    is_substantive = False
    fully_grounded = False

    if judgment is not None:
        if searched:
            qq_score = _QQ.get(judgment.get("query_quality"))  # None when not_applicable

        claims = judgment.get("claims") or []
        supported = sum(1 for c in claims if c.get("label") == "supported")
        contradicted = sum(1 for c in claims if c.get("label") == "contradicted")
        total_claims = len(claims)
        # Faithfulness only applies when the agent searched AND made substantive claims.
        faith_applicable = searched and not judgment.get("abstained") and total_claims > 0
        if faith_applicable:
            faithfulness = supported / total_claims

        # For the correctness×groundedness view: a "substantive" answer makes factual
        # claims (not a pure abstention); it is "fully grounded" only if it searched
        # and every claim is supported by the retrieved text. A correct answer that is
        # NOT fully grounded leaned on the model's memory rather than retrieval.
        is_substantive = (not judgment.get("abstained")) and total_claims > 0
        fully_grounded = is_substantive and searched and supported == total_claims

        fail_reasons = _failure_reasons(case, judgment, grounding_violation, faithfulness, contradicted)
        passed = len(fail_reasons) == 0

    return {
        "fail_reasons": fail_reasons,
        "searched": searched,
        "should_search": should_search,
        "should_abstain": should_abstain,
        "grounding_applicable": grounding_applicable,
        "grounding_violation": grounding_violation,
        "query_quality": judgment.get("query_quality") if judgment else None,
        "query_quality_score": qq_score,
        "claims_supported": supported,
        "claims_contradicted": contradicted,
        "claims_unsupported": total_claims - supported - contradicted,
        "claims_total": total_claims,
        "faith_applicable": faith_applicable,
        "faithfulness": faithfulness,
        "is_substantive": is_substantive,
        "fully_grounded": fully_grounded,
        "passed": passed,
    }


def _failure_reasons(case, j, grounding_violation, faithfulness, contradicted) -> List[str]:
    """Every gate this case trips (empty list = pass). A case can fail for >1 reason,
    so we collect all of them — that is what lets the report attribute the pass-rate
    drop to specific causes rather than just the first failing check."""
    cat = case["category"]
    reasons: List[str] = []
    if grounding_violation:
        reasons.append("grounding_violation")
    if bool(case.get("should_abstain")) and not j.get("abstained"):
        reasons.append("did_not_abstain")
    if not bool(case.get("should_abstain")) and j.get("abstained"):
        reasons.append("abstained_wrongly")  # abstained on an answerable question
    if cat == "ambiguous" and j.get("disambiguated") != "yes":
        reasons.append("not_disambiguated")
    if cat == "false_premise" and j.get("premise_handled") != "yes":
        reasons.append("premise_not_handled")
    corr = j.get("correctness")
    if corr != "correct":
        reasons.append("incorrect" if corr == "incorrect" else "partial_answer")
    if faithfulness is not None and contradicted > 0:
        reasons.append("contradicted_claim")
    if faithfulness is not None and faithfulness < _FAITH_PASS_THRESHOLD:
        reasons.append("low_faithfulness")
    return reasons


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
        judgment_obj = None if args.no_judge else judge_answer(case, result, client=client)
        judgment = judgment_obj.model_dump() if judgment_obj else None
        metrics = score_case(case, result, judgment)

        rows.append(
            {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "answer": result.answer,
                "n_searches": len(result.tool_calls),
                "queries": [tc.query for tc in result.tool_calls],
                "tokens": {"input": result.usage.input_tokens, "output": result.usage.output_tokens},
                "metrics": metrics,
                "judgment": judgment,
            }
        )

        _print_case_line(rows[-1])

    aggregates = _report(rows, args)
    _write_outputs(rows, aggregates, args)


def _print_case_line(row: Dict[str, Any]) -> None:
    m = row["metrics"]
    j = row["judgment"]
    verdict = "----" if m["passed"] is None else ("PASS" if m["passed"] else "FAIL")
    corr = (j or {}).get("correctness", "?")
    faith = f"{m['faithfulness']:.2f}" if m["faithfulness"] is not None else " n/a"
    flag_str = ("  ⚠ " + ", ".join(m["fail_reasons"])) if (m["passed"] is False and m["fail_reasons"]) else ""
    print(f"  [{row['id']:<12}] {verdict}  {corr:<9} faith={faith} qq={(m['query_quality'] or '-'):<14} n={row['n_searches']}{flag_str}")


def _report(rows: List[Dict[str, Any]], args) -> Dict[str, Any]:
    """Print the report and return aggregated metrics (single source of truth)."""
    n = len(rows)
    avg_in = sum(r["tokens"]["input"] for r in rows) / n
    avg_out = sum(r["tokens"]["output"] for r in rows) / n
    # Tool-call (search_wikipedia) efficiency: each search is one tool call.
    total_searches = sum(r["n_searches"] for r in rows)
    avg_searches = total_searches / n
    busiest = max(rows, key=lambda r: r["n_searches"])
    max_searches = busiest["n_searches"]

    # Grounding — denominator: cases that should have searched.
    g_rows = [r for r in rows if r["metrics"]["grounding_applicable"]]
    g_viol = sum(1 for r in g_rows if r["metrics"]["grounding_violation"])

    print("\n" + "=" * 72 + "\nGROUNDING & EFFICIENCY (from trace)\n" + "-" * 72)
    print(f"  grounding-violation rate   : {_pct(g_viol, len(g_rows))}  ({g_viol}/{len(g_rows)} should-search cases)  [lower better]")
    print(f"  tool calls / question      : avg {avg_searches:.2f}, max {max_searches} ({busiest['id']}), total {total_searches}")
    print(f"  avg tokens / question      : {avg_in:.0f} in / {avg_out:.0f} out")

    agg: Dict[str, Any] = {
        "n_cases": n,
        "grounding": {"violation_rate": _ratio(g_viol, len(g_rows)), "violations": g_viol, "n_should_search": len(g_rows)},
        "efficiency": {
            "avg_searches": round(avg_searches, 2),
            "max_searches": max_searches,
            "max_searches_case": busiest["id"],
            "total_searches": total_searches,
            "avg_input_tokens": round(avg_in),
            "avg_output_tokens": round(avg_out),
        },
        "judge": None,
    }

    judged = [r for r in rows if r["judgment"] is not None]
    if args.no_judge or not judged:
        if not args.no_judge:
            print("\n(no valid judgments)")
        return agg

    # pass_rate — denominator: all judged cases.
    n_pass = sum(1 for r in judged if r["metrics"]["passed"])
    # Failure attribution: which gate(s) each failing case tripped (a case can trip
    # several), tallied so the report explains *what* dropped the pass-rate.
    failures = [r for r in judged if r["metrics"]["passed"] is False]
    reason_tally: Dict[str, int] = defaultdict(int)
    for r in failures:
        for reason in r["metrics"]["fail_reasons"]:
            reason_tally[reason] += 1

    # query_quality — denominator: cases that searched (and produced a score).
    qq_scores = [r["metrics"]["query_quality_score"] for r in judged if r["metrics"]["searched"] and r["metrics"]["query_quality_score"] is not None]

    # faithfulness — denominator: cases that searched + answered substantively.
    faith_rows = [r for r in judged if r["metrics"]["faith_applicable"]]
    sup = sum(r["metrics"]["claims_supported"] for r in faith_rows)
    contra = sum(r["metrics"]["claims_contradicted"] for r in faith_rows)
    tot = sum(r["metrics"]["claims_total"] for r in faith_rows)
    unsup = tot - sup - contra
    faith_micro = _ratio(sup, tot)
    faith_macro = _mean([r["metrics"]["faithfulness"] for r in faith_rows])

    # correctness × groundedness — over substantive answers (those making factual
    # claims). Separates "right answer" from "right answer backed by retrieval", so
    # the correct-but-ungrounded quadrant (memory-leaning) is visible for debugging.
    sub = [r for r in judged if r["metrics"]["is_substantive"]]
    cg = [r for r in sub if r["judgment"]["correctness"] == "correct" and r["metrics"]["fully_grounded"]]
    cu = [r for r in sub if r["judgment"]["correctness"] == "correct" and not r["metrics"]["fully_grounded"]]
    ig = [r for r in sub if r["judgment"]["correctness"] != "correct" and r["metrics"]["fully_grounded"]]
    iu = [r for r in sub if r["judgment"]["correctness"] != "correct" and not r["metrics"]["fully_grounded"]]

    # supporting behaviour metrics
    corr_weighted = _mean([_CORRECT.get(r["judgment"]["correctness"], 0) for r in judged])
    abstain_ok = sum(1 for r in judged if bool(r["judgment"]["abstained"]) == r["metrics"]["should_abstain"])
    cite_rows = [r for r in judged if not r["judgment"]["abstained"]]
    cite_present = sum(1 for r in cite_rows if r["judgment"]["citation"] == "present")
    ambig = [r for r in judged if r["category"] == "ambiguous"]
    ambig_ok = sum(1 for r in ambig if r["judgment"]["disambiguated"] == "yes")
    fp = [r for r in judged if r["category"] == "false_premise"]
    fp_ok = sum(1 for r in fp if r["judgment"]["premise_handled"] == "yes")

    print("\n" + "=" * 72 + f"\nPASS RATE\n" + "-" * 72)
    print(f"  pass_rate                  : {_pct(n_pass, len(judged))}  ({n_pass}/{len(judged)})")
    if failures:
        print(f"  failed: {len(failures)} — attributed to:")
        for reason, cnt in sorted(reason_tally.items(), key=lambda kv: -kv[1]):
            print(f"      {reason:<22} {cnt}")
        print("  failing cases:")
        for r in failures:
            print(f"      {r['id']:<12} [{r['category']:<13}] {', '.join(r['metrics']['fail_reasons'])}")

    print("\n" + "=" * 72 + "\nQUERY QUALITY (cases that searched)\n" + "-" * 72)
    print(f"  mean query quality         : {_pct(sum(qq_scores), len(qq_scores))}  (n={len(qq_scores)})")

    print("\n" + "=" * 72 + "\nFAITHFULNESS (searched + substantive; claim-level)\n" + "-" * 72)
    print(f"  faithfulness (micro)       : {_pct(sup, tot)}  ({sup} supported / {tot} claims)")
    print(f"  faithfulness (macro)       : {_pct((faith_macro or 0), 1)}  (n={len(faith_rows)} answers)")
    print(f"  claim labels               : {sup} supported / {unsup} unsupported / {contra} contradicted")

    print("\n" + "=" * 72 + f"\nCORRECTNESS × GROUNDEDNESS ({len(sub)} substantive answers)\n" + "-" * 72)
    print(f"  {'':<14}{'fully-grounded':>16}{'not-grounded':>16}")
    print(f"  {'correct':<14}{len(cg):>16}{len(cu):>16}")
    print(f"  {'not correct':<14}{len(ig):>16}{len(iu):>16}")
    if cu:
        print("  ↳ correct but NOT grounded (right answer, leaned on memory/unsupported):")
        for r in cu:
            fa = r["metrics"]["faithfulness"]
            fstr = f"faith={fa:.2f}" if fa is not None else "no-search"
            print(f"      {r['id']:<12} [{r['category']:<13}] {fstr}")

    print("\n" + "=" * 72 + "\nCORRECTNESS & BEHAVIOUR\n" + "-" * 72)
    print(f"  correctness (weighted)     : {_pct((corr_weighted or 0), 1)}")
    print(f"  abstention accuracy        : {_pct(abstain_ok, len(judged))}  ({abstain_ok}/{len(judged)})")
    print(f"  disambiguation             : {_pct(ambig_ok, len(ambig))}  ({ambig_ok}/{len(ambig)})")
    print(f"  false-premise handled      : {_pct(fp_ok, len(fp))}  ({fp_ok}/{len(fp)})")
    print(f"  citation present           : {_pct(cite_present, len(cite_rows))}  ({cite_present}/{len(cite_rows)})")

    print("\n" + "=" * 72 + "\nPASS RATE BY CATEGORY\n" + "-" * 72)
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in judged:
        by_cat[r["category"]].append(r)
    cat_summary = {}
    for cat in sorted(by_cat):
        crows = by_cat[cat]
        cpass = sum(1 for r in crows if r["metrics"]["passed"])
        c_faith_rows = [r for r in crows if r["metrics"]["faith_applicable"]]
        c_sup = sum(r["metrics"]["claims_supported"] for r in c_faith_rows)
        c_tot = sum(r["metrics"]["claims_total"] for r in c_faith_rows)
        c_avg_searches = sum(r["n_searches"] for r in crows) / len(crows)
        creasons: Dict[str, int] = defaultdict(int)
        for r in crows:
            if r["metrics"]["passed"] is False:
                for reason in r["metrics"]["fail_reasons"]:
                    creasons[reason] += 1
        cat_summary[cat] = {
            "pass_rate": _ratio(cpass, len(crows)),
            "n": len(crows),
            "faithfulness_micro": _ratio(c_sup, c_tot),
            "avg_searches": round(c_avg_searches, 2),
            "fail_reasons": dict(creasons),
        }
        reason_str = ("   ← " + ", ".join(f"{k}×{v}" for k, v in sorted(creasons.items(), key=lambda kv: -kv[1]))) if creasons else ""
        print(f"  {cat:<16} pass {_pct(cpass, len(crows)):>4}  ({cpass}/{len(crows)})   faith {_pct(c_sup, c_tot):>4}   srch {c_avg_searches:.1f}{reason_str}")

    agg["judge"] = {
        "n_judged": len(judged),
        "pass_rate": _ratio(n_pass, len(judged)),
        "failures": {
            "by_reason": dict(reason_tally),
            "cases": [
                {"id": r["id"], "category": r["category"], "reasons": r["metrics"]["fail_reasons"]}
                for r in failures
            ],
        },
        "query_quality_mean": round(_mean(qq_scores) or 0, 3),
        "query_quality_n": len(qq_scores),
        "faithfulness": {
            "micro": faith_micro,
            "macro": round(faith_macro, 3) if faith_macro is not None else None,
            "n_answers": len(faith_rows),
            "claims": {"supported": sup, "unsupported": unsup, "contradicted": contra, "total": tot},
        },
        "correctness_weighted": round(corr_weighted, 3) if corr_weighted is not None else None,
        "correctness_x_groundedness": {
            "n_substantive": len(sub),
            "correct_grounded": len(cg),
            "correct_ungrounded": len(cu),
            "incorrect_grounded": len(ig),
            "incorrect_ungrounded": len(iu),
            "correct_ungrounded_cases": [
                {
                    "id": r["id"],
                    "category": r["category"],
                    "searched": r["metrics"]["searched"],
                    "faithfulness": r["metrics"]["faithfulness"],
                }
                for r in cu
            ],
        },
        "abstention_accuracy": {"pass": abstain_ok, "n": len(judged)},
        "disambiguation": {"pass": ambig_ok, "n": len(ambig)},
        "false_premise_handled": {"pass": fp_ok, "n": len(fp)},
        "citation_present": {"pass": cite_present, "n": len(cite_rows)},
        "by_category": cat_summary,
    }
    return agg


def _ratio(num: float, den: float) -> Optional[float]:
    return round(num / den, 3) if den else None


def _write_outputs(rows: List[Dict[str, Any]], aggregates: Dict[str, Any], args) -> None:
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    meta = {"timestamp": ts, "model": args.model, "n_cases": len(rows), "judge": not args.no_judge}
    (out_dir / f"run_{ts}.json").write_text(json.dumps({**meta, "rows": rows}, indent=2))
    (out_dir / f"report_{ts}.json").write_text(json.dumps({**meta, "report": aggregates}, indent=2))
    print(f"\nPer-case results : {out_dir / f'run_{ts}.json'}")
    print(f"Aggregated report: {out_dir / f'report_{ts}.json'}")


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
