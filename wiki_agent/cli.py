"""Command-line entry point.

    python -m wiki_agent "Who wrote Pride and Prejudice?"
    python -m wiki_agent --demo
    python -m wiki_agent --model claude-sonnet-4-6 "..."
"""

from __future__ import annotations

import argparse
import sys

from .agent import DEFAULT_MODEL, answer_question

DEMO_QUESTIONS = [
    "What is the capital of Australia?",  # simple factual
    "What is the capital of the country that hosted the 2016 Summer Olympics?",  # multi-hop
    "Who is the current Secretary-General of the United Nations?",  # temporal
    "When did Albert Einstein win the Nobel Prize for his theory of relativity?",  # false premise
    "What did I have for breakfast this morning?",  # unanswerable -> abstain
]


def _print_run(question: str, model: str) -> None:
    print(f"\n{'=' * 78}\nQ: {question}\n{'-' * 78}")
    result = answer_question(question, model=model)

    if result.searched:
        print(f"[search used: {len(result.tool_calls)} call(s)]")
        for i, tc in enumerate(result.tool_calls, 1):
            titles = ", ".join(tc.result_titles) or "(no results)"
            print(f'  {i}. query="{tc.query}" -> {titles}')
    else:
        print("[no search performed — answered directly]")

    print(f"\nA: {result.answer}\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Answer questions using Claude + Wikipedia.")
    parser.add_argument("question", nargs="*", help="The question to answer.")
    parser.add_argument("--demo", action="store_true", help="Run a set of sample questions.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Agent model (default: {DEFAULT_MODEL}).")
    args = parser.parse_args(argv)

    if args.demo:
        for q in DEMO_QUESTIONS:
            _print_run(q, args.model)
        return 0

    if not args.question:
        parser.error("provide a question, or use --demo")

    _print_run(" ".join(args.question), args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
