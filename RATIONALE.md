# Design Rationale

> Companion to the ~5-min video. Sections marked _(to fill after eval runs)_ get
> populated once we have numbers and concrete failure cases to point at.

## Models used

- **Agent:** Claude Haiku 4.5 (`claude-haiku-4-5`). Chosen deliberately *because*
  it's small: a weaker model makes the quality of the system prompt visible —
  if behaviour is good, it's the prompt doing the work, not raw model strength.
  (Note: Haiku 4.5 doesn't support adaptive thinking / the `effort` param — those
  are 4.6+ — so the loop runs without thinking and the prompt carries everything.)
  The agent model is configurable (`--model`) so the same prompt can be tried on
  Sonnet 4.6 / Opus 4.8.
- **Judge:** Claude Opus 4.8 (`claude-opus-4-8`). Strongest grader, low volume; a
  different/stronger model than the agent reduces self-grading bias.

## Prompt engineering approach

The system answers from Wikipedia, so the prompt targets the failure modes of
retrieval-augmented answering rather than generic helpfulness. Each instruction
maps to a measurable behaviour:

1. **Search-first for facts** — don't answer specific/time-sensitive facts from
   parametric memory (which is stale and unverifiable), but *don't* search
   trivial reasoning like arithmetic. Tests the model's judgement about when
   retrieval adds value.
2. **Query well + decompose** — search entity/topic terms, not the user's
   sentence; break multi-hop questions into sequential searches.
3. **Ground the answer** — answer only from retrieved text; this is the core
   anti-hallucination lever.
4. **Cite** — name the article(s), for verifiability.
5. **Calibrated honesty** — abstain when Wikipedia lacks the answer; correct
   false premises; disambiguate ambiguous entities.
6. **Concision** — answer first, then support, then citation.

### Tool design

The assignment fixes the *signature* `search_wikipedia(query)`; the **return
payload** is the real design choice. We return the **top 3–4 articles, each with
title, URL, and the intro extract** (capped to control tokens). One round-trip
gives the model groundable content *and* surfaces disambiguation candidates,
without the token blow-up of fetching full articles.

We deliberately start with the **single tool the spec names** rather than a
`search` + `read_article` pair. If the evals show the intro extract is too
shallow for deep/multi-hop facts, adding a read tool becomes an eval-driven
iteration (see below) rather than speculative complexity.

We run a **manual agentic loop**, not the SDK tool-runner, so we capture the full
trace — needed to show whether search was used and to grade faithfulness against
the actually-retrieved text.

## Eval suite design — dimensions and why

A single accuracy number hides what matters in RAG. We measure six things:

| Dimension | Why it matters | How graded |
|---|---|---|
| **Correctness** | the headline — is the answer right? | LLM judge vs. gold key facts |
| **Faithfulness** | the *point* of grounding — supported by retrieved text, not parametric memory | LLM judge, given the retrieved snippets |
| **Citation** | verifiability / trust | LLM judge + programmatic |
| **Tool-use appropriateness** | searched when it should, skipped trivial searches | programmatic, from trace |
| **Abstention accuracy** | calibration on unanswerable questions | programmatic (judge flag vs. expected) |
| **False-premise handling** | does it push back instead of confabulating? | LLM judge |

**Grading is two-layered.** Programmatic checks read the trace (deterministic,
cheap, no judge needed): did a search happen, how many, did it abstain.
LLM-as-judge (Opus 4.8, strict JSON rubric) scores the open-ended dimensions,
and crucially is shown the *retrieved context* so faithfulness is graded against
what the model actually saw — an answer that's correct but unsupported by the
retrieval is flagged.

**Test taxonomy (~30 cases across 8 categories):** simple factual · multi-hop ·
disambiguation · temporal/"current X" · unanswerable (must abstain) ·
false-premise (must correct) · comparative · answerable-from-priors (does it
still ground? does it over-search trivia?). The negative categories (abstain,
false-premise) are where most systems quietly fail and where the eval earns its
keep.

## Where it succeeds / where it fails

_(to fill after eval runs — point at specific case IDs and judge rationales.)_

## Key iterations driven by eval results

_(to fill — e.g. prompt wording changes, whether a `read_article` tool was added
after observing intro-extract shallowness, judge-rubric fixes. Keep before/after
metrics.)_

## How I'd extend with more time

- Add `read_article(title, section)` for deep facts not in the intro, gated on
  observed failures.
- Multi-judge or judge-vs-human spot-checks to quantify judge reliability.
- Adversarial expansion of the false-premise and disambiguation sets.
- Query-reformulation analysis (which query phrasings retrieve the gold article).
- Latency/cost dashboard; prompt-cache the system+tools prefix (note: Haiku 4.5's
  4096-token cache minimum means our short prefix may not cache — worth measuring).

## Time spent

_(to fill — approximate hours.)_
