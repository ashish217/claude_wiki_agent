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
4. **Cite** — name the article(s) used, in one fixed format
   (`**Sources:** [Title](URL), …`, using the titles/URLs the tool returned), so
   citations are consistent and clickable/verifiable rather than free-form.
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

## Eval suite design — metrics and why

A single accuracy number hides what matters in RAG, so the harness reports a
**headline `pass_rate`** and then decomposes failure into per-component metrics.
The key design point is that **each metric has a different denominator** — it is
scored only over the cases where it is meaningful, so one component's failures
don't distort another's.

| Metric | What it isolates | Numerator / **Denominator** | Source |
|---|---|---|---|
| **pass_rate** | did the agent do *everything* right for this case | passing cases / **all judged cases** | composite |
| **grounding_violation_rate** | did it search when it should have | non-searching cases / **cases where `should_search=true`** | trace |
| **query_quality** | did its queries retrieve the needed content | mean(good=1/adequate=.5/poor=0) / **cases that searched** | judge |
| **faithfulness** | did it use the retrieved content (vs. priors) | **Σ supported claims / Σ total claims** over **cases that searched + answered substantively** | judge, claim-level |

Supporting metrics (correctness, abstention accuracy, disambiguation,
false-premise handling, citation) round out behaviour.

**Faithfulness is claim-level.** Rather than ask the judge for one fuzzy
faithfulness score, it decomposes the answer into atomic claims and labels each
`supported` / `unsupported` / `contradicted` against the *retrieved text*;
faithfulness is then `supported / total`. This is the only reliable way to catch
an answer that is correct but smuggles in unretrieved facts — and it surfaces
*contradicted* claims (active hallucination) separately from merely unsupported
ones.

**A case "passes"** only if it does everything right *for its category*:
grounded when it should be, correct, with no contradicted claims and per-sample
faithfulness ≥ 0.7 (the bar is 0.7, not 0.8, to tolerate one unsupported claim in
a short answer and absorb judge claim-decomposition noise — contradicted claims
remain a hard fail), plus the category-specific behaviour (disambiguated /
premise-corrected / abstained). pass_rate is therefore a strict conjunction — a
deliberately demanding headline. When it dips, the report **attributes** every
failure to the specific gate(s) it tripped (`grounding_violation`,
`not_disambiguated`, `low_faithfulness`, `contradicted_claim`, …), tallied overall
and per category — so a pass-rate drop is immediately diagnosable rather than
opaque.

**Correctness vs. groundedness are measured independently**, and the report
cross-tabulates them (a 2×2 over substantive answers) to separate "right answer"
from "right answer *because of* retrieval". This surfaces the
**correct-but-ungrounded** quadrant — answers that are correct but lean on the
model's memory rather than the retrieved text. On the current suite that quadrant
holds all the residual imperfection: 0 incorrect and 0 contradicted, with ~5/24
answers correct-but-not-fully-grounded (a true peripheral aside not in the intro
extract) — isolating the weakness as groundedness of incidental detail (an
argument for deeper `read_article` retrieval), not correctness.

**Grading is two-layered.** Programmatic checks read the trace (deterministic,
cheap): did it search, how many times, tokens. The Opus 4.8 judge (Pydantic
structured output) handles the open-ended dimensions and is always shown the
*queries* and the *retrieved context*, so faithfulness and query-quality are
graded against what the agent actually did and saw.

**Test taxonomy (~28 cases across 7 categories), because category dictates the
correct behaviour:** single-hop · multi-hop · aggregation (compare/aggregate over
several entities) · temporal ("current X" — must not trust stale priors) ·
ambiguous (must disambiguate) · unanswerable (must abstain) · false-premise (must
correct). The negative categories (unanswerable, false-premise) and the
behavioural ones (ambiguous) are where most systems quietly fail and where the
eval earns its keep.

## Where it succeeds / where it fails

Current baseline (28 cases, Haiku 4.5 agent / Opus 4.8 judge): **pass_rate 89%
(25/28)** under the strict composite.

**Succeeds.** Correctness 100%, abstention 100% (4/4 unanswerable), false-premise
handling 100% (4/4), query_quality 100% (n=24), and **faithfulness 98% (129/132
claims) with zero contradicted claims** — no active hallucination anywhere.
multi_hop, temporal, and aggregation all pass 100%; multi-hop decomposition works
(e.g. 2016 Olympics → Brazil → Brasília across two searches).

**The three failures, each a different component:**
- `single-03` (Fe→iron) — **grounding violation**: answered from memory, no search
  (the lone 4% grounding violation). We deliberately don't add a case-specific
  prompt example to force it — that's overfitting the prompt to the eval. A
  small-model adherence limit, not a prompt gap.
- `ambig-04` (Michael Jordan) — answered the athlete directly without noting other
  notable namesakes → `disambiguated=no`. Borderline: arguably a fair test (the
  name *is* ambiguous) or arguably fine (the athlete dominates). Flagged as an
  eval-definition judgment, not silently "fixed."
- `fp-04` (Beatles drummer) — correctly corrected the premise (Pete Best) but added
  one ungrounded clarifying claim → faithfulness 0.75, below the 0.8 pass bar.

**The finding the richer harness exposed:** faithfulness is 98%, not the 100% the
old categorical metric reported. The 3 unsupported claims (`agg-04` Earth's
diameter, `fp-03` "1776 = the Declaration", `fp-04` "Lennon was rhythm guitarist")
are all peripheral facts the model volunteered *from memory* — true, but not in
the retrieved text. The system is honest (no fabricated falsehoods) but
occasionally garnishes with ungrounded-but-true detail. Claim-level faithfulness
makes this visible; a single categorical score hid it entirely.

## Key iterations driven by eval results

**Iteration 1 — two prompt edits in `prompts.py`** (measured on the suite):
- *Search-first boundary.* Reworded rule #1 from "search facts that may have
  changed" to "always search any external-world fact, even if confident; only skip
  pure arithmetic/logic." Removed the wiggle room the model used to answer stable
  facts from memory → fixed capital-of-Australia, lifted citation and tool-use
  (correctness 97% → 100%).
- *Disambiguation.* The first attempt (tell it to list alternatives) failed because
  the model resolves ambiguity at **query** time. The fix that worked: search the
  **bare term first** (don't pre-narrow), so results reveal the meanings. This was
  only diagnosable from the trace, and lifted disambiguation 75% → ~100% on
  Mercury/Java/Cambridge.

**Iteration in the eval itself — harness redesign.** Moved from six flat
categorical dimensions to a headline **pass_rate** plus component metrics with
**distinct denominators** (grounding-violation over should-search cases,
query-quality over searched cases, claim-level faithfulness over searched+
substantive cases). Re-baselining under it produced the more honest 89% pass_rate
and the 98% claim-level faithfulness above.

**Iteration 2 — a faithfulness nudge that we measured, then *reverted*.** We added
a rule #3 clause: "don't add tangential context from memory, even if true,"
targeting the three unsupported claims. The eval verdict: **no measurable benefit**
(faithfulness 98% → 94%, pass 89% → 86%) and **+32% input tokens** (one case went
3 → 8 searches as the model tried to ground every aside). Crucially, the
"regression" was **noise**: the set of unsupported claims was almost entirely
different between the before/after runs. The honest read — *the eval couldn't yet
resolve a change this small* — so we reverted the nudge and fixed the eval instead.
This is the headline judgment call of the project: don't ship changes you can't
measure; harden the measurement first.

**Iteration 3 — eval reproducibility (`temperature=0`).** Pinned the agent at
`temperature=0` (a factual QA system wants determinism anyway). Found along the
way that **Opus 4.8 — our judge — rejects `temperature`** ("deprecated for this
model"), so the judge can't be pinned the same way. Result, measured by running
the suite twice:
- The **agent is now stable** — identical search behaviour in 27/28 cases,
  grounding-violation a stable 0%, citation stable 100%.
- The **residual variance is judge-side**: the unpinnable judge re-decomposes
  claims differently in 9/28 cases (e.g. an ambiguous answer scored as 15 vs 12
  claims), which, against the 0.8 faithfulness pass-bar, flips pass on borderline
  cases. pass_rate still swings ≈86–89%, faithfulness ≈94–96%.

**The lesson:** agent determinism is necessary but not sufficient for a stable
LLM-judge eval; with a granular claim-level metric the *judge* becomes the
dominant noise source. The real fix is **multi-trial averaging** (report each
metric as a mean ± range over N runs), which is the top eval extension below.

**Iteration 4 — temporal grounding (inject today's date).** A spot-check ("Who won
the recent NBA championship?") returned the *2024* Finals: with no notion of "now",
the model anchors "recent/latest" to its training cutoff. Captured the failure
first as eval cases (`temporal-04` NBA, `temporal-05` FIFA World Cup), then fixed
it by prepending today's date to the system prompt with an instruction to resolve
relative time and search the specific year (and the *judge* is given the date too,
so these cases can be graded as-of-today against the retrieved text). After the
fix the agent searches "2026 NBA Finals" rather than "recent", and the temporal
category passes 5/5. `today` is a parameter (defaults to the real date), so it's
overridable for reproducibility.

## How I'd extend with more time

- **Multi-trial averaging (top priority).** Run each case N times and report every
  metric as a mean ± range, so judge-side claim-decomposition noise (the dominant
  residual variance) is averaged out and small prompt changes become measurable.
- Add `read_article(title, section)` for deep facts not in the intro, gated on
  observed failures (would also fix faithfulness false-positives like the correct
  "Darwin born in England" claim that the shallow intro couldn't support).
- Multi-judge panels or judge-vs-human spot-checks to quantify judge reliability.
- Adversarial expansion of the false-premise and disambiguation sets.
- Query-reformulation analysis (which query phrasings retrieve the gold article).
- Latency/cost dashboard; prompt-cache the system+tools prefix (note: Haiku 4.5's
  4096-token cache minimum means our short prefix may not cache — worth measuring).

## Time spent

~3–4 hours of focused work: prototype + live MediaWiki retrieval, the agent loop,
the eval harness, and two rounds of eval-driven iteration (prompt fixes, then the
metric-framework redesign). _(Adjust to your actual time before submitting.)_
