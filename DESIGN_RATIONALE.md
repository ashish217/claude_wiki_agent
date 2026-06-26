# Design Rationale — Wikipedia-grounded Q&A with Claude

A system that answers questions by searching English Wikipedia through tools, and
an eval suite that measures how well it works and *why* it fails.

- **Agent model:** Claude Haiku 4.5 (`claude-haiku-4-5`), `temperature=0`.
- **Judge model (evals):** Claude Opus 4.8 (`claude-opus-4-8`).
- **Retrieval:** live MediaWiki API — no API key, always current, and fully within
  the "no hosted search / RAG tools" constraint (we own the retrieval loop).
- I deliberately ran the agent on a **small, fast model** so that answer quality
  reflects the *system prompt and retrieval design*, not raw model horsepower.

> A companion `RATIONALE.md` is the chronological working log (every iteration,
> with raw numbers). This document is the organized writeup.

---

## 1. Prompt Engineering Approach

**Scope and goal.** The agent's single job is to produce **grounded, verifiable**
answers from Wikipedia. That goal is encoded as the system prompt; the rest of the
design follows from it.

**The key simplification — bias hard toward retrieval.** Rather than asking the
model to make a delicate call on every query about *when to trust its own
knowledge vs. when to use Wikipedia* (a hard judgment it gets wrong in subtle,
unverifiable ways), I collapsed that decision: **for essentially any fact about
the world, search Wikipedia and answer from what it returns.** The rationale:

- Wikipedia is almost certainly in the model's pretraining corpus, so retrieval
  rarely *contradicts* the model's priors — it mostly adds **verifiability,
  recency, and a citation**. The downside of "over-searching" a fact the model
  already knows is small; the downside of answering an unverifiable fact from
  stale memory is large.
- It makes behavior **consistent and citable**, and it's the honest framing of a
  retrieval-augmented system: the answer should come *from the source*.
- The only carve-out is content with **no external referent** — pure arithmetic /
  logic — which needs no lookup.

**Each instruction maps to a measurable behavior** (and to an eval dimension):

| Instruction | Behavior it drives | Why |
|---|---|---|
| **Search-first for external facts** | search before answering anything factual/time-sensitive; skip only pure computation | parametric memory is stale & unverifiable; tests judgment about when retrieval adds value |
| **Query well + decompose** | concise entity/topic queries; break multi-hop into sequential searches; for ambiguous terms search the *bare* term first | good queries are what make retrieval succeed |
| **Read deeper before abstaining** | if the intro extracts lack the fact, call `read_article` on the best result | the intro is shallow; the fact is often in the body |
| **Ground the answer** | use only retrieved text; don't add facts from memory | the core anti-hallucination lever |
| **Cite (one fixed format)** | `**Sources:** [Title](URL), …` using the tool's own titles/URLs | consistent + clickable/verifiable, not free-form |
| **Calibrated honesty** | abstain when Wikipedia lacks it; correct false premises; disambiguate | the behaviors that separate a *trustworthy* RAG system from a plausible one |
| **Concision** | answer first, then brief support, then citation | usability |

**Temporal grounding.** The model has no notion of "now," so relative time
("recent", "latest") silently anchors to its training cutoff — a spot-check
("recent NBA champion") returned a stale year. The fix: **inject today's date into
the system prompt** with an instruction to resolve relative time and search the
specific year. The judge is given the date too, so these cases grade as-of-today.

**Implementation note.** The agent runs a **manual tool loop** (not the SDK
auto-runner), specifically so we capture the full trace — which tools were called,
with what queries, and what came back. That trace is what makes "was search used"
visible and what lets the eval grade *faithfulness against the actually-retrieved
text*.

---

## 2. Tool Design

The assignment fixes the **signature** `search_wikipedia(query)`; the **return
payload is the real design choice.** `search_wikipedia` returns the **top 3–4
matching articles**, each with **title, URL, and the plain-text intro extract**
(capped to control tokens). One round-trip gives the model groundable content
*and* surfaces disambiguation candidates (e.g. "Mercury" → planet / element /
person), without the token blow-up of fetching whole articles.

I **deliberately started with the single tool the spec names**, planning to add
more only if the eval proved it necessary. It did: the benchmark results showed a
clear **retrieval-depth ceiling** (facts buried below the intro), so I added a
second tool — **`read_article(title)`**, which returns the full article prose —
and a prompt step to read deeper before abstaining. The agent decides when to
escalate from search → read based on the query and what the search returned. (See
§5, Iteration 5, for the measured payoff.)

---

## 3. Eval Suite Design — Dimensions and Why

A single accuracy number hides what matters in a RAG system, so the harness
reports a **headline `pass_rate`** and then **decomposes failure into component
metrics, each with its own denominator** so one component's failures don't distort
another's. This denominator discipline is the core of the design.

| Metric | What it isolates | Numerator / **Denominator** | Source |
|---|---|---|---|
| **pass_rate** | did the agent do *everything* right for this case | passing / **all judged cases** | composite |
| **grounding_violation_rate** | did it search when it should have | non-searching / **`should_search=true` cases** | trace |
| **query_quality** | did its queries retrieve what's needed | mean(good=1/adequate=.5/poor=0) / **cases that searched** | judge |
| **faithfulness** | did it *use* the retrieved content (vs. memory) | **Σ supported claims / Σ total claims**, over **searched + substantive** cases | judge (claim-level) |

Supporting: correctness (partial-credit), abstention accuracy, disambiguation,
false-premise handling, citation, and efficiency (tool calls, tokens).

**Faithfulness is claim-level.** Instead of one fuzzy faithfulness score, the
judge **decomposes the answer into atomic claims and labels each
`supported` / `unsupported` / `contradicted`** against the retrieved text;
faithfulness = `supported / total`. This is the only reliable way to catch an
answer that is *correct but ungrounded* (smuggling unretrieved facts), and it
separates **contradicted** claims (active hallucination — a hard fail) from merely
**unsupported** ones.

**Correctness × groundedness cross-tab.** Correctness (right vs. gold) and
groundedness (backed by retrieval) are scored *independently* and cross-tabulated,
which surfaces the diagnostic quadrant — **correct-but-ungrounded** answers that
lean on memory rather than the source.

**Failure attribution.** Every failing case records **all** the gates it trips
(`grounding_violation`, `not_disambiguated`, `low_faithfulness`,
`contradicted_claim`, `abstained_wrongly`, …), tallied overall and per category,
so a pass-rate drop is immediately traceable to a cause rather than opaque.

**What "pass" means.** A strict conjunction: grounded when it should be, correct,
behaviorally right for its category (disambiguated / premise-corrected /
abstained), with **no contradicted claims and faithfulness ≥ 0.7** (the 0.7 bar
tolerates one stray unsupported claim on short answers and absorbs judge noise;
contradicted is still fatal).

**Two-layer grading.** Programmatic checks read the trace (deterministic, cheap:
did it search, how many tool calls, tokens). The **Opus 4.8 judge** (Pydantic
structured output) handles the open-ended dimensions and is always shown the
*issued queries, the retrieved context, and today's date*, so faithfulness and
temporal correctness are graded against what the agent actually did and saw.

**Taxonomy — 82 cases, because category dictates correct behavior.** Seven
behavior-driven categories (single-hop, multi-hop, aggregation, temporal,
ambiguous, unanswerable, false-premise) plus **four public benchmarks** that form
a deliberate **difficulty ladder**:

| Benchmark | Maps to | Stresses |
|---|---|---|
| NQ-Open | single_hop | popular facts in the article *lead* |
| HotpotQA (hard) | multi_hop / aggregation | 2-hop bridge & comparison |
| MuSiQue | multi_hop | genuine 2–4-hop chains + indirect-entity resolution |
| SimpleQA | single_hop | obscure facts in article *bodies* |

**Reproducibility.** The agent is pinned at `temperature=0`. The Opus judge can't
be pinned (it rejects the `temperature` param), so residual variance is judge-side
claim re-decomposition; the standing fix is multi-trial averaging (§7).

---

## 4. Where It Succeeds / Where It Fails

The failure-attribution layer is what turns metrics into a fix-list. Full 82-case
suite, **before vs. after adding `read_article`** (same judge, `temperature=0`):

| Group | search-only | + read_article |
|---|---|---|
| **OVERALL** | **68% (56/82)** | **77% (63/82)** |
| NQ-Open (10) | 100% | 100% |
| HotpotQA (20) | 80% | 85% |
| MuSiQue (10) | 30% | 40% |
| SimpleQA (10) | 20% | **60%** |
| Sythetic (32) | 78% | 81% |
| single_hop / multi_hop | 58% / 52% | 75% / 60% |
| temporal / aggregation | 80% / 93% | 100% / 93% |
| unanswerable / ambiguous / false_premise | 100% / 75% / 80% | (same) |

**Where it succeeds.** Faithfulness ~96% with **near-zero contradicted claims** —
the system stays calibrated and **does not hallucinate**; it abstains rather than
fabricates. Abstention on unanswerable questions is 100%, false-premise correction
and disambiguation mostly work, temporal grounding hits 100% after the date fix,
and aggregation/comparison is strong (93%).

**Where it fails (19 remaining, by tier):**

- **Quick prompt-fixable (~4).** Grounding violations on *canonical* facts
  (Canberra, Fe→iron answered from memory); a dominant-name disambiguation miss
  (Michael Jordan); a false-premise-framed-as-future question (the Knicks "title
  drought") where the agent abstained as "a future event" *without searching* to
  notice it already happened.
- **Medium (~4).** Partial answers that stop short of the *specific* entity or the
  *final* hop (gave the appointing *government* not the *President*; stopped at the
  parents not the grandmother).
- **Structural — the real ceiling (~8).** (a) **Table/infobox-bound facts**
  (award-winner lists, breed ancestry) live in tables that the prose `extracts`
  strips, so even `read_article` can't see them; (b) **MuSiQue multi-hop
  reasoning** — entity-chain resolution the agent can't decompose, occasionally
  over-reading into a contradicted answer; (c) one **noisy benchmark gold**.

**The single most useful learning, made visible by the cross-tab:** failures are
**groundedness / calibration / retrieval-depth** problems — *never*
"grounded-but-wrong-reasoning." The agent reasons and stays honest; the ceiling is
**getting the right text in front of it.**

---

## 5. Key Iterations Driven by Eval Results

The eval was redesigned once mid-stream (from flat categorical dimensions to the
`pass_rate` + component-metric + attribution framework above), which made every
later iteration measurable. The system changes:

1. **Search-first + bare-term disambiguation (prompt).** Reworded the search rule
   to "always search external facts, even if confident"; for ambiguous terms,
   search the **bare term first** so results reveal the meanings (the lever was
   *query formation*, only visible from the trace). Lifted disambiguation 75%→100%
   and removed stale-fact answers on the early set.
2. **Reproducibility (`temperature=0`).** Pinned the agent (a factual QA system
   wants determinism). Discovered the Opus judge rejects `temperature`, so the
   residual noise is judge-side claim decomposition → multi-trial averaging is the
   real fix (§7).
3. **Temporal grounding.** Captured the "recent NBA champion → stale year" failure
   as eval cases, then injected today's date into the prompt and the judge.
   Temporal category → 100%.
4. **`read_article` (deeper retrieval) — the eval-driven payoff.** The benchmark
   ladder *proved* retrieval depth was the bottleneck (NQ-Open ~100% vs SimpleQA
   ~10%, same category), so I added the read tool. **68% → 77% overall pass-rate; SimpleQA
   20% → 60%**, flat on reasoning-bound MuSiQue, at +26% input tokens.

---

## 6. Learnings

- **A small eval set is enough to start iterating** — and pays for itself
  immediately. With ~30 cases I was already finding and fixing real behaviors;
  scaling to 82 (with public benchmarks) sharpened the diagnosis.
- **The model stays grounded and avoids hallucination.** Near-zero contradicted
  claims throughout. The dominant problem is **not** the model inventing
  falsehoods — it's **the system's ability to get sufficient context from
  Wikipedia** for detailed questions. When the needed fact wasn't retrievable, the
  model occasionally filled *peripheral* details from memory (ungrounded-but-true)
  to give a fuller answer — and **claim-level faithfulness is exactly what made
  that visible**, where a single correctness score would have hidden it.
- **Build the eval to explain itself.** The biggest leverage came from
  instrumentation — distinct denominators, claim-level faithfulness, the
  correctness×groundedness cross-tab, and failure attribution — which turned "the
  score dropped" into "these N cases failed for *this* reason," and pointed
  straight at the one high-value fix (`read_article`).
- **Public benchmarks as difficulty rungs.** Mapping NQ-Open / HotpotQA / MuSiQue /
  SimpleQA into our categories created a clean difficulty ladder that *isolated the
  variable*: same category (`single_hop`), opposite outcome (100% vs 10%) → the
  difference is retrieval depth, not reasoning.

---

## 7. How I'd Extend With More Time

- **Deepen `read_article`** (highest leverage now): parse **tables/infoboxes**
  (where many SimpleQA facts live) and support **section-targeted** fetches — this
  attacks the remaining structural failures directly.
- **Multi-trial averaging** in the harness: run each case N times, report each
  metric as mean ± range, so judge-side decomposition noise is averaged out and
  small changes become measurable.
- **Eval breadth:** more adversarial false-premise / ambiguous cases and larger
  benchmark samples.
- **Stronger query-formation & retrieval-quality metrics.** These two dimensions
  are currently a single coarse LLM-judge rating (good/adequate/poor), which is
  subjective and conflates "was the query well-formed" with "was the needed
  content actually retrieved." With more time I'd dig into the gaps and add
  rigorous measures — e.g. retrieval recall against the **gold supporting passages**
  the benchmarks ship (HotpotQA/MuSiQue annotate supporting facts), a query→gold
  -article hit rate, and a clean separation of query quality from retrieval success.
- **Contextualizing baselines.** Run the same suite with (a) a **larger, more
  capable agent model** and (b) a **no-RAG / closed-book** baseline. The
  closed-book run measures how much the Wikipedia grounding actually adds and
  flags cases the model can already answer from memory (where the eval isn't truly
  testing retrieval); the stronger-model run shows how much of the ceiling is the
  small agent vs. the retrieval design — and both help validate the eval set's own
  difficulty and discrimination.
- **Broaden the problem.** Natural extensions beyond single-turn English-text QA:
  **multi-turn / conversational QA** (follow-ups, context carry-over, clarifying
  questions), **internationalization** (non-English Wikipedias and cross-lingual
  retrieval), and **multi-modality** (images, tables, infoboxes — which also
  subsumes the `read_article` table-parsing gap).

---

## Appendix

- **Constraint compliance:** Anthropic models only (Haiku 4.5 agent, Opus 4.8
  judge); **no** hosted search/RAG tools — retrieval is our own MediaWiki loop.
- **Run it:** `uv run python -m wiki_agent --demo` (prototype) and
  `uv run python -m wiki_agent.evals.run_evals` (eval suite). See `README.md`.
- **Time spent:** ~6–8 hours including the benchmark integrations and iteration
  loop.
