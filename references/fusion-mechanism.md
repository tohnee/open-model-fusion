# The Fusion mechanism

This note explains *why* fusion helps, in principle, so the design choices in the
code read as deliberate rather than arbitrary. It is descriptive, not a benchmark
report.

## The pattern

Fusion is a single-layer Mixture-of-Agents (MoA): one prompt is answered
independently by a panel of models; a judge distills those answers into a structured
analysis; a synthesizer writes one final answer grounded in that analysis. There is
exactly one panel layer (no deep MoA stacks) and the judge step is forced through a
typed JSON schema rather than free-form "pick the best".

## Why it can beat a single model

- **M1 — error decorrelation.** Independent answers make uncorrelated mistakes.
  Aggregation suppresses idiosyncratic errors while reinforcing claims that survive
  across models. The structured judge step is what lets aggregation be *selective*
  (keep well-supported consensus, surface real contradictions) instead of averaging.
- **M2 — heterogeneity is the lever.** Decorrelation only pays off when the panel is
  diverse. Same-vendor / same-family panels share training data and failure modes, so
  their errors correlate and the lift shrinks. Hence heterogeneous panels are the
  default and `FusionConfig.validate()` warns on homogeneous ones.
- **M3 — coverage.** Different models raise different sub-points. `partial_coverage`
  and `unique_insights` capture breadth a single model would miss; `blind_spots`
  records what *no* model covered, which bounds the answer honestly.
- **M4 — adjudication over averaging.** The value is not "more text" but resolving
  disagreement explicitly. The judge names per-model stances; the synthesizer is told
  to pick the better-supported view rather than hedge.

## Why the structured judge step is the core lever

A free-form "summarize the answers" step launders away exactly the information that
makes fusion worth its cost: which model said what, where they actually disagree, and
what nobody addressed. Forcing a typed `Analysis` (consensus / contradictions with
per-model stances / partial coverage / unique insights / blind spots) keeps that
information legible and auditable, and makes the final synthesis a reasoning step over
structured evidence rather than a second guess.

## Why synthesis freezes the evidence

If the synthesizer could fetch new evidence, the final answer could drift away from
the analysis the judge actually produced, breaking the audit trail and reintroducing
single-pass error. Tools are therefore OFF in synthesis
(`toolset_for_phase(SYNTHESIS) == ()`): the judge is the last point new evidence
enters.

## Cost / latency shape

Fusion bills additively: every panel completion + judge + synthesis. Latency is
roughly `max(panel) + judge + synthesis` because fan-out is parallel. This is why the
skill is meant to be invoked *selectively* — for high-stakes or ambiguous questions
where being wrong is expensive — not for tactical chat.

## §5 — Benchmarking hygiene

When measuring fusion against a baseline, always exclude the answer/rubric source
domains from grounded panels (`--exclude-domains`). Otherwise a panel model can fetch
the reference answer and the comparison measures retrieval, not deliberation. Keep the
panel, judge, and (frozen) synthesis phases on the same excluded-domain set.
