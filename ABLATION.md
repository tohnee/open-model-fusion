# Ablation design — what is Fusion's lift actually made of?

The DRACO harness (`harness.evaluate`) answers *"Fusion vs one model, worth it?"* with
a single point lift and no uncertainty. That cannot settle feasibility (see REVIEW.md
H1–H3). This document specifies the experiments that can, and `eval/ablation.py`
implements them. Everything here is **runnable**: offline against canned answers
(`--ablation --demo`, for wiring) and online against real models (`--ablation`, for
evidence).

## 1. Claims under test

From `references/fusion-mechanism.md`, Fusion's value rests on four mechanisms:

| ID | Claim | The null hypothesis it must beat |
|---|---|---|
| M1 | Aggregation suppresses uncorrelated errors | Single answers are no worse |
| M2 | **Heterogeneity** is the lever; diverse panels decorrelate | Same model repeated does as well |
| M3 | Coverage — different models raise different sub-points | One model covers the same ground |
| M4 | **Structured adjudication** beats averaging | A free-form "summarize the answers" step does as well |

Plus the confound `EVAL.md` itself raises:

> A meaningful share of Fusion's lift comes from the **synthesis step itself** —
> running the same model twice and synthesizing already helps.

A credible feasibility verdict must show the lift survives **net of** that second pass.

## 2. The arms (the independent variable)

Two factors define the design, and the canonical arms are the cells that matter:

- **Panel composition:** `homogeneous` (best model ×2) vs `heterogeneous` (distinct models).
- **Aggregation:** `none` (solo), `freeform` (free-form synthesis, no typed judge),
  `structured` (the typed `Analysis` → synthesis).

| Arm | Panel | Aggregation | Adds, relative to the arm above it |
|---|---|---|---|
| `solo` | 1 model | none | — (the baseline) |
| `homo-freeform` | m0, m0 | free-form | a second pass (no diversity, no typed judge) |
| `homo-structured` | m0, m0 | structured | the typed judge (still no diversity) |
| `hetero-freeform` | m0…mk | free-form | panel diversity (no typed judge) |
| `hetero-structured` | m0…mk | structured | **the full product** |

Each arm reuses the real code path, so an arm is the pipeline, not a mock:

- `solo` → `panel._run_one` (one model, same agent loop & tools as a panel member).
- `freeform` → `panel.run` then `synthesizer.write_fallback` — the existing degraded
  path *is* exactly "no structured judge," so M4 needs no new code to ablate.
- `structured` → `orchestrator.fuse` with the arm's panel/judge.

**Controls held fixed across arms** (so a contrast isolates one change): the task set,
the grader and its `n_passes`, per-phase `Params`, `tools_enabled`, `max_in_flight`,
`fast_majority_k`, and the **merged `excluded_domains`** (task ∪ config) — the same
contamination guard the main harness uses, so no panel can fetch a rubric source.

## 3. The contrasts (the decomposition)

Each mechanism becomes **one paired contrast** — same tasks, subtract per task:

```
total_fusion_lift     = hetero-structured − solo              # is Fusion worth it at all?
synthesis_lift        = homo-freeform     − solo              # the confound: 2nd pass alone
diversity_lift        = hetero-structured − homo-structured   # M2: diverse vs repeated panel
structured_judge_lift = hetero-structured − hetero-freeform   # M4: typed Analysis vs free-form
```

Read together they attribute the total:
`total ≈ synthesis_lift + (judge contribution) + diversity_lift`, with the residual
telling you how much is interaction. The decisive feasibility questions map directly:

- **If `total_fusion_lift` is not significant** → Fusion is not worth its cost here. Stop.
- **If `synthesis_lift` ≈ `total_fusion_lift`** → you are paying a panel for what a single
  model's self-refinement gives free. Ship `homo-freeform`, not Fusion.
- **If `diversity_lift` is significant** → M2 holds; the heterogeneous panel earns its keep.
- **If `structured_judge_lift` is ~0** → M4 is decoration; `write_fallback` is enough and
  cheaper. If it's significant → the typed `Analysis` is load-bearing, as claimed.

## 4. The statistic (the missing rigor)

`paired_bootstrap(a, b, iters=2000, seed)` resamples the per-task differences `aᵢ−bᵢ`
with replacement and returns `(mean_lift, ci_low, ci_high, p_two_sided)`:

- **Paired** because every arm runs the *same* tasks — pairing removes task-difficulty
  variance, the dominant noise source at small N.
- **Percentile bootstrap** because LLM-judge scores are not normal and N is small
  (2–16 tasks); it assumes nothing about the distribution.
- A contrast is **`significant`** only when the 95% CI excludes 0. `verdict` is then
  `supported` (lift > 0), `contradicted` (lift < 0), or `null` (CI spans 0).
- Deterministic for a fixed `seed`, so reports are reproducible and diffable.

This is what lets the ablation say "no detectable effect at this sample size" instead of
reporting a noisy +1.5 as a win.

## 5. Running it

Offline wiring check (canned answers — **contrasts are not evidence**, only plumbing):

```bash
open-fusion-eval --ablation --demo --suite fusion
open-fusion-eval --ablation --demo --suite draco --json
```

Real evidence (needs `OPENROUTER_API_KEY`; panel doubles as the distinct-model pool):

```bash
open-fusion-eval --ablation --suite draco \
  --panel "openai/gpt-5.5,anthropic/claude-fable-5,google/gemini-3.1-pro-preview" \
  --judge anthropic/claude-opus-4.8 \
  --grader llm --grader-model anthropic/claude-opus-4.8 \
  --passes 3 --tools --md ablation.md
```

Library:

```python
import asyncio
from open_fusion.config import ModelSpec, FusionConfig
from open_fusion.client import ModelClient
from open_fusion.eval import load_tasks, run_ablation, render_ablation_markdown
from open_fusion.eval.grader import LLMJudgeGrader

models = [ModelSpec("openai/gpt-5.5"), ModelSpec("anthropic/claude-fable-5"),
          ModelSpec("google/gemini-3.1-pro-preview")]
judge  = ModelSpec("anthropic/claude-opus-4.8")
client = ModelClient(api_key="sk-or-...")
base   = FusionConfig(panel=models, judge=judge, tools_enabled=True)
report = asyncio.run(run_ablation(load_tasks("draco"), models, judge,
                                  LLMJudgeGrader(client, judge),
                                  client=client, base=base, n_passes=3))
print(render_ablation_markdown(report))
```

## 6. Protocol for a defensible real run

1. **Validate the slugs** against `openrouter.ai/models` first — the presets name
   speculative models (REVIEW.md §4). A 404 slug becomes an all-fail arm and a false
   regression.
2. **Pick a genuinely heterogeneous pool** (different labs/families) — M2 only pays off
   when errors decorrelate; `vendor` is a slug-prefix proxy, so eyeball it.
3. **Grade with `--grader llm`**, `--passes ≥ 3`, and report the **relative** contrasts,
   not absolute scores (judge choice moves absolutes 10–25 pts but not rankings).
4. **Exclude rubric-host domains** with `--tools` on, for every arm equally.
5. **Use ≥ 30 tasks** if you want the CIs to be tight enough to resolve a ~2-pt effect;
   the shipped suites (2–16 tasks) are for wiring and qualitative reads, not a verdict.
6. **Report all four contrasts** even when null — a null `structured_judge_lift` or a
   dominant `synthesis_lift` is a *finding*, and the honest one Fusion's own docs invite.

## 7. Beyond the canonical five (future arms)

The harness generalizes cleanly; natural extensions, in rough value order:

- **Panel-size sweep** (N = 1,2,3,5): marginal value of each added panelist — diminishing
  returns are the expected shape, and where to cap N for cost.
- **Judge-strength sweep**: strong vs weak judge at a fixed panel — how much of M4 is the
  *schema* vs the *judge model*.
- **Break the freeze**: synthesis with tools ON — does fetching new evidence at write
  time help or break the audit trail? Tests invariant #1 empirically.
- **Majority/self-consistency vote** vs synthesis — does adjudication beat plain voting
  (a stricter M4 null than free-form synthesis).
- **Error-decorrelation probe (M1 directly)**: per-task, correlate panel members'
  per-criterion error vectors; lift should track *low* inter-model error correlation.
  This measures the mechanism, not just the outcome.
