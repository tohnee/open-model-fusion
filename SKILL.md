---
name: open-fusion
description: Multi-model deliberation as a tool. Fan a single prompt out to a panel of models in parallel, have a judge model distill their outputs into a structured analysis (consensus / contradictions / partial coverage / unique insights / blind spots), then synthesize one grounded final answer. Use this skill whenever the user wants multi-model deliberation, a "panel of models", model fusion, consensus across LLMs, cross-checking an answer across several models, a second opinion, or asks to run/use "open fusion" or "fusion". Trigger on phrases like "fuse these models", "panel of models", "multi-model consensus", "get a second opinion from other models", "cross-check this across models", "MoA / mixture-of-agents", or any high-stakes/ambiguous question where being wrong is expensive (architecture decisions, research, multi-domain review). Prefer this over a single-model answer when the cost of being wrong is high; it is overkill for short tactical prompts.
compatibility: Python 3.10+. An OpenAI-compatible API key (OPENROUTER_API_KEY by default). Install with `pip install -e .` from this folder. Optional search keys (EXA_API_KEY / BRAVE_API_KEY) enable grounded panels.
---

# Open Fusion

An open, vendor-neutral implementation of the Fusion pattern: a single-layer
Mixture-of-Agents with a structured-JSON judge step. Fully implemented and
install-and-use. See `INSTALL.md` for setup and `ARCHITECTURE.md` for internals.

## What it does

Given one prompt, runs a three-phase pipeline:

1. **Fan-out** — dispatch the prompt to N panel models in parallel (each optionally
   with web/exec tools).
2. **Judge** — one judge model distills all panel responses into a typed `Analysis`
   (consensus, contradictions with per-model stances, partial coverage, unique
   insights, blind spots).
3. **Synthesis** — the judge writes the final answer grounded in that analysis,
   **with web tools turned off** (evidence is frozen).

The judge defaults to the calling model — a "second-opinion loop". Tools are OFF by
default so it works with just a model API key.

## How to run

After `pip install -e .` (see `INSTALL.md`):

```bash
open-fusion "Your question here"                       # default Quality panel
open-fusion "Your question here" --preset budget        # cheaper panel
open-fusion "Your question" --panel "a/x,b/y,c/z" --judge "a/x"   # custom
open-fusion "Your question" --tools --exclude-domains "example.com"  # grounded
open-fusion "Your question" --json                       # full result as JSON
```

The final answer goes to **stdout**; telemetry/analysis to **stderr** (so
`open-fusion "..." > answer.md` captures just the answer). `--show-analysis` prints
the structured deliberation.

Library use:
```python
import asyncio
from open_fusion import fuse, load_preset
r = asyncio.run(fuse("question", load_preset("quality")))
print(r.text); print(r.telemetry)
```

## When to use vs. not

- **Use**: deep research, multi-domain critique, architecture/selection decisions,
  anything where a wrong answer is expensive, or when the user asks for multi-model
  consensus / a second opinion.
- **Don't**: short tactical prompts, latency-sensitive chat. Fusion is ~2–3× slower
  and bills additively (every panel completion + judge + synthesis). Trigger
  selectively — that selectivity is the design, not a limitation.

## Cost & latency (so you set expectations)

- **Quality preset** ≈ 3×+ the cost of one frontier model — buys the quality
  ceiling, not savings.
- **Budget preset** ≈ 0.4–0.5× a single frontier model while approaching it.
- Latency ≈ `max(panel) + judge + synthesis`. `result.telemetry` reports
  completions, tokens, critical-path ms, and panel ok/failed counts.

## Non-negotiable design rules (correctness, not style)

- **Synthesis has no web tools** — enforced in `tools.toolset_for_phase(SYNTHESIS) == ()`.
- **Per-model attribution** — panel responses are labeled `[MODEL X]`; every
  `contradictions[].stances[].model` is populated (also the audit trail).
- **Graceful degradation** — all panels fail → `status:error`; judge fails → answer
  from raw responses (`status:judge_fallback`); ≥1 panel survives → proceed.
- **Bounded recursion** — an inner call cannot re-trigger fusion (`MAX_FUSION_DEPTH`).
- **Heterogeneous panels by default** — same-vendor panels give weaker lift (M2);
  the config emits a warning.

## Self-evaluation (is it actually helping?)

The skill ships a DRACO-style evaluation harness so you can prove whether Fusion beats
a single model on your tasks. It scores responses against weighted rubrics (four
categories — factual accuracy, breadth/depth, presentation, citation — with
negative-weight penalty criteria), grades each task 3× with a judge model, then reports
the **score lift over a solo baseline**, cost multiplier, quality-per-dollar, errors
triggered, and a **worth-it verdict**. See `EVAL.md`.

```bash
open-fusion-eval --demo                                  # zero-key demo report + verdict
open-fusion-eval --preset budget --baseline anthropic/claude-opus-4.8 \
    --grader llm --grader-model google/gemini-3.1-pro-preview --tools --md report.md
```

## Files

- `INSTALL.md` — setup + how to install into Claude Code / Codex.
- `ARCHITECTURE.md` — module map, contracts, data flow, state machine.
- `EVAL.md` — the DRACO-style evaluation standard, the three task suites (draco /
  longhorizon / semiconductor), how to run it, plugging in real DRACO.
- `DESIGN.html` — full design document: principle analysis, logical design,
  evaluation methodology (incl. pass^k long-horizon), and how eval gates skill use.
- `references/fusion-mechanism.md` — the principle-level mechanism this models.
- `tests/test_open_fusion.py` — 31 core offline tests (no network/key).
- `tests/test_eval.py` — 34 evaluation offline tests (no network/key).
- `tests/test_eval_suites.py` — 42 suite tests: 10 DRACO domains, pass^k long-horizon,
  semiconductor/DRAM, loaders, verdict overlays (no network/key).
- `src/open_fusion/` — the implementation (schema, prompts, config, cost, tools,
  client, panel, judge, synthesizer, orchestrator, cli, **eval/** with
  rubric · grader · harness · report · longhorizon · demo · cli · tasks/).
