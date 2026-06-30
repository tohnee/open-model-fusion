# open-fusion

An open, vendor-neutral implementation of OpenRouter's **Fusion** pattern: fan one
prompt out to a panel of models in parallel → a judge distills their outputs into a
structured `Analysis` (consensus / contradictions / partial coverage / unique
insights / blind spots) → synthesize one grounded final answer with web tools off.

**Status: implemented & install-and-use.** Zero required dependencies (stdlib
only). core and eval offline tests pass with no network.

## Quickstart

```bash
cd open-fusion-skill
pip install -e .
export OPENROUTER_API_KEY="sk-or-..."          # or OPEN_FUSION_API_KEY + OPEN_FUSION_BASE_URL

open-fusion "What are the strongest arguments for and against a carbon tax?"
open-fusion "Compare ridge, lasso, elastic-net." --preset budget --show-analysis
```

Library:
```python
import asyncio
from open_fusion import fuse, load_preset
r = asyncio.run(fuse("question", load_preset("quality")))
print(r.text)            # final answer
print(r.telemetry)       # completions / tokens / latency / status
```

Verify offline (no key needed):
```bash
python tests/test_open_fusion.py     # core pipeline
python tests/test_eval.py            # DRACO-style eval
python tests/test_eval_suites.py     # suites + long-horizon
python tests/test_ablation.py        # mechanism-attribution ablation
```

## Does the lift hold up? Mechanism ablation

A point lift over a solo baseline can't say *where* the lift comes from, and the demo
numbers are canned. `REVIEW.md` is a full code review + feasibility assessment;
`ABLATION.md` + `open-fusion-eval --ablation` decompose the lift into its parts
(synthesis pass vs. panel diversity vs. structured judge) with paired-bootstrap 95% CIs:

```bash
open-fusion-eval --ablation --demo --suite fusion   # offline wiring check
open-fusion-eval --ablation --suite draco --panel "...,...,..." --grader llm
```

## Is it any good? Built-in evaluation

The skill ships a DRACO-style evaluation harness (see `EVAL.md`) that scores Fusion
against a solo baseline on weighted rubrics (4 categories, negative-weight penalties),
grades each task 3× with a judge model, and prints a verdict — score lift, cost
multiplier, quality-per-dollar, errors triggered, and a "worth it / not worth it"
conclusion.

```bash
open-fusion-eval --demo                 # zero-key demo: see the report + verdict
open-fusion-eval --suite fusion --demo  # explicit Fusion-capability smoke test
open-fusion-eval --preset budget --baseline anthropic/claude-opus-4.8 \
    --grader llm --grader-model google/gemini-3.1-pro-preview --tools --md report.md
```

## Layout

```
open-fusion-skill/
├── SKILL.md                 # skill manifest + when-to-use + run cmds (Claude Code / Codex)
├── INSTALL.md               # install + agent integration + grounded-tool setup
├── ARCHITECTURE.md          # module map, contracts, data flow, state machine
├── EVAL.md                  # DRACO-style + Fusion-capability evaluation standard
├── pyproject.toml           # pip-installable; registers `open-fusion` + `open-fusion-eval`
├── config/presets.yaml      # quality / budget panels
├── references/
│   └── fusion-mechanism.md  # the principle-level mechanism this models
├── tests/
│   ├── fake_client.py       # scriptable in-memory client (zero network)
│   ├── test_open_fusion.py  # 31 core offline tests
│   └── test_eval.py         # 34 eval offline tests
└── src/open_fusion/
    ├── schema.py        data contract (Analysis, PanelResponse, FusionResult, ...)
    ├── prompts.py       judge + synthesis prompt templates
    ├── config.py        ModelSpec / Params / FusionConfig / presets / from_cli / from_plugin
    ├── cost.py          completion / token / latency accounting + Telemetry
    ├── tools.py         phase->toolset gating + web_search/web_fetch/bash execution
    ├── client.py        OpenAI-compatible async client (stdlib HTTP, retries, JSON mode)
    ├── panel.py         parallel fan-out + agent loop + timeouts + partial failure
    ├── judge.py         distill -> validate -> retry-on-bad-JSON
    ├── synthesizer.py   grounded final writer (no web tools) + fallback writer
    ├── orchestrator.py  pipeline + depth guard + degradation state machine
    ├── cli.py           `open-fusion "..." --preset quality`
    └── eval/            DRACO-style evaluation harness
        ├── rubric.py    weighted criteria, 4 categories, negative weights, 0-100 scoring
        ├── grader.py    LLM-as-judge grader (N passes) + offline rule grader
        ├── harness.py   run fusion vs solo baseline, aggregate, compare
        ├── report.py    verdict decision rules + markdown/json rendering
        ├── demo.py      zero-key demo client
        ├── cli.py       `open-fusion-eval --demo`
        └── tasks/              # sample, DRACO-style, long-horizon, Fusion capability suites
```

## Design rules that are correctness, not style

1. Synthesis phase has **no web tools** (`tools.toolset_for_phase`).
2. Panel responses carry `[MODEL X]` labels; `contradictions` carry per-model
   attribution (also the audit trail).
3. Graceful degradation: all-fail → error; judge-fail → answer from raw responses.
4. Bounded recursion: inner calls cannot re-trigger fusion (`MAX_FUSION_DEPTH`).
5. Heterogeneous panels by default (mechanism M2).

Tools are **off by default** so the skill runs with only a model API key. Enable
grounded panels with `--tools` (`web_fetch` needs no key; `web_search` needs
`EXA_API_KEY`/`BRAVE_API_KEY`; `bash` needs `OPEN_FUSION_ENABLE_BASH=1`).

See `BEGINNER_GUIDE.md` for a beginner-friendly agent/chatbot integration manual, `INSTALL.md` for setup, and `ARCHITECTURE.md` for internals.
