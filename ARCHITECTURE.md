# Architecture

Open Fusion is a single-layer Mixture-of-Agents with a structured-JSON judge step.
This document is the internals map: module responsibilities, the contracts between
them, the data that flows, and the orchestrator state machine.

## 1. Import graph (no cycles)

```
schema.py            # base types + enums (Phase, FusionStatus, TokenUsage,
   ^   ^   ^   ^      #                     PanelResponse, Analysis, FusionResult)
   |   |   |   |
config tools cost prompts
   ^     ^     \   /
   |     |      \ /
   +--- client.py        # the ONLY module that does HTTP
          ^
        panel.py  judge.py  synthesizer.py
              \      |      /
             orchestrator.py
                   ^
                 cli.py
```

`schema.py` imports nothing from the package, so it can never participate in a cycle.
`Phase` is defined in `schema.py` and re-exported by `config.py`; importing it from
either module yields the *same* enum object (this matters: `orchestrator` imports it
from `config`, `panel`/`judge`/`tools` from `schema`).

## 2. Module responsibilities

- **schema** — the data contract. Pure dataclasses + enums. `Analysis.validate()`
  encodes the correctness rules (non-empty; every stance/insight has `model`).
- **config** — `ModelSpec`, `Params`, `FusionConfig`, presets, `MAX_FUSION_DEPTH`,
  `from_cli`. `validate()` returns hard problems plus `WARNING:`-prefixed advisories.
- **tools** — `toolset_for_phase` (the gate) and `execute_tool` (offline-safe; checks
  `excluded_domains` before any network).
- **cost** — `Telemetry`: per-completion usage, token totals, panel ok/fail counts,
  judge retries, critical-path ms.
- **prompts** — judge/synthesis templates + `label_responses` (positional `[MODEL X]`).
- **client** — async wrapper over an OpenAI-compatible gateway. Blocking `urllib`
  runs in the loop's thread executor. Typed errors (`Timeout`, `RateLimit`,
  `ProviderError`). Retries `_RETRY_STATUSES` with capped backoff; drops `json` mode
  and retries once on a 400.
- **panel** — parallel fan-out. `_run_one` is a bounded tool-using agent loop that
  **never raises** (failures become `PanelResponse(status=...)`). `run` preserves
  panel order so labels are stable; optional `fast_majority_k` early-exit.
- **judge** — distill → validate → retry-once-on-bad-JSON → `JudgeError`.
- **synthesizer** — `write` (from `Analysis`) and `write_fallback` (from raw
  responses). Both run with `tools=()`.
- **orchestrator** — the pipeline + the state machine below.
- **cli** — argv → `FusionConfig` → `fuse`; answer to stdout, diagnostics to stderr.

## 3. Contracts that matter

- `panel.run` returns one `PanelResponse` per panel slot, in panel order, regardless
  of failure. `r.ok == (status == "ok")`.
- `judge.synthesize` raises `JudgeError` only after two failed parses. On success it
  returns a validated `Analysis`.
- `synthesizer.*` return a plain `str` (the final answer).
- `Telemetry.add_usage` is called exactly once per billed completion, so
  `summary()["completions"]` = panel survivors-or-attempts + judge attempt(s) +
  synthesis.

## 4. Data flow

```
question ─▶ panel.run ─▶ [PanelResponse...] ─▶ label_responses ─▶ judge.synthesize
                                                                       │
                                              Analysis (validated) ◀───┘
                                                   │
                                  synthesizer.write(question, Analysis) ─▶ final text
```

On judge failure the analysis edge is replaced by
`synthesizer.write_fallback(question, [PanelResponse...])`.

## 5. State machine

```
            depth >= MAX_FUSION_DEPTH ──▶ ERROR ("answer without fusion")
                       │ no
                       ▼
                    FAN_OUT  (panel.run)
                       │
        n_panel_ok==0 ─┼─▶ ALL_PANEL_FAILED ──▶ ERROR
                       │ >=1 survivor
                       ▼
                     JUDGE  (judge.synthesize)
                       │
         JudgeError ───┼─▶ JUDGE_FAILED ──▶ write_fallback ──▶ JUDGE_FALLBACK
                       │ ok
                       ▼
                   SYNTHESIZE  (synthesizer.write)
                       │
                       ▼
                      DONE  (OK)
```

Terminal statuses: `OK`, `JUDGE_FALLBACK`, `ERROR`. Always check `result.status`.

## 6. Why these are correctness, not style

1. **Synthesis has no web tools.** Once the judge has reasoned over the evidence,
   the answer must be written from that frozen set — otherwise the analysis and the
   answer can silently diverge. Enforced by `toolset_for_phase(SYNTHESIS) == ()`
   and by `synthesizer` passing `tools=()`.
2. **Per-model attribution.** Labels are positional and every stance/insight names
   its model; this is also the audit trail. Enforced by `Analysis.validate()`.
3. **Graceful degradation.** All panels fail → `ERROR`; judge fails → answer from
   raw responses (`JUDGE_FALLBACK`); ≥1 survivor → proceed.
4. **Bounded recursion.** An inner call cannot re-trigger fusion
   (`depth >= MAX_FUSION_DEPTH`); the client also stamps `x-open-fusion-depth`.
5. **Heterogeneous panels by default.** Same-vendor panels deliberate weakly;
   `FusionConfig.validate()` emits a `WARNING`.
