# System analysis, code review & feasibility — open-fusion

A full read of every module in `src/open_fusion/` (core + eval) and all four test
suites, plus an assessment of whether the Fusion scheme is sound and what it would
take to *prove* it. The companion document **ABLATION.md** specifies the experiments
that turn the open questions below into measurements, and `eval/ablation.py` +
`tests/test_ablation.py` implement and verify them.

---

## 1. What the system is

open-fusion is a faithful, dependency-free re-implementation of OpenRouter's **Fusion**
pattern — a **single-layer Mixture-of-Agents** with a structured-JSON adjudication step:

```
question ─▶ panel.run (N models, parallel) ─▶ [PanelResponse...]
                                                   │ label_responses (positional MODEL A/B/…)
                                                   ▼
                                          judge.synthesize ─▶ Analysis (typed, validated)
                                                   │
                                          synthesizer.write (tools OFF) ─▶ final answer
```

Three phases, each with its own sampling temperature and **toolset gate**: panel
(hot, may search/fetch/bash), judge (cool, may verify, no shell), synthesis (cool,
**no tools** — evidence is frozen). A telemetry object threads through and bills
every completion exactly once. The orchestrator is a small state machine with three
terminal states (`OK`, `JUDGE_FALLBACK`, `ERROR`).

The design is internally coherent and the four "correctness, not style" invariants
are real and enforced in code, not just documented:

| Invariant | Enforced by |
|---|---|
| Synthesis has no web tools | `toolset_for_phase(SYNTHESIS) == ()` **and** `synthesizer.write(... tools=())` (belt + braces) |
| Per-model attribution is mandatory | `Analysis.validate()` rejects any stance/insight missing `model` |
| Graceful degradation | all-fail→`ERROR`; judge-fail-twice→`write_fallback`→`JUDGE_FALLBACK`; ≥1 survivor→proceed |
| Bounded recursion | `effective_depth = max(config.depth, client.fusion_depth)` ≥ `MAX_FUSION_DEPTH` → refuse |
| Heterogeneous panels default | `FusionConfig.validate()` emits a `WARNING:` for single-vendor panels |

## 2. Engineering quality — strong

- **Clean dependency graph, no cycles.** `schema.py` imports nothing from the package
  and sits at the bottom; `client.py` is the *only* module that touches HTTP. `Phase`
  is defined once and re-exported, avoiding the classic two-enums-identity bug.
- **Panel fan-out never raises.** `_run_one` converts every failure mode (`Timeout`,
  `ClientError`, and a defensive bare `except Exception`) into a `PanelResponse` with a
  status, so one model dying can't take down the batch. `run` preserves panel order so
  `MODEL A/B/C` labels are stable even when slot B fails — this is what makes the
  judge's attribution auditable.
- **Robust judge JSON handling.** Request JSON mode → strip fences → parse → validate
  → one repair turn that feeds the parser error back → `JudgeError` → fallback. The
  client also drops `response_format` and retries once on a 400, for providers that
  reject JSON mode.
- **Inputs validated at the construction point.** `Params.__post_init__` rejects
  `max_tool_calls<0`, `max_tokens<=0`, `timeout_s<=0`, `temperature<0` — each tied to a
  concrete downstream failure it prevents (e.g. `max_tool_calls<0` would make the panel
  loop run zero rounds and hit `UnboundLocalError`).
- **Test discipline.** 177 offline tests run with zero network via `FakeClient`/
  `DemoClient`/`RuleGrader` and cover the happy path, partial failure, all-fail, judge
  retry→ok, judge→fallback, both depth-guard routes, `fast_majority_k` cancellation
  semantics, params validation, telemetry, logging on/off, and the tool loop.

This is production-credible plumbing. The architecture is not the risk.

## 3. Code-review findings

Severity: **H**igh (could mislead a real decision) · **M**edium · **L**ow / nit.
None are crashes; the suite is green. They cluster in the **evaluation** layer, which
matters because eval is what is supposed to justify paying 3× for Fusion.

### H1 — The demo "proves" nothing, but reads like it does
`run_demo` / `DemoClient` return **canned** per-task answers rigged so the fused answer
is penalty-clean and the baseline trips an error. The eval tests then assert
`mean_fusion > mean_baseline`, `worth_it is True`, `fewer errors than baseline`
(`tests/test_eval.py:152`, `test_fusion_capability_suite`). These are **tautologies
given the fixtures** — they test plumbing, not Fusion. Nothing in the repo demonstrates
Fusion beating a strong solo model on real tasks; no real-model run is committed.
*Fix:* label the demo as a wiring smoke test everywhere it appears (done in
`ablation.run_ablation_demo`'s docstring), and never cite demo numbers as evidence.

### H2 — No uncertainty on any lift
`harness.evaluate` reports a single `lift = mean_fusion - mean_baseline` over 2–16
tasks, graded by a noisy LLM judge, and `build_verdict` keys `worth_it` off thresholds
(`lift>=2` → "win"). With that N and that noise, a +2 lift is well inside the sampling
band — the verdict can flip on judge choice or task sampling. There is **no CI, no
paired test, no per-pass variance** surfaced. This is the single biggest methodology
gap. *Fix:* the ablation harness adds a paired percentile bootstrap (`paired_bootstrap`)
and marks a contrast `significant` only when its 95% CI excludes 0.

### H3 — The synthesis/diversity confound is named but never measured
`EVAL.md` and `fusion-mechanism.md` both admit "a meaningful share of Fusion's lift
comes from the synthesis step itself — run the same model twice and synthesize and it
already helps," and tell the *user* to test a same-model panel. But the shipped harness
provides **no arm to do that**, so every reported lift silently conflates "second pass"
with "diverse panel." *Fix:* `homo-*` vs `hetero-*` arms in the ablation isolate it
(`synthesis_lift` vs `diversity_lift`).

### M1 — USD cost is computed with the wrong price for mixed-model token totals
`harness._cost` prices Fusion's **aggregate** tokens (panel + judge + synthesis, all
models summed) at a **single** slug's rate — `fusion_config.caller.slug`
(`harness.py:145`). A budget panel of three cheap models judged by an expensive
synthesizer is billed entirely at the synthesizer's rate (or vice-versa). The **token**
multiplier is correct; the **dollar** figure and any `$`-based comparison are not.
*Fix:* accumulate USD per `by_label` slot using each slot's own price.

### M2 — "Majority" is defined two different ways
`harness._grade_avg` uses `math.ceil(n/2)` for penalty majority; `longhorizon.score_passes`
uses `n//2 + 1`. They agree at the default n=3 (both 2) but **disagree for even n**
(n=2 → 1 vs 2; n=4 → 2 vs 3). Two modules scoring the same passes can label a penalty
differently. *Fix:* one shared `majority(n)` helper.

### M3 — `web_fetch` has no SSRF guard (only an opt-in denylist)
When `--tools` is on, a panel model can emit `web_fetch` against **any** URL.
`_host_excluded` is an opt-in *denylist* of domains; it does not block private ranges,
loopback, or cloud metadata (`169.254.169.254`). A grounded panel is an
attacker-influenced fetch primitive. `bash` is correctly gated behind
`OPEN_FUSION_ENABLE_BASH=1`, but `web_fetch` needs no key and is on with `--tools`.
*Fix:* block private/link-local/loopback IPs and non-http(s) schemes before fetching;
consider an allowlist mode for benchmarking.

### L1 — Heterogeneity is judged by slug prefix only
`ModelSpec.vendor` = text before `/`. Two models from one lab under different prefixes
read as "heterogeneous"; a fine-tune and its base under one prefix read as
"homogeneous." The M2 warning is a useful nudge, not a correlation measurement —
fine, but worth stating.

### L2 — `executor_workers` optimization is never wired in
`client.py` can take a dedicated thread pool (comment "O6") to avoid starving the
asyncio default executor on large panels, but neither `load_preset` nor `from_cli` ever
sets it, so big panels of blocking `urllib` calls share the default pool. Latent perf
nit, not a correctness issue.

### L3 — `from_cli` always overwrites `base_url`/`api_key` with env values
Even when env is unset it writes `None`, which then re-resolves inside `ModelClient`.
Harmless today; brittle if a caller ever sets these on the config before `from_cli`.

## 4. Feasibility assessment

**Is the scheme sound? Yes — conceptually and as engineered.** Ensembling independent
LLM answers and adjudicating them is a real, published-effective pattern; the
mechanism story (M1 error decorrelation, M2 heterogeneity as the lever, M3 coverage,
M4 adjudication-over-averaging) is internally consistent and the structured-judge step
is a defensible way to keep aggregation *selective* and auditable. The cost/latency
model is honest: additive billing (~N+2 completions), wall-clock ≈ `max(panel)+judge+
synthesis`, explicitly "use selectively for high-stakes/ambiguous questions."

**Where feasibility is unproven — and it is the crux:**

1. **No empirical evidence in-repo.** The value claim ("surpasses frontier") is
   asserted from an external article; the repo's own numbers are canned (H1). Until a
   real-model run exists, "Fusion is worth it" is a hypothesis, not a result.
2. **No statistics (H2)** means even a real run on these small suites couldn't
   distinguish a true win from noise.
3. **The confound (H3)** means an apparent win might be the free synthesis pass, not
   the panel — which would change the recommendation (you'd just self-refine one cheap
   model, not pay for three).
4. **Model slugs and the cited benchmark are speculative.** Presets reference
   `claude-fable-5`, `openai/gpt-5.5`, `anthropic/claude-opus-4.8`,
   `deepseek-v4-pro`, `gemini-3.1-pro-preview`, and a DRACO paper dated 2026 with
   `arXiv:2602.11685`. These must be validated against `openrouter.ai/models` before any
   real run; treat the article's headline numbers as unverified from this codebase.

**Bottom line:** the *artifact* is feasible, well-built, and ready to run. The
*claim it exists to support* is currently untested and, with the shipped eval, untestable
with rigor. The fix is not more architecture — it is the ablation below.

## 5. What was added in this review

To make the feasibility question answerable rather than rhetorical:

- **`src/open_fusion/eval/ablation.py`** — a factorial mechanism-attribution harness.
  Five arms (`solo`, `homo-freeform`, `homo-structured`, `hetero-freeform`,
  `hetero-structured`) built **entirely from the existing pipeline modules**, so each
  arm measures the real code path. Four paired contrasts decompose the total lift into
  *synthesis*, *diversity*, and *structured-judge* components, each with a paired
  bootstrap 95% CI and a `supported / null / contradicted` verdict.
- **`--ablation` flag** on `open-fusion-eval` (works `--demo` offline or against real
  models) and exports from `open_fusion.eval`.
- **`tests/test_ablation.py`** — 35 offline tests covering arm construction, bootstrap
  determinism/CI behaviour/edge cases, end-to-end demo wiring, and rendering. These
  assert *plumbing and statistics*, never a rigged "win" (the deliberate contrast with
  H1).

See **ABLATION.md** for the experiment design, hypotheses, controls, and the real-run
protocol.
