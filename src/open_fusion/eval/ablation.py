"""
open_fusion.eval.ablation - factorial ablation harness for the Fusion mechanism.

The DRACO harness (harness.py) answers one question: "Fusion vs a solo baseline,
worth it?". It cannot say WHERE the lift comes from, and it reports a single point
lift with no uncertainty. This module fills both gaps.

It runs the SAME task suite through several pipeline ARMS that each disable one
component of Fusion, then attributes the total lift to its parts and attaches a
paired bootstrap confidence interval to every contrast. Every arm is built from the
existing pipeline modules (panel / judge / synthesizer / orchestrator) - nothing is
re-implemented, so an arm measures the real code path, not a model of it.

The canonical arms (strongest single model = `m0`, heterogeneous set = `m0..mk`):

    solo              one model, one call .......................... no panel, no synthesis
    homo-freeform     [m0, m0] -> free-form synthesis .............. + second pass, no diversity, no structured judge
    homo-structured   [m0, m0] -> judge -> synthesis ............... + structured judge,  no diversity
    hetero-freeform   [m0..mk] -> free-form synthesis .............. + diversity, no structured judge
    hetero-structured [m0..mk] -> judge -> synthesis (= FUSION) .... the full product

From those five arms each mechanism claim becomes a single paired contrast:

    total_fusion_lift     = hetero-structured - solo            (is Fusion worth it at all?)
    synthesis_lift        = homo-freeform     - solo            (M-second-pass: does a 2nd pass alone help?)
    diversity_lift        = hetero-structured - homo-structured (M2: does a heterogeneous panel help?)
    structured_judge_lift = hetero-structured - hetero-freeform (M4: does the typed Analysis beat free-form?)

This is exactly the confound EVAL.md flags ("a meaningful share of the lift comes
from the synthesis step itself"): synthesis_lift quantifies it instead of waving at it.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from .. import panel as panel_mod
from .. import synthesizer as synth_mod
from ..client import ModelClient
from ..config import FusionConfig, ModelSpec, Phase
from ..orchestrator import fuse
from .grader import Grader
from .harness import AvgScore, _grade_avg
from .rubric import Task


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
@dataclass
class Arm:
    """One pipeline variant. `kind` selects which real code path produces the answer:
      - "solo":       panel._run_one on a single model (no aggregation)
      - "freeform":   panel.run -> synthesizer.write_fallback (no structured judge)
      - "structured": the full orchestrator.fuse pipeline (judge -> synthesis)
    """
    name: str
    kind: str                       # solo | freeform | structured
    panel: list[ModelSpec]
    judge: ModelSpec
    caller: ModelSpec | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.caller is None:
            self.caller = self.judge


def build_canonical_arms(models: list[ModelSpec], judge: ModelSpec,
                         *, panel_size: int | None = None) -> list[Arm]:
    """Build the five canonical arms from a list of distinct models (strongest first).

    `models[0]` is treated as the strongest single model and is used for the solo and
    homogeneous arms; `models[:panel_size]` is the heterogeneous panel.
    """
    if not models:
        raise ValueError("need at least one model")
    m0 = models[0]
    k = panel_size or len(models)
    hetero = models[:max(2, k)] if len(models) >= 2 else [m0, m0]
    return [
        Arm("solo", "solo", [m0], judge, m0,
            "single model, single call (no panel, no synthesis)"),
        Arm("homo-freeform", "freeform", [m0, m0], judge, judge,
            "same model x2 -> free-form synthesis (isolates the second pass)"),
        Arm("homo-structured", "structured", [m0, m0], judge, judge,
            "same model x2 -> structured judge -> synthesis (adds the typed judge, no diversity)"),
        Arm("hetero-freeform", "freeform", hetero, judge, judge,
            "heterogeneous panel -> free-form synthesis (adds diversity, no structured judge)"),
        Arm("hetero-structured", "structured", hetero, judge, judge,
            "the full Fusion pipeline (diversity + structured judge + synthesis)"),
    ]


# --------------------------------------------------------------------------- #
# Running one arm
# --------------------------------------------------------------------------- #
async def _answer_for_arm(arm: Arm, task: Task, base: FusionConfig,
                          client: ModelClient) -> str:
    """Produce the arm's final answer text for one task, reusing real code paths.

    Contamination guard mirrors harness.evaluate: task exclusions are merged into the
    run's excluded domains so a grounded panel can never fetch the rubric source.
    """
    excluded = list(set(base.excluded_domains) | set(task.excluded_domains))
    panel_params = base.params[Phase.PANEL]

    if arm.kind == "solo":
        resp = await panel_mod._run_one(client, arm.panel[0], task.prompt,
                                        panel_params, base.tools_enabled, excluded)
        return resp.content

    if arm.kind == "freeform":
        responses = await panel_mod.run(
            client, arm.panel, task.prompt, panel_params,
            tools_enabled=base.tools_enabled, max_in_flight=base.max_in_flight,
            excluded_domains=excluded, fast_majority_k=base.fast_majority_k)
        if not any(r.ok for r in responses):
            return ""
        return await synth_mod.write_fallback(
            client, arm.caller or arm.judge, task.prompt, responses,
            base.params[Phase.SYNTHESIS])

    if arm.kind == "structured":
        run_cfg = FusionConfig(
            panel=arm.panel, judge=arm.judge, caller=arm.caller,
            params=base.params, depth=base.depth, tools_enabled=base.tools_enabled,
            max_in_flight=base.max_in_flight, excluded_domains=excluded,
            fast_majority_k=base.fast_majority_k,
            base_url=base.base_url, api_key=base.api_key)
        fr = await fuse(task.prompt, run_cfg, client=client)
        return fr.text or ""

    raise ValueError(f"unknown arm kind: {arm.kind}")


async def run_arm(arm: Arm, tasks: list[Task], base: FusionConfig, grader: Grader,
                  client: ModelClient, n_passes: int) -> list[AvgScore]:
    """Run one arm over every task; return per-task averaged scores (N grading passes)."""
    out: list[AvgScore] = []
    for task in tasks:
        text = await _answer_for_arm(arm, task, base, client)
        out.append(await _grade_avg(grader, task, text, n_passes))
    return out


# --------------------------------------------------------------------------- #
# Paired bootstrap: the statistic the base harness is missing
# --------------------------------------------------------------------------- #
@dataclass
class Contrast:
    a: str                  # arm whose score we expect to be higher
    b: str                  # arm we subtract
    mechanism: str          # which Fusion claim this isolates
    mean_lift: float        # observed mean(a_i - b_i) over tasks
    ci_low: float
    ci_high: float
    p_two_sided: float
    n: int

    @property
    def significant(self) -> bool:
        """95% CI excludes 0 (paired, percentile bootstrap)."""
        return self.ci_low > 0 or self.ci_high < 0

    @property
    def verdict(self) -> str:
        if not self.significant:
            return "null"          # no detectable effect at this sample size
        return "supported" if self.mean_lift > 0 else "contradicted"

    def to_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "mechanism": self.mechanism,
                "mean_lift": round(self.mean_lift, 2),
                "ci95": [round(self.ci_low, 2), round(self.ci_high, 2)],
                "p_two_sided": round(self.p_two_sided, 4), "n_tasks": self.n,
                "significant": self.significant, "verdict": self.verdict}


def paired_bootstrap(a: list[float], b: list[float], *, iters: int = 2000,
                     seed: int = 0, ci: float = 0.95) -> tuple[float, float, float, float]:
    """Paired percentile bootstrap on the per-task differences a_i - b_i.

    Returns (mean_lift, ci_low, ci_high, p_two_sided). Deterministic for a fixed seed.
    `p_two_sided` is 2 * min(P[boot<=0], P[boot>=0]) clamped to 1 - a bootstrap analogue
    of a paired test that needs no normality assumption (right for noisy LLM-judge scores
    and the small task counts these suites ship with).
    """
    if len(a) != len(b):
        raise ValueError("paired bootstrap needs equal-length samples")
    n = len(a)
    diffs = [ai - bi for ai, bi in zip(a, b)]
    mean_lift = sum(diffs) / n if n else 0.0
    if n < 2:
        return mean_lift, mean_lift, mean_lift, 1.0

    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(iters):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        boot.append(s)
    boot.sort()
    lo_i = max(0, int((1 - ci) / 2 * iters) - 1)
    hi_i = min(iters - 1, int((1 + ci) / 2 * iters) - 1)
    ci_low, ci_high = boot[lo_i], boot[hi_i]

    le0 = sum(1 for x in boot if x <= 0) / iters
    ge0 = sum(1 for x in boot if x >= 0) / iters
    p = min(1.0, 2 * min(le0, ge0))
    return mean_lift, ci_low, ci_high, p


# --------------------------------------------------------------------------- #
# Top-level ablation run
# --------------------------------------------------------------------------- #
_CONTRASTS = [
    ("total_fusion_lift",     "hetero-structured", "solo",
     "Is full Fusion worth it vs the best single model?"),
    ("synthesis_lift",        "homo-freeform",     "solo",
     "M-second-pass: does a synthesis pass alone (no diversity, no typed judge) help?"),
    ("diversity_lift",        "hetero-structured", "homo-structured",
     "M2 (heterogeneity): does a diverse panel beat the same model repeated?"),
    ("structured_judge_lift", "hetero-structured", "hetero-freeform",
     "M4 (adjudication over averaging): does the typed Analysis beat free-form fusion?"),
]


async def run_ablation(
    tasks: list[Task],
    models: list[ModelSpec],
    judge: ModelSpec,
    grader: Grader,
    *,
    client: ModelClient | None = None,
    base: FusionConfig | None = None,
    n_passes: int = 3,
    bootstrap_iters: int = 2000,
    seed: int = 0,
    panel_size: int | None = None,
) -> dict[str, Any]:
    """Run the canonical factorial ablation and decompose the lift with CIs."""
    arms = build_canonical_arms(models, judge, panel_size=panel_size)
    base = base or FusionConfig(panel=arms[-1].panel, judge=judge)
    if client is None:
        client = ModelClient(fusion_depth=base.depth + 1,
                             base_url=base.base_url, api_key=base.api_key)

    # per-arm, per-task normalized scores
    arm_scores: dict[str, list[AvgScore]] = {}
    for arm in arms:
        arm_scores[arm.name] = await run_arm(arm, tasks, base, grader, client, n_passes)

    arms_block: dict[str, Any] = {}
    for arm in arms:
        scores = arm_scores[arm.name]
        norms = [s.normalized for s in scores]
        arms_block[arm.name] = {
            "kind": arm.kind,
            "panel": [m.slug for m in arm.panel],
            "description": arm.description,
            "mean": round(sum(norms) / len(norms), 2) if norms else 0.0,
            "per_task": [round(x, 2) for x in norms],
        }

    contrasts: dict[str, Any] = {}
    for name, a_name, b_name, mechanism in _CONTRASTS:
        a = [s.normalized for s in arm_scores[a_name]]
        b = [s.normalized for s in arm_scores[b_name]]
        mean_lift, lo, hi, p = paired_bootstrap(a, b, iters=bootstrap_iters, seed=seed)
        contrasts[name] = Contrast(a_name, b_name, mechanism, mean_lift, lo, hi, p,
                                   len(a)).to_dict()

    return {
        "config": {
            "arms": [a.name for a in arms],
            "models": [m.slug for m in models],
            "judge": judge.slug,
            "grader": getattr(grader, "name", grader.__class__.__name__),
            "n_tasks": len(tasks),
            "n_passes": n_passes,
            "bootstrap_iters": bootstrap_iters,
            "tools_enabled": base.tools_enabled,
        },
        "arms": arms_block,
        "contrasts": contrasts,
        "interpretation": _interpret(contrasts),
    }


def _interpret(contrasts: dict[str, Any]) -> list[str]:
    out: list[str] = []
    tot = contrasts["total_fusion_lift"]
    syn = contrasts["synthesis_lift"]
    div = contrasts["diversity_lift"]
    jdg = contrasts["structured_judge_lift"]

    if tot["verdict"] == "supported":
        out.append(f"Fusion beats the best single model by {tot['mean_lift']:+} pts "
                   f"(95% CI {tot['ci95']}), a real effect at this sample size.")
    elif tot["verdict"] == "contradicted":
        out.append(f"Fusion UNDERPERFORMS the single model by {tot['mean_lift']:+} pts "
                   f"(95% CI {tot['ci95']}): the pipeline is not worth it on these tasks.")
    else:
        out.append(f"No detectable total lift ({tot['mean_lift']:+} pts, 95% CI {tot['ci95']}); "
                   "the task count is too small or the effect too weak to claim a win.")

    # attribution: how much of the total is just the second pass?
    if syn["verdict"] == "supported":
        share = (syn["mean_lift"] / tot["mean_lift"]) if tot["mean_lift"] > 0 else None
        frag = f" (~{share:.0%} of the total lift)" if share and 0 < share <= 1.5 else ""
        out.append(f"Synthesis confound is REAL: a second pass over the same model alone "
                   f"adds {syn['mean_lift']:+} pts{frag}. Diversity must be judged net of this.")
    else:
        out.append("Second-pass synthesis alone shows no significant lift — the value, "
                   "if any, comes from the panel, not from re-writing.")

    out.append(("Heterogeneity (M2) pays" if div["verdict"] == "supported"
                else "Heterogeneity (M2) shows no significant net lift") +
               f": diverse panel vs same-model panel = {div['mean_lift']:+} pts (95% CI {div['ci95']}).")
    out.append(("The structured judge (M4) beats free-form" if jdg["verdict"] == "supported"
                else "The structured judge (M4) is not distinguishable from free-form fusion") +
               f": {jdg['mean_lift']:+} pts (95% CI {jdg['ci95']}).")
    return out


# --------------------------------------------------------------------------- #
# Rendering + offline demo
# --------------------------------------------------------------------------- #
def render_ablation_markdown(report: dict[str, Any]) -> str:
    c = report["config"]
    L: list[str] = []
    L.append("# Fusion ablation — mechanism attribution")
    L.append("")
    L.append(f"- Models: {', '.join(c['models'])}")
    L.append(f"- Judge: {c['judge']} · grader: {c['grader']} · tasks: {c['n_tasks']} "
             f"· passes/task: {c['n_passes']} · bootstrap: {c['bootstrap_iters']} iters")
    L.append("")
    L.append("## Arm scores (0–100, mean over tasks)")
    L.append("| Arm | Kind | Panel | Mean |")
    L.append("|---|---|---|---:|")
    for name, blk in report["arms"].items():
        L.append(f"| {name} | {blk['kind']} | {', '.join(blk['panel'])} | {blk['mean']} |")
    L.append("")
    L.append("## Lift decomposition (paired bootstrap, 95% CI)")
    L.append("| Contrast | Mechanism | Lift | 95% CI | p | Verdict |")
    L.append("|---|---|---:|---|---:|---|")
    for key, ct in report["contrasts"].items():
        L.append(f"| {key} | {ct['mechanism']} | {ct['mean_lift']:+} | "
                 f"[{ct['ci95'][0]}, {ct['ci95'][1]}] | {ct['p_two_sided']} | **{ct['verdict']}** |")
    L.append("")
    L.append("## Interpretation")
    for line in report["interpretation"]:
        L.append(f"- {line}")
    return "\n".join(L)


def run_ablation_demo(tasks: list[Task], *, seed: int = 0) -> dict[str, Any]:
    """Offline ablation over canned answers (no key). Exercises every arm's real code
    path through DemoClient + RuleGrader. NOTE: the demo's answers are canned, so the
    CONTRASTS are not meaningful evidence — this only proves the harness is wired
    correctly end to end (use real models + LLM grader for real numbers)."""
    import asyncio

    from .demo import DemoClient
    from .grader import RuleGrader

    models = [ModelSpec("demo/panel-a"), ModelSpec("demo/panel-b")]
    judge = ModelSpec("demo/judge")
    client = DemoClient(tasks, baseline_slug="demo/panel-a")
    base = FusionConfig(panel=models, judge=judge)
    return asyncio.run(run_ablation(tasks, models, judge, RuleGrader(),
                                    client=client, base=base, n_passes=3, seed=seed))
