"""
open_fusion.eval.longhorizon - reliability metrics for long-horizon tasks.

DRACO grades a single deep-research answer. Long-horizon work needs a different
lens, grounded in the current SOTA agent benchmarks:

  - Milestone / sub-task completion (RoadmapBench's ~5 structured subtasks per
    instance; tau-bench's goal-state comparison): did the answer/plan actually hit
    each ordered subgoal?
  - Coherence & error propagation (RetailBench, LongDS-Bench): does it stay
    self-consistent over the horizon, or does an early mistake compound? Modeled as
    negative-weight 'drift' criteria.
  - Reliability via pass^k (tau-bench): pass^k is the probability of succeeding on
    ALL k repeated trials. Capability (pass@1) saturates with k while pass^k
    collapses, so a high mean score with low pass^k means "smart but unreliable".

This module turns the harness's N independent grading passes (each a met-map) into
those metrics. The per-pass 0-100 score still comes from rubric.score_response
(positive milestones add, drift penalties subtract), so DRACO and long-horizon share
one scoring core; only the reliability lens differs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .rubric import Task, majority


def _critical_success(task: Task, met: dict[str, bool]) -> bool:
    """One run 'passes' when every critical milestone is met and no critical
    penalty (drift/error) was triggered."""
    crit = task.critical_criteria()
    if not crit:
        # no explicit criticals -> treat all positive milestones as required
        crit = [c for c in task.rubric if c.weight > 0]
    for c in crit:
        is_met = bool(met.get(c.id, False))
        if c.weight > 0 and not is_met:
            return False
        if c.weight < 0 and is_met:     # a critical error occurred
            return False
    return True


@dataclass
class LongHorizonScore:
    milestone_completion: float        # 0-100, mean fraction of positive milestones met
    pass_k: float                      # 0/1 for one task: success on ALL k passes
    pass_at_1: float                   # 0-100, fraction of passes that individually passed
    drift_errors: int                  # distinct drift/coherence penalties triggered (>= majority)
    k: int
    per_pass_pass: list[bool] = field(default_factory=list)


def score_passes(task: Task, met_passes: list[dict[str, bool]]) -> LongHorizonScore:
    """Aggregate N grading passes for ONE long-horizon task into reliability metrics."""
    k = max(1, len(met_passes))
    milestones = [c for c in task.rubric if c.weight > 0]
    mtotal = sum(c.weight for c in milestones) or 1.0

    completions: list[float] = []
    passes: list[bool] = []
    drift_hits: dict[str, int] = {}
    for met in met_passes:
        earned = sum(c.weight for c in milestones if met.get(c.id))
        completions.append(100.0 * earned / mtotal)
        passes.append(_critical_success(task, met))
        for c in task.rubric:
            if c.weight < 0 and met.get(c.id):
                drift_hits[c.id] = drift_hits.get(c.id, 0) + 1

    maj = majority(k)
    drift = [cid for cid, h in drift_hits.items() if h >= maj]
    pass_k = 1.0 if all(passes) else 0.0          # tau-bench: all k trials succeed
    pass_at_1 = 100.0 * sum(1 for p in passes if p) / k
    return LongHorizonScore(
        milestone_completion=sum(completions) / k, pass_k=pass_k,
        pass_at_1=pass_at_1, drift_errors=len(drift), k=k, per_pass_pass=passes)


def aggregate(scores: list[LongHorizonScore]) -> dict:
    """Aggregate per-task LH scores into suite-level reliability numbers."""
    if not scores:
        return {}
    n = len(scores)
    k = scores[0].k
    return {
        "k": k,
        "milestone_completion": round(sum(s.milestone_completion for s in scores) / n, 2),
        "pass_hat_k": round(100.0 * sum(s.pass_k for s in scores) / n, 2),   # % tasks solved on ALL k passes
        "pass_at_1": round(sum(s.pass_at_1 for s in scores) / n, 2),
        "drift_errors": sum(s.drift_errors for s in scores),
        "n_tasks": n,
    }
