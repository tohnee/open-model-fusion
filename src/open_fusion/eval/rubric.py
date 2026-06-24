"""
open_fusion.eval.rubric - DRACO-style weighted rubric and scoring math.

Faithfully reproduces the DRACO grading model described in OpenRouter's
"Surpassing Frontier Performance with Fusion" (2026-06-12):

  - Each task carries a rubric of weighted criteria across FOUR categories:
        Factual Accuracy (~20 criteria), Breadth & Depth (~9),
        Presentation Quality (~6), Citation Quality (~5).
  - Criteria may carry NEGATIVE weights. "Meeting" a negative criterion means the
    response contains that error (e.g. dangerous medical advice) and is penalised.
  - A response is graded per-criterion (met / not-met), the normalized score is
        clamp( sum(weight_i * met_i) / sum(positive weights), 0, 1 ) * 100
    so verbosity cannot inflate a score (only meeting real criteria adds points;
    only errors subtract). This is the "can't bluff your way to a high score" rule.

The DRACO methodology grades each task THREE independent times with a judge model
and reports the mean normalized score; that averaging lives in the harness, this
module owns one deterministic score given a met/not-met verdict per criterion.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RubricCategory(str, Enum):
    FACTUAL_ACCURACY = "factual_accuracy"   # verifiable claims the response must get right
    BREADTH_DEPTH = "breadth_depth"         # synthesis, trade-offs, actionable guidance
    PRESENTATION = "presentation"           # terminology, formatting, readability
    CITATION = "citation"                   # primary sources, working references


# Documented DRACO target shape (~39 criteria total). Used for reporting weight
# distribution and for sanity-checking a hand-authored rubric against the paper.
DRACO_CATEGORY_TARGETS = {
    RubricCategory.FACTUAL_ACCURACY: 20,
    RubricCategory.BREADTH_DEPTH: 9,
    RubricCategory.PRESENTATION: 6,
    RubricCategory.CITATION: 5,
}


@dataclass
class Criterion:
    id: str
    description: str
    category: RubricCategory
    weight: float                              # < 0 => penalty (error) criterion
    check: dict[str, list[str]] | None = None  # optional offline rule (see RuleGrader)
    critical: bool = False                     # long-horizon: a milestone that MUST hold for pass^k
    order: int | None = None                   # long-horizon: milestone sequence position

    @property
    def is_penalty(self) -> bool:
        return self.weight < 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Criterion":
        return cls(id=d["id"], description=d.get("description", ""),
                   category=RubricCategory(d["category"]),
                   weight=float(d["weight"]), check=d.get("check"),
                   critical=bool(d.get("critical", False)), order=d.get("order"))


@dataclass
class Task:
    id: str
    domain: str
    prompt: str
    rubric: list[Criterion]
    excluded_domains: list[str] = field(default_factory=list)  # contamination guard
    reference: str | None = None                               # optional gold notes
    kind: str = "draco"                                        # "draco" | "longhorizon"
    horizon: str | None = None                                 # LH: descriptive step/time horizon

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Task":
        return cls(id=d["id"], domain=d.get("domain", "general"),
                   prompt=d["prompt"],
                   rubric=[Criterion.from_dict(c) for c in d.get("rubric", [])],
                   excluded_domains=list(d.get("excluded_domains") or []),
                   reference=d.get("reference"),
                   kind=d.get("kind", "draco"), horizon=d.get("horizon"))

    @property
    def is_longhorizon(self) -> bool:
        return self.kind == "longhorizon"

    def positive_weight(self) -> float:
        return sum(c.weight for c in self.rubric if c.weight > 0)

    def critical_criteria(self) -> list[Criterion]:
        """Milestones that must all hold for a run to 'pass' (pass^k semantics)."""
        return [c for c in self.rubric if c.critical]


@dataclass
class CategoryScore:
    category: RubricCategory
    achieved: float
    possible: float

    @property
    def normalized(self) -> float:
        return 100.0 * max(0.0, min(1.0, self.achieved / self.possible)) if self.possible else 0.0


@dataclass
class GradeResult:
    """Score for ONE response on ONE task (one judging pass)."""
    task_id: str
    normalized: float                          # 0..100
    achieved: float
    possible: float
    met: dict[str, bool]
    categories: dict[RubricCategory, CategoryScore]
    penalties_triggered: list[str]             # ids of negative criteria that were met (errors)

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "normalized": round(self.normalized, 2),
                "achieved": round(self.achieved, 3), "possible": round(self.possible, 3),
                "categories": {c.value: round(cs.normalized, 2) for c, cs in self.categories.items()},
                "penalties_triggered": self.penalties_triggered}


def score_response(task: Task, met: dict[str, bool]) -> GradeResult:
    """Turn a per-criterion met/not-met verdict into a normalized DRACO-style score."""
    achieved = 0.0
    cat_ach: dict[RubricCategory, float] = {}
    cat_pos: dict[RubricCategory, float] = {}
    penalties: list[str] = []

    for c in task.rubric:
        is_met = bool(met.get(c.id, False))
        if c.weight > 0:
            cat_pos[c.category] = cat_pos.get(c.category, 0.0) + c.weight
        if is_met:
            achieved += c.weight
            cat_ach[c.category] = cat_ach.get(c.category, 0.0) + c.weight
            if c.is_penalty:
                penalties.append(c.id)

    possible = task.positive_weight()
    normalized = 100.0 * max(0.0, min(1.0, achieved / possible)) if possible else 0.0

    categories = {
        cat: CategoryScore(cat, cat_ach.get(cat, 0.0), cat_pos.get(cat, 0.0))
        for cat in RubricCategory
        if cat in cat_pos or cat in cat_ach
    }
    return GradeResult(task_id=task.id, normalized=normalized, achieved=achieved,
                       possible=possible, met=dict(met), categories=categories,
                       penalties_triggered=penalties)


# --- offline rule evaluation (used by RuleGrader) ----------------------------
def evaluate_check(text: str, check: dict[str, list[str]] | None) -> bool:
    """Deterministic criterion check. A criterion is 'met' when ALL provided
    clauses hold: every pattern in `all` matches, at least one in `any` matches,
    and none in `none` match. Patterns are case-insensitive regexes (a plain
    substring is a valid regex). No `check` => not auto-gradable => False."""
    if not check:
        return False
    hay = text or ""

    def _search(p: str) -> bool:
        try:
            return re.search(p, hay, re.IGNORECASE | re.DOTALL) is not None
        except re.error:
            return p.lower() in hay.lower()

    ok = True
    if check.get("all"):
        ok = ok and all(_search(p) for p in check["all"])
    if check.get("any"):
        ok = ok and any(_search(p) for p in check["any"])
    if check.get("none"):
        ok = ok and not any(_search(p) for p in check["none"])
    return ok
