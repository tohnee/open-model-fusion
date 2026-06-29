"""
open_fusion.eval - DRACO-style evaluation harness for the Fusion skill.

Gives the skill an evaluation standard (weighted rubrics, 4 DRACO categories,
negative-weight penalties), feedback (per-category / per-domain breakdown, errors
triggered), and a verdict (score lift vs a solo baseline, cost multiplier, and a
"worth it / not worth it" conclusion).

Quick start (offline, no key):
    from open_fusion.eval import load_tasks, run_demo
    report = run_demo(load_tasks("sample"))
    print(report["verdict"]["conclusion"])
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from ..config import FusionConfig, ModelSpec
from .ablation import (Arm, build_canonical_arms, paired_bootstrap,
                       render_ablation_markdown, run_ablation, run_ablation_demo)
from .superiority import (run_superiority_benchmark, run_superiority_demo,
                          render_superiority_markdown, build_superiority_arms)
from .grader import Grader, LLMJudgeGrader, RuleGrader
from .harness import evaluate
from .report import render_json, render_markdown
from .rubric import (Criterion, GradeResult, RubricCategory, Task,
                     DRACO_CATEGORY_TARGETS, score_response)

__all__ = [
    "load_tasks", "evaluate", "run_demo",
    "RuleGrader", "LLMJudgeGrader", "Grader",
    "Task", "Criterion", "RubricCategory", "GradeResult", "score_response",
    "DRACO_CATEGORY_TARGETS", "render_markdown", "render_json",
    # ablation harness (mechanism attribution + paired bootstrap)
    "run_ablation", "run_ablation_demo", "render_ablation_markdown",
    "build_canonical_arms", "paired_bootstrap", "Arm",
    # superiority benchmark (prove fusion beats best single model + oracle)
    "run_superiority_benchmark", "run_superiority_demo",
    "render_superiority_markdown", "build_superiority_arms",
]

_TASK_DIR = Path(__file__).parent / "tasks"
_SUITES = {
    "sample": "sample_tasks.json",
    "draco": "draco_tasks.json",
    "longhorizon": "longhorizon_tasks.json",
    "semiconductor": "semiconductor_dram_tasks.json",
    "fusion": "fusion_capability_tasks.json",
    "superiority": "superiority_tasks.json",
    "superiority_v2": "superiority_tasks_v2.json",
    "logic_hard": "logic_hard_tasks.json",
}


def _load_file(path: Path) -> list[Task]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("tasks", data) if isinstance(data, dict) else data
    tasks = [Task.from_dict(t) for t in raw]
    for t, d in zip(tasks, raw):
        if isinstance(d, dict) and "demo" in d:
            t.demo = d["demo"]  # type: ignore[attr-defined]
    return tasks


def load_tasks(source: str = "sample") -> list[Task]:
    """Load a task set. `source` is a suite name ('sample', 'draco', 'longhorizon',
    'semiconductor', 'fusion', 'all') or a path to a DRACO-style JSON file."""
    if source == "all":
        out: list[Task] = []
        for name in ("draco", "longhorizon", "semiconductor"):
            out.extend(_load_file(_TASK_DIR / _SUITES[name]))
        return out
    if source in _SUITES:
        return _load_file(_TASK_DIR / _SUITES[source])
    return _load_file(Path(source))


def run_demo(tasks: list[Task], *, baseline: str = "demo/baseline") -> dict:
    """Run the full harness offline with canned answers (no key/network)."""
    from .demo import DemoClient
    client = DemoClient(tasks, baseline_slug=baseline)
    cfg = FusionConfig(panel=[ModelSpec("demo/panel-a"), ModelSpec("demo/panel-b")],
                       judge=ModelSpec("demo/judge"))
    kinds = sorted({t.kind for t in tasks})
    label = f"demo · {len(tasks)} tasks ({'+'.join(kinds)}) · rule grader"
    return asyncio.run(evaluate(tasks, cfg, ModelSpec(baseline), RuleGrader(),
                                client=client, n_passes=3, label=label))
