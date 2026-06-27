"""
Offline tests for the ablation harness. Zero network (DemoClient + RuleGrader).

These tests assert the harness is wired correctly and the statistics behave — NOT
that any particular mechanism "wins" (the demo answers are canned, so the contrasts
are not real evidence; that is the whole point of run_ablation_demo's docstring).

Run:  python tests/test_ablation.py   |   or: pytest tests/test_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_fusion.config import ModelSpec
from open_fusion.eval import load_tasks
from open_fusion.eval.ablation import (build_canonical_arms, paired_bootstrap,
                                       render_ablation_markdown, run_ablation_demo)

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    ok = bool(cond)
    PASS += ok
    FAIL += (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")


def test_canonical_arms():
    print("ablation: canonical arm construction")
    arms = build_canonical_arms([ModelSpec("a/x"), ModelSpec("b/y"), ModelSpec("c/z")],
                                ModelSpec("a/x"))
    names = [a.name for a in arms]
    check("five canonical arms", names == ["solo", "homo-freeform", "homo-structured",
                                           "hetero-freeform", "hetero-structured"])
    homo = next(a for a in arms if a.name == "homo-structured")
    check("homogeneous arm repeats m0", [m.slug for m in homo.panel] == ["a/x", "a/x"])
    hetero = next(a for a in arms if a.name == "hetero-structured")
    check("heterogeneous arm uses distinct models", len({m.slug for m in hetero.panel}) >= 2)
    check("solo arm has one model", len(next(a for a in arms if a.name == "solo").panel) == 1)
    # kinds map to the three real code paths
    check("kinds cover solo/freeform/structured",
          {a.kind for a in arms} == {"solo", "freeform", "structured"})


def test_bootstrap_determinism_and_ci():
    print("ablation: paired bootstrap is deterministic + CI sane")
    a = [80.0, 82.0, 78.0, 90.0, 85.0]
    b = [70.0, 72.0, 71.0, 69.0, 73.0]   # a is uniformly ~+11 higher
    m1 = paired_bootstrap(a, b, iters=1000, seed=7)
    m2 = paired_bootstrap(a, b, iters=1000, seed=7)
    check("same seed -> identical result", m1 == m2)
    mean_lift, lo, hi, p = m1
    check("mean lift positive (~11)", 9 < mean_lift < 13)
    check("CI ordered lo<=mean<=hi", lo <= mean_lift <= hi)
    check("clear separation -> CI excludes 0 (significant)", lo > 0)
    check("clear separation -> small p", p < 0.1)
    # a zero-effect pair should NOT be significant
    z = paired_bootstrap([50.0, 60.0, 40.0, 55.0], [50.0, 60.0, 40.0, 55.0],
                         iters=1000, seed=1)
    check("identical samples -> zero lift", abs(z[0]) < 1e-9)
    check("identical samples -> p == 1.0", z[3] == 1.0)


def test_bootstrap_edge_cases():
    print("ablation: bootstrap edge cases")
    one = paired_bootstrap([5.0], [3.0], iters=500, seed=0)
    check("n=1 returns the single diff with degenerate CI", one[0] == 2.0 and one[1] == 2.0)
    raised = False
    try:
        paired_bootstrap([1.0, 2.0], [1.0])
    except ValueError:
        raised = True
    check("unequal lengths rejected", raised)


def test_demo_end_to_end():
    print("ablation: end-to-end demo runs every arm offline")
    report = run_ablation_demo(load_tasks("fusion"), seed=0)
    arms = report["arms"]
    check("all five arms produced scores", set(arms) == {
        "solo", "homo-freeform", "homo-structured", "hetero-freeform", "hetero-structured"})
    check("every arm has one score per task",
          all(len(a["per_task"]) == report["config"]["n_tasks"] for a in arms.values()))
    check("scores are within 0..100",
          all(0 <= x <= 100 for a in arms.values() for x in a["per_task"]))


def test_demo_contrasts_present():
    print("ablation: lift decomposition present + well-formed")
    report = run_ablation_demo(load_tasks("draco"), seed=0)
    ct = report["contrasts"]
    check("four canonical contrasts", set(ct) == {
        "total_fusion_lift", "synthesis_lift", "diversity_lift", "structured_judge_lift"})
    for name, c in ct.items():
        check(f"{name}: CI is ordered", c["ci95"][0] <= c["ci95"][1])
        check(f"{name}: verdict is one of supported/null/contradicted",
              c["verdict"] in ("supported", "null", "contradicted"))
        check(f"{name}: p in [0,1]", 0.0 <= c["p_two_sided"] <= 1.0)
    check("interpretation lines present", len(report["interpretation"]) >= 4)


def test_demo_determinism_and_render():
    print("ablation: deterministic report + markdown renders")
    r1 = run_ablation_demo(load_tasks("fusion"), seed=3)
    r2 = run_ablation_demo(load_tasks("fusion"), seed=3)
    check("same seed -> identical contrasts", r1["contrasts"] == r2["contrasts"])
    md = render_ablation_markdown(r1)
    check("markdown has the decomposition table", "Lift decomposition" in md)
    check("markdown names each contrast", "synthesis_lift" in md and "diversity_lift" in md)
    check("markdown lists the five arms", "hetero-structured" in md and "solo" in md)


def main():
    test_canonical_arms()
    test_bootstrap_determinism_and_ci()
    test_bootstrap_edge_cases()
    test_demo_end_to_end()
    test_demo_contrasts_present()
    test_demo_determinism_and_render()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
