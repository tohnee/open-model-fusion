"""
Offline tests for the expanded evaluation system (suites + long-horizon + semiconductor).
Run:  python tests/test_eval_suites.py     |  or: pytest tests/test_eval_suites.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_fusion.eval import load_tasks, run_demo
from open_fusion.eval.rubric import RubricCategory, Task, evaluate_check, score_response
from open_fusion.eval.longhorizon import score_passes, aggregate
from open_fusion.eval.report import build_verdict

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    ok = bool(cond)
    PASS += ok
    FAIL += (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")


ARTICLE_DOMAINS = {"academic_research", "finance", "law", "medicine", "technology",
                   "ux_design", "general_knowledge", "needle_retrieval",
                   "personalized_assistance", "product_comparison"}


def test_draco_ten_domains():
    print("suite: draco covers all 10 article domains")
    t = load_tasks("draco")
    check("10 draco tasks", len(t) == 10)
    check("all 10 article domains present", {x.domain for x in t} == ARTICLE_DOMAINS)
    check("all kind=draco", all(x.kind == "draco" for x in t))
    check("every task has positive weight", all(x.positive_weight() > 0 for x in t))
    check("rubrics validate offline", all(evaluate_check("", c.check) in (True, False)
                                          for x in t for c in x.rubric))


def test_longhorizon_suite():
    print("suite: longhorizon shape")
    t = load_tasks("longhorizon")
    check("3 longhorizon tasks", len(t) == 3)
    check("all kind=longhorizon", all(x.is_longhorizon for x in t))
    check("each has critical milestones", all(len(x.critical_criteria()) >= 1 for x in t))
    check("each has a horizon descriptor", all(x.horizon for x in t))


def test_semiconductor_suite():
    print("suite: semiconductor / dram")
    t = load_tasks("semiconductor")
    check("3 semiconductor tasks", len(t) == 3)
    check("all domain=semiconductor", all(x.domain == "semiconductor" for x in t))
    check("mix of draco + longhorizon", {x.kind for x in t} == {"draco", "longhorizon"})
    ramp = next(x for x in t if x.is_longhorizon)
    check("ramp task excludes rubric-host domains", len(ramp.excluded_domains) >= 1)
    blob = " ".join(c.description for x in t for c in x.rubric).lower()
    check("grounded in DRAM facts (hbm4 / sk hynix)", "hbm4" in blob and "sk hynix" in blob)


def test_loaders():
    print("loaders")
    check("'all' = 16 tasks", len(load_tasks("all")) == 16)
    check("'sample' still loads (back-compat)", len(load_tasks("sample")) == 4)


def test_backcompat_defaults():
    print("back-compat: new fields default safely")
    t = Task.from_dict({"id": "x", "domain": "d", "prompt": "p",
                        "rubric": [{"id": "a", "category": "factual_accuracy", "weight": 2}]})
    check("kind defaults to draco", t.kind == "draco" and not t.is_longhorizon)
    check("criterion.critical defaults False", t.rubric[0].critical is False)
    gr = score_response(t, {"a": True})
    check("score_response unchanged for draco", gr.normalized == 100.0)


def test_passhat_k():
    print("longhorizon: pass^k semantics (tau-bench)")
    t = load_tasks("longhorizon")[1]  # lh_policy, has critical milestones + a critical penalty
    crit_ids = [c.id for c in t.critical_criteria()]
    all_met = {c.id: (c.weight > 0) for c in t.rubric}        # meet positives, no penalties
    s_good = score_passes(t, [all_met, all_met, all_met])
    check("all passes meet criticals -> pass^k = 1", s_good.pass_k == 1.0)
    # flip one critical positive to unmet in ONE pass -> pass^k collapses to 0
    one_bad = dict(all_met); one_bad[crit_ids[0]] = False
    s_mixed = score_passes(t, [all_met, one_bad, all_met])
    check("one failed pass -> pass^k = 0", s_mixed.pass_k == 0.0)
    check("pass@1 reflects 2/3 passes", round(s_mixed.pass_at_1, 1) == 66.7)
    # triggering a critical penalty in every pass also fails
    pen = next(c for c in t.rubric if c.weight < 0)
    bad_pen = dict(all_met); bad_pen[pen.id] = True
    s_pen = score_passes(t, [bad_pen, bad_pen, bad_pen])
    check("critical penalty -> pass^k = 0", s_pen.pass_k == 0.0)
    agg = aggregate([s_good, s_mixed])
    check("aggregate pass_hat_k = 50% (1 of 2 tasks reliable)", agg["pass_hat_k"] == 50.0)


def test_verdict_longhorizon_branches():
    print("verdict: long-horizon reliability overlay")
    base = dict(mean_fusion=80, mean_baseline=78, lift=2, cost_multiplier=3.0,
                latency_multiplier=2.5, fusion_errors=0, baseline_errors=0, label="lh")
    less = build_verdict(**base, long_horizon={"fusion": {"pass_hat_k": 60, "milestone_completion": 90, "k": 3},
                                               "baseline": {"pass_hat_k": 90, "milestone_completion": 88, "k": 3}})
    check("less reliable surfaced in verdict", any("LESS reliable" in r for r in less["recommendations"]))
    comp = build_verdict(**base, long_horizon={"fusion": {"pass_hat_k": 67, "milestone_completion": 100, "k": 3},
                                               "baseline": {"pass_hat_k": 67, "milestone_completion": 94, "k": 3}})
    check("comparable reliability surfaced", any("comparable" in r for r in comp["recommendations"]))
    check("verdict carries long_horizon summary", comp["long_horizon"]["reliability_delta"] == 0)


def test_demo_runs_each_suite():
    print("demo: every suite produces a verdict offline")
    for s in ("draco", "longhorizon", "semiconductor"):
        r = run_demo(load_tasks(s))
        check(f"{s}: verdict present", "conclusion" in r["verdict"] and r["verdict"]["conclusion"])
        check(f"{s}: per-task rows == n tasks", len(r["per_task"]) == len(load_tasks(s)))
    rl = run_demo(load_tasks("longhorizon"))
    check("longhorizon: aggregate.long_horizon present", rl["aggregate"]["long_horizon"] is not None)
    check("longhorizon: pass^k reported for both sides",
          "pass_hat_k" in rl["aggregate"]["long_horizon"]["fusion"])


def _penalty_hits(answer, rubric):
    return sum(1 for c in rubric if c.weight < 0 and evaluate_check(answer, c.check))


def test_demo_detector_integrity():
    # Deep-research suites: the fused answer is meant to be penalty-clean and the
    # baseline is meant to trip >=1 error (so error_delta / the safety story is real).
    print("integrity: deep-research fused answers trip no penalties; baselines do")
    for s in ("draco", "semiconductor"):
        tasks = load_tasks(s)
        fused_clean = all(_penalty_hits(t.demo["fused"], t.rubric) == 0 for t in tasks)
        base_hits = sum(_penalty_hits(t.demo["baseline"], t.rubric) for t in tasks)
        check(f"{s}: fused triggers zero penalty criteria", fused_clean)
        check(f"{s}: baselines trigger >=1 penalty (error_delta is real)", base_hits >= 1)

    # Long-horizon suite: reliability is DELIBERATELY symmetric — fusion and the
    # single model each fail a different critical task, so pass^k comes out
    # comparable and the verdict says "decide on cost, not reliability" (honest:
    # DRACO/Fusion targets deep research, not long-horizon execution). So here we
    # assert the asymmetry exists rather than requiring the fused side to be clean.
    print("integrity: long-horizon reliability is intentionally symmetric")
    lh = load_tasks("longhorizon")
    fused_fail_tasks = [t.id for t in lh if _penalty_hits(t.demo["fused"], t.rubric)]
    base_fail_tasks = [t.id for t in lh if _penalty_hits(t.demo["baseline"], t.rubric)]
    check("LH: fused trips a penalty on >=1 task (it is not strictly better)",
          len(fused_fail_tasks) >= 1)
    check("LH: baseline trips a penalty on >=1 task", len(base_fail_tasks) >= 1)
    check("LH: the failing tasks differ (asymmetric, comparable reliability)",
          set(fused_fail_tasks) != set(base_fail_tasks))


def main():
    test_draco_ten_domains()
    test_longhorizon_suite()
    test_semiconductor_suite()
    test_loaders()
    test_backcompat_defaults()
    test_passhat_k()
    test_verdict_longhorizon_branches()
    test_demo_runs_each_suite()
    test_demo_detector_integrity()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
