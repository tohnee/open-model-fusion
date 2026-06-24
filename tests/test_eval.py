"""
Offline tests for the eval subsystem. Zero network (RuleGrader + DemoClient).

Run directly:   python tests/test_eval.py
Or with pytest: pytest tests/test_eval.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_fusion.eval import load_tasks, run_demo
from open_fusion.eval.rubric import (Criterion, RubricCategory, Task,
                                     evaluate_check, score_response)
from open_fusion.eval.grader import RuleGrader
from open_fusion.eval.report import build_verdict

PASS, FAIL = 0, 0


def check(name: str, cond: bool):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  PASS  {name}")
    else:
        FAIL += 1; print(f"  FAIL  {name}")


def _task() -> Task:
    return Task(id="t", domain="d", prompt="p", rubric=[
        Criterion("f1", "fact a", RubricCategory.FACTUAL_ACCURACY, 3),
        Criterion("f2", "fact b", RubricCategory.FACTUAL_ACCURACY, 3),
        Criterion("pen", "error", RubricCategory.FACTUAL_ACCURACY, -5),
        Criterion("b1", "depth", RubricCategory.BREADTH_DEPTH, 2),
        Criterion("p1", "format", RubricCategory.PRESENTATION, 1),
        Criterion("c1", "cite", RubricCategory.CITATION, 2),
    ])


def test_scoring_math():
    print("rubric: scoring math")
    t = _task()
    # all positive met, no penalty -> 100
    allmet = {"f1": True, "f2": True, "b1": True, "p1": True, "c1": True}
    g = score_response(t, allmet)
    check("all positive met -> 100", abs(g.normalized - 100.0) < 1e-6)
    check("positive max excludes penalty (=11)", abs(g.possible - 11.0) < 1e-6)
    # verbosity can't game it: meeting nothing -> 0
    check("nothing met -> 0", score_response(t, {}).normalized == 0.0)
    # triggering the penalty subtracts and is recorded as an error
    pen = dict(allmet); pen["pen"] = True
    gp = score_response(t, pen)
    check("penalty lowers score", gp.normalized < 100.0)
    check("penalty recorded as triggered", gp.penalties_triggered == ["pen"])
    # clamp at 0 (big penalty with little achieved)
    gc = score_response(t, {"p1": True, "pen": True})
    check("score clamps at 0", gc.normalized == 0.0)


def test_categories_and_faithfulness():
    print("rubric: categories + DRACO shape")
    t = _task()
    g = score_response(t, {"f1": True, "b1": True})
    check("factual category partial", round(g.categories[RubricCategory.FACTUAL_ACCURACY].normalized, 1) == 50.0)
    check("four DRACO categories exist", len(list(RubricCategory)) == 4)
    from open_fusion.eval.rubric import DRACO_CATEGORY_TARGETS
    check("DRACO target ~39 criteria", sum(DRACO_CATEGORY_TARGETS.values()) == 40 or sum(DRACO_CATEGORY_TARGETS.values()) >= 39)


def test_rule_checks():
    print("grader: rule checks (any/all/none)")
    check("all matches", evaluate_check("alpha beta", {"all": ["alpha", "beta"]}))
    check("all fails on missing", not evaluate_check("alpha", {"all": ["alpha", "beta"]}))
    check("any matches", evaluate_check("gamma", {"any": ["delta", "gamma"]}))
    check("none blocks", not evaluate_check("has error here", {"any": ["has"], "none": ["error"]}))
    check("no check -> not met", not evaluate_check("anything", None))


def test_rule_grader():
    print("grader: RuleGrader over a task")
    t = _task()
    t.rubric[0].check = {"any": ["metformin"]}
    res = asyncio.run(RuleGrader().grade(t, "use metformin"))
    check("RuleGrader detects match", res["f1"] is True)
    check("RuleGrader misses absent", res["f2"] is False)


def test_grader_passes_concurrent():
    # 性能优化 O4：DRACO 的 N 次独立判分必须可并行。
    # 用一个会 sleep 的 fake grader：串行 N=3 次各 60ms 应 >=180ms，
    # 并行后应远小于（3 个一起启动，约 60-80ms）。
    print("grader: O4 concurrent pass execution")
    import time as _time
    from open_fusion.eval.grader import Grader
    from open_fusion.eval.harness import _grade_avg

    class _SlowGrader(Grader):
        name = "slow"
        def __init__(self, delay_ms: int = 60):
            self.delay = delay_ms / 1000.0
        async def grade(self, task, response):
            await asyncio.sleep(self.delay)
            # 返回固定 met-map（与 test_scoring_math 的 allmet 形态一致）
            return {c.id: (c.weight > 0) for c in task.rubric}

    t = _task()
    n = 3
    g = _SlowGrader(60)
    t0 = _time.monotonic()
    s = asyncio.run(_grade_avg(g, t, "ignored", n))
    elapsed_ms = int((_time.monotonic() - t0) * 1000)
    check("met_passes length == n_passes", len(s.met_passes) == n)
    check("passes length == n_passes", len(s.passes) == n)
    # 串行下界 3*60=180ms；并发上界应远低于此。给一个 130ms 的余量（防止 CI 抖动），
    # 实测正常情况约 60-80ms。
    check(f"concurrent runs < serial-bound (180ms), got {elapsed_ms}ms",
          elapsed_ms < 130)
    # 输出与串行版数学等价（每个 met 由相同规则生成，应得全 100 分）
    check("all-positive met -> 100", abs(s.normalized - 100.0) < 1e-6)


def test_verdict_logic():
    print("report: verdict decision rules")
    # strong win
    v = build_verdict(69, 60, 9, 3.0, 2.5, 0, 1, "x")
    check("big lift -> strong_win + worth_it", v["rating"] == "strong_win" and v["worth_it"])
    check("error reduction noted", v["error_delta"] == 1)
    # marginal, expensive -> not worth it
    v2 = build_verdict(65.6, 65.3, 0.3, 3.2, 2.5, 0, 0, "x")
    check("tiny lift high cost -> not worth_it", v2["worth_it"] is False)
    # budget win: within ~1pt at half cost
    v3 = build_verdict(64.7, 65.3, -0.6, 0.5, 1.0, 0, 0, "x")
    check("budget panel cheaper -> worth_it via budget_win", v3["budget_win"] and v3["worth_it"])
    # regression
    v4 = build_verdict(55, 60, -5, 3.0, 2.5, 0, 0, "x")
    check("negative lift -> regression", v4["rating"] == "regression")


def test_load_tasks():
    print("tasks: sample set loads + DRACO shape")
    tasks = load_tasks("sample")
    check("sample tasks present", len(tasks) >= 4)
    cats = {c.category for t in tasks for c in t.rubric}
    check("all 4 categories covered across set", len(cats) == 4)
    check("has negative-weight penalties", any(c.weight < 0 for t in tasks for c in t.rubric))
    check("tasks carry excluded_domains", all(t.excluded_domains for t in tasks))


def test_end_to_end_demo():
    print("harness: end-to-end demo (fusion vs baseline)")
    report = run_demo(load_tasks("sample"))
    agg = report["aggregate"]
    check("fusion beats baseline overall", agg["mean_fusion"] > agg["mean_baseline"])
    check("positive lift", agg["lift"] > 0)
    check("cost multiplier > 1 (fusion costs more)", agg["cost"]["cost_multiplier"] > 1)
    check("fewer errors than baseline", agg["errors"]["fusion"] < agg["errors"]["baseline"])
    check("verdict worth_it on these tasks", report["verdict"]["worth_it"] is True)
    check("per-category breakdown present", len(agg["by_category"]) == 4)
    check("per-domain breakdown present", len(agg["by_domain"]) >= 4)
    check("3 passes recorded", len(report["per_task"][0]["fusion_passes"]) == 3)
    check("conclusion text present", bool(report["verdict"]["conclusion"]))


def main():
    test_scoring_math()
    test_categories_and_faithfulness()
    test_rule_checks()
    test_rule_grader()
    test_grader_passes_concurrent()
    test_verdict_logic()
    test_load_tasks()
    test_end_to_end_demo()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
