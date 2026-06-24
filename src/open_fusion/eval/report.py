"""
open_fusion.eval.report - turn aggregate numbers into a verdict + readable report.

The verdict encodes the article's framing and the community's cost-aware critique:
Fusion's value is the score LIFT over the best single model, weighed against the
additive cost (~3x for a quality panel) and the 2-3x latency. Relative gap is the
trustworthy signal (judge choice moves absolute scores 10-25 pts but not rankings),
so the rules key off lift, cost multiplier, latency, and error reduction.
"""
from __future__ import annotations

import json
from typing import Any


def build_verdict(mean_fusion: float, mean_baseline: float, lift: float,
                  cost_multiplier: float | None, latency_multiplier: float | None,
                  fusion_errors: int, baseline_errors: int, label: str,
                  long_horizon: dict | None = None) -> dict[str, Any]:
    cm = cost_multiplier or 1.0
    lm = latency_multiplier or 1.0
    error_delta = baseline_errors - fusion_errors   # >0 => fusion made fewer errors

    # primary rating from the lift
    if lift >= 5:
        rating, headline = "strong_win", "Beyond-frontier lift"
    elif lift >= 2:
        rating, headline = "win", "Clear, worthwhile lift"
    elif lift >= 0.5:
        rating, headline = "marginal", "Small lift"
    elif lift > -0.5:
        rating, headline = "neutral", "No meaningful difference"
    else:
        rating, headline = "regression", "Fusion underperforms the baseline"

    # cost-efficiency overlay (the budget-panel story)
    budget_win = (lift >= -1.0) and (cm <= 0.7)
    # when the win is purely economic, say so in the headline instead of "underperforms"
    if budget_win and rating in ("marginal", "neutral", "regression"):
        headline = "Budget-panel win (cheaper at comparable quality)"

    recs: list[str] = []
    if rating in ("strong_win", "win"):
        recs.append(f"Worth it for high-stakes / ambiguous work: +{lift:.1f} pts "
                    f"at ~{cm:.1f}x cost and ~{lm:.1f}x latency.")
    if rating == "marginal":
        recs.append(f"Borderline: only +{lift:.1f} pts for ~{cm:.1f}x cost. Use Fusion "
                    "only when an error is expensive; otherwise call the single model.")
    if rating in ("neutral", "regression"):
        recs.append(f"Not worth it on these tasks ({lift:+.1f} pts at ~{cm:.1f}x cost). "
                    "Prefer the single model here.")
    if budget_win:
        recs.append(f"Budget-panel win: matches the baseline within {abs(lift):.1f} pt "
                    f"at ~{cm:.1f}x cost (cheaper). Good default for high-volume work.")
    if error_delta > 0:
        recs.append(f"Safety gain: Fusion triggered {error_delta} fewer penalty/error "
                    "criteria than the baseline — valuable in medical/legal/compliance.")
    elif error_delta < 0:
        recs.append(f"Caution: Fusion triggered {-error_delta} MORE error criteria than "
                    "the baseline; inspect those tasks before trusting it.")

    worth_it = rating in ("strong_win", "win") or budget_win or (error_delta > 0 and lift >= -0.5)

    # long-horizon overlay: reliability (pass^k) is the signal that matters here, not
    # the mean. DRACO has no long-horizon tasks and the article notes Fusion is not
    # built for them, so we report reliability honestly and let it move the verdict.
    lh_summary = None
    if long_horizon and long_horizon.get("fusion"):
        f = long_horizon["fusion"]
        b = long_horizon.get("baseline") or {}
        pk_f, pk_b = f.get("pass_hat_k", 0.0), b.get("pass_hat_k", 0.0)
        ms_f, ms_b = f.get("milestone_completion", 0.0), b.get("milestone_completion", 0.0)
        k = f.get("k", 1)
        rel_delta = pk_f - pk_b
        lh_summary = {"k": k, "fusion_pass_hat_k": pk_f, "baseline_pass_hat_k": pk_b,
                      "fusion_milestones": ms_f, "baseline_milestones": ms_b,
                      "reliability_delta": round(rel_delta, 2)}
        if rel_delta >= 10:
            recs.append(f"Long-horizon: Fusion is more RELIABLE — pass^{k} {pk_f:.0f}% vs "
                        f"{pk_b:.0f}% (milestones {ms_f:.0f}% vs {ms_b:.0f}%). The synthesis "
                        "step reconciles divergent trajectories before they compound.")
        elif rel_delta <= -10:
            recs.append(f"Long-horizon: Fusion is LESS reliable — pass^{k} {pk_f:.0f}% vs "
                        f"{pk_b:.0f}%. A single strong long-horizon model is the safer choice; "
                        "this matches the article's caveat that Fusion targets deep research, "
                        "not long-horizon execution.")
            worth_it = worth_it and (lift >= 2)   # need a real quality reason to accept lower reliability
        else:
            recs.append(f"Long-horizon: reliability is comparable — pass^{k} {pk_f:.0f}% vs "
                        f"{pk_b:.0f}%. Mean milestone completion {ms_f:.0f}% vs {ms_b:.0f}%; "
                        "decide on cost, not reliability.")

    conclusion = (
        f"{headline}. Fusion scored {mean_fusion:.1f} vs the {mean_baseline:.1f} baseline "
        f"({lift:+.1f} pts) at ~{cm:.1f}x cost and ~{lm:.1f}x latency. "
        + (" ".join(recs) if recs else "")
    ).strip()

    return {"rating": rating, "worth_it": bool(worth_it), "headline": headline,
            "lift": round(lift, 2), "cost_multiplier": round(cm, 2),
            "latency_multiplier": round(lm, 2), "error_delta": error_delta,
            "budget_win": budget_win, "long_horizon": lh_summary,
            "recommendations": recs, "conclusion": conclusion}


def render_markdown(report: dict[str, Any]) -> str:
    c = report["config"]
    a = report["aggregate"]
    v = report["verdict"]
    L: list[str] = []
    L.append(f"# Fusion evaluation — {c['label']}")
    L.append("")
    L.append(f"**Verdict: {v['headline']} — {'WORTH IT' if v['worth_it'] else 'NOT WORTH IT'}**  ")
    L.append(f"{v['conclusion']}")
    L.append("")
    L.append("## Setup")
    L.append(f"- Panel: {', '.join(c['panel'])}")
    L.append(f"- Judge / synthesizer (caller): {c['judge']} / {c['caller']}")
    L.append(f"- Solo baseline: {c['baseline']}")
    L.append(f"- Grader: {c['grader']} · passes/task: {c['n_passes']} · tasks: {c['n_tasks']} "
             f"· tools: {c['tools_enabled']}")
    L.append("")
    L.append("## Headline scores (0–100, mean of passes)")
    L.append("| | Fusion | Baseline | Lift |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| Overall | {a['mean_fusion']} | {a['mean_baseline']} | {a['lift']:+} |")
    L.append("")
    L.append("## By rubric category")
    L.append("| Category | Fusion | Baseline |")
    L.append("|---|---:|---:|")
    for cat, vals in a["by_category"].items():
        L.append(f"| {cat} | {vals['fusion']} | {vals['baseline']} |")
    L.append("")
    if a["by_domain"]:
        L.append("## By domain")
        L.append("| Domain | n | Fusion | Baseline |")
        L.append("|---|---:|---:|---:|")
        for d, vals in a["by_domain"].items():
            L.append(f"| {d} | {vals['n']} | {vals['fusion']} | {vals['baseline']} |")
        L.append("")
    L.append("## Cost & latency")
    cost = a["cost"]; lat = a["latency"]; err = a["errors"]
    usd = (f" (${cost['fusion_usd']} vs ${cost['baseline_usd']})"
           if cost.get("fusion_usd") is not None else "")
    cm = f"~{cost['cost_multiplier']}x" if cost.get("cost_multiplier") is not None else "n/a"
    lm = f"~{lat['latency_multiplier']}x" if lat.get("latency_multiplier") is not None else "n/a (demo)"
    qpd = a["quality_per_dollar"] if a["quality_per_dollar"] is not None else "n/a"
    L.append(f"- Cost multiplier: {cm}"
             f" ({cost['fusion_tokens']} vs {cost['baseline_tokens']} tokens){usd}")
    L.append(f"- Latency multiplier: {lm} "
             f"({lat['fusion_ms']}ms vs {lat['baseline_ms']}ms)")
    bd = lat.get("fusion_breakdown") or {}
    if any(bd.get(k) for k in ("panel_max_ms", "judge_ms", "synthesis_ms")):
        # 延迟构成：让"贵在哪一段"显式可见。比例之和约等于 critical_path_ms。
        L.append(f"  - Fusion breakdown: panel-max {bd.get('panel_max_ms', 0)}ms · "
                 f"judge {bd.get('judge_ms', 0)}ms · "
                 f"synthesis {bd.get('synthesis_ms', 0)}ms · "
                 f"critical-path {bd.get('critical_path_ms', 0)}ms")
    L.append(f"- Quality-per-dollar: {qpd} "
             f"(relative score ÷ cost multiplier; >1 favours Fusion)")
    L.append(f"- Error/penalty criteria triggered: Fusion {err['fusion']} vs baseline {err['baseline']}")
    L.append("")
    lh = a.get("long_horizon")
    if lh and lh.get("fusion"):
        f = lh["fusion"]; b = lh.get("baseline") or {}
        L.append("## Long-horizon reliability")
        L.append(f"Metrics follow tau-bench (pass^k = success on ALL k passes) and "
                 f"RoadmapBench-style milestone completion. k = {f.get('k')}.")
        L.append("| Metric | Fusion | Baseline |")
        L.append("|---|---:|---:|")
        L.append(f"| Milestone completion (%) | {f.get('milestone_completion')} | {b.get('milestone_completion')} |")
        L.append(f"| pass^{f.get('k')} (% tasks solved every pass) | {f.get('pass_hat_k')} | {b.get('pass_hat_k')} |")
        L.append(f"| pass@1 (% passes solved) | {f.get('pass_at_1')} | {b.get('pass_at_1')} |")
        L.append(f"| Drift / coherence errors | {f.get('drift_errors')} | {b.get('drift_errors')} |")
        L.append("")
    L.append("## Recommendations")
    for r in v["recommendations"]:
        L.append(f"- {r}")
    return "\n".join(L)


def render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2)
