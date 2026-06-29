"""
open_fusion.eval.harness - run the DRACO-style comparison.

For every task we run BOTH the Fusion pipeline and a single-model baseline with the
SAME tools available (the article's fair-comparison rule: configs differ only in
whether multiple models' outputs are synthesised, not in tooling). Each response is
graded N independent times (default 3) and the normalized scores are averaged. We
aggregate per category, per domain, and overall, then compute the score lift, the
cost multiplier, and a verdict.

Contamination guard: each task's `excluded_domains` are passed through to the panel's
web tools, exactly as the article excluded the rubric-hosting domains.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any

from .. import panel as panel_mod
from ..client import ModelClient
from ..config import FusionConfig, ModelSpec, Phase
from ..orchestrator import fuse
from ..schema import TokenUsage
from .grader import Grader
from .rubric import Criterion, RubricCategory, Task, majority, score_response


# slug -> (input_$/Mtok, output_$/Mtok). Illustrative; override with your own.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {}


@dataclass
class AvgScore:
    normalized: float
    by_category: dict[RubricCategory, float]
    penalties: list[str]
    passes: list[float] = field(default_factory=list)
    met_passes: list[dict] = field(default_factory=list)   # raw per-pass met-maps (for pass^k)


async def _grade_avg(grader: Grader, task: Task, response: str, n: int) -> AvgScore:
    # 性能优化 O4：DRACO 的 N 次独立判分（默认 n=3）天然可并行。
    # 用 asyncio.gather 同时启动所有 grader 调用；met_passes 顺序与传参一致，
    # 后续 scoring/penalty 聚合都对顺序不敏感（只是求和与多数投票）。
    n = max(1, n)
    met_passes: list[dict] = list(await asyncio.gather(
        *(grader.grade(task, response) for _ in range(n))))
    cat_sums: dict[RubricCategory, float] = {}
    penalty_hits: dict[str, int] = {}
    norms: list[float] = []
    for met in met_passes:
        gr = score_response(task, met)
        norms.append(gr.normalized)
        for cat, cs in gr.categories.items():
            cat_sums[cat] = cat_sums.get(cat, 0.0) + cs.normalized
        for pid in gr.penalties_triggered:
            penalty_hits[pid] = penalty_hits.get(pid, 0) + 1
    by_cat = {cat: s / n for cat, s in cat_sums.items()}
    maj = majority(n)
    penalties = [pid for pid, h in penalty_hits.items() if h >= maj]
    return AvgScore(normalized=sum(norms) / n, by_category=by_cat,
                    penalties=penalties, passes=norms, met_passes=met_passes)


def _slug_from_label(label: str) -> str:
    """Extract model slug from telemetry label ('panel:openai/gpt-4o' -> 'openai/gpt-4o')."""
    if ":" in label:
        return label.split(":", 1)[1]
    return label


def _cost_from_usage(pin: int, pout: int, slug: str, prices: dict) -> float | None:
    price = prices.get(slug)
    if not price:
        return None
    return (pin / 1e6 * price[0] + pout / 1e6 * price[1])


def _cost(usage_tokens: tuple[int, int], slug: str, prices: dict,
          by_label: dict[str, dict[str, int]] | None = None) -> dict[str, Any]:
    """Calculate cost in USD. If by_label is provided, sum per-model costs for accuracy;
    otherwise fall back to single-slug pricing (for baselines)."""
    pin, pout = usage_tokens
    tok = pin + pout

    usd = None
    if by_label:
        total_usd = 0.0
        any_priced = False
        for label, slot in by_label.items():
            model_slug = _slug_from_label(label)
            model_usd = _cost_from_usage(slot["prompt_tokens"], slot["completion_tokens"],
                                         model_slug, prices)
            if model_usd is not None:
                total_usd += model_usd
                any_priced = True
        usd = total_usd if any_priced else None
    else:
        usd = _cost_from_usage(pin, pout, slug, prices)

    return {"tokens": tok, "usd": usd}


async def _run_baseline(client: ModelClient, model: ModelSpec, task: Task,
                        cfg: FusionConfig) -> tuple[str, TokenUsage, int]:
    """Solo baseline: one model, same agent loop & tools as a panel member."""
    excluded = list(set(cfg.excluded_domains) | set(task.excluded_domains))
    t0 = time.monotonic()
    resp = await panel_mod._run_one(client, model, task.prompt,
                                    cfg.params[Phase.PANEL],
                                    cfg.tools_enabled, excluded)
    return resp.content, resp.usage, int((time.monotonic() - t0) * 1000)


async def evaluate(
    tasks: list[Task],
    fusion_config: FusionConfig,
    baseline: ModelSpec,
    grader: Grader,
    *,
    client: ModelClient | None = None,
    n_passes: int = 3,
    prices: dict[str, tuple[float, float]] | None = None,
    label: str = "fusion",
) -> dict[str, Any]:
    """Run the full comparison and return a structured report dict."""
    prices = prices or DEFAULT_PRICES
    if client is None:
        # Performance: for panels larger than 3 models, use a dedicated thread pool
        # sized 2*panel so HTTP calls don't contend on asyncio's default executor.
        panel_size = len(fusion_config.panel)
        executor_workers = max(panel_size * 2, 8) if panel_size >= 4 else None
        client = ModelClient(fusion_depth=fusion_config.depth + 1,
                             base_url=fusion_config.base_url, api_key=fusion_config.api_key,
                             executor_workers=executor_workers)

    per_task: list[dict[str, Any]] = []
    fusion_lh: list = []   # long-horizon per-task reliability scores
    base_lh: list = []
    for task in tasks:
        # contamination guard: merge task-level exclusions into the run config
        run_cfg = FusionConfig(
            panel=fusion_config.panel, judge=fusion_config.judge,
            caller=fusion_config.caller, params=fusion_config.params,
            depth=fusion_config.depth, tools_enabled=fusion_config.tools_enabled,
            max_in_flight=fusion_config.max_in_flight,
            excluded_domains=list(set(fusion_config.excluded_domains) | set(task.excluded_domains)),
            fast_majority_k=fusion_config.fast_majority_k,
            base_url=fusion_config.base_url, api_key=fusion_config.api_key,
        )

        t0 = time.monotonic()
        fr = await fuse(task.prompt, run_cfg, client=client)
        fusion_latency = int((time.monotonic() - t0) * 1000)
        fusion_text = fr.text or ""

        base_text, base_usage, base_latency = await _run_baseline(client, baseline, task, run_cfg)

        fusion_score = await _grade_avg(grader, task, fusion_text, n_passes)
        base_score = await _grade_avg(grader, task, base_text, n_passes)

        lh_task = None
        if task.is_longhorizon:
            from .longhorizon import score_passes
            lh_f = score_passes(task, fusion_score.met_passes)
            lh_b = score_passes(task, base_score.met_passes)
            fusion_lh.append(lh_f)
            base_lh.append(lh_b)
            lh_task = {
                "horizon": task.horizon,
                "fusion_milestones": round(lh_f.milestone_completion, 2),
                "baseline_milestones": round(lh_b.milestone_completion, 2),
                "fusion_pass_k": lh_f.pass_k, "baseline_pass_k": lh_b.pass_k,
                "fusion_drift": lh_f.drift_errors, "baseline_drift": lh_b.drift_errors,
            }

        tel = fr.telemetry or {}
        fusion_cost = _cost((tel.get("prompt_tokens", 0), tel.get("completion_tokens", 0)),
                            fusion_config.caller.slug, prices,
                            by_label=tel.get("by_label"))
        base_cost = _cost((base_usage.prompt_tokens, base_usage.completion_tokens),
                          baseline.slug, prices)

        # 延迟口径：wall-clock 是端到端公平对比；拆解（panel-max / judge / synth）
        # 暴露 Fusion 的延迟构成，让用户能判断慢在哪一段而不是只看一个倍数。
        fusion_panel_max_ms = max(tel.get("panel_latencies_ms") or [0]) if tel.get("panel_latencies_ms") else 0
        fusion_breakdown = {
            "panel_max_ms": fusion_panel_max_ms,
            "judge_ms": tel.get("judge_ms", 0),
            "synthesis_ms": tel.get("synthesis_ms", 0),
            "critical_path_ms": tel.get("critical_path_ms", 0),
        }

        per_task.append({
            "task_id": task.id, "domain": task.domain,
            "fusion_score": round(fusion_score.normalized, 2),
            "baseline_score": round(base_score.normalized, 2),
            "lift": round(fusion_score.normalized - base_score.normalized, 2),
            "fusion_by_category": {c.value: round(v, 2) for c, v in fusion_score.by_category.items()},
            "baseline_by_category": {c.value: round(v, 2) for c, v in base_score.by_category.items()},
            "fusion_penalties": fusion_score.penalties,
            "baseline_penalties": base_score.penalties,
            "fusion_cost": fusion_cost, "baseline_cost": base_cost,
            "fusion_latency_ms": fusion_latency, "baseline_latency_ms": base_latency,
            "fusion_latency_breakdown": fusion_breakdown,
            "fusion_status": fr.status.value if hasattr(fr.status, "value") else fr.status,
            "fusion_passes": [round(x, 2) for x in fusion_score.passes],
            "baseline_passes": [round(x, 2) for x in base_score.passes],
            "long_horizon": lh_task,
        })

    return _aggregate(per_task, tasks, fusion_config, baseline, grader, n_passes,
                      prices, label, fusion_lh, base_lh)


def _aggregate(per_task, tasks, fusion_config, baseline, grader, n_passes, prices, label,
               fusion_lh=None, base_lh=None) -> dict[str, Any]:
    n = len(per_task) or 1
    mean_fusion = sum(t["fusion_score"] for t in per_task) / n
    mean_baseline = sum(t["baseline_score"] for t in per_task) / n
    lift = mean_fusion - mean_baseline

    # per-category means
    cats = [c.value for c in RubricCategory]
    by_cat = {}
    for c in cats:
        fv = [t["fusion_by_category"].get(c) for t in per_task if c in t["fusion_by_category"]]
        bv = [t["baseline_by_category"].get(c) for t in per_task if c in t["baseline_by_category"]]
        if fv or bv:
            by_cat[c] = {
                "fusion": round(sum(fv) / len(fv), 2) if fv else None,
                "baseline": round(sum(bv) / len(bv), 2) if bv else None,
            }

    # per-domain means
    by_domain: dict[str, dict] = {}
    for t in per_task:
        d = by_domain.setdefault(t["domain"], {"fusion": [], "baseline": []})
        d["fusion"].append(t["fusion_score"])
        d["baseline"].append(t["baseline_score"])
    by_domain = {d: {"fusion": round(sum(v["fusion"]) / len(v["fusion"]), 2),
                     "baseline": round(sum(v["baseline"]) / len(v["baseline"]), 2),
                     "n": len(v["fusion"])} for d, v in by_domain.items()}

    fusion_tokens = sum(t["fusion_cost"]["tokens"] for t in per_task)
    base_tokens = sum(t["baseline_cost"]["tokens"] for t in per_task)
    cost_multiplier = (fusion_tokens / base_tokens) if base_tokens else None

    fusion_usd = sum((t["fusion_cost"]["usd"] or 0) for t in per_task) if prices else 0
    base_usd = sum((t["baseline_cost"]["usd"] or 0) for t in per_task) if prices else 0
    usd_available = bool(prices) and base_usd > 0

    fusion_lat = sum(t["fusion_latency_ms"] for t in per_task) / n
    base_lat = sum(t["baseline_latency_ms"] for t in per_task) / n
    latency_multiplier = (fusion_lat / base_lat) if base_lat else None

    # 延迟拆解：让"为什么慢"可读。三段相加约等于 wall-clock critical_path；
    # 它们的均值有助于决策（如果 judge_ms 是大头，换一个更快的 judge 比换 panel
    # 更划算）。
    breakdowns = [t.get("fusion_latency_breakdown") or {} for t in per_task]
    def _avg(field: str) -> int:
        vals = [int(b.get(field, 0) or 0) for b in breakdowns]
        return round(sum(vals) / n) if vals else 0
    fusion_breakdown_avg = {
        "panel_max_ms": _avg("panel_max_ms"),
        "judge_ms": _avg("judge_ms"),
        "synthesis_ms": _avg("synthesis_ms"),
        "critical_path_ms": _avg("critical_path_ms"),
    }

    fusion_errors = sum(len(t["fusion_penalties"]) for t in per_task)
    base_errors = sum(len(t["baseline_penalties"]) for t in per_task)

    from .report import build_verdict
    from .longhorizon import aggregate as lh_aggregate
    lh_block = None
    if fusion_lh:
        lh_block = {"fusion": lh_aggregate(fusion_lh), "baseline": lh_aggregate(base_lh or [])}
    verdict = build_verdict(mean_fusion, mean_baseline, lift, cost_multiplier,
                            latency_multiplier, fusion_errors, base_errors, label,
                            long_horizon=lh_block)

    return {
        "config": {
            "label": label,
            "panel": [m.slug for m in fusion_config.panel],
            "judge": fusion_config.judge.slug,
            "caller": fusion_config.caller.slug,
            "baseline": baseline.slug,
            "grader": getattr(grader, "name", grader.__class__.__name__),
            "n_passes": n_passes,
            "n_tasks": len(per_task),
            "tools_enabled": fusion_config.tools_enabled,
        },
        "aggregate": {
            "mean_fusion": round(mean_fusion, 2),
            "mean_baseline": round(mean_baseline, 2),
            "lift": round(lift, 2),
            "by_category": by_cat,
            "by_domain": by_domain,
            "cost": {"fusion_tokens": fusion_tokens, "baseline_tokens": base_tokens,
                     "cost_multiplier": round(cost_multiplier, 2) if cost_multiplier else None,
                     "fusion_usd": round(fusion_usd, 4) if usd_available else None,
                     "baseline_usd": round(base_usd, 4) if usd_available else None},
            "latency": {"fusion_ms": round(fusion_lat), "baseline_ms": round(base_lat),
                        "latency_multiplier": round(latency_multiplier, 2) if latency_multiplier else None,
                        "fusion_breakdown": fusion_breakdown_avg},
            "errors": {"fusion": fusion_errors, "baseline": base_errors},
            "quality_per_dollar": round((mean_fusion / mean_baseline) / cost_multiplier, 3)
                                  if cost_multiplier and mean_baseline else None,
            "long_horizon": lh_block,
        },
        "verdict": verdict,
        "per_task": per_task,
    }
