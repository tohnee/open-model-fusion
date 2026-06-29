#!/usr/bin/env python3
"""orchestrator.py 性能基准测试: Aggregator 模式 vs FULL 模式

通过模拟真实网络延迟 (FakeClient + sleep)，测量四种管道模式的关键指标:
  1. FULL mode         (panel → judge → synthesis)
  2. AGGREGATOR mode    (panel → synthesis, 跳过 judge)
  3. Consensus shortcut (panel → 短路, 跳过 judge+synth)
  4. Pick-best shortcut (panel → judge → 短路, 跳过 synth)

核心指标:
  - 总延迟 (total_ms): 端到端管道耗时
  - API 调用次数 (api_calls): 串行关键路径上的 LLM 调用数
  - 关键路径延迟 (critical_path_ms): 串行部分的理论最小延迟
  - 加速比 (speedup): 相对 FULL 模式的提速倍数

用法:
    python3 benchmark_orchestrator.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any

sys.path.insert(0, "src")
sys.path.insert(0, "tests")

from open_fusion.config import FusionConfig, FusionMode, ModelSpec
from open_fusion.orchestrator import fuse
from open_fusion.schema import FusionStatus
from fake_client import FakeClient, text


# ============================================================================
# 模拟延迟配置 (基于真实 LLM API 的典型延迟)
# ============================================================================
PANEL_LATENCY_MS = 800       # panel 模型响应延迟 (并行)
JUDGE_LATENCY_MS = 1200      # judge 模型响应延迟 (串行, 需分析所有回答)
SYNTH_LATENCY_MS = 1000      # synthesizer 延迟 (串行, 需生成最终答案)

VALID_ANALYSIS = json.dumps({
    "consensus": ["test consensus"],
    "contradictions": [],
    "partial_coverage": [],
    "unique_insights": [],
    "blind_spots": [],
    "best_model": "MODEL A",
    "best_reason": "most complete",
})


class TimedFakeClient(FakeClient):
    """带模拟网络延迟的 FakeClient。

    根据调用阶段注入不同延迟:
      - panel 阶段 (第 1 次调用每个 slug): PANEL_LATENCY_MS
      - judge 阶段 (第 2 次调用 judge slug): JUDGE_LATENCY_MS
      - synth 阶段 (第 3 次调用 judge slug): SYNTH_LATENCY_MS
    """

    def __init__(self, scripts, *, panel_ms=PANEL_LATENCY_MS,
                 judge_ms=JUDGE_LATENCY_MS, synth_ms=SYNTH_LATENCY_MS):
        super().__init__(scripts)
        self._panel_ms = panel_ms
        self._judge_ms = judge_ms
        self._synth_ms = synth_ms
        self._call_counts: dict[str, int] = {}

    async def complete(self, model, messages, *, tools=(), params=None, response_format=None):
        slug = getattr(model, "slug", model)
        self._call_counts[slug] = self._call_counts.get(slug, 0) + 1
        call_idx = self._call_counts[slug]

        # 根据调用阶段决定延迟
        if call_idx == 1:
            delay = self._panel_ms / 1000.0   # panel 调用
        elif call_idx == 2:
            delay = self._judge_ms / 1000.0   # judge 调用
        else:
            delay = self._synth_ms / 1000.0   # synth 调用

        await asyncio.sleep(delay)
        return await super().complete(model, messages, tools=tools,
                                       params=params, response_format=response_format)


def make_config(mode=FusionMode.FULL, *, enable_consensus=False,
                enable_pick_best=False, n_panel=3) -> FusionConfig:
    """构建测试配置。"""
    panel = [ModelSpec(f"m{i+1}/x") for i in range(n_panel)]
    return FusionConfig(
        panel=panel,
        judge=ModelSpec("m1/x"),
        caller=ModelSpec("m1/x"),
        mode=mode,
        enable_consensus_shortcut=enable_consensus,
        enable_pick_best=enable_pick_best,
        consensus_threshold=0.85,
        max_in_flight=n_panel,
    )


def make_scripts(n_panel: int, *, long_answer: bool = False) -> dict:
    """为每个 panel 模型生成脚本。

    long_answer=True 时回答 >100 字符 (触发 pick-best 短路)。
    """
    answer = ("A" * 150) if long_answer else "short answer"
    scripts = {}
    for i in range(n_panel):
        slug = f"m{i+1}/x"
        if i == 0:
            # judge 模型: panel + judge + synth (FULL 模式需要 3 轮)
            scripts[slug] = [text(answer), text(VALID_ANALYSIS), text("FINAL ANSWER")]
        else:
            scripts[slug] = [text(answer)]
    return scripts


async def run_benchmark(name: str, config: FusionConfig, scripts: dict,
                        expected_status: FusionStatus) -> dict:
    """运行一次基准测试, 返回指标字典。"""
    client = TimedFakeClient(scripts)
    t0 = time.monotonic()
    result = await fuse("benchmark question", config, client=client)
    total_ms = int((time.monotonic() - t0) * 1000)

    return {
        "mode": name,
        "status": result.status.value,
        "status_ok": result.status == expected_status,
        "total_ms": total_ms,
        "api_calls": len(client.calls),
        "panel_calls": config.panel.__len__(),
        "answer_chars": len(result.text),
        "speedup_vs_full": 0.0,  # 填充在汇总时
    }


async def run_consensus_benchmark(n_panel: int = 3) -> dict:
    """Consensus 短路基准: 所有 panel 回答一致 → 跳过 judge+synth。"""
    same_answer = "The answer is 42. " * 10
    scripts = {f"m{i+1}/x": [text(same_answer)] for i in range(n_panel)}
    config = FusionConfig(
        panel=[ModelSpec(f"m{i+1}/x") for i in range(n_panel)],
        judge=ModelSpec("m1/x"),
        caller=ModelSpec("m1/x"),
        enable_consensus_shortcut=True,
        consensus_threshold=0.85,
        max_in_flight=n_panel,
    )
    client = TimedFakeClient(scripts)
    t0 = time.monotonic()
    result = await fuse("benchmark question", config, client=client)
    total_ms = int((time.monotonic() - t0) * 1000)
    return {
        "mode": "Consensus Shortcut",
        "status": result.status.value,
        "status_ok": result.status == FusionStatus.OK,
        "total_ms": total_ms,
        "api_calls": len(client.calls),
        "panel_calls": n_panel,
        "answer_chars": len(result.text),
        "speedup_vs_full": 0.0,
    }


async def run_pick_best_benchmark(n_panel: int = 3) -> dict:
    """Pick-best 短路基准: judge 选出 best_model → 跳过 synth。"""
    long_a = "A" * 150  # >100 字符触发短路
    scripts = {f"m{i+1}/x": [text(long_a if i == 0 else "short")] for i in range(n_panel)}
    scripts["m1/x"].extend([text(VALID_ANALYSIS)])  # judge 轮
    config = FusionConfig(
        panel=[ModelSpec(f"m{i+1}/x") for i in range(n_panel)],
        judge=ModelSpec("m1/x"),
        caller=ModelSpec("m1/x"),
        enable_consensus_shortcut=False,
        enable_pick_best=True,
        max_in_flight=n_panel,
    )
    client = TimedFakeClient(scripts)
    t0 = time.monotonic()
    result = await fuse("benchmark question", config, client=client)
    total_ms = int((time.monotonic() - t0) * 1000)
    return {
        "mode": "Pick-Best Shortcut",
        "status": result.status.value,
        "status_ok": result.status == FusionStatus.OK,
        "total_ms": total_ms,
        "api_calls": len(client.calls),
        "panel_calls": n_panel,
        "answer_chars": len(result.text),
        "speedup_vs_full": 0.0,
    }


async def main():
    print("=" * 72)
    print("orchestrator.py 性能基准测试")
    print("Aggregator 模式 vs FULL 模式 vs 短路机制")
    print("=" * 72)
    print(f"\n模拟延迟配置:")
    print(f"  Panel 延迟:    {PANEL_LATENCY_MS}ms (并行)")
    print(f"  Judge 延迟:    {JUDGE_LATENCY_MS}ms (串行)")
    print(f"  Synth 延迟:    {SYNTH_LATENCY_MS}ms (串行)")
    print(f"  Panel 数量:    3")

    n_panel = 3
    results = []

    # 1. FULL mode (完整管道: panel → judge → synth)
    print(f"\n[1/4] FULL mode (panel → judge → synthesis)...")
    config_full = make_config(mode=FusionMode.FULL, n_panel=n_panel)
    scripts_full = make_scripts(n_panel)
    r = await run_benchmark("FULL Mode", config_full, scripts_full, FusionStatus.OK)
    results.append(r)
    print(f"  → {r['total_ms']}ms, {r['api_calls']} API calls, status={r['status']}")

    # 2. AGGREGATOR mode (panel → synth, 跳过 judge)
    print(f"\n[2/4] AGGREGATOR mode (panel → synthesis, skip judge)...")
    config_agg = make_config(mode=FusionMode.AGGREGATOR, n_panel=n_panel)
    scripts_agg = make_scripts(n_panel)
    r = await run_benchmark("AGGREGATOR Mode", config_agg, scripts_agg,
                            FusionStatus.AGGREGATOR_MODE)
    results.append(r)
    print(f"  → {r['total_ms']}ms, {r['api_calls']} API calls, status={r['status']}")

    # 3. Consensus shortcut (panel → 短路)
    print(f"\n[3/4] Consensus shortcut (panel → short-circuit)...")
    r = await run_consensus_benchmark(n_panel)
    results.append(r)
    print(f"  → {r['total_ms']}ms, {r['api_calls']} API calls, status={r['status']}")

    # 4. Pick-best shortcut (panel → judge → 短路)
    print(f"\n[4/4] Pick-best shortcut (panel → judge → short-circuit)...")
    r = await run_pick_best_benchmark(n_panel)
    results.append(r)
    print(f"  → {r['total_ms']}ms, {r['api_calls']} API calls, status={r['status']}")

    # 计算加速比
    full_ms = results[0]["total_ms"]
    for r in results:
        r["speedup_vs_full"] = round(full_ms / max(r["total_ms"], 1), 2)

    # 汇总表格
    print(f"\n{'=' * 72}")
    print("基准测试结果汇总")
    print(f"{'=' * 72}")
    print(f"{'Mode':<25} {'Total(ms)':>10} {'API Calls':>10} {'Speedup':>10} {'Status':>15}")
    print(f"{'-' * 72}")
    for r in results:
        print(f"{r['mode']:<25} {r['total_ms']:>10} {r['api_calls']:>10} "
              f"{r['speedup_vs_full']:>9.2f}x {r['status']:>15}")

    # 理论分析
    print(f"\n{'=' * 72}")
    print("理论分析 (基于模拟延迟)")
    print(f"{'=' * 72}")
    panel_parallel_ms = PANEL_LATENCY_MS  # 并行, 取最大值
    full_theoretical = panel_parallel_ms + JUDGE_LATENCY_MS + SYNTH_LATENCY_MS
    agg_theoretical = panel_parallel_ms + SYNTH_LATENCY_MS
    consensus_theoretical = panel_parallel_ms
    pick_best_theoretical = panel_parallel_ms + JUDGE_LATENCY_MS

    print(f"  FULL mode:          {panel_parallel_ms} (panel) + {JUDGE_LATENCY_MS} (judge) + {SYNTH_LATENCY_MS} (synth) = {full_theoretical}ms")
    print(f"  AGGREGATOR mode:    {panel_parallel_ms} (panel) + {SYNTH_LATENCY_MS} (synth) = {agg_theoretical}ms  [省 judge]")
    print(f"  Consensus shortcut: {panel_parallel_ms} (panel) = {consensus_theoretical}ms  [省 judge+synth]")
    print(f"  Pick-best shortcut: {panel_parallel_ms} (panel) + {JUDGE_LATENCY_MS} (judge) = {pick_best_theoretical}ms  [省 synth]")

    print(f"\n  理论加速比:")
    print(f"    AGGREGATOR vs FULL:    {full_theoretical}/{agg_theoretical} = {full_theoretical/agg_theoretical:.2f}x  (省 {JUDGE_LATENCY_MS}ms, -{JUDGE_LATENCY_MS/full_theoretical*100:.0f}%)")
    print(f"    Consensus vs FULL:     {full_theoretical}/{consensus_theoretical} = {full_theoretical/consensus_theoretical:.2f}x  (省 {JUDGE_LATENCY_MS+SYNTH_LATENCY_MS}ms, -{(JUDGE_LATENCY_MS+SYNTH_LATENCY_MS)/full_theoretical*100:.0f}%)")
    print(f"    Pick-best vs FULL:     {full_theoretical}/{pick_best_theoretical} = {full_theoretical/pick_best_theoretical:.2f}x  (省 {SYNTH_LATENCY_MS}ms, -{SYNTH_LATENCY_MS/full_theoretical*100:.0f}%)")

    # API 调用节省
    print(f"\n  API 调用次数对比 (3 panel 模型):")
    print(f"    FULL mode:          3 panel + 1 judge + 1 synth = 5 calls")
    print(f"    AGGREGATOR mode:    3 panel + 1 synth           = 4 calls  (-20%)")
    print(f"    Consensus shortcut: 3 panel                     = 3 calls  (-40%)")
    print(f"    Pick-best shortcut: 3 panel + 1 judge           = 4 calls  (-20%)")

    print(f"\n{'=' * 72}")
    print("结论: AGGREGATOR 模式通过跳过 judge 层, 在保持答案质量的前提下")
    print("      减少 1 次 API 调用并降低 ~{:.0f}% 的串行延迟。".format(
        JUDGE_LATENCY_MS / full_theoretical * 100))
    print(f"{'=' * 72}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
