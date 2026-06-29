#!/usr/bin/env python3
"""Benchmark: 单模型 glm-5.2 vs Fusion(glm-5.1, kimi-k2.6, deepseek-v4-flash)

这组配置的关键差异：
- 单模型对照组是 glm-5.2（不是 judge 模型）
- Fusion panel 使用三个不同的模型，避免 judge 自审问题
- Judge 使用 doubao-seed-2.0-pro（不参与 panel，避免偏见）
"""
import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, "src")

from open_fusion.config import FusionConfig, ModelSpec, Params
from open_fusion.schema import Phase
from open_fusion.eval import load_tasks
from open_fusion.eval.grader import RuleGrader
from open_fusion.eval.superiority import run_superiority_benchmark, render_superiority_markdown


BASE_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
API_KEY = os.environ.get("ARK_API_KEY", "")

# 单模型对照组
SOLO_MODEL = ModelSpec("glm-5.2")

# Fusion panel: 三个不同模型
FUSION_PANEL = [
    ModelSpec("glm-5.1"),
    ModelSpec("kimi-k2.6"),
    ModelSpec("deepseek-v4-flash"),
]

# Judge/Synthesizer: 不参与 panel 的独立模型
JUDGE = ModelSpec("doubao-seed-2.0-pro")


async def main():
    params = {
        Phase.PANEL: Params(temperature=0.7, max_tokens=8192, timeout_s=300.0),
        Phase.JUDGE: Params(temperature=0.2, max_tokens=4096, timeout_s=300.0),
        Phase.SYNTHESIS: Params(temperature=0.3, max_tokens=8192, timeout_s=300.0),
    }

    # 配置：panel = fusion panel, judge = doubao (不参与panel)
    base = FusionConfig(
        panel=FUSION_PANEL,
        judge=JUDGE,
        tools_enabled=False,
        base_url=BASE_URL,
        api_key=API_KEY,
        max_in_flight=1,
        params=params,
    )

    print("Loading logic_hard tasks...", flush=True)
    tasks = load_tasks("logic_hard")
    print(f"Loaded {len(tasks)} tasks\n", flush=True)

    print("=" * 70)
    print("Configuration:")
    print(f"  Solo baseline:  {SOLO_MODEL.slug}")
    print(f"  Fusion panel:   {[m.slug for m in FUSION_PANEL]}")
    print(f"  Judge/Synth:    {JUDGE.slug}")
    print(f"  Tasks:          {len(tasks)} (logic_hard)")
    print("=" * 70, flush=True)

    grader = RuleGrader()

    report = await run_superiority_benchmark(
        tasks, FUSION_PANEL, JUDGE, grader,
        client=None, base=base, n_passes=1,
        bootstrap_iters=1000, tie_margin=1.0,
        panel_size=len(FUSION_PANEL), seed=42,
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    md = render_superiority_markdown(report)
    print(md)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_glm52_vs_fusion_{ts}.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(f"\nReport saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
