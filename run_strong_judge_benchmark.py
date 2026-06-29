#!/usr/bin/env python3
"""强 Judge 方案 benchmark: glm-5.2 作为 judge/synthesizer。

配置亮点：
- Judge/Synthesizer: glm-5.2（独立于 panel，避免自审偏见）
- Panel: kimi-k2.6 + deepseek-v4-flash + doubao-seed-2.0-pro
- 开启 consensus + pick-best 双短路机制
- 分别跑 logic_hard 和 superiority_v2 两套任务集

用法：
    python3 run_strong_judge_benchmark.py              # 跑 logic_hard
    python3 run_strong_judge_benchmark.py --full        # 跑 logic_hard + superiority_v2
    python3 run_strong_judge_benchmark.py --suite superiority_v2
"""
import asyncio
import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, "src")

from open_fusion.config import FusionConfig, ModelSpec
from open_fusion.eval import load_tasks
from open_fusion.eval.grader import RuleGrader
from open_fusion.eval.superiority import run_superiority_benchmark, render_superiority_markdown


# ======================== 强 Judge 配置 ========================

BASE_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
API_KEY = os.environ.get("ARK_API_KEY", "")

# 强 Judge: glm-5.2 作为独立 judge/synthesizer
JUDGE = ModelSpec("glm-5.2")

# Panel: 三个不同厂商的模型，最大化多样性
PANEL = [
    ModelSpec("kimi-k2.6"),
    ModelSpec("deepseek-v4-flash"),
    ModelSpec("doubao-seed-2.0-pro"),
]


async def run_suite(task_suite: str):
    """运行一套任务集并输出报告。"""
    config = FusionConfig(
        panel=PANEL,
        judge=JUDGE,
        caller=JUDGE,                    # synthesizer 也用 glm-5.2
        tools_enabled=False,
        base_url=BASE_URL,
        api_key=API_KEY,
        max_in_flight=1,
        # 开启双短路机制
        enable_consensus_shortcut=True,
        enable_pick_best=True,
        consensus_threshold=0.85,
    )

    print(f"Loading tasks: {task_suite}...", flush=True)
    tasks = load_tasks(task_suite)

    from collections import Counter
    domain_counts = Counter(t.domain for t in tasks)
    print(f"\nLoaded {len(tasks)} tasks across {len(domain_counts)} domains:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  - {domain}: {count} tasks")

    print(f"\nJudge/Synthesizer: {JUDGE.slug}")
    print(f"Panel: {[m.slug for m in PANEL]}")
    print(f"Consensus shortcut: ON (threshold=0.85)")
    print(f"Pick-best: ON")
    print()

    grader = RuleGrader()

    print("=" * 70)
    print(f"Running benchmark: {task_suite} ({len(tasks)} tasks)")
    print("=" * 70, flush=True)

    report = await run_superiority_benchmark(
        tasks, PANEL, JUDGE, grader,
        client=None, base=config, n_passes=1,
        bootstrap_iters=1000, tie_margin=1.0,
        panel_size=len(PANEL), seed=42,
    )

    print()
    print("=" * 70)
    print(f"RESULTS: {task_suite}")
    print("=" * 70)

    md = render_superiority_markdown(report)
    print(md)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_{task_suite}_judge-{JUDGE.slug}_{ts}.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(f"\nReport saved to {output_file}")
    return report


async def main():
    parser = argparse.ArgumentParser(description="Strong judge benchmark with glm-5.2")
    parser.add_argument("--suite", default="logic_hard",
                        choices=["logic_hard", "superiority_v2"],
                        help="Task suite to run")
    parser.add_argument("--full", action="store_true",
                        help="Run both logic_hard and superiority_v2")
    args = parser.parse_args()

    if args.full:
        await run_suite("logic_hard")
        print("\n\n" + "=" * 70)
        print("Starting superiority_v2 suite...")
        print("=" * 70 + "\n")
        await run_suite("superiority_v2")
    else:
        await run_suite(args.suite)


if __name__ == "__main__":
    asyncio.run(main())
