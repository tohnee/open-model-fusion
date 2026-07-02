#!/usr/bin/env python3
"""能力对比测试: 编排是否真能提升模型能力？

核心实验设计:
  Arm 1: 单模型 glm-5.2 (最强单模型 baseline)
  Arm 2: 单模型 kimi-k2.6 (中游单模型 baseline)
  Arm 3: Fusion FULL (panel: kimi/deepseek-flash/doubao, judge: glm-5.2)
  Arm 4: Fusion AGGREGATOR (panel: 同上, 无 judge, 直接合成)
  Arm 5: Fusion PICK-BEST (panel: glm-5.2/kimi/deepseek-flash, judge: glm-5.2, pick-best 短路)

任务设计原则:
  - 每道题有唯一正确答案 (客观评分, 不依赖 LLM-as-judge)
  - 覆盖 5 个能力维度: 逻辑推理 / 数学 / 代码 / 常识陷阱 / 多步推理
  - 难度设在中游模型容易出错、但组合视角能纠错的区间

用法:
    export ARK_API_KEY=ark-xxx
    python3 run_capability_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, "src")

from open_fusion.config import FusionConfig, FusionMode, ModelSpec, Params
from open_fusion.schema import Phase
from open_fusion.orchestrator import fuse
from open_fusion.client import ModelClient

# ============================================================================
# 配置
# ============================================================================
BASE_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
API_KEY = os.environ.get("ARK_API_KEY", "")

if not API_KEY:
    print("ERROR: 请先设置 ARK_API_KEY 环境变量")
    sys.exit(1)

# 模型配置
SOLO_STRONG = ModelSpec("glm-5.2")
SOLO_MID = ModelSpec("kimi-k2.6")
FUSION_PANEL = [
    ModelSpec("kimi-k2.6"),
    ModelSpec("deepseek-v4-flash"),
    ModelSpec("doubao-seed-2.0-pro"),
]
FUSION_JUDGE = ModelSpec("glm-5.2")

# v1.2: pick-best arm 的 panel — 包含 glm-5.2 (最强模型加入 panel, 让 judge 可以选中它)
PICKBEST_PANEL = [
    ModelSpec("glm-5.2"),
    ModelSpec("kimi-k2.6"),
    ModelSpec("deepseek-v4-flash"),
]


# ============================================================================
# 测试任务 (每道题有唯一正确答案, 客观评分)
# ============================================================================
TASKS = [
    # --- 1. 数学: 多步计算 (中游模型易错) ---
    {
        "id": "math_1",
        "domain": "数学",
        "difficulty": "中",
        "question": """一个水池有三个进水管 A、B、C。
- A 单独开 6 小时注满
- B 单独开 8 小时注满
- C 单独开 12 小时注满

先开 A 和 B 2 小时，然后关闭 B 同时打开 C，再过几小时能注满？
请只给出最终数字（小时数，保留两位小数）。""",
        "answer": "1.20",
        "check": lambda r: "1.20" in r or "1.2" in r,
    },
    # --- 2. 代码: 找 bug ---
    {
        "id": "code_1",
        "domain": "代码",
        "difficulty": "中",
        "question": """以下 Python 函数有一个 bug，会导致在某些输入下返回错误结果。
请找出 bug 并给出修复后的完整函数。

```python
def merge_sorted_lists(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    # bug: 只追加了 a 的剩余部分
    result.extend(a[i:])
    return result
```

要求：函数应正确合并两个已排序的列表。给出修复后的完整函数代码。""",
        "answer": "result.extend(b[j:])",
        "check": lambda r: "extend(b[j:])" in r or "extend(b[j::])" in r,
    },
    # --- 3. 常识陷阱: 反直觉 ---
    {
        "id": "trap_1",
        "domain": "常识陷阱",
        "difficulty": "难",
        "question": """一根绳子对折 3 次后从中间剪断，会得到几段绳子？
请只给出数字。""",
        "answer": "9",
        "check": lambda r: re.search(r'\b9\b', r) is not None,
    },
    # --- 4. 多步推理: 因果链 ---
    {
        "id": "reasoning_1",
        "domain": "多步推理",
        "difficulty": "难",
        "question": """在一个房间里有 100 个柜子，都是关着的。
第 1 个人走过，把所有柜子都打开。
第 2 个人走过，把所有 2 的倍数的柜子切换状态（开变关，关变开）。
第 3 个人走过，把所有 3 的倍数的柜子切换状态。
...
第 100 个人走过，把第 100 号柜子切换状态。

最后有多少个柜子是开着的？请只给出数字。""",
        "answer": "10",
        "check": lambda r: re.search(r'\b10\b', r) is not None,
    },
    # --- 5. 数学: 概率 ---
    {
        "id": "math_2",
        "domain": "数学",
        "difficulty": "难",
        "question": """一个袋子里有 5 个红球和 3 个蓝球。随机抽出 2 个球（不放回），
两个球颜色不同的概率是多少？请给出最简分数。""",
        "answer": "15/28",
        "check": lambda r: "15/28" in r.replace(" ", ""),
    },
]


# ============================================================================
# 评分
# ============================================================================
def grade(task: dict, response: str) -> bool:
    """客观评分: 用 task 的 check 函数判断答案是否正确。"""
    if not response or not response.strip():
        return False
    return task["check"](response.strip())


# ============================================================================
# 运行单个 arm
# ============================================================================
async def run_solo(model: ModelSpec, tasks: list[dict], arm_name: str) -> dict:
    """运行单模型 baseline。"""
    client = ModelClient(base_url=BASE_URL, api_key=API_KEY, fusion_depth=1)
    params = Params(temperature=0.3, max_tokens=4096, timeout_s=30.0, max_tool_calls=0)

    results = []
    correct = 0
    total_ms = 0

    for task in tasks:
        messages = [{"role": "user", "content": task["question"]}]
        t0 = time.monotonic()
        try:
            comp = await client.complete(model, messages, params=params)
            response = comp.content
            latency_ms = int((time.monotonic() - t0) * 1000)
        except Exception as e:
            response = f"ERROR: {e}"
            latency_ms = int((time.monotonic() - t0) * 1000)

        is_correct = grade(task, response)
        if is_correct:
            correct += 1
        total_ms += latency_ms

        results.append({
            "task_id": task["id"],
            "domain": task["domain"],
            "difficulty": task["difficulty"],
            "correct": is_correct,
            "latency_ms": latency_ms,
            "response_preview": response[:200],
        })
        print(f"  [{arm_name}] {task['id']} ({task['domain']}): "
              f"{'✓' if is_correct else '✗'}  {latency_ms}ms")

    return {
        "arm": arm_name,
        "model": model.slug,
        "total": len(tasks),
        "correct": correct,
        "accuracy": round(correct / len(tasks) * 100, 1),
        "total_ms": total_ms,
        "avg_ms": round(total_ms / len(tasks)),
        "results": results,
    }


async def run_fusion(config: FusionConfig, tasks: list[dict], arm_name: str) -> dict:
    """运行 Fusion 编排。"""
    client = ModelClient(base_url=BASE_URL, api_key=API_KEY, fusion_depth=0)

    results = []
    correct = 0
    total_ms = 0

    for task in tasks:
        t0 = time.monotonic()
        try:
            r = await fuse(task["question"], config, client=client)
            response = r.text
            latency_ms = int((time.monotonic() - t0) * 1000)
            status = r.status.value
        except Exception as e:
            response = f"ERROR: {e}"
            latency_ms = int((time.monotonic() - t0) * 1000)
            status = "error"

        is_correct = grade(task, response)
        if is_correct:
            correct += 1
        total_ms += latency_ms

        results.append({
            "task_id": task["id"],
            "domain": task["domain"],
            "difficulty": task["difficulty"],
            "correct": is_correct,
            "latency_ms": latency_ms,
            "status": status,
            "response_preview": response[:200],
        })
        print(f"  [{arm_name}] {task['id']} ({task['domain']}): "
              f"{'✓' if is_correct else '✗'}  {latency_ms}ms  ({status})")

    return {
        "arm": arm_name,
        "total": len(tasks),
        "correct": correct,
        "accuracy": round(correct / len(tasks) * 100, 1),
        "total_ms": total_ms,
        "avg_ms": round(total_ms / len(tasks)),
        "results": results,
    }


# ============================================================================
# 主函数
# ============================================================================
async def main():
    print("=" * 72)
    print("能力对比测试: 编排是否真能提升模型能力？")
    print("=" * 72)
    print(f"\n任务数: {len(TASKS)}")
    print(f"能力维度: {', '.join(sorted(set(t['domain'] for t in TASKS)))}")
    print(f"难度分布: {', '.join(sorted(set(t['difficulty'] for t in TASKS)))}")

    print(f"\n实验设计:")
    print(f"  Arm 1: 单模型 {SOLO_STRONG.slug} (最强单模型)")
    print(f"  Arm 2: 单模型 {SOLO_MID.slug} (中游单模型)")
    print(f"  Arm 3: Fusion FULL (panel: {[m.slug for m in FUSION_PANEL]}, judge: {FUSION_JUDGE.slug})")
    print(f"  Arm 4: Fusion AGGREGATOR (panel: 同上, 无 judge)")
    print(f"  Arm 5: Fusion PICK-BEST (panel: {[m.slug for m in PICKBEST_PANEL]}, judge: {FUSION_JUDGE.slug})")

    all_results = []

    # Arm 1: 单模型 glm-5.2
    print(f"\n{'─'*72}")
    print(f"[Arm 1] 单模型: {SOLO_STRONG.slug}")
    print(f"{'─'*72}")
    r1 = await run_solo(SOLO_STRONG, TASKS, f"Solo-{SOLO_STRONG.slug}")
    all_results.append(r1)

    # Arm 2: 单模型 kimi-k2.6
    print(f"\n{'─'*72}")
    print(f"[Arm 2] 单模型: {SOLO_MID.slug}")
    print(f"{'─'*72}")
    r2 = await run_solo(SOLO_MID, TASKS, f"Solo-{SOLO_MID.slug}")
    all_results.append(r2)

    # Arm 3: Fusion FULL
    print(f"\n{'─'*72}")
    print(f"[Arm 3] Fusion FULL (panel + judge + synth)")
    print(f"{'─'*72}")
    config_full = FusionConfig(
        panel=FUSION_PANEL,
        judge=FUSION_JUDGE,
        caller=FUSION_JUDGE,
        mode=FusionMode.FULL,
        base_url=BASE_URL,
        api_key=API_KEY,
        enable_consensus_shortcut=False,  # 关闭短路, 测完整管道
        enable_pick_best=False,
        max_in_flight=3,
        params={
            Phase.PANEL: Params(temperature=0.7, max_tokens=4096, timeout_s=30.0, max_tool_calls=0),
            Phase.JUDGE: Params(temperature=0.2, max_tokens=4096, timeout_s=30.0, max_tool_calls=0),
            Phase.SYNTHESIS: Params(temperature=0.3, max_tokens=4096, timeout_s=30.0, max_tool_calls=0),
        },
    )
    r3 = await run_fusion(config_full, TASKS, "Fusion-FULL")
    all_results.append(r3)

    # Arm 4: Fusion AGGREGATOR
    print(f"\n{'─'*72}")
    print(f"[Arm 4] Fusion AGGREGATOR (panel + synth, skip judge)")
    print(f"{'─'*72}")
    config_agg = FusionConfig(
        panel=FUSION_PANEL,
        judge=FUSION_JUDGE,
        caller=FUSION_JUDGE,
        mode=FusionMode.AGGREGATOR,
        base_url=BASE_URL,
        api_key=API_KEY,
        enable_consensus_shortcut=False,
        enable_pick_best=False,
        max_in_flight=3,
        params={
            Phase.PANEL: Params(temperature=0.7, max_tokens=4096, timeout_s=30.0, max_tool_calls=0),
            Phase.SYNTHESIS: Params(temperature=0.3, max_tokens=4096, timeout_s=30.0, max_tool_calls=0),
        },
    )
    r4 = await run_fusion(config_agg, TASKS, "Fusion-AGG")
    all_results.append(r4)

    # Arm 5: Fusion FULL + PICK-BEST (glm-5.2 加入 panel, judge 可选中它)
    print(f"\n{'─'*72}")
    print(f"[Arm 5] Fusion PICK-BEST (panel + judge + pick-best 短路)")
    print(f"{'─'*72}")
    config_pickbest = FusionConfig(
        panel=PICKBEST_PANEL,
        judge=FUSION_JUDGE,
        caller=FUSION_JUDGE,
        mode=FusionMode.FULL,
        base_url=BASE_URL,
        api_key=API_KEY,
        enable_consensus_shortcut=False,
        enable_pick_best=True,          # 开启 pick-best
        pick_best_min_chars=10,         # 短答案也能触发
        max_in_flight=3,
        params={
            Phase.PANEL: Params(temperature=0.7, max_tokens=4096, timeout_s=45.0, max_tool_calls=0),
            Phase.JUDGE: Params(temperature=0.2, max_tokens=4096, timeout_s=45.0, max_tool_calls=0),
            Phase.SYNTHESIS: Params(temperature=0.3, max_tokens=4096, timeout_s=45.0, max_tool_calls=0),
        },
    )
    r5 = await run_fusion(config_pickbest, TASKS, "Fusion-PICKBEST")
    all_results.append(r5)

    # ========================================================================
    # 汇总报告
    # ========================================================================
    print(f"\n{'='*72}")
    print("能力对比汇总")
    print(f"{'='*72}")
    print(f"\n{'Arm':<25} {'Correct':>8} {'Total':>6} {'Accuracy':>10} {'Avg ms':>8}")
    print(f"{'-'*60}")
    for r in all_results:
        print(f"{r['arm']:<25} {r['correct']:>8} {r['total']:>6} "
              f"{r['accuracy']:>9.1f}% {r['avg_ms']:>8}")

    # 按维度分析
    print(f"\n{'='*72}")
    print("按能力维度分析")
    print(f"{'='*72}")
    domains = sorted(set(t["domain"] for t in TASKS))
    for domain in domains:
        domain_tasks = [t for t in TASKS if t["domain"] == domain]
        print(f"\n  [{domain}] ({len(domain_tasks)} 题)")
        for r in all_results:
            domain_results = [x for x in r["results"] if x["domain"] == domain]
            domain_correct = sum(1 for x in domain_results if x["correct"])
            print(f"    {r['arm']:<25} {domain_correct}/{len(domain_results)}")

    # 关键对比
    print(f"\n{'='*72}")
    print("关键发现")
    print(f"{'='*72}")

    solo_strong_acc = r1["accuracy"]
    solo_mid_acc = r2["accuracy"]
    fusion_full_acc = r3["accuracy"]
    fusion_agg_acc = r4["accuracy"]
    pickbest_acc = r5["accuracy"]

    print(f"\n  1. 编排 vs 最强单模型:")
    delta = fusion_full_acc - solo_strong_acc
    print(f"     Fusion FULL vs {SOLO_STRONG.slug}: {fusion_full_acc}% vs {solo_strong_acc}% "
          f"({'↑' if delta > 0 else '↓'} {abs(delta):.1f}%)")

    print(f"\n  2. 编排 vs 中游单模型:")
    delta = fusion_full_acc - solo_mid_acc
    print(f"     Fusion FULL vs {SOLO_MID.slug}: {fusion_full_acc}% vs {solo_mid_acc}% "
          f"({'↑' if delta > 0 else '↓'} {abs(delta):.1f}%)")

    print(f"\n  3. FULL vs AGGREGATOR (judge 层是否提升能力):")
    delta = fusion_full_acc - fusion_agg_acc
    print(f"     FULL vs AGGREGATOR: {fusion_full_acc}% vs {fusion_agg_acc}% "
          f"({'↑' if delta > 0 else '↓'} {abs(delta):.1f}%)")

    print(f"\n  4. PICK-BEST vs FULL (pick-best 短路是否避免投票覆盖):")
    delta = pickbest_acc - fusion_full_acc
    print(f"     PICK-BEST vs FULL: {pickbest_acc}% vs {fusion_full_acc}% "
          f"({'↑' if delta > 0 else '↓'} {abs(delta):.1f}%)")

    print(f"\n  5. PICK-BEST vs 最强单模型 (编排+短路能否超越单模型):")
    delta = pickbest_acc - solo_strong_acc
    if delta > 0:
        print(f"     ✓ PICK-BEST ({pickbest_acc}%) 超越 {SOLO_STRONG.slug} ({solo_strong_acc}%)!")
    elif delta == 0:
        print(f"     → 持平 ({pickbest_acc}% vs {solo_strong_acc}%)")
    else:
        print(f"     ✗ PICK-BEST ({pickbest_acc}%) 未超越 {SOLO_STRONG.slug} ({solo_strong_acc}%)")

    print(f"\n  6. 弱模型编排是否超越强模型单干:")
    if fusion_full_acc > solo_strong_acc:
        print(f"     ✓ 是! 编排 ({fusion_full_acc}%) 超越 {SOLO_STRONG.slug} ({solo_strong_acc}%)")
    elif fusion_full_acc == solo_strong_acc:
        print(f"     → 持平 ({fusion_full_acc}% vs {solo_strong_acc}%)")
    else:
        print(f"     ✗ 否。编排 ({fusion_full_acc}%) 未超越 {SOLO_STRONG.slug} ({solo_strong_acc}%)")

    # 逐题对比
    print(f"\n{'='*72}")
    print("逐题对比 (✓=正确, ✗=错误)")
    print(f"{'='*72}")
    print(f"\n{'Task ID':<12} {'Domain':<10} {'Strong':>8} {'Mid':>6} {'FULL':>6} {'AGG':>6} {'PICK-BEST':>10}")
    print(f"{'-'*62}")
    for i, task in enumerate(TASKS):
        s1 = "✓" if r1["results"][i]["correct"] else "✗"
        s2 = "✓" if r2["results"][i]["correct"] else "✗"
        s3 = "✓" if r3["results"][i]["correct"] else "✗"
        s4 = "✓" if r4["results"][i]["correct"] else "✗"
        s5 = "✓" if r5["results"][i]["correct"] else "✗"
        print(f"{task['id']:<12} {task['domain']:<10} {s1:>8} {s2:>6} {s3:>6} {s4:>6} {s5:>10}")

    print(f"\n{'='*72}")
    print("结论")
    print(f"{'='*72}")

    # 分析编排独有的正确题
    fusion_only = []
    for i, task in enumerate(TASKS):
        if r3["results"][i]["correct"] and not r1["results"][i]["correct"]:
            fusion_only.append(task["id"])
    if fusion_only:
        print(f"\n  编排独有的正确题 (单模型答错但编排答对): {fusion_only}")
        print(f"  → 这些题体现了编排的纠错价值")

    # 分析编排答错但单模型答对的题
    solo_only = []
    for i, task in enumerate(TASKS):
        if r1["results"][i]["correct"] and not r3["results"][i]["correct"]:
            solo_only.append(task["id"])
    if solo_only:
        print(f"\n  编排丢失的题 (单模型答对但编排答错): {solo_only}")
        print(f"  → 这些题说明编排可能在合成时丢失了正确信息")

    # 分析 pick-best 是否挽回了丢失的题
    if solo_only:
        pickbest_recovered = []
        for i, task in enumerate(TASKS):
            if (r1["results"][i]["correct"] and not r3["results"][i]["correct"]
                    and r5["results"][i]["correct"]):
                pickbest_recovered.append(task["id"])
        if pickbest_recovered:
            print(f"\n  ★ PICK-BEST 挽回的题 (FULL 丢失但 PICK-BEST 答对): {pickbest_recovered}")
            print(f"  → pick-best 短路成功避免了投票覆盖问题!")
        else:
            print(f"\n  ✗ PICK-BEST 未能挽回丢失的题")
            print(f"  → judge 可能未选中正确模型, 或 panel 中无正确答案")

    # pick-best 触发情况
    pickbest_triggered = []
    for i, task in enumerate(TASKS):
        status = r5["results"][i].get("status", "")
        if "pick_best" in status:
            pickbest_triggered.append(task["id"])
    if pickbest_triggered:
        print(f"\n  PICK-BEST 短路触发的题: {pickbest_triggered}")
    else:
        print(f"\n  PICK-BEST 短路未触发 (judge 可能未选出 best_model 或答案太短)")

    if not fusion_only and not solo_only:
        print(f"\n  编排与单模型答对/答错的题完全一致")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
