#!/usr/bin/env python3
"""v1.3 Verifier + Fallback 能力测试

用 math_hard_v13.json 的 10 道高难度数学题, 对比 4 个 arm:
  Arm 1: 单模型 glm-5.2 (baseline)
  Arm 2: Fusion PICK-BEST (无 verifier, judge 可能误判)
  Arm 3: Fusion PICK-BEST + Verifier (验证器拦截不合理答案)
  Arm 4: Fusion AGGREGATOR (跳过 judge, 直接合成)

核心问题: Verifier 能否在 math_2 这类题目上拦截 judge 误判?

用法:
    export ARK_API_KEY=ark-xxx
    python3 run_v13_verifier_test.py [--quick]  # --quick 只跑前 5 题
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
from open_fusion.schema import Phase, FusionStatus
from open_fusion.orchestrator import fuse
from open_fusion.client import ModelClient

BASE_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
API_KEY = os.environ.get("ARK_API_KEY", "")

if not API_KEY:
    print("ERROR: 请先设置 ARK_API_KEY 环境变量")
    sys.exit(1)

# 模型配置
SOLO_STRONG = ModelSpec("glm-5.2")
PANEL = [
    ModelSpec("glm-5.2"),
    ModelSpec("kimi-k2.6"),
    ModelSpec("deepseek-v4-flash"),
]
JUDGE = ModelSpec("glm-5.2")

# 超时配置 (数学题需要更长推理时间)
TIMEOUT = 45.0


def load_tasks(quick: bool = False) -> list[dict]:
    """加载 math_hard_v13.json 测试数据集。"""
    path = os.path.join(os.path.dirname(__file__), "src", "open_fusion",
                        "eval", "tasks", "math_hard_v13.json")
    with open(path) as f:
        data = json.load(f)
    tasks = data["tasks"]
    if quick:
        tasks = tasks[:5]
    # 解析 check 字符串为 lambda
    for t in tasks:
        if isinstance(t["check"], str):
            # 简单 eval (安全: 我们自己写的数据)
            t["check"] = eval(t["check"])
    return tasks


def grade(task: dict, response: str) -> bool:
    if not response or not response.strip():
        return False
    try:
        return task["check"](response.strip())
    except Exception:
        return False


def make_task_metadata(task: dict) -> dict:
    """从 task 提取 verifier 元数据。"""
    return {
        "domain": task.get("domain", ""),
        "expected_range": tuple(task["expected_range"]) if task.get("expected_range") else None,
        "expected_format": task.get("expected_format"),
        "answer_keywords": task.get("answer_keywords", []),
    }


# ============================================================================
# Arm 运行函数
# ============================================================================

async def run_solo(model: ModelSpec, tasks: list[dict], arm_name: str) -> dict:
    client = ModelClient(base_url=BASE_URL, api_key=API_KEY, fusion_depth=1)
    params = Params(temperature=0.3, max_tokens=4096, timeout_s=TIMEOUT, max_tool_calls=0)

    results = []
    correct = 0
    total_ms = 0

    for task in tasks:
        messages = [{"role": "user", "content": task["question"]}]
        t0 = time.monotonic()
        try:
            comp = await client.complete(model, messages, params=params)
            response = comp.content
        except Exception as e:
            response = f"ERROR: {e}"
        latency_ms = int((time.monotonic() - t0) * 1000)

        is_correct = grade(task, response)
        if is_correct:
            correct += 1
        total_ms += latency_ms
        results.append({"task_id": task["id"], "correct": is_correct,
                        "latency_ms": latency_ms, "status": "solo",
                        "response_preview": response[:120]})
        print(f"  [{arm_name}] {task['id']}: {'✓' if is_correct else '✗'}  {latency_ms}ms")

    return {"arm": arm_name, "correct": correct, "total": len(tasks),
            "accuracy": round(correct / len(tasks) * 100, 1),
            "avg_ms": round(total_ms / len(tasks)), "results": results}


async def run_fusion(config: FusionConfig, tasks: list[dict],
                     arm_name: str, use_verifier_meta: bool = False) -> dict:
    client = ModelClient(base_url=BASE_URL, api_key=API_KEY, fusion_depth=0)

    results = []
    correct = 0
    total_ms = 0

    for task in tasks:
        # 为 verifier 设置题目元数据
        if use_verifier_meta:
            config._task_metadata = make_task_metadata(task)

        t0 = time.monotonic()
        try:
            r = await fuse(task["question"], config, client=client)
            response = r.text
            status = r.status.value
        except Exception as e:
            response = f"ERROR: {e}"
            status = "error"
        latency_ms = int((time.monotonic() - t0) * 1000)

        is_correct = grade(task, response)
        if is_correct:
            correct += 1
        total_ms += latency_ms
        results.append({"task_id": task["id"], "correct": is_correct,
                        "latency_ms": latency_ms, "status": status,
                        "response_preview": response[:120]})
        print(f"  [{arm_name}] {task['id']}: {'✓' if is_correct else '✗'}  "
              f"{latency_ms}ms  ({status})")

    return {"arm": arm_name, "correct": correct, "total": len(tasks),
            "accuracy": round(correct / len(tasks) * 100, 1),
            "avg_ms": round(total_ms / len(tasks)), "results": results}


# ============================================================================
# 主函数
# ============================================================================

async def main():
    quick = "--quick" in sys.argv
    tasks = load_tasks(quick=quick)

    print("=" * 72)
    print(f"v1.3 Verifier + Fallback 能力测试 ({len(tasks)} 道高难度数学题)")
    print("=" * 72)
    print(f"\n实验设计:")
    print(f"  Arm 1: Solo glm-5.2 (baseline)")
    print(f"  Arm 2: Fusion PICK-BEST (无 verifier)")
    print(f"  Arm 3: Fusion PICK-BEST + Verifier (验证器拦截)")
    print(f"  Arm 4: Fusion AGGREGATOR (跳过 judge)")

    # 公共 fusion 配置
    base_params = {
        Phase.PANEL: Params(temperature=0.7, max_tokens=4096, timeout_s=TIMEOUT, max_tool_calls=0),
        Phase.JUDGE: Params(temperature=0.2, max_tokens=4096, timeout_s=TIMEOUT, max_tool_calls=0),
        Phase.SYNTHESIS: Params(temperature=0.3, max_tokens=4096, timeout_s=TIMEOUT, max_tool_calls=0),
    }

    all_results = []

    # Arm 1: Solo
    print(f"\n{'─'*72}")
    print(f"[Arm 1] Solo glm-5.2")
    print(f"{'─'*72}")
    r1 = await run_solo(SOLO_STRONG, tasks, "Solo-glm5.2")
    all_results.append(r1)

    # Arm 2: PICK-BEST (无 verifier)
    print(f"\n{'─'*72}")
    print(f"[Arm 2] Fusion PICK-BEST (无 verifier)")
    print(f"{'─'*72}")
    config_pb = FusionConfig(
        panel=PANEL, judge=JUDGE, caller=JUDGE, mode=FusionMode.FULL,
        base_url=BASE_URL, api_key=API_KEY,
        enable_consensus_shortcut=False,
        enable_pick_best=True,
        enable_verifier=False,       # 关闭 verifier
        pick_best_min_chars=10,
        pick_best_confidence_threshold=0.3,  # 低阈值, 让 pick-best 更容易触发
        max_in_flight=3,
        params=base_params,
    )
    r2 = await run_fusion(config_pb, tasks, "PICK-BEST")
    all_results.append(r2)

    # Arm 3: PICK-BEST + Verifier
    print(f"\n{'─'*72}")
    print(f"[Arm 3] Fusion PICK-BEST + Verifier")
    print(f"{'─'*72}")
    config_vb = FusionConfig(
        panel=PANEL, judge=JUDGE, caller=JUDGE, mode=FusionMode.FULL,
        base_url=BASE_URL, api_key=API_KEY,
        enable_consensus_shortcut=False,
        enable_pick_best=True,
        enable_verifier=True,        # 开启 verifier
        pick_best_min_chars=10,
        pick_best_confidence_threshold=0.3,
        max_in_flight=3,
        params=base_params,
    )
    r3 = await run_fusion(config_vb, tasks, "PICK-BEST+Verifier",
                          use_verifier_meta=True)
    all_results.append(r3)

    # Arm 4: AGGREGATOR
    print(f"\n{'─'*72}")
    print(f"[Arm 4] Fusion AGGREGATOR")
    print(f"{'─'*72}")
    config_agg = FusionConfig(
        panel=PANEL, judge=JUDGE, caller=JUDGE, mode=FusionMode.AGGREGATOR,
        base_url=BASE_URL, api_key=API_KEY,
        enable_consensus_shortcut=False,
        enable_pick_best=False,
        max_in_flight=3,
        params=base_params,
    )
    r4 = await run_fusion(config_agg, tasks, "AGGREGATOR")
    all_results.append(r4)

    # ========================================================================
    # 汇总报告
    # ========================================================================
    print(f"\n{'='*72}")
    print("汇总报告")
    print(f"{'='*72}")
    print(f"\n{'Arm':<25} {'Correct':>8} {'Total':>6} {'Accuracy':>10} {'Avg ms':>8}")
    print(f"{'-'*60}")
    for r in all_results:
        print(f"{r['arm']:<25} {r['correct']:>8} {r['total']:>6} "
              f"{r['accuracy']:>9.1f}% {r['avg_ms']:>8}")

    # 逐题对比
    print(f"\n{'='*72}")
    print("逐题对比 (✓=正确, ✗=错误, SC=pick-best短路, FB=fallback, AGG=aggregator)")
    print(f"{'='*72}")
    print(f"\n{'Task ID':<16} {'Solo':>6} {'PICK-BEST':>12} {'+Verifier':>12} {'AGG':>8}")
    print(f"{'-'*56}")
    for i, task in enumerate(tasks):
        s1 = "✓" if r1["results"][i]["correct"] else "✗"
        s2 = "✓" if r2["results"][i]["correct"] else "✗"
        s3 = "✓" if r3["results"][i]["correct"] else "✗"
        s4 = "✓" if r4["results"][i]["correct"] else "✗"
        # 标注 status
        st2 = r2["results"][i].get("status", "")
        st3 = r3["results"][i].get("status", "")
        if "pick_best" in st2:
            s2 += "(SC)"
        if "pick_best" in st3:
            s3 += "(SC)"
        elif "ok" == st3:
            s3 += "(FB)"  # verifier fallback → synth → ok
        print(f"{task['id']:<16} {s1:>6} {s2:>12} {s3:>12} {s4:>8}")

    # Verifier 效果分析
    print(f"\n{'='*72}")
    print("Verifier 效果分析")
    print(f"{'='*72}")

    verifier_intercepted = 0
    verifier_recovered = 0
    for i in range(len(tasks)):
        r2_status = r2["results"][i].get("status", "")
        r3_status = r3["results"][i].get("status", "")
        r2_correct = r2["results"][i]["correct"]
        r3_correct = r3["results"][i]["correct"]

        # Pick-best 短路在 Arm 2 触发但在 Arm 3 未触发 = verifier 拦截
        if "pick_best" in r2_status and "pick_best" not in r3_status:
            verifier_intercepted += 1
            if not r2_correct and r3_correct:
                verifier_recovered += 1
                print(f"  ✓ {tasks[i]['id']}: Verifier 拦截了错误短路 → synth 答对!")
            elif not r2_correct and not r3_correct:
                print(f"  → {tasks[i]['id']}: Verifier 拦截了, 但 synth 也答错")
            elif r2_correct and not r3_correct:
                print(f"  ✗ {tasks[i]['id']}: Verifier 误拦截了正确短路")

    if verifier_intercepted == 0:
        print("  (Verifier 未拦截任何 pick-best 短路)")

    print(f"\n  Verifier 拦截次数: {verifier_intercepted}")
    print(f"  其中成功挽回: {verifier_recovered}")

    # Fallback 效果
    print(f"\n{'='*72}")
    print("Fallback 机制效果")
    print(f"{'='*72}")

    fallback_triggered = 0
    for i in range(len(tasks)):
        r3_status = r3["results"][i].get("status", "")
        if r3_status == "ok":
            fallback_triggered += 1

    pickbest_triggered = 0
    for i in range(len(tasks)):
        r3_status = r3["results"][i].get("status", "")
        if "pick_best" in r3_status:
            pickbest_triggered += 1

    print(f"\n  Arm 3 (PICK-BEST+Verifier):")
    print(f"    Pick-best 短路触发: {pickbest_triggered}/{len(tasks)}")
    print(f"    Fallback 到 synth:  {fallback_triggered}/{len(tasks)}")

    # 结论
    print(f"\n{'='*72}")
    print("结论")
    print(f"{'='*72}")
    solo_acc = r1["accuracy"]
    pb_acc = r2["accuracy"]
    vb_acc = r3["accuracy"]
    agg_acc = r4["accuracy"]

    print(f"\n  Solo glm-5.2:        {solo_acc}%")
    print(f"  PICK-BEST (无ver):   {pb_acc}%")
    print(f"  PICK-BEST+Verifier:  {vb_acc}%")
    print(f"  AGGREGATOR:          {agg_acc}%")

    delta_vb = vb_acc - pb_acc
    if delta_vb > 0:
        print(f"\n  ★ Verifier 提升: +{delta_vb:.1f}% (相比无 verifier)")
    elif delta_vb < 0:
        print(f"\n  ✗ Verifier 退化: {delta_vb:.1f}% (可能误拦截)")
    else:
        print(f"\n  → Verifier 无影响 (可能未触发或 judge 选择一致)")

    delta_agg = agg_acc - pb_acc
    if delta_agg > 0:
        print(f"  ★ AGGREGATOR 优势: +{delta_agg:.1f}% (相比 pick-best)")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
