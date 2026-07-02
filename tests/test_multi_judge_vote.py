#!/usr/bin/env python3
"""多 Judge 投票模拟测试

构造 5 种 judge 意见不一致的场景, 验证投票逻辑是否按预期工作:
  1. 全一致 (3/3 选同一模型) → 高置信度共识
  2. 2-1 分裂 (2 选 A, 1 选 B) → 多数共识
  3. 全部分歧 (各选不同) → 无共识
  4. 含 null 票 (1 个 judge 选 null) → 有效票 2/3
  5. 全失败 (3 个 judge 都报错) → 降级

不调用真实 API, 纯模拟验证投票聚合逻辑。
"""
from __future__ import annotations

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from open_fusion.multi_judge import aggregate_votes, JudgeVote, JudgeVoteResult
from open_fusion.schema import Analysis


def make_analysis(best_model: str | None, best_reason: str = "test") -> Analysis:
    """构造一个带 best_model 的 Analysis。"""
    return Analysis(
        consensus=["test consensus"] if best_model else [],
        contradictions=[],
        partial_coverage=[],
        unique_insights=[],
        blind_spots=[],
        best_model=best_model,
        best_reason=best_reason,
    )


def make_vote(slug: str, best_model: str | None,
              best_reason: str = "test", error: str | None = None) -> JudgeVote:
    """构造一个 JudgeVote。"""
    analysis = None if error else make_analysis(best_model, best_reason)
    return JudgeVote(
        judge_slug=slug,
        best_model=best_model,
        best_reason=best_reason,
        analysis=analysis,
        error=error,
    )


# ============================================================================
# 5 种投票场景测试
# ============================================================================

def test_scenario_1_unanimous():
    """场景 1: 3 个 judge 全选 MODEL A → 100% 共识。"""
    votes = [
        make_vote("glm-5.2", "MODEL A", "most complete"),
        make_vote("kimi-k2.6", "MODEL A", "correct reasoning"),
        make_vote("deepseek-v4", "MODEL A", "best coverage"),
    ]
    result = aggregate_votes(votes)

    assert result.winner_model == "MODEL A"
    assert result.winner_vote_count == 3
    assert result.total_judges == 3
    assert result.agreement_ratio == 1.0
    assert result.is_consensus is True
    assert result.confidence == 1.0
    print("  场景 1 (全一致): ✓ winner=MODEL A, ratio=1.0, consensus=True")


def test_scenario_2_majority_split():
    """场景 2: 2 选 A, 1 选 B → 2/3 多数共识。"""
    votes = [
        make_vote("glm-5.2", "MODEL A", "correct"),
        make_vote("kimi-k2.6", "MODEL A", "best reasoning"),
        make_vote("deepseek-v4", "MODEL B", "alternative approach"),
    ]
    result = aggregate_votes(votes)

    assert result.winner_model == "MODEL A"
    assert result.winner_vote_count == 2
    assert result.agreement_ratio == 2/3
    assert result.is_consensus is True  # >= 2/3
    print(f"  场景 2 (2-1分裂): ✓ winner=MODEL A, ratio={result.agreement_ratio:.2f}, consensus=True")


def test_scenario_3_all_different():
    """场景 3: 3 个 judge 各选不同 → 无共识。"""
    votes = [
        make_vote("glm-5.2", "MODEL A"),
        make_vote("kimi-k2.6", "MODEL B"),
        make_vote("deepseek-v4", "MODEL C"),
    ]
    result = aggregate_votes(votes)

    # 票数最多的是 1 票 (并列), 取第一个
    assert result.winner_vote_count == 1
    assert result.agreement_ratio == 1/3
    assert result.is_consensus is False  # < 2/3
    print(f"  场景 3 (全分歧): ✓ winner={result.winner_model}, ratio={result.agreement_ratio:.2f}, consensus=False")


def test_scenario_4_null_votes():
    """场景 4: 1 个 judge 选 null, 2 个选 A → 有效票 2/2 一致。"""
    votes = [
        make_vote("glm-5.2", "MODEL A", "correct"),
        make_vote("kimi-k2.6", None, "no clear winner"),
        make_vote("deepseek-v4", "MODEL A", "best answer"),
    ]
    result = aggregate_votes(votes)

    assert result.winner_model == "MODEL A"
    assert result.winner_vote_count == 2
    assert result.total_judges == 3
    # ratio = 2/2 (null 不计入有效票)
    assert result.agreement_ratio == 1.0
    assert result.is_consensus is True
    print(f"  场景 4 (含null票): ✓ winner=MODEL A, ratio={result.agreement_ratio:.2f}, consensus=True")


def test_scenario_5_all_failed():
    """场景 5: 3 个 judge 全失败 → 降级, 无共识。"""
    votes = [
        make_vote("glm-5.2", None, error="JudgeError: invalid json"),
        make_vote("kimi-k2.6", None, error="timeout"),
        make_vote("deepseek-v4", None, error="connection error"),
    ]
    result = aggregate_votes(votes)

    assert result.winner_model is None
    assert result.winner_vote_count == 0
    assert result.agreement_ratio == 0.0
    assert result.is_consensus is False
    assert result.confidence == 0.0
    print(f"  场景 5 (全失败): ✓ winner=None, ratio=0.0, consensus=False")


# ============================================================================
# 边缘场景
# ============================================================================

def test_edge_single_judge():
    """边缘: 只有 1 个 judge → 1/1 = 100% (但不满足 2/3 阈值)。"""
    votes = [make_vote("glm-5.2", "MODEL A")]
    result = aggregate_votes(votes)

    assert result.winner_model == "MODEL A"
    assert result.agreement_ratio == 1.0
    # 1/1 = 100% >= 2/3, 所以 is_consensus = True
    assert result.is_consensus is True


def test_edge_two_judges_split():
    """边缘: 2 个 judge 各选不同 → 1/2 = 50% < 2/3, 无共识。"""
    votes = [
        make_vote("glm-5.2", "MODEL A"),
        make_vote("kimi-k2.6", "MODEL B"),
    ]
    result = aggregate_votes(votes)

    assert result.winner_vote_count == 1
    assert result.agreement_ratio == 0.5
    assert result.is_consensus is False


def test_edge_empty_votes():
    """边缘: 空投票列表。"""
    result = aggregate_votes([])
    assert result.winner_model is None
    assert result.total_judges == 0
    assert result.confidence == 0.0


def test_to_analysis_uses_winner():
    """to_analysis 返回胜出 judge 的 analysis。"""
    votes = [
        make_vote("glm-5.2", "MODEL A", "reason A"),
        make_vote("kimi-k2.6", "MODEL A", "reason B"),
    ]
    result = aggregate_votes(votes)
    analysis = result.to_analysis()

    assert analysis.best_model == "MODEL A"
    # 应使用第一个投胜出模型的 judge 的 reason
    assert analysis.best_reason == "reason A"


def test_to_analysis_no_consensus_returns_fallback():
    """无共识时 to_analysis 返回 fallback。"""
    votes = [
        make_vote("glm-5.2", "MODEL A"),
        make_vote("kimi-k2.6", "MODEL B"),
    ]
    result = aggregate_votes(votes)
    fallback = Analysis(consensus=["fallback"], best_model="MODEL C")
    analysis = result.to_analysis(fallback_analysis=fallback)

    # 无共识, 返回 fallback
    assert analysis.best_model == "MODEL C"


# ============================================================================
# 模拟 pick-best + 多 judge 的集成场景
# ============================================================================

def test_integration_pick_best_with_multi_judge():
    """集成场景: 多 judge 投票 → pick-best 决策。

    模拟 math_2 场景:
      - Judge A (弱 judge) 误选 MODEL B (错误答案)
      - Judge B (强 judge) 选 MODEL A (正确答案)
      - Judge C 选 MODEL A (正确答案)
      → 多数投票选 MODEL A (2/3), 避免了单 judge 误判
    """
    votes = [
        make_vote("weak-judge", "MODEL B", "seems reasonable"),
        make_vote("strong-judge", "MODEL A", "correct calculation 15/28"),
        make_vote("mid-judge", "MODEL A", "matches expected fraction format"),
    ]
    result = aggregate_votes(votes)

    # 多数投票选出 MODEL A (正确答案), 避免了单 judge 误选 MODEL B
    assert result.winner_model == "MODEL A"
    assert result.is_consensus is True
    assert result.confidence >= 2/3

    print(f"\n  集成场景 (math_2 模拟):")
    print(f"    Judge A (弱): MODEL B (错误)")
    print(f"    Judge B (强): MODEL A (正确)")
    print(f"    Judge C (中): MODEL A (正确)")
    print(f"    → 投票结果: winner=MODEL A, ratio={result.agreement_ratio:.2f}")
    print(f"    → 避免了单 judge 误判! (单 judge A 会选 MODEL B)")


def test_integration_multi_judge_plus_verifier():
    """集成场景: 多 judge 共识 + verifier 拦截错误格式。

    模拟: 3 judge 全选 MODEL A, 但 MODEL A 的答案格式不对
      → 多 judge 高置信度, 但 verifier 否决 → 回退 synth
    """
    from open_fusion.verifier import verify_answer, VerifierVerdict

    votes = [
        make_vote("glm-5.2", "MODEL A"),
        make_vote("kimi-k2.6", "MODEL A"),
        make_vote("deepseek-v4", "MODEL A"),
    ]
    result = aggregate_votes(votes)

    # 多 judge 全选 MODEL A → 高置信度
    assert result.is_consensus is True
    assert result.confidence == 1.0

    # 但 MODEL A 的答案是 "大约是 0.5 左右" (缺少精确分数)
    model_a_answer = "大约是 0.5 左右"
    task_meta = {
        "domain": "数学",
        "expected_format": "fraction",
        "expected_range": (0.5, 0.6),
    }
    vr = verify_answer(model_a_answer, task_meta)

    # Verifier 应否决 (期望 fraction 格式, 但答案是模糊描述)
    assert vr.verdict == VerifierVerdict.FAIL
    assert vr.should_fallback is True

    print(f"\n  集成场景 (多judge + verifier):")
    print(f"    3 Judge 全选 MODEL A → consensus=True, confidence={result.confidence:.2f}")
    print(f"    Verifier 检查答案 '{model_a_answer}'")
    print(f"    → Verdict={vr.verdict.value}, reason={vr.reason}")
    print(f"    → 双重保障: verifier 否决, 回退到 synthesizer")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 72)
    print("多 Judge 投票模拟测试")
    print("=" * 72)

    print("\n--- 5 种投票场景 ---\n")
    test_scenario_1_unanimous()
    test_scenario_2_majority_split()
    test_scenario_3_all_different()
    test_scenario_4_null_votes()
    test_scenario_5_all_failed()

    print("\n--- 边缘场景 ---\n")
    test_edge_single_judge()
    print("  边缘 (单judge): ✓")
    test_edge_two_judges_split()
    print("  边缘 (2judge分裂): ✓")
    test_edge_empty_votes()
    print("  边缘 (空列表): ✓")
    test_to_analysis_uses_winner()
    print("  边缘 (to_analysis): ✓")
    test_to_analysis_no_consensus_returns_fallback()
    print("  边缘 (to_analysis fallback): ✓")

    print("\n--- 集成场景 ---\n")
    test_integration_pick_best_with_multi_judge()
    test_integration_multi_judge_plus_verifier()

    print(f"\n{'=' * 72}")
    print("全部通过!")
    print(f"{'=' * 72}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
