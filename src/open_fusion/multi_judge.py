"""
open_fusion.multi_judge - v1.3 P1: 多 Judge 并行投票

多个 judge 独立分析同一组 panel 回答, 通过多数投票选出 best_model。
降低单 judge 误判概率: 3 个独立 judge 同时误判的概率 ≈ 0.3³ = 2.7%。

架构:
    Panel → ┌─ Judge A → "MODEL A" ─┐
            ├─ Judge B → "MODEL C" ─┤→ 投票聚合 → best_model
            └─ Judge C → "MODEL A" ─┘
                                        2/3 票 → "MODEL A" (高置信度)
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .client import ModelClient
from .config import ModelSpec, Params
from .cost import Telemetry
from .judge import synthesize as _single_judge, JudgeError
from .prompts import JUDGE_USER, label_responses
from .schema import Analysis, PanelResponse, Phase


@dataclass
class JudgeVote:
    """单个 judge 的投票结果。"""
    judge_slug: str
    best_model: str | None       # "MODEL A" | None
    best_reason: str | None
    analysis: Analysis | None = None
    error: str | None = None     # judge 调用失败时的错误信息

    @property
    def ok(self) -> bool:
        return self.error is None and self.analysis is not None


@dataclass
class JudgeVoteResult:
    """多 judge 投票聚合结果。"""
    votes: list[JudgeVote] = field(default_factory=list)
    winner_model: str | None = None
    winner_vote_count: int = 0
    total_judges: int = 0
    agreement_ratio: float = 0.0

    @property
    def is_consensus(self) -> bool:
        """是否达成多数共识 (>= 2/3 一致)。"""
        return self.agreement_ratio >= 2/3 and self.winner_model is not None

    @property
    def confidence(self) -> float:
        """聚合置信度: agreement_ratio (简化版, 不含 judge 自评)。"""
        if self.total_judges == 0:
            return 0.0
        return self.agreement_ratio

    def to_analysis(self, fallback_analysis: Analysis | None = None) -> Analysis:
        """将投票结果转为 Analysis 对象。

        仅在达成共识时使用胜出 judge 的 analysis。
        无共识时返回 fallback_analysis 或空 Analysis。
        """
        if self.is_consensus and self.winner_model and self.votes:
            # 找到投胜出模型的第一个 judge 的 analysis
            for v in self.votes:
                if v.best_model and v.best_model.strip().upper() == self.winner_model and v.analysis:
                    return v.analysis
        return fallback_analysis or Analysis()


async def run_multi_judge(
    client: ModelClient,
    judges: list[ModelSpec],
    question: str,
    responses: list[PanelResponse],
    params: Params,
    *,
    tel: Telemetry | None = None,
) -> JudgeVoteResult:
    """并行调用多个 judge, 聚合投票结果。

    Args:
        judges: judge 模型列表 (>= 2 个)
        question: 原始问题
        responses: panel 回答列表
        params: judge 阶段参数

    Returns:
        JudgeVoteResult: 投票聚合结果
    """
    tasks = [
        _run_single_judge_safe(client, judge, question, responses, params, tel, i)
        for i, judge in enumerate(judges)
    ]
    raw_votes = await asyncio.gather(*tasks)

    return aggregate_votes(list(raw_votes))


async def _run_single_judge_safe(
    client: ModelClient,
    judge: ModelSpec,
    question: str,
    responses: list[PanelResponse],
    params: Params,
    tel: Telemetry | None,
    index: int,
) -> JudgeVote:
    """调用单个 judge, 失败时返回 error 而非抛异常。"""
    try:
        analysis = await _single_judge(
            client, judge, question, responses, params,
            tools_enabled=False, tel=tel)
        return JudgeVote(
            judge_slug=judge.slug,
            best_model=analysis.best_model,
            best_reason=analysis.best_reason,
            analysis=analysis,
        )
    except (JudgeError, Exception) as e:
        return JudgeVote(
            judge_slug=judge.slug,
            best_model=None,
            best_reason=None,
            error=str(e),
        )


def aggregate_votes(votes: list[JudgeVote]) -> JudgeVoteResult:
    """聚合多 judge 的投票结果 (多数投票)。

    规则:
      - 统计每个 best_model 的票数 (None 不计入)
      - 票数最多的为 winner
      - agreement_ratio = winner_vote_count / total_valid_votes
      - 所有 judge 都失败 → winner=None, confidence=0
    """
    valid_votes = [v for v in votes if v.ok and v.best_model]
    total = len(valid_votes)

    if total == 0:
        return JudgeVoteResult(
            votes=votes,
            winner_model=None,
            winner_vote_count=0,
            total_judges=len(votes),
            agreement_ratio=0.0,
        )

    # 统计票数
    tally: dict[str, int] = {}
    for v in valid_votes:
        label = v.best_model.strip().upper() if v.best_model else ""
        tally[label] = tally.get(label, 0) + 1

    # 找胜出者
    winner = max(tally, key=tally.get)
    winner_count = tally[winner]
    ratio = winner_count / total

    return JudgeVoteResult(
        votes=votes,
        winner_model=winner,
        winner_vote_count=winner_count,
        total_judges=len(votes),
        agreement_ratio=ratio,
    )
