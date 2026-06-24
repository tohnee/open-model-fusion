"""
open_fusion.cost - completion / token / latency accounting.

A Telemetry object threads through the whole pipeline. Each model completion that
contributes usage calls `add_usage` exactly once, so `completions` counts billed
completion units (panel survivors + judge attempt(s) + synthesis). The orchestrator
fills the panel-level fields directly; the summary() dict is what ships on
FusionResult.telemetry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import TokenUsage


@dataclass
class Telemetry:
    status: str = "unknown"
    completions: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_label: dict[str, dict[str, int]] = field(default_factory=dict)

    n_panel_ok: int = 0
    n_panel_failed: int = 0
    n_panel_cancelled: int = 0          # fast_majority_k 早退；不是失败
    panel_latencies_ms: list[int] = field(default_factory=list)
    panel_rounds: list[int] = field(default_factory=list)

    judge_ms: int = 0
    judge_retries: int = 0
    synthesis_ms: int = 0       # 合成阶段单次完成时延（不含 fallback 重写）

    def add_usage(self, label: str, usage: TokenUsage) -> None:
        self.completions += 1
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        slot = self.by_label.setdefault(label, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
        slot["prompt_tokens"] += usage.prompt_tokens
        slot["completion_tokens"] += usage.completion_tokens
        slot["calls"] += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def critical_path_ms(self) -> int:
        # wall-clock ~= 最慢 panel（fan-out 并行） + judge + synthesis。
        # 之前 synthesis 不单独计时；现在追加 synthesis_ms 后 critical_path 与
        # 真实的 wall-clock 端到端误差 < 一次解析/分发开销，使它能作为 baseline
        # 公平对比的口径（baseline 走一次完成 = 它自己的 wall-clock）。
        slowest_panel = max(self.panel_latencies_ms) if self.panel_latencies_ms else 0
        return slowest_panel + self.judge_ms + self.synthesis_ms

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "completions": self.completions,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "panel_ok": self.n_panel_ok,
            "panel_failed": self.n_panel_failed,
            "panel_cancelled": self.n_panel_cancelled,
            "panel_latencies_ms": self.panel_latencies_ms,
            "panel_rounds": self.panel_rounds,
            "judge_ms": self.judge_ms,
            "judge_retries": self.judge_retries,
            "synthesis_ms": self.synthesis_ms,
            "critical_path_ms": self.critical_path_ms,
            "by_label": self.by_label,
        }
