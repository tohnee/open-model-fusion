"""
open_fusion.eval.demo - a zero-network client for demonstrating the eval harness.

DemoClient routes through the real pipeline (panel -> judge -> synthesis -> baseline)
but returns canned answers per task so the whole evaluation, scoring, and verdict can
be produced with no API key. It distinguishes calls by role:
  - JSON-mode call            -> the judge: return a fixed valid Analysis.
  - system prompt = SYNTHESIZER -> the caller writing the fused answer: task `fused`.
  - otherwise, by model slug   -> panel members get `panel`; the baseline slug gets
                                  `baseline` (so Fusion and the solo baseline diverge).

可选的"模拟时延"用环境变量 OPEN_FUSION_DEMO_LATENCY_MS_<PHASE> 启用（PHASE ∈
PANEL/JUDGE/SYNTH）。仅用于演示/回归验证延迟拆解通路（critical_path = panel_max +
judge + synthesis），不影响离线测试的快速性（默认全为 0）。
"""
from __future__ import annotations

import asyncio
import json
import os

from ..client import Completion
from ..schema import TokenUsage

_ANALYSIS = json.dumps({
    "consensus": ["the panel broadly agrees on the core recommendation"],
    "contradictions": [{"topic": "emphasis",
                        "stances": [{"model": "MODEL A", "stance": "cost-first"},
                                    {"model": "MODEL B", "stance": "risk-first"}]}],
    "partial_coverage": [{"models": ["MODEL A"], "point": "edge cases"}],
    "unique_insights": [{"model": "MODEL B", "insight": "a concrete actionable default"}],
    "blind_spots": ["long-horizon effects"],
})


def _env_ms(name: str) -> float:
    """读取毫秒级模拟时延环境变量；非法值视为 0，保证默认零开销。"""
    try:
        v = float(os.getenv(name, "0") or "0")
        return max(0.0, v) / 1000.0
    except ValueError:
        return 0.0


class DemoClient:
    def __init__(self, tasks, baseline_slug: str) -> None:
        self._by_prompt = {t.prompt[:48]: t for t in tasks}
        self._tasks = tasks
        self.baseline_slug = baseline_slug
        self.fusion_depth = 0
        self.calls: list[dict] = []
        # 三段模拟时延（秒）。默认 0；测试或演示时通过 env 注入。
        self._panel_delay_s = _env_ms("OPEN_FUSION_DEMO_LATENCY_MS_PANEL")
        self._judge_delay_s = _env_ms("OPEN_FUSION_DEMO_LATENCY_MS_JUDGE")
        self._synth_delay_s = _env_ms("OPEN_FUSION_DEMO_LATENCY_MS_SYNTH")

    def _find_task(self, messages):
        text = " ".join(m.get("content", "") for m in messages
                         if isinstance(m.get("content"), str))
        for t in self._tasks:
            if t.prompt[:48] in text:
                return t
        return None

    async def complete(self, model, messages, *, tools=(), params=None, response_format=None):
        self.calls.append({"model": getattr(model, "slug", model),
                           "response_format": response_format})
        usage = TokenUsage(prompt_tokens=10, completion_tokens=10)
        slug = getattr(model, "slug", model)

        # 按阶段注入模拟时延（JSON 模式 = judge；system 含 SYNTHESIZER = 合成；
        # 其它 = panel/baseline）。判定顺序与下方文本分发一致，避免误归类。
        is_synth = any("SYNTHESIZER" in (m.get("content") or "")
                       for m in messages if m.get("role") == "system")
        if response_format == "json":
            if self._judge_delay_s > 0:
                await asyncio.sleep(self._judge_delay_s)
            return Completion(content=_ANALYSIS, usage=usage, model=slug,
                              raw_message={"role": "assistant", "content": _ANALYSIS})
        elif is_synth:
            if self._synth_delay_s > 0:
                await asyncio.sleep(self._synth_delay_s)
        else:
            if self._panel_delay_s > 0:
                await asyncio.sleep(self._panel_delay_s)

        task = self._find_task(messages)
        if task is None:
            content = "no demo answer available"
        elif is_synth:
            content = task.demo["fused"]
        elif slug == self.baseline_slug:
            content = task.demo["baseline"]
        else:
            content = task.demo.get("panel", task.demo["fused"])

        return Completion(content=content, usage=usage, model=slug,
                          raw_message={"role": "assistant", "content": content})
