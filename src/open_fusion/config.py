"""
open_fusion.config - ModelSpec / Params / FusionConfig + presets + CLI parsing.

Re-exports `Phase` from schema so callers may import it from either module and get
the *same* enum object (avoids the classic two-Phase-enums identity bug).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .schema import Phase  # re-exported; single source of truth

# An inner (nested) fusion call must not re-trigger fusion. depth 0 is the only
# layer allowed to run; depth >= MAX is refused by the orchestrator.
MAX_FUSION_DEPTH = 1


@dataclass
class ModelSpec:
    """One model on the gateway. `slug` is the gateway's identifier, e.g.
    'anthropic/claude-opus-4.1'. Optional per-model overrides beat phase Params."""
    slug: str
    temperature: float | None = None
    max_tokens: int | None = None

    @property
    def vendor(self) -> str:
        return self.slug.split("/", 1)[0] if "/" in self.slug else self.slug


@dataclass
class Params:
    """Per-phase sampling/budget knobs."""
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout_s: float = 90.0
    max_tool_calls: int = 4

    def __post_init__(self) -> None:
        # 这些下界是正确性约束，不是风格：max_tool_calls=-1 会让 panel._run_one 的
        # tool loop 跑 0 次、`comp` 未定义；timeout_s<=0 会让 urllib 立即报错并被
        # 误判为模型挂掉；max_tokens<=0 会让上游 400 反复重试。把校验前置到构造点。
        if self.max_tool_calls < 0:
            raise ValueError(f"max_tool_calls must be >= 0, got {self.max_tool_calls}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")
        if self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {self.timeout_s}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")


def _default_params() -> dict[Phase, Params]:
    # judge/synthesis run cooler than the panel: divergence in fan-out, discipline
    # in distillation and writing.
    return {
        Phase.PANEL: Params(temperature=0.7, max_tokens=2048, timeout_s=90.0, max_tool_calls=4),
        Phase.JUDGE: Params(temperature=0.2, max_tokens=3072, timeout_s=90.0, max_tool_calls=3),
        Phase.SYNTHESIS: Params(temperature=0.3, max_tokens=3072, timeout_s=90.0, max_tool_calls=0),
    }


@dataclass
class FusionConfig:
    panel: list[ModelSpec]
    judge: ModelSpec
    caller: ModelSpec | None = None         # the model that invoked fusion; defaults to judge
    params: dict[Phase, Params] = field(default_factory=_default_params)
    depth: int = 0
    tools_enabled: bool = False
    max_in_flight: int = 8
    excluded_domains: list[str] = field(default_factory=list)
    fast_majority_k: int | None = None
    base_url: str | None = None
    api_key: str | None = None

    def __post_init__(self) -> None:
        if self.caller is None:
            self.caller = self.judge
        if not self.params:
            self.params = _default_params()

    def validate(self) -> list[str]:
        """Hard problems block the run; 'WARNING:'-prefixed advisories don't."""
        problems: list[str] = []
        if not self.panel:
            problems.append("panel is empty: need at least one panel model")
        if self.judge is None or not self.judge.slug:
            problems.append("judge model is required")
        if self.depth < 0:
            problems.append("depth must be >= 0")
        if self.max_in_flight < 1:
            problems.append("max_in_flight must be >= 1")
        # Heterogeneity is the mechanism (M2): same-vendor panels give weaker lift.
        vendors = {m.vendor for m in self.panel}
        if self.panel and len(vendors) == 1:
            problems.append(
                f"WARNING: homogeneous panel (all '{next(iter(vendors))}'); "
                "heterogeneous panels deliberate better")
        return problems


# --- presets (source of truth; config/presets.yaml mirrors these for reference) --
# Slugs are OpenRouter-style and drift over time -- verify at openrouter.ai/models
# or override with --panel/--judge.
_PRESETS: dict[str, dict] = {
    # Mirrors the panels OpenRouter benchmarked on DRACO (Fusion beats frontier,
    # 2026-06-12). The synthesizer/judge there was Opus 4.8.
    "quality": {
        "panel": ["anthropic/claude-fable-5", "openai/gpt-5.5"],
        "judge": "anthropic/claude-opus-4.8",
    },
    "budget": {
        "panel": ["google/gemini-3-flash-preview", "moonshotai/kimi-k2.6",
                  "deepseek/deepseek-v4-pro"],
        # Article benchmarked this panel synthesized by Opus 4.8; for an actually
        # cheap default we judge with the strongest budget reasoner instead.
        "judge": "deepseek/deepseek-v4-pro",
    },
}


def load_preset(name: str, **overrides) -> FusionConfig:
    if name not in _PRESETS:
        raise ValueError(f"unknown preset '{name}'; choose from {sorted(_PRESETS)}")
    p = _PRESETS[name]
    cfg = FusionConfig(
        panel=[ModelSpec(s) for s in p["panel"]],
        judge=ModelSpec(p["judge"]),
    )
    # 性能优化 O1：3+ panel 默认允许扔最慢一个（cancelled，不算失败）。
    # 异质性 (M2) 至少保留 2 条独立推理路径，不破坏 Fusion 的核心机制；
    # 2-panel 不启用：扔掉一个就只剩单模型，失去 Fusion 的意义。
    # 用户可通过 --fast-majority-k 0/N 覆盖。
    if len(cfg.panel) >= 3 and cfg.fast_majority_k is None:
        cfg.fast_majority_k = len(cfg.panel) - 1
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def from_plugin(spec: dict, *, depth: int = 0) -> FusionConfig:
    """Build a FusionConfig from OpenRouter's fusion *plugin* shape:

        {"id": "fusion",
         "model": "<synthesizer/judge slug>",
         "analysis_models": ["<panel slug>", ...]}

    In OpenRouter the `model` is the one that fuses the results (judge + caller);
    `analysis_models` is the panel. We map those directly so panels authored for
    the hosted Fusion plugin run unchanged here.
    """
    panel = [ModelSpec(s) for s in (spec.get("analysis_models") or [])]
    fuser = ModelSpec(spec["model"]) if spec.get("model") else (panel[0] if panel else None)
    if not panel or fuser is None:
        raise ValueError("fusion plugin spec needs 'analysis_models' and 'model'")
    return FusionConfig(panel=panel, judge=fuser, caller=fuser, depth=depth)


def from_cli(args) -> FusionConfig:
    """Build a FusionConfig from parsed argparse Namespace (see cli.py)."""
    if getattr(args, "panel", None):
        panel = [ModelSpec(s.strip()) for s in args.panel.split(",") if s.strip()]
        judge = ModelSpec((args.judge or panel[0].slug))
        cfg = FusionConfig(panel=panel, judge=judge)
        # 自定义 panel 也应享受 O1：3+ panel 默认 fast_majority_k = N-1。
        # 与 load_preset 保持一致的策略（异质性至少保留 2 条独立路径）。
        if len(cfg.panel) >= 3 and cfg.fast_majority_k is None:
            cfg.fast_majority_k = len(cfg.panel) - 1
    else:
        cfg = load_preset(getattr(args, "preset", None) or "quality")
        if getattr(args, "judge", None):
            cfg.judge = ModelSpec(args.judge)
            cfg.caller = cfg.judge

    cfg.tools_enabled = bool(getattr(args, "tools", False))
    if getattr(args, "exclude_domains", None):
        cfg.excluded_domains = [d.strip() for d in args.exclude_domains.split(",") if d.strip()]
    if getattr(args, "max_in_flight", None):
        cfg.max_in_flight = args.max_in_flight
    # 用 is not None 而非 truthy，让 `--fast-majority-k 0` 能显式关闭早退。
    if getattr(args, "fast_majority_k", None) is not None:
        cfg.fast_majority_k = args.fast_majority_k if args.fast_majority_k > 0 else None
    cfg.base_url = os.getenv("OPEN_FUSION_BASE_URL")
    cfg.api_key = os.getenv("OPEN_FUSION_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    return cfg
