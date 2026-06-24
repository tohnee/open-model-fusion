"""
open_fusion - an open, vendor-neutral implementation of the Fusion pattern
(single-layer Mixture-of-Agents with a structured-JSON judge step).

Public API:
    fuse(question, config, *, client=None) -> FusionResult   # the pipeline
    load_preset("quality" | "budget") -> FusionConfig
    FusionConfig, ModelSpec, Params, Phase                    # configuration
    FusionResult, FusionStatus, Analysis, PanelResponse       # results
"""
from __future__ import annotations

from .config import FusionConfig, ModelSpec, Params, Phase, load_preset, MAX_FUSION_DEPTH
from .orchestrator import fuse
from .schema import Analysis, FusionResult, FusionStatus, PanelResponse, TokenUsage

__all__ = [
    "fuse",
    "load_preset",
    "FusionConfig",
    "ModelSpec",
    "Params",
    "Phase",
    "MAX_FUSION_DEPTH",
    "FusionResult",
    "FusionStatus",
    "Analysis",
    "PanelResponse",
    "TokenUsage",
]

__version__ = "1.0.0"
