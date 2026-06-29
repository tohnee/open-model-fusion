"""
open_fusion.schema - the data contract. The only module everything else imports.

Defines the pipeline's enums (Phase, FusionStatus) and the typed records that flow
between phases (TokenUsage, PanelResponse, Analysis, FusionResult). Kept dependency-
free (stdlib only, no imports from sibling modules) so it sits at the bottom of the
import graph and can never create a cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# --- pipeline phases ---------------------------------------------------------
class Phase(str, Enum):
    """The three deliberation phases. Used as keys into per-phase Params and to
    gate which toolset each phase may use (see tools.toolset_for_phase)."""
    PANEL = "panel"
    JUDGE = "judge"
    SYNTHESIS = "synthesis"


class FusionStatus(str, Enum):
    """Terminal status of a run. `.value` is the stable string surfaced in JSON."""
    OK = "ok"
    ERROR = "error"
    JUDGE_FALLBACK = "judge_fallback"
    CONSENSUS_SHORTCUT = "consensus_shortcut"
    PICK_BEST_SHORTCUT = "pick_best_shortcut"
    AGGREGATOR_MODE = "aggregator_mode"


# --- token accounting --------------------------------------------------------
@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens}


# --- one panel model's result (panel.py never raises; it returns this) -------
@dataclass
class PanelResponse:
    model: str
    content: str = ""
    status: str = "ok"          # ok | timeout | error | cancelled
    error: str | None = None
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def cancelled(self) -> bool:
        """主动早退（fast_majority_k）。不是真实失败，不计入 panel_failed。"""
        return self.status == "cancelled"

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "status": self.status,
                "content": self.content, "error": self.error,
                "tool_trace": self.tool_trace, "latency_ms": self.latency_ms,
                "usage": self.usage.to_dict()}


# --- the structured intermediate representation the judge produces -----------
@dataclass
class Analysis:
    """Typed distillation of the panel. The whole system's core lever: forcing the
    judge to commit to a schema (with per-model attribution) is what turns N opaque
    answers into something a synthesizer can reason over and audit."""
    consensus: list[str] = field(default_factory=list)
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    partial_coverage: list[dict[str, Any]] = field(default_factory=list)
    unique_insights: list[dict[str, Any]] = field(default_factory=list)
    blind_spots: list[str] = field(default_factory=list)
    best_model: str | None = None       # judge-identified strongest model label (e.g. "MODEL C")
    best_reason: str | None = None      # why best_model was chosen

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Analysis":
        d = d or {}
        return cls(
            consensus=list(d.get("consensus") or []),
            contradictions=list(d.get("contradictions") or []),
            partial_coverage=list(d.get("partial_coverage") or []),
            unique_insights=list(d.get("unique_insights") or []),
            blind_spots=list(d.get("blind_spots") or []),
            best_model=d.get("best_model"),
            best_reason=d.get("best_reason"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"consensus": self.consensus, "contradictions": self.contradictions,
                "partial_coverage": self.partial_coverage,
                "unique_insights": self.unique_insights, "blind_spots": self.blind_spots,
                "best_model": self.best_model, "best_reason": self.best_reason}

    def validate(self) -> list[str]:
        """Return a list of problems. Strings starting with 'WARNING' are advisory
        (callers filter them out before treating the analysis as invalid)."""
        problems: list[str] = []

        if not any([self.consensus, self.contradictions, self.partial_coverage,
                    self.unique_insights, self.blind_spots]):
            problems.append("empty analysis: no sections populated")

        # per-model attribution is a hard correctness rule: every stance and every
        # unique insight must name the model it came from (this is the audit trail).
        for i, c in enumerate(self.contradictions):
            if not isinstance(c, dict):
                problems.append(f"contradiction[{i}] is not an object")
                continue
            stances = c.get("stances") or []
            if not stances:
                problems.append(f"contradiction[{i}] has no stances")
            for j, s in enumerate(stances):
                if not (isinstance(s, dict) and s.get("model")):
                    problems.append(
                        f"contradiction[{i}].stances[{j}] missing model attribution")

        for k, u in enumerate(self.unique_insights):
            if not (isinstance(u, dict) and u.get("model")):
                problems.append(f"unique_insights[{k}] missing model attribution")

        return problems


# --- the final object the orchestrator returns -------------------------------
@dataclass
class FusionResult:
    status: FusionStatus
    text: str = ""
    analysis: Analysis | None = None
    panel_responses: list[PanelResponse] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, FusionStatus) else self.status,
            "text": self.text,
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "panel_responses": [r.to_dict() for r in self.panel_responses],
            "telemetry": self.telemetry,
            "error": self.error,
        }
