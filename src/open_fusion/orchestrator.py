"""
open_fusion.orchestrator - the top-level pipeline.

Wires the state machine: depth guard -> FAN_OUT -> {ALL_PANEL_FAILED | JUDGE}
-> {JUDGE_FAILED -> fallback | SYNTHESIZE} -> DONE. See ARCHITECTURE.md section 5.

可观测性：所有状态机迁移点都打了结构化日志（默认静默，受 OPEN_FUSION_LOG 控制）。
日志字段包括 phase / event / duration_ms / 输入输出 size，便于定位 timing 问题。
"""
from __future__ import annotations

import time

from . import judge as judge_mod
from . import panel as panel_mod
from . import synthesizer as synth_mod
from ._logging import log_event, timed_phase
from .client import ModelClient
from .config import FusionConfig, MAX_FUSION_DEPTH, Phase
from .cost import Telemetry
from .judge import JudgeError
from .schema import FusionResult, FusionStatus


async def fuse(question: str, config: FusionConfig, *, client: ModelClient | None = None) -> FusionResult:
    """Run the full Fusion pipeline for one question.

    Pass a custom `client` (e.g. a fake) for testing; otherwise a real
    ModelClient is built from config/env.
    """
    problems = [p for p in config.validate() if not p.startswith("WARNING")]
    if problems:
        raise ValueError("invalid config: " + "; ".join(problems))

    tel = Telemetry()
    t_run0 = time.monotonic()
    log_event("orchestrator", "start",
              question_chars=len(question), n_panel=len(config.panel),
              judge=config.judge.slug, caller=(config.caller.slug if config.caller else None),
              tools_enabled=config.tools_enabled, depth=config.depth)

    # --- recursion guard ----------------------------------------------------
    # Depth守卫是双保险：既看 config.depth（调用方声明），也看 client.fusion_depth
    # （上游已发起 fusion 后传入的 client）。任一达到上限即拒绝。这条不变量被
    # ARCHITECTURE.md §6 第 4 条声明为"depth 守卫 + 头 stamp"双保险。
    effective_depth = max(config.depth, getattr(client, "fusion_depth", 0)) if client else config.depth
    if effective_depth >= MAX_FUSION_DEPTH:
        tel.status = "depth_exceeded"
        log_event("orchestrator", "depth_exceeded", effective_depth=effective_depth,
                  max_depth=MAX_FUSION_DEPTH)
        return FusionResult(status=FusionStatus.ERROR,
                            error="fusion depth exceeded; answer without fusion",
                            telemetry=tel.summary())

    if client is None:
        client = ModelClient(fusion_depth=config.depth + 1,
                             base_url=config.base_url, api_key=config.api_key)

    # --- FAN_OUT ------------------------------------------------------------
    with timed_phase("panel", n_panel=len(config.panel),
                     max_in_flight=config.max_in_flight,
                     fast_majority_k=config.fast_majority_k) as ctx:
        responses = await panel_mod.run(
            client, config.panel, question, config.params[Phase.PANEL],
            tools_enabled=config.tools_enabled,
            max_in_flight=config.max_in_flight,
            excluded_domains=config.excluded_domains,
            fast_majority_k=config.fast_majority_k,
        )
        ctx["n_ok"] = sum(1 for r in responses if r.ok)
        ctx["n_failed"] = sum(1 for r in responses if not r.ok and not r.cancelled)
        ctx["n_cancelled"] = sum(1 for r in responses if r.cancelled)

    tel.n_panel_ok = sum(1 for r in responses if r.ok)
    # cancelled（fast_majority_k 早退）不算 failed：它是策略选择，不是模型故障。
    tel.n_panel_cancelled = sum(1 for r in responses if r.cancelled)
    tel.n_panel_failed = len(responses) - tel.n_panel_ok - tel.n_panel_cancelled
    tel.panel_latencies_ms = [r.latency_ms for r in responses if r.ok]
    tel.panel_rounds = [max(1, len(r.tool_trace) + 1) for r in responses if r.ok]
    for r in responses:
        tel.add_usage(f"panel:{r.model}", r.usage)

    # --- ALL_PANEL_FAILED ---------------------------------------------------
    if tel.n_panel_ok == 0:
        tel.status = "all_panel_failed"
        first_err = next((r.error for r in responses if r.error), "unknown")
        log_event("orchestrator", "all_failed",
                  total_duration_ms=int((time.monotonic() - t_run0) * 1000),
                  first_error=first_err)
        return FusionResult(status=FusionStatus.ERROR, panel_responses=responses,
                            error=f"all panel models failed (e.g. {first_err})",
                            telemetry=tel.summary())

    # --- JUDGE --------------------------------------------------------------
    try:
        analysis = await judge_mod.synthesize(
            client, config.judge, question, responses, config.params[Phase.JUDGE],
            tools_enabled=config.tools_enabled, tel=tel)
    except JudgeError:
        # --- JUDGE_FAILED: degrade to raw-response synthesis ----------------
        # The CALLER (calling model) writes the final answer, per the Fusion design;
        # caller defaults to the judge when not set separately.
        log_event("orchestrator", "fallback", reason="judge_invalid_json_twice")
        text = await synth_mod.write_fallback(
            client, config.caller, question, responses,
            config.params[Phase.SYNTHESIS], tel=tel)
        tel.status = "judge_fallback"
        log_event("orchestrator", "end", status="judge_fallback",
                  total_duration_ms=int((time.monotonic() - t_run0) * 1000),
                  critical_path_ms=tel.critical_path_ms, answer_chars=len(text))
        return FusionResult(status=FusionStatus.JUDGE_FALLBACK, text=text,
                            panel_responses=responses, telemetry=tel.summary())

    # --- SYNTHESIZE ---------------------------------------------------------
    # "The calling model then writes the final answer grounded in that analysis."
    text = await synth_mod.write(
        client, config.caller, question, analysis, responses,
        config.params[Phase.SYNTHESIS], tel=tel)
    tel.status = "ok"
    log_event("orchestrator", "end", status="ok",
              total_duration_ms=int((time.monotonic() - t_run0) * 1000),
              critical_path_ms=tel.critical_path_ms,
              panel_max_ms=max(tel.panel_latencies_ms) if tel.panel_latencies_ms else 0,
              judge_ms=tel.judge_ms, synthesis_ms=tel.synthesis_ms,
              total_tokens=tel.total_tokens, answer_chars=len(text))
    return FusionResult(status=FusionStatus.OK, text=text, analysis=analysis,
                        panel_responses=responses, telemetry=tel.summary())
