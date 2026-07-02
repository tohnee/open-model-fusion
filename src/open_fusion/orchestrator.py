"""
open_fusion.orchestrator - the top-level pipeline.

Wires the state machine: depth guard -> FAN_OUT -> {ALL_PANEL_FAILED | JUDGE}
-> {JUDGE_FAILED -> fallback | SYNTHESIZE} -> DONE. See ARCHITECTURE.md section 5.

可观测性：所有状态机迁移点都打了结构化日志（默认静默，受 OPEN_FUSION_LOG 控制）。
日志字段包括 phase / event / duration_ms / 输入输出 size，便于定位 timing 问题。
"""
from __future__ import annotations

import logging
import time
from difflib import SequenceMatcher

from . import judge as judge_mod
from . import panel as panel_mod
from . import synthesizer as synth_mod
from ._logging import log_event, timed_phase, is_enabled_for
from .client import ModelClient
from .config import FusionConfig, MAX_FUSION_DEPTH, Phase
from .cost import Telemetry
from .judge import JudgeError
from .schema import FusionResult, FusionStatus, PanelResponse


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
        panel_size = len(config.panel)
        executor_workers = max(panel_size * 2, 8) if panel_size >= 4 else None
        client = ModelClient(fusion_depth=config.depth + 1,
                             base_url=config.base_url, api_key=config.api_key,
                             executor_workers=executor_workers)

    # --- FAN_OUT ------------------------------------------------------------
    # P1-A: Panel 裁剪输入 — 当 enable_panel_trim 时, 截断超长 prompt 省 token。
    panel_prompt = question
    if config.enable_panel_trim and len(question) > config.panel_trim_chars:
        panel_prompt = question[:config.panel_trim_chars] + "\n...(truncated)"
        log_event("orchestrator", "panel_trimmed",
                  original_chars=len(question), trimmed_chars=len(panel_prompt),
                  trim_ratio=round(len(panel_prompt) / max(len(question), 1), 2),
                  saved_chars=len(question) - len(panel_prompt))

    # 性能日志: panel 阶段开始
    log_event("orchestrator", "fan_out_start",
              n_panel=len(config.panel), mode=config.mode.value,
              tools_enabled=config.tools_enabled,
              max_in_flight=config.max_in_flight,
              fast_majority_k=config.fast_majority_k,
              panel_trim=config.enable_panel_trim,
              prompt_chars=len(panel_prompt))

    with timed_phase("panel", n_panel=len(config.panel),
                     max_in_flight=config.max_in_flight,
                     fast_majority_k=config.fast_majority_k) as ctx:
        responses = await panel_mod.run(
            client, config.panel, panel_prompt, config.params[Phase.PANEL],
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

    # 性能日志: panel 阶段完成
    panel_duration_ms = int((time.monotonic() - t_run0) * 1000)
    log_event("orchestrator", "fan_out_done",
              n_ok=tel.n_panel_ok, n_failed=tel.n_panel_failed,
              n_cancelled=tel.n_panel_cancelled,
              panel_duration_ms=panel_duration_ms,
              total_panel_tokens=tel.total_tokens,
              max_panel_latency_ms=max(tel.panel_latencies_ms) if tel.panel_latencies_ms else 0,
              response_chars=[len(r.content) for r in responses if r.ok])

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

    # --- CONSENSUS SHORT-CIRCUIT -------------------------------------------
    # 改进3：当多数模型答案高度一致时，直接采用多数答案，跳过 judge+synthesis。
    # 这节省了 2 次 API 调用，且避免了 synthesizer "折中" 破坏正确答案。
    if config.enable_consensus_shortcut:
        ok_responses = [r for r in responses if r.ok and r.content.strip()]
        if len(ok_responses) >= 2:
            consensus_resp = _check_consensus(ok_responses, config.consensus_threshold)
            if consensus_resp is not None:
                tel.status = "consensus_shortcut"
                log_event("orchestrator", "consensus_shortcut",
                          model=consensus_resp.model,
                          answer_chars=len(consensus_resp.content),
                          total_duration_ms=int((time.monotonic() - t_run0) * 1000))
                return FusionResult(
                    status=FusionStatus.CONSENSUS_SHORTCUT,
                    text=consensus_resp.content.strip(),
                    panel_responses=responses,
                    telemetry=tel.summary(),
                )

    # --- P0-A: AGGREGATOR MODE (MoA 风格: 跳过 judge, 直接合成) --------------
    # 当 mode=AGGREGATOR 时, 不调用 judge, panel 回答直接给 synthesizer。
    # 这是 Hermes MoA 的核心设计: 无中间层, 信息无损, 省 1 次 API 调用。
    from .config import FusionMode
    if config.mode == FusionMode.AGGREGATOR:
        # 性能日志: AGGREGATOR 分支入口
        log_event("orchestrator", "aggregator_mode_entry",
                  n_responses=tel.n_panel_ok,
                  panel_duration_ms=panel_duration_ms,
                  caller=config.caller.slug,
                  saved_api_calls=1,  # 相比 FULL 模式省了 judge 调用
                  total_response_chars=sum(len(r.content) for r in responses if r.ok))

        t_agg0 = time.monotonic()
        text = await synth_mod.write_fallback(
            client, config.caller, question, responses,
            config.params[Phase.SYNTHESIS], tel=tel)
        agg_synth_ms = int((time.monotonic() - t_agg0) * 1000)
        total_ms = int((time.monotonic() - t_run0) * 1000)

        tel.status = "aggregator_mode"
        # 性能日志: AGGREGATOR 分支完成
        log_event("orchestrator", "aggregator_mode_done",
                  synth_ms=agg_synth_ms,
                  total_ms=total_ms,
                  answer_chars=len(text),
                  synthesis_tokens=tel.total_tokens,
                  api_calls=tel.n_panel_ok + 1,  # N panel + 1 synth
                  critical_path_ms=tel.critical_path_ms)
        log_event("orchestrator", "end", status="aggregator_mode",
                  total_duration_ms=total_ms,
                  critical_path_ms=tel.critical_path_ms, answer_chars=len(text))
        return FusionResult(status=FusionStatus.AGGREGATOR_MODE, text=text,
                            panel_responses=responses, telemetry=tel.summary())

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

    # --- PICK-BEST SHORT-CIRCUIT -------------------------------------------
    # v1.2 改进: 如果 judge 识别出 best_model, 直接返回该模型的回答, 跳过 synthesis。
    # 这避免了 synthesizer "投票覆盖" 问题 — 当少数模型答对但多数答错时,
    # synthesizer 会被多数错误答案带偏, 而 pick-best 直接采纳 judge 选中的最佳答案。
    #
    # v1.2.1 fallback: 如果置信度评估发现 judge 判断不确定 (矛盾多/无共识/best_reason 含
    # 不确定性词汇), 则不短路, 回退到完整 synthesizer 流程。这解决了 math_2 场景下
    # judge 误判导致短路锁定错误答案的问题。
    if config.enable_pick_best:
        best_idx = _resolve_best_model(analysis.best_model, responses)
        if best_idx is not None:
            best_resp = responses[best_idx]
            min_chars = getattr(config, "pick_best_min_chars", 10)
            if best_resp.ok and len(best_resp.content.strip()) >= min_chars:
                # v1.2.1: 评估 pick-best 置信度
                confidence, reason = _assess_pick_best_confidence(analysis, responses, best_idx)
                confidence_threshold = getattr(config, "pick_best_confidence_threshold", 0.5)

                # v1.3 P0: 答案验证器 — 基于规则的格式/数值检查
                verifier_verdict = "skipped"
                verifier_reason = ""
                if getattr(config, "enable_verifier", False):
                    from .verifier import verify_answer
                    task_meta = getattr(config, "_task_metadata", None)
                    vr = verify_answer(best_resp.content, task_meta)
                    verifier_verdict = vr.verdict.value
                    verifier_reason = vr.reason
                    if vr.should_fallback:
                        # 验证器否决: 强制回退到 synthesizer
                        log_event("orchestrator", "pick_best_verifier_reject",
                                  best_model=best_resp.model,
                                  verdict=verifier_verdict,
                                  verifier_reason=verifier_reason,
                                  reason="verifier_rejected_fallback_to_synth")
                        confidence = 0.0  # 强制低置信度, 走 fallback
                        reason = f"verifier_rejected:{verifier_reason}"

                if confidence >= confidence_threshold:
                    # 高置信度: 短路返回
                    tel.status = "pick_best_shortcut"
                    log_event("orchestrator", "pick_best_shortcut",
                              best_model=best_resp.model,
                              best_reason=analysis.best_reason,
                              answer_chars=len(best_resp.content),
                              min_chars_threshold=min_chars,
                              confidence=round(confidence, 3),
                              confidence_reason=reason,
                              skipped_synth=True,
                              total_duration_ms=int((time.monotonic() - t_run0) * 1000))
                    return FusionResult(
                        status=FusionStatus.PICK_BEST_SHORTCUT,
                        text=best_resp.content.strip(),
                        analysis=analysis,
                        panel_responses=responses,
                        telemetry=tel.summary(),
                    )
                else:
                    # 低置信度: 回退到 synthesizer, 避免 judge 误判锁定错误答案
                    log_event("orchestrator", "pick_best_fallback",
                              best_model=best_resp.model,
                              confidence=round(confidence, 3),
                              confidence_reason=reason,
                              threshold=confidence_threshold,
                              reason="low_confidence_fallback_to_synth")

    # --- SYNTHESIZE ---------------------------------------------------------
    # "The calling model then writes the final answer grounded in that analysis."
    # P2-B: 当 synth_tools_enabled 时, synthesizer 保留工具能力 (支持 agent loop)。
    from .tools import toolset_for_phase
    synth_tools = toolset_for_phase(Phase.SYNTHESIS) if config.synth_tools_enabled else ()
    text = await synth_mod.write(
        client, config.caller, question, analysis, responses,
        config.params[Phase.SYNTHESIS], tel=tel, tools=synth_tools)
    tel.status = "ok"
    log_event("orchestrator", "end", status="ok",
              total_duration_ms=int((time.monotonic() - t_run0) * 1000),
              critical_path_ms=tel.critical_path_ms,
              panel_max_ms=max(tel.panel_latencies_ms) if tel.panel_latencies_ms else 0,
              judge_ms=tel.judge_ms, synthesis_ms=tel.synthesis_ms,
              total_tokens=tel.total_tokens, answer_chars=len(text))
    return FusionResult(status=FusionStatus.OK, text=text, analysis=analysis,
                        panel_responses=responses, telemetry=tel.summary())


# --------------------------------------------------------------------------- #
# Helper functions for consensus and pick-best short-circuits
# --------------------------------------------------------------------------- #

def _similarity(a: str, b: str) -> float:
    """Compute text similarity ratio between two strings (0.0 to 1.0)."""
    if not a or not b:
        return 0.0
    # Use a bounded head+tail comparison for speed, fallback to SequenceMatcher.
    # Comparing only prefixes can falsely short-circuit long answers whose setup is
    # identical but whose conclusions diverge; the tail captures those reversals.
    if len(a) > 2000 or len(b) > 2000:
        def sample(s: str) -> str:
            return s[:500] + "\n...\n" + s[-500:]
        return SequenceMatcher(None, sample(a), sample(b)).ratio()
    return SequenceMatcher(None, a, b).ratio()


def _check_consensus(responses: list[PanelResponse],
                     threshold: float = 0.85) -> PanelResponse | None:
    """改进3: Check if a majority of panel responses are highly similar.
    
    If >= 2 responses have > threshold similarity, return the longest one
    (most complete). This enables short-circuiting judge+synthesis when
    models agree, saving 2 API calls and avoiding synthesizer degradation.
    """
    if len(responses) < 2:
        log_event("orchestrator", "consensus_skip",
                  reason="insufficient_responses", n_responses=len(responses))
        return None

    contents = [r.content.strip() for r in responses]
    n = len(contents)

    # Pairwise similarity: find the largest cluster of similar responses
    best_cluster_size = 1
    best_cluster_repr = None

    # 调试日志: 记录两两相似度矩阵, 便于排查"为何未触发 consensus"
    debug_pairs = is_enabled_for(logging.DEBUG)

    for i in range(n):
        cluster = [i]
        for j in range(i + 1, n):
            sim = _similarity(contents[i], contents[j])
            if debug_pairs:
                log_event("orchestrator", "consensus_pairwise",
                          i=i, j=j, similarity=round(sim, 3),
                          threshold=threshold,
                          above=sim >= threshold,
                          len_i=len(contents[i]), len_j=len(contents[j]))
            if sim >= threshold:
                cluster.append(j)
        if len(cluster) > best_cluster_size:
            best_cluster_size = len(cluster)
            best_cluster_repr = cluster

    # Need majority (more than half) to short-circuit
    majority = (n // 2) + 1
    if best_cluster_size >= majority and best_cluster_repr:
        # Pick the longest response in the cluster (most complete answer)
        best_idx = max(best_cluster_repr,
                        key=lambda i: len(contents[i]))
        log_event("orchestrator", "consensus_reached",
                  n_responses=n, cluster_size=best_cluster_size,
                  majority_needed=majority, threshold=threshold,
                  winner_idx=best_idx, winner_model=responses[best_idx].model,
                  winner_chars=len(contents[best_idx]))
        return responses[best_idx]

    log_event("orchestrator", "consensus_missed",
              n_responses=n, best_cluster_size=best_cluster_size,
              majority_needed=majority, threshold=threshold,
              reason="cluster_below_majority")
    return None


def _resolve_best_model(best_model_label: str | None,
                        responses: list[PanelResponse]) -> int | None:
    """改进1: Resolve judge's best_model label to a panel response index.
    
    Judge labels are "MODEL A", "MODEL B", etc. (1-indexed letters).
    Returns the index into responses, or None if no match.
    """
    if not best_model_label:
        return None

    label = best_model_label.strip().upper()

    # Handle "MODEL A" / "MODEL B" / etc.
    if label.startswith("MODEL "):
        suffix = label[6:].strip()
        if len(suffix) == 1 and suffix.isalpha():
            idx = ord(suffix.upper()) - ord('A')
            if 0 <= idx < len(responses):
                return idx

    # Handle numeric labels "MODEL 1", "1", etc.
    if label.isdigit():
        idx = int(label) - 1
        if 0 <= idx < len(responses):
            return idx

    # Fallback: match by model slug if the judge used actual model names
    for i, r in enumerate(responses):
        if r.model.lower() in label.lower() or label.lower() in r.model.lower():
            return i

    return None


# v1.2.1: pick-best 置信度评估 — 低置信度时回退到 synthesizer
# 解决 math_2 问题: judge 误判导致短路锁定错误答案
_UNCERTAINTY_MARKERS = frozenset({
    "unclear", "no clear winner", "comparable", "similar quality",
    "difficult to determine", "hard to say", "tied", "equivalent",
    "no clear best", "ambiguous", "marginal", "slight", "arguable",
    "debatable", "uncertain", "not definitive", "no definitive",
})


def _assess_pick_best_confidence(
    analysis: Analysis,
    responses: list[PanelResponse],
    best_idx: int,
) -> tuple[float, str]:
    """v1.2.1: 评估 pick-best 短路的置信度。

    综合 4 个信号判断 judge 的 best_model 选择是否可靠:
      1. best_reason 是否包含不确定性词汇 (权重 -0.3)
      2. contradictions 数量 — 矛盾越多, 选择越不确定 (权重 -0.1/矛盾)
      3. consensus 是否为空 — 无共识意味着模型分歧大 (权重 -0.2)
      4. best 答案与其他答案的相似度 — 如果 best 是"异类"(低相似度),
         可能是 judge 误选了边缘答案 (权重 -0.2)

    返回: (confidence: 0.0-1.0, reason: str)
    """
    confidence = 1.0  # 起始满置信度, 逐项扣减

    # 信号 1: best_reason 含不确定性词汇
    reason_text = (analysis.best_reason or "").lower()
    if not reason_text or reason_text in ("null", "none"):
        confidence -= 0.3
        return (max(confidence, 0.0), "empty_best_reason")
    for marker in _UNCERTAINTY_MARKERS:
        if marker in reason_text:
            confidence -= 0.3
            return (max(confidence, 0.0), f"uncertain_best_reason:{marker}")

    # 信号 2: contradictions 数量
    n_contradictions = len(analysis.contradictions)
    if n_contradictions >= 2:
        confidence -= 0.1 * n_contradictions
        # 不 return, 继续检查其他信号

    # 信号 3: consensus 为空
    if not analysis.consensus:
        confidence -= 0.2

    # 信号 4: best 答案与其他答案的相似度
    ok_responses = [r for r in responses if r.ok and r.content.strip()]
    if len(ok_responses) >= 2:
        best_content = responses[best_idx].content.strip()
        similarities = []
        for i, r in enumerate(responses):
            if r.ok and r.content.strip() and i != best_idx:
                sim = _similarity(best_content, r.content.strip())
                similarities.append(sim)
        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            # 如果 best 答案与其他答案平均相似度 < 0.3, 它可能是"异类"
            if avg_sim < 0.3:
                confidence -= 0.2
                return (max(confidence, 0.0), f"low_similarity_outlier:avg_sim={avg_sim:.2f}")

    # 汇总原因
    reasons = []
    if n_contradictions >= 2:
        reasons.append(f"n_contradictions={n_contradictions}")
    if not analysis.consensus:
        reasons.append("no_consensus")
    if not reasons:
        return (max(confidence, 0.0), "high_confidence")
    return (max(confidence, 0.0), ";".join(reasons))
