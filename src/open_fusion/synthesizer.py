"""
open_fusion.synthesizer - the final-answer writers.

`write` is the normal path: it turns the judge's structured Analysis into prose.
`write_fallback` is the degraded path used when the judge could not produce valid
JSON twice; it writes from the raw panel answers instead.

Both run with tools=() unconditionally. The no-tools rule for synthesis is a hard
design invariant (evidence is frozen once the judge has reasoned over it), enforced
here and mirrored by tools.toolset_for_phase(SYNTHESIS) == ().

可观测性：两条路径都打了结构化日志（默认静默），字段含 in/out chars、tokens、
duration_ms；定位"是 caller 模型本身慢" vs "网络慢" vs "JSON 解析慢"。
"""
from __future__ import annotations

import json
import time

from ._logging import log_event
from .client import ModelClient
from .config import ModelSpec, Params
from .cost import Telemetry
from .prompts import (FALLBACK_SYSTEM, FALLBACK_USER, SYNTHESIS_SYSTEM,
                      SYNTHESIS_USER, label_responses)
from .schema import Analysis, PanelResponse


async def write(
    client: ModelClient,
    writer: ModelSpec,        # the caller/synthesizer model that writes the final answer
    question: str,
    analysis: Analysis,
    responses: list[PanelResponse],
    params: Params,
    *,
    tel: Telemetry | None = None,
) -> str:
    analysis_json = json.dumps(analysis.to_dict())
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM},
        {"role": "user", "content": SYNTHESIS_USER(question, analysis_json)},
    ]
    # synthesis_ms 来自这里的端到端测量：包含一次模型完成 + 网络往返 + JSON 编码。
    # 与 logging 共用同一个 t0，确保"日志看到的耗时 == telemetry 里的 synthesis_ms"。
    t0 = time.monotonic()
    log_event("synthesis", "start", writer=writer.slug,
              question_chars=len(question), analysis_chars=len(analysis_json),
              tools_disabled=True)
    try:
        comp = await client.complete(writer, messages, tools=(), params=params)  # no tools: frozen evidence
    except Exception as e:
        log_event("synthesis", "error", writer=writer.slug,
                  duration_ms=int((time.monotonic() - t0) * 1000),
                  error=f"{type(e).__name__}: {e}")
        raise
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    if tel:
        tel.add_usage(f"synthesis:{writer.slug}", comp.usage)
        tel.synthesis_ms = elapsed_ms
    log_event("synthesis", "end", writer=writer.slug, duration_ms=elapsed_ms,
              out_chars=len(comp.content), prompt_tokens=comp.usage.prompt_tokens,
              completion_tokens=comp.usage.completion_tokens)
    return comp.content.strip()


async def write_fallback(
    client: ModelClient,
    writer: ModelSpec,        # the caller/synthesizer model that writes the final answer
    question: str,
    responses: list[PanelResponse],
    params: Params,
    *,
    tel: Telemetry | None = None,
) -> str:
    labeled = label_responses(responses)
    messages = [
        {"role": "system", "content": FALLBACK_SYSTEM},
        {"role": "user", "content": FALLBACK_USER(question, labeled)},
    ]
    t0 = time.monotonic()
    log_event("synthesis_fallback", "start", writer=writer.slug,
              question_chars=len(question), panel_chars=len(labeled),
              tools_disabled=True)
    try:
        comp = await client.complete(writer, messages, tools=(), params=params)
    except Exception as e:
        log_event("synthesis_fallback", "error", writer=writer.slug,
                  duration_ms=int((time.monotonic() - t0) * 1000),
                  error=f"{type(e).__name__}: {e}")
        raise
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    if tel:
        tel.add_usage(f"synthesis_fallback:{writer.slug}", comp.usage)
        tel.synthesis_ms = elapsed_ms
    log_event("synthesis_fallback", "end", writer=writer.slug, duration_ms=elapsed_ms,
              out_chars=len(comp.content), prompt_tokens=comp.usage.prompt_tokens,
              completion_tokens=comp.usage.completion_tokens)
    return comp.content.strip()
