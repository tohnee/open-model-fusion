"""
open_fusion.panel - parallel fan-out of the prompt to the analysis panel.

Each panel model runs as a short tool-using agent session. Fan-out is concurrent
and tolerant of partial failure: one model timing out or erroring never kills the
batch (>=1 survivor is enough to proceed).
"""
from __future__ import annotations

import asyncio
import time

from .client import ClientError, ModelClient, Timeout
from .config import ModelSpec, Params
from .schema import PanelResponse, Phase, TokenUsage
from .tools import execute_tool, toolset_for_phase


async def _run_one(
    client: ModelClient,
    model: ModelSpec,
    question: str,
    params: Params,
    tools_enabled: bool,
    excluded_domains: list[str],
) -> PanelResponse:
    """Run ONE panel model as a bounded agent loop. Never raises."""
    t0 = time.monotonic()
    tools = toolset_for_phase(Phase.PANEL) if tools_enabled else ()
    messages: list[dict] = [{"role": "user", "content": question}]
    trace: list[dict] = []
    usage = TokenUsage()
    rounds = 0
    comp = None    # defence: 即使下面的 for 一轮没跑（max_tool_calls < 0 等异常配置），
                   # 也不会触发 UnboundLocalError；下方收口 return 走"no completion"分支。
    try:
        # 至少跑 1 轮：保证拿到一个最终 completion 或一个真实错误，而不是把配置异常
        # 静默吃掉成"空答"。Params.__post_init__ 已对 max_tool_calls < 0 做硬校验，
        # 这里的 max(1, ...) 是多写一层防御。
        loop_count = max(1, params.max_tool_calls + 1)
        for _ in range(loop_count):
            rounds += 1
            comp = await client.complete(model, messages, tools=tools, params=params)
            usage.prompt_tokens += comp.usage.prompt_tokens
            usage.completion_tokens += comp.usage.completion_tokens

            if comp.tool_calls and tools and rounds <= params.max_tool_calls:
                messages.append(comp.raw_message or
                                {"role": "assistant", "content": comp.content, "tool_calls": []})
                for tc in comp.tool_calls:
                    result = await execute_tool(tc.name, tc.arguments,
                                                excluded_domains=excluded_domains)
                    trace.append({"tool": tc.name, "args": tc.arguments, "result": result})
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": __import__("json").dumps(result)[:8000]})
                continue

            return PanelResponse(
                model=model.slug, content=comp.content, tool_trace=trace, status="ok",
                latency_ms=int((time.monotonic() - t0) * 1000), usage=usage)

        # Ran out of tool budget without a final answer: return last content if any.
        if comp is None:    # 极端 edge case：循环没进过任何一轮
            return PanelResponse(
                model=model.slug, status="error",
                error="invalid params: no completion attempted",
                tool_trace=trace, latency_ms=int((time.monotonic() - t0) * 1000),
                usage=usage)
        return PanelResponse(
            model=model.slug, content=comp.content, tool_trace=trace,
            status="ok" if comp.content else "error",
            error=None if comp.content else "tool budget exhausted without final answer",
            latency_ms=int((time.monotonic() - t0) * 1000), usage=usage)

    except Timeout as e:
        return PanelResponse(model=model.slug, status="timeout", error=str(e),
                             tool_trace=trace, latency_ms=int((time.monotonic() - t0) * 1000),
                             usage=usage)
    except ClientError as e:
        return PanelResponse(model=model.slug, status="error", error=str(e),
                             tool_trace=trace, latency_ms=int((time.monotonic() - t0) * 1000),
                             usage=usage)
    except Exception as e:  # defensive: a bug must not take down the batch
        return PanelResponse(model=model.slug, status="error",
                             error=f"{type(e).__name__}: {e}", tool_trace=trace,
                             latency_ms=int((time.monotonic() - t0) * 1000), usage=usage)


async def run(
    client: ModelClient,
    panel: list[ModelSpec],
    question: str,
    params: Params,
    *,
    tools_enabled: bool = False,
    max_in_flight: int = 8,
    excluded_domains: list[str] | None = None,
    fast_majority_k: int | None = None,
) -> list[PanelResponse]:
    """Fan out to all panel models in parallel. Returns responses in panel order
    so labels (MODEL A/B/C) are stable across runs."""
    excluded = list(excluded_domains or [])
    sem = asyncio.Semaphore(max_in_flight)

    async def guarded(m: ModelSpec) -> PanelResponse:
        async with sem:
            return await _run_one(client, m, question, params, tools_enabled, excluded)

    tasks = [asyncio.ensure_future(guarded(m)) for m in panel]

    if fast_majority_k and 0 < fast_majority_k < len(tasks):
        results: dict[int, PanelResponse] = {}
        idx = {t: i for i, t in enumerate(tasks)}
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                results[idx[t]] = t.result()
            if sum(1 for r in results.values() if r.ok) >= fast_majority_k:
                break
        for t in pending:
            t.cancel()
        # 早退场景：被取消的 panel 用独立的 "cancelled" 状态，避免和真实的 "error"
        # / "timeout" 混淆。orchestrator 据此把 cancelled 排除在 panel_failed 之外，
        # 保留 panel_failed 的诊断口径（真正不可用的模型数量）。
        return [results.get(i) or PanelResponse(model=panel[i].slug, status="cancelled",
                                                error="cancelled (fast majority)")
                for i in range(len(panel))]

    gathered = await asyncio.gather(*tasks, return_exceptions=True)
    out: list[PanelResponse] = []
    for m, r in zip(panel, gathered):
        if isinstance(r, BaseException):
            out.append(PanelResponse(model=m.slug, status="error",
                                     error=f"{type(r).__name__}: {r}"))
        else:
            out.append(r)
    return out
