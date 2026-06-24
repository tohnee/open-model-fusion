"""
open_fusion.judge - distill panel responses into a validated Analysis.

This is where the structured intermediate representation is produced - the core
lever of the whole system. Robust JSON handling: request JSON mode, strip fences,
parse, validate, and retry ONCE on failure before giving up.
"""
from __future__ import annotations

import json
import time

from .client import ModelClient
from .config import ModelSpec, Params
from .cost import Telemetry
from .prompts import JUDGE_SYSTEM, JUDGE_USER, JSON_RETRY, label_responses
from .schema import Analysis, PanelResponse, Phase
from .tools import toolset_for_phase


class JudgeError(Exception):
    """Raised after the judge fails to produce valid JSON even on retry.
    orchestrator catches this and falls back to raw-response synthesis."""


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        t = t[nl + 1:] if nl != -1 else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def parse_analysis(raw_text: str) -> Analysis:
    """Parse + validate. Raises ValueError (message reused in the retry turn)."""
    if not raw_text.strip():
        raise ValueError("empty output")
    obj = json.loads(_strip_fences(raw_text))
    analysis = Analysis.from_dict(obj)
    problems = [p for p in analysis.validate() if not p.startswith("WARNING")]
    if problems:
        raise ValueError("; ".join(problems))
    return analysis


async def synthesize(
    client: ModelClient,
    judge: ModelSpec,
    question: str,
    responses: list[PanelResponse],
    params: Params,
    *,
    tools_enabled: bool = False,
    tel: Telemetry | None = None,
) -> Analysis:
    labeled = label_responses(responses)
    tools = toolset_for_phase(Phase.JUDGE) if tools_enabled else ()
    messages: list[dict] = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER(question, labeled)},
    ]
    t0 = time.monotonic()
    comp = await client.complete(judge, messages, tools=tools, params=params,
                                 response_format="json")
    if tel:
        tel.add_usage(f"judge:{judge.slug}", comp.usage)
    try:
        analysis = parse_analysis(comp.content)
        if tel:
            tel.judge_ms = int((time.monotonic() - t0) * 1000)
        return analysis
    except (json.JSONDecodeError, ValueError) as e:
        # one repair attempt
        if tel:
            tel.judge_retries += 1
        messages.append(comp.raw_message or {"role": "assistant", "content": comp.content})
        messages.append({"role": "user", "content": JSON_RETRY(comp.content, str(e))})
        comp2 = await client.complete(judge, messages, tools=(), params=params,
                                      response_format="json")
        if tel:
            tel.add_usage(f"judge:{judge.slug}", comp2.usage)
            tel.judge_ms = int((time.monotonic() - t0) * 1000)
        try:
            return parse_analysis(comp2.content)
        except (json.JSONDecodeError, ValueError) as e2:
            raise JudgeError(f"judge produced invalid analysis twice: {e2}") from e2
