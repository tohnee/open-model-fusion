"""
open_fusion.eval.grader - per-criterion grading.

Two graders behind one async `grade(task, response) -> {criterion_id: met}` interface:

  - LLMJudgeGrader: the DRACO-faithful path. A judge model decides, per criterion,
    whether the response meets it. This is run by the harness N (=3) independent
    times and the normalized scores are averaged. Absolute scores shift with judge
    choice (DRACO reports 10-25 pt swings) so the harness emphasises the RELATIVE
    gap between Fusion and the solo baseline, which is stable across judges.
  - RuleGrader: deterministic, offline, zero-key. Evaluates each criterion's `check`
    rule. Powers the test suite and the no-key demo, and is useful for the subset of
    criteria that are mechanically checkable.
"""
from __future__ import annotations

import json
from typing import Protocol

from ..client import ClientError, ModelClient
from ..config import ModelSpec, Params
from ..judge import _strip_fences
from .rubric import Task, evaluate_check


class Grader(Protocol):
    async def grade(self, task: Task, response: str) -> dict[str, bool]: ...


class RuleGrader:
    """Deterministic grader driven by each criterion's `check` rule."""
    name = "rule"

    async def grade(self, task: Task, response: str) -> dict[str, bool]:
        return {c.id: evaluate_check(response, c.check) for c in task.rubric}


_GRADER_SYSTEM = """You are a strict evaluation judge. You are given a research QUESTION, \
a candidate RESPONSE, and a list of rubric CRITERIA. For each criterion decide whether \
the response meets it.

- A POSITIVE criterion is met when the response genuinely satisfies it (a verifiable \
claim is correct and present; a required analysis is actually performed; a citation is \
real and supports the claim).
- A NEGATIVE criterion (marked penalty=true) is "met" when the response COMMITS that \
error (e.g. states a dangerous or false claim). Mark it met=true only when the error is \
actually present.
- Do not reward length, confidence, or formatting that isn't backed by substance. A \
verbose answer that does not satisfy a criterion does NOT meet it.

Output ONLY a JSON object mapping each criterion id to a boolean, e.g.
{"f1": true, "f2": false, "p1": true}. No prose, no markdown fences."""


def _grader_user(task: Task, response: str) -> str:
    lines = [f"QUESTION:\n{task.prompt}\n", "CRITERIA:"]
    for c in task.rubric:
        tag = " (penalty=true)" if c.is_penalty else ""
        lines.append(f"- id={c.id} [{c.category.value}] weight={c.weight}{tag}: {c.description}")
    lines.append(f"\nRESPONSE:\n{response}\n")
    lines.append("Return the JSON object of {criterion_id: met_boolean} now.")
    return "\n".join(lines)


class LLMJudgeGrader:
    """DRACO-faithful LLM-as-judge grader (one judging pass per call)."""
    name = "llm"

    def __init__(self, client: ModelClient, judge: ModelSpec,
                 params: Params | None = None) -> None:
        self.client = client
        self.judge = judge
        self.params = params or Params(temperature=0.0, max_tokens=2048)

    async def grade(self, task: Task, response: str) -> dict[str, bool]:
        messages = [
            {"role": "system", "content": _GRADER_SYSTEM},
            {"role": "user", "content": _grader_user(task, response)},
        ]
        try:
            comp = await self.client.complete(self.judge, messages, tools=(),
                                              params=self.params, response_format="json")
            obj = json.loads(_strip_fences(comp.content)) or {}
        except (ClientError, json.JSONDecodeError, ValueError):
            obj = {}
        # only keep known criterion ids; coerce to bool; default missing -> False
        return {c.id: bool(obj.get(c.id, False)) for c in task.rubric}
