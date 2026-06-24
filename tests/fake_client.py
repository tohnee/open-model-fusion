"""
fake_client - a scriptable, zero-network stand-in for ModelClient.

Each model slug maps to a list of scripted turns consumed in order. A turn is one of:
  - text("...")                       -> a normal assistant message
  - with_tool_call("web_fetch", {..}) -> an assistant message that calls one tool
  - an Exception instance             -> raised when that turn is consumed (e.g. Timeout)

Because the judge/synthesis reuse the judge slug, a judge model's script is the
concatenation of: [its panel turn(s), its judge turn(s), its synthesis turn].
"""
from __future__ import annotations

import asyncio
from typing import Any

from open_fusion.client import Completion, ProviderError, ToolCall
from open_fusion.schema import TokenUsage


def text(content: str) -> dict[str, Any]:
    return {"_kind": "text", "content": content}


def with_tool_call(name: str, args: dict[str, Any], call_id: str | None = None) -> dict[str, Any]:
    return {"_kind": "tool", "name": name, "args": args, "id": call_id or f"call_{name}"}


def _completion_from_turn(turn: dict[str, Any], slug: str) -> Completion:
    usage = TokenUsage(prompt_tokens=10, completion_tokens=10)
    if turn["_kind"] == "text":
        return Completion(
            content=turn["content"], tool_calls=[], usage=usage, model=slug,
            raw_message={"role": "assistant", "content": turn["content"]})
    if turn["_kind"] == "tool":
        tc = ToolCall(id=turn["id"], name=turn["name"], arguments=turn["args"])
        raw_message = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": turn["id"], "type": "function",
                "function": {"name": turn["name"], "arguments": __import__("json").dumps(turn["args"])},
            }],
        }
        return Completion(content="", tool_calls=[tc], usage=usage, model=slug,
                          raw_message=raw_message)
    raise ValueError(f"unknown scripted turn: {turn!r}")


class FakeClient:
    def __init__(self, scripts: dict[str, list] | None = None) -> None:
        # copy so popping doesn't mutate the caller's literals across runs
        self._scripts: dict[str, list] = {k: list(v) for k, v in (scripts or {}).items()}
        self.calls: list[dict[str, Any]] = []
        self.fusion_depth = 0

    async def complete(self, model, messages, *, tools=(), params=None, response_format=None):
        self.calls.append({
            "model": getattr(model, "slug", model),
            "tools": tools,
            "response_format": response_format,
            "n_messages": len(messages),
        })
        await asyncio.sleep(0)  # let concurrent panel tasks interleave

        slug = getattr(model, "slug", model)
        script = self._scripts.get(slug)
        if not script:
            raise ProviderError(f"FakeClient: no scripted turns left for '{slug}'")
        turn = script.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        return _completion_from_turn(turn, slug)
