"""
open_fusion.client - provider-agnostic model client. The ONLY module that does HTTP.

Targets any OpenAI-compatible chat-completions endpoint; defaults to the
OpenRouter gateway so a single API key reaches every vendor's models by slug.
Zero runtime dependencies: HTTP via stdlib urllib, concurrency via the event
loop's default thread executor (urllib is blocking, so panel fan-out runs each
call in a thread).
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .config import ModelSpec, Params
from .schema import TokenUsage
from .tools import ToolSpec


# ---- typed errors so panel.py can decide failure vs. partial -----------------
class ClientError(Exception): ...
class Timeout(ClientError): ...
class RateLimit(ClientError): ...
class ProviderError(ClientError):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Completion:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    raw_message: dict[str, Any] | None = None   # assistant message verbatim (for appending)
    raw: dict[str, Any] | None = None


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_RETRY_STATUSES = {429, 500, 502, 503, 504}


def _tool_to_openai(t: ToolSpec) -> dict[str, Any]:
    return {"type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}


class ModelClient:
    """Wraps one OpenAI-compatible gateway.

    Auth/base_url resolution order:
      explicit kwargs > OPEN_FUSION_BASE_URL/OPEN_FUSION_API_KEY >
      OPENROUTER_API_KEY (with the OpenRouter base url).
    """

    def __init__(self, *, fusion_depth: int = 0, base_url: str | None = None,
                 api_key: str | None = None, max_retries: int = 3,
                 referer: str | None = None, title: str = "open-fusion",
                 executor_workers: int | None = None) -> None:
        self.fusion_depth = fusion_depth
        # Explicit kwargs take precedence; fall back to env only when not provided.
        resolved_base_url = base_url
        if not resolved_base_url:
            resolved_base_url = os.getenv("OPEN_FUSION_BASE_URL") or DEFAULT_BASE_URL
        self.base_url = resolved_base_url.rstrip("/")
        resolved_api_key = api_key
        if not resolved_api_key:
            resolved_api_key = (os.getenv("OPEN_FUSION_API_KEY")
                                or os.getenv("OPENROUTER_API_KEY") or "")
        self.api_key = resolved_api_key
        self.max_retries = max_retries
        self.referer = referer or os.getenv("OPEN_FUSION_REFERER", "https://github.com/")
        self.title = title
        # 性能优化 O6：自带线程池，避免 N>32 panel 时挤占 asyncio 默认 executor。
        # 默认 max(32, panel*2) 由调用方按需注入；None 时回退到 asyncio 默认。
        self._executor = None
        if executor_workers and executor_workers > 0:
            from concurrent.futures import ThreadPoolExecutor
            self._executor = ThreadPoolExecutor(
                max_workers=executor_workers, thread_name_prefix="of-http")

    # -- blocking HTTP (runs in a thread) -------------------------------------
    def _post(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        """默认用 self.base_url / self.api_key 发请求。"""
        return self._post_with(self.base_url, self.api_key, payload, timeout)

    def _post_with(self, base_url: str, api_key: str,
                   payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        """P1-B: 支持自定义 base_url/api_key 的 HTTP POST (多 Provider 路由)。"""
        url = f"{base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "x-open-fusion-depth": str(self.fusion_depth),
            "HTTP-Referer": self.referer,
            "X-Title": self.title,
        }
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:500]
            except Exception:
                pass
            if e.code == 429:
                raise RateLimit(f"429 rate limited: {detail}") from e
            raise ProviderError(f"HTTP {e.code}: {detail}", status=e.code) from e
        except (socket.timeout, TimeoutError) as e:
            raise Timeout(str(e)) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, (socket.timeout, TimeoutError)):
                raise Timeout(str(e)) from e
            raise ProviderError(f"network error: {e.reason}") from e

    @staticmethod
    def _parse(data: dict[str, Any], requested_model: str) -> Completion:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage_d = data.get("usage") or {}
        tcs: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
            tcs.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args))
        # Content extraction: prefer final answer, fall back to reasoning_content if present
        # (reasoning models like GLM/MiniMax/Doubao may put thinking in reasoning_content
        #  and final answer in content; if max_tokens is too low, content can be empty)
        content = msg.get("content") or ""
        if not content and msg.get("reasoning_content"):
            content = msg.get("reasoning_content", "")
        return Completion(
            content=content,
            tool_calls=tcs,
            usage=TokenUsage(prompt_tokens=int(usage_d.get("prompt_tokens", 0)),
                             completion_tokens=int(usage_d.get("completion_tokens", 0))),
            model=data.get("model", requested_model),
            raw_message=msg,
            raw=data,
        )

    async def complete(
        self,
        model: ModelSpec,
        messages: list[dict[str, Any]],
        *,
        tools: tuple[ToolSpec, ...] = (),
        params: Params,
        response_format: str | None = None,
    ) -> Completion:
        # P1-B: 多 Provider — model 有独立 base_url/api_key 时用 per-model 网关。
        eff_base_url = model.base_url or self.base_url
        eff_api_key = model.api_key or self.api_key
        if not eff_api_key:
            raise ProviderError("no API key. Set OPENROUTER_API_KEY (or OPEN_FUSION_API_KEY).")

        payload: dict[str, Any] = {
            "model": model.slug,
            "messages": messages,
            "temperature": model.temperature if model.temperature is not None else params.temperature,
            "max_tokens": model.max_tokens if model.max_tokens is not None else params.max_tokens,
        }
        if tools:
            payload["tools"] = [_tool_to_openai(t) for t in tools]
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        loop = asyncio.get_running_loop()
        attempt = 0
        while True:
            try:
                if model.base_url or model.api_key:
                    data = await loop.run_in_executor(
                        self._executor, self._post_with,
                        eff_base_url, eff_api_key, payload, params.timeout_s)
                else:
                    data = await loop.run_in_executor(
                        self._executor, self._post, payload, params.timeout_s)
                return self._parse(data, model.slug)
            except ProviderError as e:
                # If the model rejects json mode, retry once without it.
                if e.status == 400 and "response_format" in payload:
                    payload.pop("response_format", None)
                    continue
                if e.status in _RETRY_STATUSES and attempt < self.max_retries:
                    attempt += 1
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
                raise
            except RateLimit:
                if attempt < self.max_retries:
                    attempt += 1
                    await asyncio.sleep(min(2 ** attempt, 8))
                    continue
                raise
