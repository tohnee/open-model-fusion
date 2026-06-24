"""
open_fusion._logging - 极轻量结构化日志埋点。

默认完全静默（NullHandler），不影响"答案到 stdout / telemetry 到 stderr"的契约。
开启方式（运行时一次性）：
    export OPEN_FUSION_LOG=DEBUG     # 或 INFO / WARNING
或在代码里:
    from open_fusion._logging import enable_logging
    enable_logging("DEBUG")

每条日志是一个 `key=value` 单行结构（不依赖 JSON 库以避开开销，便于 grep/awk）：
    ts=2026-... phase=judge event=start
    ts=2026-... phase=judge event=end duration_ms=18 in_chars=420 out_chars=512 retries=0

埋点点位（请保持与 ARCHITECTURE.md 第 5 节状态机一致）：
    orchestrator: fuse(start/end/depth_exceeded/all_panel_failed/judge_fallback)
    synthesizer:  write(start/end), write_fallback(start/end)
    judge:        synthesize(start/end/retry/failed) —— 选择性补充
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any

_LOGGER_NAME = "open_fusion"
_logger = logging.getLogger(_LOGGER_NAME)
# NullHandler 是 library 最佳实践：不附 handler，让宿主决定要不要展示。
_logger.addHandler(logging.NullHandler())


def enable_logging(level: str = "INFO", *, stream=None) -> None:
    """打开 open_fusion 日志到 stderr（或自定义 stream）。幂等，可重复调用。"""
    # 移除已有的非 NullHandler，避免重复输出
    for h in list(_logger.handlers):
        if not isinstance(h, logging.NullHandler):
            _logger.removeHandler(h)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(level.upper() if isinstance(level, str) else level)


# 环境变量一键开启：进程启动时一次性判定。
_env_level = os.getenv("OPEN_FUSION_LOG")
if _env_level:
    try:
        enable_logging(_env_level)
    except Exception:
        # 非法 level 不应让导入失败
        pass


def _fmt_kv(fields: dict[str, Any]) -> str:
    parts = []
    for k, v in fields.items():
        if v is None:
            continue
        # 字符串避免大块换行污染日志，截断到 80 字符
        if isinstance(v, str) and len(v) > 80:
            v = v[:77] + "..."
        parts.append(f"{k}={v}")
    return " ".join(parts)


def log_event(phase: str, event: str, **fields: Any) -> None:
    """打一条结构化日志。phase: panel/judge/synthesis/orchestrator；event: start/end/...。"""
    if not _logger.isEnabledFor(logging.INFO) and event in ("start", "end"):
        return
    payload = {"phase": phase, "event": event, **fields}
    if event in ("start", "end"):
        _logger.info(_fmt_kv(payload))
    elif event in ("error", "retry", "fallback", "depth_exceeded", "all_failed"):
        _logger.warning(_fmt_kv(payload))
    else:
        _logger.debug(_fmt_kv(payload))


@contextmanager
def timed_phase(phase: str, **start_fields: Any):
    """计时上下文：记录 start/end 两条日志，end 自带 duration_ms。

    用法:
        with timed_phase("judge", in_chars=len(labeled)) as ctx:
            ...
            ctx["out_chars"] = len(analysis_json)   # 可写字段会在 end 日志里输出
    """
    t0 = time.monotonic()
    log_event(phase, "start", **start_fields)
    end_fields: dict[str, Any] = {}
    try:
        yield end_fields
    except Exception as e:
        end_fields.update({"error": f"{type(e).__name__}: {e}"})
        log_event(phase, "error",
                  duration_ms=int((time.monotonic() - t0) * 1000), **end_fields)
        raise
    else:
        end_fields["duration_ms"] = int((time.monotonic() - t0) * 1000)
        log_event(phase, "end", **end_fields)


def is_enabled_for(level: int) -> bool:
    """轻量探测：当前 logger 是否会真发出指定级别。供调用方在 DEBUG 才计算昂贵字段。"""
    return _logger.isEnabledFor(level)
