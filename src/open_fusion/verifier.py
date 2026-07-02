"""
open_fusion.verifier - v1.3 P0: 基于规则的答案验证器

在 pick-best 短路前, 验证 judge 选中的答案是否格式合理、数值在合理范围内。
如果验证失败, 返回低置信度, 触发 fallback 到 synthesizer。

当前支持:
  - 数学题答案: 分数格式 (a/b)、小数格式、整数格式
  - 数值范围检查: 基于题目元数据中的 expected_range
  - 格式一致性检查: 答案是否包含明显的非数学内容

设计原则:
  - 纯规则, 无 LLM 调用 (零延迟, 零成本)
  - 只做 "否决" 不做 "确认" — 验证器只标记明显不合理的答案
  - 向后兼容: 不配置 metadata 时, 验证器返回 neutral (不干预)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class VerifierVerdict(str, Enum):
    """验证器的判定结果。"""
    PASS = "pass"               # 答案格式合理, 不阻止短路
    FAIL = "fail"               # 答案明显不合理, 应回退到 synthesizer
    NEUTRAL = "neutral"         # 无法判断 (无元数据), 不干预


@dataclass
class VerifyResult:
    """验证器的返回值。"""
    verdict: VerifierVerdict
    confidence: float           # 0.0-1.0, FAIL 时为 0.0, PASS 时 >= 0.7
    reason: str                 # 判定原因 (用于日志)

    @property
    def should_fallback(self) -> bool:
        """是否应该回退到 synthesizer。"""
        return self.verdict == VerifierVerdict.FAIL


# ============================================================================
# 数学答案解析
# ============================================================================

# 匹配分数 (a/b)、小数 (1.23)、整数 (42)、带百分号 (15%)
_FRACTION_RE = re.compile(r'[-]?\d+\s*/\s*\d+')
_DECIMAL_RE = re.compile(r'[-]?\d+\.\d+')
_INTEGER_RE = re.compile(r'[-]?\d+')
_PERCENT_RE = re.compile(r'[-]?\d+(\.\d+)?%')

# 非数学内容的红旗词汇
_NON_MATH_MARKERS = frozenset({
    "无法", "不知道", "不确定", "cannot", "unable", "i don't know",
    "no idea", "unclear", "n/a", "none",
})


def extract_math_answer(text: str) -> list[float]:
    """从文本中提取所有数值答案。

    优先级: 分数 > 小数 > 整数。
    返回找到的所有数值列表 (分数会被转为 float)。
    """
    if not text or not text.strip():
        return []

    results: list[float] = []
    text = text.strip()

    # 先找分数 (a/b), 转为 float
    for m in _FRACTION_RE.finditer(text):
        parts = m.group().split("/")
        if len(parts) == 2 and int(parts[1]) != 0:
            results.append(int(parts[0].strip()) / int(parts[1].strip()))

    # 如果找到分数, 直接返回 (分数是最明确的数学答案格式)
    if results:
        return results

    # 找小数
    for m in _DECIMAL_RE.finditer(text):
        results.append(float(m.group()))

    # 找整数 (排除已经在小数中的)
    if not results:
        for m in _INTEGER_RE.finditer(text):
            results.append(int(m.group()))

    return results


def _check_value_range(values: list[float],
                       expected_range: tuple[float, float] | None) -> bool:
    """检查数值是否在预期范围内。返回 True 如果所有值都在范围内。"""
    if not expected_range or not values:
        return True
    lo, hi = expected_range
    return all(lo <= v <= hi for v in values)


def _has_non_math_markers(text: str) -> bool:
    """检查答案是否包含 "不知道" 类的红旗词汇。"""
    text_lower = text.lower()
    for marker in _NON_MATH_MARKERS:
        if marker in text_lower:
            return True
    return False


# ============================================================================
# 公开 API
# ============================================================================

def verify_answer(
    answer: str,
    task_metadata: dict[str, Any] | None = None,
) -> VerifyResult:
    """验证 pick-best 选中的答案是否合理。

    Args:
        answer: pick-best 选中的模型回答
        task_metadata: 题目元数据, 可选字段:
            - domain: "数学" | "代码" | "逻辑推理" | ...
            - expected_range: (min, max) 数值范围
            - expected_format: "fraction" | "decimal" | "integer" | "percentage"
            - answer_keywords: list[str] 答案必须包含的关键词

    Returns:
        VerifyResult: verdict + confidence + reason
    """
    if not answer or not answer.strip():
        return VerifyResult(VerifierVerdict.FAIL, 0.0, "empty_answer")

    text = answer.strip()
    meta = task_metadata or {}

    # 检查红旗词汇 ("不知道" 等)
    if _has_non_math_markers(text):
        return VerifyResult(VerifierVerdict.FAIL, 0.0, "non_math_marker_in_answer")

    # 非数学题不做数值验证
    domain = meta.get("domain", "")
    if domain and "数学" not in domain and "math" not in domain.lower():
        return VerifyResult(VerifierVerdict.NEUTRAL, 0.7, "non_math_domain_skipped")

    # 数学题: 提取数值并验证
    values = extract_math_answer(text)

    if not values:
        # 数学题但答案中没有数值 — 可能是纯文字描述
        # 不直接 FAIL (有些数学题答案是 "无穷大" 之类), 标记低置信度
        return VerifyResult(VerifierVerdict.NEUTRAL, 0.4, "no_numeric_value_found")

    # 检查数值范围
    expected_range = meta.get("expected_range")
    if expected_range and not _check_value_range(values, expected_range):
        # 特殊处理: 含根号/无理数的答案 (如 "9√3") 提取出的数值不是实际值
        # 如果答案包含 √ 或 sqrt, 跳过范围检查 (无法精确解析)
        if "√" in text or "sqrt" in text.lower() or "\\sqrt" in text:
            pass  # 无理数答案, 无法精确验证范围
        else:
            lo, hi = expected_range
            return VerifyResult(
                VerifierVerdict.FAIL, 0.0,
                f"value_out_of_range: {values} not in [{lo}, {hi}]"
            )

    # 检查预期格式
    expected_format = meta.get("expected_format")
    if expected_format == "fraction" and not _FRACTION_RE.search(text):
        return VerifyResult(
            VerifierVerdict.FAIL, 0.1,
            "expected_fraction_format_but_not_found"
        )
    if expected_format == "percentage" and not _PERCENT_RE.search(text):
        return VerifyResult(
            VerifierVerdict.FAIL, 0.1,
            "expected_percentage_format_but_not_found"
        )

    # 检查关键词
    answer_keywords = meta.get("answer_keywords", [])
    if answer_keywords:
        text_lower = text.lower()
        missing = [kw for kw in answer_keywords if kw.lower() not in text_lower]
        if missing:
            return VerifyResult(
                VerifierVerdict.FAIL, 0.2,
                f"missing_required_keywords: {missing}"
            )

    # 所有检查通过
    return VerifyResult(VerifierVerdict.PASS, 0.8, "all_checks_passed")
