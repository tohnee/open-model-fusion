"""v1.3 P0: 答案验证器单元测试

测试覆盖:
  - 数学答案提取 (分数/小数/整数/百分比)
  - 数值范围检查
  - 格式验证 (fraction/percentage)
  - 红旗词汇检测
  - 非数学域跳过
  - pick-best 集成 (验证器否决 → fallback)
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from open_fusion.verifier import (
    verify_answer, extract_math_answer,
    VerifierVerdict, VerifyResult,
)


# ============================================================================
# 数学答案提取测试
# ============================================================================

class TestExtractMathAnswer:
    def test_fraction(self):
        vals = extract_math_answer("答案是 15/28")
        assert vals == [15/28]

    def test_decimal(self):
        vals = extract_math_answer("P = 0.5357")
        assert vals == [0.5357]

    def test_integer(self):
        vals = extract_math_answer("共有 362880 种")
        assert vals == [362880]

    def test_multiple_values(self):
        vals = extract_math_answer("x=2, y=3")
        assert 2 in vals and 3 in vals

    def test_empty_text(self):
        assert extract_math_answer("") == []
        assert extract_math_answer("   ") == []

    def test_negative_number(self):
        vals = extract_math_answer("x = -5")
        assert -5 in vals

    def test_fraction_priority_over_decimal(self):
        """分数应优先于小数 (1/2 而非 1 和 2)"""
        vals = extract_math_answer("1/2")
        assert vals == [0.5]

    def test_no_numbers(self):
        assert extract_math_answer("无穷大") == []


# ============================================================================
# 验证器: 基本判定测试
# ============================================================================

class TestVerifyAnswer:
    def test_empty_answer_fails(self):
        r = verify_answer("")
        assert r.verdict == VerifierVerdict.FAIL
        assert r.should_fallback is True

    def test_non_math_marker_fails(self):
        r = verify_answer("我不知道答案")
        assert r.verdict == VerifierVerdict.FAIL
        assert "non_math" in r.reason

    def test_non_math_domain_skipped(self):
        r = verify_answer("some code answer", {"domain": "代码"})
        assert r.verdict == VerifierVerdict.NEUTRAL

    def test_math_answer_passes(self):
        r = verify_answer("15/28", {"domain": "数学"})
        assert r.verdict == VerifierVerdict.PASS
        assert r.confidence >= 0.7

    def test_value_out_of_range_fails(self):
        r = verify_answer("999", {
            "domain": "数学",
            "expected_range": (0, 100),
        })
        assert r.verdict == VerifierVerdict.FAIL
        assert "out_of_range" in r.reason

    def test_value_in_range_passes(self):
        r = verify_answer("72", {
            "domain": "数学",
            "expected_range": (72, 72),
        })
        assert r.verdict == VerifierVerdict.PASS

    def test_expected_fraction_format_missing(self):
        r = verify_answer("0.536", {
            "domain": "数学",
            "expected_format": "fraction",
        })
        assert r.verdict == VerifierVerdict.FAIL
        assert "fraction" in r.reason

    def test_expected_fraction_format_present(self):
        r = verify_answer("15/28", {
            "domain": "数学",
            "expected_format": "fraction",
        })
        assert r.verdict == VerifierVerdict.PASS

    def test_missing_keywords_fails(self):
        r = verify_answer("答案是 42", {
            "domain": "数学",
            "answer_keywords": ["15/28"],
        })
        assert r.verdict == VerifierVerdict.FAIL
        assert "missing_required_keywords" in r.reason

    def test_keywords_present_passes(self):
        r = verify_answer("概率是 15/28", {
            "domain": "数学",
            "answer_keywords": ["15/28"],
        })
        assert r.verdict == VerifierVerdict.PASS

    def test_no_metadata_neutral(self):
        r = verify_answer("some answer without numbers")
        assert r.verdict == VerifierVerdict.NEUTRAL

    def test_math_no_numeric_low_confidence(self):
        r = verify_answer("无穷大", {"domain": "数学"})
        assert r.verdict == VerifierVerdict.NEUTRAL
        assert r.confidence < 0.5

    def test_expected_percentage_missing(self):
        r = verify_answer("0.75", {
            "domain": "数学",
            "expected_format": "percentage",
        })
        assert r.verdict == VerifierVerdict.FAIL

    def test_expected_percentage_present(self):
        r = verify_answer("75%", {
            "domain": "数学",
            "expected_format": "percentage",
        })
        assert r.verdict == VerifierVerdict.PASS


# ============================================================================
# 测试数据集: math_hard_v13.json 验证
# ============================================================================

class TestMathDataset:
    """验证测试数据集的完整性和正确性。"""

    def test_dataset_loads(self):
        import json
        path = os.path.join(os.path.dirname(__file__), "..", "src",
                            "open_fusion", "eval", "tasks", "math_hard_v13.json")
        with open(path) as f:
            data = json.load(f)
        assert len(data["tasks"]) == 10

    def test_all_tasks_have_required_fields(self):
        import json
        path = os.path.join(os.path.dirname(__file__), "..", "src",
                            "open_fusion", "eval", "tasks", "math_hard_v13.json")
        with open(path) as f:
            data = json.load(f)
        for t in data["tasks"]:
            assert "id" in t
            assert "domain" in t
            assert "question" in t
            assert "answer" in t
            assert "check" in t
            assert "expected_format" in t
            assert "expected_range" in t

    def test_all_tasks_are_math(self):
        import json
        path = os.path.join(os.path.dirname(__file__), "..", "src",
                            "open_fusion", "eval", "tasks", "math_hard_v13.json")
        with open(path) as f:
            data = json.load(f)
        for t in data["tasks"]:
            assert t["domain"] == "数学"

    def test_verifier_on_dataset_answers(self):
        """验证器应 PASS 所有正确答案。"""
        import json
        path = os.path.join(os.path.dirname(__file__), "..", "src",
                            "open_fusion", "eval", "tasks", "math_hard_v13.json")
        with open(path) as f:
            data = json.load(f)
        for t in data["tasks"]:
            meta = {
                "domain": t["domain"],
                "expected_range": tuple(t["expected_range"]),
                "expected_format": t.get("expected_format"),
                "answer_keywords": t.get("answer_keywords", []),
            }
            r = verify_answer(t["answer"], meta)
            assert r.verdict == VerifierVerdict.PASS, \
                f"Task {t['id']} answer '{t['answer']}' failed: {r.reason}"
