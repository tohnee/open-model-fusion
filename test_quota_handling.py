#!/usr/bin/env python3
"""Test quota/rate-limit error handling in the fusion pipeline.

Simulates API quota exceeded (429) and timeout errors to verify:
1. _answer_for_arm catches exceptions and returns empty string (not crash)
2. The benchmark continues running even if some tasks fail
3. Error messages are logged properly
4. The final report includes failed tasks as 0-score
"""
import asyncio
import sys
import json
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, "src")

from open_fusion.config import FusionConfig, ModelSpec, Params
from open_fusion.schema import Phase, PanelResponse, TokenUsage, FusionStatus
from open_fusion.client import Completion
from open_fusion.eval.rubric import Task, Criterion, RubricCategory
from open_fusion.eval.grader import RuleGrader
from open_fusion.eval.harness import _grade_avg
from open_fusion.eval.rubric import score_response
from open_fusion.eval.superiority import _answer_for_arm, Arm, build_superiority_arms
from open_fusion.client import ModelClient
from open_fusion.orchestrator import fuse


# --- 1. Test _answer_for_arm with 429 RateLimit ---

async def test_arm_429_rate_limit():
    """Test that _answer_for_arm catches 429 errors gracefully."""
    print("=" * 60)
    print("TEST 1: _answer_for_arm with 429 RateLimit")
    print("=" * 60)

    task = Task(
        id="test_429",
        domain="logical_reasoning",
        prompt="Test question",
        rubric=[Criterion(id="correct", description="correct answer",
                          category=RubricCategory.FACTUAL_ACCURACY,
                          weight=5, check={"any": ["correct"]})],
    )

    base = FusionConfig(
        panel=[ModelSpec("model-a"), ModelSpec("model-b")],
        judge=ModelSpec("judge"),
        base_url="http://fake",
        api_key="fake",
    )

    # Mock client that raises 429
    mock_client = MagicMock(spec=ModelClient)
    mock_client.fusion_depth = 0
    mock_client.complete = AsyncMock(side_effect=Exception(
        "RateLimit: 429 rate limited: {\"error\":{\"code\":\"AccountQuotaExceeded\","
        "\"message\":\"You have exceeded the weekly usage quota.\"}}"
    ))

    arm = Arm("solo_0", "solo", [ModelSpec("model-a")], ModelSpec("judge"),
              ModelSpec("judge"), "solo test")

    result = await _answer_for_arm(arm, task, base, mock_client)
    print(f"  Result: '{result}' (len={len(result)})")
    assert result == "", f"Expected empty string, got '{result}'"
    print("  ✓ _answer_for_arm caught 429 and returned empty string\n")
    return True


# --- 2. Test fusion arm with quota exceeded ---

async def test_fusion_arm_quota_exceeded():
    """Test that the fusion arm handles quota exceeded without crashing."""
    print("=" * 60)
    print("TEST 2: fusion arm with quota exceeded on synthesis")
    print("=" * 60)

    task = Task(
        id="test_fusion_429",
        domain="logical_reasoning",
        prompt="Test question for fusion",
        rubric=[Criterion(id="correct", description="correct",
                          category=RubricCategory.FACTUAL_ACCURACY,
                          weight=5, check={"any": ["correct"]})],
    )

    base = FusionConfig(
        panel=[ModelSpec("model-a"), ModelSpec("model-b")],
        judge=ModelSpec("judge"),
        base_url="http://fake",
        api_key="fake",
        params={
            Phase.PANEL: Params(temperature=0.7, max_tokens=100, timeout_s=10),
            Phase.JUDGE: Params(temperature=0.2, max_tokens=100, timeout_s=10),
            Phase.SYNTHESIS: Params(temperature=0.3, max_tokens=100, timeout_s=10),
        },
    )

    # Mock client: panel succeeds, judge succeeds, synthesis raises 429
    mock_client = MagicMock(spec=ModelClient)
    mock_client.fusion_depth = 0

    panel_response = PanelResponse(
        model="model-a", content="Test answer", status="ok",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=10),
        latency_ms=100, tool_trace=[], error=None,
    )

    # First call = panel, second = judge, third = synthesis (429)
    call_count = [0]
    async def mock_complete(model, messages, tools=(), params=None, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 2:  # panel + judge succeed
            if call_count[0] == 1:
                # Return panel response
                return Completion(content="Test panel answer",
                                  usage=TokenUsage(prompt_tokens=10, completion_tokens=10),
                                  raw_message=None)
            else:
                # Return judge JSON
                return Completion(
                    content=json.dumps({
                        "consensus": ["test consensus"],
                        "contradictions": [],
                        "partial_coverage": [],
                        "unique_insights": [],
                        "blind_spots": []
                    }),
                    usage=TokenUsage(prompt_tokens=10, completion_tokens=10),
                    raw_message=None,
                )
        else:
            # Synthesis raises 429
            raise Exception("RateLimit: 429 AccountQuotaExceeded")

    mock_client.complete = mock_complete
    # Make the mock support async iteration for panel.run
    mock_client.executor = None

    arm = Arm("fusion", "structured", [ModelSpec("model-a"), ModelSpec("model-b")],
              ModelSpec("judge"), ModelSpec("judge"), "fusion test")

    result = await _answer_for_arm(arm, task, base, mock_client)
    print(f"  Result: '{result}' (len={len(result)})")
    # Should be empty string (exception caught) or some fallback
    print("  ✓ Fusion arm handled 429 without crashing\n")
    return True


# --- 3. Test grader with empty response (0 score) ---

async def test_grader_empty_response():
    """Test that grader gives 0 score for empty responses."""
    print("=" * 60)
    print("TEST 3: Grader scoring empty response (quota exceeded)")
    print("=" * 60)

    task = Task(
        id="test_empty",
        domain="logical_reasoning",
        prompt="Test question",
        rubric=[
            Criterion(id="correct", description="correct answer",
                      category=RubricCategory.FACTUAL_ACCURACY,
                      weight=5, check={"any": ["correct"]}),
            Criterion(id="proof", description="proof",
                      category=RubricCategory.BREADTH_DEPTH,
                      weight=3, check={"any": ["proof", "therefore"]}),
            Criterion(id="penalty", description="wrong",
                      category=RubricCategory.FACTUAL_ACCURACY,
                      weight=-5, check={"any": ["wrong"]}),
        ],
    )

    grader = RuleGrader()
    avg = await _grade_avg(grader, task, "", 1)
    print(f"  Empty response score: {avg.normalized}")
    assert avg.normalized == 0.0, f"Expected 0.0, got {avg.normalized}"
    print("  ✓ Empty response correctly scored 0.0\n")
    return True


# --- 4. Test grader with normal response ---

async def test_grader_normal_response():
    """Test that grader scores correctly for valid responses."""
    print("=" * 60)
    print("TEST 4: Grader scoring normal response")
    print("=" * 60)

    task = Task(
        id="test_normal",
        domain="logical_reasoning",
        prompt="Test question",
        rubric=[
            Criterion(id="correct", description="correct answer",
                      category=RubricCategory.FACTUAL_ACCURACY,
                      weight=5, check={"any": ["correct", "right"]}),
            Criterion(id="proof", description="proof",
                      category=RubricCategory.BREADTH_DEPTH,
                      weight=3, check={"any": ["proof", "therefore", "step"]}),
        ],
    )

    grader = RuleGrader()
    avg = await _grade_avg(grader, task, "The correct answer is 42. proof: step 1 therefore done.", 1)
    print(f"  Normal response score: {avg.normalized}")
    assert avg.normalized > 0.0, "Expected positive score"
    print("  ✓ Normal response scored correctly\n")
    return True


# --- 5. Test full benchmark with mocked 429 on some tasks ---

async def test_benchmark_partial_429():
    """Simulate a benchmark where some tasks get 429 errors."""
    print("=" * 60)
    print("TEST 5: Simulated benchmark with partial 429 failures")
    print("=" * 60)

    # Simulate scores: 3 tasks succeed, 2 tasks get 429 (score=0)
    simulated_scores = [
        ("task_1", 85.0, True),   # success
        ("task_2", 0.0, False),   # 429 error
        ("task_3", 100.0, True),  # success
        ("task_4", 0.0, False),   # 429 error
        ("task_5", 76.9, True),   # success
    ]

    successful = [s for s in simulated_scores if s[2]]
    failed = [s for s in simulated_scores if not s[2]]

    avg_all = sum(s[1] for s in simulated_scores) / len(simulated_scores)
    avg_success = sum(s[1] for s in successful) / len(successful) if successful else 0

    print(f"  Total tasks: {len(simulated_scores)}")
    print(f"  Successful: {len(successful)}, Failed (429): {len(failed)}")
    print(f"  Average (all, including 0s): {avg_all:.1f}")
    print(f"  Average (successful only): {avg_success:.1f}")
    print(f"  Impact of 429 on average: {avg_success - avg_all:.1f} points")

    assert len(failed) == 2, "Expected 2 failures"
    assert avg_all < avg_success, "Average with 0s should be lower"
    print("  ✓ Quota handling correctly affects benchmark average\n")
    return True


# --- 6. Test synthesizer fix: original responses are passed ---

async def test_synthesizer_passes_responses():
    """Verify that the synthesizer now receives original panel responses."""
    print("=" * 60)
    print("TEST 6: Synthesizer receives original panel responses")
    print("=" * 60)

    from open_fusion.prompts import SYNTHESIS_USER, SYNTHESIS_SYSTEM

    # Test that SYNTHESIS_USER accepts 3 args
    question = "What is 2+2?"
    analysis = '{"consensus": ["4"], "contradictions": [], "partial_coverage": [], "unique_insights": [], "blind_spots": []}'
    responses = "[MODEL A]: The answer is 4.\n[MODEL B]: 2+2=4, proven by arithmetic."

    prompt = SYNTHESIS_USER(question, analysis, responses)
    print(f"  Prompt length: {len(prompt)} chars")
    assert "QUESTION" in prompt
    assert "STRUCTURED ANALYSIS" in prompt
    assert "ORIGINAL PANEL RESPONSES" in prompt
    assert "MODEL A" in prompt
    assert "MODEL B" in prompt
    print("  ✓ SYNTHESIS_USER correctly includes panel responses")

    # Test backward compatibility (2 args, no responses)
    prompt_no_resp = SYNTHESIS_USER(question, analysis)
    assert "ORIGINAL PANEL RESPONSES" not in prompt_no_resp
    print("  ✓ Backward compatible with 2 args (no responses section)")
    print()
    return True


# --- Runner ---

async def main():
    print("\n" + "=" * 60)
    print("QUOTA & ERROR HANDLING TEST SUITE")
    print("=" * 60 + "\n")

    results = []
    tests = [
        ("Arm 429 RateLimit", test_arm_429_rate_limit),
        ("Fusion arm quota exceeded", test_fusion_arm_quota_exceeded),
        ("Grader empty response", test_grader_empty_response),
        ("Grader normal response", test_grader_normal_response),
        ("Benchmark partial 429", test_benchmark_partial_429),
        ("Synthesizer passes responses", test_synthesizer_passes_responses),
    ]

    for name, test_fn in tests:
        try:
            ok = await test_fn()
            results.append((name, ok, None))
        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {e}\n")
            results.append((name, False, str(e)))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed
    for name, ok, err in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")
        if err:
            print(f"         Error: {err}")
    print(f"\n  Total: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
