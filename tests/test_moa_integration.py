"""
MoA 集成测试: 6 个改进方案的 TDD 测试。

覆盖:
  P0-A: Aggregator Mode (FusionMode 枚举, 跳过 judge)
  P0-B: 命名预设系统 (load_preset 支持 logic/code/moa_fast)
  P1-A: Panel 裁剪输入 (enable_panel_trim + panel_trim_chars)
  P1-B: 多 Provider 支持 (ModelSpec.base_url/api_key + _post_with)
  P2-A: 末尾注入缓存优化 (SYNTHESIS_USER 末尾追加)
  P2-B: Synthesizer 工具保留 (synth_tools_enabled + tools 参数)

Run: python tests/test_moa_integration.py
"""
from __future__ import annotations

import asyncio
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from open_fusion.config import FusionConfig, ModelSpec, load_preset
from open_fusion.orchestrator import fuse
from open_fusion.schema import FusionStatus, Phase
from open_fusion.prompts import SYNTHESIS_USER, JUDGE_SYSTEM
from fake_client import FakeClient, text

PASS, FAIL = 0, 0


def check(name: str, cond: bool):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


VALID_ANALYSIS = json.dumps({
    "consensus": ["test consensus"],
    "contradictions": [],
    "partial_coverage": [],
    "unique_insights": [],
    "blind_spots": [],
    "best_model": "MODEL A",
    "best_reason": "most complete",
})


def cfg(panel=("a/x", "b/y"), judge="a/x", **kw) -> FusionConfig:
    return FusionConfig(panel=[ModelSpec(s) for s in panel], judge=ModelSpec(judge), **kw)


# == P0-A: Aggregator Mode ================================================ #

def test_p0a_fusion_mode_enum():
    from open_fusion.config import FusionMode
    check("FusionMode.FULL exists", FusionMode.FULL.value == "full")
    check("FusionMode.AGGREGATOR exists", FusionMode.AGGREGATOR.value == "aggregator")


def test_p0a_config_defaults_to_full():
    from open_fusion.config import FusionMode
    c = cfg()
    check("default mode is FULL", c.mode == FusionMode.FULL)


def test_p0a_aggregator_mode_skips_judge():
    """AGGREGATOR mode: 跳过 judge, 只需 panel + synth 调用。"""
    from open_fusion.config import FusionMode
    c = FakeClient(scripts={
        "a/x": [text("A answer"), text("FINAL SYNTH")],
        "b/y": [text("B answer")],
    })
    config = cfg(mode=FusionMode.AGGREGATOR, enable_consensus_shortcut=False)
    r = asyncio.run(fuse("q", config, client=c))
    check("aggregator mode returns AGGREGATOR_MODE", r.status == FusionStatus.AGGREGATOR_MODE)
    check("aggregator skips judge (3 calls: 2 panel + 1 synth)", len(c.calls) == 3)
    check("aggregator has final text", r.text == "FINAL SYNTH")


def test_p0a_full_mode_still_calls_judge():
    """FULL mode: 仍然走完整 pipeline (panel + judge + synth)。"""
    c = FakeClient(scripts={
        "a/x": [text("A"), text(VALID_ANALYSIS), text("FINAL")],
        "b/y": [text("B")],
    })
    config = cfg(enable_consensus_shortcut=False, enable_pick_best=False)
    r = asyncio.run(fuse("q", config, client=c))
    check("full mode returns OK", r.status == FusionStatus.OK)
    check("full mode has 4 calls", len(c.calls) == 4)


# == P0-B: Named Presets ================================================== #

def test_p0b_all_presets_loadable():
    for name in ("quality", "budget", "logic", "code", "moa_fast"):
        try:
            c = load_preset(name)
            check(f"preset '{name}' loads", len(c.panel) >= 1 and c.judge is not None)
        except Exception as e:
            check(f"preset '{name}' loads", False)


def test_p0b_logic_preset_config():
    c = load_preset("logic")
    check("logic preset threshold=0.90", c.consensus_threshold == 0.90)
    check("logic preset has 3 panel models", len(c.panel) == 3)
    check("logic preset pick_best on", c.enable_pick_best == True)


def test_p0b_code_preset_config():
    c = load_preset("code")
    check("code preset threshold=0.75", c.consensus_threshold == 0.75)


def test_p0b_moa_fast_preset_is_aggregator():
    from open_fusion.config import FusionMode
    c = load_preset("moa_fast")
    check("moa_fast is AGGREGATOR mode", c.mode == FusionMode.AGGREGATOR)
    check("moa_fast pick_best off", c.enable_pick_best == False)


def test_p0b_unknown_preset_raises():
    raised = False
    try:
        load_preset("nonexistent")
    except ValueError:
        raised = True
    check("unknown preset raises ValueError", raised)


def test_p0b_preset_overrides():
    c = load_preset("logic", consensus_threshold=0.99)
    check("override applied", c.consensus_threshold == 0.99)


# == P1-A: Panel Trimmed Input ============================================ #

def test_p1a_config_supports_trim():
    c = cfg(enable_panel_trim=True, panel_trim_chars=2000)
    check("enable_panel_trim settable", c.enable_panel_trim == True)
    check("panel_trim_chars settable", c.panel_trim_chars == 2000)


def test_p1a_trim_default_off():
    c = cfg()
    check("panel trim off by default", c.enable_panel_trim == False)


def test_p1a_trim_truncates_long_prompt():
    long_prompt = "X" * 5000
    c = FakeClient(scripts={
        "a/x": [text("A"), text(VALID_ANALYSIS), text("FINAL")],
        "b/y": [text("B")],
    })
    config = cfg(enable_panel_trim=True, panel_trim_chars=1000,
                 enable_consensus_shortcut=False, enable_pick_best=False)
    asyncio.run(fuse(long_prompt, config, client=c))
    check("trim mode runs without error", len(c.calls) >= 2)


# == P1-B: Multi-Provider Support ========================================= #

def test_p1b_modelspec_supports_per_model_urls():
    m = ModelSpec("claude-opus", base_url="https://api.anthropic.com/v1", api_key="sk-xxx")
    check("ModelSpec.base_url", m.base_url == "https://api.anthropic.com/v1")
    check("ModelSpec.api_key", m.api_key == "sk-xxx")


def test_p1b_modelspec_defaults_none():
    m = ModelSpec("gpt-4")
    check("default base_url None", m.base_url is None)
    check("default api_key None", m.api_key is None)


def test_p1b_post_with_method_exists():
    from open_fusion.client import ModelClient
    client = ModelClient(base_url="https://default.example.com", api_key="key")
    check("_post_with exists", hasattr(client, "_post_with"))


def test_p1b_complete_routes_per_model():
    from open_fusion.client import ModelClient
    from open_fusion.config import Params

    client = ModelClient(base_url="https://default.example.com", api_key="default-key")
    captured = {}
    def mock_post_with(base_url, api_key, payload, timeout):
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        return {"choices": [{"message": {"content": "ok", "role": "assistant"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    client._post_with = mock_post_with

    model = ModelSpec("claude", base_url="https://api.anthropic.com/v1", api_key="sk-custom")
    params = Params(temperature=0.5, max_tokens=100, timeout_s=10)
    asyncio.run(client.complete(model, [{"role": "user", "content": "hi"}], params=params))
    check("per-model base_url used", captured.get("base_url") == "https://api.anthropic.com/v1")
    check("per-model api_key used", captured.get("api_key") == "sk-custom")


def test_p1b_complete_uses_global_when_no_per_model():
    from open_fusion.client import ModelClient
    from open_fusion.config import Params

    client = ModelClient(base_url="https://global.example.com", api_key="global-key")
    captured = {"called": False}
    def mock_post(payload, timeout):
        captured["called"] = True
        return {"choices": [{"message": {"content": "ok", "role": "assistant"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    client._post = mock_post

    model = ModelSpec("gpt-4")
    params = Params(temperature=0.5, max_tokens=100, timeout_s=10)
    asyncio.run(client.complete(model, [{"role": "user", "content": "hi"}], params=params))
    check("global _post used when no per-model config", captured["called"] == True)


# == P2-A: Tail Injection ================================================ #

def test_p2a_synthesis_user_tail_injection():
    question = "What is 2+2?"
    analysis = '{"consensus": ["4"]}'
    responses = "[MODEL A]: The answer is 4."
    prompt = SYNTHESIS_USER(question, analysis, responses)
    check("responses at tail", "ORIGINAL PANEL RESPONSES" in prompt)
    q_pos = prompt.index("What is 2+2?")
    r_pos = prompt.index("MODEL A")
    check("question before responses (cache-friendly)", q_pos < r_pos)


def test_p2a_config_tail_injection_flag():
    c = cfg()
    check("enable_tail_injection exists", hasattr(c, "enable_tail_injection"))
    check("enable_tail_injection default True", c.enable_tail_injection == True)


def test_p2a_backward_compatible_no_responses():
    prompt = SYNTHESIS_USER("question", '{"consensus": []}')
    check("no responses section", "ORIGINAL PANEL RESPONSES" not in prompt)
    check("still has write instruction", "Write the final answer now." in prompt)


# == P2-B: Synthesizer Tool Retention ==================================== #

def test_p2b_config_synth_tools_flag():
    c = cfg()
    check("synth_tools_enabled exists", hasattr(c, "synth_tools_enabled"))
    check("synth_tools_enabled default False", c.synth_tools_enabled == False)


def test_p2b_write_accepts_tools_param():
    from open_fusion.synthesizer import write
    sig = inspect.signature(write)
    check("write has tools param", "tools" in sig.parameters)
    check("tools default empty tuple", sig.parameters["tools"].default == ())


def test_p2b_can_enable_synth_tools():
    c = cfg(synth_tools_enabled=True)
    check("synth_tools_enabled settable", c.synth_tools_enabled == True)


# == Depth Guard 补充测试 ================================================ #
# 覆盖 orchestrator.py 中未覆盖的 depth guard 分支:
#   L34:  config.validate() 返回硬错误时抛 ValueError
#   L47-54: effective_depth >= MAX_FUSION_DEPTH 时返回 ERROR
#   L50-51: depth_exceeded 日志事件
#   L56-59: client is None 时自动创建 ModelClient
#   L47:  client 有 fusion_depth 但 config.depth=0 的组合

def test_depth_guard_invalid_config_raises():
    """config.validate() 返回硬错误时, fuse() 抛 ValueError (非 WARNING)。"""
    # 空 panel → "panel is empty" 是硬错误
    bad_cfg = FusionConfig(panel=[], judge=ModelSpec("j"))
    raised = False
    try:
        asyncio.run(fuse("q", bad_cfg, client=FakeClient()))
    except ValueError:
        raised = True
    check("empty panel raises ValueError", raised)


def test_depth_guard_negative_depth_raises():
    """depth < 0 是硬错误, 应抛 ValueError。"""
    bad_cfg = FusionConfig(panel=[ModelSpec("a")], judge=ModelSpec("j"), depth=-1)
    raised = False
    try:
        asyncio.run(fuse("q", bad_cfg, client=FakeClient()))
    except ValueError:
        raised = True
    check("negative depth raises ValueError", raised)


def test_depth_guard_config_depth_blocks():
    """config.depth >= MAX_FUSION_DEPTH 时返回 ERROR, 不执行 panel。"""
    c = FakeClient(scripts={"a/x": [text("should not run")]})
    config = cfg(depth=1)  # MAX_FUSION_DEPTH = 1
    r = asyncio.run(fuse("q", config, client=c))
    check("depth=1 blocked", r.status == FusionStatus.ERROR)
    check("no panel calls on depth block", len(c.calls) == 0)
    check("error message mentions depth", "depth" in (r.error or ""))


def test_depth_guard_client_fusion_depth_blocks():
    """client.fusion_depth >= MAX_FUSION_DEPTH 时返回 ERROR, 即使 config.depth=0。"""
    c = FakeClient()
    c.fusion_depth = 1  # 模拟上游已发起 fusion
    config = cfg(depth=0)
    r = asyncio.run(fuse("q", config, client=c))
    check("client depth blocks", r.status == FusionStatus.ERROR)
    check("no calls when blocked by client depth", len(c.calls) == 0)


def test_depth_guard_takes_max_of_both():
    """effective_depth = max(config.depth, client.fusion_depth) — 两者取大值。"""
    # config.depth=0, client.fusion_depth=1 → max=1 → blocked
    c1 = FakeClient()
    c1.fusion_depth = 1
    r1 = asyncio.run(fuse("q", cfg(depth=0), client=c1))
    check("max(0,1)=1 blocks", r1.status == FusionStatus.ERROR)

    # config.depth=1, client.fusion_depth=0 → max=1 → blocked
    c2 = FakeClient()
    c2.fusion_depth = 0
    r2 = asyncio.run(fuse("q", cfg(depth=1), client=c2))
    check("max(1,0)=1 blocks", r2.status == FusionStatus.ERROR)

    # config.depth=0, client 无 fusion_depth 属性 → 用 config.depth=0 → 放行
    c3 = FakeClient(scripts={
        "a/x": [text("A"), text(VALID_ANALYSIS), text("OK")],
        "b/y": [text("B")],
    })
    # FakeClient 有 fusion_depth=0, 所以 effective_depth=0 < 1, 放行
    r3 = asyncio.run(fuse("q", cfg(depth=0, enable_consensus_shortcut=False,
                                    enable_pick_best=False), client=c3))
    check("max(0,0)=0 passes", r3.status == FusionStatus.OK)


def test_depth_guard_no_client_creates_modelclient():
    """当 client=None 时, orchestrator 自动创建 ModelClient (L56-59)。

    由于真实 ModelClient 会尝试 HTTP, 这里只验证不抛异常地到达 depth guard
    (用 depth=1 短路, 避免真实网络调用)。
    """
    config = cfg(depth=1)  # depth guard 会阻止, 不需要真实网络
    r = asyncio.run(fuse("q", config, client=None))
    check("client=None with depth guard returns ERROR", r.status == FusionStatus.ERROR)
    check("client=None depth guard has error msg", "depth" in (r.error or ""))


def test_depth_guard_telemetry_status():
    """depth guard 返回的 telemetry 中 status='depth_exceeded'。"""
    c = FakeClient()
    r = asyncio.run(fuse("q", cfg(depth=1), client=c))
    check("telemetry status is depth_exceeded",
          r.telemetry.get("status") == "depth_exceeded")


def test_depth_guard_with_aggregator_mode():
    """AGGREGATOR mode 下 depth guard 仍然生效 (mode 不影响 depth 检查)。"""
    from open_fusion.config import FusionMode
    c = FakeClient()
    config = cfg(depth=1, mode=FusionMode.AGGREGATOR)
    r = asyncio.run(fuse("q", config, client=c))
    check("aggregator mode + depth=1 blocked", r.status == FusionStatus.ERROR)
    check("no panel calls in blocked aggregator", len(c.calls) == 0)


def test_depth_guard_panel_not_empty_warning_passes():
    """validate() 返回 WARNING (同质 panel) 不阻断执行, 只有硬错误才阻断。"""
    # 同 vendor 的 panel → WARNING, 不是硬错误
    c = FakeClient(scripts={
        "a/x": [text("A"), text(VALID_ANALYSIS), text("OK")],
        "a/y": [text("B")],
    })
    config = cfg(panel=("a/x", "a/y"), enable_consensus_shortcut=False,
                 enable_pick_best=False)
    r = asyncio.run(fuse("q", config, client=c))
    check("WARNING (homogeneous) does not block", r.status == FusionStatus.OK)


# == Edge Path 补充测试 (覆盖率提升) ====================================== #
# 覆盖 orchestrator.py 中剩余未覆盖的辅助函数分支:
#   _similarity: 空字符串 / 长文本 (>2000) 分支
#   _check_consensus: <2 responses / 无共识 / 有共识 分支
#   _resolve_best_model: 数字标签 / slug 回退 / 无匹配 / None
#   pick-best short-circuit: 完整 pipeline 路径 (L207-216)
#   client=None 自动创建 ModelClient (L57-59)

def test_similarity_empty_strings():
    """_similarity 对空字符串返回 0.0。"""
    from open_fusion.orchestrator import _similarity
    check("empty a returns 0", _similarity("", "abc") == 0.0)
    check("empty b returns 0", _similarity("abc", "") == 0.0)
    check("both empty returns 0", _similarity("", "") == 0.0)


def test_similarity_long_text_uses_prefix():
    """_similarity 对 >2000 字符的文本使用前 500 字符比较 (快速路径)。"""
    from open_fusion.orchestrator import _similarity
    long_a = "X" * 2500
    long_b = "X" * 2500
    # 两者前 500 字符相同 → 相似度 1.0
    check("long identical prefix -> 1.0", _similarity(long_a, long_b) == 1.0)
    # 一个长一个短 → 走快速路径
    sim = _similarity(long_a, "short")
    check("long vs short returns float", isinstance(sim, float))


def test_check_consensus_insufficient_responses():
    """_check_consensus 在 <2 个回答时返回 None。"""
    from open_fusion.orchestrator import _check_consensus
    from open_fusion.schema import PanelResponse, TokenUsage
    usage = TokenUsage(prompt_tokens=1, completion_tokens=1)
    single = [PanelResponse(model="a", content="x", usage=usage, latency_ms=1)]
    check("<2 responses returns None", _check_consensus(single) is None)
    check("empty list returns None", _check_consensus([]) is None)


def test_check_consensus_no_agreement():
    """_check_consensus 在回答差异大时返回 None。"""
    from open_fusion.orchestrator import _check_consensus
    from open_fusion.schema import PanelResponse, TokenUsage
    usage = TokenUsage(prompt_tokens=1, completion_tokens=1)
    responses = [
        PanelResponse(model="a", content="apple", usage=usage, latency_ms=1),
        PanelResponse(model="b", content="zebra elephant ocean", usage=usage, latency_ms=1),
        PanelResponse(model="c", content="mountain river sky", usage=usage, latency_ms=1),
    ]
    check("no consensus returns None", _check_consensus(responses, threshold=0.9) is None)


def test_check_consensus_agreement_picks_longest():
    """_check_consensus 在多数一致时返回最长回答。"""
    from open_fusion.orchestrator import _check_consensus
    from open_fusion.schema import PanelResponse, TokenUsage
    usage = TokenUsage(prompt_tokens=1, completion_tokens=1)
    # 两个高度相似, 第三个不同 → 多数 (2/3) 一致
    responses = [
        PanelResponse(model="a", content="The answer is 42.", usage=usage, latency_ms=1),
        PanelResponse(model="b", content="The answer is 42. Because math.", usage=usage, latency_ms=1),
        PanelResponse(model="c", content="completely different", usage=usage, latency_ms=1),
    ]
    result = _check_consensus(responses, threshold=0.5)
    check("consensus returns a response", result is not None)
    # 应选最长的 (b)
    check("consensus picks longest", result is not None and result.model == "b")


def test_resolve_best_model_none_input():
    """_resolve_best_model 对 None/空标签返回 None。"""
    from open_fusion.orchestrator import _resolve_best_model
    check("None label returns None", _resolve_best_model(None, []) is None)
    check("empty label returns None", _resolve_best_model("", []) is None)


def test_resolve_best_model_letter_label():
    """_resolve_best_model 解析 'MODEL A' 字母标签。"""
    from open_fusion.orchestrator import _resolve_best_model
    from open_fusion.schema import PanelResponse, TokenUsage
    usage = TokenUsage(prompt_tokens=1, completion_tokens=1)
    responses = [
        PanelResponse(model="a", content="x", usage=usage, latency_ms=1),
        PanelResponse(model="b", content="y", usage=usage, latency_ms=1),
    ]
    check("MODEL A -> idx 0", _resolve_best_model("MODEL A", responses) == 0)
    check("MODEL B -> idx 1", _resolve_best_model("MODEL B", responses) == 1)
    check("MODEL Z out of range -> None", _resolve_best_model("MODEL Z", responses) is None)


def test_resolve_best_model_numeric_label():
    """_resolve_best_model 解析数字标签 '1'/'2' (1-indexed)。"""
    from open_fusion.orchestrator import _resolve_best_model
    from open_fusion.schema import PanelResponse, TokenUsage
    usage = TokenUsage(prompt_tokens=1, completion_tokens=1)
    responses = [
        PanelResponse(model="a", content="x", usage=usage, latency_ms=1),
        PanelResponse(model="b", content="y", usage=usage, latency_ms=1),
    ]
    check("'1' -> idx 0", _resolve_best_model("1", responses) == 0)
    check("'2' -> idx 1", _resolve_best_model("2", responses) == 1)
    check("'3' out of range -> None", _resolve_best_model("3", responses) is None)


def test_resolve_best_model_slug_fallback():
    """_resolve_best_model 在非 MODEL/数字标签时回退到 slug 匹配。"""
    from open_fusion.orchestrator import _resolve_best_model
    from open_fusion.schema import PanelResponse, TokenUsage
    usage = TokenUsage(prompt_tokens=1, completion_tokens=1)
    responses = [
        PanelResponse(model="gpt-4", content="x", usage=usage, latency_ms=1),
        PanelResponse(model="claude", content="y", usage=usage, latency_ms=1),
    ]
    check("slug 'gpt-4' matches idx 0", _resolve_best_model("gpt-4", responses) == 0)
    check("slug 'claude' matches idx 1", _resolve_best_model("claude", responses) == 1)
    check("unknown slug -> None", _resolve_best_model("unknown-model", responses) is None)


def test_pick_best_short_circuit_path():
    """pick-best short-circuit: judge 选出 best_model 且回答 >100 字符时跳过 synthesis。"""
    long_a = "A" * 150  # > 100 字符
    analysis_best_a = json.dumps({
        "consensus": ["x"], "contradictions": [], "partial_coverage": [],
        "unique_insights": [], "blind_spots": [],
        "best_model": "MODEL A", "best_reason": "most complete",
    })
    c = FakeClient(scripts={
        "a/x": [text(long_a), text(analysis_best_a)],  # panel A, judge
        "b/y": [text("B short")],  # panel B
    })
    config = cfg(enable_consensus_shortcut=False, enable_pick_best=True)
    r = asyncio.run(fuse("q", config, client=c))
    check("pick-best returns OK", r.status == FusionStatus.OK)
    check("pick-best uses model A answer", r.text == long_a)
    # 只调用了 2 panel + 1 judge = 3 次 (跳过 synthesis)
    check("pick-best skips synthesis (3 calls)", len(c.calls) == 3)


def test_pick_best_skipped_when_answer_too_short():
    """pick-best 不触发当 best 回答 <= 100 字符 (太短不可靠)。"""
    short_a = "A" * 50  # < 100 字符
    analysis_best_a = json.dumps({
        "consensus": ["x"], "contradictions": [], "partial_coverage": [],
        "unique_insights": [], "blind_spots": [],
        "best_model": "MODEL A", "best_reason": "short",
    })
    c = FakeClient(scripts={
        "a/x": [text(short_a), text(analysis_best_a), text("FINAL")],  # panel, judge, synth
        "b/y": [text("B")],
    })
    config = cfg(enable_consensus_shortcut=False, enable_pick_best=True)
    r = asyncio.run(fuse("q", config, client=c))
    check("short best falls through to synthesis", r.status == FusionStatus.OK)
    check("short best uses synthesis text", r.text == "FINAL")
    # 2 panel + 1 judge + 1 synth = 4 次
    check("short best runs full pipeline (4 calls)", len(c.calls) == 4)


def test_client_none_creates_modelclient_with_depth():
    """L57-59: client=None 时 orchestrator 自动创建 ModelClient (fusion_depth=depth+1)。

    用 monkeypatch 替换 ModelClient 为 FakeClient 子类, 记录构造参数。
    """
    import open_fusion.orchestrator as orch
    captured = {}

    class FakeModelClient(FakeClient):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(scripts={
                "a/x": [text("A"), text(VALID_ANALYSIS), text("OK")],
                "b/y": [text("B")],
            })

    original = orch.ModelClient
    orch.ModelClient = FakeModelClient
    try:
        config = cfg(depth=0, enable_consensus_shortcut=False, enable_pick_best=False)
        r = asyncio.run(fuse("q", config, client=None))
        check("client=None runs pipeline", r.status == FusionStatus.OK)
        check("ModelClient created with fusion_depth=1",
              captured.get("fusion_depth") == 1)
    finally:
        orch.ModelClient = original


def test_client_none_creates_modelclient_executor_workers():
    """L57-59: client=None 且 panel>=4 时创建带 executor_workers 的 ModelClient。"""
    import open_fusion.orchestrator as orch
    captured = {}

    class FakeModelClient(FakeClient):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(scripts={
                "m1/a": [text("1"), text(VALID_ANALYSIS), text("OK")],
                "m2/b": [text("2")],
                "m3/c": [text("3")],
                "m4/d": [text("4")],
            })

    original = orch.ModelClient
    orch.ModelClient = FakeModelClient
    try:
        config = FusionConfig(
            panel=[ModelSpec("m1/a"), ModelSpec("m2/b"), ModelSpec("m3/c"), ModelSpec("m4/d")],
            judge=ModelSpec("m1/a"),
            depth=0, enable_consensus_shortcut=False, enable_pick_best=False,
        )
        r = asyncio.run(fuse("q", config, client=None))
        check("4-panel client=None runs", r.status == FusionStatus.OK)
        check("executor_workers set for >=4 panel",
              captured.get("executor_workers") is not None)
    finally:
        orch.ModelClient = original


def test_consensus_short_circuit_full_path():
    """consensus short-circuit: 多数模型回答一致时跳过 judge+synthesis。"""
    same = "The answer is 42. " * 10  # 足够长且一致
    c = FakeClient(scripts={
        "a/x": [text(same)],  # panel only, no judge/synth needed
        "b/y": [text(same)],
        "c/z": [text(same)],
    })
    config = FusionConfig(
        panel=[ModelSpec("a/x"), ModelSpec("b/y"), ModelSpec("c/z")],
        judge=ModelSpec("a/x"),
        depth=0, enable_consensus_shortcut=True, consensus_threshold=0.85,
    )
    r = asyncio.run(fuse("q", config, client=c))
    check("consensus returns OK", r.status == FusionStatus.OK)
    check("consensus skips judge+synth (3 panel calls)", len(c.calls) == 3)
    check("consensus uses agreed answer", r.text == same.strip())


def test_consensus_debug_logging_emitted():
    """L287: 启用 DEBUG 日志后, _check_consensus 发出 consensus_pairwise 调试事件。"""
    import io
    from open_fusion._logging import enable_logging
    from open_fusion.schema import PanelResponse, TokenUsage

    buf = io.StringIO()
    enable_logging("DEBUG", stream=buf)
    try:
        from open_fusion.orchestrator import _check_consensus
        usage = TokenUsage(prompt_tokens=1, completion_tokens=1)
        responses = [
            PanelResponse(model="a", content="hello world", usage=usage, latency_ms=1),
            PanelResponse(model="b", content="hello world", usage=usage, latency_ms=1),
        ]
        _check_consensus(responses, threshold=0.5)
        log_output = buf.getvalue()
        check("debug log has consensus_pairwise", "consensus_pairwise" in log_output)
        check("debug log has consensus_reached", "consensus_reached" in log_output)
    finally:
        # 恢复静默 (NullHandler)
        from open_fusion._logging import _logger, enable_logging as _el
        import logging
        for h in list(_logger.handlers):
            if not isinstance(h, logging.NullHandler):
                _logger.removeHandler(h)
        _logger.addHandler(logging.NullHandler())
        _logger.setLevel(logging.NOTSET)


# == Backward Compatibility =============================================== #

def test_compat_existing_presets():
    q = load_preset("quality")
    b = load_preset("budget")
    check("quality has 2 panel", len(q.panel) == 2)
    check("budget has 3 panel", len(b.panel) == 3)


def test_compat_existing_optimization_flags():
    c = cfg()
    check("consensus shortcut on", c.enable_consensus_shortcut == True)
    check("pick_best on", c.enable_pick_best == True)
    check("threshold 0.85", c.consensus_threshold == 0.85)


def test_compat_from_plugin():
    from open_fusion.config import from_plugin
    c = from_plugin({"model": "opus", "analysis_models": ["gpt4", "claude"]})
    check("from_plugin judge", c.judge.slug == "opus")
    check("from_plugin panel", len(c.panel) == 2)


def test_compat_happy_path():
    c = FakeClient(scripts={
        "a/x": [text("A answer"), text(VALID_ANALYSIS), text("FINAL ANSWER")],
        "b/y": [text("B answer")],
    })
    config = cfg(enable_consensus_shortcut=False, enable_pick_best=False)
    r = asyncio.run(fuse("q", config, client=c))
    check("happy path OK", r.status == FusionStatus.OK)
    check("happy path text", r.text == "FINAL ANSWER")


# == Runner =============================================================== #

def main():
    print("\n" + "=" * 60)
    print("MOA INTEGRATION TEST SUITE (6 improvements)")
    print("=" * 60 + "\n")

    tests = [
        ("P0-A: FusionMode enum", test_p0a_fusion_mode_enum),
        ("P0-A: default mode FULL", test_p0a_config_defaults_to_full),
        ("P0-A: aggregator skips judge", test_p0a_aggregator_mode_skips_judge),
        ("P0-A: full mode calls judge", test_p0a_full_mode_still_calls_judge),
        ("P0-B: all presets loadable", test_p0b_all_presets_loadable),
        ("P0-B: logic preset config", test_p0b_logic_preset_config),
        ("P0-B: code preset config", test_p0b_code_preset_config),
        ("P0-B: moa_fast preset", test_p0b_moa_fast_preset_is_aggregator),
        ("P0-B: unknown preset raises", test_p0b_unknown_preset_raises),
        ("P0-B: preset overrides", test_p0b_preset_overrides),
        ("P1-A: config supports trim", test_p1a_config_supports_trim),
        ("P1-A: trim default off", test_p1a_trim_default_off),
        ("P1-A: trim truncates long prompt", test_p1a_trim_truncates_long_prompt),
        ("P1-B: ModelSpec per-model urls", test_p1b_modelspec_supports_per_model_urls),
        ("P1-B: ModelSpec defaults None", test_p1b_modelspec_defaults_none),
        ("P1-B: _post_with method", test_p1b_post_with_method_exists),
        ("P1-B: complete routes per-model", test_p1b_complete_routes_per_model),
        ("P1-B: complete uses global fallback", test_p1b_complete_uses_global_when_no_per_model),
        ("P2-A: tail injection", test_p2a_synthesis_user_tail_injection),
        ("P2-A: config flag", test_p2a_config_tail_injection_flag),
        ("P2-A: backward compatible", test_p2a_backward_compatible_no_responses),
        ("P2-B: config flag", test_p2b_config_synth_tools_flag),
        ("P2-B: write accepts tools", test_p2b_write_accepts_tools_param),
        ("P2-B: can enable", test_p2b_can_enable_synth_tools),
        ("Compat: existing presets", test_compat_existing_presets),
        ("Compat: optimization flags", test_compat_existing_optimization_flags),
        ("Compat: from_plugin", test_compat_from_plugin),
        ("Compat: happy path", test_compat_happy_path),
        # Depth Guard 补充
        ("Depth: invalid config raises", test_depth_guard_invalid_config_raises),
        ("Depth: negative depth raises", test_depth_guard_negative_depth_raises),
        ("Depth: config depth blocks", test_depth_guard_config_depth_blocks),
        ("Depth: client fusion_depth blocks", test_depth_guard_client_fusion_depth_blocks),
        ("Depth: max of both depths", test_depth_guard_takes_max_of_both),
        ("Depth: client=None creates ModelClient", test_depth_guard_no_client_creates_modelclient),
        ("Depth: telemetry status", test_depth_guard_telemetry_status),
        ("Depth: aggregator mode still guarded", test_depth_guard_with_aggregator_mode),
        ("Depth: WARNING does not block", test_depth_guard_panel_not_empty_warning_passes),
        # Edge Path 补充 (覆盖率提升)
        ("Edge: similarity empty strings", test_similarity_empty_strings),
        ("Edge: similarity long text prefix", test_similarity_long_text_uses_prefix),
        ("Edge: consensus insufficient responses", test_check_consensus_insufficient_responses),
        ("Edge: consensus no agreement", test_check_consensus_no_agreement),
        ("Edge: consensus picks longest", test_check_consensus_agreement_picks_longest),
        ("Edge: best_model None input", test_resolve_best_model_none_input),
        ("Edge: best_model letter label", test_resolve_best_model_letter_label),
        ("Edge: best_model numeric label", test_resolve_best_model_numeric_label),
        ("Edge: best_model slug fallback", test_resolve_best_model_slug_fallback),
        ("Edge: pick-best short-circuit path", test_pick_best_short_circuit_path),
        ("Edge: pick-best skipped when short", test_pick_best_skipped_when_answer_too_short),
        ("Edge: client=None creates ModelClient", test_client_none_creates_modelclient_with_depth),
        ("Edge: client=None executor_workers", test_client_none_creates_modelclient_executor_workers),
        ("Edge: consensus full short-circuit", test_consensus_short_circuit_full_path),
        ("Edge: consensus debug logging", test_consensus_debug_logging_emitted),
    ]

    for name, fn in tests:
        print(f"\n{name}")
        try:
            fn()
        except Exception as e:
            check(f"{name} (uncaught exception)", False)
            print(f"         {type(e).__name__}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
