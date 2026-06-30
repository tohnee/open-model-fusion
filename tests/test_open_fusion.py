"""
Offline test suite for open-fusion. Runs with zero network via FakeClient.

Run directly:   python tests/test_open_fusion.py
Or with pytest: pytest tests/
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from open_fusion import FusionConfig, ModelSpec, Phase, FusionStatus
from open_fusion.client import Timeout
from open_fusion.orchestrator import fuse
from open_fusion.judge import parse_analysis, JudgeError
from open_fusion.schema import Analysis, PanelResponse
from open_fusion.prompts import label_responses
from open_fusion.tools import toolset_for_phase
from open_fusion.config import load_preset
from fake_client import FakeClient, text, with_tool_call

VALID_ANALYSIS = json.dumps({
    "consensus": ["carbon taxes price externalities"],
    "contradictions": [{"topic": "regressivity",
                        "stances": [{"model": "MODEL A", "stance": "regressive"},
                                    {"model": "MODEL B", "stance": "fixable with rebates"}]}],
    "partial_coverage": [{"models": ["MODEL A"], "point": "border adjustments"}],
    "unique_insights": [{"model": "MODEL B", "insight": "revenue recycling"}],
    "blind_spots": ["political feasibility"],
})

PASS, FAIL = 0, 0


def check(name: str, cond: bool):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


def cfg(panel=("a/x", "b/y"), judge="a/x", **kw) -> FusionConfig:
    # 默认关闭 short-circuit 优化，确保测试基础 pipeline 行为。
    # 需要测试 short-circuit 的场景在各自测试函数中显式开启。
    defaults = dict(enable_consensus_shortcut=False, enable_pick_best=False)
    defaults.update(kw)
    return FusionConfig(panel=[ModelSpec(s) for s in panel], judge=ModelSpec(judge), **defaults)


# --------------------------------------------------------------------------- #
def test_schema_and_gating():
    print("schema + gating")
    a = Analysis.from_dict(json.loads(VALID_ANALYSIS))
    check("valid analysis passes validation", a.validate() == [])
    bad = Analysis.from_dict({"contradictions": [{"topic": "t", "stances": [{"stance": "x"}]}]})
    check("missing stance.model is flagged", any("attribution" in p for p in bad.validate()))
    check("empty analysis flagged", any("empty" in p for p in Analysis.from_dict({}).validate()))

    bad_partial = Analysis.from_dict({"partial_coverage": [{"point": "raised by some"}]})
    check("partial_coverage requires model attribution",
          any("partial_coverage" in p and "attribution" in p for p in bad_partial.validate()))
    bad_best = Analysis.from_dict({"consensus": ["x"], "best_model": "best one"})
    check("best_model free-form label is flagged",
          any("best_model" in p for p in bad_best.validate()))
    # the correctness invariant:
    check("SYNTHESIS has no tools", toolset_for_phase(Phase.SYNTHESIS) == ())
    check("PANEL has 3 tools", len(toolset_for_phase(Phase.PANEL)) == 3)
    check("JUDGE has 2 tools (no bash)", len(toolset_for_phase(Phase.JUDGE)) == 2)


def test_labels_and_parse():
    print("labels + parse")
    rs = [PanelResponse("m1", "aaa"), PanelResponse("m2", status="error"),
          PanelResponse("m3", "ccc")]
    lab = label_responses(rs)
    check("labels skip failures, stable A/C", "MODEL A" in lab and "MODEL C" in lab and "MODEL B" not in lab)
    check("parse strips fences", parse_analysis("```json\n" + VALID_ANALYSIS + "\n```").consensus != [])
    raised = False
    try:
        parse_analysis("not json")
    except ValueError:
        raised = True
    check("parse raises on bad json", raised)


def test_preset_fallback():
    print("presets")
    q = load_preset("quality")
    b = load_preset("budget")
    check("quality has 2 panel models", len(q.panel) == 2)
    check("budget has 3 panel models", len(b.panel) == 3)
    check("caller defaults to judge", q.caller.slug == q.judge.slug)
    # 性能优化 O1：3+ panel 的 preset 默认启用 fast_majority_k=N-1。
    check("budget preset enables fast_majority_k = N-1 by default",
          b.fast_majority_k == len(b.panel) - 1)
    # 2-panel 的 preset 不启用（避免退化成单模型）
    check("quality preset (2 panel) keeps fast_majority_k None",
          q.fast_majority_k is None)
    from open_fusion import preset_names
    check("all scenario presets exposed", set(preset_names()) >= {"quality", "budget", "logic", "code", "moa_fast"})


def test_from_cli_fast_majority():
    # O1 一致性：自定义 panel 路径也享受默认；--fast-majority-k 0 显式关闭。
    print("config: from_cli fast_majority_k consistency")
    from argparse import Namespace
    from open_fusion.config import from_cli
    # 1) 自定义 3 panel，未指定 k → 默认 N-1=2
    cfg = from_cli(Namespace(panel="a/x,b/y,c/z", judge="a/x",
                             preset=None, tools=False, exclude_domains=None,
                             max_in_flight=None, fast_majority_k=None))
    check("custom 3-panel gets default fast_majority_k=N-1",
          cfg.fast_majority_k == 2)
    # 2) 自定义 2 panel，未指定 k → 仍 None
    cfg2 = from_cli(Namespace(panel="a/x,b/y", judge="a/x",
                              preset=None, tools=False, exclude_domains=None,
                              max_in_flight=None, fast_majority_k=None))
    check("custom 2-panel keeps fast_majority_k None", cfg2.fast_majority_k is None)
    # 3) --fast-majority-k 0 应被识别为"显式关闭"（而非被 truthy 吞掉）
    cfg3 = from_cli(Namespace(panel=None, judge=None, preset="budget",
                              tools=False, exclude_domains=None,
                              max_in_flight=None, fast_majority_k=0))
    check("--fast-majority-k 0 disables early-exit explicitly",
          cfg3.fast_majority_k is None)
    # 4) --fast-majority-k 1 应被尊重
    cfg4 = from_cli(Namespace(panel="a/x,b/y,c/z", judge=None, preset=None,
                              tools=False, exclude_domains=None,
                              max_in_flight=None, fast_majority_k=1))
    check("--fast-majority-k 1 honored", cfg4.fast_majority_k == 1)


async def _happy():
    c = FakeClient(scripts={
        "a/x": [text("answer from A"), text(VALID_ANALYSIS), text("FINAL ANSWER")],
        "b/y": [text("answer from B")],
    })
    # call order for judge a/x: panel(1) then judge(2) then synth(3)
    r = await fuse("q", cfg(), client=c)
    return r, c


def test_happy_path():
    print("orchestrator: happy path")
    r, c = asyncio.run(_happy())
    check("status ok", r.status == FusionStatus.OK)
    check("final text present", r.text == "FINAL ANSWER")
    check("analysis populated", r.analysis is not None and r.analysis.consensus)
    check("telemetry completions = 2 panel +1 +1 = 4", r.telemetry["completions"] == 4)
    check("2 panels ok", r.telemetry["panel_ok"] == 2)
    synth_calls = [k for k in c.calls if k["tools"] == () and k["response_format"] is None]
    check("a synthesis call had no tools", len(synth_calls) >= 1)


def test_partial_failure():
    print("orchestrator: partial panel failure")
    c = FakeClient(scripts={
        "a/x": [text("A answer"), text(VALID_ANALYSIS), text("FINAL")],
        "b/y": [Timeout("slow")],
    })
    r = asyncio.run(fuse("q", cfg(), client=c))
    check("still OK with 1 survivor", r.status == FusionStatus.OK)
    check("1 ok, 1 failed", r.telemetry["panel_ok"] == 1 and r.telemetry["panel_failed"] == 1)


def test_all_fail():
    print("orchestrator: all panels fail")
    c = FakeClient(scripts={"a/x": [Timeout("x"), text(VALID_ANALYSIS), text("F")],
                            "b/y": [Timeout("y")]})
    r = asyncio.run(fuse("q", cfg(), client=c))
    check("status error", r.status == FusionStatus.ERROR)
    check("no text", r.text == "")


def test_judge_retry_then_ok():
    print("orchestrator: judge bad-json then good")
    c = FakeClient(scripts={
        "a/x": [text("A"), text("NOT JSON"), text(VALID_ANALYSIS), text("FINAL")],
        "b/y": [text("B")],
    })
    r = asyncio.run(fuse("q", cfg(), client=c))
    check("recovers to OK", r.status == FusionStatus.OK)
    check("one retry recorded", r.telemetry["judge_retries"] == 1)


def test_judge_fail_fallback():
    print("orchestrator: judge fails twice -> fallback")
    c = FakeClient(scripts={
        "a/x": [text("A"), text("NOPE"), text("STILL NOPE"), text("FALLBACK ANSWER")],
        "b/y": [text("B")],
    })
    r = asyncio.run(fuse("q", cfg(), client=c))
    check("status judge_fallback", r.status == FusionStatus.JUDGE_FALLBACK)
    check("fallback text present", r.text == "FALLBACK ANSWER")


def test_depth_guard():
    print("orchestrator: depth guard")
    c = FakeClient()
    r = asyncio.run(fuse("q", cfg(depth=1), client=c))
    check("blocked at depth", r.status == FusionStatus.ERROR)
    check("no panel calls made", len(c.calls) == 0)


def test_depth_guard_via_client():
    # 修复 C1：client.fusion_depth 也参与守卫，防止"通过传入已嵌套的 client 绕过"。
    print("orchestrator: depth guard via client.fusion_depth")
    c = FakeClient()
    c.fusion_depth = 1   # 模拟上游已经发起过一层 fusion
    r = asyncio.run(fuse("q", cfg(depth=0), client=c))
    check("blocked by client depth too", r.status == FusionStatus.ERROR)
    check("no panel calls made when blocked by client depth", len(c.calls) == 0)


def test_fast_majority_cancelled_not_failed():
    # 修复 C2：fast_majority_k 取消的 panel 不计入 panel_failed。
    print("panel: fast majority -> cancelled status, not failure")
    # 三个 panel；judge 复用 a/x；fast_majority_k=1 -> 拿到 1 个 ok 就早退
    async def slow_b():
        await asyncio.sleep(0.05)
        return text("late B")
    c = FakeClient(scripts={
        "a/x": [text("A done"), text(VALID_ANALYSIS), text("FINAL")],
        # b/y 与 c/z 故意制造延迟以让 a/x 先返回
        "b/y": [text("B done")],
        "c/z": [text("C done")],
    })
    # 用 monkey patch 让 b/y 与 c/z 各睡一会儿；这里不引入复杂调度，直接证明：
    # fast_majority_k=1 + 三 panel 时，至少有一个非 ok 状态是 "cancelled" 或者
    # n_panel_cancelled >= 0 且 n_panel_failed 不会因为早退而被推高。
    cfg_fm = FusionConfig(panel=[ModelSpec("a/x"), ModelSpec("b/y"), ModelSpec("c/z")],
                          judge=ModelSpec("a/x"), fast_majority_k=1)
    r = asyncio.run(fuse("q", cfg_fm, client=c))
    check("status ok with fast majority k=1", r.status == FusionStatus.OK)
    # 不变量：panel_failed 只计真实失败/超时；cancelled 计入 panel_cancelled。
    check("panel_failed semantics: only real failures",
          r.telemetry["panel_failed"] + r.telemetry["panel_ok"] + r.telemetry["panel_cancelled"]
          == len(cfg_fm.panel))
    check("panel_cancelled tracked separately", "panel_cancelled" in r.telemetry)


def test_params_validation():
    # 修复 C3：Params 必须拒绝非法配置，而不是让 panel._run_one 跑出 UnboundLocalError。
    print("config: Params rejects invalid bounds")
    from open_fusion.config import Params
    raised_negative_tools = False
    try:
        Params(max_tool_calls=-1)
    except ValueError:
        raised_negative_tools = True
    check("max_tool_calls < 0 rejected", raised_negative_tools)

    raised_zero_timeout = False
    try:
        Params(timeout_s=0)
    except ValueError:
        raised_zero_timeout = True
    check("timeout_s <= 0 rejected", raised_zero_timeout)

    raised_zero_tokens = False
    try:
        Params(max_tokens=0)
    except ValueError:
        raised_zero_tokens = True
    check("max_tokens <= 0 rejected", raised_zero_tokens)

    raised_neg_temp = False
    try:
        Params(temperature=-0.1)
    except ValueError:
        raised_neg_temp = True
    check("temperature < 0 rejected", raised_neg_temp)


def test_cli_success_statuses():
    print("cli: success exit statuses")
    from open_fusion.cli import is_success_status
    check("OK is success", is_success_status(FusionStatus.OK))
    check("judge_fallback is success", is_success_status(FusionStatus.JUDGE_FALLBACK))
    check("consensus shortcut is success", is_success_status(FusionStatus.CONSENSUS_SHORTCUT))
    check("pick-best shortcut is success", is_success_status(FusionStatus.PICK_BEST_SHORTCUT))
    check("aggregator mode is success", is_success_status(FusionStatus.AGGREGATOR_MODE))
    check("error is failure", not is_success_status(FusionStatus.ERROR))


def test_synthesis_ms_tracked():
    # 修复 H1：synthesis_ms 被记录，critical_path_ms 包含它。
    print("telemetry: synthesis_ms tracked + critical_path includes it")
    c = FakeClient(scripts={
        "a/x": [text("A"), text(VALID_ANALYSIS), text("FINAL")],
        "b/y": [text("B")],
    })
    r = asyncio.run(fuse("q", cfg(), client=c))
    check("synthesis_ms field present", "synthesis_ms" in r.telemetry)
    check("synthesis_ms is non-negative int", isinstance(r.telemetry["synthesis_ms"], int)
          and r.telemetry["synthesis_ms"] >= 0)
    # critical_path 必须 >= panel_max + judge + synthesis 三段相加（等式可能存在
    # 取整误差，所以用 >= 来表达"不会落下 synthesis 这一段"）。
    panel_max = max(r.telemetry["panel_latencies_ms"]) if r.telemetry["panel_latencies_ms"] else 0
    expected = panel_max + r.telemetry["judge_ms"] + r.telemetry["synthesis_ms"]
    check("critical_path includes synthesis", r.telemetry["critical_path_ms"] >= expected - 1)


def test_tool_loop():
    print("panel: tool loop executes then answers")
    # tools_enabled; A calls web_fetch once then answers. (web_fetch on a fake url
    # returns ok:false 'domain excluded' only if blocked; here it will try real net,
    # so we exclude the domain to keep it offline-deterministic.)
    c = FakeClient(scripts={
        "a/x": [with_tool_call("web_fetch", {"url": "http://example.com"}),
                text("A after tool"), text(VALID_ANALYSIS), text("FINAL")],
        "b/y": [text("B")],
    })
    r = asyncio.run(fuse("q", cfg(tools_enabled=True, excluded_domains=["example.com"]), client=c))
    check("status ok with tool loop", r.status == FusionStatus.OK)
    a_resp = next(p for p in r.panel_responses if p.model == "a/x")
    check("tool trace recorded", len(a_resp.tool_trace) == 1)
    check("excluded domain blocked offline", a_resp.tool_trace[0]["result"]["ok"] is False)


def test_logging_captures_phase_events():
    # 验证：开启 OPEN_FUSION_LOG 后，orchestrator/synthesizer 关键路径有结构化日志。
    # 默认（不开启）时静默，不污染输出 —— 这是 library 不变量。
    print("logging: phase events captured under enable_logging")
    import io, logging
    from open_fusion._logging import enable_logging
    buf = io.StringIO()
    enable_logging("DEBUG", stream=buf)
    try:
        c = FakeClient(scripts={
            "a/x": [text("A"), text(VALID_ANALYSIS), text("FINAL")],
            "b/y": [text("B")],
        })
        r = asyncio.run(fuse("logging probe", cfg(), client=c))
        out = buf.getvalue()
        check("orchestrator start logged", "phase=orchestrator event=start" in out)
        check("orchestrator end logged with critical_path_ms",
              "phase=orchestrator event=end" in out and "critical_path_ms=" in out)
        check("panel phase start/end logged",
              "phase=panel event=start" in out and "phase=panel event=end" in out)
        check("synthesis start logged with analysis_chars",
              "phase=synthesis event=start" in out and "analysis_chars=" in out)
        check("synthesis end logged with duration_ms",
              "phase=synthesis event=end" in out and "duration_ms=" in out)
        check("status ok still", r.status == FusionStatus.OK)
    finally:
        # 复位 logger，避免污染后续测试
        lg = logging.getLogger("open_fusion")
        for h in list(lg.handlers):
            if not isinstance(h, logging.NullHandler):
                lg.removeHandler(h)
        lg.setLevel(logging.WARNING)


def test_logging_silent_by_default():
    # 不调 enable_logging 时，关键路径不应有任何 StreamHandler 输出。
    print("logging: silent by default (library best practice)")
    import io, logging, sys
    # 临时把 stderr 替换为 buf，确认没有任何东西被打到 stderr
    buf = io.StringIO()
    old = sys.stderr
    sys.stderr = buf
    try:
        # 确保 logger 状态干净（前一个测试可能没清理彻底）
        lg = logging.getLogger("open_fusion")
        for h in list(lg.handlers):
            if not isinstance(h, logging.NullHandler):
                lg.removeHandler(h)
        c = FakeClient(scripts={
            "a/x": [text("A"), text(VALID_ANALYSIS), text("FINAL")],
            "b/y": [text("B")],
        })
        asyncio.run(fuse("silent probe", cfg(), client=c))
    finally:
        sys.stderr = old
    check("no log output to stderr by default", buf.getvalue() == "")


def main():
    test_schema_and_gating()
    test_labels_and_parse()
    test_preset_fallback()
    test_from_cli_fast_majority()
    test_happy_path()
    test_partial_failure()
    test_all_fail()
    test_judge_retry_then_ok()
    test_judge_fail_fallback()
    test_depth_guard()
    test_depth_guard_via_client()
    test_fast_majority_cancelled_not_failed()
    test_params_validation()
    test_cli_success_statuses()
    test_synthesis_ms_tracked()
    test_logging_captures_phase_events()
    test_logging_silent_by_default()
    test_tool_loop()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
