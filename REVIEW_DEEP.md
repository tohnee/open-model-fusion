# Open Fusion 深度 Review 报告

日期：2026-06-30

## 1. Review 范围

本报告对 `open_fusion` 的核心实现、OpenRouter Fusion 机制对齐度、Hermes MoA 机制对齐度、可审计性、容错性、成本/延迟路径和测试覆盖进行了交叉验证。

重点文件：

- `src/open_fusion/orchestrator.py`：顶层状态机与 shortcut 路由。
- `src/open_fusion/panel.py`：并行 fan-out 与工具循环。
- `src/open_fusion/judge.py`：结构化 JSON judge 与 retry。
- `src/open_fusion/synthesizer.py`：最终写作与 fallback 写作。
- `src/open_fusion/schema.py`：跨阶段数据契约与验证。
- `src/open_fusion/tools.py`：phase-to-tool gate。
- `src/open_fusion/config.py`：Fusion/MoA 配置、预设、多 provider 字段。

## 2. 总体结论

当前实现已经形成一个可运行的单层 Fusion / MoA hybrid：

1. **FULL 模式**基本对齐 OpenRouter Fusion：panel 独立并行回答，judge 产出结构化 `Analysis`，synthesizer 基于 frozen evidence 写最终答案。
2. **AGGREGATOR 模式**吸收了 Hermes MoA 的“跳过 judge、原始回答直聚合”思想，但仍是一次性 synthesis，并非完整 Hermes 式可携带完整系统提示、工具 schema、多轮 agent loop 的 aggregator。
3. 代码具备较完整的容错：单 panel 失败不会中断；全部 panel 失败返回 error；judge 两次 JSON 失败后降级为 raw-response synthesis；depth guard 防止递归 fusion。
4. 主要缺口在机制语义边界：shortcut status 不明确、partial coverage 归因校验不足、长文本 consensus 只看前缀有误判风险、`synth_tools_enabled` 与 evidence-freeze 设计存在语义张力。

## 3. 与 OpenRouter Fusion 原理对比

### 3.1 对齐项

- **单层 MoA**：只有一层 panel fan-out，没有深层递归 MoA stack。
- **结构化 judge**：judge 输出 `consensus / contradictions / partial_coverage / unique_insights / blind_spots / best_model / best_reason`。
- **可审计归因**：contradiction stances、unique insights 必须带 model attribution。
- **冻结证据**：FULL synthesis 默认无工具，避免最终答案漂移出 judge analysis。
- **异构 panel 提醒**：同 vendor panel 会给 warning。

### 3.2 不足项

- Shortcut 直接绕过 judge/synthesis，可能削弱“显式 adjudication”的 Fusion 核心价值。
- 原实现中 shortcut 返回 `OK`，外部调用方无法从 `status` 上区分完整 Fusion 和 shortcut 结果。
- 原 `partial_coverage` 没有强制模型归因，削弱覆盖面审计能力。
- 原长文本 consensus 只比较前 500 字符，若开头相同但结论相反，可能误触发 shortcut。

## 4. 与 Hermes MoA 原理对比

### 4.1 对齐项

- 已有 `FusionMode.AGGREGATOR`，可跳过 judge，走 `panel -> raw-response synthesis`。
- 已有场景化 preset，例如 `logic`、`code`、`moa_fast`。
- 已有 per-model `base_url/api_key`，支持 OpenAI-compatible 多网关路由。
- 已有 panel trim 与 tail injection 思路。

### 4.2 不足项

- Hermes MoA 的 aggregator 通常可以携带完整系统提示和工具 schema；当前 aggregator mode 复用 fallback synthesis，且 fallback 明确无工具。
- `synth_tools_enabled` 字段存在，但 FULL synthesis 的正确性仍应保持 no-tools。若未来要支持 Hermes-style tool aggregator，建议新增独立 `AGGREGATOR` phase，而不是改变 `SYNTHESIS` phase 的 frozen-evidence 不变量。
- Panel trim 当前是简单字符截断，可能破坏结构化 prompt、代码块或后置约束。

## 5. 本次落地修复

本次根据 review 结论优先修复“不改变主架构但提升可审计性/正确性”的问题：

1. **Shortcut status 显式化**
   - consensus shortcut 返回 `FusionStatus.CONSENSUS_SHORTCUT`。
   - pick-best shortcut 返回 `FusionStatus.PICK_BEST_SHORTCUT`。
   - 这样调用方可以区分完整 judge+synth 结果和 shortcut 结果。

2. **加强 `Analysis.validate()`**
   - `partial_coverage` 必须是 object。
   - `partial_coverage.models` 必须是非空 list 且元素为非空字符串。
   - `best_model` 如果非 null，必须是 `MODEL X`、数字标签或包含 `/` 的模型 slug，避免 free-form label 破坏后续解析。

3. **降低长文本 consensus 误判**
   - 长文本相似度由“只比较前 500 字符”改为“head + tail 采样”。
   - 这样能捕获“相同铺垫 + 相反结论”的典型误判场景。

4. **补充测试**
   - 增加 partial coverage 归因校验测试。
   - 增加 best model free-form label 校验测试。
   - 更新 shortcut status 测试。
   - 更新长文本 similarity 测试，覆盖 tail divergence。

## 6. 保留建议

后续建议按优先级继续推进：

### P0：明确 `synth_tools_enabled` 的产品语义

如果目标是忠实 OpenRouter Fusion：删除或隐藏 `synth_tools_enabled`，保持 synthesis no-tools。

如果目标是 Fusion + Hermes 双模式：新增独立 `AGGREGATOR` phase 和 aggregator tool loop，仅 aggregator mode 可启用工具，FULL synthesis 继续 frozen evidence。

### P1：pick-best 策略改为 best-as-spine

当前 pick-best 直接返回 best model 原文，可能损失其他模型的 unique insights。建议增加策略：以 best answer 为主干，但仍让 synthesizer 合并 judge analysis 中的 consensus、unique insights 和 blind spots。

### P1：改进 panel trim

简单字符截断应升级为 head-tail 或结构化裁剪，尤其对代码、JSON、长上下文任务更安全。

### P2：多 provider adapter

当前多 provider 仍假设 OpenAI-compatible `/chat/completions`。如果要支持 Anthropic/Gemini 原生 API，应引入 provider adapter 层。

## 7. 测试验证

本次修复后运行：

```bash
python tests/test_open_fusion.py
python tests/test_moa_integration.py
python tests/test_eval.py
python tests/test_ablation.py
python tests/test_eval_suites.py
python -m compileall src tests
```

详见最终回复中的测试结果。
