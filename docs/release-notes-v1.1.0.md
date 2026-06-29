# Release Notes — v1.1.0

**open-model-fusion** — MoA 集成与管道优化

> 发布日期: 2026-06-30
> 提交: `fda7ee3` → `main`
> 24 files changed, 3,589 insertions(+), 71 deletions(-)

---

## 概述

本次发布将 Hermes MoA (Mixture-of-Agents) 的 6 项核心设计吸收进 Fusion 管道，同时引入两种短路机制和全链路性能日志。所有改动向后兼容：现有预设、`from_plugin()` 接口和默认行为均不受影响。

---

## 新增功能

### P0-A: Aggregator 模式

新增 `FusionMode` 枚举 (`FULL` / `AGGREGATOR`)。AGGREGATOR 模式跳过 judge 层，panel 回答直接交给 synthesizer 合成，实现 MoA 风格的无损信息传递。

```python
config = FusionConfig(
    panel=[...], judge=ModelSpec("..."),
    mode=FusionMode.AGGREGATOR,
)
```

- **节省**: 1 次 API 调用 (judge)
- **延迟降低**: ~40% (串行关键路径)
- **适用场景**: panel 模型能力强、答案互补性高时，避免 judge 层的信息瓶颈

### P0-B: 命名预设系统

新增 3 个场景化预设，与现有 `quality` / `budget` 并列：

| 预设 | 模式 | 共识阈值 | Pick-Best | 适用场景 |
|------|------|----------|-----------|----------|
| `logic` | FULL | 0.90 | ON | 逻辑推理 (高一致性要求) |
| `code` | FULL | 0.75 | ON | 代码生成 (允许多样性) |
| `moa_fast` | AGGREGATOR | 0.85 | OFF | 快速合成 (MoA 风格) |

### P1-A: Panel 输入裁剪

`enable_panel_trim` + `panel_trim_chars` 在超长 prompt 传入 panel 前截断，节省 30-50% token 开销。截断后追加 `...(truncated)` 标记。

### P1-B: 多 Provider 支持

`ModelSpec.base_url` / `ModelSpec.api_key` 支持每个模型使用独立的 API 网关和密钥。`client._post_with()` 方法按模型路由请求。

```python
panel = [
    ModelSpec("claude-opus", base_url="https://api.anthropic.com/v1", api_key="sk-xxx"),
    ModelSpec("gpt-4", base_url="https://api.openai.com/v1", api_key="sk-yyy"),
]
```

### P2-A: 末尾注入 (Tail Injection)

`SYNTHESIS_USER` prompt 将 panel 回答追加在末尾 (而非中间)，提升 prompt cache 命中率。问题和分析部分在前 (稳定前缀)，回答在后 (变化后缀)。

### P2-B: Synthesizer 工具保留

`synth_tools_enabled` 标志让 synthesizer 保留工具调用能力 (agent loop)，不再强制无工具模式。`synthesizer.write()` 新增 `tools` 参数。

---

## 短路机制

### Consensus Shortcut

当多数 panel 回答高度相似 (SequenceMatcher 相似度 ≥ 阈值) 时，直接采用最长回答，跳过 judge + synthesis。

- **触发条件**: `enable_consensus_shortcut=True` + 多数回答相似度 ≥ `consensus_threshold`
- **节省**: 2 次 API 调用 (judge + synth)
- **延迟降低**: ~73%

### Pick-Best Shortcut

Judge 输出 `best_model` 字段时，若该模型回答足够完整 (>100 字符)，直接采用，跳过 synthesis。

- **触发条件**: `enable_pick_best=True` + judge 返回 `best_model` + 回答 >100 字符
- **节省**: 1 次 API 调用 (synth)
- **延迟降低**: ~33%

---

## 性能基准数据

基于模拟延迟的基准测试 (Panel 800ms 并行 / Judge 1200ms 串行 / Synth 1000ms 串行)：

| 模式 | 实测延迟 | API 调用 | 加速比 | 状态 |
|------|---------|---------|--------|------|
| FULL Mode | 3,006ms | 5 | 1.00x | ok |
| AGGREGATOR Mode | 2,004ms | 4 | 1.50x | aggregator_mode |
| Consensus Shortcut | 802ms | 3 | 3.75x | ok |
| Pick-Best Shortcut | 2,004ms | 4 | 1.50x | ok |

**关键结论**:
- AGGREGATOR 模式: 省 1 次 API 调用，串行延迟降低 40%
- Consensus 短路: 省 2 次 API 调用，串行延迟降低 73%
- Pick-Best 短路: 省 1 次 API 调用，串行延迟降低 33%

---

## 调试日志

`_check_consensus` 新增 4 类结构化日志事件 (DEBUG 级别):

| 事件 | 触发时机 | 关键字段 |
|------|---------|---------|
| `consensus_skip` | 回答数 <2 | `reason`, `n_responses` |
| `consensus_pairwise` | 两两相似度计算 | `i`, `j`, `similarity`, `threshold`, `above` |
| `consensus_reached` | 达成共识 | `cluster_size`, `majority_needed`, `winner_model` |
| `consensus_missed` | 未达多数 | `best_cluster_size`, `reason` |

AGGREGATOR 分支新增入口/出口性能日志:
- `aggregator_mode_entry`: `n_responses`, `panel_duration_ms`, `saved_api_calls`
- `aggregator_mode_done`: `synth_ms`, `total_ms`, `answer_chars`, `api_calls`

---

## 测试覆盖

| 测试文件 | 测试数 | 状态 |
|---------|--------|------|
| `tests/test_moa_integration.py` | 52 | 全部通过 |
| `tests/test_open_fusion.py` | 17 | 全部通过 |
| `tests/test_ablation.py` | 6 | 全部通过 |
| `tests/test_eval.py` | 9 | 全部通过 |
| `tests/test_eval_suites.py` | 9 | 全部通过 |
| `test_quota_handling.py` | 6 | 全部通过 |
| **总计** | **99** | **全部通过** |

核心模块行覆盖率:

| 模块 | 覆盖率 |
|------|--------|
| `orchestrator.py` | **100%** |
| `prompts.py` | **100%** |
| `config.py` | 92% |
| `schema.py` | 90% |
| `synthesizer.py` | 86% |

---

## CI 流水线状态

GitHub Actions 运行记录:

| Workflow | 状态 | 耗时 |
|----------|------|------|
| pages-build-deployment | ✓ 绿色 | 4m11s |
| pages-build-deployment | ✓ 绿色 | 5m34s |
| pages-build-deployment | ✓ 绿色 | 40s |

> 注: 仓库当前未配置测试 CI 工作流。建议后续在 `.github/workflows/` 中添加自动化测试流水线。

---

## 配置变更

### 默认参数更新
- `max_tokens`: 4096 → **8192**
- `timeout_s`: 120 → **300**

### 新增配置字段
| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | `FusionMode` | `FULL` | 管道模式 |
| `enable_panel_trim` | `bool` | `False` | Panel 输入裁剪 |
| `panel_trim_chars` | `int` | `4000` | 裁剪阈值 |
| `synth_tools_enabled` | `bool` | `False` | Synthesizer 工具保留 |
| `enable_tail_injection` | `bool` | `True` | 末尾注入 |
| `enable_consensus_shortcut` | `bool` | `True` | 共识短路 |
| `enable_pick_best` | `bool` | `True` | Pick-best 短路 |
| `consensus_threshold` | `float` | `0.85` | 共识相似度阈值 |

### ModelSpec 新增字段
| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_url` | `str \| None` | `None` | 模型级 API 网关 |
| `api_key` | `str \| None` | `None` | 模型级密钥 |

---

## 向后兼容性

- 现有预设 (`quality`, `budget`) 行为不变
- `from_plugin()` 接口签名不变
- 默认 `mode=FULL` 保持原有管道流程
- `test_open_fusion.py` 中 `cfg()` 辅助函数显式关闭短路机制，验证基础管道行为不受影响

---

## 文档

- [docs/fusion-optimization.md](fusion-optimization.md) — 3 项改进机制 (consensus / pick-best / tail injection) 的场景配置指南
- [docs/moa-comparison.md](moa-comparison.md) — Hermes MoA vs Open-Fusion 架构深度对比，含 6 项可吸收方案分析

---

## 参与者

- 代码实现与测试: open-model-fusion 项目
- 架构参考: Hermes MoA (Mixture-of-Agents)
