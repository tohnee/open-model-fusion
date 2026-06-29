# Fusion Pipeline 优化机制文档

## 概述

本目录下的 Fusion pipeline 通过三层 short-circuit 机制优化多模型聚合的**质量**和**速度**。
每个 short-circuit 点在命中时跳过后续步骤，减少 API 调用并避免 synthesizer 破坏好答案。

## Pipeline 流程

```
Panel 执行 (N 次 API 调用)
    │
    ▼
┌─────────────────────────────┐
│ ① Consensus Short-Circuit   │  多数模型答案一致?
│    threshold = 0.85         │  → 直接采用最长回答，跳过 Judge + Synthesis
└──────────┬──────────────────┘
           │ 未命中
           ▼
┌─────────────────────────────┐
│ ② Judge 分析 (1 次 API 调用) │  Judge 产出 JSON 分析 + best_model
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ ③ Pick-Best Short-Circuit   │  Judge 识别出 best_model 且回答 >100 字符?
│                             │  → 直接采用该模型回答，跳过 Synthesis
└──────────┬──────────────────┘
           │ 未命中
           ▼
┌─────────────────────────────┐
│ ④ Synthesis (1 次 API 调用)  │  Synthesizer 融合所有回答，产出最终答案
└─────────────────────────────┘
```

## API 调用次数对比

| 场景 | Panel | Judge | Synthesis | 总调用 | vs 默认 |
|------|------:|------:|----------:|-------:|--------:|
| 默认（无 short-circuit） | N | 1 | 1 | N+2 | — |
| ① Consensus 命中 | N | 0 | 0 | **N** | -2 |
| ② Pick-Best 命中 | N | 1 | 0 | **N+1** | -1 |
| ④ Synthesis 兜底 | N | 1 | 1 | N+2 | 0 |

> N = panel 模型数量（通常 3）。实际场景中 Consensus 命中率约 20-30%，Pick-Best 命中率约 40-60%。

## 配置参数

```python
from open_fusion.config import FusionConfig, ModelSpec

base = FusionConfig(
    panel=[ModelSpec("glm-5.1"), ModelSpec("kimi-k2.6"), ModelSpec("deepseek-v4-flash")],
    judge=ModelSpec("doubao-seed-2.0-pro"),
    caller=ModelSpec("doubao-seed-2.0-pro"),  # synthesizer，可独立配置

    # --- 优化开关 ---
    enable_consensus_shortcut=True,   # ① 多数一致时跳过 judge+synthesis
    enable_pick_best=True,            # ③ judge 识别 best_model 时直接采用
    consensus_threshold=0.85,         # 文本相似度阈值 (0.0-1.0)
)
```

## 三个改进点详解

### ① Consensus Short-Circuit（多数一致短路）

**位置**: orchestrator.py → panel 执行之后、judge 之前

**机制**: 对所有成功的 panel 回答做两两文本相似度比较（SequenceMatcher）。
如果有 ≥ ⌈N/2⌉+1 个回答相似度 ≥ threshold，直接采用其中最长的回答。

**适用场景**:
- 数学计算题：多个模型算出相同数值答案
- 事实检索题：多个模型给出一致的事实陈述
- 代码生成题：多个模型写出结构相同的代码
- 简单问答：答案明确无歧义

**不适用场景**:
- 开放式论述题（不同表述方式但都正确）
- 创意写作题（答案天然多样化）
- 推理证明题（不同证明路径但结论相同）

**调参建议**:
| threshold | 效果 | 适用场景 |
|-----------|------|---------|
| 0.90 | 严格，几乎只有完全相同才命中 | 数学/事实题 |
| 0.85 (默认) | 平衡 | 通用 |
| 0.75 | 宽松，结构相似即命中 | 代码题 |

**关闭方式**: `enable_consensus_shortcut=False`

---

### ② Pick-Best Short-Circuit（择优短路）

**位置**: orchestrator.py → judge 分析之后、synthesis 之前

**机制**: Judge 在 JSON 分析中额外输出 `best_model` 和 `best_reason` 字段。
如果 best_model 对应的回答 >100 字符（确保非空且有一定完整度），直接采用该回答。

**Judge prompt 新增字段**:
```json
{
  "consensus": [...],
  "contradictions": [...],
  "partial_coverage": [...],
  "unique_insights": [...],
  "blind_spots": [...],
  "best_model": "MODEL C",        // 最强模型的标签
  "best_reason": "最完整的枚举证明，结论正确"  // 选择理由
}
```

**适用场景**:
- 逻辑推理题：一个模型给出完整证明，其他模型推理有误
- 知识深度题：一个模型覆盖了其他模型缺失的关键点
- 代码调试题：一个模型定位了 bug，其他模型遗漏

**不适用场景**:
- 多模型各有亮点且互补（需要 synthesis 融合）
- 所有模型答案质量接近（judge 难以区分）
- best_model 回答过短（<100 字符，可能是空响应或错误）

**关闭方式**: `enable_pick_best=False`

---

### ③ Synthesizer 传入原始回答（信息保全）

**位置**: synthesizer.py + prompts.py

**机制**: Synthesizer 现在同时收到：
1. Judge 的结构化 JSON 分析（consensus/contradictions/insights）
2. **完整的原始 panel 回答**（labeled responses）

Synthesizer prompt 新增指示：
- "If one model's response is clearly superior for a section, use that section directly"
- "Preserve key details from the original responses (proofs, code, step-by-step reasoning)"

**适用场景**: 始终启用（无开关），当 ①② 都未命中时，synthesis 仍有完整信息可用。

---

## 场景推荐配置

### 场景 A: 逻辑推理/数学题为主的评测

```python
base = FusionConfig(
    ...,
    enable_consensus_shortcut=True,
    consensus_threshold=0.90,   # 严格，避免不同证明路径误判
    enable_pick_best=True,      # 一个模型正确证明时直接采用
)
```
预期效果: Pick-Best 命中率高（逻辑题通常有一个模型明显更强），API 调用减少 ~40%。

### 场景 B: 代码生成评测

```python
base = FusionConfig(
    ...,
    enable_consensus_shortcut=True,
    consensus_threshold=0.75,   # 宽松，代码结构相似即视为一致
    enable_pick_best=True,
)
```
预期效果: Consensus 命中率高（简单代码多模型一致），API 调用减少 ~30%。

### 场景 C: 开放式问答/创意写作

```python
base = FusionConfig(
    ...,
    enable_consensus_shortcut=False,  # 答案天然多样化，不短路
    enable_pick_best=False,           # 需要 synthesis 融合多模型创意
)
```
预期效果: 每次都走完整 pipeline，API 调用 = N+2，但融合质量最高。

### 场景 D: 追求速度的实时场景

```python
base = FusionConfig(
    ...,
    enable_consensus_shortcut=True,
    consensus_threshold=0.80,   # 降低阈值提高命中率
    enable_pick_best=True,
)
```
预期效果: 最大限度减少 API 调用，延迟降低 ~30-50%。

## 与更强 Judge 模型配合

当使用 Claude 3.5 / GPT-4o 作为 judge 时，Pick-Best 的准确率显著提升：

```python
base = FusionConfig(
    panel=[ModelSpec("glm-5.1"), ModelSpec("kimi-k2.6"), ModelSpec("deepseek-v4-flash")],
    judge=ModelSpec("claude-3.5-sonnet"),       # 强 judge
    caller=ModelSpec("claude-3.5-sonnet"),       # 强 synthesizer
    enable_pick_best=True,
)
```

强 judge 能更准确识别哪个模型回答最优，Pick-Best 命中率可达 60-80%，大幅减少 synthesis 调用。
