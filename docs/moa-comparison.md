# Open-Fusion vs Hermes MoA 深度对比分析

## 一、架构总览对比

### Hermes MoA 架构

```
用户问题
    │
    ▼
┌──────────────────────────────────────┐
│ Reference Models (并行, 无工具, 无系统提示) │
│  ┌────────┐ ┌────────┐ ┌────────┐   │
│  │Model A │ │Model B │ │Model C │   │
│  └───┬────┘ └───┬────┘ └───┬────┘   │
│      └──────────┼──────────┘        │
│                 ▼                    │
│    (原始回答直接拼接到末尾)              │
└─────────────────┬────────────────────┘
                  ▼
┌──────────────────────────────────────┐
│ Aggregator (完整工具, 完整系统提示)      │
│  接收: 原始问题 + 参考模型回答            │
│  输出: 最终答案 (可调用工具)              │
└──────────────────────────────────────┘
```

### Open-Fusion 架构

```
用户问题
    │
    ▼
┌──────────────────────────────────────┐
│ Panel Models (并行, 可选工具)           │
│  ┌────────┐ ┌────────┐ ┌────────┐   │
│  │Model A │ │Model B │ │Model C │   │
│  └───┬────┘ └───┬────┘ └───┬────┘   │
└─────────────────┬────────────────────┘
                  ▼
         ┌────────────────┐
         │ Consensus 检查  │ → (多数一致? 直接返回)
         └───────┬────────┘
                 ▼
┌──────────────────────────────────────┐
│ Judge (结构化 JSON 分析, 无工具)        │
│  输出: consensus, contradictions,     │
│        unique_insights, blind_spots, │
│        best_model, best_reason       │
└─────────────────┬────────────────────┘
                  ▼
         ┌────────────────┐
         │ Pick-Best 检查  │ → (有最优? 直接返回)
         └───────┬────────┘
                 ▼
┌──────────────────────────────────────┐
│ Synthesizer (融合产出, 无工具)          │
│  接收: JSON 分析 + 原始 panel 回答      │
│  输出: 最终答案                        │
└──────────────────────────────────────┘
```

## 二、核心差异逐项对比

### 1. 中间层设计

| 维度 | Hermes MoA | Open-Fusion |
|------|-----------|-------------|
| **中间层** | 无 (参考回答直接给 aggregator) | 有 (Judge 产出结构化 JSON) |
| **信息传递** | 原始文本拼接 | JSON 摘要 + 原始文本 |
| **可审计性** | 低 (黑盒拼接) | 高 (结构化分析可追溯) |
| **信息损失** | 无 (完整原文传递) | Judge 摘要有损失 (已通过传入原文缓解) |
| **API 调用** | N + 1 (N 参考 + 1 聚合) | N + 1 或 N + 2 (加 Judge) |

**分析**: MoA 的最大优势是**信息无损**——参考模型的完整回答直接给 aggregator。Open-Fusion 的 Judge 层有信息压缩，但我们已修复为同时传入原始回答。

**MoA 可吸收点**: MoA 的"无中间层"设计在简单场景下更高效。可以考虑当 panel ≤ 2 时跳过 Judge，直接走 aggregator 模式。

### 2. 工具调用策略

| 维度 | Hermes MoA | Open-Fusion |
|------|-----------|-------------|
| **Panel/Reference 工具** | 禁用 (省 token, 避免严格 provider 拒绝) | 可选 (tools_enabled) |
| **Judge 工具** | 无 Judge 层 | 禁用 |
| **Aggregator/Synth 工具** | 完整工具 schema | 禁用 (evidence is frozen) |
| **多轮工具循环** | 支持 (每轮重新跑 MoA) | 不支持 (单轮 fusion) |

**分析**: 这是最大的架构差异。MoA 的 aggregator 可以调用工具并多轮迭代，而 Fusion 的 synthesizer 是"证据冻结"的单次输出。

**MoA 可吸收点**: 
- Reference 模型禁用工具的设计值得吸收——能省大量 token 且避免 provider 兼容问题
- 可以考虑在 Fusion 中增加"aggregator mode"，让 synthesizer 保留工具能力

### 3. Prompt 缓存优化

| 维度 | Hermes MoA | Open-Fusion |
|------|-----------|-------------|
| **缓存策略** | 精心设计: 参考输出追加在 user turn 末尾，不破坏前缀缓存 | 无显式缓存策略 |
| **参考模型输入** | 裁剪版 (去掉系统提示和工具转录，稳定可缓存) | 完整问题 |
| **缓存命中率** | 高 (前缀字节稳定) | 低 (每条 panel 消息独立) |

**分析**: MoA 在缓存优化上做得非常精细。参考模型收到的是"对话文本的确定性裁剪"，前缀跨迭代可缓存；aggregator 的参考输出注入在末尾，不破坏前缀。

**MOA 可吸收点**: 
- Panel 模型不应收到完整系统提示，只需问题本身——这既省 token 又提高缓存命中
- Synthesizer 的输入应保持前缀稳定，只在末尾追加新内容

### 4. 错误处理

| 维度 | Hermes MoA | Open-Fusion |
|------|-----------|-------------|
| **单模型失败** | 错误注入上下文，继续执行 | 错误记录在 PanelResponse，继续执行 |
| **全部失败** | 未明确文档 | 返回 FusionStatus.ERROR |
| **Judge 失败** | 无 Judge | 降级为 fallback synthesis (原始回答直接合成) |
| **凭证过期** | 不中断，注入错误信息 | 同样处理 |

**分析**: 两者在错误韧性上基本对等。Fusion 多了 Judge 失败的降级路径。

### 5. 递归防护

| 维度 | Hermes MoA | Open-Fusion |
|------|-----------|-------------|
| **机制** | 显式禁止 aggregator 指向另一个 MoA preset | depth guard (MAX_FUSION_DEPTH=1) |
| **设计** | 配置层面阻断 | 运行时检测 |

**分析**: MoA 的配置层面阻断更简洁，但 Fusion 的 depth guard 更通用（支持嵌套调用的有限递归）。

### 6. 配置与预设

| 维度 | Hermes MoA | Open-Fusion |
|------|-----------|-------------|
| **预设系统** | 命名预设 (default, chinese, coding...) | 单一配置 |
| **切换方式** | /model preset --provider moa | 修改代码/环境变量 |
| **多 provider** | 原生支持 (openrouter, openai, anthropic...) | 统一网关 (单 base_url) |
| **温度控制** | reference_temperature + aggregator_temperature | 三阶段独立温度 |

**MOA 可吸收点**: 
- **命名预设系统**是最值得吸收的特性——让用户预定义多套配置，按场景切换
- **多 provider 支持**也很重要——不同模型走不同网关，需要 per-model client

### 7. 成本与延迟

| 维度 | Hermes MoA | Open-Fusion |
|------|-----------|-------------|
| **API 调用** | N + 1 (无 Judge) | N + 2 (有 Judge) 或 N + 1 (pick-best) 或 N (consensus) |
| **实测成本** | 单次 Opus 的 ~80x (3 参考 + Opus 聚合) | 未精确测量 |
| **延迟** | ~4-6x 单模型 | ~3-5x 单模型 |
| **Token 消耗** | 参考模型 ~41K input (含上下文注入) | Panel 各自独立，无上下文累积 |

**分析**: Fusion 的 short-circuit 机制在成本上有优势——consensus 命中时只需 N 次调用。但 MoA 无 Judge 层，常规情况下少一次调用。

## 三、MOA 可吸收的设计方案

### 方案 A: 无 Judge 直聚合模式 (Aggregator Mode)

**来源**: MoA 的核心设计——参考回答直接给 aggregator，无需中间 Judge

**实现思路**: 在 FusionConfig 中增加 `mode` 字段:

```python
class FusionMode(Enum):
    FULL = "full"            # 完整 pipeline: panel → judge → synth (当前默认)
    AGGREGATOR = "aggregator"  # MoA 模式: panel → synth (跳过 judge)
```

**适用场景**: panel ≤ 2 个模型时，或简单任务无需结构化分析时

**预期收益**: 减少 1 次 API 调用，降低延迟 ~30%

### 方案 B: Panel 裁剪输入 (Trimmed Panel Input)

**来源**: MoA 的参考模型只接收对话文本，不含系统提示和工具转录

**实现思路**: Panel 模型的 prompt 做裁剪——只传问题本身，不传完整上下文

```python
# 当前: panel 收到完整 question
# 改进: 对需要节省 token 的场景，panel 收到裁剪后的问题
trimmed_prompt = question if len(question) < 2000 else question[:2000] + "..."
```

**适用场景**: 长 prompt 场景，参考模型不需要完整上下文

**预期收益**: Panel token 消耗减少 30-50%

### 方案 C: 命名预设系统 (Named Presets)

**来源**: MoA 的 default/chinese/coding 预设系统

**实现思路**: 在 config 中支持命名预设，通过 CLI 参数选择:

```python
PRESETS = {
    "default": FusionConfig(
        panel=[ModelSpec("glm-5.1"), ModelSpec("kimi-k2.6"), ModelSpec("deepseek-v4-flash")],
        judge=ModelSpec("doubao-seed-2.0-pro"),
        ...
    ),
    "logic": FusionConfig(
        panel=[ModelSpec("kimi-k2.6"), ModelSpec("doubao-seed-2.0-pro"), ModelSpec("glm-5.2")],
        judge=ModelSpec("glm-5.2"),
        consensus_threshold=0.90,
        ...
    ),
    "code": FusionConfig(
        panel=[ModelSpec("deepseek-v4-flash"), ModelSpec("kimi-k2.6"), ModelSpec("glm-5.1")],
        judge=ModelSpec("doubao-seed-2.0-pro"),
        consensus_threshold=0.75,
        ...
    ),
}
```

**预期收益**: 用户无需改代码即可切换配置，按场景选择最优组合

### 方案 D: 多 Provider 支持 (Multi-Gateway)

**来源**: MoA 原生支持 openrouter/openai/anthropic 等多 provider

**实现思路**: 为每个 ModelSpec 关联独立的 base_url 和 api_key:

```python
@dataclass
class ModelSpec:
    slug: str
    vendor: str = ""
    base_url: str | None = None    # 新增: per-model 网关
    api_key: str | None = None     # 新增: per-model 密钥
```

**适用场景**: panel 模型分布在不同平台（如 GPT 走 OpenAI, Claude 走 Anthropic）

**预期收益**: 真正的异构多模型融合，不受单一网关限制

### 方案 E: Aggregator 工具保留 (Tool-Capable Synthesizer)

**来源**: MoA 的 aggregator 保留完整工具 schema

**实现思路**: 在 FusionConfig 中增加 `synth_tools_enabled` 标志:

```python
# 当 synth_tools_enabled=True 时, synthesizer 可以调用工具
# 适用于需要多轮迭代的 agent 场景
```

**适用场景**: 需要工具调用的复杂任务（代码执行、文件操作等）

**预期收益**: Fusion 可嵌入 agent loop，支持多轮工具调用

### 方案 F: 参考输出末尾注入 (Tail Injection)

**来源**: MoA 将参考输出追加在 user turn 末尾，不破坏前缀缓存

**实现思路**: Synthesizer 的消息构造保持前缀稳定:

```python
# 当前: SYNTHESIS_USER 重新构造完整消息
# 改进: 保持 user question 前缀不变, 在末尾追加 panel 回答
messages = [
    {"role": "system", "content": SYNTHESIS_SYSTEM},
    {"role": "user", "content": question + "\n\n" + panel_responses},
]
```

**预期收益**: 提高 synthesizer 的 prompt 缓存命中率

## 四、优先级建议

| 方案 | 优先级 | 难度 | 预期收益 |
|------|--------|------|---------|
| **A: Aggregator Mode** | P0 | 低 | 减少 1 次 API 调用, 降延迟 30% |
| **C: 命名预设** | P0 | 低 | 用户体验大幅提升 |
| **B: Panel 裁剪** | P1 | 中 | Token 消耗减少 30-50% |
| **D: 多 Provider** | P1 | 高 | 真正异构融合 |
| **F: 末尾注入** | P2 | 低 | 缓存命中提升 |
| **E: 工具保留** | P2 | 高 | 支持 agent loop |

## 五、结论

Open-Fusion 和 Hermes MoA 代表了多模型融合的两种哲学:

- **MoA**: 简单直接, 无中间层, 信息无损, 但缺乏结构化分析
- **Fusion**: 结构化分析, 可审计, 有 short-circuit 优化, 但多一层开销

两者各有优势。最佳策略是**让用户选择模式**——简单任务走 aggregator mode (类 MoA)，复杂任务走 full pipeline (当前 Fusion)。这正是方案 A (Aggregator Mode) 的核心价值。
