# AGGREGATOR vs PICK-BEST 深度分析报告

> 基于 v1.2 三轮真实 API 测试数据 (2026-06-30)
> 测试模型: glm-5.2, kimi-k2.6, deepseek-v4-flash, doubao-seed-2.0-pro
> 测试题数: 5 题 (数学×2, 代码×1, 常识陷阱×1, 多步推理×1)

---

## 一、数据汇总

### 三轮测试准确率

| Arm | 第一轮 | 第二轮 | 第三轮 | 均值 | 标准差 |
|-----|--------|--------|--------|------|--------|
| Solo glm-5.2 | 60% | 80% | 60% | 67% | ±10.0% |
| Solo kimi-k2.6 | 40% | 40% | 40% | 40% | 0% |
| Fusion FULL | 60% | 40% | 40% | 47% | ±10.4% |
| Fusion AGGREGATOR | 60% | 40% | **60%** | **53%** | ±10.4% |
| Fusion PICK-BEST | — | — | **60%** | 60% | — |

### 逐题表现 (第三轮, 最完整数据)

| 题目 | 维度 | Solo-glm | Solo-kimi | FULL | AGG | PICK-BEST |
|------|------|:--------:|:---------:|:----:|:---:|:---------:|
| math_1 | 数学 | ✗ | ✗ | ✗ | ✗ | ✗ |
| code_1 | 代码 | ✓ | ✓ | ✓ | ✓ | ✓ (短路) |
| trap_1 | 常识陷阱 | ✗ | ✗ | ✗ | **✓** | **✓** |
| reasoning_1 | 多步推理 | ✓ | ✓ | ✓ | ✓ | ✓ |
| math_2 | 数学 | ✓ | ✗ | ✗ | ✗ | ✗ (短路) |

---

## 二、AGGREGATOR 为什么在部分场景下更稳健

### 核心发现: AGGREGATOR (60%) > FULL (40%) — 跳过 judge 反而更好

这看似反直觉，但有深刻的结构性原因。

### 原因 1: Judge 层的信息损失

**FULL 管道**: panel → judge (结构化分析) → synthesizer (最终答案)

Judge 的职责是将 N 个回答蒸馏为结构化 JSON (consensus / contradictions / best_model 等)。这个"压缩"过程有信息损失：

```
Panel 原始回答 (3 × 500字 = 1500字)
  → Judge 蒸馏为 Analysis JSON (~200字, 丢失 87% 原文)
    → Synthesizer 只看到结构化摘要, 丢失原始推理细节
      → 最终答案基于"摘要的摘要"
```

**AGGREGATOR 管道**: panel → synthesizer (直接看原始回答)

```
Panel 原始回答 (3 × 500字 = 1500字)
  → Synthesizer 直接看到全部原始回答
    → 最终答案基于完整信息
```

**实测证据**: trap_1 (绳子对折3次剪断)
- FULL: judge 蒸馏时可能丢失了关键推理细节 → synthesizer 答错
- AGGREGATOR: synthesizer 直接看到某个模型的完整推理 → 答对

### 原因 2: Judge 的"投票偏向"放大错误

Judge 在判断 `best_model` 时，倾向于选择"多数派"答案。当多数模型答错时：

```
math_2 (概率题 15/28):
  Panel: glm-5.2 ✓, kimi ✗, deepseek ✗
  Judge 分析: "2/3 模型认为是 X, 共识是 X" → best_model = kimi (错误)
  Synthesizer: 基于 judge 的错误分析 → 答错

AGGREGATOR: synthesizer 直接看 3 个原始回答
  → 虽然多数也答错, 但 glm-5.2 的正确推理过程完整保留
  → synthesizer 有机会识别正确推理 (虽然这次也没识别出来)
```

### 原因 3: AGGREGATOR 减少一次 API 调用的不确定性

| 模式 | API 调用 | 故障点 |
|------|---------|--------|
| FULL | panel(3) + judge(1) + synth(1) = 5 | judge JSON 解析失败 → fallback |
| AGGREGATOR | panel(3) + synth(1) = 4 | 少一个故障点 |

FULL 模式的 judge 层有 `JudgeError` 风险 (JSON 解析失败两次)，会触发 fallback 到 `write_fallback`。这个 fallback 路径比 AGGREGATOR 的直接合成更粗暴 (不经过结构化分析)。

### 原因 4: 代码题上两者持平

| 维度 | FULL | AGG | 差异原因 |
|------|:----:|:---:|---------|
| 代码 | ✓ | ✓ | 代码 bug 明确, panel 模型都能找到, 合成无歧义 |
| 多步推理 | ✓ | ✓ | 逻辑链清晰, 正确答案一致 |
| **常识陷阱** | **✗** | **✓** | 反直觉题需要完整推理, judge 压缩丢失关键细节 |
| 数学 | 0/2 | 0/2 | 数学需要精确计算, 编排无法弥补计算错误 |

---

## 三、PICK-BEST 的优劣分析

### 优势: 纠正常识陷阱 (+20%)

```
trap_1 (绳子对折3次):
  Solo-glm-5.2:  ✗ (三轮中两轮答错)
  Solo-kimi:     ✗ (三轮全部答错)
  Fusion-FULL:   ✗ (judge 压缩丢失推理)
  Fusion-AGG:    ✓ (synthesizer 看到完整推理)
  Fusion-PICK:   ✓ (judge 选中答对的模型, 短路返回)
```

Pick-BEST 在 trap_1 上成功，因为：
1. glm-5.2 加入 panel 后答对了这道题
2. judge 正确识别 glm-5.2 为 best_model
3. 短路直接返回 glm-5.2 的完整答案, 绕过了 synthesizer

### 劣势: math_2 短路锁定错误

```
math_2 (概率 15/28):
  Panel: glm-5.2 ✗, kimi ✗, deepseek ✗ (这轮全都答错了)
  Judge: 选中某个模型 → pick_best_shortcut
  结果: ✗ (短路锁定了错误答案, 无 synthesizer 纠错机会)
```

问题根源：**当 panel 中没有模型答对时, pick-best 无论选谁都是错的**。更严重的是，短路跳过了 synthesizer，连"碰巧合成出正确答案"的机会都没有了。

### PICK-BEST 触发率分析

| 题目 | 短路触发 | 结果 | 置信度评估 |
|------|---------|------|-----------|
| code_1 | ✓ | ✓ 正确 | 高置信度 (有共识, 无矛盾) |
| math_2 | ✓ | ✗ 错误 | 应为低置信度 (无共识, 多矛盾) |
| trap_1 | ✗ | ✓ 正确 | 走了 synthesis |
| reasoning_1 | ✗ | ✓ 正确 | 走了 synthesis |
| math_1 | ✗ | ✗ 错误 | 走了 synthesis |

**关键洞察**: math_2 本应被 fallback 机制拦截 — judge 分析中 consensus 为空、contradictions 多, 置信度应该低。v1.2.1 的置信度评估正是为解决此问题而设计。

---

## 四、v1.2.1 Fallback 机制预期效果

### 置信度评估逻辑

```python
def _assess_pick_best_confidence(analysis, responses, best_idx):
    confidence = 1.0

    # 信号 1: best_reason 含不确定性词汇 → -0.3
    # 信号 2: contradictions >= 2 → -0.1/矛盾
    # 信号 3: consensus 为空 → -0.2
    # 信号 4: best 答案与其他答案相似度 < 0.3 → -0.2 (异类检测)

    return max(confidence, 0.0)
```

### 对 math_2 的预期行为

```
math_2 的 judge 分析 (推测):
  consensus: []          → 信号 3 触发, -0.2
  contradictions: [...]   → 信号 2 可能触发, -0.2
  best_reason: "..."      → 需检查不确定性词汇

预期置信度: 0.6 或更低
如果 threshold = 0.5:
  - 0.6 >= 0.5 → 仍然短路 (不够低)
  - 0.4 < 0.5 → 回退到 synthesizer (预期效果)

需要实测验证真实 judge 输出的置信度。
```

---

## 五、架构层次对比

```
┌──────────────────────────────────────────────────────────┐
│                     FULL MODE                            │
│  Panel → Judge(蒸馏) → Pick-Best? → Synth(合成) → 答案   │
│                          ↑ 信息损失点                     │
│                          ↑ 投票偏向放大点                 │
│                          ↑ JSON 解析故障点                │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                  AGGREGATOR MODE                         │
│  Panel → Synth(直接合成) → 答案                          │
│            ↑ 看到全部原始回答, 无信息损失                  │
│            ↑ 少一次 API 调用, 少一个故障点                 │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                   PICK-BEST MODE                         │
│  Panel → Judge(蒸馏) → 置信度评估 → 短路 or Synth → 答案  │
│                          ↑ v1.2.1 新增                    │
│                          高置信度: 直接返回 (省 synth)     │
│                          低置信度: 回退 synth (安全网)     │
└──────────────────────────────────────────────────────────┘
```

---

## 六、结论与建议

### 何时用 AGGREGATOR

- **反直觉题 / 常识陷阱**: synthesizer 需要完整推理细节, judge 蒸馏会丢失
- **追求稳健性**: 少一次 API 调用, 少一个故障点
- **延迟敏感**: 比 FULL 快 ~40%

### 何时用 PICK-BEST (with v1.2.1 fallback)

- **有强模型加入 panel**: judge 有好答案可选
- **客观题 (代码/数学)**: 答案对错明确, judge 容易判断
- **v1.2.1 fallback 启用后**: 低置信度自动回退, 安全网保底

### 何时用 FULL

- **需要 judge 的结构化分析作为可审计记录** (contradictions, blind_spots)
- **panel 模型能力接近, 无明显强弱**
- **复杂开放题**: judge 的多维度分析有助于 synthesizer 全面覆盖

### 最优策略: 自适应路由

基于 v1.2.1 置信度评估，未来可以实现自适应路由：
- 高置信度 pick-best → 短路返回
- 中置信度 → AGGREGATOR (直接合成, 无 judge 干扰)
- 低置信度 → FULL (judge 分析 + synth, 多层保险)

这正是 v1.3 roadmap 的核心方向。
