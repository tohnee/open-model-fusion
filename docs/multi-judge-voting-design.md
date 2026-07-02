# 多 Judge 投票方案设计 (v1.3 P1)

> 解决核心瓶颈: 单 judge 误判导致 pick-best 锁定错误答案 (math_2 场景)

---

## 一、问题分析

### 当前瓶颈

```
Panel (3 模型) → Judge (1 个) → best_model → pick-best 短路
                    ↑
              单点故障: judge 误判 = 短路锁定错误
```

**实测数据** (math_2 题): judge 选中了错误模型, pick-best 短路返回了错误答案。即使 glm-5.2 在 panel 中答对了, judge 未选中它。

### 根因

1. **单 judge 无交叉验证**: 一个 judge 的判断 = 最终判断, 无纠错机会
2. **Judge 和 panel 可能同质化**: 如果 judge 模型本身不擅长某类题, 它选 best 也会错
3. **Judge prompt 缺乏置信度**: judge 被迫选一个 best_model, 即使没有信心

---

## 二、方案设计

### 架构: 多 Judge 并行投票

```
Panel (3 模型) → ┌─ Judge A → Analysis{best_model: "MODEL A"}
                  ├─ Judge B → Analysis{best_model: "MODEL C"}
                  └─ Judge C → Analysis{best_model: "MODEL A"}
                                    │
                          投票聚合器 (多数投票)
                                    │
                    best_model = "MODEL A" (2/3 票)
                                    │
                         Pick-Best 短路 (高置信度)
```

### 投票策略

```python
@dataclass
class JudgeVote:
    """单个 judge 的投票结果。"""
    judge_slug: str
    best_model: str | None       # "MODEL A" | None
    best_reason: str | None
    confidence: float            # judge 自评置信度 (0.0-1.0)


@dataclass
class JudgeVoteResult:
    """多 judge 投票聚合结果。"""
    winner_model: str | None     # 多数票胜出的模型
    vote_count: int              # 胜出票数
    total_judges: int            # 总 judge 数
    agreement_ratio: float       # vote_count / total_judges
    votes: list[JudgeVote]       # 所有 judge 的投票详情

    @property
    def is_consensus(self) -> bool:
        """是否达成多数共识 (>= 2/3 一致)。"""
        return self.agreement_ratio >= 2/3

    @property
    def confidence(self) -> float:
        """聚合置信度: agreement_ratio * mean(judge confidences)。"""
        if not self.votes:
            return 0.0
        avg_judge_conf = sum(v.confidence for v in self.votes) / len(self.votes)
        return self.agreement_ratio * avg_judge_conf
```

### 聚合规则

| 场景 | 处理 |
|------|------|
| 3 judge 全选同一模型 | 高置信度, 直接 pick-best 短路 |
| 2/3 选同一模型 | 中置信度, 结合 verifier 决定是否短路 |
| 3 judge 各选不同 | 无共识, 跳过 pick-best, 走 synthesizer |
| 多数 judge 选 null | 无最佳, 走 synthesizer |

---

## 三、核心代码修改文件清单

### 新增文件

| 文件 | 职责 |
|------|------|
| `src/open_fusion/multi_judge.py` | 多 judge 并行调用 + 投票聚合器 |
| `tests/test_multi_judge.py` | 多 judge 投票逻辑的 TDD 测试 |

### 修改文件

| 文件 | 改动内容 | 改动量 |
|------|---------|--------|
| **`config.py`** | 新增 `judges: list[ModelSpec]` 字段 (向后兼容: 为空时退化为单 judge); 新增 `multi_judge_strategy: str` ("majority" \| "weighted") | ~15 行 |
| **`orchestrator.py`** | JUDGE 阶段: 检测 `config.judges` 是否非空 → 并行调用多 judge → 投票聚合 → 将投票结果传入 pick-best 置信度评估 | ~40 行 |
| **`judge.py`** | 抽取 `synthesize()` 为可复用的单 judge 调用; 新增 `synthesize_multi()` 并行调度 + 投票 | ~50 行 |
| **`schema.py`** | 新增 `JudgeVote` / `JudgeVoteResult` 数据类; `Analysis` 新增 `judge_votes` 字段 | ~25 行 |
| **`prompts.py`** | JUDGE_SYSTEM 新增 `confidence` 字段 (judge 自评 0.0-1.0); 新增 `JUDGE_CONFIDENCE_INSTRUCTION` | ~10 行 |
| **`verifier.py`** | (已实现) 验证器在多 judge 场景下作为额外 gate | 0 行 |
| **`eval/superiority.py`** | 传递 `judges` 配置到 fusion arm | ~5 行 |

### 配置示例

```python
# v1.3: 多 judge 配置
config = FusionConfig(
    panel=[
        ModelSpec("glm-5.2"),
        ModelSpec("kimi-k2.6"),
        ModelSpec("deepseek-v4-flash"),
    ],
    judge=ModelSpec("glm-5.2"),          # 主 judge (向后兼容)
    judges=[                              # v1.3: 多 judge 列表
        ModelSpec("glm-5.2"),
        ModelSpec("kimi-k2.6"),
        ModelSpec("deepseek-v4-pro"),
    ],
    multi_judge_strategy="majority",      # 或 "weighted"
    enable_pick_best=True,
    enable_verifier=True,
    pick_best_confidence_threshold=0.5,
)
```

### 向后兼容

```python
# 当 judges 为空时, 退化为现有单 judge 逻辑 (零改动)
if not config.judges:
    analysis = await judge_mod.synthesize(...)  # 现有逻辑
else:
    vote_result = await judge_mod.synthesize_multi(...)
    analysis = vote_result.to_analysis()        # 聚合后的 Analysis
```

---

## 四、实施计划 (TDD)

### Step 1: 数据结构 + 投票聚合器

```
1. 在 schema.py 定义 JudgeVote / JudgeVoteResult
2. 在 multi_judge.py 实现 aggregate_votes() 函数
3. TDD: 测试全一致 / 2-1 分裂 / 全不同 / null 票
```

### Step 2: 并行多 Judge 调用

```
1. 在 judge.py 抽取单 judge 调用为 _run_single_judge()
2. 实现 synthesize_multi(): asyncio.gather 并行调用 N 个 judge
3. TDD: mock 3 个 judge, 验证并行调用 + 聚合
```

### Step 3: Orchestrator 集成

```
1. orchestrator.py JUDGE 阶段: 检测 config.judges
2. 多 judge → 投票 → Analysis
3. pick-best 置信度评估加入 judge agreement_ratio 信号
4. TDD: 集成测试 — 3 judge 全选同一模型 → 高置信度短路
```

### Step 4: Judge Prompt 增强

```
1. JUDGE_SYSTEM 新增 confidence 字段
2. Judge 自评置信度作为投票权重
3. TDD: 验证 confidence 解析
```

---

## 五、预期收益

| 指标 | v1.2 (单 judge) | v1.3 (多 judge) | 提升 |
|------|----------------|-----------------|------|
| Judge 误判率 | ~30% (3 题中 1 题选错) | ~10% (3 judge 中 2 个对 = 多数对) | -67% |
| Pick-best 短路准确率 | 50% (2 触发, 1 对 1 错) | ~80% (2/3 共识时短路) | +30% |
| API 调用 (judge 阶段) | 1 | 3 | +2 (但并行, 延迟不增) |
| 关键路径延迟 | panel + 1 judge | panel + 1 judge (并行) | 不变 |

### 关键洞察

多 judge 的核心价值不是"更多判断", 而是**降低单点误判概率**。3 个独立 judge 同时误判同一道题的概率 ≈ 0.3³ = 2.7%, 远低于单 judge 的 30%。

### 成本分析

- **额外 API 调用**: +2 次 judge 调用 (但并行执行, 不增加关键路径延迟)
- **额外 token**: judge 阶段 token × 3 (但 judge 的 prompt 比 panel 短)
- **ROI**: 每次避免 1 个错误答案 ≈ 节省 1 次用户重新提问 + 1 次 fusion 重跑

---

## 六、与 Verifier 的协同

```
多 Judge 投票 → agreement_ratio >= 0.67?
  → 否: 跳过 pick-best, 走 synthesizer (无共识)
  → 是: Verifier 检查答案格式/数值
           → FAIL: 回退到 synthesizer (验证器否决)
           → PASS: pick-best 短路 (双重保障)
```

多 judge 解决"选谁"的问题, verifier 解决"选的答案是否合理"的问题。两者互补:
- 多 judge 可能全选同一模型, 但该模型答案格式不对 → verifier 拦截
- Verifier 通过, 但 judge 之间分歧大 → 多 judge 的低 agreement_ratio 拦截
