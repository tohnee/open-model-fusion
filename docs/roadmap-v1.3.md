# v1.3 Roadmap — Judge 判断力瓶颈与多模型干扰

> 基线版本: v1.2.1 (pick-best + confidence fallback)
> 目标版本: v1.3.0
> 核心主题: 从"依赖 judge 判断"到"多维度验证 + 自适应路由"

---

## 一、问题诊断: judge 为何会误判?

### 1.1 实测数据 (v1.2 测试)

| 场景 | judge 行为 | 结果 | 根因 |
|------|-----------|------|------|
| math_2 (概率题) | 选中错误模型 | ✗ 短路锁定错误 | judge 本身缺乏数学验证能力, 被"看起来合理"的错误推理欺骗 |
| code_1 (代码题) | 选中正确模型 | ✓ 短路成功 | 代码正确性可通过结构判断, judge 擅长此类验证 |
| trap_1 (常识陷阱) | 未选出 best_model | ✓ synth 答对 | 无短路时 synthesizer 反而能综合多视角纠错 |

### 1.2 judge 判断力的 3 个结构性瓶颈

**瓶颈 1: judge 无法验证事实正确性**

judge 只能看到 panel 的回答文本, 无法执行代码、验算数学、查证事实。当所有 panel 模型都给出"看起来合理但实际错误"的推理时, judge 无能力识别。

```
math_2 案例:
  Panel A (glm-5.2): "15/28" (正确)
  Panel B (kimi): "5/14" (错误, 但推理看似合理)
  Panel C (deepseek): "5/14" (错误, 与 B 一致)
  
  Judge 看到: B 和 C 答案一致, 推理详细 → 选 B 为 best_model
  实际: A 才是对的, 但被"多数一致"误导
```

**瓶颈 2: judge 被"多数一致性"干扰**

当 2/3 的 panel 模型给出相同 (错误) 答案时, judge 倾向于选择多数派 — 即使第三方的答案才是正确的。这是"投票效应"在 judge 层的延续。

**瓶颈 3: judge 的 best_reason 缺乏可验证性**

judge 输出的 `best_reason` 是自然语言解释, 没有"我验算了, 答案确实等于 15/28"这样的执行验证。置信度评估只能靠文本启发式, 无法做事实检查。

### 1.3 多模型干扰的 2 种模式

**模式 A: 错误传染 (Error Contagion)**

当多个中游模型犯同样的错误时, 这个错误会成为"共识", 污染 judge 和 synthesizer 的判断。

```
传染路径: kimi 答错 → deepseek 答错 (相同错误) → judge 看到"共识" → 选错误答案
                                                    → synthesizer 看到"共识" → 合成错误答案
```

**模式 B: 正确答案边缘化 (Correct Answer Marginalization)**

当只有一个模型答对时, 它的回答会显得"与众不同", 被 judge 和 synthesizer 视为"异类"而忽略。

```
边缘化路径: glm-5.2 答对 (15/28) → 其他 2 个答错 (5/14)
  → judge: "15/28 与多数不一致, 可能是计算错误" → 选 5/14
  → synthesizer: "大多数模型同意 5/14" → 合成 5/14
```

---

## 二、v1.3 五个方案 (按优先级)

### P0: 验证型 Judge (Verification-Based Judging)

**目标**: judge 不再只"看"答案, 而是"验算"答案。

**设计**:

```python
# 新增 Phase: VERIFY
class Phase(str, Enum):
    PANEL = "panel"
    JUDGE = "judge"
    VERIFY = "verify"       # v1.3: 对 panel 答案做独立验证
    SYNTHESIS = "synthesis"

# VERIFY 阶段: 对数学/代码题, judge 用工具 (code interpreter) 独立验算
# 对事实题, judge 用 web search 交叉验证
# 输出: 每个 panel 答案的 verified: true/false
```

```python
@dataclass
class Analysis:
    # ... 现有字段 ...
    verified_answers: list[dict[str, Any]] = field(default_factory=list)
    # [{"model": "MODEL A", "verified": True, "method": "code_execution", "detail": "executed, result=15/28"}]
```

**管道变化**:

```
v1.2: Panel → Judge (选 best) → Pick-Best / Synth
v1.3: Panel → Judge (分析) → VERIFY (验算) → Pick-Best (带验证) / Synth
```

**预估收益**: math_2 类题目中, VERIFY 阶段执行 `15/28 == 5*3/(8*7)` 返回 True, 直接标记 MODEL A 为已验证正确, pick-best 可高置信度短路。

**改动文件**: `schema.py`, `orchestrator.py`, `judge.py`, 新增 `verify.py`, `tools.py` (增加 code_interpreter 工具)
**测试**: 数学验证, 代码执行验证, 事实查证, 验证失败回退

---

### P0: 加权投票 (Weighted Voting)

**目标**: 不再 1 模型 1 票, 根据模型历史表现和题型适配度加权。

**设计**:

```python
@dataclass
class ModelSpec:
    slug: str
    # v1.3: 模型权重 (基于历史准确率或手动配置)
    weight: float = 1.0
    # v1.3: 模型擅长领域 (用于题型匹配)
    strengths: list[str] = field(default_factory=list)  # ["math", "code", "logic"]
```

```python
# judge 输出新增字段
@dataclass
class Analysis:
    # ... 现有字段 ...
    answer_clusters: list[dict[str, Any]] = field(default_factory=list)
    # [{"answer_hash": "15/28", "models": ["MODEL A"], "weighted_score": 1.5},
    #  {"answer_hash": "5/14", "models": ["MODEL B", "MODEL C"], "weighted_score": 2.0}]
```

**加权策略**:
- 数学题: glm-5.2 weight=1.5, kimi weight=0.8, deepseek weight=0.7
- 代码题: 所有模型 weight=1.0 (代码验证靠 VERIFY 阶段)
- 当加权后少数派的分数 > 多数派时, pick-best 选少数派

**预估收益**: math_2 中, glm-5.2 (weight=1.5) 的"15/28"得分 1.5, 而两个错误模型 (weight=0.7+0.8=1.5) 平分, 单个错误答案的得分只有 0.7-0.8, 低于 glm-5.2。

**改动文件**: `config.py`, `schema.py`, `orchestrator.py`, `judge.py`
**测试**: 加权计算, 题型匹配, 权重边界

---

### P1: 答案聚类与异常检测 (Answer Clustering)

**目标**: 自动识别"少数派正确答案"模式, 防止正确答案被边缘化。

**设计**:

```python
def _cluster_answers(responses: list[PanelResponse]) -> list[AnswerCluster]:
    """将 panel 回答按语义相似度聚类, 识别"异类"答案。
    
    返回聚类结果, 每个簇包含:
    - 答案摘要 (extracted answer)
    - 成员模型列表
    - 簇大小
    - 是否为"少数派异类" (size=1 且与其他簇差异大)
    """
    # 用 _similarity() 做层次聚类
    # 提取答案核心 (如数字、分数、代码关键行)
    ...
```

**在 pick-best 中的使用**:

```python
# v1.3: 如果发现少数派异类, 且该异类来自高权重模型, 优先考虑
clusters = _cluster_answers(responses)
if len(clusters) == 2 and clusters[0].size == 1:
    outlier = clusters[0]
    majority = clusters[1]
    # 检查异类模型是否为高权重
    outlier_model = responses[outlier.members[0]]
    if model_weight(outlier_model) >= 1.5:
        # 高权重模型的"异类"答案可能是唯一正确的
        # 触发 VERIFY 阶段验算
        ...
```

**预估收益**: math_2 中, glm-5.2 的"15/28"会被识别为异类簇 (size=1), 触发验算确认正确性。

**改动文件**: 新增 `clustering.py`, `orchestrator.py`
**测试**: 聚类准确性, 异类检测, 边界情况

---

### P1: 自适应路由 (Adaptive Routing)

**目标**: 根据题型自动选择最佳管道模式, 不再"一刀切"。

**设计**:

```python
@dataclass
class AdaptiveRouter:
    """根据问题特征自动选择管道模式。"""
    
    def detect_domain(self, question: str) -> str:
        """轻量级分类器: 检测问题领域 (math/code/logic/factual/open)。"""
        # 基于关键词匹配 + 简单启发式
        if re.search(r'\d+.*[+\-*/].*\d+|计算|概率|方程|求解', question):
            return "math"
        if re.search(r'```|def |function |bug|代码|python|java', question):
            return "code"
        ...
    
    def select_mode(self, domain: str, panel: list[ModelSpec]) -> FusionMode:
        """根据领域选择模式。"""
        if domain in ("math", "code"):
            # 精确计算题: 用 FULL + VERIFY + pick-best
            return FusionMode.FULL
        elif domain == "open":
            # 开放题: 用 AGGREGATOR (synth 直接综合, 无需 judge)
            return FusionMode.AGGREGATOR
        else:
            # 逻辑/事实题: 用 FULL + pick-best
            return FusionMode.FULL
```

**路由策略**:

| 题型 | 模式 | 理由 |
|------|------|------|
| 数学/计算 | FULL + VERIFY | 需要验算, pick-best 选已验证答案 |
| 代码 | FULL + VERIFY | 需要执行验证, pick-best 选已验证答案 |
| 常识/陷阱 | AGGREGATOR | synthesizer 综合多视角比 judge 判断更有效 |
| 开放问答 | AGGREGATOR | 无客观答案, synth 综合最优 |
| 逻辑推理 | FULL + pick-best | judge 可评估推理链完整性 |

**预估收益**: trap_1 用 AGGREGATOR (已验证有效), math_2 用 FULL+VERIFY (验算确认), 不再"一刀切"。

**改动文件**: 新增 `router.py`, `orchestrator.py`, `config.py`
**测试**: 题型检测, 模式选择, 混合题型

---

### P2: 历史性能追踪 (Performance Tracking)

**目标**: 记录每个模型在不同领域的准确率, 动态调整权重。

**设计**:

```python
@dataclass
class ModelPerformance:
    """追踪模型在特定领域的历史准确率。"""
    model_slug: str
    domain: str
    correct: int = 0
    total: int = 0
    
    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)


class PerformanceRegistry:
    """全局性能注册表, 跨 fusion 调用共享。"""
    _records: dict[tuple[str, str], ModelPerformance] = {}
    
    @classmethod
    def record(cls, model: str, domain: str, correct: bool):
        key = (model, domain)
        rec = cls._records.setdefault(key, ModelPerformance(model, domain))
        if correct:
            rec.correct += 1
        rec.total += 1
    
    @classmethod
    def get_weight(cls, model: str, domain: str) -> float:
        """根据历史准确率计算权重: accuracy > 0.7 → weight=1.5, < 0.4 → weight=0.5。"""
        rec = cls._records.get((model, domain))
        if not rec or rec.total < 3:
            return 1.0  # 数据不足, 默认权重
        acc = rec.accuracy
        if acc > 0.7:
            return 1.5
        elif acc < 0.4:
            return 0.5
        return 1.0
```

**预估收益**: 长期运行后, glm-5.2 在数学题的权重自动升到 1.5, kimi 在数学题降到 0.5, pick-best 更倾向选 glm-5.2 的答案。

**改动文件**: 新增 `performance.py`, `orchestrator.py`, `config.py`
**测试**: 准确率追踪, 权重计算, 冷启动

---

## 三、实施计划

### 阶段 1: 验证能力 (P0 VERIFY + 加权投票)

```
Week 1-3:
  - 新增 verify.py: code_interpreter 工具集成
  - Phase.VERIFY 在 judge 后、pick-best 前执行
  - ModelSpec.weight 字段 + judge 输出 answer_clusters
  - TDD: 验证测试 (15+ cases) + 加权测试 (10+ cases)

验收标准:
  - math_2: VERIFY 阶段执行计算, 标记 "15/28" 为 verified=True
  - pick-best 优先选 verified=True 的答案
  - 加权投票: 高权重模型的"异类"答案不被忽略
```

### 阶段 2: 智能路由 (P1 聚类 + 自适应)

```
Week 4-5:
  - 新增 clustering.py: 答案聚类 + 异类检测
  - 新增 router.py: 题型检测 + 模式选择
  - TDD: 聚类测试 (10+ cases) + 路由测试 (8+ cases)

验收标准:
  - 数学题自动路由到 FULL+VERIFY
  - 常识题自动路由到 AGGREGATOR
  - 少数派高权重答案触发验算
```

### 阶段 3: 自学习 (P2 性能追踪)

```
Week 6-7:
  - 新增 performance.py: 历史准确率追踪
  - 权重动态调整: 准确率 > 0.7 → 1.5x, < 0.4 → 0.5x
  - 集成测试: 多轮 fusion 后权重收敛
  - 基准测试: 对比 v1.2 和 v1.3 的准确率

验收标准:
  - 5 轮测试后, glm-5.2 数学权重升至 1.5
  - 整体准确率 vs v1.2 提升 >= 15%
```

---

## 四、预期效果对比

| 场景 | v1.2 行为 | v1.3 预期 | 改进来源 |
|------|----------|----------|---------|
| math_2 (概率题) | judge 误选错误答案, 短路锁定错误 | VERIFY 验算确认 15/28 正确, 加权投票让 glm-5.2 答案胜出 | P0 VERIFY + 加权 |
| trap_1 (常识题) | PICK-BEST 通过 synth 答对 | 自适应路由到 AGGREGATOR, 省去 judge 直接 synth | P1 自适应路由 |
| code_1 (代码题) | pick-best 选中正确模型 | VERIFY 执行代码确认, pick-best 高置信度短路 | P0 VERIFY |
| math_1 (计算题) | 所有模式都答错 | VERIFY 验算后发现无 panel 答对, synth 可基于验算结果修正 | P0 VERIFY |
| 重复数学题 | 每次都靠 judge 判断 | 历史追踪: glm-5.2 数学准确率高, 权重升至 1.5x | P2 性能追踪 |

---

## 五、风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| VERIFY 阶段增加延迟 | 高 | +3-5s | 仅对 math/code 题触发, 开放题跳过 |
| code_interpreter 执行失败 | 中 | 验证不可用 | 回退到 v1.2 的置信度评估 |
| 题型分类错误 | 中 | 路由到次优模式 | 分类器有 fallback 到 FULL |
| 加权权重需要冷启动 | 中 | 初期权重不准 | 前 3 轮用默认权重, 积累数据后调整 |
| 聚类算法误判 | 低 | 异类检测失败 | 阈值可配置, 有 fallback 到 judge |

---

## 六、向后兼容性

1. **VERIFY 阶段可选**: `enable_verify: bool = False` 默认关闭, 不影响现有管道
2. **权重默认 1.0**: 不配置 weight 的模型行为不变
3. **自适应路由可选**: `enable_adaptive_routing: bool = False` 默认关闭
4. **性能追踪可选**: `enable_performance_tracking: bool = False` 默认关闭
5. **所有 v1.2 配置和模式不受影响**

---

## 七、成功指标

| 指标 | v1.2 基线 | v1.3 目标 |
|------|----------|----------|
| math_2 准确率 | 0% (3 轮全错) | >80% (VERIFY 验算) |
| 整体准确率 | 40-60% (波动) | >70% (稳定) |
| pick-best 误判率 | 50% (2 次触发, 1 次错) | <10% (VERIFY 保护) |
| judge 瓶颈影响 | 严重 (误判锁定错误) | 可控 (验证 + 加权 + 聚类三重保护) |
| 模式自适应 | 无 (手动配置) | 自动 (题型检测路由) |
