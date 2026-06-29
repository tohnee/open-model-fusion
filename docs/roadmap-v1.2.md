# Roadmap v1.2 — 多 Provider 负载均衡与弹性路由

> 基线版本: v1.1.0 (commit `7354fee`, 2026-06-30)
> 目标版本: v1.2.0
> 核心主题: 从"能连多 Provider"到"智能调度多 Provider"

---

## 一、当前基线 (v1.1.0 Baseline)

### 1.1 测试与覆盖率

| 指标 | 数值 |
|------|------|
| 测试总数 | 99 (全部通过) |
| orchestrator.py 覆盖率 | 100% |
| 总覆盖率 | 64% |
| 核心模块覆盖率 | prompts 100%, config 92%, schema 90%, synthesizer 86% |

### 1.2 性能基线 (模拟延迟: Panel 800ms / Judge 1200ms / Synth 1000ms)

| 模式 | 延迟 | API 调用 | 加速比 |
|------|------|---------|--------|
| FULL Mode | 3,003ms | 5 | 1.00x |
| AGGREGATOR Mode | 2,004ms | 4 | 1.50x |
| Consensus Shortcut | 803ms | 3 | 3.74x |
| Pick-Best Shortcut | 2,002ms | 4 | 1.50x |

### 1.3 多 Provider 现状 (P1-B 已实现)

当前能力:
- `ModelSpec.base_url` / `ModelSpec.api_key` 支持每模型独立网关
- `_post_with()` 方法按模型路由 HTTP 请求
- `complete()` 在 model 有独立配置时自动切换网关

```
┌─────────────┐    ┌──────────────────────────┐
│  FusionConfig │    │       ModelClient         │
│  base_url    │───▶│  base_url (全局回退)       │
│  api_key     │    │  _post()    → 全局网关     │
│              │    │  _post_with() → 模型网关   │
│  panel[0]    │    └──────────────────────────┘
│   .base_url  │         │
│   .api_key   │         ▼
│  panel[1]    │    ┌──────────┐  ┌──────────┐  ┌──────────┐
│   .base_url  │    │ Provider A│  │ Provider B│  │ Provider C│
│   .api_key   │    │ (OpenAI)  │  │ (Anthropic)│  │ (Volcengine)│
└─────────────┘    └──────────┘  └──────────┘  └──────────┘
```

---

## 二、问题诊断: 7 个关键缺口

### 缺口 1: 无跨 Provider 故障转移

**现状**: `client.py` L207-216 的重试逻辑只对**同一 Provider** 做指数退避重试。如果 Provider A 宕机 (500/503)，会重试 3 次同一个 Provider，全部失败后 panel 该模型标记为 failed。

**影响**: 单 Provider 故障 = 该模型完全不可用，即使配置了等效备选模型。

```python
# 当前: 只重试同一 Provider
while True:
    try:
        data = await loop.run_in_executor(
            self._executor, self._post_with,
            eff_base_url, eff_api_key, payload, params.timeout_s)
        return self._parse(data, model.slug)
    except ProviderError as e:
        if e.status in _RETRY_STATUSES and attempt < self.max_retries:
            attempt += 1
            await asyncio.sleep(min(2 ** attempt, 8))  # 重试同一 Provider
            continue
        raise  # 3 次后放弃，无备选
```

### 缺口 2: 无加权路由

**现状**: 每个 `ModelSpec` 是一个固定的 slug，无法表达"70% 流量到 Provider A 的 gpt-4，30% 到 Provider B 的 gpt-4"。

**影响**: 无法在多个等效 Provider 间分散负载，单个 Provider 可能过载。

### 缺口 3: 无健康追踪 / 熔断器

**现状**: 没有记录 Provider 的历史成功率/延迟。即使 Provider A 连续失败 50 次，下一次请求仍会尝试 A。

**影响**: 故障 Provider 拖慢整体管道 (重试 3 次 × 指数退避 = 最多 ~14s 浪费)。

### 缺口 4: 无每 Provider 并发限制

**现状**: `max_in_flight` 是全局信号量。如果 3 个 panel 模型都在 Provider A，它们会竞争同一个全局信号量，而非 Provider A 的容量上限。

**影响**: 单 Provider 过载时无法隔离影响，慢 Provider 拖累快 Provider。

### 缺口 5: 无成本感知路由

**现状**: 不追踪每个 Provider 的 token 单价。无法在"贵但快"和"便宜但慢"间做权衡。

**影响**: 无法优化总体成本，尤其在大规模批量推理时。

### 缺口 6: 无延迟感知路由

**现状**: 不记录 Provider 的历史响应延迟。P2-A 的 prompt cache 命中率优化只覆盖了 prompt 层，未覆盖网络层。

**影响**: 无法自动避开当前延迟高的 Provider。

### 缺口 7: 无模型等价组

**现状**: `ModelSpec` 是 1:1 映射到一个 slug。无法表达"gpt-4 在 OpenAI 和 Azure 上是等价的，可以互为备选"。

**影响**: 故障转移必须手动配置，无法自动在等效模型间切换。

---

## 三、v1.2 Roadmap: 5 个优先级方案

### P0: Provider 熔断器 + 自动故障转移 (最高优先级)

**目标**: Provider 故障时自动切换到等效备选，不再重试死掉的 Provider。

**设计**:

```python
@dataclass
class ProviderHealth:
    """追踪单个 Provider 的健康状态 (滑动窗口)。"""
    base_url: str
    successes: int = 0
    failures: int = 0
    last_failure_ts: float = 0.0
    consecutive_failures: int = 0

    @property
    def is_open(self) -> bool:
        """熔断器是否处于开启状态 (连续失败 >= 5 次且在冷却期内)。"""
        if self.consecutive_failures < 5:
            return False
        # 冷却期 30s, 过后进入 half-open 尝试
        return (time.monotonic() - self.last_failure_ts) < 30.0

    def record_success(self):
        self.successes += 1
        self.consecutive_failures = 0

    def record_failure(self):
        self.failures += 1
        self.consecutive_failures += 1
        self.last_failure_ts = time.monotonic()


class ProviderHealthRegistry:
    """全局 Provider 健康注册表 (单例, 跨 fusion 调用共享)。"""
    _health: dict[str, ProviderHealth] = {}

    @classmethod
    def get(cls, base_url: str) -> ProviderHealth:
        return cls._health.setdefault(base_url, ProviderHealth(base_url))

    @classmethod
    def is_available(cls, base_url: str) -> bool:
        return not cls.get(base_url).is_open
```

```python
# ModelSpec 新增 fallback 字段
@dataclass
class ModelSpec:
    slug: str
    temperature: float | None = None
    max_tokens: int | None = None
    base_url: str | None = None
    api_key: str | None = None
    # v1.2 新增: 等效备选模型列表 (按优先级排序)
    fallbacks: list["ModelSpec"] = field(default_factory=list)
```

**管道变化**:

```
请求 ModelSpec("gpt-4", base_url="openai")
  → 检查 ProviderHealth["openai"]: 熔断器开?
    → 否: 发送请求
      → 成功: record_success, 返回结果
      → 失败 (429/5xx): record_failure, 尝试 fallbacks[0]
    → 是: 直接尝试 fallbacks[0]
      → fallbacks[0] = ModelSpec("gpt-4", base_url="azure")
      → 检查 ProviderHealth["azure"]...
```

**预估收益**: 单 Provider 故障时，管道从 "重试 3 次 + 失败 (~14s)" 变为 "熔断 + 切换 (<1s)"。

**改动文件**: `client.py`, `config.py`, 新增 `health.py`
**测试**: 熔断器状态机 (closed/open/half-open), fallback 链遍历, 健康恢复

---

### P1: 模型等价组 (Model Equivalence Group)

**目标**: 声明式定义"哪些模型可以互为备选"，故障转移时自动在组内切换。

**设计**:

```python
@dataclass
class ModelGroup:
    """一组功能等价的模型, 分布在不同 Provider 上。

    用法:
        group = ModelGroup("gpt-4-class", models=[
            ModelSpec("openai/gpt-4", base_url="https://api.openai.com/v1"),
            ModelSpec("gpt-4", base_url="https://azure.example.com/v1"),
        ], strategy="failover")  # 或 "round_robin" / "weighted"
    """
    name: str
    models: list[ModelSpec]
    strategy: str = "failover"  # failover | round_robin | weighted | latency_first
    weights: list[float] | None = None  # strategy=weighted 时使用

    def select(self, health: ProviderHealthRegistry) -> ModelSpec | None:
        """根据策略和健康状态选择一个可用模型。"""
        available = [m for m in self.models if health.is_available(m.base_url or "")]
        if not available:
            return None
        if self.strategy == "failover":
            return available[0]
        elif self.strategy == "round_robin":
            # 轮询 (维护一个内部计数器)
            ...
        elif self.strategy == "latency_first":
            # 选历史延迟最低的
            ...
```

**配置示例**:

```python
config = FusionConfig(
    panel=[
        ModelGroup("gpt4-class", models=[
            ModelSpec("openai/gpt-4", base_url="https://api.openai.com/v1"),
            ModelSpec("gpt-4", base_url="https://azure.example.com/v1"),
        ], strategy="failover"),
        ModelSpec("anthropic/claude-opus"),
    ],
    judge=ModelSpec("zhipu/glm-5.2"),
)
```

**改动文件**: `config.py`, `client.py`, `panel.py`
**测试**: failover/round_robin/weighted 策略, 组内全部不可用, 混合 ModelSpec + ModelGroup panel

---

### P1: 每 Provider 并发隔离

**目标**: 每个 Provider 有独立的并发限制，慢 Provider 不阻塞快 Provider。

**设计**:

```python
class ProviderRateLimiter:
    """每 Provider 独立的信号量池。"""
    _sems: dict[str, asyncio.Semaphore] = {}
    _limits: dict[str, int] = {}  # base_url → max_concurrent

    @classmethod
    def configure(cls, base_url: str, max_concurrent: int):
        cls._limits[base_url] = max_concurrent
        cls._sems[base_url] = asyncio.Semaphore(max_concurrent)

    @classmethod
    async def acquire(cls, base_url: str):
        if base_url not in cls._sems:
            cls._sems[base_url] = asyncio.Semaphore(cls._limits.get(base_url, 8))
        await cls._sems[base_url].acquire()

    @classmethod
    def release(cls, base_url: str):
        if base_url in cls._sems:
            cls._sems[base_url].release()
```

**在 panel.py 中的使用**:

```python
async def _run_one(client, model, question, params, ...):
    eff_base_url = model.base_url or client.base_url
    await ProviderRateLimiter.acquire(eff_base_url)
    try:
        return await _run_one_inner(client, model, question, params, ...)
    finally:
        ProviderRateLimiter.release(eff_base_url)
```

**配置示例**:

```python
FusionConfig(
    panel=[...],
    provider_limits={
        "https://api.openai.com/v1": 3,   # OpenAI 限 3 并发
        "https://api.anthropic.com/v1": 5,  # Anthropic 限 5 并发
    },
)
```

**预估收益**: 当 Provider A 限流时，Provider B 的模型不受影响，整体管道不被拖慢。

**改动文件**: 新增 `ratelimit.py`, `panel.py`, `config.py`
**测试**: 并发隔离, 限流后排队, 混合 Provider 并发

---

### P2: 延迟感知路由

**目标**: 追踪每 Provider 的历史 P50/P95 延迟，自动优先选择快的 Provider。

**设计**:

```python
@dataclass
class ProviderMetrics:
    """滑动窗口延迟追踪。"""
    latencies: deque  # maxlen=100, 存最近 100 次请求延迟
    p50: float = 0.0
    p95: float = 0.0

    def record(self, latency_ms: float):
        self.latencies.append(latency_ms)
        sorted_l = sorted(self.latencies)
        self.p50 = sorted_l[len(sorted_l) // 2]
        self.p95 = sorted_l[int(len(sorted_l) * 0.95)]
```

`ModelGroup(strategy="latency_first")` 在选择时优先取 `p95` 最低的 Provider。

**改动文件**: `health.py` (合并 ProviderHealth + ProviderMetrics), `config.py`
**测试**: 延迟记录, P50/P95 计算, 策略切换

---

### P2: 成本感知路由

**目标**: 追踪每 Provider 的 token 单价，支持"预算内选最优"策略。

**设计**:

```python
@dataclass
class ProviderPricing:
    """每 Provider 的 token 单价 (per 1K tokens)。"""
    input_per_1k: float
    output_per_1k: float

# ModelGroup 新增 budget_aware 模式
ModelGroup("gpt4-class", models=[...], strategy="budget_aware",
           pricing={
               "openai": ProviderPricing(input_per_1k=0.03, output_per_1k=0.06),
               "azure": ProviderPricing(input_per_1k=0.025, output_per_1k=0.05),
           },
           max_cost_per_call=0.10)
```

**改动文件**: `config.py`, `health.py`, `client.py` (记录 usage)
**测试**: 预算耗尽时降级, 成本对比选择

---

## 四、实施计划

### 阶段 1: 弹性基础 (P0 熔断器 + 故障转移)

```
Week 1-2:
  - 新增 health.py: ProviderHealth + ProviderHealthRegistry
  - ModelSpec.fallbacks 字段 + client.py 故障转移逻辑
  - TDD: 熔断器状态机测试 (12+ cases)
  - 集成测试: Provider A 故障 → 自动切换 Provider B

验收标准:
  - 单 Provider 故障时管道不中断
  - 熔断器 5 次连续失败后 30s 冷却
  - half-open 状态下探活成功则恢复
```

### 阶段 2: 智能调度 (P1 等价组 + 并发隔离)

```
Week 3-4:
  - 新增 ModelGroup + 4 种选择策略
  - 新增 ratelimit.py: ProviderRateLimiter
  - panel.py 集成每 Provider 信号量
  - TDD: 策略选择测试 (15+ cases) + 并发隔离测试 (8+ cases)

验收标准:
  - ModelGroup 可替代 ModelSpec 作为 panel 成员
  - failover/round_robin/weighted/latency_first 四种策略可切换
  - 每 Provider 并发独立限制, 互不阻塞
```

### 阶段 3: 观测性 (P2 延迟 + 成本追踪)

```
Week 5-6:
  - ProviderMetrics 滑动窗口延迟追踪
  - ProviderPricing 成本模型
  - log_event 新增 provider_health / provider_latency 事件
  - 基准测试: 对比 v1.1 的延迟/成本

验收标准:
  - 每个 Provider 有 P50/P95 延迟记录
  - 成本感知策略在预算内选最优 Provider
  - 日志可追溯每次请求的路由决策
```

---

## 五、架构演进对比

### v1.1 (当前): 静态路由

```
ModelSpec("gpt-4", base_url="openai")
  → 固定发往 openai
  → 失败重试 3 次 openai
  → 3 次后 panel 标记 failed
```

### v1.2 (目标): 弹性路由

```
ModelGroup("gpt4-class", models=[
    ModelSpec("gpt-4", base_url="openai"),     # 主力
    ModelSpec("gpt-4", base_url="azure"),      # 备选
    ModelSpec("gpt-4", base_url="openrouter"), # 兜底
], strategy="latency_first")
  → 检查健康: openai 熔断? → 跳过
  → 检查延迟: azure P95=800ms, openrouter P95=1200ms → 选 azure
  → 每 Provider 独立并发: openai 限流不影响 azure
  → 成功后更新延迟/健康指标
```

---

## 六、风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 熔断器误判 (网络抖动触发熔断) | 中 | 可用性下降 | half-open 探活机制 + 冷却期可配置 |
| ModelGroup 增加 API 复杂度 | 中 | 学习成本 | 向后兼容: ModelSpec 仍可直接用 |
| 延迟追踪内存开销 | 低 | 内存增长 | 滑动窗口 maxlen=100, 总量可控 |
| 并发限制死锁 | 低 | 管道卡住 | 信号量 + try/finally 保证释放 |
| 成本数据需要人工维护 | 中 | 路由不准 | 提供默认定价表 + 环境变量覆盖 |

---

## 七、向后兼容性承诺

1. **ModelSpec 不变**: 现有代码无需改动，`fallbacks` 默认空列表
2. **FusionConfig 不变**: 新增字段全部有默认值
3. **panel.py 接口不变**: ModelGroup 在 `_run_one` 内部解析为 ModelSpec
4. **预设系统不变**: `quality`/`budget`/`logic`/`code`/`moa_fast` 行为不受影响
5. **from_plugin() 不变**: 仍接受 `{"model": ..., "analysis_models": [...]}` 格式

---

## 八、成功指标

| 指标 | v1.1 基线 | v1.2 目标 |
|------|----------|----------|
| 单 Provider 故障恢复时间 | ~14s (重试耗尽) | <1s (熔断+切换) |
| Provider 故障时管道成功率 | 依赖手动 fallback | 自动 >95% |
| 并发隔离 | 全局信号量 | 每 Provider 独立 |
| 路由策略 | 固定 | 4 种可选 |
| 延迟可观测性 | 无 | P50/P95 per Provider |
| 成本可观测性 | 无 | per-call 成本追踪 |
