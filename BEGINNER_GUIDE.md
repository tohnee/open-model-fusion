# Open Fusion 小白上手手册：把多模型 Fusion 集成进 Coding Agent / Chatbot

> 结论：**可以集成**。Open Fusion 本质上是一个“多模型复核工具”：你的 coding agent 或 chatbot 平时仍用主模型回答；当问题高风险、复杂、需要第二意见时，再调用 `open-fusion` 让多个模型并行审查、由 judge 结构化归纳、最后输出一条融合答案。

本手册按“完全新手”的顺序写：先跑通命令行，再接入 coding agent，再接入自己的 chatbot / 后端服务。

---

## 0. 你需要先理解的 3 个词

| 词 | 小白解释 | 在 Open Fusion 里的位置 |
|---|---|---|
| Panel | 多个参考模型，像评审小组 | `FusionConfig.panel` / `--panel` |
| Judge | 裁判模型，整理共识、分歧、盲点 | `FusionConfig.judge` / `--judge` |
| Synthesizer | 最终写答案的模型 | 默认就是 caller/judge |

默认 FULL 流程：

```text
用户问题 -> Panel 并行回答 -> Judge 输出结构化 Analysis -> Synthesizer 写最终答案
```

还有一个更像 Hermes MoA 的快速模式：

```text
用户问题 -> Panel 并行回答 -> Aggregator 直接综合原始回答
```

它通过 `moa_fast` preset 或 `FusionMode.AGGREGATOR` 使用。

---

## 1. 适合接入日常 coding agent / chatbot 吗？

适合，但**不要把它当默认聊天模型**。

### 适合自动触发的场景

- 架构设计：数据库选型、缓存策略、限流方案、微服务边界。
- 代码审查：安全漏洞、并发 bug、性能瓶颈、迁移风险。
- Debug 复盘：多个可能原因需要交叉验证。
- 产品 / 技术选型：多个方案需要列共识、分歧、盲点。
- 高风险输出：合规、财务、医疗、法律等需要明确“不能保证”的边界。
- 用户明确说：`多模型共识`、`second opinion`、`fusion`、`MoA`、`panel of models`。

### 不建议自动触发的场景

- 改一个变量名、写一个很短的函数。
- 低风险问答、闲聊、格式转换。
- 用户明确要求最快或最低成本。

### 一个简单触发规则

在 agent system prompt / tool policy 里写：

```text
当任务满足以下任一条件时调用 open-fusion：
1. 用户明确要求多模型、共识、复核、Fusion、MoA、second opinion。
2. 任务涉及架构/安全/性能/合规/长期维护决策。
3. 你对单模型答案把握不足，且答错代价高。
否则直接用主模型回答。
```

---

## 2. 安装

### 2.1 准备 Python

需要 Python 3.10+：

```bash
python --version
```

### 2.2 安装 Open Fusion

在仓库根目录运行：

```bash
pip install -e .
```

验证命令是否可用：

```bash
open-fusion --help
```

---

## 3. 配置 API Key

默认走 OpenRouter，因为它能用一个 key 路由多个厂商模型：

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

如果你用自己的 OpenAI-compatible gateway：

```bash
export OPEN_FUSION_BASE_URL="https://your-gateway/v1"
export OPEN_FUSION_API_KEY="..."
```

> 注意：模型 slug 会随服务商变化。若 preset 模型不可用，请用 `--panel` / `--judge` 传你自己的模型 slug。

---

## 4. 先从命令行跑通

### 4.1 最简单调用

```bash
open-fusion "请评审：PostgreSQL + Redis 做多租户限流是否合理？"
```

### 4.2 使用便宜 preset

```bash
open-fusion "评审这个 Python 并发爬虫设计" --preset budget
```

### 4.3 使用 coding preset

```bash
open-fusion "审查这段认证中间件可能有哪些安全漏洞" --preset code
```

### 4.4 使用 logic preset

```bash
open-fusion "这个分布式锁方案是否存在竞态条件？" --preset logic
```

### 4.5 使用 MoA 快速模式

```bash
open-fusion "给这个重构方案做一次快速多模型复核" --preset moa_fast
```

### 4.6 查看结构化分析

```bash
open-fusion "评审 GraphQL vs REST 的取舍" --show-analysis
```

### 4.7 输出 JSON 给程序处理

```bash
open-fusion "评审这个缓存方案" --json > fusion-result.json
```

---

## 5. 选择哪个 preset？

| preset | 适合场景 | 特点 |
|---|---|---|
| `quality` | 高风险/重要决策 | 默认质量优先，成本更高 |
| `budget` | 日常复核 | 成本较低，适合作为默认 agent 工具 |
| `logic` | 推理、并发、分布式一致性 | consensus 阈值更高 |
| `code` | 代码审查、bug 分析、实现方案 | 更偏 coding 模型组合 |
| `moa_fast` | 快速第二意见 | 跳过 structured judge，延迟更低 |

推荐默认策略：

```text
日常 coding agent：budget
代码/安全审查：code
架构/高风险：quality
逻辑/并发/一致性：logic
用户要快：moa_fast
```

---

## 6. 集成到 Codex / Claude Code / 其他 Coding Agent

### 6.1 最低成本集成：让 agent 直接调用 CLI

在你的 `AGENTS.md`、system prompt 或 agent instructions 中加入：

```markdown
## Multi-model review tool
When the task is high-risk, ambiguous, architectural, security-sensitive, or the user asks for multi-model consensus / second opinion / Fusion / MoA, run:

`open-fusion "<the exact question to review>" --preset budget --show-analysis`

Use `--preset code` for code review, `--preset logic` for concurrency/distributed reasoning, and `--preset quality` for high-stakes architecture decisions. Do not use it for trivial edits.
```

然后 agent 就可以像调用普通 shell 命令一样调用 `open-fusion`。

### 6.2 推荐给 coding agent 的调用模板

```bash
open-fusion "
请作为多模型评审团审查以下问题：

任务：<粘贴用户任务>
代码/设计：<粘贴关键上下文，不要粘整个仓库>

请重点输出：
1. 主要共识
2. 关键分歧
3. 风险和盲点
4. 建议采用的方案
5. 需要进一步验证的测试
" --preset code --show-analysis
```

### 6.3 Agent 应如何使用结果

建议 agent 不要机械复制 Fusion 输出，而是：

1. 读最终答案。
2. 若有 `--show-analysis`，检查 contradictions / blind_spots。
3. 回到本地代码仓库做实际验证。
4. 把 Fusion 结果作为“外部评审意见”，不是绝对真理。

---

## 7. 集成到自己的 Chatbot / Web 后端

### 7.1 最简单：后端用 subprocess 调 CLI

适合快速接入、内部工具、低 QPS。

```python
import subprocess


def fusion_review(question: str) -> str:
    proc = subprocess.run(
        ["open-fusion", question, "--preset", "budget"],
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return proc.stdout.strip()
```

### 7.2 更推荐：直接用 Python API

适合生产后端、需要拿 telemetry/status/analysis 的场景。

```python
import asyncio
from open_fusion import fuse, load_preset, FusionStatus

SUCCESS = {
    FusionStatus.OK,
    FusionStatus.JUDGE_FALLBACK,
    FusionStatus.CONSENSUS_SHORTCUT,
    FusionStatus.PICK_BEST_SHORTCUT,
    FusionStatus.AGGREGATOR_MODE,
}

async def fusion_review(question: str, preset: str = "budget") -> dict:
    cfg = load_preset(preset)
    result = await fuse(question, cfg)
    if result.status not in SUCCESS:
        return {"ok": False, "error": result.error, "telemetry": result.telemetry}
    return {
        "ok": True,
        "status": result.status.value,
        "answer": result.text,
        "analysis": result.analysis.to_dict() if result.analysis else None,
        "telemetry": result.telemetry,
    }

# asyncio.run(fusion_review("评审这个缓存方案"))
```

### 7.3 Chatbot 中的 UX 建议

在 UI 上展示：

- “正在进行多模型复核...”
- 最终答案。
- 可展开的“共识 / 分歧 / 盲点”。
- cost / latency telemetry 仅给开发者或高级用户看。

---

## 8. 成本、延迟与稳定性

Fusion 会调用多个模型：

```text
总成本 ≈ panel 模型调用 + judge 调用 + synthesis 调用
总延迟 ≈ 最慢 panel + judge + synthesis
```

优化建议：

- 默认用 `budget`。
- 只有高风险才用 `quality`。
- 对 3+ panel 使用 `fast_majority_k=N-1`，跳过最慢模型。
- 用 `moa_fast` 给低风险第二意见提速。
- 不要把整个代码仓库塞进 prompt，只给关键文件/片段。

---

## 9. 联网工具怎么用？

默认不联网。需要检索当前资料时：

```bash
export EXA_API_KEY="..."  # 或 BRAVE_API_KEY
open-fusion "调研最近的向量数据库选型" --tools --exclude-domains "example.com"
```

工具策略：

- Panel 阶段可以联网/执行工具。
- Synthesis 阶段默认不使用工具，保证最终答案不漂移出 judge 分析。
- benchmarking 时一定用 `--exclude-domains` 排除答案源，避免污染。

---

## 10. 常见问题

### Q1：它会不会太贵？

会比单模型贵，所以不要默认每条消息都用。把它放在“重要决策前的复核按钮”上最合适。

### Q2：它能保证答案一定正确吗？

不能。它能降低单模型盲区，暴露分歧和 blind spots，但仍然需要你对关键结论做测试/查证。

### Q3：我可以只用两个便宜模型吗？

可以，用 `--panel` 自定义：

```bash
open-fusion "评审这个设计" \
  --panel "model-a,model-b" \
  --judge "model-a"
```

### Q4：为什么有时 status 不是 `ok` 但也有答案？

以下 status 都表示有可用答案：

- `ok`：完整 FULL 路径。
- `judge_fallback`：judge JSON 失败后，从原始 panel 回答综合。
- `consensus_shortcut`：panel 多数高度一致，跳过 judge/synthesis。
- `pick_best_shortcut`：judge 识别某个 panel 回答明显最好，直接采用。
- `aggregator_mode`：MoA-style aggregator 模式。

只有 `error` 表示失败。

### Q5：如何验证它真的适合我的业务？

先跑离线 demo：

```bash
open-fusion-eval --demo
```

再做自己的小评测集：准备 10-30 个真实任务，用 `open-fusion-eval` 比较 baseline 和 fusion 的得分、成本、错误数。

---

## 11. 生产落地 checklist

- [ ] 只在高价值场景触发，不做默认聊天路径。
- [ ] 设置预算：哪些用户/哪些任务可用 `quality`。
- [ ] 记录 `status`、`telemetry`、`panel_ok`、`panel_failed`。
- [ ] 对 `error` 做降级：回退主模型或提示稍后再试。
- [ ] 对联网 benchmark 设置 `--exclude-domains`。
- [ ] 对代码任务只传关键上下文，避免 prompt 过长。
- [ ] 定期更新模型 slug 和 preset。
- [ ] 用自己的任务集评测是否 `worth_it`。
