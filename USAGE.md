# Open Fusion · 使用手册

> Multi-model deliberation as a tool. 把一个问题扇出到一组模型并行作答，由裁判
> 蒸馏成结构化分析，最后由调用方模型合成一条有据可循的最终答案。

## 1. 安装

```bash
cd open-fusion-skill
pip install -e .
```

零运行期依赖（仅标准库）。安装后会注册两个命令行工具：

- `open-fusion`：单次提问的推理 CLI
- `open-fusion-eval`：DRACO 风格评测 CLI

验证：

```bash
open-fusion --help
python tests/test_open_fusion.py      # 52 个离线测试，全过
```

## 2. 设置 API Key

默认走 OpenRouter（一个 key 覆盖所有厂商模型）：

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

或者用任何 OpenAI 兼容的 gateway：

```bash
export OPEN_FUSION_BASE_URL="https://your-gateway/v1"
export OPEN_FUSION_API_KEY="..."
```

## 3. CLI 用法

### 3.1 基础调用

```bash
# 默认 Quality preset（两个旗舰模型 + Opus 做 judge）
open-fusion "为什么 Rust 的 borrow checker 让并发代码更安全？"

# Budget preset（三个便宜模型，自动启用 fast_majority_k=2 早退最慢一个）
open-fusion "对比 ridge / lasso / elastic-net" --preset budget

# 自定义 panel + judge
open-fusion "设计一个多租户的限流器" \
    --panel "anthropic/claude-opus-4.1,openai/gpt-5,deepseek/deepseek-chat" \
    --judge "anthropic/claude-opus-4.1" \
    --show-analysis

# 启用联网检索（需要 EXA_API_KEY 或 BRAVE_API_KEY）
export EXA_API_KEY="..."
open-fusion "最新的 speculative decoding 吞吐数据" --tools \
    --exclude-domains "en.wikipedia.org"

# 显式控制早退策略（拿到 K 个 ok 就 cancel 剩下的）
open-fusion "..." --fast-majority-k 2

# 机器可读输出
open-fusion "..." --json
```

### 3.2 输出契约

| 流 | 内容 |
|---|---|
| **stdout** | 最终答案（或 `--json` 时的完整结果 JSON） |
| **stderr** | `[status]` / `[telemetry]` / `[analysis]` 等诊断信息 |

```bash
open-fusion "你的问题" > answer.md          # 只保留答案
open-fusion "你的问题" 2> telemetry.log     # 只保留指标
```

### 3.3 完整 CLI 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--preset` | `quality` | `quality` 或 `budget` |
| `--panel` | — | 逗号分隔的模型 slug，覆盖 preset |
| `--judge` | panel[0] | 裁判 / 合成模型 |
| `--tools` | off | 开启 panel 阶段的 web 工具 |
| `--exclude-domains` | — | 逗号分隔，从联网检索中排除的域 |
| `--max-in-flight` | 8 | 并发上限 |
| `--fast-majority-k` | preset 决定 | 拿到 K 个 ok 后早退；3+ panel preset 默认 = N-1 |
| `--json` | off | 全量结果 JSON 到 stdout |
| `--show-analysis` | off | 把结构化分析打到 stderr |

## 4. 运行时日志（调试 / 性能分析）

**默认完全静默**（library 最佳实践，不污染 stdout/stderr）。需要时打开：

```bash
# 全程结构化日志到 stderr，含 duration_ms / 输入输出 chars / tokens
OPEN_FUSION_LOG=DEBUG open-fusion "你的问题"

# 仅看阶段开始/结束
OPEN_FUSION_LOG=INFO open-fusion "你的问题"
```

输出样例（实测）：

```
2026-06-24 01:00:09 [INFO] phase=orchestrator event=start n_panel=2 judge=anthropic/claude-opus-4.1
2026-06-24 01:00:09 [INFO] phase=panel event=end n_ok=2 n_failed=0 n_cancelled=0 duration_ms=2840
2026-06-24 01:00:09 [INFO] phase=synthesis event=end duration_ms=1100 out_chars=2350 prompt_tokens=420 completion_tokens=512
2026-06-24 01:00:09 [INFO] phase=orchestrator event=end status=ok total_duration_ms=4900 critical_path_ms=4850 panel_max_ms=2840 judge_ms=910 synthesis_ms=1100
```

每条日志的可解析字段：

| 字段 | 含义 |
|---|---|
| `phase` | `orchestrator` / `panel` / `synthesis` / `synthesis_fallback` |
| `event` | `start` / `end` / `error` / `fallback` / `depth_exceeded` / `all_failed` |
| `duration_ms` | 该阶段端到端耗时 |
| `panel_max_ms` / `judge_ms` / `synthesis_ms` | 关键路径三段拆解 |
| `critical_path_ms` | 三段相加，约等于 wall-clock |
| `n_ok` / `n_failed` / `n_cancelled` | panel 三分类统计 |
| `question_chars` / `analysis_chars` / `out_chars` | 输入输出数据量 |
| `prompt_tokens` / `completion_tokens` | 计费用量 |

## 5. 库 API

### 5.1 一次性使用

```python
import asyncio
from open_fusion import fuse, load_preset

async def main():
    cfg = load_preset("quality")
    result = await fuse("你的问题", cfg)
    print(result.text)                # 最终答案
    print(result.status)              # ok | error | judge_fallback
    print(result.analysis.to_dict())  # 结构化裁判分析
    print(result.telemetry)           # 含 critical_path_ms / synthesis_ms 等

asyncio.run(main())
```

### 5.2 自定义 panel

```python
from open_fusion import FusionConfig, ModelSpec

cfg = FusionConfig(
    panel=[
        ModelSpec("anthropic/claude-opus-4.1"),
        ModelSpec("openai/gpt-5"),
        ModelSpec("deepseek/deepseek-chat"),
    ],
    judge=ModelSpec("anthropic/claude-opus-4.1"),
    tools_enabled=False,
    fast_majority_k=2,                  # 拿到 2 个 ok 就早退最慢的
    excluded_domains=["en.wikipedia.org"],
)
```

### 5.3 程序化打开日志

```python
from open_fusion._logging import enable_logging
enable_logging("DEBUG")                 # 或写入自定义 stream:
# enable_logging("DEBUG", stream=open("of.log", "w"))
```

## 6. 自带评测：判断 Fusion 在你的任务上「值不值得用」

### 6.1 离线 demo（零 key）

```bash
# 看 verdict 报告格式
open-fusion-eval --demo

# 注入模拟时延，验证延迟拆解通路
OPEN_FUSION_DEMO_LATENCY_MS_PANEL=30 \
OPEN_FUSION_DEMO_LATENCY_MS_JUDGE=15 \
OPEN_FUSION_DEMO_LATENCY_MS_SYNTH=10 \
    open-fusion-eval --demo --suite draco --md /tmp/report.md
```

### 6.2 真实评测

```bash
open-fusion-eval --tasks sample --preset budget \
    --baseline anthropic/claude-opus-4.8 \
    --grader llm --grader-model google/gemini-3.1-pro-preview \
    --passes 3 --tools --md report.md
```

### 6.3 自带的 task suites

| `--suite` | 内容 |
|---|---|
| `sample` | 4 个示例任务（向后兼容） |
| `draco` | DRACO 10 个领域 |
| `longhorizon` | pass^k 长程可靠性 |
| `semiconductor` | 半导体 / DRAM 行业知识 |
| `all` | 全部 16 个任务 |

### 6.4 报告里能拿到什么

- 总分 lift（Fusion vs baseline）
- 按 4 类 rubric（factual_accuracy / breadth_depth / presentation / citation）
- 按 domain 细分
- cost / latency 倍数（含 **panel-max / judge / synthesis 三段拆解**）
- 触发的 penalty 数（安全增益）
- pass^k 长程可靠性
- `worth_it` 布尔结论 + recommendations

## 7. 集成到 agent

### 7.1 Claude Code

把整个 `open-fusion-skill/` 目录拷到下列任一位置：

- **项目级**：`.claude/skills/open-fusion/`
- **个人级**：`~/.claude/skills/open-fusion/`

`SKILL.md` 已写好触发短语，agent 在用户说 *"多模型共识 / 第二意见 / fuse 几个模型"* 时会自动调用。

### 7.2 Codex / 自家 agent

在 `AGENTS.md` 或自定义 prompt 里加一行：

```markdown
## Multi-model deliberation
For high-stakes/ambiguous questions, run:
`open-fusion "<question>" --preset quality`
```

也可以把 `open-fusion` 注册成函数 / tool，让 agent 自主决定何时调用。

## 8. 什么时候用 / 什么时候不用

| ✅ 适合 | ❌ 不适合 |
|---|---|
| 深度研究、多领域批判、架构 / 选型决策 | 短战术指令 |
| 答错代价高的问题、医疗 / 法律 / 合规 | 对延迟敏感的聊天 |
| 用户明确要 "多模型共识 / 第二意见" | 答案明确的转换 / 摘要（单模型一次更快更省） |

成本约 2-3× 单模型；按每次完成累加计费（panel + judge + synthesis）。


## 9. 真实业务场景落地指南

> 关键原则：不要把 Fusion 当作默认聊天模型；把它放在**高风险 / 高不确定 / 需要可审计交叉验证**的决策闸门上。Fusion 的价值不是“让每个回答都更贵地生成一次”，而是让关键问题在输出前经过多模型复核、结构化分歧整理和盲点暴露。

### 9.1 什么时候应该触发 Fusion

| 业务场景 | 适合原因 | 示例 prompt |
|---|---|---|
| 技术架构决策 | 需要从成本、复杂度、可靠性、团队维护等多个角度交叉审查 | “评审这个多租户限流设计，输出共识、争议、盲点和上线建议。” |
| 方案 / PRD / RFC 审查 | 容易遗漏边界条件、反例和风险 | “审查这个支付重试方案，找出失败模式和必须补充的保护措施。” |
| 法律 / 医疗 / 金融 / 合规辅助分析 | 错误代价高，必须显式暴露不确定性与需人工确认点 | “辅助审查这段合同条款，区分明确风险、潜在风险和需要律师确认的问题。” |
| 供应商 / 工具 / 模型选型 | 需要多维比较，避免单模型偏见 | “在 Snowflake、BigQuery、ClickHouse 中为日志分析场景做选型。” |
| 竞品 / 市场 / 深度研究 | 需要联网、引用、交叉验证和盲点提示 | “分析某竞品最近 6 个月的产品变化和对我们路线的影响。” |
| 关键客户回复 / 对外发布内容 | 需要降低事实错误、措辞风险和遗漏重要 caveat 的概率 | “给企业客户解释本次数据事故原因、影响范围和补救计划。” |

不建议触发 Fusion 的场景：简单事实问答、普通文案润色、实时客服首轮回复、低风险格式转换、答案已经明确的短任务。此类任务用单模型更快、更省。

### 9.2 推荐架构：风险路由 + Fusion 决策闸门

生产系统里建议加一层路由，不要所有请求都直接走 `open-fusion`：

```text
用户 / 业务系统请求
        ↓
风险分类器 / 规则判断
        ↓
低风险：单模型直接回答
中风险：budget fusion 或单模型后按不确定性升级
高风险：quality fusion + 人工复核
        ↓
保存最终答案 + structured analysis + telemetry
        ↓
必要时进入审批 / 审计 / 复盘
```

一个最小可用的路由规则：

```python
def should_use_fusion(task) -> bool:
    if task.domain in {"legal", "medical", "finance", "compliance", "security"}:
        return True
    if task.risk_level >= 0.7:
        return True
    if task.requires_citations:
        return True
    if task.user_requested_consensus:
        return True
    if task.estimated_business_impact_usd > 1000:
        return True
    return False
```

对应的分层策略：

| 风险级别 | 推荐路径 | 说明 |
|---|---|---|
| 低风险 | 单模型 | 快速、低成本；不需要多模型复核 |
| 中风险 | `--preset budget` | 用较低成本做 second opinion / 批量抽检 |
| 高风险 | `--preset quality` | 追求质量上限，适合关键决策 |
| 高风险且对外影响大 | `--preset quality` + 人工复核 | Fusion 只做决策增强，不直接自动执行 |

### 9.3 业务集成方式

#### A. CLI 集成：最快落地

适合内部脚本、CI、Agent、运维工具：

```bash
open-fusion "评审这份数据库迁移方案，输出共识、争议、盲点和最终建议。" \
  --preset quality \
  --show-analysis \
  --json
```

需要联网查证时：

```bash
open-fusion "分析最近三个月某竞品的产品变化，并给出风险判断。" \
  --preset quality \
  --tools \
  --exclude-domains "internal-rubric.example.com" \
  --json
```

#### B. 后端 library 集成：适合产品化

```python
import asyncio
from open_fusion import fuse, load_preset

async def fusion_review(question: str, *, risk: str):
    cfg = load_preset("quality" if risk == "high" else "budget")
    cfg.tools_enabled = True
    cfg.excluded_domains = ["your-rubric-host.example.com"]

    result = await fuse(question, cfg)
    return {
        "status": result.status.value,
        "answer": result.text,
        "analysis": result.analysis.to_dict() if result.analysis else None,
        "telemetry": result.telemetry,
    }
```

#### C. Agent skill / tool 集成：让模型选择性调用

在你的 Agent system prompt 或 tool policy 中加入：

```text
When the user asks for high-stakes analysis, multi-model consensus, architecture
review, legal/medical/financial/compliance assistance, or cross-checking, call the
open-fusion tool. For short low-risk questions, answer directly.
```

这样更接近 Fusion 的真实价值：由主 Agent 判断“是否值得多模型审议”，而不是把每个请求都强制走 Fusion。

### 9.4 业务 prompt 写法：要求“决策材料”，不要只问“答案”

Fusion 更适合结构化评审 prompt。建议显式要求：共识、分歧、部分覆盖、独特洞察、盲点、最终建议和需人工确认的问题。

架构评审模板：

```text
我们正在评审以下架构方案：

[贴方案]

请用 Fusion 做多模型审查，重点输出：
1. 所有模型都认可的共识；
2. 模型之间的主要分歧；
3. 只有部分模型提到但可能重要的点；
4. 独特洞察；
5. 所有模型都遗漏的盲点；
6. 是否建议上线；
7. 如果不建议，必须修改哪些点；
8. 哪些结论需要人工架构委员会确认。
```

合规 / 法律辅助模板：

```text
请对以下条款做风险审查：

[贴条款]

要求：
1. 区分明确风险、潜在风险、需要律师确认的问题；
2. 不要给最终法律意见，只给辅助分析；
3. 标出模型之间的分歧；
4. 给出建议修改方向；
5. 输出必须包含“不能替代专业法律意见”的说明。
```

竞品 / 市场研究模板：

```text
请分析 [公司/产品] 最近 [时间范围] 的变化。

要求：
1. 启用联网查证；
2. 给出来源；
3. 区分事实、推断、假设；
4. 汇总共识；
5. 标出模型之间矛盾；
6. 标出信息不足或需要继续验证的地方；
7. 给出对我们产品路线的建议。
```

### 9.5 上线时必须保存的审计字段

真实业务不要只保存最终答案。至少保存以下字段，便于质量复盘、问题追责和成本优化：

| 字段 | 用途 |
|---|---|
| `status` | 判断是否 `ok` / `judge_fallback` / `error` |
| `panel_ok` / `panel_failed` / `panel_cancelled` | 判断 panel 稳定性和早退行为 |
| `prompt_tokens` / `completion_tokens` | 成本核算 |
| `critical_path_ms` / `panel_latencies_ms` / `judge_ms` / `synthesis_ms` | 延迟 SLA 与瓶颈定位 |
| `analysis.consensus` | 审计模型共识 |
| `analysis.contradictions` | 审计关键分歧 |
| `analysis.partial_coverage` | 识别只被部分模型覆盖的点 |
| `analysis.unique_insights` | 保留单个模型贡献的独特洞察 |
| `analysis.blind_spots` | 触发人工复核或补充调研 |
| `final_answer` | 对用户或业务系统的最终输出 |
| `human_override` | 记录人工是否改写 / 驳回 |
| `downstream_outcome` | 关联后续真实效果，用于评测闭环 |

### 9.6 用评测决定“值不值得”

`--suite fusion --demo` 只能验证机制和报告格式；真实业务必须用自己的历史任务集评测。建议流程：

1. 从真实业务中抽样 50–200 个 case。
2. 为每类任务写 rubric：`factual_accuracy`、`breadth_depth`、`presentation`、`citation`，并加入 negative penalties（危险建议、虚假引用、遗漏关键风险等）。
3. 跑真实评测：

```bash
open-fusion-eval --tasks your_business_tasks.json \
  --preset budget \
  --baseline your/current-single-model \
  --grader llm \
  --grader-model your/judge-model \
  --passes 3 \
  --tools \
  --exclude-domains "your-rubric-host.example.com" \
  --md business_fusion_report.md
```

4. 只在以下条件满足时上线 Fusion：
   - Fusion 相对 baseline 的 lift 明显为正；
   - penalty/error criteria 更少；
   - 成本和延迟可接受；
   - 高风险类别中结果更稳定；
   - 人工复核认为 `analysis` 的分歧和盲点有实际帮助。

### 9.7 推荐上线节奏

| 阶段 | 做法 | 目标 |
|---|---|---|
| 1. 内部研究助手 | 手动运行 CLI，服务架构评审 / 竞品研究 / 方案审查 | 验证 prompt、panel、成本和延迟 |
| 2. 半自动审批助手 | 高风险 case 自动触发 Fusion，输出给人工 reviewer | 降低漏判和单模型偏差 |
| 3. 自动化 + 抽检 | 中风险批量任务用 `budget`，高风险用 `quality + human review` | 控制成本，同时覆盖关键风险 |
| 4. 动态路由 | 根据历史评测自动选择 single / budget / quality / human review | 最大化 ROI |

### 9.8 业务化后续增强建议

如果继续完善项目，优先级建议如下：

1. **Risk router**：按领域、风险、引用需求、业务影响自动选择 single / budget / quality / human review。
2. **Business prompt templates**：内置架构评审、合规审查、供应商选型、竞品研究模板。
3. **Result persistence schema**：标准化保存 `analysis`、telemetry、模型配置、最终答案和人工反馈。
4. **Evaluation dataset builder**：从历史 case 生成 `your_business_tasks.json`。
5. **Policy YAML**：用配置管理不同业务域的 Fusion 策略。
6. **Uncertainty trigger**：单模型回答不确定、引用不足或自相矛盾时自动升级 Fusion。
7. **Human review report**：把 `consensus / contradictions / blind_spots` 渲染成 reviewer 友好的 Markdown 或 UI。

## 10. 调试 / 排错速查

| 现象 | 看哪里 |
|---|---|
| 答案不对 / 不一致 | `--show-analysis` 看裁判的 `consensus` / `contradictions` |
| 慢 | `OPEN_FUSION_LOG=INFO` 看 `panel_max_ms / judge_ms / synthesis_ms` 谁是大头 |
| 部分 panel 失败 | telemetry 里 `panel_failed > 0` + stderr 上的 panel slot error |
| 早退（不是失败） | `panel_cancelled > 0` —— 这是 `fast_majority_k` 的策略选择 |
| 同厂家 panel 警告 | `FusionConfig.validate()` 返回 `WARNING: homogeneous panel` |
| 嵌套调用被拒 | status=error 且 error="fusion depth exceeded" |
| 评测说 not_worth_it | 看 `verdict.recommendations`，可能 cost 不划算或 lift 太小 |

## 11. 不可妥协的设计不变量

这些是正确性约束，不是风格选择。改它们会破坏 Fusion 的核心机制：

1. **合成阶段无 web 工具**。`tools.toolset_for_phase(SYNTHESIS) == ()` 强制保证。
   证据在裁判完成蒸馏后被冻结，最终答案不能引入未经裁判审视的新材料。
2. **per-model attribution**。每条 `contradictions[].stances[].model` 和
   `unique_insights[].model` 必填，由 `Analysis.validate()` 检查。审计链。
3. **优雅降级**。所有 panel 失败 → `status:error`；裁判失败两次 →
   `status:judge_fallback`（从原始 panel 答案重写）；只要 ≥1 个 panel 活着就继续。
4. **bounded recursion**。`MAX_FUSION_DEPTH=1`，嵌套调用被 orchestrator 拒绝；
   守卫既看 `config.depth` 也看 `client.fusion_depth`（双保险）。
5. **异质 panel by default**。同厂家 panel 给警告 (`WARNING: homogeneous panel`)。

## 12. 常用命令速查

```bash
# 一句话调用
open-fusion "你的问题"

# 调试性能
OPEN_FUSION_LOG=INFO open-fusion "你的问题"

# JSON 输出
open-fusion "你的问题" --json

# 评测 demo（零 key）
open-fusion-eval --demo --md /tmp/r.md

# 真实评测
open-fusion-eval --preset budget --baseline anthropic/claude-opus-4.8 \
    --grader llm --grader-model google/gemini-3.1-pro-preview --md report.md
```

## 13. 文件入口速查

| 文档 | 内容 |
|---|---|
| `SKILL.md` | 给 agent 看的 manifest，含触发短语 |
| `INSTALL.md` | 详细安装 / agent 集成 / 联网工具配置 |
| `ARCHITECTURE.md` | 模块图、状态机、不变量 |
| `EVAL.md` | DRACO-style 评测方法 |
| `USAGE.md` | 本文档：使用手册 |
| `DESIGN.html` | 完整设计文档（HTML 富版） |
| `references/fusion-mechanism.md` | 原理级机制说明 |

## 14. 性能优化已落地的项

| 优化 | 行为 |
|---|---|
| **O1** | 3+ panel 的 preset 默认 `fast_majority_k = N-1`（扔最慢一个，保留至少 2 条独立路径） |
| **O4** | DRACO N 次独立判分（默认 n=3）改为 `asyncio.gather` 并行；3×60ms 串行 → ~60ms |
| **O6** | `ModelClient(executor_workers=N)` 可注入自带线程池，避免大 panel 挤占 asyncio 默认池 |
| **H1** | `synthesis_ms` 单独计时；`critical_path_ms = panel_max + judge + synthesis` 三段守恒 |
| **C2** | `panel_cancelled` 独立计量（不计入 panel_failed） |
| **C1** | depth 守卫双保险：config.depth + client.fusion_depth |

测试覆盖：`tests/test_eval.py::test_grader_passes_concurrent` 验证 O4 并发下界
（实测 61ms vs 串行下界 180ms）。

进一步的优化建议（O2/O3/O5/O7/O8）见 review 报告。
