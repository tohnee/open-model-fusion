# TODO: 多模态大模型复杂图像处理能力提升研究

> 创建日期: 2026-06-30
> 状态: 待研究
> 优先级: 中

## 背景

当前 Fusion 框架的核心思路（多模型编排、pick-best、verifier、多 judge 投票）在文本推理场景已验证有效（数学题准确率提升 20%+）。
如何将这些架构思路迁移到**多模态大模型 (MLLM)** 处理**复杂图像**的场景？

## 核心问题

复杂图像（医学影像、工程图纸、图表推理、表格提取、数学几何图、化学结构图）存在以下挑战：

1. **单模型视觉盲点**: 不同 MLLM 对不同图像类型有专长，例如 GPT-4V 擅长自然图像但弱于数学图表
2. **OCR/布局错误级联**: 一个模型读错一个数字 → 后续推理全错
3. **图像细节遗漏**: 高分辨率图像中小字/标注容易被忽略
4. **推理链断裂**: 看到了图像但推理链错了（和文本场景的 math_2 问题类似）
5. **幻觉风险**: MLLM 容易"编造"图中不存在的内容

## 可迁移的 Fusion 思路

### 1. Panel 多视角 (Multi-Crop / Multi-Model FAN_OUT)
- **多模型 panel**: 不同 MLLM (GPT-4V, Claude Vision, Gemini, Qwen-VL, InternVL) 并行看图
- **多裁剪 panel**: 同一 MLLM 对同一张图看全图 + 左上/右上/左下/右下 4 个 crop + OCR 文本层
- **多粒度 panel**: 低分辨率全图 (语义理解) + 高分辨率 crop (细节提取) + OCR 纯文本 (数字精确)

### 2. Vision-Judge (视觉裁判)
- Judge 拿到所有 panel 的视觉描述+回答，判断谁看对了
- 关键：judge 本身也需要看图（不是纯文本 judge）
- 可做"视觉一致性检查": 不同 crop 提取的信息是否矛盾

### 3. Verifier 视觉验证器 (P0 优先)
- **OCR 交叉验证**: panel 提取的数字/文字是否一致？不一致 → 标记低置信度
- **几何约束验证**: 数学题中, 答案是否满足几何约束（如角度和为180°）
- **表格一致性**: 表格提取的行列数/数值是否自洽
- **幻觉检测**: 回答中提到的视觉元素是否存在于至少 2 个 panel 的描述中？

### 4. Pick-Best 视觉短路
- judge 识别哪个 model/crop 看得最清楚 → 直接采用该回答
- 触发条件：verifier 通过 + 高置信度
- fallback: 多视角融合 synthesis

### 5. Multi-Judge 视觉投票
- 多个 vision judge 独立评判 panel 回答
- 多数投票决定 best_model
- 降低单 judge 视觉误判概率

### 6. AGGREGATOR 模式 (视觉 MoA)
- 跳过 judge，直接把所有 panel 的视觉描述+回答给 synthesizer
- synthesizer 综合多视角信息生成最终答案
- 适合：图像信息密度高、需要互补的场景

## 关键架构差异（vs 文本 Fusion）

| 维度 | 文本 Fusion | 多模态 Fusion |
|------|-----------|--------------|
| 输入 | 纯文本 | 图像 + 文本 prompt |
| Panel 多样性来源 | 不同模型 | 不同模型 + 不同 crop + 不同分辨率 |
| Judge 输入 | 文本回答 | 图像 + 文本回答 |
| Verifier 类型 | 数值/格式检查 | OCR 一致性 + 几何约束 + 幻觉检测 |
| Token 成本 | 文本 token | 视觉 token (贵 10x) |
| 延迟瓶颈 | API 调用 | 高分辨率图像处理 |

## 实施路线草图

### Phase 1: 多模型 Vision Panel (MVP)
- 支持图片输入的 ModelSpec (image_url field)
- Panel 同时调用 2-3 个 MLLM 看图回答
- AGGREGATOR 模式: synthesizer 综合多模型视觉描述
- 测试集: 数学几何题 + 图表理解题

### Phase 2: Vision Verifier
- OCR 一致性检查 (不同模型提取的文字是否一致)
- 数值范围验证 (答案是否在合理区间)
- 幻觉标记 (被单个模型提及但其他模型未看到的元素)

### Phase 3: Multi-Crop Panel
- 自动裁剪: 全图 + 4 象限 + OCR 文本层
- 同一模型多视角并行
- Panel size = N models × (1 + K crops)

### Phase 4: Vision Judge + Pick-Best
- Vision judge 看图 + 看 panel 回答, 选 best
- Verifier 通过 → 短路返回
- Verifier 否决 → synthesis

## 相关资源
- [ ] 调研各 MLLM 的 API 图像支持 (GPT-4V, Claude 3 Vision, Gemini, Qwen-VL, InternVL)
- [ ] 收集多模态 benchmark 数据集 (MathVista, ChartQA, DocVQA, MMMU)
- [ ] 评估视觉 token 成本模型 (图片分辨率 × 模型单价)
