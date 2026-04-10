# 14 LLM 后训练 (Post-Training)

> 📺 [Lecture 14 - LLM Post-Training](https://hanlab.mit.edu/courses/2024-fall-65940)
> 📄 [Slides](https://hanlab.mit.edu/courses/2024-fall-65940)

---

## 14.1 Post-Training Pipeline

预训练后的 LLM 需要 post-training 才能"好用"：

```
Pretrained LLM (只会预测下一个token)
    ↓ SFT (Supervised Fine-Tuning)
Chat Model (能按指令回答)
    ↓ RLHF / DPO (Alignment)
Aligned Model (符合人类偏好，更安全)
```

### 14.1.1 SFT (Supervised Fine-Tuning)

用指令数据微调预训练模型：
```
Input: "解释什么是量化"
Output: "量化是把高精度数值映射到低精度的过程..."
```
- 数据量: 几千到几万条高质量指令数据
- 方法: 全量微调或 LoRA
- 成本: 相对低（几小时到一天）

### 14.1.2 RLHF (Reinforcement Learning from Human Feedback)

```
1. 收集人类偏好数据: 对同一 prompt，人类标注哪个回答更好
2. 训练 Reward Model (RM): 学习人类偏好
3. PPO 强化学习: 用 RM 的奖励信号优化 LLM
```

**PPO**:
- Policy: LLM 本身
- Reward: Reward Model 的评分
- 约束: KL 散度惩罚（不偏离 SFT 模型太远）
$$L_{PPO} = E[r(y)] - \beta \cdot KL[\pi_{LM} \| \pi_{ref}]$$

### 14.1.3 DPO (Direct Preference Optimization)

不需要 Reward Model，直接用偏好数据训练：

$$L_{DPO} = -E\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

- $y_w$: 人类偏好的回答 (winner)
- $y_l$: 人类不偏好的回答 (loser)
- $\sigma$: sigmoid

**DPO vs RLHF**:

| | RLHF | DPO |
|---|------|-----|
| 需要 Reward Model？ | 是 | 否 |
| 训练稳定性 | 较差 (PPO 不稳定) | 较好 |
| 效果上限 | 可能更高 | 略低 |
| 实现复杂度 | 高 | 低 |
| 工业界采用 | OpenAI, Anthropic | Meta, 开源社区 |

---

## 14.2 Parameter-Efficient Fine-Tuning (PEFT)

### 14.2.1 为什么需要 PEFT？

全量微调 LLM 的显存需求：

| 模型 | 参数量 | 全量微调显存 | LoRA 微调显存 |
|------|--------|------------|-------------|
| LLaMA-7B | 7B | ~100GB | ~16GB |
| LLaMA-13B | 13B | ~180GB | ~24GB |
| LLaMA-70B | 70B | ~1TB | ~80GB |

全量微调需要存: 模型参数 + 梯度 + 优化器状态 (Adam 2份) ≈ 4-16x 模型大小。

### 14.2.2 PEFT 方法分类

| 类型 | 方法 | 核心思路 | 可训练参数占比 |
|------|------|---------|-------------|
| Additive | Adapter, Prompt Tuning | 插入新模块 | ~5% |
| Selective | BitFit | 只调 bias | ~0.1% |
| **Reparameterized** | **LoRA** | 低秩分解 | ~0.5% |

### 14.2.3 LoRA (Low-Rank Adaptation)

**核心公式**:

$$W' = W_0 + \Delta W = W_0 + BA$$

- $W_0$: 原始权重 [d × d]，**冻结不训练**
- $B$: [d × r]，可训练
- $A$: [r × d]，可训练
- $r \ll d$: 秩 (如 r=8, d=4096)

**参数量**: $2 \times d \times r$ vs 原始 $d \times d$

| d=4096, r=8 | 参数量 | 占比 |
|-------------|--------|------|
| 原始 W | 16.8M | 100% |
| LoRA B+A | 65.5K | **0.39%** |

**为什么 work？**
- 模型更新 $\Delta W$ 本质上是**低秩的**（Aghajanyan et al., 2020）
- LoRA 直接用低秩矩阵建模这个更新
- 推理时可以把 $BA$ 合并回 $W_0$: $W_{merged} = W_0 + BA$ → **零推理开销**

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """LoRA Linear Layer — 从零实现"""
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.original = nn.Linear(in_features, out_features, bias=True)
        self.original.weight.requires_grad_(False)  # 冻结原始权重
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

        # Kaiming 初始化 A
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 初始化为 0 → 训练开始时 ΔW = 0

    def forward(self, x):
        # 原始路径 + LoRA 路径
        base = self.original(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora

    def merge(self):
        """推理时合并: 零开销"""
        self.original.weight.data += (self.lora_B @ self.lora_A * self.scaling).T
        # 合并后可以删掉 lora_A, lora_B

# 使用示例
layer = LoRALinear(4096, 4096, rank=8)
x = torch.randn(1, 10, 4096)
out = layer(x)
print(f"LoRA 参数: {sum(p.numel() for p in [layer.lora_A, layer.lora_B])}")
print(f"原始参数: {layer.original.weight.numel()}")
print(f"LoRA 占比: {sum(p.numel() for p in [layer.lora_A, layer.lora_B]) / layer.original.weight.numel():.2%}")
```

### 14.2.4 QLoRA: 量化 + LoRA

在 4-bit 量化模型上做 LoRA 微调：

```
原始 FP16 模型 → NF4 量化 (4-bit) → 加 LoRA adapter (FP16) → 微调 LoRA
```

- NF4 (NormalFloat4): 专门为正态分布权重设计的 4-bit 格式
- 双重量化: 量化 scale 本身也量化（从 FP32 → FP8）
- 分页优化器: 用 CPU 内存缓解 GPU 压力

**效果**: 单张 A100 (80GB) 可以微调 65B 模型。

```python
# QLoRA 使用 (bitsandbytes)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # 双重量化
)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b", quantization_config=bnb_config)
# 然后挂 LoRA adapter 微调
```

### 14.2.5 BitDelta

量化微调前后的**差异**到 1 bit:
- $\Delta W = W_{finetuned} - W_{pretrained}$
- $\Delta W \approx \alpha \cdot \text{sign}(\Delta W)$
- 只存 1-bit mask + 一个 scale → 极小存储

---

## 14.3 Prompt Engineering

### 14.3.1 In-Context Learning

| 方法 | 描述 | 示例 |
|------|------|------|
| Zero-shot | 不给示例 | "翻译: Hello" |
| Few-shot | 给几个示例 | "Hi→你好, Bye→再见, Hello→?" |
| CoT (Chain-of-Thought) | 让模型一步步推理 | "让我们一步一步思考..." |

### 14.3.2 RAG (Retrieval Augmented Generation)

```
用户提问 → 检索相关文档 → [文档 + 问题] 一起输入 LLM → 生成回答
```

- 解决: LLM 知识过时、幻觉问题
- 不需要重新训练模型
- 向量数据库: FAISS, Milvus, Pinecone

---

## Infra 实战映射

### vLLM
- vLLM 本身不做训练，但可以部署 LoRA adapter
- vLLM 支持多 LoRA: 不同请求可以用不同的 LoRA adapter
- 关键: 合并后的 LoRA 不增加推理开销

### TensorRT-LLM
- 支持 INT4/FP8 的 LoRA 推理
- LoRA kernel fusion: 把 LoRA 计算融合到 GEMM 中

### 沐曦 MACA
- LoRA 在 MACA 上不需要特殊硬件支持（只是额外矩阵乘法）
- QLoRA 的 NF4 量化需要软件实现
- 多 LoRA 部署: 可以在 MACA 上实现动态 adapter 切换

---

## 跨 Lecture 关联

- **前置 ←** [Lec05: 量化](../lec05-quantization-I/README.md) — QLoRA 的量化部分
- **前置 ←** [Lec06: QAT](../lec06-quantization-II/README.md) — 量化感知训练
- **前置 ←** [Lec12: Transformer](../lec12-transformer/README.md) — LLaMA 架构
- **延伸 →** [Lec13: LLM 部署](../lec13-llm-deploy/README.md) — 量化模型推理

---

## 面试高频题

**Q1: LoRA 为什么不损失性能？**
> A: 因为模型更新的本质是低秩的（Aghajanyan 2020 证明了预训练模型的 intrinsic dimension 很低）。LoRA 直接用低秩矩阵建模更新，参数少但能捕获主要变化。r=8 已经能覆盖大部分更新。

**Q2: LoRA 推理有额外开销吗？**
> A: 训练时有两个矩阵乘法（B·A），但推理前可以合并: $W_{merged} = W_0 + BA$，合并后和原始模型完全一样，零额外开销。

**Q3: QLoRA 怎么在单卡微调 65B？**
> A: 模型本身 NF4 量化到 ~35GB (65B×4bit/8)。LoRA adapter 是 FP16 但参数极少 (~0.5%)。加上分页优化器（CPU offload），总显存需求约 48GB，单张 A100-80GB 够用。

**Q4: DPO vs RLHF，工业上怎么选？**
> A: 小团队 / 开源项目用 DPO（简单稳定）。大厂有资源做 RLHF（上限更高，但需要训 Reward Model + 调 PPO）。Meta LLaMA-2 用 RLHF，很多开源项目用 DPO。
