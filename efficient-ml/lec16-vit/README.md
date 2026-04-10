# 16 Vision Transformer (ViT)

> 📺 [Lecture 16 - Vision Transformer](https://hanlab.mit.edu/courses/2024-fall-65940)
> 📄 [Slides](https://hanlab.mit.edu/courses/2024-fall-65940)

---

## 16.1 ViT 核心

### 16.1.1 从图像到 Token

Transformer 原本处理 1D token 序列。ViT 把 2D 图像转换成 token：

$$\text{Patch}: x_p \in \mathbb{R}^{P^2 \cdot C} \xrightarrow{E} z_0 \in \mathbb{R}^D$$

- 图像 $x \in \mathbb{R}^{H \times W \times C}$ 切成 $N = HW/P^2$ 个 patch
- 每个 patch (如 16×16×3) 通过线性投影变成 D 维向量
- 加上 position embedding → 变成 token 序列

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """ViT Patch Embedding — 把图像切成 patch 并投影"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        # 用 Conv2d 实现分块 + 投影 (一步到位)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        # 加 CLS token + position embedding
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, N+1, embed_dim]
        x = x + self.pos_embed
        return x

# 224×224 图像, patch=16 → 14×14 = 196 个 patch
embed = PatchEmbedding(img_size=224, patch_size=16)
x = torch.randn(1, 3, 224, 224)
out = embed(x)
print(f"输出: {out.shape}")  # [1, 197, 768]  (196 patches + 1 CLS)
```

### 16.1.2 ViT vs CNN

| | CNN | ViT |
|---|-----|-----|
| 感受野 | 局部 (受限于 kernel) | 全局 (self-attention) |
| 归纳偏置 | 强 (平移不变性、局部性) | 弱 (需要更多数据学习) |
| 计算复杂度 | O(HW × K²) | O(N² × D) |
| 数据需求 | 少 (inductive bias 帮助) | 多 (需要大数据集预训练) |
| 扩展性 | 架构设计限制 | 增大模型就行 |

> **关键洞察**: ViT 的成功依赖于大规模预训练。在小数据集上 CNN 通常更好（inductive bias 帮助），但数据够大时 ViT 的上限更高。

---

## 16.2 高效视觉 Transformer

### 16.2.1 Swin Transformer: Window Attention

**问题**: ViT 的全局 attention 计算量 O(N²)，高分辨率图像 N 很大。

**方案**: 只在局部窗口内做 attention:
- 窗口大小固定 (如 7×7 = 49 tokens)
- 每个窗口内计算量 = O(49²) → 与图像大小无关!
- **Shifted Window**: 交替移动窗口位置，让相邻窗口信息交流

```
Window Partition (regular):
┌─────┬─────┐
│  W1 │  W2 │
├─────┼─────┤
│  W3 │  W4 │
└─────┴─────┘

Shifted Window:
  ┌───┬───┬───┐
  │   │   │   │
──┼───┼───┼───┤
  │   │   │   │
──┼───┼───┼───┤
  │   │   │   │
  └───┴───┴───┘
→ 窗口边界不同 → 信息跨窗口流动
```

### 16.2.2 EfficientViT: ReLU Linear Attention

把 softmax attention 替换为线性注意力:

$$\text{LinearAttn}(Q, K, V) = \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q)(\phi(K)^T \mathbf{1})}$$

其中 $\phi$ = ReLU（替代 softmax）

- softmax 是使计算变成 O(n²) 的罪魁祸首
- 用 ReLU 后: $\phi(K)^T V$ 可以预计算 → **O(n)** 复杂度!
- 代价: 没有 softmax 的归一化，精度可能下降

### 16.2.3 SparseViT

动态稀疏 attention:
- 不是每个 patch 都参与 attention
- 根据输入重要性动态选择参与 attention 的 token
- 类似 DuoAttention (Lec15) 的思路

---

## 16.3 多模态应用

### 16.3.1 CLIP: 对比学习

双塔架构: ViT (视觉) + Transformer (文本)

$$\text{Loss} = -\frac{1}{N}\sum_i \log \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_j \exp(\text{sim}(v_i, t_j)/\tau)}$$

- 同一对图片-文本的 embedding 靠近
- 不同对的 embedding 远离
- 训练后可以做 zero-shot 分类

### 16.3.2 SAM (Segment Anything)

Meta 的通用分割模型:
- ViT-H 编码器: 提取图像特征
- Prompt encoder: 处理点/框/mask 提示
- Mask decoder: 预测分割 mask
- 可以分割任何物体（零样本泛化）

### 16.3.3 EfficientViT-SAM

用 EfficientViT 替换 SAM 的 ViT-H 编码器:
- 速度提升 10-100x
- 精度几乎不变
- 可以在移动端实时运行

### 16.3.4 多模态 LLM

| 模型 | 视觉编码器 | 语言模型 |
|------|-----------|---------|
| Flamingo | ViT | Chinchilla |
| LLaVA | CLIP ViT | LLaMA |
| PaLM-E | ViT | PaLM |

---

## 16.4 AR 图像生成

### VAR (Visual AutoRegressive)

从粗到细的自回归图像生成:
- 先生成低分辨率 token → 再逐步生成高分辨率
- 传统 AR 是从左到右，VAR 是从粗到细
- 比传统 AR 快 10x+

### HART: 高效 AR 图像生成
- VAR 的加速版
- 用更好的 tokenization 减少序列长度

---

## Infra 实战映射

### ViT 推理优化
- **FlashAttention**: 加速 ViT 的 attention 计算（和 LLM 相同）
- **TensorRT**: NVIDIA 提供 ViT 的专门优化
- **Batching**: ViT 推理通常是 compute-bound，batching 效果好

### 多模态 LLM 部署
- 视觉编码器 (ViT) 和语言模型可以分别优化
- ViT 部分可以用 INT8 量化（比 LLM 量化更容易）
- 图像预处理 (patch embedding) 可以离线完成

### 沐曦 MACA
- ViT 的 Conv2d patch embedding 在 MACA 上应该有良好支持
- FlashAttention 需要在 MACA 上实现对应 kernel
- 多模态推理的显存管理: 图像 token + 文本 token 的 KV Cache

---

## 跨 Lecture 关联

- **前置 ←** [Lec12: Transformer](../lec12-transformer/README.md) — Attention, Position Encoding
- **前置 ←** [Lec15: 长上下文](../lec15-long-context/README.md) — 稀疏注意力用于视觉
- **横向 ↔** [Lec05: 量化](../lec05-quantization-I/README.md) — ViT 量化比 LLM 容易
- **横向 ↔** [Lec03: 剪枝](../lec03-pruning-I/README.md) — token pruning 剪掉不重要的 patch

---

## 面试高频题

**Q1: ViT 为什么需要大规模预训练？**
> A: ViT 的 inductive bias 弱（没有 CNN 的平移不变性和局部性假设），需要从数据中学习这些。在小数据集上 CNN 的强归纳偏置有优势，但数据够大时 ViT 的弱归纳偏置反而让模型更灵活、上限更高。

**Q2: Swin Transformer 的 Window Attention 怎么实现跨窗口信息交流？**
> A: Shifted Window — 交替使用两种窗口划分方式（regular 和 shifted），相邻层的窗口边界不同，信息通过层间传播跨窗口流动。计算量保持 O(N)（与图像大小线性关系）。

**Q3: Linear Attention 为什么能 O(n)？**
> A: 核心技巧: $\phi(Q)(\phi(K)^T V)$ 中，先算 $\phi(K)^T V$ (d×d)，再乘 Q (n×d) → O(n×d²)。而标准 attention $QK^T$ (n×n) → O(n²)。代价是 φ 替代 softmax 后没有概率归一化。

**Q4: CLIP 为什么能做 zero-shot 分类？**
> A: 训练时学了"图片-文本对齐"，推理时把类别名变成文本（如 "a photo of [class]"），和图片 embedding 算相似度，最相似的就是预测类别。不需要专门训练分类器。
