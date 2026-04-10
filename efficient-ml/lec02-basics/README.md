# Lec02 深度学习基础与效率指标

> 📺 [课程视频](https://www.youtube.com/watch?v=I0nKjPpZmMU&feature=youtu.be) &nbsp;|&nbsp; 📄 [Slides](https://www.dropbox.com/scl/fi/pxvvqyq2yu6mwgk79bq5x/Lec02-Basics.pdf?rlkey=tsumfkhrglic55jnjs4yu66ni&e=1&st=cmwnvuvn&dl=0)

---

## 目录

- [2.1 模型参数量](#21-模型参数量)
- [2.2 FLOPs（浮点运算次数）](#22-flops)
- [2.3 内存占用](#23-内存占用)
- [2.4 数值精度格式](#24-数值精度格式)
- [2.5 效率指标体系](#25-效率指标体系)
- [数学推导](#数学推导)
- [代码：从手写到工业实践](#代码从手写到工业实践)
- [Infra 实战映射](#infra-实战映射)
- [跨 Lecture 关联](#跨-lecture-关联)
- [面试高频题](#面试高频题)

---

## 2.1 模型参数量

参数量是模型的"体重"。一个 7B 模型，就是有 70 亿个浮点数要存、要搬、要算。搞效率优化，第一步永远是搞清楚参数在哪里、有多少。

### Linear 层

全连接层本质就是矩阵乘加偏置：y = xW^T + b。

```
params = C_in × C_out + C_out (bias)
```

`Linear(768, 3072)` → 768 × 3072 + 3072 = **2,362,368**。

这个数字在 Transformer 里会反复出现——FFN 的 up projection 就是这个规模。

### Conv2d 层

```
params = C_out × C_in × K_H × K_W + C_out (bias)
```

`Conv2d(64, 128, kernel_size=3)` → 128 × 64 × 3 × 3 + 128 = **73,856**。

卷积参数量跟输入图像多大没关系。不管输入是 224×224 还是 1024×1024，同一个卷积层参数量一样——因为同一组 kernel 在所有空间位置滑动复用（weight sharing）。这个性质对理解后面 FLOPs 和参数量的"脱钩"很关键。

### Transformer 中的 Multi-Head Attention

MHA 里有四个线性投影：Q、K、V、O。

| 投影 | 形状 |
|------|------|
| W_Q, W_K, W_V | d_model × d_model |
| W_O | d_model × d_model |

忽略 bias 时总参数量：**4 × d_model²**。

GPT-2 小模型（d_model = 768）单层 attention 参数 = 4 × 768² ≈ 2.36M。但别忘了还有 FFN（通常比 attention 参数更多），后面会拆解。

---

## 2.2 FLOPs

FLOPs = Floating Point Operations，是硬件无关的计算量度量。先把几个容易混淆的术语理清：

| 缩写 | 含义 | 举例 |
|------|------|------|
| **FLOPs** | 运算总次数（大写 O，复数 s） | "这个模型 forward 一次要 4.1 GFLOPs" |
| **FLOPS** | 每秒运算次数（吞吐） | "A100 FP16 峰值 312 TFLOPS" |
| **MACs** | Multiply-Accumulate | 1 MAC = 1 次乘法 + 1 次加法 = 2 FLOPs |

很多论文和工具（比如 `thop`、`fvcore`）报告的是 MACs，跟 FLOPs 差 2 倍，读数据时注意看清楚单位。

### Linear 层

```
FLOPs = 2 × B × C_in × C_out
```

系数 2 就来自 MAC：一次乘、一次累加。

### Conv2d 层

```
FLOPs = 2 × B × C_in × C_out × K_H × K_W × H_out × W_out
```

跟参数量相比，多了 H_out × W_out 这一项。这就是参数量和 FLOPs 能"脱钩"的原因：3×3 卷积参数可能只有几十个，但在 1000×1000 的 feature map 上做一次 forward，FLOPs 是几百兆。

### Self-Attention

设序列长度 N，模型维度 d：

```
FLOPs_attn ≈ 4·N·d²  +  2·N²·d
              ─────      ─────
              QKV+O投影    注意力矩阵(QK^T + score×V)
```

两项分别是 O(Nd²) 和 O(N²d)。谁占主导取决于 N 和 d 的相对大小：

- N = 512, d = 768 → 第一项主导（短序列）
- N = 32768, d = 768 → 第二项主导（长序列）

所有长上下文优化工作（FlashAttention、Ring Attention、Sparse Attention）都是在处理 N²d 这一项。

---

## 2.3 内存占用

新手最常犯的错误："7B 模型，FP16 就是 14GB，显卡放得下就能训练"——差远了。

训练时的内存由四个部分组成：

```
Memory_train = 权重 + 梯度 + Optimizer States + Activations
```

| 组成 | FP32 Adam | 说明 |
|------|-----------|------|
| 权重 | 4P bytes | 需要参与前向和反向 |
| 梯度 | 4P bytes | 反向传播产生，与权重等大 |
| Adam m（一阶矩） | 4P bytes | 维护梯度的指数移动平均 |
| Adam v（二阶矩） | 4P bytes | 维护梯度平方的指数移动平均 |
| **仅参数相关合计** | **16P bytes** | 还没算 activations |

7B 模型，纯 FP32 Adam 训练：16 × 7 × 10⁹ = **112 GB**，这还不算 activations。一张 A100 80GB 连参数相关内存都塞不下。

所以实践中必须：
1. 混合精度——权重和梯度存 FP16/BF16，optimizer states 存 FP32
2. ZeRO——把 optimizer states 切分到多张卡
3. Gradient checkpointing——用计算换 activation 内存

### 推理内存与 memory-bandwidth 瓶颈

推理时不需要梯度和 optimizer states，但有另一个问题：**KV cache**。

LLM decode 阶段（自回归生成），每个 token 的生成过程：
1. 把全部模型权重从 HBM 读一遍
2. 做一个矩阵-向量乘法（因为 batch=1 时新 token 只有一个）
3. 更新 KV cache

```
7B FP16 → 14 GB 权重
A100 HBM 带宽 ≈ 2 TB/s
→ 光搬权重的延迟 ≈ 14 / 2000 = 7 ms/token ≈ ~143 tokens/s 上限
```

这就是 decode 阶段是 memory-bound 的根本原因：算力根本没用满，瓶颈在搬数据。

---

## 2.4 数值精度格式

### 浮点数回顾

IEEE 754 浮点数 = 符号位 + 指数 + 尾数：

```
value = (-1)^sign × 2^(exponent - bias) × (1 + mantissa)
```

指数决定"能表示多大/多小的数"（动态范围），尾数决定"相邻两个可表示数之间的间距"（精度）。

### 格式对比

| 格式 | 位宽 | 指数 | 尾数 | 最大值 | 典型用途 |
|------|------|------|------|--------|----------|
| FP32 | 32 | 8 | 23 | ~3.4×10³⁸ | 精确计算、optimizer states |
| FP16 | 16 | 5 | 10 | 65504 | 推理（需 loss scaling） |
| BF16 | 16 | 8 | 7 | ~3.4×10³⁸ | 训练主流格式 |
| INT8 | 8 | — | — | [-128, 127] | 量化推理 |
| INT4 | 4 | — | — | [-8, 7] | 激进量化 |

### FP16 vs BF16：为什么 LLM 训练选 BF16

FP16 的问题在 5 bit 指数：最大值只有 65504。训练 LLM 时 attention score（尤其是 pre-softmax logits）或 loss 值稍微大一点就溢出成 inf，整个训练崩掉。所以用 FP16 训练必须搭配 loss scaling：先把 loss 乘一个大数（比如 1024），让梯度的数值范围抬上来避免 underflow，反向传播之后再除回去。这套流程能 work，但增加了工程复杂度，而且 scale factor 选不好还会出问题。

BF16 的 8 bit 指数跟 FP32 完全一样，动态范围不受限，根本不需要 loss scaling。代价是尾数只有 7 bit（FP16 有 10 bit），精度更粗糙。但实践证明对训练收敛几乎没影响。

还有个工程上的好处：FP32 转 BF16 只要截掉低 16 位就行，不需要任何舍入逻辑。

现在 A100/H100 上几乎所有大模型训练都用 BF16。

### 整数格式的能效优势

量化的核心收益不只是省内存，还有**能耗**。整数运算电路比浮点简单得多。

Horowitz 2014 (45nm) 的经典数据：

| 运算 | 能耗 (pJ) |
|------|----------|
| FP32 MUL | 4.6 |
| FP32 ADD | 0.9 |
| INT8 MUL | 0.2 |
| INT8 ADD | 0.03 |

INT8 矩阵乘相比 FP32 大约省 **18–20× 能耗**。在移动端和边缘设备上，这不是"nice to have"而是"must have"。

---

## 2.5 效率指标体系

### Latency

LLM 场景有特殊的延迟指标拆分：

| 指标 | 含义 | 主要影响因素 |
|------|------|-------------|
| TTFT | Time To First Token，首 token 延迟 | prompt 长度 → prefill 是 compute-bound |
| TPOT | Time Per Output Token | 模型大小 × 带宽 → decode 是 memory-bound |
| P99 | 第 99 百分位延迟 | 线上 SLA 几乎都看 P99，不看平均 |

TTFT 和 TPOT 由不同因素主导，优化手段也不同。这是 vLLM 做 continuous batching 时需要平衡的核心 trade-off。

### Throughput

```
Throughput = batch_size / latency
```

加大 batch 能提吞吐（GPU 利用率上去了），但单条请求的延迟会变长。生产环境的核心问题就是在 latency SLA 约束下最大化 throughput。

### Arithmetic Intensity 与 Roofline 模型

```
Arithmetic Intensity (AI) = FLOPs / Bytes_accessed
```

AI 这个指标告诉你一个 kernel 每搬一个 byte 做多少次计算。

| 场景 | AI | 瓶颈 |
|------|-----|------|
| LLM decode (batch=1) | ~1（每读 2B 做 2 FLOPs） | Memory-bound |
| 大 batch matmul / prefill | 很高 | Compute-bound |
| Softmax / LayerNorm | 极低（element-wise） | Memory-bound |

**Roofline 模型**把这件事画成图：

```
实际性能 = min(峰值算力, 峰值带宽 × AI)
```

横轴是 AI，纵轴是实际 FLOPS。AI 低于某个拐点（= 峰值算力 / 峰值带宽）时，性能被带宽限住；高于拐点才能被算力限住。

A100 的拐点大约在 AI ≈ 156（312 TFLOPS / 2 TB/s）。也就是说一个 kernel 的 AI 低于 156 时，你再怎么优化计算也没用，瓶颈在搬数据。

### MBU（Memory Bandwidth Utilization）

```
MBU = 实际带宽利用 / 硬件峰值带宽
```

Decode 阶段是纯 memory-bound，MBU 直接反映实现效率。好的 LLM serving engine 在 decode 阶段 MBU 能做到 70-80%+。

---

## 数学推导

### ResNet-50 第一层

`Conv2d(3, 64, kernel_size=7, stride=2, padding=3)`，输入 224×224。

参数量：

```
P = 64 × 3 × 7 × 7 = 9,408（无 bias）
```

输出分辨率：

```
H_out = floor((224 + 2×3 - 7) / 2 + 1) = 112
```

FLOPs：

```
FLOPs = 2 × 64 × 3 × 7 × 7 × 112 × 112 ≈ 236 MFLOPs
```

一个只有 9408 个参数的层，FLOPs 就有 236M。这就是 Conv 的特点：参数少、计算重（因为 weight sharing 在空间维度上重复计算）。

### GPT-2 (117M) 单层参数拆解

d_model = 768，d_ff = 3072（= 4 × d_model）。

| 模块 | 参数量 |
|------|--------|
| Attention: 4 × (768² + 768) | 2,362,368 |
| FFN: 768×3072 + 3072 + 3072×768 + 768 | 4,722,432 |
| 2 × LayerNorm: 2 × (768 + 768) | 3,072 |
| **单层合计** | **~7.09M** |

12 层 → ~85M。加上 token embedding（50257×768 ≈ 38.6M）就是 ~124M。实际公开的 GPT-2 small 约 117M，因为它的 token embedding 和 output head 权重共享（weight tying），少算了一份。

一个值得注意的比例：FFN 参数 / Attention 参数 ≈ 2:1。所以 Transformer 里参数大头在 FFN，不在 attention。

### 混合精度训练内存（1B 模型）

| 组件 | 精度 | 内存 |
|------|------|------|
| 前向/反向用的参数 | BF16 | 2 GB |
| 梯度 | BF16 | 2 GB |
| Master weights（optimizer 维护） | FP32 | 4 GB |
| Adam m | FP32 | 4 GB |
| Adam v | FP32 | 4 GB |
| **合计（不含 activations）** | | **16 GB** |

跟纯 FP32 的 16 GB 一样？是的——混合精度的主要收益在于 **计算速度**（BF16 matmul 在 Tensor Core 上快 2-4 倍）和 **activation 内存**（中间结果存 BF16），而不是 optimizer states 的内存。Optimizer states 始终是 FP32 的大头。要砍 optimizer states 内存，得靠 ZeRO 或 FSDP 分片。

---

## 代码：从手写到工业实践

### 基础工具函数

这几个函数是日常 debug 和 profiling 的起点：

```python
import torch
import torch.nn as nn
from typing import Tuple


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """统计可训练 / 总参数量。任何模型丢进来就能用。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def param_memory_mb(model: nn.Module, dtype_bytes: int = 2) -> float:
    """参数内存估算 (MB)。dtype_bytes: FP32=4, BF16/FP16=2, INT8=1"""
    total = sum(p.numel() for p in model.parameters())
    return total * dtype_bytes / (1024 ** 2)
```

### HuggingFace Transformers 里的实际写法

在 HuggingFace transformers 仓库里，参数量统计不是简单的 `sum(p.numel())`。看 `modeling_utils.py` 里 `PreTrainedModel` 的实现：

```python
# transformers/modeling_utils.py 中 num_parameters() 的核心逻辑
# 简化版，展示关键设计决策

def num_parameters(self, only_trainable: bool = False, 
                   exclude_embeddings: bool = False) -> int:
    """
    HF 比朴素版本多考虑两件事：
    1. exclude_embeddings —— 很多论文报告参数量时不算 embedding，
       因为 embedding 不参与主要计算（只是查表），而且如果有 weight tying
       会导致重复计算。
    2. 去重 —— 用 data_ptr() 检测共享权重，避免重复统计。
       典型场景：GPT-2 的 token embedding 和 lm_head 共享权重。
    """
    if exclude_embeddings:
        embedding_params = sum(
            p.numel() for p in self.get_input_embeddings().parameters()
        )
    
    # 关键：用 set(data_ptr) 去重共享参数
    seen = set()
    total = 0
    for name, p in self.named_parameters():
        if only_trainable and not p.requires_grad:
            continue
        ptr = p.data_ptr()
        if ptr in seen:
            continue  # 跳过共享权重
        seen.add(ptr)
        total += p.numel()
    
    if exclude_embeddings:
        total -= embedding_params
    return total
```

这段代码揭示了两个实战中容易踩的坑：
1. **Weight tying**：GPT-2、LLaMA 等模型的 input embedding 和 output head 共享权重。朴素的 `sum(p.numel())` 会重复计算。
2. **Embedding 的特殊性**：Embedding 层参数量可以很大（vocab_size × d_model），但它的"计算"只是查表（gather），FLOPs 贡献微乎其微。

### FLOPs 估算：手写 vs 工具

手写：

```python
def flops_linear(B: int, in_f: int, out_f: int) -> int:
    """FLOPs = 2 × B × in × out（MAC 的 2）"""
    return 2 * B * in_f * out_f


def flops_attention(B: int, N: int, d: int) -> int:
    """Self-attention FLOPs 估算（单层）
    
    4Nd² 来自 Q/K/V/O 四个 linear projection（每个是 2×N×d²，共 4 个）
    2N²d 来自 QK^T（N×N 矩阵乘，2N²d）和 attn×V（同量级）
    """
    proj = 4 * N * d * d     # QKV + O projection: 4 × (2×N×d×d) / 2
    attn = 2 * N * N * d     # QK^T + score @ V
    return B * 2 * (proj // (2) + attn // (2))
    # 更直接的写法：
    # return B * (4 * 2 * N * d * d + 2 * 2 * N * N * d)
```

实际项目中常用 `calflops` 或 Meta 的 `fvcore` 来自动算：

```python
# 用 calflops（pip install calflops）
# 这是 LLM 社区比较常用的 FLOPs profiler
from calflops import calculate_flops

flops, macs, params = calculate_flops(
    model=model,
    input_shape=(1, 128),      # (batch, seq_len)
    transformer_tokenizer=tokenizer,
)
print(f"FLOPs: {flops}  MACs: {macs}  Params: {params}")
```

但注意：**自动工具的数字不一定对**。特别是涉及动态形状（attention mask、变长序列）时，自动 profiler 经常给出误导性的数字。手动推导公式能力是基本功，不能完全依赖工具。

### 数值精度：实际观察

```python
import torch

# BF16 vs FP16 精度差异
x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
print(f"FP16 最大误差: {(x - x.half().float()).abs().max():.2e}")
print(f"BF16 最大误差: {(x - x.bfloat16().float()).abs().max():.2e}")
# FP16 精度更高（10 bit 尾数），BF16 更粗糙（7 bit 尾数）

# FP16 溢出 —— 这就是训练炸掉的典型原因
big = torch.tensor(65505.0)
print(f"65505 → FP16: {big.half()}")     # inf！
print(f"65505 → BF16: {big.bfloat16()}")  # 65536.0，安全

# 更贴近实际的场景：大 attention score
# pre-softmax logits 在训练初期可能很大
scores = torch.randn(1, 12, 2048, 2048) * 30  # 模拟未归一化的 attention
print(f"FP16 overflow count: {torch.isinf(scores.half()).sum()}")
print(f"BF16 overflow count: {torch.isinf(scores.bfloat16()).sum()}")
```

### Transformer Block 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """简化版 Transformer Block，展示参数分布和计算流程。
    
    结构对标 GPT-2 / LLaMA 的基本 pattern：
    - Pre-norm（LayerNorm 在 attention / FFN 之前）
    - GELU activation
    - 无 bias（现代 LLM 通常 bias=False 省参数）
    """
    
    def __init__(self, d_model: int = 256, n_heads: int = 4, d_ff: int = 1024):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        
        # Attention projections — 合并成一个 Linear 更高效
        # 很多开源实现（LLaMA、Mistral）都把 QKV 合并
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # FFN — LLaMA 用 SwiGLU，这里简化为 GELU
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        
        # Pre-norm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        
        # --- Self-Attention ---
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, d_head)
        q, k, v = qkv.unbind(0)
        
        # Scaled dot-product attention
        # PyTorch 2.0+ 提供了 F.scaled_dot_product_attention，
        # 会自动 dispatch 到 FlashAttention kernel
        attn_out = F.scaled_dot_product_attention(q, k, v)
        
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        x = x + self.o_proj(attn_out)
        
        # --- FFN ---
        h = self.ln2(x)
        x = x + self.down(F.gelu(self.up(h)))
        
        return x


if __name__ == "__main__":
    block = TransformerBlock(d_model=256, n_heads=4, d_ff=1024)
    trainable, total = count_parameters(block)
    print(f"Transformer Block 参数量: {total:,}")
    
    # 逐模块拆解
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            n = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {n:,}")
    
    # 不同精度内存
    for label, nbytes in [("FP32", 4), ("BF16", 2), ("INT8", 1)]:
        print(f"{label} 参数内存: {total * nbytes / 1024**2:.2f} MB")
    
    # FLOPs 手动估算 (batch=32, seq_len=128)
    B, N, D, D_FF = 32, 128, 256, 1024
    qkv_flops = 2 * B * N * D * (3 * D)     # QKV projection
    o_flops = 2 * B * N * D * D              # O projection
    attn_flops = 2 * B * N * N * D * 2       # QK^T + score@V（两次 N×N×D）
    ffn_flops = 2 * B * N * D * D_FF * 2     # up + down
    total_flops = qkv_flops + o_flops + attn_flops + ffn_flops
    print(f"\nForward FLOPs (B={B}, N={N}): {total_flops / 1e9:.2f} GFLOPs")
```

### vLLM 中的内存估算逻辑

vLLM 在初始化时需要估算模型占多少显存、剩余显存能放多少 KV cache blocks。这段逻辑在 `worker/worker.py` 中：

```python
# vLLM 内存估算的简化版核心逻辑
# 实际代码在 vllm/worker/worker.py 的 determine_num_available_blocks()

def estimate_kv_cache_memory(
    num_layers: int,
    num_kv_heads: int,  # GQA 时 kv_heads < q_heads
    head_dim: int,
    block_size: int,     # 每个 block 存多少个 token，默认 16
    dtype_bytes: int,    # BF16=2, FP8=1
) -> int:
    """单个 KV cache block 的内存 (bytes)。
    
    每个 block 要存：
    - 每层的 K 和 V（所以 ×2）
    - 每个 head 的 head_dim 个数值
    - block_size 个 token
    """
    return (2 * num_layers * num_kv_heads * head_dim 
            * block_size * dtype_bytes)

# 举例：LLaMA-2 7B
# 32 layers, 32 kv_heads, head_dim=128, block_size=16, BF16
block_mem = estimate_kv_cache_memory(32, 32, 128, 16, 2)
print(f"单个 KV block 内存: {block_mem / 1024:.1f} KB")  # 4096 KB = 4 MB

# A100 80GB，模型本身 ~14GB (7B × 2B)
# 剩余 ~66GB 给 KV cache
# 能放 66 * 1024 / 4 ≈ 16896 个 blocks
# 每个 block 16 tokens → 最多缓存 ~270K tokens
# 对 batch_size=256, seq_len=1024 来说是够的
```

这里体现了一个重要的工程 pattern：**先做粗估算，再实际 profile**。vLLM 的做法是先按公式估一个上界，然后用一个 dummy forward pass 来测量实际占用，取两者的保守值。

### PyTorch 混合精度训练 Pattern

```python
# 标准混合精度训练写法（PyTorch native）
# 来自 PyTorch 官方文档和各开源项目的通用 pattern

from torch.amp import autocast, GradScaler

model = model.cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()  # 仅 FP16 需要，BF16 不需要

for batch in dataloader:
    optimizer.zero_grad()
    
    # autocast 区域内，matmul 等算子自动用低精度执行
    # 但 softmax、layernorm、loss 会保持 FP32
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(batch)
    
    # BF16 时可以直接 loss.backward()，不需要 scaler
    # FP16 时必须 scaler.scale(loss).backward()
    loss.backward()
    
    # 梯度裁剪 —— 大模型训练必备
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
```

为什么 BF16 不需要 GradScaler？因为 BF16 的动态范围跟 FP32 一样，梯度不会 underflow。FP16 的动态范围小，微小的梯度值会 flush 到 0，所以需要 loss scaling 把梯度数值抬上去。

---

## Infra 实战映射

### vLLM

vLLM 的设计处处对应本讲的分析：

**PagedAttention**——OS 虚拟内存的思想用到 KV cache 上。传统实现为每个序列预分配一块连续内存（按 max_seq_len），实际用不满就浪费了。PagedAttention 把 KV cache 切成固定大小的 block（默认 16 tokens/block），按需分配，不连续也没关系。这把 KV cache 的内存利用率从 ~60% 提到 ~95%+。

**Continuous Batching**——不等一个 batch 里所有序列都生成完才开始下一个 batch，而是序列完成就立刻填入新请求。本质是在 compute-bound（prefill）和 memory-bound（decode）之间动态调度。

### TensorRT-LLM

NVIDIA 的编译器从 FLOPs 和 memory access 分析出发做优化：

**Layer Fusion**——把 element-wise ops（bias add、activation、residual add）融合进前面的 matmul kernel，减少中间结果的 HBM 读写。一个被 fuse 掉的 op 意味着少一次 HBM round trip，在 memory-bound 的场景下收益很大。

**GEMM Plugin**——根据矩阵的具体尺寸选 tiling 策略，让数据搬运和计算 overlap。不同 shape 的最优策略不同，所以 TRT-LLM 会在构建阶段做 auto-tuning。

---

## 跨 Lecture 关联

| 方向 | Lecture | 关联点 |
|------|---------|--------|
| ← 前置 | Lec01 | 效率优化的动机与全局视角 |
| → | Lec03/04 | 剪枝——直接减参数量和 FLOPs，本讲公式是衡量标准 |
| → | Lec05/06 | 量化——降低每个参数的 bit 数，本讲的精度格式分析是基础 |
| → | Lec11 | Tiny Engine——MCU 上的极限内存约束，本讲的内存分析方法论直接沿用 |
| → | Lec12/13 | Transformer/LLM 部署——本讲所有公式的大规模实际应用 |

---

## 面试高频题

**Q1：Linear(1024, 4096) 有多少参数？FP16 推理占多少内存？**

参数量 = 1024 × 4096 + 4096(bias) = 4,198,400 ≈ 4.2M。FP16 每参数 2 bytes → 4.2M × 2 = **8.4 MB**。

注意：很多现代模型 bias=False（LLaMA、Mistral 都是），这时参数量 = 4,194,304，内存 ≈ 8.0 MB。面试时可以主动提这一点，展示你知道当前的工程实践。

---

**Q2：FLOPs 和参数量能脱钩吗？举例。**

典型例子：`Conv2d(3, 3, 3)` 只有 81 个参数，但作用在 1000×1000 图像上 FLOPs ≈ 162M。原因就是 weight sharing——同一组参数在不同空间位置反复使用。反过来，Embedding 层参数量巨大（vocab_size × d_model），但 FLOPs 几乎为零（只是查表）。

---

**Q3：BF16 vs FP16，为什么 LLM 训练选 BF16？**

两个原因：
1. 动态范围——BF16 指数 8 bit（同 FP32），最大值 ~3.4×10³⁸。FP16 指数 5 bit，最大值 65504。训练中 attention score 或 loss 超过 65504 就溢出为 inf。BF16 不需要 loss scaling，FP16 必须。
2. 转换简单——FP32 到 BF16 截断低 16 位即可，硬件实现几乎零开销。

代价是 BF16 尾数只有 7 bit（FP16 有 10 bit），精度更低，但实际训练不敏感。

---

**Q4：LLM decode 为什么是 memory-bound？怎么缓解？**

Decode 时每个 step 只生成一个 token（或很少几个），矩阵乘退化为矩阵-向量乘。每个参数读 2 bytes（FP16），做 2 FLOPs，arithmetic intensity ≈ 1，远低于 GPU 的 compute/bandwidth 比（A100 约 156）。算力大量空转。

缓解方法：
- Continuous batching——攒多个请求一起 decode，提高 batch size 从而提高 AI
- 模型量化——INT8/INT4 减少搬运量
- Speculative decoding——用小模型快速猜多个 token，大模型一次性验证

---

**Q5：推理内存 = 参数 × dtype_bytes 吗？少了什么？**

少了 **KV cache**。每生成一个新 token，前面所有 token 的 K、V 向量都要保留（否则每步重算，延迟不可接受）。KV cache 大小：

```
KV_cache = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch_size × dtype_bytes
```

长序列、大 batch 下 KV cache 可以超过模型权重本身。LLaMA-2 7B 处理 batch=128, seq=4096 的 KV cache 约 64 GB，远超 14 GB 的模型权重。

这就是 PagedAttention、GQA（Grouped Query Attention）、MQA（Multi Query Attention）等技术的出发点——压缩 KV cache。
