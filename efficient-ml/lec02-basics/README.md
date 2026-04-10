# Lec02 深度学习基础与效率指标

> 📺 [课程视频](https://www.youtube.com/watch?v=5HpLyZd1h0I) &nbsp;|&nbsp; 📄 [Slides](https://hanlab.mit.edu/courses/2024-fall-65940)

---

## 目录

- [2.1 模型参数量](#21-模型参数量-parameter-count)
- [2.2 FLOPs（浮点运算次数）](#22-flops浮点运算次数)
- [2.3 内存占用](#23-内存占用-memory-footprint)
- [2.4 数值精度格式](#24-数值精度格式)
- [2.5 效率指标体系](#25-效率指标体系)
- [数学推导](#数学推导)
- [代码示例](#代码示例)
- [Infra 实战映射](#infra-实战映射)
- [跨 Lecture 关联](#跨-lecture-关联)
- [面试高频题](#面试高频题)

---

## 2.1 模型参数量 (Parameter Count)

参数量衡量模型的"体量"——需要存储多少个可学习的数值。它直接决定模型文件大小，也是推理内存占用的基础。

### Linear 层

```
params = C_in × C_out + C_out (bias)
```

例：`Linear(768, 3072)` 有 768 × 3072 + 3072 = **2,362,368** 个参数。

### Conv2d 层

```
params = C_out × C_in × K_H × K_W + C_out (bias)
```

例：`Conv2d(64, 128, kernel_size=3)` 有 128 × 64 × 3 × 3 + 128 = **73,856** 个参数。

> **要点**：卷积层参数量与空间分辨率无关，只取决于 channel 数和 kernel 尺寸。这源于权重共享——同一组卷积核在所有空间位置滑动复用。

### Transformer 中的 Multi-Head Attention

| 投影矩阵 | 形状 |
|----------|------|
| W_Q, W_K, W_V | d_model × d_model |
| W_O | d_model × d_model |

总参数量（忽略 bias）：**4 × d_model²**

以 GPT-2 (117M) 为例：d_model = 768，12 层，单层 attention 参数 = 4 × 768² = 2,359,296。

---

## 2.2 FLOPs（浮点运算次数）

FLOPs 衡量计算量，是硬件无关的指标。注意术语区分：

| 缩写 | 含义 | 说明 |
|------|------|------|
| **FLOPs** | Floating Point Operations | 运算总次数 |
| **FLOPS** | FLOPs per Second | 硬件吞吐能力 |
| **MACs** | Multiply-Accumulate Operations | 1 MAC = 2 FLOPs |

### Linear 层

```
FLOPs = 2 × B × C_in × C_out
```

系数 2 来自 multiply-accumulate：一次乘法加一次累加。

### Conv2d 层

```
FLOPs = 2 × B × C_in × C_out × K_H × K_W × H_out × W_out
```

> **与参数量的关键区别**：Conv 的 FLOPs 与输出分辨率正相关。高分辨率特征图的计算代价非常高，这也是下采样策略如此重要的原因。

### Self-Attention

设序列长度 N，模型维度 d：

```
FLOPs_attn ≈ 4·N·d²  +  2·N²·d
              ─────      ─────
              QKV+O投影    注意力矩阵计算
              O(Nd²)      O(N²d)
```

当 N >> d 时（长序列场景），N²d 项占主导，这正是 FlashAttention、Sparse Attention 等长上下文优化工作的出发点。

---

## 2.3 内存占用 (Memory Footprint)

内存 ≠ 参数量。训练时实际占用由四部分构成：

```
Memory_train = 权重 + 梯度 + Activations + Optimizer States
```

| 组成部分 | FP32 Adam 训练 | 推理 |
|----------|---------------|------|
| 权重 | 4P bytes | 4P bytes |
| 梯度 | 4P bytes | — |
| Adam 状态 (m, v) | 8P bytes | — |
| Activations | 与 batch size 正比 | 很小（可重算） |

**估算示例**：7B 参数模型用 FP32 Adam 训练，仅参数相关内存就达到 (4+4+8) × 7B = **112 GB**，还没算 activations。这就是混合精度训练和 gradient checkpointing 的必要性所在。

### 推理时的内存带宽瓶颈

LLM decode 阶段（batch=1）每生成一个 token，都需要把全部权重从 HBM 加载一遍：

```
7B FP16 模型 → 14 GB 数据搬运
A100 HBM 带宽 ≈ 2 TB/s
→ 单 token 延迟 ≈ 14 GB / 2 TB/s = 7 ms
```

---

## 2.4 数值精度格式

### 格式对比

| 格式 | 位宽 | 指数位 | 尾数位 | 最大值 | 精度 | 溢出风险 |
|------|------|--------|--------|--------|------|----------|
| FP32 | 32 | 8 | 23 | ~3.4×10³⁸ | ~7 位十进制 | 低 |
| FP16 | 16 | 5 | 10 | 65504 | ~3-4 位 | **高** |
| BF16 | 16 | 8 | 7 | ~3.4×10³⁸ | ~2-3 位 | 低 |
| INT8 | 8 | — | — | [-128, 127] | 整数 | — |
| INT4 | 4 | — | — | [-8, 7] | 整数 | — |

### FP32

```
v = (-1)^s × 2^(e-127) × (1 + mantissa)
```

1 bit 符号 + 8 bit 指数（偏置 127）+ 23 bit 尾数。

### FP16 vs BF16

FP16 的 5 bit 指数带来的有限动态范围（最大 65504）是实际训练中的大问题——attention score 或 loss 值稍大就会溢出，必须搭配 loss scaling。

BF16 保留了 FP32 的 8 bit 指数，动态范围完全一致，代价是尾数精度更低（7 bit vs FP16 的 10 bit）。实践中这个精度损失对训练收敛影响很小。另一个工程优势：FP32 → BF16 转换只需截断低 16 位，几乎零开销。

> **现状**：主流 LLM 训练（A100/H100）几乎全部使用 BF16。FP16 需要 loss scaling 防溢出，BF16 不需要。

### 整数格式与能效

INT8/INT4 没有指数和尾数的复杂处理，电路简单，能效显著优于浮点。根据 Horowitz 2014 在 45nm 工艺下的估算：

| 运算 | 能耗 (pJ) | 相对 FP32 MUL |
|------|----------|---------------|
| FP32 MUL | 4.6 | 1× |
| FP32 ADD | 0.9 | 0.2× |
| INT32 ADD | 0.1 | 0.02× |
| INT8 ADD | 0.03 | 0.007× |

INT8 矩阵乘法相比 FP32 可节省约 **18–20×** 能耗——这是量化技术的根本动机。

---

## 2.5 效率指标体系

### Latency（延迟）

端到端推理时间。LLM 场景下的关键细分：

| 指标 | 含义 | 影响因素 |
|------|------|----------|
| **TTFT** | Time To First Token，首 token 延迟 | prompt 长度，prefill 计算量 |
| **TPOT** | Time Per Output Token，逐 token 延迟 | 模型大小，内存带宽 |
| **P99** | 第 99 百分位延迟 | SLA 通常基于此指标 |

### Throughput（吞吐量）

```
Throughput = batch_size / latency
```

Latency 和 Throughput 之间存在 trade-off：加大 batch 提高吞吐，但单请求延迟上升。

### Arithmetic Intensity 与 Roofline 模型

```
Arithmetic Intensity (AI) = FLOPs / Bytes of Memory Access
```

| AI 高低 | 瓶颈类型 | 典型场景 |
|---------|---------|---------|
| AI 高 | Compute-bound | 大 batch matmul、prefill |
| AI 低 | Memory-bound | LLM decode (batch=1)、element-wise ops |

**Roofline 模型**：

```
Performance = min(Peak_FLOPS, Peak_BW × AI)
```

AI 低于拐点时受带宽限制，高于拐点时受算力限制。不同硬件的拐点不同，这也是调优的出发点。

### Memory Bandwidth Utilization (MBU)

```
MBU = 实际内存带宽使用 / 硬件峰值内存带宽
```

decode 阶段是典型的 memory-bound 场景，MBU 是衡量实现效率的核心指标。

---

## 数学推导

### ResNet-50 第一层分析

`Conv2d(3, 64, kernel_size=7, stride=2, padding=3)`，输入 224×224：

**参数量**：

```
P = 64 × 3 × 7 × 7 = 9,408
```

（无 bias 时；若有 bias 加 64）

**输出尺寸**：

```
H_out = floor((224 + 2×3 - 7) / 2 + 1) = 112
```

**FLOPs**：

```
FLOPs = 2 × 64 × 3 × 7 × 7 × 112 × 112 = 235,929,600 ≈ 236 MFLOPs
```

仅这一层就占 ResNet-50 总 FLOPs（~4.1 GFLOPs）的约 5.7%。

### GPT-2 Transformer 层参数拆解

d_model = 768，d_ff = 4 × 768 = 3072。

| 模块 | 计算 | 参数量 |
|------|------|--------|
| Attention (Q/K/V/O) | 4 × 768² + 4 × 768 | 2,362,368 |
| FFN (up + down) | 768×3072 + 3072 + 3072×768 + 768 | 4,722,432 |
| **单层合计** | | ~7.08M |

12 层 × 7.08M ≈ 85M，加上 token embedding（vocab×768）和 position embedding，总计约 117M。

> **注**：上面的 attention 参数量若忽略 bias 则为 4 × 768² = 2,359,296，课程正文中采用的是这个数字。含 bias 时每个 Linear 多 768（即 d_model）个参数。

### 混合精度训练内存分析（1B 模型）

| 组件 | 精度 | 计算 | 内存 |
|------|------|------|------|
| 模型参数 (fwd/bwd) | FP16 | 1×10⁹ × 2B | 2 GB |
| 梯度 | FP16 | 1×10⁹ × 2B | 2 GB |
| 参数 master copy | FP32 | 1×10⁹ × 4B | 4 GB |
| Adam m | FP32 | 1×10⁹ × 4B | 4 GB |
| Adam v | FP32 | 1×10⁹ × 4B | 4 GB |
| **合计** | | | **16 GB** |

> **勘误说明**：原文中 FP16 参数和梯度各算了 4 GB（即按 2×10⁹ × 2 计算），这里的 "2P" 可能来自把参数量记为 P 个参数、每个 2 bytes，写法上容易混淆。按 1B 参数 × 2 bytes/param，FP16 部分各为 2 GB，总计应为 **16 GB**；原文写 20 GB 对应的是把模型参数也保留了一份 FP32 副本用于更新后再转回 FP16 的标准做法，此时 FP32 master copy 那一行已经包含了这部分，FP16 参数实际是 forward 用的工作副本。两种算法的差异在于是否将工作副本和 master copy 同时计入，实际峰值更接近 20 GB（因为更新时两份同时存在）。

Activations 未计入。ZeRO 优化通过在多卡间切分 optimizer states 来降低单卡内存。

---

## 代码示例

```python
import torch
import torch.nn as nn
from typing import Tuple


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """统计可训练参数量和总参数量"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def estimate_flops_linear(in_features: int, out_features: int, batch_size: int = 1) -> int:
    """Linear 层 FLOPs 估算。1 MAC = 2 FLOPs"""
    macs = batch_size * in_features * out_features
    return 2 * macs


def estimate_flops_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    output_h: int,
    output_w: int,
    batch_size: int = 1,
) -> int:
    """Conv2d 层 FLOPs 估算（与输出分辨率正相关）"""
    macs = (
        batch_size * out_channels * in_channels
        * kernel_size * kernel_size * output_h * output_w
    )
    return 2 * macs


def model_memory_mb(model: nn.Module, dtype: torch.dtype = torch.float32) -> float:
    """
    估算推理时参数内存占用 (MB)。
    训练时需额外加上梯度和 optimizer states。
    """
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }.get(dtype, 4)
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * bytes_per_param / (1024 ** 2)


# ──── 简化版 Transformer Block ────

class SmallTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, d_ff: int = 1024):
        super().__init__()
        self.attn_q = nn.Linear(d_model, d_model)
        self.attn_k = nn.Linear(d_model, d_model)
        self.attn_v = nn.Linear(d_model, d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
        scale = (D // self.n_heads) ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = self.attn_out(attn @ v)
        x = self.ln1(x + out)
        x = self.ln2(x + self.ffn(x))
        return x


if __name__ == "__main__":
    model = SmallTransformerBlock(d_model=256, n_heads=4, d_ff=1024)
    trainable, total = count_parameters(model)
    print(f"可训练参数: {trainable:,}")
    print(f"总参数量:   {total:,}")

    # 手动验证
    attn_params = 4 * (256 * 256 + 256)            # Q/K/V/O 四个 Linear
    ffn_params = (256 * 1024 + 1024) + (1024 * 256 + 256)  # up + down
    ln_params = 2 * (256 + 256)                     # 两个 LayerNorm
    print(f"手动估算: {attn_params + ffn_params + ln_params:,}")

    # 不同精度下的内存占用
    for name, dt in [("FP32", torch.float32), ("FP16", torch.float16), ("BF16", torch.bfloat16)]:
        print(f"{name} 内存: {model_memory_mb(model, dt):.3f} MB")

    # FLOPs 示例
    flops = estimate_flops_linear(256, 1024, batch_size=32 * 128)
    print(f"\nFFN up-proj FLOPs (batch=32, seq=128): {flops / 1e6:.2f} MFLOPs")

    # 数值精度对比
    print("\n=== 数值精度实验 ===")
    x32 = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    x16 = x32.half()
    xb16 = x32.bfloat16()
    print(f"FP32: {x32}")
    print(f"FP16 精度损失: {(x32 - x16.float()).abs().max():.6f}")
    print(f"BF16 精度损失: {(x32 - xb16.float()).abs().max():.6f}")

    # FP16 溢出演示
    large = torch.tensor(65505.0)
    print(f"\n65505 → FP16: {large.half()}  (FP16 max = 65504，溢出为 inf)")
    print(f"65505 → BF16: {large.bfloat16()}  (BF16 动态范围与 FP32 一致，无溢出)")
```

---

## Infra 实战映射

### vLLM

vLLM 的核心设计直接对应本讲的内存与计算分析：

- **PagedAttention**：KV cache 按固定大小的"页"分配（默认 16 tokens/page），借鉴 OS 虚拟内存思想，解决连续分配带来的内存碎片问题。
- **Continuous Batching**：根据当前算术强度动态调整 prefill/decode 比例，在 compute-bound 和 memory-bound 之间寻找平衡点。
- **默认 BF16**：`LLMEngine` 初始化时通过 `model_config.dtype` 指定，H100 上可用 `quantization="fp8"` 进一步压缩。

### TensorRT-LLM

NVIDIA 的编译器将 FLOPs 分析直接嵌入优化流程：

- **Layer Fusion**：profiling 每层算术强度后决定是否融合（把 element-wise ops 合并到 matmul kernel，减少内存读写）。
- **GEMM Plugin**：根据矩阵尺寸自动选择 tiling/blocking 策略，最大化 Tensor Core 利用率。
- **混合精度**：`--weight_dtype float16 --kv_cache_dtype int8` 分别控制权重和 KV cache 精度。

### 沐曦 MACA

国产 GPU 的差异化考量：

- 早期硬件可能不支持 BF16，需 fallback 到 FP16 + loss scaling。
- HBM 带宽通常低于 A100（2 TB/s），同模型 decode 延迟更高，MBU 优化更关键。
- MACA 提供类 CUDA 的 API，FLOPs/参数量计算方法不变，但 Roofline 拐点不同，需要针对性 profiling。

---

## 跨 Lecture 关联

| 方向 | Lecture | 关系 |
|------|---------|------|
| ← 前置 | Lec01 | 课程总览，效率优化的动机 |
| → 后续 | Lec03/04 | 剪枝：直接削减参数量和 FLOPs |
| → 后续 | Lec05/06 | 量化：降低每个参数的 bit 数，压缩内存和带宽 |
| → 后续 | Lec11 | Tiny Engine：MCU 上的极限内存压缩 |
| → 后续 | Lec12/13 | Transformer 与 LLM 部署：本讲公式的大规模应用 |

---

## 面试高频题

**Q1：Linear(1024, 4096) 有多少参数？FP16 推理占多少内存？**

参数量 = 1024 × 4096 + 4096 = 4,198,400 ≈ 4.2M。FP16 内存 = 4.2M × 2 bytes = **8.4 MB**。

---

**Q2：FLOPs 和参数量的区别？能否 FLOPs 大但参数少？**

完全可以。`Conv2d(3, 3, 3)` 只有 3×3×3×3 = 81 个参数，但作用在 1000×1000 的图像上 FLOPs 约 162M。本质就是权重共享——同一组参数在不同空间位置反复使用。

---

**Q3：为什么 BF16 比 FP16 更适合 LLM 训练？**

两点。① 动态范围：BF16 的 8 bit 指数与 FP32 一致，最大值约 3.4×10³⁸；FP16 最大仅 65504，attention score 或 loss 偏大时容易溢出，需要额外的 loss scaling 机制。② 转换开销：FP32 → BF16 只需截断低 16 位，近似零成本。代价是 BF16 尾数精度（7 bit）低于 FP16（10 bit），但实际训练中这个差距影响不大。

---

**Q4：LLM decode 阶段为什么是 memory-bound？**

Decode 时 batch_size 通常很小（甚至为 1），矩阵乘法退化为矩阵-向量乘法。每个参数只做 2 次 FLOPs 却要从 HBM 读一次（若 FP16 则读 2 bytes），算术强度极低，远低于 GPU 的 compute/bandwidth 比值。增大 batch size（Continuous Batching）是提高 decode 阶段吞吐的有效手段。

---

**Q5：推理内存只等于参数量 × dtype_bytes 吗？**

不是。推理内存 = 参数 + KV cache + activations。小 batch 下 activations 可忽略，但 KV cache 随序列长度和 batch size 线性增长。长序列场景下 KV cache 甚至会超过参数本身的内存，这正是 vLLM PagedAttention 等工作要解决的问题。
