# Lecture 05 · 量化基础(Part I)

> 课程：MIT 6.5940 Efficient AI Computing (2024 Fall)
> 讲者：Song Han
> 配套资料：[Slides](https://hanlab.mit.edu/courses/2024-fall-65940) · [视频](https://youtu.be/91stHPsxwig) 

这一讲是整门课的一个分水岭：前几讲谈的是"少算"（剪枝、稀疏），从这一讲开始谈的是"算得便宜"。两条路在工业界经常一起用，但量化的工程价值更大——因为它直接吃到了硬件红利：从 Volta 的 INT8 Tensor Core、Hopper 的 FP8、到 Blackwell 的 FP4，每一代 NVIDIA GPU 的峰值算力都是建立在更窄数据类型之上的。理解量化，本质上是理解"硬件提供什么，模型怎么去匹配"。

下面的笔记按"先讲清楚机器在算什么、再讲清楚我们怎么把数塞进去"的顺序展开，并在每一节末尾标注 *业界落点* —— 这部分不是 lecture 内容，是我作为工程师在落地时实际用到的判断。

---

## 5.1 数据类型：先把硬件账本算清楚

量化讨论的第一性原理只有一句话：**bit width 越窄，每次乘加的能耗、面积、内存带宽都成比例下降**。Bill Dally 的那张经典图（45nm 工艺下的能耗表）是所有 efficient AI 的圣经：

| 操作 | 能耗 (pJ) | 相对 FP32 加 |
|---|---|---|
| FP32 MUL | 3.7 | 3.7× |
| FP32 ADD | 0.9 | 1× |
| INT32 MUL | 3.1 | 3.4× |
| INT8 MUL | 0.2 | 0.22× |
| INT8 ADD | 0.03 | 0.03× |
| 32b SRAM 读 | 5 | 5.5× |
| 32b DRAM 读 | 640 | 711× |

两个直接结论：第一，INT8 乘法比 FP32 乘法省约 19 倍能量；第二，**任何片上运算的能耗都被一次 DRAM 访存瞬间秒杀**。这就是为什么 LLM 推理几乎永远是 memory-bound，也是为什么 weight-only quantization（只压权重、激活仍走 FP16）在 decode 阶段就能拿到接近线性的加速——瓶颈本来就在搬权重，权重小了一半，速度自然上去。这条结论在后面 Lec13 讲 AWQ 时会再用一次，记住它。

### 5.1.1 整数：补码不是细节，是硬件接口

无符号 INT8 范围 [0, 255]，有符号 INT8 用二补码表示 [−128, 127]。补码的好处是加法器和乘法器不需要为符号位单独设计电路，所以所有现代 CPU/GPU 的整数 ALU 默认就是补码的。这点在你写 quant kernel 的时候会反复遇到——比如把 INT8 累加进 INT32 时，硬件只保证位扩展（sign-extend）正确，不会替你处理 overflow。

### 5.1.2 定点数：被低估的中间形态

`fixed<8,4>` 的意思是 8 位总宽、其中 4 位小数。它本质上是"带固定 scale 的整数"——硬件用整数 ALU 算，软件知道在哪里有"虚拟小数点"。DSP 和 NPU 上仍然大量使用定点，因为：

1. 不用浮点单元，面积和功耗都低；
2. scale 是 2 的幂时，dequantize 就是一次 shift，零成本。

业界里 Qualcomm Hexagon、寒武纪 MLU、地平线 BPU 这类边缘 NPU 的核心算力都建在定点之上。理解定点的"shift = scale"这个事实，是后面理解 **power-of-two quantization** 的基础。

### 5.1.3 IEEE FP32 与 subnormal

$$(-1)^{\text{sign}} \times (1 + \text{Fraction}) \times 2^{\text{Exponent} - 127}$$

1 位符号、8 位指数（bias=127）、23 位尾数。指数位决定动态范围，尾数位决定相对精度。需要记住的 corner case：

- **Exponent = 0**：进入 subnormal，公式变成 $(-1)^s \times \text{Fraction} \times 2^{-126}$，去掉了隐含的前导 1，目的是让 0 附近的数值密度不至于断崖式下降。
- **Exponent = 255**：表示 ±Inf 和 NaN。

在 quant kernel 里 subnormal 是个真实的坑——某些硬件（包括早期的 GPU SM）处理 subnormal 会走慢路径，导致一个本来不应该的 100x 性能塌方。CUDA 里有 `__ftz` (flush-to-zero) 编译选项就是为此存在的。

### 5.1.4 FP16 vs BF16：为什么训练几乎都用 BF16

这是面试必考题，也是新人最容易答错的题。

| | FP16 | BF16 |
|---|---|---|
| Sign / Exp / Mantissa | 1 / 5 / 10 | 1 / 8 / 7 |
| 最大值 | 65504 | ~3.4×10³⁸ |
| 最小正规值 | ~6×10⁻⁵ | ~1.2×10⁻³⁸ |
| 与 FP32 互转 | 需要 loss scaling | 直接截断尾数 |

BF16 是 Google Brain 为 TPU 设计、后来被 NVIDIA 在 Ampere（A100）跟进的格式。它的 **指数位和 FP32 完全一致**，所以从 FP32 转 BF16 只是把后 16 位截掉，不会触发上下溢出。代价是只有 7 位尾数，相对精度大约 0.4%（FP16 大约 0.05%）。

训练为什么选 BF16 而不是 FP16？因为训练中梯度的动态范围非常宽，FP16 的 5 位指数（最大 65504）经常装不下小梯度，需要 loss scaling 这种 hack 来维持训练稳定。BF16 直接消灭了这个问题。代价是模型权重和 activation 的相对精度差一点，但对 SGD 这种本身就有噪声的算法影响很小。

> 业界落点：H100 的 Tensor Core 同时支持 FP16/BF16/FP8/TF32/FP64 几种 dtype，但实际生产里 LLM 训练几乎是 BF16 一统天下（GPT、LLaMA、Qwen 全是）。FP16 训练只在视觉小模型和一些 legacy 代码里出现。

### 5.1.5 FP8：H100 时代的新主角

NVIDIA 在 Hopper 引入了两种 FP8：

- **E4M3**：4 位指数 + 3 位尾数，最大 ±448。精度优先，用于 **前向激活和权重**。
- **E5M2**：5 位指数 + 2 位尾数，最大 ±57344。范围优先，用于 **梯度**（梯度分布尾巴长）。

FP8 的关键是它已经不能用"对 FP32 的近似"这种心态去想了——它的尾数太短（3 位），必须配合 **per-tensor scale**（甚至 per-block scale）才能用。这就是为什么 NVIDIA Transformer Engine 做的事情 90% 都是"在合适的位置插 amax/scale 计算"。

> 业界落点：H100 上 FP8 GEMM 的峰值算力是 BF16 的 2 倍（约 1979 vs 989 TFLOPS），但你实际跑出来通常只有 1.3–1.6 倍加速，因为 amax 同步、scale 维护、kernel launch 都在吃时间。这个 gap 是 MLSys 工程师调优的主战场。

### 5.1.6 MXFP / NVFP：Blackwell 之后的新格局

Lecture 没讲，但这是 2024–2026 业界最大的变化，必须补：

- **MXFP4 / MXFP6 / MXFP8**（OCP Microscaling 标准）：每 32 个元素共享一个 8 位 E8M0 scale。Blackwell B200/B300 原生支持。
- **NVFP4**：NVIDIA 自家的 FP4 变体，每 16 个元素一个 FP8 scale，比 MXFP4 精度更好。
- **FP4 GEMM 算力**：B200 在 FP4 下号称 20 PFLOPS（dense）/ 40 PFLOPS（sparse），是 H100 BF16 的 20 倍。

这套东西的工程含义是：**未来 2 年大模型推理的主流会从 W8A8 / W4A16 走向 W4A4 / FP4**。如果你现在写量化代码，必须把 group quantization（block-wise scale）当作 first-class citizen 来设计接口，否则两年后要重写。

---

## 5.2 线性量化：把浮点塞进整数的最朴素方法

### 5.2.1 公式

量化（fp → int）：
$$q = \text{round}\!\left(\frac{r}{S}\right) + Z$$

反量化（int → fp）：
$$\hat r = S \cdot (q - Z)$$

其中 $S>0$ 是 scale（一个浮点数），$Z$ 是 zero-point（一个整数）。这两个参数加起来叫 quantization parameters，简称 qparams。

量化误差 $e = r - \hat r$ 来自三个独立的源头：

1. **Rounding error**：连续值落在最近的格点上，最坏情况 $|e| \le S/2$。
2. **Clipping error**：超出 $[r_{\min}, r_{\max}]$ 的值被截断到边界。这个误差可以非常大。
3. **Scale 本身的精度**：S 通常存成 FP16 或 FP32，但极低 bit 时会引入二阶误差。

工程上 99% 的精度问题是 clipping 引起的，而不是 rounding。这导出了一个非常重要的判断：**量化的核心不是 round，而是怎么选 $r_{\min}, r_{\max}$**。这个问题在下一讲 calibration 里会变成主线。

### 5.2.2 对称量化

令 $Z=0$，$S = \dfrac{\max(|r|)}{2^{b-1}-1}$，则 $q\in[-2^{b-1}, 2^{b-1}-1]$。

优点是反量化只剩一次乘法（没有 zero-point 的减法），矩阵乘里每个累加项只需要 $S_x \cdot S_w \cdot \sum x_q w_q$，可以把 scale 完全提到 GEMM 外面去算。**这就是为什么权重几乎永远用对称量化**——权重分布天然以 0 为中心，而且 GEMM 内层循环不能容忍额外的零点修正。

缺点是如果分布不对称（例如 ReLU 之后的激活全为非负），会浪费一半的格点。

### 5.2.3 非对称量化

$$S = \frac{r_{\max}-r_{\min}}{2^b - 1},\quad Z = \text{round}\!\left(-\frac{r_{\min}}{S}\right)$$

完整利用了量化范围。代价是 GEMM 展开后会出现 cross terms：

$$\sum_i (x_i - Z_x)(w_i - Z_w) = \sum x_i w_i - Z_w \sum x_i - Z_x \sum w_i + N Z_x Z_w$$

后面三项都需要预先计算或在 epilogue 里补回来。TensorRT 和 cuDNN 的 INT8 GEMM 都为这种 "asymmetric activation × symmetric weight" 的组合写了专门的 kernel。

> 业界落点：现代 LLM 量化方案里几乎统一用 **symmetric weight + asymmetric activation**（如果 activation 量化的话）。weight-only 方案（GPTQ/AWQ/W4A16）则连 activation 都不动，更简单。

### 5.2.4 一段可跑的代码

```python
import numpy as np

def symmetric_quantize(x, n_bits=8):
    qmax = 2 ** (n_bits - 1) - 1
    scale = np.abs(x).max() / qmax
    q = np.clip(np.round(x / scale), -qmax - 1, qmax).astype(np.int8)
    return q, scale

def asymmetric_quantize(x, n_bits=8):
    qmin, qmax = 0, 2 ** n_bits - 1
    scale = (x.max() - x.min()) / (qmax - qmin)
    zp = int(round(-x.min() / scale))
    q = np.clip(np.round(x / scale) + zp, qmin, qmax).astype(np.uint8)
    return q, scale, zp

x = np.random.randn(4096).astype(np.float32) * 0.1
q1, s1 = symmetric_quantize(x)
q2, s2, z2 = asymmetric_quantize(x)

mse_sym = np.mean((x - q1.astype(np.float32) * s1) ** 2)
mse_asym = np.mean((x - (q2.astype(np.float32) - z2) * s2) ** 2)
print(f"sym  MSE = {mse_sym:.3e}")
print(f"asym MSE = {mse_asym:.3e}")
```

如果你把上面 `x` 改成 `np.random.randn(...) ** 2`（模拟 ReLU 后的 activation），会看到非对称量化的 MSE 比对称低一倍以上。这就是非对称存在的全部理由。

---

## 5.3 量化粒度：精度和硬件友好的 trade-off

| 粒度 | scale 数量 | 精度 | 硬件成本 | 典型使用方 |
|---|---|---|---|---|
| Per-tensor | 1 | 低 | 最低，GEMM 外提 | TFLite INT8、TRT 早期方案 |
| Per-token (activation) | seq_len | 中 | 中等 | SmoothQuant、INT8 LLM |
| Per-channel (weight) | out_channels | 中高 | 中等 | PyTorch QAT、TRT 现代方案 |
| Per-group (weight) | out × ⌈in/g⌉ | 高 | 较高，需特殊 kernel | AWQ、GPTQ、MX/NV FP4 |

**Per-tensor** 把整个张量塞进一个 scale，最省，但只要存在一个 outlier 就会撑大 scale 让其他值的有效精度全面塌方。在 CNN 上还能用，在 LLM 上几乎不可用——因为 LLM activation 的 outlier 系统性地集中在某些 channel 上，比其他 channel 大 100 倍以上（这是 SmoothQuant 论文的核心观察，Lec13 会展开）。

**Per-channel** 给 weight 的每个 output channel 独立 scale。对于 `Y = X W^T`，每个 output channel 对应 W 的一行，这一行用自己的 scale 不会破坏 GEMM 的展开形式。Activation 的 per-channel 量化则不行——因为同一个 channel 在 GEMM 里要和 W 的不同 row 相乘，scale 会和 reduction 维度纠缠。这就是为什么有 **per-token activation + per-channel weight** 这个经典组合：两个 scale 都不在 reduction 维度上。

**Per-group**（也叫 sub-channel、block-wise）把一行权重切成长度 g（典型 64/128）的小块，每块独立 scale。AWQ 和 GPTQ 用 g=128 的 W4 是事实标准。代价是：GEMM kernel 必须在内层循环里做 dequantize（因为 scale 在 reduction 维度上跳变），这就是为什么 Marlin、Machete、CUTLASS 的 W4A16 kernel 写起来比标准 GEMM 复杂一个数量级。

```python
def per_channel_symmetric_quantize(W, n_bits=8):
    """W: [out, in]，沿 out 维度独立量化"""
    qmax = 2 ** (n_bits - 1) - 1
    scales = np.abs(W).max(axis=1, keepdims=True) / qmax  # [out, 1]
    q = np.clip(np.round(W / scales), -qmax - 1, qmax).astype(np.int8)
    return q, scales.squeeze(1)

def group_symmetric_quantize(W, group_size=128, n_bits=4):
    """W: [out, in]，每行切成 in/g 个 group"""
    qmax = 2 ** (n_bits - 1) - 1
    out, in_ = W.shape
    assert in_ % group_size == 0
    Wg = W.reshape(out, in_ // group_size, group_size)
    scales = np.abs(Wg).max(axis=-1, keepdims=True) / qmax
    q = np.clip(np.round(Wg / scales), -qmax - 1, qmax).astype(np.int8)
    return q.reshape(out, in_), scales.squeeze(-1)  # scales: [out, in/g]
```

> 业界落点：vLLM 0.6+ 加载 AWQ 模型默认走 Marlin kernel，吞吐比朴素 dequant+GEMM 高 2–3 倍。如果你在公司里跑 vLLM 性能不达标，第一件事是确认 `quantization=awq_marlin` 而不是 `awq`。这是非常常见的性能事故。

---

## 5.4 Uniform vs Non-uniform

Uniform 量化（也就是上面整节讲的线性量化）用等距格点。Non-uniform 量化用任意格点位置——典型做法是 K-means 聚类找最优 codebook：

```python
from sklearn.cluster import KMeans

def kmeans_quantize(W, n_bits=4):
    K = 2 ** n_bits
    flat = W.reshape(-1, 1)
    km = KMeans(n_clusters=K, n_init=10).fit(flat)
    Wq = km.cluster_centers_[km.labels_].reshape(W.shape)
    return Wq, km.cluster_centers_.flatten()
```

这就是 Han Song 自己 2016 年 *Deep Compression* 论文的做法：迭代剪枝 → K-means 量化 → Huffman 编码，三件套压缩 35×。在当时是 ICLR best paper。

但 non-uniform quantization 在工业界基本死了，原因只有一个：**没有硬件能直接对 codebook indices 做 GEMM**。每次乘法之前必须先查表把 4-bit index 翻译回 FP16，这个查表开销在 GPU 上吞掉了所有收益。它现在只活在两个地方：

1. **存储压缩**（不参与计算），比如 LLM weight 的离线压缩传输；
2. **向量数据库的 PQ**（Product Quantization），见下一节。

---

## 5.5 Product Quantization

把 d 维向量切成 M 段，每段独立做 K-means。总 codebook 大小从 $K^d$ 降到 $M\cdot K^{d/M}$，距离查询可以用查表加加法完成。

PQ 的主战场不是模型量化，是 **向量检索**。FAISS、ScaNN、Milvus 这些向量数据库的核心索引（IVF-PQ、IVFADC）全部建在 PQ 之上。一个 768 维的 BERT embedding，用 PQ 可以压到 96 字节，亿级向量库在单机就能跑。如果你做 RAG 或者推荐召回，PQ 是绕不开的。

最近两年比较新的发展是 **RaBitQ / QINCo / RVQ**——把 PQ 和神经网络结合，用残差量化（Residual Quantization）做更精细的 codebook。这条线做向量检索的人需要关注。

---

## 5.6 一段更接近生产的代码

下面这段加入了 STE（Straight-Through Estimator），是 QAT 训练的最小骨架。Lec06 会详细讲为什么 round 操作的梯度要直通。

```python
import torch
import torch.nn as nn

class FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, qmin, qmax):
        q = torch.clamp(torch.round(x / scale), qmin, qmax)
        return q * scale

    @staticmethod
    def backward(ctx, g):
        # STE：把不可导的 round 当作恒等映射
        return g, None, None, None


class QLinear(nn.Module):
    def __init__(self, in_f, out_f, n_bits=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.n_bits = n_bits
        self.register_buffer("scale", torch.tensor(1.0))

    @torch.no_grad()
    def calibrate(self):
        qmax = 2 ** (self.n_bits - 1) - 1
        # per-tensor 对称，简单起见
        self.scale.copy_(self.weight.abs().max() / qmax)

    def forward(self, x):
        qmax = 2 ** (self.n_bits - 1) - 1
        qmin = -qmax - 1
        w_q = FakeQuant.apply(self.weight, self.scale, qmin, qmax)
        return x @ w_q.T + self.bias
```

这段代码本身不会带来推理加速（fake quant 在 FP32 里跑），它的作用是让训练过程"感受到"量化误差，从而把权重往对量化友好的方向调。真正的加速来自把训好的 `weight` 转成 INT8 张量，放进 INT8 GEMM kernel。这两步在 PyTorch 里是 `torch.ao.quantization` 模块负责的。

---

## 5.7 Infra 实战映射

这一节是我自己加的，lecture 不讲，但面试和上线都问。

**vLLM**。默认 BF16/FP16 推理，量化通过 `--quantization {awq, gptq, awq_marlin, gptq_marlin, fp8, compressed-tensors}` 加载。生产里要点：
- W4A16 用 `awq_marlin` 或 `gptq_marlin`，吞吐对比朴素方案 2–3×。
- Hopper 上跑 FP8 用 `fp8`，配合 `kv_cache_dtype=fp8` 把 KV cache 也压掉，长 context 收益巨大。
- W4A8 / W4A4 还在 `compressed-tensors` 这套 schema 下持续演进，QServe 是目前的 SOTA 之一。

**TensorRT-LLM**。NVIDIA 自家的极致优化方案。原生支持 INT8 SmoothQuant、FP8、INT4 weight-only、FP4（B200）。优势是 kernel fusion 做到极致——dequantize、GEMM、bias、activation、residual add 全部 fuse 在一个 CUDA kernel 里。劣势是模型支持速度比 vLLM 慢一拍，且 build engine 时间长。

**SGLang**。介于 vLLM 和 TRT-LLM 之间，量化策略基本对齐 vLLM，但 scheduler 和 RadixAttention 让前缀复用场景吞吐更高。Agent / 多轮对话场景值得评测。

**llama.cpp / GGUF**。CPU 推理事实标准。它的 K-quants（Q4_K, Q5_K, Q6_K）是 group quantization 的一个变体，在 CPU 上手写 SIMD kernel。本地推理几乎都用它。

**国产硬件**。华为昇腾、寒武纪 MLU、沐曦 MetaX、摩尔线程都有自己的 INT8/FP8 路线，但 kernel 生态远不如 CUDA。如果你在国内 infra 团队，一个常见任务是把 vLLM/TRT-LLM 的某个 quant kernel "翻译"到目标硬件的算子库上。这种工作的难点 90% 不在数学，而在硬件文档不全、工具链不成熟。

---

## 5.8 跨 Lecture 关联

- **前置 ←** Lec02 数值基础、Lec03 剪枝（量化和剪枝在 Deep Compression 里联合使用）
- **后续 →** Lec06 进阶量化：PTQ vs QAT、calibration 算法（min-max / KL / MSE / percentile）、STE 推导
- **后续 →** Lec13 LLM 部署：SmoothQuant、AWQ、GPTQ、QServe，是这一讲在 LLM 场景的直接延伸
- **横向 ↔** Lec07 NAS：硬件感知的 NAS 经常把"是否能 INT8 化"作为搜索目标之一

---

## 5.9 面试高频题（答题口径）

**Q1. 对称 vs 非对称，怎么选？**
权重几乎永远对称，因为分布以 0 为中心，且对称量化让 GEMM 累加项里的 scale 可以提到外层循环之外，硬件友好。激活看情况：ReLU 之后全正用非对称；GELU/SiLU 之后用对称（因为有负值）。LLM 时代更常见的做法是 weight 对称 + activation 非对称（per-token），或者干脆 weight-only 不动激活。

**Q2. FP16 vs BF16，训练为什么用 BF16？**
BF16 的指数位和 FP32 一样多（8 bit），动态范围相同，从 FP32 转 BF16 只是截断尾数，不会上下溢出，也不需要 loss scaling。代价是只有 7 位尾数，相对精度比 FP16 差 8 倍，但 SGD 本身有梯度噪声，不敏感。FP16 训练需要 dynamic loss scaling，工程复杂度更高，现在只在小模型和 legacy 代码里出现。

**Q3. Per-tensor vs Per-channel vs Per-group 的 trade-off？**
精度递增、硬件成本递增。Per-tensor 在 LLM 上不可用（outlier 撑大 scale），per-channel 是 CNN 时代的事实标准，per-group（g=128）是现在 W4 LLM 的事实标准。Group 的代价是 GEMM kernel 内层循环要做 dequant，需要 Marlin / Machete 这种特化 kernel 才能跑出收益。

**Q4. 为什么 LLM 量化比 CNN 难？**
两个原因。第一，LLM activation 存在 systematic outlier——某些 channel 的值持续比其他大 100 倍以上，这些 channel 通常出现在 attention 之后的 down_proj 输入上。Per-tensor 量化时这些 outlier 撑大 scale，让正常 channel 的有效 bit 数从 8 掉到 2-3。SmoothQuant 的解决方案是把 outlier 从 activation 迁移到 weight（数学上等价的缩放变换）。第二，LLM 的精度对量化误差更敏感，因为 autoregressive 解码会让 token-level 误差累积。

**Q5. W4 vs W8 vs FP8，生产里怎么选？**
看瓶颈。Decode 阶段是 memory-bound，weight 越小越好，W4A16（AWQ/GPTQ）吞吐最高，单卡能装更大模型。Prefill 阶段是 compute-bound，W4 的 dequant 开销吃掉收益，FP8 或 W8A8 (SmoothQuant) 更合适。所以 SOTA 方案（QServe 的 W4A8）是 weight 用 W4 省内存、激活用 INT8 跑 GEMM 拿算力。Hopper 之后 FP8 是更省心的选择，因为有原生 Tensor Core 支持，不需要 dequant。

**Q6. 4-bit 模型实际能省多少？**
朴素地算，FP16 → INT4 是 4×。但要扣掉：scale 元数据（per-group g=128 时大约 6.25% overhead）、KV cache（如果不压）、activation buffer。所以 7B 模型 FP16 14 GB → W4 实际 4–4.5 GB。如果你看到一个 4-bit 模型号称 "exactly 1/4"，多半是没算 scale。

---

## 5.10 自检清单

读完这一讲，你应该能不查资料地回答：

1. INT8 乘法比 FP32 乘法省多少能量？比 DRAM 一次访存呢？
2. 写出对称量化的 scale 公式，并解释为什么 weight 用对称、activation 经常用非对称。
3. Per-channel 量化为什么对 weight 可行、对 activation 不可行？
4. BF16 的指数位有几位？为什么这件事对训练这么重要？
5. 一个 g=128 的 W4 量化的 7B 模型，加上 scale 元数据，实际占多少显存？
6. 你的推理框架（vLLM/TRT-LLM/SGLang）跑 AWQ 模型时，怎么确认走的是 Marlin kernel 而不是朴素 dequant？

如果有任何一题答不上来，回去看对应小节。
