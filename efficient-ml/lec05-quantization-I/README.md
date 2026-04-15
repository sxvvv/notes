# Lec 05 · 量化基础 (Part I)

> **课程**: MIT 6.5940 *TinyML and Efficient Deep Learning Computing* (Fall 2024)
> **讲者**: Song Han
> **配套**: [Slides (PDF)](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec05-Quantization-I.pdf) · [Video](https://youtu.be/91stHPsxwig) · [课程主页](https://hanlab.mit.edu/courses/2024-fall-65940)

这一讲是整门课的分水岭。前面几讲讲的是"少算"(剪枝、稀疏),从这讲开始讲"算得便宜"。两条路在工业界常常一起上,但量化的工程价值更直接——它吃的是硬件红利。从 Volta 的 INT8 Tensor Core、Turing 的 INT4、Hopper 的 FP8,到 Blackwell 的 FP4,每一代 NVIDIA 数据中心 GPU 的峰值算力都建立在更窄的数据类型上。理解量化,本质上是理解"硬件提供什么,模型怎么去匹配"。

下面按"先讲清楚机器在算什么,再讲清楚我们怎么把数塞进去"的顺序展开。每节末尾的 *业界落点* 不是课件内容,是我自己在推理部署里踩过的一些坑。

---

## 目录

- [5.1 数据类型:把硬件账本算清楚](#51-数据类型把硬件账本算清楚)
- [5.2 线性量化:最朴素的映射](#52-线性量化最朴素的映射)
- [5.3 量化粒度:精度和硬件友好的拉锯](#53-量化粒度精度和硬件友好的拉锯)
- [5.4 Uniform vs Non-uniform](#54-uniform-vs-non-uniform)
- [5.5 Product Quantization:量化在向量检索里的第二春](#55-product-quantization量化在向量检索里的第二春)
- [5.6 一段更接近生产的 QAT 骨架](#56-一段更接近生产的-qat-骨架)
- [5.7 Infra 实战:框架到底在做什么](#57-infra-实战框架到底在做什么)
- [5.8 跨 Lecture 关联](#58-跨-lecture-关联)
- [5.9 面试高频题](#59-面试高频题)
- [5.10 自检清单](#510-自检清单)
- [参考文献](#参考文献)

---

## 5.1 数据类型:把硬件账本算清楚

量化讨论的第一性原理只有一句话: **bit width 越窄,每次乘加的能耗、面积、访存带宽成比例下降**。这张经典的 45 nm 能耗表(Horowitz ISSCC 2014[^horowitz],Song Han 的课件和 Dally 的 NIPS'15 slides 都在用它)是 efficient AI 的圣经:

<p align="center">
  <img src="./images/computation_cost.png
" width="640" alt="Horowitz 2014 energy table"/>
</p>

| Operation | Energy (pJ) | 相对 FP32 乘法 |
|---|---:|---:|
| 8-bit Add (INT8) | 0.03 | 0.008× |
| 16-bit Add (INT16) | 0.05 | 0.014× |
| 32-bit Add (INT32) | 0.1 | 0.027× |
| 16-bit FP Add (FP16) | 0.4 | 0.11× |
| 32-bit FP Add (FP32) | 0.9 | 0.24× |
| **8-bit Mult (INT8)** | **0.2** | **0.054×** |
| 16-bit FP Mult (FP16) | 1.1 | 0.30× |
| 32-bit Int Mult | 3.1 | 0.84× |
| **32-bit FP Mult (FP32)** | **3.7** | **1.00×** |
| 32-bit SRAM Read (8 KB) | 5 | 1.35× |
| **32-bit DRAM Read** | **640** | **173×** |

> 数据来源:Horowitz 2014[^horowitz],45 nm 0.9 V。表格数字从 Song Han 的 CS231n 2017 slides 转引[^han231n]。

两个结论要刻进脑子:

1. **INT8 乘法比 FP32 乘法省约 18×**(3.7 / 0.2)。只算乘法的话,位宽减半能量近似降 4×。
2. **任何片上运算的能耗都被一次 DRAM 读秒杀**。一次 DRAM 读 = 173 次 FP32 乘法 = 3200 次 INT8 乘法。

第二点是理解 LLM 推理经济学的钥匙。Decode 阶段(autoregressive,batch=1)的瓶颈是**搬权重**,不是算——这就是为什么 **weight-only quantization**(只压权重、激活照跑 FP16/BF16)在 decode 阶段能拿到接近线性的加速。权重小一半,搬运时间就少一半,而 GEMM 本来就不是 bottleneck。这条结论在 Lec13 讲 AWQ[^awq] 时还会再用一次。

### 5.1.1 整数:补码是硬件接口

无符号 INT8 范围 `[0, 255]`,有符号 INT8 用二补码表示 `[-128, 127]`。补码的好处是加法器和乘法器不需要为符号位单独设计电路——硬件 ALU 默认就是补码。写 quant kernel 时会反复遇到一个坑:INT8 累加进 INT32 必须 **显式 sign-extend**,否则 `-1` 会变成 `0x000000FF` 而不是 `0xFFFFFFFF`。CUDA 的 `__dp4a` intrinsic 帮你处理好了,手写 SASS 或 AVX-VNNI 就得自己来。

### 5.1.2 定点数:被低估的中间形态

`fixed<8,4>` 表示 8 位总宽、其中 4 位小数。它本质是"带固定 scale 的整数"——硬件用整数 ALU 算,软件知道"虚拟小数点"在哪里。DSP 和边缘 NPU 上仍然大量使用定点,因为:

1. 不需要浮点单元,面积和功耗低。
2. **scale 是 2 的幂时,dequantize 就是一次 shift,零成本。**

Qualcomm Hexagon、寒武纪 MLU、地平线 BPU 这类边缘 NPU 的核心算力都建在定点之上。"shift = scale"这条事实是后面理解 **power-of-two quantization** 的基础。

### 5.1.3 IEEE FP32 与 subnormal

$$(-1)^{\mathrm{sign}} \times (1 + \mathrm{Fraction}) \times 2^{\mathrm{Exponent} - 127}$$

1 位符号、8 位指数(bias = 127)、23 位尾数。指数位决定动态范围,尾数位决定相对精度。两个 corner case 要记住:

- **Exponent = 0**:进入 subnormal,公式变成 $(-1)^s \times \mathrm{Fraction} \times 2^{-126}$,去掉隐含前导 1,让 0 附近数值密度不会断崖式下降。
- **Exponent = 255**:表示 ±Inf 和 NaN。

在 quant kernel 里 subnormal 是个真实的坑:某些硬件(包括早期的 GPU SM)处理 subnormal 走慢路径,会导致一个本来不该发生的 100× 性能塌方。CUDA 的 `-ftz=true` 编译选项就是为此存在的,它把 subnormal 一律刷成零。

### 5.1.4 FP16 vs BF16:为什么训练几乎都用 BF16

<p align="center">
  <img src="./images/02-fp16-vs-bf16-bits.png" width="640" alt="FP16 vs BF16 bit layout"/>
</p>

|  | FP16 | BF16 |
|---|---|---|
| Sign / Exp / Mantissa | 1 / 5 / 10 | 1 / 8 / 7 |
| 动态范围 (max normal) | ≈ 6.55 × 10⁴ | ≈ 3.39 × 10³⁸ |
| 最小 normal | ≈ 6.10 × 10⁻⁵ | ≈ 1.18 × 10⁻³⁸ |
| 相对精度 (≈ ULP) | 2⁻¹⁰ ≈ 0.1% | 2⁻⁷ ≈ 0.8% |
| 与 FP32 互转 | 需要 loss scaling | 直接截断尾数 |

BF16 是 Google Brain 为 TPU 设计、NVIDIA 在 Ampere (A100) 跟进的格式[^bf16]。它的 **指数位和 FP32 完全一致**(都是 8 位),所以 FP32 → BF16 就是把后 16 位尾数截掉,**不会触发上下溢出**。代价是尾数只剩 7 位,相对精度比 FP16 差 8×(≈ 0.8% vs 0.1%)。

训练为什么选 BF16 不选 FP16?因为训练中梯度的动态范围非常宽,FP16 的最大 normal 只有 65504,小梯度经常装不下,必须用 **loss scaling** 这种 hack(把 loss 先乘 2¹⁵ 再 backward,update 前再除回来)来防止下溢。BF16 动态范围和 FP32 一样,这个问题消失了。代价是前向激活的相对精度差一些,但 SGD 本身带梯度噪声,不敏感。

> **业界落点**:H100 Tensor Core 同时支持 FP16 / BF16 / FP8 / TF32 / FP64,但生产环境里 LLM 训练几乎是 BF16 一统天下(GPT、LLaMA、Qwen、DeepSeek 全是)。FP16 训练只在 CV 小模型和 legacy 代码里出现。

### 5.1.5 FP8:Hopper 开始的新主角

NVIDIA 在 Hopper (H100) 引入了两种 FP8[^fp8]:

| | E4M3 | E5M2 |
|---|---|---|
| Exp / Mant | 4 / 3 | 5 / 2 |
| Max normal | ±448 | ±57344 |
| 用途 | **前向激活 / 权重** | **反向梯度** |
| 原因 | 精度优先 | 范围优先 |

FP8 已经不能再用"FP32 的近似"那套心态去想了——尾数只剩 3 位,必须配合 **per-tensor scale**(甚至 per-block scale)才能用。这就是为什么 NVIDIA [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) 90% 的工作是"在合适的位置插 amax / scale 的计算和同步"。

**H100 SXM 算力数字**[^h100ds](都是 dense,sparse 翻倍):

| Precision | TFLOPS |
|---|---:|
| FP64 Tensor Core | 67 |
| TF32 Tensor Core | 989 |
| **FP16 / BF16 Tensor Core** | **989** |
| **FP8 Tensor Core** | **1,979** |
| INT8 Tensor Core | 1,979 TOPS |

> **业界落点**:H100 FP8 GEMM 的峰值算力是 BF16 的 **2×**,但实际跑出来通常只有 1.3–1.6× 加速。差距被 amax 同步、scale 维护、kernel launch 吃掉了。这个 gap 是 MLSys 工程师调优的主战场。DeepSeek-V3 的 FP8 训练是 2024 年最精彩的案例之一,他们的 [技术报告](https://arxiv.org/abs/2412.19437) 里讲了怎么做 fine-grained scaling 把 FP8 训练稳住。

### 5.1.6 MXFP / NVFP:Blackwell 之后的新格局

Lecture 本身没讲,但这是 2024–2026 年业界最大的变化,必须补:

- **MXFP4 / MXFP6 / MXFP8** (OCP Microscaling 标准[^mx]):每 32 个元素共享一个 8 位 E8M0 scale。Blackwell 原生支持。
- **NVFP4**:NVIDIA 自家的 FP4 变体,每 **16** 个元素一个 FP8 (E4M3) scale,比 MXFP4 精度更好。
- **B200 FP4 算力**:**9 PFLOPS dense / 18 PFLOPS sparse** [^b200ds][^cudo]。是 H100 BF16 dense (989 TFLOPS) 的 **约 9×**(dense-to-dense),**约 18×**(sparse-to-dense)。NVIDIA 市场材料里有时把 GB200 NVL72 整机或和 sparsity 混算到一起得到 "20× / 25×" 这种数字,注意区分。

这套东西的工程含义很直接: **未来两年大模型推理的主流会从 W8A8 / W4A16 走向 W4A4 / FP4**。如果你现在写量化代码,必须把 group-wise scale 当作 first-class citizen 设计进接口,否则两年后要重写。

---

## 5.2 线性量化:最朴素的映射

### 5.2.1 公式

**量化**(fp → int):

$$q = \mathrm{round}\!\left(\frac{r}{S}\right) + Z$$

**反量化**(int → fp):

$$\hat{r} = S \cdot (q - Z)$$

其中 $S > 0$ 是 scale(一个浮点数),$Z$ 是 zero-point(一个整数)。这两个加起来叫 **qparams**。

量化误差 $e = r - \hat{r}$ 有三个独立来源:

1. **Rounding error**:连续值落在最近格点上,最坏情况 $|e| \le S/2$。
2. **Clipping error**:超出 $[r_{\min}, r_{\max}]$ 的值被截断到边界,误差可以任意大。
3. **Scale 本身的精度**:S 通常存 FP16 / FP32,极低 bit 时引入二阶误差。

工程上 **99% 的精度问题是 clipping 引起的,不是 rounding**。这导出一个非常重要的判断: **量化的核心不是 round,而是怎么选 $r_{\min}, r_{\max}$**。这个问题在下一讲的 calibration 里会变成主线。

<p align="center">
  <img src="./images/03-quant-error-clip-vs-round.png" width="640" alt="Clipping error dominates rounding error"/>
</p>

### 5.2.2 对称量化

令 $Z = 0$,$S = \dfrac{\max(|r|)}{2^{b-1}-1}$,则 $q \in [-2^{b-1}, 2^{b-1}-1]$。

优点:反量化只剩一次乘法(没有 zero-point 减法),矩阵乘里每个累加项变成

$$y = \sum_i x_i w_i \;=\; S_x S_w \sum_i q^x_i q^w_i$$

scale 可以完全提到 GEMM 外面算。**这就是为什么权重几乎永远用对称量化**——权重分布天然以 0 为中心,而且 GEMM 内层循环不能容忍额外的零点修正开销。

缺点:如果分布不对称(比如 ReLU 之后激活全为非负),一半的格点被浪费。

### 5.2.3 非对称量化

$$S = \frac{r_{\max} - r_{\min}}{2^b - 1}, \qquad Z = \mathrm{round}\!\left(-\frac{r_{\min}}{S}\right)$$

完整利用量化范围。代价是 GEMM 展开后出现 cross terms:

$$\sum_i (q^x_i - Z_x)(q^w_i - Z_w) = \sum q^x_i q^w_i - Z_w \sum q^x_i - Z_x \sum q^w_i + N Z_x Z_w$$

后三项需要预先计算或在 epilogue 里补回来。TensorRT 和 cuDNN 的 INT8 GEMM 都为 "asymmetric activation × symmetric weight" 的组合写了专门 kernel(TFLite 的 [量化白皮书](https://arxiv.org/abs/1806.08342) 里有经典推导[^jacob])。

> **业界落点**:现代 LLM 量化方案几乎统一是 **symmetric weight + (optional) asymmetric activation**。Weight-only 方案(GPTQ[^gptq] / AWQ / W4A16)则连 activation 都不动,最简单。

### 5.2.4 可运行代码

```python
import numpy as np

def symmetric_quantize(x, n_bits=8):
    qmax = 2 ** (n_bits - 1) - 1          # 127 for INT8
    scale = np.abs(x).max() / qmax
    q = np.clip(np.round(x / scale), -qmax - 1, qmax).astype(np.int8)
    return q, scale

def asymmetric_quantize(x, n_bits=8):
    qmin, qmax = 0, 2 ** n_bits - 1        # 0..255 for UINT8
    scale = (x.max() - x.min()) / (qmax - qmin)
    zp = int(round(-x.min() / scale))
    q = np.clip(np.round(x / scale) + zp, qmin, qmax).astype(np.uint8)
    return q, scale, zp

# 对比:Gaussian(对称分布) vs Chi-square(偏正,模拟 ReLU 输出)
rng = np.random.default_rng(0)
x_sym  = rng.standard_normal(4096).astype(np.float32) * 0.1
x_asym = rng.chisquare(df=2, size=4096).astype(np.float32) * 0.1

for name, x in [("gaussian", x_sym), ("chi-square", x_asym)]:
    qs, s = symmetric_quantize(x)
    qa, sa, za = asymmetric_quantize(x)
    mse_s = np.mean((x - qs.astype(np.float32) * s) ** 2)
    mse_a = np.mean((x - (qa.astype(np.float32) - za) * sa) ** 2)
    print(f"{name:11s}  sym MSE={mse_s:.3e}  asym MSE={mse_a:.3e}  ratio={mse_s/mse_a:.2f}×")
```

在偏态分布上,非对称量化的 MSE 比对称低一半以上。这就是非对称存在的全部理由。

---

## 5.3 量化粒度:精度和硬件友好的拉锯

<p align="center">
  <img src="./images/04-granularity-tensor-channel-group.png" width="720" alt="Per-tensor / per-channel / per-group granularity"/>
</p>

| 粒度 | scale 数量 | 精度 | 硬件成本 | 代表方案 |
|---|---|---|---|---|
| **Per-tensor** | 1 | 低 | 最低(GEMM 外提) | TFLite INT8、TRT 早期 |
| **Per-token (activation)** | `seq_len` | 中 | 中等 | SmoothQuant、LLM.int8() |
| **Per-channel (weight)** | `out_channels` | 中高 | 中等 | PyTorch QAT、TRT 现代 |
| **Per-group (weight)** | `out × ⌈in / g⌉` | 高 | 需特化 kernel | AWQ、GPTQ、MXFP4 / NVFP4 |

**Per-tensor** 整个张量一个 scale,最省,但**只要存在一个 outlier 就会撑大 scale 让其他值的有效精度塌方**。在 CNN 上还能用,在 LLM 上基本不可用——LLM activation 的 outlier 系统性地集中在少数 channel 上,比其他 channel 大 100× 以上。这是 SmoothQuant[^smoothquant] 和 LLM.int8()[^llmint8] 两篇论文的共同观察(Lec13 会展开)。

**Per-channel** 给 weight 每个 output channel 独立 scale。对于 $Y = X W^\top$,每个 output channel 对应 W 的一行,用自己的 scale 不会破坏 GEMM 展开。**Activation 的 per-channel 量化则不行**——同一个 channel 在 GEMM 里要和 W 的不同 row 相乘,scale 会和 reduction 维度纠缠。这就是为什么有 **per-token activation + per-channel weight** 这个经典组合:两个 scale 都避开 reduction 维度。

**Per-group**(也叫 sub-channel 或 block-wise)把一行权重切成长度 `g`(典型 64 / 128)的小块,每块独立 scale。AWQ 和 GPTQ 默认 `g=128` 的 W4 是 LLM 推理的事实标准。代价是:GEMM kernel 必须在内层循环里 dequantize(scale 在 reduction 维度上跳变),所以 [Marlin](https://github.com/IST-DASLab/marlin)[^marlin] / Machete / CUTLASS 的 W4A16 kernel 比标准 GEMM 复杂一个量级。

```python
def per_channel_symmetric_quantize(W, n_bits=8):
    """W: [out, in],沿 out 维度独立量化。"""
    qmax = 2 ** (n_bits - 1) - 1
    scales = np.abs(W).max(axis=1, keepdims=True) / qmax      # [out, 1]
    q = np.clip(np.round(W / scales), -qmax - 1, qmax).astype(np.int8)
    return q, scales.squeeze(1)

def group_symmetric_quantize(W, group_size=128, n_bits=4):
    """W: [out, in],每行切成 in/g 个 group。"""
    qmax = 2 ** (n_bits - 1) - 1
    out, in_ = W.shape
    assert in_ % group_size == 0
    Wg = W.reshape(out, in_ // group_size, group_size)
    scales = np.abs(Wg).max(axis=-1, keepdims=True) / qmax     # [out, in/g, 1]
    q = np.clip(np.round(Wg / scales), -qmax - 1, qmax).astype(np.int8)
    return q.reshape(out, in_), scales.squeeze(-1)             # scales: [out, in/g]
```

> **业界落点**:vLLM 0.6+ 加载 AWQ 模型默认走 Marlin kernel,吞吐比朴素 dequant + GEMM 高 2–3×[^marlin]。如果你在公司跑 vLLM 性能不达标,第一件事是确认 `quantization="awq_marlin"` 而不是 `"awq"`(后者是 fallback 的 Python dequant kernel)。这是非常常见的上线事故。

---

## 5.4 Uniform vs Non-uniform

Uniform 量化(也就是上面整节讲的线性量化)用等距格点。Non-uniform 量化用任意格点位置——典型做法是 K-means 聚类找最优 codebook:

```python
from sklearn.cluster import KMeans

def kmeans_quantize(W, n_bits=4):
    K = 2 ** n_bits
    flat = W.reshape(-1, 1)
    km = KMeans(n_clusters=K, n_init=10).fit(flat)
    Wq = km.cluster_centers_[km.labels_].reshape(W.shape)
    return Wq, km.cluster_centers_.flatten()
```

这就是 Song Han 自己 2016 年 *Deep Compression*[^dc] 论文的做法:迭代剪枝 → K-means 量化 → Huffman 编码,三件套压缩 35×。当年拿了 ICLR best paper。

但 **non-uniform quantization 在 inference infra 里基本死了**,原因只有一个:**没有硬件能直接对 codebook index 做 GEMM**。每次乘法之前必须先查表把 4-bit index 翻译回 FP16,这个查表开销在 GPU 上吞掉所有收益。它现在只活在两个地方:

1. **存储压缩**(不参与计算),例如 LLM 权重的离线传输压缩。
2. **向量数据库的 PQ**(Product Quantization),见下一节。

---

## 5.5 Product Quantization:量化在向量检索里的第二春

把 $d$ 维向量切成 $M$ 段,每段独立做 K-means。总 codebook 大小从 $K^d$ 降到 $M \cdot K^{d/M}$,距离查询可以用查表加加法完成(ADC,Asymmetric Distance Computation[^pq])。

PQ 的主战场不是模型量化,是 **向量检索**。[FAISS](https://github.com/facebookresearch/faiss)、ScaNN、Milvus 的核心索引(IVF-PQ、IVFADC)都建在 PQ 之上。一个 768 维的 BERT embedding,用 PQ 可以压到 ~96 字节,亿级向量库在单机跑得动。做 RAG 或推荐召回的人绕不开它。

最近两年比较新的发展:

- **OPQ** (Optimized PQ):先做正交变换再 PQ,降低子空间相关性。
- **RVQ / RQ** (Residual VQ):多层残差量化,每层量化上一层的残差。Meta [Encodec](https://arxiv.org/abs/2210.13438) 和 Google SoundStream 的音频 tokenizer 都在用。
- **RaBitQ** (SIGMOD 2024):1-bit 的理论最优码本,在 FAISS 里已经合入。

---

## 5.6 一段更接近生产的 QAT 骨架

下面这段加入了 **STE**(Straight-Through Estimator),是 QAT 训练的最小骨架。STE 是 Bengio 2013 年[^ste] 提出来的技巧:反向时把不可导的 round 当成恒等映射。为什么可以这么做,Lec06 会详细讲。

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
        # STE: round 的梯度直通
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
        # per-tensor 对称,简单起见
        self.scale.copy_(self.weight.abs().max() / qmax)

    def forward(self, x):
        qmax = 2 ** (self.n_bits - 1) - 1
        qmin = -qmax - 1
        w_q = FakeQuant.apply(self.weight, self.scale, qmin, qmax)
        return x @ w_q.T + self.bias
```

这段代码本身不会带来推理加速(fake quant 仍在 FP32 里跑),它的作用是让训练过程"感受到"量化误差,把权重往对量化友好的方向调。真正的加速来自把训好的 `weight` 转成 INT8 tensor、塞进 INT8 GEMM kernel——这两步在 PyTorch 里是 [`torch.ao.quantization`](https://docs.pytorch.org/docs/stable/quantization.html) 模块负责的。

---

## 5.7 Infra 实战:框架到底在做什么

Lecture 不讲的一节,但面试和上线都会问。

### vLLM
默认 BF16 / FP16 推理。量化通过 `--quantization {awq, gptq, awq_marlin, gptq_marlin, fp8, compressed-tensors, ...}` 切换。[官方文档](https://docs.vllm.ai/en/latest/features/quantization/index.html)[^vllm] 里列了完整支持矩阵。
- **W4A16** 用 `awq_marlin` 或 `gptq_marlin`,吞吐比朴素方案 2–3×。
- Hopper 上跑 `fp8`,配合 `kv_cache_dtype=fp8` 把 KV cache 也压掉,长 context 收益巨大。
- **W4A8 / W4A4** 走 `compressed-tensors` schema 持续演进,[QServe](https://arxiv.org/abs/2405.04532)[^qserve] 是目前 W4A8 的 SOTA 之一(Han Lab 自己的工作)。

### TensorRT-LLM
NVIDIA 自家方案。原生支持 INT8 SmoothQuant、FP8、INT4 weight-only、FP4(Blackwell)。优势是 kernel fusion 做到极致——dequant / GEMM / bias / activation / residual add 全 fuse 进一个 CUDA kernel。劣势是模型支持速度比 vLLM 慢一拍,build engine 时间长。

### SGLang
介于 vLLM 和 TRT-LLM 之间,量化策略基本对齐 vLLM,但 scheduler 和 [RadixAttention](https://arxiv.org/abs/2312.07104) 让前缀复用场景吞吐更高。Agent / 多轮对话场景值得评测。

### llama.cpp / GGUF
CPU 推理事实标准。它的 [K-quants](https://github.com/ggerganov/llama.cpp/pull/1684)(`Q4_K_M`、`Q5_K_M`、`Q6_K`)是 group quantization 的变体,在 CPU 上手写 SIMD kernel(AVX2 / AVX-512 / NEON)。Mac 和笔记本本地推理几乎都用它。

### 国产硬件
昇腾、寒武纪 MLU、沐曦、摩尔线程都有自己的 INT8 / FP8 路线,kernel 生态远不如 CUDA。在国内 infra 团队常见的任务是把 vLLM / TRT-LLM 的某个 quant kernel "翻译"到目标硬件的算子库上。这种工作的难点 90% 不在数学,在硬件文档不全、工具链不成熟。

---

## 5.8 跨 Lecture 关联

- **前置 ←** Lec02 数值基础、Lec03 剪枝(量化和剪枝在 *Deep Compression* 里是联合使用的)
- **后续 →** Lec06 进阶量化:PTQ vs QAT、calibration 算法(min-max / KL / MSE / percentile)、STE 推导
- **后续 →** Lec13 LLM 部署:SmoothQuant、AWQ、GPTQ、QServe,是这一讲在 LLM 场景的直接延伸
- **横向 ↔** Lec07 NAS:硬件感知的 NAS 经常把"是否能 INT8 化"作为搜索目标之一

---

## 5.9 面试高频题

**Q1. 对称 vs 非对称,怎么选?**
权重几乎永远对称,因为分布以 0 为中心,且对称量化让 GEMM 累加项里的 scale 可以提到外层循环之外,硬件友好。激活看情况:ReLU 之后全正用非对称;GELU / SiLU 有负值可以用对称。LLM 时代更常见的是 weight 对称 + activation 非对称(per-token),或者干脆 weight-only 不动激活。

**Q2. FP16 vs BF16,训练为什么用 BF16?**
BF16 指数位和 FP32 一致(都是 8 bit),动态范围相同,FP32 → BF16 是截断尾数,不会上下溢出,不需要 loss scaling。代价是尾数只有 7 位,相对精度比 FP16 差 8×,但 SGD 本身有梯度噪声不敏感。FP16 训练需要 dynamic loss scaling,工程复杂度高,现在只在小模型和 legacy 代码里出现。

**Q3. Per-tensor / per-channel / per-group 的 trade-off?**
精度递增、硬件成本递增。Per-tensor 在 LLM 上不可用(outlier 撑大 scale),per-channel 是 CNN 时代的事实标准,per-group (g=128) 是现在 W4 LLM 的事实标准。Group 的代价是 GEMM kernel 内层循环要 dequant,需要 Marlin / Machete 这种特化 kernel 才能跑出收益。

**Q4. 为什么 LLM 量化比 CNN 难?**
两个原因。第一,LLM activation 存在 **systematic outlier**——某些 channel 的值持续比其他大 100× 以上(通常出现在 `down_proj` / `o_proj` 的输入上[^llmint8][^smoothquant])。Per-tensor 量化时这些 outlier 撑大 scale,让正常 channel 的有效比特数从 8 掉到 2–3。SmoothQuant 的解法是把 outlier 从 activation 数学等价地迁移到 weight。第二,LLM 的精度对量化误差更敏感,autoregressive 解码让 token 级误差累积,长序列尤其明显。

**Q5. W4 vs W8 vs FP8,生产里怎么选?**
看瓶颈。Decode 阶段 memory-bound,weight 越小越好,W4A16 (AWQ / GPTQ) 吞吐最高,单卡装更大模型。Prefill 阶段 compute-bound,W4 的 dequant 开销吃掉收益,FP8 或 W8A8 (SmoothQuant) 更合适。所以 SOTA 方案(QServe 的 W4A8)是 weight 用 W4 省内存、activation 用 INT8 跑 GEMM 吃算力。Hopper 之后 FP8 更省心,因为原生 Tensor Core 支持,不需要 dequant。

**Q6. 4-bit 模型实际能省多少?**
朴素算 FP16 → INT4 是 4×。但要扣掉:
- **Scale 元数据**。`g=128, W=[out, in]` 的 group 量化,每 128 个 W4 权重配一个 FP16 scale,scale overhead ≈ $\frac{16}{4 \cdot 128} = 3.1\%$。如果还带 zero-point,再加一倍。
- **KV cache**(不单独压的话)。
- **Activation buffer / workspace**。

以 LLaMA-2-7B 为例:FP16 权重 ≈ 13.5 GB,AWQ W4 权重 ≈ 3.9 GB(不是 3.4 GB);加上 scale、KV cache 和激活 buffer,单序列实际显存大概 5–6 GB。如果看到一个 4-bit 模型号称 "exactly 1/4",多半是没算 scale。

---

## 5.10 自检清单

读完这一讲,你应该能不查资料地回答:

1. INT8 乘法比 FP32 乘法省多少能量?比 DRAM 一次访存呢?(参考 5.1 表格)
2. 写出对称量化的 scale 公式,解释为什么 weight 用对称、activation 常用非对称。
3. Per-channel 量化为什么对 weight 可行、对 activation 不可行?
4. BF16 的指数位有几位?为什么这件事对训练这么重要?
5. 一个 `g=128` 的 W4 量化 7B 模型,加 scale 元数据后实际占多少显存?
6. 你的推理框架跑 AWQ 模型时,怎么确认走的是 Marlin kernel 而不是朴素 dequant?

任何一题答不上来,回去看对应小节。

---

## 参考文献

[^horowitz]: M. Horowitz. *1.1 Computing's Energy Problem (and what we can do about it).* ISSCC 2014. [[PDF]](https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf)
[^han231n]: Song Han. *Efficient Methods and Hardware for Deep Learning.* CS231n 2017 Guest Lecture. [[PDF]](https://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture15.pdf) — 45 nm 能耗表在第 23 页,明确标注源自 Horowitz 2014。
[^bf16]: S. Wang & P. Kanwar. *BFloat16: The secret to high performance on Cloud TPUs.* Google Cloud Blog, 2019. [[link]](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
[^fp8]: P. Micikevicius et al. *FP8 Formats for Deep Learning.* arXiv:2209.05433, 2022. [[PDF]](https://arxiv.org/abs/2209.05433)
[^h100ds]: NVIDIA. *H100 Tensor Core GPU Datasheet.* [[PDF]](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet) — H100 SXM dense: FP16/BF16 = 989 TFLOPS,FP8 = 1,979 TFLOPS;sparse 数字翻倍。
[^mx]: Open Compute Project. *OCP Microscaling Formats (MX) Specification v1.0*, 2023. [[PDF]](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
[^b200ds]: NVIDIA. *Blackwell B200 Datasheet.* [[PDF]](https://resources.nvidia.com/en-us-blackwell-architecture/datasheet) — B200 dense FP4 = 9 PFLOPS,sparse FP4 = 18 PFLOPS。
[^cudo]: CUDO Compute. *NVIDIA Blackwell Architecture Breakdown: B100, B200, GB200.* 2024. [[link]](https://www.cudocompute.com/blog/nvidias-blackwell-architecture-breaking-down-the-b100-b200-and-gb200)
[^jacob]: B. Jacob et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR 2018. arXiv:1712.05877. [[PDF]](https://arxiv.org/abs/1712.05877) — TFLite 线性量化的经典参考。
[^gptq]: E. Frantar et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. arXiv:2210.17323. [[PDF]](https://arxiv.org/abs/2210.17323)
[^awq]: J. Lin et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys 2024. arXiv:2306.00978. [[PDF]](https://arxiv.org/abs/2306.00978)
[^smoothquant]: G. Xiao et al. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML 2023. arXiv:2211.10438. [[PDF]](https://arxiv.org/abs/2211.10438)
[^llmint8]: T. Dettmers et al. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS 2022. arXiv:2208.07339. [[PDF]](https://arxiv.org/abs/2208.07339)
[^marlin]: E. Frantar et al. *Marlin: a Mixed-Precision Inference Kernel for Int4 Weight × FP16 Activation.* 2024. [[repo]](https://github.com/IST-DASLab/marlin)
[^dc]: S. Han, H. Mao, W. J. Dally. *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.* ICLR 2016 (Best Paper). arXiv:1510.00149. [[PDF]](https://arxiv.org/abs/1510.00149)
[^pq]: H. Jégou, M. Douze, C. Schmid. *Product Quantization for Nearest Neighbor Search.* TPAMI 2011. [[PDF]](https://hal.inria.fr/inria-00514462v2/document)
[^ste]: Y. Bengio, N. Léonard, A. Courville. *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.* arXiv:1308.3432, 2013. [[PDF]](https://arxiv.org/abs/1308.3432)
[^vllm]: vLLM Documentation. *Quantization.* [[link]](https://docs.vllm.ai/en/latest/features/quantization/index.html)
[^qserve]: Y. Lin et al. *QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving.* MLSys 2025. arXiv:2405.04532. [[PDF]](https://arxiv.org/abs/2405.04532)
