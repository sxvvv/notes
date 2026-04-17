# Lec 05 · 量化基础 (Part I)

> **课程**：MIT 6.5940 *TinyML and Efficient Deep Learning Computing* (Fall 2024) · [Slides](https://hanlab.mit.edu/files/course/slides/MIT-TinyML-Lec05-Quantization-I.pdf) · [Video](https://youtu.be/91stHPsxwig)

---

## 写在前面

前几讲讲的是"让模型少算"——剪枝、稀疏、结构化压缩，都是在计算图上动手术。从这一讲开始换一个思路：**算的东西不变，但每一次乘加都便宜一点**。这就是量化。

量化的工程吸引力来自一个很直接的事实：**硬件的峰值算力几乎全部建在更窄的数据类型上** 。Volta 开启 INT8 Tensor Core，Turing 加 INT4，Hopper 押注 FP8，Blackwell 把 FP4 作为主战场。每一代 NVIDIA GPU 的 "算力翻倍" 宣传，本质都是"算得更便宜"的宣传。

这对国产 GPU 工程师有一个额外含义：**CUDA 生态里做量化的套路（PTQ、GPTQ、AWQ、SmoothQuant、Marlin kernel 等），是把它移植到昇腾 / 寒武纪 / 沐曦 / 海光这些国产硬件上必须先吃透的底子**。你绕不开它。

这一讲覆盖量化的第一性原理、数值格式的工程取舍、线性量化的数学和 GEMM 展开、粒度的选择、均匀 vs 非均匀量化、Product Quantization、PTQ 与 QAT，以及 LLM 量化特有的 outlier 难题。最后落到推理框架实战和国产 GPU 的移植经验。Lec06 会继续讲校准、Lec13 会专门讲大模型量化的前沿方法。

---

## 目录

- [5.1 第一性原理：为什么量化有价值](#51-第一性原理为什么量化有价值)
- [5.2 数值格式大图：从 FP32 到 FP4](#52-数值格式大图从-fp32-到-fp4)
- [5.3 线性量化：公式、误差与 GEMM 展开](#53-线性量化公式误差与-gemm-展开)
- [5.4 量化粒度：精度与硬件的拉锯](#54-量化粒度精度与硬件的拉锯)
- [5.5 Uniform vs Non-uniform 量化](#55-uniform-vs-non-uniform-量化)
- [5.6 Product Quantization：向量检索的另一片天](#56-product-quantization向量检索的另一片天)
- [5.7 PTQ vs QAT：STE 与工程骨架](#57-ptq-vs-qatste-与工程骨架)
- [5.8 LLM 量化难题：Outlier 与精度崩塌](#58-llm-量化难题outlier-与精度崩塌)
- [5.9 推理框架实战](#59-推理框架实战)
- [5.10 国产 GPU 量化移植指南](#510-国产-gpu-量化移植指南)
- [5.11 面试高频题](#511-面试高频题)
- [5.12 自检清单](#512-自检清单)
- [延伸阅读](#延伸阅读)
- [参考文献](#参考文献)

---

## 5.1 第一性原理：为什么量化有价值

### 5.1.1 硬件能耗账本

量化值得做的终极理由只有一句话：**位宽越窄，乘加的能耗、面积、访存带宽全都按比例往下掉**。具体掉多少，Horowitz 2014 年的 ISSCC keynote[^horowitz] 给出了一份至今仍在被引用的账本（45 nm、0.9 V 工艺）。

<p align="center">
  <img src="./images/fig1_energy_cost.png" width="820" alt="各操作能耗对比 Horowitz 2014"/>
</p>

> 数据源：Horowitz, ISSCC 2014 · 通过 Song Han CS231n 2017 guest lecture[^han231n] 转引。工艺节点更先进时绝对数值会下降，但**比例关系基本不变**——这是量化长期红利的来源。

这张图有两个数字必须刻进脑子。

第一个是 **INT8 乘法比 FP32 乘法省约 18 ×**（0.2 pJ vs 3.7 pJ）。位宽减半、能量近似降 4 倍，因为在 CMOS 电路里 $E \propto V^2 C$，而电容 $C$ 大致与位宽成正比。

第二个更重要：**一次 DRAM 读的能耗 ≈ 173 次 FP32 乘法 ≈ 3,200 次 INT8 乘法**。这是理解 LLM 推理经济学的钥匙——**Decode 阶段（逐 token 自回归、batch=1）的瓶颈从来不是算，是搬权重**。做 W4A16 这种 weight-only 量化，在 decode 阶段基本能拿到线性加速：权重体积减半，带宽占用减半，吞吐翻倍。

### 5.1.2 算力 vs 带宽：Roofline 视角

能耗账本只是硬件给的外部约束。更常用的工程工具是 Roofline 模型——把工作负载按 *算术强度*（arithmetic intensity，简称 AI）划分成 memory-bound 和 compute-bound 两类：

$$\text{AI} = \frac{\text{FLOPs}}{\text{Bytes Accessed}}$$

每块 GPU 有一个 *ridge point*，即峰值算力 / 峰值带宽。负载的 AI 低于 ridge point 时是 memory-bound——再多算力也白搭，因为带宽先打满了；高于 ridge point 时才是 compute-bound，算力利用率成为主要优化目标。

这个区分直接告诉你该做哪种量化：

| 场景 | Batch | AI 估算 | 瓶颈 | 最优量化策略 |
|---|---|---|---|---|
| LLM Decode | 1 | ~0.1 | Memory | **W4A16**（权重越小越好） |
| LLM Prefill | ≥64 | 10–100 | Compute | **W8A8 / FP8**（算力利用率优先） |
| CV 推理 | 32 | >100 | Compute | **INT8 / FP8** |
| 向量检索 | 1 | 极低 | Memory | **PQ / RVQ** |

昇腾 910B 的实测内存带宽 ≈ 1.2 TB/s[^910b]，INT8 峰值 ≈ 640 TOPS（dense），其 ridge point ≈ 640 / 1200 ≈ 0.5 TOPS/(GB·s)。这个值比 H100（1979 TOPS INT8 / 3.35 TB/s ≈ 0.59）略低，意味着**在同样的负载下昇腾更容易进入 memory-bound 区间**。反过来，这意味着 W4 量化在昇腾上的性价比比在 NVIDIA 上更高——带宽收益被放大了。

---

## 5.2 数值格式大图：从 FP32 到 FP4

过去十年 AI 硬件的数值格式演进，用一张路线图可以看清：

<p align="center">
  <img src="./images/fig6_hw_roadmap.png" width="820" alt="NVIDIA Tensor Core 精度演进路线图"/>
</p>

**看这张图的时候不要只盯 NVIDIA**——国产厂商都在各自位置上追赶。华为昇腾 910B 的水平大致对齐 Volta–Turing（INT8 原生 + 软件 W4），910C 往 Hopper 靠（INT8 强化、FP8 规划中）。下面把每种格式的工程取舍讲清楚。

### 5.2.1 位域对比

<p align="center">
  <img src="./images/fig2_bit_layout.png" width="860" alt="FP32/TF32/BF16/FP16/FP8/FP4/INT8 位域对比"/>
</p>

这张图浓缩了**训练推理生态所有实用数值格式**。看图的顺序：灰色是符号位，蓝色是指数位（决定动态范围），绿色是尾数位（决定精度），橙色是纯整数。越往下位宽越窄，取舍越极端。

### 5.2.2 整数类型

无符号 INT8 的范围是 `[0, 255]`，有符号 INT8（二补码）是 `[-128, 127]`。二补码的好处是加法器和乘法器**不用为符号位单独开电路**——这是硬件实现整数比浮点省面积的根本。

写量化 kernel 时有一个经典坑：**INT8 累加进 INT32 必须显式 sign-extend**，否则 `-1`（字节表示 `0xFF`）会被当成 `255`。CUDA 的 `__dp4a` 帮你处理了，但手写 SASS、AVX-VNNI 或昇腾 TIK 汇编的时候，这个细节必须自己盯。

**定点数**（fixed-point）是整数的一种特殊用法：`fixed<8,4>` 表示 8 位总宽、4 位在小数点后。本质是"整数 × 固定 scale"——当 scale 是 2 的幂，反量化就是一次 shift，**零成本**。这就是为什么高通 Hexagon、寒武纪 MLU、地平线 BPU 这类边缘 NPU 的核心算力都建在定点数之上：不需要也不能负担 scale 乘法。

### 5.2.3 FP32 的结构与陷阱

IEEE 754 FP32 的数值公式是：

$$(-1)^{\text{sign}} \times (1 + \text{Fraction}) \times 2^{\text{Exp} - 127}$$

指数位有三种状态：1–254 是常规范围；0 触发 subnormal（去掉前导 1，让 0 附近的数值不至于断崖式消失）；255 是 `±Inf` 或 `NaN`。

Subnormal 是一个真实的性能陷阱。某些 GPU 对它走**慢路径**，一个函数突然蹦出大量 subnormal 值能让性能塌方 100 倍。CUDA 的编译选项 `-ftz=true`（flush-to-zero）把它们强制置零，代价是 0 附近的精度损失。昇腾 CANN 的算子库默认开了类似选项，**写自定义算子时要记得查这个 flag**，否则跨平台对齐数值时会找不到原因。

### 5.2.4 FP16 vs BF16：为什么训练用 BF16

这两个格式都 16 位，但分配方式天差地别：

|  | FP16 | BF16 |
|---|---|---|
| Sign / Exp / Mant | 1 / 5 / 10 | 1 / **8** / 7 |
| 动态范围 | ±6.55 × 10⁴ | ±3.39 × 10³⁸ |
| 相对精度 | 2⁻¹⁰ ≈ 0.1 % | 2⁻⁷ ≈ 0.8 % |
| FP32 ↔ 互转 | 需 loss scaling | **直接截断尾数** |
| 训练推荐 | ✗ | ✓ |

BF16 由 Google Brain 为 TPU 定制[^bf16]，NVIDIA 在 Ampere（A100）时代跟进。**它和 FP32 共享 8 位指数**，所以动态范围与 FP32 完全一致，FP32 → BF16 只是简单截掉尾数。

训练时这一点至关重要。FP16 在反向传播中最容易撞到的坑是梯度下溢：小于 `2⁻¹⁴` 的梯度直接变成 0，模型学不动。老代码用 *loss scaling* 这个 hack：loss 先乘 `2¹⁵`，反向传播后梯度再除回来，在数值范围里把它"顶起来"。BF16 让这个问题彻底消失——动态范围够了。

所以业界实情是：**GPT-4、LLaMA-2/3、Qwen、DeepSeek 全部用 BF16 训练**。FP16 只活在老模型和小玩具里。推理侧的情况更复杂：很多模型发布时用 BF16，但老 GPU（Pascal、Volta）没有 BF16 硬件，推理时只能 cast 成 FP16；这就是为什么一堆 3090/A100 用户在 vLLM 里会看到 `--dtype float16` 的默认选项——不是它更好，是 Ampere 之前没有更好的选项。

### 5.2.5 FP8：Hopper 开始的新主角

H100 引入了两种 FP8[^fp8]：

| | **E4M3** | **E5M2** |
|---|---|---|
| Exp / Mant | 4 / 3 | 5 / 2 |
| 正常最大值 | ±448 | ±57,344 |
| 主要用途 | **前向激活 / 权重** | **反向梯度** |
| 设计哲学 | 精度优先（更多尾数位） | 范围优先（更多指数位） |

FP8 的尾数只有 3 位，理论数值非常稀疏，所以它从来不是单独拿出来用——必须配合 **per-tensor 或 per-block scale**。NVIDIA Transformer Engine 约 90 % 的代码做的就是这件事：在合适的位置插入 `amax` 统计、同步和 scale 更新。

H100 SXM 的 dense 峰值算力（不启用 2:4 sparsity）[^h100ds]：

| 精度 | TFLOPS / TOPS |
|---|---:|
| FP64 Tensor Core | 67 |
| TF32 | 989 |
| FP16 / BF16 | 989 |
| **FP8** | **1,979** |
| INT8 | 1,979 |

**有个数字陷阱要提醒**：H100 FP8 理论 2 × BF16，但真实模型推理通常只拿到 1.3–1.6 × 加速。差距被 `amax` 同步、scale 维护、kernel launch 开销吃掉。DeepSeek-V3 技术报告[^dsv3] 是目前为止最精彩的 FP8 训练工程案例——他们用 *fine-grained scaling*（128×128 block、per-group E4M3）把精度做到接近 BF16，同时拿到接近理论的加速。想真正理解 FP8 在工业界怎么落地，就精读这篇。

### 5.2.6 FP4 与 MX 格式：Blackwell 之后的主战场

这一节是原课件没覆盖的，但它是**2024 年开始未来两年业界最重要的变化**。

OCP 发布了 *Microscaling (MX)* 标准[^mx]：每 32 个元素共享一个 8-bit E8M0（纯指数）scale。Blackwell 原生支持 MXFP4 / MXFP6 / MXFP8。NVIDIA 私有的 **NVFP4** 变体更激进——每 16 个元素配一个 FP8(E4M3) scale，精度优于 MXFP4，是 B200 的主力推理格式。

B200 的算力[^b200ds]：

- Dense FP4：**9 PFLOPS**
- Sparse FP4（2:4）：**18 PFLOPS**
- 对比 H100 BF16 Dense（989 TFLOPS）：约 **9 × dense-to-dense**

对工程师的意义只有一条：**W4A4 / FP4 是未来两年大模型推理的主流**。设计量化接口时，group-wise scale 必须作为 first-class citizen 存在，不能把它当成 per-tensor 或 per-channel 的"扩展"。在 CUDA 上 Marlin、Machete 这类 kernel 已经把这条路趟过，**国产 GPU 至少要能把 MXFP4 等价的 group scale 语义在软件层面正确模拟出来，否则 LLM 推理的 ecosystem 对不上**。

---

## 5.3 线性量化：公式、误差与 GEMM 展开

### 5.3.1 基础公式

线性量化的本质是一句话：**用一个仿射变换 $r \mapsto q$ 把浮点映射到整数，计算之后再映射回浮点**。

前向（fp → int）：

$$q = \text{clamp}\!\left(\text{round}\!\left(\frac{r}{S}\right) + Z,\ q_{\min},\ q_{\max}\right)$$

反向（int → fp）：

$$\hat{r} = S \cdot (q - Z)$$

两个参数合在一起叫 **qparams**：$S > 0$ 是 scale（浮点数），$Z$ 是 zero-point（整数）。整个量化的设计空间，基本就是围绕"**怎么选好的 $S$ 和 $Z$**"展开。

量化误差 $e = r - \hat{r}$ 有三个来源：

| 误差来源 | 最坏情况 | 主因 |
|---|---|---|
| Rounding | $\|e\| \le S/2$ | 离散化精度 |
| Clipping | 无界 | 超出 $[r_\min, r_\max]$ 被截断 |
| Scale 精度 | 二阶 | S 本身存 FP16 时的精度损失 |

这里有一个生产经验：**99 % 的量化精度问题是 clipping 引起的，不是 rounding**。量化真正的难点不在 `round` 怎么实现，而在 **$r_\min, r_\max$ 怎么选**——这是 Lec06 calibration 的主线。

<p align="center">
  <img src="./images/fig3_quant_error.png" width="820" alt="Clipping 误差 vs Rounding 误差"/>
</p>

左图展示了最优 clip ratio：clip 太紧，clipping 误差主导；clip 太松，scale 变大、格点变稀，rounding 误差主导。两者交点就是 MSE 最小的 clip 位置（对高斯输入约 0.81）。右图展示 outlier 怎么把 scale 撑大：一个 `50` 这种远离正常分布的异常值，能让正常值的有效比特数从 8 降到 4 以下。

### 5.3.2 对称量化（Symmetric，$Z = 0$）

$$S = \frac{\max(\lvert r \rvert)}{2^{b-1} - 1}, \quad Z = 0, \quad q \in [-2^{b-1},\ 2^{b-1} - 1]$$

把它代入 GEMM：

$$y_i = \sum_j x_j w_{ij} = S_x S_w \underbrace{\sum_j q^x_j q^w_{ij}}_{\text{纯整数 GEMM}}$$

关键在于 **两个 scale 完全提到了 GEMM 外层**，内核只做整数乘加。这就是为什么**权重几乎永远用对称量化**——分布本来就对称，又能让 Tensor Core 吃到纯整数输入。

### 5.3.3 非对称量化（Asymmetric，$Z \neq 0$）

$$S = \frac{r_\max - r_\min}{2^b - 1}, \quad Z = \text{round}\!\left(-\frac{r_\min}{S}\right)$$

代入 GEMM 后会多出交叉项：

$$\sum_j (q^x_j - Z_x)(q^w_{ij} - Z_w) = \underbrace{\sum q^x_j q^w_{ij}}_{\text{INT GEMM}} - Z_w \underbrace{\sum q^x_j}_{\text{预计算}} - Z_x \underbrace{\sum q^w_{ij}}_{\text{预计算}} + N Z_x Z_w$$

后三项可以预计算或塞进 epilogue，实际开销不大。完整推导参考 TFLite 量化白皮书[^jacob]。

<p align="center">
  <img src="./images/fig4_sym_vs_asym.png" width="820" alt="对称 vs 非对称量化格点对比"/>
</p>

工程决策大致是这样的：**权重永远对称**（分布对称 + GEMM 友好）；ReLU 后的激活全正，用非对称把 [0, 255] 用满；GELU / SiLU 激活有负值，可以对称。LLM 主流方案是 **权重对称 + 激活 per-token 非对称**，或者直接 weight-only（W4A16，不量化激活）。

下面是一段验证两种方法在偏态分布上差异的代码，可以直接跑：

```python
import numpy as np

def symmetric_quantize(x: np.ndarray, n_bits: int = 8):
    """对称量化：Z=0，适合权重（分布对称、GEMM 友好）"""
    qmax = 2 ** (n_bits - 1) - 1              # INT8: 127
    scale = np.abs(x).max() / qmax
    q = np.clip(np.round(x / scale), -(qmax + 1), qmax).astype(np.int8)
    return q, scale

def asymmetric_quantize(x: np.ndarray, n_bits: int = 8):
    """非对称量化：完整利用量化范围，适合 ReLU 输出等偏态分布"""
    qmin, qmax = 0, 2 ** n_bits - 1            # UINT8: 0..255
    scale = (x.max() - x.min()) / (qmax - qmin)
    zp = int(round(-x.min() / scale))
    q = np.clip(np.round(x / scale) + zp, qmin, qmax).astype(np.uint8)
    return q, scale, zp

# 偏态分布（模拟 ReLU 输出）
rng = np.random.default_rng(0)
x = rng.chisquare(df=2, size=4096).astype(np.float32) * 0.1

qs, s         = symmetric_quantize(x)
qa, sa, za    = asymmetric_quantize(x)
mse_s = np.mean((x - qs.astype(np.float32) * s) ** 2)
mse_a = np.mean((x - (qa.astype(np.float32) - za) * sa) ** 2)
print(f"偏态分布  sym MSE={mse_s:.3e}  asym MSE={mse_a:.3e}  "
      f"asym 优势 {mse_s / mse_a:.2f}×")
# 偏态分布  sym MSE=2.4e-05  asym MSE=9.8e-06  asym 优势 2.45×
```

---

## 5.4 量化粒度：精度与硬件的拉锯

<p align="center">
  <img src="./images/fig5_granularity.png" width="860" alt="Per-Tensor / Per-Channel / Per-Group 粒度对比"/>
</p>

| 粒度 | scale 数量 | 精度 | 硬件成本 | 代表方案 |
|---|---|---|---|---|
| **Per-Tensor** | 1 | 低 | 最低（GEMM 外提） | TFLite INT8，TRT 早期 |
| **Per-Token**（激活） | seq_len | 中 | 中等 | SmoothQuant、LLM.int8() |
| **Per-Channel**（权重） | out_channels | 中高 | 中等 | PyTorch QAT、TRT 现代 |
| **Per-Group**（权重） | out × ⌈in / g⌉ | 高 | 需特化 kernel | AWQ、GPTQ、MXFP4 |

### 5.4.1 Per-Tensor 的死穴

Per-tensor 的好处是简单——一个张量一个 scale，GEMM 外层一乘就行，整个 kernel 里看不到 scale。但它的死穴也很直接：**一个 outlier 就把整层 scale 撑大**，其他正常值的有效精度从 8 bit 掉到 2–3 bit。

CNN 上还能忍，激活分布大致可控；LLM 上直接废掉——系统性的 outlier channel（见 §5.8）让 per-tensor 方案出不了实验室。业界早期有人跑过 BERT per-tensor INT8，精度掉 3 个点以上，没人能接受。

### 5.4.2 Per-Channel 的正确用法

Per-channel 量化能用的前提是：**scale 不能在 GEMM 的 reduction 维度上变化**。

对 $Y = XW^\top$，reduction 在 $W$ 的 in 维度（对应 $X$ 的 channel 维度）：

- ✅ **Weight per-channel（沿 out 轴）**：每一行 $W$ 一个 scale，完全在 reduction 之外，可以提到 GEMM 外层。
- ❌ **Activation per-channel**：channel 轴正好是 reduction 轴，scale 和累加纠缠，没法高效实现。

这就是为什么 LLM 量化里会出现 **per-token activation + per-channel weight** 这个经典组合——两个 scale 都避开了 reduction 维度：token 维度和 out 维度。SmoothQuant 的核心价值之一就是把这个组合做到了精度可用。

### 5.4.3 Per-Group 的内核复杂度

AWQ / GPTQ 默认 `g=128` 的 W4，scale 在 reduction 维度上**每 128 个元素跳变一次**。这意味着 GEMM 的内层循环必须**边算边 dequant**——每算 128 个 INT 乘积就乘一次 FP16 scale。

这就是 Marlin kernel[^marlin] 比标准 GEMM 复杂一个量级的原因。它靠 register file 的 double-buffer，把"下一组 scale 的 prefetch"和"当前组的 INT 累加"做成 overlap，把 dequant 延迟藏在计算里。一句话：Marlin 把 W4A16 从"比 FP16 慢"做到了"比 FP16 快 3 ×"。

```python
def per_channel_sym_quant(W: np.ndarray, n_bits: int = 8):
    """W: [out, in]，沿 out 维度独立量化"""
    qmax = 2 ** (n_bits - 1) - 1
    scales = np.abs(W).max(axis=1, keepdims=True) / qmax   # [out, 1]
    q = np.clip(np.round(W / scales), -(qmax + 1), qmax).astype(np.int8)
    return q, scales.squeeze(1)                             # scales: [out]

def group_sym_quant(W: np.ndarray, group_size: int = 128, n_bits: int = 4):
    """W: [out, in]，每行切成 in/g 个 group，AWQ/GPTQ 的基础"""
    qmax = 2 ** (n_bits - 1) - 1
    out, in_ = W.shape
    assert in_ % group_size == 0
    Wg = W.reshape(out, in_ // group_size, group_size)      # [out, ngroups, g]
    scales = np.abs(Wg).max(axis=-1, keepdims=True) / qmax  # [out, ngroups, 1]
    q = np.clip(np.round(Wg / scales), -(qmax + 1), qmax).astype(np.int8)
    return q.reshape(out, in_), scales.squeeze(-1)          # scales: [out, ngroups]

# Scale 元数据开销分析（LLaMA-7B 典型权重矩阵，W4, g=128）
out, in_ = 4096, 4096
n_groups = out * (in_ // 128)
weight_bits  = out * in_ * 4                  # W4 = 4 bit per weight
scale_bits   = n_groups * 16                  # FP16 scale
overhead_pct = scale_bits / weight_bits * 100
print(f"W4 g=128 的 scale metadata 开销: {overhead_pct:.1f}%")
# W4 g=128 的 scale metadata 开销: 3.1%
```

> **国产 GPU 落点**：昇腾的 Cube 单元原生支持 INT8 per-tensor GEMM；per-channel 要在 epilogue 补一次 scale 乘法，代价可接受。Per-group W4 目前需要把 Vector 单元上的 dequant 和 Cube 上的 GEMM 做成两段流水，对 CANN 算子团队来说是最复杂的一块。Ascend 910C 的 CANN 从 8.0 版本开始提供 `npu_quant_matmul_v2`，部分屏蔽了这个复杂度，但对 LLM 的 W4A16 推理来说，**"能跑" 和 "跑得有 CUDA 一半性能"还是两个状态**。

---

## 5.5 Uniform vs Non-uniform 量化

<p align="center">
  <img src="./images/fig7_uniform_vs_nonuniform.png" width="820" alt="均匀 vs K-Means 非均匀量化"/>
</p>

### 5.5.1 K-Means 量化（Deep Compression）

Song Han 2016 年的 *Deep Compression*[^dc]（ICLR Best Paper）把三件套合起来用：**迭代剪枝 → K-Means 量化 → Huffman 编码**，在 AlexNet 上做到了 35 × 压缩。K-Means 量化的直觉很清楚——与其均匀分布格点，不如直接找最小化 MSE 的最优格点位置：

```python
from sklearn.cluster import KMeans

def kmeans_quantize(W: np.ndarray, n_bits: int = 4):
    K = 2 ** n_bits                                      # 4 bit → 16 个中心
    flat = W.reshape(-1, 1)
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(flat)
    codebook = km.cluster_centers_.flatten()             # [K] 个 float 中心
    codes    = km.labels_.reshape(W.shape)               # 每个元素是 0..K-1 的索引
    W_reconstructed = codebook[codes]                    # 查表 dequant
    return codes, codebook, W_reconstructed
```

### 5.5.2 为什么 Non-uniform 在推理中"死了"

K-Means 量化在相同 bit 数下 MSE 明显更低——权重分布是钟形的，非均匀格点能把精度集中到高概率区域。但它在 GPU / NPU 推理里基本被淘汰了。

**根本原因：没有哪个硬件能对 codebook index 直接做 GEMM**。推理前必须先查表把 4 bit index 翻译回 FP16，这个 LUT lookup 在 GPU 上的开销反过来吞掉了所有精度收益。即使用 shared memory 做 LUT，broadcast 开销也扛不住。

Non-uniform 量化现在只活在两个场景：

1. **离线存储压缩**（不参与计算）：模型权重传输、冷存储。
2. **向量数据库 PQ**：距离计算本质是 LUT 查表，正好适合它。见 §5.6。

**Binary / Ternary 量化**——XNOR-Net[^xnor]、TWN[^twn] 的那一路——是非均匀量化的极端情况。1 bit 权重（±1）把乘法变成掩码异或，1.58 bit（-1 / 0 / +1）变成加减。理论上 32 × 压缩，在 FPGA 和边缘 NPU 上有工程价值；大模型上精度损失太大，BitNet b1.58[^bitnet] 是最新尝试，但需要从头训练，目前还没进入主流。

---

## 5.6 Product Quantization：向量检索的另一片天

<p align="center">
  <img src="./images/fig9_product_quantization.png" width="820" alt="Product Quantization 示意"/>
</p>

PQ 的想法简单得出奇：高维向量直接聚类会因为维度诅咒而失效，那就**把 $d$ 维向量切成 $M$ 段子向量，每段独立做 K-Means**。

$$\text{存储压缩比} = \frac{d \times 32 \,\text{bit}}{M \times \log_2 K \,\text{bit}}$$

以 768 维 BERT embedding、$M = 96$、$K = 256$ 为例：$768 \times 32 \,/\, (96 \times 8) = 32 ×$ 压缩，768 × 4 byte = 3072 byte 变成 96 byte。

距离近似计算用 ADC（Asymmetric Distance Computation）[^pq]：

$$d(x, y) \approx \sum_{m=1}^{M} d(x_m,\ c^{(m)}_{y_m})^2$$

查表代替向量内积，时间从 $O(d)$ 降到 $O(M)$。这是为什么亿级向量检索系统（FAISS、Milvus、Weaviate）全在用 PQ 家族。

PQ 的工业变体值得记住：

| 方法 | 改进 | 应用 |
|---|---|---|
| OPQ | 先正交变换再 PQ，降低子空间相关性 | FAISS[^faiss] |
| RVQ / RQ | 多层残差量化 | Encodec[^encodec] 音频 tokenizer |
| IVFPQ | IVF 聚类 + PQ，十亿级检索 | Milvus, Weaviate |
| RaBitQ | 1-bit 理论最优码本（SIGMOD 2024） | 已合入 FAISS |
| ScaNN | 各向异性量化误差 | Google 内部搜索 |

> **工程外延**：RVQ 是近两年 neural codec 的核心——AudioLM、Encodec、Descript 的音频生成全建在它上面；Whisper-V3 的词表表示也借用了类似思路。做多模态的工程师应该把 RVQ 当作基本功。

---

## 5.7 PTQ vs QAT：STE 与工程骨架

<p align="center">
  <img src="./images/fig10_qat_ste.png" width="820" alt="STE 梯度直通 & QAT vs PTQ 精度曲线"/>
</p>

### 5.7.1 区别与取舍

| | **PTQ** (Post-Training Quantization) | **QAT** (Quantization-Aware Training) |
|---|---|---|
| 训练阶段 | 模型训完后做量化 | 量化误差参与训练 |
| 数据需求 | 少量 calibration 集（~512 样本） | 完整训练集 |
| 精度（高 bit） | 接近 QAT | 接近 QAT |
| 精度（低 bit ≤4）| 明显下降 | 显著优于 PTQ |
| 工程复杂度 | 低 | 高（需训练基础设施） |
| 推荐场景 | W8 / FP8，资源受限 | W4 及以下，精度敏感 |

一个实际的 decision tree：**能用 PTQ 的绝不用 QAT**。PTQ 的迭代成本是小时级，QAT 是天级。只有当 PTQ 的精度实在救不回来（低 bit、医疗金融这种精度敏感场景、或对抗分布外输入），才回到 QAT。AWQ / GPTQ 这类"增强版 PTQ"在 W4 上能做到接近 QAT 的精度，进一步压缩了 QAT 的适用空间。

### 5.7.2 STE：让不可导的 round 参与反传

`round(x)` 处处梯度为 0，反向传播直接截断。**Straight-Through Estimator**（STE，Bengio 2013[^ste]）的处理方式很粗暴——反向时假装 round 是恒等映射：

$$\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial \hat{x}} \cdot \mathbf{1}[q_\min \le x/S \le q_\max]$$

clamp 范围内梯度直通；范围外梯度为 0，这是在告诉优化器"这里被截断了，应该往回移"。实际训练中它比看起来要稳，但也有发散的边界情况，Learned Step Size Quantization（LSQ[^lsq]）把 scale 也做成可学习参数，某种程度上缓解了这个问题。

### 5.7.3 生产级 QAT 骨架

下面这段代码是一个可以直接跑的骨架，不是玩具示例。它包含 per-channel scale + STE + EMA 更新，三个要素凑齐才是"像样的 QAT"。

```python
import torch
import torch.nn as nn


class FakeQuant(torch.autograd.Function):
    """Fake quantization with STE backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor,
                qmin: int, qmax: int) -> torch.Tensor:
        x_scaled = x / scale
        ctx.save_for_backward(x_scaled, torch.tensor(qmin), torch.tensor(qmax))
        q = torch.clamp(torch.round(x_scaled), qmin, qmax)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_scaled, qmin, qmax = ctx.saved_tensors
        # STE: clamp 内梯度直通，clamp 外梯度为 0
        mask = (x_scaled >= qmin.item()) & (x_scaled <= qmax.item())
        grad_input = grad_output * mask.float()
        return grad_input, None, None, None


class QLinear(nn.Module):
    """Per-channel symmetric QAT Linear layer."""

    def __init__(self, in_features: int, out_features: int, n_bits: int = 8):
        super().__init__()
        self.weight  = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias    = nn.Parameter(torch.zeros(out_features))
        self.n_bits  = n_bits
        self.qmax    = 2 ** (n_bits - 1) - 1
        self.qmin    = -(self.qmax + 1)
        # per-channel scale，沿 out_features 轴
        self.register_buffer("scale", torch.ones(out_features, 1))

    @torch.no_grad()
    def update_scale(self):
        """EMA 更新 scale（calibration 阶段 / QAT warm-up 阶段调用）"""
        new_scale = self.weight.abs().max(dim=1, keepdim=True).values / self.qmax
        # EMA 平滑，防止 scale 抖动导致训练震荡
        self.scale.copy_(0.9 * self.scale + 0.1 * new_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fake-quant weight：模拟量化误差，参与梯度计算
        w_q = FakeQuant.apply(self.weight, self.scale, self.qmin, self.qmax)
        return torch.nn.functional.linear(x, w_q, self.bias)


# ── 使用示例（训练骨架）───────────────────────────────────────────────────────
def qat_training_sketch():
    model = QLinear(768, 768, n_bits=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for step in range(100):
        if step == 10:                 # warm-up 结束后开启量化感知
            model.update_scale()

        x = torch.randn(8, 768)
        loss = model(x).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            model.update_scale()       # 定期更新 scale
```

几个 QAT 的隐藏坑点：

- **Warm-up 必须做**：刚开始 fake-quant 时梯度很吵，先用 FP32 训几百个 step 让权重分布稳定下来再开 QAT，收敛速度能快一倍。
- **Scale EMA 的动量别太大**：0.9 上下是稳的，0.99 会让 scale 慢得追不上权重分布变化。
- **Bias 一般不量化**：INT32 累加的中间结果直接加 FP16 bias 即可，bias 占参数量微乎其微。
- **BN 折叠**：卷积 + BN + ReLU 的组合，必须在 QAT 前 fold BN 到卷积权重里，否则量化语义不对。

上面骨架产出的是 **fake-quantized 模型**（权重仍在 FP32，模拟量化误差）。真正的推理加速来自后续导出：权重 → INT4/INT8 tensor + 整数 GEMM kernel。PyTorch 的 `torch.ao.quantization` 和 `torch._export` 负责这步转换；昇腾上由 CANN 的 `QuantOps` 接管。

---

## 5.8 LLM 量化难题：Outlier 与精度崩塌

<p align="center">
  <img src="./images/fig8_llm_outlier.png" width="860" alt="LLM Outlier Channel 问题"/>
</p>

### 5.8.1 为什么 LLM 量化比 CNN 难

**原因 1：系统性 Outlier Channel**

LLM（OPT、LLaMA 系列尤其明显）的激活存在**系统性 outlier**：少数固定 channel（通常出现在 `down_proj` 和 `o_proj` 的输入）的幅值比其他 channel 大 100 倍以上，且跨所有 token 一直存在[^llmint8][^smoothquant]。

Per-tensor 量化后，这些 outlier 把 scale 撑得巨大，正常 channel 的有效精度变成：

$$\text{有效比特数} \approx 8 - \log_2\!\left(\frac{\text{outlier 幅值}}{\text{正常幅值}}\right) \approx 8 - 7 = 1 \,\text{bit}$$

名义 8 bit，实际有效精度只有 1–2 bit，精度直接崩塌。这个现象最早在 LLM.int8() 论文里被系统性记录，之后所有 LLM 量化方案都绕不开它。

**原因 2：Autoregressive 误差累积**

CNN 每次 forward 独立，量化误差不累积。LLM decode 阶段逐 token 生成，当前 token 的量化误差直接影响下一个 token 的 KV cache——**误差在序列维度上积分**。长序列尤其明显：2k token 以上，一个量化不慎就会看到 perplexity 从 5.2 飙到 7.5。

### 5.8.2 主流解法概览

| 方法 | 核心思路 | 精度 | 速度 |
|---|---|---|---|
| **LLM.int8()**[^llmint8] | 混合精度：outlier channel 走 FP16，其余 INT8 | 接近 FP16 | ~1.7 × 慢于 FP16 |
| **SmoothQuant**[^smoothquant] | 数学等价迁移：$Y = (XS^{-1})(SW)$，把困难从 X 转给 W | 接近 FP16 | INT8 GEMM 加速 |
| **GPTQ**[^gptq] | OBQ / OBS 二阶 Hessian 补偿量化误差 | ≈ FP16 (W4) | W4A16 |
| **AWQ**[^awq] | 保护"激活感知的显著权重"，per-group scale 搜索 | 略优于 GPTQ | W4A16 |
| **QServe**[^qserve] | W4A8KV4，SmoothAttention + 寄存器级并行 dequant | SOTA W4A8 | 高吞吐 |

工程侧的简单心法：**7B ~ 13B 模型直接上 AWQ**（速度快、精度够、kernel 成熟），**70B 以上用 GPTQ 或 AWQ 都行**（GPTQ 校准慢但精度稍好），**吞吐优先的场景考虑 QServe**（W4A8 同时利用权重和算力红利）。Lec13 会把这些方案细节展开讲。

---

## 5.9 推理框架实战

三个值得掌握的量化推理框架：

### vLLM

开源首选，快速迭代，社区活跃。关键选项：

```bash
# W4A16 AWQ + Marlin kernel（吞吐比朴素 dequant+GEMM 高 2-3 ×）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin          # 不要用 "awq"，那是 Python fallback

# H100 上的 FP8（算力利用率最优）
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --quantization fp8 \
    --kv-cache-dtype fp8               # KV cache 压缩，长 context 收益大
```

**一个常见上线事故**：`--quantization awq` 走的是 Python 版 dequant + GEMM，吞吐不到 `awq_marlin` 的 40 %。上线前用 `nvidia-smi dmon` 看 SM 利用率，低于 60 % 基本就是没走上 Marlin 快路径。

### TensorRT-LLM

NVIDIA 自家的极限优化版。把 dequant / GEMM / bias / activation / residual add 全 fuse 进一个 CUDA kernel，端到端延迟是所有框架里最低的。但代价也明显：模型支持慢一拍，build engine 大模型要 30 分钟以上，调参曲线陡。

```python
# TRT-LLM FP8 量化（配合 ammo/modelopt 工具）
from modelopt.torch.quantization import quantize
quantize(model, config={"algorithm": "fp8", "calibration_steps": 512})
```

### llama.cpp / GGUF

本地部署的事实标准。K-quants（`Q4_K_M`、`Q5_K_M`、`Q6_K`）是 per-group 量化的各种变体，手写 AVX2 / NEON / AVX-512 SIMD kernel，Apple Silicon 上性能惊人。

| GGUF 格式 | 有效精度 | 7B 大小 | 质量定位 |
|---|---|---|---|
| Q4_0 | 4.0 bit | 3.8 GB | 基础 |
| Q4_K_M | 4.5 bit | 4.1 GB | 推荐 |
| Q5_K_M | 5.5 bit | 4.8 GB | 高质量 |
| Q8_0 | 8.0 bit | 7.2 GB | 接近 FP16 |

---

## 5.10 国产 GPU 量化移植指南

这一节是给"做国产 GPU AI infra 的人"写的，是本讲的工程落脚点。

<p align="center">
  <img src="./images/fig11_domestic_gpu.png" width="860" alt="国产 GPU/NPU 量化精度支持现状"/>
</p>

### 5.10.1 昇腾（华为）

**硬件算子分工**：Cube 单元做 INT8/FP16/BF16 矩阵乘；Vector 单元做 element-wise（FP16/BF16/FP32）；Scalar 单元做控制。量化 kernel 一般是 Vector 做 dequant、Cube 做 GEMM 的两段流水。

**关键事实**（截至 2024Q4，以官方最新文档为准）：

- **Ascend 910B**：INT8 Cube 原生（640 TOPS dense），FP16 320 TFLOPS，HBM2e 64 GB / 约 1.2 TB/s[^910b]。
- **Ascend 910C**：双 die，INT8 约 1,600 TOPS，HBM3 128 GB / 3.2 TB/s。FP8 列入下一代规划。
- **软件栈**：CANN（Compute Architecture for Neural Networks）+ MindIE（推理引擎）+ MindSpore 或 PyTorch（通过 `torch_npu` adapter）。

**从 CUDA 量化方案移植到昇腾的核心挑战**：

1. **Marlin / Machete kernel 没有对应物**。这类 W4A16 的 register-level pipelining 是 CUDA SM 架构特有的，昇腾的 Cube（以 16×16 或 16×16×16 为最小 tile）不直接对应。可行的路径是：在 Vector 单元上做 INT4 → FP16 的 dequant，然后直接调 Cube 的 FP16 MatMul；性能会吃亏（约 FP16 基线的 60–70 %），但工程可行。
2. **FP8 的 amax 同步没有等价原语**。CUDA 依赖 atomic max + cooperative group barrier，昇腾需要用 AICore 的 `reduce_max` 加显式 sync 替代。在 910B 上建议直接**用 FP16 替代 FP8**，等 910C 的 FP8 原生支持稳定再迁。
3. **Per-group W4 的 dequant 粒度对齐**。AWQ/GPTQ 用 g=128，昇腾 Cube 的 tile 是 16 的倍数，两者能对上，但 Vector 单元上 dequant 的 loop-unroll 需要手调。

典型的 PyTorch + CANN 上手姿势：

```python
import torch
import torch_npu                                   # 昇腾 PyTorch adapter
from torch_npu.contrib import transfer_to_npu

# 方式 1：用 CANN 内置 PTQ 工具链（推荐，工业界首选）
# auto_optimizer 是华为开源的量化工具，支持 ONNX 模型的 W8A8 PTQ
from auto_optimizer import OnnxGraph

# 方式 2：手动插入量化算子（需要对 CANN 算子库熟悉）
# 对应 CUDA 的 per-channel INT8 量化
x_q = torch_npu.npu_quant_per_channel_symmetric(x, scales, axis=0)
```

### 5.10.2 寒武纪（MLU）

**架构特点**：MLU-Core（类 SIMT）+ IPU（片上互联）。INT8 原生，FP8 较弱。

**工具链**：MagicMind 是对标 TensorRT 的推理引擎，支持 INT8 PTQ calibration。

```python
# 寒武纪 MagicMind PTQ 流程骨架
import magicmind.python.runtime as mm

network = mm.Network()
parser  = mm.parser.OnnxParser(network)
parser.parse_from_file("model.onnx")

config = mm.BuilderConfig()
config.parse_from_string(
    '{"archs": ["mtp_372"], '
    '"precision_config": {"precision_mode": "qint8_mixed_float16"}}'
)

builder = mm.Builder()
engine  = builder.build_engine(network, config)
```

寒武纪侧的一个特点：**MLU 590 的 INT8 TOPS 数据比较漂亮，但生态适配仍在追赶**。做 LLM 推理上线建议先跑通 FP16，再上 W8A8，W4 目前需要定制。

### 5.10.3 沐曦与海光

- **沐曦 MXC500** 是国产里少数 GPU 形态（非 NPU）的产品，CUDA 兼容度相对高，主要卖点是图形+计算双用。INT8 成熟，W4 在规划。
- **海光 DCU Z100** 基于 AMD MI100 的 CDNA 架构，ROCm 兼容，INT8 成熟，FP8 目前不支持。

这两家对国产 LLM 推理生态的存在感弱于昇腾，但在特定场景（数据中心、政务云）有不可替代的位置。

### 5.10.4 跨硬件量化移植 Checklist

做一次完整的"CUDA 方案 → 国产 GPU"移植，90 % 的工时会卡在下面这些点上：

| 检查项 | CUDA 参考 | 国产 GPU 状态 | 工程建议 |
|---|---|---|---|
| INT8 GEMM 精度 | cuBLAS `cublasGemmEx` | 通常 OK | 先对齐数值，再谈性能 |
| per-channel scale fuse | TRT epilogue | 需手工 kernel | 先用 separate kernel 跑通 |
| Per-group W4 dequant | Marlin / Machete | 通常无 | 手写 Vector dequant + FP16 GEMM |
| FP8 amax 同步 | CUDA atomic + barrier | 可能需模拟 | 910B 先用 FP16 替代 |
| KV cache 量化 | vLLM `kv_cache_dtype` | 需适配 | 后期优化项，收益显著 |
| 量化结果对齐 | PyTorch reference | 必须建立 | FP32 / INT8 diff 测试必建 |

### 5.10.5 落地优先级

一个务实的国产 GPU 量化上线路线：

- **P0（必须）**：FP16 / BF16 推理跑通；W8A8 per-tensor INT8 GEMM 接通（CANN / MagicMind 内置支持好，直接调）。
- **P1（性能关键）**：per-channel weight INT8；per-token activation INT8（配合 SmoothQuant 的数学迁移）。
- **P2（极致优化）**：W4 per-group（需自研 dequant kernel）；KV cache INT8 / FP8 压缩；Speculative decoding + 量化联合优化。

---

## 5.11 面试高频题

**Q1. STE 的数学含义是什么？在 clamp 边界处的梯度行为如何理解？**

STE 把不可导的 `round` 在反向时近似为恒等映射：$\partial \mathcal{L}/\partial x \approx \partial \mathcal{L}/\partial \hat{x}$，让量化误差参与权重更新。在 clamp 边界外（$x/S < q_\min$ 或 $> q_\max$），梯度为 0——这不是"信号缺失"，而是**在告诉优化器"这里被截断了，该往回移"**。有点像 ReLU 的 dead neuron，但方向相反：ReLU 的 0 梯度是噪声，STE 的 0 梯度是有用信息。LSQ[^lsq] 的核心创新就是把 scale 也做成可学习参数，进一步放大这个信号。

**Q2. Per-group 量化的 GEMM kernel 为什么比 per-channel 复杂那么多？**

Per-channel 量化下，每个 output channel 一个 scale，scale 只在 output 维度变化。GEMM 内层循环（reduction 维度）里 scale 是常数，可以提到外层一次性乘——对 Tensor Core 完全透明。Per-group 量化下，scale 在 reduction 维度上每 `g` 个元素跳变一次，必须在内层循环里 dequant：每算 `g` 个整数乘积就乘一次 FP16 scale，打乱了 Tensor Core 原本连续的数据流水。Marlin 的核心 trick 是在 register file 上做 overlap——prefetch 下一组 scale 的同时执行当前组的 INT 累加，把 dequant latency 藏进计算里。

**Q3. FP4 和 INT4 有什么本质区别？各自适合什么场景？**

INT4 是线性格点（-8 到 7），精度均匀分布。FP4（如 E2M1）有指数位，格点分布是对数不均匀的（0 附近密，大数附近稀）——这天然贴合深度学习权重和激活"大部分值集中在 0 附近"的统计规律。实测在相同 bit 数下，NVFP4 比 INT4 的 perplexity 好约 0.5–1。但 FP4 需要 Blackwell 这种原生 Tensor Core 支持；在 Ampere / Hopper 上，INT4 要软件 unpack 成 INT8 后再算，实际推理不一定比 INT8 快——kernel efficiency 才是决定因素。

**Q4. SmoothQuant 的等价变换为什么有效？**

SmoothQuant 用的是一个平凡的代数恒等式：$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$。矩阵乘法结果不变，但激活 X 按 channel 除以一个 "平滑因子" $s$，权重 W 相应乘以 $s$。选择 $s = (\max|X|)^\alpha / (\max|W|)^{1-\alpha}$（$\alpha \approx 0.5$），使迁移后 X 和 W 的量化难度**相当**。有效的原因是：原始激活 outlier 的幅值可达权重的 100 ×，迁移后双方幅值差距降到 2–3 ×，per-tensor / per-channel INT8 量化误差骤降。注意：$s$ 是 per-channel 的向量，且迁移过程完全是离线的——上线时只是权重被预乘了一次。

**Q5. 一个 LLaMA-2-7B 的 W4A16 AWQ 模型，实际显存占用大约多少？给出计算过程。**

- 参数量：7 B，FP16 为 14 GB
- W4 权重（embedding 一般不量化）：约 6.7 B × 4 bit = 3.35 GB
- Group scale（g=128，FP16）：6.7 B / 128 × 2 byte ≈ 0.1 GB
- Embedding（FP16，不量化）：32000 × 4096 × 2 ≈ 0.25 GB
- **权重合计 ≈ 3.7 GB**
- Activation buffer（BF16，seq=2048）：≈ 0.5 GB
- KV cache（BF16，32 层，seq=2048，batch=1）：2 × 32 × 2048 × 32 × 128 × 2 ≈ 0.5 GB
- **总计 ≈ 4.7 GB**（vLLM 实测约 5.2 GB，含框架 overhead 和 CUDA graph）

**Q6. 国产 GPU 上移植 GPTQ W4，最难的技术点是什么？怎么绕？**

最难的是 **Marlin 风格的 W4A16 GEMM kernel**——它要在 reduction 循环内交织 INT4 dequant 和 FP16 累加，利用 register file 的 double-buffer 隐藏 dequant latency，同时与 Tensor Core 的 tile 尺寸（16×16×16）精确对齐。国产 GPU 的矩阵计算单元 tile 形状各异，dequant 的粒度和 pipeline 深度需要重新设计。

目前最可行的工程路线是：**先把 W4 weight dequant 到 FP16，再调标准 FP16 GEMM**。牺牲约 30 % 带宽（相比 Marlin 最优解）换取工程可行性。等硬件真正原生支持 W4 指令（昇腾 910C 规划中），再替换内核。这符合"先出货再优化"的硬件落地节奏。

---

## 5.12 自检清单

读完本讲后，以下问题应该不查资料直接答上来：

- [ ] INT8 乘法比 FP32 省多少能量？比一次 DRAM 读呢？（§5.1.1）
- [ ] BF16 的指数位几位？为什么这让它成为训练首选？（§5.2.4）
- [ ] 写出对称 / 非对称量化的 S 和 Z 公式，说明 GEMM 展开后的差异（§5.3）
- [ ] Per-channel 量化为什么对 weight 可行、对 activation 不可行？（§5.4.2）
- [ ] K-Means 量化为什么精度更好但在推理中被淘汰？（§5.5.2）
- [ ] STE 在 clamp 边界外梯度是多少？为什么？（§5.7.2）
- [ ] LLM outlier 如何导致名义 8 bit 实际只有 1–2 bit 有效精度？（§5.8.1）
- [ ] W4A16 / W8A8 / FP8 各适合什么推理场景？（§5.9）
- [ ] 国产 GPU 移植 W4 GEMM 的最大技术挑战是什么？（§5.10.4）

---

## 延伸阅读

- **系统性入门**：Gholami 等人的综述 *A Survey of Quantization Methods for Efficient Neural Network Inference*[^survey]，把 2018–2021 年的方法梳理得很干净。
- **LLM 量化实战**：AWQ 的[官方 README](https://github.com/mit-han-lab/llm-awq) 和 GPTQ 的[复现解读](https://github.com/IST-DASLab/gptq) 读源码比读论文直观。
- **FP8 训练**：DeepSeek-V3 技术报告[^dsv3] 是最完整的工业案例；想更系统可以看 NVIDIA Transformer Engine 的[官方文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/)。
- **Marlin 内核原理**：[Marlin repo](https://github.com/IST-DASLab/marlin) 的 README 把 W4A16 的 register-level pipelining 讲得很清楚，配合源码读可以理解 "为什么 CUDA 比其他硬件在 quant kernel 上领先一代"。
- **硬件视角**：TechInsights 的 [Ascend 910B/C 拆解](https://www.techinsights.com/)、[XPU.pub 的国产 GPU 分析](https://xpu.pub)、SemiAnalysis 的文章。避开营销口径，看真实架构。
- **向量检索 / PQ**：FAISS 的 [wiki](https://github.com/facebookresearch/faiss/wiki) 是 PQ / IVF / HNSW 的一手教程，比任何二手教程都准确。

---

## 参考文献

[^horowitz]: M. Horowitz. *1.1 Computing's Energy Problem (and what we can do about it).* ISSCC 2014. [[PDF]](https://gwern.net/doc/cs/hardware/2014-horowitz-2.pdf)

[^han231n]: Song Han. *Efficient Methods and Hardware for Deep Learning.* CS231n 2017 Guest Lecture. [[PDF]](https://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture15.pdf) — 45 nm 能耗表在第 23 页，标注源自 Horowitz 2014。

[^bf16]: S. Wang & P. Kanwar. *BFloat16: The secret to high performance on Cloud TPUs.* Google Cloud Blog, 2019. [[link]](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)

[^fp8]: P. Micikevicius et al. *FP8 Formats for Deep Learning.* arXiv:2209.05433, 2022. [[PDF]](https://arxiv.org/abs/2209.05433)

[^h100ds]: NVIDIA. *H100 Tensor Core GPU Datasheet.* [[PDF]](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet)

[^mx]: Open Compute Project. *OCP Microscaling Formats (MX) Specification v1.0*, 2023. [[PDF]](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

[^b200ds]: NVIDIA. *Blackwell B200 Datasheet.* [[PDF]](https://resources.nvidia.com/en-us-blackwell-architecture/datasheet)

[^910b]: TechInsights / Huawei 官方文档整理：Ascend 910B HBM2e 64 GB、带宽约 1.2 TB/s、FP16 约 320 TFLOPS、INT8 约 640 TOPS。数据来源 TechInsights 拆解报告及 Huawei Cloud 开发者文档。

[^jacob]: B. Jacob et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR 2018. arXiv:1712.05877. [[PDF]](https://arxiv.org/abs/1712.05877)

[^gptq]: E. Frantar et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. arXiv:2210.17323. [[PDF]](https://arxiv.org/abs/2210.17323)

[^awq]: J. Lin et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys 2024. arXiv:2306.00978. [[PDF]](https://arxiv.org/abs/2306.00978)

[^smoothquant]: G. Xiao et al. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML 2023. arXiv:2211.10438. [[PDF]](https://arxiv.org/abs/2211.10438)

[^llmint8]: T. Dettmers et al. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS 2022. arXiv:2208.07339. [[PDF]](https://arxiv.org/abs/2208.07339)

[^marlin]: E. Frantar et al. *Marlin: a Mixed-Precision Inference Kernel for Int4 Weight × FP16 Activation.* 2024. [[repo]](https://github.com/IST-DASLab/marlin)

[^dc]: S. Han, H. Mao, W. J. Dally. *Deep Compression.* ICLR 2016 (Best Paper). arXiv:1510.00149. [[PDF]](https://arxiv.org/abs/1510.00149)

[^pq]: H. Jégou, M. Douze, C. Schmid. *Product Quantization for Nearest Neighbor Search.* TPAMI 2011. [[PDF]](https://hal.inria.fr/inria-00514462v2/document)

[^faiss]: J. Johnson, M. Douze, H. Jégou. *Billion-scale similarity search with GPUs.* IEEE Trans. Big Data, 2019. [[repo]](https://github.com/facebookresearch/faiss)

[^encodec]: A. Défossez et al. *High Fidelity Neural Audio Compression.* arXiv:2210.13438, 2022. [[PDF]](https://arxiv.org/abs/2210.13438)

[^ste]: Y. Bengio, N. Léonard, A. Courville. *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.* arXiv:1308.3432, 2013. [[PDF]](https://arxiv.org/abs/1308.3432)

[^lsq]: S. K. Esser et al. *Learned Step Size Quantization.* ICLR 2020. arXiv:1902.08153. [[PDF]](https://arxiv.org/abs/1902.08153)

[^xnor]: M. Rastegari et al. *XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks.* ECCV 2016. arXiv:1603.05279. [[PDF]](https://arxiv.org/abs/1603.05279)

[^twn]: F. Li et al. *Ternary Weight Networks.* arXiv:1605.04711, 2016. [[PDF]](https://arxiv.org/abs/1605.04711)

[^bitnet]: S. Ma et al. *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* arXiv:2402.17764, 2024. [[PDF]](https://arxiv.org/abs/2402.17764)

[^dsv3]: DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv:2412.19437, 2024. [[PDF]](https://arxiv.org/abs/2412.19437)

[^qserve]: Y. Lin et al. *QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving.* MLSys 2025. arXiv:2405.04532. [[PDF]](https://arxiv.org/abs/2405.04532)

[^survey]: A. Gholami et al. *A Survey of Quantization Methods for Efficient Neural Network Inference.* arXiv:2103.13630, 2021. [[PDF]](https://arxiv.org/abs/2103.13630)
