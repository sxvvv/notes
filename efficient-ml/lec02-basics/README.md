# Lec02 深度学习基础与效率指标

> 📺 [课程视频](https://www.youtube.com/watch?v=I0nKjPpZmMU&feature=youtu.be) &nbsp;|&nbsp; 📄 [Slides](https://www.dropbox.com/scl/fi/pxvvqyq2yu6mwgk79bq5x/Lec02-Basics.pdf?rlkey=tsumfkhrglic55jnjs4yu66ni&e=1&st=cmwnvuvn&dl=0)
>
> 基于 MIT 6.5940 EfficientML 课程整理，加入个人理解补充。

---

## 目录

- [一、模型参数量](#一模型参数量)
- [二、计算量：FLOPs 与 MACs](#二计算量flops-与-macs)
- [三、训练内存占用](#三训练内存占用)
- [四、数值精度](#四数值精度)
- [五、效率指标体系](#五效率指标体系)
- [六、推理内存与 KV Cache](#六推理内存与-kv-cache)
- [代码实践](#代码实践)
- [Infra 实战映射](#infra-实战映射)
- [跨 Lecture 关联](#跨-lecture-关联)
- [面试高频题](#面试高频题)

---

## 一、模型参数量

参数量决定了模型的存储开销和搬运代价。一个 7B 模型有 70 亿个浮点数需要保存、搬运和参与运算。效率优化的起点，永远是搞清楚参数在哪里、有多少。

### Linear 层

全连接层就是矩阵乘加偏置： $y = xW^T + b$

$$
\mathrm{params} = C_{\mathrm{in}} \times C_{\mathrm{out}} + C_{\mathrm{out}} \quad (\text{bias})
$$

例如 `Linear(768, 3072)` → $768 \times 3072 + 3072 = 2{,}362{,}368$ 。

这个数字在 Transformer 中反复出现——FFN 的 up projection 就是这个规模。

### Conv2d 层

$$
\mathrm{params} = C_{\mathrm{out}} \times C_{\mathrm{in}} \times K_H \times K_W + C_{\mathrm{out}} \quad (\text{bias})
$$

`Conv2d(64, 128, kernel_size=3)` → $128 \times 64 \times 3 \times 3 + 128 = 73{,}856$ 。

卷积参数量与输入空间分辨率无关。无论输入是 $224 \times 224$ 还是 $1024 \times 1024$ ，参数量不变——因为同一组 kernel 在所有空间位置滑动复用（weight sharing）。这个性质是后面理解 FLOPs 和参数量"脱钩"的关键。

### Transformer 中的 Multi-Head Attention

MHA 包含四组线性投影：Q、K、V、O。

| 投影 | 形状 |
|------|------|
| $W_Q, W_K, W_V$ | $d_{\mathrm{model}} \times d_{\mathrm{model}}$ |
| $W_O$ | $d_{\mathrm{model}} \times d_{\mathrm{model}}$ |

忽略 bias 时总参数量： $4 \times d_{\mathrm{model}}^2$ 。

### 实例：GPT-2 small 参数量拆解

GPT-2 small 是由 12 层 Transformer Block 组成的 Decoder-only 结构， $d_{\mathrm{model}} = 768$ ， $d_{\mathrm{ff}} = 3072 = 4 \times d_{\mathrm{model}}$ 。

流程：Input Embedding → 12 × Transformer Block → Output Embedding（与 Input Embedding 共享权重）

**(1) Input Embedding**

- Token Embedding：对每个 token $t_i$ ，查表得到稠密向量 $e_i = W_e[t_i] \in \mathbb{R}^{768}$ ，其中 $W_e \in \mathbb{R}^{50257 \times 768}$ 。Params = $50257 \times 768 = 38{,}597{,}376$
- Position Embedding：绝对可学习位置嵌入 $W_p \in \mathbb{R}^{1024 \times 768}$ 。Params = $1024 \times 768 = 786{,}432$
- 合计： $39{,}383{,}808$

**(2) 单层 Transformer Block**

$$
x' = x + \mathrm{Attn}(\mathrm{LN}_1(x)), \quad x'' = x' + \mathrm{MLP}(\mathrm{LN}_2(x'))
$$

| 模块 | 参数量 |
|------|--------|
| LayerNorm 1： $2 \times 768$ （scale + bias） | 1,536 |
| Attention： $W_{QKV}$ ( $768 \times 768 \times 3 + 768 \times 3$ ) + $W_{\mathrm{proj}}$ ( $768 \times 768 + 768$ ) | 2,362,368 |
| LayerNorm 2： $2 \times 768$ | 1,536 |
| FFN up： $768 \times 3072 + 3072$ | 2,362,368 |
| FFN down： $3072 \times 768 + 768$ | 2,360,064 |
| **单层合计** | **~7.09M** |

12 层 → $7.09\text{M} \times 12 = 85{,}054{,}464$

**(3) 总参数量**

Input Embedding + 12 层 Block = $39.4\text{M} + 85.1\text{M} = 124.4\text{M}$

实际公开的 GPT-2 small 约 117M。差异来源于 Output Head 与 Input Embedding 共享权重（weight tying），朴素 `sum(p.numel())` 会重复计算这 38.6M 参数。HuggingFace 的 `num_parameters()` 用 `data_ptr()` 去重来规避这个问题。

值得注意的比例：FFN 参数 / Attention 参数 $\approx 2 : 1$ 。Transformer 中参数的大头在 FFN，不在 attention。

<p align="center">
  <img src="images/fig_transformer_params.png" width="700"/>
</p>
<p align="center"><b>图 1</b>：Transformer block 参数分布。FFN 占比约 2/3，attention 约 1/3。</p>

---

## 二、计算量：FLOPs 与 MACs

先把几个容易混淆的术语对齐：

| 缩写 | 含义 | 举例 |
|------|------|------|
| **MACs** | Multiply-Accumulate 次数 | 1 MAC = 一次乘法 + 一次累加 |
| **FLOPs** | 浮点运算总次数（大写 O，复数 s） | 1 MAC = 2 FLOPs |
| **FLOPS** | 每秒运算次数（吞吐能力） | "A100 FP16 峰值 312 TFLOPS" |

很多论文和工具（如 `thop`、`fvcore`）报告的是 MACs 而非 FLOPs，两者差 2 倍。读数据时务必看清单位。

> **本文约定：** 后续公式统一以 MACs 为单位给出。如需换算成 FLOPs，乘以 2 即可。

### Linear 层

$$
\mathrm{MACs} = B \times C_{\mathrm{in}} \times C_{\mathrm{out}}
$$

每个输出元素需要 $C_{\mathrm{in}}$ 次 multiply-accumulate。

### Conv2d 层

$$
\mathrm{MACs} = B \times C_{\mathrm{in}} \times C_{\mathrm{out}} \times K_H \times K_W \times H_{\mathrm{out}} \times W_{\mathrm{out}}
$$

与参数量相比，多出 $H_{\mathrm{out}} \times W_{\mathrm{out}}$ 。这就是参数量与计算量能"脱钩"的原因—— $3 \times 3$ 卷积可能只有 81 个参数，但在 $1000 \times 1000$ 的 feature map 上做一次 forward，MACs 就有约 81M。

反过来，Embedding 层参数量巨大（ $\mathrm{vocab\\_size} \times d_{\mathrm{model}}$ ），但计算约等于零——只是一次 gather 查表。

<p align="center">
  <img src="images/fig_flops_vs_params.png" width="600"/>
</p>
<p align="center"><b>图 2</b>：参数量与计算量可以显著脱钩。Embedding 参数量大但计算近乎为零（查表）；Conv 参数少但因 weight sharing 而计算密集。</p>

### Self-Attention

设序列长度 $N$ ，模型维度 $d$ ，单层 self-attention 的 MACs：

$$
\underbrace{4Nd^2}\_{\text{QKV + O 投影}} + \underbrace{2N^2 d}\_{\text{注意力矩阵 } (QK^T + \mathrm{score} \cdot V)}
$$

两项分别对应 $O(Nd^2)$ 和 $O(N^2 d)$ 。谁占主导取决于 $N$ 和 $d$ 的相对大小：

- 当 $N < 2d$ 时，投影项主导（如 $N=512, d=768$ ，典型的短序列场景）
- 当 $N > 2d$ 时，注意力矩阵项主导（如 $N=32768, d=768$ ，长序列场景）

长上下文优化（FlashAttention、Ring Attention、Sparse Attention）本质上都是在压缩 $N^2 d$ 这一项。

<p align="center">
  <img src="images/fig_attention_flops.png" width="600"/>
</p>
<p align="center"><b>图 3</b>：Self-Attention 计算量随序列长度的变化。交叉点在 $N = 2d$ ，之后注意力矩阵的二次项成为瓶颈。</p>

> **关于公式中的系数约定：** 这里写 $4Nd^2$ 而非 $8Nd^2$ ，是因为以 MAC 计数。每个线性投影对矩阵 $(N, d) \times (d, d)$ 做乘-累加，MACs $= N \cdot d \cdot d$ 。四个投影（Q/K/V/O）共 $4Nd^2$ MACs。注意力矩阵部分， $QK^T$ 是 $(N, d) \times (d, N)$ ，MACs $= N^2 d$ ；score $\times V$ 是 $(N, N) \times (N, d)$ ，MACs $= N^2 d$ ；合计 $2N^2 d$ MACs。如果要转换成 FLOPs，整个公式乘 2 得到 $8Nd^2 + 4N^2 d$ ，这与部分文献（如 Megatron-LM 论文）的写法一致。

### 实例：ResNet-50 第一层

`Conv2d(3, 64, kernel_size=7, stride=2, padding=3)`，输入 $224 \times 224$ 。

参数量： $P = 64 \times 3 \times 7 \times 7 = 9{,}408$ （无 bias）

输出分辨率： $H_{\mathrm{out}} = \lfloor (224 + 2 \times 3 - 7) / 2 + 1 \rfloor = 112$

MACs： $64 \times 3 \times 7 \times 7 \times 112 \times 112 \approx 118\text{M}$

一个不到 1 万参数的层，MACs 约 1.18 亿。这就是 Conv 的典型特征：参数少但计算密集（weight sharing 在空间维度上反复使用同一组参数）。

---

## 三、训练内存占用

> 训练和推理的显存构成完全不同。推理只需要放模型权重（加上 KV cache），而训练要额外维护梯度、优化器状态和中间激活值。搞清楚显存花在了哪里，是所有显存优化的前提。

训练显存 = **Model States** + **Residual States**

这是 ZeRO 论文（Rajbhandari et al., 2019）给出的经典划分。Model States 是跟模型参数直接相关的那些东西（权重、梯度、优化器状态），Residual States 是其余所有东西（激活值、临时 buffer、碎片）。

### 3.1 Model States

Model States 由三部分构成：权重、梯度、优化器状态。我们用 $\Psi$ 表示模型参数量。

**(1) 权重（Parameters）**

就是模型本身的参数。内存大小 = 参数量 × 每个参数的字节数。

- FP32： $4\Psi$ bytes
- BF16 / FP16： $2\Psi$ bytes

7B 模型 FP16 权重 = $2 \times 7 \times 10^9 = 14$ GB。

**(2) 梯度（Gradients）**

反向传播过程中，每个参数都会产生一个对应的梯度，大小跟权重一模一样。

- FP32： $4\Psi$ bytes
- BF16 / FP16： $2\Psi$ bytes

**(3) 优化器状态（Optimizer States）**

这是 Model States 中的大头，也是最容易被低估的部分。

以最常用的 **AdamW** 为例，它需要为每个参数额外保存两个状态：

| 状态 | 含义 | 精度 | 内存 |
|------|------|------|------|
| 参数副本（master weights） | optimizer 内部维护的 FP32 精度参数 | FP32 | $4\Psi$ |
| 一阶矩 $m$ （momentum） | 梯度的指数移动平均 | FP32 | $4\Psi$ |
| 二阶矩 $v$ （variance） | 梯度平方的指数移动平均 | FP32 | $4\Psi$ |

为什么 optimizer states 始终是 FP32？因为 Adam 的更新公式涉及除法和开根号，低精度下数值误差会累积，导致训练不稳定。这不是工程偷懒，是数学上的硬约束。

### Model States 合计

**纯 FP32 训练（已经很少用了）：**

纯 FP32 时不需要额外的 master weights（权重本身就是 FP32），所以是 $4\Psi$ （权重）+ $4\Psi$ （梯度）+ $4\Psi$ （ $m$ ）+ $4\Psi$ （ $v$ ）= $16\Psi$ 。

**混合精度训练（主流做法）：**

| 组件 | 精度 | 内存 |
|------|------|------|
| 前向/反向用的权重 | BF16 | $2\Psi$ |
| 梯度 | BF16 | $2\Psi$ |
| Master weights | FP32 | $4\Psi$ |
| Adam $m$ | FP32 | $4\Psi$ |
| Adam $v$ | FP32 | $4\Psi$ |
| **合计** | | **$16\Psi$** |

一个反直觉的事实：**混合精度训练的 Model States 总量跟纯 FP32 一样，都是 $16\Psi$ 。** 混合精度省的不是 optimizer states——optimizer states 无论如何都得 FP32。混合精度真正省的是：（a）计算速度，BF16 matmul 在 Tensor Core 上快 2–4 倍；（b）激活值内存，中间结果存 BF16 直接减半。要真正砍掉 optimizer states 的冗余，只能靠 ZeRO / FSDP 把它切分到多张卡上。

**实际算一下：**

| 模型 | 参数量 $\Psi$ | Model States（ $16\Psi$ ） | 单卡放得下？ |
|------|-------------|--------------------------|------------|
| GPT-2 1.5B | 1.5B | 24 GB | A100 80GB ✅ |
| LLaMA-2 7B | 7B | 112 GB | A100 80GB ❌ |
| LLaMA-2 13B | 13B | 208 GB | A100 80GB ❌ |
| LLaMA-2 70B | 70B | 1120 GB | 8×A100 都不够 |

7B 模型光 Model States 就 112 GB，一张 A100 80GB 装都装不下——而这还没算激活值。

<p align="center">
  <img src="images/fig_memory_breakdown.png" width="700"/>
</p>
<p align="center"><b>图 4</b>：7B 模型的训练内存拆解。混合精度并不能显著节省 optimizer states（仍为 FP32），其主要收益在于计算加速和 activation 内存的缩减。</p>

### 3.2 Residual States

Model States 之外的显存消耗统称 Residual States，包括三部分：激活值、临时 buffer、内存碎片。

**(1) 激活值（Activations）**

激活值是前向传播过程中产生的中间结果。之所以要保留它们，是因为反向传播计算梯度时需要用到这些中间结果。

> 举个简单的例子：要算 $y = \text{ReLU}(Wx)$ 对 $W$ 的梯度，反向传播时需要知道 $x$ 的值和 ReLU 的输入值。这些就是"激活值"。

对于 Transformer 模型，**单层**的激活值内存（BF16 存储）大致为：

$$
\text{Activation}\_{\text{per\\_layer}} \approx sbh \times \left(34 + 5 \frac{as}{h}\right)
$$

其中 $s$ = 序列长度， $b$ = batch size， $h$ = hidden size， $a$ = attention heads 数量。

- 第一项 $34sbh$ 来自 QKV 投影、FFN 线性层、LayerNorm 等操作的输入缓存
- 第二项 $5as^2b$ 来自注意力矩阵（softmax 输出 + dropout mask），这一项跟 $s^2$ 成正比

两个关键观察：

**激活值跟 batch size 和序列长度线性（甚至平方）相关，但跟参数量无关。** 你可以有一个参数量不大的模型，但在长序列 + 大 batch 下，激活值照样炸显存。

**不用 FlashAttention 时，注意力矩阵的 $s^2$ 项非常恐怖。** 比如 $s=4096, a=32, b=4, h=4096$ ，光注意力矩阵这一项就要 $5 \times 32 \times 4096^2 \times 4 \approx 10$ GB 每层，32 层就是 320 GB。FlashAttention 通过 tiling + 重计算把这个 $s^2$ 项基本消除了。

**具体例子——GPT-2 1.5B**（ $h=1600, L=48, a=25$ ）， $s=1024, b=32$ ：不用 activation checkpointing 时，激活值约 60 GB，比 Model States 的 24 GB 还大得多。

**Activation Checkpointing（梯度检查点）**

既然激活值这么吃显存，一个自然的想法是：不存它们，反向传播时重新算。

- **Full checkpointing：** 只保存每一层的输入，反向传播时重新做一遍该层的前向。激活值内存从 $O(L \times sbh \times 34)$ 降到 $O(L \times 2sbh)$ ，代价是增加约 33% 的计算开销（多做一次前向）。
- **Selective checkpointing：** 只重计算那些"占显存多但算得快"的操作。典型代表就是注意力矩阵——softmax 输出占了大量显存（ $s^2$ 项），但重算成本不高（相比线性投影的 $d^2$ 开销）。Megatron-LM 的 selective recomputation 把重计算开销从 33% 降到约 4%，同时仍然显著减少了激活值内存。

FlashAttention 本质上也是在做这件事：它不把完整的 $N \times N$ 注意力矩阵写到 HBM，而是分块计算并重计算 softmax，从而把注意力部分的激活值内存从 $O(s^2)$ 降到 $O(s)$ 。

**(2) 临时 Buffer（Temporary Buffers）**

训练过程中一些操作需要临时分配额外的内存，用完即释放。典型的例子：

- **梯度 all-reduce buffer：** 分布式训练时，gradient all-reduce 操作倾向于把所有梯度拼成一个大的连续 tensor 再通信，这样可以增大消息体积、提高带宽利用率。梯度本身是 FP16/BF16，但 all-reduce 时这个 buffer 可能是 FP32 的。对于 1.5B 参数的模型，一个 FP32 的 flattened buffer 就需要 6 GB。
- **梯度 norm 计算：** gradient clipping 之前需要计算所有梯度的 L2 norm，也需要一个临时 buffer。

这些 buffer 是"隐形杀手"——你在计算 Model States 和 Activation 时不会算到它们，但它们在运行时是实打实占显存的。

**(3) 内存碎片（Memory Fragmentation）**

即使总的空闲显存足够，如果这些空闲空间不连续，大的 tensor 分配请求照样会 OOM。这跟操作系统的内存碎片问题本质相同。

实际观察到的现象：训练超大模型时，OOM 发生时显存还有 30% 以上是空闲的——但都是碎片，拼不成连续块。PyTorch 的 CUDA caching allocator 会做一些碎片整理，但在极端情况下仍然不够。

### 3.3 完整公式与实例

把上面的部分合在一起，训练总显存：

$$
\text{Memory}\_{\text{train}} = \underbrace{16\Psi}\_{\text{Model States}} + \underbrace{\text{Activations}}\_{\text{跟 } b, s, L \text{ 相关}} + \underbrace{\text{Buffers + Fragmentation}}\_{\text{难以精确预估}}
$$

实践中的粗估公式（混合精度 + Adam + 不做 activation checkpointing）：

$$
\text{Memory}\_{\text{train}} \approx 16\Psi + 2 \cdot L \cdot s \cdot b \cdot h \cdot 34
$$

最后一项在做了 full activation checkpointing 后近似为 $2 \cdot L \cdot s \cdot b \cdot h \cdot 2$ ，降幅非常显著。

**实例：LLaMA-2 7B 训练显存估算**（ $L=32, h=4096, a=32, \Psi=7\text{B}$ ）

| 组件 | 显存 | 说明 |
|------|------|------|
| Model States | 112 GB | $16 \times 7\text{B}$ |
| Activations（ $s=2048, b=4$ ，无 checkpointing） | ~72 GB | $32 \times 2048 \times 4 \times 4096 \times 34 \times 2$ bytes |
| Activations（ $s=2048, b=4$ ，full checkpointing） | ~4 GB | $32 \times 2048 \times 4 \times 4096 \times 2 \times 2$ bytes |
| 临时 buffer | 数 GB | 视实现而定 |
| **总计（无 checkpointing）** | **~190 GB** | 3 张 A100 80GB 才放得下 |
| **总计（有 checkpointing）** | **~120 GB** | 2 张 A100 80GB |

这还只是数据并行中单副本的显存需求。如果不做 ZeRO 切分，每张卡都要放完整的 Model States。

### 3.4 怎么把显存压下去

上面分析了显存花在哪里，自然引出了对应的优化手段：

| 显存大头 | 优化手段 | 原理 |
|----------|----------|------|
| Optimizer States（ $12\Psi$ ） | ZeRO Stage 1 / FSDP | 把 optimizer states 切分到多张卡，每张卡只存 $1/N$ |
| Optimizer States + Gradients | ZeRO Stage 2 | 在 Stage 1 基础上，梯度也切分 |
| 全部 Model States | ZeRO Stage 3 / FSDP Full Shard | 权重、梯度、optimizer states 全切分，forward 时按需 all-gather |
| Activations | Activation Checkpointing | 不存中间激活值，反向传播时重算 |
| Activations（注意力部分） | FlashAttention | 分块计算，不具化 $N \times N$ 注意力矩阵 |
| 权重 + Optimizer States | LoRA / QLoRA | 冻结大部分参数，只训练低秩适配器 |
| 全部 | Offloading（CPU / NVMe） | 把暂时不用的 states 卸载到 CPU 内存或 SSD |

一个常见的组合配置（8×A100 80GB 训练 7B）：BF16 混合精度 + ZeRO Stage 2 + Selective Activation Checkpointing + FlashAttention。这套组合下单卡显存压力降到约 30–40 GB，剩余空间可以放更大的 batch。

### 3.5 与推理显存的对比

把训练和推理放在一起看，就能理解为什么"能推理的卡不一定能训练"：

| 对比项 | 训练 | 推理 |
|--------|------|------|
| 权重 | ✅ 需要（前向 + 反向） | ✅ 需要 |
| 梯度 | ✅ 需要 | ❌ 不需要 |
| Optimizer States | ✅ 需要（Adam: $12\Psi$ bytes） | ❌ 不需要 |
| Activations | ✅ 需要（为反向传播保留） | ❌ 不需要保留（前向即丢弃） |
| KV Cache | ❌ 训练时一次算完整个序列 | ✅ 自回归生成时逐步累积 |

推理的显存 ≈ 权重 + KV Cache。训练的显存 ≈ 权重 × 8（mixed precision Adam）+ Activations。

同样一个 7B 模型：推理 FP16 只需要 14 GB（加上 KV cache 也就 20–30 GB），训练却需要 100+ GB。差了将近一个数量级。

### 3.6 SFT 与 Pretrain 的区别

微调（SFT）和预训练在显存公式上没有本质区别——只要你做全参数微调，Model States 是一样的。但有几点实践差异：

- **数据侧：** SFT 需要额外存 loss\_mask（标记哪些 token 参与 loss 计算），pretrain 通常不需要。数据集占内存会稍大。
- **序列长度：** SFT 数据通常含 padding（因为不同样本长度不一），实际有效计算比 pretrain 低，但显存该用还是得用。
- **常见搭配：** SFT 更多使用 LoRA/QLoRA，因为全参数微调对大模型来说显存成本太高，而微调任务通常不需要动所有参数。

---

## 四、数值精度

> 精度格式的选择直接影响内存占用、计算速度和训练稳定性。搞懂每种格式的位宽分配和数值特性，是理解混合精度训练、量化推理的基础。

### 4.1 浮点数是怎么存的

所有浮点数都由三部分构成：**符号位（sign）+ 指数（exponent）+ 尾数（mantissa）**。

$$
\text{value} = (-1)^{S} \times 2^{(E - \text{bias})} \times (1 + M)
$$

其中 $S$ 是符号位（0 正 1 负）， $E$ 是指数域的原始值，bias 是偏移量（让指数能表示负数）， $M$ 是尾数的小数部分（前面隐含一个 1）。

两个核心参数：

- **指数位数** → 决定**动态范围**（能表示的最大值和最小值之间的跨度）
- **尾数位数** → 决定**精度**（相邻两个可表示数值之间有多密）

这是一个 tradeoff：总位宽固定时，给指数多分一位，动态范围翻倍，但精度减半。深度学习中不同场景对这个 tradeoff 的偏好不同，这就是为什么会有这么多格式。

### 4.2 常见格式一览

| 格式 | 总位宽 | 符号 | 指数 | 尾数 | 动态范围（约） | 每个数占内存 |
|------|--------|------|------|------|----------------|-------------|
| FP32 | 32 bit | 1 | 8 | 23 | $\pm 3.4 \times 10^{38}$ | 4 bytes |
| TF32 | 19 bit* | 1 | 8 | 10 | 同 FP32 | 4 bytes（存储仍 32 bit） |
| FP16 | 16 bit | 1 | 5 | 10 | $\pm 65504$ | 2 bytes |
| BF16 | 16 bit | 1 | 8 | 7 | 同 FP32 | 2 bytes |
| FP8 E5M2 | 8 bit | 1 | 5 | 2 | $\pm 57344$ | 1 byte |
| FP8 E4M3 | 8 bit | 1 | 4 | 3 | $\pm 448$ | 1 byte |
| INT8 | 8 bit | — | — | — | $-128 \sim 127$ | 1 byte |
| INT4 | 4 bit | — | — | — | $-8 \sim 7$ | 0.5 byte |

> *TF32 是 NVIDIA A100 引入的"计算格式"：存储仍然是 32 bit，但 Tensor Core 计算时只用 19 bit（8 位指数 + 10 位尾数），所以算得快但精度略低于 FP32。用户代码不需要显式转换，硬件自动处理。

<p align="center">
  <img src="images/fig_precision_formats.png" width="700"/>
</p>
<p align="center"><b>图 5</b>：常用数值格式的位宽分配。BF16 和 FP32 共享相同的 8-bit 指数，因而动态范围一致；FP16 的 5-bit 指数使其最大值仅为 65504。</p>

### 4.3 FP16 vs BF16：为什么 LLM 训练用 BF16

这是面试最常问的问题之一。答案的核心在**指数位数**。

**FP16 的问题：动态范围太小**

FP16 只有 5 位指数，最大可表示值仅 **65504**。这在 LLM 训练中很容易出事：

- Attention score（softmax 之前的 logits）在长序列或特定 head 下可以轻松超过 65504
- Loss 值在训练初期也可能很大
- 一旦溢出就变成 `inf`，梯度变 `nan`，训练直接崩掉

为了应对这个问题，FP16 训练**必须搭配 loss scaling**：先把 loss 乘一个大常数（比如 1024 或动态调整的 scale factor），抬高梯度数值避免 underflow，反向传播后再除回来。这套机制就是 PyTorch 中 `GradScaler` 的作用。可行，但增加了工程复杂度，而且 scale factor 选不好还是会出问题。

**BF16 的设计思路：保范围、砍精度**

BF16 的 8 位指数与 FP32 完全一致，动态范围达到 $\pm 3.4 \times 10^{38}$ ，根本不会溢出。代价是尾数只有 7 位（FP16 有 10 位），精度更低。

但训练中这点精度损失基本可以忽略。原因是：训练本身就是一个带噪声的优化过程（SGD 的随机性），尾数少几位引入的误差远小于梯度本身的方差。

**工程上的额外好处**

FP32 转 BF16 只需要截断低 16 位，不需要额外的 rounding 逻辑。硬件实现极其简单，几乎零开销。而 FP32 转 FP16 需要处理指数域的映射和可能的溢出，复杂得多。

**总结对比：**

| 维度 | FP16 | BF16 |
|------|------|------|
| 指数位 | 5 | 8 |
| 尾数位 | 10 | 7 |
| 最大值 | 65504 | ~$3.4 \times 10^{38}$ |
| 需要 loss scaling？ | 是 | 否 |
| FP32 转换复杂度 | 高 | 低（截断即可） |
| 精度 | 更高 | 更低 |
| 训练稳定性 | 需要小心 | 开箱即用 |

当前 A100 / H100 上的大模型训练几乎统一使用 BF16。FP16 主要出现在早期的模型（BERT 时代）和部分推理场景。

### 4.4 FP8：下一代训练和推理精度

FP8 由 NVIDIA、ARM、Intel 联合提出，已在 H100 / H200 的 Tensor Core 上原生支持。相比 BF16 再省一半内存和带宽。

**两种变体，用在不同地方：**

| 变体 | 指数 | 尾数 | 最大值 | 适用场景 |
|------|------|------|--------|----------|
| E4M3 | 4 | 3 | ±448 | **前向传播**：权重和激活值需要更高精度 |
| E5M2 | 5 | 2 | ±57344 | **反向传播**：梯度数值波动大，需要更宽的动态范围 |

为什么前向和反向用不同的格式？因为权重和激活值的分布通常比较集中，精度比范围更重要（所以用 E4M3，尾数多一位）。而梯度的分布跨度大、尾部值多，动态范围比精度更关键（所以用 E5M2，指数多一位）。

NVIDIA Transformer Engine 的默认配置就是 `HYBRID` 模式：前向 E4M3，反向 E5M2。

**关键机制：per-tensor scaling**

FP8 的动态范围比 BF16 小很多（E4M3 最大只到 448），所以不能像 BF16 那样直接截断使用。每个 tensor 需要乘一个 FP32 的 scale factor，把数值范围映射到 FP8 能表示的区间内。

这个 scale factor 怎么定？有两种主流策略：

- **Delayed scaling（延迟缩放）：** 用前几步的 amax（绝对值最大值）历史来估算当前步的 scale，延迟一步更新。H100 上的标准做法。
- **Block scaling（MXFP8）：** 每 32 个连续元素共享一个 scale factor，粒度更细，能用 E4M3 表示所有 tensor（包括梯度），不需要 E5M2。Blackwell 架构开始支持。

**FP8 vs INT8**

同样是 8 bit，FP8 比 INT8 更适合深度学习的原因：

INT8 是定点格式，所有数值的间距均匀。但神经网络中的 tensor 分布通常是"中间密、两头稀"——大量值集中在零附近，少量值很大。FP8 的浮点特性天然匹配这种分布：靠近零的地方可表示值更密，远离零的地方更稀疏。

具体表现在：attention 机制中 softmax 前的 score 可以从接近 0 到数千，INT8 的 $[-128, 127]$ 固定范围很难覆盖，而 FP8 E4M3 到 448、E5M2 到 57344，灵活得多。

### 4.5 混合精度训练的完整流程

理解了各种格式的特性后，混合精度训练的逻辑就很清楚了。核心原则是：**计算用低精度（快），存储用高精度（准）**。

**BF16 混合精度（当前主流）：**

```
Forward:  权重 BF16 → matmul 在 Tensor Core 上用 BF16 → 激活值存 BF16
Backward: 梯度计算用 BF16 → 梯度存 BF16
Update:   optimizer 内部全部 FP32（master weights + Adam m/v）
          更新后的 FP32 权重截断为 BF16 用于下一步 forward
```

哪些操作保持 FP32？那些对数值误差敏感的操作：softmax、LayerNorm、loss 计算。PyTorch 的 `autocast` 会自动处理这个——matmul 走 BF16，element-wise ops 走 FP32。

**FP8 混合精度（H100+ 上的新选项）：**

```
Forward:  权重和激活量化为 E4M3 → matmul 在 FP8 Tensor Core 上
Backward: 梯度量化为 E5M2 → matmul 在 FP8 Tensor Core 上
其余:     与 BF16 混合精度相同，optimizer 仍然 FP32
```

FP8 的收益主要在两个方面：matmul 吞吐翻倍（相比 BF16），以及激活值内存再减半。但需要 Transformer Engine 库来管理 scaling factor，不是简单换个 dtype 就行的。

**代码示例：BF16 vs FP16 的关键差异**

```python
import torch
from torch.amp import autocast, GradScaler

# === BF16 训练（简洁，不需要 scaler） ===
for batch in dataloader:
    optimizer.zero_grad()
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(batch)
    loss.backward()
    optimizer.step()

# === FP16 训练（必须用 GradScaler） ===
scaler = GradScaler()
for batch in dataloader:
    optimizer.zero_grad()
    with autocast(device_type="cuda", dtype=torch.float16):
        loss = model(batch)
    scaler.scale(loss).backward()    # loss 先放大
    scaler.step(optimizer)            # 更新前先缩回来
    scaler.update()                   # 动态调整 scale factor
```

BF16 不需要 `GradScaler`，因为动态范围跟 FP32 一样，梯度不会 underflow。FP16 动态范围小，微小的梯度值会被 flush 成 0，必须用 loss scaling 抬高数值范围。

### 4.6 量化的能效视角

量化不只是省内存，更省电。这一点在移动端和大规模推理场景下尤其关键。

Horowitz（2014, 45nm 工艺）测量的单次运算能耗：

| 操作 | 能耗 (pJ) | 相对 FP32 乘法 |
|------|-----------|---------------|
| FP32 乘法 | 3.7 | 1× |
| FP32 加法 | 0.9 | 0.24× |
| FP16 乘法 | 1.1 | 0.30× |
| INT8 乘法 | 0.2 | 0.05× |
| INT8 加法 | 0.03 | 0.008× |

INT8 乘法只需 FP32 乘法约 **1/18** 的能耗。FP16 乘法约 FP32 的 1/3。

另一个常被忽略的事实是：**数据搬运的能耗远大于计算本身**。从 DRAM 读 32 bit 的能耗约 640 pJ，比一次 FP32 乘法贵 170 倍。所以降低精度不仅减少计算能耗，更重要的是减少了搬运量——同样的带宽下可以搬运两倍（FP16）甚至四倍（INT8）的数据。

这也是为什么量化在 memory-bound 的推理场景（尤其是 LLM decode）收益如此显著——瓶颈本来就在搬运，搬运量直接砍半。

<p align="center">
  <img src="images/fig_energy_cost.png" width="600"/>
</p>
<p align="center"><b>图 6</b>：不同精度下单次运算的能耗（Horowitz 2014, 45nm）。INT8 乘法相比 FP32 乘法约省 18 倍能耗。</p>

### 4.7 精度选择速查

| 场景 | 权重 | 计算 | Optimizer | 备注 |
|------|------|------|-----------|------|
| 大模型预训练（主流） | BF16 | BF16 Tensor Core | FP32 | 最稳，不需要 loss scaling |
| 大模型预训练（H100+） | FP8 E4M3 | FP8 Tensor Core | FP32 | 需要 Transformer Engine |
| 推理（高质量） | FP16 / BF16 | FP16 / BF16 | — | 无精度损失 |
| 推理（高吞吐） | INT8 / FP8 | INT8 / FP8 | — | 需要 PTQ 或 QAT 校准 |
| 推理（极限压缩） | INT4 / GPTQ / AWQ | FP16 dequant | — | 权重 INT4，计算时反量化为 FP16 |
| 边缘设备 | INT8 / INT4 | INT8 | — | 能耗是首要约束 |
| LoRA 微调 | 基座 INT4 + adapter BF16 | BF16 | FP32 | QLoRA：4bit 量化基座 + 低秩适配器 |

一个实用的判断原则：**训练看动态范围**（选 BF16 / FP8 E5M2），**推理看精度和效率**（选 FP8 E4M3 / INT8 / INT4）。训练中数值波动大，溢出是致命的；推理中数值分布已经稳定，可以更激进地压缩。

---

## 五、效率指标体系

### Latency

LLM 场景的延迟指标需要拆分为两个阶段：

| 指标 | 含义 | 主要影响因素 |
|------|------|-------------|
| TTFT | Time To First Token，首 token 延迟 | prompt 长度 → prefill 阶段是 compute-bound |
| TPOT | Time Per Output Token | 模型大小 × 带宽 → decode 阶段是 memory-bound |
| P99 | 第 99 百分位延迟 | 线上 SLA 通常看 P99 而非平均 |

TTFT 和 TPOT 由不同因素主导，优化手段也不同。这正是 vLLM continuous batching 需要平衡的核心 trade-off。

### Throughput

$$
\mathrm{Throughput} = \mathrm{batch\\_size} / \mathrm{latency}
$$

增大 batch 可提升吞吐（GPU 利用率上升），但单条请求的延迟会增加。生产环境的核心问题是在 latency SLA 约束下最大化 throughput。

### Arithmetic Intensity 与 Roofline 模型

$$
\text{Arithmetic Intensity (AI)} = \frac{\mathrm{FLOPs}}{\mathrm{Bytes\\_accessed}}
$$

AI 衡量一个 kernel 每搬运一个 byte 做多少次计算。

| 场景 | AI | 瓶颈 |
|------|-----|------|
| LLM decode (batch=1) | ~1 | Memory-bound |
| 大 batch matmul / prefill | 很高 | Compute-bound |
| Softmax / LayerNorm | 极低（element-wise） | Memory-bound |

**Roofline 模型**将这个关系可视化：

$$
\text{实际性能} = \min(\text{峰值算力},\;\text{峰值带宽} \times \mathrm{AI})
$$

<p align="center">
  <img src="images/fig_roofline.png" width="650"/>
</p>
<p align="center"><b>图 7</b>：Roofline 模型（A100 FP16）。Ridge point 约在 AI = 156。低于此值的 kernel 性能受限于带宽，高于此值才受限于算力。</p>

A100 的 ridge point $\approx 312 \text{ TFLOPS} / 2 \text{ TB/s} = 156$ 。对于 AI 低于 156 的 kernel，优化计算逻辑收效甚微，瓶颈在数据搬运。

<p align="center">
  <img src="images/fig_arithmetic_intensity.png" width="650"/>
</p>
<p align="center"><b>图 8</b>：常见操作的 Arithmetic Intensity 在频谱上的位置。LLM decode（batch=1）的 AI 约为 1，远低于 A100 ridge point。</p>

### MBU（Memory Bandwidth Utilization）

$$
\mathrm{MBU} = \frac{\text{实际带宽利用}}{\text{硬件峰值带宽}}
$$

Decode 阶段是纯 memory-bound 场景，MBU 直接反映工程实现效率。优秀的 LLM serving engine 在 decode 阶段 MBU 可达 70–80%+。

---

## 六、推理内存与 KV Cache

推理时不需要梯度和 optimizer states，但有另一个大户：**KV cache**。

### Decode 阶段的 memory-bandwidth 瓶颈

LLM decode 阶段（自回归生成）每生成一个 token 的过程：

1. 把全部模型权重从 HBM 读一遍
2. 做一次矩阵-向量乘（batch=1 时新 token 只有一个）
3. 更新 KV cache

```
7B FP16 → 14 GB 权重
A100 HBM 带宽 ≈ 2 TB/s
光搬权重的延迟 ≈ 14 / 2000 = 7 ms/token ≈ ~143 tokens/s 上限
```

这是 decode 阶段 memory-bound 的根本原因：算力大量空闲，瓶颈在数据搬运。

### KV Cache 内存公式

自回归生成时必须保留已生成所有 token 的 K、V 向量：

$$
\mathrm{KV\\_cache} = 2 \times L \times n_{\mathrm{kv\\_heads}} \times d_{\mathrm{head}} \times N \times B \times \mathrm{dtype\\_bytes}
$$

长序列 + 大 batch 下 KV cache 可远超模型权重。LLaMA-2 7B 在 batch=128、seq=4096 下 KV cache 约 64 GB，模型权重仅 14 GB。

这正是 PagedAttention、GQA（Grouped Query Attention）、MQA（Multi Query Attention）等技术的出发点——压缩 KV cache。

<p align="center">
  <img src="images/fig_kv_cache.png" width="600"/>
</p>
<p align="center"><b>图 9</b>：KV cache 随序列长度线性增长，随 batch size 线性放大。大 batch + 长序列下 KV cache 内存可远超模型权重本身。</p>

---

## 代码实践

### 基础工具函数

日常 debug 和 profiling 的起点：

```python
import torch
import torch.nn as nn
from typing import Tuple


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """统计可训练 / 总参数量。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def param_memory_mb(model: nn.Module, dtype_bytes: int = 2) -> float:
    """参数内存估算 (MB)。dtype_bytes: FP32=4, BF16/FP16=2, INT8=1"""
    total = sum(p.numel() for p in model.parameters())
    return total * dtype_bytes / (1024 ** 2)
```

### HuggingFace Transformers 的参数量统计

HuggingFace 不是简单的 `sum(p.numel())`。关键设计考量：

```python
# transformers/modeling_utils.py 中 num_parameters() 的核心逻辑（简化）

def num_parameters(self, only_trainable=False, exclude_embeddings=False):
    """
    两个工程细节：
    1. exclude_embeddings —— 论文常不算 embedding 参数
       （embedding 只做查表，不贡献实际计算）
    2. 去重 —— 用 data_ptr() 检测共享权重，避免重复统计
       典型场景：GPT-2 的 token embedding 和 lm_head 共享权重
    """
    if exclude_embeddings:
        embedding_params = sum(
            p.numel() for p in self.get_input_embeddings().parameters()
        )
    
    # 用 set(data_ptr) 去重共享参数
    seen = set()
    total = 0
    for name, p in self.named_parameters():
        if only_trainable and not p.requires_grad:
            continue
        ptr = p.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += p.numel()
    
    if exclude_embeddings:
        total -= embedding_params
    return total
```

### 计算量估算：手写 vs 工具

手写（以 MACs 为单位）：

```python
def macs_linear(B: int, in_f: int, out_f: int) -> int:
    """Linear 层的 MACs。"""
    return B * in_f * out_f


def macs_attention(B: int, N: int, d: int) -> int:
    """Self-attention MACs 估算（单层）。
    
    4*N*d^2 : Q/K/V/O 四个 projection（每个 N*d*d MACs）
    2*N^2*d : QK^T + attn @ V（每个 N*N*d MACs）
    """
    proj = 4 * N * d * d
    attn = 2 * N * N * d
    return B * (proj + attn)
```

实际项目中常用 `calflops` 或 Meta 的 `fvcore` 自动计算：

```python
from calflops import calculate_flops

flops, macs, params = calculate_flops(
    model=model,
    input_shape=(1, 128),
    transformer_tokenizer=tokenizer,
)
print(f"FLOPs: {flops}  MACs: {macs}  Params: {params}")
```

但注意：**自动工具不一定准确**。涉及动态形状（attention mask、变长序列）时，自动 profiler 容易给出误导性数字。手动推导能力是基本功，不能完全依赖工具。

### 数值精度：实际验证

```python
import torch

# BF16 vs FP16 精度差异
x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
print(f"FP16 最大误差: {(x - x.half().float()).abs().max():.2e}")
print(f"BF16 最大误差: {(x - x.bfloat16().float()).abs().max():.2e}")

# FP16 溢出
big = torch.tensor(65505.0)
print(f"65505 → FP16: {big.half()}")     # inf
print(f"65505 → BF16: {big.bfloat16()}")  # 65536.0，安全

# 模拟未归一化的 attention score
scores = torch.randn(1, 12, 2048, 2048) * 30
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
    - 无 bias（现代 LLM 通常 bias=False）
    """
    
    def __init__(self, d_model=256, n_heads=4, d_ff=1024):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        
        # QKV 合并为一个 Linear（LLaMA、Mistral 均如此）
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # FFN（LLaMA 用 SwiGLU，这里简化为 GELU）
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        B, N, D = x.shape
        
        # Self-Attention
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # PyTorch 2.0+ 会自动 dispatch 到 FlashAttention kernel
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        x = x + self.o_proj(attn_out)
        
        # FFN
        h = self.ln2(x)
        x = x + self.down(F.gelu(self.up(h)))
        
        return x
```

### vLLM 中的 KV cache 内存估算

```python
def estimate_kv_block_memory(
    num_layers: int,
    num_kv_heads: int,  # GQA 时 kv_heads < q_heads
    head_dim: int,
    block_size: int = 16,  # 每个 block 容纳的 token 数
    dtype_bytes: int = 2,  # BF16=2, FP8=1
) -> int:
    """单个 KV cache block 的内存 (bytes)。"""
    # K 和 V 各一份（×2），每层、每个 head、block_size 个 token
    return 2 * num_layers * num_kv_heads * head_dim * block_size * dtype_bytes

# LLaMA-2 7B: 32 layers, 32 kv_heads, head_dim=128, block_size=16, BF16
block_mem = estimate_kv_block_memory(32, 32, 128, 16, 2)
print(f"单个 KV block: {block_mem / 1024:.1f} KB")  # 4096 KB = 4 MB

# A100 80GB，模型 ~14GB (7B × 2B)
# 剩余 ~66GB 可用于 KV cache
# 可放 66 * 1024 / 4 ≈ 16896 个 blocks
# 每个 block 16 tokens → 最多缓存 ~270K tokens
```

---

## Infra 实战映射

### vLLM

vLLM 的设计与本讲的分析一一对应：

**PagedAttention** —— 借鉴 OS 虚拟内存思想管理 KV cache。传统做法为每个序列按 `max_seq_len` 预分配连续内存，实际利用率低。PagedAttention 将 KV cache 切成固定大小的 block（默认 16 tokens/block），按需分配且无需连续。内存利用率从约 60% 提升到 95%+。

**Continuous Batching** —— 不等一个 batch 里所有序列生成完毕再处理下一个 batch，而是序列完成即填入新请求。本质是在 compute-bound（prefill）和 memory-bound（decode）之间动态调度。

### TensorRT-LLM

NVIDIA 的推理编译器从计算量和访存分析出发做优化：

**Layer Fusion** —— 将 element-wise ops（bias add、activation、residual add）融合进前序 matmul kernel，减少中间结果的 HBM 读写。每个被融合的 op 意味着少一次 HBM round trip，在 memory-bound 场景下收益显著。

**GEMM Plugin** —— 根据矩阵具体尺寸选择 tiling 策略，让数据搬运与计算 overlap。不同 shape 的最优策略不同，因此 TRT-LLM 在构建阶段做 auto-tuning。

---

## 跨 Lecture 关联

| 方向 | Lecture | 关联点 |
|------|---------|--------|
| ← 前置 | Lec01 | 效率优化的动机与全局视角 |
| → | Lec03/04 | 剪枝——直接减参数量和计算量，本讲公式是衡量标准 |
| → | Lec05/06 | 量化——降低每个参数的 bit 数，本讲的精度格式分析是基础 |
| → | Lec11 | Tiny Engine——MCU 上的极限内存约束，本讲的内存分析方法论直接沿用 |
| → | Lec12/13 | Transformer / LLM 部署——本讲所有公式的大规模实际应用 |

---

## 面试高频题

**Q1：Linear(1024, 4096) 有多少参数？FP16 推理占多少内存？**

参数量 $= 1024 \times 4096 + 4096 = 4{,}198{,}400 \approx 4.2\text{M}$ 。FP16 每参数 2 bytes → $4.2\text{M} \times 2 = 8.4 \text{ MB}$ 。

补充：现代模型多设 bias=False（LLaMA、Mistral 等），此时参数量 $= 4{,}194{,}304$ ，内存 $\approx 8.0 \text{ MB}$ 。

---

**Q2：参数量和计算量能脱钩吗？举例。**

可以，而且很常见。

- `Conv2d(3, 3, 3)` 只有 81 个参数，作用在 $1000 \times 1000$ 图像上 MACs $\approx 81\text{M}$ 。原因：weight sharing 在空间维度反复使用同一组参数。
- Embedding 层参数量巨大（ $\mathrm{vocab\\_size} \times d_{\mathrm{model}}$ ），但 MACs 几乎为零——仅做一次 gather 查表。

---

**Q3：BF16 vs FP16，为什么 LLM 训练选 BF16？**

两个原因：

1. **动态范围** —— BF16 指数 8 bit（与 FP32 一致），最大值 $\sim 3.4 \times 10^{38}$ 。FP16 指数 5 bit，最大值 65504。训练中 attention score 或 loss 超过 65504 即溢出。BF16 不需要 loss scaling，FP16 必须。
2. **转换简单** —— FP32 转 BF16 截断低 16 位即可，硬件几乎零开销。

代价是 BF16 尾数仅 7 bit（FP16 有 10 bit），精度更低，但对训练收敛影响可忽略。

---

**Q4：LLM decode 为什么是 memory-bound？怎么缓解？**

Decode 时每步只生成一个 token（或极少量），矩阵乘退化为矩阵-向量乘。每个参数读 2 bytes（FP16），做 2 FLOPs，arithmetic intensity $\approx 1$ ，远低于 A100 的 ridge point（约 156）。算力大量空闲。

缓解方法：

- **Continuous batching** —— 攒多请求一起 decode，提高 batch size 从而提升 AI
- **模型量化** —— INT8 / INT4 减少搬运量
- **Speculative decoding** —— 小模型快速草拟多个 token，大模型一次验证

---

**Q5：推理内存 = 参数 × dtype\_bytes 吗？少了什么？**

少了 **KV cache**。自回归生成时必须保留已生成所有 token 的 K、V 向量：

$$
\mathrm{KV\\_cache} = 2 \times L \times n_{\mathrm{kv\\_heads}} \times d_{\mathrm{head}} \times N \times B \times \mathrm{dtype\\_bytes}
$$

长序列 + 大 batch 下 KV cache 可远超模型权重。LLaMA-2 7B 在 batch=128、seq=4096 下 KV cache 约 64 GB，模型权重仅 14 GB。

---

**Q6：混合精度训练能省多少 Model States 内存？**

答案是：**不省**。混合精度下 Model States 仍然是 $16\Psi$ （BF16 权重 $2\Psi$ + BF16 梯度 $2\Psi$ + FP32 master weights $4\Psi$ + FP32 Adam $m$ $4\Psi$ + FP32 Adam $v$ $4\Psi$ ）。跟纯 FP32 一样。

混合精度真正省的是：（a）计算速度——BF16 Tensor Core 快 2–4×；（b）激活值内存——中间结果存 BF16 直接减半。要减 optimizer states 只能靠 ZeRO / FSDP 切分。

---

**Q7：FP8 的两种变体 E4M3 和 E5M2 分别用在哪里？为什么？**

E4M3（4 位指数 + 3 位尾数）用于**前向传播**——权重和激活值的分布比较集中，精度比范围更重要。E5M2（5 位指数 + 2 位尾数）用于**反向传播**——梯度数值波动大，需要更宽的动态范围。

FP8 的动态范围比 BF16 小很多（E4M3 最大 448），所以每个 tensor 需要额外维护一个 FP32 的 scale factor（per-tensor scaling），不能直接截断使用。
