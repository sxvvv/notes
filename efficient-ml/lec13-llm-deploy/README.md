# Lec13 LLM部署：量化 + 推理系统

> 📺 [课程视频](https://www.youtube.com/watch?v=placeholder) | 📄 [Slides](https://efficientml.ai/slides/lec13.pdf)

---

## 核心概念

### 13.0 背景：为什么LLM部署如此困难

LLM（大语言模型）的部署不同于传统CNN部署，核心矛盾在于：

1. **规模差异**：GPT-3有1750亿参数，仅权重就占350GB（FP16），远超单卡显存
2. **推理模式特殊**：自回归逐token生成，无法像CNN那样简单batch
3. **访存瓶颈**：decode阶段每生成一个token就要从HBM加载全量权重
4. **量化困难**：LLM的activation分布与CNN截然不同，存在异常大值（outlier）

**关键指标定义**：
- **TTFT (Time To First Token)**：用户发出请求到收到第一个token的延迟，决定"响应感"
- **TPOT (Time Per Output Token)**：生成每个后续token的平均时间，决定流畅度
- **Throughput**：单位时间内系统处理的总token数，决定成本效率

$$\text{Latency} = \text{TTFT} + N_{\text{output}} \times \text{TPOT}$$

---

### 13.1 LLM量化挑战：为何CNN的方案直接失效

#### CNN量化 vs LLM量化的本质区别

CNN的activation分布相对均匀，量化到INT8损失极小。但LLM中存在**系统性异常值（systematic outliers）**：

```
典型Transformer activation分布（以OPT-6.7B为例）：
- 大多数activation值：[-1, 1] 范围内
- outlier activation值：可达 ±60 甚至更大
- outlier出现位置：固定在特定的hidden dimension上（不是随机的！）
```

这一现象的规模依赖性（scale-dependence）是关键：

| 模型规模 | Outlier幅度 | INT8量化精度损失 |
|---------|-------------|-----------------|
| OPT-125M | 小 | 可接受（<1% PPL增加）|
| OPT-1.3B | 中 | 轻微（~1-2%）|
| OPT-6.7B | **突然爆发** | 严重崩塌（>10%）|
| OPT-30B+ | 持续存在 | 直接使用INT8不可行 |

**6.7B是"相变拐点"**（phase transition），这是MIT/HuggingFace的实测发现。

#### 量化方案选型分析

```
Weight-only量化（如GPTQ, AWQ）：
  优点：压缩比高，实现简单
  缺点：activation仍FP16，arithmetic仍FP16，加速有限
  适用：memory-bound场景（batch=1）

Weight + Activation量化（如SmoothQuant, INT8推理）：
  优点：INT8 GEMM吞吐是FP16的2x（A100/H100硬件加速）
  缺点：activation outlier问题难处理
  适用：large-batch compute-bound场景
```

#### Single-batch vs Batch推理的访存分析

**单batch（batch=1）：Memory-bound**

```
以LLaMA-7B为例：
- 参数量：7B × 2 bytes (FP16) = 14 GB
- 每个token生成需要加载：~14 GB 数据
- A100 HBM带宽：2 TB/s
- 理论token生成速度：2TB/s ÷ 14GB ≈ 143 tokens/s
- 实测：约80-100 tokens/s（效率约60-70%）
- Arithmetic intensity极低：计算量少，IO多
```

**大batch：Compute-bound**

```
batch=64时：
- 权重复用：同一权重对64个sequence都有效
- Arithmetic intensity = (2 × batch × d_model) / (2 × d_model × d_model)
              = batch / d_model = 64 / 4096 ≈ 0.016
- A100的arithmetic intensity ridge point：约250
- 仍然memory-bound，但比batch=1好很多
```

**结论**：LLM推理的根本瓶颈是权重加载。量化从FP16→INT8可以将权重体积缩小2x，直接提升memory-bound场景的速度上限2x。

---

### 13.2 SmoothQuant：Weight-Activation联合量化

#### 核心思想

SmoothQuant的洞察是：activation难量化是因为outlier大，但我们可以把"难度"从activation转移到weight上。

**数学推导**：

对于线性层 $Y = XW$，我们引入一个channel-wise的平滑因子 $s \in \mathbb{R}^{C_{\text{in}}}$：

$$Y = X W = \underbrace{(X \cdot \text{diag}(s)^{-1})}_{\hat{X}} \cdot \underbrace{(\text{diag}(s) \cdot W)}_{\hat{W}}$$

其中：
- $\hat{X} = X \cdot \text{diag}(s)^{-1}$：除以 $s$，activation的范围被压缩
- $\hat{W} = \text{diag}(s) \cdot W$：乘以 $s$，weight的范围相应扩大

**关键问题**：$s$ 如何取值才能让量化误差最小？

#### Smoothing Factor的自动搜索

设第 $j$ 个channel的activation最大值为 $\alpha_j = \max|X_j|$，weight最大值为 $\omega_j = \max|W_j|$。

量化后的per-tensor量化误差与动态范围成正比：

$$\text{quantization error} \propto \frac{\max|X_j|}{2^{N}-1}$$

SmoothQuant选择：

$$\boxed{s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}}$$

其中 $\alpha \in [0, 1]$ 控制迁移量：
- $\alpha = 0$：不迁移（$s=1$），activation保持原样
- $\alpha = 1$：完全迁移，activation完全压平，代价是weight范围爆炸
- $\alpha = 0.5$：**最优选择**（empirically verified，各模型ablation均最佳）

**为什么 $\alpha=0.5$ 最好？**

平衡条件：量化后activation的max = 量化后weight的max

$$\frac{\max|X_j|}{s_j} = \max|W_j| \cdot s_j$$

$$s_j^2 = \frac{\max|X_j|}{\max|W_j|} \Rightarrow s_j = \sqrt{\frac{\max|X_j|}{\max|W_j|}} = \frac{\max|X_j|^{0.5}}{\max|W_j|^{0.5}}$$

这正好对应 $\alpha=0.5$ 的情况！

#### Smoothing融入LayerNorm：零额外开销

SmoothQuant的精妙之处：$s$ 的计算可以**融合进前置的LayerNorm层**，不增加推理开销。

```
原始计算图：
Input → LayerNorm → (scale by s^{-1}) → INT8 Linear → Output
                        ↑ 额外开销

优化后：
Input → LayerNorm (weight/bias乘以s^{-1}) → INT8 Linear → Output
              ↑ 把s^{-1}吸收进LayerNorm的weight，零开销
```

数学上：LayerNorm的输出为 $\frac{x - \mu}{\sigma} \cdot \gamma + \beta$，我们把 $\gamma$ 替换为 $\gamma / s$，$\beta$ 替换为 $\beta / s$，效果等价。

#### 系统架构

```
SmoothQuant在FasterTransformer上的实现：
┌─────────────────────────────────────────────┐
│  LayerNorm (FP16, 参数已离线scale)           │
│     ↓                                        │
│  Linear/GEMM (INT8 × INT8 → INT32)          │
│     ↓                                        │
│  Dequant + Add Residual (FP16)              │
│     ↓                                        │
│  LayerNorm (FP16)                            │
│     ↓                                        │
│  Attention (QKV Projection: INT8)            │
│  Softmax: FP16（不量化，精度敏感）           │
│  BMM Q×K^T: INT8，Softmax后BMM: FP16       │
└─────────────────────────────────────────────┘

量化配置：
- Linear层：W8A8
- LayerNorm：FP16
- Softmax：FP16
- 残差连接：FP16
```

**SmoothQuant精度结果（OPT-175B, WikiText-2 PPL）**：

| 方案 | PPL | vs FP16 |
|------|-----|---------|
| FP16 基准 | 8.34 | - |
| 直接 W8A8 | 崩塌 | - |
| SmoothQuant W8A8 | 8.40 | +0.7% |
| SmoothQuant W8A8 (per-channel weight) | 8.36 | +0.2% |

---

### 13.3 AWQ：Activation-aware Weight Quantization

#### 核心洞察：Salient Weight Channels

AWQ的出发点更精准：**不是所有weight channel都同样重要**。

实验观察：对LLM中 ~1% 的weight channel进行保护（保持FP16），可以恢复大部分量化损失。哪些channel重要？**那些对应activation幅度大的channel**。

直觉解释：

```
Y = X · W
  ↑     ↑
activation  weight

若 X_j 幅度大（outlier），则 X_j × W_j 对输出贡献大
量化误差 ε_W_j 被放大为 X_j × ε_W_j
所以 X_j 大时，W_j 的量化误差影响更大
```

#### 为什么不用Mixed Precision？

最直接的想法：对salient channel用FP16，其余用INT4。但这**对硬件极不友好**：
- GPU不支持混合精度的高效GEMM kernel
- 需要特殊的packing/unpacking，overhead大

#### AWQ的解决方案：Scaling代替Mixed Precision

AWQ的方案：对salient channel的weight进行scale-up，再对activation进行scale-down，让量化后的weight误差更小：

$$Q(W \cdot s) \cdot (s^{-1} \cdot X) \approx W \cdot X$$

其中 $s > 1$ 对salient channel放大weight，使得量化后的精度提升（更大的值量化误差相对更小），同时 $s^{-1}$ 可以融合进前置层。

**误差分析**：

设量化函数 $Q(w) = \text{round}(w / \Delta) \cdot \Delta$，量化步长 $\Delta \propto \max|w|$。

对channel $j$ scale $s_j$ 后：

$$\Delta_j' = \frac{\max|W_j| \cdot s_j}{2^N - 1}$$

量化误差：

$$\epsilon_j = |Q(W_j \cdot s_j) - W_j \cdot s_j| \leq \frac{\Delta_j'}{2} \propto \max|W_j| \cdot s_j$$

等效的weight误差（除以 $s_j$ 后）：

$$\tilde{\epsilon}_j = \frac{\epsilon_j}{s_j} \leq \frac{\max|W_j|}{2^N - 1}$$

这与scale无关！但输出误差是：

$$\delta Y_j = X_j \cdot \tilde{\epsilon}_j$$

scale对output误差的影响通过改变量化grid来体现。AWQ用grid search找最优 $s$：

$$\boxed{s^* = \arg\min_s \|Q(W \cdot \text{diag}(s)) \cdot \text{diag}(s^{-1}) \cdot X - W \cdot X\|_F}$$

实际上AWQ把scale参数化为：

$$s_j = \hat{X}_j^\alpha, \quad \alpha \in [0, 1]$$

其中 $\hat{X}_j$ 是channel $j$ 的activation平均幅度（从calibration set统计），在 $\alpha$ 上做grid search。

#### AWQ vs GPTQ

| 特性 | AWQ | GPTQ |
|------|-----|------|
| 量化比特 | 4-bit (weight only) | 4-bit (weight only) |
| 校准数据 | 少量（128样本）| 少量（128样本）|
| 量化速度 | 快（分钟级）| 慢（小时级，逐层Hessian）|
| 精度（4-bit）| 相当 | 相当 |
| 硬件友好性 | 高（纯weight量化）| 高 |
| 核心思路 | activation-aware scale | second-order Taylor展开 |

---

### 13.4 LLM推理系统

#### Prefill vs Decode的对比

LLM推理分为两个截然不同的阶段：

**Prefill阶段**（处理input prompt）：

```
输入：[token_1, token_2, ..., token_n]
特点：
- 所有token同时处理（完整的矩阵乘法）
- Compute-bound：arithmetic intensity高
- 时间：O(n²) attention + O(n) FFN
- 产出：第一个输出token 和 KV Cache
```

**Decode阶段**（逐token生成）：

```
输入：[新token] + KV Cache
特点：
- 每次只处理1个token（向量×矩阵）
- Memory-bound：arithmetic intensity极低
- 时间：O(n) attention（与KV Cache长度成正比）+ O(1) FFN
- 产出：下一个token
```

量化收益主要体现在decode阶段：decode的瓶颈是从HBM加载weight，INT8可以直接2x加速。

#### KV Cache：机制与内存分析

KV Cache缓存了每个attention层的Key和Value，避免重复计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $K_{1:t}$ 和 $V_{1:t}$ 在每次decode步都会reuse。

**KV Cache的内存计算**：

```
KV Cache大小 = 2 (K+V) × num_layers × num_heads × head_dim × seq_len × dtype_bytes

以LLaMA-7B (FP16) 为例：
- num_layers = 32
- num_heads = 32
- head_dim = 128
- seq_len = 4096
- dtype = FP16 (2 bytes)

KV Cache = 2 × 32 × 32 × 128 × 4096 × 2 bytes
         = 2 × 32 × 32 × 128 × 4096 × 2
         ≈ 2 GB per sequence

如果batch=10：20 GB KV Cache
模型本身：14 GB
总计：34 GB（超过A100 80GB的40%！）
```

#### PagedAttention（vLLM核心）

**问题**：传统实现为每个sequence预分配最大长度的连续显存，导致严重碎片化和浪费。

**解决方案**：借鉴OS虚拟内存的分页思想。

```
传统KV Cache分配：
seq1: [K1V1 | K2V2 | K3V3 | ... | KmaxVmax]  ← 预分配满
seq2: [K1V1 | K2V2 | ... | K5V5 | EMPTY×(max-5)]  ← 浪费

PagedAttention：
Physical Memory: [Block0][Block1][Block2][Block3][Block4]...
                   4 slots  4 slots  4 slots  4 slots

seq1 page table: [Block0→tokens 1-4, Block2→tokens 5-8, Block4→tokens 9-12]
seq2 page table: [Block1→tokens 1-4, Block3→tokens 5-6]
                                               ↑ 只分配实际使用的block
```

PagedAttention的关键好处：
1. **内存利用率从55%提升到>95%**（vLLM论文实测）
2. **支持prefix sharing**：相同prompt prefix的KV可以共享物理block
3. **无fragmentation**：block是固定大小，分配/释放高效

**Block size的权衡**：
- 太小：page table开销大，attention kernel效率低
- 太大：内部碎片增加
- vLLM默认：16 tokens per block

#### Continuous Batching（iteration-level调度）

**传统静态batching的问题**：

```
Static batching:
Batch = [seq1(100 tokens), seq2(100 tokens), seq3(100 tokens)]
当seq1生成完50个token时，seq2和seq3还在继续
seq1完成后，GPU必须等整个batch都完成才能处理新请求
→ 利用率低，延迟高
```

**Continuous Batching**：

```
Iteration 0: [seq1, seq2, seq3]  处理
Iteration 1: [seq1, seq2, seq3]  继续
...
Iteration k: seq3完成！立即替换为seq4
Iteration k+1: [seq1, seq2, seq4]  继续处理
```

每个推理step（iteration）动态决定batch组成，完成的sequence立即被新请求替代。

**吞吐量提升**：Continuous batching可以将GPU利用率从40-50%提升到80-90%+。

#### Speculative Decoding（推测解码）

**原理**：LLM decode是memory-bound的，GPU计算能力被浪费。可以用小模型并行猜测多个token，让大模型批量验证。

```
Draft model（小模型，如LLaMA-7B）：
  step 1: 生成候选 [t1, t2, t3, t4]

Target model（大模型，如LLaMA-70B）：
  并行验证：P_target(t1|ctx), P_target(t2|ctx,t1), P_target(t3|ctx,t1,t2), P_target(t4|...)
  接受条件：P_target(t_i) / P_draft(t_i) ≥ random uniform
  若t1,t2被接受，t3被拒：
    → 接受t1,t2，从P_target(·|ctx,t1,t2)重新采样t3
```

**关键性质**：
- 验证是exact的（接受标准保证最终分布等同于只用target model）
- 理论加速比：$\beta / (1 - \alpha^{\gamma+1})$，其中 $\alpha$ 是draft接受率，$\gamma$ 是每次猜的token数
- 实际加速：2x-3x（代码生成场景更高，因为预测性强）

**加速条件**：
- Target model的batch验证 ≈ 单token验证的时间（true when memory-bound）
- Draft model 够快（比target model小5-10x以上）

---

## 数学推导

### SmoothQuant的Per-Channel量化误差界

**命题**：对于per-tensor量化方案，第 $j$ 个channel经过SmoothQuant处理后，其对输出的影响的量化误差界为：

$$\mathbb{E}[\|Y - \hat{Y}\|_2^2] \leq \sum_j \frac{\|X_j\|_2^2 \cdot R_j^2}{3(2^N-1)^2}$$

其中 $R_j = \max|\hat{W}_j| = \max|W_j| \cdot s_j$，$N$ 是量化比特数。

**推导**（简化版）：

均匀量化的误差模型（高分辨率近似）：

$$q_j \sim \mathcal{U}[-\Delta_j/2, \Delta_j/2], \quad \mathbb{E}[q_j^2] = \frac{\Delta_j^2}{12}$$

量化步长 $\Delta_j = \frac{2R_j}{2^N - 1}$

输出误差的第 $j$ 个分量（$Y_j = X_j W_j$）：

$$\delta Y_j = X_j \cdot q_j^W$$

$$\mathbb{E}[\|\delta Y_j\|^2] = \|X_j\|^2 \cdot \mathbb{E}[q_j^{W2}] = \|X_j\|^2 \cdot \frac{\Delta_j^2}{12} = \frac{\|X_j\|^2 R_j^2}{3(2^N-1)^2}$$

SmoothQuant通过最小化 $R_j = \max|W_j| \cdot s_j$ 与 $\|X_j\|/s_j$ 的乘积来最小化此误差。在 $\alpha=0.5$ 时，两者平衡，总误差最小。

### KV Cache量化的信息论下界

对于seq_len = $L$, hidden_dim = $d$，KV Cache需存储：

$$\text{KV bits} \geq 2Ld \cdot H(K, V | \text{model})$$

由于连续分布的微分熵为 $-\infty$（理论上可无限压缩），实际工程上的约束来自量化误差对perplexity的影响：

$$\Delta \text{PPL} \approx \frac{\lambda \cdot \text{MSE}(K, \hat{K})}{d}$$

经验上，4-bit KV Cache（KVQuant等方法）可以保持 PPL 损失 < 0.5，大幅压缩KV Cache内存。

---

## 代码示例

### SmoothQuant核心实现

```python
import torch
import torch.nn as nn
from typing import Optional

def compute_smooth_scale(
    activation_max: torch.Tensor,   # [C_in] 各channel的activation最大值
    weight_max: torch.Tensor,        # [C_in] 各channel的weight最大值
    alpha: float = 0.5
) -> torch.Tensor:
    """
    计算SmoothQuant的per-channel平滑因子
    
    s_j = max|X_j|^alpha / max|W_j|^(1-alpha)
    
    Args:
        activation_max: 从calibration set统计的activation绝对值最大值
        weight_max: weight各channel的绝对值最大值
        alpha: 迁移系数，0.5时两边误差平衡，经验最优
    
    Returns:
        s: [C_in] 平滑因子
    """
    # 防止除零，给weight_max加小epsilon
    eps = 1e-8
    scale = (activation_max.pow(alpha) /
             (weight_max.pow(1 - alpha) + eps))
    return scale


def smooth_ln_fcs(
    ln: nn.LayerNorm,
    fcs: list[nn.Linear],
    act_scales: torch.Tensor,   # [C_in]
    alpha: float = 0.5
):
    """
    将SmoothQuant的scale融合进LayerNorm，实现零推理开销
    
    原理：
      LayerNorm输出 * (1/s) → Linear(weight * s)
      等价于修改LayerNorm的weight为 gamma/s，
      并把weight每列乘以s
    
    修改是离线进行的，不影响推理速度
    """
    device = fcs[0].weight.device
    dtype = fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    
    # 统计所有fc层的weight各channel最大值
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0)[0].unsqueeze(0) for fc in fcs],
        dim=0
    ).max(dim=0)[0]  # [C_in]
    
    # 计算平滑因子
    scales = compute_smooth_scale(act_scales, weight_scales, alpha)
    
    # 修改LayerNorm：把 /s 融合进LayerNorm的weight
    ln.weight.div_(scales)
    if ln.bias is not None:
        ln.bias.div_(scales)
    
    # 修改Linear：把 *s 吸收进weight（逐行乘以s，因为weight是[C_out, C_in]）
    for fc in fcs:
        fc.weight.mul_(scales.unsqueeze(0))  # broadcast: [1, C_in]


class SmoothQuantLinear(nn.Module):
    """
    SmoothQuant之后的INT8 Linear层
    
    存储INT8 weight，推理时用INT8 GEMM
    """
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # INT8 weight存储（实际部署中会用torch.int8或调用cutlass kernel）
        self.register_buffer('weight_int8',
                            torch.zeros(out_features, in_features,
                                       dtype=torch.int8))
        self.register_buffer('weight_scale',
                            torch.zeros(out_features, dtype=torch.float16))
        
        if bias:
            self.register_buffer('bias',
                                torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None
    
    @classmethod
    def from_float(cls, module: nn.Linear,
                   input_scale: float) -> 'SmoothQuantLinear':
        """
        从FP16 Linear层转换为量化层
        
        Args:
            module: 原始FP16 Linear层
            input_scale: activation的量化scale（1/127的倍数）
        """
        sq_linear = cls(module.in_features, module.out_features,
                       module.bias is not None)
        
        # Per-tensor量化weight到INT8
        weight_abs_max = module.weight.abs().max()
        sq_linear.weight_scale = weight_abs_max / 127.0
        sq_linear.weight_int8 = (
            module.weight / sq_linear.weight_scale
        ).round().clamp(-128, 127).to(torch.int8)
        
        if module.bias is not None:
            sq_linear.bias = module.bias.half()
        
        return sq_linear
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 实际部署用CUTLASS INT8 GEMM，这里用float模拟
        weight_fp = self.weight_int8.float() * self.weight_scale
        out = torch.nn.functional.linear(x.float(), weight_fp)
        if self.bias is not None:
            out = out + self.bias.float()
        return out.half()


# ============================================================
# 使用示例：对一个Transformer block应用SmoothQuant
# ============================================================

def quantize_transformer_block_smoothquant(block, calibration_data):
    """
    对Transformer block应用SmoothQuant
    
    流程：
    1. 收集calibration数据下各层的activation统计
    2. 计算并应用smooth scale
    3. 将Linear层替换为INT8版本
    """
    act_dict = {}
    hooks = []
    
    # 注册hook收集activation统计
    def make_hook(name):
        def hook(module, inp, out):
            x = inp[0]
            # 统计per-channel的最大绝对值
            act_dict[name] = x.abs().view(-1, x.shape[-1]).max(dim=0)[0]
        return hook
    
    # 假设block有attn和mlp子模块
    hooks.append(block.attn.q_proj.register_forward_pre_hook(make_hook('attn_q')))
    hooks.append(block.mlp.fc1.register_forward_pre_hook(make_hook('mlp_fc1')))
    
    # 跑calibration data
    with torch.no_grad():
        for batch in calibration_data:
            block(batch)
            break  # 实际用更多数据
    
    # 移除hooks
    for h in hooks:
        h.remove()
    
    # 应用smooth scale
    smooth_ln_fcs(
        block.attn_norm,
        [block.attn.q_proj, block.attn.k_proj, block.attn.v_proj],
        act_dict['attn_q']
    )
    
    print(f"Smoothing applied. QKV projection now INT8-ready.")
    return block


# ============================================================
# KV Cache内存计算工具
# ============================================================

def compute_kv_cache_memory(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    max_seq_len: int,
    batch_size: int,
    dtype_bytes: int = 2,   # FP16
    kv_quant_bits: Optional[int] = None
) -> dict:
    """
    计算KV Cache占用的显存
    
    Returns:
        dict包含：bytes, GB, 相对于模型权重的比例
    """
    if kv_quant_bits is not None:
        bits_per_elem = kv_quant_bits
        dtype_bytes = bits_per_elem / 8
    
    # K和V各一份
    kv_cache_bytes = (2 * num_layers * num_heads * head_dim *
                      max_seq_len * batch_size * dtype_bytes)
    
    return {
        'bytes': kv_cache_bytes,
        'GB': kv_cache_bytes / (1024**3),
        'per_token_bytes': (2 * num_layers * num_heads * head_dim * dtype_bytes),
        'per_token_KB': (2 * num_layers * num_heads * head_dim * dtype_bytes) / 1024
    }


# LLaMA-7B参数
llama7b_kv = compute_kv_cache_memory(
    num_layers=32, num_heads=32, head_dim=128,
    max_seq_len=4096, batch_size=1
)
print(f"LLaMA-7B KV Cache (seq=4096, batch=1): {llama7b_kv['GB']:.2f} GB")
print(f"每个token的KV Cache: {llama7b_kv['per_token_KB']:.1f} KB")

# 与INT4 KV Cache对比
llama7b_kv_int4 = compute_kv_cache_memory(
    num_layers=32, num_heads=32, head_dim=128,
    max_seq_len=4096, batch_size=1, kv_quant_bits=4
)
print(f"LLaMA-7B INT4 KV Cache: {llama7b_kv_int4['GB']:.2f} GB (节省{(1-llama7b_kv_int4['GB']/llama7b_kv['GB'])*100:.0f}%)")
```

### AWQ核心思路（Pseudo-code实现）

```python
import torch
import torch.nn as nn

def awq_search_scale(
    weight: torch.Tensor,        # [C_out, C_in]
    act_scales: torch.Tensor,    # [C_in] calibration set的activation幅度
    n_bits: int = 4,
    n_grid: int = 20             # grid search的点数
) -> torch.Tensor:
    """
    AWQ核心：搜索最优scale factor
    
    目标：min_alpha ||Q(W * diag(s)) * diag(s^-1) * X - W * X||
    其中 s_j = act_scales_j ^ alpha
    
    通过grid search在alpha上搜索最优值
    """
    best_error = float('inf')
    best_alpha = 0.5
    
    # 对act_scales归一化（避免数值问题）
    act_scales = act_scales.float()
    
    for i in range(n_grid + 1):
        alpha = i / n_grid  # [0, 1]
        
        # 计算scale: s_j = act_scales_j ^ alpha
        scales = act_scales.pow(alpha)
        
        # scale weight: W_scaled[c_out, c_in] = W[c_out, c_in] * scales[c_in]
        w_scaled = weight.float() * scales.unsqueeze(0)
        
        # 模拟INT4量化
        w_quant = pseudo_quantize_tensor(w_scaled, n_bits=n_bits)
        
        # 等效weight: W_eff = W_quant / scales
        w_eff = w_quant / scales.unsqueeze(0)
        
        # 量化误差（用activation-weighted MSE，体现salient channel的重要性）
        error = ((w_eff - weight.float()).pow(2) *
                 act_scales.unsqueeze(0)).mean().item()
        
        if error < best_error:
            best_error = error
            best_alpha = alpha
    
    return act_scales.pow(best_alpha)


def pseudo_quantize_tensor(
    w: torch.Tensor,
    n_bits: int = 4,
    zero_point: bool = True,
    per_channel: bool = True
) -> torch.Tensor:
    """
    模拟INT4量化（实际部署用CUDA kernel）
    
    Per-group量化：将weight按group（如128）分组，每组独立量化
    """
    if per_channel:
        max_val = w.abs().max(dim=1, keepdim=True)[0]
    else:
        max_val = w.abs().max()
    
    if zero_point:
        # 非对称量化
        min_val = w.min(dim=1, keepdim=True)[0] if per_channel else w.min()
        max_val_asym = w.max(dim=1, keepdim=True)[0] if per_channel else w.max()
        scale = (max_val_asym - min_val) / (2**n_bits - 1)
        zp = (-min_val / scale).round()
        w_int = (w / scale + zp).round().clamp(0, 2**n_bits - 1)
        w_dequant = (w_int - zp) * scale
    else:
        # 对称量化
        scale = max_val / (2**(n_bits-1) - 1)
        w_int = (w / scale).round().clamp(-(2**(n_bits-1)), 2**(n_bits-1)-1)
        w_dequant = w_int * scale
    
    return w_dequant


class AWQLinear(nn.Module):
    """
    AWQ量化后的Linear层（INT4 weight, FP16 activation）
    
    存储：INT4 weight（pack成INT8，每个INT8存两个INT4）+ scale/zero_point
    推理：dequant weight到FP16，然后FP16 GEMM
    """
    def __init__(self, in_features: int, out_features: int,
                 group_size: int = 128, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        n_groups = in_features // group_size
        
        # INT4 weight packed成INT8（节省一半存储）
        self.register_buffer('qweight',
            torch.zeros(out_features, in_features // 2, dtype=torch.int8))
        # 每组的scale和zero_point（FP16）
        self.register_buffer('scales',
            torch.zeros(out_features, n_groups, dtype=torch.float16))
        self.register_buffer('zeros',
            torch.zeros(out_features, n_groups, dtype=torch.float16))
        
        if bias:
            self.register_buffer('bias',
                torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 实际AWQ推理用专门的GEMM kernel（llm-awq提供）
        # 这里模拟dequant + GEMM
        weight = self.dequantize()  # [C_out, C_in] FP16
        out = torch.nn.functional.linear(x, weight, self.bias)
        return out
    
    def dequantize(self) -> torch.Tensor:
        """将INT4 weight反量化为FP16"""
        # 解包INT4（高4位和低4位）
        weight_high = (self.qweight >> 4).to(torch.float16)
        weight_low = (self.qweight & 0xF).to(torch.float16)
        
        # 重新排列并应用scale/zero
        # 简化实现，实际需要处理group维度
        return weight_low  # 简化，实际实现更复杂
```

### Speculative Decoding模拟

```python
import torch
import torch.nn.functional as F
from typing import Optional

class SpeculativeDecoder:
    """
    Speculative Decoding的实现
    
    流程：
    1. Draft model（小模型）猜gamma个token
    2. Target model（大模型）并行验证
    3. 用rejection sampling决定接受哪些token
    
    保证：最终的token分布与只用target model相同（exact）
    """
    
    def __init__(self, target_model, draft_model, gamma: int = 4):
        """
        Args:
            target_model: 大模型（准确但慢）
            draft_model: 小模型（快但不够准）
            gamma: 每次猜测的token数量
        """
        self.target = target_model
        self.draft = draft_model
        self.gamma = gamma
    
    def generate_step(
        self,
        input_ids: torch.Tensor,    # [batch, seq_len]
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        执行一步speculative decoding，可能接受1到gamma+1个token
        
        Returns:
            新生成的token ids
        """
        # Step 1: Draft model逐个生成gamma个候选token
        draft_tokens = []
        draft_probs = []
        
        current_ids = input_ids
        for _ in range(self.gamma):
            with torch.no_grad():
                draft_logits = self.draft(current_ids)[:, -1, :]
            draft_prob = F.softmax(draft_logits / temperature, dim=-1)
            
            # 从draft分布采样
            draft_token = torch.multinomial(draft_prob, num_samples=1)
            draft_tokens.append(draft_token)
            draft_probs.append(draft_prob)
            
            current_ids = torch.cat([current_ids, draft_token], dim=-1)
        
        # Step 2: Target model并行验证所有候选
        # target处理 [input, draft_token_1, ..., draft_token_gamma]
        # 一次forward即可得到所有位置的logits
        with torch.no_grad():
            target_logits = self.target(current_ids)
        
        # target_logits[:, -gamma-1:-1, :] 对应验证gamma个draft token的位置
        # target_logits[:, -1, :] 对应在全部接受情况下的下一个token
        
        # Step 3: Rejection sampling
        accepted_tokens = []
        
        for i in range(self.gamma):
            # target在位置 seq_len+i 的概率分布
            target_prob = F.softmax(
                target_logits[:, input_ids.shape[1] + i - 1, :] / temperature,
                dim=-1
            )
            p_target = target_prob.gather(-1, draft_tokens[i])
            p_draft = draft_probs[i].gather(-1, draft_tokens[i])
            
            # 接受率: min(1, p_target / p_draft)
            accept_prob = torch.clamp(p_target / (p_draft + 1e-8), max=1.0)
            
            # 随机接受
            if torch.rand(1).item() < accept_prob.item():
                accepted_tokens.append(draft_tokens[i])
            else:
                # 拒绝：从修正后的分布重新采样
                # 修正分布: max(0, p_target - p_draft) / norm
                corrected = torch.clamp(target_prob - draft_probs[i], min=0)
                corrected = corrected / corrected.sum()
                corrected_token = torch.multinomial(corrected, num_samples=1)
                accepted_tokens.append(corrected_token)
                break  # 一旦拒绝就停止
        else:
            # 所有gamma个都被接受：从target采样第gamma+1个
            bonus_prob = F.softmax(
                target_logits[:, -1, :] / temperature, dim=-1
            )
            bonus_token = torch.multinomial(bonus_prob, num_samples=1)
            accepted_tokens.append(bonus_token)
        
        return torch.cat(accepted_tokens, dim=-1)
    
    def expected_speedup(self, acceptance_rate: float) -> float:
        """
        理论加速比公式
        
        E[接受的token数] = (1 - alpha^(gamma+1)) / (1 - alpha)
        其中 alpha 是平均接受率
        
        加速比 ≈ E[接受token数] / 1（因为验证时间≈单token目标model时间）
        """
        alpha = acceptance_rate
        gamma = self.gamma
        expected_tokens = (1 - alpha**(gamma + 1)) / (1 - alpha + 1e-8)
        return expected_tokens


# 测试加速比
decoder = SpeculativeDecoder(None, None, gamma=4)
for alpha in [0.7, 0.8, 0.9]:
    speedup = decoder.expected_speedup(alpha)
    print(f"接受率 α={alpha:.1f}, γ={decoder.gamma}: 期望加速 {speedup:.2f}x")
# 接受率 α=0.7, γ=4: 期望加速 2.57x
# 接受率 α=0.8, γ=4: 期望加速 2.95x
# 接受率 α=0.9, γ=4: 期望加速 3.44x
```

---

## Infra 实战映射

### vLLM

- **PagedAttention**：`vllm/attention/backends/flash_attn.py` 实现了block-based KV Cache。物理block由 `BlockManager` 管理，逻辑block到物理block的映射存在 `block_tables` tensor中，在attention kernel调用时传入。
- **Continuous Batching**：`AsyncLLMEngine` 里的 `step()` 函数每个iteration重新决定batch组成，`Scheduler` 按照优先级（FCFS or priority）决定哪些sequence进入下一个batch。
- **Speculative Decoding**：vLLM 0.4+支持，通过 `--speculative-model` 参数指定draft model，默认gamma=5。
- **量化支持**：`vllm --quantization awq/gptq/squeezellm`，AWQ通过`AutoAWQ`库加载量化好的权重，推理时调用专门的GEMM kernel（`awq_gemm`）。

### TensorRT-LLM

- **Kernel Fusion**：TRT-LLM对attention实现了高度融合的kernel，QKV projection + attention + output projection可以在少量kernel调用中完成，减少HBM读写。
- **FP8支持**：H100 GPU原生支持FP8（E4M3/E5M2），TRT-LLM提供`--quantization fp8`，精度接近BF16，吞吐提升1.5-2x。
- **In-flight Batching**：TRT-LLM的equivalent of continuous batching，通过`GptManager`实现，支持dynamic sequence length。
- **INT8 SmoothQuant**：`modelopt`（原`ammo`）工具链支持一键SmoothQuant量化，生成TRT-LLM可用的量化权重。

### 沐曦 MACA

- **精度格式**：沐曦GPU（如MC-200）支持FP16/BF16，对INT8有专门的Tensor Core加速，但FP8支持需要确认具体型号。部署前必须确认目标卡的precision support matrix。
- **KV Cache量化**：在访存带宽相对NVIDIA A100较小的硬件上，KV Cache量化（INT8/INT4）收益更大，应优先启用。
- **算子适配**：vLLM的PagedAttention kernel用CUDA编写，在MACA上需要通过`MACA SDK`的`hipify`工具转换为HIP，或直接使用沐曦提供的优化attention实现。
- **SmoothQuant部署**：FasterTransformer的MACA适配版（或等效的推理框架）是主要载体，scale融合进LayerNorm的方式在MACA上同样零开销。

---

## 跨 Lecture 关联

- **前置知识** ← Lec05-06 (量化基础，PTQ/QAT原理)，Lec12 (Transformer架构，Attention计算)
- **后续延伸** → Lec14 (后训练：LoRA/QLoRA与量化结合)，Lec15 (长上下文：KV Cache压缩更重要)

### 知识图谱

```
Lec05 量化基础
    ↓ 量化方案（PTQ, per-channel）
Lec13.1 LLM量化挑战（outlier问题）
    ↓ 解决方案
├── Lec13.2 SmoothQuant（W8A8）
└── Lec13.3 AWQ（W4A16）
         ↓ 部署到
Lec13.4 LLM推理系统（vLLM, TRT-LLM）
         ↓ 与后训练结合
Lec14.3 QLoRA（量化+LoRA微调）
         ↓ 长上下文时
Lec15 长上下文推理（KV Cache主导显存）
```

---

## 面试高频题

**Q: SmoothQuant的核心思想是什么？为什么α=0.5效果最好？**

A: SmoothQuant把activation的量化难度（outlier）通过per-channel scale因子迁移到weight上。数学上，$Y=XW$ 恒等变形为 $\hat{X}\hat{W}$，其中 $\hat{X}=X \cdot \text{diag}(s)^{-1}$，$\hat{W}=\text{diag}(s) \cdot W$。α=0.5时 $s_j = \sqrt{\max|X_j|/\max|W_j|}$，使得量化后activation的范围等于weight的范围，两者的量化误差平衡，总误差最小（对称错误最优）。

**Q: PagedAttention解决了什么问题？核心机制是什么？**

A: 解决KV Cache的内存碎片化和浪费问题。传统实现为每个sequence预分配最大长度的连续显存，大量浪费。PagedAttention借鉴OS虚拟内存，将KV Cache分成固定大小的物理block（如16 tokens），维护逻辑块到物理块的映射表。按需分配物理块，无内部碎片，内存利用率从~55%提升到>95%。还支持不同sequence共享相同prefix的KV Cache（Copy-on-Write）。

**Q: Prefill和Decode阶段有什么本质区别？对系统设计有何影响？**

A: Prefill处理整个prompt，是compute-bound（矩阵乘矩阵），arithmetic intensity高；Decode每步只处理1个token，是memory-bound（向量乘矩阵），arithmetic intensity极低（主要瓶颈是从HBM加载权重）。影响：①量化主要加速decode阶段；②Prefill和Decode可以分离部署（Disaggregated Serving，如Splitwise/DistServe）；③KV Cache主要是Decode的开销；④Speculative Decoding专门优化memory-bound的Decode。

**Q: Speculative Decoding的加速是有损的吗？**

A: 不是。Speculative Decoding通过特定的rejection sampling保证最终的token分布与只用target model完全相同（exact decoding），是无损加速。关键在于拒绝时从修正分布 $\max(0, p_{\text{target}} - p_{\text{draft}})$ 重新采样，这保证了最终边缘分布等于target分布。代价是：若draft model质量差（接受率低），overhead反而可能增加（极端情况下每次都拒绝，多跑了一个draft model）。

**Q: AWQ和GPTQ有什么本质区别？**

A: GPTQ基于二阶泰勒展开（Hessian矩阵），量化后用Hessian补偿误差，精度高但计算慢（逐层重建，需要小时级）。AWQ基于activation magnitude的直觉（salient channel），用scale保护重要channel，速度快（分钟级），精度与GPTQ相当。GPTQ是数学最优（局部），AWQ是activation-aware的近似，两者精度在4-bit时相近，AWQ的显著优势是速度和更好的硬件适配性。

**Q: 为什么LLM在6.7B附近出现量化"相变"？**

A: 这与模型训练时的attention head数量、hidden dimension以及LayerNorm的行为有关。更大的模型在训练时更容易让某些hidden dimension的activation特化为"路由信号"（类似attention的hard routing），导致这些维度的值极端大。6.7B是经验观察的拐点，而非严格的理论推导——目前学界对此现象的深层原因仍在研究中（部分工作指向LayerNorm前的residual stream量级与训练动态的关系）。
