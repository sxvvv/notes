# 06 进阶量化 (Quantization Part II)

> 📺 [Lecture 06 - Quantization (Part II)](https://youtu.be/sTz2tXG1T0c)
> 📄 [Slides](https://hanlab.mit.edu/courses/2024-fall-65940)

---

## 6.1 Post-Training Quantization (PTQ)

PTQ = 训练完成后量化，**不需要重新训练**。这是工业界最常用的方式。

### 6.1.1 Weight 量化

权重是静态的，量化相对简单：

| 粒度 | 公式 | 精度 | 存储 overhead |
|------|------|------|-------------|
| Per-tensor | 一个 S, Z 用于整个 W | 低 | 2 个数 |
| Per-channel | 每个输出 channel 一个 S, Z | 高 | 2×C_out 个数 |
| Group (per-vector) | 每 group_size 个值一组 | 最高 | 2×(total/group_size) 个数 |

**Weight Equalization**: 相邻层的 scale 可以合并重分配
- 思路: `y = ReLU(W2 · W1 · x)`，把 W1 的 scale 迁移到 W2
- 效果: 让两层的值域都更均匀

**Adaptive Rounding**: 不用 nearest rounding，而是学习最优的 round 方向
- `q = floor(r/S) + p`，p ∈ {0, 1} 是学习的
- 目标: 最小化重建误差 $\min \|\hat{W}x - Wx\|^2$

### 6.1.2 Activation 量化

Activation 随输入变化，需要统计其范围：

**方法一: 训练时统计 (EMA)**
```python
# 指数移动平均统计 activation 范围
ema_max = momentum * ema_max + (1 - momentum) * current_max
scale = ema_max / qmax
```

**方法二: Calibration (校准)** — 用少量代表性数据

| Calibration 方法 | 原理 | 效果 |
|-----------------|------|------|
| MinMax | scale = (max - min) / (qmax - qmin) | 简单但 outlier 敏感 |
| Percentile | 取 99.9% 分位数作为 max | 抗 outlier，工程常用 |
| KL-divergence (Entropy) | 找使量化前后分布 KL 散度最小的 S | 精度最好，TensorRT 默认 |
| MSE | 最小化 $\|\text{round}(r/S) \times S - r\|^2$ | 平衡方案 |

```python
import numpy as np

def calibrate_kl(fp32_tensor, n_bits=8):
    """KL-divergence calibration — TensorRT 方式"""
    n_bins = 2 ** n_bits
    hist, edges = np.histogram(fp32_tensor, bins=4096)
    # 逐步缩小范围，找 KL 散度最小的阈值
    best_kl = float('inf')
    best_threshold = edges[-1]
    for i in range(128, len(hist)):
        threshold = edges[i]
        # 量化到 n_bins 个 bin，计算 KL
        reference = hist[:i].copy()
        reference[-1] += hist[i:].sum()  # 超出范围的合并到最后
        # ... 简化版: 实际需要更精细的实现
        quantized = np.interp(
            np.linspace(0, i, n_bins),
            np.arange(i),
            reference
        )
        kl = np.sum(reference * np.log(reference / (quantized + 1e-10) + 1e-10))
        if kl < best_kl:
            best_kl = kl
            best_threshold = threshold
    return best_threshold
```

### 6.1.3 Bias Correction

量化后的权重可能引入系统性偏差（均值偏移）。纠正方法：

$$W_{corrected} = W_q + (E[W] - E[W_q])$$

其中 $E[\cdot]$ 是对校正数据集的期望。**ZeroQ** 论文提出用 generated data（不需要真实数据）做 bias correction。

---

## 6.2 Quantization-Aware Training (QAT)

QAT = 在训练过程中模拟量化，让模型适应量化误差。精度比 PTQ 好，但需要训练数据和算力。

### 6.2.1 Fake Quantization

在训练图中插入 `quantize → dequantize` 节点：

```
FP32 weight → [Quantize] → INT8 → [Dequantize] → FP32 → 前向传播
                                                                    ↓
                                                              反向传播(梯度)
```

模型"看到"的是量化后的权重，从而学会容忍量化误差。

### 6.2.2 Straight-Through Estimator (STE)

**核心问题**: `round()` 不可导（梯度几乎处处为 0），怎么反向传播？

**STE 的回答**: forward 做 round，backward 假装 round 不存在。

$$\frac{\partial L}{\partial r} \approx \frac{\partial L}{\partial \hat{r}}$$

```python
import torch

class StraightThroughEstimator(torch.autograd.Function):
    """STE: forward 量化, backward 梯度直通"""
    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        q = torch.round(x / scale + zero_point).clamp(qmin, qmax)
        return (q - zero_point) * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None

# 使用
x = torch.randn(10, requires_grad=True)
scale = torch.tensor(0.1)
zp = torch.tensor(0.0)
x_q = StraightThroughEstimator.apply(x, scale, zp, -128, 127)
loss = x_q.sum()
loss.backward()
print(f"梯度 = STE 直通: {x.grad}")  # 梯度约等于 1
```

> **STE 为什么 work？** 本质上是告诉优化器："你就当量化不存在，继续按 FP32 的梯度更新"。经过多轮迭代，FP32 权重会自然收敛到量化友好的区域。

### 6.2.3 QAT vs PTQ 对比

| | PTQ | QAT |
|---|-----|-----|
| 需要训练？ | 不需要（只需少量校准数据） | 需要 |
| 精度 | 中等（INT8 可接受，INT4 崩） | 高（INT4 也能保精度） |
| 成本 | 几分钟 | 几小时到几天 |
| 适用场景 | 快速部署、大模型 | 精度敏感、小模型 |

---

## 6.3 低比特量化 (Low-bit Quantization)

### 6.3.1 Binary Quantization (1-bit)

权重只有 +1 和 -1：

$$w_b = \text{sign}(w) = \begin{cases} +1 & w \geq 0 \\ -1 & w < 0 \end{cases}$$

**XNOR-Net**: 用 XNOR 门替代浮点乘法
- 乘法变成 XNOR → popcount（数 1 的个数）
- 速度提升 ~58x，但精度下降严重

**Deterministic vs Stochastic**:
- Deterministic: 直接 sign()
- Stochastic: 以概率做随机二值化

### 6.3.2 Ternary Quantization (2-bit)

权重取 {-1, 0, +1}:

$$w_t = \begin{cases} +1 & w > \Delta \\ 0 & |w| \leq \Delta \\ -1 & w < -\Delta \end{cases}$$

比 binary 多一个"0"状态，精度提升明显，但计算优化不如 binary 极端。

### 6.3.3 混合精度量化 (Mixed Precision)

不同层用不同 bit-width：

```python
# 混合精度策略示例
mixed_precision_config = {
    "layer0": 8,   # 第一层敏感，用 INT8
    "layer1": 4,   # 中间层不敏感，用 INT4
    "layer2": 4,
    "layer3": 8,   # 最后一层敏感，用 INT8
}
```

**HAWQ/HAWQ-V3**: 用 Hessian 谱（特征值）衡量每层的敏感度
- Hessian 特征值大 → 层对扰动敏感 → 用高精度
- Hessian 特征值小 → 层鲁棒 → 可以低精度

---

## 代码示例: 完整 QAT 训练循环

```python
import torch
import torch.nn as nn

class QATLinear(nn.Module):
    def __init__(self, in_f, out_f, n_bits=8):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.n_bits = n_bits
        self.scale = None

    def _fake_quant(self, x):
        """Fake quantize with STE"""
        qmax = 2 ** (self.n_bits - 1) - 1
        qmin = -(qmax + 1)
        if self.scale is None:
            self.scale = x.abs().max() / qmax
        q = torch.round(x / self.scale).clamp(qmin, qmax)
        return q * self.scale  # STE 梯度自动直通

    def forward(self, x):
        w_q = self._fake_quant(self.linear.weight)
        return nn.functional.linear(x, w_q, self.linear.bias)

    def prepare_for_export(self):
        """训练完成后，真正量化权重"""
        qmax = 2 ** (self.n_bits - 1) - 1
        qmin = -(qmax + 1)
        self.scale = self.linear.weight.data.abs().max() / qmax
        q = torch.round(self.linear.weight.data / self.scale).clamp(qmin, qmax)
        self.linear.weight.data = q * self.scale

# 训练
model = nn.Sequential(
    QATLinear(784, 256, n_bits=8),
    nn.ReLU(),
    QATLinear(256, 10, n_bits=8),
)
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(10):
    x = torch.randn(32, 784)
    y = torch.randint(0, 10, (32,))
    loss = nn.CrossEntropyLoss()(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 3 == 0:
        # 重新校准 scale
        for m in model.modules():
            if isinstance(m, QATLinear):
                m.scale = None
print("QAT 训练完成")
```

---

## Infra 实战映射

### vLLM
- 不做 QAT，只支持加载 PTQ 量化后的模型
- 支持 AWQ (W4)、GPTQ (W4/W8)、FP8 量化格式
- 量化在模型层面完成，推理时直接加载

### TensorRT-LLM (NVIDIA)
- 支持 INT8 PTQ + QAT
- H100 原生 FP8 (E4M3) 支持
- 自动混合精度: 敏感层保持 FP16，其余 INT8

### 沐曦 MACA
- 需要确认硬件支持哪些低比特运算
- INT4 如果没有硬件支持，可以拆成两个 INT8 运算
- QAT 需要在 MACA 软件栈上实现 STE 算子

---

## 跨 Lecture 关联

- **前置 ←** [Lec05: 量化基础](../lec05-quantization-I/README.md) — 量化公式、数值类型
- **延伸 →** [Lec13: LLM 部署](../lec13-llm-deploy/README.md) — LLM 专用量化 (SmoothQuant/AWQ)
- **延伸 →** [Lec14: LLM 后训练](../lec14-llm-post-training/README.md) — QLoRA = 量化 + LoRA
- **横向 ↔** [Lec09: 蒸馏](../lec09-distillation/README.md) — 量化后蒸馏恢复精度

---

## 面试高频题

**Q1: STE 是什么？为什么需要它？**
> A: Straight-Through Estimator。round() 函数梯度几乎处处为 0，无法反向传播。STE 在 forward 时做 round（真实量化），backward 时用 identity（梯度 = 1 直通）。本质是"欺骗"优化器继续正常更新。

**Q2: PTQ 和 QAT 怎么选？**
> A: 快速部署 / 大模型用 PTQ（几分钟搞定）。精度敏感 / 小模型 / 极低比特(4bit以下)用 QAT。实际工业中常见: 先 PTQ 试试，精度不够再 QAT。

**Q3: Calibration 的 KL-divergence 方法原理？**
> A: 用不同阈值截断 activation 范围，把截断后的 FP32 分布和量化后的 INT8 分布计算 KL 散度，选 KL 最小的阈值。本质是找"信息损失最少"的量化参数。TensorRT 默认用这个。

**Q4: Binary/Ternary 量化实际有用吗？**
> A: 学术价值大（理论极限），工业落地少。精度损失太大。目前主流还是 INT8 (PTQ) 和 INT4 (AWQ/GPTQ)。1-bit/2-bit 在极端边缘设备上可能有应用。

**Q5: 混合精度量化的关键问题是什么？**
> A: 怎么决定每层用什么 bit-width。太激进会崩精度，太保守浪费压缩率。HAWQ 用 Hessian 谱衡量层敏感度，自动化这个决策。
