# 08 高效 NAS (NAS Part II)

> 📺 [Lecture 08 - NAS (Part II)](https://hanlab.mit.edu/courses/2024-fall-65940)
> 📄 [Slides](https://hanlab.mit.edu/courses/2024-fall-65940)

---

## 8.1 Weight Sharing

### 8.1.1 问题：NAS 计算爆炸

如果有 N 个候选 operation × L 层，需要评估 $N^L$ 个架构。每个从头训练太慢。

### 8.1.2 方案：所有子架构共享一组权重

训练一个 **supernet**（超网络），包含所有可能的子架构。每个子架构只是 supernet 的一个子集，**不需要单独训练**。

```
Supernet (训练一次):
[Conv3x3, Conv5x5, DepthConv, MaxPool, Skip] × L层

子网络 A (直接从supernet取权重):
[Conv3x3, DepthConv, Skip, Conv5x5, ...] × L层

子网络 B (直接从supernet取权重):
[Conv5x5, Conv3x3, Conv3x3, MaxPool, ...] × L层
```

**效果**: 训练成本从 $O(N^L \times T)$ 降到 $O(T)$，T 是单个网络训练时间。

---

## 8.2 One-Shot NAS

### 8.2.1 核心流程

```
1. 定义搜索空间 → 构建 supernet
2. 训练 supernet（一次）
3. 搜索最优子网络（直接评估，不重训练）
```

**训练技巧**:
- **Path Dropout**: 随机丢弃某些 path，防止 co-adaptation
- **Sandwich Rule**: 每个 batch 同时训练最大、最小和随机子网络
- **Inplace Distillation**: 大子网络的输出作为小子网络的 soft label

### 8.2.2 Once-for-All (OFA)

MIT Song Han 团队的工作：
- 训练**一个**大网络，从中提取**多种**大小的子网络
- 支持: 不同宽度(elastic depth)、不同深度(elastic width)、不同分辨率(elastic resolution)
- 各个子网络共享权重，不重新训练
- 应用: 在不同设备上部署不同大小的模型

```python
# OFA 概念示意
class SupernetBlock(nn.Module):
    def __init__(self, in_ch, out_ch_list, kernel_list):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, max(out_ch_list), k, padding=k//2)
            for k in kernel_list
        ])
        self.out_ch_list = out_ch_list  # [32, 48, 64]

    def forward(self, x, kernel_idx=0, out_ch_idx=1):
        # 动态选择子网络
        conv = self.convs[kernel_idx]
        out = conv(x)
        out_ch = self.out_ch_list[out_ch_idx]
        return out[:, :out_ch]  # 截取前 out_ch 个通道
```

---

## 8.3 ProxylessNAS

### 8.3.1 问题：Proxy Gap

大部分 NAS 在 **proxy**（代理设置）上搜索：
- 用 CIFAR-10 搜索，再迁移到 ImageNet
- 用小模型搜索，再 scale 到大模型
- Proxy 和真实目标之间的 gap 可能导致次优结果

### 8.3.2 直接在目标上搜索

ProxylessNAS:
- 直接在**目标硬件**和**目标数据集**上搜索
- 把硬件 latency 建模为可微分函数
- 搜索空间: MBConv 的 expansion ratio, kernel size, 等

### 8.3.3 可微分 Latency 延迟建模

$$L_{total} = L_{acc} + \lambda \times \text{latency}(\alpha)$$

其中 latency(α) 是架构参数 α 的可微分近似：

$$\text{latency}(\alpha) = \sum_i \text{softmax}(\alpha_i)^T \cdot \text{latency\_lookup\_table}[i]$$

- 预先测量每种 operation 在目标硬件上的 latency
- 用 softmax 加权求和得到可微分的 latency 估计
- λ 控制 accuracy-latency trade-off

```python
# 可微分 latency 计算
def differentiable_latency(alpha, latency_table):
    """
    alpha: [n_ops] 架构参数
    latency_table: [n_ops] 每种 op 的实测 latency
    """
    weights = torch.softmax(alpha, dim=0)
    return torch.dot(weights, latency_table)
```

---

## 8.4 性能评估策略

| 方法 | 原理 | 成本 | 精度 |
|------|------|------|------|
| 从头训练 | 直接训练候选架构 | 极高 | 准确 |
| Weight Inheritance | 继承 supernet 权重评估 | 低 | 一般 |
| HyperNetwork | 用小网络预测大网络性能 | 中 | 较好 |
| Zero-cost Proxy | 不训练直接评估 | 极低 | 粗糙 |
| Zen-NAS | 用 training dynamics 评估 | 低 | 较好 |

**Zero-cost Proxy 示例**:
- NASWOT: 用 Jacobian 的奇异值评估
- GradSign: 用梯度的符号一致性
- Snip: 连接敏感度 (Lec03 提到的)

---

## 8.5 从 NAS 到 LLM

NAS 的思想在 LLM 时代也在用：
- **MoE (Mixture of Experts)**: 每个 token 动态选择专家 → 类似于搜索最优路径
- **Router 设计**: MoE 的路由器本质上是一个轻量 NAS
- **模型缩放**: NAS 的 width/depth 搜索 → LLM 的 scaling law 指导
- **AutoML for LLM**: 自动搜索 attention pattern, FFN ratio 等

---

## 代码示例: 简单 Supernet

```python
import torch
import torch.nn as nn

class SupernetLayer(nn.Module):
    """单层 supernet: 包含多个候选 operation"""
    def __init__(self, channels):
        super().__init__()
        self.ops = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1),   # Conv3x3
            nn.Conv2d(channels, channels, 5, padding=2),   # Conv5x5
            nn.Conv2d(channels, channels, 1),               # Conv1x1
            nn.Identity(),                                    # Skip
        ])
        self.alpha = nn.Parameter(torch.zeros(len(self.ops)))

    def forward(self, x, hard=False):
        if hard:
            # 推理: 选最优 op
            idx = self.alpha.argmax()
            return self.ops[idx](x)
        else:
            # 训练: 加权混合 (DARTS)
            weights = torch.softmax(self.alpha, dim=0)
            return sum(w * op(x) for w, op in zip(weights, self.ops))

class SimpleSupernet(nn.Module):
    def __init__(self, channels=64, n_layers=4):
        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.layers = nn.ModuleList([
            SupernetLayer(channels) for _ in range(n_layers)
        ])

    def forward(self, x, hard=False):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x, hard=hard)
        return x

# 训练 supernet
model = SimpleSupernet()
x = torch.randn(2, 3, 32, 32)
out = model(x, hard=False)
print(f"Supernet 输出 shape: {out.shape}")
# 提取最优子网络
out_sub = model(x, hard=True)
print(f"子网络输出 shape: {out_sub.shape}")
```

---

## Infra 实战映射

### vLLM / LLM 推理
- NAS 在 LLM 推理优化中不直接使用
- 但 MoE 的 router 设计借鉴了 NAS 的思路
- vLLM 对 MoE 模型有专门优化（expert parallelism）

### TensorRT-LLM
- NVIDIA 提供 NAS-like 工具自动优化 tensor core 配置
- 自动选择最优 GEMM kernel（也是一种搜索）

### 沐曦 MACA
- 可以用 ProxylessNAS 思路：在 MACA 硬件上实测 latency，指导模型架构选择
- 不同硬件的最优架构可能不同 → hardware-aware 是关键

---

## 跨 Lecture 关联

- **前置 ←** [Lec07: NAS 基础](../lec07-nas-I/README.md) — 搜索空间、DARTS
- **延伸 →** [Lec10: MCUNet](../lec10-mcunet/README.md) — TinyNAS 就是 hardware-aware NAS
- **横向 ↔** [Lec03: 剪枝](../lec03-pruning-I/README.md) — Once-for-All + 剪枝 = 自动模型压缩

---

## 面试高频题

**Q1: Weight Sharing 的问题是什么？**
> A: 子网络之间会 co-adapt（互相适应），导致单个子网络的权重不是最优的。缓解方法: Path Dropout、Sandwich Rule。

**Q2: ProxylessNAS 为什么比传统 NAS 好？**
> A: 传统 NAS 在 proxy（CIFAR-10/小模型）上搜索再迁移，proxy gap 导致次优。ProxylessNAS 直接在目标硬件和数据集上搜索，消除了 proxy gap。关键创新是把 latency 建模为可微分函数纳入搜索目标。

**Q3: Once-for-All 的实际价值？**
> A: 训练一次，部署到多种设备。移动端用小子网络，服务器用大子网络，权重共享。省去了每个设备单独训练/搜索的成本。
