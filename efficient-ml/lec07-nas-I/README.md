# Lec07 神经架构搜索 I — 基础与 DARTS

> 📺 [课程视频](https://hanlab.mit.edu/courses/2023-fall-65940) | 📄 [Slides](https://hanlab.mit.edu/courses/2023-fall-65940)

---

## 核心概念

### 7.1 手工设计网络的演进轨迹

在 NAS 出现之前，研究人员靠直觉和反复实验手工设计网络。回顾这条线索，能看出每一代设计解决了什么瓶颈，也能理解 NAS 究竟在自动化什么。

**AlexNet (2012)** — 暴力堆砌深度 + ReLU + Dropout，首次证明深度 CNN 在大规模图像识别上可行。5 个卷积层 + 3 个全连接，参数量 ~60M。问题：太宽、太深、太重。

**VGGNet (2014)** — 系统化探索深度的影响，全部使用 3×3 conv，以小 kernel 叠加模拟大感受野。规律：3×3 conv × 2 ≈ 5×5 conv，参数量更少，非线性更丰富。VGG-16 参数量 ~138M，仍然过重。

**SqueezeNet (2016)** — Fire Module 是核心创新：

```
Fire Module:
  Squeeze: 1×1 conv (压缩 channel 数 → s₁ₓ₁)
  Expand:  1×1 conv (e₁ₓ₁) + 3×3 conv (e₃ₓ₃) 并行 → concat
```

设计哲学：先用 1×1 conv 压缩，再用混合 kernel 展开。AlexNet 精度相当，参数量缩小 50×。

**ResNet (2015)** — 残差连接解决梯度消失，允许训练 100+ 层网络：

$$y = \mathcal{F}(x, \{W_i\}) + x$$

关键洞察：学习残差比学习原始映射更容易。当残差为 0 时退化为恒等映射，保证"加深不变差"。

**MobileNet (2017)** — Depthwise-Separable Convolution，将标准卷积分解为两步：

```
标准卷积: (H×W×Cin) → (H×W×Cout)  计算量: H·W·Cin·Cout·K·K
Depthwise: (H×W×Cin) → (H×W×Cin)  每个 channel 独立卷积  计算量: H·W·Cin·K·K
Pointwise: (H×W×Cin) → (H×W×Cout) 1×1 conv               计算量: H·W·Cin·Cout
```

计算量比值：

$$\frac{D_W + P_W}{\text{Standard}} = \frac{1}{C_{out}} + \frac{1}{K^2}$$

对 K=3, Cout=256，压缩比约 8-9×。

MobileNet 还引入两个超参数：
- **宽度乘子 α**：按比例缩减每层 channel 数，Cin → αCin
- **分辨率乘子 ρ**：缩减输入图像尺寸

**MobileNetV2 (2018)** — Inverted Bottleneck Block：

```
标准 Bottleneck（ResNet风格）: wide → narrow → wide  (先压缩再还原)
Inverted Bottleneck:          narrow → wide → narrow (先扩张再压缩)

结构:
  1×1 conv (expand, ×t)     # t 通常为 6
  3×3 depthwise conv
  1×1 conv (project)
  + shortcut (仅当 stride=1 且 in_channels == out_channels)
```

为什么 inverted？DW conv 在高维空间里表达能力更强；低维的 linear bottleneck 避免信息损失（ReLU 在低维会破坏流形）。

**ShuffleNet (2018)** — Pointwise conv 是 MobileNet 的计算瓶颈。ShuffleNet 用 group conv 替换 1×1 conv，再通过 channel shuffle 打破组间信息隔离：

```
Group Conv → Channel Shuffle → Group Depthwise Conv → Group Conv
```

Channel Shuffle 操作：将 g 组、每组 n 个 channel 的 tensor reshape 为 (n, g) 再转置为 (g, n)。

**SENet (2018)** — Squeeze-and-Excitation：用全局 average pooling + 两层 FC 生成 per-channel 注意力权重：

$$\hat{x}_c = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot \text{GAP}(x))) \cdot x_c$$

这些手工设计的共同主题：**在 accuracy-efficiency tradeoff 上做人工搜索**，本质上是在离散空间里的人工优化。NAS 要做的，就是把这个过程自动化。

---

### 7.2 NAS 三要素

NAS 问题的形式化定义：

$$\alpha^* = \arg\max_{\alpha \in \mathcal{A}} \text{acc}(\mathcal{N}(\alpha), \mathcal{D}_{val})$$

其中 $\alpha$ 是架构描述，$\mathcal{A}$ 是搜索空间，$\mathcal{N}(\alpha)$ 是对应网络，在验证集上评估精度。

#### 搜索空间 (Search Space)

**Macro Search Space** — 搜索整体骨架：每层用什么操作、跳跃连接怎么连。自由度最高，但搜索难度极大。

**Chain-Structured** — 网络是线性链，搜索每个 stage 的宽度、深度、kernel size。Early NAS 常用。

**Cell-based（NASNet 风格）** — 将网络分解为可复用的 cell，搜索 cell 的内部结构。两种 cell：
- **Normal Cell**：stride=1，维持分辨率
- **Reduction Cell**：stride=2，降低分辨率

Cell 搜索到之后，堆叠固定数量形成完整网络。搜索空间大幅缩小，且搜到的 cell 有更好的迁移性。

**Hierarchical** — Auto-DeepLab 风格，在多个尺度上同时搜索：cell 内部结构 + cell 间的路径（高分辨率/低分辨率流的选择）。

#### 搜索策略 (Search Strategy)

| 策略 | 原理 | 优缺点 |
|------|------|--------|
| **Random Search** | 随机采样架构 | 简单强基准，NAS 需要明显超越 |
| **Evolutionary** | 变异 + 选择，维持架构种群 | 无梯度，灵活，但需大量评估 |
| **RL** | 用 RNN controller 生成架构，以 val acc 作为 reward | NASNet 用此，搜索耗费 ~450 GPU days |
| **Differentiable (DARTS)** | 连续化搜索空间，用梯度优化 | 快（~4 GPU days），但有收敛问题 |

#### 性能评估 (Performance Estimation)

**从头训练** — 最准确，也最慢。搜索 N 个架构就需要训练 N 次。

**Early Stopping** — 只训练少量 epoch 看排名是否稳定。问题：early rank ≠ final rank。

**Weight Sharing / Supernet** — 所有子架构共享权重，一次训练，免重训。Lec08 详细讲。

**Zero-cost Proxy** — 不训练就能预测性能：
- **NASWOT**：随机初始化网络，看 batch 内样本的 kernel matrix 是否满秩（满秩 → 激活多样 → 好架构）
- **GradNorm / Synflow**：前向+后向一次，统计梯度信号强度
- **FLOPs 分布分析**：好的搜索空间应该在 FLOPs 目标附近有密集分布，避免大量无效架构

---

### 7.3 DARTS — 可微分架构搜索

**核心思想**：将离散的"选哪个 op"变成连续的"各 op 加权求和"，从而可以用反向传播优化架构参数。

#### 搜索空间参数化

在 cell 内，每条边 $(i, j)$ 上有 $|\mathcal{O}|$ 个候选操作（如 3×3 conv, 5×5 conv, max pool, skip, none）。

混合操作（Soft 选择）：

$$\bar{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o' \in \mathcal{O}} \exp(\alpha_{o'}^{(i,j)})} \cdot o(x)$$

其中 $\alpha_o^{(i,j)}$ 是架构参数，$\text{softmax}$ 保证权重和为 1，可解释为"选择概率"。

搜索结束后，**离散化**：对每条边选权重最大的操作：

$$o^{*(i,j)} = \arg\max_{o \in \mathcal{O}} \alpha_o^{(i,j)}$$

#### Bi-level Optimization

DARTS 的优化目标是双层优化问题：

$$\min_\alpha \mathcal{L}_{val}(w^*(\alpha), \alpha)$$

$$\text{s.t.} \quad w^*(\alpha) = \arg\min_w \mathcal{L}_{train}(w, \alpha)$$

外层：最小化验证集损失（优化架构参数 $\alpha$）
内层：最小化训练集损失（优化权重 $w$）

两者**交替优化**（近似求解）：
1. 固定 $\alpha$，在训练集上用 SGD 更新 $w$ 一步
2. 固定 $w$，在验证集上用 Adam 更新 $\alpha$ 一步

内层近似（一阶 DARTS）：

$$\nabla_\alpha \mathcal{L}_{val}(w^*(\alpha), \alpha) \approx \nabla_\alpha \mathcal{L}_{val}(w - \xi \nabla_w \mathcal{L}_{train}(w, \alpha), \alpha)$$

二阶近似展开（二阶 DARTS，更精确但更慢）：

$$\approx \nabla_\alpha \mathcal{L}_{val}(w, \alpha) - \xi \nabla^2_{\alpha, w} \mathcal{L}_{train}(w, \alpha) \cdot \nabla_w \mathcal{L}_{val}(w, \alpha)$$

#### DARTS 的已知问题

1. **性能坍塌（Performance Collapse）**：搜索后期 `skip connection` 的权重会异常高，因为 skip 不修改特征，梯度更容易流过，导致 $\alpha_{skip}$ 虚高，离散化后选了大量 skip，网络退化。

2. **train/val 分布不一致**：架构参数在验证集上优化，权重在训练集上优化，可能过拟合到验证集的架构偏好。

3. **内存问题**：同时保存所有操作的中间激活，搜索期间显存需求是单操作的 $|\mathcal{O}|$ 倍。

---

## 数学推导

### MobileNet 计算量推导

设输入特征图 $\mathbf{X} \in \mathbb{R}^{H \times W \times C_{in}}$，卷积核大小 $K \times K$，输出 channel 数 $C_{out}$。

**标准卷积 MACs**（Multiply-Accumulate Operations）：

$$\text{MACs}_{std} = H \cdot W \cdot K \cdot K \cdot C_{in} \cdot C_{out}$$

**Depthwise Separable Convolution MACs**：

$$\text{MACs}_{DW} = H \cdot W \cdot K \cdot K \cdot C_{in} \qquad \text{(depthwise)}$$

$$\text{MACs}_{PW} = H \cdot W \cdot C_{in} \cdot C_{out} \qquad \text{(pointwise)}$$

$$\text{MACs}_{DS} = H \cdot W \cdot C_{in}(K^2 + C_{out})$$

**压缩比**：

$$\frac{\text{MACs}_{DS}}{\text{MACs}_{std}} = \frac{H \cdot W \cdot C_{in}(K^2 + C_{out})}{H \cdot W \cdot K^2 \cdot C_{in} \cdot C_{out}} = \frac{1}{C_{out}} + \frac{1}{K^2}$$

对 $K=3$：$\frac{1}{C_{out}} + \frac{1}{9}$，当 $C_{out} \gg 1$ 时约为 $\frac{1}{9} \approx 11\%$，即节省 ~8.9×。

### DARTS 梯度推导

设 $w' = w - \xi \nabla_w \mathcal{L}_{train}(w, \alpha)$，二阶展开：

$$\nabla_\alpha \mathcal{L}_{val}(w', \alpha) = \nabla_\alpha \mathcal{L}_{val}(w', \alpha)$$

用链式法则：

$$= \nabla_\alpha \mathcal{L}_{val} \Big|_{w'} - \xi \nabla^2_{\alpha, w} \mathcal{L}_{train}(w, \alpha) \cdot \nabla_{w'} \mathcal{L}_{val}(w', \alpha)$$

混合偏导 $\nabla^2_{\alpha, w}$ 用有限差分近似（避免直接计算 Hessian）：

$$\nabla^2_{\alpha, w} \mathcal{L}_{train} \cdot v \approx \frac{\nabla_\alpha \mathcal{L}_{train}(w^+, \alpha) - \nabla_\alpha \mathcal{L}_{train}(w^-, \alpha)}{2\epsilon}$$

其中 $w^\pm = w \pm \epsilon v$，$v = \nabla_{w'} \mathcal{L}_{val}(w', \alpha)$。

---

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ #
# 7.1 Depthwise-Separable Conv（MobileNet 核心构件）
# ------------------------------------------------------------------ #
class DepthwiseSeparableConv(nn.Module):
    """
    将标准 K×K conv 分解为:
      depthwise: groups=in_channels 的 K×K conv
      pointwise: 1×1 conv
    """
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel,
            stride=stride, padding=kernel // 2,
            groups=in_ch,   # 每个 channel 独立卷积
            bias=False
        )
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu6(self.bn1(self.dw(x)))
        x = F.relu6(self.bn2(self.pw(x)))
        return x


# ------------------------------------------------------------------ #
# 7.2 MobileNetV2 Inverted Bottleneck Block
# ------------------------------------------------------------------ #
class InvertedBottleneck(nn.Module):
    """
    expand_ratio: 扩张倍数 t（论文中 t=6）
    stride=2 时无 shortcut（分辨率变化）
    """
    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=6):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            # Pointwise expand
            layers += [nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                       nn.BatchNorm2d(mid_ch), nn.ReLU6(inplace=True)]
        layers += [
            # Depthwise
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride,
                      padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU6(inplace=True),
            # Pointwise project（无激活函数！线性 bottleneck）
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)  # 残差
        return self.conv(x)


# ------------------------------------------------------------------ #
# 7.3 简化版 DARTS 混合操作
# ------------------------------------------------------------------ #
OPS = {
    'skip':    lambda C: nn.Identity(),
    'conv3x3': lambda C: nn.Sequential(
        nn.Conv2d(C, C, 3, padding=1, bias=False), nn.BatchNorm2d(C), nn.ReLU()),
    'conv5x5': lambda C: nn.Sequential(
        nn.Conv2d(C, C, 5, padding=2, bias=False), nn.BatchNorm2d(C), nn.ReLU()),
    'maxpool': lambda C: nn.MaxPool2d(3, stride=1, padding=1),
}


class MixedOp(nn.Module):
    """
    DARTS 的核心：对每条边，定义所有候选操作的加权混合。
    前向 = Σ softmax(α_k) * op_k(x)
    """
    def __init__(self, C):
        super().__init__()
        self.ops = nn.ModuleList([op(C) for op in OPS.values()])
        # 架构参数：每个操作一个 logit，初始化为 0（均匀分布）
        self.alpha = nn.Parameter(torch.zeros(len(OPS)))

    def forward(self, x):
        weights = F.softmax(self.alpha, dim=0)  # shape: [num_ops]
        # 加权求和所有候选操作的输出
        return sum(w * op(x) for w, op in zip(weights, self.ops))

    def discretize(self):
        """搜索结束后，取权重最大的操作"""
        best_idx = self.alpha.argmax().item()
        op_name = list(OPS.keys())[best_idx]
        return op_name, self.alpha.softmax(0)[best_idx].item()


class DARTSCell(nn.Module):
    """
    简化的 DARTS cell：3 个节点，每对节点之间一条边（Mixed Op）。
    真实 DARTS 有 4 个中间节点，边更多。
    """
    def __init__(self, C):
        super().__init__()
        # 边 0→1, 0→2, 1→2
        self.edge_01 = MixedOp(C)
        self.edge_02 = MixedOp(C)
        self.edge_12 = MixedOp(C)

    def forward(self, x):
        n0 = x
        n1 = self.edge_01(n0)
        n2 = self.edge_02(n0) + self.edge_12(n1)  # 节点聚合：求和
        return n2


# ------------------------------------------------------------------ #
# 示例：DARTS 搜索一步（bi-level 简化）
# ------------------------------------------------------------------ #
def darts_search_step(cell, arch_optimizer, weight_optimizer,
                      train_x, val_x, criterion):
    """
    一次 DARTS 迭代：
      1. 用 train_x 更新网络权重 w
      2. 用 val_x 更新架构参数 alpha
    """
    # Step 1: 更新权重（冻结 alpha）
    weight_optimizer.zero_grad()
    loss_train = criterion(cell(train_x), torch.zeros_like(cell(train_x)))
    loss_train.backward()
    weight_optimizer.step()

    # Step 2: 更新 alpha（冻结 w）
    arch_optimizer.zero_grad()
    loss_val = criterion(cell(val_x), torch.zeros_like(cell(val_x)))
    loss_val.backward()
    arch_optimizer.step()

    return loss_train.item(), loss_val.item()


if __name__ == '__main__':
    C = 16
    cell = DARTSCell(C)

    # 分离架构参数和权重参数
    arch_params = [cell.edge_01.alpha, cell.edge_02.alpha, cell.edge_12.alpha]
    weight_params = [p for n, p in cell.named_parameters() if 'alpha' not in n]

    arch_opt = torch.optim.Adam(arch_params, lr=3e-4)
    weight_opt = torch.optim.SGD(weight_params, lr=0.025, momentum=0.9)

    x = torch.randn(4, C, 8, 8)
    for step in range(5):
        lt, lv = darts_search_step(
            cell, arch_opt, weight_opt, x, x,
            nn.MSELoss()
        )
        print(f"Step {step}: train_loss={lt:.4f}, val_loss={lv:.4f}")

    # 查看搜索结果
    for name, edge in [('0→1', cell.edge_01), ('0→2', cell.edge_02), ('1→2', cell.edge_12)]:
        op, weight = edge.discretize()
        print(f"  Edge {name}: best_op={op} (weight={weight:.3f})")
```

---

## Infra 实战映射

- **vLLM**: vLLM 的 KV Cache 管理借鉴了 NAS 中对内存约束的显式建模思路——PagedAttention 本质上是在运行时做动态内存规划，类似 TinyEngine 的编译时内存规划，只是时机不同。LLM 服务的 continuous batching 也类似 NAS 的 performance estimation：在不重跑推理的情况下预测 batch 完成时间。

- **TensorRT-LLM**: NVIDIA 的 AutoDeploy 工具对 LLM 做 op-level 搜索（kernel fusion 组合、precision 选择），是 NAS 在推理优化侧的工程化应用。TRT 的 layer fusion 决策可看作一个小型 search problem：在延迟约束下搜最优 kernel 组合。

- **沐曦 MACA**: 国产 GPU 的 roofline model 与 NVIDIA 不同（内存带宽/算力比不同），手工设计的网络（针对 A100 调优的 MobileNet 变体）搬到沐曦硬件上可能不是最优。NAS + 硬件 latency lookup table（针对 MACA 后端实测）能找到真正适配的架构。这是 ProxylessNAS 思路的直接应用（见 Lec08）。

---

## 跨 Lecture 关联

- 前置知识 ← Lec02（基础卷积计算量分析）、Lec03-04（剪枝理解参数冗余）、Lec05-06（量化了解精度-效率 tradeoff）
- 后续延伸 → Lec08（高效 NAS：weight sharing、ProxylessNAS）→ Lec10（MCUNet：NAS + 硬件约束的极致应用）→ Lec12（Transformer 的 NAS 变体：AutoFormer、NAS-BERT）

---

## 面试高频题

**Q: DARTS 为什么会出现性能坍塌？如何缓解？**

A: Skip connection 在反向传播中梯度损失最小，导致架构参数 $\alpha_{skip}$ 持续增大，离散化后网络被 skip 主导，表达能力丧失。缓解方法：(1) DARTS+ 在离散化时限制每条边最多选 1 个 skip；(2) PC-DARTS 对 channel 随机采样，降低 skip 的梯度优势；(3) 用 Gumbel-Softmax 替代连续 softmax，增加离散性。

**Q: Cell-based search space 相比 macro search space 有什么优势？**

A: (1) 搜索空间更小，从 $\mathcal{O}(K^N)$ 降为 $\mathcal{O}(K^{n_{cell}})$，$n_{cell} \ll N$；(2) 搜到的 cell 可以在不同规模网络间迁移（如从 CIFAR 搜索迁移到 ImageNet 部署）；(3) Normal/Reduction cell 的分工与 CNN 的 stride 设计天然对齐。

**Q: NAS 的 performance estimation 有哪些方法？各有什么代价？**

A: (1) 从头训练：最准，代价 O(N × full training)；(2) Early stopping：快，但 rank correlation 弱；(3) Weight sharing：训练一次 supernet，评估代价低，但 co-adaptation 导致子网精度估计偏差；(4) Zero-cost proxy（NASWOT、GradNorm）：无训练代价，但与最终精度的相关性在不同搜索空间差异大。

**Q: MobileNetV2 的 linear bottleneck 为什么不加 ReLU？**

A: 理论依据是流形假设（Manifold Hypothesis）：低维 bottleneck 中的激活流形在高维时可以用 ReLU 不损失信息，但 bottleneck 很窄时 ReLU 会把负值截断，造成信息坍塌（information collapse）。实验也验证：去掉 projection layer 后的 ReLU，精度提升明显。

**Q: Depthwise-separable conv 的计算量压缩比是多少？有没有缺点？**

A: 压缩比为 $\frac{1}{C_{out}} + \frac{1}{K^2}$，对 K=3 约为 8-9×。缺点：(1) 硬件利用率低——DW conv 的 MAC 密度低，在 GPU 上实际加速比远小于理论值（受内存带宽限制）；(2) 表达能力弱于标准 conv（跨 channel 交互只在 PW 发生）；(3) 在现代 GPU 上 1×1 conv（PW）反而可能是瓶颈（小 batch 下 cublas gemm 效率低）。
