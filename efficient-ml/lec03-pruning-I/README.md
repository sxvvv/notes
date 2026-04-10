# Lec03 神经网络剪枝基础 (Pruning I)

> 📺 [课程视频](https://www.youtube.com/watch?v=sZzc6tAtTrM) | 📄 [Slides](https://hanlab.mit.edu/courses/2023-fall-65940)

---

## 核心概念

### 3.1 剪枝动机与直觉

神经网络在训练时往往是**过参数化**的 (over-parameterized)。直觉上，我们需要大量参数来让优化过程顺利收敛（loss landscape 更平滑，更容易找到好的极小值），但推理时不需要那么多参数就能保持精度。

几个关键观察：

1. **权重分布往往接近零**：训练好的网络里，权重的分布通常是以 0 为中心的钟形分布。绝对值很小的权重对输出影响有限。
2. **激活稀疏性**：ReLU 激活函数会直接把负值置零。统计表明 AlexNet 在 ImageNet 上的激活稀疏度高达 62%。
3. **冗余特征**：不同 filter 可能学到相似的特征，存在大量冗余。

**剪枝 vs 量化的本质区别**：
- **量化**：改变权重的数值精度（float32 → int8），结构不变，参数数量不变
- **剪枝**：改变网络结构，直接删除权重/神经元/层，参数数量减少

两者可以组合使用（先剪枝再量化，或同时做）。

### 3.2 剪枝粒度 (Pruning Granularity)

剪枝粒度是一个谱系，从最细到最粗：

```
weight → vector → kernel → channel → filter → layer → block
  细 ←————————————————————————————————————————————→ 粗
  灵活 ←—————————————————————————————————————→ 硬件友好
```

#### 非结构化剪枝 (Unstructured Pruning)

最细粒度，逐个权重决定是否删除。对于权重矩阵 $W \in \mathbb{R}^{m \times n}$，产生一个二值掩码 $M \in \{0, 1\}^{m \times n}$：

$$\hat{W} = W \odot M$$

其中 $\odot$ 是逐元素乘法（Hadamard product）。

**优点**：
- 剪枝灵活度最高，可以精确控制每个权重
- 精度损失最小（在相同稀疏度下）

**缺点**：
- 产生**不规则稀疏矩阵**，无法直接用标准 BLAS 库加速
- 内存访问模式随机，cache miss 率高
- 需要专门的稀疏计算库（如 cuSPARSE）或专用硬件

#### 结构化剪枝 (Structured Pruning)

按照规则结构删除，保持矩阵的规则性。

**细粒度结构化**：

- **Pattern-based**：每个 kernel 按固定 pattern 剪枝，例如十字形、对角线形
- **Vector-level**：按向量（行或列）剪枝
- **Kernel-level**：删整个卷积核（$k \times k$），输入通道和输出通道连接的一个 kernel 块

**粗粒度结构化**：

- **Channel-level**：删整个输入/输出 channel。卷积层的 channel 数从 $C_{in}$ 减小
- **Filter-level**：删整个 filter（所有输入 channel 对应一个输出 channel 的所有 kernel）
- **Layer-level**：直接删整个层

**结构化剪枝的优点**：
- 剪枝后模型仍然是稠密矩阵，可以直接用标准 cuBLAS/cuDNN 加速
- 普通 GPU 无需任何特殊库支持
- 模型可以无损地序列化为更小的网络

**结构化剪枝的缺点**：
- 精度损失通常比非结构化剪枝大（约束更强）
- 要达到同等精度，需要的稀疏度（sparsity）更低

#### Channel 剪枝的参数量计算

对于一个卷积层，参数量为：
$$\text{Params} = C_{out} \times C_{in} \times k_H \times k_W$$

剪掉 $p$ 个输出 channel 后：
$$\text{Params}' = (C_{out} - p) \times C_{in} \times k_H \times k_W$$

同时，下一层的输入 channel 也需要对应减少，级联减参效果明显。

### 3.3 剪枝准则 (Pruning Criterion)

如何判断哪些权重"不重要"？这是剪枝的核心问题。

#### Magnitude-based 准则（幅度剪枝）

最直观的方法：绝对值越小越不重要。

**L1-norm 准则**（逐权重）：
$$\text{score}(w_{ij}) = |w_{ij}|$$

**L2-norm 准则**（按 filter/channel）：
$$\text{score}(\mathbf{f}_j) = \|\mathbf{f}_j\|_2 = \sqrt{\sum_{i,k,l} w_{ijkl}^2}$$

**L1-norm 按 filter**：
$$\text{score}(\mathbf{f}_j) = \|\mathbf{f}_j\|_1 = \sum_{i,k,l} |w_{ijkl}|$$

**优点**：计算简单，不需要数据，训练完直接用  
**缺点**：忽略了权重之间的相关性，忽略了激活分布

#### Saliency-based 准则（显著性剪枝）

考虑删掉某个权重对 loss 的影响。

**一阶 Taylor 展开**：

设当前权重为 $w$，删除该权重等价于 $\Delta w = -w$，loss 变化为：

$$\Delta \mathcal{L} \approx \frac{\partial \mathcal{L}}{\partial w} \cdot (-w) = -\frac{\partial \mathcal{L}}{\partial w} \cdot w$$

权重的重要性定义为 loss 变化的绝对值：

$$\text{saliency}(w) = \left| \frac{\partial \mathcal{L}}{\partial w} \cdot w \right|$$

这就是 **SNIP (Single-shot Network Pruning)** 的核心：

$$s_j = \left| \frac{\partial \mathcal{L}(\mathbf{w}; \mathcal{D})}{\partial w_j} \cdot w_j \right|$$

**优势**：同时考虑了权重大小 $|w|$ 和梯度大小 $|\partial \mathcal{L}/\partial w|$，一个权重大但梯度小（loss 对它不敏感），也可以剪掉。

#### 二阶 Taylor 展开（Optimal Brain Damage / Surgeon）

更精确的估计，包含曲率信息。

$$\Delta \mathcal{L} \approx \frac{\partial \mathcal{L}}{\partial w} \Delta w + \frac{1}{2} \Delta w^T H \Delta w$$

其中 $H = \frac{\partial^2 \mathcal{L}}{\partial w^2}$ 是 Hessian 矩阵。

假设在局部最优点 $\frac{\partial \mathcal{L}}{\partial w} = 0$，且只删单个权重 $w_i$（即 $\Delta w = -w_i \mathbf{e}_i$）：

$$\Delta \mathcal{L}_i \approx \frac{1}{2} \frac{w_i^2}{[H^{-1}]_{ii}}$$

**OBD (Optimal Brain Damage)**：假设 Hessian 是对角矩阵，只用对角元素：
$$\text{saliency}_{\text{OBD}}(w_i) = \frac{1}{2} h_{ii} w_i^2$$

**OBS (Optimal Brain Surgeon)**：使用完整 Hessian 逆，不需要假设对角：
$$\delta w^* = -\frac{w_i}{[H^{-1}]_{ii}} H^{-1} \mathbf{e}_i$$

OBS 同时给出最优的权重更新方向（其他权重如何补偿这个删除）。

#### APoZ (Average Percentage of Zeros)

基于激活稀疏性的准则。对一批数据前向传播，统计每个 channel 的激活有多少比例为 0：

$$\text{APoZ}(c) = \frac{1}{N \cdot H \cdot W} \sum_{n, h, w} \mathbf{1}[\text{ReLU}(\mathbf{z}_{n,c,h,w}) = 0]$$

APoZ 越高 → 该 channel 的激活大部分是 0 → 该 channel 可以删掉。

#### Reconstruction Error 最小化

删除权重后，尽量让下一层的输入分布不变。设第 $l$ 层输出为 $\mathbf{y}^{(l)}$，剪枝后的输出为 $\hat{\mathbf{y}}^{(l)}$：

$$\min_{M} \|\mathbf{y}^{(l)} - \hat{\mathbf{y}}^{(l)}\|_2^2 \quad \text{s.t.} \|M\|_0 \leq k$$

即在给定稀疏度约束下，最小化特征重建误差。这个方法需要数据，但通常精度比 magnitude pruning 好。

### 3.4 剪枝流程

#### 标准流程：训练-剪枝-微调

```
预训练 → 剪枝（按准则去掉权重）→ 微调（恢复精度）
```

微调的必要性：剪枝后模型精度下降，需要在小学习率下继续训练来恢复。微调通常比从头训练快很多。

#### 迭代剪枝 (Iterative Pruning)

每次剪掉一小部分，微调恢复，再继续剪：

```
训练 → [剪10% → 微调 → 剪10% → 微调 → ...]直到目标稀疏度
```

**优点**：比一次性剪到目标稀疏度精度更好（每次剪枝量小，模型有时间自适应）  
**缺点**：训练开销大，每一轮都需要微调

#### 一次性剪枝 (One-shot Pruning)

直接剪到目标稀疏度，然后微调一次。计算开销小，但精度可能略差。

#### 全局剪枝 vs 局部剪枝

- **局部 (Layer-wise)**：每层独立设定剪枝率，例如每层都剪 30%
- **全局 (Global)**：设全局阈值，所有层共享同一个评分排名，有些层可能剪很多，有些层几乎不剪

全局剪枝更灵活，因为不同层对精度的敏感度不同（第一层和最后一层通常更敏感）。

---

## 数学推导

### 推导 1：L1 Filter Pruning 的 FLOPs 减少量

设原始卷积层：输入 $(C_{in}, H, W)$，输出 $(C_{out}, H', W')$，kernel size $k \times k$。

FLOPs（乘加操作数）：
$$\text{FLOPs} = C_{out} \times C_{in} \times k^2 \times H' \times W' \times 2$$

（每个输出元素需要 $C_{in} \times k^2$ 次乘加，共 $C_{out} \times H' \times W'$ 个输出元素，乘 2 是因为乘法和加法各算一次）

剪掉 $m$ 个 filter（输出 channel）后：
$$\text{FLOPs}' = (C_{out} - m) \times C_{in} \times k^2 \times H' \times W' \times 2$$

FLOPs 减少比例：
$$\frac{\text{FLOPs} - \text{FLOPs}'}{\text{FLOPs}} = \frac{m}{C_{out}}$$

即剪枝率直接等于 FLOPs 减少比例（对于这一层）。

### 推导 2：SNIP Saliency 推导

设网络参数为 $\mathbf{w}$，数据集为 $\mathcal{D}$，损失函数为 $\mathcal{L}(\mathbf{w}; \mathcal{D})$。

引入掩码 $\mathbf{c} \in \{0, 1\}^{|\mathbf{w}|}$，修改后的损失为：

$$\mathcal{L}(c \odot \mathbf{w}; \mathcal{D})$$

对 $c_j$ 的影响：令 $\delta c_j = c_j - 1$（从保留到删除的变化），一阶近似：

$$\Delta \mathcal{L} \approx \frac{\partial \mathcal{L}(c \odot \mathbf{w}; \mathcal{D})}{\partial c_j} \Bigg|_{c=\mathbf{1}} \cdot \delta c_j$$

计算偏导数（链式法则）：

$$\frac{\partial \mathcal{L}(c \odot \mathbf{w}; \mathcal{D})}{\partial c_j} \Bigg|_{c=\mathbf{1}} = \frac{\partial \mathcal{L}(\mathbf{w}; \mathcal{D})}{\partial w_j} \cdot w_j$$

因此 saliency 定义为对 loss 影响的绝对值：

$$s_j = \left| \frac{\partial \mathcal{L}(\mathbf{w}; \mathcal{D})}{\partial w_j} \cdot w_j \right|$$

**注意**：这只需要做一次前向+反向传播，计算开销是 $O(|\mathbf{w}|)$，和梯度计算一样高效。

### 推导 3：全局阈值的推导

给定目标稀疏度 $s$（比如保留 30% 的权重），全局阈值 $\tau$ 满足：

$$\tau = \text{quantile}_{1-s}\left(\{|w| : w \in \mathbf{W}\}\right)$$

即找到所有权重绝对值的第 $(1-s)$ 百分位数。掩码为：

$$M_{ij} = \mathbf{1}[|w_{ij}| \geq \tau]$$

---

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# ============================================================
# Part 1: 不同粒度剪枝的 PyTorch 实现
# ============================================================

class SimpleCNN(nn.Module):
    """用于演示剪枝的简单 CNN"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # (B,32,14,14)
        x = self.pool(self.relu(self.conv2(x)))   # (B,64,7,7)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def count_sparsity(model):
    """统计模型的全局稀疏度（0 值占比）"""
    total = 0
    zeros = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total += param.numel()
            zeros += (param == 0).sum().item()
    return zeros / total


# 1. 非结构化剪枝：逐权重 L1 magnitude pruning
def unstructured_pruning(model, amount=0.5):
    """
    对所有卷积层和全连接层做非结构化 L1 剪枝
    amount: 剪掉的比例 (0.5 = 50% 的权重置为 0)
    """
    model_copy = deepcopy(model)
    for name, module in model_copy.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # L1 非结构化剪枝：找绝对值最小的 amount 比例的权重，置 0
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model_copy


# 2. 结构化剪枝：按 L2-norm 删整个 filter (输出 channel)
def structured_filter_pruning(model, amount=0.3):
    """
    对卷积层按 L2-norm 做结构化 filter 剪枝
    amount: 删掉的 filter 比例
    """
    model_copy = deepcopy(model)
    for name, module in model_copy.named_modules():
        if isinstance(module, nn.Conv2d):
            # ln_structured: 按 L2-norm 剪枝，dim=0 是输出 channel 维度
            prune.ln_structured(
                module, name='weight', amount=amount, n=2, dim=0
            )
    return model_copy


# 3. 全局剪枝：所有层共用一个阈值
def global_unstructured_pruning(model, amount=0.5):
    """全局 L1 非结构化剪枝：所有层参数一起排序，统一阈值"""
    model_copy = deepcopy(model)
    # 收集所有 (module, weight_name) 对
    parameters_to_prune = []
    for name, module in model_copy.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return model_copy


# ============================================================
# Part 2: 手动实现 Magnitude Pruning（不用 torch.prune）
# ============================================================

def magnitude_pruning_manual(weight_tensor, sparsity):
    """
    手动实现 L1 magnitude pruning
    weight_tensor: 权重张量
    sparsity: 目标稀疏度，例如 0.9 = 保留最大的 10%
    返回: 掩码 mask (1 = 保留, 0 = 删除)
    """
    # 计算阈值：找到绝对值的第 sparsity 百分位数
    threshold = torch.quantile(weight_tensor.abs().float(), sparsity)
    mask = (weight_tensor.abs() >= threshold).float()
    return mask


def apply_mask_and_check(model, sparsity=0.9):
    """对 fc1 层手动做 magnitude pruning 并验证"""
    w = model.fc1.weight.data
    mask = magnitude_pruning_manual(w, sparsity)

    # 应用掩码
    pruned_w = w * mask

    actual_sparsity = (pruned_w == 0).float().mean().item()
    print(f"目标稀疏度: {sparsity:.1%}")
    print(f"实际稀疏度: {actual_sparsity:.1%}")
    print(f"保留权重数: {(mask == 1).sum().item()} / {mask.numel()}")
    return mask


# ============================================================
# Part 3: SNIP Saliency 计算
# ============================================================

def compute_snip_saliency(model, dataloader, criterion, device='cpu'):
    """
    计算 SNIP saliency: s_j = |grad_j * w_j|
    只需要一个 batch 的数据
    """
    model = model.to(device)
    model.train()

    # 只取一个 batch 来计算 saliency
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    # 前向+反向传播
    model.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()

    # 计算每层权重的 saliency
    saliency_dict = {}
    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            # saliency = |grad * weight|
            saliency = (param.grad * param.data).abs()
            saliency_dict[name] = saliency.detach().cpu()
            print(f"  {name}: mean saliency = {saliency.mean().item():.6f}, "
                  f"shape = {saliency.shape}")

    return saliency_dict


# ============================================================
# Part 4: 稀疏度-精度曲线
# ============================================================

def sparsity_accuracy_curve(model, test_loader, sparsity_levels, device='cpu'):
    """
    对不同稀疏度做剪枝，记录精度，画出稀疏度-精度曲线
    这里只用非结构化 L1 剪枝做演示
    """
    criterion = nn.CrossEntropyLoss()
    results = []

    for sparsity in sparsity_levels:
        # 对原始模型做剪枝（每次都从原始模型开始）
        pruned_model = unstructured_pruning(model, amount=sparsity)

        # 评估精度
        pruned_model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = pruned_model(images)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        actual_sparsity = count_sparsity(pruned_model)
        results.append((sparsity, actual_sparsity, acc))
        print(f"目标稀疏度={sparsity:.1%} | "
              f"实际稀疏度={actual_sparsity:.1%} | "
              f"精度={acc:.2%}")

    return results


# ============================================================
# Part 5: 演示主程序
# ============================================================

def demo():
    torch.manual_seed(42)
    model = SimpleCNN()

    print("=" * 50)
    print("原始模型统计")
    print("=" * 50)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    print(f"初始稀疏度: {count_sparsity(model):.2%}")

    print("\n" + "=" * 50)
    print("非结构化剪枝（50%）")
    print("=" * 50)
    pruned_unstructured = unstructured_pruning(model, amount=0.5)
    print(f"稀疏度: {count_sparsity(pruned_unstructured):.2%}")
    # 验证：权重结构没变，只有值被置 0
    print(f"conv1.weight shape: {pruned_unstructured.conv1.weight.shape}")

    print("\n" + "=" * 50)
    print("结构化 Filter 剪枝（30%）")
    print("=" * 50)
    pruned_structured = structured_filter_pruning(model, amount=0.3)
    print(f"稀疏度: {count_sparsity(pruned_structured):.2%}")

    print("\n" + "=" * 50)
    print("手动 Magnitude Pruning 验证")
    print("=" * 50)
    apply_mask_and_check(model, sparsity=0.9)

    print("\n" + "=" * 50)
    print("全局剪枝（70%）")
    print("=" * 50)
    pruned_global = global_unstructured_pruning(model, amount=0.7)
    print(f"稀疏度: {count_sparsity(pruned_global):.2%}")

    # 打印不同层的稀疏度（全局剪枝下各层稀疏度不同）
    for name, module in pruned_global.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight
            layer_sparsity = (w == 0).float().mean().item()
            print(f"  {name}: 稀疏度 = {layer_sparsity:.2%}, shape = {w.shape}")


if __name__ == '__main__':
    demo()
```

---

## Infra 实战映射

### vLLM

vLLM 本身主要针对 LLM serving，**不直接实现剪枝**，但与剪枝的交互体现在：

- **加载剪枝后的模型**：vLLM 可以加载用 SparseGPT / Wanda 等工具剪枝后的 HuggingFace 模型（非结构化稀疏权重）。此时 vLLM 仍用稠密计算，剪枝没有实际加速。
- **结构化剪枝友好**：如果做了 channel pruning 并把模型保存为更小的 `config.json`（hidden_dim 减小），vLLM 加载后可以直接享受加速。
- **2:4 稀疏**：vLLM roadmap 上有 NVIDIA 2:4 稀疏支持，利用 `cusparseLt` 实现 2x matmul 加速。

实际操作：用 `neural-compressor` 或 `llm-compressor` 做稀疏，然后 vLLM 加载：
```python
# 加载稀疏模型（如 SparseGPT 处理过的 LLaMA）
from vllm import LLM
llm = LLM(model="neuralmagic/llama-2-7b-pruned50-retrained", dtype="float16")
```

### TensorRT-LLM

NVIDIA 对剪枝的支持更深入：

- **2:4 结构化稀疏（Ampere+）**：TRT-LLM 原生支持，通过 `prune_weights_for_sparse_gpu` 把权重转换为 2:4 格式，推理时用 `cusparseLt` 实现 ~2x matmul 加速，且对精度影响极小。
- **SparseGPT 集成**：TRT-LLM 的 `ModelOpt` 工具链（`nvidia-modelopt`）集成了 SparseGPT，可以一键对 LLM 做 2:4 剪枝 + 量化：
  ```bash
  python hf_ptq.py --model_dir llama-7b --sparsity_fmt dense_and_sparse --qformat int4_awq
  ```
- **Structured pruning**：`ModelOpt` 支持 channel pruning，剪枝后直接 export 为更小的 TRT engine。

### 沐曦 MACA

沐曦 GPU（曦云系列）基于 MACA（Metax Advanced Computing Architecture）：

- **非结构化稀疏**：目前 MACA 尚未有像 cuSPARSELt 那样成熟的稀疏加速库。推理非结构化稀疏模型时基本是稠密计算，稀疏没有加速收益。
- **结构化剪枝**：是目前最务实的选择。Channel pruning 后模型变小，直接在 MACA 上用标准矩阵乘法跑，没有硬件依赖问题。
- **MACA 特有考虑**：
  - MACA 的 `mxlib`（对应 cuBLAS）对小 batch size 下的 matmul 优化较弱，结构化剪枝减小 hidden_dim 后效果更明显
  - 曦思（元脑）推理框架支持加载标准 HuggingFace 格式的剪枝模型
  - 国产硬件上通常优先做量化（int8/int4）而不是剪枝，因为量化工具链更成熟

---

## 跨 Lecture 关联

- **前置知识** ← Lec02（模型效率基础：FLOPs、参数量、内存带宽）
- **后续延伸** → Lec04（剪枝进阶：LTH、自动剪枝、硬件稀疏支持）
- **横向关联** ↔ Lec05/06（量化，可与剪枝组合），Lec09（蒸馏，结构化剪枝后用蒸馏恢复精度）

---

## 面试高频题

**Q1: 非结构化剪枝和结构化剪枝的本质区别是什么？各自适用场景？**

A: 非结构化剪枝删除单个权重，产生不规则稀疏矩阵，精度损失小但需要专门稀疏硬件（如 NVIDIA 2:4 Sparse）才能加速；结构化剪枝删除整个 channel/filter/layer，保持矩阵规则性，普通 GPU 直接加速，但精度损失通常更大。生产环境优先结构化剪枝，或在 Ampere+ 卡上做 2:4 非结构化剪枝。

**Q2: 为什么 magnitude pruning 不一定是最优准则？**

A: Magnitude pruning 只看权重的绝对值大小，忽略了权重对 loss 的实际影响。一个权重很大，但如果 loss 对它的梯度接近 0（即 loss 对它不敏感），删掉它精度损失也很小。SNIP 的 $|g \cdot w|$ 同时考虑了权重大小和梯度大小，更准确反映权重的重要性。

**Q3: 剪枝后为什么需要微调（fine-tune）？微调时学习率怎么设置？**

A: 剪枝破坏了网络中已经学到的特征表示和层间协作关系，直接剪枝后精度大幅下降。微调让剩余权重重新适应新的网络结构。学习率通常设为原始训练 lr 的 1/10 到 1/100，避免破坏已有的权重分布。一般微调 10-20% 的原始训练 epoch 数就够了。

**Q4: 迭代剪枝比一次性剪枝好在哪里？代价是什么？**

A: 迭代剪枝每次只剪一小部分权重，每次微调后网络有机会重新分配重要性，后续剪枝的判断更准确；一次性剪枝可能误删了重要权重，微调无法完全恢复。代价是训练总开销是 $O(N \times \text{iterations})$，迭代剪枝轮数越多越耗时。

**Q5: 全局剪枝和逐层剪枝，哪个更好？**

A: 通常全局剪枝更好。不同层对精度的敏感度不同：第一层（输入附近）和最后一层（分类头附近）通常更重要，中间层可以剪更多。逐层统一剪枝率会对敏感层剪太多，对不敏感层剪太少。全局剪枝让网络自动平衡。

**Q6: 给一个已部署的 LLM，你会怎么做剪枝？**

A: 首先选择准则：对 LLM 来说，SparseGPT（二阶信息，逐层重建误差最小化）或 Wanda（$|W| \cdot \|X\|_2$，激活感知）效果好于纯 magnitude pruning。其次选择粒度：如果目标是在 NVIDIA Ampere+ GPU 上跑，做 2:4 结构化稀疏；如果是通用场景，做 channel/attention head pruning。最后决定是否微调：小模型 fine-tune 成本低，建议做；70B+ 的 LLM fine-tune 成本极高，可以考虑 SparseGPT 的 one-shot 方案（无需微调）。
