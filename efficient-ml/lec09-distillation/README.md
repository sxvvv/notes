# 09 知识蒸馏 (Knowledge Distillation)

> 📺 [Lecture 09 - Knowledge Distillation](https://hanlab.mit.edu/courses/2024-fall-65940)
> 📄 [Slides](https://hanlab.mit.edu/courses/2024-fall-65940)

---

## 9.1 核心思想

### 9.1.1 Dark Knowledge

> "The dark knowledge is in the relative probabilities of the incorrect answers." — Geoffrey Hinton

大模型 (Teacher) 的输出不只是"正确答案"，还包含了大量**类别间关系信息**。

```
Teacher 预测:
  猫: 0.7  狗: 0.2  车: 0.05  树: 0.03  鸟: 0.02

Hard Label (ground truth):
  猫: 1.0  狗: 0.0  车: 0.0   树: 0.0   鸟: 0.0
```

Teacher 的 soft label 告诉 Student: "猫和狗很像，车和树不太相关" — 这就是 dark knowledge。

### 9.1.2 Temperature

$$\text{softmax}(z_i, T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- $T = 1$: 标准 softmax，输出尖锐
- $T \to \infty$: 输出趋向均匀分布
- **蒸馏用 $T > 1$**: 让输出更 soft，传递更多关系信息

```
T=1:   [0.7, 0.2, 0.05, 0.05]  ← 信息集中在最大值
T=5:   [0.35, 0.30, 0.18, 0.17] ← 信息更均匀，类别关系更清晰
```

### 9.1.3 蒸馏损失函数

$$L = \alpha \cdot L_{soft} + (1 - \alpha) \cdot L_{hard}$$

- $L_{soft} = \text{KL}(\text{softmax}(z_s/T) \| \text{softmax}(z_t/T)) \times T^2$
- $L_{hard} = \text{CE}(y_s, y_{true})$ — 标准交叉熵
- $\alpha$ 通常取 0.5-0.9
- **乘 $T^2$**: 因为 soft target 的梯度被 $1/T$ 缩小了，需要补偿

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=5.0, alpha=0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels):
        # Soft target loss (蒸馏)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits / self.T, dim=1),
            reduction='batchmean'
        ) * (self.T ** 2)

        # Hard target loss (标准分类)
        hard_loss = F.cross_entropy(student_logits, labels)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

---

## 9.2 蒸馏什么 (What to Match)

### 9.2.1 Logits 蒸馏 (Response-Based)
- 最基本：只匹配最终输出
- Hinton 2015 原始方法
- 简单但信息量有限

### 9.2.2 Feature 蒸馏 (Feature-Based)
- 匹配中间层特征 → **FitNets**
- Teacher 的中间层特征比 Student 大 → 需要一个 adapter（小网络）对齐维度

```python
class FeatureDistillation(nn.Module):
    def __init__(self, teacher_dim, student_dim):
        super().__init__()
        self.adapter = nn.Linear(student_dim, teacher_dim)

    def forward(self, student_feat, teacher_feat):
        # Student 特征映射到 Teacher 维度，然后 MSE
        adapted = self.adapter(student_feat)
        return F.mse_loss(adapted, teacher_feat)
```

### 9.2.3 Attention 蒸馏
- 匹配 Teacher 的 attention map
- 传递: "模型应该关注哪里"的信息

### 9.2.4 关系蒸馏 (Relational)
- 不匹配单个样本的特征，而是匹配**样本间的关系**
- 例: Teacher 认为样本 A 和 B 很像，Student 也应该学到这种关系

---

## 9.3 蒸馏方案

| 方案 | 描述 | 优缺点 |
|------|------|--------|
| Offline Distillation | Teacher 固定，只训练 Student | 简单，最常用 |
| Online Distillation | Teacher 和 Student 同时训练 | 互惠学习，效果可能更好 |
| Self-Distillation | 用模型自身的 earlier checkpoint 做 Teacher | 不需要额外模型 |

**Self-Distillation**: 用训练早期的模型教后期的模型。或者用 deeper layers 教 shallower layers。

---

## 9.4 应用

### 9.4.1 DistilBERT
- BERT → DistilBERT: 参数量减少 40%，保留 97% 能力
- 方法: Logits 蒸馏 + hidden state 蒸馏 + embedding cosine loss
- 6 层 Student 蒸馏 12 层 Teacher

### 9.4.2 GAN 蒸馏
- Generator 蒸馏: 把大 GAN 压缩成小 GAN
- Anycost GAN: 可变成本的生成器，灵活控制质量/速度

### 9.4.3 NetAug (反向思路)
- 传统: 大教小
- NetAug: **用更大的网络增强小网络训练**
- 在小网络外面套辅助结构（augmentation），训练完去掉
- 效果: 小网络精度提升

### 9.4.4 LLM 蒸馏实践

```
大模型 (Teacher) → 小模型 (Student)
GPT-4 → GPT-3.5 (推测)
LLaMA-70B → LLaMA-7B (通过生成的数据)
```

常见做法:
1. 用大模型生成高质量训练数据
2. 用这些数据微调小模型
3. Alpaca/Vicuna 的思路

```python
# LLM 蒸馏的简化流程
def llm_distill(teacher, student, prompts, tokenizer):
    """用 Teacher 生成数据，训练 Student"""
    for prompt in prompts:
        # Teacher 生成
        teacher_output = teacher.generate(tokenizer(prompt))
        # Student 学习
        loss = F.cross_entropy(
            student(tokenizer(prompt)).logits,
            teacher_output.logits.detach()
        )
        loss.backward()
```

---

## 代码示例: 完整蒸馏训练循环

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Teacher 和 Student
class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.net(x)

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )
    def forward(self, x):
        return self.net(x)

def distill_train(teacher, student, dataloader, epochs=10, T=5.0, alpha=0.7):
    teacher.eval()  # Teacher 冻结
    optimizer = torch.optim.Adam(student.parameters())

    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            with torch.no_grad():
                teacher_logits = teacher(x)

            student_logits = student(x)

            # 蒸馏 loss
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction='batchmean'
            ) * (T ** 2)

            hard_loss = F.cross_entropy(student_logits, y)
            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# 使用
teacher = TeacherNet()  # 预训练好的
student = StudentNet()
# distill_train(teacher, student, train_loader)
```

---

## Infra 实战映射

### vLLM / LLM 推理
- vLLM 本身不做蒸馏，但可以部署蒸馏后的小模型
- 蒸馏是训练阶段的工作，推理系统只关心模型格式兼容

### 工业界实践
- 很多公司的"小模型"背后是大模型蒸馏的结果
- Speculative Decoding (Lec13) 中的 draft model 可以看作一种推理时蒸馏
- RAG + 蒸馏: 用大模型+RAG生成高质量数据，再教小模型

### 沐曦 MACA
- 在 MACA 上部署蒸馏后的小模型，不需要特殊支持
- 可以用 MACA 硬件训练 Student 模型（蒸馏训练本身计算量不大）

---

## 跨 Lecture 关联

- **前置 ←** [Lec03: 剪枝](../lec03-pruning-I/README.md) — 剪枝+蒸馏组合
- **前置 ←** [Lec05: 量化](../lec05-quantization-I/README.md) — 量化+蒸馏恢复精度
- **延伸 →** [Lec10: MCUNet](../lec10-mcunet/README.md) — NetAug 反向思路
- **延伸 →** [Lec13: LLM 部署](../lec13-llm-deploy/README.md) — Speculative Decoding

---

## 面试高频题

**Q1: 为什么 Temperature 要 > 1？**
> A: T=1 时 softmax 输出太尖锐，"猫0.99, 狗0.01" 几乎等于 hard label，dark knowledge 很少。T>1 让输出更均匀，暴露更多类别间关系。但 T 太大信息就太模糊了，通常 T=3~10。

**Q2: 蒸馏损失中为什么要乘 T²？**
> A: soft target 的梯度被 1/T 缩小了（因为 logits 除以 T），如果不补偿，soft loss 的梯度会远小于 hard loss。乘 T² 让 soft 和 hard loss 的梯度量级匹配。

**Q3: 除了 logits 还能蒸馏什么？**
> A: Feature (FitNets — 中间层特征), Attention map, 关系(样本间距离)。越底层的蒸馏信息越丰富，但对齐成本也越高（维度不匹配需要 adapter）。

**Q4: 蒸馏和剪枝/量化怎么组合？**
> A: 常见 pipeline: 先剪枝/量化 → 精度下降 → 用蒸馏恢复精度。Teacher 用原始模型，Student 用压缩后模型。三者可以叠加: 剪枝+量化+蒸馏。
