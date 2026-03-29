# Seq2Seq、注意力机制与 Transformer

## 一、Seq2Seq：编码器-解码器框架

### 1.1 任务定义

Seq2Seq（Sequence-to-Sequence）指输入一个序列、输出另一个序列的任务。机器翻译是最典型的例子："今天天气真好" → "Today is a good day"。事实上几乎所有 NLP 任务都可以归为 Seq2Seq：文本分类是输出长度为 1 的目标序列，词性标注是输出与输入等长的序列。

### 1.2 Encoder-Decoder 结构

处理 Seq2Seq 的通用思路是**编码再解码**：

```
源序列 ──Encoder──▶ 上下文向量 c ──Decoder──▶ 目标序列
```

- **编码器**：逐词读入源序列，将整句信息压缩为一个固定长度的上下文向量 $c$（通常取 RNN 最后一个时间步的隐藏状态）
- **解码器**：以 $c$ 为初始输入，自回归地逐词生成目标序列——每一步的输出作为下一步的输入

### 1.3 信息瓶颈

问题出在"固定长度的上下文向量"上。无论源序列是 5 个词还是 50 个词，编码器都必须把全部信息压缩到同一个向量里。序列越长，信息损失越严重。Cho et al. (2014) 的实验表明，基于 RNN 的 Seq2Seq 模型在句子长度超过 20 词后性能急剧下降。

这就是注意力机制要解决的问题。

## 二、注意力机制：动态对齐

### 2.1 核心思想

与其把整个源序列压缩成一个向量，不如让解码器在生成每个词时**动态地"回看"源序列**，选择性地关注最相关的部分。

Bahdanau et al. (2014) 提出的注意力机制做了一个关键改动：解码器在每个时间步都能访问编码器的**所有**隐藏状态 $h_1, h_2, \dots, h_T$，而非仅最后一个。

### 2.2 计算流程（交叉注意力）

以解码器在时间步 $t$ 生成词为例：

1. **计算注意力分数**：用解码器当前状态 $s_t$ 与编码器每个隐藏状态 $h_j$ 计算相关性分数 $e_{tj} = \text{score}(s_t, h_j)$
2. **归一化**：对分数做 Softmax 得到注意力权重 $\alpha_{tj} = \text{softmax}(e_{tj})$
3. **加权求和**：用权重对编码器隐藏状态求和，得到上下文向量 $c_t = \sum_j \alpha_{tj} h_j$
4. **融合输出**：将 $c_t$ 与 $s_t$ 拼接后送入输出层

关键点：每个时间步的上下文向量 $c_t$ 是**不同的**——翻译"天气"时关注源序列中的"天气"，翻译"good"时关注"好"。这就是"动态对齐"。

### 2.3 QKV 范式

Luong et al. (2015) 将注意力计算统一为 Query-Key-Value 框架：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Query (Q)**：当前要生成的位置（"我在找什么"）
- **Key (K)**：源序列各位置的索引（"这里有什么"）
- **Value (V)**：源序列各位置的实际内容（"匹配上了就取这个"）
- $\sqrt{d_k}$：缩放因子，防止点积过大导致 Softmax 梯度消失

在交叉注意力中，Q 来自解码器，K 和 V 来自编码器。

### 2.4 注意力的局限

注意力解决了信息瓶颈，但底层仍然依赖 RNN 逐步处理序列——无法并行，长距离依赖问题仍未根治。

## 三、Transformer：完全基于注意力的架构

Vaswani et al. (2017) 的 *"Attention Is All You Need"* 彻底抛弃 RNN，整个模型只用注意力机制构建。

### 3.1 自注意力（Self-Attention）

与交叉注意力（Q 和 K/V 来自不同序列）不同，自注意力的 **Q、K、V 全部来自同一个输入序列**。序列中的每个词元同时充当查询者、被查询的索引和信息提供者。

计算过程：

1. 对输入 $X$ 分别乘以三个可学习的权重矩阵，得到 $Q = XW^Q$，$K = XW^K$，$V = XW^V$
2. 计算注意力：$Z = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

$QK^T$ 是一个 $[\text{seq\_len}, \text{seq\_len}]$ 的矩阵，第 $i$ 行第 $j$ 列表示第 $i$ 个词对第 $j$ 个词的关注程度。整个计算是矩阵乘法，可以完全并行。

**为什么需要三个独立矩阵而不是直接用 $X$？** 因为"查询"、"索引"、"内容"在信息检索中承担不同角色。$W^Q$、$W^K$、$W^V$ 让模型学会把同一个输入投影到三个功能不同的空间中。

**PyTorch 实现：**

```python
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_size)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)
```

### 3.2 多头注意力（Multi-Head Attention）

单组 $W^Q, W^K, W^V$ 只能从一个"视角"捕获关系。多头注意力并行运行 $h$ 组独立的注意力计算，每组关注不同类型的依赖（语法、语义、指代等），最后拼接结果并投影：

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

实践中，将 `hidden_size` 均分给 $h$ 个头（如 512 维 / 8 头 = 每头 64 维），总计算量与单头相当。

**高效实现的关键**：不是创建 $h$ 个独立的注意力模块，而是一次性做完所有头的线性变换，通过 `view` + `transpose` 拆分多头维度，利用批量矩阵乘法并行计算。

```python
# 拆分多头：[batch, seq_len, hidden] → [batch, num_heads, seq_len, head_dim]
q = q.view(B, T, num_heads, head_dim).transpose(1, 2)
# 并行计算所有头的注意力
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
```

### 3.3 整体架构

Transformer 仍然是 Encoder-Decoder 结构，但内部用注意力替代了 RNN。编码器和解码器各由 $N$ 层（原论文 $N=6$）堆叠而成。

**编码器层**（每层两个子层）：
1. 多头**自注意力**（双向，每个词能看到所有其他词）
2. 逐位置前馈网络（FFN）

**解码器层**（每层三个子层）：
1. **带掩码的**多头自注意力（单向，只能看到已生成的词）
2. 多头**交叉注意力**（Q 来自解码器，K/V 来自编码器输出）
3. 逐位置前馈网络（FFN）

每个子层后都接**残差连接 + 层归一化**（Add & Norm）。

### 3.4 关键组件详解

#### 逐位置前馈网络（FFN）

独立作用于每个位置的两层全连接网络，结构是"升维 → 激活 → 降维"：

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

$W_1$ 将维度从 $d_{model}$ 升到 $4 \times d_{model}$（投射到高维空间提取复杂模式），$W_2$ 再降回来。注意力子层提供的主要是 Softmax 归一化，逐位置的非线性变换由 FFN 承担。

#### 残差连接（Add）

每个子层的输出加上其输入：$\text{output} = \text{Sublayer}(x) + x$。作用是让梯度能直接跨层传播，解决深层网络的训练困难（类似 ResNet 的思路）。

#### 层归一化（Norm）

对每个样本的特征维度做归一化（均值为 0，方差为 1），加速训练收敛并稳定梯度。与 Batch Norm 不同，Layer Norm 不依赖 batch size，更适合变长序列。

#### 位置编码

自注意力本身是置换不变的——打乱词序不影响计算结果。为了注入位置信息，Transformer 在输入嵌入上加上位置编码。原论文使用正弦/余弦函数生成：

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

不同频率的正弦波编码不同尺度的位置关系。后续模型（如 RoPE）改用相对位置编码，在处理超长文本时泛化性更好。

#### 掩码（Masking）

两种用途：

- **因果掩码（Causal Mask）**：在解码器自注意力中，将未来位置的分数设为 $-\infty$，Softmax 后权重为 0，确保自回归特性
- **填充掩码（Padding Mask）**：batch 内句子长度不同时，标记填充位置，避免对 padding token 计算注意力

### 3.5 三种架构变体

Transformer 的 Encoder 和 Decoder 可以独立使用，衍生出三种主流架构：

| 架构 | 代表模型 | 注意力类型 | 擅长任务 |
|------|----------|-----------|----------|
| Encoder-Only | BERT | 双向自注意力 | 文本分类、NER 等理解任务 |
| Decoder-Only | GPT 系列 | 单向自注意力（带掩码） | 文本生成、当前 LLM 主流 |
| Encoder-Decoder | T5、原始 Transformer | 编码器双向 + 解码器单向 + 交叉注意力 | 翻译、摘要等 Seq2Seq 任务 |

## 四、从 RNN 到 Transformer 的演进脉络

```
RNN Seq2Seq          →  注意力 Seq2Seq       →  Transformer
(固定长度瓶颈)         (动态对齐，仍依赖 RNN)   (完全并行，长距离依赖)
```

| 维度 | RNN Seq2Seq | + Attention | Transformer |
|------|-------------|-------------|-------------|
| 序列建模 | 隐藏状态逐步传递 | 同左，但解码器可回看编码器全部状态 | 自注意力直接计算任意两个位置的关系 |
| 并行性 | 不可并行（逐时间步） | 不可并行 | 完全可并行（矩阵乘法） |
| 长距离依赖 | 梯度消失，实际只能捕获短距离 | 注意力缓解了瓶颈，但 RNN 本身仍受限 | $O(1)$ 路径长度，任意距离直接连接 |
| 计算复杂度 | $O(T)$ 顺序步 | $O(T)$ 顺序步 + $O(T^2)$ 注意力 | $O(T^2)$ 注意力（可并行） |

Transformer 用 $O(T^2)$ 的并行计算换掉了 $O(T)$ 的串行计算，在 GPU 上反而更快。当 $T$ 极大时（数万 token），$T^2$ 会成为瓶颈——这催生了后续的稀疏注意力、线性注意力等优化。

## 五、PyTorch 实现与验证

> 以下所有输出均为实际运行结果。完整代码见 [base-llm / code / C4](https://github.com/datawhalechina/base-llm/tree/main/code/C4)

### 5.1 自注意力

```python
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.q_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, mask=None):
        q, k, v = self.q_linear(x), self.k_linear(x), self.v_linear(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_size)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v), weights
```

```
输入形状:        (2, 5, 16)   # batch=2, seq_len=5, hidden=16
输出形状:        (2, 5, 16)   # 形状不变，但每个位置已融合全局上下文
注意力权重形状:  (2, 5, 5)    # seq_len × seq_len 的关注矩阵
每行权重和:      [1.0, 1.0, 1.0, 1.0, 1.0]  # Softmax 归一化
```

### 5.2 因果掩码（Causal Mask）

解码器的关键：用下三角掩码确保位置 $i$ 只能注意到 $\leq i$ 的位置。

```python
causal_mask = torch.tril(torch.ones(T, T))  # 下三角矩阵
```

```
掩码矩阵 (T=5):
  [[1 0 0 0 0]
   [1 1 0 0 0]
   [1 1 1 0 0]
   [1 1 1 1 0]
   [1 1 1 1 1]]

带掩码的注意力权重 (前4行):
  位置0: [1.0000, 0.0000, 0.0000, 0.0000]  ← 只能看到自己
  位置1: [0.5612, 0.4388, 0.0000, 0.0000]  ← 只能看到 0,1
  位置2: [0.4026, 0.2895, 0.3078, 0.0000]  ← 只能看到 0,1,2
  位置3: [0.1913, 0.0995, 0.3555, 0.3538]  ← 只能看到 0,1,2,3
```

上三角全为零——未来信息被彻底屏蔽，保证了自回归特性。

### 5.3 多头注意力

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wo = nn.Linear(hidden_size, hidden_size, bias=False)  # 输出投影

    def forward(self, x):
        B, T, _ = x.shape
        # 一次线性变换后拆分多头
        q = self.q_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # (B, num_heads, T, head_dim) — num_heads 作为批次维度并行计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)
        # 合并多头：(B, num_heads, T, head_dim) → (B, T, hidden_size)
        context = context.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(context)
```

```
输入形状:          (2, 5, 16)
输出形状:          (2, 5, 16)
注意力权重形状:    (2, 4, 5, 5)   # 4 个头各自有独立的 5×5 注意力矩阵
每头维度:          16 / 4 = 4
参数量:  单头 768  vs  多头 1024 (多了 W_O)
```

### 5.4 逐位置前馈网络（FFN）

```python
class FFN(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff)    # 升维
        self.w2 = nn.Linear(d_ff, d_model)    # 降维
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w2(self.relu(self.w1(x)))
```

```
中间升维: 16 → 64 → 16
FFN 参数量: 2128
```

### 5.5 编码器层（Encoder Layer）

把多头注意力和 FFN 用 Add & Norm 串起来：

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FFN(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.self_attn(x)
        x = self.norm1(x + attn_out)       # 残差 + 层归一化
        x = self.norm2(x + self.ffn(x))    # 残差 + 层归一化
        return x
```

```
编码器层参数量: 3216
```

### 5.6 位置编码

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

```
前4个位置, 前4维的编码值:
  pos=0: [ 0.0000,  1.0000,  0.0000,  1.0000]
  pos=1: [ 0.8415,  0.5403,  0.3110,  0.9504]
  pos=2: [ 0.9093, -0.4161,  0.5911,  0.8066]
  pos=3: [ 0.1411, -0.9900,  0.8126,  0.5828]
```

偶数维是 sin，奇数维是 cos，不同维度频率不同——低维变化快（编码细粒度位置），高维变化慢（编码粗粒度位置）。

### 5.7 堆叠 6 层编码器

```python
class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])

    def forward(self, x):
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        return x
```

```
编码器层数:   6
输入形状:     (2, 5, 16)
输出形状:     (2, 5, 16)
总参数量:     19296
每层参数量:   3216
```

与 PyTorch 官方 `nn.TransformerEncoder` 的输出形状一致——验证通过。

### 5.8 维度变换全流程

以 `batch=2, seq_len=5, d_model=16, num_heads=4` 为例：

```
输入 x:                             (2, 5, 16)
  │
  ├─ 位置编码 PE:                   (2, 5, 16)  加法，形状不变
  │
  ├─ 线性变换 → Q, K, V:           (2, 5, 16)  各自独立
  │    └─ view + transpose 拆头:    (2, 4, 5, 4)  4头×4维
  │
  ├─ Q × K^T (缩放):               (2, 4, 5, 5)  每头一个注意力矩阵
  ├─ Softmax:                       (2, 4, 5, 5)
  ├─ × V:                           (2, 4, 5, 4)
  │    └─ transpose + view 合并头:  (2, 5, 16)
  ├─ W_O 输出投影:                  (2, 5, 16)
  │
  ├─ Add & Norm:                    (2, 5, 16)
  │
  ├─ FFN (16→64→16):               (2, 5, 16)
  ├─ Add & Norm:                    (2, 5, 16)
  │
  └─ 重复 ×6 层 → 最终输出:         (2, 5, 16)
```

## 参考文献

1. Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
2. Bahdanau, D., Cho, K., Bengio, Y. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR 2015.
3. Luong, M., Pham, H., Manning, C. D. (2015). *Effective Approaches to Attention-based Neural Machine Translation*. EMNLP 2015.
4. Cho, K., et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*. EMNLP 2014.
5. Ba, J. L., Kiros, J. R., Hinton, G. E. (2016). *Layer Normalization*.
6. Su, J., et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*.
