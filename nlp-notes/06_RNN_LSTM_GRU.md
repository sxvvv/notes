# 从 RNN 到 LSTM、GRU：序列建模的演进

## 一、核心问题：如何编码一个变长序列

经过分词和 Word2Vec，已经能把一句话变成一组词向量。但"播放周杰伦的《稻香》"是 4 个向量，"今天天气怎么样"是 5 个——长度不固定，而分类器要求一个定长输入。需要一种方法把**词向量序列**压缩成一个**文本向量**。

早期方案的问题：

| 方法 | 做法 | 缺陷 |
|------|------|------|
| 向量求和/平均 | 所有词向量逐元素相加 | "我爱你"和"你爱我"结果相同，丢失语序 |
| 全连接网络 | 对每个时间步共享权重做变换 | 每个词被孤立处理，无法利用上下文 |
| 一维 CNN | 滑动窗口捕获局部 n-gram | 感受野固定，难以捕获长距离依赖 |

共同缺陷：没有"记忆"机制，无法让后面的计算利用前面的信息。

## 二、RNN：引入循环记忆

### 2.1 结构

RNN（Recurrent Neural Network）的核心思想：处理第 $t$ 个词时，不仅看当前输入 $x_t$，还接收上一步的隐藏状态 $h_{t-1}$ 作为"记忆"。公式：

$$h_t = \tanh(U \cdot x_t + W \cdot h_{t-1} + b)$$

其中 $U$（输入→隐藏）和 $W$（隐藏→隐藏）在所有时间步共享，这使得 RNN 能处理任意长度的序列。

> **时间步**：处理序列中第 $t$ 个元素的那一步。"播放"是 $t=1$，"周杰伦"是 $t=2$，以此类推。

### 2.2 工作流程

以"播放周杰伦的《稻香》"→ 分类为"音乐播放"为例（$E=128, H=3$）：

1. $t=1$：输入 $x_1$（"播放"）+ $h_0$（零向量）→ $h_1$ 包含"播放"的信息
2. $t=2$：输入 $x_2$（"周杰伦"）+ $h_1$ → $h_2$ 融合了"播放 周杰伦"
3. $t=3$：输入 $x_3$（"的"）+ $h_2$ → $h_3$ 融合了"播放 周杰伦 的"
4. $t=4$：输入 $x_4$（"《稻香》"）+ $h_3$ → $h_4$ 融合了整句信息

最终 $h_4$ 就是整句话的文本向量，送入全连接层做分类。

### 2.3 从零实现

**数据准备：**

```python
import numpy as np

B, E, H = 1, 128, 3  # batch_size, embedding_dim, hidden_dim

def prepare_inputs():
    np.random.seed(42)
    vocab = {"播放": 0, "周杰伦": 1, "的": 2, "《稻香》": 3}
    tokens = ["播放", "周杰伦", "的", "《稻香》"]
    ids = [vocab[t] for t in tokens]
    emb_table = np.random.randn(len(vocab), E).astype(np.float32)
    x_np = emb_table[ids][None]  # (1, 4, 128)
    return tokens, x_np
```

**NumPy 手写 RNN：**

```python
def manual_rnn_numpy(x_np, U_np, W_np):
    B_local, T_local, _ = x_np.shape
    h_prev = np.zeros((B_local, H), dtype=np.float32)
    steps = []
    for t in range(T_local):
        x_t = x_np[:, t, :]
        h_t = np.tanh(x_t @ U_np + h_prev @ W_np)  # 核心公式
        steps.append(h_t)
        h_prev = h_t
    return np.stack(steps, axis=1), h_prev
```

> 代码使用行向量 $x_t U$（矩阵在右），与公式 $U x_t$（列向量，矩阵在左）是转置关系，本质一致。

**运行输出：**

```
各时间步隐藏状态 h_t:
  t=1 '播放':    [ 0.919036,  0.273749, -0.330602]
  t=2 '周杰伦':  [ 0.367867,  0.615103, -0.868026]
  t=3 '的':      [-0.431706,  0.844129,  0.600168]
  t=4 '《稻香》': [-0.049530,  0.121081,  0.910190]
```

可以看到每一步的 $h_t$ 都在变化——新输入不断与累积记忆融合。

**与 PyTorch `nn.RNN` 对齐：**

```python
import torch, torch.nn as nn

def pytorch_rnn_forward(x, U, W):
    rnn = nn.RNN(input_size=E, hidden_size=H, num_layers=1,
                 nonlinearity='tanh', bias=False, batch_first=True)
    with torch.no_grad():
        rnn.weight_ih_l0.copy_(U.T)
        rnn.weight_hh_l0.copy_(W.T)
    return rnn(x)
```

```
NumPy 与 PyTorch nn.RNN 输出一致: True
```

`nn.RNN` 的关键参数：

| 参数 | 含义 |
|------|------|
| `input_size` | 输入维度 $E$（词向量维度） |
| `hidden_size` | 隐藏状态维度 $H$ |
| `num_layers` | 堆叠层数，>1 则前层输出作后层输入 |
| `batch_first` | `True` 时输入形状为 `(B, T, E)`；`False` 为 `(T, B, E)` |
| `bidirectional` | 是否双向 |

### 2.4 双向 RNN

单向 RNN 只能利用前文。但"**苹果**味道不错"和"**苹果**股票大涨"中，"苹果"的含义取决于后文。BiRNN 用两个独立的 RNN 分别从左到右和从右到左处理序列，最终拼接：

$$h_t = [\overrightarrow{h_t} \; ; \; \overleftarrow{h_t}]$$

每个时间步的输出维度变为 $2H$：

```python
birnn = nn.RNN(input_size=E, hidden_size=H, batch_first=True, bidirectional=True)
y, _ = birnn(x_torch)
print(y.shape)  # → torch.Size([1, 4, 6])  即 2×3=6
```

BiRNN 的局限：需要完整序列才能计算反向信息，无法用于实时逐词生成（如自回归语言模型）。

### 2.5 RNN 的致命缺陷

**训练机制**：BPTT（随时间反向传播）将 RNN 沿时间展开，本质是一个各层共享参数的深层网络。梯度从 $t$ 传到 $k$ 时需要连乘：

$$\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}$$

每一步乘以 $W$ 的导数矩阵和 $\tanh'$（最大值为 1，通常 < 1）。连乘的后果：

- $\|W\| < 1$ → 梯度指数衰减 → **梯度消失**：远距离信号传不回来，模型学不到长程依赖
- $\|W\| > 1$ → 梯度指数增长 → **梯度爆炸**：训练崩溃（可通过梯度裁剪缓解）

梯度消失更棘手——它意味着 RNN 在实践中只能捕获短距离依赖，长句中早期词的信息会被逐步"遗忘"。

## 三、LSTM：用门控管理记忆

### 3.1 设计思路

RNN 的状态更新是"粗暴"的：每一步都把新输入和旧记忆无差别地混合，再强制过一遍 $W$ 的矩阵乘法。LSTM（Long Short-Term Memory, Hochreiter & Schmidhuber 1997）的思路是**让网络自己决定信息的取舍**——通过可学习的"门"来控制哪些信息保留、哪些遗忘、哪些输出。

与 RNN 只有单一隐藏状态 $h_t$ 不同，LSTM 维护两条并行轨道：

- **细胞状态 $ c_t $**（长期记忆）：信息在上面以加法方式流动，不经过矩阵连乘——这是缓解梯度消失的关键
- **隐藏状态 $ h_t $**（短期记忆/输出）：由 $ c_t $ 经门控过滤后生成

### 3.2 四步计算

LSTM 单元在每个时间步接收 $ x_t $、$ h_{t-1} $、$ c_{t-1} $，输出 $ h_t $、$ c_t $。

**第一步：遗忘门** — 决定从长期记忆中丢弃什么

$$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

$ f_t $ 的每个元素在 (0,1) 之间。接近 1 = 保留，接近 0 = 丢弃。

**第二步：输入门 + 候选记忆** — 决定写入什么新信息

$$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \qquad \tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

$ i_t $ 控制写入强度，$ \tilde{c}_t $ 是准备写入的内容。

**第三步：更新细胞状态** — 旧记忆选择性保留 + 新信息选择性写入

$$ c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t $$

这是 LSTM 的核心：$c_t$ 的更新只涉及**逐元素乘法和加法**，没有额外的权重矩阵乘法。

**第四步：输出门** — 决定输出细胞状态的哪些部分

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \qquad h_t = o_t \odot \tanh(c_t)$$

### 3.3 为什么能缓解梯度消失

梯度沿细胞状态反向传播时：

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

从 $t$ 回溯到 $k$：$\frac{\partial L}{\partial c_k} = \frac{\partial L}{\partial c_t} \odot (f_t \odot f_{t-1} \odot \cdots \odot f_{k+1})$

与 RNN 对比：RNN 的梯度要连乘权重矩阵 $W$（结构性衰减），而 LSTM 只连乘遗忘门的值。如果模型学到某个信息需要长期保持，它可以把中间各步的 $f_t$ 都推向 1，梯度就能近乎无损地传回去。

所以说 LSTM **缓解**了梯度消失（足够长的序列上 $0.99^n$ 仍会衰减），但相比 RNN 几十步就失效的窘境，已经是本质性的改进。

### 3.4 从零实现

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def manual_lstm_numpy(x_np, weights):
    U_f, W_f, U_i, W_i, U_c, W_c, U_o, W_o = weights
    B_local, T_local, _ = x_np.shape
    h_prev = np.zeros((B_local, H), dtype=np.float32)
    c_prev = np.zeros((B_local, H), dtype=np.float32)
    steps = []
    for t in range(T_local):
        x_t = x_np[:, t, :]
        f_t = sigmoid(x_t @ U_f + h_prev @ W_f)       # 遗忘门
        i_t = sigmoid(x_t @ U_i + h_prev @ W_i)       # 输入门
        c_tilde = np.tanh(x_t @ U_c + h_prev @ W_c)   # 候选记忆
        c_t = f_t * c_prev + i_t * c_tilde             # 更新细胞状态
        o_t = sigmoid(x_t @ U_o + h_prev @ W_o)       # 输出门
        h_t = o_t * np.tanh(c_t)                       # 隐藏状态
        steps.append(h_t)
        h_prev, c_prev = h_t, c_t
    return np.stack(steps, axis=1), h_prev, c_prev
```

**运行输出：**

```
LSTM 各时间步隐藏状态 h_t:
  t=1 '播放':    [ 0.002236, -0.038506, -0.115164]
  t=2 '周杰伦':  [ 0.023647,  0.161795, -0.019757]
  t=3 '的':      [-0.035689,  0.002108,  0.141641]
  t=4 '《稻香》': [ 0.245469,  0.329092,  0.013907]

最终细胞状态 c_T: [0.309988, 0.41179, 0.330677]
NumPy 与 PyTorch nn.LSTM 输出一致: True
```

注意 LSTM 的 $h_t$ 值明显比 RNN 小（受 sigmoid 门控的压制），但细胞状态 $c_t$ 中积累了更稳定的信息。

## 四、GRU：LSTM 的轻量替代

GRU（Gated Recurrent Unit, Cho et al. 2014）对 LSTM 做了两项简化：

1. **合并双轨为单轨**：不再区分 $ c_t $ 和 $ h_t $，只保留一个状态 $ h_t $
2. **三门变两门**：用"更新门"同时承担遗忘和输入的职责

### 4.1 计算公式

**重置门**（控制旧信息参与候选计算的程度）：

$$ r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) $$

**更新门**（控制新旧信息的混合比例）：

$$ z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z) $$

**候选状态**：

$$ \tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, \; x_t] + b_h) $$

**最终状态**：

$$ h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t $$

$z_t$ 接近 1 时保留旧信息，接近 0 时采用新信息——一个门同时管了"忘"和"记"。

### 4.2 从零实现

```python
def manual_gru_numpy(x_np, weights):
    U_r, W_r, U_z, W_z, U_h, W_h = weights
    B_local, T_local, _ = x_np.shape
    h_prev = np.zeros((B_local, H), dtype=np.float32)
    steps = []
    for t in range(T_local):
        x_t = x_np[:, t, :]
        r_t = sigmoid(x_t @ U_r + h_prev @ W_r)          # 重置门
        z_t = sigmoid(x_t @ U_z + h_prev @ W_z)          # 更新门
        h_tilde = np.tanh(x_t @ U_h + (r_t * h_prev) @ W_h)  # 候选状态
        h_t = z_t * h_prev + (1 - z_t) * h_tilde          # 最终状态
        steps.append(h_t)
        h_prev = h_t
    return np.stack(steps, axis=1), h_prev
```

**运行输出：**

```
GRU 各时间步隐藏状态 h_t:
  t=1 '播放':    [ 0.325388,  0.866400, -0.595028]
  t=2 '周杰伦':  [ 0.101285,  0.667204, -0.206921]
  t=3 '的':      [-0.286829, -0.335360, -0.429926]
  t=4 '《稻香》': [-0.239465,  0.089522, -0.304712]
```

## 五、三者对比

同一输入"播放周杰伦的《稻香》"（$E=128, H=3$），三种模型的最终隐藏状态：

```
RNN  h_T: [-0.049530,  0.121081,  0.910190]
LSTM h_T: [ 0.245469,  0.329092,  0.013907]
GRU  h_T: [-0.239465,  0.089522, -0.304712]
```

结构差异总结：

| | RNN | LSTM | GRU |
|---|---|---|---|
| 状态数 | 1（$h_t$） | 2（$h_t$ + $c_t$） | 1（$h_t$） |
| 门控 | 无 | 3 个（遗忘/输入/输出） | 2 个（重置/更新） |
| 状态更新方式 | $\tanh(Ux + Wh)$（矩阵乘法） | $f \odot c + i \odot \tilde{c}$（逐元素加法） | $z \odot h + (1-z) \odot \tilde{h}$（逐元素加法） |
| 梯度传播 | 连乘权重矩阵 → 消失/爆炸 | 连乘遗忘门值 → 可学习的保持 | 连乘更新门值 → 可学习的保持 |
| 参数量 | 最少 | 最多（约 RNN 的 4 倍） | 居中（约 RNN 的 3 倍） |
| 长距离依赖 | 差 | 好 | 好（略逊于 LSTM） |

实践中的选择：LSTM 和 GRU 效果通常接近，GRU 参数更少、训练更快，适合数据/算力有限的场景。

## 六、LSTM 变体补充

### 6.1 窥孔连接（Peephole Connections）

标准 LSTM 的门只看 $ h_{t-1}$ 和 $ x_t $，看不到细胞状态 $ c_t$。窥孔连接让门直接访问 $c$：

- 遗忘门和输入门"窥视" $c_{t-1}$
- 输出门"窥视" $c_t$

对需要精确计时/计数的任务有帮助。

### 6.2 耦合输入/遗忘门（CIFG）

令 $ i_t = 1 - f_t $，遗忘多少就写入多少，省掉一个独立门：

$$ c_t = f_t \odot c_{t-1} + (1 - f_t) \odot \tilde{c}_t $$

Greff et al. (2017) 的大规模实验表明，这种简化不损失性能，且减少了参数。同一研究还指出：遗忘门和输出激活函数是 LSTM 最关键的组件，去掉它们性能显著下降。

### 6.3 从 Type 到 Token

Word2Vec 给出的是静态词向量（Type）——"周杰伦"不管出现在哪里向量都一样。RNN/LSTM/GRU 的隐藏状态 $h_t$ 则是**动态表示**（Token）——"播放 **周杰伦**"和"我喜欢 **周杰伦**"中，同一个词因上下文不同而获得不同的表示。

Elman (1990) 的实验发现，仅通过"预测下一个词"训练 RNN，隐藏状态空间会自发地将动词/名词、有生命/无生命等范畴区分开来，说明 RNN 不只是在记忆历史，还隐式地学到了句法和语义结构。

> 后来的 ELMo (2018) 在此基础上将这种动态表示做成了通用的预训练特征，改变了 NLP 的开发范式。

## 参考文献

1. Elman, J. L. (1990). *Finding structure in time*. Cognitive Science, 14(2), 179-211.
2. Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory*. Neural Computation, 9(8), 1735-1780.
3. Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). *Learning to forget: Continual prediction with LSTM*.
4. Cho, K., et al. (2014). *Learning phrase representations using RNN encoder-decoder for statistical machine translation*. EMNLP.
5. Schuster, M., & Paliwal, K. K. (1997). *Bidirectional recurrent neural networks*. IEEE TSP.
6. Greff, K., et al. (2017). *LSTM: A search space odyssey*. IEEE TNNLS, 28(10), 2222-2232.
7. Bengio, Y., Simard, P., & Frasconi, P. (1994). *Learning long-term dependencies with gradient descent is difficult*. IEEE TNN.
