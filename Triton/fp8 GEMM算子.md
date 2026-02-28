## 前置知识：浮点精度格式

| 格式 | 总位数 | 符号位 | 指数位 | 尾数位 | 最大值 | 典型用途 |
|------|--------|--------|--------|--------|--------|----------|
| FP32 | 32 | 1 | 8 | 23 | ~3.4×10³⁸ | 训练默认精度 |
| FP16 | 16 | 1 | 5 | 10 | 65504 | 混合精度训练 |
| BF16 | 16 | 1 | 8 | 7 | ~3.4×10³⁸ | 大模型训练（动态范围大） |
| FP8 (E4M3) | 8 | 1 | 4 | 3 | **448** | 推理/量化计算 |
| FP8 (E5M2) | 8 | 1 | 5 | 2 | 57344 | 梯度表示 |
| INT8 | 8 | 1 | — | 7 | 127 | 推理量化 |

- 指数位越多 → 动态范围越大（能表示的数值跨度越大）
- 尾数位越多 → 精度越高（能区分的数值越细）
- FP8 E4M3 的最大值 448，就是因为 4 位指数 + 3 位尾数的编码上限
- FP8 相比 FP16/FP32，存储和计算开销大幅降低，但精度损失需要通过量化技术来补偿

## 为什么要量化？

FP32 权重（32bit）→ FP8 权重（8bit）= 存储减少 4 倍，计算吞吐提升

### 核心公式
未量化的矩阵乘法：

$$C = A \times W$$

量化后的矩阵乘法：

$$C = (A_q \times W_q) \times (S_A \times S_W)$$

### 分块动态量化策略

不同于全局量化（整个张量共享一个缩放因子），这里采用 分块量化：每 BLOCK_SIZE（默认 128）个元素共享一个独立的缩放因子。

优势：每个块根据自身数据分布独立调整，局部精度更高。

### 量化流程

```text
输入张量 x (FP32)
    │
    ▼
┌─────────────────────────────────────┐
│ 分块：将 x 划分为 ceil(size/BLOCK_SIZE)个块 │
└─────────────────────────────────────┘
    │
    ▼  对每个块：
┌─────────────────────────────────────┐
│ 1. 求块内最大绝对值：max_val = max(|x|)│
│ 2. 计算缩放因子：s = max_val / 448.0  │
│ 3. 量化：y = x / s，然后转为 FP8      │
└─────────────────────────────────────┘
    │
    ▼
输出：y (FP8) + s (FP32)

```

448.0 的来源：FP8 E4M3 格式能表示的最大正数就是 448。将块内最大值映射到 448，确保量化后数据充分利用 FP8 的表示范围，同时不溢出。

**Triton Kernel 实现**

```python
@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    # 当前线程块的 ID，每个线程块处理一个数据块
    pid = tl.program_id(axis=0)

    # 计算当前块内元素的全局索引
    # 例如 pid=2, BLOCK_SIZE=128 → offs = [256, 257, ..., 383]
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 加载当前块的所有元素，转为 FP32 保证计算精度
    x = tl.load(x_ptr + offs).to(tl.float32)

    # 计算缩放因子：块内最大绝对值 / FP8最大值
    s = tl.max(tl.abs(x)) / 448.0

    # 量化：原始值 / 缩放因子 → 映射到 FP8 可表示范围
    y = x / s

    # 转换为 FP8 类型存储
    y = y.to(y_ptr.dtype.element_ty)

    # 写回量化结果和缩放因子
    tl.store(y_ptr + offs, y)       # 每个元素都存
    tl.store(s_ptr + pid, s)        # 每个块存一个缩放因子
```

**Kernel 启动函数**

```python
def act_quant(x: torch.Tensor, block_size: int = 128):
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0

    # 输出：与 x 同形状的 FP8 张量
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)

    # 缩放因子：每 block_size 个元素对应一个 s
    # 形状为 (..., x.size(-1) // block_size)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)

    # 网格大小 = 总元素数 / BLOCK_SIZE（即总块数）
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s
```

## FP8 GEMM 实现

目标公式

  $$C = A \times W^T \times S_A \times S_W$$

  - $A$：输入激活（已量化为 FP8）
  - $W$：权重（离线阶段已量化为 FP8）
  - $S_A$、$S_W$：对应的缩放因子（FP32）
注意这里是 $W^T$（权重转置），下文会解释原因

### Triton 分块矩阵乘法回顾

矩阵维度：

$$A_{M \times K} \times B^T_{K \times N} = C_{M \times N}$$

```python
pid_m = tl.program_id(axis=0)  # M 维度的块索引
pid_n = tl.program_id(axis=1)  # N 维度的块索引
k = tl.cdiv(K, BLOCK_SIZE_K)   # K 维度需要迭代的次数
# 当前块在 M、N 维度的元素索引
offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
```

### 指针计算与转置理解

```python
# A 矩阵指针：形状 (M, K)，行主序
# a_ptrs[i,j] = a_ptr + offs_m[i] * K + offs_k[j]
a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
# B 矩阵指针：形状 (N, K)，行主序
# b_ptrs[j,i] = b_ptr + offs_n[i] * K + offs_k[j]
b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
```

**为什么是 $W^T$?**

B 矩阵在内存中存储为(N, K)，即每行是一个输出神经元的权重向量。

但 b_ptrs 的索引维度是 (offs_k, offs_n)，加载出来的子块形状是 $(BLOCK_{SIZE_K}, BLOCK_{SIZE_N})$。这等效于对 B 做了转置：从(N, K)读取为(K, N)的视图。

循环中 b_ptrs += BLOCK_SIZE_K：移动的是 B 的第二维（K 维），说明 K 是 B 在内存中的连续维度，N 才是归约后的输出维度。因此实际计算的是 $A \times W^T$。

### 核心循环：分块归约 + 反量化

```python
for i in range(k):
    # 加载 A 的子块 (BLOCK_SIZE_M, BLOCK_SIZE_K)，带边界 mask
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)

    # 加载 B 的子块 (BLOCK_SIZE_K, BLOCK_SIZE_N)，带边界 mask
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)

    # 加载当前块的缩放因子
    a_s = tl.load(a_s_ptrs)  # 形状 (BLOCK_SIZE_M,)
    b_s = tl.load(b_s_ptrs)  # 形状 (BLOCK_SIZE_N,)

    # FP8 点积 + 反量化
    # tl.dot(a, b) → (BLOCK_SIZE_M, BLOCK_SIZE_N)，FP8 计算
    # a_s[:, None]  → (BLOCK_SIZE_M, 1)  广播到每一列
    # b_s[None, :]  → (1, BLOCK_SIZE_N)  广播到每一行
    accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]

    # 沿 K 轴移动到下一个子块
    a_ptrs += BLOCK_SIZE_K
    b_ptrs += BLOCK_SIZE_K
```

反量化的直觉理解：
  FP8 点积结果（缩放后的小数值）
       × a_s（恢复 A 的量级）
       × b_s（恢复 W 的量级）
       = 近似 FP32 的真实结果

每 $BLOCK_{SIZE_K}$ 个元素共享一个缩放因子，所以 $a_s$ 的维度是 $(BLOCK_{SIZE_M},)$（M 维度上每行一个），$b_s$ 的维度是 $(BLOCK_{SIZE_N},)$（N 维度上每列一个）。

### 结果写回

```python
# 计算输出矩阵 C 中当前子块的全局索引
offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

# C 矩阵指针：形状 (M, N)
c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]

# 边界保护
mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

# 将 FP32 累加结果写回
tl.store(c_ptrs, c, mask=mask)
```

```text
┌──────────────────────────────────────────────────┐
│                  离线阶段（权重）                   │
│  W (FP32) → 分块量化 → W_q (FP8) + S_W (FP32)    │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│                  在线阶段（推理）                   │
│                                                    │
│  1. 激活量化                                        │
│     A (FP32) → act_quant → A_q (FP8) + S_A (FP32) │
│                                                    │
│  2. FP8 GEMM                                       │
│     沿 K 轴分块循环：                                │
│       ┌─────────────────────────────────┐          │
│       │ 加载 A_q, W_q 子块 (FP8)         │          │
│       │ 加载 S_A, S_W 子块 (FP32)        │          │
│       │ dot(A_q, W_q) × S_A × S_W       │          │
│       │ 累加到 accumulator (FP32)        │          │
│       └─────────────────────────────────┘          │
│                                                    │
│  3. 写回结果 C (FP32)                               │
└──────────────────────────────────────────────────┘

```
