# Triton 实现 Softmax
## 一、Softmax 数学定义

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$

直接算会溢出。减去行最大值后数学等价，但 exp 输入 ≤ 0，数值安全：

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j} e^{x_j - \max(x)}}$$

## 二、并行策略

输入矩阵 shape = `(n_rows, n_cols)`。Softmax 沿行独立计算，行与行之间零依赖。
每个 Triton program（类似 CUDA block）负责若干行
每一行用一个长度为 BLOCK_SIZE 的向量一次性处理所有列

## 三、Triton 语法速查

在看 kernel 之前，先搞清楚几个核心 API：
| 语法 | 含义 | 类比 |
|------|------|------|
| `tl.program_id(axis)` | 当前 block 在第 axis 维的编号 | CUDA `blockIdx.x` |
| `tl.arange(0, N)` | 生成 `[0,1,...,N-1]` 向量 | `torch.arange` |
| `tl.load(ptrs, mask, other)` | 按指针向量批量读取，mask=False 的位置填 other | 带边界检查的批量读 |
| `tl.store(ptrs, vals, mask)` | 按指针向量批量写入 | 带边界检查的批量写 |
| `tl.max(x, axis)` | 向量归约求最大值 | `torch.max` |
| `tl.sum(x, axis)` | 向量归约求和 | `torch.sum` |
| `tl.exp(x)` | 逐元素 exp（快速近似，精度 ~1e-6） | `torch.exp` |
| `tl.range(a, b)` | 循环迭代器，用于 `for` | `range(a, b)` |
| `tl.constexpr` | 编译期常量，必须是 kernel 参数的类型标注 | C++ `constexpr` |
**关键限制**：`tl.arange` 的参数必须是 2 的幂。`for`/`while` 内不能用 `return`。
## 四、Triton 的指针寻址
Triton 没有多维张量索引，全靠指针算术。理解这一点是理解 Triton 的关键。
内存是一维的：

row 0: [a00, a01, a02, a03]    地址: base+0, base+1, base+2, base+3

row 1: [a10, a11, a12, a13]    地址: base+4, base+5, base+6, base+7

row 2: [a20, a21, a22, a23]    地址: base+8, base+9, base+10, base+11

访问第 row_idx 行的所有列:
```python
row_ptr   = base + row_idx * row_stride    # 行首地址
col_offs  = tl.arange(0, BLOCK_SIZE)       # [0, 1, 2, ..., BLOCK_SIZE-1]
ptrs      = row_ptr + col_offs             # 该行每个元素的地址向量
```

`row_stride` = 一行有多少个元素（连续存储时等于 `n_cols`），通过 `tensor.stride(0)` 获取。


## 五、Kernel 完整实现（带逐行注释）
```python
import torch
import triton
import triton.language as tl
@triton.jit
def softmax_kernel(
    input_ptr,          # 输入张量的起始地址
    output_ptr,         # 输出张量的起始地址
    n_rows,             # 总行数
    n_cols,             # 总列数
    input_row_stride,   # 输入每行的步长（元素个数）
    output_row_stride,  # 输出每行的步长
    ROWS_PER_BLOCK: tl.constexpr,  # 每个 block 处理多少行（编译期常量）
    BLOCK_SIZE: tl.constexpr,      # 列方向的向量长度（必须是 2 的幂）
):
    # ---- 1. 确定当前 block 负责哪些行 ----
    block_id = tl.program_id(0)                # 当前 block 编号
    row_start = block_id * ROWS_PER_BLOCK      # 起始行号
    # 如果起始行已超出范围，直接退出
    # （注意：return 只能在循环外使用）
    if row_start >= n_rows:
        return
    # ---- 2. 遍历当前 block 的每一行 ----
    for row_idx in tl.range(row_start, row_start + ROWS_PER_BLOCK):
        # 不能用 return，用 if 守卫跳过越界行
        if row_idx < n_rows:
            # ---- 3. 构造指针 ----
            row_ptr = input_ptr + row_idx * input_row_stride
            col_offsets = tl.arange(0, BLOCK_SIZE)     # [0, 1, ..., BLOCK_SIZE-1]
            input_ptrs = row_ptr + col_offsets          # 当前行所有元素的地址
            # ---- 4. 带 mask 加载 ----
            # BLOCK_SIZE >= n_cols，超出部分填 -inf
            # -inf 不影响 max，exp(-inf)=0 不影响 sum
            mask = col_offsets < n_cols
            row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
            # ---- 5. 数值稳定的 softmax ----
            row_max = tl.max(row, axis=0)
            row_safe = row - row_max           # 全部 <= 0，exp 不会溢出
            numerator = tl.exp(row_safe)
            denominator = tl.sum(numerator, axis=0)
            softmax_output = numerator / denominator
            # ---- 6. 写回 ----
            output_row_ptr = output_ptr + row_idx * output_row_stride
            output_ptrs = output_row_ptr + col_offsets
            tl.store(output_ptrs, softmax_output, mask=mask)
```
## 六、Host 端调用
```python
def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """x: (n_rows, n_cols) 的 float32 CUDA 张量"""
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    # BLOCK_SIZE 必须是 2 的幂（tl.arange 的要求）
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    ROWS_PER_BLOCK = 2
    # grid: 一共需要多少个 block
    # cdiv = ceil division = 向上取整除法
    grid = (triton.cdiv(n_rows, ROWS_PER_BLOCK),)
    softmax_kernel[grid](
        x, output,
        n_rows, n_cols,
        x.stride(0),        # 行步长
        output.stride(0),
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
```
kernel[grid](args) 是 Triton 的 kernel 启动语法，等价于 CUDA 的 kernel<<<grid, block>>>(args)。

## 七、执行流程图
```text
Host                                  GPU
  │                                    │
  │  分配 output, 计算 grid            │
  │  kernel[grid](...) ───────────────→│
  │                                    │  block 0: row 0~1
  │                                    │  block 1: row 2~3
  │                                    │  block 2: row 4~5
  │                                    │  ...
  │                                    │  每个 block 内部:
  │                                    │    for 每行:
  │                                    │      load → max → sub → exp → sum → div → store
  │  等待完成 ←────────────────────────│
  │  读取 output                       │
```