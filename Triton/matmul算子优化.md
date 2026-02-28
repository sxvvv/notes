# 分组矩阵乘法优化

## 基础分块后还剩什么问题？

分块解决了块内的数据复用，但块间的调度顺序仍是朴素的行主序：

```text
pid:   0     1     2     3     4     5     6     7     8
映射: (0,0) (0,1) (0,2) (0,3) (0,4) (0,5) (0,6) (0,7) (0,8)
       ──────────── 全部在第 0 行 ────────────────────────
pid:   9     10    11    ...
映射: (1,0) (1,1) (1,2) ...
       ──────── 第 1 行 ────────
```

问题出在：GPU 有几十个 SM（流多处理器），同一时刻可能有 pid=0, 1, 2, ..., 7 同时执行。它们都在第 0 行，共享 A 的第 0 行子块，但 B 的列子块分别是第 0~7 列，在内存中跨度很大。

```text
SM 0 处理 pid=0: 需要 A[row0] + B[col0]
SM 1 处理 pid=1: 需要 A[row0] + B[col1]
SM 2 处理 pid=2: 需要 A[row0] + B[col2]
...
SM 7 处理 pid=7: 需要 A[row0] + B[col7]  ← B 的访问范围横跨整个 N 轴！
```

L2 cache 容量有限，当 N 很大时，B[col0] 还没等到被 pid=9 复用，就已经被 B[col7] 挤出缓存了。

## 分组调度如何解决块间缓存问题 

不改变计算，只改变 pid → (pid_m, pid_n) 的映射关系，让同时执行的 block 访问的数据在内存中更紧凑。

做法：将 GROUP_SIZE_M 个相邻行的 block 编为一组，组内按列主序排列（先走 M 再走 N）。

分组前（行主序）：

```text
pid: 0    1    2    3    4    5    6    7
     (0,0)(0,1)(0,2)(0,3)(1,0)(1,1)(1,2)(1,3)
     
同时执行的 pid 0~3 → B 列跨度 = 4 列（全部 N 轴）
```

分组后（组内列主序）：

```text
pid: 0    1    2    3    4    5    6    7
     (0,0)(1,0)(0,1)(1,1)(0,2)(1,2)(0,3)(1,3)
      ├─组0──┤  ├─组0──┤  ├─组0──┤  ├─组0──┤
      
同时执行的 pid 0~3 → B 列跨度 = 2 列，A 行跨度 = 2 行
```

实现

```python
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # ======== 分组调度（唯一与基础版不同的部分）========
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ======== 以下与基础分块版完全一致 ========
    
    # 构造 A、B 子块的指针
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # K 轴规约循环
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    # 写回结果
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

```

## autotune自动搜索最优参数

```python
def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': bm, 'BLOCK_SIZE_N': bn, 
                       'BLOCK_SIZE_K': bk, 'GROUP_SIZE_M': gm},
                      num_warps=nw, num_stages=ns)
        for bm in [16, 32, 64, 128, 256]
        for bn in [16, 32, 64, 128, 256]
        for bk in [16, 32, 64]
        for gm in [4, 8]
        for nw in [2, 4, 8]
        for ns in [3, 4, 5]
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],  # 这些值变化时重新搜索
)
@triton.jit
def matmul_kernel(...):
    ...  # kernel 代码不变

```