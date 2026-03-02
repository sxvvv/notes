"""
FP8 分块量化 & GEMM 实现（初学者友好版）

=== 什么是量化？===
把高精度数字（FP32, 32位）压缩成低精度数字（FP8, 8位）
好处：省内存、算得快
代价：精度有损失，需要用缩放因子(scale)来尽量保持精度

=== 什么是 FP8 E4M3？===
一种 8 位浮点格式：1位符号 + 4位指数 + 3位尾数
最大能表示的正数是 448.0

=== 核心公式 ===
原始矩阵乘法：C = A @ W
量化矩阵乘法：C = (A_q @ W_q^T) * S_A * S_W

其中 A_q, W_q 是 FP8 格式，S_A, S_W 是 FP32 缩放因子
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

# ============================================================
#  全局常量
# ============================================================

FP8_MAX = 448.0                     # FP8 E4M3 能表示的最大值
FP8_DTYPE = torch.float8_e4m3fn     # PyTorch 中对应的 FP8 类型
DEFAULT_BLOCK = 128                 # 默认分块大小：每128个元素共享一个缩放因子


# ============================================================
#  第一部分：激活值量化（FP32 → FP8）
# ============================================================

@triton.jit
def _act_quant_kernel(
    x_ptr,          # 输入数据指针（FP32）
    y_ptr,          # 输出数据指针（FP8）
    s_ptr,          # 缩放因子指针（FP32）
    BLOCK_SIZE: tl.constexpr,   # 每个块的元素数量
):
    """
    每个线程块处理 BLOCK_SIZE 个元素：
    1. 找到这些元素中的最大绝对值
    2. 用 最大绝对值 / 448 算出缩放因子
    3. 把每个元素除以缩放因子，压缩到 FP8 范围内
    """
    # === 第1步：确定当前线程块要处理哪些元素 ===
    # pid 就是当前线程块的编号（第0块、第1块、第2块...）
    pid = tl.program_id(0)

    # 当前块负责的元素索引
    # 比如 pid=2, BLOCK_SIZE=128 → offs = [256, 257, 258, ..., 383]
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # === 第2步：加载数据，统一转成 FP32 做计算 ===
    x = tl.load(x_ptr + offs).to(tl.float32)

    # === 第3步：计算缩放因子 ===
    # 思路：把当前块的最大绝对值映射到 FP8 最大值 448
    # 比如块内最大绝对值是 89.6，则 s = 89.6 / 448 = 0.2
    s = tl.max(tl.abs(x)) / 448.0

    # === 第4步：量化 = 原始值 / 缩放因子 ===
    # 接上例：原始值 89.6 / 0.2 = 448（刚好是 FP8 最大值）
    # 这样就把数据"压缩"到 FP8 能表示的范围了
    y = (x / s).to(y_ptr.dtype.element_ty)  # 转成 FP8 存储

    # === 第5步：写回结果 ===
    tl.store(y_ptr + offs, y)     # 每个元素都要存（FP8）
    tl.store(s_ptr + pid, s)      # 每个块只存一个缩放因子（FP32）


def act_quant(
    x: torch.Tensor,
    block_size: int = DEFAULT_BLOCK,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对激活值进行分块量化

    参数：
        x: 输入张量，FP32，必须是 contiguous 的
        block_size: 分块大小，默认128（每128个元素共享一个缩放因子）

    返回：
        y: 量化后的张量，FP8
        s: 缩放因子张量，FP32，形状为 (..., 最后一维 // block_size)

    举例：
        x 形状 (2, 512)，block_size=128
        → y 形状 (2, 512)，dtype=FP8
        → s 形状 (2, 4)，dtype=FP32  （512/128=4，每行4个缩放因子）
    """
    assert x.is_contiguous(), "输入必须是 contiguous 的"
    assert x.size(-1) % block_size == 0, \
        f"最后一维({x.size(-1)})必须能被 block_size({block_size}) 整除"

    # 创建输出张量
    y = torch.empty_like(x, dtype=FP8_DTYPE)

    # 缩放因子：最后一维缩小 block_size 倍
    s = x.new_empty(*x.shape[:-1], x.size(-1) // block_size, dtype=torch.float32)

    # 启动 kernel：总共需要 (总元素数 / block_size) 个线程块
    num_blocks = triton.cdiv(x.numel(), block_size)
    _act_quant_kernel[(num_blocks,)](x, y, s, BLOCK_SIZE=block_size)

    return y, s


# ============================================================
#  第二部分：权重反量化（FP8 → FP32，用于调试验证）
# ============================================================

@triton.jit
def _weight_dequant_kernel(
    x_ptr,      # 量化权重指针（FP8）
    s_ptr,      # 缩放因子指针（FP32）
    y_ptr,      # 输出指针（FP32）
    M, N,       # 权重矩阵的行数和列数
    BLOCK_SIZE: tl.constexpr,
):
    """反量化 = 量化值 × 缩放因子，就是量化的逆操作"""
    pid_m = tl.program_id(0)    # 行方向的块编号
    pid_n = tl.program_id(1)    # 列方向的块编号
    num_col_blocks = tl.cdiv(N, BLOCK_SIZE)

    # 当前块的行列索引
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # 加载量化数据，转成 FP32
    flat_offs = offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(x_ptr + flat_offs, mask=mask).to(tl.float32)

    # 加载这个块对应的缩放因子（每个块一个 s）
    s = tl.load(s_ptr + pid_m * num_col_blocks + pid_n)

    # 反量化：乘回缩放因子
    tl.store(y_ptr + flat_offs, x * s, mask=mask)


def weight_dequant(
    x: torch.Tensor,
    s: torch.Tensor,
    block_size: int = DEFAULT_BLOCK,
) -> torch.Tensor:
    """
    权重反量化：FP8 → FP32

    参数：
        x: 量化权重 (M, N)，FP8
        s: 缩放因子 (M // block_size, N // block_size)，FP32

    返回：
        反量化后的权重 (M, N)，FP32
    """
    assert x.is_contiguous() and s.is_contiguous()
    assert x.ndim == 2 and s.ndim == 2

    M, N = x.shape
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    _weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# ============================================================
#  第三部分：FP8 GEMM 矩阵乘法
#  计算：C = A_q @ W_q^T * S_A * S_W
# ============================================================

# Triton 自动调优配置：尝试不同的分块大小，自动选最快的
_FP8_GEMM_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 128},
        num_stages=ns,
        num_warps=8,
    )
    for bm in [16, 32, 64]
    for bn in [32, 64, 128]
    for ns in [3, 4, 5, 6]
]


@triton.autotune(configs=_FP8_GEMM_CONFIGS, key=["N", "K"])
@triton.jit
def _fp8_gemm_kernel(
    # 数据指针
    a_ptr, b_ptr, c_ptr,
    # 缩放因子指针
    a_s_ptr, b_s_ptr,
    # 矩阵维度
    M, N: tl.constexpr, K: tl.constexpr,
    # 分块大小
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    分块矩阵乘法 + 反量化

    矩阵布局：
        A:  (M, K) — 量化激活，FP8
        B:  (N, K) — 量化权重，FP8（存储为 N×K，计算时隐式转置）
        C:  (M, N) — 输出结果，FP32

    缩放因子布局：
        a_s: (M, K // BLOCK_K) — 每 BLOCK_K 列共享一个
        b_s: (N // BLOCK_K, K // BLOCK_K) — 二维分块

    计算流程（伪代码）：
        for 每个 K 方向的子块:
            加载 A 的子块 (BLOCK_M, BLOCK_K)
            加载 B 的子块 (BLOCK_K, BLOCK_N)  ← 隐式转置
            局部矩阵乘 + 乘缩放因子（反量化）
            累加到结果
    """
    pid_m = tl.program_id(0)     # C 矩阵行方向的块编号
    pid_n = tl.program_id(1)     # C 矩阵列方向的块编号
    num_k_blocks = tl.cdiv(K, BLOCK_K)

    # === 计算当前块在 M、N 维度的元素索引 ===
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # === 构造数据指针 ===
    # A: 形状 (M, K)，a_ptrs 指向子块 (offs_m, offs_k)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]

    # B: 形状 (N, K)，但我们按 (offs_k, offs_n) 索引
    #    → 等效读取 B^T 的子块 (offs_k, offs_n)
    #    这就是为什么说计算的是 A @ B^T
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

    # === 构造缩放因子指针 ===
    # a_s: 每行有 num_k_blocks 个缩放因子，起始位置 = 行号 × num_k_blocks
    a_s_ptrs = a_s_ptr + offs_m * num_k_blocks
    # b_s: 类似的二维索引
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_K) * num_k_blocks

    # === 沿 K 轴分块归约 ===
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(num_k_blocks):
        # 边界保护：K 可能不是 BLOCK_K 的整数倍
        k_remaining = K - i * BLOCK_K
        k_mask = offs_k < k_remaining

        # 加载 A 子块 (BLOCK_M, BLOCK_K) 和 B 子块 (BLOCK_K, BLOCK_N)
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

        # 加载缩放因子（第 i 个 K-block 对应第 i 个缩放因子）
        sa = tl.load(a_s_ptrs + i)     # 形状 (BLOCK_M,)
        sb = tl.load(b_s_ptrs + i)     # 形状 (BLOCK_N,)

        # 核心计算：FP8矩阵乘 × 缩放因子（反量化）
        # tl.dot(a, b)  → (BLOCK_M, BLOCK_N)
        # sa[:, None]   → (BLOCK_M, 1) 广播到每一列
        # sb[None, :]   → (1, BLOCK_N) 广播到每一行
        acc += tl.dot(a, b) * sa[:, None] * sb[None, :]

        # 移动到下一个 K-block
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    # === 写回结果到 C ===
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
) -> torch.Tensor:
    """
    FP8 量化矩阵乘法：C = A_q @ W_q^T * S_A * S_W

    参数：
        a:   量化激活 (..., K)，FP8
        a_s: 激活缩放因子 (..., K // 128)，FP32
        b:   量化权重 (N, K)，FP8
        b_s: 权重缩放因子 (N // 128, K // 128)，FP32

    返回：
        C (..., N)，默认浮点精度

    举例：
        a 形状 (4, 512)，b 形状 (256, 512)
        → C 形状 (4, 256)
    """
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()

    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)

    c = a.new_empty(*a.shape[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    _fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c


# ============================================================
#  测试代码
# ============================================================

def test_act_quant():
    """测试激活值量化和反量化的精度"""
    print("=" * 60)
    print("测试1：激活值量化")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.randn(4, 512, device="cuda", dtype=torch.float32)
    print(f"输入形状: {x.shape}, 类型: {x.dtype}")

    # 量化
    y, s = act_quant(x, block_size=128)
    print(f"量化后形状: {y.shape}, 类型: {y.dtype}")
    print(f"缩放因子形状: {s.shape}, 类型: {s.dtype}")

    # 手动反量化验证
    # 将 FP8 转回 FP32，乘以缩放因子
    y_fp32 = y.to(torch.float32)  # (4, 512)
    s_expanded = s.repeat_interleave(128, dim=-1)  # (4, 4) → (4, 512)
    x_reconstructed = y_fp32 * s_expanded

    # 计算误差
    abs_err = (x - x_reconstructed).abs()
    rel_err = abs_err / (x.abs() + 1e-8)
    print(f"绝对误差 - 均值: {abs_err.mean().item():.6f}, 最大: {abs_err.max().item():.6f}")
    print(f"相对误差 - 均值: {rel_err.mean().item():.6f}, 最大: {rel_err.max().item():.6f}")
    print()


def test_weight_dequant():
    """测试权重反量化"""
    print("=" * 60)
    print("测试2：权重反量化")
    print("=" * 60)

    torch.manual_seed(42)

    # 模拟：先量化一个权重矩阵，再反量化，看误差
    M, N = 256, 512
    w = torch.randn(M, N, device="cuda", dtype=torch.float32)
    print(f"原始权重形状: {w.shape}")

    # 手动量化（模拟离线量化过程）
    block_size = 128
    w_reshaped = w.reshape(M // block_size, block_size, N // block_size, block_size)
    w_reshaped = w_reshaped.permute(0, 2, 1, 3)  # (M//bs, N//bs, bs, bs)
    w_abs_max = w_reshaped.abs().amax(dim=(-2, -1))  # (M//bs, N//bs)
    s = w_abs_max / FP8_MAX

    # 量化
    s_expanded = s[:, :, None, None].expand_as(w_reshaped)
    w_q_float = w_reshaped / s_expanded
    w_q = w_q_float.permute(0, 2, 1, 3).reshape(M, N).to(FP8_DTYPE)

    print(f"量化权重形状: {w_q.shape}, 类型: {w_q.dtype}")
    print(f"缩放因子形状: {s.shape}")

    # 用 Triton kernel 反量化
    w_recovered = weight_dequant(w_q, s, block_size=block_size)
    abs_err = (w - w_recovered).abs()
    print(f"反量化误差 - 均值: {abs_err.mean().item():.6f}, 最大: {abs_err.max().item():.6f}")
    print()


def test_fp8_gemm():
    """测试 FP8 GEMM 的正确性：对比 FP32 矩阵乘法"""
    print("=" * 60)
    print("测试3：FP8 GEMM 正确性")
    print("=" * 60)

    torch.manual_seed(42)
    M, N, K = 64, 256, 512
    block_size = 128

    # 随机生成激活和权重
    a_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
    w_fp32 = torch.randn(N, K, device="cuda", dtype=torch.float32)

    # FP32 基准结果：C = A @ W^T
    c_ref = a_fp32 @ w_fp32.T
    print(f"A 形状: ({M}, {K}), W 形状: ({N}, {K})")
    print(f"FP32 基准结果形状: {c_ref.shape}")

    # 量化激活
    a_q, a_s = act_quant(a_fp32, block_size=block_size)

    # 量化权重（模拟离线量化）
    num_n_blocks = N // block_size
    num_k_blocks = K // block_size
    w_reshaped = w_fp32.reshape(num_n_blocks, block_size, num_k_blocks, block_size)
    w_reshaped = w_reshaped.permute(0, 2, 1, 3)
    w_abs_max = w_reshaped.abs().amax(dim=(-2, -1))
    b_s = w_abs_max / FP8_MAX
    s_expanded = b_s[:, :, None, None].expand_as(w_reshaped)
    w_q_float = w_reshaped / s_expanded
    b_q = w_q_float.permute(0, 2, 1, 3).reshape(N, K).to(FP8_DTYPE)

    print(f"量化激活: {a_q.shape} ({a_q.dtype}), 缩放因子: {a_s.shape}")
    print(f"量化权重: {b_q.shape} ({b_q.dtype}), 缩放因子: {b_s.shape}")

    # FP8 GEMM
    c_fp8 = fp8_gemm(a_q, a_s, b_q, b_s)
    print(f"FP8 GEMM 结果形状: {c_fp8.shape}")

    # 对比误差
    abs_err = (c_ref - c_fp8).abs()
    rel_err = abs_err / (c_ref.abs() + 1e-8)
    print(f"绝对误差 - 均值: {abs_err.mean().item():.4f}, 最大: {abs_err.max().item():.4f}")
    print(f"相对误差 - 均值: {rel_err.mean().item():.4f}, 最大: {rel_err.max().item():.4f}")

    # cosine similarity（越接近1越好）
    cos_sim = torch.nn.functional.cosine_similarity(
        c_ref.flatten().unsqueeze(0),
        c_fp8.flatten().unsqueeze(0),
    )
    print(f"余弦相似度: {cos_sim.item():.6f}（越接近1越好）")
    print()


def test_performance():
    """简单的性能对比：FP8 GEMM vs FP32 GEMM"""
    print("=" * 60)
    print("测试4：性能对比")
    print("=" * 60)

    import time

    M, N, K = 1024, 4096, 4096
    block_size = 128
    num_warmup = 10
    num_iters = 100

    a_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
    w_fp32 = torch.randn(N, K, device="cuda", dtype=torch.float32)

    # 量化
    a_q, a_s = act_quant(a_fp32, block_size=block_size)
    num_n_blocks = N // block_size
    num_k_blocks = K // block_size
    w_reshaped = w_fp32.reshape(num_n_blocks, block_size, num_k_blocks, block_size)
    w_reshaped = w_reshaped.permute(0, 2, 1, 3)
    w_abs_max = w_reshaped.abs().amax(dim=(-2, -1))
    b_s = w_abs_max / FP8_MAX
    s_expanded = b_s[:, :, None, None].expand_as(w_reshaped)
    w_q_float = w_reshaped / s_expanded
    b_q = w_q_float.permute(0, 2, 1, 3).reshape(N, K).to(FP8_DTYPE)

    # FP32 基准
    for _ in range(num_warmup):
        _ = a_fp32 @ w_fp32.T
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        _ = a_fp32 @ w_fp32.T
    torch.cuda.synchronize()
    fp32_time = (time.perf_counter() - start) / num_iters * 1000

    # FP8 GEMM
    for _ in range(num_warmup):
        _ = fp8_gemm(a_q, a_s, b_q, b_s)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        _ = fp8_gemm(a_q, a_s, b_q, b_s)
    torch.cuda.synchronize()
    fp8_time = (time.perf_counter() - start) / num_iters * 1000

    print(f"矩阵规模: M={M}, N={N}, K={K}")
    print(f"FP32 GEMM: {fp32_time:.3f} ms")
    print(f"FP8  GEMM: {fp8_time:.3f} ms")
    print(f"加速比: {fp32_time / fp8_time:.2f}x")
    print()


if __name__ == "__main__":
    print("需要 CUDA GPU 和 Triton 支持\n")

    test_act_quant()
    test_weight_dequant()
    test_fp8_gemm()

    # 性能测试可选（矩阵较大，首次运行会触发 autotune 编译，较慢）
    # test_performance()
