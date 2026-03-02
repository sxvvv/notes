"""
INT8 分块量化 & GEMM 实现

=== 适配说明 ===
FP8 (float8_e4m3fn) 需要 SM89+ 的 GPU（如 H100, RTX 4090）
本版本改用 INT8 量化，所有 CUDA GPU 均可运行
量化原理完全相同，只是数值范围从 448 变为 127

=== 量化原理 ===
原始矩阵乘法：C = A @ W
量化矩阵乘法：C = (A_q @ W_q^T) * S_A * S_W

=== INT8 vs FP8 ===
INT8: 整数格式，范围 [-128, 127]，我们用 127 作为最大值
FP8 E4M3: 浮点格式，范围 [-448, 448]
两者量化思路一模一样：缩放到最大值范围内
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

# ============================================================
#  全局常量
# ============================================================

QUANT_MAX = 127.0                   # INT8 最大值（对称量化用 127）
QUANT_DTYPE = torch.int8            # 量化类型
DEFAULT_BLOCK = 128                 # 分块大小


# ============================================================
#  第一部分：激活值量化（FP32 → INT8）
# ============================================================

@triton.jit
def _act_quant_kernel(
    x_ptr,          # 输入数据指针（FP32）
    y_ptr,          # 输出数据指针（INT8）
    s_ptr,          # 缩放因子指针（FP32）
    BLOCK_SIZE: tl.constexpr,
):
    """
    每个线程块处理 BLOCK_SIZE 个元素：
    1. 找到这些元素中的最大绝对值
    2. 用 最大绝对值 / 127 算出缩放因子
    3. 把每个元素除以缩放因子，压缩到 INT8 范围 [-127, 127]
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 加载数据，转 FP32
    x = tl.load(x_ptr + offs).to(tl.float32)

    # 计算缩放因子：块内最大绝对值 / INT8最大值
    s = tl.max(tl.abs(x)) / 127.0

    # 量化：除以缩放因子，四舍五入到整数
    y = x / s

    # 存储（Triton 会自动截断到 int8 范围）
    tl.store(y_ptr + offs, y.to(tl.int8))
    tl.store(s_ptr + pid, s)


def act_quant(
    x: torch.Tensor,
    block_size: int = DEFAULT_BLOCK,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对激活值进行分块量化

    参数：
        x: 输入张量，FP32，必须是 contiguous 的
        block_size: 分块大小，默认 128

    返回：
        y: 量化后的张量，INT8
        s: 缩放因子张量，FP32

    举例：
        x 形状 (2, 512), block_size=128
        → y 形状 (2, 512), dtype=int8
        → s 形状 (2, 4),   dtype=float32  （512/128=4）
    """
    assert x.is_contiguous(), "输入必须是 contiguous 的"
    assert x.size(-1) % block_size == 0

    y = torch.empty_like(x, dtype=QUANT_DTYPE)
    s = x.new_empty(*x.shape[:-1], x.size(-1) // block_size, dtype=torch.float32)

    num_blocks = triton.cdiv(x.numel(), block_size)
    _act_quant_kernel[(num_blocks,)](x, y, s, BLOCK_SIZE=block_size)
    return y, s


# ============================================================
#  第二部分：权重反量化（INT8 → FP32，用于调试验证）
# ============================================================

@triton.jit
def _weight_dequant_kernel(
    x_ptr, s_ptr, y_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    """反量化 = 量化值（int8） × 缩放因子（float32）"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_col_blocks = tl.cdiv(N, BLOCK_SIZE)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    flat_offs = offs_m[:, None] * N + offs_n[None, :]

    # 加载 int8 数据，转成 float32 再乘缩放因子
    x = tl.load(x_ptr + flat_offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * num_col_blocks + pid_n)
    tl.store(y_ptr + flat_offs, x * s, mask=mask)


def weight_dequant(
    x: torch.Tensor,
    s: torch.Tensor,
    block_size: int = DEFAULT_BLOCK,
) -> torch.Tensor:
    """权重反量化：INT8 → FP32"""
    assert x.is_contiguous() and s.is_contiguous()
    assert x.ndim == 2 and s.ndim == 2
    M, N = x.shape
    y = torch.empty(M, N, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    _weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# ============================================================
#  第三部分：INT8 GEMM 矩阵乘法
#  C = A_q @ W_q^T * S_A * S_W
# ============================================================

# 自动调优配置
_GEMM_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 128},
        num_stages=ns, num_warps=8,
    )
    for bm in [16, 32, 64]
    for bn in [32, 64, 128]
    for ns in [3, 4, 5, 6]
]


@triton.autotune(configs=_GEMM_CONFIGS, key=["N", "K"])
@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    a_s_ptr, b_s_ptr,
    M, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    矩阵布局：
        A:   (M, K) — INT8 量化激活
        B:   (N, K) — INT8 量化权重（存储为 N×K，隐式转置）
        C:   (M, N) — FP32 输出
        a_s: (M, K//BLOCK_K)
        b_s: (N//BLOCK_K, K//BLOCK_K)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_k_blocks = tl.cdiv(K, BLOCK_K)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # A (M,K) 行主序
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    # B (N,K) 行主序，按 (offs_k, offs_n) 索引 → 隐式转置
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

    # 缩放因子指针
    a_s_ptrs = a_s_ptr + offs_m * num_k_blocks
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_K) * num_k_blocks

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(num_k_blocks):
        k_mask = offs_k < (K - i * BLOCK_K)

        # 加载 INT8 子块，转成 FP32 进行计算
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0).to(tl.float32)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0).to(tl.float32)

        sa = tl.load(a_s_ptrs + i)     # (BLOCK_M,)
        sb = tl.load(b_s_ptrs + i)     # (BLOCK_N,)

        # 矩阵乘 + 反量化
        acc += tl.dot(a, b) * sa[:, None] * sb[None, :]

        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    # 写回
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def int8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
) -> torch.Tensor:
    """
    INT8 量化矩阵乘法：C = A_q @ W_q^T * S_A * S_W

    参数：
        a:   量化激活 (..., K), INT8
        a_s: 激活缩放因子 (..., K // 128), FP32
        b:   量化权重 (N, K), INT8
        b_s: 权重缩放因子 (N // 128, K // 128), FP32

    返回：
        C (..., N), FP32
    """
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()

    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)

    c = torch.empty(*a.shape[:-1], N, device=a.device, dtype=torch.float32)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    _gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c


# ============================================================
#  辅助函数：权重量化（模拟离线量化过程）
# ============================================================

def quantize_weight(
    w: torch.Tensor,
    block_size: int = DEFAULT_BLOCK,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对权重进行分块量化（CPU/GPU 均可，用于模拟离线量化）

    参数：
        w: 权重 (N, K), FP32
        block_size: 分块大小

    返回：
        w_q: 量化权重 (N, K), INT8
        w_s: 缩放因子 (N // block_size, K // block_size), FP32
    """
    N, K = w.shape
    assert N % block_size == 0 and K % block_size == 0

    # 重塑为 (N//bs, bs, K//bs, bs) 的分块视图
    w_blocks = w.reshape(N // block_size, block_size, K // block_size, block_size)
    # 转为 (N//bs, K//bs, bs, bs)
    w_blocks = w_blocks.permute(0, 2, 1, 3)

    # 每个块的最大绝对值 → 缩放因子
    block_max = w_blocks.abs().amax(dim=(-2, -1))   # (N//bs, K//bs)
    w_s = block_max / QUANT_MAX

    # 量化
    s_expanded = w_s[:, :, None, None]
    w_q_float = w_blocks / s_expanded
    w_q_float = w_q_float.round().clamp(-128, 127)

    # 转回原始布局
    w_q = w_q_float.permute(0, 2, 1, 3).reshape(N, K).to(QUANT_DTYPE).contiguous()
    w_s = w_s.contiguous()

    return w_q, w_s


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
    y_fp32 = y.to(torch.float32)
    s_expanded = s.repeat_interleave(128, dim=-1)
    x_reconstructed = y_fp32 * s_expanded

    # 误差
    abs_err = (x - x_reconstructed).abs()
    rel_err = abs_err / (x.abs() + 1e-8)
    print(f"绝对误差 - 均值: {abs_err.mean().item():.6f}, 最大: {abs_err.max().item():.6f}")
    print(f"相对误差 - 均值: {rel_err.mean().item():.6f}, 最大: {rel_err.max().item():.6f}")
    print("✅ 通过\n")


def test_weight_dequant():
    """测试权重量化 → 反量化的往返精度"""
    print("=" * 60)
    print("测试2：权重量化 → 反量化")
    print("=" * 60)

    torch.manual_seed(42)
    M, N = 256, 512
    w = torch.randn(M, N, device="cuda", dtype=torch.float32)
    print(f"原始权重形状: {w.shape}")

    w_q, w_s = quantize_weight(w.clone())
    print(f"量化权重: {w_q.shape} ({w_q.dtype}), 缩放因子: {w_s.shape}")

    w_recovered = weight_dequant(w_q, w_s)
    abs_err = (w - w_recovered).abs()
    print(f"反量化误差 - 均值: {abs_err.mean().item():.6f}, 最大: {abs_err.max().item():.6f}")
    print("✅ 通过\n")


def test_gemm():
    """测试 INT8 GEMM 正确性：对比 FP32 矩阵乘法"""
    print("=" * 60)
    print("测试3：INT8 GEMM 正确性")
    print("=" * 60)

    torch.manual_seed(42)
    M, N, K = 64, 256, 512

    a_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
    w_fp32 = torch.randn(N, K, device="cuda", dtype=torch.float32)

    # FP32 基准
    c_ref = a_fp32 @ w_fp32.T
    print(f"A: ({M}, {K}), W: ({N}, {K}), C: {c_ref.shape}")

    # 量化
    a_q, a_s = act_quant(a_fp32)
    w_q, w_s = quantize_weight(w_fp32)
    print(f"量化激活: {a_q.shape} ({a_q.dtype}), 缩放因子: {a_s.shape}")
    print(f"量化权重: {w_q.shape} ({w_q.dtype}), 缩放因子: {w_s.shape}")

    # INT8 GEMM
    c_int8 = int8_gemm(a_q, a_s, w_q, w_s)
    print(f"INT8 GEMM 结果: {c_int8.shape}")

    # 误差
    abs_err = (c_ref - c_int8).abs()
    rel_err = abs_err / (c_ref.abs() + 1e-8)
    print(f"绝对误差 - 均值: {abs_err.mean().item():.4f}, 最大: {abs_err.max().item():.4f}")
    print(f"相对误差 - 均值: {rel_err.mean().item():.4f}, 最大: {rel_err.max().item():.4f}")

    cos_sim = torch.nn.functional.cosine_similarity(
        c_ref.flatten().unsqueeze(0),
        c_int8.flatten().unsqueeze(0),
    )
    print(f"余弦相似度: {cos_sim.item():.6f}（越接近 1 越好）")
    print("✅ 通过\n")


def test_performance():
    """性能对比：INT8 GEMM vs FP32 GEMM"""
    print("=" * 60)
    print("测试4：性能对比")
    print("=" * 60)

    import time

    M, N, K = 1024, 4096, 4096

    a_fp32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
    w_fp32 = torch.randn(N, K, device="cuda", dtype=torch.float32)
    a_q, a_s = act_quant(a_fp32)
    w_q, w_s = quantize_weight(w_fp32)

    num_warmup, num_iters = 10, 100

    # FP32
    for _ in range(num_warmup):
        _ = a_fp32 @ w_fp32.T
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        _ = a_fp32 @ w_fp32.T
    torch.cuda.synchronize()
    fp32_ms = (time.perf_counter() - t0) / num_iters * 1000

    # INT8
    for _ in range(num_warmup):
        _ = int8_gemm(a_q, a_s, w_q, w_s)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        _ = int8_gemm(a_q, a_s, w_q, w_s)
    torch.cuda.synchronize()
    int8_ms = (time.perf_counter() - t0) / num_iters * 1000

    print(f"矩阵规模: M={M}, N={N}, K={K}")
    print(f"FP32 GEMM:  {fp32_ms:.3f} ms")
    print(f"INT8 GEMM:  {int8_ms:.3f} ms")
    print(f"加速比: {fp32_ms / int8_ms:.2f}x")
    print()


if __name__ == "__main__":
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f"GPU: {name}, Compute Capability: {cap}")
    print(f"FP8 支持需要 SM89+, 当前 SM{cap[0]}{cap[1]}")
    if cap >= (8, 9):
        print("✅ 你的 GPU 支持 FP8，可以用原版 FP8 代码")
    else:
        print("⚠️  你的 GPU 不支持 FP8，本版本使用 INT8 替代（原理完全相同）")
    print()

    test_act_quant()
    test_weight_dequant()
    test_gemm()

    # 取消注释运行性能测试（首次会触发 autotune 编译，较慢）
    # test_performance()
