"""
FP8 Block-wise Quantization & GEMM Implementation
核心流程：
  1. act_quant:    激活值分块量化 FP32 → FP8 E4M3
  2. weight_dequant: 权重反量化 FP8 → FP32（调试/验证用）
  3. fp8_gemm:     量化矩阵乘法 C = A_q @ W_q^T * S_A * S_W
"""
from typing import Tuple
import torch
import triton
import triton.language as tl
# ============================================================
#  常量
# ============================================================
FP8_MAX = 448.0          # float8_e4m3fn 最大可表示值
FP8_DTYPE = torch.float8_e4m3fn
DEFAULT_BLOCK = 128
# ============================================================
#  激活值量化  FP32 → FP8
# ============================================================
@triton.jit
def _act_quant_kernel(
    x_ptr, y_ptr, s_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    y = (x / s).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)
def act_quant(
    x: torch.Tensor,
    block_size: int = DEFAULT_BLOCK,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """激活值分块量化。
    Args:
        x: 输入张量 (任意形状, 最后一维须被 block_size 整除, contiguous)
        block_size: 每块共享一个缩放因子的元素数
    Returns:
        (y, s) — 量化结果 FP8, 缩放因子 FP32
    """
    assert x.is_contiguous()
    assert x.size(-1) % block_size == 0
    y = torch.empty_like(x, dtype=FP8_DTYPE)
    s = x.new_empty(*x.shape[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = (triton.cdiv(x.numel(), block_size),)
    _act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s
# ============================================================
#  权重反量化  FP8 → FP32（调试 / 精度验证）
# ============================================================
@triton.jit
def _weight_dequant_kernel(
    x_ptr, s_ptr, y_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    n_blocks = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    offs = offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n_blocks + pid_n)
    tl.store(y_ptr + offs, x * s, mask=mask)
def weight_dequant(
    x: torch.Tensor,
    s: torch.Tensor,
    block_size: int = DEFAULT_BLOCK,
) -> torch.Tensor:
    """权重反量化 (M, N) → FP32。"""
    assert x.is_contiguous() and s.is_contiguous()
    assert x.ndim == 2 and s.ndim == 2
    M, N = x.shape
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    _weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y
# ============================================================
#  FP8 GEMM:  C = A_q @ W_q^T  ⊙  (S_A ⊗ S_W)
# ============================================================
_FP8_GEMM_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 128},
        num_stages=ns, num_warps=8,
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
    # 分块参数
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    A (M, K) @ B^T (K, N) → C (M, N)
    B 存储为 (N, K)，通过索引转置实现 B^T
    缩放因子布局：
      a_s: (M, K // BLOCK_K)  每 BLOCK_K 列共享一个 s
      b_s: (N // BLOCK_K, K // BLOCK_K)  与 A 对齐的分块缩放
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_k_blocks = tl.cdiv(K, BLOCK_K)
    # ---- 行列偏移 ----
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    # ---- 数据指针：A (M,K) 行主序, B (N,K) 行主序 ----
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]    # (BM, BK)
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]    # (BK, BN) — 隐式转置
    # ---- 缩放因子指针 ----
    a_s_ptrs = a_s_ptr + offs_m * num_k_blocks                # (BM,) 起始列
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_K) * num_k_blocks   # (BN,) 起始列
    # ---- 分块归约 ----
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(num_k_blocks):
        k_mask = offs_k < (K - i * BLOCK_K)
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        sa = tl.load(a_s_ptrs + i)         # (BM,)
        sb = tl.load(b_s_ptrs + i)         # (BN,)
        # dot → (BM, BN)  再乘缩放因子完成反量化
        acc += tl.dot(a, b) * sa[:, None] * sb[None, :]
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    # ---- 写回 ----
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
    """FP8 分块量化矩阵乘法  C = A_q @ W_q^T ⊙ (S_A ⊗ S_W)
    Args:
        a:   量化激活 (..., K)  FP8
        a_s: 激活缩放因子 (..., K // block_size)  FP32
        b:   量化权重 (N, K)  FP8
        b_s: 权重缩放因子 (N // block_size, K // block_size)  FP32
    Returns:
        C (..., N)  默认浮点精度
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
