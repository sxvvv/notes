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
    """
    Triton 核函数：完成单个 BLOCK_SIZE×BLOCK_SIZE 块的反量化
    Args:
        x_ptr: FP8 权重矩阵的显存指针
        s_ptr: 缩放因子矩阵的显存指针
        y_ptr: 输出 FP32 矩阵的显存指针
        M: 权重矩阵的行数
        N: 权重矩阵的列数
        BLOCK_SIZE: 分块大小（编译期常量）
    """
    # 1. 获取当前线程块的行/列分块 ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # 2. 计算列方向的总块数（向上取整）
    n_blocks = tl.cdiv(N, BLOCK_SIZE)
    # 3. 生成当前块内的全局行/列索引
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 4. 生成掩码：过滤超出矩阵维度的越界元素
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # 5. 计算显存绝对偏移（2D→1D，行优先）
    offs = offs_m[:, None] * N + offs_n[None, :]
    # 6. 加载 FP8 权重并转换为 FP32
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32) 
    # 7. 加载当前块对应的缩放因子（单个标量）
    s = tl.load(s_ptr + pid_m * n_blocks + pid_n)
    # 8. 执行反量化计算并存储结果
    tl.store(y_ptr + offs, x * s, mask=mask)
def weight_dequant(
    x: torch.Tensor,
    s: torch.Tensor,
    block_size: int = DEFAULT_BLOCK,
) -> torch.Tensor:
    """权重反量化 (M, N) → FP32。"""
    # 1. 检查输入张量是否连续（Triton 要求显存连续，否则指针访问会出错）
    assert x.is_contiguous() and s.is_contiguous()
    # 2. 检查张量维度：x 是 2D 权重矩阵，s 是 2D 缩放因子矩阵
    assert x.ndim == 2 and s.ndim == 2
    # 3. 获取权重矩阵维度
    M, N = x.shape
    # 4. 创建输出张量（FP32 精度，和 x 同设备）
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    # 5. 定义核函数的启动网格（grid）：决定多少个核函数实例并行运行
    # triton.cdiv(M, block_size)：行方向的块数；triton.cdiv(N, block_size)：列方向的块数
    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    # 6. 启动 Triton 核函数
    _weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    # 7. 返回反量化后的 FP32 矩阵
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
    pid_m = tl.program_id(0)  # 行方向分块 ID（对应 C 的 M 维度）
    pid_n = tl.program_id(1)  # 列方向分块 ID（对应 C 的 N 维度）
    num_k_blocks = tl.cdiv(K, BLOCK_K)  # K 维度的总块数（向上取整）
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
        # 9.1 生成 K 维度掩码（过滤最后一块的越界元素）
        k_mask = offs_k < (K - i * BLOCK_K)
        
        # 9.2 加载 A/B 的 FP8 数据（带掩码）
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        
        # 9.3 加载对应分块的缩放因子
        sa = tl.load(a_s_ptrs + i)         # (BM,)
        sb = tl.load(b_s_ptrs + i)         # (BN,)
        
        # 9.4 核心运算：矩阵乘法 + 反量化（乘缩放因子）
        acc += tl.dot(a, b) * sa[:, None] * sb[None, :]
        
        # 9.5 指针偏移：处理下一个 K 分块
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
