"""
Triton 分组矩阵乘法 
===========================================
功能：计算 C = A @ B
  - A: (M, K)
  - B: (K, N)
  - C: (M, N)

优化思路（两层，递进关系）：
  1. 分块：把大矩阵切成小块，每个 GPU block 算一个小块 → 并行 + 块内复用
  2. 分组：改变 block 的调度顺序，让相邻 block 访问相邻数据 → 提高缓存命中率
"""

import torch
import triton
import triton.language as tl


# ╔══════════════════════════════════════════════════════════════╗
# ║  第一部分：Autotune 参数配置                                  ║
# ║  作用：告诉 Triton "请在这些参数组合里自动找最快的那个"          ║
# ╚══════════════════════════════════════════════════════════════╝

def get_autotune_configs():
    """
    返回一组候选参数，Triton 会逐个尝试，选出最快的。
    
    参数含义：
      BLOCK_SIZE_M/N/K : 每个 block 负责的子矩阵大小
      GROUP_SIZE_M     : 分组大小（几行 block 编为一组）
      num_warps        : 每个 block 内的 warp 数（影响并行度）
      num_stages       : 流水线深度（load 和 compute 重叠程度）
    """
    return [
        # ---- 小块配置（适合小矩阵）----
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4},
            num_warps=4, num_stages=3,
        ),
        # ---- 中块配置（通用）----
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_warps=4, num_stages=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_warps=8, num_stages=3,
        ),
        # ---- 大块配置（适合大矩阵）----
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_warps=8, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_warps=8, num_stages=4,
        ),
    ]


# ╔══════════════════════════════════════════════════════════════╗
# ║  第二部分：Triton Kernel（GPU 上实际执行的代码）               ║
# ╚══════════════════════════════════════════════════════════════╝

@triton.autotune(configs=get_autotune_configs(), key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    # --- 数据指针 ---
    a_ptr, b_ptr, c_ptr,
    # --- 矩阵维度 ---
    M, N, K,
    # --- 步长（每跨一行/一列要跳过多少个元素）---
    stride_am, stride_ak,   # A 的行步长、列步长
    stride_bk, stride_bn,   # B 的行步长、列步长
    stride_cm, stride_cn,   # C 的行步长、列步长
    # --- 编译期常量（由 autotune 填入）---
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    每个 GPU block 执行这个函数一次，负责计算 C 中一个 (BLOCK_M × BLOCK_N) 的子块。
    """

    # ================================================================
    # 步骤 1：分组调度 —— 确定当前 block 负责 C 的哪个子块
    # ================================================================
    #
    # 目标：从一维的 pid 算出二维坐标 (pid_m, pid_n)
    #
    # 朴素方式（行主序）：
    #   pid_m = pid // num_pid_n
    #   pid_n = pid % num_pid_n
    #   问题：同时运行的 block 访问的 B 列跨度太大，缓存不友好
    #
    # 分组方式：
    #   把 GROUP_SIZE_M 行的 block 编为一组，组内先走 M 再走 N
    #   效果：同时运行的 block 访问的数据更紧凑，缓存命中率↑
    #
    pid = tl.program_id(axis=0)  # 当前 block 的全局编号

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)  # M 轴有几个 block
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  # N 轴有几个 block

    # 每组有多少个 block
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # 当前 block 属于第几组
    group_id = pid // num_pid_in_group

    # 这个组在 M 轴从第几行开始
    first_pid_m = group_id * GROUP_SIZE_M

    # 这个组实际有几行（最后一组可能不足 GROUP_SIZE_M）
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    # 最终的二维坐标
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ================================================================
    # 步骤 2：构造指针 —— 定位 A 和 B 的起始子块
    # ================================================================
    #
    # A 子块位置：第 pid_m 行，K 轴从 0 开始
    # B 子块位置：第 pid_n 列，K 轴从 0 开始
    #
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A 子块指针：shape = (BLOCK_M, BLOCK_K)
    #   offs_am[:, None] 是列向量 (BLOCK_M, 1)
    #   offs_k[None, :]  是行向量 (1, BLOCK_K)
    #   广播相加 → (BLOCK_M, BLOCK_K) 的 2D 偏移
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak

    # B 子块指针：shape = (BLOCK_K, BLOCK_N)
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # ================================================================
    # 步骤 3：K 轴循环累加
    # ================================================================
    #
    # 每轮循环：
    #   1. 加载 A 的一个 (BLOCK_M, BLOCK_K) 子块
    #   2. 加载 B 的一个 (BLOCK_K, BLOCK_N) 子块
    #   3. 做矩阵乘法，累加到 accumulator
    #   4. 指针沿 K 轴滑动 BLOCK_K 步
    #
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 边界检查：K 轴末尾可能不足 BLOCK_K
        k_mask = offs_k < (K - k * BLOCK_SIZE_K)

        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

        accumulator += tl.dot(a, b)

        # 指针沿 K 轴滑动
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 累加用 float32 保精度，最后转 float16
    c = accumulator.to(tl.float16)

    # ================================================================
    # 步骤 4：写回结果
    # ================================================================
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


# ╔══════════════════════════════════════════════════════════════╗
# ║  第三部分：Python 调用接口                                    ║
# ╚══════════════════════════════════════════════════════════════╝

def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    调用 Triton kernel 计算 C = A @ B

    参数：
        a: (M, K) float16 CUDA 张量
        b: (K, N) float16 CUDA 张量
    返回：
        c: (M, N) float16 CUDA 张量
    """
    # --- 输入检查 ---
    assert a.is_cuda and b.is_cuda, "需要 CUDA 张量"
    assert a.dim() == 2 and b.dim() == 2, "需要 2D 矩阵"
    assert a.shape[1] == b.shape[0], f"K 维度不匹配: {a.shape[1]} vs {b.shape[0]}"

    M, K = a.shape
    _, N = b.shape

    # 统一转 float16
    if a.dtype != torch.float16:
        a = a.half()
    if b.dtype != torch.float16:
        b = b.half()

    # 确保连续存储（stride 才有意义）
    a = a.contiguous()
    b = b.contiguous()

    # 分配输出
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # 计算 grid：总共启动多少个 block
    # （具体值由 autotune 选出的 BLOCK_SIZE_M/N 决定）
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    # 启动 kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# ╔══════════════════════════════════════════════════════════════╗
# ║  第四部分：测试                                               ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"

    # ---------- 正确性测试 ----------
    print("=" * 50)
    print("正确性测试")
    print("=" * 50)

    test_shapes = [
        (64, 64, 64),
        (128, 256, 192),
        (1024, 512, 256),
    ]

    for M, N, K in test_shapes:
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)

        c_triton = triton_matmul(a, b)
        c_ref = torch.matmul(a.float(), b.float()).half()

        max_diff = (c_triton.float() - c_ref.float()).abs().max().item()
        ok = max_diff < 0.05  # float16 精度下允许的误差

        print(f"  ({M:4d}, {K:4d}) @ ({K:4d}, {N:4d})  "
              f"{'✓ 通过' if ok else '✗ 失败'}  max|diff|={max_diff:.2e}")

    # ---------- 速度测试 ----------
    print()
    print("=" * 50)
    print("速度测试 (4096 × 4096 × 4096)")
    print("=" * 50)

    M, N, K = 4096, 4096, 4096
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N, device=device, dtype=torch.float16)

    # 预热（让 autotune 完成搜索）
    for _ in range(3):
        triton_matmul(a, b)
    torch.cuda.synchronize()

    # 计时：Triton
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(20):
        triton_matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / 20

    # 计时：PyTorch
    start.record()
    for _ in range(20):
        torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / 20

    print(f"  Triton : {triton_ms:.3f} ms")
    print(f"  PyTorch: {torch_ms:.3f} ms")
    print(f"  比率   : {torch_ms / triton_ms:.2f}x")
