import torch
import triton
import triton.language as tl


# ============================================================
# 第一部分：Triton Kernel 定义
# ============================================================
@triton.jit
def softmax_kernel(
    # --- 指针参数 ---
    input_ptr,          # 输入矩阵的起始指针
    output_ptr,         # 输出矩阵的起始指针
    # --- 矩阵形状 ---
    n_rows,             # 矩阵行数
    n_cols,             # 矩阵列数
    # --- 步幅信息 ---
    input_row_stride,   # 输入矩阵的行步幅（相邻两行首元素的地址差）
    output_row_stride,  # 输出矩阵的行步幅
    # --- 编译期常量 ---
    ROWS_PER_BLOCK: tl.constexpr,  # 每个 block 处理的行数
    BLOCK_SIZE: tl.constexpr,      # 列方向的块大小（≥ n_cols 的最小 2 的幂）
):
    """
    Softmax kernel：每个 block 处理 ROWS_PER_BLOCK 行
    
    算法步骤（对每一行）：
        1. 加载一整行到 SRAM
        2. 减去行最大值（数值稳定性）
        3. 计算 exp
        4. 除以 sum(exp)
        5. 写回结果
    """

    # ----------------------------------------------------------
    # Step 0: 确定当前 block 负责哪些行
    # ----------------------------------------------------------
    # tl.program_id(0) 返回当前 block 在 grid 中的编号（从 0 开始）
    block_id = tl.program_id(0)
    row_start = block_id * ROWS_PER_BLOCK  # 当前 block 的起始行号

    # 提前退出：如果起始行号已超出矩阵范围
    if row_start >= n_rows:
        return

    # ----------------------------------------------------------
    # Step 1: 遍历当前 block 负责的每一行
    # ----------------------------------------------------------
    for row_idx in tl.range(row_start, row_start + ROWS_PER_BLOCK):

        # 边界检查：最后一个 block 可能不足 ROWS_PER_BLOCK 行
        if row_idx < n_rows:
            # ------------------------------------------------------
            # Step 2: 计算当前行的指针
            # ------------------------------------------------------
            # 当前行首元素地址 = 起始指针 + 行号 × 行步幅
            row_ptr = input_ptr + row_idx * input_row_stride

            # 生成列偏移: [0, 1, 2, ..., BLOCK_SIZE-1]
            col_offsets = tl.arange(0, BLOCK_SIZE)

            # 当前行每个元素的指针
            input_ptrs = row_ptr + col_offsets

            # 掩码：BLOCK_SIZE 可能 > n_cols，超出部分不能访问
            mask = col_offsets < n_cols

            # ------------------------------------------------------
            # Step 3: 加载数据到 SRAM（片上高速存储）
            # ------------------------------------------------------
            # other=-inf: 超出部分填 -inf，不影响 max 和 softmax 计算
            row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

            # ------------------------------------------------------
            # Step 4: 计算 Softmax（数值稳定版本）
            # ------------------------------------------------------
            # 4a. 减去最大值，防止 exp 溢出
            #     softmax(x) = softmax(x - max(x))，数学上等价
            row_max = tl.max(row, axis=0)
            row_safe = row - row_max

            # 4b. 计算 exp（Triton 使用快速近似 exp，类似 CUDA 的 __expf）
            numerator = tl.exp(row_safe)

            # 4c. 求和
            denominator = tl.sum(numerator, axis=0)

            # 4d. 归一化
            softmax_output = numerator / denominator

            # ------------------------------------------------------
            # Step 5: 写回 DRAM（全局显存）
            # ------------------------------------------------------
            output_row_ptr = output_ptr + row_idx * output_row_stride
            output_ptrs = output_row_ptr + col_offsets
            tl.store(output_ptrs, softmax_output, mask=mask)


# ============================================================
# 第二部分：Host 端调用逻辑
# ============================================================
def triton_softmax(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    使用 Triton kernel 计算 softmax（沿最后一维）
    
    Args:
        input_tensor: shape (n_rows, n_cols) 的 CUDA 张量
    Returns:
        output_tensor: 与输入同形状的 softmax 结果
    """
    # --- 基本检查 ---
    assert input_tensor.is_cuda, "输入必须在 GPU 上"
    assert input_tensor.dim() == 2, "仅支持二维矩阵"

    n_rows, n_cols = input_tensor.shape

    # --- 分配输出张量 ---
    output_tensor = torch.empty_like(input_tensor)

    # --- 计算 BLOCK_SIZE ---
    # 必须是 2 的幂（Triton 的 tl.arange 要求）
    # 且 ≥ n_cols，确保一个 block 能处理一整行
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # --- 每个 block 处理的行数 ---
    ROWS_PER_BLOCK = 2

    # --- 定义 grid（block 的数量）---
    # 总行数 / 每 block 处理行数，向上取整
    grid = (triton.cdiv(n_rows, ROWS_PER_BLOCK),)

    # --- 启动 kernel ---
    softmax_kernel[grid](
        input_tensor,
        output_tensor,
        n_rows,
        n_cols,
        input_tensor.stride(0),   # 行步幅
        output_tensor.stride(0),
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output_tensor


# ============================================================
# 第三部分：测试与验证
# ============================================================
if __name__ == "__main__":
    # 1. 构造测试数据
    input_tensor = torch.randn(1000, 512, device='cuda')

    # 2. Triton 计算
    triton_output = triton_softmax(input_tensor)

    # 3. PyTorch 参考结果
    torch_output = torch.softmax(input_tensor, dim=1)

    # 4. 对比验证
    is_close = torch.allclose(triton_output, torch_output, atol=1e-6)
    print(f"结果是否一致: {is_close}")

    if not is_close:
        max_diff = (triton_output - torch_output).abs().max()
        print(f"最大误差: {max_diff:.2e}")
