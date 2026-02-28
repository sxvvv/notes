import time
import numpy as np
import matplotlib.pyplot as plt

def matrix_multiply(A, B):
    # A: (M, K), B: (K, N)
    M, K = A.shape
    K_inner, N = B.shape
    assert K == K_inner, "维度不匹配: A 列数 {} != B 行数 {}".format(K, K_inner)
    C = np.zeros((M, N), dtype=A.dtype)
    for i in range(M):
        for j in range(N):
            # 用切片做内积，比逐元素乘加更高效
            C[i, j] = np.dot(A[i, :], B[:, j])
    return C

def matrix_multiply_blocked(A, B, BLOCK_SIZE):
    M, K = A.shape
    K_inner, N = B.shape
    assert K == K_inner
    C = np.zeros((M, N), dtype=np.float32)
    for m in range(0, M, BLOCK_SIZE):
        m_end = min(m + BLOCK_SIZE, M)
        for n in range(0, N, BLOCK_SIZE):
            n_end = min(n + BLOCK_SIZE, N)
            # 直接累加到 C 的块上，避免边界越界和多余临时数组
            C[m:m_end, n:n_end] = 0.0
            for k in range(0, K, BLOCK_SIZE):
                k_end = min(k + BLOCK_SIZE, K)
                a = A[m:m_end, k:k_end]
                b = B[k:k_end, n:n_end]
                C[m:m_end, n:n_end] += np.dot(a, b)
    return C

def benchmark_matrix_multiplication(sizes, BLOCK_SIZE):
    times_naive = []
    times_blocked = []

    for size in sizes:
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)

        # Naive matrix multiplication（用 perf_counter 更精确）
        t1 = time.perf_counter()
        c2 = matrix_multiply(a, b)
        t2 = time.perf_counter()
        times_naive.append(t2 - t1)

        # Blocked matrix multiplication
        t1 = time.perf_counter()
        c3 = matrix_multiply_blocked(a, b, BLOCK_SIZE)
        t2 = time.perf_counter()
        times_blocked.append(t2 - t1)

        diff = np.abs(c2.astype(np.float32) - c3).max()
        print("size: {}  max|diff|: {:.2e}".format(size, diff))
        assert np.allclose(c2, c3, rtol=1e-4, atol=1e-5), "结果不一致"

    return times_naive, times_blocked

if __name__ == '__main__':
    sizes = [32,64,128, 256,512]  # Different matrix sizes to benchmark
    BLOCK_SIZE = 32

    times_naive, times_blocked = benchmark_matrix_multiplication(sizes, BLOCK_SIZE)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_naive, label='Naive (matrix_multiply)', marker='o')
    plt.plot(sizes, times_blocked, label='Blocked (matrix_multiply_blocked)', marker='o')

    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Multiplication Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()