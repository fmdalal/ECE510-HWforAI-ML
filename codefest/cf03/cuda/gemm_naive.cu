%%writefile gemm_naive.cu

// matmul_naive.cu
// Naive FP32 matrix multiply: C = A * B, row-major, N x N (default N=1024).
// One CUDA thread per output element C[row, col]; 8x8 thread blocks.
//
// Build: nvcc -O3 -std=c++14 -arch=sm_70 matmul_naive.cu -o matmul_naive
// Run:   ./matmul_naive            (N=1024, 20 timed iterations)
//        ./matmul_naive 2048 50    (optional overrides)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",              \
                         cudaGetErrorName(_e), __FILE__, __LINE__,            \
                         cudaGetErrorString(_e));                             \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// ------------------------------ kernel --------------------------------------
// One thread computes one C[row, col]. Accumulates in a register; issues
// 2*N global loads per output element.
__global__
void matmul_naive_kernel(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < N; ++k) {
        acc += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

// ------------------------------ host ----------------------------------------
static void init_matrix(float* M, int N, unsigned seed) {
    unsigned s = seed;
    for (int i = 0; i < N * N; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;      // xorshift32
        M[i] = ((s & 0xFFFF) / 32768.0f) - 1.0f;      // -> [-1, 1)
    }
}

// CPU reference on a BS x BS sub-block of C anchored at (r0, c0).
static void cpu_reference_block(const float* A, const float* B, float* Cref,
                                int N, int r0, int c0, int bs) {
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < bs; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < N; ++k) {
                acc += A[(r0 + i) * N + k] * B[k * N + (c0 + j)];
            }
            Cref[i * bs + j] = acc;
        }
    }
}

static bool verify_block(const float* hC, const float* Cref,
                         int N, int r0, int c0, int bs) {
    float max_abs = 0.0f, max_rel = 0.0f;
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < bs; ++j) {
            float got = hC[(r0 + i) * N + (c0 + j)];
            float ref = Cref[i * bs + j];
            float d = std::fabs(got - ref);
            float r = d / (std::fabs(ref) + 1e-6f);
            if (d > max_abs) max_abs = d;
            if (r > max_rel) max_rel = r;
        }
    }
    bool ok = (max_rel < 1e-3f) || (max_abs < 1e-3f);
    std::printf("  spot-check on %dx%d block at (%d,%d): "
                "max_abs=%.3e max_rel=%.3e -> %s\n",
                bs, bs, r0, c0, max_abs, max_rel, ok ? "OK" : "FAIL");
    return ok;
}

int main(int argc, char** argv) {
    int N     = (argc > 1) ? std::atoi(argv[1]) : 1024;
    int iters = (argc > 2) ? std::atoi(argv[2]) : 20;
    if (N <= 0) { std::fprintf(stderr, "N must be positive\n"); return 1; }

    size_t bytes = (size_t)N * N * sizeof(float);
    std::printf("[naive] N=%d iters=%d  bytes/matrix=%.1f MiB\n",
                N, iters, bytes / (1024.0 * 1024.0));

    int dev = 0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("Device: %s (SM %d.%d, %d SMs)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    float* hA = (float*)std::malloc(bytes);
    float* hB = (float*)std::malloc(bytes);
    float* hC = (float*)std::malloc(bytes);
    if (!hA || !hB || !hC) { std::fprintf(stderr, "host alloc failed\n"); return 1; }
    init_matrix(hA, N, 0xC0FFEEu);
    init_matrix(hB, N, 0xDEADBEEFu);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    // Launch config: 8x8 block; one thread per output element of C.
    dim3 block(8, 8);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Warm up (first launch includes JIT/context overhead).
    matmul_naive_kernel<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time `iters` launches with CUDA events.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        matmul_naive_kernel<<<grid, block>>>(dA, dB, dC, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iters;

    // Correctness spot-check on a 8x8 sub-block.
    CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
    const int BS = 8;
    float* Cref = (float*)std::malloc(BS * BS * sizeof(float));
    cpu_reference_block(hA, hB, Cref, N, 0, 0, BS);
    verify_block(hC, Cref, N, 0, 0, BS);

    double gflops = 2.0 * (double)N * (double)N * (double)N / 1e9;
    std::printf("  avg time: %.3f ms   perf: %.2f GFLOP/s\n",
                avg_ms, gflops / (avg_ms / 1e3));

    std::free(Cref);
    std::free(hA); std::free(hB); std::free(hC);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}