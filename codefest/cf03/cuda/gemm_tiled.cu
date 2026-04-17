// matmul_tiled.cu
// Shared-memory tiled FP32 matrix multiply: C = A * B, row-major, N x N.
// Tile size TILE = 8 per spec. Each 8x8 thread block computes one 8x8
// output tile of C. Each thread computes exactly one C element.
// Requires N % TILE == 0 (1024 qualifies).
//
// Build: nvcc -O3 -std=c++14 -arch=sm_70 matmul_tiled.cu -o matmul_tiled
// Run:   ./matmul_tiled            (N=1024, 20 timed iterations)
//        ./matmul_tiled 2048 50    (optional overrides)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#ifndef TILE
#define TILE 8
#endif

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
// Shared-memory tile plan: per K-slab, cooperatively load an 8x8 piece of A
// and an 8x8 piece of B, __syncthreads, then each thread does 8 FMAs out of
// those shared tiles, __syncthreads, and move to the next slab.
// Global-memory traffic is reduced by a factor of TILE vs. the naive kernel.
__global__
void matmul_tiled_kernel(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         int N)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;                   // col within tile (0..TILE-1)
    int ty = threadIdx.y;                   // row within tile (0..TILE-1)
    int row = blockIdx.y * TILE + ty;       // global C row
    int col = blockIdx.x * TILE + tx;       // global C col

    float acc = 0.0f;
    int numTiles = N / TILE;                // exact because N % TILE == 0

    for (int t = 0; t < numTiles; ++t) {
        // Each thread loads exactly one element into each shared tile.
        //   As[ty][tx] <- A[row,          t*TILE + tx]
        //   Bs[ty][tx] <- B[t*TILE + ty,  col]
        As[ty][tx] = A[row * N + (t * TILE + tx)];
        Bs[ty][tx] = B[(t * TILE + ty) * N + col];

        __syncthreads();                    // tiles fully loaded

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();                    // safe to overwrite tiles next iter
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
    if (N <= 0 || N % TILE != 0) {
        std::fprintf(stderr, "N must be a positive multiple of TILE=%d (got %d)\n",
                     TILE, N);
        return 1;
    }

    size_t bytes = (size_t)N * N * sizeof(float);
    std::printf("[tiled TILE=%d] N=%d iters=%d  bytes/matrix=%.1f MiB\n",
                TILE, N, iters, bytes / (1024.0 * 1024.0));

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

    // Launch config: TILE x TILE block (one thread per element of the 8x8
    // output tile); grid covers the full N x N output with no remainder.
    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    // Warm up (first launch includes JIT/context overhead).
    matmul_tiled_kernel<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time `iters` launches with CUDA events.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        matmul_tiled_kernel<<<grid, block>>>(dA, dB, dC, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iters;

    // Correctness spot-check on a 16x16 sub-block.
    CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
    const int BS = 16;
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
