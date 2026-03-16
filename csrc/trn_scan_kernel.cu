/*
 * TRN Kogge-Stone scan CUDA kernel.
 * Computes r_t = alpha_t * r_{t-1} + drive_t for all (B, T, K) in parallel.
 *
 * Each block handles one batch element.
 * Threads within a block collaboratively scan T positions using shared memory.
 * The associative operator is: (a2,b2) o (a1,b1) = (a2*a1, a2*b1+b2).
 *
 * Compile (WSL2):
 *   nvcc -arch=sm_120 -O3 --shared -Xcompiler -fPIC \
 *     csrc/trn_scan_kernel.cu -o csrc/trn_scan.so
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

// Kogge-Stone inclusive scan in shared memory.
// Each thread handles one position t, iterating over K channels.
__global__ void kogge_stone_scan_kernel(
    const float* __restrict__ alpha,   // (B, T, K)
    const float* __restrict__ drive,   // (B, T, K)
    float* __restrict__ output,        // (B, T, K)
    int T, int K
) {
    const int b = blockIdx.x;       // batch index
    const int t = threadIdx.x;      // position within sequence
    if (t >= T) return;

    const int base = b * T * K;

    // Process K channels in chunks to manage register pressure
    for (int k_start = 0; k_start < K; k_start += blockDim.y) {
        const int k = k_start + threadIdx.y;
        if (k >= K) continue;

        const int idx = base + t * K + k;
        float a = alpha[idx];
        float d = drive[idx];

        // Kogge-Stone: log2(T) rounds of parallel prefix
        for (int offset = 1; offset < T; offset <<= 1) {
            // Read from neighbor at t - offset
            float a_left = 1.0f;
            float d_left = 0.0f;
            if (t >= offset) {
                // Neighbor's values from previous round are in registers
                // But we need inter-thread communication -> shared memory
            }
            __syncthreads();
        }

        output[idx] = d;  // placeholder
    }
}

// Simpler approach: sequential per (b, k) pair, parallelized across B*K.
// This eliminates the shared memory complexity and is still much faster
// than the Python loop because there's zero kernel launch overhead.
__global__ void sequential_scan_kernel(
    const float* __restrict__ alpha,   // (B, T, K)
    const float* __restrict__ drive,   // (B, T, K)
    float* __restrict__ output,        // (B, T, K)
    int T, int K
) {
    // Each thread handles one (batch, channel) pair across all T positions
    const int bk = blockIdx.x * blockDim.x + threadIdx.x;
    const int B_times_K = gridDim.x * blockDim.x;  // unused, just for clarity
    const int b = bk / K;
    const int k = bk % K;

    // Bounds check
    if (k >= K) return;

    const int base = b * T * K + k;  // start of this (b, k) channel
    float h = 0.0f;

    for (int t = 0; t < T; t++) {
        const int idx = base + t * K;
        h = alpha[idx] * h + drive[idx];
        output[idx] = h;
    }
}


// PyTorch wrapper
std::vector<torch::Tensor> trn_scan_forward(
    torch::Tensor alpha,    // (B, T, K) fp32
    torch::Tensor drive_r,  // (B, T, K) fp32
    torch::Tensor drive_i   // (B, T, K) fp32
) {
    TORCH_CHECK(alpha.is_cuda(), "alpha must be CUDA");
    TORCH_CHECK(alpha.dtype() == torch::kFloat32, "alpha must be fp32");

    const int B = alpha.size(0);
    const int T = alpha.size(1);
    const int K = alpha.size(2);

    auto output_r = torch::empty_like(drive_r);
    auto output_i = torch::empty_like(drive_i);

    // Launch: one thread per (b, k) pair
    const int threads = 256;
    const int blocks = (B * K + threads - 1) / threads;

    sequential_scan_kernel<<<blocks, threads>>>(
        alpha.data_ptr<float>(),
        drive_r.data_ptr<float>(),
        output_r.data_ptr<float>(),
        T, K
    );

    sequential_scan_kernel<<<blocks, threads>>>(
        alpha.data_ptr<float>(),
        drive_i.data_ptr<float>(),
        output_i.data_ptr<float>(),
        T, K
    );

    return {output_r, output_i};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("trn_scan_forward", &trn_scan_forward,
          "TRN scan forward (sequential per channel, parallel across B*K)");
}
