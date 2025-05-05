#include <cuda_runtime.h>
#include <stdio.h>

// Tiled kernel using shared memory
__global__ void tiled_shared_add(float* A, float* B, float* C, int N) {
    extern __shared__ float tile[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float* tileA = tile;
    float* tileB = tile + blockDim.x;

    for (int i = idx; i < N; i += stride) {
        // Load tile
        tileA[tid] = A[i];
        tileB[tid] = B[i];

        __syncthreads();

        // Compute
        C[i] = tileA[tid] + tileB[tid];

        __syncthreads();  // optional if no shared reuse
    }
}

extern "C"
void run_tiled_shared(int N) {
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    size_t size = N * sizeof(float);
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    for (int i = 0; i < N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = min((N + threads - 1) / threads, 1024);
    size_t sharedMemSize = 2 * threads * sizeof(float);

    tiled_shared_add<<<blocks, threads, sharedMemSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
}
