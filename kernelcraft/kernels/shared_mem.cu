// shared_mem.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void shared_mem_add(float* A, float* B, float* C, int N) {
    extern __shared__ float tile[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < N) {
        // Split shared memory for A and B
        float* tileA = tile;
        float* tileB = tile + blockDim.x;

        // Load to shared memory
        tileA[tid] = A[idx];
        tileB[tid] = B[idx];

        __syncthreads();

        // Compute and write result
        C[idx] = tileA[tid] + tileB[tid];
    }
}

extern "C"
void run_shared_mem(int N) {
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

    dim3 threads(256);
    dim3 blocks((N + 255) / 256);
    size_t sharedMemSize = 2 * threads.x * sizeof(float);

    shared_mem_add<<<blocks, threads, sharedMemSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
}
