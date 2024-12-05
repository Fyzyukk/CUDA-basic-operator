#include <cuda.h>
#include "cuda_runtime.h"
#include "cstdio"

#define blockSize 256

__global__ void reduce_v0(int* input, int* output, size_t n) {

    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockSize + threadIdx.x;

    __shared__ float mem[ blockSize];
    mem[tid] = input[gtid];
    __syncthreads();
    for (int idx = blockSize/2; idx < blockDim.x; idx *= 2) {
        if (tid % (2 * idx) == 0) {
            mem[tid] += mem[tid + idx];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = mem[0];
    }
}


void CheckResult(int *out, int groudtruth, int n) {
    float res = 0;
    for (int i = 0; i < n; i++){
        res += out[i];
    }
    if (res != groudtruth) {
        printf("groudtruth: %d\n", groudtruth);
        printf("result: %d\n", res);
        printf("the ans is wrong\n");
    }

    printf("the ans is right\n");
}

int main() {

    // 获取设备属性
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // thread
    // 向上加 1 -> 防止 N = 255
    const int N = 25600000;
    int gridSize  = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);

    dim3 Grid(gridSize);
    dim3 Block(blockSize);

    // allocate memory
    int* host_in  = (int* )malloc(N * sizeof(int));
    int* host_out = (int* )malloc(gridSize * sizeof(int));
    int groudtruth = 0;
    for (int i = 0; i < N; ++i) {
        host_in[i] = 1;
        groudtruth += host_in[i];
    }
    int* device_in;
    int* device_out;
    cudaMalloc((void** )&device_in, N * sizeof(int));
    cudaMalloc((void** )&device_out, gridSize * sizeof(int));
    cudaMemcpy(device_in, host_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // 开始处理
    // grid: (100000, 1, 1)   block: (256, 1, 1)   thread = 25600000
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v0<<<Grid, Block>>>(device_in, device_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(host_out, device_out, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    // 验证
    CheckResult(host_out, groudtruth, gridSize);

    printf("reducev0 latency = %f ms\n", ms);

    // free
    cudaFree(device_in);
    cudaFree(device_out);
    free(host_in);
    free(host_out);


    return 0;
}