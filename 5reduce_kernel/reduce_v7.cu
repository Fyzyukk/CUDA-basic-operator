// reduce 类算子 -> 累加
// v7: 最后结果是串行循环 100000 次对核函数返回的结果进行求和 -> 并行
// baseline:  519.084167 ms
// reduce_v0: 0.696928 ms      bank conflict: 834
// reduce_v1: 0.510880 ms(old) bank conflict: 7001056
// reduce_v1: 0.478528 ms(new) bank conflict: 689
// reduce_v2: 0.454912 ms      bank conflict: 891
// reduce_v3: 0.237376 ms      bank conflict: 987
// reduce_v4: 0.197024 ms      bank conflict: 2343
// reduce_v5: 0.236256 ms
// reduce_v6: 0.202784 ms(blockSize = 128)
// reduce_v7: 0.422016 ms(两次 reduce)


#include <cuda.h>
#include "cuda_runtime.h"
//#include <bits/stdc++.h>
#include <cstdio>


void CheckResult(int *out, float res, int groudtruth, int n) {

    if (res != groudtruth) {
        printf("groudtruth: %d\n", groudtruth);
        printf("result: %d\n", res);
        printf("the ans is wrong\n");
    }

    else
        printf("the ans is right\n");
}

// __device__ GPU 端的函数, 编译器自行决定是否 inline
// 将 for 循环展开, 节省位运算

template <int blockSize>
__device__ void BlockSharedMemReduce(float* smem) {

    if (blockSize >= 1024) {
        if (threadIdx.x < 512)
            smem[threadIdx.x] += smem[threadIdx.x + 512];
    }
    __syncthreads();
    if (blockSize >=512) {
        if (threadIdx.x < 256)
            smem[threadIdx.x] += smem[threadIdx.x + 256];
    }
    __syncthreads();
    if (blockSize >= 256) {
        if (threadIdx.x< 128)
            smem[threadIdx.x] += smem[threadIdx.x + 128];
    }
    __syncthreads();
    if (blockSize >= 128) {
        if (threadIdx.x < 64)
            smem[threadIdx.x] += smem[threadIdx.x + 64];
    }
    __syncthreads();

    // final warp
    // volatile: 
    if (threadIdx.x < 32) {
        volatile float* vshm = smem;
        if (blockDim.x >= 64)
            vshm[threadIdx.x] += vshm[threadIdx.x + 32];
        vshm[threadIdx.x] += vshm[threadIdx.x + 16];
        vshm[threadIdx.x] += vshm[threadIdx.x + 8];
        vshm[threadIdx.x] += vshm[threadIdx.x + 4];
        vshm[threadIdx.x] += vshm[threadIdx.x + 2];
        vshm[threadIdx.x] += vshm[threadIdx.x + 1];
    }
}

template<int blockSize>
__global__ void reduce_v7(int* input, int* output, size_t n) {

    __shared__ float smem[blockSize];
    unsigned int tid = threadIdx.x;

    // 不显式指定每个 thread 处理多少个数据
    // unsigned int gtid = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_tid = blockDim.x * gridDim.x;

    // load
    // 更灵活, 不局限于 n thread -> 2n 数据   
    // 根据 for 自动确定每个 thread 处理多少个数据
    // smem[tid] = input[gtid] + input[gtid + blockDim.x];
    float sum = 0.0f;
    for (int32_t i = gtid; i < n; i += total_tid) 
        sum += input[i];

    smem[tid] = sum;
    __syncthreads();

    //operation
    BlockSharedMemReduce<blockSize>(smem);

    // result
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

int main() {

    float ms = 0;
    int groudtruth = 0;
    const int N = 25600000; // 数据量

    // 获取设备属性
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // thread
    const int blockSize = 256;
    const int blockSize_v3 = blockSize / 2;
    const int blockSize_v6 = blockSize / 4;
    int gridSize  = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]); // 向上加 1 -> 防止 N = 255 -> gridSize = 0

    dim3 Grid(gridSize);
    dim3 Block(blockSize_v3);

    // allocate memory
    int* device_in;
    int* device_out;
    int* device_res;
    int* host_in  = (int* )malloc(N * sizeof(int));
    int* host_res = (int* )malloc(1 * sizeof(int));
    int* host_out = (int* )malloc(gridSize * sizeof(int));
    for (int i = 0; i < N; ++i) {
        host_in[i] = 1;
        groudtruth += host_in[i];
    }
    cudaMalloc((void** )&device_in, N * sizeof(int));
    cudaMalloc((void** )&device_res, 1 * sizeof(int));
    cudaMalloc((void** )&device_out, gridSize * sizeof(int));
    cudaMemcpy(device_in, host_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // 开始处理
    // grid: (100000, 1, 1)   block: (256, 1, 1)   thread = 25600000
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v7<blockSize_v3><<<Grid, Block>>>(device_in, device_out, N);
    reduce_v7<blockSize_v3><<<1, Block>>>(device_out, device_res, gridSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(host_res, device_res, 1 * sizeof(int), cudaMemcpyDeviceToHost);

    // 验证
    CheckResult(host_out, *host_res, groudtruth, gridSize);
    printf("reducev7 latency = %f ms\n", ms);

    // free
    cudaFree(device_in);
    cudaFree(device_out);
    cudaFree(device_res);

    free(host_in);
    free(host_out);
    free(host_res);

    return 0;
}