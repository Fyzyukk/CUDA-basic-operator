#include <cuda_runtime.h>
#include "cstdio"


// 一维
__global__ void hello_cuda() {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("hello cuda!");
}

int main() {
    
    // 核函数
    hello_cuda<<<10, 5>>>(); // 10 个 block, 每个 block 中有 5 个 thread -> 50 个 thread
    
    // 同步等待所有 thread 结束
    cudaDeviceSynchronize();

    return 0;
}