// 针对warp divergent
// v0运行时间: 933.44us
// v0带宽利用率: 62.67%
// v0内存吞吐量: 144.33GB/s
// v1运行时间: 675.90us
// v1带宽利用率: 86.46%
// v1内存吞吐量: 199.35GB/s

#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <vector>
#include <iostream>

#define N 32 * 1024 * 1024 // 32MB
#define BLOCK_SIZE 256


void CPU_reduce(std::vector<float> &input_, double &output_) {
    for (auto x : input_) {
        output_ += x;
    }
}

bool checkout(float output_, float output_host) {
    if (std::abs(output_ - output_host) > 0.0001) {
        return false;
    } else {
        return true;
    }
}

__global__ void reduce_v1(float *g_idata, float *g_odata) {
    
    // 256 * 32/8 = 1024Byte -> 1KB
    // 3080: 单个SM的L1 cache 128KB
    __shared__ float smem[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // SM, 每个block独享
    // <<<N / BLOCK_SIZE, BLOCK_SIZE(向上取整)>>>
    smem[tid] = g_idata[gid];
    __syncthreads(); // 使用smem, 同步

    // =============================== change ===============================
    for (unsigned int i = 1; i < blockDim.x; i *= 2) {
        
        // 第一次迭代: i = 1
        // 0-3号warp: tid: 0 -> 127     均进入if分支
        // 4-7号warp: tid: 128 -> 255   均未进入if分支
        // 第二次迭代: i = 2
        // 同上
        // 第四次迭代: i = 8
        // 0号warp: tid: 0 -> 15    进入if分支
        //          tid: 16 -> 31   未进入if分支 -> warp divergent
        int index = 2 * i * tid;
        if (index < blockDim.x) {
            smem[index] += smem[index + i];
        }
        __syncthreads();
    }

    if (tid == 0) {
        // 写回每个block的sum
        g_odata[blockIdx.x] = smem[0];
    }
}

int main() {
    float *input_device;
    float *output_device;
    float *input_host = (float*)malloc(N * sizeof(float));
    float *output_host = (float*)malloc(N / BLOCK_SIZE * sizeof(float));
    cudaMalloc((void**)&input_device, N * sizeof(float));
    cudaMalloc((void**)&output_device, (N / BLOCK_SIZE) * sizeof(float));
    for (int i = 0; i < N; ++i) {
        input_host[i] = 1.0;
    }
    cudaMemcpy(input_device, input_host, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(N / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    int block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; // 向上取整
    reduce_v1<<<grid, block>>>(input_device, output_device);
    cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 1; i < N / BLOCK_SIZE; ++i) {
        output_host[0] += output_host[i];
    }

    bool res;
    std::vector<float> input_(N, 1.0);
    double output_ = 0;
    CPU_reduce(input_, output_);
    res = checkout(output_, output_host[0]);
    if (res) {
        std::cout << "PASSED!" << std::endl;
    } else {
        std::cout << "FAILED!" << std::endl;
    }

    free(input_host);
    free(output_host);
    cudaFree(input_device);
    cudaFree(output_device);

    return 0;
}