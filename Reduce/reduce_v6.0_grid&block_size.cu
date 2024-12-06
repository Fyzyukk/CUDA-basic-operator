// grid_size
// v0运行时间: 933.44us
// v0带宽利用率: 62.67%
// v0内存吞吐量: 144.33GB/s
// v1运行时间: 675.90us
// v1带宽利用率: 86.46%
// v1内存吞吐量: 199.35GB/s
// v2运行时间: 649.50us
// v2带宽利用率: 89.95%
// v2内存吞吐量: 207.42GB/s
// v3运行时间: 337.25us
// v3带宽利用率: 89.86%
// v3内存吞吐量: 398.79GB/s
// v4运行时间: 199.33us
// v4带宽利用率: 92.26%
// v4内存吞吐量: 674.85GB/s(比较接近760GB/s的理论值)
// v4: L1/TEX Cache Throughput [%]	68.56
// v4: L1/TEX Hit Rate [%]	0.27
// v5运行时间: 196.67us
// v5带宽利用率: 93.59%
// v5内存吞吐量: 683.72GB/s(比较接近760GB/s的理论值)
// v5: L1/TEX Cache Throughput [%]	69.61
// v5: L1/TEX Hit Rate [%]	0.35
// v6.0运行时间: 191.62us
// v6.0带宽利用率: 96.51%
// v6.0内存吞吐量: 700.47(比较接近760GB/s的理论值)
// v6.0: grid_size: [1024, 1, 1]


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

template<unsigned int blockSize>
__device__ void warpReduce(volatile float *smem, int tid) {
    if(blockSize >= 64)smem[tid] += smem[tid + 32];
    if(blockSize >= 32)smem[tid] += smem[tid + 16];
    if(blockSize >= 16)smem[tid] += smem[tid + 8];
    if(blockSize >= 8)smem[tid] += smem[tid + 4];
    if(blockSize >= 4)smem[tid] += smem[tid + 2];
    if(blockSize >= 2)smem[tid] += smem[tid + 1];
}

template<unsigned int blockSize, unsigned int NUM_PER_THREAD>
__global__ void reduce_v6(float *g_idata, float *g_odata) {
    __shared__ float smem[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * (blockDim.x * NUM_PER_THREAD) + threadIdx.x;
    smem[tid] = 0;

    // TODO: unrolling
    for (int i = 0; i < NUM_PER_THREAD; ++i) {
        smem[tid] += g_idata[gid + i * blockSize];
    }    
    __syncthreads(); // 使用smem, 同步

    // do reduction in shared mem
    if(blockSize >= 512){
        if(tid < 256){
            smem[tid] += smem[tid + 256];
        }
        __syncthreads();
    }
    if(blockSize >= 256){
        if(tid < 128){
            smem[tid] += smem[tid + 128];
        }
        __syncthreads();
    }
    if(blockSize >= 128){
        if(tid < 64){
            smem[tid] += smem[tid + 64];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid < 32) {
        warpReduce<blockSize>(smem, tid);
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

    const int block_num = 1024;
    const int NUM_PER_BLOCK = N / block_num;                // 每个block处理多少数据
    const int NUM_PER_THREAD = NUM_PER_BLOCK / BLOCK_SIZE;  // 每个thread处理多少数据
    dim3 grid(block_num);
    dim3 block(BLOCK_SIZE);
    reduce_v6<BLOCK_SIZE, NUM_PER_THREAD><<<grid, block>>>(input_device, output_device);
    cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 1; i < block_num; ++i) {
        output_host[0] += output_host[i];
    }

    bool res;
    std::vector<float> input_(N, 1.0);
    double output_ = 0;
    CPU_reduce(input_, output_);
    res = checkout(output_, output_host[0]);
    if (res) {
        std::cout << "PASSED!" << std::endl;
        std::cout << "CPU: " << output_ << std::endl;
        std::cout << "GPU: " << output_host[0] << std::endl;
    } else {
        std::cout << "FAILED!" << std::endl;
        std::cout << "CPU: " << output_ << std::endl;
        std::cout << "GPU: " << output_host[0] << std::endl;
    }

    free(input_host);
    free(output_host);
    cudaFree(input_device);
    cudaFree(output_device);

    return 0;
}