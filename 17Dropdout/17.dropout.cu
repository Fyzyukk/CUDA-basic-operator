# if 0
import torch
from torch import nn
class MyNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        output = self.dropout(x)
        print(f"dropout的输出: \n {output}")
        output = self.fc1(output)
        return output

input_size = 10
num_classes = 5
model = MyNet(input_size, num_classes)
x = torch.arange(0, 10).reshape(-1).float()
print('输入向量: ', x)
model.train()
print("训练模式下:", model(x))
model.eval()
print("测试模式下:", model(x))
# endif

# include <cuda.h>
# include <curand.h>    
# include <curand_kernel.h> // 生成随机数的接口
# include <cstdio>


#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

// 自定义向量化类型，主要用于VecType_u8
template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) VectorType {
  T val[VecSize];
};

template <typename T>
struct uniform_distribution {
    __device__ T operator () (curandStatePhilox4_32_10_t* state) {
        return static_cast<T>(curand_uniform(state));
    }
    static constexpr int Count = 1;
};

template <>
struct uniform_distribution<float> {
    __device__ float4 operator()(curandStatePhilox4_32_10_t* state) {
        return curand_uniform4(state);
    }
    static constexpr int Count = 4;
};

template <typename T, int VecSize>
struct DstMaskFunctor {
    float prob_;
    bool is_upscale_in_train_;
    float inv_prob;

    __device__ DstMaskFunctor(const float prob, const bool is_upscale_in_train) : 
    prob_(prob), is_upscale_in_train_(is_upscale_in_train) {
        inv_prob = 1.0f / (1 - prob_);
    }

    __device__ void operator () (T* dst, const T* src_val, const T* rand) {
        
        for (int i = 0; i < VecSize; ++i) {
            if (rand[i] > prob_) {
                dst[i] = is_upscale_in_train_ ? static_cast<T>(src_val[i] * inv_prob) : src_val[i]; // dropout的输出
                dst[i + VecSize] = static_cast<T>(1); // mask
            } else {
                dst[i] = 0; // dropout的输出
                dst[i + VecSize] = static_cast<T>(0); // mask
            }
        }
    }
};


template <typename T, typename MaskType>
__global__ void VectorizedDstMask(
    const size_t n,
    const int seed,
    const float dropout_prob,
    const T* src,
    MaskType* mask,
    T* dst,
    bool is_upscale_in_train,
    uint64_t increment,
    int64_t main_offset)
{
    int thread_idx  = threadIdx.x;
    int thread_nums = blockDim.x;
    int block_idx   = blockIdx.x;
    int block_nums  = gridDim.x;
    int gid         = blockIdx.x * blockDim.x + threadIdx.x;

    // block_offset: 0 or 256
    int block_offset = block_idx * thread_nums;
    
    // VecSize: 4
    constexpr int VecSize = uniform_distribution<T>::Count;

    // stride: 所有block在一次iter中所能处理的data
    // stride: 2 * 256 * 4 = 2048
    int stride = block_nums * thread_nums * VecSize;

    // 初始化随机数
    curandStatePhilox4_32_10_t state;
    curand_init(seed, gid, increment, &state);

    // 以每次向量化读取的data为最小单位, 不是将mask全部生成后在进行dropout
    // 而是每次向量化读取过程中就进行dropout
    // 声明寄存器, 暂存输出数据, 随机数, mask
    // dst: 0 ~ VecSize - 1
    // mask: VecSize ~ 2 * VecSize - 1
    T dst_mask[VecSize * 2];
    float rands[VecSize];
    MaskType mask_result[VecSize];

    // 初始化随机数的functor, 计算mask和输出结果的functor
    using Rand = uniform_distribution<float>;
    
    // dst_functor: DstMaskFunctor是一个struce, dst_functor是其实例化
    auto dst_functor = DstMaskFunctor<T, VecSize>(dropout_prob, is_upscale_in_train);

    using VecType = float4;
    using VecType_u8 = VectorType<MaskType, VecSize>;
    VecType vec_temp_input;

    // 可以向量化的部分(0 ~ 4095)
    // block_offset = block_idx * thread_nums;
    // block_offset: 0, 256
    int start = block_offset * VecSize;
    for (;start < main_offset; start += stride) {

        // 取出数据
        int thread_offset = thread_idx;

        // 向量化的读, 一次读float4
        // start: 0, 1024
        // stride: 2048
        // vec_input: 地址
        //  start: 0
        //      vec_input -> src[0]     ~   src[1023]
        //      vec_input -> src[2048]  ~   src[3071] 
        //  start: 1024
        //      vec_input -> src[1024]  ~   src[2047]
        //      vec_input -> src[3072]  ~   src[4095]
        const VecType* vec_input = reinterpret_cast<const VecType*>(src + start);
        
        // 向量化的读
        // ===================================
        // ============ 是否需要 * 4 ==========
        // ===================================
        // 不需要 *4, 因为vec_input是float4类型, CUDA会自动以float4为单位进行寻址
        //  eg: thread_offset = 0 -> vec_temp_input[0]指向vec_input[0] ~ dst_mask[3]
        //      thread_offset = 1 -> vec_temp_input[1]指向vec_input[4] ~ dst_mask[7]
        // vec_temp_input = vec_input[thread_offset * 4];
        vec_temp_input = vec_input[thread_offset];

        // 加载数据
        auto random_tuple = Rand()(&state);
        for (int i = 0; i < VecSize; ++i) {
            dst_mask[i] = *(reinterpret_cast<T*>(&vec_temp_input) + i);
            rands[i] = static_cast<float>((&random_tuple.x)[i]);
        }

        // 计算数据
        dst_functor(&dst_mask[0], &dst_mask[0], &rands[0]);

        // 向量化的写, 一次写float4
        // start: 0, 1024
        // stride: 2048
        // vec_dst_output: 地址
        //  start: 0
        //      vec_dst_output -> dst[0]     ~   dst[1023]
        //      vec_dst_output -> dst[2048]  ~   dst[3071] 
        //  start: 1024
        //      vec_dst_output -> dst[1024]  ~   dst[2047]
        //      vec_dst_output -> dst[3072]  ~   dst[4095]
        T* res = dst + start;
        VecType* vec_dst_output = reinterpret_cast<VecType*>(res);

        // ===================================
        // ============ 是否需要 * 4 ==========
        // ===================================
        // 不需要 *4, 因为vec_dst_output是float4类型, CUDA会自动以float4为单位进行寻址
        //  eg: thread_offset = 0 -> vec_dst_output[0]指向dst_mask[0] ~ dst_mask[3]
        //      thread_offset = 1 -> vec_dst_output[1]指向dst_mask[4] ~ dst_mask[7]
        // vec_dst_output[thread_offset * 4] = *(reinterpret_cast<VecType*>(&dst_mask[0]));
        vec_dst_output[thread_offset] = *(reinterpret_cast<VecType*>(&dst_mask[0]));

        # if 1
        // 记录生成的mask
        for (int i = 0; i < VecSize; ++i) {

            // 前VecSize位: 保存result   后VecSize位: 保存mask
            mask_result[i] = static_cast<MaskType>(dst_mask[i + VecSize]);
        }

        MaskType* mask_res = mask + start;
        VecType_u8* vec_mask_output = reinterpret_cast<VecType_u8*>(mask_res);
        vec_mask_output[thread_offset] = *(reinterpret_cast<VecType_u8*>(mask_result));
        # endif
    }

    // 不可向量化的部分, 标量读写
    // 此时的start和offset等价, start += stride == main_offset
    // int remain = n - start;
    int remain = n - main_offset;
    if (remain > 0) {

        // 取出数据
        int thread_offset = thread_idx * VecSize;

        // const T* src_remain = src + start;
        const T* src_remain = src + main_offset;
        auto random_tuple = Rand()(&state);
        for (int i = 0; i < VecSize; i++) {
            if (i + thread_offset < remain) {
                dst_mask[i] = src_remain[thread_offset + i];
            }
            rands[i] = static_cast<float>((&random_tuple.x)[i]);
        }

        // 算出数据
        dst_functor(&dst_mask[0], &dst_mask[0], &rands[0]);

        // 写回数据
        // T* res = dst + start;
        // MaskType* mask_res = mask + start;
        T* res = dst + main_offset;
        MaskType* mask_res = mask + main_offset;
        
        for (int i = 0; i < VecSize; ++i) {
            if ((thread_offset + i) < remain) {
                res[thread_offset + i] = dst_mask[i];
                mask_result[i] = static_cast<MaskType>(dst_mask[i + VecSize]);
                mask_res[thread_offset + i] = mask_result[i];
            }
        }
    }
}

template <typename T>
void DropoutKernel(
    const bool is_test,
    const bool is_upscale_in_train,
    const size_t num_eles,
    const float dropout_prob,
    const int seed_val,
    const float* x_data,
    uint8_t* mask_data,
    float* y_data
)
{
    if (!is_test) {
        if (dropout_prob == 1.0f) {
            cudaMemset(y_data, 0, num_eles);
            cudaMemset(mask_data, 0, num_eles);
            return;
        }

        // 共分配512个threads
        // 一共4100个data, 循环8次, 剩下4个标量读写
        size_t num_block = 2;
        size_t num_thread = 256;
        dim3 grid(num_block);
        dim3 block(num_thread);

        uint64_t increment = 0;

        // 向量化访存的数据量
        // 4100 / (2 * 256 * 4) * (2 * 256 * 4) = 4096, 剩下4个data, 标量读取
        uint64_t main_offset = num_eles / (num_block * num_thread * 4) * (num_block * num_thread * 4);

        VectorizedDstMask<T, uint8_t><<<grid, block>>> \ 
        (num_eles, seed_val, dropout_prob, x_data, mask_data, y_data, is_upscale_in_train, increment, main_offset);
    } else {
        cudaMemcpy(y_data, x_data, num_eles, cudaMemcpyDeviceToDevice);
    }
}


int main() 
{
    constexpr size_t num_eles = 4100; // 512 * 8 + 2
    
    float* x = (float*)malloc(num_eles * sizeof(float));
    float* d_x;
    CHECK(cudaMalloc((void **)&d_x, num_eles * sizeof(float)));

    float* y = (float*)malloc(num_eles * sizeof(float));
    float* d_y;
    CHECK(cudaMalloc((void **)&d_y, num_eles * sizeof(float)));

    uint8_t* mask = (uint8_t*)malloc(num_eles * sizeof(uint8_t));
    uint8_t* d_mask;
    CHECK(cudaMalloc((void **)&d_mask, num_eles * sizeof(uint8_t)));

    // 输入为全1, 输出为 0 和 1-p
    for(int i = 0; i < num_eles; i++){
        x[i] = 1;
    }
    CHECK(cudaMemcpy(d_x, x, num_eles * sizeof(float), cudaMemcpyHostToDevice));
    
    const bool is_test = false;
    const bool is_upscale_in_train = true;
    const float dropout_prob = 0.5;
    const int seed_val = 10000;    

    DropoutKernel<float>(is_test, is_upscale_in_train, num_eles, dropout_prob, seed_val, \
        d_x, d_mask, d_y);
    
    CHECK(cudaMemcpy(y, d_y, num_eles * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(mask, d_mask, num_eles * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    // 打印最后位于可向量化和不可向量化边界的三个结果
    for (int i = num_eles - 3; i < num_eles; i++){
      printf("[%d] y is %f\n",i, y[i]);
      printf("[%d] mask is %d\n",i, mask[i]);
    }

    int count_res = 0;
    int count_mask = 0;
    for (int i = 0; i < num_eles; ++i) {
        if (y[i] != 0) {
            count_res++;
        }
        if (mask[i] != 0) {
            count_mask++;
        }
    }

    printf("count_res: %d\n", count_res);
    printf("count_mask: %d\n", count_mask);


    cudaFree(d_x);                                                                                                        
    cudaFree(d_y);                                                                                                        
    cudaFree(d_mask);                                                                                                        
    free(x);                                                                                                              
    free(y);                                                                                                              
    free(mask);


    return 0;
}