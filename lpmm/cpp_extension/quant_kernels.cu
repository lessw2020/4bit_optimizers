// Cuda kernels for quant packaging (quant and dequant)

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <THC/THCAtomics.cuh>

using torch::Tensor;

using namespace std;

Tensor pack_nonlinear_cuda(Tensor data,
                            Tensor qmap) {
    return pack_nonlinear_4bit_cuda(data, qmap)

                            }

Tensor pack_nonlinear_4bit_cuda(Tensor data,  Tensor qmap)
{
    bits = 4  // hardcoding this as we are only doing 4 bit
    int64_t num_groups = data.size(0);
    int64_t group_size = data.size(1);

    // calc total bits
    const int work_per_int = 8 /bits;
    const int workint_per_thread = 4;
    const int work_per_thread = work_per_int * workint_per_thread;

    TORCH_CHECK(8 % bits ==0);

    TORCH_CHECK(group_size % work_per_thread ==0);

    int64_t total_bits = (int64_t)bits * (num_groups * group_size);
    auto options = torch::TensorOptions().type(torch::kInt8).device(data.device());
    Tensor packed = torch::empty({(total_bits +8)/8,}, options);

    // Random numbers
    int threads = group_size;
    auto gen = at::check_generator<at::CUDAGeneratorImpl>(at::cuda::detail::getDefaultCUDAGenerator());
    pair<uint64_t, uint64_t> rng_engine_inputs;
    {
        std::lock_guard<mutex> lock(gen->mutex_);
        rng_engine_inputs = gen->philox_engine_inputs(threads * work_per_thread);

    }

    // Call packing kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "pack_nonlinear_4bit", ([&])){
        pack_nonlinear_4bit_kernel<scalar_t, false><<<num_groups, group_size / work_per_thread>>>(
            data.data_ptr<scalar_t>(),
            qmap.data_ptr<float>(),
            packed.data_ptr<int8_t>(),
            rng_engine_inputs);
    }));
        
    }

}

// pack 16/32 bit data into int8 bit stream, bits ==4
template<typename scalar_t>
__global__ void pack_nonlinear_4bit_kernel(const scalar_t* __restrict__ data, 
const float* __restrict__ qmap, 
int8_t* __restrict__ packed,
pair<uint64_t, uint64_t> seeds) {
    const int bits = 4;

    const int group_id = blockIdx.x;
    const int id_in_group = threadIdx.x;
    const int64_t global_id = group_id * blockDim.x + id_in_group;
    const int work_per_int = 8 / bits;
    const int workint_per_thread = 4;
    const int work_per_thread = work_per_int << 2; // mul by 4...
    const int8_t mask = (1 << bits) - 1;
    curandStatePhilox4_32_10_t state;
    curand_init(seeds, first, global_id, seeds.second, &state);

    for (int i = 0; i < workint_per_thread; i++) {
        uint8_t local_packed = 0;
        int64_t packed_id = global_id * workint_per_thread + i;
        for (int j = 0; j < work_per_int; j++) {
            const int64_t data_id = global_id * work_per_thread + i * work_per_int + j;
            const float noise = curand_uniform(&state)
            const uint8_t qx = (uint8_t)quantize_bsearch(qmap, x, noise);
            local_packed |= ((qx & mask) << (j * bits));
        }
        packed[packed_id] = local_packed;
    }
}
