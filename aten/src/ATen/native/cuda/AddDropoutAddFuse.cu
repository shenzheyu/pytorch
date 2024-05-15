#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Utils.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/macros/Macros.h>
#include <curand_kernel.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_masked_scale_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/native_dropout_backward_native.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace at::native {

namespace {

// philox generates 128 bits of randomness at a time. Kernel uses this
// explicitly by putting suitably transformed result into float4 for all members
// of float4 to be consumed UNROLL has to be 4. Don't change! Note: VEC <= 4
// (and in most real-world cases will be 4), so same logic applies.
const int UNROLL = 4;

template <
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    int ADims,
    int VEC,
    typename mask_t>
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(256, 4)
#endif
__global__ void fused_dropout_kernel_vec(
    at::cuda::detail::TensorInfo<scalar_t, IndexType> a,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> b,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> c,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> d,
    at::cuda::detail::TensorInfo<mask_t, IndexType> e,
    IndexType totalElements,
    accscalar_t p,
    PhiloxCudaState philox_args) {
  // make sure we don't break assumption that we can't have > 4 elements /
  // thread
  static_assert(VEC <= 4, "Value of VEC must be in [2, 4]");

  using LoadT = memory::aligned_vector<scalar_t, VEC>;
  using MaskLoadT = memory::aligned_vector<mask_t, VEC>;

  auto seeds = at::cuda::philox::unpack(philox_args);
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);

  // Helps align the total number of times curand_uniform4 is called by each
  // thread for the same totalElements in the vec=2 and vec=4 cases.
  bool gridxvec_loop_state = 0;
  accscalar_t scale = 1.0 / p;

  float4 rand;

  // Note: Vectorized loads means we'll stride each thread by an additional VEC
  // factor, as we'll load VEC elements at a time
  for (IndexType linearIndex = idx * VEC; linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x * VEC) {
    // local storage
    scalar_t input_src[VEC];
    scalar_t bias_src[VEC];
    scalar_t residual_src[VEC];
    // We'll use this to actually cause vectorized loads later
    LoadT* input_value = reinterpret_cast<LoadT*>(&input_src);
    LoadT* bias_value = reinterpret_cast<LoadT*>(&bias_src);
    LoadT* residual_value = reinterpret_cast<LoadT*>(&residual_src);

    // curand_uniform_double was pure evil anyway, not doing what it promises,
    // and there's nothing for halfs, so generate float for everything
    //  Note: need a new set of random values per 4 elements -- we'll handle VEC
    //  elements in this thread, so need ceil(VEC / 4) sets of rand.
    if ((VEC == 4) || (gridxvec_loop_state == 0)) {
      rand = curand_uniform4(&state);
    } else {
      // sets up the last two values we generated last iteration to be used this
      // iteration.
      rand.x = rand.z;
      rand.y = rand.w;
      gridxvec_loop_state ^= 1;
    }

    rand.x = rand.x < p;
    rand.y = rand.y < p;
    if (VEC == 4) {
      rand.z = rand.z < p;
      rand.w = rand.w < p;
    }

    // Note: We explicitly check for is_contiguous() before launching the
    // vectorized kernel and replace IndexToOffset call with linearIndex to
    // allow vectorization of NHWC (or other) ordering. Single vectorized load
    *input_value = *reinterpret_cast<LoadT*>(&a.data[linearIndex]);
    *bias_value = *reinterpret_cast<LoadT*>(&b.data[linearIndex]);
    *residual_value = *reinterpret_cast<LoadT*>(&c.data[linearIndex]);

    scalar_t r[VEC];
    mask_t mask[VEC];


// Perform the actual computation
#pragma unroll
    for (int ii = 0; ii < VEC; ii++) {
      r[ii] = (input_src[ii] + bias_src[ii]) * (&rand.x)[ii] * scale +
          residual_src[ii];
      // r[ii] = (input_src[ii] + bias_src[ii] * (&rand.x)[ii] * scale;
      mask[ii] = (mask_t)(&rand.x)[ii];
    }
    // Vectorized writes for both mask & result
    *(reinterpret_cast<LoadT*>(&d.data[linearIndex])) =
        *reinterpret_cast<LoadT*>(&r[0]);
    *(reinterpret_cast<MaskLoadT*>(&e.data[linearIndex])) =
        *reinterpret_cast<MaskLoadT*>(&mask[0]);

    // printf(
    //     "idx: %d,\n input_src: %f, %f, %f, %f,\n bias_src: %f, %f, %f, %f,\n residual_src: %f, %f, %f, %f,\n, output: %f, %f, %f, %f,\n mask: %d, %d, %d, %d\n\n",  
    //     idx,
    //     input_src[0],
    //     input_src[1],
    //     input_src[2],
    //     input_src[3],
    //     bias_src[0],
    //     bias_src[1],
    //     bias_src[2],
    //     bias_src[3],
    //     residual_src[0],
    //     residual_src[1],
    //     residual_src[2],
    //     residual_src[3],
    //     r[0],
    //     r[1],
    //     r[2],
    //     r[3],
    //     mask[0],
    //     mask[1],
    //     mask[2],
    //     mask[3]);

    __syncthreads();
  }
}

template <
    typename scalar_t,
    typename accscalar_t,
    typename IndexType,
    int ADims,
    int BDims = ADims,
    typename mask_t>
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
C10_LAUNCH_BOUNDS_2(256, 4)
#endif
__global__ void fused_dropout_kernel(
    at::cuda::detail::TensorInfo<scalar_t, IndexType> a,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> b,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> c,
    at::cuda::detail::TensorInfo<scalar_t, IndexType> d,
    at::cuda::detail::TensorInfo<mask_t, IndexType> e,
    IndexType totalElements,
    accscalar_t p,
    PhiloxCudaState philox_args) {
  auto seeds = at::cuda::philox::unpack(philox_args);
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);
  accscalar_t scale = 1.0 / p;

  IndexType rounded_size =
      ((totalElements - 1) / (blockDim.x * gridDim.x * UNROLL) + 1) *
      blockDim.x * gridDim.x * UNROLL;
  for (IndexType linearIndex = idx; linearIndex < rounded_size;
       linearIndex += gridDim.x * blockDim.x * UNROLL) {
    // curand_uniform_double was pure evil anyway, not doing what it promises,
    // and there's nothing for halfs, so generate float for everything
    float4 rand = curand_uniform4(&state);
    scalar_t input_src[UNROLL];
    scalar_t bias_src[UNROLL];
    scalar_t residual_src[UNROLL];
    rand.x = rand.x < p;
    rand.y = rand.y < p;
    rand.z = rand.z < p;
    rand.w = rand.w < p;
    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
      if (li < totalElements) {
        // Convert `linearIndex` into an offset of `a`
        const IndexType aOffset =
            cuda::detail::IndexToOffset<scalar_t, IndexType, ADims>::get(
                li, a);
        input_src[ii] = a.data[aOffset];
        bias_src[ii] = b.data[aOffset];
        residual_src[ii] = c.data[aOffset];
      }
    }
    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
      if (li < totalElements) {
        // Convert `linearIndex` into an offset of `b`
        const IndexType bOffset =
            cuda::detail::IndexToOffset<scalar_t, IndexType, BDims>::get(li, d);
        d.data[bOffset] =
            (input_src[ii] + bias_src[ii]) * (&rand.x)[ii] * scale +
            residual_src[ii];
        e.data[bOffset] = (mask_t)(&rand.x)[ii];
      }
    }
    __syncthreads();
  }
}

template <typename mask_t, typename scalar_t, typename accscalar_t>
void masked_scale_kernel(
    at::Tensor& ret,
    const at::Tensor& src,
    const at::Tensor& mask,
    accscalar_t scale) {
  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(ret)
                  .add_input(src)
                  .add_input(mask)
                  .build();

  at::native::gpu_kernel(
      iter,
      [=] GPU_LAMBDA(const scalar_t src_val, const mask_t mask_val)
          -> scalar_t { return (float)mask_val * src_val * scale; });
}

template <typename scalar_t>
int get_vector_size(at::Tensor self, at::Tensor ret, at::Tensor mask) {
  int vec_size = 4;
  // get the vector size
  if (!self.is_non_overlapping_and_dense() ||
      !ret.is_non_overlapping_and_dense() ||
      !mask.is_non_overlapping_and_dense()) {
    vec_size = 1;
  } else {
    vec_size = memory::can_vectorize_up_to<scalar_t>(
        (char*)self.data_ptr());
  }

  // check that we'd have no remainders - prefer a smaller vector size with no
  // remainders over a larger vector and remainder.
  bool can_vectorize = true;
  do {
    can_vectorize = self.numel() % vec_size == 0 &&
        ret.numel() % vec_size == 0 && mask.numel() % vec_size == 0;
    if (!can_vectorize)
      vec_size /= 2;
  } while (vec_size > 1 && !can_vectorize);
  return can_vectorize ? vec_size : 1;
}

template <typename index_type, typename mask_t>
inline void launcher(
    const Tensor& input1,
    const Tensor& bias1,
    const Tensor& residual1,
    const Tensor& input2,
    const Tensor& bias2,
    const Tensor& residual2,
    Tensor& ret1,
    Tensor& mask1,
    Tensor& ret2,
    Tensor& mask2,
    double p,
    const int64_t nelem,
    const PhiloxCudaState rng_engine_inputs,
    dim3 grid,
    dim3 dim_block) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto stream1 = at::cuda::getStreamFromPool().stream();
  
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input1.scalar_type(),
      "fused_dropout",
      [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        accscalar_t pa = (accscalar_t)(p);
        auto input1_info =
            cuda::detail::getTensorInfo<scalar_t, index_type>(input1);
        auto bias1_info =
            cuda::detail::getTensorInfo<scalar_t, index_type>(bias1);
        auto residual1_info =
            cuda::detail::getTensorInfo<scalar_t, index_type>(residual1);
        auto ret1_info =
            cuda::detail::getTensorInfo<scalar_t, index_type>(ret1);
        auto mask1_info =
            cuda::detail::getTensorInfo<mask_t, index_type>(mask1);
        input1_info.collapseDims();
        bias1_info.collapseDims();
        residual1_info.collapseDims();
        ret1_info.collapseDims();
        mask1_info.collapseDims(); // ret and mask are collapsed to 1d
                                   // contiguous tensor

        auto input2_info =
            cuda::detail::getTensorInfo<scalar_t, index_type>(input2);
        auto bias2_info =
            cuda::detail::getTensorInfo<scalar_t, index_type>(bias2);
        auto residual2_info =
            cuda::detail::getTensorInfo<scalar_t, index_type>(residual2);
        auto ret2_info =
            cuda::detail::getTensorInfo<scalar_t, index_type>(ret2);
        auto mask2_info =
            cuda::detail::getTensorInfo<mask_t, index_type>(mask2);
        input2_info.collapseDims();
        bias2_info.collapseDims();
        residual2_info.collapseDims();
        ret2_info.collapseDims();
        mask2_info.collapseDims(); // ret and mask are collapsed to 1d
                                   // contiguous tensor

        int vec_size = get_vector_size<scalar_t>(input1, ret1, mask1);

        if (vec_size > 1) {
          switch (vec_size) {
            case 4:
              fused_dropout_kernel_vec<scalar_t, accscalar_t, index_type, 1, 4>
                  <<<grid, dim_block, 0, stream>>>(
                      input1_info,
                      bias1_info,
                      residual1_info,
                      ret1_info,
                      mask1_info,
                      nelem,
                      pa,
                      rng_engine_inputs);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              fused_dropout_kernel_vec<scalar_t, accscalar_t, index_type, 1, 4>
                  <<<grid, dim_block, 0, stream1>>>(
                      input2_info,
                      bias2_info,
                      residual2_info,
                      ret2_info,
                      mask2_info,
                      nelem,
                      pa,
                      rng_engine_inputs);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              break;
            case 2:
              fused_dropout_kernel_vec<scalar_t, accscalar_t, index_type, 1, 2>
                  <<<grid, dim_block, 0, stream>>>(
                      input1_info,
                      bias1_info,
                      residual1_info,
                      ret1_info,
                      mask1_info,
                      nelem,
                      pa,
                      rng_engine_inputs);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              fused_dropout_kernel_vec<scalar_t, accscalar_t, index_type, 1, 2>
                  <<<grid, dim_block, 0, stream1>>>(
                      input2_info,
                      bias2_info,
                      residual2_info,
                      ret2_info,
                      mask2_info,
                      nelem,
                      pa,
                      rng_engine_inputs);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              break;
          }
        } else {
          switch (input1_info.dims) {
            case 1:
              fused_dropout_kernel<scalar_t, accscalar_t, index_type, 1>
                  <<<grid, dim_block, 0, stream>>>(
                      input1_info,
                      bias1_info,
                      residual1_info,
                      ret1_info,
                      mask1_info,
                      nelem,
                      pa,
                      rng_engine_inputs);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              fused_dropout_kernel<scalar_t, accscalar_t, index_type, 1>
                  <<<grid, dim_block, 0, stream1>>>(
                      input2_info,
                      bias2_info,
                      residual2_info,
                      ret2_info,
                      mask2_info,
                      nelem,
                      pa,
                      rng_engine_inputs);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              break;
            default:
              if (!input1.is_contiguous() && !bias1.is_contiguous() &&
                  !residual1.is_contiguous() && ret1.is_contiguous() &&
                  mask1.is_contiguous()) {
                fused_dropout_kernel<scalar_t, accscalar_t, index_type, -1, 1>
                    <<<grid, dim_block, 0, stream>>>(
                        input1_info,
                        bias1_info,
                        residual1_info,
                        ret1_info,
                        mask1_info,
                        nelem,
                        pa,
                        rng_engine_inputs);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              } else {
                fused_dropout_kernel<scalar_t, accscalar_t, index_type, -1>
                    <<<grid, dim_block, 0, stream>>>(
                        input1_info,
                        bias1_info,
                        residual1_info,
                        ret1_info,
                        mask1_info,
                        nelem,
                        pa,
                        rng_engine_inputs);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }
              if (!input2.is_contiguous() && !bias2.is_contiguous() &&
                  !residual2.is_contiguous() && ret2.is_contiguous() &&
                  mask2.is_contiguous()) {
                fused_dropout_kernel<scalar_t, accscalar_t, index_type, -1, 1>
                    <<<grid, dim_block, 0, stream1>>>(
                        input2_info,
                        bias2_info,
                        residual2_info,
                        ret2_info,
                        mask2_info,
                        nelem,
                        pa,
                        rng_engine_inputs);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              } else {
                fused_dropout_kernel<scalar_t, accscalar_t, index_type, -1>
                    <<<grid, dim_block, 0, stream1>>>(
                        input2_info,
                        bias2_info,
                        residual2_info,
                        ret2_info,
                        mask2_info,
                        nelem,
                        pa,
                        rng_engine_inputs);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }
          }
        }
      });
}

} // anonymous namespace

template <typename mask_t>
std::tuple<Tensor, Tensor, Tensor, Tensor> dropout_cuda(
    CUDAGeneratorImpl* gen,
    const Tensor& input1,
    const Tensor& bias1,
    const Tensor& residual1,
    const Tensor& input2,
    const Tensor& bias2,
    const Tensor& residual2,
    double p) {
  Tensor mask1 = at::empty_like(
      input1, input1.options().dtype(c10::CppTypeToScalarType<mask_t>::value));
  Tensor mask2 = at::empty_like(
      input2, input2.options().dtype(c10::CppTypeToScalarType<mask_t>::value));
  const int64_t nelem = input1.numel();
  // empty tensors should not get here, but just in case, avoid FPE
  // non-training shot-cut
  if (nelem == 0)
    return std::tuple<Tensor, Tensor, Tensor, Tensor>(
        input1.clone(), mask1, input2.clone(), mask2);

  Tensor ret1 = at::empty_like(input1);
  Tensor ret2 = at::empty_like(input2);
  const int64_t block_size = 256;
  unsigned int blocks_per_sm =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor /
      block_size;
  dim3 dim_block(block_size);
  dim3 grid((nelem + block_size - 1) / block_size);
  grid.x = std::min(
      (unsigned int)at::cuda::getCurrentDeviceProperties()
              ->multiProcessorCount *
          blocks_per_sm,
      grid.x);
  // number of times random will be generated per thread, to offset philox
  // counter in thc random state
  int64_t counter_offset =
      ((nelem - 1) / (block_size * grid.x * UNROLL) + 1) * UNROLL;
  PhiloxCudaState rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(counter_offset);
  }
  if (cuda::detail::canUse32BitIndexMath(input1)) {
    launcher<unsigned int, mask_t>(
        input1,
        bias1,
        residual1,
        input2,
        bias2,
        residual2,
        ret1,
        mask1,
        ret2,
        mask2,
        p,
        nelem,
        rng_engine_inputs,
        grid,
        dim_block);
  } else {
    launcher<uint64_t, mask_t>(
        input1,
        bias1,
        residual1,
        input2,
        bias2,
        residual2,
        ret1,
        mask1,
        ret2,
        mask2,
        p,
        nelem,
        rng_engine_inputs,
        grid,
        dim_block);
  }
  // TODO: remove the clone() calls
  // ret2 = ret1.clone();
  // mask2 = mask1.clone();
  return std::tuple<Tensor, Tensor, Tensor, Tensor>(ret1, mask1, ret2, mask2);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> native_add_dropout_add_fuse_cuda(
    const Tensor& input1,
    const Tensor& bias1,
    const Tensor& residual1,
    const Tensor& input2,
    const Tensor& bias2,
    const Tensor& residual2,
    double p,
    c10::optional<bool> train) {
  // short-cut for train == false
  if (train.has_value() && !train.value()) {
    auto ret1 = input1.add(bias1).add(residual1);
    auto ret2 = input2.add(bias2).add(residual2);
    return std::make_tuple(
        ret1,
        at::ones_like(
            input1,
            input1.options().dtype(c10::CppTypeToScalarType<bool>::value)),
        ret2,
        at::ones_like(
            input2,
            input2.options().dtype(c10::CppTypeToScalarType<bool>::value)));
  }
  // short-cut
  if (p == 1) {
    // native_dropout_cuda is in derivatives.yaml, so we don't need to add data
    // dependency from output to input for autograd
    auto ret1 = at::zeros_like(input1);
    auto mask1 = at::zeros_like(
        input1, input1.options().dtype(c10::CppTypeToScalarType<bool>::value));
    auto ret2 = at::zeros_like(input2);
    auto mask2 = at::zeros_like(
        input2, input2.options().dtype(c10::CppTypeToScalarType<bool>::value));
    return std::tuple<Tensor, Tensor, Tensor, Tensor>(ret1, mask1, ret1, mask1);
  }

  auto gen = get_generator_or_default<CUDAGeneratorImpl>(
      c10::nullopt, cuda::detail::getDefaultCUDAGenerator());
  double p1m = 1. - p;
  return dropout_cuda<bool>(
      gen, input1, bias1, residual1, input2, bias2, residual2, p1m);
}

template <typename mask_t>
Tensor dropout_backward_cuda(
    const Tensor& grad,
    const Tensor& mask,
    double scale) {
  Tensor ret = at::empty_like(grad, grad.suggest_memory_format());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      ret.scalar_type(),
      "masked_scale",
      [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        masked_scale_kernel<mask_t, scalar_t>(
            ret, grad, mask, (accscalar_t)scale);
      });
  return ret;
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
native_add_dropout_add_fuse_2_cuda(
    const Tensor& grad_output1,
    const Tensor& mask1,
    const Tensor& grad_output2,
    const Tensor& mask2,
    double scale) {
  TORCH_CHECK(
      mask1.scalar_type() == at::ScalarType::Bool,
      "Mask should be Bool Scalar Type",
      mask1.scalar_type());
  Tensor grad_input1 = dropout_backward_cuda<bool>(grad_output1, mask1, scale);
  Tensor grad_bias1 = grad_input1;
  Tensor grad_residual1 = grad_output1;
  Tensor grad_input2 = dropout_backward_cuda<bool>(grad_output2, mask2, scale);
  Tensor grad_bias2 = grad_input2;
  Tensor grad_residual2 = grad_output2;
  return std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>(
      grad_input1,
      grad_bias1,
      grad_residual1,
      grad_input2,
      grad_bias2,
      grad_residual2);
}

} // namespace at::native

