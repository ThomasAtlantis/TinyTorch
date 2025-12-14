/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "OpTransformCuda.cuh"

namespace tinytorch::op {

// Helper struct to avoid ambiguous conversions between __half and __nv_bfloat16
// Use float as intermediate type for conversions involving __half or __nv_bfloat16
template <typename SrcT, typename DstT>
struct SafeCast {
  __device__ __forceinline__ static DstT cast(const SrcT& value) {
    return static_cast<DstT>(value);
  }
};

// Specializations for __half to __nv_bfloat16
template <>
struct SafeCast<__half, __nv_bfloat16> {
  __device__ __forceinline__ static __nv_bfloat16 cast(const __half& value) {
    return __float2bfloat16(__half2float(value));
  }
};

// Specializations for __half to int64_t
template <>
struct SafeCast<__half, int64_t> {
  __device__ __forceinline__ static int64_t cast(const __half& value) {
    return static_cast<int64_t>(__half2float(value));
  }
};

// Specializations for __half to uint8_t
template <>
struct SafeCast<__half, uint8_t> {
  __device__ __forceinline__ static uint8_t cast(const __half& value) {
    return static_cast<uint8_t>(__half2float(value));
  }
};

// Specializations for __nv_bfloat16 to __half
template <>
struct SafeCast<__nv_bfloat16, __half> {
  __device__ __forceinline__ static __half cast(const __nv_bfloat16& value) {
    return __float2half(__bfloat162float(value));
  }
};

// Specializations for __nv_bfloat16 to int64_t
template <>
struct SafeCast<__nv_bfloat16, int64_t> {
  __device__ __forceinline__ static int64_t cast(const __nv_bfloat16& value) {
    return static_cast<int64_t>(__bfloat162float(value));
  }
};

// Specializations for __nv_bfloat16 to uint8_t
template <>
struct SafeCast<__nv_bfloat16, uint8_t> {
  __device__ __forceinline__ static uint8_t cast(const __nv_bfloat16& value) {
    return static_cast<uint8_t>(__bfloat162float(value));
  }
};

// Specializations for int64_t to __half
template <>
struct SafeCast<int64_t, __half> {
  __device__ __forceinline__ static __half cast(const int64_t& value) {
    return __float2half(static_cast<float>(value));
  }
};

// Specializations for int64_t to __nv_bfloat16
template <>
struct SafeCast<int64_t, __nv_bfloat16> {
  __device__ __forceinline__ static __nv_bfloat16 cast(const int64_t& value) {
    return __float2bfloat16(static_cast<float>(value));
  }
};

template <typename SrcT, typename DstT>
__global__ void kDtypeCast(const SrcT* src, DstT* dst, int64_t numel) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    dst[idx] = SafeCast<SrcT, DstT>::cast(src[idx]);
  }
}

template <typename SrcT, typename DstT>
void dtypeCastCudaKernelLauncher(const void* src, void* dst, int64_t numel, const Device& device) {
  using CudaSrcT = typename cuda::CudaTypeCast<SrcT>::type;
  using CudaDstT = typename cuda::CudaTypeCast<DstT>::type;

  const auto* srcPtr = static_cast<const CudaSrcT*>(src);
  auto* dstPtr = static_cast<CudaDstT*>(dst);

  auto params = cuda::getKernelLaunchParams(device.index, numel);
  CUDA_LAUNCH_KERNEL((kDtypeCast<CudaSrcT, CudaDstT>), params, srcPtr, dstPtr, numel);
}

template <typename SrcT>
struct DTypeDstDispatchCuda {
  const Tensor& src;
  Tensor& dst;

  void operator()() const { dtypeCastDispatch(dst.dtype(), *this); }

  template <typename DstT>
  void operator()() const {
    dtypeCastCudaKernelLauncher<SrcT, DstT>(src.dataPtr(), dst.dataPtr(), src.numel(), src.device());
  }
};

struct DTypeSrcDispatchCuda {
  const Tensor& src;
  Tensor& dst;

  template <typename SrcT>
  void operator()() const {
    DTypeDstDispatchCuda<SrcT> dstDispatch{src, dst};
    dtypeCastDispatch(dst.dtype(), dstDispatch);
  }
};

void dtypeCastOpCudaImpl(Tensor& dst, const Tensor& src) {
  ASSERT(src.numel() == dst.numel());
  DTypeSrcDispatchCuda srcDispatch{src, dst};
  dtypeCastDispatch(src.dtype(), srcDispatch);
}

void registerTransformCuda() {
  // dtype cast
  REGISTER_OP_IMPL_ALL_DTYPES(dtypeCast, CUDA, dtypeCastOpCudaImpl);

  // check
  REGISTER_OP_IMPL_DTYPE_TPL(check, CUDA, checkOpCudaImpl);

  // permute
  REGISTER_OP_IMPL_DTYPE_TPL(permute, CUDA, permuteOpCudaImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(permuteAll, CUDA, permuteAllOpCudaImpl);

  // transpose
  REGISTER_OP_IMPL_DTYPE_TPL(transpose, CUDA, transposeOpCudaImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(transpose2d, CUDA, transpose2dOpCudaImpl);

  // indexAdvance
  REGISTER_OP_IMPL_DTYPE_TPL(indexAdvance, CUDA, indexAdvanceOpCudaImpl);

  // indexPutAdvance
  REGISTER_OP_IMPL_DTYPE_TPL(indexPutAdvance, CUDA, indexPutAdvanceOpCudaImpl);

  // tril/triu
  REGISTER_OP_IMPL_DTYPE_TPL(tril, CUDA, trilOpCudaImpl);
  REGISTER_OP_IMPL_DTYPE_TPL(triu, CUDA, triuOpCudaImpl);

  // gather
  REGISTER_OP_IMPL_DTYPE_TPL(gather, CUDA, gatherOpCudaImpl)

  // scatter
  REGISTER_OP_IMPL_DTYPE_TPL(scatter, CUDA, scatterOpCudaImpl)
  REGISTER_OP_IMPL_DTYPE_TPL(scatterInplace, CUDA, scatterOpInplaceCudaImpl)

  // expand
  REGISTER_OP_IMPL_DTYPE_TPL(expand, CUDA, expandOpCudaImpl)

  // indexSelect
  REGISTER_OP_IMPL_DTYPE_TPL(indexSelect, CUDA, indexSelectOpCudaImpl)

  // repeatInterleave
  REGISTER_OP_IMPL_DTYPE_TPL(repeatInterleave, CUDA, repeatInterleaveOpCudaImpl)
}

}  // namespace tinytorch::op