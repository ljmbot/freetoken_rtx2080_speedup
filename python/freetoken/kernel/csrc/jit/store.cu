#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>
#include <freetoken/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <concepts>
#include <cstddef>
#include <cstdint>

namespace {

struct StoreKernelParams {
  void *__restrict__ k_cache;
  void *__restrict__ v_cache;
  const void *__restrict__ indices;
  const void *__restrict__ k;
  const void *__restrict__ v;
  std::size_t kv_cache_stride_k;
  std::size_t kv_cache_stride_v;
  std::size_t kv_input_stride_k;
  std::size_t kv_input_stride_v;
  std::size_t length;
};

// kElementSize / vElementSize are the per-row byte counts of the K and V
// slabs. They differ when the V cache is sub-byte quantized (K8V4/K8V5:
// K rows are int8 x head_dim, V rows are packed VBITS x head_dim / 8).
template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kElementSize, std::size_t vElementSize,
          std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    store_kv_cache(const __grid_constant__ StoreKernelParams params) {
  using namespace device;

  constexpr auto kWarpPerBlock =
      static_cast<unsigned>(kNumThreads / kWarpThreads);
  static_assert(kNumThreads % kWarpThreads == 0);

  const auto &[k_cache, v_cache, indices, k, v, kv_cache_stride_k,
               kv_cache_stride_v, kv_input_stride_k, kv_input_stride_v,
               length] = params;
  const auto warp_id =
      (threadIdx.x / kWarpThreads) + blockIdx.x * kWarpPerBlock;
  PDL::wait<kUsePDL>();

  // each warp handles one element
  if (warp_id < length) {
    const auto pos = static_cast<const T *>(indices)[warp_id];
    const auto dst_k = pointer::offset(k_cache, pos * kv_cache_stride_k);
    const auto src_k = pointer::offset(k, warp_id * kv_input_stride_k);
    warp::copy<kElementSize>(dst_k, src_k);
    const auto dst_v = pointer::offset(v_cache, pos * kv_cache_stride_v);
    const auto src_v = pointer::offset(v, warp_id * kv_input_stride_v);
    warp::copy<vElementSize>(dst_v, src_v);
  }

  PDL::launch<kUsePDL>();
}

template <std::size_t kElementSize, std::size_t vElementSize,
          std::size_t num_threads = 128,   // number of threads per block
          std::size_t max_concurrency = 1, // max blocks per SM
          bool use_pdl = false>
struct StoreKernel {
  static void run(const tvm::ffi::TensorView k_cache,
                  const tvm::ffi::TensorView v_cache,
                  const tvm::ffi::TensorView indices,
                  const tvm::ffi::TensorView k, const tvm::ffi::TensorView v) {
    using namespace host;
    auto Dk = SymbolicSize{"Dk"}; // K element count per row
    auto Dv = SymbolicSize{"Dv"}; // V element count per row (packed when sub-byte)
    auto L = SymbolicSize{"L"};   // length
    auto Xk = SymbolicSize{"Xk"}; // stride kv cache (K)
    auto Xv = SymbolicSize{"Xv"}; // stride kv cache (V)
    auto Yk = SymbolicSize{"Yk"}; // stride kv input (K)
    auto Yv = SymbolicSize{"Yv"}; // stride kv input (V)
    auto indices_dtype_ = SymbolicDType{};
    auto k_dtype_ = SymbolicDType{};
    auto v_dtype_ = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({-1, Dk}) //
        .with_strides({Xk, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(k_dtype_)
        .verify(k_cache);
    TensorMatcher({-1, Dv}) //
        .with_strides({Xv, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(v_dtype_)
        .verify(v_cache);
    TensorMatcher({L, Dk}) //
        .with_strides({Yk, 1})
        .with_dtype(k_dtype_)
        .verify(k);
    TensorMatcher({L, Dv}) //
        .with_strides({Yv, 1})
        .with_dtype(v_dtype_)
        .verify(v);
    TensorMatcher({L}) //
        .with_device<kDLCUDA>(device_)
        .with_dtype<int32_t, int64_t>(indices_dtype_)
        .verify(indices);

    const auto k_dtype_size = dtype_bytes(k_dtype_.unwrap());
    const auto v_dtype_size = dtype_bytes(v_dtype_.unwrap());
    RuntimeCheck(kElementSize == k_dtype_size * Dk.unwrap());
    RuntimeCheck(vElementSize == v_dtype_size * Dv.unwrap());

    const auto device = device_.unwrap();
    const auto use_int32 = indices_dtype_.unwrap().bits == 32;
    const auto length = static_cast<std::size_t>(L.unwrap());
    const auto kv_cache_stride_k = Xk.unwrap() * k_dtype_size;
    const auto kv_cache_stride_v = Xv.unwrap() * v_dtype_size;
    const auto kv_input_stride_k = Yk.unwrap() * k_dtype_size;
    const auto kv_input_stride_v = Yv.unwrap() * v_dtype_size;

    const auto params = StoreKernelParams{
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .indices = indices.data_ptr(),
        .k = k.data_ptr(),
        .v = v.data_ptr(),
        .kv_cache_stride_k = kv_cache_stride_k,
        .kv_cache_stride_v = kv_cache_stride_v,
        .kv_input_stride_k = kv_input_stride_k,
        .kv_input_stride_v = kv_input_stride_v,
        .length = length,
    };

    constexpr auto kWarpPerBlock = num_threads / 32;
    static_assert(num_threads % 32 == 0);
    const auto num_blocks = div_ceil(length, kWarpPerBlock);
    const auto kernel = use_int32
                            ? store_kv_cache<num_threads, max_concurrency,
                                             use_pdl, kElementSize,
                                             vElementSize, int32_t>
                            : store_kv_cache<num_threads, max_concurrency,
                                             use_pdl, kElementSize,
                                             vElementSize, int64_t>;
    LaunchKernel(num_blocks, num_threads, device)
        .with_attr(use_pdl)(kernel, params);
  }
};

} // namespace
