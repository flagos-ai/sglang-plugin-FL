#include "../common.h"
#include "../device_utils.h"
#include "../norm/common.mu"

#include <musa_runtime.h>
#include <musa_bf16.h>
#include <musa_fp16.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

#ifndef SGL_CUSTOM_AR_THREADS
#define SGL_CUSTOM_AR_THREADS 512
#endif

#ifndef SGL_CUSTOM_AR_BLOCKS
#define SGL_CUSTOM_AR_BLOCKS 36
#endif

#ifndef SGL_CUSTOM_AR_VECTOR_LOAD
#define SGL_CUSTOM_AR_VECTOR_LOAD 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_H8192_BLOCKS
#define SGL_CUSTOM_AR_FUSED_RMSNORM_H8192_BLOCKS 16
#endif

#ifndef SGL_CUSTOM_AR_ATOMIC_BARRIER
#define SGL_CUSTOM_AR_ATOMIC_BARRIER 1
#endif

#ifndef SGL_CUSTOM_AR_MAX_BLOCKS
#define SGL_CUSTOM_AR_MAX_BLOCKS 120
#endif

#ifndef SGL_CUSTOM_AR_DYNAMIC_BLOCKS
#define SGL_CUSTOM_AR_DYNAMIC_BLOCKS 1
#endif

#ifndef SGL_CUSTOM_AR_RMSNORM_CACHE_HIDDEN_LIMIT
#define SGL_CUSTOM_AR_RMSNORM_CACHE_HIDDEN_LIMIT 2048
#endif

#ifndef SGL_CUSTOM_AR_RMSNORM_T_CACHE_HIDDEN_LIMIT
#define SGL_CUSTOM_AR_RMSNORM_T_CACHE_HIDDEN_LIMIT 8192
#endif

#ifndef SGL_CUSTOM_AR_RMSNORM_WEIGHT_CACHE_HIDDEN_LIMIT
#define SGL_CUSTOM_AR_RMSNORM_WEIGHT_CACHE_HIDDEN_LIMIT 8192
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_TOKEN_2STAGE
#define SGL_CUSTOM_AR_FUSED_RMSNORM_TOKEN_2STAGE 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_SHFL_2STAGE
#define SGL_CUSTOM_AR_FUSED_RMSNORM_SHFL_2STAGE 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_PACKED_1STAGE
#define SGL_CUSTOM_AR_FUSED_RMSNORM_PACKED_1STAGE 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_SAFE_PACKED_1STAGE
#define SGL_CUSTOM_AR_FUSED_RMSNORM_SAFE_PACKED_1STAGE 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2RANK_1STAGE
#define SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2RANK_1STAGE 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_PARTIAL_PACKED_NON8
#define SGL_CUSTOM_AR_FUSED_RMSNORM_PARTIAL_PACKED_NON8 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2_NON8
#define SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2_NON8 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_VEC4_NON8
#define SGL_CUSTOM_AR_FUSED_RMSNORM_VEC4_NON8 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_REGCACHE_2RANK
#define SGL_CUSTOM_AR_FUSED_RMSNORM_REGCACHE_2RANK 1
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WEIGHT_CACHE
#define SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WEIGHT_CACHE 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WEIGHT_CACHE_MIN_ROWS
#define SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WEIGHT_CACHE_MIN_ROWS 2048
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SHARED_INV_RMS
#define SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SHARED_INV_RMS 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WARP_INV_RMS
#define SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WARP_INV_RMS 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SKIP_END_BARRIER
#define SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SKIP_END_BARRIER 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD
#define SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_SMALL_ROWS
#define SGL_CUSTOM_AR_FUSED_RMSNORM_SMALL_ROWS 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_WARP_ROWS
#define SGL_CUSTOM_AR_FUSED_RMSNORM_WARP_ROWS 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_POLLING
#define SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_POLLING 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_MIN_ROWS
#define SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_MIN_ROWS 1
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_MAX_ROWS
#define SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_MAX_ROWS 128
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_SLOTS
#define SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_SLOTS 2
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_SKIP_END_BARRIER
#define SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_SKIP_END_BARRIER 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_LAMPORT_PUSH
#define SGL_CUSTOM_AR_FUSED_RMSNORM_LAMPORT_PUSH 0
#endif

#ifndef SGL_CUSTOM_AR_FUSED_RMSNORM_LAMPORT_END_BARRIER
#define SGL_CUSTOM_AR_FUSED_RMSNORM_LAMPORT_END_BARRIER 1
#endif

#ifndef SGL_CUSTOM_AR_PUSH_16B_ASM
#define SGL_CUSTOM_AR_PUSH_16B_ASM 0
#endif

constexpr int kMaxBlocks = SGL_CUSTOM_AR_MAX_BLOCKS;
constexpr int kMaxThreadsPerBlock = 1024;
constexpr int kDefaultThreads = SGL_CUSTOM_AR_THREADS;
constexpr int kDefaultBlockLimit = SGL_CUSTOM_AR_BLOCKS;
constexpr int kH4096BlockLimit = 32;
constexpr int kH8192BlockLimit = SGL_CUSTOM_AR_FUSED_RMSNORM_H8192_BLOCKS;
constexpr int kCacheHiddenLimit = SGL_CUSTOM_AR_RMSNORM_CACHE_HIDDEN_LIMIT;
constexpr int kTypedCacheHiddenLimit = SGL_CUSTOM_AR_RMSNORM_T_CACHE_HIDDEN_LIMIT;
constexpr int kWeightCacheHiddenLimit = SGL_CUSTOM_AR_RMSNORM_WEIGHT_CACHE_HIDDEN_LIMIT;
constexpr bool kRowSharedInvRms = SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SHARED_INV_RMS != 0;
constexpr bool kRowWarpInvRms = SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WARP_INV_RMS != 0;
constexpr bool kRowSkipEndBarrier =
    SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_SKIP_END_BARRIER != 0;
constexpr int kMaxRanks = 8;
constexpr int kOneShotMaxToken = 128;
using FlagType = uint32_t;

__device__ __host__ __forceinline__ bool rmsnorm_cache_weight_hidden(
    int hidden,
    int rows) {
  return (hidden == 8192 || hidden == 1536) && hidden <= kWeightCacheHiddenLimit;
}

struct alignas(128) Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][kMaxRanks];
  alignas(128) FlagType peer_counter[2][kMaxBlocks][kMaxRanks];
  alignas(128) FlagType push_epoch[kMaxBlocks];
  alignas(128) FlagType lamport_counter;
  alignas(128) FlagType lamport_flag;
  alignas(128) FlagType lamport_clear_packed;
};

struct __align__(16) RankData {
  const void* ptrs[kMaxRanks];
};

struct __align__(16) RankSignals {
  Signal* signals[kMaxRanks];
};

__global__ void reset_signal_kernel(Signal* signal) {
  auto* words = reinterpret_cast<uint32_t*>(signal);
  constexpr int words_count = static_cast<int>(sizeof(Signal) / sizeof(uint32_t));
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < words_count;
       idx += blockDim.x * gridDim.x) {
    words[idx] = 0;
  }
}

int get_musa_sm_count() {
  static int sm_count = 0;
  if (sm_count == 0) {
    int device_id = 0;
    if (musaGetDevice(&device_id) == musaSuccess) {
      int queried = 0;
      if (musaDeviceGetAttribute(
              &queried, musaDevAttrMultiProcessorCount, device_id) ==
              musaSuccess &&
          queried > 0) {
        sm_count = queried;
      }
    }
    if (sm_count <= 0) {
      sm_count = kDefaultBlockLimit;
    }
  }
  return sm_count;
}

template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T>
struct packed_t {
  using P = array_t<T, 16 / sizeof(T)>;
  using A = array_t<float, 16 / sizeof(T)>;
};

template <typename T>
struct fp_trait {};

template <>
struct fp_trait<half> {
  using type = uint16_t;
  static constexpr uint16_t pos_zero = 0x0000u;
  static constexpr uint16_t neg_zero = 0x8000u;
};

template <>
struct fp_trait<__mt_bfloat16> {
  using type = uint16_t;
  static constexpr uint16_t pos_zero = 0x0000u;
  static constexpr uint16_t neg_zero = 0x8000u;
};

template <>
struct fp_trait<float> {
  using type = uint32_t;
  static constexpr uint32_t pos_zero = 0x00000000u;
  static constexpr uint32_t neg_zero = 0x80000000u;
};

template <typename DType>
__device__ __forceinline__ void clear_pos_zero(DType& val) {
  using Trait = fp_trait<DType>;
  auto* ptr = reinterpret_cast<typename Trait::type*>(&val);
  if (*ptr == Trait::pos_zero) {
    *ptr = Trait::neg_zero;
  }
}

template <typename DType>
__device__ __forceinline__ bool is_pos_zero(const DType& val) {
  using Trait = fp_trait<DType>;
  const auto* ptr = reinterpret_cast<const typename Trait::type*>(&val);
  return *ptr == Trait::pos_zero;
}

template <typename DType>
__device__ __forceinline__ DType get_pos_zero() {
  using Trait = fp_trait<DType>;
  const auto value = Trait::pos_zero;
  return *reinterpret_cast<const DType*>(&value);
}

template <typename T, int N>
__device__ __forceinline__ void clear_pos_zero(array_t<T, N>& val) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    clear_pos_zero(val.data[i]);
  }
}

template <typename T, int N>
__device__ __forceinline__ bool has_pos_zero(array_t<T, N> val) {
  bool found = false;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    found |= is_pos_zero(val.data[i]);
  }
  return found;
}

template <typename P>
__device__ __forceinline__ P make_pos_zero_packet() {
  P out;
#pragma unroll
  for (int i = 0; i < P::size; ++i) {
    out.data[i] = get_pos_zero<typename P::type>();
  }
  return out;
}

template <typename P>
__device__ __forceinline__ P load_volatile_packet(const P* ptr) {
  static_assert(sizeof(P) == 16 && alignof(P) == 16);
#if SGL_CUSTOM_AR_PUSH_16B_ASM
  using i4 = int __attribute__((ext_vector_type(4)));
  uint4 raw;
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310)
  i4 tmp = __musa_ldcv_v4i32(reinterpret_cast<const i4*>(ptr));
  raw = *reinterpret_cast<uint4*>(&tmp);
#else
  raw = *reinterpret_cast<const uint4*>(ptr);
#endif
  return *reinterpret_cast<const P*>(&raw);
#else
  uint4 raw;
  const auto* src = reinterpret_cast<const volatile uint32_t*>(ptr);
  raw.x = src[0];
  raw.y = src[1];
  raw.z = src[2];
  raw.w = src[3];
  return *reinterpret_cast<const P*>(&raw);
#endif
}

template <typename P>
__device__ __forceinline__ void store_volatile_packet(P* ptr, P value) {
  static_assert(sizeof(P) == 16 && alignof(P) == 16);
  const auto raw = *reinterpret_cast<const uint4*>(&value);
#if SGL_CUSTOM_AR_PUSH_16B_ASM
  using i4 = int __attribute__((ext_vector_type(4)));
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310)
  __musa_stwb_v4i32(*reinterpret_cast<const i4*>(&raw), reinterpret_cast<i4*>(ptr));
#else
  *reinterpret_cast<uint4*>(ptr) = raw;
#endif
#else
  auto* dst = reinterpret_cast<volatile uint32_t*>(ptr);
  dst[0] = raw.x;
  dst[1] = raw.y;
  dst[2] = raw.z;
  dst[3] = raw.w;
#endif
}

__device__ __forceinline__ float upcast_s(half value) {
  return __half2float(value);
}

__device__ __forceinline__ float upcast_s(__mt_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ float upcast_s(float value) {
  return value;
}

template <typename T>
__device__ __forceinline__ T downcast_s(float value);

template <>
__device__ __forceinline__ half downcast_s<half>(float value) {
  return __float2half(value);
}

template <>
__device__ __forceinline__ __mt_bfloat16 downcast_s<__mt_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <>
__device__ __forceinline__ float downcast_s<float>(float value) {
  return value;
}

template <typename T>
__device__ __forceinline__ T& assign_add(T& a, T b) {
  a = downcast_s<T>(upcast_s(a) + upcast_s(b));
  return a;
}

template <>
__device__ __forceinline__ float& assign_add<float>(float& a, float b) {
  a += b;
  return a;
}

template <typename T, int N>
__device__ __forceinline__ array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    assign_add(a.data[i], b.data[i]);
  }
  return a;
}

template <typename T, int N>
__device__ __forceinline__ array_t<float, N> upcast(array_t<T, N> value) {
  if constexpr (std::is_same<T, float>::value) {
    return value;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      out.data[i] = upcast_s(value.data[i]);
    }
    return out;
  }
}

template <typename O>
__device__ __forceinline__ O downcast(array_t<float, O::size> value) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return value;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; ++i) {
      out.data[i] = downcast_s<typename O::type>(value.data[i]);
    }
    return out;
  }
}

template <typename P, int nranks, typename A>
__device__ __forceinline__ P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < nranks; ++i) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}

template <typename P>
__device__ __forceinline__ P* get_tmp_buf(Signal* signal) {
  return reinterpret_cast<P*>(signal + 1);
}

template <typename T, int nranks, int vlen = 8>
__device__ __forceinline__ void shfl_reduce(float* res) {
  if constexpr (nranks >= 4) {
#pragma unroll
    for (int i = 0; i < vlen; ++i) {
      res[i] += __shfl_xor_sync(0xffffffff, res[i], 16);
    }
  }
#pragma unroll
  for (int i = 0; i < vlen; ++i) {
    res[i] += __shfl_xor_sync(0xffffffff, res[i], 8);
  }
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  return value;
}

__device__ __forceinline__ float block_rms_scale_16warps(
    float value,
    float* warp_sums,
    float* inv_rms,
    float inv_hidden,
    float eps) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = lane < 16 ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      *inv_rms = fast_rsqrt(value * inv_hidden + eps);
    }
  }
  __syncthreads_lm();
  return *inv_rms;
}

__device__ __forceinline__ float block_sum_16warps_tid0(float value, float* warp_sums) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = lane < 16 ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
  }
  return value;
}

__device__ __forceinline__ float block_sum_12warps(float value, float* warp_sums) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0 && warp < 12) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = lane < 12 ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  __syncthreads_lm();
  return warp_sums[0];
}

__device__ __forceinline__ float block_sum_12warps_tid0(float value, float* warp_sums) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0 && warp < 12) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = lane < 12 ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
  }
  return value;
}

template <int kActiveWarps>
__device__ __forceinline__ float block_sum_nwarps(float value, float* warp_sums) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0 && warp < kActiveWarps) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = lane < kActiveWarps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  __syncthreads_lm();
  return warp_sums[0];
}

template <int kActiveWarps>
__device__ __forceinline__ float block_sum_nwarps_tid0(
    float value,
    float* warp_sums) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0 && warp < kActiveWarps) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = lane < kActiveWarps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
  }
  return value;
}

__device__ __forceinline__ float block_sum_tid0(float value, float* warp_sums) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int num_warps = (static_cast<int>(blockDim.x) + 31) >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = tid < num_warps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
  }
  return value;
}

__device__ __forceinline__ float row_group_sum_tid0(
    float value,
    float* warp_sums,
    int packed_hidden) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int tid_in_row = tid % packed_hidden;
  const int row_group = tid / packed_hidden;
  const int warp_in_row = tid_in_row >> 5;
  const int warps_per_row = packed_hidden >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value =
      (warp_in_row == 0 && lane < warps_per_row)
          ? warp_sums[row_group * warps_per_row + lane]
          : 0.0f;
  if (warp_in_row == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
  }
  return value;
}

__device__ __forceinline__ float row_group_sum_all(
    float value,
    float* warp_sums,
    int packed_hidden) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row_group = tid / packed_hidden;
  const int warps_per_row = packed_hidden >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (i < warps_per_row) {
      sum += warp_sums[row_group * warps_per_row + i];
    }
  }
  __syncthreads_lm();
  return sum;
}

__device__ __forceinline__ float row_group_rms_scale_shared(
    float value,
    float* warp_sums,
    float* inv_rms,
    float inv_hidden,
    float eps,
    int packed_hidden) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warps_per_row = packed_hidden >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = lane < warps_per_row ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      *inv_rms = fast_rsqrt(value * inv_hidden + eps);
    }
  }
  __syncthreads_lm();
  return *inv_rms;
}

__device__ __forceinline__ float row_group_sum_all_wide(
    float value,
    float* warp_sums,
    int packed_hidden) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row_group = tid / packed_hidden;
  const int warps_per_row = packed_hidden >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    if (i < warps_per_row) {
      sum += warp_sums[row_group * warps_per_row + i];
    }
  }
  __syncthreads_lm();
  return sum;
}

constexpr size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

__device__ __forceinline__ void signal_store(FlagType* ptr, FlagType value) {
  volatile_store(static_cast<uint32_t>(value), reinterpret_cast<uint32_t*>(ptr));
}

__device__ __forceinline__ void signal_store_volatile(FlagType* ptr, FlagType value) {
  *reinterpret_cast<volatile FlagType*>(ptr) = value;
}

__device__ __forceinline__ FlagType signal_load(FlagType* ptr) {
  flushInv_byp();
  return static_cast<uint32_t>(volatile_load(reinterpret_cast<uint32_t*>(ptr)));
}

__device__ __forceinline__ FlagType signal_load_volatile(FlagType* ptr) {
  return *reinterpret_cast<volatile FlagType*>(ptr);
}

template <int nranks, bool start, bool fence = false>
__device__ __forceinline__ void multi_rank_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
  static_assert(!(start && fence));
  if constexpr (!start) {
    __syncthreads_lm();
  }
  if (threadIdx.x < nranks) {
#if SGL_CUSTOM_AR_ATOMIC_BARRIER
    auto flag = atomicAdd(&self_sg->self_counter[blockIdx.x][threadIdx.x], 1) + 1;
    auto* peer = &sg.signals[threadIdx.x]->peer_counter[flag & 1][blockIdx.x][rank];
    auto* local = &self_sg->peer_counter[flag & 1][blockIdx.x][threadIdx.x];
    atomicExch(peer, flag);
    while (atomicAdd(local, 0) != flag) {
    }
#else
    auto flag = self_sg->self_counter[blockIdx.x][threadIdx.x] + 1;
    self_sg->self_counter[blockIdx.x][threadIdx.x] = flag;
    auto* peer = &sg.signals[threadIdx.x]->peer_counter[flag & 1][blockIdx.x][rank];
    auto* local = &self_sg->peer_counter[flag & 1][blockIdx.x][threadIdx.x];
    if constexpr (fence) {
      signal_store(peer, flag);
      while (signal_load(local) != flag) {
      }
    } else {
      signal_store_volatile(peer, flag);
      while (signal_load_volatile(local) != flag) {
      }
    }
#endif
  }
  if constexpr (start || fence) {
    __syncthreads_lm();
  }
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_rmsnorm_kernel(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    T* __restrict__ norm_out,
    const T* __restrict__ weight,
    int rank,
    int rows,
    int hidden,
    float eps) {
  if (data_ptr != nullptr) {
    data = *data_ptr;
  }
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ float inv_rms;
  const int tid = threadIdx.x;
  const bool cache_float_values =
      hidden <= kCacheHiddenLimit &&
      !(nranks == 2 && hidden == 2048) &&
      !(nranks == 2 && hidden == 4096 && rows < 2048);
  const bool cache_t_values =
      !cache_float_values && !(nranks == 2 && hidden == 2048) &&
      hidden != 4096 && hidden <= kTypedCacheHiddenLimit;
  const bool cache_weight = rmsnorm_cache_weight_hidden(hidden, rows);
  float* cached = reinterpret_cast<float*>(smem);
  T* cached_t = reinterpret_cast<T*>(smem);
  const size_t cache_bytes =
      cache_float_values
          ? static_cast<size_t>(hidden) * sizeof(float)
          : (cache_t_values ? static_cast<size_t>(hidden) * sizeof(T) : 0);
  const size_t weight_offset = align_up(cache_bytes, alignof(T));
  T* cached_weight = reinterpret_cast<T*>(smem + weight_offset);
  const size_t weight_bytes = cache_weight ? static_cast<size_t>(hidden) * sizeof(T) : 0;
  float* warp_sums =
      reinterpret_cast<float*>(smem + align_up(weight_offset + weight_bytes, alignof(float)));
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack = P::size;
  const int packed_hidden = hidden / pack;
  const bool use_packed =
      (SGL_CUSTOM_AR_FUSED_RMSNORM_PACKED_1STAGE ||
       SGL_CUSTOM_AR_FUSED_RMSNORM_SAFE_PACKED_1STAGE ||
       SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2RANK_1STAGE) &&
      (hidden % pack) == 0;
  const bool use_partial_packed_non8 =
      (nranks == 4 || nranks == 8) && !use_packed &&
      SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2RANK_1STAGE &&
      SGL_CUSTOM_AR_FUSED_RMSNORM_PARTIAL_PACKED_NON8 &&
      hidden >= pack;
  const bool use_vec4_non8 =
      (nranks == 4 || nranks == 8) && !use_packed &&
      SGL_CUSTOM_AR_FUSED_RMSNORM_VEC4_NON8 &&
      !use_partial_packed_non8 && sizeof(T) == 2 && ((hidden & 3) == 0);
  const bool use_vec2_non8 =
      (nranks == 4 || nranks == 8) && !use_packed &&
      SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2_NON8 &&
      !use_partial_packed_non8 && !use_vec4_non8 && sizeof(T) == 2 &&
      ((hidden & 1) == 0);

  const P* ptrs[nranks];
#pragma unroll
  for (int i = 0; i < nranks; ++i) {
    ptrs[i] = reinterpret_cast<const P*>(data.ptrs[i]);
  }

  if (cache_weight && use_packed) {
    for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
      const int col0 = packed_col * pack;
      *(Vec8<T>*)(cached_weight + col0) = Vec8<T>::load(weight, col0);
    }
  }

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);

  if constexpr (nranks == 2) {
    if (SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2RANK_1STAGE &&
        hidden <= 2048 && (hidden % pack) == 0 &&
        (blockDim.x % packed_hidden) == 0) {
      const int rows_per_block = blockDim.x / packed_hidden;
      const int row_slot = tid / packed_hidden;
      const int packed_col = tid - row_slot * packed_hidden;
      const auto* local = reinterpret_cast<const P*>(data.ptrs[rank]);
      const auto* peer = reinterpret_cast<const P*>(data.ptrs[rank ^ 1]);
      const auto* residual = reinterpret_cast<const P*>(residual_in);
      auto* residual_dst = reinterpret_cast<P*>(residual_out);
      auto* norm_dst = reinterpret_cast<P*>(norm_out);
      const auto* weight_vec = reinterpret_cast<const P*>(weight);

      for (int row_base = blockIdx.x * rows_per_block; row_base < rows;
           row_base += gridDim.x * rows_per_block) {
        const int row = row_base + row_slot;
        float values[pack] = {0.0f};
        float square_sum = 0.0f;
        P residual_packet_out;
        if (row < rows) {
          const int packed_idx = row * packed_hidden + packed_col;
          const P local_vec = local[packed_idx];
          const P peer_vec = peer[packed_idx];
          const P residual_vec = residual[packed_idx];
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            float val = upcast_s(local_vec.data[i]);
            val += upcast_s(peer_vec.data[i]);
            val += upcast_s(residual_vec.data[i]);
            values[i] = val;
            residual_packet_out.data[i] = downcast_s<T>(val);
            square_sum += val * val;
          }
          residual_dst[packed_idx] = residual_packet_out;
        }

        const float row_square_sum =
            row_group_sum_all(square_sum, warp_sums, packed_hidden);
        const float scale =
            fast_rsqrt(row_square_sum / static_cast<float>(hidden) + eps);

        if (row < rows) {
          const int packed_idx = row * packed_hidden + packed_col;
          const P w = weight_vec[packed_col];
          P norm_packet;
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            const float gamma = upcast_s(w.data[i]);
            norm_packet.data[i] =
                downcast_s<T>(values[i] * scale * gamma);
          }
          norm_dst[packed_idx] = norm_packet;
        }
      }
      if constexpr (!kRowSkipEndBarrier) {
        multi_rank_barrier<nranks, false>(sg, self_sg, rank);
      }
      return;
    }
  }

  if constexpr (nranks == 4 || nranks == 8) {
    if (SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2RANK_1STAGE &&
        hidden <= 2048 && (hidden % pack) == 0 &&
        blockDim.x >= packed_hidden && (blockDim.x % packed_hidden) == 0) {
      const int rows_per_block = blockDim.x / packed_hidden;
      const int row_slot = tid / packed_hidden;
      const int packed_col = tid - row_slot * packed_hidden;
      const int col0 = packed_col * pack;
      for (int row_base = blockIdx.x * rows_per_block; row_base < rows;
           row_base += gridDim.x * rows_per_block) {
        const int row = row_base + row_slot;
        Float8 sum_float;
        float square_sum_row = 0.0f;
        if (row < rows) {
          float acc[pack] = {0.0f};
#pragma unroll
          for (int r = 0; r < nranks; ++r) {
            const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
            Vec8<T> x = Vec8<T>::load(peer + row * hidden, col0);
#pragma unroll
            for (int i = 0; i < pack; ++i) {
              acc[i] += to_float<T>(x.val.elem[i]);
            }
          }
          Vec8<T> residual_vec = Vec8<T>::load(residual_in + row * hidden, col0);
          Vec8<T> residual_vec_out;
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            float val = acc[i] + to_float<T>(residual_vec.val.elem[i]);
            sum_float.val.elem[i] = val;
            residual_vec_out.val.elem[i] = from_float<T>(val);
            square_sum_row += val * val;
          }
          *(Vec8<T>*)(residual_out + row * hidden + col0) = residual_vec_out;
        }

        const float row_square_sum =
            row_group_sum_all(square_sum_row, warp_sums, packed_hidden);
        const float scale =
            fast_rsqrt(row_square_sum / static_cast<float>(hidden) + eps);

        if (row < rows) {
          Vec8<T> w = Vec8<T>::load(weight, col0);
          Vec8<T> dst;
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            const float gamma = to_float<T>(w.val.elem[i]);
            dst.val.elem[i] =
                from_float<T>(sum_float.val.elem[i] * scale * gamma);
          }
          *(Vec8<T>*)(norm_out + row * hidden + col0) = dst;
        }
      }
      if constexpr (!kRowSkipEndBarrier) {
        multi_rank_barrier<nranks, false>(sg, self_sg, rank);
      }
      return;
    }

    if (SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2RANK_1STAGE &&
        (hidden == 4096 || hidden == 8192) && (hidden % pack) == 0 &&
        blockDim.x >= packed_hidden && (blockDim.x % packed_hidden) == 0) {
      const int rows_per_block = static_cast<int>(blockDim.x) / packed_hidden;
      const int row_slot = tid / packed_hidden;
      const int packed_col = tid - row_slot * packed_hidden;
      const int col0 = packed_col * pack;
      for (int row_base = blockIdx.x * rows_per_block; row_base < rows;
           row_base += gridDim.x * rows_per_block) {
        const int row = row_base + row_slot;
        Float8 sum_float;
        float square_sum_row = 0.0f;
        if (row < rows) {
          float acc[pack] = {0.0f};
#pragma unroll
          for (int r = 0; r < nranks; ++r) {
            const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
            Vec8<T> x = Vec8<T>::load(peer + row * hidden, col0);
#pragma unroll
            for (int i = 0; i < pack; ++i) {
              acc[i] += to_float<T>(x.val.elem[i]);
            }
          }
          Vec8<T> residual_vec = Vec8<T>::load(residual_in + row * hidden, col0);
          Vec8<T> residual_vec_out;
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            float val = acc[i] + to_float<T>(residual_vec.val.elem[i]);
            sum_float.val.elem[i] = val;
            residual_vec_out.val.elem[i] = from_float<T>(val);
            square_sum_row += val * val;
          }
          *(Vec8<T>*)(residual_out + row * hidden + col0) = residual_vec_out;
        }

        const float row_square_sum =
            packed_hidden <= 256
                ? row_group_sum_all(square_sum_row, warp_sums, packed_hidden)
                : row_group_sum_all_wide(square_sum_row, warp_sums, packed_hidden);
        const float scale =
            fast_rsqrt(row_square_sum / static_cast<float>(hidden) + eps);

        if (row < rows) {
          Vec8<T> w = Vec8<T>::load(weight, col0);
          Vec8<T> dst;
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            const float gamma = to_float<T>(w.val.elem[i]);
            dst.val.elem[i] =
                from_float<T>(sum_float.val.elem[i] * scale * gamma);
          }
          *(Vec8<T>*)(norm_out + row * hidden + col0) = dst;
        }
      }
      if constexpr (!kRowSkipEndBarrier) {
        multi_rank_barrier<nranks, false>(sg, self_sg, rank);
      }
      return;
    }
  }

  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int base = row * hidden;
    float square_sum = 0.0f;
    if constexpr (nranks == 2) {
      if (SGL_CUSTOM_AR_FUSED_RMSNORM_REGCACHE_2RANK &&
          (hidden % pack) == 0 && blockDim.x == packed_hidden &&
          packed_hidden <= kMaxThreadsPerBlock &&
          (hidden == 4096 || (hidden & (hidden - 1)) != 0)) {
        const auto* local = reinterpret_cast<const T*>(data.ptrs[rank]);
        const auto* peer = reinterpret_cast<const T*>(data.ptrs[rank ^ 1]);
        const int col0 = tid * pack;
        Float8 sum_float;
        Vec8<T> local_vec = Vec8<T>::load(local + base, col0);
        Vec8<T> peer_vec = Vec8<T>::load(peer + base, col0);
        Vec8<T> residual_vec = Vec8<T>::load(residual_in + base, col0);
        Vec8<T> residual_vec_out;
#pragma unroll
        for (int i = 0; i < pack; ++i) {
          float val = to_float<T>(local_vec.val.elem[i]);
          val += to_float<T>(peer_vec.val.elem[i]);
          val += to_float<T>(residual_vec.val.elem[i]);
          sum_float.val.elem[i] = val;
          residual_vec_out.val.elem[i] = from_float<T>(val);
          square_sum += val * val;
        }
        *(Vec8<T>*)(residual_out + base + col0) = residual_vec_out;

        const float row_square_sum = block_sum(square_sum, warp_sums);
        if (tid == 0) {
          inv_rms = fast_rsqrt(row_square_sum / static_cast<float>(hidden) + eps);
        }
        __syncthreads_lm();

        Vec8<T> w = Vec8<T>::load(weight, col0);
        Vec8<T> dst;
#pragma unroll
        for (int i = 0; i < pack; ++i) {
          const float gamma = to_float<T>(w.val.elem[i]);
          dst.val.elem[i] = from_float<T>(sum_float.val.elem[i] * inv_rms * gamma);
        }
        *(Vec8<T>*)(norm_out + base + col0) = dst;
        if (row + gridDim.x < rows) {
          __syncthreads_lm();
        }
        continue;
      }
    }
    if constexpr (nranks == 2) {
      if (SGL_CUSTOM_AR_FUSED_RMSNORM_VEC2RANK_1STAGE && (hidden % pack) == 0) {
        const auto* local = reinterpret_cast<const T*>(data.ptrs[rank]);
        const auto* peer = reinterpret_cast<const T*>(data.ptrs[rank ^ 1]);
        for (int col0 = tid * pack; col0 < hidden; col0 += blockDim.x * pack) {
          Vec8<T> local_vec = Vec8<T>::load(local + base, col0);
          Vec8<T> peer_vec = Vec8<T>::load(peer + base, col0);
          Vec8<T> residual_vec = Vec8<T>::load(residual_in + base, col0);
          Vec8<T> residual_vec_out;
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            const int col = col0 + i;
            float val = to_float<T>(local_vec.val.elem[i]);
            val += to_float<T>(peer_vec.val.elem[i]);
            val += to_float<T>(residual_vec.val.elem[i]);
            residual_vec_out.val.elem[i] = from_float<T>(val);
            if (cache_float_values) {
              cached[col] = val;
            } else if (cache_t_values) {
              cached_t[col] = residual_vec_out.val.elem[i];
            }
            square_sum += val * val;
          }
          *(Vec8<T>*)(residual_out + base + col0) = residual_vec_out;
        }
      } else if (use_packed && SGL_CUSTOM_AR_FUSED_RMSNORM_PACKED_1STAGE) {
        const int packed_base = row * packed_hidden;
        for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
          const int col0 = packed_col * pack;
          float acc[pack] = {0.0f};
#pragma unroll
          for (int r = 0; r < nranks; ++r) {
            const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
            Vec8<T> x = Vec8<T>::load_byp_slc(peer + base, col0);
#pragma unroll
            for (int i = 0; i < pack; ++i) {
              acc[i] += to_float<T>(x.val.elem[i]);
            }
          }
          Vec8<T> residual_vec = Vec8<T>::load_byp_slc(residual_in + base, col0);
          Vec8<T> residual_vec_out;
          Float8 sum_float;
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            const int col = col0 + i;
            float val = acc[i] + to_float<T>(residual_vec.val.elem[i]);
            residual_vec_out.val.elem[i] = from_float<T>(val);
            sum_float.val.elem[i] = val;
            if (cache_float_values) {
              cached[col] = val;
            } else if (cache_t_values) {
              cached_t[col] = residual_vec_out.val.elem[i];
            }
            square_sum += val * val;
          }
          *(Vec8<T>*)(residual_out + base + col0) = residual_vec_out;
        }
      } else {
        for (int col = tid; col < hidden; col += blockDim.x) {
          float ar_sum = 0.0f;
#pragma unroll
          for (int r = 0; r < nranks; ++r) {
            const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
            ar_sum += to_float(peer[base + col]);
          }
          const float val = ar_sum + to_float(residual_in[base + col]);
          const T residual_value = from_float<T>(val);
          residual_out[base + col] = residual_value;
          if (cache_float_values) {
            cached[col] = val;
          } else if (cache_t_values) {
            cached_t[col] = residual_value;
          }
          square_sum += val * val;
        }
      }
    } else if (use_packed && SGL_CUSTOM_AR_FUSED_RMSNORM_PACKED_1STAGE) {
      const int packed_base = row * packed_hidden;
      for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
        const int col0 = packed_col * pack;
        float acc[pack] = {0.0f};
#pragma unroll
        for (int r = 0; r < nranks; ++r) {
          const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
          Vec8<T> x = Vec8<T>::load_byp_slc(peer + base, col0);
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            acc[i] += to_float<T>(x.val.elem[i]);
          }
        }
        Vec8<T> residual_vec = Vec8<T>::load_byp_slc(residual_in + base, col0);
        Vec8<T> residual_vec_out;
        Float8 sum_float;
#pragma unroll
        for (int i = 0; i < pack; ++i) {
          const int col = col0 + i;
          float val = acc[i] + to_float<T>(residual_vec.val.elem[i]);
          residual_vec_out.val.elem[i] = from_float<T>(val);
          sum_float.val.elem[i] = val;
          if (cache_float_values) {
            cached[col] = val;
          } else if (cache_t_values) {
            cached_t[col] = residual_vec_out.val.elem[i];
          }
          square_sum += val * val;
        }
        *(Vec8<T>*)(residual_out + base + col0) = residual_vec_out;
      }
    } else if (use_packed) {
      const int packed_base = row * packed_hidden;
      for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
        const int packed_idx = packed_base + packed_col;
        const int col0 = packed_col * pack;
        P reduced = packed_reduce<P, nranks, A>(ptrs, packed_idx);
        P residual_packet = reinterpret_cast<const P*>(residual_in)[packed_idx];
        P residual_packet_out;
#pragma unroll
        for (int i = 0; i < pack; ++i) {
          const int col = col0 + i;
          float val = upcast_s(reduced.data[i]);
          val += upcast_s(residual_packet.data[i]);
          residual_packet_out.data[i] = downcast_s<T>(val);
          if (cache_float_values) {
            cached[col] = val;
          } else if (cache_t_values) {
            cached_t[col] = residual_packet_out.data[i];
          }
          square_sum += val * val;
        }
        reinterpret_cast<P*>(residual_out)[packed_idx] = residual_packet_out;
      }
    } else if (use_partial_packed_non8) {
      const int row_mod = base & (pack - 1);
      const int aligned_start = row_mod == 0 ? 0 : min(hidden, pack - row_mod);
      const int aligned_packed_hidden = (hidden - aligned_start) / pack;
      const int aligned_end = aligned_start + aligned_packed_hidden * pack;
      for (int col = tid; col < aligned_start; col += blockDim.x) {
        float ar_sum = 0.0f;
#pragma unroll
        for (int r = 0; r < nranks; ++r) {
          const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
          ar_sum += to_float(peer[base + col]);
        }
        const float val = ar_sum + to_float(residual_in[base + col]);
        const T residual_value = from_float<T>(val);
        residual_out[base + col] = residual_value;
        if (cache_float_values) {
          cached[col] = val;
        } else if (cache_t_values) {
          cached_t[col] = residual_value;
        }
        square_sum += val * val;
      }
      for (int packed_col = tid; packed_col < aligned_packed_hidden;
           packed_col += blockDim.x) {
        const int col0 = aligned_start + packed_col * pack;
        float acc[pack] = {0.0f};
#pragma unroll
        for (int r = 0; r < nranks; ++r) {
          const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
          Vec8<T> x = Vec8<T>::load_byp_slc(peer + base, col0);
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            acc[i] += to_float<T>(x.val.elem[i]);
          }
        }
        Vec8<T> residual_vec = Vec8<T>::load_byp_slc(residual_in + base, col0);
        Vec8<T> residual_vec_out;
#pragma unroll
        for (int i = 0; i < pack; ++i) {
          const int col = col0 + i;
          float val = acc[i] + to_float<T>(residual_vec.val.elem[i]);
          residual_vec_out.val.elem[i] = from_float<T>(val);
          if (cache_float_values) {
            cached[col] = val;
          } else if (cache_t_values) {
            cached_t[col] = residual_vec_out.val.elem[i];
          }
          square_sum += val * val;
        }
        *(Vec8<T>*)(residual_out + base + col0) = residual_vec_out;
      }
      for (int col = aligned_end + tid; col < hidden; col += blockDim.x) {
        float ar_sum = 0.0f;
#pragma unroll
        for (int r = 0; r < nranks; ++r) {
          const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
          ar_sum += to_float(peer[base + col]);
        }
        const float val = ar_sum + to_float(residual_in[base + col]);
        const T residual_value = from_float<T>(val);
        residual_out[base + col] = residual_value;
        if (cache_float_values) {
          cached[col] = val;
        } else if (cache_t_values) {
          cached_t[col] = residual_value;
        }
        square_sum += val * val;
      }
    } else if (use_vec4_non8) {
      using Vec4 = int64_t;
      const int packed4_hidden = hidden >> 2;
      const int base4 = row * packed4_hidden;
      const Vec4* ptrs4[nranks];
#pragma unroll
      for (int r = 0; r < nranks; ++r) {
        ptrs4[r] = reinterpret_cast<const Vec4*>(data.ptrs[r]);
      }
      const Vec4* residual4 = reinterpret_cast<const Vec4*>(residual_in);
      Vec4* residual_out4 = reinterpret_cast<Vec4*>(residual_out);
      Vec4* cached_t4 = reinterpret_cast<Vec4*>(cached_t);
      for (int packed_col = tid; packed_col < packed4_hidden;
           packed_col += blockDim.x) {
        float vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int r = 0; r < nranks; ++r) {
          const Vec4 raw = ptrs4[r][base4 + packed_col];
          const T* src = reinterpret_cast<const T*>(&raw);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            vals[i] += upcast_s(src[i]);
          }
        }
        const Vec4 residual_raw = residual4[base4 + packed_col];
        const T* residual_src = reinterpret_cast<const T*>(&residual_raw);
        Vec4 out;
        T* dst = reinterpret_cast<T*>(&out);
        const int col0 = packed_col << 2;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          vals[i] += upcast_s(residual_src[i]);
          dst[i] = downcast_s<T>(vals[i]);
          if (cache_float_values) {
            cached[col0 + i] = vals[i];
          }
          square_sum += vals[i] * vals[i];
        }
        residual_out4[base4 + packed_col] = out;
        if (cache_t_values) {
          cached_t4[packed_col] = out;
        }
      }
    } else if (use_vec2_non8) {
      using Vec2 = int32_t;
      const int packed2_hidden = hidden >> 1;
      const int base2 = row * packed2_hidden;
      const Vec2* ptrs2[nranks];
#pragma unroll
      for (int r = 0; r < nranks; ++r) {
        ptrs2[r] = reinterpret_cast<const Vec2*>(data.ptrs[r]);
      }
      const Vec2* residual2 = reinterpret_cast<const Vec2*>(residual_in);
      Vec2* residual_out2 = reinterpret_cast<Vec2*>(residual_out);
      Vec2* cached_t2 = reinterpret_cast<Vec2*>(cached_t);
      for (int packed_col = tid; packed_col < packed2_hidden;
           packed_col += blockDim.x) {
        float vals[2] = {0.0f, 0.0f};
#pragma unroll
        for (int r = 0; r < nranks; ++r) {
          const Vec2 raw = ptrs2[r][base2 + packed_col];
          const T* src = reinterpret_cast<const T*>(&raw);
          vals[0] += upcast_s(src[0]);
          vals[1] += upcast_s(src[1]);
        }
        const Vec2 residual_raw = residual2[base2 + packed_col];
        const T* residual_src = reinterpret_cast<const T*>(&residual_raw);
        vals[0] += upcast_s(residual_src[0]);
        vals[1] += upcast_s(residual_src[1]);

        Vec2 out;
        T* dst = reinterpret_cast<T*>(&out);
        dst[0] = downcast_s<T>(vals[0]);
        dst[1] = downcast_s<T>(vals[1]);
        residual_out2[base2 + packed_col] = out;
        const int col0 = packed_col << 1;
        if (cache_float_values) {
          cached[col0] = vals[0];
          cached[col0 + 1] = vals[1];
        } else if (cache_t_values) {
          cached_t2[packed_col] = out;
        }
        square_sum += vals[0] * vals[0] + vals[1] * vals[1];
      }
    } else {
      for (int col = tid; col < hidden; col += blockDim.x) {
        float ar_sum = 0.0f;
#pragma unroll
        for (int r = 0; r < nranks; ++r) {
          const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
          ar_sum += to_float(peer[base + col]);
        }
        const float val = ar_sum + to_float(residual_in[base + col]);
        const T residual_value = from_float<T>(val);
        residual_out[base + col] = residual_value;
        if (cache_float_values) {
          cached[col] = val;
        } else if (cache_t_values) {
          cached_t[col] = residual_value;
        }
        square_sum += val * val;
      }
    }

    const float row_square_sum =
        (nranks == 8 && hidden == 1536 && blockDim.x == 512)
            ? block_sum_nwarps<6>(square_sum, warp_sums)
            : block_sum(square_sum, warp_sums);
    if (tid == 0) {
      inv_rms = fast_rsqrt(row_square_sum / static_cast<float>(hidden) + eps);
    }
    __syncthreads_lm();

    if (use_packed) {
      for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
        const int col0 = packed_col * pack;
        Float8 sum_float;
        if (cache_float_values) {
          sum_float = *(Float8*)(cached + col0);
        } else if (cache_t_values) {
          Vec8<T> r = *(Vec8<T>*)(cached_t + col0);
          #pragma unroll
          for (int i = 0; i < pack; ++i) {
            sum_float.val.elem[i] = to_float<T>(r.val.elem[i]);
          }
        } else {
          Vec8<T> r = Vec8<T>::load(residual_out + base, col0);
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            sum_float.val.elem[i] = to_float<T>(r.val.elem[i]);
          }
        }
        Vec8<T> w = cache_weight ? *(Vec8<T>*)(cached_weight + col0) : Vec8<T>::load(weight, col0);
        Vec8<T> dst;
#pragma unroll
        for (int i = 0; i < pack; ++i) {
          const float gamma = to_float<T>(w.val.elem[i]);
          dst.val.elem[i] = from_float<T>(sum_float.val.elem[i] * inv_rms * gamma);
        }
        *(Vec8<T>*)(norm_out + base + col0) = dst;
      }
    } else if (use_partial_packed_non8) {
      const int row_mod = base & (pack - 1);
      const int aligned_start = row_mod == 0 ? 0 : min(hidden, pack - row_mod);
      const int aligned_packed_hidden = (hidden - aligned_start) / pack;
      const int aligned_end = aligned_start + aligned_packed_hidden * pack;
      for (int col = tid; col < aligned_start; col += blockDim.x) {
        float val;
        if (cache_float_values) {
          val = cached[col];
        } else if (cache_t_values) {
          val = to_float(cached_t[col]);
        } else {
          val = to_float(residual_out[base + col]);
        }
        const float gamma = to_float(weight[col]);
        norm_out[base + col] = from_float<T>(val * inv_rms * gamma);
      }
      for (int packed_col = tid; packed_col < aligned_packed_hidden;
           packed_col += blockDim.x) {
        const int col0 = aligned_start + packed_col * pack;
        Float8 sum_float;
        if (cache_float_values) {
          sum_float = *(Float8*)(cached + col0);
        } else if (cache_t_values) {
          Vec8<T> r = *(Vec8<T>*)(cached_t + col0);
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            sum_float.val.elem[i] = to_float<T>(r.val.elem[i]);
          }
        } else {
          Vec8<T> r = Vec8<T>::load(residual_out + base, col0);
#pragma unroll
          for (int i = 0; i < pack; ++i) {
            sum_float.val.elem[i] = to_float<T>(r.val.elem[i]);
          }
        }
        Vec8<T> w = Vec8<T>::load(weight, col0);
        Vec8<T> dst;
#pragma unroll
        for (int i = 0; i < pack; ++i) {
          const float gamma = to_float<T>(w.val.elem[i]);
          dst.val.elem[i] = from_float<T>(sum_float.val.elem[i] * inv_rms * gamma);
        }
        *(Vec8<T>*)(norm_out + base + col0) = dst;
      }
      for (int col = aligned_end + tid; col < hidden; col += blockDim.x) {
        float val;
        if (cache_float_values) {
          val = cached[col];
        } else if (cache_t_values) {
          val = to_float(cached_t[col]);
        } else {
          val = to_float(residual_out[base + col]);
        }
        const float gamma = to_float(weight[col]);
        norm_out[base + col] = from_float<T>(val * inv_rms * gamma);
      }
    } else if (use_vec4_non8) {
      using Vec4 = int64_t;
      const int packed4_hidden = hidden >> 2;
      const int base4 = row * packed4_hidden;
      const Vec4* cached_t4 = reinterpret_cast<const Vec4*>(cached_t);
      const Vec4* residual_out4 = reinterpret_cast<const Vec4*>(residual_out);
      const Vec4* weight4 = reinterpret_cast<const Vec4*>(weight);
      Vec4* norm_out4 = reinterpret_cast<Vec4*>(norm_out);
      for (int packed_col = tid; packed_col < packed4_hidden;
           packed_col += blockDim.x) {
        float vals[4];
        const int col0 = packed_col << 2;
        if (cache_float_values) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            vals[i] = cached[col0 + i];
          }
        } else {
          const Vec4 raw = cache_t_values ? cached_t4[packed_col]
                                          : residual_out4[base4 + packed_col];
          const T* src = reinterpret_cast<const T*>(&raw);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            vals[i] = upcast_s(src[i]);
          }
        }

        const Vec4 weight_raw = weight4[packed_col];
        const T* weight_src = reinterpret_cast<const T*>(&weight_raw);
        Vec4 out;
        T* dst = reinterpret_cast<T*>(&out);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          dst[i] = downcast_s<T>(vals[i] * inv_rms * upcast_s(weight_src[i]));
        }
        norm_out4[base4 + packed_col] = out;
      }
    } else if (use_vec2_non8) {
      using Vec2 = int32_t;
      const int packed2_hidden = hidden >> 1;
      const int base2 = row * packed2_hidden;
      const Vec2* cached_t2 = reinterpret_cast<const Vec2*>(cached_t);
      const Vec2* residual_out2 = reinterpret_cast<const Vec2*>(residual_out);
      const Vec2* weight2 = reinterpret_cast<const Vec2*>(weight);
      Vec2* norm_out2 = reinterpret_cast<Vec2*>(norm_out);
      for (int packed_col = tid; packed_col < packed2_hidden;
           packed_col += blockDim.x) {
        float vals[2];
        if (cache_float_values) {
          const int col0 = packed_col << 1;
          vals[0] = cached[col0];
          vals[1] = cached[col0 + 1];
        } else {
          const Vec2 raw = cache_t_values ? cached_t2[packed_col]
                                          : residual_out2[base2 + packed_col];
          const T* src = reinterpret_cast<const T*>(&raw);
          vals[0] = upcast_s(src[0]);
          vals[1] = upcast_s(src[1]);
        }

        const Vec2 weight_raw = weight2[packed_col];
        const T* weight_src = reinterpret_cast<const T*>(&weight_raw);
        Vec2 out;
        T* dst = reinterpret_cast<T*>(&out);
        dst[0] = downcast_s<T>(vals[0] * inv_rms * upcast_s(weight_src[0]));
        dst[1] = downcast_s<T>(vals[1] * inv_rms * upcast_s(weight_src[1]));
        norm_out2[base2 + packed_col] = out;
      }
    } else {
      for (int col = tid; col < hidden; col += blockDim.x) {
        float val;
        if (cache_float_values) {
          val = cached[col];
        } else if (cache_t_values) {
          val = to_float(cached_t[col]);
        } else {
          val = to_float(residual_out[base + col]);
        }
        const float gamma = to_float(weight[col]);
        norm_out[base + col] = from_float<T>(val * inv_rms * gamma);
      }
    }
    if (row + gridDim.x < rows) {
      __syncthreads_lm();
    }
  }

  if constexpr (!(kRowSkipEndBarrier && nranks == 2)) {
    multi_rank_barrier<nranks, false>(sg, self_sg, rank);
  }
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_rmsnorm_2stage_kernel(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    T* __restrict__ norm_out,
    const T* __restrict__ weight,
    int rank,
    int rows,
    int hidden,
    float eps,
    int packed_size) {
  if (data_ptr != nullptr) {
    data = *data_ptr;
  }
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack = P::size;

  extern __shared__ __align__(16) unsigned char smem[];

  const int tid = threadIdx.x;
  const bool cache_float_values = hidden <= kCacheHiddenLimit;
  const bool cache_t_values =
      !cache_float_values && hidden != 4096 && hidden <= kTypedCacheHiddenLimit;
  const bool cache_weight = rmsnorm_cache_weight_hidden(hidden, rows);
  float* cached = reinterpret_cast<float*>(smem);
  T* cached_t = reinterpret_cast<T*>(smem);
  const size_t cache_bytes =
      cache_float_values
          ? static_cast<size_t>(hidden) * sizeof(float)
          : (cache_t_values ? static_cast<size_t>(hidden) * sizeof(T) : 0);
  const size_t weight_offset = align_up(cache_bytes, alignof(T));
  T* cached_weight = reinterpret_cast<T*>(smem + weight_offset);
  const size_t weight_bytes = cache_weight ? static_cast<size_t>(hidden) * sizeof(T) : 0;
  float* warp_sums =
      reinterpret_cast<float*>(smem + align_up(weight_offset + weight_bytes, alignof(float)));

  const P* ptrs[nranks];
#pragma unroll
  for (int i = 0; i < nranks; ++i) {
    ptrs[i] = reinterpret_cast<const P*>(data.ptrs[i]);
  }
  const P* residual_p = reinterpret_cast<const P*>(residual_in);
  const P* weight_p = reinterpret_cast<const P*>(weight);
  P* residual_out_p = reinterpret_cast<P*>(residual_out);
  P* norm_out_p = reinterpret_cast<P*>(norm_out);
  P* self_tmp = get_tmp_buf<P>(self_sg);

  if (cache_weight) {
    for (int packed_col = tid; packed_col < hidden / pack; packed_col += blockDim.x) {
      const int col0 = packed_col * pack;
      *(Vec8<T>*)(cached_weight + col0) = Vec8<T>::load(weight, col0);
    }
  }

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);

  const int packed_hidden = hidden / pack;
  const int rows_per_rank = rows / nranks;
  const int row_remainder = rows % nranks;

  const int owner_begin =
      rank * rows_per_rank + (row_remainder > rank ? rank : row_remainder);
  const int owner_rows = rows_per_rank + (row_remainder > rank ? 1 : 0);
  const int owner_end = owner_begin + owner_rows;

  for (int row = owner_begin + blockIdx.x; row < owner_end; row += gridDim.x) {
    const int packed_base = row * packed_hidden;
    for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
      const int packed_idx = packed_base + packed_col;
      P reduced = packed_reduce<P, nranks, A>(ptrs, packed_idx);
      self_tmp[packed_idx] = reduced;
    }
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (tid == 0) {
    __threadfence_system_noflush();
  }
  multi_rank_barrier<nranks, false, true>(sg, self_sg, rank);

  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int base = row * hidden;
    const int packed_base = row * packed_hidden;
    float square_sum = 0.0f;
    int owner = row_remainder == 0 ? row / rows_per_rank : nranks - 1;
    int scan_begin = 0;
    if (row_remainder != 0) {
#pragma unroll
      for (int r = 0; r < nranks; ++r) {
        const int scan_rows = rows_per_rank + (row_remainder > r ? 1 : 0);
        if (row >= scan_begin && row < scan_begin + scan_rows) {
          owner = r;
        }
        scan_begin += scan_rows;
      }
    }
    const P* owner_tmp = get_tmp_buf<P>(sg.signals[owner]);

    for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
      const int packed_idx = packed_base + packed_col;
      P reduced = owner_tmp[packed_idx];
      P residual_packet = residual_p[packed_idx];
      P residual_packet_out;

#pragma unroll
      for (int i = 0; i < pack; ++i) {
        const int col = packed_col * pack + i;
        float val = upcast_s(reduced.data[i]);
        val += upcast_s(residual_packet.data[i]);
        T residual_value = downcast_s<T>(val);
        residual_packet_out.data[i] = residual_value;
        if (cache_float_values) {
          cached[col] = val;
        } else if (cache_t_values) {
          cached_t[col] = residual_value;
        }
        square_sum += val * val;
      }
      residual_out_p[packed_idx] = residual_packet_out;
    }

    const float row_square_sum = block_sum(square_sum, warp_sums);
    const float inv_rms = fast_rsqrt(row_square_sum / static_cast<float>(hidden) + eps);

    for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
      const int packed_idx = packed_base + packed_col;
      P weight_packet =
          cache_weight ? reinterpret_cast<const P*>(cached_weight)[packed_col]
                       : weight_p[packed_col];
      P norm_packet;
#pragma unroll
      for (int i = 0; i < pack; ++i) {
        const int col = packed_col * pack + i;
        float val;
        if (cache_float_values) {
          val = cached[col];
        } else if (cache_t_values) {
          val = upcast_s(cached_t[col]);
        } else {
          val = upcast_s(residual_out[base + col]);
        }
        const float gamma = upcast_s(weight_packet.data[i]);
        norm_packet.data[i] = downcast_s<T>(val * inv_rms * gamma);
      }
      norm_out_p[packed_idx] = norm_packet;
    }
    if (hidden != 3072) {
      __syncthreads_lm();
    }
  }

  if constexpr (!kRowSkipEndBarrier) {
    if (hidden != 3072) {
      multi_rank_barrier<nranks, false>(sg, self_sg, rank);
    }
  }
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_kernel(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    int rank,
    int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  constexpr int pack = P::size;
  const int packed_size = size / pack;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  const P* ptrs[nranks];
#pragma unroll
  for (int i = 0; i < nranks; ++i) {
    ptrs[i] = reinterpret_cast<const P*>(data.ptrs[i]);
  }
  const P* residual = reinterpret_cast<const P*>(residual_in);
  P* dst = reinterpret_cast<P*>(residual_out);

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  for (int idx = tid; idx < packed_size; idx += stride) {
    P reduced = packed_reduce<P, nranks, A>(ptrs, idx);
    P residual_packet = residual[idx];
    P out;
#pragma unroll
    for (int i = 0; i < pack; ++i) {
      out.data[i] =
          downcast_s<T>(upcast_s(reduced.data[i]) + upcast_s(residual_packet.data[i]));
    }
    dst[idx] = out;
  }
  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T, int nranks, int vlen = 8>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_2shot_kernel(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    int rank,
    int packed_size) {
  if (data_ptr != nullptr) {
    data = *data_ptr;
  }
  constexpr int nranks_sft = (nranks >> 1) - (nranks >> 3);
  constexpr int coalesce_num = 8;
  constexpr int coalesce_sft = 3;
  constexpr int group_stride_sft = nranks_sft + coalesce_sft;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int target_rank = (tid >> coalesce_sft) & (nranks - 1);
  const int group_id = tid >> group_stride_sft;
  const int coalesce_tid = tid & (coalesce_num - 1);
  const int stride = gridDim.x * blockDim.x;

  using Vec = array_t<T, vlen>;
  int idx_base = blockIdx.x * blockDim.x;
  int idx_in_blk = coalesce_tid + (rank << coalesce_sft) + (group_id << group_stride_sft);

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  Vec* target_ptr = reinterpret_cast<Vec*>(const_cast<void*>(data.ptrs[target_rank]));
  Vec* buffer_ptr = get_tmp_buf<Vec>(sg.signals[rank]);
  do {
    const int idx = idx_in_blk + idx_base;
    float acc[vlen] = {0};
    if (idx < packed_size) {
#if SGL_CUSTOM_AR_VECTOR_LOAD
      Vec raw = target_ptr[idx];
      const T* src = reinterpret_cast<const T*>(&raw);
#else
      const T* src = reinterpret_cast<const T*>(&target_ptr[idx]);
#endif
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        acc[i] = upcast_s(src[i]);
      }
    }
    shfl_reduce<T, nranks, vlen>(acc);
    if constexpr (nranks == 8) {
      __shared__ float smem[kMaxThreadsPerBlock << 1];
      if (lane < coalesce_num) {
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          smem[warp * vlen * coalesce_num + coalesce_tid * vlen + i] = acc[i];
        }
      }
      __syncthreads_lm();
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        acc[i] += smem[(warp ^ 1) * vlen * coalesce_num + coalesce_tid * vlen + i];
      }
    }
    if (rank == target_rank && idx < packed_size) {
      Vec res;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        reinterpret_cast<T*>(&res)[i] = downcast_s<T>(acc[i]);
      }
      buffer_ptr[idx] = res;
    }
    idx_base += stride;
  } while (idx_base < packed_size);

  __musa_barrier_slc();
  __syncthreads_lm();
  if (tid == 0) {
    __threadfence_system_noflush();
  }
  if (tid < nranks) {
#if SGL_CUSTOM_AR_ATOMIC_BARRIER
    auto flag = atomicAdd(&self_sg->self_counter[blockIdx.x][tid], 1) + 1;
    auto* peer = &sg.signals[tid]->peer_counter[flag & 1][blockIdx.x][rank];
    auto* local = &self_sg->peer_counter[flag & 1][blockIdx.x][tid];
    atomicExch(peer, flag);
    while (atomicAdd(local, 0) != flag) {
    }
#else
    auto flag = self_sg->self_counter[blockIdx.x][tid] + 1;
    self_sg->self_counter[blockIdx.x][tid] = flag;
    auto* peer = &sg.signals[tid]->peer_counter[flag & 1][blockIdx.x][rank];
    auto* local = &self_sg->peer_counter[flag & 1][blockIdx.x][tid];
    signal_store(peer, flag);
    while (signal_load(local) != flag) {
    }
#endif
  }
  __syncthreads_lm();

  buffer_ptr = get_tmp_buf<Vec>(sg.signals[target_rank]);
  const Vec* residual = reinterpret_cast<const Vec*>(residual_in);
  Vec* dst = reinterpret_cast<Vec*>(residual_out);
  idx_in_blk = coalesce_tid + (target_rank << coalesce_sft) + (group_id << group_stride_sft);
  idx_base = blockIdx.x * blockDim.x;
  do {
    const int idx = idx_in_blk + idx_base;
    if (idx < packed_size) {
      const Vec reduced = buffer_ptr[idx];
      const Vec residual_packet = residual[idx];
      Vec out;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        reinterpret_cast<T*>(&out)[i] = downcast_s<T>(
            upcast_s(reinterpret_cast<const T*>(&reduced)[i]) +
            upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]));
      }
      dst[idx] = out;
    }
    idx_base += stride;
  } while (idx_base < packed_size);

  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T, int nranks, int vlen = 8>
__global__ void __launch_bounds__(256, 1)
custom_all_reduce_residual_rmsnorm_warp_rows_kernel(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    T* __restrict__ norm_out,
    const T* __restrict__ weight,
    int rank,
    int rows,
    int hidden,
    float eps) {
  if (data_ptr != nullptr) {
    data = *data_ptr;
  }
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const float inv_hidden = 1.0f / static_cast<float>(hidden);

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);

  for (int row = blockIdx.x * warps_per_block + warp; row < rows;
       row += gridDim.x * warps_per_block) {
    const int base = row * hidden;
    float square_sum = 0.0f;
    for (int col0 = lane * vlen; col0 < hidden; col0 += 32 * vlen) {
      float acc[vlen] = {0.0f};
#pragma unroll
      for (int r = 0; r < nranks; ++r) {
        const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
        Vec8<T> x = Vec8<T>::load(peer + base, col0);
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          acc[i] += to_float<T>(x.val.elem[i]);
        }
      }
      Vec8<T> residual_vec = Vec8<T>::load(residual_in + base, col0);
      Vec8<T> residual_vec_out;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        const float val = acc[i] + to_float<T>(residual_vec.val.elem[i]);
        residual_vec_out.val.elem[i] = from_float<T>(val);
        square_sum += val * val;
      }
      *(Vec8<T>*)(residual_out + base + col0) = residual_vec_out;
    }

    const float row_square_sum = warp_sum(square_sum);
    const float scale =
        fast_rsqrt(__shfl_sync(0xffffffff, row_square_sum, 0, 32) * inv_hidden + eps);

    for (int col0 = lane * vlen; col0 < hidden; col0 += 32 * vlen) {
      Vec8<T> residual_vec = Vec8<T>::load(residual_out + base, col0);
      Vec8<T> weight_vec = Vec8<T>::load(weight, col0);
      Vec8<T> dst;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        const float val = to_float<T>(residual_vec.val.elem[i]);
        const float gamma = to_float<T>(weight_vec.val.elem[i]);
        dst.val.elem[i] = from_float<T>(val * scale * gamma);
      }
      *(Vec8<T>*)(norm_out + base + col0) = dst;
    }
  }

  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T, int nranks, int vlen = 8>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_sums_direct_kernel(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    float* __restrict__ row_sums,
    int rank,
    int rows,
    int hidden) {
  static_assert(nranks == 2, "direct residual sums is specialized for TP2");
  const int tid = threadIdx.x;
  constexpr int pack = vlen;
  const int packed_hidden = hidden / pack;
  __shared__ float warp_sums[(kMaxThreadsPerBlock + 31) / 32];

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);

  const auto* local = reinterpret_cast<const T*>(data.ptrs[rank]);
  const auto* peer = reinterpret_cast<const T*>(data.ptrs[rank ^ 1]);
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int base = row * hidden;
    float square_sum = 0.0f;
    for (int packed_col = tid; packed_col < packed_hidden;
         packed_col += blockDim.x) {
      const int col0 = packed_col * pack;
      Vec8<T> local_vec = Vec8<T>::load_byp_slc(local + base, col0);
      Vec8<T> peer_vec = Vec8<T>::load_byp_slc(peer + base, col0);
      Vec8<T> residual_vec = Vec8<T>::load_byp_slc(residual_in + base, col0);
      Vec8<T> residual_vec_out;
#pragma unroll
      for (int i = 0; i < pack; ++i) {
        const float val = to_float<T>(local_vec.val.elem[i]) +
                          to_float<T>(peer_vec.val.elem[i]) +
                          to_float<T>(residual_vec.val.elem[i]);
        residual_vec_out.val.elem[i] = from_float<T>(val);
        square_sum += val * val;
      }
      *(Vec8<T>*)(residual_out + base + col0) = residual_vec_out;
    }

    const float row_square_sum = block_sum_tid0(square_sum, warp_sums);
    if (tid == 0) {
      row_sums[row] = row_square_sum;
    }
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (tid == 0) {
    __threadfence_system_noflush();
  }
  __syncthreads_lm();
  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_sums_scalar_kernel(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    float* __restrict__ row_sums,
    int rank,
    int rows,
    int hidden) {
  const int tid = threadIdx.x;
  __shared__ float warp_sums[(kMaxThreadsPerBlock + 31) / 32];
  T* self_tmp = get_tmp_buf<T>(self_sg);

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);

  const T* ptrs[nranks];
#pragma unroll
  for (int r = 0; r < nranks; ++r) {
    ptrs[r] = reinterpret_cast<const T*>(data.ptrs[r]);
  }

  if (rows > 1) {
    if ((hidden & 1) == 0) {
      using Vec2 = int32_t;
      const int packed2_hidden = hidden >> 1;
      const Vec2* ptrs2[nranks];
#pragma unroll
      for (int r = 0; r < nranks; ++r) {
        ptrs2[r] = reinterpret_cast<const Vec2*>(ptrs[r]);
      }
      const Vec2* residual2 = reinterpret_cast<const Vec2*>(residual_in);
      Vec2* residual_out2 = reinterpret_cast<Vec2*>(residual_out);

      for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const int base2 = row * packed2_hidden;
        float square_sum = 0.0f;
        for (int packed_col = tid; packed_col < packed2_hidden;
             packed_col += blockDim.x) {
          float vals[2] = {0.0f, 0.0f};
#pragma unroll
          for (int r = 0; r < nranks; ++r) {
            const Vec2 raw = ptrs2[r][base2 + packed_col];
            const T* src = reinterpret_cast<const T*>(&raw);
            vals[0] += upcast_s(src[0]);
            vals[1] += upcast_s(src[1]);
          }
          const Vec2 residual_raw = residual2[base2 + packed_col];
          const T* residual_src = reinterpret_cast<const T*>(&residual_raw);
          vals[0] += upcast_s(residual_src[0]);
          vals[1] += upcast_s(residual_src[1]);

          Vec2 out;
          T* dst = reinterpret_cast<T*>(&out);
          dst[0] = downcast_s<T>(vals[0]);
          dst[1] = downcast_s<T>(vals[1]);
          residual_out2[base2 + packed_col] = out;
          square_sum += vals[0] * vals[0] + vals[1] * vals[1];
        }

        const float row_square_sum = block_sum_tid0(square_sum, warp_sums);
        if (tid == 0) {
          row_sums[row] = row_square_sum;
        }
      }

      __musa_barrier_slc();
      __syncthreads_lm();
      if (tid == 0) {
        __threadfence_system_noflush();
      }
      __syncthreads_lm();
      multi_rank_barrier<nranks, false>(sg, self_sg, rank);
      return;
    }

    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
      const int base = row * hidden;
      float square_sum = 0.0f;
      for (int col = tid; col < hidden; col += blockDim.x) {
        float val = upcast_s(residual_in[base + col]);
#pragma unroll
        for (int r = 0; r < nranks; ++r) {
          val += upcast_s(ptrs[r][base + col]);
        }
        residual_out[base + col] = downcast_s<T>(val);
        square_sum += val * val;
      }

      const float row_square_sum = block_sum_tid0(square_sum, warp_sums);
      if (tid == 0) {
        row_sums[row] = row_square_sum;
      }
    }

    __musa_barrier_slc();
    __syncthreads_lm();
    if (tid == 0) {
      __threadfence_system_noflush();
    }
    __syncthreads_lm();
    multi_rank_barrier<nranks, false>(sg, self_sg, rank);
    return;
  }

  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int base = row * hidden;
    for (int col = rank + tid * nranks; col < hidden;
         col += blockDim.x * nranks) {
      float val = 0.0f;
#pragma unroll
      for (int r = 0; r < nranks; ++r) {
        val += upcast_s(ptrs[r][base + col]);
      }
      self_tmp[base + col] = downcast_s<T>(val);
    }
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (tid == 0) {
    __threadfence_system_noflush();
  }
  multi_rank_barrier<nranks, false, true>(sg, self_sg, rank);

  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int base = row * hidden;
    float square_sum = 0.0f;
    for (int col = tid; col < hidden; col += blockDim.x) {
      const int owner = col & (nranks - 1);
      const T* owner_tmp = get_tmp_buf<T>(sg.signals[owner]);
      const float val =
          upcast_s(owner_tmp[base + col]) + upcast_s(residual_in[base + col]);
      residual_out[base + col] = downcast_s<T>(val);
      square_sum += val * val;
    }

    const float row_square_sum = block_sum_tid0(square_sum, warp_sums);
    if (tid == 0) {
      row_sums[row] = row_square_sum;
    }
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (tid == 0) {
    __threadfence_system_noflush();
  }
  __syncthreads_lm();
  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T, int nranks, int vlen = 8>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_2shot_sums_kernel(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    float* __restrict__ row_sums,
    int rank,
    int rows,
    int hidden,
    int packed_size) {
  constexpr int nranks_sft = (nranks >> 1) - (nranks >> 3);
  constexpr int coalesce_num = 8;
  constexpr int coalesce_sft = 3;
  constexpr int group_stride_sft = nranks_sft + coalesce_sft;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int target_rank = (tid >> coalesce_sft) & (nranks - 1);
  const int group_id = tid >> group_stride_sft;
  const int coalesce_tid = tid & (coalesce_num - 1);
  const int stride = gridDim.x * blockDim.x;
  const int packed_hidden = hidden / vlen;

  using Vec = array_t<T, vlen>;
  __shared__ float warp_sums[(kMaxThreadsPerBlock + 31) / 32];
  int idx_base = blockIdx.x * blockDim.x;
  int idx_in_blk = coalesce_tid + (rank << coalesce_sft) + (group_id << group_stride_sft);

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);

  Vec* target_ptr = reinterpret_cast<Vec*>(const_cast<void*>(data.ptrs[target_rank]));
  Vec* buffer_ptr = get_tmp_buf<Vec>(sg.signals[rank]);
  do {
    const int idx = idx_in_blk + idx_base;
    float acc[vlen] = {0};
    if (idx < packed_size) {
#if SGL_CUSTOM_AR_VECTOR_LOAD
      Vec raw = target_ptr[idx];
      const T* src = reinterpret_cast<const T*>(&raw);
#else
      const T* src = reinterpret_cast<const T*>(&target_ptr[idx]);
#endif
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        acc[i] = upcast_s(src[i]);
      }
    }
    shfl_reduce<T, nranks, vlen>(acc);
    if constexpr (nranks == 8) {
      __shared__ float smem[kMaxThreadsPerBlock << 1];
      if (lane < coalesce_num) {
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          smem[warp * vlen * coalesce_num + coalesce_tid * vlen + i] = acc[i];
        }
      }
      __syncthreads_lm();
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        acc[i] += smem[(warp ^ 1) * vlen * coalesce_num + coalesce_tid * vlen + i];
      }
    }
    if (rank == target_rank && idx < packed_size) {
      Vec res;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        reinterpret_cast<T*>(&res)[i] = downcast_s<T>(acc[i]);
      }
      buffer_ptr[idx] = res;
    }
    idx_base += stride;
  } while (idx_base < packed_size);

  __musa_barrier_slc();
  __syncthreads_lm();
  if (tid == 0) {
    __threadfence_system_noflush();
  }
  if (tid < nranks) {
#if SGL_CUSTOM_AR_ATOMIC_BARRIER
    auto flag = atomicAdd(&self_sg->self_counter[blockIdx.x][tid], 1) + 1;
    auto* peer = &sg.signals[tid]->peer_counter[flag & 1][blockIdx.x][rank];
    auto* local = &self_sg->peer_counter[flag & 1][blockIdx.x][tid];
    atomicExch(peer, flag);
    while (atomicAdd(local, 0) != flag) {
    }
#else
    auto flag = self_sg->self_counter[blockIdx.x][tid] + 1;
    self_sg->self_counter[blockIdx.x][tid] = flag;
    auto* peer = &sg.signals[tid]->peer_counter[flag & 1][blockIdx.x][rank];
    auto* local = &self_sg->peer_counter[flag & 1][blockIdx.x][tid];
    signal_store(peer, flag);
    while (signal_load(local) != flag) {
    }
#endif
  }
  __syncthreads_lm();

  buffer_ptr = get_tmp_buf<Vec>(sg.signals[target_rank]);
  const Vec* residual = reinterpret_cast<const Vec*>(residual_in);
  Vec* dst = reinterpret_cast<Vec*>(residual_out);
  if constexpr (nranks == 4 || nranks == 8) {
    if ((hidden == 1536 || hidden == 3072 || hidden == 4096) &&
        blockDim.x == 512) {
      const Vec* owner_tmp = get_tmp_buf<Vec>(sg.signals[target_rank]);
      for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const int idx = row * packed_hidden + tid;
        float square_sum = 0.0f;
        if (tid < packed_hidden) {
          const Vec reduced = owner_tmp[idx];
          const Vec residual_packet = residual[idx];
          Vec out;
#pragma unroll
          for (int i = 0; i < vlen; ++i) {
            const float val =
                upcast_s(reinterpret_cast<const T*>(&reduced)[i]) +
                upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
            reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
            square_sum += val * val;
          }
          dst[idx] = out;
        }
        const float row_square_sum =
            hidden == 4096
                ? block_sum_16warps_tid0(square_sum, warp_sums)
                : (hidden == 3072
                       ? block_sum_12warps_tid0(square_sum, warp_sums)
                       : block_sum_nwarps_tid0<6>(square_sum, warp_sums));
        if (tid == 0) {
          row_sums[row] = row_square_sum;
        }
      }

      if constexpr (!kRowSkipEndBarrier) {
        multi_rank_barrier<nranks, false>(sg, self_sg, rank);
      }
      return;
    }
  }
  idx_in_blk = coalesce_tid + (target_rank << coalesce_sft) + (group_id << group_stride_sft);
  idx_base = blockIdx.x * blockDim.x;
  do {
    const int idx = idx_in_blk + idx_base;
    float values[vlen] = {0};
    float square_sum = 0.0f;
    if (idx < packed_size) {
#if SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 1 || \
    SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 2
      const Vec8<T> reduced =
          Vec8<T>::load_byp_slc(reinterpret_cast<const T*>(buffer_ptr), idx * vlen);
#else
      const Vec reduced = buffer_ptr[idx];
#endif
#if SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 1 || \
    SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 3
      const Vec8<T> residual_packet = Vec8<T>::load_byp_slc(residual_in, idx * vlen);
#else
      const Vec residual_packet = residual[idx];
#endif
      Vec out;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
#if SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 1
        const float val =
            upcast_s(reinterpret_cast<const T*>(&reduced)[i]) +
            upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
#elif SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 2
        const float val =
            upcast_s(reinterpret_cast<const T*>(&reduced)[i]) +
            upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
#elif SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 3
        const float val =
            upcast_s(reinterpret_cast<const T*>(&reduced)[i]) +
            upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
#else
        const float val =
            upcast_s(reinterpret_cast<const T*>(&reduced)[i]) +
            upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
#endif
        reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
        square_sum += val * val;
      }
      dst[idx] = out;
    }
    const float row_square_sum = block_sum_tid0(square_sum, warp_sums);
    if (tid == 0) {
      const int row = idx_base / packed_hidden;
      if (row < rows) {
        row_sums[row] = row_square_sum;
      }
    }
    idx_base += stride;
  } while (idx_base < packed_size);

  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <
    typename T,
    int nranks,
    bool kSharedInvRms,
    bool kCacheWeightStatic,
    int vlen = 8>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_2shot_rmsnorm_row_kernel(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    T* __restrict__ norm_out,
    const T* __restrict__ weight,
    int rank,
    int rows,
    int hidden,
    float eps,
    int packed_size) {
  if (data_ptr != nullptr) {
    data = *data_ptr;
  }
  constexpr int nranks_sft = (nranks >> 1) - (nranks >> 3);
  constexpr int coalesce_num = 8;
  constexpr int coalesce_sft = 3;
  constexpr int group_stride_sft = nranks_sft + coalesce_sft;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int target_rank = (tid >> coalesce_sft) & (nranks - 1);
  const int group_id = tid >> group_stride_sft;
  const int coalesce_tid = tid & (coalesce_num - 1);
  const int stride = gridDim.x * blockDim.x;
  const int packed_hidden = hidden / vlen;
  const float inv_hidden = 1.0f / static_cast<float>(hidden);

  using Vec = array_t<T, vlen>;
  __shared__ float inv_rms;
  extern __shared__ __align__(16) unsigned char row_smem[];
  const size_t weight_bytes =
      kCacheWeightStatic ? static_cast<size_t>(hidden) * sizeof(T) : 0;
  T* cached_weight = reinterpret_cast<T*>(row_smem);
  float* warp_sums =
      reinterpret_cast<float*>(row_smem + align_up(weight_bytes, alignof(float)));
  const Vec* weight_vec = reinterpret_cast<const Vec*>(weight);

  int idx_base = blockIdx.x * blockDim.x;
  int idx_in_blk = coalesce_tid + (rank << coalesce_sft) + (group_id << group_stride_sft);

  if constexpr (kCacheWeightStatic) {
    for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
      const int col0 = packed_col * vlen;
      *(Vec*)(cached_weight + col0) = weight_vec[packed_col];
    }
    __syncthreads_lm();
  }
  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  if (vlen != packed_t<T>::P::size && (hidden % vlen) == 0 &&
      blockDim.x >= packed_hidden && packed_hidden <= kMaxThreadsPerBlock) {
    const Vec* residual = reinterpret_cast<const Vec*>(residual_in);
    Vec* residual_dst = reinterpret_cast<Vec*>(residual_out);
    Vec* norm_dst = reinterpret_cast<Vec*>(norm_out);
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
      const int packed_col = tid;
      const int packed_idx = row * packed_hidden + packed_col;
      float values[vlen] = {0};
      float square_sum = 0.0f;
      Vec out;
      if (packed_col < packed_hidden) {
#pragma unroll
        for (int r = 0; r < nranks; ++r) {
          const Vec packet = reinterpret_cast<const Vec*>(data.ptrs[r])[packed_idx];
#pragma unroll
          for (int i = 0; i < vlen; ++i) {
            values[i] += upcast_s(reinterpret_cast<const T*>(&packet)[i]);
          }
        }
        const Vec residual_packet = residual[packed_idx];
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          values[i] += upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
          reinterpret_cast<T*>(&out)[i] = downcast_s<T>(values[i]);
          square_sum += values[i] * values[i];
        }
      }
      const float row_square_sum = block_sum(square_sum, warp_sums);
      const float scale = fast_rsqrt(row_square_sum * inv_hidden + eps);
      if (packed_col < packed_hidden) {
        const Vec w = kCacheWeightStatic
            ? *(reinterpret_cast<const Vec*>(cached_weight) + packed_col)
            : weight_vec[packed_col];
        Vec norm_packet;
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          const float gamma = upcast_s(reinterpret_cast<const T*>(&w)[i]);
          reinterpret_cast<T*>(&norm_packet)[i] =
              downcast_s<T>(values[i] * scale * gamma);
        }
        residual_dst[packed_idx] = out;
        norm_dst[packed_idx] = norm_packet;
      }
    }

    if constexpr (!kRowSkipEndBarrier) {
      multi_rank_barrier<nranks, false>(sg, self_sg, rank);
    }
    return;
  }
  if constexpr (nranks == 2) {
    if (hidden == 6144 && (hidden % vlen) == 0 && blockDim.x == 512) {
      const Vec* local = reinterpret_cast<const Vec*>(data.ptrs[rank]);
      const Vec* peer = reinterpret_cast<const Vec*>(data.ptrs[rank ^ 1]);
      const Vec* residual = reinterpret_cast<const Vec*>(residual_in);
      Vec* residual_dst = reinterpret_cast<Vec*>(residual_out);
      Vec* norm_dst = reinterpret_cast<Vec*>(norm_out);
      for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const int row_base = row * packed_hidden;
        const int packed_col0 = tid;
        const int packed_col1 = tid + blockDim.x;
        const int packed_idx0 = row_base + packed_col0;
        const int packed_idx1 = row_base + packed_col1;
        float values0[vlen] = {0};
        float values1[vlen] = {0};
        float square_sum = 0.0f;
        Vec out0;
        Vec out1;

        const Vec local0 = local[packed_idx0];
        const Vec peer0 = peer[packed_idx0];
        const Vec residual0 = residual[packed_idx0];
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          float val = upcast_s(reinterpret_cast<const T*>(&local0)[i]);
          val += upcast_s(reinterpret_cast<const T*>(&peer0)[i]);
          val += upcast_s(reinterpret_cast<const T*>(&residual0)[i]);
          values0[i] = val;
          reinterpret_cast<T*>(&out0)[i] = downcast_s<T>(val);
          square_sum += val * val;
        }

        if (packed_col1 < packed_hidden) {
          const Vec local1 = local[packed_idx1];
          const Vec peer1 = peer[packed_idx1];
          const Vec residual1 = residual[packed_idx1];
#pragma unroll
          for (int i = 0; i < vlen; ++i) {
            float val = upcast_s(reinterpret_cast<const T*>(&local1)[i]);
            val += upcast_s(reinterpret_cast<const T*>(&peer1)[i]);
            val += upcast_s(reinterpret_cast<const T*>(&residual1)[i]);
            values1[i] = val;
            reinterpret_cast<T*>(&out1)[i] = downcast_s<T>(val);
            square_sum += val * val;
          }
        }

        const float row_square_sum = block_sum(square_sum, warp_sums);
        const float scale = fast_rsqrt(row_square_sum * inv_hidden + eps);

        const Vec w0 = kCacheWeightStatic
            ? *(reinterpret_cast<const Vec*>(cached_weight) + packed_col0)
            : weight_vec[packed_col0];
        Vec norm0;
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          const float gamma = upcast_s(reinterpret_cast<const T*>(&w0)[i]);
          reinterpret_cast<T*>(&norm0)[i] =
              downcast_s<T>(values0[i] * scale * gamma);
        }
        residual_dst[packed_idx0] = out0;
        norm_dst[packed_idx0] = norm0;

        if (packed_col1 < packed_hidden) {
          const Vec w1 = kCacheWeightStatic
              ? *(reinterpret_cast<const Vec*>(cached_weight) + packed_col1)
              : weight_vec[packed_col1];
          Vec norm1;
#pragma unroll
          for (int i = 0; i < vlen; ++i) {
            const float gamma = upcast_s(reinterpret_cast<const T*>(&w1)[i]);
            reinterpret_cast<T*>(&norm1)[i] =
                downcast_s<T>(values1[i] * scale * gamma);
          }
          residual_dst[packed_idx1] = out1;
          norm_dst[packed_idx1] = norm1;
        }
      }

      if constexpr (!kRowSkipEndBarrier) {
        multi_rank_barrier<nranks, false>(sg, self_sg, rank);
      }
      return;
    }
    if ((hidden == 1536 || hidden == 3072 ||
         (hidden == 6144 && rows >= 8192)) &&
        (hidden % vlen) == 0 && blockDim.x == packed_hidden) {
      const Vec* local = reinterpret_cast<const Vec*>(data.ptrs[rank]);
      const Vec* peer = reinterpret_cast<const Vec*>(data.ptrs[rank ^ 1]);
      const Vec* residual = reinterpret_cast<const Vec*>(residual_in);
      Vec* residual_dst = reinterpret_cast<Vec*>(residual_out);
      Vec* norm_dst = reinterpret_cast<Vec*>(norm_out);
      for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const int packed_idx = row * packed_hidden + tid;
        const Vec local_packet = local[packed_idx];
        const Vec peer_packet = peer[packed_idx];
        const Vec residual_packet = residual[packed_idx];
        float values[vlen] = {0};
        float square_sum = 0.0f;
        Vec out;
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          float val = upcast_s(reinterpret_cast<const T*>(&local_packet)[i]);
          val += upcast_s(reinterpret_cast<const T*>(&peer_packet)[i]);
          val += upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
          values[i] = val;
          reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
          square_sum += val * val;
        }
        residual_dst[packed_idx] = out;

        const float row_square_sum =
            row_group_sum_all_wide(square_sum, warp_sums, packed_hidden);
        const float scale = fast_rsqrt(row_square_sum * inv_hidden + eps);

        const Vec w = kCacheWeightStatic
            ? *(reinterpret_cast<const Vec*>(cached_weight) + tid)
            : weight_vec[tid];
        Vec norm_packet;
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          const float gamma = upcast_s(reinterpret_cast<const T*>(&w)[i]);
          reinterpret_cast<T*>(&norm_packet)[i] =
              downcast_s<T>(values[i] * scale * gamma);
        }
        norm_dst[packed_idx] = norm_packet;
      }

      if constexpr (!kRowSkipEndBarrier) {
        multi_rank_barrier<nranks, false>(sg, self_sg, rank);
      }
      return;
    }
  }
  Vec* target_ptr = reinterpret_cast<Vec*>(const_cast<void*>(data.ptrs[target_rank]));
  Vec* buffer_ptr = get_tmp_buf<Vec>(sg.signals[rank]);
  do {
    const int idx = idx_in_blk + idx_base;
    float acc[vlen] = {0};
    if (idx < packed_size) {
#if SGL_CUSTOM_AR_VECTOR_LOAD
      Vec raw = target_ptr[idx];
      const T* src = reinterpret_cast<const T*>(&raw);
#else
      const T* src = reinterpret_cast<const T*>(&target_ptr[idx]);
#endif
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        acc[i] = upcast_s(src[i]);
      }
    }
    shfl_reduce<T, nranks, vlen>(acc);
    if constexpr (nranks == 8) {
      __shared__ float smem[kMaxThreadsPerBlock << 1];
      if (lane < coalesce_num) {
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          smem[warp * vlen * coalesce_num + coalesce_tid * vlen + i] = acc[i];
        }
      }
      __syncthreads_lm();
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        acc[i] += smem[(warp ^ 1) * vlen * coalesce_num + coalesce_tid * vlen + i];
      }
    }
    if (rank == target_rank && idx < packed_size) {
      Vec res;
#if SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
      const Vec residual_packet =
          reinterpret_cast<const Vec*>(residual_in)[idx];
#endif
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
#if SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
        const float val =
            acc[i] + upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
        reinterpret_cast<T*>(&res)[i] = downcast_s<T>(val);
#else
        reinterpret_cast<T*>(&res)[i] = downcast_s<T>(acc[i]);
#endif
      }
      buffer_ptr[idx] = res;
    }
    idx_base += stride;
  } while (idx_base < packed_size);

  __musa_barrier_slc();
  __syncthreads_lm();
  if (tid == 0) {
    __threadfence_system_noflush();
  }
  if (tid < nranks) {
#if SGL_CUSTOM_AR_ATOMIC_BARRIER
    auto flag = atomicAdd(&self_sg->self_counter[blockIdx.x][tid], 1) + 1;
    auto* peer = &sg.signals[tid]->peer_counter[flag & 1][blockIdx.x][rank];
    auto* local = &self_sg->peer_counter[flag & 1][blockIdx.x][tid];
    atomicExch(peer, flag);
    while (atomicAdd(local, 0) != flag) {
    }
#else
    auto flag = self_sg->self_counter[blockIdx.x][tid] + 1;
    self_sg->self_counter[blockIdx.x][tid] = flag;
    auto* peer = &sg.signals[tid]->peer_counter[flag & 1][blockIdx.x][rank];
    auto* local = &self_sg->peer_counter[flag & 1][blockIdx.x][tid];
    signal_store(peer, flag);
    while (signal_load(local) != flag) {
    }
#endif
  }
  __syncthreads_lm();

  const Vec* residual = reinterpret_cast<const Vec*>(residual_in);
  Vec* residual_dst = reinterpret_cast<Vec*>(residual_out);
  Vec* norm_dst = reinterpret_cast<Vec*>(norm_out);

  if constexpr (nranks == 4 || nranks == 8) {
    if (vlen != packed_t<T>::P::size && (hidden % vlen) == 0 &&
        blockDim.x >= packed_hidden && packed_hidden <= kMaxThreadsPerBlock) {
      const int packed_col = tid;
      for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        float values[vlen] = {0};
        float square_sum = 0.0f;
        Vec out;
        if (tid < packed_hidden) {
          const int packed_idx = row * packed_hidden + packed_col;
          const int idx_target_rank =
              (packed_idx >> coalesce_sft) & (nranks - 1);
          const Vec* owner_tmp = get_tmp_buf<Vec>(sg.signals[idx_target_rank]);
          const Vec reduced = owner_tmp[packed_idx];
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
          const Vec residual_packet = residual[packed_idx];
#endif
#pragma unroll
          for (int i = 0; i < vlen; ++i) {
            float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
            val += upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
#endif
            values[i] = val;
            reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
            square_sum += val * val;
          }
        }
        const float row_square_sum = block_sum(square_sum, warp_sums);
        const float scale = fast_rsqrt(row_square_sum * inv_hidden + eps);
        if (tid < packed_hidden) {
          const int packed_idx = row * packed_hidden + packed_col;
          const Vec w = kCacheWeightStatic
              ? *(reinterpret_cast<const Vec*>(cached_weight) + packed_col)
              : weight_vec[packed_col];
          Vec norm_packet;
#pragma unroll
          for (int i = 0; i < vlen; ++i) {
            const float gamma = upcast_s(reinterpret_cast<const T*>(&w)[i]);
            reinterpret_cast<T*>(&norm_packet)[i] =
                downcast_s<T>(values[i] * scale * gamma);
          }
          residual_dst[packed_idx] = out;
          norm_dst[packed_idx] = norm_packet;
        }
      }

      if constexpr (!kRowSkipEndBarrier) {
        multi_rank_barrier<nranks, false>(sg, self_sg, rank);
      }
      return;
    }
  }

  if (SGL_CUSTOM_AR_FUSED_RMSNORM_SMALL_ROWS &&
      (hidden == 512 || hidden == 1024 || hidden == 2048) &&
      blockDim.x == packed_hidden) {
    const int packed_col = tid;
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
      float values[vlen] = {0};
      float square_sum = 0.0f;
      const int packed_idx = row * packed_hidden + packed_col;
      const int col_target_rank = (packed_col >> coalesce_sft) & (nranks - 1);
      const Vec* owner_tmp = get_tmp_buf<Vec>(sg.signals[col_target_rank]);
      const Vec reduced = owner_tmp[packed_idx];
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
      const Vec residual_packet = residual[packed_idx];
#endif
      Vec out;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
        val += upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
#endif
        values[i] = val;
        reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
        square_sum += val * val;
      }
      const float scale =
          row_group_rms_scale_shared(square_sum, warp_sums, &inv_rms, inv_hidden,
                                     eps, packed_hidden);
      const Vec w = weight_vec[packed_col];
      Vec norm_packet;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        const float gamma = upcast_s(reinterpret_cast<const T*>(&w)[i]);
        reinterpret_cast<T*>(&norm_packet)[i] =
            downcast_s<T>(values[i] * scale * gamma);
      }
      residual_dst[packed_idx] = out;
      norm_dst[packed_idx] = norm_packet;
    }

    if constexpr (!kRowSkipEndBarrier) {
      multi_rank_barrier<nranks, false>(sg, self_sg, rank);
    }
    return;
  }

  if ((hidden == 512 || hidden == 1024 || hidden == 2048) &&
      blockDim.x >= packed_hidden && (blockDim.x % packed_hidden) == 0) {
    const int rows_per_block = static_cast<int>(blockDim.x) / packed_hidden;
    const int row_slot = tid / packed_hidden;
    const int packed_col = tid - row_slot * packed_hidden;
    for (int row_base = blockIdx.x * rows_per_block; row_base < rows;
         row_base += gridDim.x * rows_per_block) {
      const int row = row_base + row_slot;
      float values[vlen] = {0};
      float square_sum = 0.0f;
      Vec out;
      if (row < rows) {
        const int packed_idx = row * packed_hidden + packed_col;
        const int col_target_rank = (packed_col >> coalesce_sft) & (nranks - 1);
        const Vec* owner_tmp = get_tmp_buf<Vec>(sg.signals[col_target_rank]);
        const Vec reduced = owner_tmp[packed_idx];
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
        const Vec residual_packet = residual[packed_idx];
#endif
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
          val += upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
#endif
          values[i] = val;
          reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
          square_sum += val * val;
        }
      }
      const float row_square_sum =
          row_group_sum_all(square_sum, warp_sums, packed_hidden);
      const float scale = fast_rsqrt(row_square_sum * inv_hidden + eps);
      if (row < rows) {
        const int packed_idx = row * packed_hidden + packed_col;
        const Vec w = weight_vec[packed_col];
        Vec norm_packet;
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          const float gamma = upcast_s(reinterpret_cast<const T*>(&w)[i]);
          reinterpret_cast<T*>(&norm_packet)[i] =
              downcast_s<T>(values[i] * scale * gamma);
        }
        residual_dst[packed_idx] = out;
        norm_dst[packed_idx] = norm_packet;
      }
    }

    if constexpr (!kRowSkipEndBarrier) {
      multi_rank_barrier<nranks, false>(sg, self_sg, rank);
    }
    return;
  }

  if (SGL_CUSTOM_AR_FUSED_RMSNORM_SMALL_ROWS &&
      hidden == 4096 && blockDim.x == packed_hidden) {
      const Vec* owner_tmp = get_tmp_buf<Vec>(sg.signals[target_rank]);
      for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const int packed_idx = row * packed_hidden + tid;
        float values[vlen] = {0};
        float square_sum = 0.0f;
        const Vec reduced = owner_tmp[packed_idx];
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
        const Vec residual_packet = residual[packed_idx];
#endif
        Vec out;
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
          val += upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
#endif
          values[i] = val;
          reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
          square_sum += val * val;
        }
        const float row_square_sum =
            hidden == 3072 ? block_sum_12warps(square_sum, warp_sums)
                           : block_sum(square_sum, warp_sums);
        const float scale = fast_rsqrt(row_square_sum * inv_hidden + eps);

        Vec norm_packet;
        const Vec w = kCacheWeightStatic
            ? *(reinterpret_cast<const Vec*>(cached_weight) + tid)
            : weight_vec[tid];
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          const float gamma = upcast_s(reinterpret_cast<const T*>(&w)[i]);
          reinterpret_cast<T*>(&norm_packet)[i] =
              downcast_s<T>(values[i] * scale * gamma);
        }
        residual_dst[packed_idx] = out;
        norm_dst[packed_idx] = norm_packet;
      }

      if constexpr (!kRowSkipEndBarrier) {
        multi_rank_barrier<nranks, false>(sg, self_sg, rank);
      }
      return;
  }

  if constexpr (nranks == 4 || nranks == 8) {
    if ((hidden == 1536 || hidden == 3072 || hidden == 4096) &&
        blockDim.x == 512) {
      const Vec* owner_tmp = get_tmp_buf<Vec>(sg.signals[target_rank]);
      for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const int packed_idx = row * packed_hidden + tid;
        float values[vlen] = {0};
        float square_sum = 0.0f;
        if (tid < packed_hidden) {
          const Vec reduced = owner_tmp[packed_idx];
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
          const Vec residual_packet = residual[packed_idx];
#endif
          Vec out;
#pragma unroll
          for (int i = 0; i < vlen; ++i) {
            float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
            val += upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
#endif
            values[i] = val;
            reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
            square_sum += val * val;
          }
          residual_dst[packed_idx] = out;
        }
        const float row_square_sum =
            hidden == 4096
                ? block_sum(square_sum, warp_sums)
                : (hidden == 3072
                       ? block_sum_12warps(square_sum, warp_sums)
                       : block_sum_nwarps<6>(square_sum, warp_sums));
        const float scale = fast_rsqrt(row_square_sum * inv_hidden + eps);

        if (tid < packed_hidden) {
          Vec norm_packet;
          const Vec w = kCacheWeightStatic
              ? *(reinterpret_cast<const Vec*>(cached_weight) + tid)
              : weight_vec[tid];
#pragma unroll
          for (int i = 0; i < vlen; ++i) {
            const float gamma = upcast_s(reinterpret_cast<const T*>(&w)[i]);
            reinterpret_cast<T*>(&norm_packet)[i] =
                downcast_s<T>(values[i] * scale * gamma);
          }
          norm_dst[packed_idx] = norm_packet;
        }
      }

      if constexpr (!kRowSkipEndBarrier) {
        multi_rank_barrier<nranks, false>(sg, self_sg, rank);
      }
      return;
    }
  }
  buffer_ptr = get_tmp_buf<Vec>(sg.signals[target_rank]);
  idx_in_blk = coalesce_tid + (target_rank << coalesce_sft) + (group_id << group_stride_sft);
  idx_base = blockIdx.x * blockDim.x;
  do {
    const int idx = idx_in_blk + idx_base;
    float values[vlen] = {0};
    float square_sum = 0.0f;
    if (idx < packed_size) {
#if SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 1 || \
    SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 2
      const Vec8<T> reduced =
          Vec8<T>::load_byp_slc(reinterpret_cast<const T*>(buffer_ptr), idx * vlen);
#else
      const Vec reduced = buffer_ptr[idx];
#endif
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
#if SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 1 || \
    SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 3
      const Vec8<T> residual_packet = Vec8<T>::load_byp_slc(residual_in, idx * vlen);
#else
      const Vec residual_packet = residual[idx];
#endif
#endif
      Vec out;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
#if SGL_CUSTOM_AR_FUSED_RMSNORM_TMP_WITH_RESIDUAL
#if SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 1 || \
    SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 2
        float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
#else
        float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
#endif
#else
#if SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 1
        float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
        val += upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
#elif SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 2
        float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
        val += upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
#elif SGL_CUSTOM_AR_FUSED_RMSNORM_SUMS_BYPASS_LOAD == 3
        float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
        val += upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
#else
        float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
        val += upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
#endif
#endif
        values[i] = val;
        reinterpret_cast<T*>(&out)[i] = downcast_s<T>(val);
        square_sum += val * val;
      }
      residual_dst[idx] = out;
    }
    const float row_square_sum = block_sum(square_sum, warp_sums);
    if constexpr (kSharedInvRms) {
      if (tid == 0) {
        inv_rms = fast_rsqrt(row_square_sum * inv_hidden + eps);
      }
      __syncthreads_lm();
    }
    if (idx < packed_size) {
      const int packed_col = idx % packed_hidden;
      const Vec w = kCacheWeightStatic
          ? *(reinterpret_cast<const Vec*>(cached_weight) + packed_col)
          : weight_vec[packed_col];
      Vec norm_packet;
      float scale;
      if constexpr (kSharedInvRms) {
        scale = inv_rms;
      } else if constexpr (kRowWarpInvRms) {
        scale = lane == 0 ? fast_rsqrt(row_square_sum * inv_hidden + eps) : 0.0f;
        scale = __shfl_sync(0xffffffff, scale, 0, 32);
      } else {
        scale = fast_rsqrt(row_square_sum * inv_hidden + eps);
      }
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        const float gamma = upcast_s(reinterpret_cast<const T*>(&w)[i]);
        reinterpret_cast<T*>(&norm_packet)[i] =
            downcast_s<T>(values[i] * scale * gamma);
      }
      norm_dst[idx] = norm_packet;
    }
    idx_base += stride;
  } while (idx_base < packed_size);

  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T, int nranks, int vlen = 8>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_rmsnorm_shfl_2stage_kernel(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    T* __restrict__ norm_out,
    const T* __restrict__ weight,
    int rank,
    int rows,
    int hidden,
    float eps,
    int packed_size) {
  if (data_ptr != nullptr) {
    data = *data_ptr;
  }
  constexpr int nranks_sft = (nranks >> 1) - (nranks >> 3);
  constexpr int coalesce_num = 8;
  constexpr int coalesce_sft = 3;
  constexpr int group_stride_sft = nranks_sft + coalesce_sft;
  constexpr int group_stride = 1 << group_stride_sft;
  constexpr int pack = vlen;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int target_rank = (tid >> coalesce_sft) & (nranks - 1);
  const int group_id = tid >> group_stride_sft;
  const int coalesce_tid = tid & (coalesce_num - 1);
  const int stride = gridDim.x * blockDim.x;
  const int packed_hidden = hidden / pack;

  typedef int16_t Vec __attribute__((vector_size(16)));

  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ float inv_rms;

  const bool cache_float_values = hidden <= kCacheHiddenLimit;
  const bool cache_t_values =
      !cache_float_values && hidden != 4096 && hidden <= kTypedCacheHiddenLimit;
  const bool cache_weight = rmsnorm_cache_weight_hidden(hidden, rows);
  float* cached = reinterpret_cast<float*>(smem);
  T* cached_t = reinterpret_cast<T*>(smem);
  const size_t cache_bytes =
      cache_float_values
          ? static_cast<size_t>(hidden) * sizeof(float)
          : (cache_t_values ? static_cast<size_t>(hidden) * sizeof(T) : 0);
  const size_t weight_offset = align_up(cache_bytes, alignof(T));
  T* cached_weight = reinterpret_cast<T*>(smem + weight_offset);
  const size_t weight_bytes = cache_weight ? static_cast<size_t>(hidden) * sizeof(T) : 0;
  float* warp_sums =
      reinterpret_cast<float*>(smem + align_up(weight_offset + weight_bytes, alignof(float)));

  if (cache_weight) {
    for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
      const int col0 = packed_col * pack;
      *(Vec8<T>*)(cached_weight + col0) = Vec8<T>::load(weight, col0);
    }
  }

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);

  int idx_base = blockIdx.x * blockDim.x;
  int idx_in_blk = coalesce_tid + (rank << coalesce_sft) + (group_id << group_stride_sft);
  Vec* target_ptr = reinterpret_cast<Vec*>(const_cast<void*>(data.ptrs[target_rank]));
  Vec* buffer_ptr = get_tmp_buf<Vec>(sg.signals[rank]);
  do {
    const int idx = idx_in_blk + idx_base;
    float acc[vlen] = {0};
    if (idx < packed_size) {
#if SGL_CUSTOM_AR_VECTOR_LOAD
      Vec raw = target_ptr[idx];
      const T* src = reinterpret_cast<const T*>(&raw);
#else
      const T* src = reinterpret_cast<const T*>(&target_ptr[idx]);
#endif
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        acc[i] = upcast_s(src[i]);
      }
    }
    shfl_reduce<T, nranks, vlen>(acc);
    if constexpr (nranks == 8) {
      __shared__ float reduce_smem[kMaxThreadsPerBlock << 1];
      if (lane < coalesce_num) {
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          reduce_smem[warp * vlen * coalesce_num + coalesce_tid * vlen + i] = acc[i];
        }
      }
      __syncthreads_lm();
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        acc[i] += reduce_smem[(warp ^ 1) * vlen * coalesce_num + coalesce_tid * vlen + i];
      }
    }
    if (rank == target_rank && idx < packed_size) {
      Vec res;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        reinterpret_cast<T*>(&res)[i] = downcast_s<T>(acc[i]);
      }
      buffer_ptr[idx] = res;
    }
    idx_base += stride;
  } while (idx_base < packed_size);

  __musa_barrier_slc();
  __syncthreads_lm();
  if (tid == 0) {
    __threadfence_system_noflush();
  }
  if (tid < nranks) {
#if SGL_CUSTOM_AR_ATOMIC_BARRIER
    auto flag = atomicAdd(&self_sg->self_counter[blockIdx.x][tid], 1) + 1;
    auto* peer = &sg.signals[tid]->peer_counter[flag & 1][blockIdx.x][rank];
    auto* local = &self_sg->peer_counter[flag & 1][blockIdx.x][tid];
    atomicExch(peer, flag);
    while (atomicAdd(local, 0) != flag) {
    }
#else
    auto flag = self_sg->self_counter[blockIdx.x][tid] + 1;
    self_sg->self_counter[blockIdx.x][tid] = flag;
    auto* peer = &sg.signals[tid]->peer_counter[flag & 1][blockIdx.x][rank];
    auto* local = &self_sg->peer_counter[flag & 1][blockIdx.x][tid];
    signal_store(peer, flag);
    while (signal_load(local) != flag) {
    }
#endif
  }
  __syncthreads_lm();

  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int base = row * hidden;
    const int packed_base = row * packed_hidden;
    float square_sum = 0.0f;
    for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
      const int packed_idx = packed_base + packed_col;
      const int owner = (packed_idx & (group_stride - 1)) >> coalesce_sft;
      const Vec* owner_tmp = get_tmp_buf<Vec>(sg.signals[owner]);
      const Vec reduced = owner_tmp[packed_idx];
      const Vec residual_packet = reinterpret_cast<const Vec*>(residual_in)[packed_idx];
      Vec residual_packet_out;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        const int col = packed_col * pack + i;
        float val = upcast_s(reinterpret_cast<const T*>(&reduced)[i]);
        val += upcast_s(reinterpret_cast<const T*>(&residual_packet)[i]);
        const T residual_value = downcast_s<T>(val);
        reinterpret_cast<T*>(&residual_packet_out)[i] = residual_value;
        if (cache_float_values) {
          cached[col] = val;
        } else if (cache_t_values) {
          cached_t[col] = residual_value;
        }
        square_sum += val * val;
      }
      reinterpret_cast<Vec*>(residual_out)[packed_idx] = residual_packet_out;
    }

    const float row_square_sum = block_sum(square_sum, warp_sums);
    if (tid == 0) {
      inv_rms = fast_rsqrt(row_square_sum / static_cast<float>(hidden) + eps);
    }
    __syncthreads_lm();

    for (int packed_col = tid; packed_col < packed_hidden; packed_col += blockDim.x) {
      const int col0 = packed_col * pack;
      Float8 sum_float;
      if (cache_float_values) {
        sum_float = *(Float8*)(cached + col0);
      } else if (cache_t_values) {
        Vec8<T> r = *(Vec8<T>*)(cached_t + col0);
#pragma unroll
        for (int i = 0; i < pack; ++i) {
          sum_float.val.elem[i] = to_float<T>(r.val.elem[i]);
        }
      } else {
        Vec8<T> r = Vec8<T>::load(residual_out + base, col0);
#pragma unroll
        for (int i = 0; i < pack; ++i) {
          sum_float.val.elem[i] = to_float<T>(r.val.elem[i]);
        }
      }
      Vec8<T> w = cache_weight ? *(Vec8<T>*)(cached_weight + col0) : Vec8<T>::load(weight, col0);
      Vec8<T> dst;
#pragma unroll
      for (int i = 0; i < pack; ++i) {
        const float gamma = to_float<T>(w.val.elem[i]);
        dst.val.elem[i] = from_float<T>(sum_float.val.elem[i] * inv_rms * gamma);
      }
      *(Vec8<T>*)(norm_out + base + col0) = dst;
    }
    __syncthreads_lm();
  }

  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_rmsnorm_push_polling_kernel(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ input,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    T* __restrict__ norm_out,
    const T* __restrict__ weight,
    int rank,
    int rows,
    int hidden,
    float eps,
    int packed_size) {
  static_assert(nranks == 4, "push-polling fused RMSNorm is only tuned for TP4");
  extern __shared__ __align__(16) unsigned char smem[];
  auto* warp_sums = reinterpret_cast<float*>(smem);
  using P = typename packed_t<T>::P;
  constexpr int pack = P::size;
  const int tid = threadIdx.x;
  const int packed_hidden = hidden / pack;
  const int rows_per_block = blockDim.x / packed_hidden;
  const int row_slot = tid / packed_hidden;
  const int packed_col = tid - row_slot * packed_hidden;
  const int col0 = packed_col * pack;
  const int bytes = packed_size * static_cast<int>(sizeof(P));
  constexpr int push_slots = SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_SLOTS;
  const int epoch = self_sg->push_epoch[blockIdx.x] % push_slots;
  const int slot_offset = epoch * nranks * packed_size;

  const auto* src = reinterpret_cast<const P*>(input);
#pragma unroll
  for (int peer = 0; peer < nranks; ++peer) {
    auto* dst = reinterpret_cast<P*>(
        reinterpret_cast<char*>(const_cast<void*>(data.ptrs[peer])) +
        (epoch * nranks + rank) * bytes);
    for (int idx = tid + blockIdx.x * blockDim.x; idx < packed_size;
         idx += blockDim.x * gridDim.x) {
      P value = src[idx];
      clear_pos_zero(value);
      store_volatile_packet(&dst[idx], value);
    }
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (tid == 0) {
    __threadfence_system_noflush();
  }
  __syncthreads_lm();

  const auto* local_buffer =
      reinterpret_cast<const P*>(data.ptrs[rank]) + slot_offset;
  P* reset_ptr = reinterpret_cast<P*>(
      reinterpret_cast<char*>(const_cast<void*>(data.ptrs[rank])) +
      epoch * nranks * bytes);
  const P pos_zero = make_pos_zero_packet<P>();

  for (int row_base = blockIdx.x * rows_per_block; row_base < rows;
       row_base += gridDim.x * rows_per_block) {
    const int row = row_base + row_slot;
    Float8 sum_float;
    float square_sum_row = 0.0f;
    int packed_idx = 0;
    if (row < rows) {
      packed_idx = row * packed_hidden + packed_col;
      P values[nranks];
      while (true) {
        bool waiting = false;
        flushInv_byp();
#pragma unroll
        for (int i = 0; i < nranks; ++i) {
          values[i] = load_volatile_packet(local_buffer + i * packed_size + packed_idx);
          waiting |= has_pos_zero(values[i]);
        }
        if (!waiting) {
          break;
        }
      }
      auto acc = upcast(values[0]);
#pragma unroll
      for (int i = 1; i < nranks; ++i) {
        packed_assign_add(acc, upcast(values[i]));
      }
      Vec8<T> residual_vec = Vec8<T>::load(residual_in + row * hidden, col0);
      Vec8<T> residual_vec_out;
#pragma unroll
      for (int i = 0; i < pack; ++i) {
        float val = acc.data[i] + to_float<T>(residual_vec.val.elem[i]);
        sum_float.val.elem[i] = val;
        residual_vec_out.val.elem[i] = from_float<T>(val);
        square_sum_row += val * val;
      }
      *(Vec8<T>*)(residual_out + row * hidden + col0) = residual_vec_out;
    }

    const float row_square_sum =
        row_group_sum_all_wide(square_sum_row, warp_sums, packed_hidden);
    const float scale =
        fast_rsqrt(row_square_sum / static_cast<float>(hidden) + eps);

    if (row < rows) {
      Vec8<T> w = Vec8<T>::load(weight, col0);
      Vec8<T> dst;
#pragma unroll
      for (int i = 0; i < pack; ++i) {
        const float gamma = to_float<T>(w.val.elem[i]);
        dst.val.elem[i] = from_float<T>(sum_float.val.elem[i] * scale * gamma);
      }
      *(Vec8<T>*)(norm_out + row * hidden + col0) = dst;
#pragma unroll
      for (int i = 0; i < nranks; ++i) {
        reset_ptr[i * packed_size + packed_idx] = pos_zero;
      }
    }
  }

  __syncthreads_lm();
  if (tid == 0) {
    self_sg->push_epoch[blockIdx.x] = (epoch + 1) % push_slots;
  }
#if !SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_SKIP_END_BARRIER
  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
#endif
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_rmsnorm_lamport_init_kernel(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    int rank,
    int packed_size,
    int slot_stride_packed) {
  using P = typename packed_t<T>::P;
  constexpr int push_slots = 3;
  const int tid = threadIdx.x;
  const P pos_zero = make_pos_zero_packet<P>();

  if (blockIdx.x == 0 && tid == 0) {
    self_sg->lamport_counter = 0;
    self_sg->lamport_flag = 0;
    self_sg->lamport_clear_packed = 0;
  }

  auto* base = reinterpret_cast<P*>(const_cast<void*>(data.ptrs[rank]));
  const int total_packets = push_slots * nranks * packed_size;
  for (int linear = tid + blockIdx.x * blockDim.x; linear < total_packets;
       linear += blockDim.x * gridDim.x) {
    const int slot_rank = linear / packed_size;
    const int offset = linear - slot_rank * packed_size;
    store_volatile_packet(base + slot_rank * slot_stride_packed + offset, pos_zero);
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (tid == 0) {
    __threadfence_system_noflush();
  }
  __syncthreads_lm();
  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_rmsnorm_lamport_kernel(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ input,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    T* __restrict__ norm_out,
    const T* __restrict__ weight,
    int rank,
    int rows,
    int hidden,
    float eps,
    int packed_size,
    int slot_stride_packed) {
  static_assert(
      nranks == 2 || nranks == 4,
      "Lamport fused RMSNorm is implemented for TP2/TP4");
  extern __shared__ __align__(16) unsigned char smem[];
  auto* warp_sums = reinterpret_cast<float*>(smem);
  auto* shared_inv_rms = warp_sums + ((blockDim.x + 31) >> 5);
  using P = typename packed_t<T>::P;
  constexpr int pack = P::size;
  constexpr int push_slots = 3;
  const int tid = threadIdx.x;
  const int packed_hidden = hidden / pack;
  const int rows_per_block = blockDim.x / packed_hidden;
  const int row_slot = tid / packed_hidden;
  const int packed_col = tid - row_slot * packed_hidden;
  const int col0 = packed_col * pack;
  const int flag = self_sg->lamport_flag % push_slots;
  const int clear_slot = (flag + 2) % push_slots;
  const int clear_packed = static_cast<int>(self_sg->lamport_clear_packed);
  const auto* src = reinterpret_cast<const P*>(input);

  __syncthreads_lm();
  if (tid == 0) {
    atomicAdd(&self_sg->lamport_counter, 1);
  }

#pragma unroll
  for (int peer = 0; peer < nranks; ++peer) {
    auto* dst = reinterpret_cast<P*>(const_cast<void*>(data.ptrs[peer])) +
                (flag * nranks + rank) * slot_stride_packed;
    for (int idx = tid + blockIdx.x * blockDim.x; idx < packed_size;
         idx += blockDim.x * gridDim.x) {
      P value = src[idx];
      clear_pos_zero(value);
      store_volatile_packet(&dst[idx], value);
    }
  }

  auto* clear_ptr = reinterpret_cast<P*>(const_cast<void*>(data.ptrs[rank])) +
                    clear_slot * nranks * slot_stride_packed;
  const P pos_zero = make_pos_zero_packet<P>();
  const int clear_limit = min(clear_packed, nranks * slot_stride_packed);
  for (int idx = tid + blockIdx.x * blockDim.x; idx < clear_limit;
       idx += blockDim.x * gridDim.x) {
    store_volatile_packet(&clear_ptr[idx], pos_zero);
  }

#if !SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_SKIP_END_BARRIER
  __musa_barrier_slc();
  __syncthreads_lm();
  if (tid == 0) {
    __threadfence_system_noflush();
  }
  __syncthreads_lm();
#endif

  const auto* local_buffer =
      reinterpret_cast<const P*>(data.ptrs[rank]) + flag * nranks * slot_stride_packed;

  for (int row_base = blockIdx.x * rows_per_block; row_base < rows;
       row_base += gridDim.x * rows_per_block) {
    const int row = row_base + row_slot;
    Float8 sum_float;
    float square_sum_row = 0.0f;
    if (row < rows) {
      const int packed_idx = row * packed_hidden + packed_col;
      P values[nranks];
      while (true) {
        bool waiting = false;
        flushInv_byp();
#pragma unroll
        for (int i = 0; i < nranks; ++i) {
          values[i] = load_volatile_packet(local_buffer + i * slot_stride_packed + packed_idx);
          waiting |= has_pos_zero(values[i]);
        }
        if (!waiting) {
          break;
        }
      }
      auto acc = upcast(values[0]);
#pragma unroll
      for (int i = 1; i < nranks; ++i) {
        packed_assign_add(acc, upcast(values[i]));
      }
      Vec8<T> residual_vec = Vec8<T>::load(residual_in + row * hidden, col0);
      Vec8<T> residual_vec_out;
#pragma unroll
      for (int i = 0; i < pack; ++i) {
        float val = acc.data[i] + to_float<T>(residual_vec.val.elem[i]);
        sum_float.val.elem[i] = val;
        residual_vec_out.val.elem[i] = from_float<T>(val);
        square_sum_row += val * val;
      }
      *(Vec8<T>*)(residual_out + row * hidden + col0) = residual_vec_out;
    }

    const float scale =
        packed_hidden == 512 && rows_per_block == 1
            ? block_rms_scale_16warps(
                  square_sum_row, warp_sums, shared_inv_rms,
                  1.0f / static_cast<float>(hidden), eps)
            : fast_rsqrt(
                  row_group_sum_all_wide(square_sum_row, warp_sums, packed_hidden) /
                      static_cast<float>(hidden) +
                  eps);

    if (row < rows) {
      Vec8<T> w = Vec8<T>::load(weight, col0);
      Vec8<T> dst;
#pragma unroll
      for (int i = 0; i < pack; ++i) {
        const float gamma = to_float<T>(w.val.elem[i]);
        dst.val.elem[i] = from_float<T>(sum_float.val.elem[i] * scale * gamma);
      }
      *(Vec8<T>*)(norm_out + row * hidden + col0) = dst;
    }
  }

  __syncthreads_lm();
  if (blockIdx.x == 0 && tid == 0) {
    while (*reinterpret_cast<volatile FlagType*>(&self_sg->lamport_counter) !=
           static_cast<FlagType>(gridDim.x)) {
    }
    self_sg->lamport_flag = (flag + 1) % push_slots;
    self_sg->lamport_clear_packed = packed_size * nranks;
    self_sg->lamport_counter = 0;
  }
#if SGL_CUSTOM_AR_FUSED_RMSNORM_LAMPORT_END_BARRIER
  if (rows <= 32 && blockIdx.x == 0) {
    __syncthreads_lm();
    multi_rank_barrier<nranks, false>(sg, self_sg, rank);
  }
#endif
}

template <typename T, int nranks>
void launch_fused_ar_rmsnorm(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* input_src,
    const T* residual_in,
    T* residual_out,
    T* norm_out,
    const T* weight,
    int rank,
    int rows,
    int hidden,
    float eps,
    int slot_stride_packed,
    musaStream_t stream) {
  constexpr int pack = packed_t<T>::P::size;
  const int threads = std::min(kDefaultThreads, kMaxThreadsPerBlock);
  const int block_limit =
      hidden == 4096
          ? (nranks == 2 && rows >= 1024
                 ? (rows >= 8192 ? 42 : (rows >= 4096 ? 44 : 42))
                 : kH4096BlockLimit)
          : (hidden == 8192 ? kH8192BlockLimit : kDefaultBlockLimit);
  const int blocks = std::min(block_limit, rows);
  if (blocks <= 0) {
    return;
  }
  const size_t smem_bytes =
      align_up(
          align_up(
              hidden <= kCacheHiddenLimit
                  ? static_cast<size_t>(hidden) * sizeof(float)
                  : (hidden != 4096 && hidden <= kTypedCacheHiddenLimit
                         ? static_cast<size_t>(hidden) * sizeof(T)
                         : 0),
              alignof(T)) +
              (rmsnorm_cache_weight_hidden(hidden, rows)
                   ? static_cast<size_t>(hidden) * sizeof(T)
                   : 0),
          alignof(float)) +
      static_cast<size_t>((threads + 31) / 32) * sizeof(float);
  const int64_t packed_size64 =
      (static_cast<int64_t>(rows) * static_cast<int64_t>(hidden)) / pack;
#if SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_POLLING
  if constexpr ((nranks == 2 || nranks == 4) && !std::is_same<T, float>::value) {
    if (input_src != nullptr &&
        rows >= SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_MIN_ROWS &&
        rows <= SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_MAX_ROWS &&
        rows <= kOneShotMaxToken && hidden == 4096 && (hidden % pack) == 0 &&
        packed_size64 <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      const int push_threads = 512;
      const int push_blocks = std::max(1, std::min(get_musa_sm_count(), rows));
      const size_t push_smem =
          static_cast<size_t>((push_threads + 31) / 32 + 1) * sizeof(float);
#if SGL_CUSTOM_AR_FUSED_RMSNORM_LAMPORT_PUSH
      if (slot_stride_packed >= static_cast<int>(packed_size64)) {
        custom_all_reduce_residual_rmsnorm_lamport_kernel<T, nranks>
            <<<push_blocks, push_threads, push_smem, stream>>>(
                data, sg, self_sg, input_src, residual_in, residual_out, norm_out,
                weight, rank, rows, hidden, eps, static_cast<int>(packed_size64),
                slot_stride_packed);
        return;
      }
#endif
      if constexpr (nranks == 4) {
        custom_all_reduce_residual_rmsnorm_push_polling_kernel<T, nranks>
            <<<push_blocks, push_threads, push_smem, stream>>>(
                data, sg, self_sg, input_src, residual_in, residual_out, norm_out,
                weight, rank, rows, hidden, eps, static_cast<int>(packed_size64));
        return;
      }
    }
  }
#endif
#if SGL_CUSTOM_AR_FUSED_RMSNORM_WARP_ROWS
  if constexpr ((nranks == 2 || nranks == 4 || nranks == 8) &&
                !std::is_same<T, float>::value) {
    if (rows <= 512 && (hidden == 512 || hidden == 1024 || hidden == 2048)) {
      constexpr int warp_threads = 256;
      constexpr int warps_per_block = warp_threads / 32;
      const int warp_blocks = std::min(
          block_limit, (rows + warps_per_block - 1) / warps_per_block);
      custom_all_reduce_residual_rmsnorm_warp_rows_kernel<T, nranks>
          <<<warp_blocks, warp_threads, 0, stream>>>(
              data, data_ptr, sg, self_sg, residual_in, residual_out, norm_out, weight,
              rank, rows, hidden, eps);
      return;
    }
  }
#endif
#if SGL_CUSTOM_AR_FUSED_RMSNORM_SHFL_2STAGE
  if constexpr ((nranks == 4 || nranks == 8) && !std::is_same<T, float>::value) {
    if (hidden % pack == 0 &&
        packed_size64 <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      custom_all_reduce_residual_rmsnorm_shfl_2stage_kernel<T, nranks>
          <<<blocks, threads, smem_bytes, stream>>>(
              data, data_ptr, sg, self_sg, residual_in, residual_out, norm_out, weight,
              rank, rows, hidden, eps, static_cast<int>(packed_size64));
      return;
    }
  }
#endif
#if SGL_CUSTOM_AR_FUSED_RMSNORM_TOKEN_2STAGE
  if (hidden % pack == 0 && rows >= 128) {
    if (packed_size64 <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      custom_all_reduce_residual_rmsnorm_2stage_kernel<T, nranks>
          <<<blocks, threads, smem_bytes, stream>>>(
              data, data_ptr, sg, self_sg, residual_in, residual_out, norm_out, weight,
              rank, rows, hidden, eps, static_cast<int>(packed_size64));
      return;
    }
  }
#endif
  custom_all_reduce_residual_rmsnorm_kernel<T, nranks>
      <<<blocks, threads, smem_bytes, stream>>>(
          data, data_ptr, sg, self_sg, residual_in, residual_out, norm_out, weight,
          rank, rows, hidden, eps);
}

template <typename T, int nranks>
int select_residual_2shot_block_limit(int packed_size) {
  int limit = kDefaultBlockLimit;
#if SGL_CUSTOM_AR_DYNAMIC_BLOCKS
  if constexpr (nranks == 4) {
    if (packed_size >= 4 * 1024 * 1024) {
      limit = 40;
    }
  } else if constexpr (nranks == 8) {
    limit = 60;
  }
#endif
  return std::min(limit, kMaxBlocks);
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
custom_all_reduce_residual_scalar_kernel(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual_in,
    T* __restrict__ residual_out,
    int rank,
    int size) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);

  for (int idx = tid; idx < size; idx += stride) {
    float ar_sum = 0.0f;
#pragma unroll
    for (int r = 0; r < nranks; ++r) {
      const auto* peer = reinterpret_cast<const T*>(data.ptrs[r]);
      ar_sum += to_float(peer[idx]);
    }
    residual_out[idx] = from_float<T>(ar_sum + to_float(residual_in[idx]));
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (threadIdx.x == 0) {
    __threadfence_system_noflush();
  }
  __syncthreads_lm();
  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T, int nranks>
void launch_residual_ar(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* residual_in,
    T* residual_out,
    int rank,
    int size,
    musaStream_t stream) {
  constexpr int pack = packed_t<T>::P::size;
  const int threads = std::min(kDefaultThreads, kMaxThreadsPerBlock);
  if (size % pack != 0) {
    const int blocks = std::min(kMaxBlocks, (size + threads - 1) / threads);
    if (blocks <= 0) {
      return;
    }
    custom_all_reduce_residual_scalar_kernel<T, nranks>
        <<<blocks, threads, 0, stream>>>(
            data, sg, self_sg, residual_in, residual_out, rank, size);
    return;
  }
  const int packed_size = size / pack;
  if constexpr ((nranks == 4 || nranks == 8) && !std::is_same<T, float>::value) {
    const int block_limit = select_residual_2shot_block_limit<T, nranks>(packed_size);
    const int blocks = std::min(block_limit, (packed_size + threads - 1) / threads);
    if (blocks <= 0) {
      return;
    }
    custom_all_reduce_residual_2shot_kernel<T, nranks>
        <<<blocks, threads, 0, stream>>>(
            data, data_ptr, sg, self_sg, residual_in, residual_out, rank,
            packed_size);
  } else {
    const int blocks = std::min(kMaxBlocks, (packed_size + threads - 1) / threads);
    if (blocks <= 0) {
      return;
    }
    custom_all_reduce_residual_kernel<T, nranks>
        <<<blocks, threads, 0, stream>>>(
            data, sg, self_sg, residual_in, residual_out, rank, size);
  }
}

template <typename T, int nranks>
bool launch_residual_ar_sums(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    const T* residual_in,
    T* residual_out,
    float* row_sums,
    int rank,
    int rows,
    int hidden,
    musaStream_t stream) {
  constexpr int pack = packed_t<T>::P::size;
  const int packed_hidden = hidden / pack;
  const int threads =
      nranks == 2 ? std::min(packed_hidden, kMaxThreadsPerBlock)
                  : (packed_hidden <= kMaxThreadsPerBlock ? packed_hidden : 0);
  if constexpr (!((nranks == 2 || nranks == 4 || nranks == 8) && !std::is_same<T, float>::value)) {
    return false;
  }
  if ((hidden % pack) != 0) {
    const int scalar_threads = std::min(hidden, kMaxThreadsPerBlock);
    const int blocks = std::min(kDefaultBlockLimit, rows);
    if (scalar_threads <= 0 || blocks <= 0) {
      return true;
    }
    custom_all_reduce_residual_sums_scalar_kernel<T, nranks>
        <<<blocks, scalar_threads, 0, stream>>>(
            data, sg, self_sg, residual_in, residual_out, row_sums, rank, rows,
            hidden);
    return true;
  }
  if constexpr (nranks == 4 || nranks == 8) {
    if (packed_hidden > kMaxThreadsPerBlock) {
      return false;
    }
  }
  if (threads <= 0 || threads > kMaxThreadsPerBlock) {
    return false;
  }
  const int64_t packed_size64 =
      (static_cast<int64_t>(rows) * static_cast<int64_t>(hidden)) / pack;
  if (packed_size64 > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    return false;
  }
  const int packed_size = static_cast<int>(packed_size64);
  if constexpr (nranks == 2) {
    const int blocks = std::min(kDefaultBlockLimit, rows);
    if (blocks <= 0) {
      return true;
    }
    custom_all_reduce_residual_sums_direct_kernel<T, 2>
        <<<blocks, threads, 0, stream>>>(
            data, sg, self_sg, residual_in, residual_out, row_sums, rank, rows,
            hidden);
    return true;
  }
  TVM_FFI_ICHECK_EQ(threads, packed_hidden);
  const int block_limit = select_residual_2shot_block_limit<T, nranks>(packed_size);
  const int blocks = std::min(block_limit, (packed_size + threads - 1) / threads);
  if (blocks <= 0) {
    return true;
  }
  custom_all_reduce_residual_2shot_sums_kernel<T, nranks>
      <<<blocks, threads, 0, stream>>>(
          data, sg, self_sg, residual_in, residual_out, row_sums, rank, rows,
          hidden, packed_size);
  return true;
}

template <typename T, int nranks, int vlen>
bool launch_residual_ar_rmsnorm_row_vlen(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* residual_in,
    T* residual_out,
    T* norm_out,
    const T* weight,
    int rank,
    int rows,
    int hidden,
    float eps,
    musaStream_t stream) {
  if constexpr (!((nranks == 2 || nranks == 4 || nranks == 8) && !std::is_same<T, float>::value)) {
    return false;
  }
  constexpr int pack = vlen;
  if constexpr (vlen != packed_t<T>::P::size) {
    return false;
  }
  const int packed_hidden = hidden / pack;
  const bool use_direct_512 =
      hidden == 4096 || (nranks == 2 && hidden == 6144) ||
      ((nranks == 4 || nranks == 8) &&
       (hidden == 1536 || hidden == 3072) &&
       !(hidden == 1536 && rows <= 128));
  const int threads =
      (SGL_CUSTOM_AR_FUSED_RMSNORM_SMALL_ROWS &&
         (hidden == 512 || hidden == 1024 || hidden == 2048))
          ? packed_hidden
      : use_direct_512
          ? 512
      : (hidden == 512 || hidden == 1024 || hidden == 2048)
          ? 512
      : (hidden <= 2048 && packed_hidden <= kMaxThreadsPerBlock)
          ? packed_hidden
          : (packed_hidden <= kMaxThreadsPerBlock ? packed_hidden : 0);
  if (threads <= 0 || (hidden % pack) != 0) {
    return false;
  }
  if (hidden <= 2048 && !use_direct_512) {
    TVM_FFI_ICHECK_EQ(threads % packed_hidden, 0);
  } else if (use_direct_512) {
    TVM_FFI_ICHECK_EQ(threads, 512);
  } else {
    TVM_FFI_ICHECK_EQ(threads, packed_hidden);
  }
  const int64_t packed_size64 =
      (static_cast<int64_t>(rows) * static_cast<int64_t>(hidden)) / pack;
  if (packed_size64 > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    return false;
  }
  const int packed_size = static_cast<int>(packed_size64);
  const int block_limit = select_residual_2shot_block_limit<T, nranks>(packed_size);
  const bool one_block_per_row =
      threads == packed_hidden && rows <= kOneShotMaxToken;
  const int blocks = one_block_per_row
      ? std::min(rows, kMaxBlocks)
      : std::min(block_limit, (packed_size + threads - 1) / threads);
  if (blocks <= 0) {
    return true;
  }
  const bool cache_weight =
      SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WEIGHT_CACHE &&
      rows >= SGL_CUSTOM_AR_FUSED_RMSNORM_ROW_WEIGHT_CACHE_MIN_ROWS &&
      hidden <= kWeightCacheHiddenLimit;
  const size_t smem_bytes =
      align_up(cache_weight ? static_cast<size_t>(hidden) * sizeof(T) : 0, alignof(float)) +
      static_cast<size_t>((threads + 31) / 32) * sizeof(float);
  if (cache_weight) {
    custom_all_reduce_residual_2shot_rmsnorm_row_kernel<T, nranks, true, true, vlen>
        <<<blocks, threads, smem_bytes, stream>>>(
            data, data_ptr, sg, self_sg, residual_in, residual_out, norm_out, weight,
            rank, rows, hidden, eps, packed_size);
  } else if constexpr (kRowSharedInvRms) {
    custom_all_reduce_residual_2shot_rmsnorm_row_kernel<T, nranks, true, false, vlen>
        <<<blocks, threads, smem_bytes, stream>>>(
            data, data_ptr, sg, self_sg, residual_in, residual_out, norm_out, weight,
            rank, rows, hidden, eps, packed_size);
  } else {
    custom_all_reduce_residual_2shot_rmsnorm_row_kernel<T, nranks, false, false, vlen>
        <<<blocks, threads, smem_bytes, stream>>>(
            data, data_ptr, sg, self_sg, residual_in, residual_out, norm_out, weight,
            rank, rows, hidden, eps, packed_size);
  }
  return true;
}

template <typename T, int nranks>
bool launch_residual_ar_rmsnorm_row(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* residual_in,
    T* residual_out,
    T* norm_out,
    const T* weight,
    int rank,
    int rows,
    int hidden,
    float eps,
    musaStream_t stream) {
  if ((hidden % 8) == 0) {
    return launch_residual_ar_rmsnorm_row_vlen<T, nranks, 8>(
        data, data_ptr, sg, self_sg, residual_in, residual_out, norm_out, weight, rank,
        rows, hidden, eps, stream);
  }
  return false;
}

template <typename T>
void dispatch_world_size(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* input_src,
    const T* residual_in,
    T* residual_out,
    T* norm_out,
    const T* weight,
    int rank,
    int world_size,
    int rows,
    int hidden,
    float eps,
    int slot_stride_packed,
    musaStream_t stream) {
  switch (world_size) {
    case 2:
      launch_fused_ar_rmsnorm<T, 2>(
          data, data_ptr, sg, self_sg, input_src, residual_in, residual_out, norm_out,
          weight, rank, rows, hidden, eps, slot_stride_packed, stream);
      break;
    case 4:
      launch_fused_ar_rmsnorm<T, 4>(
          data, data_ptr, sg, self_sg, input_src, residual_in, residual_out, norm_out,
          weight, rank, rows, hidden, eps, slot_stride_packed, stream);
      break;
    case 6:
      launch_fused_ar_rmsnorm<T, 6>(
          data, data_ptr, sg, self_sg, input_src, residual_in, residual_out, norm_out,
          weight, rank, rows, hidden, eps, slot_stride_packed, stream);
      break;
    case 8:
      launch_fused_ar_rmsnorm<T, 8>(
          data, data_ptr, sg, self_sg, input_src, residual_in, residual_out, norm_out,
          weight, rank, rows, hidden, eps, slot_stride_packed, stream);
      break;
    default:
      TVM_FFI_THROW(ValueError) << "world_size must be one of 2/4/6/8";
  }
}

template <typename T>
void dispatch_residual_world_size(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* residual_in,
    T* residual_out,
    int rank,
    int world_size,
    int size,
    musaStream_t stream) {
  switch (world_size) {
    case 2:
      launch_residual_ar<T, 2>(
          data, data_ptr, sg, self_sg, residual_in, residual_out, rank, size,
          stream);
      break;
    case 4:
      launch_residual_ar<T, 4>(
          data, data_ptr, sg, self_sg, residual_in, residual_out, rank, size,
          stream);
      break;
    case 6:
      launch_residual_ar<T, 6>(
          data, data_ptr, sg, self_sg, residual_in, residual_out, rank, size,
          stream);
      break;
    case 8:
      launch_residual_ar<T, 8>(
          data, data_ptr, sg, self_sg, residual_in, residual_out, rank, size,
          stream);
      break;
    default:
      TVM_FFI_THROW(ValueError) << "world_size must be one of 2/4/6/8";
  }
}

template <typename T>
bool dispatch_residual_sums_world_size(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    const T* residual_in,
    T* residual_out,
    float* row_sums,
    int rank,
    int world_size,
    int rows,
    int hidden,
    musaStream_t stream) {
  switch (world_size) {
    case 2:
      return launch_residual_ar_sums<T, 2>(
          data, sg, self_sg, residual_in, residual_out, row_sums, rank, rows,
          hidden, stream);
    case 4:
      return launch_residual_ar_sums<T, 4>(
          data, sg, self_sg, residual_in, residual_out, row_sums, rank, rows,
          hidden, stream);
    case 8:
      return launch_residual_ar_sums<T, 8>(
          data, sg, self_sg, residual_in, residual_out, row_sums, rank, rows,
          hidden, stream);
    default:
      return false;
  }
}

template <typename T>
bool dispatch_residual_rmsnorm_row_world_size(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* residual_in,
    T* residual_out,
    T* norm_out,
    const T* weight,
    int rank,
    int world_size,
    int rows,
    int hidden,
    float eps,
    musaStream_t stream) {
  switch (world_size) {
    case 2:
      return launch_residual_ar_rmsnorm_row<T, 2>(
          data, data_ptr, sg, self_sg, residual_in, residual_out, norm_out, weight, rank,
          rows, hidden, eps, stream);
    case 4:
      return launch_residual_ar_rmsnorm_row<T, 4>(
          data, data_ptr, sg, self_sg, residual_in, residual_out, norm_out, weight, rank,
          rows, hidden, eps, stream);
    case 8:
      return launch_residual_ar_rmsnorm_row<T, 8>(
          data, data_ptr, sg, self_sg, residual_in, residual_out, norm_out, weight, rank,
          rows, hidden, eps, stream);
    default:
      return false;
  }
}

}  // namespace

void sgl_musa_custom_ar_fused_allreduce_rmsnorm_unregistered(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView norm_out,
    ffi::TensorView residual_in,
    ffi::TensorView residual_out,
    ffi::TensorView weight,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t max_size_bytes,
    int64_t rank,
    int64_t world_size,
    float eps,
    int64_t reset_lamport) {
  CHECK_MUSA_CONTIGUOUS(inp);
  CHECK_MUSA_CONTIGUOUS(norm_out);
  CHECK_MUSA_CONTIGUOUS(residual_in);
  CHECK_MUSA_CONTIGUOUS(residual_out);
  CHECK_MUSA_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS_2D(inp);
  CHECK_CONTIGUOUS_2D(norm_out);
  CHECK_CONTIGUOUS_2D(residual_in);
  CHECK_CONTIGUOUS_2D(residual_out);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 1);
  TVM_FFI_ICHECK_EQ(inp.size(0), norm_out.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), norm_out.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(0), residual_in.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), residual_in.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(0), residual_out.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), residual_out.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(1), weight.size(0));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), norm_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), residual_in.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), residual_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), weight.dtype()));
  TVM_FFI_ICHECK_EQ(rank_data.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);

  RankSignals sg{};
  const auto* sig_ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(sig_ptrs[i]);
  }

  RankData data{};
  const auto* rank_ptrs = static_cast<const int64_t*>(rank_data.data_ptr());
  for (int i = 0; i < kMaxRanks; ++i) {
    data.ptrs[i] = reinterpret_cast<const void*>(rank_ptrs[i]);
  }

  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(inp.device());
  const int64_t numel64 = tensor_numel(inp);
  TVM_FFI_ICHECK_LE(numel64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  const int64_t nbytes = numel64 * ((static_cast<int64_t>(inp.dtype().bits) * inp.dtype().lanes + 7) / 8);
  TVM_FFI_ICHECK_LE(nbytes, max_size_bytes);

  const int rows = static_cast<int>(inp.size(0));
  const int hidden = static_cast<int>(inp.size(1));
  constexpr int configured_push_slots = SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_SLOTS;
  constexpr int push_slots =
      SGL_CUSTOM_AR_FUSED_RMSNORM_LAMPORT_PUSH ? 3 : configured_push_slots;
  static_assert(push_slots >= 2, "push polling needs at least two slots");
  const bool use_push_polling =
      SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_POLLING &&
      (world_size == 2 || world_size == 4) && hidden == 4096 &&
      rows >= SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_MIN_ROWS &&
      rows <= SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_MAX_ROWS &&
      rows <= kOneShotMaxToken &&
      nbytes * push_slots * world_size <= max_size_bytes;
  if (SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_POLLING &&
      (world_size == 2 || world_size == 4) && hidden == 4096 &&
      rows >= SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_MIN_ROWS &&
      rows <= SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_MAX_ROWS &&
      rows <= kOneShotMaxToken) {
    TVM_FFI_ICHECK_LE(nbytes * push_slots * world_size, max_size_bytes);
  }
  if (!use_push_polling) {
    const musaError_t copy_err = musaMemcpyAsync(
        reinterpret_cast<void*>(self_buffer_ptr),
        inp.data_ptr(),
        static_cast<size_t>(nbytes),
        musaMemcpyDeviceToDevice,
        stream);
    TVM_FFI_ICHECK_EQ(copy_err, musaSuccess)
        << "MUSA custom AR fused RMSNorm copy failed: "
        << musaGetErrorString(copy_err);
  }
  const int slot_stride_packed = static_cast<int>(
      max_size_bytes / (static_cast<int64_t>(push_slots) *
                        static_cast<int64_t>(world_size) * 16));

  if (dtype_equal(inp.dtype(), dl_float16)) {
#if SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_POLLING && SGL_CUSTOM_AR_FUSED_RMSNORM_LAMPORT_PUSH
    if (use_push_polling && reset_lamport) {
      constexpr int init_threads = 512;
      const int packed_size = static_cast<int>(numel64 / packed_t<half>::P::size);
      const int total_packets = push_slots * static_cast<int>(world_size) * packed_size;
      const int init_blocks =
          std::min(32, (total_packets + init_threads - 1) / init_threads);
      if (init_blocks > 0) {
        if (world_size == 2) {
          custom_all_reduce_residual_rmsnorm_lamport_init_kernel<half, 2>
              <<<init_blocks, init_threads, 0, stream>>>(
                  data, sg, self_sg, static_cast<int>(rank), packed_size,
                  slot_stride_packed);
        } else {
          custom_all_reduce_residual_rmsnorm_lamport_init_kernel<half, 4>
              <<<init_blocks, init_threads, 0, stream>>>(
                  data, sg, self_sg, static_cast<int>(rank), packed_size,
                  slot_stride_packed);
        }
      }
    }
#endif
    dispatch_world_size(
        data, nullptr, sg, self_sg,
        use_push_polling ? static_cast<const half*>(inp.data_ptr()) : nullptr,
        static_cast<const half*>(residual_in.data_ptr()),
        static_cast<half*>(residual_out.data_ptr()),
        static_cast<half*>(norm_out.data_ptr()),
        static_cast<const half*>(weight.data_ptr()),
        static_cast<int>(rank), static_cast<int>(world_size),
        rows, hidden, eps, slot_stride_packed, stream);
  } else if (dtype_equal(inp.dtype(), dl_bfloat16)) {
#if SGL_CUSTOM_AR_FUSED_RMSNORM_PUSH_POLLING && SGL_CUSTOM_AR_FUSED_RMSNORM_LAMPORT_PUSH
    if (use_push_polling && reset_lamport) {
      constexpr int init_threads = 512;
      const int packed_size =
          static_cast<int>(numel64 / packed_t<__mt_bfloat16>::P::size);
      const int total_packets = push_slots * static_cast<int>(world_size) * packed_size;
      const int init_blocks =
          std::min(32, (total_packets + init_threads - 1) / init_threads);
      if (init_blocks > 0) {
        if (world_size == 2) {
          custom_all_reduce_residual_rmsnorm_lamport_init_kernel<__mt_bfloat16, 2>
              <<<init_blocks, init_threads, 0, stream>>>(
                  data, sg, self_sg, static_cast<int>(rank), packed_size,
                  slot_stride_packed);
        } else {
          custom_all_reduce_residual_rmsnorm_lamport_init_kernel<__mt_bfloat16, 4>
              <<<init_blocks, init_threads, 0, stream>>>(
                  data, sg, self_sg, static_cast<int>(rank), packed_size,
                  slot_stride_packed);
        }
      }
    }
#endif
    dispatch_world_size(
        data, nullptr, sg, self_sg,
        use_push_polling ? static_cast<const __mt_bfloat16*>(inp.data_ptr()) : nullptr,
        static_cast<const __mt_bfloat16*>(residual_in.data_ptr()),
        static_cast<__mt_bfloat16*>(residual_out.data_ptr()),
        static_cast<__mt_bfloat16*>(norm_out.data_ptr()),
        static_cast<const __mt_bfloat16*>(weight.data_ptr()),
        static_cast<int>(rank), static_cast<int>(world_size),
        rows, hidden, eps, slot_stride_packed, stream);
  } else {
    TVM_FFI_THROW(ValueError) << "custom AR fused RMSNorm only supports fp16/bf16";
  }

  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA custom AR fused RMSNorm kernel failed: " << musaGetErrorString(err);
}

void sgl_musa_custom_ar_fused_allreduce_residual_unregistered(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView residual_in,
    ffi::TensorView residual_out,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t max_size_bytes,
    int64_t rank,
    int64_t world_size) {
  CHECK_MUSA_CONTIGUOUS(inp);
  CHECK_MUSA_CONTIGUOUS(residual_in);
  CHECK_MUSA_CONTIGUOUS(residual_out);
  TVM_FFI_ICHECK_EQ(tensor_numel(inp), tensor_numel(residual_in));
  TVM_FFI_ICHECK_EQ(tensor_numel(inp), tensor_numel(residual_out));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), residual_in.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), residual_out.dtype()));
  TVM_FFI_ICHECK_EQ(rank_data.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);

  RankSignals sg{};
  const auto* sig_ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(sig_ptrs[i]);
  }

  RankData data{};
  const auto* rank_ptrs = static_cast<const int64_t*>(rank_data.data_ptr());
  for (int i = 0; i < kMaxRanks; ++i) {
    data.ptrs[i] = reinterpret_cast<const void*>(rank_ptrs[i]);
  }

  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(inp.device());
  const int64_t numel64 = tensor_numel(inp);
  TVM_FFI_ICHECK_LE(numel64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  const int64_t nbytes =
      numel64 * ((static_cast<int64_t>(inp.dtype().bits) * inp.dtype().lanes + 7) / 8);
  TVM_FFI_ICHECK_LE(nbytes, max_size_bytes);

  const musaError_t copy_err = musaMemcpyAsync(
      reinterpret_cast<void*>(self_buffer_ptr),
      inp.data_ptr(),
      static_cast<size_t>(nbytes),
      musaMemcpyDeviceToDevice,
      stream);
  TVM_FFI_ICHECK_EQ(copy_err, musaSuccess)
      << "MUSA custom AR fused residual copy failed: " << musaGetErrorString(copy_err);

  const int size = static_cast<int>(numel64);
  if (dtype_equal(inp.dtype(), dl_float16)) {
    dispatch_residual_world_size(
        data, nullptr, sg, self_sg,
        static_cast<const half*>(residual_in.data_ptr()),
        static_cast<half*>(residual_out.data_ptr()),
        static_cast<int>(rank), static_cast<int>(world_size), size, stream);
  } else if (dtype_equal(inp.dtype(), dl_bfloat16)) {
    dispatch_residual_world_size(
        data, nullptr, sg, self_sg,
        static_cast<const __mt_bfloat16*>(residual_in.data_ptr()),
        static_cast<__mt_bfloat16*>(residual_out.data_ptr()),
        static_cast<int>(rank), static_cast<int>(world_size), size, stream);
  } else {
    TVM_FFI_THROW(ValueError) << "custom AR fused residual only supports fp16/bf16";
  }

  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA custom AR fused residual kernel failed: " << musaGetErrorString(err);
}

void sgl_musa_custom_ar_fused_allreduce_residual_registered(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView residual_in,
    ffi::TensorView residual_out,
    int64_t self_signal_ptr,
    int64_t rank,
    int64_t world_size) {
  CHECK_MUSA_CONTIGUOUS(residual_in);
  CHECK_MUSA_CONTIGUOUS(residual_out);
  TVM_FFI_ICHECK_EQ(tensor_numel(residual_in), tensor_numel(residual_out));
  TVM_FFI_ICHECK(dtype_equal(residual_in.dtype(), residual_out.dtype()));
  TVM_FFI_ICHECK_EQ(
      rank_data.device().device_type, residual_in.device().device_type);
  TVM_FFI_ICHECK_EQ(
      rank_data.device().device_id, residual_in.device().device_id);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);

  RankSignals sg{};
  const auto* sig_ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(sig_ptrs[i]);
  }

  RankData data{};
  const auto* device_data_ptr =
      reinterpret_cast<const RankData*>(rank_data.data_ptr());

  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(residual_in.device());
  const int64_t numel64 = tensor_numel(residual_in);
  TVM_FFI_ICHECK_LE(numel64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));

  const int size = static_cast<int>(numel64);
  if (dtype_equal(residual_in.dtype(), dl_float16)) {
    dispatch_residual_world_size(
        data, device_data_ptr, sg, self_sg,
        static_cast<const half*>(residual_in.data_ptr()),
        static_cast<half*>(residual_out.data_ptr()),
        static_cast<int>(rank), static_cast<int>(world_size), size, stream);
  } else if (dtype_equal(residual_in.dtype(), dl_bfloat16)) {
    dispatch_residual_world_size(
        data, device_data_ptr, sg, self_sg,
        static_cast<const __mt_bfloat16*>(residual_in.data_ptr()),
        static_cast<__mt_bfloat16*>(residual_out.data_ptr()),
        static_cast<int>(rank), static_cast<int>(world_size), size, stream);
  } else {
    TVM_FFI_THROW(ValueError) << "custom AR fused residual only supports fp16/bf16";
  }

  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA custom AR fused residual registered kernel failed: "
      << musaGetErrorString(err);
}

void sgl_musa_custom_ar_fused_allreduce_rmsnorm_registered(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView residual_in,
    ffi::TensorView residual_out,
    ffi::TensorView norm_out,
    ffi::TensorView weight,
    int64_t self_signal_ptr,
    int64_t rank,
    int64_t world_size,
    float eps) {
  CHECK_MUSA_CONTIGUOUS(residual_in);
  CHECK_MUSA_CONTIGUOUS(residual_out);
  CHECK_MUSA_CONTIGUOUS(norm_out);
  CHECK_MUSA_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS_2D(residual_in);
  CHECK_CONTIGUOUS_2D(residual_out);
  CHECK_CONTIGUOUS_2D(norm_out);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 1);
  TVM_FFI_ICHECK_EQ(residual_in.size(0), residual_out.size(0));
  TVM_FFI_ICHECK_EQ(residual_in.size(1), residual_out.size(1));
  TVM_FFI_ICHECK_EQ(residual_in.size(0), norm_out.size(0));
  TVM_FFI_ICHECK_EQ(residual_in.size(1), norm_out.size(1));
  TVM_FFI_ICHECK_EQ(residual_in.size(1), weight.size(0));
  TVM_FFI_ICHECK(dtype_equal(residual_in.dtype(), residual_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(residual_in.dtype(), norm_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(residual_in.dtype(), weight.dtype()));
  TVM_FFI_ICHECK_EQ(
      rank_data.device().device_type, residual_in.device().device_type);
  TVM_FFI_ICHECK_EQ(
      rank_data.device().device_id, residual_in.device().device_id);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);

  RankSignals sg{};
  const auto* sig_ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(sig_ptrs[i]);
  }

  RankData data{};
  const auto* device_data_ptr =
      reinterpret_cast<const RankData*>(rank_data.data_ptr());

  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(residual_in.device());
  const int rows = static_cast<int>(residual_in.size(0));
  const int hidden = static_cast<int>(residual_in.size(1));
  const int64_t numel64 = tensor_numel(residual_in);
  TVM_FFI_ICHECK_LE(numel64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));

  if (dtype_equal(residual_in.dtype(), dl_float16)) {
    dispatch_world_size(
        data, device_data_ptr, sg, self_sg,
        static_cast<const half*>(nullptr),
        static_cast<const half*>(residual_in.data_ptr()),
        static_cast<half*>(residual_out.data_ptr()),
        static_cast<half*>(norm_out.data_ptr()),
        static_cast<const half*>(weight.data_ptr()),
        static_cast<int>(rank), static_cast<int>(world_size),
        rows, hidden, eps, 0, stream);
  } else if (dtype_equal(residual_in.dtype(), dl_bfloat16)) {
    dispatch_world_size(
        data, device_data_ptr, sg, self_sg,
        static_cast<const __mt_bfloat16*>(nullptr),
        static_cast<const __mt_bfloat16*>(residual_in.data_ptr()),
        static_cast<__mt_bfloat16*>(residual_out.data_ptr()),
        static_cast<__mt_bfloat16*>(norm_out.data_ptr()),
        static_cast<const __mt_bfloat16*>(weight.data_ptr()),
        static_cast<int>(rank), static_cast<int>(world_size),
        rows, hidden, eps, 0, stream);
  } else {
    TVM_FFI_THROW(ValueError) << "custom AR fused RMSNorm only supports fp16/bf16";
  }

  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA custom AR fused RMSNorm registered kernel failed: "
      << musaGetErrorString(err);
}

void sgl_musa_custom_ar_fused_allreduce_rmsnorm_row_unregistered(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView residual_in,
    ffi::TensorView residual_out,
    ffi::TensorView norm_out,
    ffi::TensorView weight,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t max_size_bytes,
    int rank,
    int world_size,
    int hidden,
    double eps) {
  CHECK_MUSA_CONTIGUOUS(inp);
  CHECK_MUSA_CONTIGUOUS(residual_in);
  CHECK_MUSA_CONTIGUOUS(residual_out);
  CHECK_MUSA_CONTIGUOUS(norm_out);
  CHECK_MUSA_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS_2D(inp);
  CHECK_CONTIGUOUS_2D(residual_in);
  CHECK_CONTIGUOUS_2D(residual_out);
  CHECK_CONTIGUOUS_2D(norm_out);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 1);
  TVM_FFI_ICHECK_EQ(inp.size(0), residual_in.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), residual_in.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(0), residual_out.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), residual_out.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(0), norm_out.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), norm_out.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(1), hidden);
  TVM_FFI_ICHECK_EQ(weight.size(0), hidden);
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), residual_in.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), residual_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), norm_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), weight.dtype()));
  TVM_FFI_ICHECK_EQ(residual_in.device().device_id, inp.device().device_id);
  TVM_FFI_ICHECK_EQ(residual_out.device().device_id, inp.device().device_id);
  TVM_FFI_ICHECK_EQ(norm_out.device().device_id, inp.device().device_id);
  TVM_FFI_ICHECK_EQ(weight.device().device_id, inp.device().device_id);
  TVM_FFI_ICHECK_EQ(rank_data.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);

  RankData data{};
  const auto* rank_ptrs = static_cast<const int64_t*>(rank_data.data_ptr());
  for (int i = 0; i < kMaxRanks; ++i) {
    data.ptrs[i] = reinterpret_cast<const void*>(rank_ptrs[i]);
  }

  RankSignals sg{};
  const auto* sig_ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(sig_ptrs[i]);
  }

  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(inp.device());
  const int64_t numel64 = tensor_numel(inp);
  TVM_FFI_ICHECK_LE(numel64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  const int64_t nbytes =
      numel64 * ((static_cast<int64_t>(inp.dtype().bits) * inp.dtype().lanes + 7) / 8);
  TVM_FFI_ICHECK_LE(nbytes, max_size_bytes);

  const musaError_t copy_err = musaMemcpyAsync(
      reinterpret_cast<void*>(self_buffer_ptr),
      inp.data_ptr(),
      static_cast<size_t>(nbytes),
      musaMemcpyDeviceToDevice,
      stream);
  TVM_FFI_ICHECK_EQ(copy_err, musaSuccess)
      << "MUSA custom AR fused rmsnorm row copy failed: "
      << musaGetErrorString(copy_err);

  const int rows = static_cast<int>(inp.size(0));
  bool launched = false;
  if (dtype_equal(inp.dtype(), dl_float16)) {
    launched = dispatch_residual_rmsnorm_row_world_size(
        data, nullptr, sg, self_sg,
        static_cast<const half*>(residual_in.data_ptr()),
        static_cast<half*>(residual_out.data_ptr()),
        static_cast<half*>(norm_out.data_ptr()),
        static_cast<const half*>(weight.data_ptr()), rank, world_size, rows,
        hidden, static_cast<float>(eps), stream);
  } else if (dtype_equal(inp.dtype(), dl_bfloat16)) {
    launched = dispatch_residual_rmsnorm_row_world_size(
        data, nullptr, sg, self_sg,
        static_cast<const __mt_bfloat16*>(residual_in.data_ptr()),
        static_cast<__mt_bfloat16*>(residual_out.data_ptr()),
        static_cast<__mt_bfloat16*>(norm_out.data_ptr()),
        static_cast<const __mt_bfloat16*>(weight.data_ptr()), rank, world_size,
        rows, hidden, static_cast<float>(eps), stream);
  } else {
    TVM_FFI_THROW(ValueError)
        << "custom AR fused rmsnorm row only supports fp16/bf16";
  }
  TVM_FFI_ICHECK(launched)
      << "MUSA custom AR fused rmsnorm row unsupported shape";
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA custom AR fused rmsnorm row unregistered kernel failed: "
      << musaGetErrorString(err);
}

void sgl_musa_custom_ar_fused_allreduce_rmsnorm_row_registered(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView residual_in,
    ffi::TensorView residual_out,
    ffi::TensorView norm_out,
    ffi::TensorView weight,
    int64_t self_signal_ptr,
    int rank,
    int world_size,
    int hidden,
    double eps) {
  CHECK_MUSA_CONTIGUOUS(residual_in);
  CHECK_MUSA_CONTIGUOUS(residual_out);
  CHECK_MUSA_CONTIGUOUS(norm_out);
  CHECK_MUSA_CONTIGUOUS(weight);
  TVM_FFI_ICHECK_EQ(residual_in.ndim(), 2);
  TVM_FFI_ICHECK_EQ(residual_out.ndim(), 2);
  TVM_FFI_ICHECK_EQ(norm_out.ndim(), 2);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 1);
  TVM_FFI_ICHECK_EQ(residual_in.size(0), residual_out.size(0));
  TVM_FFI_ICHECK_EQ(residual_in.size(1), residual_out.size(1));
  TVM_FFI_ICHECK_EQ(residual_in.size(0), norm_out.size(0));
  TVM_FFI_ICHECK_EQ(residual_in.size(1), norm_out.size(1));
  TVM_FFI_ICHECK_EQ(residual_in.size(1), hidden);
  TVM_FFI_ICHECK_EQ(weight.size(0), hidden);
  TVM_FFI_ICHECK(dtype_equal(residual_in.dtype(), residual_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(residual_in.dtype(), norm_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(residual_in.dtype(), weight.dtype()));
  TVM_FFI_ICHECK_EQ(norm_out.device().device_id, residual_in.device().device_id);
  TVM_FFI_ICHECK_EQ(weight.device().device_id, residual_in.device().device_id);
  TVM_FFI_ICHECK_EQ(
      rank_data.device().device_type, residual_in.device().device_type);
  TVM_FFI_ICHECK_EQ(
      rank_data.device().device_id, residual_in.device().device_id);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);

  RankData data{};
  const auto* device_data_ptr =
      reinterpret_cast<const RankData*>(rank_data.data_ptr());
  const auto* sig_ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  RankSignals sg{};
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(sig_ptrs[i]);
  }
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(residual_in.device());
  const int rows = static_cast<int>(residual_in.size(0));
  bool launched = false;
  if (dtype_equal(residual_in.dtype(), dl_float16)) {
    launched = dispatch_residual_rmsnorm_row_world_size(
        data, device_data_ptr, sg, self_sg, static_cast<const half*>(residual_in.data_ptr()),
        static_cast<half*>(residual_out.data_ptr()),
        static_cast<half*>(norm_out.data_ptr()),
        static_cast<const half*>(weight.data_ptr()), rank, world_size, rows,
        hidden, static_cast<float>(eps), stream);
  } else if (dtype_equal(residual_in.dtype(), dl_bfloat16)) {
    launched = dispatch_residual_rmsnorm_row_world_size(
        data, device_data_ptr, sg, self_sg,
        static_cast<const __mt_bfloat16*>(residual_in.data_ptr()),
        static_cast<__mt_bfloat16*>(residual_out.data_ptr()),
        static_cast<__mt_bfloat16*>(norm_out.data_ptr()),
        static_cast<const __mt_bfloat16*>(weight.data_ptr()), rank, world_size,
        rows, hidden, static_cast<float>(eps), stream);
  } else {
    TVM_FFI_THROW(ValueError) << "custom AR fused rmsnorm row only supports fp16/bf16";
  }
  TVM_FFI_ICHECK(launched)
      << "MUSA custom AR fused rmsnorm row unsupported shape";
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA custom AR fused rmsnorm row registered kernel failed: "
      << musaGetErrorString(err);
}

void sgl_musa_custom_ar_reset_signal(
    int64_t self_signal_ptr,
    ffi::TensorView stream_ref) {
  TVM_FFI_ICHECK_EQ(stream_ref.device().device_type, kDLExtDev);
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(stream_ref.device());
  reset_signal_kernel<<<1, 256, 0, stream>>>(self_sg);
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA custom AR signal reset failed: " << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_custom_ar_fused_allreduce_rmsnorm_unregistered,
    sgl_musa_custom_ar_fused_allreduce_rmsnorm_unregistered);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_custom_ar_fused_allreduce_residual_unregistered,
    sgl_musa_custom_ar_fused_allreduce_residual_unregistered);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_custom_ar_fused_allreduce_residual_registered,
    sgl_musa_custom_ar_fused_allreduce_residual_registered);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_custom_ar_fused_allreduce_rmsnorm_registered,
    sgl_musa_custom_ar_fused_allreduce_rmsnorm_registered);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_custom_ar_fused_allreduce_rmsnorm_row_unregistered,
    sgl_musa_custom_ar_fused_allreduce_rmsnorm_row_unregistered);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_custom_ar_fused_allreduce_rmsnorm_row_registered,
    sgl_musa_custom_ar_fused_allreduce_rmsnorm_row_registered);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    sgl_musa_custom_ar_reset_signal,
    sgl_musa_custom_ar_reset_signal);
