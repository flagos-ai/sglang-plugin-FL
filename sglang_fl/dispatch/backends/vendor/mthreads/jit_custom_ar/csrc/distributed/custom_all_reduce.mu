#include "../common.h"
#include "../device_utils.h"

#include <musa_runtime.h>
#include <musa_bf16.h>
#include <musa_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
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

#ifndef SGL_CUSTOM_AR_ATOMIC_BARRIER
#define SGL_CUSTOM_AR_ATOMIC_BARRIER 1
#endif

#ifndef SGL_CUSTOM_AR_MAX_BLOCKS
#define SGL_CUSTOM_AR_MAX_BLOCKS 120
#endif

#ifndef SGL_CUSTOM_AR_DYNAMIC_BLOCKS
#define SGL_CUSTOM_AR_DYNAMIC_BLOCKS 1
#endif

#ifndef SGL_CUSTOM_AR_PUSH_POLLING
#define SGL_CUSTOM_AR_PUSH_POLLING 0
#endif

#ifndef SGL_CUSTOM_AR_PUSH_16B_ASM
#define SGL_CUSTOM_AR_PUSH_16B_ASM 0
#endif

#ifndef SGL_CUSTOM_AR_2SHOT_DOUBLE_STORE
#define SGL_CUSTOM_AR_2SHOT_DOUBLE_STORE 0
#endif

#ifndef SGL_CUSTOM_AR_1SHOT_2RANK_SPECIAL
#define SGL_CUSTOM_AR_1SHOT_2RANK_SPECIAL 1
#endif

#ifndef SGL_CUSTOM_AR_PUSH_SKIP_START_BARRIER
#define SGL_CUSTOM_AR_PUSH_SKIP_START_BARRIER 0
#endif

constexpr int kMaxBlocks = SGL_CUSTOM_AR_MAX_BLOCKS;
constexpr int kMaxThreadsPerBlock = 1024;
constexpr int kDefaultThreads = SGL_CUSTOM_AR_THREADS;
constexpr int kDefaultBlockLimit = SGL_CUSTOM_AR_BLOCKS;
constexpr int kMaxRanks = 8;
using FlagType = uint32_t;

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

struct ArLaunchContext {
  RankData data;
  RankSignals sg;
  Signal* self_sg;
  int rank;
  int world_size;
};

struct PushLaunchContext {
  RankData data;
  RankSignals sg;
  Signal* self_sg;
  const void* input;
  void* self_buffer;
  int rank;
  int world_size;
};

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
  if (*ptr == Trait::pos_zero) *ptr = Trait::neg_zero;
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

template <typename P, int nranks, typename A>
__device__ __forceinline__ P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < nranks; ++i) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) cross_device_reduce_1stage(
    RankData data, const RankData* data_ptr, RankSignals sg, Signal* self_sg,
    T* __restrict__ out, int rank, int size) {
  if (data_ptr != nullptr) {
    data = *data_ptr;
  }
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
    reinterpret_cast<P*>(out)[idx] = packed_reduce<P, nranks, A>(reinterpret_cast<const P**>(&data.ptrs[0]), idx);
  }
  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) cross_device_reduce_1stage_2rank(
    RankData data, const RankData* data_ptr, RankSignals sg, Signal* self_sg,
    T* __restrict__ out, int rank, int size) {
  if (data_ptr != nullptr) {
    data = *data_ptr;
  }
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  const auto* local = reinterpret_cast<const P*>(data.ptrs[rank]);
  const auto* peer = reinterpret_cast<const P*>(data.ptrs[rank ^ 1]);
  auto* dst = reinterpret_cast<P*>(out);
  multi_rank_barrier<2, true>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
    A tmp = upcast(local[idx]);
    packed_assign_add(tmp, upcast(peer[idx]));
    dst[idx] = downcast<P>(tmp);
  }
  multi_rank_barrier<2, false>(sg, self_sg, rank);
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) cross_device_push_1stage(
    RankData data, const RankData* data_ptr, RankSignals sg, Signal* self_sg,
    const T* __restrict__ input, T* __restrict__ out, int rank, int size) {
  if (data_ptr != nullptr) {
    data = *data_ptr;
  }
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const int bytes = size * static_cast<int>(sizeof(P));
  const P* src = reinterpret_cast<const P*>(input);

#if !SGL_CUSTOM_AR_PUSH_POLLING
  if constexpr (nranks == 2) {
#if !SGL_CUSTOM_AR_PUSH_SKIP_START_BARRIER
    multi_rank_barrier<2, true>(sg, self_sg, rank);
#endif
    P* peer_dst = reinterpret_cast<P*>(
        reinterpret_cast<char*>(const_cast<void*>(data.ptrs[rank ^ 1])) + rank * bytes);
    for (int idx = tid; idx < size; idx += stride) {
      peer_dst[idx] = src[idx];
    }
    __musa_barrier_slc();
    __syncthreads_lm();
    if (threadIdx.x == 0) {
      __threadfence_system_noflush();
    }
    multi_rank_barrier<2, false, true>(sg, self_sg, rank);

    const auto* local_buffer = reinterpret_cast<const char*>(data.ptrs[rank]);
    const P* peer_src = reinterpret_cast<const P*>(local_buffer + (rank ^ 1) * bytes);
    auto* dst = reinterpret_cast<P*>(out);
    for (int idx = tid; idx < size; idx += stride) {
      A tmp = upcast(src[idx]);
      packed_assign_add(tmp, upcast(peer_src[idx]));
      dst[idx] = downcast<P>(tmp);
    }
    multi_rank_barrier<2, false>(sg, self_sg, rank);
    return;
  }

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
#pragma unroll
  for (int peer = 0; peer < nranks; ++peer) {
    P* dst = reinterpret_cast<P*>(
        reinterpret_cast<char*>(const_cast<void*>(data.ptrs[peer])) + rank * bytes);
    for (int idx = tid; idx < size; idx += stride) {
      dst[idx] = src[idx];
    }
  }
  __musa_barrier_slc();
  __syncthreads_lm();
  if (threadIdx.x == 0) {
    __threadfence_system_noflush();
  }
  multi_rank_barrier<nranks, false, true>(sg, self_sg, rank);

  const P* ptrs[nranks];
  const auto* local_buffer = reinterpret_cast<const char*>(data.ptrs[rank]);
#pragma unroll
  for (int i = 0; i < nranks; ++i) {
    ptrs[i] = reinterpret_cast<const P*>(local_buffer + i * bytes);
  }
  for (int idx = tid; idx < size; idx += stride) {
    reinterpret_cast<P*>(out)[idx] = packed_reduce<P, nranks, A>(ptrs, idx);
  }
  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
#else
  const int epoch = self_sg->push_epoch[blockIdx.x] & 1;
  // Push data uses a v2-style two-stage ring:
  //   slot = epoch * nranks + source_rank
#pragma unroll
  for (int peer = 0; peer < nranks; ++peer) {
    P* dst = reinterpret_cast<P*>(
        reinterpret_cast<char*>(const_cast<void*>(data.ptrs[peer])) + (epoch * nranks + rank) * bytes);
    for (int idx = tid; idx < size; idx += stride) {
      P value = src[idx];
      clear_pos_zero(value);
      store_volatile_packet(&dst[idx], value);
    }
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (threadIdx.x == 0) {
    __threadfence_system_noflush();
  }
  __syncthreads_lm();

  const P* ptrs[nranks];
  auto* local_buffer = reinterpret_cast<char*>(const_cast<void*>(data.ptrs[rank]));
#pragma unroll
  for (int i = 0; i < nranks; ++i) {
    ptrs[i] = reinterpret_cast<const P*>(local_buffer + (epoch * nranks + i) * bytes);
  }
  for (int idx = tid; idx < size; idx += stride) {
    P values[nranks];
    while (true) {
      bool waiting = false;
      flushInv_byp();
#pragma unroll
      for (int i = 0; i < nranks; ++i) {
        values[i] = load_volatile_packet(&ptrs[i][idx]);
        waiting |= has_pos_zero(values[i]);
      }
      if (!waiting) break;
    }
    A tmp = upcast(values[0]);
#pragma unroll
    for (int i = 1; i < nranks; ++i) {
      packed_assign_add(tmp, upcast(values[i]));
    }
    reinterpret_cast<P*>(out)[idx] = downcast<P>(tmp);
    P* reset_ptr = reinterpret_cast<P*>(local_buffer + epoch * nranks * bytes);
    const P pos_zero = make_pos_zero_packet<P>();
#pragma unroll
    for (int i = 0; i < nranks; ++i) {
      reset_ptr[i * size + idx] = pos_zero;
    }
  }
  __syncthreads_lm();
  if (threadIdx.x == 0) {
    self_sg->push_epoch[blockIdx.x] = (epoch + 1) & 1;
  }
  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
#endif
}

template <typename P>
__device__ __forceinline__ P* get_tmp_buf(Signal* signal) {
  return reinterpret_cast<P*>(signal + 1);
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) cross_device_reduce_2stage(
    RankData data, RankSignals sg, Signal* self_sg, T* __restrict__ out, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / nranks;
  int start = rank * part;
  int end = rank == nranks - 1 ? size : start + part;
  int largest_part = part + size % nranks;
  const P* ptrs[nranks];
  P* tmps[nranks];
#pragma unroll
  for (int i = 0; i < nranks; ++i) {
    int target = (rank + i) % nranks;
    ptrs[i] = reinterpret_cast<const P*>(data.ptrs[target]);
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, nranks, A>(ptrs, idx);
  }
  multi_rank_barrier<nranks, false, true>(sg, self_sg, rank);
  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < nranks; ++i) {
      int gather_rank = (rank + i) % nranks;
      if (gather_rank == nranks - 1 || idx < part) {
        reinterpret_cast<P*>(out)[gather_rank * part + idx] = tmps[i][idx];
      }
    }
  }
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

template <typename T, int nranks, int vlen = 8>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) custom_all_reduce_2shot(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    T* __restrict__ out,
    int rank,
    int size) {
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

  typedef int16_t Vec __attribute__((vector_size(16)));
  int idx_base = blockIdx.x * blockDim.x;
  int idx_in_blk = coalesce_tid + (rank << coalesce_sft) + (group_id << group_stride_sft);

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  Vec* target_ptr = reinterpret_cast<Vec*>(const_cast<void*>(data.ptrs[target_rank]));
  Vec* buffer_ptr = get_tmp_buf<Vec>(sg.signals[rank]);
  do {
    int idx = idx_in_blk + idx_base;
    float acc[vlen] = {0};
    if (idx < size) {
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
    if (rank == target_rank && idx < size) {
      Vec res;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        reinterpret_cast<T*>(&res)[i] = downcast_s<T>(acc[i]);
      }
      buffer_ptr[idx] = res;
#if SGL_CUSTOM_AR_2SHOT_DOUBLE_STORE
      buffer_ptr[idx] = res;
#endif
    }
    idx_base += stride;
  } while (idx_base < size);

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
  idx_in_blk = coalesce_tid + (target_rank << coalesce_sft) + (group_id << group_stride_sft);
  idx_base = blockIdx.x * blockDim.x;
  do {
    int idx = idx_in_blk + idx_base;
    if (idx < size) {
      reinterpret_cast<Vec*>(out)[idx] = buffer_ptr[idx];
    }
    idx_base += stride;
  } while (idx_base < size);
}

template <typename T, int nranks>
int select_block_limit(int packed_size, int shot) {
  int limit = kDefaultBlockLimit;
#if SGL_CUSTOM_AR_DYNAMIC_BLOCKS
  if (shot == 0) {
    if constexpr (nranks == 2) {
      if (packed_size < 32 * 1024) {
        limit = std::min(limit, 12);
      }
    }
  } else if (shot == 1) {
    if constexpr (nranks == 2) {
      if (packed_size <= 128 * 1024) {
        limit = std::min(limit, 24);
      }
    }
  } else if ((shot == 2 || shot == 4) && !std::is_same<T, float>::value) {
    if constexpr (nranks == 4) {
      if (packed_size >= 4 * 1024 * 1024) {
        limit = 40;
      } else if (packed_size >= 1024 * 1024) {
        limit = 80;
      }
    } else if constexpr (nranks == 8) {
      if (packed_size <= 128 * 1024) {
        limit = 60;
      }
    }
  }
#endif
  return std::min(limit, kMaxBlocks);
}

template <typename T, int nranks>
void launch_ar(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* input,
    T* out,
    int rank,
    int size,
    int shot,
    musaStream_t stream) {
  const int pack = packed_t<T>::P::size;
  TVM_FFI_ICHECK_EQ(size % pack, 0);
  int packed_size = size / pack;
  int block_limit = select_block_limit<T, nranks>(packed_size, shot);
  int blocks = std::min(block_limit, (packed_size + kDefaultThreads - 1) / kDefaultThreads);
  if (blocks <= 0) {
    return;
  }
  if (shot == 0 || shot == 3) {
    cross_device_push_1stage<T, nranks><<<blocks, kDefaultThreads, 0, stream>>>(data, data_ptr, sg, self_sg, input, out, rank, packed_size);
  } else if (shot == 1) {
    if constexpr (nranks == 2 && SGL_CUSTOM_AR_1SHOT_2RANK_SPECIAL) {
      cross_device_reduce_1stage_2rank<T><<<blocks, kDefaultThreads, 0, stream>>>(data, data_ptr, sg, self_sg, out, rank, packed_size);
    } else {
      cross_device_reduce_1stage<T, nranks><<<blocks, kDefaultThreads, 0, stream>>>(data, data_ptr, sg, self_sg, out, rank, packed_size);
    }
  } else if (shot == 2 || shot == 4) {
    if constexpr (std::is_same<T, float>::value) {
      cross_device_reduce_2stage<T, nranks><<<blocks, kDefaultThreads, 0, stream>>>(data, sg, self_sg, out, rank, packed_size);
    } else {
      custom_all_reduce_2shot<T, nranks><<<blocks, kDefaultThreads, 0, stream>>>(
          data, data_ptr, sg, self_sg, out, rank, packed_size);
    }
  } else {
    TVM_FFI_THROW(ValueError) << "shot must be 0, 1, 2, 3, or 4";
  }
}

template <typename T>
void dispatch_world_size(
    RankData data,
    const RankData* data_ptr,
    RankSignals sg,
    Signal* self_sg,
    const T* input,
    T* out,
    int rank,
    int world_size,
    int size,
    int shot,
    musaStream_t stream) {
  switch (world_size) {
    case 2:
      launch_ar<T, 2>(data, data_ptr, sg, self_sg, input, out, rank, size, shot, stream);
      break;
    case 4:
      launch_ar<T, 4>(data, data_ptr, sg, self_sg, input, out, rank, size, shot, stream);
      break;
    case 6:
      launch_ar<T, 6>(data, data_ptr, sg, self_sg, input, out, rank, size, shot, stream);
      break;
    case 8:
      launch_ar<T, 8>(data, data_ptr, sg, self_sg, input, out, rank, size, shot, stream);
      break;
    default:
      TVM_FFI_THROW(ValueError) << "world_size must be one of 2/4/6/8";
  }
}

}  // namespace

int64_t sgl_musa_custom_ar_meta_size() {
  return static_cast<int64_t>(sizeof(Signal));
}

__global__ void __launch_bounds__(1, 1) empty_kernel() {}

void sgl_musa_custom_ar_launch_empty(ffi::TensorView out) {
  CHECK_MUSA_CONTIGUOUS(out);
  auto stream = get_stream(out.device());
  empty_kernel<<<1, 1, 0, stream>>>();
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess) << "MUSA empty kernel failed: " << musaGetErrorString(err);
}

void sgl_musa_custom_ar_launch(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView out,
    int64_t self_signal_ptr,
    int64_t rank,
    int64_t world_size,
    int64_t shot) {
  CHECK_MUSA_CONTIGUOUS(out);
  TVM_FFI_ICHECK_EQ(
      rank_data.device().device_type, out.device().device_type);
  TVM_FFI_ICHECK_EQ(rank_data.device().device_id, out.device().device_id);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);

  RankSignals sg{};
  const auto* ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(ptrs[i]);
  }
  RankData data{};
  const auto* device_data_ptr =
      reinterpret_cast<const RankData*>(rank_data.data_ptr());
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(out.device());
  const int64_t numel64 = tensor_numel(out);
  TVM_FFI_ICHECK_LE(numel64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  const int size = static_cast<int>(numel64);

  if (dtype_equal(out.dtype(), dl_float16)) {
    auto* out_ptr = static_cast<half*>(out.data_ptr());
    dispatch_world_size(data, device_data_ptr, sg, self_sg, out_ptr, out_ptr, static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), stream);
  } else if (dtype_equal(out.dtype(), dl_bfloat16)) {
    auto* out_ptr = static_cast<__mt_bfloat16*>(out.data_ptr());
    dispatch_world_size(data, device_data_ptr, sg, self_sg, out_ptr, out_ptr, static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), stream);
  } else if (dtype_equal(out.dtype(), dl_float32)) {
    auto* out_ptr = static_cast<float*>(out.data_ptr());
    dispatch_world_size(data, device_data_ptr, sg, self_sg, out_ptr, out_ptr, static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), stream);
  } else {
    TVM_FFI_THROW(ValueError) << "custom ar only supports fp16/bf16/fp32";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess) << "MUSA custom AR kernel failed: " << musaGetErrorString(err);
}

void sgl_musa_custom_ar_launch_unregistered(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView out,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t max_size_bytes,
    int64_t rank,
    int64_t world_size,
    int64_t shot) {
  CHECK_MUSA_CONTIGUOUS(inp);
  CHECK_MUSA_CONTIGUOUS(out);
  TVM_FFI_ICHECK_EQ(rank_data.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), out.dtype()));
  TVM_FFI_ICHECK_EQ(tensor_numel(inp), tensor_numel(out));

  RankSignals sg{};
  const auto* ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(ptrs[i]);
  }
  RankData data{};
  const auto* rank_ptrs = static_cast<const int64_t*>(rank_data.data_ptr());
  for (int i = 0; i < kMaxRanks; ++i) {
    data.ptrs[i] = reinterpret_cast<const void*>(rank_ptrs[i]);
  }
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(out.device());
  const int64_t numel64 = tensor_numel(out);
  TVM_FFI_ICHECK_LE(numel64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  const int64_t nbytes = numel64 * ((static_cast<int64_t>(out.dtype().bits) * out.dtype().lanes + 7) / 8);
  TVM_FFI_ICHECK_LE(nbytes, max_size_bytes);
  if (shot == 0 || shot == 3) {
    TVM_FFI_ICHECK_LE(nbytes * 2 * world_size, max_size_bytes);
  }
  if (shot != 0 && shot != 3) {
    const musaError_t copy_err = musaMemcpyAsync(
        reinterpret_cast<void*>(self_buffer_ptr),
        inp.data_ptr(),
        static_cast<size_t>(nbytes),
        musaMemcpyDeviceToDevice,
        stream);
    TVM_FFI_ICHECK_EQ(copy_err, musaSuccess) << "MUSA custom AR copy failed: " << musaGetErrorString(copy_err);
  }
  const int size = static_cast<int>(numel64);

  if (dtype_equal(out.dtype(), dl_float16)) {
    dispatch_world_size(data, nullptr, sg, self_sg, static_cast<const half*>(inp.data_ptr()), static_cast<half*>(out.data_ptr()), static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), stream);
  } else if (dtype_equal(out.dtype(), dl_bfloat16)) {
    dispatch_world_size(data, nullptr, sg, self_sg, static_cast<const __mt_bfloat16*>(inp.data_ptr()), static_cast<__mt_bfloat16*>(out.data_ptr()), static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), stream);
  } else if (dtype_equal(out.dtype(), dl_float32)) {
    dispatch_world_size(data, nullptr, sg, self_sg, static_cast<const float*>(inp.data_ptr()), static_cast<float*>(out.data_ptr()), static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), stream);
  } else {
    TVM_FFI_THROW(ValueError) << "custom ar only supports fp16/bf16/fp32";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess) << "MUSA custom AR kernel failed: " << musaGetErrorString(err);
}

int64_t sgl_musa_custom_ar_create_context(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    int64_t self_signal_ptr,
    int64_t rank,
    int64_t world_size) {
  TVM_FFI_ICHECK_EQ(rank_data.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);

  auto* ctx = new ArLaunchContext{};
  const auto* rank_ptrs = static_cast<const int64_t*>(rank_data.data_ptr());
  for (int i = 0; i < kMaxRanks; ++i) {
    ctx->data.ptrs[i] = reinterpret_cast<const void*>(rank_ptrs[i]);
  }
  const auto* sig_ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    ctx->sg.signals[i] = reinterpret_cast<Signal*>(sig_ptrs[i]);
  }
  ctx->self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  ctx->rank = static_cast<int>(rank);
  ctx->world_size = static_cast<int>(world_size);
  return reinterpret_cast<int64_t>(ctx);
}

void sgl_musa_custom_ar_dispose_context(int64_t ctx_ptr) {
  delete reinterpret_cast<ArLaunchContext*>(ctx_ptr);
}

void sgl_musa_custom_ar_launch_context(
    int64_t ctx_ptr,
    ffi::TensorView out,
    int64_t shot) {
  CHECK_MUSA_CONTIGUOUS(out);
  auto* ctx = reinterpret_cast<ArLaunchContext*>(ctx_ptr);
  TVM_FFI_ICHECK(ctx != nullptr);
  auto stream = get_stream(out.device());
  const int64_t numel64 = tensor_numel(out);
  TVM_FFI_ICHECK_LE(numel64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  const int size = static_cast<int>(numel64);

  if (dtype_equal(out.dtype(), dl_float16)) {
    auto* out_ptr = static_cast<half*>(out.data_ptr());
    dispatch_world_size(ctx->data, nullptr, ctx->sg, ctx->self_sg, out_ptr, out_ptr, ctx->rank, ctx->world_size, size, static_cast<int>(shot), stream);
  } else if (dtype_equal(out.dtype(), dl_bfloat16)) {
    auto* out_ptr = static_cast<__mt_bfloat16*>(out.data_ptr());
    dispatch_world_size(ctx->data, nullptr, ctx->sg, ctx->self_sg, out_ptr, out_ptr, ctx->rank, ctx->world_size, size, static_cast<int>(shot), stream);
  } else if (dtype_equal(out.dtype(), dl_float32)) {
    auto* out_ptr = static_cast<float*>(out.data_ptr());
    dispatch_world_size(ctx->data, nullptr, ctx->sg, ctx->self_sg, out_ptr, out_ptr, ctx->rank, ctx->world_size, size, static_cast<int>(shot), stream);
  } else {
    TVM_FFI_THROW(ValueError) << "custom ar only supports fp16/bf16/fp32";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess) << "MUSA custom AR kernel failed: " << musaGetErrorString(err);
}

extern "C" void sgl_musa_custom_ar_launch_context_raw_nocheck(
    int64_t ctx_ptr,
    void* out,
    int64_t numel,
    int dtype_code,
    int64_t stream_value,
    int64_t shot) {
  auto* ctx = reinterpret_cast<ArLaunchContext*>(ctx_ptr);
  auto stream = reinterpret_cast<musaStream_t>(stream_value);
  const int size = static_cast<int>(numel);
  if (dtype_code == 0) {
    dispatch_world_size(ctx->data, nullptr, ctx->sg, ctx->self_sg, static_cast<half*>(out), static_cast<half*>(out), ctx->rank, ctx->world_size, size, static_cast<int>(shot), stream);
  } else if (dtype_code == 1) {
    dispatch_world_size(ctx->data, nullptr, ctx->sg, ctx->self_sg, static_cast<__mt_bfloat16*>(out), static_cast<__mt_bfloat16*>(out), ctx->rank, ctx->world_size, size, static_cast<int>(shot), stream);
  } else {
    dispatch_world_size(ctx->data, nullptr, ctx->sg, ctx->self_sg, static_cast<float*>(out), static_cast<float*>(out), ctx->rank, ctx->world_size, size, static_cast<int>(shot), stream);
  }
}

extern "C" void sgl_musa_custom_ar_launch_unregistered_raw_nocheck(
    const int64_t* rank_ptrs,
    const int64_t* signal_ptrs,
    const void* inp,
    void* out,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t numel,
    int dtype_code,
    int64_t stream_value,
    int64_t rank,
    int64_t world_size,
    int64_t shot) {
  RankSignals sg{};
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(signal_ptrs[i]);
  }
  RankData data{};
  for (int i = 0; i < kMaxRanks; ++i) {
    data.ptrs[i] = reinterpret_cast<const void*>(rank_ptrs[i]);
  }
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = reinterpret_cast<musaStream_t>(stream_value);
  const int size = static_cast<int>(numel);
  const int64_t nbytes = dtype_code == 2 ? numel * 4 : numel * 2;
  if (shot != 0 && shot != 3) {
    musaMemcpyAsync(
        reinterpret_cast<void*>(self_buffer_ptr),
        inp,
        static_cast<size_t>(nbytes),
        musaMemcpyDeviceToDevice,
        stream);
  }

  if (dtype_code == 0) {
    dispatch_world_size(data, nullptr, sg, self_sg, static_cast<const half*>(inp), static_cast<half*>(out), static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), stream);
  } else if (dtype_code == 1) {
    dispatch_world_size(data, nullptr, sg, self_sg, static_cast<const __mt_bfloat16*>(inp), static_cast<__mt_bfloat16*>(out), static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), stream);
  } else {
    dispatch_world_size(data, nullptr, sg, self_sg, static_cast<const float*>(inp), static_cast<float*>(out), static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), stream);
  }
}

extern "C" int64_t sgl_musa_custom_ar_create_unregistered_context_raw(
    const int64_t* rank_ptrs,
    const int64_t* signal_ptrs,
    const void* inp,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t rank,
    int64_t world_size) {
  auto* ctx = new PushLaunchContext{};
  for (int i = 0; i < world_size; ++i) {
    ctx->sg.signals[i] = reinterpret_cast<Signal*>(signal_ptrs[i]);
  }
  for (int i = 0; i < kMaxRanks; ++i) {
    ctx->data.ptrs[i] = reinterpret_cast<const void*>(rank_ptrs[i]);
  }
  ctx->self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  ctx->input = inp;
  ctx->self_buffer = reinterpret_cast<void*>(self_buffer_ptr);
  ctx->rank = static_cast<int>(rank);
  ctx->world_size = static_cast<int>(world_size);
  return reinterpret_cast<int64_t>(ctx);
}

extern "C" void sgl_musa_custom_ar_dispose_unregistered_context_raw(int64_t ctx_ptr) {
  delete reinterpret_cast<PushLaunchContext*>(ctx_ptr);
}

extern "C" void sgl_musa_custom_ar_launch_unregistered_context_raw_nocheck(
    int64_t ctx_ptr,
    void* out,
    int64_t numel,
    int dtype_code,
    int64_t stream_value,
    int64_t shot) {
  auto* ctx = reinterpret_cast<PushLaunchContext*>(ctx_ptr);
  auto stream = reinterpret_cast<musaStream_t>(stream_value);
  const int size = static_cast<int>(numel);
  const int64_t nbytes = dtype_code == 2 ? numel * 4 : numel * 2;
  if (shot != 0 && shot != 3) {
    musaMemcpyAsync(ctx->self_buffer, ctx->input, static_cast<size_t>(nbytes), musaMemcpyDeviceToDevice, stream);
  }

  if (dtype_code == 0) {
    dispatch_world_size(ctx->data, nullptr, ctx->sg, ctx->self_sg, static_cast<const half*>(ctx->input), static_cast<half*>(out), ctx->rank, ctx->world_size, size, static_cast<int>(shot), stream);
  } else if (dtype_code == 1) {
    dispatch_world_size(ctx->data, nullptr, ctx->sg, ctx->self_sg, static_cast<const __mt_bfloat16*>(ctx->input), static_cast<__mt_bfloat16*>(out), ctx->rank, ctx->world_size, size, static_cast<int>(shot), stream);
  } else {
    dispatch_world_size(ctx->data, nullptr, ctx->sg, ctx->self_sg, static_cast<const float*>(ctx->input), static_cast<float*>(out), ctx->rank, ctx->world_size, size, static_cast<int>(shot), stream);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_custom_ar_meta_size, sgl_musa_custom_ar_meta_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_custom_ar_launch_empty, sgl_musa_custom_ar_launch_empty);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_custom_ar_launch, sgl_musa_custom_ar_launch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_custom_ar_launch_unregistered, sgl_musa_custom_ar_launch_unregistered);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_custom_ar_create_context, sgl_musa_custom_ar_create_context);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_custom_ar_dispose_context, sgl_musa_custom_ar_dispose_context);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_musa_custom_ar_launch_context, sgl_musa_custom_ar_launch_context);
