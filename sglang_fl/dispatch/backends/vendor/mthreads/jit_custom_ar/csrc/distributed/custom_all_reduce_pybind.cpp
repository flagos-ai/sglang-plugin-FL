#include <cstdint>

#include <torch/extension.h>
#include <torch/library.h>
#include <torch_musa/csrc/aten/musa/MUSAContext.h>

#ifndef SGL_CUSTOM_AR_TORCH_NS
#define SGL_CUSTOM_AR_TORCH_NS sglang_musa_jit_ar
#endif

extern "C" void sgl_musa_custom_ar_launch_context_raw_nocheck(
    int64_t ctx_ptr, void *out, int64_t numel, int dtype_code,
    int64_t stream_value, int64_t shot);

extern "C" void sgl_musa_custom_ar_launch_unregistered_raw_nocheck(
    const int64_t *rank_ptrs, const int64_t *signal_ptrs, const void *inp,
    void *out, int64_t self_signal_ptr, int64_t self_buffer_ptr, int64_t numel,
    int dtype_code, int64_t stream_value, int64_t rank, int64_t world_size,
    int64_t shot);

extern "C" int64_t sgl_musa_custom_ar_create_unregistered_context_raw(
    const int64_t *rank_ptrs, const int64_t *signal_ptrs, const void *inp,
    int64_t self_signal_ptr, int64_t self_buffer_ptr, int64_t rank,
    int64_t world_size);

extern "C" void
sgl_musa_custom_ar_dispose_unregistered_context_raw(int64_t ctx_ptr);

extern "C" void sgl_musa_custom_ar_launch_unregistered_context_raw_nocheck(
    int64_t ctx_ptr, void *out, int64_t numel, int dtype_code,
    int64_t stream_value, int64_t shot);

namespace {

int dtype_code(torch::Tensor &out) {
  switch (out.scalar_type()) {
  case at::ScalarType::Half:
    return 0;
  case at::ScalarType::BFloat16:
    return 1;
  case at::ScalarType::Float:
    return 2;
  default:
    TORCH_CHECK(false, "MUSA custom AR only supports fp16/bf16/fp32");
  }
}

void launch_context(int64_t ctx_ptr, torch::Tensor out, int64_t shot) {
  const auto stream = at::musa::getCurrentMUSAStream();
  sgl_musa_custom_ar_launch_context_raw_nocheck(
      ctx_ptr, out.data_ptr(), out.numel(), dtype_code(out),
      reinterpret_cast<int64_t>(stream.stream()), shot);
}

void launch_unregistered(torch::Tensor rank_data, torch::Tensor signal_ptrs_cpu,
                         torch::Tensor input, torch::Tensor out,
                         int64_t self_signal_ptr, int64_t self_buffer_ptr,
                         int64_t rank, int64_t world_size, int64_t shot) {
  const auto stream = at::musa::getCurrentMUSAStream();
  sgl_musa_custom_ar_launch_unregistered_raw_nocheck(
      rank_data.data_ptr<int64_t>(), signal_ptrs_cpu.data_ptr<int64_t>(),
      input.data_ptr(), out.data_ptr(), self_signal_ptr, self_buffer_ptr,
      out.numel(), dtype_code(out), reinterpret_cast<int64_t>(stream.stream()),
      rank, world_size, shot);
}

int64_t create_unregistered_context(torch::Tensor rank_data,
                                    torch::Tensor signal_ptrs_cpu,
                                    torch::Tensor input,
                                    int64_t self_signal_ptr,
                                    int64_t self_buffer_ptr, int64_t rank,
                                    int64_t world_size) {
  return sgl_musa_custom_ar_create_unregistered_context_raw(
      rank_data.data_ptr<int64_t>(), signal_ptrs_cpu.data_ptr<int64_t>(),
      input.data_ptr(), self_signal_ptr, self_buffer_ptr, rank, world_size);
}

void dispose_unregistered_context(int64_t ctx_ptr) {
  sgl_musa_custom_ar_dispose_unregistered_context_raw(ctx_ptr);
}

void launch_unregistered_context(int64_t ctx_ptr, torch::Tensor out,
                                 int64_t shot) {
  const auto stream = at::musa::getCurrentMUSAStream();
  sgl_musa_custom_ar_launch_unregistered_context_raw_nocheck(
      ctx_ptr, out.data_ptr(), out.numel(), dtype_code(out),
      reinterpret_cast<int64_t>(stream.stream()), shot);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_context", &launch_context);
  m.def("launch_unregistered", &launch_unregistered);
  m.def("create_unregistered_context", &create_unregistered_context);
  m.def("dispose_unregistered_context", &dispose_unregistered_context);
  m.def("launch_unregistered_context", &launch_unregistered_context);
}

TORCH_LIBRARY_FRAGMENT(SGL_CUSTOM_AR_TORCH_NS, m) {
  m.def("launch_context(int ctx_ptr, Tensor! out, int shot) -> ()");
  m.impl("launch_context", torch::kMUSA, &launch_context);
}
