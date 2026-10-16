# Iluvatar empty import compatibility

This optional package supplies a bounded import face for SGLang **0.5.18** on
Iluvatar CoreX, where `torch.cuda` is available but NVIDIA `sgl_kernel` and
FlashInfer builds are unavailable. It contains **no accelerator kernels**.
Every declared kernel raises `RuntimeError` if called. Unknown symbols remain
absent, so capability checks using `hasattr` cannot invent kernel support.

Install this addon **explicitly**, only inside a dedicated Iluvatar empty
container or environment, after installing vendor PyTorch and Triton:

```bash
# Run from the sglang-plugin-FL checkout.
python -m pip install --no-deps --no-build-isolation ./addon/iluvatar-compat
export SGLANG_IS_FLASHINFER_AVAILABLE=0
```

Do not install this package in NVIDIA environments or alongside real
`sgl-kernel` / FlashInfer packages: it intentionally owns the same Python import
names and would overwrite or shadow their files. The main plugin does not
install this addon automatically. The distribution name is
`sglang-iluvatar-compat`, not `sgl-kernel` or `flashinfer-python`; it does not
pretend to satisfy either real library's package requirement.

Use the plugin's Iluvatar defaults (`triton` attention and PyTorch sampling),
and disable custom all-reduce, NVIDIA graph features, and other unsupported
backends. Installing the addon does not make those implementations available.
The FlashInfer availability environment variable is required because SGLang's
probe tests package presence, which this import compatibility package provides.

The reviewed import names come from `python/sglang/srt` and
`python/sglang/kernels/ops` in upstream tag `v0.5.18`; see `import_manifest.py`.
The only working capability function is
`sgl_kernel.utils.is_arch_support_pdl()`, which truthfully returns `False`.
`scalar_type.py` preserves upstream's pure-Python scalar type metadata and
license. `FlashMLASchedMeta` can be used in annotations but refuses construction.
Both root packages expose `__flagos_import_shim__ = True` for preflight
diagnostics; `sgl_kernel.__version__` is `0.0.0+iluvatar.empty` and does not
claim to be an NVIDIA kernel release. Skip SGLang's real-kernel version check
with `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`.
No `sgl_kernel_npu`, `flash_attn`, or `cuda` packages are supplied. The optional
`sgl_kernel.allreduce` module is deliberately absent so SGLang can detect that
custom all-reduce is unavailable and choose its native communicator.

This only resolves import-time dependencies. CoreX runtime adaptation still
needs the native clamp-position path, the Iluvatar dtype gate, and vendor
Triton PDL compatibility in the plugin. SGLang's native CUDA MoE path can call
`sgl_kernel.moe_sum_reduce` at runtime even with a Triton MoE runner; that path
needs an implemented replacement and will fail clearly here. Host HiCache,
quantized NVIDIA kernels, speculative-only kernels, and FlashAttention remain
unsupported unless an actual implementation replaces them. CUDA Python is not
needed for the basic disabled-graph Triton path; features that require it must
remain disabled or be implemented separately.
