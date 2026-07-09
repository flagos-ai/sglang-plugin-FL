import sys
import types


def _install_module(monkeypatch, name):
    mod = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def test_sampler_sgl_kernel_exports_are_patched(monkeypatch):
    from sglang_fl.dispatch.backends.vendor.musa import patch

    _install_module(monkeypatch, "sglang")
    _install_module(monkeypatch, "sglang.srt")
    _install_module(monkeypatch, "sglang.srt.layers")
    sampler = _install_module(monkeypatch, "sglang.srt.layers.sampler")
    sgl_kernel = _install_module(monkeypatch, "sgl_kernel")

    def min_p_sampling_from_probs():
        pass

    def top_k_renorm_prob():
        pass

    def top_k_top_p_sampling_from_probs():
        pass

    def top_p_renorm_prob():
        pass

    sgl_kernel.min_p_sampling_from_probs = min_p_sampling_from_probs
    sgl_kernel.top_k_renorm_prob = top_k_renorm_prob
    sgl_kernel.top_k_top_p_sampling_from_probs = top_k_top_p_sampling_from_probs
    sgl_kernel.top_p_renorm_prob = top_p_renorm_prob

    patch._patch_sampler_sgl_kernel_exports()

    assert sampler.min_p_sampling_from_probs is min_p_sampling_from_probs
    assert sampler.top_k_renorm_prob is top_k_renorm_prob
    assert sampler.top_k_top_p_sampling_from_probs is top_k_top_p_sampling_from_probs
    assert sampler.top_p_renorm_prob is top_p_renorm_prob


def test_apply_musa_patches_is_idempotent(monkeypatch):
    from sglang_fl.dispatch.backends.vendor.musa import patch

    calls = []
    monkeypatch.setattr(patch, "_patches_applied", False)
    monkeypatch.setattr(
        patch, "_patch_pp_send_recv_order", lambda: calls.append("pp_order")
    )
    monkeypatch.setattr(
        patch, "_patch_pp_launch_batch_add_sync", lambda: calls.append("pp_sync")
    )
    monkeypatch.setattr(
        patch,
        "_patch_sampler_sgl_kernel_exports",
        lambda: calls.append("sampler"),
    )

    patch.apply_musa_patches()
    patch.apply_musa_patches()

    assert calls == ["pp_order", "pp_sync", "sampler"]
