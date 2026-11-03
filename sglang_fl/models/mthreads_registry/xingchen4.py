"""SGLang model-registry entry point for XingChen4."""

from ..xingchen4_impl.model import XingChen4ForCausalLM

EntryClass = XingChen4ForCausalLM

__all__ = ["EntryClass", "XingChen4ForCausalLM"]
