"""SGLang model-registry entry point for Xing4.0."""

from ..xingchen4_impl.model import Xing4_0ForCausalLM

EntryClass = Xing4_0ForCausalLM

__all__ = ["EntryClass", "Xing4_0ForCausalLM"]
