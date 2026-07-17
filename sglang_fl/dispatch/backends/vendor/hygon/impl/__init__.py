# Bridge layer: translates between SGLang framework-specific parameters
# and the standardized dispatch op signatures.
#
# SGLang's AROUND hook calls bridge functions which handle framework-specific
# parameters (post_residual_addition, fused_set_kv_buffer_arg, etc.),
# then call dispatch.call_op() with the standardized signature.

from sglang_fl.dispatch.bridge.silu_and_mul import silu_and_mul_bridge
from sglang_fl.dispatch.bridge.rms_norm import rms_norm_bridge
from sglang_fl.dispatch.bridge.gemma_rms_norm import gemma_rms_norm_bridge
from sglang_fl.dispatch.bridge.rotary_embedding import rotary_embedding_bridge
from sglang_fl.dispatch.bridge.mrotary_embedding import mrotary_embedding_bridge
from sglang_fl.dispatch.bridge.topk import topk_bridge
from sglang_fl.dispatch.bridge.fused_moe import fused_moe_bridge
from sglang_fl.dispatch.bridge.fla_chunk import chunk_gated_delta_rule_bridge
from sglang_fl.dispatch.bridge.fla_fused_recurrent import (
    fused_recurrent_gated_delta_rule_bridge,
)
from sglang_fl.dispatch.bridge.fla_packed_decode import (
    fused_recurrent_gated_delta_rule_packed_decode_bridge,
)

__all__ = [
    "silu_and_mul_bridge",
    "rms_norm_bridge",
    "gemma_rms_norm_bridge",
    "rotary_embedding_bridge",
    "mrotary_embedding_bridge",
    "topk_bridge",
    "fused_moe_bridge",
    "chunk_gated_delta_rule_bridge",
    "fused_recurrent_gated_delta_rule_bridge",
    "fused_recurrent_gated_delta_rule_packed_decode_bridge",
]
