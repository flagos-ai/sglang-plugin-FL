from __future__ import annotations

import torch


def rotary_embedding_iluvatar(
    obj,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    del obj
    q_out = query if inplace else query.clone()
    k_out = key if inplace else key.clone()

    from .triton_ops import rotary_embedding

    if rotary_embedding(
        q_out, k_out, cos, sin, position_ids, rotary_interleaved
    ):
        return q_out, k_out

    pos = position_ids.flatten()
    cos = cos.index_select(0, pos).unsqueeze(1).to(query.dtype)
    sin = sin.index_select(0, pos).unsqueeze(1).to(query.dtype)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        if rotary_interleaved:
            x1, x2 = x[..., ::2], x[..., 1::2]
            rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
            scale_cos = torch.stack((cos, cos), dim=-1).flatten(-2)
            scale_sin = torch.stack((sin, sin), dim=-1).flatten(-2)
        else:
            x1, x2 = x.chunk(2, dim=-1)
            rotated = torch.cat((-x2, x1), dim=-1)
            scale_cos = torch.cat((cos, cos), dim=-1)
            scale_sin = torch.cat((sin, sin), dim=-1)
        return x * scale_cos + rotated * scale_sin

    q_out, k_out = rotate(query), rotate(key)
    if inplace:
        query.copy_(q_out)
        key.copy_(k_out)
        return query, key
    return q_out, k_out
