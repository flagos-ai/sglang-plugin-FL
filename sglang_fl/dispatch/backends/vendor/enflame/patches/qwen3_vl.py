import torch

def gcu_fast_pos_embed_interpolate_from_list(self, grid_thw):
    num_grid_per_side = self.num_grid_per_side
    m_size = self.spatial_merge_size
    hidden_dim = self.pos_embed.embedding_dim

    outputs = []
    for t, h, w in grid_thw:
        h_idxs = torch.linspace(
            0, num_grid_per_side - 1, h, dtype=torch.float32, device=self.device
        )
        w_idxs = torch.linspace(
            0, num_grid_per_side - 1, w, dtype=torch.float32, device=self.device
        )

        h_floor = h_idxs.to(torch.int32) # for gcu
        w_floor = w_idxs.to(torch.int32) # for gcu
        h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
        w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

        dh = h_idxs - h_floor
        dw = w_idxs - w_floor

        # Create meshgrid view for all h, w vars
        dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
        h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
        h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

        # original computation of weights
        # w00 = (1 - dh_grid) * (1 - dw_grid)
        # w01 = (1 - dh_grid) * dw_grid
        # w10 = dh_grid * (1 - dw_grid)
        # w11 = dh_grid * dw_grid
        # we reuse w11 here to avoid duplicate
        # dh_grid * dw_grid computation
        w11 = dh_grid * dw_grid
        w10 = dh_grid - w11
        w01 = dw_grid - w11
        w00 = 1 - dh_grid - w01

        h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
        w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
        h_grid_idx = h_grid * num_grid_per_side

        indices = (h_grid_idx + w_grid).reshape(4, -1)
        weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
        weights = weights.to(dtype=self.dtype)

        embeds = self.pos_embed(indices)
        embeds *= weights
        combined = embeds.sum(dim=0)

        combined = combined.reshape(
            h // m_size, m_size, w // m_size, m_size, hidden_dim
        )
        combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
        repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
        outputs.append(repeated)

    return torch.cat(outputs, dim=0)

def patch_qwen3_vl():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    server_args_class = "sglang.srt.models.qwen3_vl.Qwen3VLMoeVisionModel"
    HookRegistry.register(f"{server_args_class}.fast_pos_embed_interpolate_from_list", gcu_fast_pos_embed_interpolate_from_list, HookType.REPLACE)