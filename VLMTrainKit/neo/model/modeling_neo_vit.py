from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel

from neo.model.configuration_neo_vit import NEOVisionConfig


def precompute_rope_freqs_sincos(
    dim: int, max_position: int, base: float = 10000.0, device=None
):
    # Explicitly use float32 to avoid precision loss from global default dtype (e.g., bfloat16)
    # bfloat16 has limited mantissa bits, causing position indices to be incorrectly rounded
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    t = torch.arange(max_position, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def build_abs_positions_from_grid(grid_hw: torch.Tensor, device=None):
    device = grid_hw.device
    B = grid_hw.shape[0]
    H = grid_hw[:, 0]
    W = grid_hw[:, 1]
    N = H * W
    N_total = N.sum()

    patch_to_sample = torch.repeat_interleave(torch.arange(B, device=device), N)

    patch_id_within_image = torch.arange(N_total, device=device)
    patch_id_within_image = (
        patch_id_within_image
        - torch.cumsum(torch.cat([torch.tensor([0], device=device), N[:-1]]), dim=0)[
            patch_to_sample
        ]
    )

    W_per_patch = W[patch_to_sample]
    abs_x = patch_id_within_image % W_per_patch
    abs_y = patch_id_within_image // W_per_patch

    return abs_x, abs_y


def apply_rotary_emb_1d(
    x: torch.tensor,
    cos_cached: torch.tensor,
    sin_cached: torch.tensor,
    positions: torch.tensor,
):
    cos = cos_cached[positions]
    sin = sin_cached[positions]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    x_rotated = torch.empty_like(x)
    x_rotated[..., 0::2] = rotated_x1
    x_rotated[..., 1::2] = rotated_x2
    return x_rotated


def apply_2d_rotary_pos_emd(
    x: torch.Tensor,
    cos_cached_x: torch.Tensor,
    sin_cached_x: torch.Tensor,
    cos_cached_y: torch.Tensor,
    sin_cached_y: torch.Tensor,
    abs_positions_x: torch.Tensor,
    abs_positions_y: torch.Tensor,
):
    dim = x.shape[-1]
    dim_half = dim // 2

    x_part_1, x_part_2 = x[..., :dim_half], x[..., dim_half:]
    rotated_part_1 = apply_rotary_emb_1d(
        x_part_1, cos_cached_x, sin_cached_x, abs_positions_x
    )
    rotated_part_2 = apply_rotary_emb_1d(
        x_part_2, cos_cached_y, sin_cached_y, abs_positions_y
    )
    return torch.cat((rotated_part_1, rotated_part_2), dim=-1)


class NEOVisionEmbeddings(nn.Module):

    def __init__(
        self,
        config: NEOVisionConfig,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.llm_embed_dim = config.llm_hidden_size
        self.downsample_factor = int(1 / config.downsample_ratio)
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.dense_embedding = nn.Conv2d(
            in_channels=self.embed_dim,
            out_channels=self.llm_embed_dim,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor,
        )
        self.gelu = nn.GELU()
        self.rope_dim_part = self.embed_dim // 2
        cos_x, sin_x = precompute_rope_freqs_sincos(
            self.rope_dim_part,
            config.max_position_embeddings_vision,
            base=config.rope_theta_vision,
            device=None,
        )
        cos_y, sin_y = precompute_rope_freqs_sincos(
            self.rope_dim_part,
            config.max_position_embeddings_vision,
            base=config.rope_theta_vision,
            device=None,
        )

        self.register_buffer("cos_cached_x", cos_x, persistent=False)
        self.register_buffer("sin_cached_x", sin_x, persistent=False)
        self.register_buffer("cos_cached_y", cos_y, persistent=False)
        self.register_buffer("sin_cached_y", sin_y, persistent=False)

    def _apply_2d_rotary_pos_emb(self, patch_embeds, grid_hw):
        abs_pos_x, abs_pos_y = build_abs_positions_from_grid(
            grid_hw, device=patch_embeds.device
        )
        embeddings = apply_2d_rotary_pos_emd(
            patch_embeds.to(torch.float32),
            self.cos_cached_x,
            self.sin_cached_x,
            self.cos_cached_y,
            self.sin_cached_y,
            abs_pos_x,
            abs_pos_y,
        ).to(self.patch_embedding.weight.dtype)
        return embeddings

    def forward(self, pixel_values: torch.FloatTensor, grid_hw: None):
        assert (
            pixel_values.dim() == 2
        ), f"pixel_values must be 2D for native resolution, got: {pixel_values.dim()}"

        pixel_values = pixel_values.view(-1, 3, self.patch_size, self.patch_size)
        patch_embeds = self.gelu(
            self.patch_embedding(pixel_values).view(-1, self.embed_dim)
        )
        self.cos_cached_x = self.cos_cached_x.to(patch_embeds.device)
        self.sin_cached_x = self.sin_cached_x.to(patch_embeds.device)
        self.cos_cached_y = self.cos_cached_y.to(patch_embeds.device)
        self.sin_cached_y = self.sin_cached_y.to(patch_embeds.device)
        self.sin_cached_y = self.sin_cached_y.to(patch_embeds.device)

        patch_embeds = self._apply_2d_rotary_pos_emb(patch_embeds, grid_hw)
        assert (grid_hw[:, 0] * grid_hw[:, 1]).sum() == patch_embeds.shape[0]

        patches_list = []
        cur_position = 0
        for i in range(grid_hw.shape[0]):
            h, w = grid_hw[i]
            patches_per_img = (
                patch_embeds[cur_position : cur_position + h * w]
                .view(h, w, -1)
                .unsqueeze(0)
            )
            patches_per_img = self.dense_embedding(patches_per_img.permute(0, 3, 1, 2))
            patches_per_img = patches_per_img.permute(0, 2, 3, 1)
            patches_list.append(patches_per_img.view(-1, patches_per_img.shape[-1]))
            cur_position += h * w

        embeddings = torch.cat(
            patches_list, dim=0
        )  # (N_total // downsample_factor**2, C)
        assert cur_position == patch_embeds.shape[0]
        assert embeddings.shape[0] == int(
            patch_embeds.shape[0] / self.downsample_factor**2
        )

        return embeddings


class NEOVisionModel(PreTrainedModel):
    main_input_name = "pixel_values"
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    config_class = NEOVisionConfig
    # support transformers 4.51.+
    _tp_plan = ""

    def __init__(self, config: NEOVisionConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = NEOVisionEmbeddings(config)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_embeds: Optional[torch.FloatTensor] = None,
        grid_hw: Optional[torch.Tensor] = None,
    ):
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            assert (
                pixel_values.dim() == 2
            ), f"pixel_values must be 2D for native resolution, got: {pixel_values.dim()}"
            hidden_states = self.embeddings(pixel_values, grid_hw=grid_hw)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=None,
            hidden_states=None,
            attentions=None,
        )
