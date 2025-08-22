# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py
# https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/models/attention.py
import os
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import CrossAttention, FeedForward, AdaLayerNorm
from einops import rearrange, repeat

from libs.rgb_flow import match_flow_warp_kv, rgb_flow_warp_kv, rgb_flow_warp_nose

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        use_temporal: Optional[bool] = False
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    use_temporal=use_temporal
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, flow_mask=None, timestep=None, return_dict: bool = True):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length,
                height=height,
                width=weight,
                inner_dim=inner_dim,
                batch=batch,
                flow_mask=flow_mask
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        use_temporal: bool = False
    ):
        super().__init__()
        self.use_temporal = use_temporal
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.warp = use_temporal

        # key-frame attention
        if use_temporal:
            self.attn1 = KeyFrameAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn1 = CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # cross attention
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # FNN
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        # temporal attention
        # if use_temporal:
        #     self.attn_temp = CrossAttention(
        #         query_dim=dim,
        #         heads=num_attention_heads,
        #         dim_head=attention_head_dim,
        #         dropout=dropout,
        #         bias=attention_bias,
        #         upcast_attention=upcast_attention,
        #     )
        #     self.conv_temp = zero_module(nn.Conv2d(dim, dim, 1, padding=0))
        # else:
        #     self.attn_temp = None

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            # self.attn_temp._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None,
                height=None, width=None, inner_dim=None, batch=None, flow_mask=None):
        # 判断decode
        warp = self.warp
        # key-frame attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )
        if self.warp:
            if self.only_cross_attention:
                hidden_states = (
                    self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask) + hidden_states
                )
            else:
                hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length, flow_mask=flow_mask, warp=warp) + hidden_states
        else:
            # for controlnet  and unet的encode
            if self.only_cross_attention:
                hidden_states = (
                        self.attn1(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states,
                                   attention_mask=attention_mask) + hidden_states
                )
            else:
                hidden_states = (self.attn1(hidden_states=norm_hidden_states, encoder_hidden_states=None,
                                            attention_mask=attention_mask) + hidden_states)

        # temporal attention  这是为了训练用的，但是为了zero-shot，不用temporal attention
        # if self.attn_temp is not None:
        #     d = norm_hidden_states.shape[1]
        #     norm_hidden_states = rearrange(norm_hidden_states, "(b f) d c -> (b d) f c", f=video_length)
        #     hidden_states1 = self.attn_temp(norm_hidden_states)
        #     # transfer to conv input
        #     hidden_states1 = rearrange(hidden_states1, "(b d) f c -> (b f) d c", d=d)
        #     hidden_states1 = (
        #         hidden_states1.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        #     )  # (b*f, h, w, c)-> (b*f, c, h, w)
        #     hidden_states1 = self.conv_temp(hidden_states1)
        #     # transfer to attention input
        #     #  (b*f, c, h, w) -> (b*f, h, w, c) -> (b*f, h*w, c)
        #     hidden_states1 = hidden_states1.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        #     # add with self attention output
        #     hidden_states = hidden_states + hidden_states1

        if self.attn2 is not None:
            # cross-attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                )
                + hidden_states
            )
        # FNN
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


class KeyFrameAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head**-0.5

        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = False
        self.added_kv_proj_dim = added_kv_proj_dim

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def set_attention_slice(self, slice_size):
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        self._slice_size = slice_size

    def _attention(self, query, key, value, attention_mask=None):
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        if attention_mask is not None:  # 128 256 256
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)

        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)   # 128 256 256 与 128 256 160

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)  # 128 256 160
        return hidden_states

    def _sliced_attention(self, query, key, value, sequence_length, dim, attention_mask):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]

            if self.upcast_attention:
                query_slice = query_slice.float()
                key_slice = key_slice.float()

            attn_slice = torch.baddbmm(
                torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query_slice.dtype, device=query.device),
                query_slice,
                key_slice.transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )

            if attention_mask is not None:
                attn_slice = attn_slice + attention_mask[start_idx:end_idx]

            if self.upcast_softmax:
                attn_slice = attn_slice.float()

            attn_slice = attn_slice.softmax(dim=-1)

            # cast back to the original dtype
            attn_slice = attn_slice.to(value.dtype)
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        # TODO attention_mask
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
        if len(hidden_states.shape) == 4:
            pass
        else:
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def forwards(self, hidden_states, encoder_hidden_states=None, 
                                        attention_mask=None, video_length=None, k=0,
                                        flow_mask=None, warp=None):
        
        batch_size, sequence_length, _ = hidden_states.shape
        encoder_hidden_states = encoder_hidden_states
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        _, res, dim = query.shape
        if warp:
            if res in [256, 1024, 4096]:  # res is 256 256 1024 4096
                query = rearrange(query, "(b f) d c -> b f d c", f=video_length)
                query = rgb_flow_warp_nose(query, flow=flow_mask, video_length=video_length)

        # 测试fully
        # query = rearrange(query, "(b f) d c -> b (f d) c", f=video_length)
        # 测试fully
        query = self.reshape_heads_to_batch_dim(query)
        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        key = key[:, [k] * video_length]
        key = rearrange(key, "b f d c -> (b f) d c")
        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        value = value[:, [k] * video_length]
        value = rearrange(value, "b f d c -> (b f) d c")
        # 测试fully
        # key = rearrange(key, "(b f) d c -> b (f d) c", f=video_length)
        # value = rearrange(value, "(b f) d c -> b (f d) c", f=video_length)
        # 测试fully
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        # All frames
        # hidden_states = rearrange(hidden_states, "b (f d) c -> (b f) d c", f=video_length)
        return hidden_states

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, k=0,
                flow_mask=None, warp=None):
        batch_size, sequence_length, _ = hidden_states.shape
        encoder_hidden_states = encoder_hidden_states
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        _, res, dim = query.shape
        if warp:
            # if visualize query token:
            # self.visualize_and_save_features_pca(query, 0, 'pip_vis/warp_f/', 0)
            query = rearrange(query, "(b f) d c -> b f d c", f=video_length)
            query = rgb_flow_warp_nose(query, flow=flow_mask, video_length=video_length)
            # self.visualize_and_save_features_pca(query, 0, 'pip_vis/warp_after/', 0)

        query = self.reshape_heads_to_batch_dim(query)
        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        if warp:
            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            if video_length < 61:  # first不适合超过40帧的
                key_0 = match_flow_warp_kv(key, flow=flow_mask, video_length=video_length)
            else:
                key_0 = key[:, [0] * video_length]  
            key_warp = rgb_flow_warp_kv(key, flow=flow_mask, video_length=video_length)
            key = torch.cat([key_0, key_warp], dim=2)
            # key = key_0
            # key = key_warp
            key = rearrange(key, "b f d c -> (b f) d c")

            # value 
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            if video_length < 41:  # 40帧以下，可以一次性infer能接受
                value_0 = match_flow_warp_kv(value, flow=flow_mask, video_length=video_length)
            else:
                value_0 = value[:, [0] * video_length]
            value_warp = rgb_flow_warp_kv(value, flow=flow_mask, video_length=video_length)
            value = torch.cat([value_0, value_warp], dim=2)
            # value = value_0
            # value = value_warp
            value = rearrange(value, "b f d c -> (b f) d c")
        else: 
            assert ValueError('error of flow warp operation in error module')

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)
        # 变成fresco的head
        query = rearrange(query, "(b h) d c -> b d h c", b=batch_size, h=self.heads)
        key = rearrange(key, "(b h) d c -> b d h c", b=batch_size, h=self.heads)
        value = rearrange(value, "(b h) d c -> b d h c", b=batch_size, h=self.heads)
        # # #

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)
        
        # fresco的multi-head
        hidden_states = rearrange(hidden_states, "b d h c -> (b h) d c", h=self.heads)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states
    
    def forward_cat(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, k=0,
                flow_mask=None, warp=None):
        batch_size, sequence_length, _ = hidden_states.shape
        encoder_hidden_states = encoder_hidden_states
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = self.to_q(hidden_states)
        _, res, dim = query.shape

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        # 先全局，后temporal
        query_raw = query.clone()
        key_raw = key.clone()
        # 这里的key 和 value 是第一帧+遮挡区域
        if warp:
            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            # key_0 = key[:, [0] * video_length]
            key_warp = match_flow_warp_kv(key, flow=flow_mask, video_length=video_length)
            # key = torch.cat([key_0, key_warp], dim=2)
            key = key_warp
            key = rearrange(key, "b f d c -> (b f) d c")
            # value 
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            # value_0 = value[:, [0] * video_length]
            value_warp = match_flow_warp_kv(value, flow=flow_mask, video_length=video_length)
            # value = torch.cat([value_0, value_warp], dim=2)
            value = value_warp
            value = rearrange(value, "b f d c -> (b f) d c")
        else: 
            assert ValueError('error of flow warp operation in error module')

        query = self.reshape_heads_to_batch_dim(query)
        if self.added_kv_proj_dim is not None:
            raise NotImplementedError
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # 变成fresco的head
        query = rearrange(query, "(b h) d c -> b d h c", b=batch_size, h=self.heads)
        key = rearrange(key, "(b h) d c -> b d h c", b=batch_size, h=self.heads)
        value = rearrange(value, "(b h) d c -> b d h c", b=batch_size, h=self.heads)
        # # #

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)
        
        hidden_states = rearrange(hidden_states, "b d h c -> (b h) d c", h=self.heads)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        
        # 在对q k v进行warp
        if warp:
            value = rearrange(hidden_states, "(b f) d c -> b f d c", f=video_length)  # (bf) d c(c*8) --> b f d c
            query = rearrange(query_raw, "(b f) d c -> b f d c", f=video_length)
            key = rearrange(key_raw, "(b f) d c -> b f d c", f=video_length)
            # warp
            query = rgb_flow_warp_nose(query, flow=flow_mask, video_length=video_length)
            # query = rgb_flow_warp_kv(query, flow=flow_mask, video_length=video_length)
            key_warp = rgb_flow_warp_kv(key, flow=flow_mask, video_length=video_length)
            if key_warp is None:
                hidden_states = hidden_states # 没有后续操作
            else:
                value_warp = rgb_flow_warp_kv(value, flow=flow_mask, video_length=video_length)
                
                # query = rearrange(query, "b f d c -> (b f) d c")
                key = rearrange(key_warp, "b f d c -> (b f) d c")
                value = rearrange(value_warp, "b f d c -> (b f) d c")

                query = self.reshape_heads_to_batch_dim(query)
                key = self.reshape_heads_to_batch_dim(key)
                value = self.reshape_heads_to_batch_dim(value)
                # 变成fresco的head
                query = rearrange(query, "(b h) d c -> b d h c", b=batch_size, h=self.heads)
                key = rearrange(key, "(b h) d c -> b d h c", b=batch_size, h=self.heads)
                value = rearrange(value, "(b h) d c -> b d h c", b=batch_size, h=self.heads)
                # # #

                if attention_mask is not None:
                    if attention_mask.shape[-1] != query.shape[1]:
                        target_length = query.shape[1]
                        attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                        attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

                if self._use_memory_efficient_attention_xformers:
                    hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
                    hidden_states = hidden_states.to(query.dtype)
                else:
                    if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                        hidden_states = self._attention(query, key, value, attention_mask)
                    else:
                        hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)
                        
                hidden_states = rearrange(hidden_states, "b d h c -> (b h) d c", h=self.heads)
                hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

                # query是区域型的，key和value是全局的
                # 对query进行warp或者ind，这里还需要对hideen_states进行warp。类似去掉重合部分
                # hidden_states = rearrange(hidden_states, "(b f) d c -> b f d c", f=video_length)
                # hidden_states = rgb_flow_warp_kv(hidden_states, flow=flow_mask, video_length=video_length)
                # hidden_states = rearrange(hidden_states, "b f d c -> (b f) d c")

        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states

    @staticmethod
    def visualize_and_save_features_pca(feats_map, t, save_dir, layer_idx):
        """
        feats_map: [B, N, D]
        """
        from sklearn.decomposition import PCA
        from PIL import Image
        B = len(feats_map)
        feats_map = feats_map.flatten(0, -2)
        feats_map = feats_map.cpu().numpy()
        pca = PCA(n_components=3)
        pca.fit(feats_map)
        feature_maps_pca = pca.transform(feats_map)  # N X 3
        feature_maps_pca = feature_maps_pca.reshape(B, -1, 3)  # B x (H * W) x 3
        if feature_maps_pca.shape[1] in [4096]:
            for i, experiment in enumerate(feature_maps_pca):
                pca_img = feature_maps_pca[i]  # (H * W) x 3
                h = w = int(np.sqrt(pca_img.shape[0]))
                pca_img = pca_img.reshape(h, w, 3)
                pca_img_min = pca_img.min(axis=(0, 1))
                pca_img_max = pca_img.max(axis=(0, 1))
                pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
                pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
                pca_img.save(os.path.join(save_dir, f"{i}_time_{t}_layer_{layer_idx}_res{h}.png"))