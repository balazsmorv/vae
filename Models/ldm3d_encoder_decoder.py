"""
Adopted from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
"""

# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
import importlib
from einops import rearrange, einsum
from einops_exts import rearrange_many


def get_obj_from_str(string, reload=False):
    """
    Source: https://github.com/CompVis/latent-diffusion/blob/main/ldm/util.py#L88
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """
    Source: https://github.com/CompVis/latent-diffusion/blob/main/ldm/util.py#L78
    """
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


def Normalize(in_channels, num_groups=8):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Sequential(
                torch.nn.Conv3d(in_channels,
                                in_channels,
                                kernel_size=(1, 3, 3),
                                stride=1,
                                padding=(0, 1, 1)),
                torch.nn.Conv3d(in_channels,
                                in_channels,
                                kernel_size=(3, 1, 1),
                                stride=1,
                                padding=(1, 0, 0)),
            )

    def forward(self, x, size=None):
        if size is not None:
            x = torch.nn.functional.interpolate(x, size=size, mode="nearest")
        else:
            x = torch.nn.functional.interpolate(x, scale_factor=(2., 2., 2.), mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Sequential(
                torch.nn.Conv3d(in_channels,
                                in_channels,
                                kernel_size=(1, 3, 3),
                                stride=(1, 2, 2),
                                padding=0),
                torch.nn.Conv3d(in_channels,
                                in_channels,
                                kernel_size=(3, 1, 1),
                                stride=(2, 1, 1),
                                padding=0),
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Sequential(
            torch.nn.Conv3d(in_channels,
                            out_channels,
                            kernel_size=(1, 3, 3),
                            stride=1,
                            padding=(0, 1, 1)),
            torch.nn.Conv3d(out_channels,
                            out_channels,
                            kernel_size=(3, 1, 1),
                            stride=1,
                            padding=(1, 0, 0)),
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Sequential(
            torch.nn.Conv3d(out_channels,
                            out_channels,
                            kernel_size=(1, 3, 3),
                            stride=1,
                            padding=(0, 1, 1)),
            torch.nn.Conv3d(out_channels,
                            out_channels,
                            kernel_size=(3, 1, 1),
                            stride=1,
                            padding=(1, 0, 0)),
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Sequential(
                    torch.nn.Conv3d(in_channels,
                                    out_channels,
                                    kernel_size=(1, 3, 3),
                                    stride=1,
                                    padding=(0, 1, 1)),
                    torch.nn.Conv3d(out_channels,
                                    out_channels,
                                    kernel_size=(3, 1, 1),
                                    stride=1,
                                    padding=(1, 0, 0)),
                )
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class ResidualTemplate(nn.Module):
    def __init__(self, in_channels, fn):
        super().__init__()
        self.norm = LayerNorm(in_channels)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs) + x


class SpatialLinearAttention(nn.Module):
    """
    Source: https://github.com/lucidrains/video-diffusion-pytorch/blob/main/video_diffusion_pytorch/video_diffusion_pytorch.py#L193
    """

    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)


class EinopsToAndFrom(nn.Module):
    """
    Source: https://github.com/lucidrains/video-diffusion-pytorch/blob/main/video_diffusion_pytorch/video_diffusion_pytorch.py#L248
    """

    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
            self,
            x,
            pos_bias=None,
            focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if focus_present_mask is not None and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if pos_bias is not None:
            sim = sim + pos_bias

        if focus_present_mask is not None and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)


class Encoder3D(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, attn_heads=4, attn_dim_head=32,
                 **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c',
                                                    Attention(dim, heads=attn_heads, dim_head=attn_dim_head,
                                                              rotary_emb=rotary_emb))
        # downsampling
        self.conv_in = nn.Sequential(
            torch.nn.Conv3d(in_channels,
                            self.ch,
                            kernel_size=(3, 3, 3),
                            stride=1,
                            padding=(1, 1, 1)),
            # torch.nn.Conv3d(self.ch,
            #                 self.ch,
            #                 kernel_size=(3, 1, 1),
            #                 stride=1,
            #                 padding=(1, 0, 0))
        )

        self.temporal_attn_in = ResidualTemplate(in_channels, temporal_attn(in_channels))

        curr_res = resolution[1]
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(ResidualTemplate(block_in, SpatialLinearAttention(block_in)))
                    attn.append(ResidualTemplate(block_in, temporal_attn(block_in)))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(block_in, heads=attn_heads))

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = nn.Sequential(
            ResidualTemplate(block_in, spatial_attn),
            ResidualTemplate(block_in, temporal_attn(block_in))
        )
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=(3, 3, 3),
                                        stride=1,
                                        padding=(1, 1, 1))

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder3D(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_heads=4, attn_dim_head=32,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = np.array(resolution)
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        # self.z_shape = (1,z_channels,curr_res,curr_res)
        # print("Working with z of shape {} = {} dimensions.".format(
        #     self.z_shape, np.prod(self.z_shape)))
        curr_res = resolution[-2] // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, *(self.resolution // 2 ** (self.num_resolutions - 1)))
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv3d(z_channels,
                                       block_in,
                                       kernel_size=(3, 3, 3),
                                       stride=1,
                                       padding=(1, 1, 1))

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        temporal_attn = lambda dim: EinopsToAndFrom('b c f h w', 'b (h w) f c',
                                                    Attention(dim, heads=attn_heads, dim_head=attn_dim_head,
                                                              rotary_emb=rotary_emb))
        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(block_in, heads=attn_heads))

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = nn.Sequential(
            ResidualTemplate(block_in, spatial_attn),
            ResidualTemplate(block_in, temporal_attn(block_in))
        )
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(ResidualTemplate(block_in, SpatialLinearAttention(block_in)))
                    attn.append(ResidualTemplate(block_in, temporal_attn(block_in)))

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        out_ch,
                                        kernel_size=(3, 3, 3),
                                        stride=1,
                                        padding=(1, 1, 1))

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h, tuple(self.resolution // 2 ** (i_level - 1)))

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


from math import pi, log

import torch
from torch import nn, einsum

from einops import rearrange, repeat


# helper functions

def exists(val):
    return val is not None


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]

    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


# rotary embedding helper functions

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


def apply_rotary_emb(freqs, t, start_index=0, scale=1.):
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs = freqs[-seq_len:, :]

    freqs = freqs.to(t)
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[
        -1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim=-1)


# learned rotation helpers


def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)


# classes


class RotaryEmbedding(nn.Module):
    """
    Source: https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    """

    def __init__(
            self,
            dim,
            custom_freqs=None,
            freqs_for='lang',
            theta=10000,
            max_freq=10,
            num_freqs=1,
            learned_freq=False,
            use_xpos=False,
            xpos_scale_base=512,
            interpolate_factor=1.,
            theta_rescale_factor=1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        theta *= theta_rescale_factor ** (dim / (dim - 2))

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.cache = dict()
        self.cache_scale = dict()
        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)

        # interpolation factors

        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        # xpos

        self.use_xpos = use_xpos
        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.register_buffer('scale', scale)

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim=-2, offset=0, freq_seq_len=None):
        assert not self.use_xpos, 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        if exists(freq_seq_len):
            assert freq_seq_len >= seq_len
            seq_len = freq_seq_len

        freqs = self.forward(lambda: self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset),
                             cache_key=f'freqs:{seq_len}|offset:{offset}')
        return apply_rotary_emb(freqs, t)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim=-2):
        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len
        q = self.rotate_queries_or_keys(q, seq_dim=seq_dim, freq_seq_len=k_len)
        k = self.rotate_queries_or_keys(k, seq_dim=seq_dim)
        return q, k

    def rotate_queries_and_keys(self, q, k, seq_dim=-2):
        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]
        seq = self.get_seq_pos(seq_len, dtype=dtype, device=device)
        freqs = self.forward(lambda: seq, cache_key=f'freqs:{seq_len}')
        scale = self.get_scale(lambda: seq, cache_key=f'scale:{seq_len}').to(dtype)
        rotated_q = apply_rotary_emb(freqs, q, scale=scale)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale ** -1)
        return rotated_q, rotated_k

    def get_scale(self, t, cache_key=None):
        assert self.use_xpos

        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        if callable(t):
            t = t()

        scale = 1.
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')
            scale = torch.cat((scale, scale), dim=-1)

        if exists(cache_key):
            self.cache[cache_key] = scale

        return scale

    def forward(self, t, cache_key=None):
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        if callable(t):
            t = t()

        freqs = self.freqs

        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)

        if exists(cache_key):
            self.cache[cache_key] = freqs

        return freqs
