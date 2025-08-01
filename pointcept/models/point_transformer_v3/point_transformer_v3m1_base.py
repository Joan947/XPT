"""
Point Transformer - V3 Mode1

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.dynamic_tanh import DynamicTanh


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out

class RotaryEmbedding3D(nn.Module):
    def __init__(self, dim, base=10000, device=None):
        """
        Args:
            dim (int): Head dimension. Must be divisible by 6.
            base (float): Base value for frequency calculation.
        """
        super().__init__()
        if dim % 6 != 0:
            raise ValueError(f"Dimension {dim} must be divisible by 6 for 3D RoPE.")
        self.dim = dim
        self.dim_per_coord = dim // 3  # Dimension allocated for RoPE along one coordinate (e.g., X)
        self.base = base
        self.device = device

        # Precompute inverse frequencies
        # Each coordinate (X, Y, Z) will use self.dim_per_coord features.
        # RoPE operates on pairs, so inv_freq is for self.dim_per_coord / 2 pairs.
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim_per_coord, 2, device=device).float() / self.dim_per_coord))
        self.register_buffer("inv_freq", inv_freq, persistent=False) # (dim_per_coord / 2)

    def _apply_rope_single_coord(self, x, cos_emb, sin_emb):
        """
        Apply RoPE for a single coordinate to a part of the input tensor.
        Args:
            x (Tensor): Input tensor part, shape (..., seq_len, dim_per_coord)
            cos_emb (Tensor): Cosine embeddings, shape (..., seq_len, dim_per_coord / 2)
            sin_emb (Tensor): Sine embeddings, shape (..., seq_len, dim_per_coord / 2)
        Returns:
            Tensor: Rotated tensor part.
        """
        # Reshape x to (..., seq_len, dim_per_coord / 2, 2)
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]

        # Broadcast cos_emb and sin_emb if necessary (e.g., if x has a head dimension)
        # If x is (batch, heads, seq_len, dim_per_coord) and cos/sin are (batch, seq_len, dim_per_coord/2),
        # then unsqueeze cos/sin at dim 1.
        # Current design assumes cos/sin are already broadcastable or match x's leading dims.
        # For PTv3, Q/K are (N_patches_or_total_points, H, K_or_1, C_head_part)
        # Coords are (N_patches_or_total_points, K_or_1, 3)
        # Cos/Sin embeddings will be (N_patches_or_total_points, K_or_1, C_head_part / 2)
        # So they need .unsqueeze(1) to match H dimension in Q/K
        
        # If cos_emb/sin_emb are (..., K, Dpc/2) and x1/x2 are (..., H, K, Dpc/2), add head dim
        if x1.dim() > cos_emb.dim(): # Add head dimension
            cos_emb = cos_emb.unsqueeze(1)
            sin_emb = sin_emb.unsqueeze(1)
            
        rotated_x1 = x1 * cos_emb - x2 * sin_emb
        rotated_x2 = x1 * sin_emb + x2 * cos_emb
        
        return torch.stack((rotated_x1, rotated_x2), dim=-1).flatten(start_dim=-2)

    def forward(self, q, k, coords):
        """
        Apply 3D RoPE to query and key tensors.
        Args:
            q (Tensor): Query tensor, shape (batch_size, num_heads, seq_len, head_dim) or (total_tokens, num_heads, head_dim)
            k (Tensor): Key tensor, shape (batch_size, num_heads, seq_len, head_dim) or (total_tokens, num_heads, head_dim)
            coords (Tensor): Coordinates, shape (batch_size, seq_len, 3) or (total_tokens, 3)
                               These are the grid_coords.
        Returns:
            Tuple[Tensor, Tensor]: Rotated q and k tensors.
        """
        # coords: (N, K, 3) or (N_total, 3)
        # self.inv_freq: (dim_per_coord / 2)
        # We need t: (N, K, dim_per_coord / 2) or (N_total, dim_per_coord / 2)
        
        # Ensure coords are on the same device as inv_freq
        coords = coords.to(self.inv_freq.device)

        # Add a sequence dimension if coords is (total_tokens, 3) for consistency
        # This happens in flash attention path where Q/K are (total_tokens, H, C_h)
        # and coords are (total_tokens, 3). We treat seq_len=1 for each token in this case
        # for the purpose of generating embeddings, but then apply to the (total_tokens, H, C_h) tensor.
        # The "sequence" is effectively the flattened batch of points.
        if q.dim() == 3 and coords.dim() == 2: # Flash path (total_tokens, H, C_h), coords (total_tokens, 3)
            # Coords: (total_tokens, 3) -> t: (total_tokens, dim_per_coord/2)
            t_x = coords[..., 0:1] * self.inv_freq
            t_y = coords[..., 1:2] * self.inv_freq
            t_z = coords[..., 2:3] * self.inv_freq
        elif q.dim() == 4 and coords.dim() == 3: # Non-flash path (N_patches, H, K, C_h), coords (N_patches, K, 3)
            # Coords: (N_patches, K, 3) -> t: (N_patches, K, dim_per_coord/2)
            t_x = coords[..., None, 0] * self.inv_freq # (N, K, 1) * (Dpc/2) -> (N, K, Dpc/2)
            t_y = coords[..., None, 1] * self.inv_freq
            t_z = coords[..., None, 2] * self.inv_freq
        else:
            raise ValueError(f"Mismatch in q/k ({q.shape}) and coords ({coords.shape}) dimensions")

        cos_x, sin_x = t_x.cos(), t_x.sin()
        cos_y, sin_y = t_y.cos(), t_y.sin()
        cos_z, sin_z = t_z.cos(), t_z.sin()

        # Split q and k along the head dimension for X, Y, Z parts
        q_chunks = q.split(self.dim_per_coord, dim=-1)
        k_chunks = k.split(self.dim_per_coord, dim=-1)

        q_rotated_x = self._apply_rope_single_coord(q_chunks[0], cos_x, sin_x)
        q_rotated_y = self._apply_rope_single_coord(q_chunks[1], cos_y, sin_y)
        q_rotated_z = self._apply_rope_single_coord(q_chunks[2], cos_z, sin_z)

        k_rotated_x = self._apply_rope_single_coord(k_chunks[0], cos_x, sin_x)
        k_rotated_y = self._apply_rope_single_coord(k_chunks[1], cos_y, sin_y)
        k_rotated_z = self._apply_rope_single_coord(k_chunks[2], cos_z, sin_z)

        q_out = torch.cat((q_rotated_x, q_rotated_y, q_rotated_z), dim=-1)
        k_out = torch.cat((k_rotated_x, k_rotated_y, k_rotated_z), dim=-1)
        
        return q_out.type_as(q), k_out.type_as(k) # Cast back to original type

class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        norm_layer=DynamicTanh,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_rope=False, # New flag for RoPE
        rope_base=10000,   # RoPE base frequency
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        head_dim = channels // num_heads
        if enable_rope:
            assert not enable_rpe, "RoPE and RPE cannot both be enabled."
            assert head_dim % 6 == 0, "Head dimension must be divisible by 6 for 3D RoPE."

        self.channels = channels
        self.num_heads = num_heads
        # self.norm_q = norm_layer(channels // num_heads)
        # self.norm_k = norm_layer(channels // num_heads)
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_rope = enable_rope # Store the flag

        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None
        if self.enable_rope:
            self.rope = RotaryEmbedding3D(dim=channels // num_heads, base=rope_base)
        else:
            self.rope = None

    @torch.no_grad()
    def get_grid_coords_for_patch(self, point, order_padded):
        # order_padded are the indices of points after padding and serialization
        # point.grid_coord are the original grid coordinates
        # We need the grid_coord for the points in `order_padded`
        # Shape: (Total padded points, 3)
        patch_grid_coords = point.grid_coord[order_padded]
        return patch_grid_coords

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels
        C_head = C // H

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]
        # print(f"===========================================qkv shape: {qkv.shape}==================================================")
        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            #applying normalization
            # q = self.norm_q(q)
            # k = self.norm_k(k)
            # print(f"===========================================q per head shape: {q.shape}==================================================")
            # print(f"===========================================k per head shape: {k.shape}==================================================")
            
            # q = self.norm_q(q.reshape(-1,K,C_head)).reshape(-1, H, K, C_head) # Apply norm per-head token
            # k = self.norm_k(k.reshape(-1,K,C_head)).reshape(-1, H, K, C_head) # Apply norm per-head token
            # print(f"===========================================q shape: {q.shape}==================================================")
            # print(f"===========================================k shape: {k.shape}==================================================")
            
            
            if self.enable_rope:
                # Coords for RoPE: (N_patches, K, 3)
                # These are the grid coordinates of the points within each patch
                patch_grid_coords = self.get_grid_coords_for_patch(point, order).reshape(-1, K, 3)
                if self.rope.device != q.device: # Ensure RoPE module is on correct device
                    self.rope.to(q.device)
                q_dtype, k_dtype = q.dtype, k.dtype # Store original dtype
                q, k = self.rope(q.float(), k.float(), patch_grid_coords)
                q, k = q.to(q_dtype), k.to(k_dtype) # Cast back

            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            qkv_reshaped = qkv.reshape(-1, 3, H, C // H)  # (N', 3, H, C')
            q, k, v = qkv_reshaped[:, 0], qkv_reshaped[:, 1], qkv_reshaped[:, 2]  # (N', H, C')

            # Apply Norm across last dim
            # q = self.norm_q(q)
            # k = self.norm_k(k)
            # print(f"===========================================q shape: {q.shape}==================================================")
            # print(f"===========================================k shape: {k.shape}==================================================")
            
            # q = self.norm_q(q.reshape(-1, C_head)).reshape(q.shape)
            # k = self.norm_k(k.reshape(-1, C_head)).reshape(k.shape)
            print(f"===========================================q per head shape: {q.shape}==================================================")
            print(f"===========================================k per head shape: {k.shape}==================================================")
            

            if self.enable_rope:
                # Coords for RoPE: (N_total_padded, 3)
                # These are the grid coordinates of each point in the flattened, padded list
                # q_f, k_f are (N_total_padded, H, C_head)
                flat_grid_coords = self.get_grid_coords_for_patch(point, order) # (N_total_padded, 3)
                if self.rope.device != q.device:
                    self.rope.to(q.device)
                q_dtype, k_dtype = q.dtype, k.dtype
                q_f, k_f = self.rope(q.float(), k.float(), flat_grid_coords)
                q, k = q_f.to(q_dtype), k_f.to(k_dtype)

            # attn
            # Stack back into shape expected by flash_attn: (N', 3, H, C')
            qkv_normed = torch.stack([q, k, v], dim=1)
            print(f"===========================================qkv_normed shape: {qkv_normed.shape}==================================================")
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv_normed.half(),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)

            # feat = flash_attn.flash_attn_varlen_qkvpacked_func(
            #     qkv.half().reshape(-1, 3, H, C // H),
            #     cu_seqlens,
            #     max_seqlen=self.patch_size,
            #     dropout_p=self.attn_drop if self.training else 0,
            #     softmax_scale=self.scale,
            # ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        #norm_layer=nn.LayerNorm,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_rope=False, # New
        rope_base=10000,   # New
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_rope=enable_rope, # Pass through
            rope_base=rope_base,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


@MODELS.register_module("PT-v3m1")
class PointTransformerV3(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_rope=False, # New
        rope_base=10000, 
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1
        if enable_rope:
            assert not enable_rpe, "RoPE and RPE cannot both be enabled at the model level."
            # Check head dimensions for RoPE compatibility early
            for s in range(self.num_stages):
                head_dim_enc = enc_channels[s] // enc_num_head[s]
                if head_dim_enc % 6 != 0:
                    raise ValueError(f"Encoder stage {s}: head_dim {head_dim_enc} not div by 6 for RoPE")
            if not self.cls_mode:
                for s in range(len(dec_depths)): # dec_depths has one less element
                    head_dim_dec = dec_channels[s] // dec_num_head[s]
                    if head_dim_dec % 6 != 0:
                        raise ValueError(f"Decoder stage {s}: head_dim {head_dim_dec} not div by 6 for RoPE")
                    

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(DynamicTanh, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            #ln_layer = nn.LayerNorm
            ln_layer = DynamicTanh
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_rope=enable_rope, # <<<<< ADDED
                        rope_base=rope_base,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            #dec_channels = list(dec_channels) 
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_rope=enable_rope, # <<<<< ADDED
                            rope_base=rope_base,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        if not self.cls_mode:
            point = self.dec(point)
        # else:
        #     point.feat = torch_scatter.segment_csr(
        #         src=point.feat,
        #         indptr=nn.functional.pad(point.offset, (1, 0)),
        #         reduce="mean",
        #     )
        return point