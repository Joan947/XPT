from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import DropPath

try:
    import natten
    from natten.functional import na1d
except ImportError:
    natten = None
    natten_functional = None
    # We will raise an error in SerializedAttention if NATTEN is not found,
    # as it's now a core dependency for the attention mechanism.

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        # Heuristic for relative position bound, assuming patch_size is K (num points in patch)
        # This formula might need review if patch_size represents something else in a 3D context.
        # Original code comment: "clamp into bnd" suggests these are grid coordinate differences.
        self.pos_bnd = int((self.patch_size) ** (1 / 3) * 4 * 2) # Adjusted as per typical RPEs in vision, was (4*patch_size)**(1/3)*2
        if self.pos_bnd == 0: # Ensure pos_bnd is at least 1 to avoid issues with rpe_num
            self.pos_bnd = 1
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord_diff):
        # coord_diff is expected to be (N_groups, K, K, 3) representing relative coordinates
        # N_groups = num_patches, K = patch_size (num_points_in_patch)
        # Ensure coord_diff is integer for indexing, if it's float from grid_coord differences
        coord_diff = coord_diff.long()
        idx = (
            coord_diff.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord_diff.device).view(1, 1, 1, 3) * self.rpe_num  # x, y, z stride
        ) # Shape: (N_groups, K, K, 3)
        
        # Flatten idx for index_select
        # Original idx.reshape(-1) would be (N_groups * K * K * 3)
        # RPE table size: (3 * rpe_num, num_heads)
        # index_select expects 1D index
        selected_rpe = self.rpe_table.index_select(0, idx.reshape(-1)) # (N_groups*K*K*3, num_heads)
        
        # Reshape and sum over the 3 spatial dimensions (x, y, z)
        # Original: out.view(idx.shape + (-1,)).sum(3) -> (N_groups, K, K, 3, H).sum(3) -> (N_groups, K, K, H)
        out = selected_rpe.view(idx.shape + (self.num_heads,)).sum(dim=3) # (N_groups, K, K, H)
        out = out.permute(0, 3, 1, 2)  # (N_groups, K, K, H) -> (N_groups, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size, # This is K, number of points per patch
        qkv_bias=True,
        qk_scale=None, # Note: NATTEN functional API uses its own scale, NATTEN module can take qk_scale
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
    ):
        super().__init__()
        if natten is None:
            raise ImportError(
                "NATTEN library is not installed, but it is required for SerializedAttention. "
                "Please install NATTEN (e.g., pip install natten) and try again."
            )
            
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.patch_size = patch_size  # K (number of points in a patch/window)
        self.order_index = order_index
        self.enable_rpe = enable_rpe

        if self.enable_rpe:
            self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
            self.rpe_module = RPE(patch_size, num_heads)
            # NATTEN's functional API (natten1d_forward) handles scaling internally (dim_head ** -0.5).
            # If a custom qk_scale is provided, it won't be directly used by natten1d_forward.
            # This is a slight behavior change if qk_scale was non-default.
            # For simplicity, we rely on NATTEN's default scaling.
            self.attn_drop = torch.nn.Dropout(attn_drop) # Applied to attention output (context)
            self.proj = torch.nn.Linear(channels, channels)
            self.proj_drop = torch.nn.Dropout(proj_drop)
        else:
            # Use NATTEN's NeighborhoodAttention1D module.
            # This module includes its own QKV projection, attention calculation, and output projection.
            self.na_attn = natten.NeighborhoodAttention1D(
                dim=channels,
                kernel_size=self.patch_size, # Full attention within the K points
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale, # NATTEN module can use this
                attn_drop=attn_drop, # Dropout on attention output (context)
                proj_drop=proj_drop, # Dropout on final projection within NATTEN module
                # rpb=True, # Can be enabled to use NATTEN's own 1D RPB
            )
            # self.qkv, self.rpe_module, self.proj, self.proj_drop are not needed here

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        # This function calculates relative grid coordinates for RPE
        # Input `order` is the order of points after padding and serialization
        # Output shape: (N_groups, K, K, 3)
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}_k{K}" # Make key K-specific if K can vary (not in this version)
        if rel_pos_key not in point.keys():
            # grid_coord should be integer-valued for RPE indexing purposes
            grid_coord = point.grid_coord[order].long() # (N_padded_total, 3)
            # Reshape to (N_groups, K, 3)
            grid_coord_grouped = grid_coord.reshape(-1, K, 3) # N_groups = N_padded_total / K
            # Calculate relative positions: (N_groups, K, 1, 3) - (N_groups, 1, K, 3) -> (N_groups, K, K, 3)
            point[rel_pos_key] = grid_coord_grouped.unsqueeze(2) - grid_coord_grouped.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        # This function remains largely the same, it prepares padded indices
        # based on self.patch_size (K)
        pad_key = f"pad_k{self.patch_size}"
        unpad_key = f"unpad_k{self.patch_size}"
        cu_seqlens_key = f"cu_seqlens_k{self.patch_size}"

        # Recompute if K changed or keys not present. K is fixed in this version.
        keys_exist = pad_key in point.keys() and unpad_key in point.keys() and cu_seqlens_key in point.keys()
        
        if not keys_exist:
            offset = point.offset
            bincount = offset2bincount(offset)
            
            # Calculate padding to make each segment a multiple of self.patch_size
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            
            # Mask for segments that actually need padding vs those that are too small
            # The original logic regarding (bincount > self.patch_size) seems to interact with how
            # small point clouds are handled. For NATTEN with fixed kernel size K,
            # all segments are padded to be multiples of K.
            # mask_pad = bincount > self.patch_size # Original condition
            # bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad # Original logic
            # Simpler: always pad to multiple of self.patch_size

            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            
            total_padded_points = _offset_pad[-1]
            pad = torch.empty(total_padded_points, dtype=torch.long, device=offset.device)
            unpad = torch.empty(_offset[-1], dtype=torch.long, device=offset.device) # Original number of points
            
            cu_seqlens_list = []

            for i in range(len(offset)): # Iterate over batch items
                start_orig, end_orig = _offset[i], _offset[i+1]
                start_pad, end_pad = _offset_pad[i], _offset_pad[i+1]
                
                num_orig_pts_in_segment = end_orig - start_orig
                num_pad_pts_in_segment = end_pad - start_pad

                # Original points
                pad[start_pad : start_pad + num_orig_pts_in_segment] = torch.arange(
                    start_orig, end_orig, device=offset.device
                )
                unpad[start_orig : end_orig] = torch.arange(
                    start_pad, start_pad + num_orig_pts_in_segment, device=offset.device
                )
                
                # Handle padding points: repeat the last valid point
                if num_pad_pts_in_segment > num_orig_pts_in_segment:
                    num_to_pad = num_pad_pts_in_segment - num_orig_pts_in_segment
                    # Pad with the index of the last original point in the segment if segment not empty
                    # otherwise pad with index 0 (or handle as invalid - but RPE needs valid coords)
                    # For safety, if num_orig_pts_in_segment is 0, this needs care.
                    # Assume num_orig_pts_in_segment > 0 if num_pad_pts_in_segment > 0.
                    # If a segment is empty, bincount[i] = 0, bincount_pad[i] = 0 (or K if we force min K).
                    # If bincount[i]=0, bincount_pad[i] could be 0 or self.patch_size. If self.patch_size, all are pad.
                    # Let's assume all segments have at least one point or are handled before this.
                    # If num_orig_pts_in_segment == 0 and num_pad_pts_in_segment > 0, all are padding.
                    # These padding points need valid feature/coord data. Often replicated from a real point.
                    # Here, `pad` indices point to original features. We need to make sure these are valid.
                    # The simplest for `pad` indices is to replicate the last point's index.
                    if num_orig_pts_in_segment > 0:
                        pad_val = end_orig - 1 
                    else: # Segment was empty, pad with dummy index 0 (features at 0 better be generic)
                        pad_val = 0 # This might require careful handling of features for padded elements.
                                    # For RPE, grid_coord[0] will be used.
                    
                    pad[start_pad + num_orig_pts_in_segment : end_pad] = pad_val
            
                # cu_seqlens for NATTEN/FlashAttention (not used by NATTEN here, but generated)
                cu_seqlens_list.append(
                    torch.arange(
                        start_pad,
                        end_pad, # Up to, but not including, end_pad
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens_list), (0, 1), value=total_padded_points # end offset for last segment
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]


    def forward(self, point: Point):
        H = self.num_heads
        K = self.patch_size # Fixed K, number of points in each attention window
        C = self.channels

        pad_indices, unpad_indices, _ = self.get_padding_and_inverse(point)

        # Get features and coordinates for serialized and padded points
        # `order` defines how global points are grouped into patches.
        # Here, `point.serialized_order` is (num_serialization_types, N_total_original_points)
        # We need to select one serialization order and apply padding.
        # `order` should be indices into the original point cloud features/coords
        order = point.serialized_order[self.order_index][pad_indices]
        # `inverse` maps from processed flat tensor back to original point structure (after unpadding)
        inverse = unpad_indices[point.serialized_inverse[self.order_index]]

        if self.enable_rpe:
            # 1. Project to QKV
            qkv_feat = self.qkv(point.feat)[order] # (N_padded_total, 3 * C)
            
            # 2. Reshape QKV for NATTEN
            # (N_padded_total, 3*C) -> (N_groups, K, 3, H, C//H) where N_groups = N_padded_total / K
            qkv_reshaped = qkv_feat.view(-1, K, 3, H, C // H)
            # Permute and unbind: (3, N_groups, H, K, C//H) -> Q, K, V each (N_groups, H, K, C//H)
            q, k, v = qkv_reshaped.permute(2, 0, 3, 1, 4).unbind(dim=0)

            # 3. Calculate RPE bias
            # get_rel_pos uses grid_coord of the *ordered and padded* points
            # Input to rpe_module: (N_groups, K, K, 3), relative grid coordinates
            # Output from rpe_module: (N_groups, H, K, K), the bias tensor
            relative_pos_bias = self.rpe_module(self.get_rel_pos(point, order))

            # 4. NATTEN forward pass
            # natten1d_forward expects query, key, value: (Batch, Heads, SequenceLength, DimHead)
            # Here, Batch=N_groups, SequenceLength=K
            # bias: (Batch, Heads, SequenceLength, KernelSize). For full attention in patch, KernelSize=K.
            # NATTEN's natten1d_forward handles QK scaling internally (dim_head ** -0.5)
            feat = na1d(
                query=q,
                key=k,
                value=v,
                bias=relative_pos_bias,
                kernel_size=K, # Full attention within the K points
                dilation=1     # Standard non-dilated attention
            ) # Output shape: (N_groups, H, K, C//H)
            
            # 5. Reshape output and apply dropout
            # (N_groups, H, K, C//H) -> (N_groups, K, H, C//H) -> (N_padded_total, C)
            feat = feat.transpose(1, 2).reshape(-1, C)
            feat = self.attn_drop(feat) # Dropout on attention output (context)

            # 6. Final projection
            feat = self.proj(feat)
            feat = self.proj_drop(feat)
        else:
            # Use NATTEN NeighborhoodAttention1D module
            # Input to NATTEN module: (Batch, SequenceLength, Channels)
            # Here, Batch=N_groups, SequenceLength=K, Channels=C
            x = point.feat[order].view(-1, K, C) # (N_groups, K, C)
            feat = self.na_attn(x) # Output: (N_groups, K, C)
            feat = feat.reshape(-1, C) # (N_padded_total, C)

        # Inverse operation: unpad and reorder to original point structure
        point.feat = feat[inverse]
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
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash = False,
        # Removed: enable_flash, upcast_attention, upcast_softmax
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
        self.attn = SerializedAttention( # Uses NATTEN
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
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
        point = self.drop_path(self.attn(point)) # Attn is now NATTEN-based
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
        if point.sparse_conv_feat is not None: # Check if sparse_conv_feat exists
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    # No changes needed in SerializedPooling, SerializedUnpooling, Embedding
    # unless they indirectly depend on the removed flags, which they don't seem to.
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
            code[0], # Operate on the first serialization order for pooling structure
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]

        # generate down code, order, inverse for all serialization types
        # Pooling structure is based on the first serialization, but all orderings are downsampled
        
        # Create new set of codes, orders, inverses for the pooled points
        # The number of pooled points is code_.shape[0]
        num_pooled_points = code_.shape[0]
        new_serialized_code = torch.empty_like(point.serialized_code[:, :num_pooled_points])
        new_serialized_order = torch.empty_like(point.serialized_order[:, :num_pooled_points])
        new_serialized_inverse = torch.empty_like(point.serialized_inverse[:, :num_pooled_points])

        # For each serialization type, select the codes of the new head_indices
        # and then recompute order/inverse for these pooled points
        for i in range(point.serialized_code.shape[0]):
            # Get the codes for the unique points selected by the first serialization order
            # These unique points (head_indices) are indexed based on the *original* point indexing
            # We need their codes from the i-th serialization.
            current_codes_for_heads = point.serialized_code[i, head_indices] >> pooling_depth * 3
            new_serialized_code[i] = current_codes_for_heads
            
            current_order = torch.argsort(current_codes_for_heads)
            new_serialized_order[i] = current_order
            
            # Inverse mapping for these pooled points under current_order
            current_inverse = torch.zeros_like(current_order)
            current_inverse.scatter_(
                dim=0, # Scatter along the single dimension
                index=current_order,
                src=torch.arange(0, num_pooled_points, device=current_order.device)
            )
            new_serialized_inverse[i] = current_inverse


        if self.shuffle_orders and new_serialized_code.shape[0] > 1: # Only shuffle if multiple orders exist
            perm = torch.randperm(new_serialized_code.shape[0])
            new_serialized_code = new_serialized_code[perm]
            new_serialized_order = new_serialized_order[perm] # Orders are specific to codes, so permute orders themselves
            # Inverse maps from order index to position, so they should also be permuted
            new_serialized_inverse = new_serialized_inverse[perm]


        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=(point.grid_coord[head_indices] >> pooling_depth), # Ensure integer grid coords
            serialized_code=new_serialized_code,
            serialized_order=new_serialized_order,
            serialized_inverse=new_serialized_inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices], # Batch index for each new pooled point
        )
        
        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster # Maps original points to pooled points
            point_dict["pooling_parent"] = point   # Store the original point object
        
        new_point = Point(point_dict)
        if self.norm is not None:
            new_point = self.norm(new_point)
        if self.act is not None:
            new_point = self.act(new_point)
        new_point.sparsify() # Create sparse tensor from new (pooled) points
        return new_point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,
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

    def forward(self, point): # current (pooled) point
        assert "pooling_parent" in point.keys(), "pooling_parent not found in point for unpooling"
        assert "pooling_inverse" in point.keys(), "pooling_inverse not found in point for unpooling"
        
        parent = point.pop("pooling_parent") # The point object before pooling
        inverse_cluster_map = point.pop("pooling_inverse") # Map from parent's points to current points

        # Project features of current (pooled) points and parent (skip-connection) points
        point = self.proj(point)
        parent_projected = self.proj_skip(parent) # This projects parent.feat

        # Add features: parent_projected.feat are original features projected
        # point.feat are upsampled features (from pooled points)
        # We need to add point.feat[inverse_cluster_map] to parent_projected.feat
        parent_projected.feat = parent_projected.feat + point.feat[inverse_cluster_map]
        
        # Other attributes from parent are largely kept (coord, grid_coord, serialization info, batch)
        # We are essentially updating the features of the parent point cloud
        
        if self.traceable:
            parent_projected["unpooling_parent"] = point # Store the (projected) pooled point object
        return parent_projected


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

        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1, # Should this be 2 for kernel_size 5 to maintain size? Or is it 'SAME' like?
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


@MODELS.register_module("PT-v3m1")# Changed name to reflect NATTEN usage
class PointTransformerV3Natten(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48), # K for NATTEN
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48), # K for NATTEN
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False, # For NATTEN-based attention
        enable_flash = False,
        # Removed: enable_flash, upcast_attention, upcast_softmax
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        super().__init__()
        if natten is None: # Check NATTEN availability at model initialization
            raise ImportError("NATTEN library is required for PointTransformerV3Natten but not found.")

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
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc_stage = PointSequential()
            if s > 0:
                enc_stage.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        shuffle_orders=self.shuffle_orders, # Pass shuffle_orders to pooling
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc_stage.add(
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
                        cpe_indice_key=f"enc_stage{s}_block{i}", # More specific key
                        enable_rpe=enable_rpe,
                    ),
                    name=f"block{i}",
                )
            if len(enc_stage) != 0: # Check if anything was added to enc_stage
                self.enc.add(module=enc_stage, name=f"enc{s}")

        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            # Note: dec_channels in config usually has one less element than enc_channels
            # The last decoder stage outputs to a channel size that matches a mid-encoder stage.
            # Example: enc_channels = (32, 64, 128, 256, 512)
            #          dec_channels = (64, 64, 128, 256) implies upsample from 512 to 256, then 256 to 128 etc.
            # The skip connection for the first upsampling (from enc_channels[-1]) comes from enc_channels[-2].
            # So, in_channels for SerializedUnpooling is enc_channels[s+1], skip_channels is enc_channels[s].
            # out_channels for SerializedUnpooling is dec_channels[s].

            # Let's adjust indexing for clarity.
            # dec_channels_full = [enc_channels[0]] + list(dec_channels) # Target output channels for each decoder stage
            # Example: dec_channels_cfg = (C3, C2, C1, C0_skip)
            # Stage s (from num_stages-2 down to 0):
            #   Unpool from enc_channel[s+1] (or previous decoder output)
            #   Skip from enc_channel[s]
            #   Output dec_channel[s_dec_idx]

            current_up_in_channels = enc_channels[-1] # Start with the bottleneck features
            for s_idx, s in enumerate(reversed(range(self.num_stages - 1))): # s = num_stages-2, ..., 0
                dec_s_idx = self.num_stages - 2 - s # s_idx mapping for dec_depths etc.
                                                     # s_idx = 0 for s = num_stages-2
                                                     # s_idx = num_stages-2 for s = 0

                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:dec_s_idx]) : sum(dec_depths[: dec_s_idx + 1])
                ]
                dec_drop_path_.reverse()
                
                dec_stage = PointSequential()
                dec_stage.add(
                    SerializedUnpooling(
                        in_channels=current_up_in_channels,
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[dec_s_idx],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                current_up_in_channels = dec_channels[dec_s_idx] # Input for next unpooling or final output

                for i in range(dec_depths[dec_s_idx]):
                    dec_stage.add(
                        Block(
                            channels=dec_channels[dec_s_idx],
                            num_heads=dec_num_head[dec_s_idx],
                            patch_size=dec_patch_size[dec_s_idx],
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
                            cpe_indice_key=f"dec_stage{s}_block{i}", # More specific key
                            enable_rpe=enable_rpe,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec_stage, name=f"dec{s}")

    def forward(self, data_dict):
        # point = Point(data_dict)
        # point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        # point.sparsify() # Initial sparsify for embedding

        # point = self.embedding(point)
        
        # # Store encoder features for skip connections
        # skip_connections = []

        # # Encoder pass
        # for s in range(self.num_stages):
        #     enc_module = getattr(self.enc, f"enc{s}")
        #     # If enc_module contains downsampling, it will handle point.sparsify()
        #     # If it's just blocks, they update point.feat and point.sparse_conv_feat
        #     point = enc_module(point)
        #     if not self.cls_mode and s < self.num_stages -1: # Store for all but last encoder stage
        #          # Create a new Point object for skip to avoid modification by later stages
        #         skip_point = Point(
        #             feat=point.feat.clone(),
        #             coord=point.coord.clone(),
        #             grid_coord=point.grid_coord.clone(),
        #             batch=point.batch.clone(),
        #             offset=point.offset.clone(),
        #             sparse_conv_feat=point.sparse_conv_feat.clone() if point.sparse_conv_feat is not None else None,
        #             # Serialization info also needed if unpooling reconstructs it, but SerializedUnpooling uses pooling_parent
        #             serialized_code = point.serialized_code.clone(),
        #             serialized_order = point.serialized_order.clone(),
        #             serialized_inverse = point.serialized_inverse.clone(),
        #             serialized_depth = point.serialized_depth
        #         )
        #         skip_connections.append(skip_point)


        # if not self.cls_mode:
        #     # Decoder pass
        #     # Skip connections are used in reverse order
        #     skip_connections.reverse()
        #     for s_idx, s in enumerate(reversed(range(self.num_stages - 1))): # s from num_stages-2 down to 0
        #         dec_module = getattr(self.dec, f"dec{s}")
        #         # The 'up' layer in dec_module (SerializedUnpooling) expects point.pooling_parent and point.pooling_inverse
        #         # These are set by SerializedPooling.
        #         # For unpooling, the `point` is the output of the previous decoder stage (or bottleneck)
        #         # and it needs to be combined with `skip_connections[s_idx]`.
        #         # This requires SerializedUnpooling to take two inputs or be refactored.
        #         # The current SerializedUnpooling takes one `point` and assumes `point.pooling_parent` exists.
        #         # This structure is for U-Nets where `point` is from deeper layer, `parent` is skip.
        #         # The `point.pooling_parent` in the current `SerializedUnpooling` refers to the state *before* the corresponding pooling.
        #         # Let's adjust how `SerializedUnpooling` gets its skip connection.
        #         # The `point` passed to `dec_module` is the output from the deeper layer.
        #         # `SerializedUnpooling` needs to access the skip connection.
        #         # We can make `pooling_parent` in `point` be the skip connection.
                
        #         # The `point` object passed to the decoder stage `dec{s}` is the output from the previous, deeper stage `dec{s+1}`.
        #         # This `point` object itself should contain `pooling_parent` (which is skip_connections[s_idx])
        #         # and `pooling_inverse` (which maps `pooling_parent` points to `point`'s points).
        #         # This means SerializedPooling must store the skip features in `point.pooling_parent.feat_for_skip` or similar.
        #         # Or, the loop here needs to manage it.
        #         # The current `SerializedUnpooling` expects `point.pop("pooling_parent")` to be the skip.
        #         # This `point` object is the one from the deeper layer of the U-Net.
        #         # Its `pooling_parent` was set when it (or its ancestor) was created by `SerializedPooling`.
        #         # So this should work: `point` from deeper layer is passed, its `pooling_parent` is the skip.
        #         point = dec_module(point)
        # # else: # cls_mode
        # #     # Global average pooling if classification mode
        # #     # This needs to be careful with batching if point.offset is used.
        # #     # torch_scatter.segment_csr is safer.
        # #     if point.offset is not None and len(point.offset) > 0 :
        # #         indptr = nn.functional.pad(point.offset, (1, 0)) # Convert offset to indptr
        # #         point.feat = torch_scatter.segment_csr(
        # #             src=point.feat,
        # #             indptr=indptr,
        # #             reduce="mean",
        # #         )
        # #     else: # Single point cloud in batch, or already pooled
        # #         point.feat = point.feat.mean(dim=0, keepdim=True)
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
        return point