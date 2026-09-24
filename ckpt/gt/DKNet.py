from functools import partial
import torch
import torch.nn as nn
from einops import rearrange

try:
    from timm.layers import DropPath
except ImportError:
    from timm.models.layers import DropPath

from model.block.gcn_conv import Gcn_block
from model.block.graph import Graph
from model.block.transformer import Attention, Mlp, KinematicHubOrthogonalFusion


class KinematicAnchoredPooling(nn.Module):
    """
    Kinematic-Anchored Biomechanical Pooling Layer (KAP).
    Injects proximal bone direction vectors into distal vulnerable joints
    while aggregating peripheral motion trends toward trunk hubs.
    """
    def __init__(self):
        super().__init__()
        # Distal joint kinematic chain index definitions (Human3.6M layout)
        self.r_leg = [2, 3]    # Right knee, right ankle
        self.l_leg = [5, 6]    # Left knee, left ankle
        self.l_arm = [12, 13]  # Left elbow, left wrist
        self.r_arm = [15, 16]  # Right elbow, right wrist

        # Learnable gating scalar initialized to zero for safe identity mapping at epoch 0
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = x.clone()

        # Phase 1: Bottom-up centripetal aggregation to proximal trunk hubs
        out[:, 1, :] = out[:, 1, :] + x[:, self.r_leg, :].mean(dim=1)    # Right hip
        out[:, 4, :] = out[:, 4, :] + x[:, self.l_leg, :].mean(dim=1)    # Left hip
        out[:, 11, :] = out[:, 11, :] + x[:, self.l_arm, :].mean(dim=1)  # Left shoulder
        out[:, 14, :] = out[:, 14, :] + x[:, self.r_arm, :].mean(dim=1)  # Right shoulder

        # Phase 2: Top-down kinematic bone displacement vector injection
        scale = torch.tanh(self.gamma)
        out[:, 3, :] = out[:, 3, :] + scale * (x[:, 1, :] - x[:, 2, :])    # Right ankle
        out[:, 6, :] = out[:, 6, :] + scale * (x[:, 4, :] - x[:, 5, :])    # Left ankle
        out[:, 13, :] = out[:, 13, :] + scale * (x[:, 11, :] - x[:, 12, :]) # Left wrist
        out[:, 16, :] = out[:, 16, :] + scale * (x[:, 14, :] - x[:, 15, :]) # Right wrist

        return out


class LocalTopologyBlock(nn.Module):
    """
    Local spatial topological modeling unit based on spatial GCN and KAP.
    """
    def __init__(self, dim, h_dim, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.graph = Graph('hm36_gt', 'spatial', pad=0)
        A_tensor = torch.tensor(self.graph.A, dtype=torch.float32)
        self.register_buffer('A', A_tensor)
        kernel_size = self.A.size(0)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.anat_pool = KinematicAnchoredPooling()

        self.gcn1 = Gcn_block(in_channels=dim, out_channels=h_dim, kernel_size=kernel_size, residual=False)
        self.norm_gcn1 = norm_layer(dim)

        self.gcn2 = Gcn_block(in_channels=h_dim, out_channels=dim, kernel_size=kernel_size, residual=False)
        self.norm_gcn2 = norm_layer(dim)

    def forward(self, x):
        res = x
        x_pooled = self.anat_pool(x)
        x_gcn, _ = self.gcn1(self.norm_gcn1(x_pooled), self.A)
        x_gcn, _ = self.gcn2(x_gcn, self.A)
        return res + self.drop_path(self.norm_gcn2(x_gcn))


class GlobalContextBlock(nn.Module):
    """
    Global spatial self-attention context unit based on Multi-Head Self-Attention (MHSA).
    """
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_attn = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        return x + self.drop_path(self.attn(self.norm_attn(x)))


class HeteroDualBranchBlock(nn.Module):
    """
    Heterogeneous dual-stream synergistic block combining local GCN topology
    and global self-attention context with orthogonal feature fusion.
    """
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim1 = dim1 = int(dim / 5)     # Default: 32 (local stream)
        self.dim2 = dim2 = dim - dim1       # Default: 128 (global stream)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.local1 = LocalTopologyBlock(dim1, dim1 * 2, drop_path=drop_path, norm_layer=norm_layer)
        self.global1 = GlobalContextBlock(dim1, num_heads=2, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0.1,
                                          attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer)

        self.norm_fusion = norm_layer(dim)
        self.fusion = KinematicHubOrthogonalFusion(
            in_features=dim,
            hidden_features=dim2,
            dim1=self.dim1,
            dim2=self.dim2,
            drop=drop
        )

        self.local2 = LocalTopologyBlock(dim2, dim2 * 2, drop_path=drop_path, norm_layer=norm_layer)
        self.global2 = GlobalContextBlock(dim2, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0.1,
                                          attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer)

        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim * 4, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim1, self.dim2], -1)

        x1 = self.local1(x1)
        x2 = self.global2(x2)

        x_fusion = torch.cat([x1, x2], -1)
        x_fusion_temp = x_fusion + self.fusion(self.norm_fusion(x_fusion))
        x_fusion_1, x_fusion_2 = torch.split(x_fusion_temp, [self.dim1, self.dim2], -1)

        x1 = self.global1(x1 + x_fusion_1)
        x2 = self.local2(x2 + x_fusion_2)

        x = torch.cat([x1, x2], -1) + x_fusion
        return x + self.drop_path(self.mlp(self.norm_mlp(x)))


class DKNet(nn.Module):
    """
    DualKine-Net: Dual-Stream Kinematic Network for Lightweight 3D Human Pose Estimation.
    """
    def __init__(self, args=None, depth=3, embed_dim=160, mlp_hidden_dim=1024, h=8, drop_rate=0.1):
        super().__init__()
        depth = getattr(args, 'layers', depth) if args is not None else depth
        embed_dim = getattr(args, 'channel', embed_dim) if args is not None else embed_dim
        mlp_hidden_dim = getattr(args, 'd_hid', mlp_hidden_dim) if args is not None else mlp_hidden_dim
        self.num_joints_in = getattr(args, 'n_joints', 17) if args is not None else 17
        self.num_joints_out = getattr(args, 'out_joints', 17) if args is not None else 17

        drop_path_rate = 0.3
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None

        self.patch_embed = nn.Linear(2, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_joints_in, embed_dim))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0.1, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            HeteroDualBranchBlock(
                dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        self.head_norm = norm_layer(embed_dim)
        self.fcn = nn.Linear(embed_dim, 3)

    def forward(self, x):
        if x.dim() == 4:
            x = rearrange(x, 'b f j c -> (b f) j c').contiguous()
        elif not x.is_contiguous():
            x = x.contiguous()

        x = self.patch_embed(x) + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.head_norm(x)
        output_3d = self.fcn(x)
        return output_3d.view(output_3d.shape[0], -1, self.num_joints_out, output_3d.shape[-1])